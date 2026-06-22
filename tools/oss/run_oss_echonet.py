"""run_oss_echonet — ECHO-NETTED out-of-sample (OOS) sweep for the boba feature harness.

Part-1 (run_oss_all.py -> oss_ic_matrix.csv) scored each feature's marginal rank-IC
over the base controls [rate_momentum, vol_momentum] against its HEAD forward target.
That RAW number includes the *contemporaneous echo*: a feature can post a high forward
IC just by re-reporting the move already underway at the anchor. The honest TRADEABLE
number nets that echo out by adding the already-happened (trailing) outcome to the
controls.

This script computes, per feature / leg / joint, per block, TWO marginal rank-ICs over
the feature's HEAD forward target via core.marginal_ic:

  RAW          base = [rate_momentum, vol_momentum]                  (reproduces Part-1)
  ECHO-NETTED  base = [rate_momentum, vol_momentum, trailing_outcome] (net-of-echo edge)

The trailing outcome is the already-happened analogue of the forward target, on the
same byb grid, over the strictly-past window [anchor-100ms, anchor]:

  PRICE head:  log(mid@anchor / mid@(anchor-100ms)) / sigma_ev@anchor
               (symmetric to the forward target: log(mid_fwd/mid_now)/sigma_ev)
  RATE head:   (byb mid-move count over (anchor-100ms, anchor]) / lambda_ev@anchor
               (symmetric to the forward target's fwd_count/lambda_ev)

Both windows are STRICTLY already-happened (end at the anchor) — no forward leak.

Spans are the FROZEN block[0] spans from fixed_spans_block0.json (the same fixed-span
OOS discipline as Part 1 — the spans never see blocks 1..57). Each feature/leg/joint
cell that errors is recorded as NaN and the run continues. Sequential, one block in
memory at a time (RAM-bounded). Checkpointed to oss_echonet_matrix.csv after every block.
"""
import sys, os, csv, json, time, traceback, importlib, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import oss_core as core

HARNESS = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HARNESS, "oss_echonet_matrix.csv")
SPANS_PATH = os.path.join(HARNESS, "fixed_spans_block0.json")
N_BLOCKS = 58
HORIZON_NS = core.HORIZON_NS                       # 100 ms, same window as the forward target

# Module order = the Part-1 report order (cross-venue / price-head keepers first).
MODULES = [
    "microprice", "ofi_normalised", "ofi", "price_momentum", "flow_persistence",
    "xv_book_pressure", "trade_rate_surge", "mid_rate_surge", "trade_rate_normalised",
    "volume_normalised", "vol_over_rate", "volume_surge", "gap_dynamics",
    "range_breakout", "flow_imbalance",
]


# ── the TRAILING (already-happened) outcome, symmetric to the forward target ──
def trailing_outcome(arrays, grid, head):
    """The already-happened analogue of grid.{price,rate}_target on the byb grid,
    over the strictly-past window [anchor-100ms, anchor]. STRICTLY past — ends at the
    anchor, so it carries the contemporaneous echo with NO forward leak.

    Mirrors oss_core.build_grid's forward-target construction exactly, but with the
    window shifted from (anchor, anchor+100ms] back to (anchor-100ms, anchor]:

      PRICE: log(mid@anchor / mid@(anchor-100ms)) / sigma_ev   (cf. log(mid_fwd/mid_now)/sigma_ev)
      RATE:  byb mid-moves over (anchor-100ms, anchor] / lambda_ev  (cf. fwd_count/lambda_ev)
    """
    anchor_ts = grid.anchor_ts
    byb_rx, byb_mid = arrays.byb_rx, arrays.byb_mid
    past_ts = anchor_ts - HORIZON_NS                        # 100 ms before each anchor

    if head == "price":
        # symmetric to: mid_now = byb_mid[ss(byb_rx, anchor_ts)-1];
        #               mid_fwd = byb_mid[ss(byb_rx, anchor_ts+H)-1]; log(mid_fwd/mid_now)/sigma_ev
        mid_anchor = byb_mid[np.searchsorted(byb_rx, anchor_ts, "right") - 1]
        mid_past   = byb_mid[np.searchsorted(byb_rx, past_ts,   "right") - 1]
        trailing_return = np.log(mid_anchor / mid_past)
        return trailing_return / grid.sigma_ev

    # rate: byb mid-move COUNT over (anchor-100ms, anchor], / lambda_ev.
    # cum_mv is the running byb mid-move count; rebuild it the same way build_grid does
    # (one move per byb timestamp where the byb log-mid changed). cum_mv[i] = #moves
    # strictly before byb index i, so the count over (a, b] is the same searchsorted
    # difference the forward target uses, with the window shifted back.
    byb_lm = np.log(byb_mid)
    byb_blr = np.empty_like(byb_lm)
    byb_blr[0] = 0.0
    byb_blr[1:] = np.diff(byb_lm)
    mv = byb_blr != 0.0                                     # a REAL byb mid-move per timestamp
    cum_mv = np.concatenate([[0.0], np.cumsum(mv.astype(float))])
    trailing_count = (cum_mv[np.searchsorted(byb_rx, anchor_ts, "right")]
                      - cum_mv[np.searchsorted(byb_rx, past_ts, "right")])
    return trailing_count / np.maximum(grid.lambda_ev, 1e-9)


# ── per-module FIXED-span adapter (block[0] selection), reused from run_oss_all ──
def head_of_record(rec):
    return rec.get("head", "price")


def compute_fixed(mod, A, G, rec):
    """Invoke compute with the FROZEN block[0] spans from fixed_spans_block0.json,
    per module signature. Mirrors run_oss_all.compute_fixed but reads the persisted
    record (kind + fixed_spans) instead of re-deriving on block[0]."""
    kind = rec["kind"]
    fixed = rec["fixed_spans"]
    if kind == "flow_imbalance":
        return {ex: mod.imbalance(A, G, ex, int(fixed[ex])) for ex in mod.EXCHANGES}
    # "spans": rebuild the payload compute() expects (ints or (fast, slow) tuples)
    payload = {}
    for leg, v in fixed.items():
        payload[leg] = (int(v[0]), int(v[1])) if isinstance(v, list) else int(v)
    return mod.compute(A, G, spans=payload)


# ── CSV columns ──────────────────────────────────────────────────────────────
def build_columns(sel):
    """Per feature: <name>__joint__raw, <name>__joint__net, then per-leg
    <name>__<leg>__raw / __net (legs in the persisted block[0] order)."""
    cols = ["block_idx", "block_name", "n_anchors"]
    for name in MODULES:
        if name not in sel:
            continue
        legs = sel[name]["legs"]
        cols += [f"{name}__joint__raw", f"{name}__joint__net"]
        for leg in legs:
            cols += [f"{name}__{leg}__raw", f"{name}__{leg}__net"]
    return cols


def score_block(mods, sel, A, G, cols):
    """Compute the RAW + ECHO-NETTED marginal IC for every feature/leg/joint on one
    block. Returns the row dict (all unset cells NaN). Robust: a feature that errors
    leaves its cells NaN and the rest continue."""
    base = [G.controls["rate_momentum"], G.controls["vol_momentum"]]
    # trailing outcomes for both heads (cheap; reused across features sharing a head)
    trail = {}
    for h in ("price", "rate"):
        try:
            trail[h] = trailing_outcome(A, G, h)
        except Exception:
            print(f"  [trailing {h}] FAILED -> echo-net NaN for {h}-head feats:\n{traceback.format_exc()}", flush=True)
            trail[h] = None

    row = {c: float("nan") for c in cols}
    for mod in mods:
        name = mod.NAME
        rec = sel[name]
        head = head_of_record(rec)
        target = G.price_target if head == "price" else G.rate_target
        base_net = base if trail[head] is None else base + [trail[head]]
        try:
            feats = compute_fixed(mod, A, G, rec)
            venue_arrs = list(feats.values())
            # joint
            row[f"{name}__joint__raw"] = core.marginal_ic(venue_arrs, target, base)
            if trail[head] is not None:
                row[f"{name}__joint__net"] = core.marginal_ic(venue_arrs, target, base_net)
            # per-leg
            for leg, arr in feats.items():
                rc, nc = f"{name}__{leg}__raw", f"{name}__{leg}__net"
                if rc in row:
                    row[rc] = core.marginal_ic(arr, target, base)
                if nc in row and trail[head] is not None:
                    row[nc] = core.marginal_ic(arr, target, base_net)
            jr = row[f"{name}__joint__raw"]; jn = row[f"{name}__joint__net"]
            print(f"    {name:22s} head={head:5s} joint raw={jr:+.4f} net={jn:+.4f}", flush=True)
        except Exception:
            print(f"    {name} FAILED -> NaN:\n{traceback.format_exc()}", flush=True)
    return row


# ── block[0] validation gate ──────────────────────────────────────────────────
def validate_block0(mods, sel):
    """Sanity-check before the full run: RAW block[0] numbers MUST match the Part-1
    oss_ic_matrix.csv block-0 row; the ECHO-NETTED block[0] numbers should roughly
    track the notebooks' echo-net findings."""
    print("[validate] loading block[0] ...", flush=True)
    A, G = core.load_cached(0)
    cols = build_columns(sel)
    row = score_block(mods, sel, A, G, cols)

    # the RAW reference values from oss_ic_matrix.csv block-0 row (Part-1)
    raw_ref = {}
    with open(os.path.join(HARNESS, "oss_ic_matrix.csv")) as f:
        r = csv.reader(f); header = next(r); b0 = next(r)
        for h, v in zip(header, b0):
            if "__" in h:
                try:
                    raw_ref[h] = float(v)
                except ValueError:
                    pass

    print("\n[validate] RAW block[0] vs Part-1 oss_ic_matrix.csv (must match):", flush=True)
    worst = 0.0
    for name in MODULES:
        ref_key = f"{name}__joint"                 # Part-1 column name (no raw/net suffix)
        mine = row.get(f"{name}__joint__raw", float("nan"))
        ref = raw_ref.get(ref_key, float("nan"))
        d = abs(mine - ref) if np.isfinite(mine) and np.isfinite(ref) else float("nan")
        if np.isfinite(d):
            worst = max(worst, d)
        flag = "OK" if (np.isfinite(d) and d < 5e-4) else "**MISMATCH**"
        print(f"  {name:22s} raw_joint mine={mine:+.4f}  part1={ref:+.4f}  |d|={d:.2e}  {flag}", flush=True)
    print(f"\n[validate] worst joint RAW |diff| vs Part-1 = {worst:.2e}  "
          f"({'PASS' if worst < 5e-4 else 'FAIL'})", flush=True)

    print("\n[validate] ECHO-NETTED block[0] spot-checks (should roughly track notebooks):", flush=True)
    for k in ("microprice__joint", "ofi_normalised__joint", "range_breakout__joint",
              "ofi_normalised__bin", "ofi_normalised__okx", "ofi_normalised__byb",
              "microprice__prem_byb", "range_breakout__byb", "price_momentum__byb"):
        rk, nk = f"{k}__raw", f"{k}__net"
        if rk in row:
            print(f"  {k:30s} raw={row[rk]:+.4f}  net={row[nk]:+.4f}  "
                  f"retain={ (row[nk]/row[rk]*100) if row[rk] not in (0.0,) and np.isfinite(row[rk]) and np.isfinite(row[nk]) else float('nan'):.0f}%",
                  flush=True)
    del A, G; gc.collect()
    return worst < 5e-4


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    validate_only = "--validate" in sys.argv
    mods = [importlib.import_module(f"oss_features.{m}") for m in MODULES]
    with open(SPANS_PATH) as f:
        sel = json.load(f)                          # FROZEN block[0] spans + kind + legs + head

    ok = validate_block0(mods, sel)
    if not ok:
        print("\n[validate] RAW block[0] did NOT reproduce Part-1 — aborting before the full run.", flush=True)
        sys.exit(1)
    if validate_only:
        print("\n[validate] --validate only; not running the full sweep.", flush=True)
        return

    cols = build_columns(sel)
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(cols)
    print(f"\n[csv] header written ({len(cols)} cols): {CSV_PATH}", flush=True)

    for bi in range(N_BLOCKS):
        t0 = time.time()
        cache_path = os.path.join(core.CACHE_DIR, f"block_{bi:03d}.npz")
        if not os.path.exists(cache_path):
            continue                                # not in the Part-1 cache; skip (no rebuild)
        print(f"\n[block {bi:2d}/{N_BLOCKS-1}] loading from cache ...", flush=True)
        try:
            A, G = core.load_cached(bi)
        except Exception:
            print(f"[block {bi}] LOAD FAILED:\n{traceback.format_exc()}", flush=True)
            row = {c: float("nan") for c in cols}
            row["block_idx"] = bi; row["block_name"] = ""
            with open(CSV_PATH, "a", newline="") as f:
                csv.writer(f).writerow([row[c] for c in cols])
            continue

        row = score_block(mods, sel, A, G, cols)
        row["block_idx"] = bi
        row["block_name"] = A.block
        row["n_anchors"] = len(G.anchor_ts)
        with open(CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow([row[c] for c in cols])
        del A, G; gc.collect()
        print(f"[block {bi:2d}] done in {time.time()-t0:.0f}s -> checkpointed CSV", flush=True)

    print(f"\n[ALL DONE] {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
