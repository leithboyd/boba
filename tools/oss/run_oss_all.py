"""run_oss_all — the multi-block OUT-OF-SAMPLE (OOS) sweep for the boba feature harness.

A TRUE OOS test (not per-block re-optimisation):

  1. On block[0] ONLY, derive each feature's in-sample best span(s) per leg
     (via the module's best_spans() / spans=None selection). FIX those spans.
  2. For blocks 0..57 (SEQUENTIAL — one block in memory at a time, to bound RAM):
     load + build_grid, and for each feature call compute(spans=FIXED), scoring the
     per-block purged+embargoed walk-forward MARGINAL rank-IC over the base controls
     (rate_momentum + vol_momentum) against the feature's HEAD target — both the JOINT
     (all legs) and each leg.
  3. block[0] is the SELECTION block; the honest OOS estimate is over blocks 1..57.

The per-block marginal-IC matrix is checkpointed to oss_ic_matrix.csv after EVERY
block (rows = block, cols = <feature>__joint and <feature>__<leg>). A feature that
errors on a block records NaN for its cells and the run continues.

Spans are FIXED on block[0] and reused unchanged on every other block — the spans
never see blocks 1..57, which is what makes the 1..57 estimate honest OOS.
"""
import sys, os, csv, json, time, traceback, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import oss_core as core

HARNESS = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HARNESS, "oss_ic_matrix.csv")
SPANS_PATH = os.path.join(HARNESS, "fixed_spans_block0.json")
N_BLOCKS = 58

# Module order = the report order (cross-venue / price-head keepers first).
MODULES = [
    "microprice", "ofi_normalised", "ofi", "price_momentum", "flow_persistence",
    "xv_book_pressure", "trade_rate_surge", "mid_rate_surge", "trade_rate_normalised",
    "volume_normalised", "vol_over_rate", "volume_surge", "gap_dynamics",
    "range_breakout", "flow_imbalance",
]


def head_of(mod):
    return getattr(mod, "HEAD", "price")


# ── per-module FIXED-span adapter (block[0] selection) ───────────────────────
def derive_fixed(mod, A, G):
    """Return (kind, payload): how to FIX this module's block[0] spans.
    kind == "flow_imbalance" -> payload = {venue: span}, built directly via imbalance().
    kind == "spans"          -> payload = the dict handed to compute(spans=payload).
    """
    name = mod.NAME
    if name == "flow_imbalance":
        picks = mod.best_spans(A, G, head="price")          # {venue: span}
        return "flow_imbalance", {ex: int(picks[ex]) for ex in picks}
    bs = mod.best_spans(A, G)                                # keyed by the same legs compute() returns
    spans = {}
    for leg, v in bs.items():
        if isinstance(v, tuple):
            if len(v) == 3 and isinstance(v[0], str):       # ofi_normalised: (variant, span, ic)
                spans[leg] = int(v[1])
            else:                                            # (fast, slow) pair
                spans[leg] = (int(v[0]), int(v[1]))
        else:
            spans[leg] = int(v)
    return "spans", spans


def compute_fixed(mod, A, G, kind, payload):
    """Invoke compute with the FIXED block[0] spans, per module signature."""
    if kind == "flow_imbalance":
        return {ex: mod.imbalance(A, G, ex, payload[ex]) for ex in mod.EXCHANGES}
    return mod.compute(A, G, spans=payload)


# ── block[0] selection ───────────────────────────────────────────────────────
def select_block0(mods):
    """Derive + persist the block[0] fixed spans for every module. Returns
    {name: (kind, payload, head, legs)}."""
    print(f"[selection] loading block[0] ...", flush=True)
    A, G = core.load_cached(0)
    sel = {}
    record = {}
    for mod in mods:
        name = mod.NAME
        head = head_of(mod)
        kind, payload = derive_fixed(mod, A, G)
        feats = compute_fixed(mod, A, G, kind, payload)
        legs = list(feats.keys())
        sel[name] = (kind, payload, head, legs)
        record[name] = {
            "head": head, "kind": kind, "legs": legs,
            "fixed_spans": {k: (list(v) if isinstance(v, tuple) else v) for k, v in payload.items()},
        }
        print(f"[selection] {name:22s} head={head:5s} legs={legs} spans={record[name]['fixed_spans']}", flush=True)
    with open(SPANS_PATH, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[selection] wrote {SPANS_PATH}", flush=True)
    del A, G
    return sel


# ── CSV columns ──────────────────────────────────────────────────────────────
def build_columns(sel):
    cols = ["block_idx", "block_name", "n_anchors"]
    for name in MODULES:
        if name not in sel:
            continue
        _, _, _, legs = sel[name]
        cols.append(f"{name}__joint")
        for leg in legs:
            cols.append(f"{name}__{leg}")
    return cols


def main():
    mods = [importlib.import_module(f"oss_features.{m}") for m in MODULES]
    sel = select_block0(mods)
    cols = build_columns(sel)

    # fresh CSV with header
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(cols)
    print(f"[csv] header written ({len(cols)} cols): {CSV_PATH}", flush=True)

    for bi in range(N_BLOCKS):
        t0 = time.time()
        print(f"\n[block {bi:2d}/{N_BLOCKS-1}] loading ...", flush=True)
        try:
            A, G = core.load_cached(bi)
        except Exception:
            print(f"[block {bi}] LOAD FAILED:\n{traceback.format_exc()}", flush=True)
            # write an all-NaN row so the matrix stays rectangular
            row = {c: float("nan") for c in cols}
            row["block_idx"] = bi
            row["block_name"] = ""
            with open(CSV_PATH, "a", newline="") as f:
                csv.writer(f).writerow([row[c] for c in cols])
            continue

        n_anchors = len(G.anchor_ts)
        base = [G.controls["rate_momentum"], G.controls["vol_momentum"]]
        row = {c: float("nan") for c in cols}
        row["block_idx"] = bi
        row["block_name"] = A.block
        row["n_anchors"] = n_anchors

        for mod in mods:
            name = mod.NAME
            kind, payload, head, _legs = sel[name]
            target = G.price_target if head == "price" else G.rate_target
            try:
                feats = compute_fixed(mod, A, G, kind, payload)
                joint = core.marginal_ic(list(feats.values()), target, base)
                row[f"{name}__joint"] = joint
                for leg, arr in feats.items():
                    col = f"{name}__{leg}"
                    if col in row:
                        row[col] = core.marginal_ic(arr, target, base)
                pv = "  ".join(f"{leg}={row[f'{name}__{leg}']:+.3f}"
                               for leg in feats if f"{name}__{leg}" in row)
                print(f"[block {bi:2d}] {name:22s} joint={joint:+.4f}  ({pv})", flush=True)
            except Exception:
                print(f"[block {bi}] {name} FAILED -> NaN:\n{traceback.format_exc()}", flush=True)
                # cells already NaN; continue

        with open(CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow([row[c] for c in cols])
        del A, G, feats
        import gc; gc.collect()
        print(f"[block {bi:2d}] done in {time.time()-t0:.0f}s  -> checkpointed CSV", flush=True)

    print(f"\n[ALL DONE] {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
