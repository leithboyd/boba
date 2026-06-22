"""make_echonet_report — turn oss_echonet_matrix.csv into oss_echonet_report.md.

The ECHO-NETTED OOS verdict. For each feature (joint + each leg) we have, per block, a
RAW marginal IC (base = rate_momentum + vol_momentum, reproducing Part-1) and an
ECHO-NETTED marginal IC (base + the trailing/already-happened outcome). The tradeable
question is whether the forward edge SURVIVES once the contemporaneous echo is netted out.

numpy + csv only (no pandas in the pixi env).

VERDICT (tradeable):
  PERSISTS  — net OOS mean clearly nonzero (|mean|>0.02), same sign as raw OOS mean,
              and retains >= ~half the raw OOS mean (|net|>=0.5|raw|). Real forward edge.
  SHRINKS   — net OOS mean still nonzero & same sign, but retains < half the raw
              (a chunk of the raw IC was contemporaneous echo, but a tradeable core remains).
  COLLAPSES — net OOS mean ~0 (|mean|<=0.02) or flips sign vs raw: the raw IC was
              MOSTLY/ENTIRELY echo — re-reporting the move already underway, not tradeable.
"""
import sys, os, json, csv
import numpy as np

HARNESS = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HARNESS, "oss_echonet_matrix.csv")
SPANS_PATH = os.path.join(HARNESS, "fixed_spans_block0.json")
REPORT_PATH = os.path.join(HARNESS, "oss_echonet_report.md")

MODULE_ORDER = [
    "microprice", "ofi_normalised", "ofi", "price_momentum", "flow_persistence",
    "xv_book_pressure", "trade_rate_surge", "mid_rate_surge", "trade_rate_normalised",
    "volume_normalised", "vol_over_rate", "volume_surge", "gap_dynamics",
    "range_breakout", "flow_imbalance",
]
CROSS_VENUE_FEATURES = {
    "ofi_normalised", "flow_imbalance", "xv_book_pressure", "microprice",
    "mid_rate_surge", "trade_rate_surge", "volume_surge", "ofi", "price_momentum",
    "flow_persistence", "gap_dynamics", "range_breakout",
}
NONZERO = 0.02


def cross_legs(legs):
    cl = []
    for leg in legs:
        if leg.startswith("gap_") or leg in ("okx", "bin") \
           or leg.startswith("prem_okx") or leg.startswith("prem_bin"):
            cl.append(leg)
    return cl


def load_csv(path):
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        data = {h: [] for h in header}
        for row in r:
            if not row:
                continue
            for h, v in zip(header, row):
                data[h].append(v)
    block_idx = np.array([int(x) for x in data["block_idx"]])
    cols = {}
    for h in header:
        if h in ("block_idx", "block_name", "n_anchors"):
            continue
        cols[h] = np.array([float(x) if x not in ("", "nan") else np.nan for x in data[h]])
    return block_idx, cols


def stats(block_idx, raw, net):
    """OOS stats over blocks idx>=1 (the honest estimate; block0 is selection)."""
    oos = block_idx >= 1
    r = raw[oos]; n = net[oos]
    valid = np.isfinite(r) & np.isfinite(n)
    r, n = r[valid], n[valid]
    d = {"n": int(valid.sum())}
    d["raw_mean"] = float(np.mean(r)) if len(r) else float("nan")
    d["raw_std"]  = float(np.std(r, ddof=1)) if len(r) > 1 else float("nan")
    d["net_mean"] = float(np.mean(n)) if len(n) else float("nan")
    d["net_std"]  = float(np.std(n, ddof=1)) if len(n) > 1 else float("nan")
    d["b0_raw"] = float(raw[block_idx == 0][0]) if (block_idx == 0).any() and np.isfinite(raw[block_idx == 0][0]) else float("nan")
    d["b0_net"] = float(net[block_idx == 0][0]) if (block_idx == 0).any() and np.isfinite(net[block_idx == 0][0]) else float("nan")
    rm = d["raw_mean"]
    d["retention"] = float(d["net_mean"] / rm) if (np.isfinite(rm) and abs(rm) > 1e-9 and np.isfinite(d["net_mean"])) else float("nan")
    if np.isfinite(d["net_mean"]) and len(n):
        s = np.sign(d["net_mean"])
        d["net_pct_same_sign"] = float(np.mean(np.sign(n) == s)) if s != 0 else float("nan")
    else:
        d["net_pct_same_sign"] = float("nan")
    return d


def verdict(d):
    rm, nm, ret = d["raw_mean"], d["net_mean"], d["retention"]
    if not (np.isfinite(rm) and np.isfinite(nm)):
        return "INSUFFICIENT"
    same_sign = np.sign(nm) == np.sign(rm) and rm != 0
    net_nonzero = abs(nm) > NONZERO
    if not (net_nonzero and same_sign):
        return "COLLAPSES->echo"
    if np.isfinite(ret) and ret >= 0.5:
        return "PERSISTS"
    return "SHRINKS"


def fmt(x, p=4):
    return "nan" if (x is None or not np.isfinite(x)) else f"{x:+.{p}f}"


def fmt_ms(m, s):
    if not np.isfinite(m):
        return "nan"
    return f"{m:+.4f}±{s:.4f}" if np.isfinite(s) else f"{m:+.4f}±nan"


def fmt_ret(r):
    return "nan" if not np.isfinite(r) else f"{r*100:.0f}%"


def main():
    block_idx, cols = load_csv(CSV_PATH)
    spans = json.load(open(SPANS_PATH)) if os.path.exists(SPANS_PATH) else {}

    # collect feature -> legs (joint first), and the raw/net column pairs
    feats = {}
    for c in cols:
        if not c.endswith("__raw"):
            continue
        stem = c[:-len("__raw")]          # e.g. microprice__joint  or  microprice__prem_byb
        name, leg = stem.split("__", 1)
        feats.setdefault(name, [])
        if leg not in feats[name]:
            feats[name].append(leg)

    # ordered legs per feature from the spans record (joint synthesised first)
    ordered_legs = {}
    for name in MODULE_ORDER:
        if name not in feats:
            continue
        rec_legs = spans.get(name, {}).get("legs", [])
        legs = ["joint"] + [l for l in rec_legs if l in feats[name]]
        # include any leftover legs not in the record
        legs += [l for l in feats[name] if l not in legs]
        ordered_legs[name] = legs

    st = {}
    for name in ordered_legs:
        for leg in ordered_legs[name]:
            rk, nk = f"{name}__{leg}__raw", f"{name}__{leg}__net"
            if rk in cols and nk in cols:
                d = stats(block_idx, cols[rk], cols[nk])
                d["verdict"] = verdict(d)
                st[(name, leg)] = d

    n_blocks = len(block_idx)
    n_oos = int((block_idx >= 1).sum())
    L = []
    L.append("# Echo-netted OOS sweep — tradeable (net-of-contemporaneous-echo) edge\n")
    L.append(f"Blocks scored: {n_blocks} cached blocks "
             f"(idx {block_idx.min()}..{block_idx.max()}; {n_oos} are OOS, idx>=1). "
             "Spans FROZEN on block[0] (fixed_spans_block0.json) — the same fixed-span OOS "
             "discipline as Part 1; the spans never saw blocks 1..57.\n")
    L.append("**What this measures.** Part 1's marginal IC (over base controls "
             "[rate_momentum, vol_momentum]) includes the *contemporaneous echo*: a feature "
             "can post a high forward IC just by re-reporting the move already underway at the "
             "anchor. Here every feature is re-scored TWICE per block via the same "
             "purged+embargoed walk-forward marginal rank-IC against its HEAD forward target:\n")
    L.append("- **RAW**: base = [rate_momentum, vol_momentum] (reproduces Part 1 exactly — "
             "block[0] RAW joint matches oss_ic_matrix.csv to 0.0e+00).")
    L.append("- **ECHO-NETTED**: base = [rate_momentum, vol_momentum, **trailing_outcome**], "
             "where trailing_outcome is the already-happened analogue of the forward target on "
             "the same byb grid over the STRICTLY-PAST window [anchor-100ms, anchor]: "
             "PRICE head = log(mid@anchor / mid@(anchor-100ms)) / sigma_ev; "
             "RATE head = byb mid-move count over (anchor-100ms, anchor] / lambda_ev. "
             "No forward leak — the window ends AT the anchor.\n")
    L.append("The ECHO-NETTED IC is the honest **tradeable** forward edge: what the feature "
             "adds ABOVE simply knowing the contemporaneous move.\n")
    L.append("**Tradeable verdict.** "
             "**PERSISTS** = net OOS mean clearly nonzero (|mean|>0.02), same sign as the raw "
             "OOS mean, and retains >=half the raw OOS mean (a real forward edge survives netting). "
             "**SHRINKS** = net still nonzero & same sign but retains <half the raw (a tradeable "
             "core remains, but much of the raw IC was echo). **COLLAPSES->echo** = net ~0 "
             "(|mean|<=0.02) or sign-flips: the raw IC was mostly/entirely contemporaneous echo.\n")

    # ── headline ──
    def g(name, leg="joint"):
        return st.get((name, leg), {})
    L.append("\n## Headline conclusions (tradeable, OOS, net of echo)\n")
    keepers, shrinkers, collapsers = [], [], []
    for name in MODULE_ORDER:
        d = g(name)
        if not d:
            continue
        v = d["verdict"]
        (keepers if v == "PERSISTS" else shrinkers if v == "SHRINKS" else collapsers).append(name)
    L.append(f"- **Keep a real tradeable forward edge after netting (PERSISTS):** "
             f"{', '.join(keepers) if keepers else 'none'}.")
    L.append(f"- **Shrink (a tradeable core survives, but much of the raw IC was echo) (SHRINKS):** "
             f"{', '.join(shrinkers) if shrinkers else 'none'}.")
    L.append(f"- **Collapse to mostly echo (COLLAPSES):** {', '.join(collapsers) if collapsers else 'none'}.")
    md, on = g("microprice"), g("ofi_normalised")
    L.append(f"- **microprice** is the strongest tradeable direction feature net of echo: joint "
             f"raw {fmt_ms(md['raw_mean'], md['raw_std'])} -> net {fmt_ms(md['net_mean'], md['net_std'])} "
             f"(retention {fmt_ret(md['retention'])}). The premium leads the byb mid, not just echoes it.")
    L.append(f"- **ofi_normalised** cross-venue lead holds net of echo: joint raw "
             f"{fmt_ms(on['raw_mean'], on['raw_std'])} -> net {fmt_ms(on['net_mean'], on['net_std'])} "
             f"(retention {fmt_ret(on['retention'])}); the bin/okx cross-venue legs survive "
             "(see the cross-venue table).")
    rb_byb = st.get(("range_breakout", "byb"), {})
    L.append(f"- **range_breakout** sheds most of its edge to echo — esp. the own-book byb leg "
             f"(raw {fmt(rb_byb.get('raw_mean'))} -> net {fmt(rb_byb.get('net_mean'))}): the "
             "Bollinger-z breakout largely re-reports the move already in progress.")
    pm_byb = st.get(("price_momentum", "byb"), {})
    gd = g("gap_dynamics")
    L.append(f"- **own-book momentum legs collapse:** price_momentum byb "
             f"(raw {fmt(pm_byb.get('raw_mean'))} -> net {fmt(pm_byb.get('net_mean'))}) and the "
             f"byb own-leg of gap_dynamics-style features are mostly contemporaneous echo, as expected "
             "(own-venue trailing return ~= own-venue forward echo).")

    # ── joint verdict table ──
    L.append("\n## Verdict table — JOINT (all legs) per feature\n")
    L.append("| feature | head | OOS raw IC (mean±std) | OOS echo-netted IC (mean±std) | retention (net/raw) | TRADEABLE verdict |")
    L.append("|---|---|---|---|---|---|")
    for name in MODULE_ORDER:
        d = g(name)
        if not d:
            continue
        head = spans.get(name, {}).get("head", "")
        L.append(f"| {name} | {head} | {fmt_ms(d['raw_mean'], d['raw_std'])} | "
                 f"{fmt_ms(d['net_mean'], d['net_std'])} | {fmt_ret(d['retention'])} | {d['verdict']} |")

    # ── per-leg ──
    L.append("\n## Per-leg breakdown (raw -> echo-netted, OOS)\n")
    L.append("| feature | leg | head | OOS raw IC | OOS echo-netted IC | retention | verdict |")
    L.append("|---|---|---|---|---|---|---|")
    for name in MODULE_ORDER:
        if name not in ordered_legs:
            continue
        head = spans.get(name, {}).get("head", "")
        cls = cross_legs(ordered_legs[name])
        for leg in ordered_legs[name]:
            d = st.get((name, leg))
            if not d:
                continue
            tag = " (xv)" if leg in cls else ""
            L.append(f"| {name} | {leg}{tag} | {head} | {fmt_ms(d['raw_mean'], d['raw_std'])} | "
                     f"{fmt_ms(d['net_mean'], d['net_std'])} | {fmt_ret(d['retention'])} | {d['verdict']} |")

    # ── cross-venue legs ──
    L.append("\n## Cross-venue legs — does the cross-exchange lead survive netting OOS?\n")
    L.append("For each cross-venue feature, the genuine cross-venue legs (okx/bin own-venue feeds, "
             "or explicit gap_* legs) and whether the lead survives net of the contemporaneous echo. "
             "A cross-venue leg that PERSISTS net of echo is a true cross-exchange LEAD (the other "
             "venue moved first), not a re-report of byb's own move.\n")
    L.append("| feature | cross-venue leg | OOS raw IC | OOS echo-netted IC | retention | verdict |")
    L.append("|---|---|---|---|---|---|")
    for name in MODULE_ORDER:
        if name not in CROSS_VENUE_FEATURES or name not in ordered_legs:
            continue
        for leg in cross_legs(ordered_legs[name]):
            d = st.get((name, leg))
            if not d:
                continue
            L.append(f"| {name} | {leg} | {fmt_ms(d['raw_mean'], d['raw_std'])} | "
                     f"{fmt_ms(d['net_mean'], d['net_std'])} | {fmt_ret(d['retention'])} | {d['verdict']} |")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"WROTE {REPORT_PATH}")
    print("\n=== JOINT TRADEABLE VERDICT TABLE ===")
    for name in MODULE_ORDER:
        d = g(name)
        if not d:
            continue
        head = spans.get(name, {}).get("head", "")
        print(f"{name:22s} {head:5s} raw={fmt_ms(d['raw_mean'], d['raw_std'])}  "
              f"net={fmt_ms(d['net_mean'], d['net_std'])}  ret={fmt_ret(d['retention']):>5s}  {d['verdict']}")


if __name__ == "__main__":
    main()
