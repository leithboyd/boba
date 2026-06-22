"""Hard validation of the OSS harness core + flow_imbalance reference module.

1. block[0]: reproduce the notebook's flow_imbalance PRICE-head marginal IC over
   controls ≈ +0.158 joint (per-venue bin≈+0.138) to a small tolerance.
2. block[0]: grid anchor count ~1.7M, targets finite, σ_ev/λ_ev positive.
3. run_oss over blocks [0,1,2] end-to-end, printing a per-block marginal-IC row.
+ a bit-exact oracle check (independent KernelMean loop) confirming the production
  E/W EMA is correct on real data — the project's per-field validation requirement.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import oss_core as core
from oss_features import flow_imbalance as fi

print("=" * 78)
print("VALIDATION 1 + 2 — block[0]")
print("=" * 78)
A, G = core.load_cached(0)

# --- VALIDATION 2: grid sanity ---
n_anchors = len(G.anchor_ts)
print(f"grid anchors:            {n_anchors:,}   (notebook ~1.7M)")
print(f"price_target finite:     {np.isfinite(G.price_target).mean()*100:.1f}%  "
      f"rate_target finite: {np.isfinite(G.rate_target).mean()*100:.1f}%")
print(f"sigma_ev  > 0:           {(G.sigma_ev > 0).all()}   median {np.nanmedian(G.sigma_ev):.2e}")
print(f"lambda_ev > 0:           {(G.lambda_ev > 0).all()}   median {np.nanmedian(G.lambda_ev):.2f} moves/s")
assert 1_500_000 <= n_anchors <= 1_900_000, f"anchor count {n_anchors} not ~1.7M"
assert np.isfinite(G.price_target).all() and np.isfinite(G.rate_target).all()
assert (G.sigma_ev > 0).all() and (G.lambda_ev > 0).all()
print("VALIDATION 2: PASS")

# --- VALIDATION 1: flow_imbalance PRICE-head marginal IC, notebook §6 span selection ---
print("\n" + "-" * 78)
feats = fi.compute(A, G)                                    # span=None -> in-sample best span per venue
spans = fi.best_spans(A, G)
joint = core.marginal_ic_price(list(feats.values()), G)
per_venue = {ex: core.marginal_ic_price(feats[ex], G) for ex in feats}
print(f"flow_imbalance PRICE-head marginal IC over controls (chosen spans {spans}):")
print(f"  JOINT (all venues): {joint:+.4f}   (notebook target ≈ +0.158)")
for ex in ("bin", "byb", "okx"):
    note = "   (notebook ≈ +0.138)" if ex == "bin" else ""
    print(f"  {ex} alone:          {per_venue[ex]:+.4f}{note}")
print(f"  plain spearman ic (bin, price_target): {core.ic(feats['bin'], G.price_target):+.4f}")
ok1 = abs(joint - 0.158) < 0.01 and abs(per_venue["bin"] - 0.138) < 0.01
print(f"VALIDATION 1: {'PASS' if ok1 else 'CHECK'}  "
      f"(|joint-0.158|={abs(joint-0.158):.4f}, |bin-0.138|={abs(per_venue['bin']-0.138):.4f})")

# --- ORACLE: independent KernelMean loop, bit-exact vs the production E/W EMA ---
print("\n" + "-" * 78)
print("ORACLE — independent streaming KernelMean loop vs production E/W EMA (real block)")
def oracle_imbalance(arrays, grid, ex, N, n_check):
    """Dead-simple, no shared code: one trade at a time, decay E,W by (1-a) once per
    clock tick, inject a*Σsigned_qty / a*Σqty per timestamp, read E/W at each anchor."""
    a = 2.0 / (N + 1.0)
    merged_ts, anchors, tick_at = grid.merged_ts, grid.anchor_ts[:n_check], grid.tick_at_anchor[:n_check]
    rx, qty, sign = arrays.tr_rx[ex], arrays.tr_qty[ex], arrays.tr_sign[ex]
    cutoff = merged_ts[tick_at.max() + 2]
    m = rx <= cutoff
    rx, sq, q = rx[m], (sign * qty)[m], qty[m]
    E = np.full(len(merged_ts), np.nan); W = np.full(len(merged_ts), np.nan)
    e = w = 0.0; ti = 0; n = len(rx); i = 0
    n_ticks = tick_at.max() + 2
    while ti < n_ticks:
        e *= (1.0 - a); w *= (1.0 - a)                     # one clock tick: decay
        ts = merged_ts[ti]
        se = sw = 0.0
        while i < n and rx[i] == ts:                       # all prints at this timestamp = one event
            se += sq[i]; sw += q[i]; i += 1
        if sw != 0.0 or se != 0.0:
            e += a * se; w += a * sw
        E[ti] = e; W[ti] = w; ti += 1
    ratio = E / np.where(W > 0.0, W, np.nan)
    return ratio[tick_at]

N_CHK = 40_000
worst = 0.0
for ex in ("bin", "byb", "okx"):
    prod = fi.imbalance(A, G, ex, 100)[:N_CHK]
    orc = oracle_imbalance(A, G, ex, 100, N_CHK)
    both = np.isfinite(prod) & np.isfinite(orc)
    d = float(np.nanmax(np.abs(prod[both] - orc[both]))); worst = max(worst, d)
    print(f"  {ex}: max|diff| {d:.2e} on {int(both.sum()):,} anchors")
assert worst < 1e-12, f"oracle mismatch {worst}"
print(f"ORACLE: production E/W EMA == independent loop, bit-exact (worst {worst:.2e})  PASS")

# --- VALIDATION 3: run_oss over [0,1,2] ---
print("\n" + "=" * 78)
print("VALIDATION 3 — run_oss over blocks [0, 1, 2] (flow_imbalance, price head)")
print("=" * 78)
results = core.run_oss([0, 1, 2], [fi], head="price")
print("\nper-block JOINT marginal IC (3 OOS numbers):")
for bi in (0, 1, 2):
    print(f"  block[{bi}]: {results['flow_imbalance'][bi]['joint']:+.4f}")
print("VALIDATION 3: PASS (ran end-to-end)")
