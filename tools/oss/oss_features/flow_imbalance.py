"""flow_imbalance — the REFERENCE feature module for the OSS harness.

Definition (verbatim from notebooks/features/flow_imbalance.ipynb / build_flow_imbalance.py §3):
the normalised signed trade-flow imbalance, per venue (bin/byb/okx), as an E/W
KernelMean EMA on the SHARED trade clock:

    feature(venue, N) = EMA(signed_qty) / EMA(qty)          ∈ [-1, 1]
    signed_qty = +qty if the trade lifts the ask (buy), -qty if it hits the bid (sell)

- Injected ONLY on that venue's own trades (summed over same-timestamp prints — one
  event), decayed once per trade-timestamp on the shared clock (α = 2/(N+1)).
- Read committed at the last clock tick at-or-before each anchor (causal,
  piecewise-constant between this venue's trades — no live front needed).
- NO σ-division: it is a bounded ratio already comparable across regimes.

This module is the TEMPLATE every other feature module follows: a single
`compute(arrays, grid) -> {venue: feature_array_on_grid}` with the on-grid
causal-read convention. See INTERFACE.md for the contract.
"""
import numpy as np
from scipy.signal import lfilter
from scipy.stats import spearmanr

NAME = "flow_imbalance"
EXCHANGES = ["bin", "byb", "okx"]
# The trade-span family the notebook sweeps (EMA memory in trades). The reported
# marginal-IC numbers come from the IN-SAMPLE best span per venue against the price
# head (notebook §6), which on block[0] lands on span 5 for every venue.
SPANS = [5, 20, 100, 500, 2000, 8000]
DEFAULT_SPAN = 100          # a single fixed span (used only by the bit-exact oracle check)


def _inject(arrays, ex):
    """Per-venue per-timestamp injected mass on the shared clock:
    E_inj = Σ signed_qty, W_inj = Σ qty at each clock tick. Same-timestamp prints are
    summed into ONE injection (searchsorted maps each trade to its clock tick)."""
    merged_ts = arrays.merged_ts
    n_ticks = len(merged_ts)
    rx, qty, sign = arrays.tr_rx[ex], arrays.tr_qty[ex], arrays.tr_sign[ex]
    k = np.searchsorted(merged_ts, rx, "left")                    # clock-tick index of each trade (exact match)
    e_inj = np.bincount(k, weights=sign * qty, minlength=n_ticks)  # Σ signed_qty per clock tick
    w_inj = np.bincount(k, weights=qty,        minlength=n_ticks)  # Σ qty per clock tick
    return e_inj, w_inj


def imbalance(arrays, grid, ex, N):
    """The E/W EMA-ratio for one venue at span N, read committed at each anchor's last
    clock tick (causal, piecewise-constant). Returns an array on the anchor grid."""
    a = 2.0 / (N + 1.0)
    e_inj, w_inj = _inject(arrays, ex)
    E = lfilter([a], [1.0, -(1.0 - a)], e_inj)
    W = lfilter([a], [1.0, -(1.0 - a)], w_inj)
    ratio = E / np.where(W > 0.0, W, np.nan)                       # E/W; nan until this venue has traded (W>0)
    return ratio[grid.tick_at_anchor]                             # value as of the last clock tick <= anchor


def best_spans(arrays, grid, head="price"):
    """The notebook §6 pick: per venue, the IN-SAMPLE best span against the head
    target (Spearman). This selection is in-sample only — the chosen feature is then
    re-scored OUT-OF-SAMPLE by the harness's walk-forward marginal IC. Returns
    {venue: span}."""
    target = grid.price_target if head == "price" else grid.rate_target
    out = {}
    for ex in EXCHANGES:
        scores = []
        for N in SPANS:
            d = imbalance(arrays, grid, ex, N)
            if head == "rate":
                d = np.abs(d)
            scores.append(spearmanr(d, target).statistic)
        out[ex] = SPANS[int(np.nanargmax(scores))]
    return out


def compute(arrays, grid, span=None, head="price"):
    """The module contract: return {venue: feature_array_on_grid} for flow_imbalance.

    arrays — BlockArrays from oss_core.load_block_arrays / load_cached.
    grid   — Grid from oss_core.build_grid (provides anchor_ts, tick_at_anchor, merged_ts).
    span   — None (default) -> per-venue in-sample best span (notebook §6, the
             reported number); or an int -> that fixed span for every venue.

    Returns one SIGNED array per venue, length len(grid.anchor_ts), read causally at
    every anchor (never |·| — the model is fed the signed feature for both heads)."""
    if span is None:
        spans = best_spans(arrays, grid, head=head)
        return {ex: imbalance(arrays, grid, ex, spans[ex]) for ex in EXCHANGES}
    return {ex: imbalance(arrays, grid, ex, span) for ex in EXCHANGES}
