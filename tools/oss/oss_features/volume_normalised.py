"""volume_normalised — feature module for the OSS harness.

Definition (verbatim from notebooks/features/volume_normalised.ipynb / build_volume_normalised.py
§2/§3/§6/§10): a SINGLE span-N KernelMeanEMA (E/W) of a venue's per-trade-event traded qty, divided
by a regime yardstick:

    volume_normalised(ex; N) = (E/W)_N(qty_ex) / yardstick

- E/W is the exp-weighted MEAN traded qty per venue trade-event: on each `ex` trade-timestamp,
  inject Σqty of that timestamp's simultaneous prints (ONE event); decay once per shared trade-
  timestamp (α = 2/(N+1)). Read E/W = mean qty per venue trade (non-venue ticks cancel; never push
  a 0 on a non-event).
- Read committed at the last shared trade-clock tick at-or-before each anchor (grid.tick_at_anchor):
  the EMA is piecewise-constant between shared ticks (flat-between-ticks committed-E read, exactly
  §3's `_ew_flow_at` — NO live-front correction; the volume EMA is sparse-flow, decay rides the
  shared clock, and the §3 build reads the committed E at the last shared tick).
- NORMALISATION: the SHIPPED form is `/ σ_ev` (the §6 hard regime-invariance scale gate disqualifies
  the un-normalised baseline at ≈3.65× and rejects `/λ_ev` on a negative marginal; `/σ_ev` passes at
  ≈2.5× and is the shipped invariant form). σ_ev is byb's RMS-per-move yardstick at span YARDSTICK_N,
  carried on grid.sigma_ev. Applied as a per-anchor division.

HEAD = "rate" — this is an INTENSITY feature (traded volume is an activity clock, not a direction
signal); it is scored against grid.rate_target. The §6 span pick is per venue the SIGNED rate-head IC
argmax over the span family (the model is fed the signed feature; never |·|).

Legs: one per venue (byb / okx / bin) — each that venue's OWN normalised traded volume. There is no
constructed cross-venue "gap" leg; the cross-venue analysis (§9) just adds okx+bin's own legs.
"""
import numpy as np
from scipy.signal import lfilter
from scipy.stats import spearmanr

NAME = "volume_normalised"
HEAD = "rate"
EXCHANGES = ["byb", "okx", "bin"]
# the SINGLE-EMA span family swept (§3 SPANS). span=1 is degenerate for an E/W flow (α=1 fully
# decays E to 0 each tick), so the family starts at 2.
SPANS = [2, 10, 50, 200, 1000, 5000]
DEFAULT_SPAN = 50           # a single fixed span (used only as a fallback)
EPS = 1e-300


def _qty_per_ts(arrays, ex):
    """Per-venue total qty summed over simultaneous prints into ONE event per trade-TIMESTAMP, plus
    the shared-clock tick index of each such timestamp. (Same-timestamp prints are one event.)"""
    rx, qty = arrays.tr_rx[ex], arrays.tr_qty[ex]
    u, inv = np.unique(rx, return_inverse=True)              # unique venue trade-timestamps
    q_sum = np.bincount(inv, weights=qty)                    # Σ qty over that timestamp's prints
    kt = np.searchsorted(arrays.merged_ts, u, "left")        # shared-clock tick index of each (exact match)
    return kt, q_sum


def volume_ema(arrays, grid, ex, N):
    """ema(volume) = E/W = exp-weighted MEAN qty per venue trade at span N, read committed at each
    anchor's last shared trade-clock tick (causal, flat between shared ticks). Returns an array on
    the anchor grid. NaN until the venue has traded (W <= 0)."""
    a = 2.0 / (N + 1.0)
    n_ticks = len(arrays.merged_ts)
    kt, q_sum = _qty_per_ts(arrays, ex)
    # E leg: inject a*qty at the venue-trade tick, decay once per shared tick (§3 `_ew_flow_at`).
    e_inj = np.zeros(n_ticks)
    np.add.at(e_inj, kt, a * q_sum)
    E = lfilter([1.0], [1.0, -(1.0 - a)], (1.0 - a) * e_inj)
    # W leg: same recursion with weight 1 per venue trade-event.
    w_inj = np.zeros(n_ticks)
    np.add.at(w_inj, kt, a * np.ones_like(q_sum))
    W = lfilter([1.0], [1.0, -(1.0 - a)], (1.0 - a) * w_inj)
    # E/W with the denominator clamped at EPS — VERBATIM the notebook §3 read
    # (`E/np.maximum(W,1e-300)`); this is bit-exact in the live region and, at the
    # degenerate small span (N=2) where W can decay to a SUBNORMAL ~1e-313, reproduces
    # the notebook's clamped value rather than the raw E/W.
    ratio = E / np.maximum(W, EPS)
    ratio[W <= 0.0] = np.nan                                 # NaN-where-undefined: before this venue's first trade
    return ratio[grid.tick_at_anchor]                        # committed value at the last shared tick <= anchor


def normalised(arrays, grid, ex, N):
    """THE SHIPPED FEATURE for one venue at span N: ema(volume) / σ_ev (the §6/§10 invariant form)."""
    v = volume_ema(arrays, grid, ex, N)
    return v / np.maximum(grid.sigma_ev, EPS)


def best_spans(arrays, grid):
    """The §6 pick: per venue, the IN-SAMPLE best span by SIGNED rate-head IC (Spearman vs
    grid.rate_target), matching the notebook's `rate_member = argmax(rate_grid[ex])`. In-sample only;
    the chosen feature is re-scored OOS by the harness's walk-forward marginal IC. Returns {ex: span}."""
    out = {}
    for ex in EXCHANGES:
        scores = []
        for N in SPANS:
            d = normalised(arrays, grid, ex, N)
            scores.append(spearmanr(d, grid.rate_target).statistic)   # SIGNED rate-head IC (the feature's home)
        out[ex] = SPANS[int(np.nanargmax(scores))]
    return out


def compute(arrays, grid, spans=None):
    """The module contract: return {leg: feature_array_on_grid} for volume_normalised.

    arrays — BlockArrays from oss_core.
    grid   — Grid from oss_core.build_grid (anchor_ts, tick_at_anchor, merged_ts, sigma_ev).
    spans  — None (default) -> per-venue IN-SAMPLE best span by signed rate-head IC (the §6 headline);
             or a {leg: N} dict -> force those fixed spans (the harness FIXES block[0]'s pick for OOS).

    Returns one SIGNED array per venue, length len(grid.anchor_ts), read causally at every anchor
    (the model is fed the signed feature for both heads; never |·|). NaN where undefined."""
    if spans is None:
        spans = best_spans(arrays, grid)
    elif isinstance(spans, int):
        spans = {ex: spans for ex in EXCHANGES}
    return {ex: normalised(arrays, grid, ex, spans[ex]) for ex in EXCHANGES}
