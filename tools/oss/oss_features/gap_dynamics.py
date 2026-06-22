"""gap_dynamics — the cross-venue gap z-score (the OU reversion displacement).

Definition (verbatim from notebooks/features/gap_dynamics.ipynb / build_gap_dynamics.py
§2-§3). The SHIPPED reading is `gap_zscore` — a standardised displacement of the
cross-venue log price gap, per src venue (okx, bin):

    g      = log(mid_byb) - log(mid_src)                 # the cross-venue gap (byb is the reference)
    z      = (g_fresh - ema(g, N)) / sigma_g
    sigma_g = sqrt( ema(g^2, N) - ema(g, N)^2 )          # the gap's own recent spread

- The gap is a forward-filled LEVEL: `ema(g,N)`, `ema(g^2,N)` are COMMITTED trade-clock
  EMAs (alpha = 2/(N+1), y[-1]=0) on the merged trade clock, read at the last tick
  at-or-before each anchor (`grid.tick_at_anchor`). The displacement numerator `g_fresh`
  is the LIVE FRONT: the freshest gap as of the anchor (every book update since the last
  trade) — searchsorted(rx, anchor, "right")-1 on each venue's own mid stream. So the
  z's mean/spread are committed at the last tick, its displacement is current. This is
  exactly the notebook's LiveFrontEMA philosophy applied to the z.
- LEGS = the two CROSS-VENUE gaps (byb<->okx, byb<->bin). byb is the reference/target,
  NOT a separate leg (the gap is intrinsically byb-vs-src — there is no byb-vs-byb gap),
  exactly as the template's price_dislocation has no byb leg.
- SIGN: z>0 means byb sits HIGH vs src -> byb is expected to FALL back (reversion). The
  directional prediction is sign(-z); we ship the SIGNED z (never |z|) — the price head
  learns the sign, the rate head recovers the magnitude itself.
- NORMALISATION: ships RAW. §5's hard regime-invariance gate measures the z's scale
  across volatility buckets and it sits well under the 3x bar (a z-score is standardised
  by construction), so /sigma_ev and /lambda_ev do NOT improve it -> no yardstick divide.
- HEAD = "price": the feature feeds the price head (sign(-z) = byb's reversion direction),
  scored vs grid.price_target.

The `gap_halflife`/ac1 reading is a sub-tick degenerate DIAGNOSTIC in the notebook (the
gap is white at the trade-tick clock), NOT a shipped feature column, so it is not emitted
here — only the z legs are the model features.

Same module contract as the flow_imbalance reference: one
`compute(arrays, grid, spans=None) -> {leg: feature_array_on_grid}`, on-grid causal read.
See INTERFACE.md.
"""
import numpy as np
from scipy.signal import lfilter
from scipy.stats import spearmanr

import oss_core as core

NAME = "gap_dynamics"
HEAD = "price"                 # signed z feeds the price head (scored vs grid.price_target)

SRCS = ["okx", "bin"]          # the cross-venue gap legs; byb is the REFERENCE, not a leg
# The full gap-EMA span family the notebook §6 sweeps (EMA memory in trades).
SPANS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
DEFAULT_SPAN = 100             # a single fixed span (diagnostic / oracle convenience)
EPS_SG = 1e-9                  # sigma_g floor: same in §3 and the §4 oracle (~4 orders below a real gap std ~1e-4)


def _mid_on_clock(arrays, ex):
    """Causal: each venue's most-recent mid at-or-before every trade-clock tick.
    (rx, mid) come from the boba.io policy mid stream for the venue.)"""
    rx, mid = core.mid_stream(arrays, ex)
    idx = np.clip(np.searchsorted(rx, arrays.merged_ts, "right") - 1, 0, len(mid) - 1)
    return mid[idx]


def _mid_at(arrays, ex, t):
    """The freshest mid for a venue at arbitrary times t (the live front)."""
    rx, mid = core.mid_stream(arrays, ex)
    idx = np.clip(np.searchsorted(rx, t, "right") - 1, 0, len(mid) - 1)
    return mid[idx]


def _ema_commit(x, N):
    """Committed per-trade EMA, alpha = 2/(N+1), y[-1]=0 init — the project convention."""
    a = 2.0 / (N + 1.0)
    return lfilter([a], [1.0, -(1.0 - a)], x)


def _gap_committed(arrays):
    """g = log(mid_byb) - log(mid_src) on the trade clock, per src leg (one value/tick)."""
    log_mid_byb = np.log(_mid_on_clock(arrays, "byb"))
    return {ex: log_mid_byb - np.log(_mid_on_clock(arrays, ex)) for ex in SRCS}


def gap_zscore(arrays, grid, ex, N, g_committed=None):
    """The standardised cross-venue z for one leg at span N, read causally at each anchor.

    Mean/spread committed at the last trade tick (grid.tick_at_anchor); displacement
    g_fresh = log(mid_byb) - log(mid_src) read FRESH as of the anchor (live front).
    Returns a SIGNED array on the anchor grid; NaN where mids are undefined (clipped to
    the first valid mid in practice). sign(-z) is byb's expected reversion direction."""
    g = (g_committed if g_committed is not None else _gap_committed(arrays))[ex]
    tick = grid.tick_at_anchor
    em_g = _ema_commit(g, N)[tick]
    em_g2 = _ema_commit(g * g, N)[tick]
    sig_g = np.sqrt(np.maximum(em_g2 - em_g * em_g, 0.0))          # sigma_g = sqrt(E[g^2]-E[g]^2)
    g_fresh = (np.log(_mid_at(arrays, "byb", grid.anchor_ts))
               - np.log(_mid_at(arrays, ex, grid.anchor_ts)))       # the freshest gap (live front)
    return (g_fresh - em_g) / np.maximum(sig_g, EPS_SG)            # large +z -> byb high -> expected to fall


def best_spans(arrays, grid):
    """The notebook §6 pick: per leg, the IN-SAMPLE best span by |IC| (plain Spearman of
    the SIGNED z vs the price-head target). In-sample only — re-scored OOS by the harness
    walk-forward marginal IC. Returns {leg: span}."""
    target = grid.price_target
    g_committed = _gap_committed(arrays)
    out = {}
    for ex in SRCS:
        scores = []
        for N in SPANS:
            z = gap_zscore(arrays, grid, ex, N, g_committed=g_committed)
            ok = np.isfinite(z) & np.isfinite(target)
            scores.append(spearmanr(z[ok], target[ok]).statistic)
        out[ex] = SPANS[int(np.nanargmax(np.abs(scores)))]       # |IC|: the sign is the reversion verdict, not the strength
    return out


def compute(arrays, grid, spans=None):
    """The module contract: return {leg: signed_z_on_grid} for gap_dynamics.

    arrays — BlockArrays from oss_core.load_cached.
    grid   — Grid from oss_core.build_grid.
    spans  — None (default) -> per-leg in-sample best span (notebook §6, the reported
             number); or {leg: N} -> force that fixed span per leg (the harness uses this
             to FIX block[0]'s pick for the OOS run).

    Returns one SIGNED array per leg (okx, bin), length len(grid.anchor_ts), read causally
    at every anchor (never |z| — the model is fed the signed feature for both heads).
    Ships RAW (no /sigma_ev or /lambda_ev — the regime-invariance gate passes by
    construction)."""
    g_committed = _gap_committed(arrays)
    if spans is None:
        spans = best_spans(arrays, grid)
    return {ex: gap_zscore(arrays, grid, ex, spans[ex], g_committed=g_committed) for ex in SRCS}
