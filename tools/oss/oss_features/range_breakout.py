"""range_breakout — per-venue Bollinger-style breakout, for the OSS harness.

Definition (verbatim from notebooks/features/range_breakout.ipynb / build_range_breakout.py §2-§3):
a per-venue σ-normalised breakout `z` computed on each venue's OWN mid, entirely from
trade-clock EMAs (NO rolling max/min, NO boxcar, NO fixed-N buffer):

    σ_band = √( ema(mid²,N) − ema(mid,N)² )
    z      = (mid − ema(mid,N)) / σ_band          (the price-head `breakout_magnitude`)

- The two band legs (`ema(mid,N)`, `ema(mid²,N)`) are forward-filled LEVELS, so each is a
  LiveFrontEMA: decay/commit once per trade-timestamp on the SHARED clock (α = 2/(N+1)),
  but READ on the live front at every anchor — `(1−α)·committed[tick_at_anchor] + α·fresh`,
  where `fresh` is the venue's mid as of the anchor.
- Numerically stabilised by centering on a fixed per-venue REF (each venue's first mid) before
  forming the band — z is EXACTLY shift-invariant to this, it only keeps the catastrophically
  cancelling variance `ema(mid²) − ema(mid)²` off the precision cliff (price ~2600).
- NORMALISATION: shipped RAW (the notebook §5 hard regime-invariance gate passes well under 3×
  for a z-score — no /σ_ev, no /λ_ev). The feature is self-normalised by its own band width.

HEAD = "price": a signed breakout (mid above/below its band) is a DIRECTION feature, scored
against grid.price_target. The IC sign decides continuation vs reversion per span/venue.

Legs: one per venue — "byb" (its own breakout vs its own future), "bin", "okx" (does the venue
breaking out lead byb?). The notebook ships per-venue legs; there is no cross-venue gap leg.

span selection (spans=None): per-venue IN-SAMPLE best span by |Spearman(z, price_target)| over
the band-span family (notebook §6 best_member), reproducing the §6 picks (byb N=50, bin/okx
N=200 on block[0]). spans={leg:N} forces a fixed span per leg (the harness FIXES block[0]'s
pick for the OOS run).
"""
import numpy as np
from scipy.signal import lfilter
from scipy.stats import spearmanr

import oss_core as core

NAME = "range_breakout"
HEAD = "price"                                   # signed direction feature -> scored vs grid.price_target

# the venues the notebook ships a leg for (EXCHANGES order from the notebook §2)
EXCHANGES = ["byb", "bin", "okx"]
# the band-span family swept in §6 (SHORT + LONG); each leg is ONE span N (not a fast/slow pair)
SHORT = [50, 200, 1000]
LONG  = [2000, 8000, 30000]
SPANS = SHORT + LONG


def _mid_streams(arrays):
    """Each venue's freshest-mid stream (rx, mid) per the boba.io MID_STREAM policy:
    byb/okx = merged_levels, bin = front_levels. byb is the already same-timestamp-collapsed
    byb merged mid; okx/bin via oss_core.mid_stream."""
    return {ex: core.mid_stream(arrays, ex) for ex in EXCHANGES}


def _refs(mids):
    """Fixed per-venue reference price (the venue's FIRST observed mid) — subtracted before the
    band purely for numerical stability; z is exactly shift-invariant to it."""
    return {ex: float(mids[ex][1][0]) for ex in EXCHANGES}


def _event_ema(series, N):
    """Plain committed EventEMA over a trade-clock series (α = 2/(N+1), y[-1]=0): the committed
    part of a LiveFrontEMA. Same recursion the production class runs (scipy lfilter)."""
    a = 2.0 / (N + 1.0)
    return lfilter([a], [1.0, -(1.0 - a)], series)


def breakout(arrays, grid, ex, N, mids=None, refs=None):
    """The σ-normalised breakout z for one venue at band span N, read on the live front at each
    anchor (causal). Returns an array on the anchor grid; NaN where the band has no width.

    Band legs (ema(mid), ema(mid²)) are LiveFrontEMAs:
      committed[k] = EventEMA over the (REF-centred) mid forward-filled to each trade tick,
      read = (1−α)·committed[tick_at_anchor] + α·fresh   (fresh = REF-centred mid at the anchor).
    z = (fresh − ema(mid)) / √(ema(mid²) − ema(mid)²)."""
    if mids is None:
        mids = _mid_streams(arrays)
    if refs is None:
        refs = _refs(mids)
    merged_ts = arrays.merged_ts
    tick_at_anchor = grid.tick_at_anchor
    anchor_ts = grid.anchor_ts
    rx, mid = mids[ex]
    ref = refs[ex]
    a = 2.0 / (N + 1.0)

    # committed (REF-centred) mid + mid² on the shared trade clock (last mid at-or-before each tick)
    clk_idx = np.clip(np.searchsorted(rx, merged_ts, "right") - 1, 0, len(mid) - 1)
    mid_clk = mid[clk_idx] - ref
    mid2_clk = mid_clk * mid_clk
    com_m = _event_ema(mid_clk, N)
    com_m2 = _event_ema(mid2_clk, N)

    # fresh (REF-centred) mid as of each anchor — the live front (every book update, never stale)
    fresh_idx = np.clip(np.searchsorted(rx, anchor_ts, "right") - 1, 0, len(mid) - 1)
    fresh = mid[fresh_idx] - ref

    em = (1.0 - a) * com_m[tick_at_anchor] + a * fresh
    em2 = (1.0 - a) * com_m2[tick_at_anchor] + a * (fresh * fresh)
    var = np.maximum(em2 - em * em, 0.0)                              # band variance, clamp tiny negative round-off
    sig_band = np.sqrt(var)
    z = (fresh - em) / np.where(sig_band > 0.0, sig_band, np.nan)     # σ-normalised; nan where the band has no width
    return z


def best_spans(arrays, grid, mids=None, refs=None):
    """The notebook §6 pick: per venue, the IN-SAMPLE best band span by |Spearman(z,
    price_target)| over the band-span family (best_member = |IC|-argmax). In-sample selection
    only; the chosen feature is re-scored OUT-OF-SAMPLE by the harness's walk-forward marginal IC.
    Returns {venue: span}."""
    if mids is None:
        mids = _mid_streams(arrays)
    if refs is None:
        refs = _refs(mids)
    target = grid.price_target
    out = {}
    for ex in EXCHANGES:
        scores = []
        for N in SPANS:
            z = breakout(arrays, grid, ex, N, mids=mids, refs=refs)
            v = np.isfinite(z) & np.isfinite(target)
            s = spearmanr(z[v], target[v]).statistic if v.sum() > 100 else np.nan
            scores.append(abs(s) if np.isfinite(s) else np.nan)       # pick by |IC| (sign carried separately)
        out[ex] = SPANS[int(np.nanargmax(scores))]
    return out


def compute(arrays, grid, spans=None):
    """The module contract: return {leg: feature_array_on_grid} for range_breakout.

    arrays — BlockArrays from oss_core.load_cached.
    grid   — Grid from oss_core.build_grid (anchor_ts, tick_at_anchor, merged_ts, price_target).
    spans  — None (default) -> per-venue IN-SAMPLE best span (notebook §6, reproduces the
             headline); or {leg: N} -> that fixed span per leg (the harness fixes block[0]'s pick
             for the OOS run; a missing leg falls back to its in-sample pick).

    Returns one SIGNED array per venue, length len(grid.anchor_ts), read causally on the live
    front at every anchor. Shipped RAW (no /σ_ev, /λ_ev) — self-normalised by the band width."""
    mids = _mid_streams(arrays)
    refs = _refs(mids)
    if spans is None:
        chosen = best_spans(arrays, grid, mids=mids, refs=refs)
    else:
        chosen = dict(spans)
        missing = [ex for ex in EXCHANGES if ex not in chosen]
        if missing:
            auto = best_spans(arrays, grid, mids=mids, refs=refs)
            for ex in missing:
                chosen[ex] = auto[ex]
    return {ex: breakout(arrays, grid, ex, chosen[ex], mids=mids, refs=refs) for ex in EXCHANGES}
