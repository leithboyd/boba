"""price_momentum — per-venue trade-clock momentum (the OSS feature module).

Definition (verbatim from notebooks/features/price_momentum.ipynb / build_price_momentum.py
§2/§3/§5/§6) — the SHIPPED headline form is `signed_momentum` (form 1), built PER VENUE
(byb's own leg + the cross-venue okx/bin legs, all kept; no fixed leader):

    atom  signed_ret_N(ex) = EMA(Δlog mid, N)   — smoothed signed drift of a venue's mid,
          built as a decayed-SUM EMA on the SHARED trade clock with a LIVE FRONT read:
          inject α·Δlog on every venue mid-move, decay by (1−α) once per trade-timestamp,
          read committed[last tick ≤ anchor] + α·(Σ Δlog since that tick).  α = 2/(N+1).

    signed_momentum(ex, N) = signed_ret_N            (RAW)         if the scale gate PASSES
                           = signed_ret_N / σ_ev     (/σ_ev)       if it FAILS  (raw_scale > 3)

The §5 RAW-FIRST scale gate is MEASURED on this block (max/min of the per-vol-decile std of
byb's signed_ret(50)); we ship the form the gate picks — never assumed. (k drops out of every
rank-IC, so the only real choice is raw vs /σ_ev; we use σ_ev with no constant.)

Span selection (§6): spans=None → per venue, the IN-SAMPLE best span = argmax over SPANS of
|Spearman(signed_momentum(ex, N), price_target)| (the directional/price head — momentum's
home; continuation + / reversion −). spans={leg:N} forces a fixed span per leg (the harness
uses this to FIX block[0]'s pick for the OOS run).

This module follows the flow_imbalance template's `compute(arrays, grid) -> {leg: array_on_grid}`
contract (INTERFACE.md). HEAD = "price": the signed feature is scored vs grid.price_target.
"""
import numpy as np
from scipy.signal import lfilter
from scipy.stats import spearmanr

import oss_core as core

NAME = "price_momentum"
HEAD = "price"                      # directional feature -> scored vs grid.price_target
EXCHANGES = ["byb", "okx", "bin"]   # built per venue off its OWN mid-moves; keep ALL (no fixed leader)

# the single-span family swept in §6 = sorted(set(FAST) | set(SLOW)) with
# FAST=[1,10,50,200], SLOW=[100,500,2000,8000]
SPANS = [1, 10, 50, 100, 200, 500, 2000, 8000]
SCALE_SPAN = 50                     # span at which §5 measures the raw-vs-/σ_ev scale gate
SCALE_GATE = 3.0                    # ship /σ_ev iff byb signed_ret(50) scale across vol buckets > 3×


# ── per-venue mid-move stream (the sparse Δlog flow the momentum leg is built from) ──
def _move_stream(arrays, ex):
    """THIS venue's signed mid-move flow on its own collapsed mid stream: (move_rx, move_dlog).
    Same-TIMESTAMP mid rows collapse to ONE update (the final mid) — simultaneous events are not
    a sequence; only rows where the mid actually CHANGED are real moves."""
    rx0, mid0 = core.mid_stream(arrays, ex)
    keep = np.concatenate([rx0[1:] != rx0[:-1], [True]])      # collapse same-timestamp rows -> final mid
    rx, mid = rx0[keep], mid0[keep]
    lm = np.log(mid)
    dlr = np.empty_like(lm)
    dlr[0] = 0.0
    dlr[1:] = np.diff(lm)                                     # Δlog mid per timestamp (signed)
    mv = dlr != 0.0                                           # a REAL mid-move (one per timestamp)
    return rx[mv], dlr[mv]


def _flow_sum_at(arrays, grid, mv_rx, val, span):
    """Decayed SUM of `val` over a venue's MOVE stream, decayed once per trade-timestamp on the
    shared clock, read AT each anchor with the partial epoch since the last trade folded in (the
    LIVE FRONT -> fresh, not stale). Verbatim port of the notebook §2 `_flow_sum_at`."""
    merged_ts = grid.merged_ts
    anchors = grid.anchor_ts
    n_ticks = len(merged_ts)
    a = 2.0 / (span + 1.0)
    k = np.searchsorted(merged_ts, mv_rx, "left")             # trades strictly before each move
    ep = np.bincount(k, weights=val, minlength=n_ticks + 1)   # per-trade-epoch sums
    x = np.zeros(n_ticks + 1)
    x[1:] = a * (1.0 - a) * ep[:-1]
    com = lfilter([1.0], [1.0, -(1.0 - a)], x)                # committed sum just after each trade
    ta = np.searchsorted(merged_ts, anchors, "right") - 1     # last trade <= anchor (== grid.tick_at_anchor)
    cs = np.concatenate([[0.0], np.cumsum(val)])              # prefix sums over the move stream
    partial = cs[np.searchsorted(mv_rx, anchors, "right")] - cs[np.searchsorted(mv_rx, merged_ts[ta], "right")]
    return com[ta + 1] + a * partial


def signed_ret(arrays, grid, ex, span):
    """signed_ret_N(ex) = EMA(Δlog mid, N) — the venue's smoothed signed drift, read live-front
    at every anchor (committed + α·moves since the last trade)."""
    mv_rx, mv_d = _move_stream(arrays, ex)
    return _flow_sum_at(arrays, grid, mv_rx, mv_d, span)


def _normalise_signed(arrays, grid):
    """The §5 RAW-FIRST scale gate, MEASURED on this block: max/min of the per-vol-decile std of
    byb's signed_ret(50). Returns True iff it FAILS (> 3×) -> ship /σ_ev; else ship RAW."""
    vol_level = np.log(grid.sigma_ev)
    finite = np.isfinite(vol_level)
    edges = np.nanpercentile(vol_level[finite], np.arange(10, 100, 10))
    vol_decile = np.digitize(vol_level, edges)
    raw = signed_ret(arrays, grid, "byb", SCALE_SPAN)
    bands = [np.nanstd(raw[vol_decile == d]) for d in range(10)]
    bands = [v for v in bands if np.isfinite(v) and v > 0]
    raw_scale = max(bands) / min(bands)
    return raw_scale > SCALE_GATE


def signed_momentum(arrays, grid, ex, span, normalise):
    """The SHIPPED form-1: signed drift, RAW or /σ_ev per the gate. SIGNED (never |·|)."""
    sret = signed_ret(arrays, grid, ex, span)
    if normalise:
        return sret / np.maximum(grid.sigma_ev, 1e-12)
    return sret


def best_spans(arrays, grid, normalise=None):
    """The §6 pick: per venue, the IN-SAMPLE best span = argmax |Spearman(signed_momentum, price_target)|
    over SPANS (price head — momentum's home; |IC| picks continuation OR reversion). Returns {leg: span}."""
    if normalise is None:
        normalise = _normalise_signed(arrays, grid)
    target = grid.price_target
    out = {}
    for ex in EXCHANGES:
        scores = []
        for N in SPANS:
            f = signed_momentum(arrays, grid, ex, N, normalise)
            v = np.isfinite(f) & np.isfinite(target)
            scores.append(spearmanr(f[v], target[v]).statistic if v.sum() > 100 else np.nan)
        out[ex] = SPANS[int(np.nanargmax(np.abs(scores)))]
    return out


def compute(arrays, grid, spans=None):
    """Module contract: return {leg: signed_momentum_on_grid} for price_momentum.

    spans=None  -> per-venue IN-SAMPLE best span (notebook §6, reproduces the headline).
    spans={leg:N} -> force that fixed span per leg (the harness FIXes block[0]'s pick for OOS).

    Each array is length len(grid.anchor_ts), index-aligned, SIGNED, NaN-where-undefined, strictly
    causal (built on grid.merged_ts, live-front read at each anchor). Normalisation (RAW vs /σ_ev) is
    the §5 gate's decision measured on this block — the SAME form the notebook ships."""
    normalise = _normalise_signed(arrays, grid)
    if spans is None:
        spans = best_spans(arrays, grid, normalise=normalise)
    return {ex: signed_momentum(arrays, grid, ex, spans[ex], normalise) for ex in EXCHANGES}
