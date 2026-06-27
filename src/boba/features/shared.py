"""Construction of `shared_data` from `raw_data` -- the IMPLEMENTATION behind the standalone feature
inputs whose TYPES (`RawData` / `SharedData` / `Config` / `Series` / …) are the contracts in
`boba.features.base`. Kept in its own module so the data-model contracts and the precompute that fills
them stay fully separated.

  * `build_shared_data(raw, config)` -- the single precompute every vectorized feature shares.
  * `flow_at(...)`                    -- the shared sparse-flow EMA primitive features build on
                                        (σ_ev, OFI, trade flow, …).

Standalone: no eval grid, no prediction target, no sampling -- and nothing imported from
`boba.research`. The feature layer stands alone on top of this.
"""
from __future__ import annotations

import numpy as np

from boba.features.base import Config, ListingRaw, ListingShared, RawData, Series, SharedData


def _ffill(rx: np.ndarray, val: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Causal forward-fill: `val` of the last `rx <= t[i]`; NaN before the first `rx` (no wrap)."""
    idx = np.searchsorted(rx, t, "right") - 1
    return np.where(idx < 0, np.nan, val[np.clip(idx, 0, len(val) - 1)])


def hold_last(vec: np.ndarray) -> np.ndarray:
    """Carry the last valid (finite) value forward through any MID-STREAM NaN, so a model never gets a
    NaN once the feature has warmed up. Only the LEADING warm-up NaN (before the first finite value)
    remains. A flow EMA at `span = 1` (α = 1) momentarily reads 0/0 = NaN at a bare trade-tick after
    warm-up; that is held over here (the freshest real reading persists) — NaN is not a valid model input."""
    v = np.asarray(vec, dtype=float)
    finite = np.isfinite(v)
    if not finite.any():
        return v
    idx = np.where(finite, np.arange(len(v)), -1)
    idx = np.maximum.accumulate(idx)                          # last finite index at-or-before each point
    return np.where(idx >= 0, v[np.clip(idx, 0, len(v) - 1)], np.nan)


def flow_at(clock: np.ndarray, src_rx: np.ndarray, val: np.ndarray, out_ts: np.ndarray, span: float) -> np.ndarray:
    """EWMA of a sparse `val` stamped at `src_rx`, decayed once per `clock` tick, READ at each `out_ts`
    (the committed-per-tick EMA + the partial epoch of injections since the last tick). The shared
    sparse-flow primitive every feature builds on (σ_ev, OFI, trade flow, …) -- one implementation, so the
    gate path and the features can't drift. `val` is aligned to `src_rx`; both decay on the shared `clock`."""
    from scipy.signal import lfilter

    a = 2.0 / (span + 1.0)
    n = len(clock)
    k = np.searchsorted(clock, src_rx, "left")
    ep = np.bincount(k, weights=val, minlength=n + 1)
    x = np.zeros(n + 1)
    x[1:] = a * (1.0 - a) * ep[:-1]
    com = lfilter([1.0], [1.0, -(1.0 - a)], x)
    ta = np.searchsorted(clock, out_ts, "right") - 1
    cs = np.concatenate([[0.0], np.cumsum(val)])
    last_tick = clock[np.clip(ta, 0, n - 1)]                  # ta<0 (pre-first-tick, warm-up) clamps; dropped downstream
    partial = cs[np.searchsorted(src_rx, out_ts, "right")] - cs[np.searchsorted(src_rx, last_tick, "right")]
    return com[ta + 1] + a * partial


def _listing_mid(lr: ListingRaw, policy: str) -> Series:
    """The `(rx, (bid+ask)/2)` mid `Series` from the listing's mid-policy stream -- `merged_levels`
    (trade-fused) where the venue carries it, else raw `front_levels`. The single derived 'mid' every
    mid-based feature reads."""
    src = lr.merged_levels if (policy == "merged_levels" and lr.merged_levels is not None) else lr.front_levels
    return Series(np.asarray(src.rx), (np.asarray(src.bid) + np.asarray(src.ask)) / 2.0)


def _yardsticks(target_mid: Series, clock: np.ndarray, out_ts: np.ndarray, span: int) -> tuple[np.ndarray, np.ndarray]:
    """σ_ev (RMS target mid-move per move) and λ_ev (target moves per second) at each `out_ts`, both
    decayed on `clock`. σ_ev = sqrt(E[move²]/E[1]); λ_ev = E[1]/E[Δt]. A 'move' is a target mid CHANGE."""
    from scipy.signal import lfilter

    rx, mid = target_mid
    keep = np.concatenate([rx[1:] != rx[:-1], [True]]) if len(rx) else np.zeros(0, bool)
    t_rx, t_mid = rx[keep], mid[keep]                         # same-rx mids collapse to the last
    lm = np.log(t_mid)
    dlr = np.empty_like(lm)
    if len(dlr):
        dlr[0] = 0.0
        dlr[1:] = np.diff(lm)
    mv = dlr != 0.0
    mv_rx, mv_r2 = t_rx[mv], dlr[mv] ** 2
    a = 2.0 / (span + 1.0)
    clock_dt = np.zeros(len(clock))
    clock_dt[1:] = np.diff(clock) / 1e9
    e_sq = flow_at(clock, mv_rx, mv_r2, out_ts, span)
    e_mv = flow_at(clock, mv_rx, np.ones(mv_r2.size), out_ts, span)
    e_dt = lfilter([a], [1.0, -(1.0 - a)], clock_dt)[np.searchsorted(clock, out_ts, "right") - 1]
    # Warm-up = NaN, NOT a floored huge/zero value: σ_ev is undefined until a move is seen (e_mv > 0) and
    # λ_ev until an inter-tick gap is seen (e_dt > 0). This is exactly what the streaming VolYardstick /
    # RateYardstick return there, so the two builds AGREE on the warm-up region instead of silently diverging.
    sig = np.where(e_mv > 0.0, np.sqrt(e_sq / np.where(e_mv > 0.0, e_mv, 1.0)), np.nan)
    lam = np.where(e_dt > 0.0, e_mv / np.where(e_dt > 0.0, e_dt, 1.0), np.nan)
    return sig, lam


def _union(arrays: list[np.ndarray]) -> np.ndarray:
    """Sorted-unique int64 union of timestamp arrays (empty-safe)."""
    nonempty = [np.asarray(a, np.int64) for a in arrays if len(a)]
    return np.unique(np.concatenate(nonempty)) if nonempty else np.empty(0, np.int64)


def build_shared_data(raw: RawData, config: Config) -> SharedData:
    """Construct `shared_data` from `raw_data` -- the single precompute every vectorized feature shares.
    Standalone: no research / eval-grid / prediction-target logic. It builds

      * the DECAY clock  -- the sorted-unique union of every listing's trade timestamps (EMAs tick here);
      * the EVENT grid   -- every timestamp that carried ANY event (book or trade) on any listing: the
                            feature OUTPUT index;
      * the σ_ev / λ_ev yardsticks from the TARGET listing's mid-moves, read at every `event_ts`;
      * each listing's derived mid `(rx, (bid+ask)/2)` per its mid policy.
    """
    listings = config.all_listings
    clock = _union([raw.listings[l].trade.rx for l in listings])

    mids: dict[str, Series] = {}
    grids = [clock]
    for l in listings:
        lr = raw.listings[l]
        grids.append(np.asarray(lr.front_levels.rx))
        if lr.merged_levels is not None:
            grids.append(np.asarray(lr.merged_levels.rx))
        grids.append(np.asarray(lr.trade.rx))
        mids[l] = _listing_mid(lr, config.mid_stream.get(l, "front_levels"))
    event_ts = _union(grids)

    vol, rate = _yardsticks(mids[config.target_listing], clock, event_ts, config.yardstick_span)
    return SharedData(
        event_ts=event_ts, clock=clock, vol_yardstick=vol, rate_yardstick=rate,
        listings={l: ListingShared(mid=mids[l]) for l in listings},
    )
