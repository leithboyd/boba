"""`rate_momentum` — log-ratio of a FAST to a SLOW rate yardstick (the target's trade-rate acceleration).

WHAT it measures. The target's `λ_ev` (mid-moves per second) at a fast span over `λ_ev` at a slow span, in
logs: `log(λ_ev_fast / λ_ev_slow)`. A single target-based leg. `params = (n_fast, n_slow)`. Positive =
short-horizon activity running ABOVE its slower baseline (the tape speeding up); negative = quieting down.

WHY it might predict (falsifiable). Trade intensity clusters like volatility (a self-exciting / Hawkes
arrival process), and a fast-vs-slow move-rate ratio is its acceleration coordinate. The hypothesis is that
rate momentum leads byb's near-term move COUNT — the natural target for the rate head — rather than the
signed direction. FALSIFIED if the ratio carries no forward information about byb's move rate beyond the
slow level itself.

RESEARCH. Hawkes, A.G. (1971) 'Spectra of Some Self-Exciting and Mutually Exciting Point Processes',
Biometrika 58(1):83-90 (self-exciting arrivals); Engle, R. & Russell, J. (1998) 'Autoregressive Conditional
Duration', Econometrica 66(5):1127-1162 (clustered trade timing).

Two implementations of the same maths, tied by `boba.research.screening.parity_check`:
  - `vectorized(raw, shared, config, (n_fast, n_slow))` -> {target -> log(λ_ev_fast/λ_ev_slow) per `event_ts`}
  - `LiveRateMomentum` -> two composed `RateYardstick`s (fast, slow); `value = log(fast.lam()/slow.lam())`

Mirror augmentation: `λ_ev` counts mid CHANGES (a move is a move regardless of sign), so it is EVEN under
the reflection of the tape through byb's mid; a ratio of two even quantities is even -> the feature is EVEN
-> `SPEC.mirror` is the identity.

See `AUTHORING.md` (this directory) for the EMA-type and inject/decay rules these obey.
"""
from __future__ import annotations

import math

import numpy as np

from boba.features.base import Config, FeatureSpec, ParamKind, Params, RawData, SharedData, register
from boba.features.shared import _yardsticks
from boba.features.streaming import LiveMergedBook, RateYardstick


def _ex(listing: str) -> str:
    """The short exchange key (leg key) for a full listing id, e.g. 'byb_eth_usdt_p' -> 'byb'."""
    return listing.split("_", 1)[0]


def _identity(v: np.ndarray) -> np.ndarray:
    """Mirror for an EVEN feature — the reflection of the tape through byb's mid leaves it unchanged."""
    return v


def _log_ratio(fast: np.ndarray, slow: np.ndarray) -> np.ndarray:
    """`log(fast / slow)`, NaN where either side is undefined (warm-up) or non-positive (a quiet window where
    λ_ev = 0) — so the model never sees an `inf` from a `log(0)` (held downstream by `hold_last`)."""
    ok = np.isfinite(fast) & np.isfinite(slow) & (fast > 0.0) & (slow > 0.0)
    return np.where(ok, np.log(np.where(ok, fast, 1.0) / np.where(ok, slow, 1.0)), np.nan)


def vectorized(raw: RawData, shared: SharedData, config: Config, params: Params) -> dict[str, np.ndarray]:
    """{target -> log(λ_ev_fast / λ_ev_slow)} at every `event_ts` (causal); NaN until both yardsticks warm up."""
    n_fast, n_slow = params
    target_mid = shared.listings[config.target_listing].mid
    _, lam_fast = _yardsticks(target_mid, shared.clock, shared.event_ts, n_fast)
    _, lam_slow = _yardsticks(target_mid, shared.clock, shared.event_ts, n_slow)
    return {_ex(config.target_listing): _log_ratio(lam_fast, lam_slow)}


class LiveRateMomentum:
    """O(1) streaming build: two composed `RateYardstick`s (fast, slow) on the TARGET mid. Each `tick`
    needs the trade's receive-time for Δt, so `on_trade` captures `ev.rx`. `refresh()` feeds the target
    log-mid to both yardsticks, then decays both once with that `rx` iff a trade landed. One leg."""

    def __init__(self, config: Config, params: Params):
        n_fast, n_slow = params
        self.target = config.target_listing
        self.key = _ex(self.target)
        self.keys = (self.key,)
        fuse_tick = {l: config.tick_size[l] for l in config.all_listings
                     if config.mid_stream.get(l) == "merged_levels"}     # KeyError -> no tick for a fused listing
        self.book = LiveMergedBook(fuse_tick)        # shared merged-book reconstruction (fuse + un-cross); read .quote()
        self.fast = RateYardstick(n_fast)
        self.slow = RateYardstick(n_slow)
        self.was_trade_present = False
        self._rx = 0

    def on_book(self, ev) -> None:
        self.book.on_book(ev)
        self._rx = ev.rx

    def on_trade(self, ev) -> None:
        self.book.on_trade(ev)
        self._rx = ev.rx
        self.was_trade_present = True

    def refresh(self) -> None:
        traded, self.was_trade_present = self.was_trade_present, False
        q = self.book.quote(self.target)
        lt = None
        if q is not None:
            mid = 0.5 * (q[0] + q[1])
            if mid > 0.0:
                lt = math.log(mid)
        self.fast.on_target_logmid(lt)               # injects one move into each iff the target moved
        self.slow.on_target_logmid(lt)
        if traded:
            self.fast.tick(self._rx)                 # the trade tick's rx -> Δt for λ_ev
            self.slow.tick(self._rx)

    def value(self) -> dict[str, float]:
        lf, ls = self.fast.lam(), self.slow.lam()
        ok = lf > 0.0 and ls > 0.0 and math.isfinite(lf) and math.isfinite(ls)
        return {self.key: math.log(lf / ls) if ok else float("nan")}


SPEC = FeatureSpec(
    name="rate_momentum",
    vectorized=vectorized,
    make_streaming=lambda config, params: LiveRateMomentum(config, params),
    keys_for=lambda config, params: (_ex(config.target_listing),),
    mirror=_identity,   # λ_ev counts mid CHANGES (sign-free) -> even; a ratio of two even quantities is EVEN
    param_kind=ParamKind.FAST_SLOW,                          # params = (n_fast, n_slow)
)
register(SPEC)
