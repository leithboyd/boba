"""`rate_momentum` — log-ratio of a FAST to a SLOW rate yardstick (per-venue trade-rate acceleration).

WHAT it measures. For EACH venue, that venue's own `λ_ev` (mid-moves per second) at a fast span over `λ_ev`
at a slow span, in logs: `log(λ_ev_fast / λ_ev_slow)`. It **fans out over every listing** (`config.all_listings`
— the target plus each source), one leg per venue built from that venue's OWN mid; `params = (n_fast, n_slow)`.
Positive = short-horizon activity running ABOVE its slower baseline (the tape speeding up); negative = quieting.
(To restrict to the target alone, pass a `config` whose `other_listings` is empty.)

WHY it might predict (falsifiable). Trade intensity clusters like volatility (a self-exciting / Hawkes
arrival process), and a fast-vs-slow move-rate ratio is its acceleration coordinate. The hypothesis is that a
venue's rate momentum (especially a faster source's, ahead of byb) leads byb's near-term move COUNT — the
natural target for the rate head — rather than the signed direction. FALSIFIED if no leg carries forward
information about byb's move rate beyond the slow level itself.

RESEARCH. Hawkes, A.G. (1971) 'Spectra of Some Self-Exciting and Mutually Exciting Point Processes',
Biometrika 58(1):83-90 (self-exciting arrivals); Engle, R. & Russell, J. (1998) 'Autoregressive Conditional
Duration', Econometrica 66(5):1127-1162 (clustered trade timing).

Two implementations of the same maths, tied by `boba.research.screening.parity_check`:
  - `vectorized(raw, shared, config, (n_fast, n_slow))` -> {venue -> log(λ_ev_fast/λ_ev_slow) per `event_ts`}
  - `LiveRateMomentum` -> two composed `RateYardstick`s (fast, slow) per venue; `value = log(fast.lam()/slow.lam())`

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
    """{venue -> log(λ_ev_fast / λ_ev_slow)} from each venue's OWN mid-moves, one value per `event_ts`
    (causal); NaN until both yardsticks warm up. Fans out over `config.all_listings`."""
    n_fast, n_slow = params
    out: dict[str, np.ndarray] = {}
    for l in config.all_listings:
        mid = shared.listings[l].mid
        _, lam_fast = _yardsticks(mid, shared.clock, shared.event_ts, n_fast)
        _, lam_slow = _yardsticks(mid, shared.clock, shared.event_ts, n_slow)
        out[_ex(l)] = _log_ratio(lam_fast, lam_slow)
    return out


class LiveRateMomentum:
    """O(1) streaming build, ONE leg per venue (fans out over `config.all_listings`). Each venue carries two
    `RateYardstick`s (fast, slow) over its OWN mid; each `tick` needs the trade's receive-time for Δt, so
    `on_trade` captures `ev.rx`. `refresh()` feeds each venue's log-mid to its two yardsticks, then decays
    all of them once with that `rx` iff a trade landed."""

    def __init__(self, config: Config, params: Params):
        n_fast, n_slow = params
        self.listings = list(config.all_listings)
        self.keys = tuple(_ex(l) for l in self.listings)
        self._key_of = {l: _ex(l) for l in self.listings}
        fuse_tick = {l: config.tick_size[l] for l in config.all_listings
                     if config.mid_stream.get(l) == "merged_levels"}     # KeyError -> no tick for a fused listing
        self.book = LiveMergedBook(fuse_tick)        # shared merged-book reconstruction (fuse + un-cross); read .quote()
        self.fast = {l: RateYardstick(n_fast) for l in self.listings}
        self.slow = {l: RateYardstick(n_slow) for l in self.listings}
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
        for l in self.listings:
            q = self.book.quote(l)
            lt = None
            if q is not None:
                mid = 0.5 * (q[0] + q[1])
                if mid > 0.0:
                    lt = math.log(mid)
            self.fast[l].on_target_logmid(lt)        # injects one move into each iff this venue's mid moved
            self.slow[l].on_target_logmid(lt)
        if traded:
            for l in self.listings:
                self.fast[l].tick(self._rx)          # the trade tick's rx -> Δt for λ_ev
                self.slow[l].tick(self._rx)

    def value(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for l in self.listings:
            lf, ls = self.fast[l].lam(), self.slow[l].lam()
            ok = lf > 0.0 and ls > 0.0 and math.isfinite(lf) and math.isfinite(ls)
            out[self._key_of[l]] = math.log(lf / ls) if ok else float("nan")
        return out


SPEC = FeatureSpec(
    name="rate_momentum",
    vectorized=vectorized,
    make_streaming=lambda config, params: LiveRateMomentum(config, params),
    keys_for=lambda config, params: tuple(_ex(l) for l in config.all_listings),
    mirror=_identity,   # λ_ev counts mid CHANGES (sign-free) -> even; a ratio of two even quantities is EVEN
    param_kind=ParamKind.FAST_SLOW,                          # params = (n_fast, n_slow)
)
register(SPEC)
