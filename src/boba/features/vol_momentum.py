"""`vol_momentum` — log-ratio of a FAST to a SLOW volatility yardstick (per-venue vol acceleration).

WHAT it measures. For EACH venue, that venue's own `σ_ev` (RMS mid-move per move) at a fast span over `σ_ev`
at a slow span, in logs: `log(σ_ev_fast / σ_ev_slow)`. It **fans out over every listing** (`config.all_listings`
— the target plus each source), one leg per venue built from that venue's OWN mid; `params = (n_fast, n_slow)`.
Positive = short-horizon vol running ABOVE its slower baseline (vol picking up); negative = vol cooling off.
(To restrict to the target alone, pass a `config` whose `other_listings` is empty.)

WHY it might predict (falsifiable). Volatility clusters, and the clustering builds and decays — a fast-vs-slow
σ_ev ratio is the canonical vol-of-vol / vol-acceleration coordinate. The hypothesis is that a venue's vol
momentum (especially a faster source's, ahead of byb) leads byb's near-term realised-vol regime, and through
it the SCALE of the next move (a |move| / squared-return target), not its direction. FALSIFIED if no leg
carries forward information about byb's realised vol beyond the slow level itself.

RESEARCH. Corsi, F. (2009) 'A Simple Approximate Long-Memory Model of Realized Volatility', Journal of
Financial Econometrics 7(2):174-196 (HAR-RV: a fast vs slow realised-vol decomposition); Engle, R. (1982)
'Autoregressive Conditional Heteroscedasticity', Econometrica 50(4):987-1007 (volatility clustering).

Two implementations of the same maths, tied by `boba.research.screening.parity_check`:
  - `vectorized(raw, shared, config, (n_fast, n_slow))` -> {venue -> log(σ_ev_fast/σ_ev_slow) per `event_ts`}
  - `LiveVolMomentum` -> two composed `VolYardstick`s (fast, slow) per venue; `value = log(fast.sigma()/slow.sigma())`

Mirror augmentation: `σ_ev` is built from SQUARED moves, so it is EVEN under the reflection of the tape
through byb's mid; a ratio of two even quantities is even -> the feature is EVEN -> `SPEC.mirror` is the
identity (the reflection leaves it unchanged).

See `AUTHORING.md` (this directory) for the EMA-type and inject/decay rules these obey.
"""
from __future__ import annotations

import math

import numpy as np

from boba.features.base import Config, FeatureSpec, ParamKind, Params, RawData, SharedData, register
from boba.features.shared import _yardsticks
from boba.features.streaming import LiveMergedBook, VolYardstick


def _ex(listing: str) -> str:
    """The short exchange key (leg key) for a full listing id, e.g. 'byb_eth_usdt_p' -> 'byb'."""
    return listing.split("_", 1)[0]


def _identity(v: np.ndarray) -> np.ndarray:
    """Mirror for an EVEN feature — the reflection of the tape through byb's mid leaves it unchanged."""
    return v


def _log_ratio(fast: np.ndarray, slow: np.ndarray) -> np.ndarray:
    """`log(fast / slow)`, NaN where either side is undefined (warm-up NaN) or non-positive — so the model
    never sees an `inf` from a `log(0)` / `x/0` (held downstream by `hold_last`)."""
    ok = np.isfinite(fast) & np.isfinite(slow) & (fast > 0.0) & (slow > 0.0)
    return np.where(ok, np.log(np.where(ok, fast, 1.0) / np.where(ok, slow, 1.0)), np.nan)


def vectorized(raw: RawData, shared: SharedData, config: Config, params: Params) -> dict[str, np.ndarray]:
    """{venue -> log(σ_ev_fast / σ_ev_slow)} from each venue's OWN mid-moves, one value per `event_ts`
    (causal); NaN until both yardsticks warm up. Fans out over `config.all_listings`."""
    n_fast, n_slow = params
    out: dict[str, np.ndarray] = {}
    for l in config.all_listings:
        mid = shared.listings[l].mid
        sig_fast, _ = _yardsticks(mid, shared.clock, shared.event_ts, n_fast)
        sig_slow, _ = _yardsticks(mid, shared.clock, shared.event_ts, n_slow)
        out[_ex(l)] = _log_ratio(sig_fast, sig_slow)
    return out


class LiveVolMomentum:
    """O(1) streaming build, ONE leg per venue (fans out over `config.all_listings`). Each venue carries two
    `VolYardstick`s (fast, slow) over its OWN mid; the mid follows that venue's mid policy via the shared
    `LiveMergedBook`. `refresh()` feeds each venue's log-mid to its two yardsticks, then decays all of them
    once iff a trade landed (shared clock)."""

    def __init__(self, config: Config, params: Params):
        n_fast, n_slow = params
        self.listings = list(config.all_listings)
        self.keys = tuple(_ex(l) for l in self.listings)
        self._key_of = {l: _ex(l) for l in self.listings}
        fuse_tick = {l: config.tick_size[l] for l in config.all_listings
                     if config.mid_stream.get(l) == "merged_levels"}     # KeyError -> no tick for a fused listing
        self.book = LiveMergedBook(fuse_tick)        # shared merged-book reconstruction (fuse + un-cross); read .quote()
        self.fast = {l: VolYardstick(n_fast) for l in self.listings}
        self.slow = {l: VolYardstick(n_slow) for l in self.listings}
        self.was_trade_present = False

    def on_book(self, ev) -> None:
        self.book.on_book(ev)

    def on_trade(self, ev) -> None:
        self.book.on_trade(ev)
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
            self.fast[l].on_target_logmid(lt)        # injects (Δlog)^2 into each iff this venue's mid moved
            self.slow[l].on_target_logmid(lt)
        if traded:
            for l in self.listings:
                self.fast[l].tick()
                self.slow[l].tick()

    def value(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for l in self.listings:
            sf, ss = self.fast[l].sigma(), self.slow[l].sigma()
            ok = sf > 0.0 and ss > 0.0 and math.isfinite(sf) and math.isfinite(ss)
            out[self._key_of[l]] = math.log(sf / ss) if ok else float("nan")
        return out


SPEC = FeatureSpec(
    name="vol_momentum",
    vectorized=vectorized,
    make_streaming=lambda config, params: LiveVolMomentum(config, params),
    keys_for=lambda config, params: tuple(_ex(l) for l in config.all_listings),
    mirror=_identity,   # σ_ev is built from SQUARED moves (even); a ratio of two even quantities is EVEN
    param_kind=ParamKind.FAST_SLOW,                          # params = (n_fast, n_slow)
)
register(SPEC)
