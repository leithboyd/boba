"""`price_dislocation` — the worked-example feature, as a registered `FeatureSpec`.

WHAT it measures. How far another source's price has drifted from the target's: the log-price gap,
smoothed two ways (a fast and a slow `LiveFrontEMA` leg) and divided by the volatility yardstick
`σ_ev`. One leg per foreign source (a fan-out feature). `params = (n_fast, n_slow)`.

WHY it might predict. Venues discover price at different speeds, so a faster source's mid leads byb:
a fresh smoothed cross-venue gap is the foreign price having moved before byb has caught up, and it
should predict byb's catch-up *toward* that foreign price (a signed lead-lag edge). FALSIFIED if the
gap carries no forward signed IC on byb's mid — i.e. byb does not catch up and the gap is just noise.

RESEARCH. Hasbrouck, J. (1995) 'One Security, Many Markets', Journal of Finance 50(4):1175-1199;
Hayashi, T. & Yoshida, N. (2005) 'On covariance estimation of non-synchronously observed diffusion
processes', Bernoulli 11(2):359-379 (cross-venue lead-lag).

Two implementations of the same maths, tied by `boba.research.screening.parity_check`:
  - `vectorized(raw, shared, config, params)` -> {source -> feature vector per `event_ts`}  (offline; lfilter)
  - `LiveDislocation`                         -> the O(1) streaming build (composes `VolYardstick` for σ_ev)

Mirror augmentation: the feature is ODD under the reflection of the tape through byb's mid, so `SPEC.mirror`
is `np.negative` (see the SPEC comment below and `AUTHORING.md` → Mirror augmentation).

See `AUTHORING.md` (this directory) for the EMA-type and inject/decay rules these obey.
"""
from __future__ import annotations

import math

import numpy as np

from boba.ema import LiveFrontEMA
from boba.features.base import Config, FeatureSpec, ParamKind, Params, RawData, SharedData
from boba.features.base import register
from boba.features.shared import _ffill
from boba.features.streaming import LiveMergedBook, VolYardstick


def _ex(listing: str) -> str:
    """The short exchange key (leg key) for a full listing id, e.g. 'bin_eth_usdt_p' -> 'bin'."""
    return listing.split("_", 1)[0]


def _ema(gap: np.ndarray, span: int) -> np.ndarray:
    """Offline trade-clock EMA of `gap` (committed per tick). `span == 1` collapses to the gap itself."""
    if span == 1:
        return gap
    from scipy.signal import lfilter

    a = 2.0 / (span + 1.0)
    return lfilter([a], [1.0, -(1.0 - a)], gap)


def _dislocation_leg(src_mid, target_logmid_clock: np.ndarray, target_logmid_event: np.ndarray,
                     clock: np.ndarray, event_ts: np.ndarray, tick_at_event: np.ndarray,
                     n_fast: int, n_slow: int, sigma: np.ndarray) -> np.ndarray:
    """One source's `(fast − slow live-front EMA of the log gap) / σ_ev` at every `event_ts` (causal).

    The committed leg is the trade-clock EMA of the gap `g = log(mid_src) − log(mid_target)` read at each
    event-grid timestamp's tick; the live front adds the freshest gap AT the event timestamp:
    `(1−α)·committed[tick] + α·g_fresh`. `g_fresh` is NaN where either side has not quoted -> the leg is
    NaN there. The committed EMA uses the gap forward-filled on the trade clock (a missing mid contributes
    a 0 gap, matching the old `nan_to_num`); the σ_ev divide carries the regime scale out."""
    af, as_ = 2.0 / (n_fast + 1.0), 2.0 / (n_slow + 1.0)
    g_committed = np.nan_to_num(np.log(_ffill(src_mid.rx, src_mid.value, clock)) - target_logmid_clock, nan=0.0)
    g_fresh = np.log(_ffill(src_mid.rx, src_mid.value, event_ts)) - target_logmid_event
    fast = (1.0 - af) * _ema(g_committed, n_fast)[tick_at_event] + af * g_fresh
    slow = (1.0 - as_) * _ema(g_committed, n_slow)[tick_at_event] + as_ * g_fresh
    return (fast - slow) / sigma


def vectorized(raw: RawData, shared: SharedData, config: Config, params: Params) -> dict[str, np.ndarray]:
    """{source -> (fast − slow live-front EMA of the log gap) / σ_ev}, one value per `event_ts` (causal)."""
    n_fast, n_slow = params
    target_mid = shared.listings[config.target_listing].mid
    target_logmid_clock = np.log(_ffill(target_mid.rx, target_mid.value, shared.clock))
    target_logmid_event = np.log(_ffill(target_mid.rx, target_mid.value, shared.event_ts))
    tick_at_event = np.searchsorted(shared.clock, shared.event_ts, "right") - 1
    sigma = np.where(shared.vol_yardstick == 0.0, np.nan, shared.vol_yardstick)   # σ_ev=0 only in warm-up -> clean NaN, not inf
    return {_ex(l): _dislocation_leg(shared.listings[l].mid, target_logmid_clock, target_logmid_event,
                                     shared.clock, shared.event_ts, tick_at_event, n_fast, n_slow, sigma)
            for l in config.other_listings}


class LiveDislocation:
    """O(1) streaming build. σ_ev via the shared `VolYardstick`; each gap leg a `LiveFrontEMA`
    (live-front level read). Driver applies events then calls `refresh()` once per timestamp."""

    def __init__(self, config: Config, params: Params):
        n_fast, n_slow = params
        self.target = config.target_listing
        self.others = list(config.other_listings)
        self.keys = tuple(_ex(o) for o in self.others)
        self._key_of = {o: _ex(o) for o in self.others}
        fuse_tick = {l: config.tick_size[l] for l in config.all_listings
                     if config.mid_stream.get(l) == "merged_levels"}     # KeyError -> no tick for a fused listing
        self.book = LiveMergedBook(fuse_tick)        # shared merged-book reconstruction (fuse + un-cross); read .quote()
        self.yard = VolYardstick(config.yardstick_span)
        self.leg_f = {o: LiveFrontEMA(n_fast) for o in self.others}
        self.leg_s = {o: LiveFrontEMA(n_slow) for o in self.others}
        self.was_trade_present = False

    def on_book(self, ev) -> None:
        self.book.on_book(ev)

    def on_trade(self, ev) -> None:
        self.book.on_trade(ev)
        self.was_trade_present = True

    def refresh(self) -> None:
        traded, self.was_trade_present = self.was_trade_present, False
        q = self.book.quote(self.target)
        if q is None:
            return
        lt = math.log(0.5 * (q[0] + q[1]))          # mid from the un-crossed (bid, ask)
        self.yard.on_target_logmid(lt)              # injects (Δlog)^2 iff the target moved
        for o in self.others:
            qo = self.book.quote(o)
            if qo is not None:
                g = math.log(0.5 * (qo[0] + qo[1])) - lt
                self.leg_f[o].add(g); self.leg_s[o].add(g)   # refresh the live front
        if traded:                                  # advance the clock once: decay σ_ev, commit each leg
            self.yard.tick()
            for o in self.others:
                self.leg_f[o].tick(); self.leg_s[o].tick()

    def value(self) -> dict[str, float]:
        sig = self.yard.sigma()
        return {self._key_of[o]: (self.leg_f[o].value() - self.leg_s[o].value()) / sig for o in self.others}


SPEC = FeatureSpec(
    name="price_dislocation",
    vectorized=vectorized,
    make_streaming=lambda config, params: LiveDislocation(config, params),
    keys_for=lambda config, params: tuple(_ex(l) for l in config.other_listings),
    param_kind=ParamKind.FAST_SLOW,                          # params = (n_fast, n_slow)
    # Mirror augmentation: reflecting the tape through byb's mid negates this feature. The legs are linear
    # in the log gap and the gap is a price-DIFFERENCE (the reflection level cancels), so gap -> -gap and
    # each leg -> -leg; σ_ev is built from squared byb moves (even), so it is unchanged. Hence the feature
    # is ODD -> np.negative. (The signed target negates too; the engines handle that.) See AUTHORING.md.
    mirror=np.negative,
)
register(SPEC)
