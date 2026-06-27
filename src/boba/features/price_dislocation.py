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
  - `vectorized(ctx, params)` -> {source -> feature vector on the grid}   (offline, may use lfilter)
  - `LiveDislocation`         -> the O(1) streaming build (composes `LiveYardstick` for σ_ev)

Mirror augmentation: the feature is ODD under the reflection of the tape through byb's mid, so `SPEC.mirror`
is `np.negative` (see the SPEC comment below and `AUTHORING.md` → Mirror augmentation).

See `AUTHORING.md` (this directory) for the EMA-type and inject/decay rules these obey.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from boba.ema import LiveFrontEMA
from boba.features.base import FeatureSpec, ParamKind, Params, register
from boba.research.screening import LiveYardstick, ScreeningContext


def _ema(gap: np.ndarray, span: int) -> np.ndarray:
    """Offline trade-clock EMA of `gap` (committed per tick). `span == 1` collapses to the gap itself."""
    if span == 1:
        return gap
    from scipy.signal import lfilter

    a = 2.0 / (span + 1.0)
    return lfilter([a], [1.0, -(1.0 - a)], gap)


def vectorized(ctx: ScreeningContext, params: Params) -> dict[str, np.ndarray]:
    """{source -> (fast − slow live-front EMA of the log gap) / σ_ev}, on the anchor grid (causal)."""
    n_fast, n_slow = params
    af, as_ = 2.0 / (n_fast + 1.0), 2.0 / (n_slow + 1.0)
    target_ex = ctx.target.split("_", 1)[0]
    log_mid_target_clock = ctx.target_logmid_on_clock
    g_fresh_target = np.log(ctx.mid_at_anchor(target_ex))
    out: dict[str, np.ndarray] = {}
    for ex in ctx.sources:
        g_committed = np.nan_to_num(np.log(ctx.mid_on_clock(ex)) - log_mid_target_clock, nan=0.0)
        g_fresh = np.log(ctx.mid_at_anchor(ex)) - g_fresh_target
        fast = (1.0 - af) * _ema(g_committed, n_fast)[ctx.tick_at_anchor] + af * g_fresh
        slow = (1.0 - as_) * _ema(g_committed, n_slow)[ctx.tick_at_anchor] + as_ * g_fresh
        out[ex] = (fast - slow) / ctx.sigma_at_anchor
    return out


class LiveDislocation:
    """O(1) streaming build. σ_ev via the shared `LiveYardstick`; each gap leg a `LiveFrontEMA`
    (live-front level read). Driver applies events then calls `refresh()` once per timestamp."""

    def __init__(self, ctx: ScreeningContext, params: Params):
        n_fast, n_slow = params
        target_ex = ctx.target.split("_", 1)[0]
        self.target = ctx.target
        self.others = [f"{s}_{ctx.coin}" for s in ctx.sources]
        self.keys = tuple(ctx.sources)
        self._key_of = {f"{s}_{ctx.coin}": s for s in ctx.sources}
        self.fuse_trades = frozenset(
            f"{ex}_{ctx.coin}" for ex in (target_ex,) + tuple(ctx.sources)
            if ctx.mid_stream[ex] == "merged_levels")
        self.bid: dict = {}; self.bid_t: dict = {}; self.ask: dict = {}; self.ask_t: dict = {}
        self.yard = LiveYardstick(ctx.yardstick_span)
        self.leg_f = {o: LiveFrontEMA(n_fast) for o in self.others}
        self.leg_s = {o: LiveFrontEMA(n_slow) for o in self.others}
        self.was_trade_present = False

    def _side(self, listing: str, is_ask, px: float, t: int) -> None:
        held_t = self.ask_t if is_ask else self.bid_t
        if t > held_t.get(listing, -1):
            (self.ask if is_ask else self.bid)[listing] = px
            held_t[listing] = t

    def _mid(self, listing: str) -> Optional[float]:
        b, a = self.bid.get(listing), self.ask.get(listing)
        return None if b is None or a is None else 0.5 * (b + a)

    def on_book(self, ev) -> None:                  # uses only prices; sizes (ev.bid_qty/ask_qty) ignored
        listing = ev.listing
        if listing in self.fuse_trades:
            self._side(listing, False, ev.bid, ev.exch_time); self._side(listing, True, ev.ask, ev.exch_time)
        else:
            self.bid[listing] = ev.bid; self.ask[listing] = ev.ask

    def on_trade(self, ev) -> None:
        if ev.listing in self.fuse_trades:
            self._side(ev.listing, ev.lifts_ask, ev.px, ev.exch_time)
        self.was_trade_present = True

    def refresh(self) -> None:
        traded, self.was_trade_present = self.was_trade_present, False
        tgt = self._mid(self.target)
        if tgt is None:
            return
        lt = math.log(tgt)
        self.yard.on_target_logmid(lt)              # injects (Δlog)^2 iff the target moved
        for o in self.others:
            m = self._mid(o)
            if m is not None:
                g = math.log(m) - lt
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
    make_streaming=lambda ctx, params: LiveDislocation(ctx, params),
    keys_for=lambda ctx, params: tuple(ctx.sources),
    param_kind=ParamKind.FAST_SLOW,                          # params = (n_fast, n_slow)
    # Mirror augmentation: reflecting the tape through byb's mid negates this feature. The legs are linear
    # in the log gap and the gap is a price-DIFFERENCE (the reflection level cancels), so gap -> -gap and
    # each leg -> -leg; σ_ev is built from squared byb moves (even), so it is unchanged. Hence the feature
    # is ODD -> np.negative. (The signed target negates too; the engines handle that.) See AUTHORING.md.
    mirror=np.negative,
)
register(SPEC)
