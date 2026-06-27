"""`price_momentum` -- per-venue EMA of log-mid moves in target volatility units.

WHAT it measures. For each venue, collapse same-timestamp mid rows to the final state, take non-zero
`Δlog(mid)` moves, and smooth them as a sparse flow with `KernelMeanEMA` on the shared trade clock.
The feature is:

    EMA(Δlog mid) / σ_ev

where `σ_ev` is the target volatility yardstick from `ScreeningContext` / `LiveYardstick`. It fans
out over every exchange -- the target plus each foreign source; `params = N` (the EMA span).

WHY it might predict (a falsifiable hypothesis). Recent log-mid moves tend to *continue* at short
microstructure horizons (momentum / trend), rather than immediately reverting. So a positive EMA of
a venue's recent moves -- read in `σ_ev` units, so the strength is measured against the target's own
volatility -- predicts byb's mid continuing *up* (and a negative EMA, down). The hypothesis is
falsified if there is no forward signed IC, or if the relationship is the wrong sign (the moves
mean-revert rather than persist).

Research. Moskowitz, T., Ooi, Y.H. & Pedersen, L.H. (2012) 'Time Series Momentum', Journal of
Financial Economics 104(2):228-250 -- an EWMA-of-returns trend signal that predicts the same
instrument's own future returns, the direct macro-horizon analogue of this microstructure EMA.

Mirror augmentation: log returns negate under price reflection while `σ_ev` is even, so the feature
is ODD and `SPEC.mirror` is `np.negative`.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from boba.ema import KernelMeanEMA
from boba.features.base import FeatureSpec, ParamKind, Params, register
from boba.research.screening import LiveYardstick, ScreeningContext


def _exchanges(ctx: ScreeningContext) -> tuple[str, ...]:
    return (ctx.target.split("_", 1)[0],) + tuple(ctx.sources)


def _move_stream(mid_stream: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """`(rx, mid)` level rows -> non-zero `(rx, Δlog(mid))`, one final mid per timestamp."""
    rx, mid = mid_stream
    ok = (mid > 0.0) & np.isfinite(mid)
    rx, mid = rx[ok], mid[ok]
    if len(rx) < 2:
        return rx[:0], mid[:0].astype(float)
    keep = np.concatenate([rx[1:] != rx[:-1], [True]])
    rx, mid = rx[keep], mid[keep]
    if len(rx) < 2:
        return rx[:0], mid[:0].astype(float)
    dlog = np.diff(np.log(mid))
    move = dlog != 0.0
    return rx[1:][move], dlog[move]


def _leg(ctx: ScreeningContext, rx: np.ndarray, ret: np.ndarray, span: int) -> np.ndarray:
    E = ctx._flow_at(ctx.anchor_ts, ret, span, src_rx=rx)
    W = ctx._flow_at(ctx.anchor_ts, np.ones(ret.size), span, src_rx=rx)
    return E / np.where(W == 0.0, np.nan, W)


def vectorized(ctx: ScreeningContext, params: Params) -> dict[str, np.ndarray]:
    """{exchange -> E/W EMA of that venue's log-mid moves divided by target σ_ev}."""
    n = params
    out: dict[str, np.ndarray] = {}
    for ex in _exchanges(ctx):
        rx, ret = _move_stream(ctx._mids[ex])
        out[ex] = _leg(ctx, rx, ret, n) / ctx.sigma_at_anchor
    return out


class LivePriceMomentum:
    """O(1) streaming build. Price state follows the context's mid policy; returns are sparse flows."""

    def __init__(self, ctx: ScreeningContext, params: Params):
        self.target = ctx.target
        self.exes = _exchanges(ctx)
        self.keys = self.exes
        self._key_of = {f"{ex}_{ctx.coin}": ex for ex in self.exes}
        self.fuse_trades = frozenset(
            f"{ex}_{ctx.coin}" for ex in self.exes if ctx.mid_stream.get(ex, "front_levels") == "merged_levels"
        )
        self.bid: dict[str, float] = {}
        self.ask: dict[str, float] = {}
        self.bid_t: dict[str, int] = {}
        self.ask_t: dict[str, int] = {}
        self.prev_log = {ex: None for ex in self.exes}
        self.leg = {ex: KernelMeanEMA(params) for ex in self.exes}
        self.yard = LiveYardstick(ctx.yardstick_span)
        self.was_trade_present = False

    def _side(self, listing: str, is_ask, px: float, t: int) -> None:
        held_t = self.ask_t if is_ask else self.bid_t
        if t > held_t.get(listing, -1):
            (self.ask if is_ask else self.bid)[listing] = px
            held_t[listing] = t

    def _mid(self, listing: str) -> Optional[float]:
        b, a = self.bid.get(listing), self.ask.get(listing)
        return None if b is None or a is None else 0.5 * (b + a)

    def on_book(self, ev) -> None:
        if ev.listing not in self._key_of:
            return
        if ev.listing in self.fuse_trades:
            self._side(ev.listing, False, ev.bid, ev.exch_time)
            self._side(ev.listing, True, ev.ask, ev.exch_time)
        else:
            self.bid[ev.listing] = ev.bid
            self.ask[ev.listing] = ev.ask

    def on_trade(self, ev) -> None:
        if ev.listing in self.fuse_trades:
            self._side(ev.listing, ev.lifts_ask, ev.px, ev.exch_time)
        self.was_trade_present = True

    def refresh(self) -> None:
        traded, self.was_trade_present = self.was_trade_present, False
        target_mid = self._mid(self.target)
        self.yard.on_target_logmid(math.log(target_mid) if target_mid is not None and target_mid > 0.0 else None)

        for listing, ex in self._key_of.items():
            mid = self._mid(listing)
            if mid is None or mid <= 0.0:
                continue
            log_mid = math.log(mid)
            prev = self.prev_log[ex]
            if prev is not None and log_mid != prev:
                self.leg[ex].add(log_mid - prev)
            self.prev_log[ex] = log_mid

        if traded:
            self.yard.tick()
            for ex in self.exes:
                self.leg[ex].tick()

    def value(self) -> dict[str, float]:
        sig = self.yard.sigma()
        return {ex: self.leg[ex].value() / sig for ex in self.exes}


SPEC = FeatureSpec(
    name="price_momentum",
    vectorized=vectorized,
    make_streaming=lambda ctx, params: LivePriceMomentum(ctx, params),
    keys_for=lambda ctx, params: (ctx.target.split("_", 1)[0],) + tuple(ctx.sources),
    mirror=np.negative,
    param_kind=ParamKind.SINGLE,
)
register(SPEC)
