"""`trade_flow_imbalance` -- signed trade notional imbalance as a single EMA feature.

For each venue and receive timestamp, aggregate valid trades into:

    signed_value = sum(side * price * qty)
    value        = sum(price * qty)

where `side` is +1 for trades lifting the ask and -1 for trades hitting the bid. The feature is the
trade-clock EMA ratio `EMA(signed_value) / EMA(value)`, built as two sparse-flow `KernelMeanEMA`
legs whose common event weights cancel in the ratio. It fans out over every exchange -- the target
plus each foreign source; `params = N` (the EMA span).

Mirror augmentation: the positive notional weight is a magnitude, while trade side flips, so the
feature is ODD under tape reflection and `SPEC.mirror` is `np.negative`.
"""
from __future__ import annotations

import math

import numpy as np

from boba.ema import KernelMeanEMA
from boba.features.base import FeatureSpec, ParamKind, Params, register
from boba.research.screening import ScreeningContext


def _exchanges(ctx: ScreeningContext) -> tuple[str, ...]:
    return (ctx.target.split("_", 1)[0],) + tuple(ctx.sources)


def _trade_sign(lifts_ask: float) -> float:
    return 1.0 if lifts_ask > 0.0 else -1.0


def _trade_value_stream(trades: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(rx, px, lifts_ask, qty)` rows -> `(ts, signed_value_sum, value_sum)` per receive timestamp."""
    rx, px, lifts, qty = trades
    ok = ((px > 0.0) & (qty > 0.0) & np.isfinite(px) & np.isfinite(qty) & np.isfinite(lifts))
    if not np.any(ok):
        return rx[:0], px[:0].astype(float), px[:0].astype(float)
    rx, px, lifts, qty = rx[ok], px[ok], lifts[ok], qty[ok]
    value = px * qty
    signed = np.where(lifts > 0.0, value, -value)
    uniq, inv = np.unique(rx, return_inverse=True)
    return uniq, np.bincount(inv, weights=signed), np.bincount(inv, weights=value)


def vectorized(ctx: ScreeningContext, params: Params) -> dict[str, np.ndarray]:
    """{exchange -> EMA(signed notional) / EMA(notional)}, on the anchor grid."""
    n = params
    out: dict[str, np.ndarray] = {}
    for ex in _exchanges(ctx):
        rx, signed, value = _trade_value_stream(ctx._trades[ex])
        num = ctx._flow_at(ctx.anchor_ts, signed, n, src_rx=rx)
        den = ctx._flow_at(ctx.anchor_ts, value, n, src_rx=rx)
        out[ex] = num / np.where(den == 0.0, np.nan, den)
    return out


class LiveTradeFlowImbalance:
    """O(1) streaming build. Same-timestamp trades are summed into one signed/total notional flow event."""

    def __init__(self, ctx: ScreeningContext, params: Params):
        coin = ctx.coin
        self.exes = _exchanges(ctx)
        self.keys = self.exes
        self._key_of = {f"{ex}_{coin}": ex for ex in self.exes}
        self.fuse_trades = frozenset()
        self.num = {ex: KernelMeanEMA(params) for ex in self.exes}
        self.den = {ex: KernelMeanEMA(params) for ex in self.exes}
        self.ts_signed = {ex: 0.0 for ex in self.exes}
        self.ts_value = {ex: 0.0 for ex in self.exes}
        self.ts_got = {ex: False for ex in self.exes}
        self.was_trade_present = False

    def on_book(self, ev) -> None:
        pass

    def on_trade(self, ev) -> None:
        self.was_trade_present = True
        ex = self._key_of.get(ev.listing)
        if ex is None:
            return
        if not (ev.px > 0.0 and ev.qty > 0.0 and math.isfinite(ev.px)
                and math.isfinite(ev.qty) and math.isfinite(ev.lifts_ask)):
            return
        value = ev.px * ev.qty
        self.ts_signed[ex] += _trade_sign(ev.lifts_ask) * value
        self.ts_value[ex] += value
        self.ts_got[ex] = True

    def refresh(self) -> None:
        traded, self.was_trade_present = self.was_trade_present, False
        for ex in self.exes:
            if self.ts_got[ex] and self.ts_value[ex] > 0.0:
                self.num[ex].add(self.ts_signed[ex])
                self.den[ex].add(self.ts_value[ex])
            self.ts_signed[ex] = 0.0
            self.ts_value[ex] = 0.0
            self.ts_got[ex] = False
        if traded:
            for ex in self.exes:
                self.num[ex].tick()
                self.den[ex].tick()

    def value(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for ex in self.exes:
            num, den = self.num[ex].value(), self.den[ex].value()
            out[ex] = num / den if (num == num and den == den and den != 0.0) else float("nan")
        return out


SPEC = FeatureSpec(
    name="trade_flow_imbalance",
    vectorized=vectorized,
    make_streaming=lambda ctx, params: LiveTradeFlowImbalance(ctx, params),
    keys_for=lambda ctx, params: (ctx.target.split("_", 1)[0],) + tuple(ctx.sources),
    mirror=np.negative,
    param_kind=ParamKind.SINGLE,
)
register(SPEC)
