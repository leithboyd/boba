"""`flow_persistence` -- EMA of consecutive per-timestamp trade-flow sign agreement.

For each venue, trades sharing a receive timestamp are first netted by side count:

    eps_t = sign(sum(side_i))

Exact ties are skipped. The feature is a sparse-flow `KernelMeanEMA` of `eps_t * eps_prev`, where
`eps_prev` is the venue's previous non-zero per-timestamp sign. Positive values mean trade-flow sign
tends to persist; negative values mean it tends to alternate. It fans out over every exchange -- the
target plus each foreign source; `params = N` (the EMA span).

Mirror augmentation: under tape reflection both `eps_t` and `eps_prev` flip sign, so their product is
EVEN. `SPEC.mirror` is therefore the identity.
"""
from __future__ import annotations

import math

import numpy as np

from boba.ema import KernelMeanEMA
from boba.features.base import FeatureSpec, ParamKind, Params, register
from boba.research.screening import ScreeningContext


def _identity(v: np.ndarray) -> np.ndarray:
    return v


def _exchanges(ctx: ScreeningContext) -> tuple[str, ...]:
    return (ctx.target.split("_", 1)[0],) + tuple(ctx.sources)


def _trade_sign(lifts_ask: float) -> float:
    return 1.0 if lifts_ask > 0.0 else -1.0


def _eps_stream(trades: tuple) -> tuple[np.ndarray, np.ndarray]:
    """`(rx, px, lifts_ask, qty)` rows -> non-zero `(ts, eps)` after same-timestamp side netting."""
    rx, px, lifts, qty = trades
    ok = ((px > 0.0) & (qty > 0.0) & np.isfinite(px) & np.isfinite(qty) & np.isfinite(lifts))
    if not np.any(ok):
        return rx[:0], px[:0].astype(float)
    rx, lifts = rx[ok], lifts[ok]
    signs = np.where(lifts > 0.0, 1.0, -1.0)
    uniq, inv = np.unique(rx, return_inverse=True)
    net = np.bincount(inv, weights=signs)
    eps = np.sign(net)
    keep = eps != 0.0
    return uniq[keep], eps[keep]


def _persistence_stream(trades: tuple) -> tuple[np.ndarray, np.ndarray]:
    rx, eps = _eps_stream(trades)
    if len(eps) < 2:
        return rx[:0], eps[:0]
    return rx[1:], eps[1:] * eps[:-1]


def vectorized(ctx: ScreeningContext, params: Params) -> dict[str, np.ndarray]:
    """{exchange -> E/W EMA of `eps_t * eps_prev`}, on the anchor grid."""
    n = params
    out: dict[str, np.ndarray] = {}
    for ex in _exchanges(ctx):
        rx, per = _persistence_stream(ctx._trades[ex])
        E = ctx._flow_at(ctx.anchor_ts, per, n, src_rx=rx)
        W = ctx._flow_at(ctx.anchor_ts, np.ones(per.size), n, src_rx=rx)
        out[ex] = E / np.where(W == 0.0, np.nan, W)
    return out


class LiveFlowPersistence:
    """O(1) streaming build. Same-timestamp prints become one net sign per venue."""

    def __init__(self, ctx: ScreeningContext, params: Params):
        coin = ctx.coin
        self.exes = _exchanges(ctx)
        self.keys = self.exes
        self._key_of = {f"{ex}_{coin}": ex for ex in self.exes}
        self.fuse_trades = frozenset()
        self.leg = {ex: KernelMeanEMA(params) for ex in self.exes}
        self.net = {ex: 0.0 for ex in self.exes}
        self.dirty = {ex: False for ex in self.exes}
        self.prev = {ex: 0.0 for ex in self.exes}
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
        self.net[ex] += _trade_sign(ev.lifts_ask)
        self.dirty[ex] = True

    def refresh(self) -> None:
        traded, self.was_trade_present = self.was_trade_present, False
        for ex in self.exes:
            if self.dirty[ex]:
                eps = 1.0 if self.net[ex] > 0.0 else (-1.0 if self.net[ex] < 0.0 else 0.0)
                if eps != 0.0:
                    if self.prev[ex] != 0.0:
                        self.leg[ex].add(eps * self.prev[ex])
                    self.prev[ex] = eps
                self.net[ex] = 0.0
                self.dirty[ex] = False
        if traded:
            for ex in self.exes:
                self.leg[ex].tick()

    def value(self) -> dict[str, float]:
        return {ex: self.leg[ex].value() for ex in self.exes}


SPEC = FeatureSpec(
    name="flow_persistence",
    vectorized=vectorized,
    make_streaming=lambda ctx, params: LiveFlowPersistence(ctx, params),
    keys_for=lambda ctx, params: (ctx.target.split("_", 1)[0],) + tuple(ctx.sources),
    mirror=_identity,
    param_kind=ParamKind.SINGLE,
)
register(SPEC)
