"""`stoikov_premium_fast_slow` -- size-weighted-mid premium, as a fast/slow oscillator.

Each venue's L1 book gives a size-weighted fair value between the quotes:

    microprice = (bid_qty * ask_prc + ask_qty * bid_prc) / (bid_qty + ask_qty)
    prem       = (microprice - mid) / mid

This `microprice` is the SIZE-WEIGHTED (imbalance-adjusted) MID -- the *leading / first-order term*
of Stoikov's micro-price, NOT the full martingale micro-price estimator. Stoikov's estimator adds a
mean-reverting correction fitted from the imbalance->future-mid dynamics; here we keep only the
leading size-weighted-mid term, which is exact at L1 from `(bid, bid_qty, ask, ask_qty)` and needs no
fit. (Name kept for continuity; read "stoikov premium" as "size-weighted-mid premium".)

A positive premium means the touch leans up: bid size is large relative to ask size, pulling the
fair value toward the ask. This feature smooths that LEVEL two ways with `LiveFrontEMA` and returns
`fast - slow`. It fans out over every exchange -- the target plus each foreign source -- using each
venue's OWN raw `front_levels` book because the sizes are required; `params = (n_fast, n_slow)`.

Reference: Stoikov, S. (2018) 'The Micro-Price: A High-Frequency Estimator of Future Prices',
Quantitative Finance 18(12):1959-1966 (SSRN 2970694); Gatheral, J. & Oomen, R. (size-weighted mid).

Two implementations of the same maths, tied by `boba.research.screening.parity_check`:
  - `vectorized(ctx, params)` -> {exchange -> feature vector on the grid}   (offline; may use lfilter)
  - `LiveStoikovPremiumFastSlow` -> O(1) streaming build (two `LiveFrontEMA` legs per venue)

Mirror augmentation: the premium is ODD under book reflection. The reflected book swaps bid/ask and
their sizes, so `(microprice - mid) / mid` negates exactly. Therefore `SPEC.mirror` is `np.negative`.

See `AUTHORING.md` (this directory) for the EMA-type and inject/decay rules these obey.
"""
from __future__ import annotations

import math

import numpy as np

from boba.ema import LiveFrontEMA
from boba.features.base import FeatureSpec, ParamKind, Params, register
from boba.research.screening import ScreeningContext


def _exchanges(ctx: ScreeningContext) -> tuple[str, ...]:
    """Every venue we build a premium leg for: the target plus each foreign source."""
    return (ctx.target.split("_", 1)[0],) + tuple(ctx.sources)


def _premium_stream(book: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Raw `(rx, bid, bid_qty, ask, ask_qty)` rows -> valid `(rx, premium)` level stream.

    The premium needs both prices and both side sizes. Rows with non-positive or non-finite inputs are
    ignored, so they do not overwrite the previous valid level in either build.
    """
    rx, bid, bq, ask, aq = book
    ok = ((bid > 0.0) & (ask > 0.0) & (bq > 0.0) & (aq > 0.0)
          & np.isfinite(bid) & np.isfinite(ask) & np.isfinite(bq) & np.isfinite(aq))
    if not np.any(ok):
        return rx[:0], bid[:0].astype(float)
    bid, bq, ask, aq = bid[ok], bq[ok], ask[ok], aq[ok]
    mid = 0.5 * (bid + ask)
    micro = (bq * ask + aq * bid) / (bq + aq)
    return rx[ok], (micro - mid) / mid


def _ffill(rx: np.ndarray, val: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Causal forward-fill with NaN before the first valid source row."""
    out = np.full(len(t), np.nan)
    if len(rx) == 0:
        return out
    idx = np.searchsorted(rx, t, "right") - 1
    ok = idx >= 0
    out[ok] = val[idx[ok]]
    return out


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """Offline trade-clock EMA of a level sampled on the trade clock."""
    if span == 1:
        return x
    a = 2.0 / (span + 1.0)
    try:
        from scipy.signal import lfilter
    except ModuleNotFoundError:
        out = np.empty_like(x, dtype=float)
        ema = 0.0
        for i, v in enumerate(x):
            ema = (1.0 - a) * ema + a * v
            out[i] = ema
        return out
    return lfilter([a], [1.0, -(1.0 - a)], x)


def _live_front_at(ctx: ScreeningContext, rx: np.ndarray, prem: np.ndarray, span: int) -> np.ndarray:
    """`LiveFrontEMA(span)` for the premium stream, read at each anchor."""
    out = np.full(len(ctx.anchor_ts), np.nan)
    if len(rx) == 0 or len(ctx.merged_ts) == 0 or len(ctx.anchor_ts) == 0:
        return out

    a = 2.0 / (span + 1.0)
    prem_on_clock = np.nan_to_num(_ffill(rx, prem, ctx.merged_ts), nan=0.0)
    fresh = _ffill(rx, prem, ctx.anchor_ts)
    tick = ctx.tick_at_anchor
    tick_ok = tick >= 0
    first_commit = np.searchsorted(ctx.merged_ts, rx[0], "left")
    started = tick_ok & (tick >= first_commit) & (first_commit < len(ctx.merged_ts))
    if not np.any(started):
        return out

    committed = _ema(prem_on_clock, span)
    out[started] = (1.0 - a) * committed[tick[started]] + a * fresh[started]
    return out


def _premium_scalar(bid: float, bq: float, ask: float, aq: float) -> float | None:
    """Scalar Stoikov premium for one book event; `None` means the row is not a valid L1 state."""
    if not (bid > 0.0 and ask > 0.0 and bq > 0.0 and aq > 0.0):
        return None
    if not (math.isfinite(bid) and math.isfinite(ask) and math.isfinite(bq) and math.isfinite(aq)):
        return None
    mid = 0.5 * (bid + ask)
    micro = (bq * ask + aq * bid) / (bq + aq)
    return (micro - mid) / mid


def vectorized(ctx: ScreeningContext, params: Params) -> dict[str, np.ndarray]:
    """{exchange -> fast - slow `LiveFrontEMA` of that venue's Stoikov premium}, on the anchor grid."""
    n_fast, n_slow = params
    out: dict[str, np.ndarray] = {}
    for ex in _exchanges(ctx):
        rx, prem = _premium_stream(ctx._books[ex])
        out[ex] = _live_front_at(ctx, rx, prem, n_fast) - _live_front_at(ctx, rx, prem, n_slow)
    return out


class LiveStoikovPremiumFastSlow:
    """O(1) streaming build, one premium oscillator per venue.

    Book rows refresh the timestamp's last valid premium; `refresh()` pushes one level update per
    venue, then decays/commits both legs once iff a trade landed on the shared clock.
    """

    def __init__(self, ctx: ScreeningContext, params: Params):
        n_fast, n_slow = params
        coin = ctx.coin
        self.exes = _exchanges(ctx)
        self.keys = self.exes
        self._key_of = {f"{ex}_{coin}": ex for ex in self.exes}
        self.fuse_trades = frozenset()                    # premium needs raw book sizes; trades do not refresh it
        self.leg_f = {ex: LiveFrontEMA(n_fast) for ex in self.exes}
        self.leg_s = {ex: LiveFrontEMA(n_slow) for ex in self.exes}
        self.ts_prem = {ex: 0.0 for ex in self.exes}
        self.ts_got = {ex: False for ex in self.exes}
        self.was_trade_present = False

    def on_book(self, ev) -> None:
        ex = self._key_of.get(ev.listing)
        if ex is None:
            return
        prem = _premium_scalar(ev.bid, ev.bid_qty, ev.ask, ev.ask_qty)
        if prem is None:
            return
        self.ts_prem[ex] = prem
        self.ts_got[ex] = True

    def on_trade(self, ev) -> None:
        self.was_trade_present = True

    def refresh(self) -> None:
        traded, self.was_trade_present = self.was_trade_present, False
        for ex in self.exes:
            if self.ts_got[ex]:
                self.leg_f[ex].add(self.ts_prem[ex])
                self.leg_s[ex].add(self.ts_prem[ex])
                self.ts_got[ex] = False
        if traded:
            for ex in self.exes:
                self.leg_f[ex].tick()
                self.leg_s[ex].tick()

    def value(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for ex in self.exes:
            f, s = self.leg_f[ex].value(), self.leg_s[ex].value()
            out[ex] = (f - s) if (f == f and s == s) else float("nan")
        return out


SPEC = FeatureSpec(
    name="stoikov_premium_fast_slow",
    vectorized=vectorized,
    make_streaming=lambda ctx, params: LiveStoikovPremiumFastSlow(ctx, params),
    keys_for=lambda ctx, params: (ctx.target.split("_", 1)[0],) + tuple(ctx.sources),
    mirror=np.negative,                                      # bid/ask + sizes swap under reflection -> premium negates
    param_kind=ParamKind.FAST_SLOW,                          # params = (n_fast, n_slow)
)
register(SPEC)
