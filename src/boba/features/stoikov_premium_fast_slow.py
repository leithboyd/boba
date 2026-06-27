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
  - `vectorized(raw, shared, config, params)` -> {exchange -> fast-slow per `shared.event_ts`}  (offline)
  - `LiveStoikovPremiumFastSlow`               -> O(1) streaming build (two `LiveFrontEMA` legs per venue)

Mirror augmentation: the premium is ODD under book reflection. The reflected book swaps bid/ask and
their sizes, so `(microprice - mid) / mid` negates exactly. Therefore `SPEC.mirror` is `np.negative`.

See `AUTHORING.md` (this directory) for the EMA-type and inject/decay rules these obey.
"""
from __future__ import annotations

import math

import numpy as np

from boba.ema import LiveFrontEMA
from boba.features.base import Config, FeatureSpec, FrontLevels, ParamKind, Params, RawData, SharedData, register
from boba.features.shared import _ffill


def _ex(listing: str) -> str:
    """The short exchange key (leg key) for a full listing id, e.g. 'byb_eth_usdt_p' -> 'byb'."""
    return listing.split("_", 1)[0]


def _premium_stream(front: FrontLevels) -> tuple[np.ndarray, np.ndarray]:
    """`front_levels` rows -> valid `(rx, premium)` LEVEL stream. The premium needs both prices and both
    side sizes; rows with non-positive or non-finite inputs are dropped so they never overwrite the
    previous valid level in either build."""
    rx, bid, bq, ask, aq = front.rx, front.bid, front.bid_qty, front.ask, front.ask_qty
    ok = ((bid > 0.0) & (ask > 0.0) & (bq > 0.0) & (aq > 0.0)
          & np.isfinite(bid) & np.isfinite(ask) & np.isfinite(bq) & np.isfinite(aq))
    if not np.any(ok):
        return rx[:0].astype(np.int64), bid[:0].astype(float)
    bid, bq, ask, aq = bid[ok], bq[ok], ask[ok], aq[ok]
    mid = 0.5 * (bid + ask)
    micro = (bq * ask + aq * bid) / (bq + aq)
    return np.asarray(rx[ok], np.int64), (micro - mid) / mid


def _live_front_at(rx: np.ndarray, prem: np.ndarray, clock: np.ndarray, event_ts: np.ndarray,
                   span: int) -> np.ndarray:
    """`LiveFrontEMA(span)` of the premium LEVEL, read live at every `event_ts`: the committed EMA on the
    decay `clock` (the premium ffilled/sampled to the clock) carried one step toward the fresh premium
    at `event_ts`. `fast = (1-a)*committed[tick_at_event] + a*fresh`, with `tick_at_event` the last clock
    tick at-or-before each event_ts. NaN before the first commit / first valid level (warm-up)."""
    out = np.full(len(event_ts), np.nan)
    if len(rx) == 0 or len(clock) == 0 or len(event_ts) == 0:
        return out
    from scipy.signal import lfilter

    a = 2.0 / (span + 1.0)
    prem_on_clock = np.nan_to_num(_ffill(rx, prem, clock), nan=0.0)      # premium sampled to the clock
    fresh = _ffill(rx, prem, event_ts)                                   # freshest premium at each event_ts
    committed = prem_on_clock if span == 1 else lfilter([a], [1.0, -(1.0 - a)], prem_on_clock)

    tick = np.searchsorted(clock, event_ts, "right") - 1                 # last clock tick <= each event_ts
    first_commit = np.searchsorted(clock, rx[0], "left")                 # first tick that sees a premium
    started = (tick >= 0) & (tick >= first_commit) & (first_commit < len(clock))
    if not np.any(started):
        return out
    out[started] = (1.0 - a) * committed[tick[started]] + a * fresh[started]
    return out


def _premium_leg(front: FrontLevels, clock: np.ndarray, event_ts: np.ndarray,
                 n_fast: int, n_slow: int) -> np.ndarray:
    """One venue's fast-slow size-weighted-mid premium oscillator, read live at every `event_ts`: two
    `LiveFrontEMA` legs (fast, slow) over that venue's premium LEVEL, returned as `fast - slow`."""
    rx, prem = _premium_stream(front)
    return (_live_front_at(rx, prem, clock, event_ts, n_fast)
            - _live_front_at(rx, prem, clock, event_ts, n_slow))


def vectorized(raw: RawData, shared: SharedData, config: Config, params: Params) -> dict[str, np.ndarray]:
    """{exchange -> fast - slow `LiveFrontEMA` of that venue's Stoikov premium}, one value per `event_ts`."""
    n_fast, n_slow = params
    return {_ex(l): _premium_leg(raw.listings[l].front_levels, shared.clock, shared.event_ts, n_fast, n_slow)
            for l in config.all_listings}


def _premium_scalar(bid: float, bq: float, ask: float, aq: float) -> float | None:
    """Scalar Stoikov premium for one book event; `None` means the row is not a valid L1 state."""
    if not (bid > 0.0 and ask > 0.0 and bq > 0.0 and aq > 0.0):
        return None
    if not (math.isfinite(bid) and math.isfinite(ask) and math.isfinite(bq) and math.isfinite(aq)):
        return None
    mid = 0.5 * (bid + ask)
    micro = (bq * ask + aq * bid) / (bq + aq)
    return (micro - mid) / mid


class LiveStoikovPremiumFastSlow:
    """O(1) streaming build, one premium oscillator per venue.

    Book rows refresh the timestamp's last valid premium; `refresh()` pushes one level update per
    venue, then decays/commits both legs once iff a trade landed on the shared clock.
    """

    def __init__(self, config: Config, params: Params):
        n_fast, n_slow = params
        self.exes = tuple(_ex(l) for l in config.all_listings)
        self.keys = self.exes
        self._key_of = {l: _ex(l) for l in config.all_listings}   # full listing -> short key
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
    make_streaming=lambda config, params: LiveStoikovPremiumFastSlow(config, params),
    keys_for=lambda config, params: tuple(_ex(l) for l in config.all_listings),
    mirror=np.negative,                                      # bid/ask + sizes swap under reflection -> premium negates
    param_kind=ParamKind.FAST_SLOW,                          # params = (n_fast, n_slow)
)
register(SPEC)
