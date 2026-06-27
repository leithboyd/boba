"""`price_momentum` -- per-venue EMA of log-mid moves in target volatility units.

WHAT it measures. For each venue, collapse same-timestamp mid rows to the final state, take non-zero
`Δlog(mid)` moves, and smooth them as a sparse flow with `KernelMeanEMA` on the shared trade clock.
The feature is:

    EMA(Δlog mid) / σ_ev

where `σ_ev` is the target volatility yardstick (`shared_data.vol_yardstick` / `VolYardstick`). It
fans out over every exchange -- the target plus each foreign source; `params = N` (the EMA span).

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

See `AUTHORING.md` (this directory) for the EMA-type and inject/decay rules these obey.
"""
from __future__ import annotations

import math

import numpy as np

from boba.ema import KernelMeanEMA
from boba.features.base import Config, Params, RawData, Series, SharedData, FeatureSpec, ParamKind, register
from boba.features.shared import flow_at
from boba.features.streaming import LiveMergedBook, VolYardstick


def _ex(listing: str) -> str:
    """The short exchange key (leg key) for a full listing id, e.g. 'byb_eth_usdt_p' -> 'byb'."""
    return listing.split("_", 1)[0]


def _move_stream(mid: Series) -> tuple[np.ndarray, np.ndarray]:
    """A venue's mid `Series` -> non-zero `(rx, Δlog(mid))`, keeping the FINAL mid per timestamp.
    Same-rx mids collapse to the last row, then consecutive non-zero log-moves are kept."""
    rx, m = np.asarray(mid.rx), np.asarray(mid.value)
    ok = (m > 0.0) & np.isfinite(m)
    rx, m = rx[ok], m[ok]
    if len(rx) < 2:
        return rx[:0], m[:0].astype(float)
    keep = np.concatenate([rx[1:] != rx[:-1], [True]])        # last mid per timestamp
    rx, m = rx[keep], m[keep]
    if len(rx) < 2:
        return rx[:0], m[:0].astype(float)
    dlog = np.diff(np.log(m))
    move = dlog != 0.0
    return rx[1:][move], dlog[move]


def _pm_leg(mid: Series, clock: np.ndarray, event_ts: np.ndarray, n: int) -> np.ndarray:
    """One venue's `EMA(Δlog mid)` read as `E/W` at every `event_ts`: its non-zero log-mid moves decayed
    once per trade-clock tick, read live (committed-per-tick + the partial epoch since the last tick).
    Volatility normalisation (`/ σ_ev`) is applied by the caller."""
    rx, ret = _move_stream(mid)
    e = flow_at(clock, rx, ret, event_ts, n)
    w = flow_at(clock, rx, np.ones(ret.size), event_ts, n)
    return e / np.where(w == 0.0, np.nan, w)


def vectorized(raw: RawData, shared: SharedData, config: Config, params: Params) -> dict[str, np.ndarray]:
    """{exchange -> E/W EMA of that venue's log-mid moves / σ_ev}, one value per `event_ts` (causal)."""
    n = params
    sig = np.where(shared.vol_yardstick == 0.0, np.nan, shared.vol_yardstick)   # σ_ev=0 only in warm-up -> clean NaN, not inf
    return {_ex(l): _pm_leg(shared.listings[l].mid, shared.clock, shared.event_ts, n) / sig
            for l in config.all_listings}


class LivePriceMomentum:
    """O(1) streaming build, one leg per venue. Each venue carries a `KernelMeanEMA` over its non-zero
    log-mid moves; the mid follows the listing's mid policy (book-only, or trade-fused for the listings
    in `fuse_trades`). `refresh()` injects each venue's move, feeds the target log-mid to a composed
    `VolYardstick` (σ_ev), then decays all legs + the yardstick once iff a trade landed (shared clock)."""

    def __init__(self, config: Config, params: Params):
        self.target = config.target_listing
        self.exes = tuple(_ex(l) for l in config.all_listings)
        self.keys = self.exes
        self._key_of = {l: _ex(l) for l in config.all_listings}   # full listing -> short key
        fuse_tick = {l: config.tick_size[l] for l in config.all_listings
                     if config.mid_stream.get(l, "front_levels") == "merged_levels"}   # KeyError -> no tick for a fused listing
        self.book = LiveMergedBook(fuse_tick)        # shared merged-book reconstruction (fuse + un-cross); read .quote()
        self.prev_log = {ex: None for ex in self.exes}
        self.leg = {ex: KernelMeanEMA(params) for ex in self.exes}
        self.yard = VolYardstick(config.yardstick_span)
        self.was_trade_present = False

    def on_book(self, ev) -> None:
        self.book.on_book(ev)

    def on_trade(self, ev) -> None:
        self.book.on_trade(ev)
        self.was_trade_present = True

    def refresh(self) -> None:
        traded, self.was_trade_present = self.was_trade_present, False
        tq = self.book.quote(self.target)
        target_mid = 0.5 * (tq[0] + tq[1]) if tq is not None else None      # mid from the un-crossed (bid, ask)
        self.yard.on_target_logmid(math.log(target_mid) if target_mid is not None and target_mid > 0.0 else None)

        for listing, ex in self._key_of.items():
            q = self.book.quote(listing)
            if q is None:
                continue
            mid = 0.5 * (q[0] + q[1])
            if mid <= 0.0:
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
    make_streaming=lambda config, params: LivePriceMomentum(config, params),
    keys_for=lambda config, params: tuple(_ex(l) for l in config.all_listings),
    mirror=np.negative,   # log returns negate under price reflection; σ_ev is even -> the feature is ODD
    param_kind=ParamKind.SINGLE,                             # params = N (a single EMA span)
)
register(SPEC)
