"""`trade_flow_imbalance` -- signed trade-VOLUME imbalance as a single EMA feature.

WHAT: for each venue and receive timestamp, aggregate valid trades into the net and total VOLUME:

    signed_volume = sum(side * qty)        # side = +1 lifts the ask, -1 hits the bid
    total_volume  = sum(qty)

and read the trade-clock EMA ratio `EMA(signed_volume) / EMA(total_volume)` in [-1, 1] -- the
exponentially-weighted fraction of recent volume that was buyer-initiated minus seller-initiated. Built
as two sparse-flow `KernelMeanEMA` legs whose common event weights cancel in the ratio. It fans out over
every exchange -- the target plus each foreign source; `params = N` (the EMA span).

WHY (falsifiable hypothesis): aggressive order flow moves price and is autocorrelated (large parent
orders are split and executed over time), so a venue leaning net-buy should predict the target's mid
drifting UP over the ~100 ms horizon (net-sell, down); a foreign venue can lead. Falsified if the
signed-volume fraction carries no forward signed IC on byb's mid once the regime controls are netted.

RESEARCH: this is the signed order-flow / trade imbalance underlying the VPIN flow-toxicity metric --
Easley, D., Lopez de Prado, M. & O'Hara, M. (2012), "Flow Toxicity and Liquidity in a High-frequency
World", Review of Financial Studies 25(5):1457-1493. We VOLUME-weight (not notional): a `px*qty` weight
would break the mirror, since price reflects under the book flip while `qty` is invariant.

Mirror augmentation: `qty` is invariant under the tape reflection while trade side flips, so the feature
is ODD and `SPEC.mirror` is `np.negative` -- it commutes with the FULL book reflection (price + side),
exactly. See `AUTHORING.md` for the EMA-type and inject/decay rules these obey.
"""
from __future__ import annotations

import math

import numpy as np

from boba.ema import KernelMeanEMA
from boba.features.base import Config, FeatureSpec, ParamKind, Params, RawData, SharedData, Trade, register
from boba.features.shared import flow_at


def _ex(listing: str) -> str:
    """The short exchange key (leg key) for a full listing id, e.g. 'byb_eth_usdt_p' -> 'byb'."""
    return listing.split("_", 1)[0]


def _trade_sign(lifts_ask: float) -> float:
    return 1.0 if lifts_ask > 0.0 else -1.0


def _trade_volume_stream(trade: Trade) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`Trade` rows -> `(ts, signed_volume_sum, total_volume_sum)` per receive timestamp. VOLUME-weighted
    (`qty`, not notional) so the feature is exactly odd under reflection; bad prc<=0 / qty<=0 prints dropped."""
    rx, px, lifts, qty = trade.rx, trade.price, trade.lifts_ask, trade.qty
    ok = ((px > 0.0) & (qty > 0.0) & np.isfinite(px) & np.isfinite(qty) & np.isfinite(lifts))
    if not np.any(ok):
        return rx[:0], qty[:0].astype(float), qty[:0].astype(float)
    rx, lifts, qty = rx[ok], lifts[ok], qty[ok]
    signed = np.where(lifts > 0.0, qty, -qty)
    uniq, inv = np.unique(rx, return_inverse=True)
    return uniq, np.bincount(inv, weights=signed), np.bincount(inv, weights=qty)


def _imbalance_leg(trade: Trade, clock: np.ndarray, event_ts: np.ndarray, n: int) -> np.ndarray:
    """One venue's signed-volume imbalance `EMA(signed)/EMA(total)` at every `event_ts` (in [-1, 1]):
    the common per-event EMA weights cancel in the ratio, leaving the volume-weighted signed fraction."""
    rx, signed, total = _trade_volume_stream(trade)
    num = flow_at(clock, rx, signed, event_ts, n)
    den = flow_at(clock, rx, total, event_ts, n)
    return num / np.where(den == 0.0, np.nan, den)


def vectorized(raw: RawData, shared: SharedData, config: Config, params: Params) -> dict[str, np.ndarray]:
    """{exchange -> EMA(signed volume) / EMA(volume)}, one value per `event_ts` (causal, in [-1, 1])."""
    n = params
    return {_ex(l): _imbalance_leg(raw.listings[l].trade, shared.clock, shared.event_ts, n)
            for l in config.all_listings}


class LiveTradeFlowImbalance:
    """O(1) streaming build, one signed/total-volume EMA pair per venue. Same-timestamp trades are summed
    into one signed/total volume flow event; `refresh()` injects each venue's SUMMED flow, then decays all
    legs once iff a trade landed (shared trade clock)."""

    def __init__(self, config: Config, params: Params):
        self.exes = tuple(_ex(l) for l in config.all_listings)
        self.keys = self.exes
        self._key_of = {l: _ex(l) for l in config.all_listings}    # full listing -> short key
        self.num = {ex: KernelMeanEMA(params) for ex in self.exes}
        self.den = {ex: KernelMeanEMA(params) for ex in self.exes}
        self.ts_signed = {ex: 0.0 for ex in self.exes}
        self.ts_value = {ex: 0.0 for ex in self.exes}
        self.ts_got = {ex: False for ex in self.exes}
        self.was_trade_present = False

    def on_book(self, ev) -> None:                                 # book-only does nothing — this is a pure trade flow
        pass

    def on_trade(self, ev) -> None:                                # accumulate this venue's signed/total volume at the ts
        self.was_trade_present = True
        ex = self._key_of.get(ev.listing)
        if ex is None:
            return
        if not (ev.px > 0.0 and ev.qty > 0.0 and math.isfinite(ev.px)
                and math.isfinite(ev.qty) and math.isfinite(ev.lifts_ask)):
            return
        self.ts_signed[ex] += _trade_sign(ev.lifts_ask) * ev.qty
        self.ts_value[ex] += ev.qty
        self.ts_got[ex] = True

    def refresh(self) -> None:                                     # ONE per timestamp: inject SUMMED flow, then decay AT MOST once
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
    make_streaming=lambda config, params: LiveTradeFlowImbalance(config, params),
    keys_for=lambda config, params: tuple(_ex(l) for l in config.all_listings),
    mirror=np.negative,
    param_kind=ParamKind.SINGLE,
)
register(SPEC)
