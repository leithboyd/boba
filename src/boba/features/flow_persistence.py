"""`flow_persistence` -- EMA of consecutive per-timestamp trade-flow sign agreement.

WHAT it measures. For each venue, trades sharing a receive timestamp are first netted by side count:

    eps_t = sign(sum(side_i))

Exact ties are skipped. The feature is a sparse-flow `KernelMeanEMA` of `eps_t * eps_prev`, where
`eps_prev` is the venue's previous non-zero per-timestamp sign. Positive values mean trade-flow sign
tends to persist; negative values mean it tends to alternate. It fans out over every exchange -- the
target plus each foreign source; `params = N` (the EMA span).

WHY it might predict (a falsifiable hypothesis). Trade-flow sign has long memory: large parent orders
are split into many child slices that print in one direction over time, so a buy is far more likely to
be followed by another buy than by a sell. When `eps_t * eps_prev > 0` (consecutive signs agree) the
venue is mid-campaign and the directional pressure is ongoing -- so the move count over the next
horizon should be elevated. Because the feature measures persistence *magnitude* and is sign-symmetric
(a buy-campaign and a sell-campaign produce the same product), it is an EVEN feature and belongs to the
RATE head; its mirror is the identity. Falsified if `|feature|` has no forward power on the move count.

RESEARCH. Bouchaud, J.-P., Gefen, Y., Potters, M. & Wyart, M. (2004) 'Fluctuations and response in
financial markets: the subtle nature of random price changes', Quantitative Finance 4(2):176-190
(long-memory / autocorrelation of order-flow sign).

Mirror augmentation: under tape reflection both `eps_t` and `eps_prev` flip sign, so their product is
EVEN. `SPEC.mirror` is therefore the identity.

Two implementations of the same maths, tied by the parity driver:
  - `vectorized(raw, shared, config, N)` -> {exchange -> E/W EMA per `shared.event_ts`}  (offline; `flow_at`)
  - `LiveFlowPersistence`                -> the O(1) streaming build (one `KernelMeanEMA` E/W leg per venue)

See `AUTHORING.md` (this directory) for the EMA-type and inject/decay rules these obey.
"""
from __future__ import annotations

import math

import numpy as np

from boba.ema import KernelMeanEMA
from boba.features.base import Config, FeatureSpec, ParamKind, Params, RawData, SharedData, Trade, register


def _ex(listing: str) -> str:
    """The short exchange key (leg key) for a full listing id, e.g. 'byb_eth_usdt_p' -> 'byb'."""
    return listing.split("_", 1)[0]


def _identity(v: np.ndarray) -> np.ndarray:
    return v


def _eps_stream(trade: Trade) -> tuple[np.ndarray, np.ndarray]:
    """`trade` rows -> non-zero `(ts, eps)` after same-timestamp side netting. `eps_t = sign(sum of side)`
    per receive-ts (a buy lifts the ask = +1, a sell hits the bid = -1); exact ties (net == 0) are dropped."""
    rx, px, lifts, qty = trade.rx, trade.price, trade.lifts_ask, trade.qty
    ok = (px > 0.0) & (qty > 0.0) & np.isfinite(px) & np.isfinite(qty) & np.isfinite(lifts)
    if not np.any(ok):
        return rx[:0].astype(np.int64), px[:0].astype(float)
    rx, lifts = rx[ok], lifts[ok]
    signs = np.where(lifts > 0.0, 1.0, -1.0)
    uniq, inv = np.unique(rx, return_inverse=True)
    net = np.bincount(inv, weights=signs)
    eps = np.sign(net)
    keep = eps != 0.0
    return uniq[keep], eps[keep]


def _persistence_stream(trade: Trade) -> tuple[np.ndarray, np.ndarray]:
    """`(ts, eps_t * eps_prev)` over consecutive non-zero per-timestamp signs (eps_prev = the previous one,
    'seen through' any skipped tie). The sparse flow the EMA is taken over, stamped at the CURRENT ts."""
    rx, eps = _eps_stream(trade)
    if len(eps) < 2:
        return rx[:0], eps[:0]
    return rx[1:], eps[1:] * eps[:-1]


def _persistence_leg(trade: Trade, clock: np.ndarray, event_ts: np.ndarray, n: int) -> np.ndarray:
    """One venue's persistence EMA read as `E/W` at every `event_ts`: its `eps_t * eps_prev` flow decayed
    once per trade-clock tick, read live (committed-per-tick + the partial epoch since the last tick)."""
    from boba.features.shared import flow_at

    per_rx, per = _persistence_stream(trade)
    e = flow_at(clock, per_rx, per, event_ts, n)
    w = flow_at(clock, per_rx, np.ones(per.size), event_ts, n)
    return e / np.where(w == 0.0, np.nan, w)


def vectorized(raw: RawData, shared: SharedData, config: Config, params: Params) -> dict[str, np.ndarray]:
    """{exchange -> E/W EMA of `eps_t * eps_prev` at span N}, one value per `event_ts` (causal)."""
    n = params
    return {_ex(l): _persistence_leg(raw.listings[l].trade, shared.clock, shared.event_ts, n)
            for l in config.all_listings}


def _trade_sign(lifts_ask: float) -> float:
    return 1.0 if lifts_ask > 0.0 else -1.0


class LiveFlowPersistence:
    """O(1) streaming build, one persistence leg per venue. Same-timestamp prints net to one sign per
    venue; `refresh()` injects `eps_t * eps_prev` (against the venue's previous non-zero sign, skipping
    ties), then decays all legs once iff a trade landed (shared clock)."""

    def __init__(self, config: Config, params: Params):
        self.exes = tuple(_ex(l) for l in config.all_listings)
        self.keys = self.exes
        self._key_of = {l: _ex(l) for l in config.all_listings}   # full listing -> short key
        self.leg = {ex: KernelMeanEMA(params) for ex in self.exes}
        self.net = {ex: 0.0 for ex in self.exes}                  # summed side this timestamp, per venue
        self.dirty = {ex: False for ex in self.exes}              # did a usable trade land this ts, per venue
        self.prev = {ex: 0.0 for ex in self.exes}                 # the venue's previous non-zero per-ts sign
        self.was_trade_present = False                            # did any trade land this ts? -> one decay

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
                if eps != 0.0:                                    # exact ties are skipped, prev unchanged
                    if self.prev[ex] != 0.0:
                        self.leg[ex].add(eps * self.prev[ex])
                    self.prev[ex] = eps
                self.net[ex] = 0.0
                self.dirty[ex] = False
        if traded:
            for ex in self.exes:
                self.leg[ex].tick()

    def value(self) -> dict[str, float]:
        return {ex: self.leg[ex].value() for ex in self.exes}     # E/W per venue, nan during warm-up


SPEC = FeatureSpec(
    name="flow_persistence",
    vectorized=vectorized,
    make_streaming=lambda config, params: LiveFlowPersistence(config, params),
    keys_for=lambda config, params: tuple(_ex(l) for l in config.all_listings),
    mirror=_identity,   # eps_t and eps_prev both flip under reflection -> their product is EVEN
    param_kind=ParamKind.SINGLE,
)
register(SPEC)
