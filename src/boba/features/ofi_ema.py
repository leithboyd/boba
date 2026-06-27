"""`ofi_ema` — a single EMA of byb's path-sum level-1 Order-Flow Imbalance.

The raw-EMA sibling of `ofi_fast_slow`: the same path-sum OFI flow (per book change, the SUM of the
Cont–Kukanov–Stoikov increments at that timestamp), smoothed by ONE `KernelMeanEMA` leg read as `E/W`
(a sparse flow per AUTHORING). It **fans out over every exchange** — one leg per venue, each from that
venue's OWN book; `params = N` (the EMA span). `N = 1` is the freshest, unsmoothed OFI.

Two implementations of the same maths, tied by `boba.research.screening.parity_check`:
  - `vectorized(raw, shared, config, N)` -> {exchange -> value per `shared.event_ts`}  (offline; uses `flow_at`)
  - `LiveOFIEma`                          -> the O(1) streaming build (one `KernelMeanEMA` E/W leg per venue)

Mirror augmentation: OFI is signed order flow, ODD under the reflection of the tape through byb's mid
(the bid/ask sides swap, so the increment negates). So `SPEC.mirror` is `np.negative`.

WHY it might predict (falsifiable): the path-sum L1 OFI flow measures net depth *added* on the bid minus
that added on the ask — pressure from limit-order arrivals/cancellations at the top of book. The
hypothesis is that this signed pressure leads byb's next mid-move in the same direction: a positive EMA
(more bid-side than ask-side depth accruing) precedes an up-move, a negative one a down-move. A single
EMA span carries it; `N = 1` (α = 1) is the freshest, unsmoothed read and the leg the sweep leans on
hardest. It is FALSIFIED if the EMA shows no forward signed information coefficient against byb's
mid-move at any span (a flat, sign-indifferent IC) — order-book imbalance then carries no directional edge.

Research: Cont, R., Kukanov, A. & Stoikov, S. (2014) 'The Price Impact of Order Book Events', Journal of
Financial Econometrics 12(1):47-88 — the order-flow-imbalance increment summed here is theirs, and they
document its (contemporaneous, here-tested-forward) linear relation to price moves.

See `AUTHORING.md` (this directory) for the EMA-type and inject/decay rules these obey.
"""
from __future__ import annotations

import numpy as np

from boba.ema import KernelMeanEMA
from boba.features._ofi import ofi_increment, ofi_leg
from boba.features.base import Config, FeatureSpec, ParamKind, Params, RawData, SharedData, register


def _ex(listing: str) -> str:
    """The short exchange key (leg key) for a full listing id, e.g. 'byb_eth_usdt_p' -> 'byb'."""
    return listing.split("_", 1)[0]


def vectorized(raw: RawData, shared: SharedData, config: Config, params: Params) -> dict[str, np.ndarray]:
    """{exchange -> E/W EMA of that venue's path-sum OFI flow at span N}, one value per `event_ts` (causal)."""
    n = params
    return {_ex(l): ofi_leg(raw.listings[l].front_levels, shared.clock, shared.event_ts, n)
            for l in config.all_listings}


class LiveOFIEma:
    """O(1) streaming build, one OFI per venue. Each venue carries one `KernelMeanEMA` E/W leg over its
    own path-sum OFI flow + its previous raw row; each book row accumulates one CKS increment, `refresh()`
    injects each venue's SUMMED increment, then decays all legs once iff a trade landed (shared clock)."""

    def __init__(self, config: Config, params: Params):
        self.exes = tuple(_ex(l) for l in config.all_listings)
        self.keys = self.exes
        self._key_of = {l: _ex(l) for l in config.all_listings}   # full listing -> short key
        self.leg = {ex: KernelMeanEMA(params) for ex in self.exes}
        self.prev = {ex: None for ex in self.exes}        # previous raw (bid, bid_qty, ask, ask_qty) per venue
        self.ts_e = {ex: 0.0 for ex in self.exes}
        self.ts_got = {ex: False for ex in self.exes}     # did a book change land this timestamp, per venue
        self.was_trade_present = False                    # did a trade land this timestamp? -> one decay

    def on_book(self, ev) -> None:                        # a venue book row -> accumulate its OFI increment vs the prev raw row
        ex = self._key_of.get(ev.listing)
        if ex is None:
            return
        bid, bq, ask, aq = ev.bid, ev.bid_qty, ev.ask, ev.ask_qty
        p = self.prev[ex]
        if p is not None:
            self.ts_e[ex] += ofi_increment(*p, bid, bq, ask, aq)   # scalar twin of ofi_stream
            self.ts_got[ex] = True
        self.prev[ex] = (bid, bq, ask, aq)

    def on_trade(self, ev) -> None:                       # any venue's trade -> just flag the timestamp (OFI is book-only)
        self.was_trade_present = True

    def refresh(self) -> None:                            # ONE per timestamp: inject each venue's SUMMED OFI, then decay AT MOST once
        traded, self.was_trade_present = self.was_trade_present, False
        for ex in self.exes:
            if self.ts_got[ex]:
                self.leg[ex].add(self.ts_e[ex])
                self.ts_e[ex] = 0.0; self.ts_got[ex] = False
        if traded:
            for ex in self.exes:
                self.leg[ex].tick()

    def value(self) -> dict[str, float]:
        return {ex: self.leg[ex].value() for ex in self.exes}   # E/W per venue, nan during warm-up


SPEC = FeatureSpec(
    name="ofi_ema",
    vectorized=vectorized,
    make_streaming=lambda config, params: LiveOFIEma(config, params),
    keys_for=lambda config, params: tuple(_ex(l) for l in config.all_listings),
    mirror=np.negative,   # signed order flow: reflecting the book swaps bid/ask -> the OFI increment negates
    param_kind=ParamKind.SINGLE,                             # params = N (a single EMA span)
)
register(SPEC)
