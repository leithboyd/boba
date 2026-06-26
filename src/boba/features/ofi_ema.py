"""`ofi_ema` — a single EMA of byb's path-sum level-1 Order-Flow Imbalance.

The raw-EMA sibling of `ofi_fast_slow`: the same path-sum OFI flow (per book change, the SUM of the
Cont–Kukanov–Stoikov increments at that timestamp), smoothed by ONE `KernelMeanEMA` leg read as `E/W`
(a sparse flow per AUTHORING). It **fans out over every exchange** — one leg per venue, each from that
venue's OWN book; `params = N` (the EMA span). `N = 1` is the freshest, unsmoothed OFI.

Two implementations of the same maths, tied by `boba.research.screening.parity_check`:
  - `vectorized(ctx, params)` -> {exchange -> feature vector on the grid}   (offline; reuses `ctx._flow_at`)
  - `LiveOFIEma`              -> the O(1) streaming build (one `KernelMeanEMA` E/W leg per venue)

Mirror augmentation: OFI is signed order flow, ODD under the reflection of the tape through byb's mid
(the bid/ask sides swap, so the increment negates). So `SPEC.mirror` is `np.negative`.

See `AUTHORING.md` (this directory) for the EMA-type and inject/decay rules these obey.
"""
from __future__ import annotations

import numpy as np

from boba.ema import KernelMeanEMA
from boba.features.base import FeatureSpec, Params, register
from boba.research.screening import ScreeningContext


def _ofi_stream(book: tuple) -> tuple[np.ndarray, np.ndarray]:
    """`(rx, bid, bid_qty, ask, ask_qty)` raw front_levels rows -> `(ts, summed_increment)`: one path-sum
    OFI sample per book-update timestamp. The CKS increment is formed for EVERY consecutive RAW row, then
    increments sharing an rx_time are SUMMED into one sample (records at one ns are ONE flow event)."""
    rx, bid, bq, ask, aq = book
    pbp, pbq, pap, paq = bid[:-1], bq[:-1], ask[:-1], aq[:-1]     # previous raw row
    cbp, cbq, cap, caq = bid[1:], bq[1:], ask[1:], aq[1:]         # current raw row
    e = (np.where(cbp >= pbp, cbq, 0.0) - np.where(cbp <= pbp, pbq, 0.0)
         - np.where(cap <= pap, caq, 0.0) + np.where(cap >= pap, paq, 0.0))
    uniq, inv = np.unique(rx[1:], return_inverse=True)           # increments are stamped at the CUR row's rx
    return uniq, np.bincount(inv, weights=e)                     # value = the full intra-ns path sum per ts


def _exchanges(ctx: ScreeningContext) -> tuple[str, ...]:
    """Every venue we build an OFI leg for: the target plus each foreign source (its OWN-book OFI)."""
    return (ctx.target.split("_", 1)[0],) + tuple(ctx.sources)


def vectorized(ctx: ScreeningContext, params: Params) -> dict[str, np.ndarray]:
    """{exchange -> E/W EMA of that venue's path-sum OFI flow at span N}, on the anchor grid (causal)."""
    n = params
    out: dict[str, np.ndarray] = {}
    for ex in _exchanges(ctx):
        ofi_rx, ofi_e = _ofi_stream(ctx._books[ex])
        E = ctx._flow_at(ctx.anchor_ts, ofi_e, n, src_rx=ofi_rx)
        W = ctx._flow_at(ctx.anchor_ts, np.ones(ofi_e.size), n, src_rx=ofi_rx)
        out[ex] = E / np.where(W == 0.0, np.nan, W)
    return out


class LiveOFIEma:
    """O(1) streaming build, one OFI per venue. Each venue carries one `KernelMeanEMA` E/W leg over its
    own path-sum OFI flow + its previous raw row; each book row accumulates one CKS increment, `refresh()`
    injects each venue's SUMMED increment, then decays all legs once iff a trade landed (shared clock)."""

    def __init__(self, ctx: ScreeningContext, params: Params):
        coin = ctx.coin
        self.exes = _exchanges(ctx)
        self.keys = self.exes
        self._key_of = {f"{ex}_{coin}": ex for ex in self.exes}   # full listing -> short key
        self.fuse_trades = frozenset()                    # OFI is book-only — no trade fusion
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
            pb, pq, pa, paq = p
            e = ((bq if bid >= pb else 0.0) - (pq if bid <= pb else 0.0)
                 - (aq if ask <= pa else 0.0) + (paq if ask >= pa else 0.0))
            self.ts_e[ex] += e; self.ts_got[ex] = True
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
    make_streaming=lambda ctx, params: LiveOFIEma(ctx, params),
    keys_for=lambda ctx, params: (ctx.target.split("_", 1)[0],) + tuple(ctx.sources),
    mirror=np.negative,   # signed order flow: reflecting the book swaps bid/ask -> the OFI increment negates
)
register(SPEC)
