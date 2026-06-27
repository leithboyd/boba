"""`ofi_fast_slow` — the path-sum level-1 Order-Flow Imbalance, as a fast/slow oscillator.

WHAT it measures. Each time byb's top-of-book changes, the **OFI increment** (Cont–Kukanov–Stoikov)
counts the net depth added on the bid minus the ask: positive = net buying pressure. The **path-sum**
variant injects, per receive-timestamp, the SUM of the per-raw-row increments (the full intra-ns book
path) as ONE flow sample. We smooth that flow two ways — a fast and a slow `KernelMeanEMA` leg, each
read as `E/W` (a sparse flow per AUTHORING) — and take `fast − slow`: a sign-stable lean-vs-baseline
oscillator. It **fans out over every exchange** — one leg per venue, each built from that venue's OWN
book (byb's own order flow, plus each foreign venue's as a cross-venue lead); `params = (n_fast, n_slow)`.

WHY it might predict (falsifiable hypothesis). The L1 OFI increment counts net depth added to the bid
minus the ask, so its sign is the direction the book is being pushed. A *fresh* fast-minus-slow lean —
recent order flow leaning harder than its own slower baseline — should anticipate byb's next move in that
direction (positive lean → up). Falsified if the feature shows no forward signed information coefficient.

RESEARCH. Cont, R., Kukanov, A. & Stoikov, S. (2014) 'The Price Impact of Order Book Events', Journal
of Financial Econometrics 12(1):47-88.

Two implementations of the same maths, tied by `boba.research.screening.parity_check`:
  - `vectorized(raw, shared, config, params)` -> {exchange -> (fast − slow) value per `shared.event_ts`}
  - `LiveOFIFastSlow`                         -> the O(1) streaming build (two `KernelMeanEMA` E/W legs per venue)

Mirror augmentation: OFI is signed order flow, ODD under the reflection of the tape through byb's mid
(the bid/ask sides swap, so the increment negates — see `AUTHORING.md` → Mirror augmentation and the
commutation test). So `SPEC.mirror` is `np.negative`.

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
    """{exchange -> (fast − slow) E/W EMAs of that venue's path-sum OFI flow}, one value per `event_ts` (causal)."""
    n_fast, n_slow = params
    out: dict[str, np.ndarray] = {}
    for l in config.all_listings:
        front = raw.listings[l].front_levels
        out[_ex(l)] = (ofi_leg(front, shared.clock, shared.event_ts, n_fast)
                       - ofi_leg(front, shared.clock, shared.event_ts, n_slow))
    return out


class LiveOFIFastSlow:
    """O(1) streaming build, one OFI per venue. Each venue carries two `KernelMeanEMA` E/W legs over its
    own path-sum OFI flow + its previous raw row; each book row accumulates one CKS increment, `refresh()`
    injects each venue's SUMMED increment, then decays all legs once iff a trade landed (shared clock)."""

    def __init__(self, config: Config, params: Params):
        n_fast, n_slow = params
        self.exes = tuple(_ex(l) for l in config.all_listings)
        self.keys = self.exes
        self._key_of = {l: _ex(l) for l in config.all_listings}   # full listing -> short key
        self.leg_f = {ex: KernelMeanEMA(n_fast) for ex in self.exes}
        self.leg_s = {ex: KernelMeanEMA(n_slow) for ex in self.exes}
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
                self.leg_f[ex].add(self.ts_e[ex]); self.leg_s[ex].add(self.ts_e[ex])
                self.ts_e[ex] = 0.0; self.ts_got[ex] = False
        if traded:
            for ex in self.exes:
                self.leg_f[ex].tick(); self.leg_s[ex].tick()

    def value(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for ex in self.exes:
            f, s = self.leg_f[ex].value(), self.leg_s[ex].value()
            out[ex] = (f - s) if (f == f and s == s) else float("nan")   # nan in either leg -> undefined
        return out


SPEC = FeatureSpec(
    name="ofi_fast_slow",
    vectorized=vectorized,
    make_streaming=lambda config, params: LiveOFIFastSlow(config, params),
    keys_for=lambda config, params: tuple(_ex(l) for l in config.all_listings),
    mirror=np.negative,   # signed order flow: reflecting the book swaps bid/ask -> the OFI increment negates
    param_kind=ParamKind.FAST_SLOW,                          # params = (n_fast, n_slow)
)
register(SPEC)
