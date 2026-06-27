"""Shared path-sum level-1 Order-Flow-Imbalance primitives — the Cont–Kukanov–Stoikov (CKS) increment and
its sparse-flow stream, used by BOTH `ofi_ema` and `ofi_fast_slow`. Keeping the OFI maths here means a
change to the increment is ONE edit, not four, and the two features cannot drift apart on it.

  * `ofi_increment(...)` -- the scalar CKS L1 increment for ONE book change (the streaming `on_book` step).
  * `ofi_stream(front)`  -- the vectorized path-sum OFI flow: one summed increment per book-update timestamp
                            (the same algebra as `ofi_increment`, applied to consecutive raw rows).
  * `ofi_leg(front, …)`  -- one venue's OFI flow as an `E/W` EMA read live at every `event_ts`.

The increment is depth ADDED on the bid minus depth added on the ask. Research: Cont, R., Kukanov, A. &
Stoikov, S. (2014) 'The Price Impact of Order Book Events', Journal of Financial Econometrics 12(1):47-88.
"""
from __future__ import annotations

import numpy as np

from boba.features.base import FrontLevels
from boba.features.shared import flow_at


def ofi_increment(pb: float, pq: float, pa: float, paq: float,
                  bid: float, bq: float, ask: float, aq: float) -> float:
    """The scalar CKS L1 OFI increment for ONE book change, previous raw row `(pb, pq, pa, paq)` ->
    current `(bid, bq, ask, aq)`: bid-side depth added minus ask-side depth added. The streaming `on_book`
    step; `ofi_stream` is its vectorized twin (identical algebra over consecutive rows)."""
    return ((bq if bid >= pb else 0.0) - (pq if bid <= pb else 0.0)
            - (aq if ask <= pa else 0.0) + (paq if ask >= pa else 0.0))


def ofi_stream(front: FrontLevels) -> tuple[np.ndarray, np.ndarray]:
    """`front_levels` rows -> `(ts, summed_increment)`: one path-sum OFI sample per book-update timestamp.
    The CKS increment is formed for EVERY consecutive RAW row, then increments sharing an rx_time are
    SUMMED into one sample (records at one ns are ONE flow event)."""
    rx, bid, bq, ask, aq = front.rx, front.bid, front.bid_qty, front.ask, front.ask_qty
    pbp, pbq, pap, paq = bid[:-1], bq[:-1], ask[:-1], aq[:-1]     # previous raw row
    cbp, cbq, cap, caq = bid[1:], bq[1:], ask[1:], aq[1:]         # current raw row
    e = (np.where(cbp >= pbp, cbq, 0.0) - np.where(cbp <= pbp, pbq, 0.0)
         - np.where(cap <= pap, caq, 0.0) + np.where(cap >= pap, paq, 0.0))
    uniq, inv = np.unique(rx[1:], return_inverse=True)           # increments are stamped at the CUR row's rx
    return uniq, np.bincount(inv, weights=e)                     # value = the full intra-ns path sum per ts


def ofi_leg(front: FrontLevels, clock: np.ndarray, event_ts: np.ndarray, n: int) -> np.ndarray:
    """One venue's OFI EMA read as `E/W` at every `event_ts`: its path-sum OFI flow decayed once per
    trade-clock tick, read live (committed-per-tick + the partial epoch since the last tick)."""
    ofi_rx, ofi_e = ofi_stream(front)
    e = flow_at(clock, ofi_rx, ofi_e, event_ts, n)
    w = flow_at(clock, ofi_rx, np.ones(ofi_e.size), event_ts, n)
    return e / np.where(w == 0.0, np.nan, w)
