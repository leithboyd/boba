"""Pre-processed market data for one block: per-listing event arrays + target book."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


# Rolling window (samples) for the feed-latency baseline (rolling min of rx − exchange time).
_FEED_BASELINE_WINDOW: int = 30_000


@dataclass
class SessionData:
    """Pre-processed market data for one session/block.

    All per-listing dicts are keyed by listing token (e.g. 'bin_doge_usdt_p').
    """
    # ── Per-listing book snapshots ────────────────────────────────────────────
    listing_book_t:       dict[str, np.ndarray]   # rx timestamps (ns)
    listing_book_bid:     dict[str, np.ndarray]   # best-bid price
    listing_book_ask:     dict[str, np.ndarray]   # best-ask price
    listing_book_bid_qty: dict[str, np.ndarray]   # best-bid quantity
    listing_book_ask_qty: dict[str, np.ndarray]   # best-ask quantity

    # ── Per-listing trades ────────────────────────────────────────────────────
    trade_ts:          dict[str, np.ndarray]   # rx timestamps (ns)
    trade_exchange_ts: dict[str, np.ndarray]   # exchange timestamps (ns) — same per sweep
    trade_prc:         dict[str, np.ndarray]
    trade_qty:         dict[str, np.ndarray]
    trade_dir:         dict[str, np.ndarray]   # +1 buy / -1 sell

    # ── Per-listing feed latency excess ───────────────────────────────────────
    listing_feed_latency_excess_ns: dict[str, np.ndarray]

    # ── Target listing book + feed latency ────────────────────────────────────
    target_listing:        str
    book_t:                 np.ndarray
    book_bid:               np.ndarray
    book_ask:               np.ndarray
    book_mid:               np.ndarray
    feed_latency_raw_ns:    np.ndarray
    feed_latency_excess_ns: np.ndarray

    # ── Sorted union of all event timestamps ──────────────────────────────────
    all_rx: np.ndarray


def build_target_book(fl: dict[str, pl.DataFrame], listing: str) -> pl.DataFrame:
    """Sorted best-bid / best-ask / mid table for the target listing.

    Includes ``exchange_time`` for feed-latency computation.
    """
    return (
        fl[listing].sort("rx_time").select([
            "rx_time",
            "exchange_time",
            pl.col("bid_prc").alias("target_bid"),
            pl.col("ask_prc").alias("target_ask"),
            ((pl.col("bid_prc") + pl.col("ask_prc")) / 2).alias("target_mid"),
        ]).drop_nulls()
    )


def build_session_data(
    fl: dict[str, pl.DataFrame],
    td: dict[str, pl.DataFrame],
    listings: list[str],
    target_listing: str,
) -> SessionData:
    """Convert per-listing front-levels and trade frames into a :class:`SessionData`.

    The returned dicts are keyed by listing token; the target book/latency arrays
    come from ``target_listing``.
    """
    listing_book_t: dict[str, np.ndarray] = {}
    listing_book_bid: dict[str, np.ndarray] = {}
    listing_book_ask: dict[str, np.ndarray] = {}
    listing_book_bid_qty: dict[str, np.ndarray] = {}
    listing_book_ask_qty: dict[str, np.ndarray] = {}
    listing_feed_latency_excess_ns: dict[str, np.ndarray] = {}
    for e in listings:
        b = fl[e].sort("rx_time").drop_nulls(subset=["bid_prc", "ask_prc", "bid_qty", "ask_qty", "exchange_time"])
        listing_book_t[e]       = b["rx_time"].cast(pl.Int64).to_numpy()
        listing_book_bid[e]     = b["bid_prc"].to_numpy()
        listing_book_ask[e]     = b["ask_prc"].to_numpy()
        listing_book_bid_qty[e] = b["bid_qty"].to_numpy()
        listing_book_ask_qty[e] = b["ask_qty"].to_numpy()
        rx_t   = listing_book_t[e]
        exch_t = b["exchange_time"].cast(pl.Int64).to_numpy()
        feed_raw_e = rx_t - exch_t
        feed_baseline_e = (pl.Series(feed_raw_e.astype(np.float64))
                           .rolling_min(window_size=_FEED_BASELINE_WINDOW, min_samples=1).to_numpy())
        listing_feed_latency_excess_ns[e] = (feed_raw_e - feed_baseline_e).astype(np.int64)

    trade_ts: dict[str, np.ndarray] = {}
    trade_exchange_ts: dict[str, np.ndarray] = {}
    trade_prc: dict[str, np.ndarray] = {}
    trade_qty: dict[str, np.ndarray] = {}
    trade_dir: dict[str, np.ndarray] = {}
    for e in listings:
        t = td[e].sort("rx_time")
        trade_ts[e]          = t["rx_time"].cast(pl.Int64).to_numpy()
        trade_exchange_ts[e] = t["exchange_time"].cast(pl.Int64).to_numpy()
        trade_prc[e]         = t["prc"].to_numpy()
        trade_qty[e]         = t["qty"].to_numpy()
        trade_dir[e]         = np.where(t["aggressor"].to_numpy() == "Bid", 1.0, -1.0)

    all_rx_parts = []
    for e in listings:
        all_rx_parts.append(listing_book_t[e]); all_rx_parts.append(trade_ts[e])
    all_rx = np.sort(np.concatenate(all_rx_parts))

    target_book = build_target_book(fl, target_listing)
    book_t   = target_book["rx_time"].cast(pl.Int64).to_numpy()
    book_bid = target_book["target_bid"].to_numpy()
    book_ask = target_book["target_ask"].to_numpy()
    book_mid = target_book["target_mid"].to_numpy()
    book_exchange_t       = target_book["exchange_time"].cast(pl.Int64).to_numpy()
    feed_latency_raw_ns   = book_t - book_exchange_t
    feed_latency_baseline = (pl.Series(feed_latency_raw_ns.astype(np.float64))
                             .rolling_min(window_size=_FEED_BASELINE_WINDOW, min_samples=1).to_numpy())
    feed_latency_excess_ns = (feed_latency_raw_ns - feed_latency_baseline).astype(np.int64)

    return SessionData(
        listing_book_t=listing_book_t, listing_book_bid=listing_book_bid,
        listing_book_ask=listing_book_ask, listing_book_bid_qty=listing_book_bid_qty,
        listing_book_ask_qty=listing_book_ask_qty,
        listing_feed_latency_excess_ns=listing_feed_latency_excess_ns,
        trade_ts=trade_ts, trade_exchange_ts=trade_exchange_ts,
        trade_prc=trade_prc, trade_qty=trade_qty, trade_dir=trade_dir,
        target_listing=target_listing,
        book_t=book_t, book_bid=book_bid, book_ask=book_ask, book_mid=book_mid,
        feed_latency_raw_ns=feed_latency_raw_ns,
        feed_latency_excess_ns=feed_latency_excess_ns,
        all_rx=all_rx,
    )
