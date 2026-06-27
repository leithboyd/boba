"""Parquet data loading.

Data lives directly in DATA_DIR (no session sub-dirs), written in blocks:
    holocron.{ts}.{idx}.{listing}.{data_type}.parquet
A "listing" is the full book token incl. the perp suffix, e.g. "bin_doge_usdt_p"
(perp) or "bin_doge_usdt" (spot). A "block" is the "holocron.{ts}.{idx}" prefix —
a contiguous ~24h chunk.

Synthetic data types (computed, not captured) live under SYNTHETIC_DATA_DIR — a
parallel tree mirroring DATA_DIR's structure, same block/listing filenames, new
data_type suffix. They are built lazily on first access and cached: ``load_block``
computes from the raw streams, writes the parquet under SYNTHETIC_DATA_DIR, and
returns it; subsequent loads read the cache.
  * ``merged_levels`` — the trade-augmented ("merged") price stream from
    notebooks/02_merged_price_stream.ipynb. Per side, hold the price of the
    newest-by-exchange_time event among {BBO snapshots, qualifying trades}
    (buy/'Bid'-aggressor → ask, sell/'Ask' → bid; trades filtered to prc>0 & qty>0).
    A fresher top-of-book *price* feed (big win for the slow byb/okx feeds; it HURTS
    bin's already-sub-ms PERP feed, so it is DISALLOWED for bin perp — bin spot is
    slower and ~neutral, so allowed). The aggressor→side convention is VENUE-SPECIFIC:
    Binance SPOT inverts it — a 'Bid'-aggressor trade is a SELL that hits the bid
    (verified across all coins) — while every other listing is standard ('Bid' lifts
    the ask). It is PRICE-ONLY: there are deliberately
    no quantity columns, because the merge does not inform size (carrying the last
    snapshot's qty forward would be a stale-data trap). Schema: rx_time, bid_prc,
    ask_prc, bid_exchange_time, ask_exchange_time (one row per distinct rx event).
    The two sides are fused INDEPENDENTLY (each newest-by-exchange_time), which can momentarily CROSS
    (ask < bid, ~0.15% of rows). The result is UN-CROSSED before return: where crossed, the side with
    the newer exchange_time is trusted and the stale side is pushed one TICK past it (`tick_size`,
    raising if the listing is not configured), so `ask >= bid` always holds.

Known data quirks (consumers beware):
  * BBO (front_levels) cadence is venue-dependent. On eth_usdt_p, bin snapshots are
    sub-millisecond, but byb/okx only refresh every ~10-20 ms (p90 100-160 ms) — so
    byb/okx top-of-book is stale between snapshots. Fusing trade prints (hold the
    newest-by-exchange_time price per side) gives a fresher *price-only* stream that
    removes ~30-50% of byb/okx next-snapshot mid error, but hurts bin (its quote is
    already fresher than its trades). Quantities stay snapshot-stale either way.
    Holds across the archive on quiet and busy days — see
    notebooks/02_merged_price_stream.ipynb.
  * Binance perp (bin_*_p) `trade` streams carry ~0.25% rows with prc == 0 AND
    qty == 0, at valid contiguous trade-ids. These are USD-M futures aggTrade
    entries for insurance-fund / ADL fills, which Binance excludes from aggregation
    (spot and byb/okx are clean). They are NOT market fills — filter `prc > 0`
    (equivalently `qty > 0`) before any price/volume use.
  * **Binance SPOT inverts the `aggressor` convention.** On Binance SPOT (bin_* with
    no `_p` suffix) a trade with `aggressor == "Bid"` is a SELL that hits the bid, and
    `"Ask"` is a BUY that lifts the ask — the OPPOSITE of every other listing (all
    perp, and byb/okx spot), where `"Bid"` = buy/lifts-ask. Verified empirically across
    all coins (Binance maps `isBuyerMaker` the other way for spot vs perp). The inverted
    venues are configured by the `aggressor_inverted` setting (default `["bin_spot"]`).
    Consequence: any code that assumes `aggressor=="Bid" → buy/+1` uniformly is WRONG on
    Binance spot — e.g. `dataset/session_data.py` builds `trade_dir = where(aggressor==
    "Bid", +1, -1)`, so its trade-direction sign is inverted for bin spot. Use
    `_trade_lifts_ask(listing, agg)` / `_aggressor_inverted(listing)` rather than
    assuming. (The merged_levels builder already does.)
"""
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import polars as pl

from boba.settings import SETTINGS, PROJECT_ROOT


def _resolve_data_dir() -> Optional[Path]:
    """Resolve the data root from settings. Returns ``None`` when ``data_dir`` is
    not configured (settings.local.toml at the project root), so importing
    :mod:`boba.io` never breaks on a machine without data — the loaders raise a
    clear error if called while this is ``None``.
    """
    raw = SETTINGS.get("data_dir")
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _resolve_synthetic_data_dir() -> Optional[Path]:
    """Resolve the synthetic (computed-and-cached) data root: a parallel tree
    mirroring ``data_dir``'s structure. Uses the explicit ``synthetic_data_dir``
    setting if present, else derives it from ``data_dir`` by swapping the
    ``mando_data`` path segment for ``synthetic_mando_data`` (so e.g.
    ``.../bob/mando_data/corelliantkdata1/holocron`` →
    ``.../bob/synthetic_mando_data/corelliantkdata1/holocron``). Returns ``None``
    when neither is resolvable — callers then compute without caching.
    """
    raw = SETTINGS.get("synthetic_data_dir")
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p.resolve()
    if DATA_DIR is None:
        return None
    parts = list(DATA_DIR.parts)
    if "mando_data" in parts:
        parts[parts.index("mando_data")] = "synthetic_mando_data"
        return Path(*parts)
    return None


DATA_DIR: Optional[Path] = _resolve_data_dir()
SYNTHETIC_DATA_DIR: Optional[Path] = _resolve_synthetic_data_dir()

# Captured straight off the feed.
RawDataType = Literal["front_levels", "trade", "fri"]
# Computed from the raw streams and cached under SYNTHETIC_DATA_DIR (see module docstring).
SyntheticDataType = Literal["merged_levels"]
DataType = Literal["front_levels", "trade", "fri", "merged_levels"]

_SYNTHETIC_TYPES = ("merged_levels",)


# Trade `aggressor` convention is venue-specific (see the BINANCE-SPOT quirk above). A
# listing whose "{exchange}_{spot|perp}" token is in the `aggressor_inverted` setting
# INVERTS the standard convention: there a 'Bid'-aggressor trade is a SELL (hits bid).
_AGGRESSOR_INVERTED: set = set(SETTINGS.get("aggressor_inverted", ["bin_spot"]))


def _venue_market(listing: str) -> str:
    return f"{listing.split('_', 1)[0]}_{'perp' if listing.endswith('_p') else 'spot'}"


def _aggressor_inverted(listing: str) -> bool:
    return _venue_market(listing) in _AGGRESSOR_INVERTED


# price tick (minimum increment) per listing — exchange reference data, kept in `tick_sizes.toml` at
# the project root (verified from the data). Used e.g. to un-cross the merged book one tick from the
# fresh side. `tick_size(listing)` looks it up (raising on an unconfigured listing).
def _load_tick_sizes() -> dict:
    import tomllib

    from boba.settings import PROJECT_ROOT

    path = PROJECT_ROOT / "tick_sizes.toml"
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f).get("tick_size", {})


_TICK_SIZES: dict = _load_tick_sizes()


def tick_size(listing: str) -> float:
    """The price tick (minimum increment) for `listing`, from `tick_sizes.toml`. Raises if the listing
    is not configured — the tick is exchange reference data, so infer it from a block's `front_levels`
    prices (`min` positive increment) and add it to the file rather than guessing at call time."""
    try:
        return _TICK_SIZES[listing]
    except KeyError:
        raise KeyError(
            f"no tick_size configured for {listing!r}; add it to tick_sizes.toml (have {sorted(_TICK_SIZES)})"
        ) from None


def _merged_levels_blocked(listing: str) -> bool:
    # bin's PERP feed is sub-millisecond fresh, so the merge feeds a staler trade price
    # into a live quote and HURTS it (validated: next-snapshot R2 ~ -13). bin spot is
    # slower/neutral, so only bin perp is blocked.
    return listing.startswith("bin_") and listing.endswith("_p")


def list_blocks(listing: str = "bin_doge_usdt_p", data_type: DataType = "front_levels") -> list[str]:
    """Sorted distinct block ids ('holocron.{ts}.{idx}') for a listing/type.

    For a synthetic type this is the set of blocks that *can be built* — i.e. those
    present in every raw stream the build needs — not just those already cached.
    """
    if DATA_DIR is None:
        raise RuntimeError("data_dir is not configured in settings.local.toml")
    if data_type == "merged_levels":   # needs both front_levels and trade for the block
        if _merged_levels_blocked(listing):
            return []                   # disallowed for bin perp (merge degrades the sub-ms feed)
        return sorted(set(list_blocks(listing, "front_levels")) & set(list_blocks(listing, "trade")))
    blocks = {".".join(f.name.split(".")[:3])
              for f in DATA_DIR.glob(f"holocron.*.{listing}.{data_type}.parquet")}
    return sorted(blocks)


def load_block(block: str, listing: str, data_type: DataType) -> pl.DataFrame:
    """Load one block's file(s) for a book listing.

    ``block``   — a 'holocron.{ts}.{idx}' prefix (see :func:`list_blocks`).
    ``listing`` — full book token incl. the ``_p`` perp suffix, e.g. "bin_doge_usdt_p".

    Rows come back **rx_time-sorted**: the raw parquet files are stored in arrival
    order and :func:`list_blocks` is chronological, so a single block — or an in-order
    concat of consecutive blocks — needs no re-sort. Synthetic types are built
    rx-sorted too. (To merge streams across *different* listings you still sort the
    combined stream, since their events interleave in time.)

    Synthetic types are built lazily and cached: on a cache miss the stream is
    computed from the raw streams, written under SYNTHETIC_DATA_DIR, then returned
    (when SYNTHETIC_DATA_DIR is unresolvable it is computed but not cached).
    """
    if DATA_DIR is None:
        raise RuntimeError("data_dir is not configured in settings.local.toml")
    if data_type in _SYNTHETIC_TYPES:
        if data_type == "merged_levels" and _merged_levels_blocked(listing):
            raise ValueError(
                f"merged_levels is disabled for {listing!r}: bin's PERP BBO is already sub-millisecond "
                "fresh, so the merge feeds a staler trade price into a live quote and hurts prediction. "
                "Use the raw 'front_levels' stream instead.")
        return _load_or_build_synthetic(block, listing, data_type)
    files = sorted(DATA_DIR.glob(f"{block}.{listing}.{data_type}.parquet"))
    if not files:
        raise FileNotFoundError(f"No files: {block} / {listing} / {data_type} under {DATA_DIR}")
    return pl.concat([pl.read_parquet(f) for f in files])  # files stored rx_time-sorted -> no re-sort needed


# ── Synthetic streams: lazy compute + cache ──────────────────────────────────

_BUILDERS = {}   # data_type -> builder(block, listing) -> pl.DataFrame; populated below


def _load_or_build_synthetic(block: str, listing: str, data_type: SyntheticDataType) -> pl.DataFrame:
    if SYNTHETIC_DATA_DIR is not None:
        out = SYNTHETIC_DATA_DIR / f"{block}.{listing}.{data_type}.parquet"
        if out.exists():
            return pl.read_parquet(out)
    df = _BUILDERS[data_type](block, listing)
    if SYNTHETIC_DATA_DIR is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".parquet.tmp")   # write-then-rename so a crash can't leave a partial cache file
        df.write_parquet(tmp)
        tmp.rename(out)
    return df


_I64MIN = np.iinfo(np.int64).min


def _merge_side(snap_rx, snap_ex, snap_px, tr_rx, tr_ex, tr_px):
    """One price side of the merge: combine the side's BBO snapshot prices with its
    relevant trades, in rx (then arrival-sequence) order, holding the price of the
    newest-by-exchange_time event seen so far. On an exchange_time TIE the LATEST event
    in sequence wins -- so a single aggressive order that sweeps several levels at one
    exchange_time (its prints share it, arriving in order) resolves to its LAST/deepest
    print, not its first (holding the first would understate the swept side and pull the
    mid the wrong way). Returns (rx_sorted, held_price, held_exchange_time)."""
    rx = np.concatenate([snap_rx, tr_rx])
    ex = np.concatenate([snap_ex, tr_ex])
    px = np.concatenate([snap_px, tr_px])
    o = np.argsort(rx, kind="stable")           # stable: snapshot before trade on equal rx; arrival order kept within ties
    rx, ex, px = rx[o], ex[o], px[o]
    run = np.maximum.accumulate(ex)
    pos = np.where(ex >= run, np.arange(len(ex)), -1)   # the LATEST event at the running-max exchange_time (ties -> last)
    held = np.maximum.accumulate(pos)                   # carry that index forward (>=0 from i=0)
    return rx, px[held], ex[held]


def _trade_lifts_ask(listing: str, agg: np.ndarray) -> np.ndarray:
    """Mask of trades that hit (lift) the ASK. The aggressor→side convention is
    venue-specific — configured by the `aggressor_inverted` setting (see the
    BINANCE-SPOT quirk in the module docstring): inverted venues have 'Bid'-aggressor =
    SELL (hits the bid); standard venues have 'Bid' lift the ask."""
    return (agg == "Ask") if _aggressor_inverted(listing) else (agg == "Bid")


def _uncross_book(bid, ask, bid_ex, ask_ex, tick):
    """Un-cross a fused book: where `ask < bid` (the two sides fused independently and crossed), TRUST the
    side with the newer `exchange_time` and push the STALE side exactly one `tick` past it — the freshest
    valid (non-crossed) book. Ties (`ask_ex == bid_ex`) treat the ask as fresher. Returns `(bid, ask)`.
    The scalar online twin is `boba.features.streaming.uncross_quote` (parity-tied)."""
    crossed = ask < bid
    ask_fresher = ask_ex >= bid_ex                       # the side updated at least as recently is trusted
    new_bid = np.where(crossed & ask_fresher, ask - tick, bid)
    new_ask = np.where(crossed & ~ask_fresher, bid + tick, ask)
    return new_bid, new_ask


def _build_merged_levels(block: str, listing: str) -> pl.DataFrame:
    """Build the trade-augmented price stream (see module docstring). Price-only:
    no quantity columns by design (the merge does not inform size). The independently-fused sides can
    cross, so the result is UN-CROSSED with the listing's tick (`tick_size`, raising if not configured)."""
    fl = (load_block(block, listing, "front_levels")
          .select("rx_time", "exchange_time", "bid_prc", "ask_prc")
          .drop_nulls().sort("rx_time"))
    td = (load_block(block, listing, "trade")
          .select("rx_time", "exchange_time", "aggressor", "prc", "qty")
          .filter((pl.col("prc") > 0) & (pl.col("qty") > 0))   # drop bin-perp insurance/ADL zero prints
          .drop_nulls().sort("rx_time"))
    s_rx = fl["rx_time"].cast(pl.Int64).to_numpy()
    s_ex = fl["exchange_time"].cast(pl.Int64).to_numpy()
    bid, ask = fl["bid_prc"].to_numpy(), fl["ask_prc"].to_numpy()
    t_rx = td["rx_time"].cast(pl.Int64).to_numpy()
    t_ex = td["exchange_time"].cast(pl.Int64).to_numpy()
    t_px = td["prc"].to_numpy()
    lift_ask = _trade_lifts_ask(listing, td["aggressor"].to_numpy())   # venue-specific aggressor→side
    hit_bid = ~lift_ask

    ask_rx, ask_px, ask_ex = _merge_side(s_rx, s_ex, ask, t_rx[lift_ask], t_ex[lift_ask], t_px[lift_ask])
    bid_rx, bid_px, bid_ex = _merge_side(s_rx, s_ex, bid, t_rx[hit_bid], t_ex[hit_bid], t_px[hit_bid])

    # one row per distinct rx, from the first snapshot on (before it neither side is defined)
    all_rx = np.concatenate([s_rx, t_rx])
    uniq = np.unique(all_rx[all_rx >= s_rx[0]])      # sorted unique; both sides defined since s_rx[0] feeds both
    ai = np.searchsorted(ask_rx, uniq, "right") - 1
    bi = np.searchsorted(bid_rx, uniq, "right") - 1
    b, a = _uncross_book(bid_px[bi], ask_px[ai], bid_ex[bi], ask_ex[ai], tick_size(listing))
    return pl.DataFrame({
        "rx_time": uniq,
        "bid_prc": b,
        "ask_prc": a,
        "bid_exchange_time": bid_ex[bi],
        "ask_exchange_time": ask_ex[ai],
    }).with_columns(
        pl.col("rx_time").cast(pl.Datetime("ns", "UTC")),
        pl.col("bid_exchange_time").cast(pl.Datetime("ns", "UTC")),
        pl.col("ask_exchange_time").cast(pl.Datetime("ns", "UTC")),
    )


_BUILDERS["merged_levels"] = _build_merged_levels
