"""Parquet data loading.

Data lives directly in DATA_DIR (no session sub-dirs), written in blocks:
    holocron.{ts}.{idx}.{listing}.{data_type}.parquet
A "listing" is the full book token incl. the perp suffix, e.g. "bin_doge_usdt_p"
(perp) or "bin_doge_usdt" (spot). A "block" is the "holocron.{ts}.{idx}" prefix —
a contiguous ~24h chunk.

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
"""
from pathlib import Path
from typing import Literal, Optional

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


DATA_DIR: Optional[Path] = _resolve_data_dir()

DataType = Literal["front_levels", "trade", "fri"]


def list_blocks(listing: str = "bin_doge_usdt_p", data_type: DataType = "front_levels") -> list[str]:
    """Sorted distinct block ids ('holocron.{ts}.{idx}') present for a listing/type."""
    if DATA_DIR is None:
        raise RuntimeError("data_dir is not configured in settings.local.toml")
    blocks = {".".join(f.name.split(".")[:3])
              for f in DATA_DIR.glob(f"holocron.*.{listing}.{data_type}.parquet")}
    return sorted(blocks)


def load_block(block: str, listing: str, data_type: DataType) -> pl.DataFrame:
    """Load one block's file(s) for a book listing.

    ``block``   — a 'holocron.{ts}.{idx}' prefix (see :func:`list_blocks`).
    ``listing`` — full book token incl. the ``_p`` perp suffix, e.g. "bin_doge_usdt_p".
    """
    if DATA_DIR is None:
        raise RuntimeError("data_dir is not configured in settings.local.toml")
    files = sorted(DATA_DIR.glob(f"{block}.{listing}.{data_type}.parquet"))
    if not files:
        raise FileNotFoundError(f"No files: {block} / {listing} / {data_type} under {DATA_DIR}")
    return pl.concat([pl.read_parquet(f) for f in files])
