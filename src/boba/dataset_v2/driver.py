"""Real-data driver for the v2 sequential builder.

Wires ``boba.io`` block loading → ``SessionData`` → the chunk engine. The pure-compute pieces
(engine, cache, planner) are unit-tested on synthetic data; the ``build_from_blocks`` glue
here needs ``DATA_DIR`` and is exercised end-to-end against real data. ``build_dataset_v2``
itself is loader-agnostic and unit-testable.
"""
from __future__ import annotations

from pathlib import Path

from boba.dataset_v2.engine import MS, Block, build_chunked
from boba.dataset_v2.raw import DatasetRawConfig
from boba.dataset_v2.session_data import SessionData


def tail_window_ns_for(cfg: DatasetRawConfig, k: int = 25, floor_ms: int = 1000) -> int:
    """A carried-tail window that re-warms the widest selected feature. ``k·max_span`` ms (ms
    is also a fine proxy for event spans on dense streams). Generous by design — an over-long
    tail is harmless (the EMA forgets it); too short would under-warm."""
    spans = [u.n for u in cfg.expanded().units if u.n is not None]
    return int(max(floor_ms, k * (max(spans) if spans else floor_ms)) * MS)


def build_dataset_v2(cfg: DatasetRawConfig, blocks: list[Block], cache_dir: Path, *,
                     tail_window_ns: int | None = None, load=None, verbose: bool = True) -> list[str]:
    """Build the per-column cache for ``blocks`` (ordered). Derives the cache directory key
    (``cfg.grid_hash()``) and the tail window from ``cfg``, then runs the chunk engine
    (memory-bounded via ``cfg.mem_budget_gb``). ``load(block_id) -> SessionData`` supplies data
    on demand when ``Block.session`` is None."""
    if tail_window_ns is None:
        tail_window_ns = tail_window_ns_for(cfg)
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    return build_chunked(blocks, cfg, Path(cache_dir), cfg.grid_hash(),
                         tail_window_ns=tail_window_ns, load=load, log=log)


# ── Integration glue (needs DATA_DIR) ───────────────────────────────────────────────────────

def _load_session(cfg: DatasetRawConfig, block: str) -> SessionData:
    import polars as pl
    from boba import io as _io
    from boba.dataset_v2.session_data import build_session_data
    listings = list(cfg.listings)
    fl = {e: _io.load_block(block, e, "front_levels") for e in listings}
    td = {e: _io.load_block(block, e, "trade").filter((pl.col("prc") > 0) & (pl.col("qty") > 0))
          for e in listings}
    return build_session_data(fl, td, listings, cfg.target_listing)


def _bounds_ms(sd: SessionData) -> tuple[int, int]:
    starts = [a[0] for a in sd.listing_book_t.values() if len(a)]
    ends = [a[-1] for a in sd.listing_book_t.values() if len(a)]
    return int(min(starts) // MS), int(max(ends) // MS) + 1


def build_from_blocks(cfg: DatasetRawConfig, block_ids: list[str], cache_dir: Path, *,
                      verbose: bool = True) -> list[str]:
    """Full real-data path: derive each block's [start, end) bounds (one bounded pass), then
    build with on-demand loading. Blocks must be in dataset order and contiguous."""
    metas: list[Block] = []
    for bid in block_ids:                                  # bounded: one block loaded at a time
        s, e = _bounds_ms(_load_session(cfg, bid))
        metas.append(Block(id=bid, start_ms=s, end_ms=e))
    return build_dataset_v2(cfg, metas, cache_dir, load=lambda bid: _load_session(cfg, bid),
                            verbose=verbose)
