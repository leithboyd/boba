"""Memory-aware chunk planning for the v2 sequential builder.

The compute unit is a *chunk*; the cache unit stays the *block* (see
docs/v2_dataset_design.md §5). A chunk is a contiguous span of block rows processed in one
pass — small blocks are **coalesced** up to a memory budget, and an oversized block is
**split** into sub-chunks below it. Because continuous-carry semantics make a block's output
independent of how it was chunked, the budget steers only peak memory / throughput, never
the result — so chunk planning is deliberately kept OUT of ``config_str`` / ``cache_key``.

This lets the same build run on a 128 GB box (big chunks) or a 16 GB laptop (split chunks)
and produce byte-identical caches.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# Peak memory of one chunk is dominated by the (event_ticks × F) float32 output matrix;
# the rest (full-grid microprice transients, shared state, cost tables) scales with it.
# Calibrated to the v1 reference point: ~25 GB peak at 13M event-ticks × 225 cols
# (x_out ≈ 11.7 GB → overhead ≈ 2.1); 2.2 leaves a little headroom.
_MEM_OVERHEAD = 2.2


@dataclass(frozen=True)
class Segment:
    """A contiguous row range ``[start, end)`` (event-tick indices) of one block."""
    block: str
    start: int
    end: int

    @property
    def n_ticks(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Chunk:
    """One compute pass: either several whole blocks (coalesced) or one block's sub-range
    (split). Segments are contiguous and in dataset order."""
    segments: tuple[Segment, ...]

    @property
    def n_ticks(self) -> int:
        return sum(s.n_ticks for s in self.segments)

    @property
    def blocks(self) -> tuple[str, ...]:
        return tuple(s.block for s in self.segments)


def chunk_capacity_ticks(mem_budget_gb: float, n_features: int,
                         overhead: float = _MEM_OVERHEAD) -> int:
    """Max event-ticks one chunk may hold under ``mem_budget_gb`` for an ``n_features``-wide
    build. Lower budget → smaller capacity → more (and possibly split) chunks."""
    if mem_budget_gb <= 0:
        raise ValueError(f"mem_budget_gb must be positive, got {mem_budget_gb}")
    per_tick_bytes = overhead * max(n_features, 1) * 4          # float32 output, + transients
    return max(1, int((mem_budget_gb * 1e9) / per_tick_bytes))


def _split_block(block: str, n_ticks: int, capacity: int) -> list[Chunk]:
    """Split one oversized block into ``ceil(n/capacity)`` balanced sub-chunks, each ≤ capacity,
    together covering ``[0, n_ticks)`` contiguously."""
    n_parts = math.ceil(n_ticks / capacity)
    part = math.ceil(n_ticks / n_parts)                        # balanced, still ≤ capacity
    out: list[Chunk] = []
    start = 0
    while start < n_ticks:
        end = min(start + part, n_ticks)
        out.append(Chunk((Segment(block, start, end),)))
        start = end
    return out


def plan_chunks(blocks: list[tuple[str, int]], capacity: int) -> list[Chunk]:
    """Pack ``blocks`` (ordered ``(block_id, n_event_ticks)``) into chunks ≤ ``capacity``.

    - Contiguous blocks that fit together are **coalesced** into one chunk.
    - A block larger than ``capacity`` is **split** into its own sub-chunks (not coalesced
      with neighbours, so the split boundaries stay clean).

    Every block row is covered exactly once, in order. The result is the compute plan; the
    cache is still written per block (a split block's sub-chunks reassemble into its files).
    """
    if capacity < 1:
        raise ValueError(f"capacity must be ≥ 1, got {capacity}")
    chunks: list[Chunk] = []
    cur: list[Segment] = []
    cur_ticks = 0

    def flush():
        nonlocal cur, cur_ticks
        if cur:
            chunks.append(Chunk(tuple(cur)))
            cur, cur_ticks = [], 0

    for block, n in blocks:
        if n < 0:
            raise ValueError(f"block {block!r} has negative ticks {n}")
        if n > capacity:                       # oversized → split, on its own
            flush()
            chunks.extend(_split_block(block, n, capacity))
            continue
        if cur_ticks + n > capacity:           # wouldn't fit → start a new chunk
            flush()
        cur.append(Segment(block, 0, n))
        cur_ticks += n
    flush()
    return chunks


def plan_chunks_for(mem_budget_gb: float, n_features: int,
                    blocks: list[tuple[str, int]]) -> list[Chunk]:
    """Convenience: derive capacity from the budget + feature count, then plan."""
    return plan_chunks(blocks, chunk_capacity_ticks(mem_budget_gb, n_features))
