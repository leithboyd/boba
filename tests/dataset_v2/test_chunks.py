"""v2: memory-aware chunk planner + the invariant that perf/memory config never affects
the cache key."""
from __future__ import annotations

from collections import defaultdict

import pytest

from boba.dataset_v2 import DatasetRawConfig, col
from boba.dataset_v2.chunks import (
    Chunk, Segment, chunk_capacity_ticks, plan_chunks, plan_chunks_for,
)


def _cfg(**kw) -> DatasetRawConfig:
    return DatasetRawConfig(columns=(col("{LISTING}_ema_ofi_{N}b", LISTING="bin", N=[3, 10]),), **kw)


# ── Perf/memory config must NOT touch config_str / cache_key ───────────────────────────────

class TestPerfConfigNotHashed:
    def test_workers_and_mem_budget_excluded(self):
        base = _cfg()
        # a beefy box vs a small laptop — different parallelism + memory budget
        small = _cfg(n_workers=2, mem_budget_gb=8.0)
        big = _cfg(n_workers=64, mem_budget_gb=900.0)
        assert base.config_str() == small.config_str() == big.config_str()
        assert base.cache_key() == small.cache_key() == big.cache_key()

    def test_output_affecting_change_still_moves_the_key(self):
        # sanity: the key IS sensitive to things that change output
        base = _cfg()
        assert base.config_str() != _cfg(horizon_ms=999.0).config_str()
        assert base.config_str() != DatasetRawConfig(
            columns=(col("{LISTING}_microprice", LISTING="bin"),)).config_str()

    def test_defaults_target_the_dev_box(self):
        c = _cfg()
        assert c.n_workers == 18           # ≈ the 18 cores
        assert c.mem_budget_gb == 96.0     # headroom under 128 GB


# ── chunk_capacity_ticks ───────────────────────────────────────────────────────────────────

class TestCapacity:
    def test_monotonic_in_budget(self):
        assert chunk_capacity_ticks(8, 100) < chunk_capacity_ticks(96, 100)

    def test_inverse_in_features(self):
        assert chunk_capacity_ticks(96, 400) < chunk_capacity_ticks(96, 100)

    def test_positive_and_floored_at_one(self):
        assert chunk_capacity_ticks(96, 100) > 0
        # a pathologically tiny budget never returns 0 (would stall the planner)
        assert chunk_capacity_ticks(1e-9, 10_000) == 1

    def test_rejects_nonpositive_budget(self):
        with pytest.raises(ValueError):
            chunk_capacity_ticks(0, 100)

    def test_dev_box_fits_a_few_blocks(self):
        # 96 GB / ~225 cols → tens of millions of ticks; a ~13M-tick block coalesces ~3/chunk
        cap = chunk_capacity_ticks(96, 225)
        assert 2 <= cap // 13_000_000 <= 6


# ── plan_chunks: coalesce / split / coverage ───────────────────────────────────────────────

def _assert_full_coverage_in_order(chunks: list[Chunk], blocks: list[tuple[str, int]]):
    """Every block's rows [0, n) covered exactly once, contiguous, and blocks in input order."""
    covered: dict[str, list[tuple[int, int]]] = defaultdict(list)
    seen_block_order: list[str] = []
    for c in chunks:
        for s in c.segments:
            covered[s.block].append((s.start, s.end))
            if s.block not in seen_block_order:
                seen_block_order.append(s.block)
    assert seen_block_order == [b for b, _ in blocks]              # order preserved
    for block, n in blocks:
        segs = sorted(covered[block])
        assert segs[0][0] == 0 and segs[-1][1] == n               # spans [0, n)
        for (a0, a1), (b0, b1) in zip(segs, segs[1:]):
            assert a1 == b0                                       # no gap / overlap


class TestPlanner:
    def test_coalesces_small_blocks(self):
        blocks = [("b0", 100), ("b1", 100), ("b2", 100)]
        chunks = plan_chunks(blocks, capacity=250)
        assert [c.blocks for c in chunks] == [("b0", "b1"), ("b2",)]
        _assert_full_coverage_in_order(chunks, blocks)

    def test_no_chunk_exceeds_capacity(self):
        blocks = [("b0", 90), ("b1", 90), ("b2", 90), ("b3", 90)]
        cap = 200
        chunks = plan_chunks(blocks, capacity=cap)
        assert all(c.n_ticks <= cap for c in chunks)
        _assert_full_coverage_in_order(chunks, blocks)

    def test_splits_oversized_block(self):
        blocks = [("big", 1000)]
        cap = 300
        chunks = plan_chunks(blocks, capacity=cap)
        assert len(chunks) == 4                                   # ceil(1000/300)
        assert all(c.n_ticks <= cap for c in chunks)
        assert all(c.blocks == ("big",) for c in chunks)          # split, never coalesced
        _assert_full_coverage_in_order(chunks, blocks)

    def test_split_block_isolated_from_neighbours(self):
        blocks = [("a", 100), ("big", 1000), ("c", 100)]
        chunks = plan_chunks(blocks, capacity=300)
        # 'a' and 'c' are their own (small) chunks; 'big' is split into its own sub-chunks
        assert chunks[0].blocks == ("a",)
        assert all(c.blocks == ("big",) for c in chunks[1:-1])
        assert chunks[-1].blocks == ("c",)
        assert all(c.n_ticks <= 300 for c in chunks)
        _assert_full_coverage_in_order(chunks, blocks)

    def test_full_size_blocks_big_budget_one_or_few_per_chunk_no_split(self):
        blocks = [(f"b{i}", 13_000_000) for i in range(10)]
        chunks = plan_chunks_for(mem_budget_gb=96.0, n_features=225, blocks=blocks)
        # none split (each chunk has whole blocks), each chunk within capacity
        for c in chunks:
            assert all(s.start == 0 for s in c.segments)          # whole blocks only
        _assert_full_coverage_in_order(chunks, blocks)

    def test_scale_down_forces_splits(self):
        # same blocks, tiny budget → blocks get split, build still covers everything
        blocks = [(f"b{i}", 13_000_000) for i in range(3)]
        chunks = plan_chunks_for(mem_budget_gb=4.0, n_features=225, blocks=blocks)
        cap = chunk_capacity_ticks(4.0, 225)
        assert any(len(c.segments) == 1 and c.n_ticks < 13_000_000 for c in chunks)  # at least one split
        assert all(c.n_ticks <= cap for c in chunks)
        _assert_full_coverage_in_order(chunks, blocks)

    def test_empty_and_rejects_bad_capacity(self):
        assert plan_chunks([], capacity=100) == []
        with pytest.raises(ValueError):
            plan_chunks([("b", 10)], capacity=0)
