"""v2 per-column cache: round-trip, column selection, cross-block concat, and the
alignment invariants that per-column storage makes load-bearing."""
from __future__ import annotations

import numpy as np
import pytest

from boba.dataset_v2.raw import SampleArraysRaw
from boba.dataset_v2.cache import (
    AlignmentError, write_block, read_block, read_blocks, cached_columns,
)

GH = "gridhash123"
NAMES = ["bin_microprice", "bin_ema_ofi_10b", "byb_microprice", "byb_dt_100b"]


def _sample(n, names=NAMES, t0=0, seed=0, cost=True):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, len(names))).astype(np.float32)
    t = np.arange(t0, t0 + n, dtype=np.float64)
    kw = {}
    if cost:
        kw["eval_mid"] = rng.standard_normal(n).astype(np.float32)
        kw["c_mid_exit_l"] = rng.standard_normal(n).astype(np.float32)
    return SampleArraysRaw(x=x, timestamp_ms=t, column_names=list(names), **kw)


class TestRoundTrip:
    def test_block_roundtrip_all_columns_and_cost(self, tmp_path):
        s = _sample(50)
        write_block(tmp_path, GH, "b0", s)
        r = read_block(tmp_path, GH, "b0", NAMES, cost_fields=("eval_mid", "c_mid_exit_l"))
        np.testing.assert_array_equal(r.x, s.x)
        np.testing.assert_array_equal(r.timestamp_ms, s.timestamp_ms)
        np.testing.assert_array_equal(r.eval_mid, s.eval_mid)
        np.testing.assert_array_equal(r.c_mid_exit_l, s.c_mid_exit_l)

    def test_column_subset_and_order(self, tmp_path):
        s = _sample(30)
        write_block(tmp_path, GH, "b0", s)
        pick = [NAMES[2], NAMES[0]]                    # reversed subset
        r = read_block(tmp_path, GH, "b0", pick)
        assert r.column_names == pick
        np.testing.assert_array_equal(r.x[:, 0], s.x[:, 2])
        np.testing.assert_array_equal(r.x[:, 1], s.x[:, 0])

    def test_cached_columns_lists_what_is_there(self, tmp_path):
        write_block(tmp_path, GH, "b0", _sample(10))
        assert cached_columns(tmp_path, GH, "b0") == sorted(NAMES)
        assert cached_columns(tmp_path, GH, "missing") == []


class TestConcat:
    def test_read_blocks_concatenates_in_order(self, tmp_path):
        sizes = [20, 35, 15]
        samples, blocks, t0 = [], [], 0
        for i, n in enumerate(sizes):
            s = _sample(n, t0=t0, seed=i)
            write_block(tmp_path, GH, f"b{i}", s)
            samples.append(s); blocks.append(f"b{i}"); t0 += n
        r = read_blocks(tmp_path, GH, blocks, NAMES, cost_fields=("eval_mid",))
        assert r.x.shape == (sum(sizes), len(NAMES))
        np.testing.assert_array_equal(r.x, np.vstack([s.x for s in samples]))
        np.testing.assert_array_equal(r.timestamp_ms, np.concatenate([s.timestamp_ms for s in samples]))
        np.testing.assert_array_equal(r.eval_mid, np.concatenate([s.eval_mid for s in samples]))
        # timestamps are monotonic across the join (no discontinuity)
        assert np.all(np.diff(r.timestamp_ms) > 0)


class TestAlignment:
    def test_write_rejects_misaligned_cost_field(self, tmp_path):
        s = _sample(40)
        s.eval_mid = s.eval_mid[:-1]                   # one short — a slicing bug
        with pytest.raises(AlignmentError):
            write_block(tmp_path, GH, "b0", s)

    def test_write_rejects_columnname_count_mismatch(self, tmp_path):
        s = _sample(40)
        s.column_names = s.column_names[:-1]           # x has F cols, names has F-1
        with pytest.raises(AlignmentError):
            write_block(tmp_path, GH, "b0", s)

    def test_read_catches_a_shifted_column_file(self, tmp_path):
        # the corruption per-column storage risks: one column file the wrong length.
        write_block(tmp_path, GH, "b0", _sample(40))
        bad = tmp_path / GH / "b0" / f"{NAMES[1]}.npy"
        np.save(bad, np.zeros(39, np.float32))         # 39 ≠ grid 40
        with pytest.raises(AlignmentError):
            read_block(tmp_path, GH, "b0", NAMES)
