"""v2 driver: grid_hash (cache dir key) invariance, checkpoint persistence, the loader-based
build path, and the v1-oracle sanity (instantaneous features match v1 exactly)."""
from __future__ import annotations

import dataclasses

import numpy as np

from boba.dataset_v2 import DatasetRawConfig, col
from boba.dataset_v2.raw import build_features_raw
from boba.dataset_v2.cache import read_blocks
from boba.dataset_v2.engine import (
    MS, Block, build_chunked, concat_sessions, tail_of, save_checkpoint, load_checkpoint,
)
from boba.dataset_v2.driver import build_dataset_v2, tail_window_ns_for

from tests.test_dataset_v2_engine import _blocks, _cfg, _continuous, LISTINGS


def _gcfg(**kw):
    kw.setdefault("listings", ("bin", "byb"))
    kw.setdefault("target_listing", "bin")
    kw.setdefault("columns", (col("{LISTING}_ema_ofi_{N}b", LISTING="bin", N=[3, 10]),))
    return DatasetRawConfig(**kw)


class TestGridHash:
    def test_excludes_columns_cost_warmup_perf(self):
        base = _gcfg()
        # different column selection → SAME grid dir (columns are carried by filenames)
        diff = DatasetRawConfig(columns=(col("{LISTING}_microprice", LISTING="bin"),),
                                listings=("bin", "byb"), target_listing="bin")
        assert base.grid_hash() == diff.grid_hash()
        # warmup / cost_fields / perf knobs never in the grid hash
        assert base.grid_hash() == _gcfg(
            warmup_ms=999, n_workers=2, mem_budget_gb=8.0, cost_fields=("eval_mid",)).grid_hash()

    def test_sensitive_to_grid_identity(self):
        base = _gcfg()
        assert base.grid_hash() != _gcfg(horizon_ms=999.0).grid_hash()
        assert base.grid_hash() != _gcfg(event_mask="none").grid_hash()
        assert base.grid_hash() != _gcfg(target_listing="byb").grid_hash()
        assert base.grid_hash().startswith("v2_")


class TestCheckpoint:
    def test_roundtrip(self, tmp_path):
        blocks = _blocks(2)
        sd = concat_sessions([b.session for b in blocks])
        tail = tail_of(sd, 1000 * MS)
        save_checkpoint(tmp_path / "ckpt.npz", tail)
        back = load_checkpoint(tmp_path / "ckpt.npz")
        assert back.target_listing == tail.target_listing
        np.testing.assert_array_equal(back.all_rx, tail.all_rx)
        np.testing.assert_array_equal(back.book_mid, tail.book_mid)
        for l in LISTINGS:
            np.testing.assert_array_equal(back.listing_book_t[l], tail.listing_book_t[l])
            np.testing.assert_array_equal(back.listing_book_bid[l], tail.listing_book_bid[l])
            np.testing.assert_array_equal(back.trade_prc[l], tail.trade_prc[l])
            np.testing.assert_array_equal(back.trade_dir[l], tail.trade_dir[l])


class TestDriver:
    def test_tail_window_scales_with_span(self):
        small = tail_window_ns_for(_gcfg())                               # max span 10
        big = tail_window_ns_for(_gcfg(columns=(
            col("{LISTING}_ema_ofi_{N}b", LISTING="bin", N=[1000]),)))
        assert 0 < small < big

    def test_loader_path_matches_continuous(self, tmp_path):
        # blocks with no in-memory session — fetched via load() per chunk; output must equal
        # the continuous build, and land under cfg.grid_hash().
        src = _blocks(4)
        cfg = _cfg()
        by_id = {b.id: b.session for b in src}
        metas = [Block(id=b.id, start_ms=b.start_ms, end_ms=b.end_ms) for b in src]
        build_dataset_v2(cfg, metas, tmp_path, tail_window_ns=2500 * MS,
                         load=lambda i: by_id[i], verbose=False)
        cont = _continuous(cfg, src)
        got = read_blocks(tmp_path, cfg.grid_hash(), [b.id for b in src], list(cont.column_names))
        idx = {int(t): i for i, t in enumerate(cont.timestamp_ms)}
        rows = [idx[int(t)] for t in got.timestamp_ms]
        np.testing.assert_allclose(got.x, cont.x[rows], rtol=1e-5, atol=1e-4)


class TestV1Oracle:
    def test_instantaneous_features_match_v1_exactly(self, tmp_path):
        # v1 = a block built standalone with a real warmup (per-block reset); v2 = the engine's
        # continuous output for that block. On shared timestamps, INSTANTANEOUS features have no
        # model difference, so they must be bit-exact — the alignment/plumbing gate (§11.1/§11.2).
        blocks = _blocks(3)
        cfg = _cfg()
        names = list(_continuous(cfg, blocks).column_names)
        build_chunked(blocks, cfg, tmp_path, "gh", tail_window_ns=2500 * MS, capacity=4000)
        v2 = read_blocks(tmp_path, "gh", ["b1"], names)                   # warmed continuous block

        v1 = build_features_raw(blocks[1].session, list(cfg.listings),
                                dataclasses.replace(cfg, warmup_ms=100))  # standalone, cold+warmup
        i1 = {int(t): i for i, t in enumerate(v1.timestamp_ms)}
        pairs = [(i, i1[int(t)]) for i, t in enumerate(v2.timestamp_ms) if int(t) in i1]
        assert len(pairs) > 100
        v2r = [p[0] for p in pairs]; v1r = [p[1] for p in pairs]
        inst = [i for i, n in enumerate(names) if n.endswith("microprice") or n.endswith("spread_width")]
        assert inst
        for ci in inst:
            np.testing.assert_array_equal(v2.x[v2r][:, ci], v1.x[v1r][:, ci])

    def test_full_feature_ladder_vs_v1_per_block(self, tmp_path):
        # The complete §11 ladder against v1's ACTUAL per-block output (reset + warmup), every
        # column, timestamp-joined:
        #   • instantaneous columns → BIT-EXACT (alignment + correctness, no model difference)
        #   • stateful columns      → agree in the interior, past v1's reset transient
        blocks = _blocks(3)
        cfg = _cfg()
        names = list(_continuous(cfg, blocks).column_names)
        nmap = {u.name: u.n for u in cfg.expanded().units}     # column → span (None = instantaneous)

        build_chunked(blocks, cfg, tmp_path, "gh", tail_window_ns=2500 * MS, capacity=2000)
        v2 = read_blocks(tmp_path, "gh", [b.id for b in blocks], names)

        cfg_v1 = dataclasses.replace(cfg, warmup_ms=80)        # v1: each block standalone, cold
        parts = [build_features_raw(b.session, list(cfg.listings), cfg_v1) for b in blocks]
        v1x = np.vstack([p.x for p in parts])
        v1t = np.concatenate([p.timestamp_ms for p in parts])
        v1bs = np.concatenate([np.full(len(p.timestamp_ms), b.start_ms, np.float64)
                               for p, b in zip(parts, blocks)])

        i1 = {int(t): i for i, t in enumerate(v1t)}
        common = [(i, i1[int(t)]) for i, t in enumerate(v2.timestamp_ms) if int(t) in i1]
        assert len(common) > 200
        r2 = np.array([c[0] for c in common]); r1 = np.array([c[1] for c in common])
        age = v2.timestamp_ms[r2] - v1bs[r1]                   # rows since this row's block start

        checked_instant = checked_stateful = 0
        for ci, nm in enumerate(names):
            a, b = v2.x[r2][:, ci], v1x[r1][:, ci]
            n = nmap.get(nm)
            if n is None:                                      # instantaneous: exact everywhere
                np.testing.assert_array_equal(a, b, err_msg=f"{nm}: instantaneous mismatch (alignment?)")
                checked_instant += 1
            else:
                # mid-block, past v1's cold-start transient, the EMAs/windows are essentially
                # IDENTICAL between v1 and v2 (both converged to the same recent-weighted value —
                # measured max rel-diff ~1.5e-7). A misaligned EMA column would diverge here (a
                # 50-row shift ⇒ ~1.3e-2), so the tight tolerance is a real alignment guard.
                interior = age >= 12 * n
                if interior.sum() > 10:
                    np.testing.assert_allclose(a[interior], b[interior], rtol=1e-4, atol=5e-4,
                                               err_msg=f"{nm}: mid-block disagreement vs v1")
                    checked_stateful += 1
        assert checked_instant >= 2 and checked_stateful >= 4   # both kinds actually exercised
        # (v2 ≠ v1 *inside* the reset transient — the bug v2 fixes — is proven by the
        #  chunked==continuous engine test, which v1-per-block fails at boundaries.)
