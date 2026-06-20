"""v2 engine: chunked processing with carried-tail checkpoints must reproduce the
un-chunked continuous build (docs/v2_dataset_design.md §3, §5, §11).

This is the core correctness gate: the same data built in chunks (with the tail carry) ==
the same data built in one continuous pass, on every shared row. If that holds, chunking +
carry + per-block slicing + per-column cache are all correct and a block's output is
chunk-independent.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from boba.dataset_v2 import DatasetRawConfig, col
from boba.dataset_v2.raw import build_features_raw
from boba.dataset_v2.session_data import SessionData
from boba.dataset_v2.engine import Block, build_chunked, concat_sessions
from boba.dataset_v2.cache import read_blocks

LISTINGS = ["bin", "byb"]
TARGET = "bin"
MS = 1_000_000          # ns per ms
SPAN_MS = 2000          # each synthetic block spans 2000 ms


def _synth_block(lo_ns, hi_ns, seed, book_dt_ns=MS, trade_every=5):
    rng = np.random.default_rng(seed)
    bf = {f: {} for f in ("listing_book_t", "listing_book_bid", "listing_book_ask",
                          "listing_book_bid_qty", "listing_book_ask_qty",
                          "listing_feed_latency_excess_ns")}
    tf = {f: {} for f in ("trade_ts", "trade_exchange_ts", "trade_prc", "trade_qty", "trade_dir")}
    for l in LISTINGS:
        bt = np.arange(lo_ns, hi_ns, book_dt_ns, dtype=np.int64)
        n = len(bt)
        mid = 100.0 + np.cumsum(rng.choice([-1.0, 0.0, 1.0], size=n) * 0.001)
        bf["listing_book_t"][l] = bt
        bf["listing_book_bid"][l] = mid - 0.01
        bf["listing_book_ask"][l] = mid + 0.01
        bf["listing_book_bid_qty"][l] = np.full(n, 10.0) + rng.standard_normal(n) * 0.1
        bf["listing_book_ask_qty"][l] = np.full(n, 10.0) + rng.standard_normal(n) * 0.1
        bf["listing_feed_latency_excess_ns"][l] = np.zeros(n, np.int64)
        tt = bt[::trade_every]
        m = len(tt)
        tf["trade_ts"][l] = tt
        tf["trade_exchange_ts"][l] = tt
        tf["trade_prc"][l] = mid[::trade_every]
        tf["trade_qty"][l] = np.full(m, 1.0)
        tf["trade_dir"][l] = rng.choice([-1.0, 1.0], size=m)
    tb = bf["listing_book_t"][TARGET]
    bmid = (bf["listing_book_bid"][TARGET] + bf["listing_book_ask"][TARGET]) / 2.0
    z = np.zeros(len(tb), np.int64)
    all_rx = np.sort(np.concatenate(
        [bf["listing_book_t"][l] for l in LISTINGS] + [tf["trade_ts"][l] for l in LISTINGS]))
    return SessionData(
        target_listing=TARGET, all_rx=all_rx,
        book_t=tb, book_bid=bf["listing_book_bid"][TARGET], book_ask=bf["listing_book_ask"][TARGET],
        book_mid=bmid, feed_latency_raw_ns=z, feed_latency_excess_ns=z, **bf, **tf)


def _cfg():
    cols = (
        col("{LISTING}_microprice", LISTING=LISTINGS),                         # instantaneous
        col("{LISTING}_spread_width", LISTING=LISTINGS),                       # instantaneous
        col("{LISTING}_ema_ofi_{N}b", LISTING=LISTINGS, N=[3, 10]),            # event-clock EMA
        col("{LISTING}_ema_microprice_centered_{N}ms", LISTING=[TARGET], N=[20, 50]),  # calendar EMA
        col("{LISTING}_return_{N}ms", LISTING=[TARGET], N=[20]),               # ms window
        col("{LISTING}_dt_{N}t", LISTING=[TARGET], N=[5]),                     # event window
    )
    return DatasetRawConfig(
        columns=cols, listings=tuple(LISTINGS), target_listing=TARGET,
        warmup_ms=0, horizon_ms=50.0, event_mask="none", cost_fields=(),
        wide_threshold={l: 1.0e-4 for l in LISTINGS})


def _blocks(n_blocks):
    out = []
    for k in range(n_blocks):
        sd = _synth_block(k * SPAN_MS * MS, (k + 1) * SPAN_MS * MS, seed=k)
        out.append(Block(id=f"b{k}", session=sd, start_ms=k * SPAN_MS, end_ms=(k + 1) * SPAN_MS))
    return out


def _continuous(cfg, blocks):
    sd = concat_sessions([b.session for b in blocks])
    return build_features_raw(sd, list(cfg.listings), cfg)   # cfg already warmup_ms=0


def _assert_chunked_matches_continuous(tmp_path, cfg, blocks, capacity_ms, tail_ms):
    gh = "gh"
    build_chunked(blocks, cfg, tmp_path, gh, tail_window_ns=tail_ms * MS, capacity=capacity_ms)
    cont = _continuous(cfg, blocks)
    got = read_blocks(tmp_path, gh, [b.id for b in blocks], list(cont.column_names))

    # chunked rows are a subset of continuous (block-local horizon holes at the seams)
    idx = {int(t): i for i, t in enumerate(cont.timestamp_ms)}
    rows = [idx[int(t)] for t in got.timestamp_ms]
    assert len(rows) > 0
    assert np.all(np.diff(got.timestamp_ms) > 0)             # seamless, strictly increasing
    np.testing.assert_allclose(got.x, cont.x[rows], rtol=1e-5, atol=1e-4)
    return got, cont


class TestChunkedEqualsContinuous:
    def test_two_blocks_per_chunk(self, tmp_path):
        # 4 blocks → chunks [b0,b1],[b2,b3]; the second chunk is warmed by the carried tail
        _assert_chunked_matches_continuous(tmp_path, _cfg(), _blocks(4),
                                           capacity_ms=4000, tail_ms=2500)

    def test_one_block_per_chunk_carry_every_boundary(self, tmp_path):
        # capacity = one block → carry across every boundary
        _assert_chunked_matches_continuous(tmp_path, _cfg(), _blocks(4),
                                           capacity_ms=2000, tail_ms=2500)

    def test_single_chunk_trivially_matches(self, tmp_path):
        # whole range in one chunk (no carry) — must equal the continuous build exactly
        _assert_chunked_matches_continuous(tmp_path, _cfg(), _blocks(2),
                                           capacity_ms=10_000, tail_ms=2500)

    def test_block_outputs_are_chunk_independent(self, tmp_path):
        # the same block built under different chunkings yields byte-identical column files
        gh = "gh"
        blocks = _blocks(4)
        cfg = _cfg()
        build_chunked(blocks, cfg, tmp_path / "a", gh, tail_window_ns=2500 * MS, capacity=2000)
        build_chunked(blocks, cfg, tmp_path / "b", gh, tail_window_ns=2500 * MS, capacity=700)
        names = list(_continuous(cfg, blocks).column_names)
        for b in blocks[2:]:                                 # blocks past the cold first chunk
            a = read_blocks(tmp_path / "a", gh, [b.id], names)   # capacity=2000: whole blocks
            c = read_blocks(tmp_path / "b", gh, [b.id], names)   # capacity=700: SPLIT blocks
            np.testing.assert_allclose(a.x, c.x, rtol=1e-5, atol=1e-4)

    def test_split_blocks_match_continuous(self, tmp_path):
        # capacity below one block's span → blocks are SPLIT (scale-down path); the forward
        # data on each sub-range must reproduce the block's grid with no internal holes.
        _assert_chunked_matches_continuous(tmp_path, _cfg(), _blocks(3),
                                           capacity_ms=700, tail_ms=2500)

    def test_cost_fields_correct_through_splits(self, tmp_path):
        # cost fields are forward-looking; splits must carry horizon_ms of lookahead so the
        # outcome stays correct at internal sub-range boundaries.
        cfg = dataclasses.replace(_cfg(), cost_fields=("eval_mid", "c_mid_exit_l"))
        blocks = _blocks(3)
        gh = "gh"
        build_chunked(blocks, cfg, tmp_path, gh, tail_window_ns=2500 * MS, capacity=700)
        cont = build_features_raw(concat_sessions([b.session for b in blocks]), list(cfg.listings), cfg)
        got = read_blocks(tmp_path, gh, [b.id for b in blocks], list(cont.column_names),
                          cost_fields=cfg.cost_fields)
        idx = {int(t): i for i, t in enumerate(cont.timestamp_ms)}
        rows = [idx[int(t)] for t in got.timestamp_ms]
        np.testing.assert_allclose(got.x, cont.x[rows], rtol=1e-5, atol=1e-4)
        for f in cfg.cost_fields:
            np.testing.assert_allclose(getattr(got, f), getattr(cont, f)[rows], rtol=1e-5, atol=1e-4)


# ── event_mask: which event type drives the output grid ─────────────────────────────────────
import pytest


def _sparse_session(span_ms=800):
    """Book ticks every 2 ms, trades every 5 ms (offset) — so book/trade/both/none give
    distinct, properly-nested row sets."""
    bf = {f: {} for f in ("listing_book_t","listing_book_bid","listing_book_ask",
                          "listing_book_bid_qty","listing_book_ask_qty","listing_feed_latency_excess_ns")}
    tf = {f: {} for f in ("trade_ts","trade_exchange_ts","trade_prc","trade_qty","trade_dir")}
    for l in LISTINGS:
        bt = (np.arange(0, span_ms, 2)).astype(np.int64) * MS          # book at 0,2,4,…
        n = len(bt)
        bf["listing_book_t"][l]=bt; bf["listing_book_bid"][l]=np.full(n,99.99); bf["listing_book_ask"][l]=np.full(n,100.01)
        bf["listing_book_bid_qty"][l]=np.full(n,10.0); bf["listing_book_ask_qty"][l]=np.full(n,10.0)
        bf["listing_feed_latency_excess_ns"][l]=np.zeros(n,np.int64)
        tt = (np.arange(1, span_ms, 5)).astype(np.int64) * MS          # trades at 1,6,11,… (offset)
        m=len(tt)
        tf["trade_ts"][l]=tt; tf["trade_exchange_ts"][l]=tt; tf["trade_prc"][l]=np.full(m,100.0)
        tf["trade_qty"][l]=np.full(m,1.0); tf["trade_dir"][l]=np.ones(m)
    tb=bf["listing_book_t"][TARGET]; bmid=(bf["listing_book_bid"][TARGET]+bf["listing_book_ask"][TARGET])/2
    z=np.zeros(len(tb),np.int64)
    arx=np.sort(np.concatenate([bf["listing_book_t"][l] for l in LISTINGS]+[tf["trade_ts"][l] for l in LISTINGS]))
    return SessionData(target_listing=TARGET, all_rx=arx, book_t=tb, book_bid=bf["listing_book_bid"][TARGET],
        book_ask=bf["listing_book_ask"][TARGET], book_mid=bmid, feed_latency_raw_ns=z, feed_latency_excess_ns=z, **bf, **tf)


class TestEventMask:
    def test_options_select_expected_rows(self):
        sd = _sparse_session(800)
        cfg = dataclasses.replace(_cfg(), horizon_ms=20.0)
        def ts(m):
            c = dataclasses.replace(cfg, event_mask=m)
            return set(int(t) for t in build_features_raw(sd, LISTINGS, c).timestamp_ms)
        none, book, trade, both = ts("none"), ts("book"), ts("trade"), ts("both")
        assert both == (book | trade)                 # union, by construction
        assert book < both and trade < both           # each a strict subset of the union
        assert both < none                            # the dense grid also has quiet ms
        assert len(trade) < len(book) < len(both) < len(none)

    def test_invalid_mask_rejected(self):
        with pytest.raises(ValueError, match="event_mask"):
            DatasetRawConfig(columns=_cfg().columns, event_mask="bogus")


# ── gap-aware warmup trim (read-time) ────────────────────────────────────────────────────────
class TestGapAwareTrim:
    def test_gap_trim_ms_logic(self):
        from boba.dataset_v2.cache import gap_trim_ms
        assert gap_trim_ms(100, float("inf"), k=20) == 2000   # cold origin (G=∞) → full re-warm
        assert gap_trim_ms(100, 200, k=20) == 2000            # gap ≥ span → re-warm (short EMA)
        assert gap_trim_ms(100, 100, k=20) == 2000            # gap == span → re-warm (≥)
        assert gap_trim_ms(1000, 200, k=20) == 0              # gap < span → resilient (long EMA)
        assert gap_trim_ms(100, 50, k=20) == 0                # gap < span → resilient
        assert gap_trim_ms(0, float("inf")) == 0              # instantaneous → never trimmed

    def test_origin_trimmed_contiguous_blocks_not(self, tmp_path):
        from boba.dataset_v2.cache import read_blocks, read_dataset, _block_gap_before_ms
        blocks = _blocks(3); cfg = _cfg(); gh = "gh"           # contiguous → gap 0 after b0
        build_chunked(blocks, cfg, tmp_path, gh, tail_window_ns=2500 * MS, capacity=2000)
        assert _block_gap_before_ms(tmp_path, gh, "b0") == float("inf")   # origin
        assert _block_gap_before_ms(tmp_path, gh, "b1") == 0.0            # contiguous
        names = list(_continuous(cfg, blocks).column_names)              # max span 50 → trim 1000
        full = read_blocks(tmp_path, gh, ["b0", "b1", "b2"], names)
        warm = read_dataset(tmp_path, gh, ["b0", "b1", "b2"], names)
        b0t = read_blocks(tmp_path, gh, ["b0"], names).timestamp_ms
        dropped = int((b0t < b0t[0] + 1000).sum())
        assert dropped > 0
        assert warm.x.shape[0] == full.x.shape[0] - dropped   # only b0's warmup gone; b1/b2 untouched

    def test_gap_rewarms_short_span_only(self, tmp_path):
        from boba.dataset_v2.cache import read_blocks, _block_gap_before_ms
        # b1 starts 200ms after b0 ends → a real 200ms gap
        b0 = Block("b0", 0, 2000, _synth_block(0, 2000 * MS, seed=0))
        b1 = Block("b1", 2200, 4200, _synth_block(2200 * MS, 4200 * MS, seed=1))
        cfg = dataclasses.replace(_cfg(), columns=(
            col("{LISTING}_ema_microprice_centered_{N}ms", LISTING=[TARGET], N=[50, 1000]),))
        gh = "gh"
        build_chunked([b0, b1], cfg, tmp_path, gh, tail_window_ns=25_000 * MS, capacity=2000)
        assert _block_gap_before_ms(tmp_path, gh, "b1") == 200.0
        short = f"{TARGET}_ema_microprice_centered_50ms"
        long_ = f"{TARGET}_ema_microprice_centered_1000ms"
        # 200ms gap ≥ 50ms span → short EMA re-warms (rows dropped); < 1000ms span → long is resilient
        assert read_blocks(tmp_path, gh, ["b1"], [short], trim_warmup=True).x.shape[0] \
            < read_blocks(tmp_path, gh, ["b1"], [short]).x.shape[0]
        assert read_blocks(tmp_path, gh, ["b1"], [long_], trim_warmup=True).x.shape[0] \
            == read_blocks(tmp_path, gh, ["b1"], [long_]).x.shape[0]

    def test_warmup_trim_ms_cold_convenience(self):
        from boba.dataset_v2.cache import warmup_trim_ms
        assert warmup_trim_ms(["bin_microprice"]) == 0                 # instantaneous
        assert warmup_trim_ms(["bin_ema_ofi_100b", "bin_microprice"], k=20) == 2000
        assert warmup_trim_ms(["a_b_ema_log_microprice_ratio_1000ms"], k=10) == 10000
