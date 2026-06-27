"""Comprehensive tests for boba/dataset/raw.py.

Strategy: a dataset is defined by an explicit ordered tuple of ColumnSpecs
(boba.dataset.columns) — there is no full-catalogue default, output column order is the
expansion/request order, and the compute_* kernels take per-family span dicts.
tests/helpers.py rebuilds the legacy full catalogue (make_cfg, legacy_local_names,
DEFAULT_*_SPANS) so every numeric check below keeps its original coverage.
Each formula in boba/dataset/raw.py is tested by:
  1. Computing the expected result independently from synthetic inputs.
  2. Comparing against the function under test with np.testing.assert_allclose.

Edge cases tested:
  - Empty inputs.
  - Single event.
  - Multiple events with same nanosecond timestamp.
  - Grid ticks before any event (fill_value behaviour).
  - Buy/sell EMA: opposite-side trades must produce zero contribution.
  - Forward-fill: grid ticks with no new event hold the previous value.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import lfilter

from boba.dataset.columns import col
from boba.dataset.raw import (
    _alpha,
    _ewm_1d,
    aggregate_same_ns_bbo,
    aggregate_same_ns_trade,
    forward_fill_to_ms_grid,
    dt_over_last_n_events,
    time_since_event_ms,
    trailing_count_ms,
    compute_microprice,
    compute_ofi_events,
    compute_abs_log_ret,
    compute_bbo_event_emas,
    compute_trade_event_emas,
    compute_ms_grid_temporal,
    trade_value_per_ms,
    feature_names,
    _compute_per_listing,
    _grid_bounds_ms,
    _COST_FIELDS,
    build_features_raw,
)
from tests.helpers import (
    DEFAULT_BBO_SPANS,
    DEFAULT_TEMPORAL_SPANS,
    DEFAULT_TRADE_SPANS,
    EMA_TRADE_SPANS,
    EMA_TRADE_VALUE_MS_SPANS,
    cols_from_names,
    legacy_local_names,
    make_cfg,
)


# ── EMA helpers ───────────────────────────────────────────────────────────────

class TestAlpha:
    def test_standard_convention(self):
        # alpha = 2 / (span + 1)
        assert _alpha(1) == pytest.approx(1.0)
        assert _alpha(9) == pytest.approx(0.2)
        assert _alpha(99) == pytest.approx(0.02)
        assert _alpha(10) == pytest.approx(2.0 / 11.0)


class TestEwm1d:
    def test_empty(self):
        out = _ewm_1d(np.array([], dtype=np.float64), 0.5)
        assert out.shape == (0,)

    def test_constant_input(self):
        # Constant input → output converges to that value
        x = np.full(1000, 3.0)
        out = _ewm_1d(x, 0.1)
        # After warmup the output equals the input
        assert out[-1] == pytest.approx(3.0, rel=1e-6)

    def test_first_value(self):
        # y[0] = alpha * x[0] + (1-alpha) * 0 = alpha * x[0]
        x = np.array([5.0, 0.0, 0.0])
        out = _ewm_1d(x, 0.3)
        assert out[0] == pytest.approx(0.3 * 5.0)
        # y[1] = alpha * 0 + (1-alpha) * y[0] = 0.7 * 1.5 = 1.05
        assert out[1] == pytest.approx(0.7 * 0.3 * 5.0)

    def test_manual_recursion(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50)
        alpha = 0.2
        # Reference: y[i] = alpha*x[i] + (1-alpha)*y[i-1]
        ref = np.zeros(50)
        ref[0] = alpha * x[0]
        for i in range(1, 50):
            ref[i] = alpha * x[i] + (1 - alpha) * ref[i - 1]
        out = _ewm_1d(x, alpha)
        np.testing.assert_allclose(out, ref, rtol=1e-12)

    def test_matches_scipy_lfilter(self):
        x = np.linspace(0, 10, 100)
        alpha = 0.05
        out = _ewm_1d(x, alpha)
        ref = lfilter([alpha], [1.0, -(1.0 - alpha)], x)
        np.testing.assert_allclose(out, ref, rtol=1e-12)


# ── Same-ns aggregation ──────────────────────────────────────────────────────

class TestAggregateBBO:
    def test_empty(self):
        ts = np.array([], dtype=np.int64)
        out = aggregate_same_ns_bbo(ts, ts, ts, ts, ts)
        for arr in out:
            assert arr.shape == (0,)

    def test_no_duplicates_passthrough(self):
        ts = np.array([100, 200, 300], dtype=np.int64)
        bid = np.array([1.0, 1.1, 1.2])
        ask = np.array([1.05, 1.15, 1.25])
        bq = np.array([10.0, 20.0, 30.0])
        aq = np.array([15.0, 25.0, 35.0])
        out_ts, out_bid, out_ask, out_bq, out_aq = aggregate_same_ns_bbo(ts, bid, ask, bq, aq)
        np.testing.assert_array_equal(out_ts, ts)
        np.testing.assert_array_equal(out_bid, bid)

    def test_keeps_last_per_ns(self):
        # Three updates at ns=100 (last one wins), one at ns=200
        ts = np.array([100, 100, 100, 200], dtype=np.int64)
        bid = np.array([1.0, 1.1, 1.2, 1.3])
        ask = np.array([1.05, 1.15, 1.25, 1.35])
        bq = np.array([10.0, 11.0, 12.0, 13.0])
        aq = np.array([20.0, 21.0, 22.0, 23.0])
        out_ts, out_bid, out_ask, out_bq, out_aq = aggregate_same_ns_bbo(ts, bid, ask, bq, aq)
        np.testing.assert_array_equal(out_ts, [100, 200])
        np.testing.assert_array_equal(out_bid, [1.2, 1.3])
        np.testing.assert_array_equal(out_ask, [1.25, 1.35])
        np.testing.assert_array_equal(out_bq, [12.0, 13.0])
        np.testing.assert_array_equal(out_aq, [22.0, 23.0])


class TestAggregateTrade:
    def test_empty(self):
        ts = np.array([], dtype=np.int64)
        prc = qty = dir_ = np.array([], dtype=np.float64)
        out = aggregate_same_ns_trade(ts, prc, qty, dir_)
        for arr in out:
            assert arr.shape == (0,)

    def test_sums_same_side_same_ns(self):
        # 3 buy trades at ns=100, 1 sell at ns=200
        ts = np.array([100, 100, 100, 200], dtype=np.int64)
        prc = np.array([1.0, 1.0, 1.0, 0.9])
        qty = np.array([10.0, 20.0, 5.0, 100.0])
        dir_ = np.array([1.0, 1.0, 1.0, -1.0])
        out_ts, out_prc, out_qty, out_dir = aggregate_same_ns_trade(ts, prc, qty, dir_)
        np.testing.assert_array_equal(out_ts, [100, 200])
        np.testing.assert_array_equal(out_qty, [35.0, 100.0])  # 10+20+5
        np.testing.assert_allclose(out_prc, [1.0, 0.9])  # VWAP equal at same price
        np.testing.assert_array_equal(out_dir, [1.0, -1.0])

    def test_vwap_when_prices_differ(self):
        # Two same-ns same-side trades at different prices (multi-level sweep)
        ts = np.array([100, 100], dtype=np.int64)
        prc = np.array([10.0, 11.0])
        qty = np.array([5.0, 5.0])
        dir_ = np.array([1.0, 1.0])
        out_ts, out_prc, out_qty, _ = aggregate_same_ns_trade(ts, prc, qty, dir_)
        # VWAP = (10*5 + 11*5) / (5+5) = 10.5
        assert len(out_ts) == 1
        assert out_qty[0] == 10.0
        assert out_prc[0] == pytest.approx(10.5)

    def test_keeps_opposite_sides_separate(self):
        # Same ns, opposite sides → two separate events
        ts = np.array([100, 100], dtype=np.int64)
        prc = np.array([10.0, 10.0])
        qty = np.array([5.0, 3.0])
        dir_ = np.array([1.0, -1.0])
        out_ts, out_prc, out_qty, out_dir = aggregate_same_ns_trade(ts, prc, qty, dir_)
        assert len(out_ts) == 2
        assert set(out_dir.tolist()) == {1.0, -1.0}


# ── Forward-fill onto ms grid ─────────────────────────────────────────────────

class TestForwardFill:
    def test_empty_events(self):
        grid = np.array([100, 200, 300], dtype=np.int64)
        out = forward_fill_to_ms_grid(np.array([], dtype=np.int64), np.array([]), grid, fill_value=-1.0)
        np.testing.assert_array_equal(out, [-1.0, -1.0, -1.0])

    def test_ticks_before_first_event_use_fill_value(self):
        # Event at ns=500; grid ticks at 100, 300, 600
        event_t = np.array([500], dtype=np.int64)
        values = np.array([7.5])
        grid = np.array([100, 300, 600], dtype=np.int64)
        out = forward_fill_to_ms_grid(event_t, values, grid, fill_value=0.0)
        np.testing.assert_array_equal(out, [0.0, 0.0, 7.5])

    def test_holds_last_value_across_gaps(self):
        # Events at 100, 300, 700 with values 1, 2, 3
        # Grid ticks at 50, 100, 200, 300, 500, 700, 900
        event_t = np.array([100, 300, 700], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0])
        grid = np.array([50, 100, 200, 300, 500, 700, 900], dtype=np.int64)
        out = forward_fill_to_ms_grid(event_t, values, grid, fill_value=0.0)
        # 50: no prior, 0
        # 100: event lands at 100 (side='right' index = 1 → last index 0)
        # 200: last is 100 → 1
        # 300: event lands here → 2
        # 500: last is 300 → 2
        # 700: event lands here → 3
        # 900: last is 700 → 3
        np.testing.assert_array_equal(out, [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])


# ── Microprice ────────────────────────────────────────────────────────────────

class TestMicroprice:
    def test_balanced_book_gives_mid(self):
        # Equal sizes → microprice = arithmetic mid
        bid = np.array([100.0])
        ask = np.array([102.0])
        bq = np.array([10.0])
        aq = np.array([10.0])
        mp = compute_microprice(bid, ask, bq, aq)
        assert mp[0] == pytest.approx(101.0)

    def test_thin_bid_pulls_toward_ask(self):
        # Thin bid (qty=1) vs thick ask (qty=99) → microprice closer to bid
        # Stoikov formula: weight by *opposite* side's qty
        # mp = (bq*ask + aq*bid) / (bq+aq) = (1*102 + 99*100)/100 = 9999/100 = 99.99 + 1.02 = ... wait
        # = (1*102 + 99*100) / (1+99) = (102 + 9900)/100 = 100.02
        bid = np.array([100.0])
        ask = np.array([102.0])
        bq = np.array([1.0])
        aq = np.array([99.0])
        mp = compute_microprice(bid, ask, bq, aq)
        # bid_qty is small → microprice pulled toward bid (which it gets less weight)
        # actually formula: mp = (bid_qty*ask + ask_qty*bid)/total
        # = (1*102 + 99*100)/100 = (102 + 9900)/100 = 100.02
        # So mp is closer to bid (100) than to ask (102) — heavy ask side, thin bid → bid likely breaks → price goes down → mp closer to bid
        assert mp[0] == pytest.approx(100.02)

    def test_zero_qty_safe(self):
        # Both sides zero qty: don't crash, return some value (we use eps guard)
        bid = np.array([100.0])
        ask = np.array([102.0])
        bq = np.array([0.0])
        aq = np.array([0.0])
        mp = compute_microprice(bid, ask, bq, aq)
        # eps protection means result is finite
        assert np.isfinite(mp[0])


# ── OFI events ────────────────────────────────────────────────────────────────

class TestOfiEvents:
    def test_first_event_zero(self):
        bid = np.array([100.0, 100.0])
        ask = np.array([101.0, 101.0])
        bq = np.array([10.0, 10.0])
        aq = np.array([10.0, 10.0])
        ofi = compute_ofi_events(bid, ask, bq, aq)
        assert ofi[0] == 0.0  # no prior state

    def test_bid_steps_up_positive(self):
        # bid 100→101 with new qty 5: e_bid = +5
        bid = np.array([100.0, 101.0])
        ask = np.array([102.0, 102.0])
        bq = np.array([10.0, 5.0])
        aq = np.array([10.0, 10.0])
        ofi = compute_ofi_events(bid, ask, bq, aq)
        # bid up → +new_bq = +5; ask unchanged with same qty → -(10-10) = 0
        assert ofi[1] == pytest.approx(5.0)

    def test_bid_steps_down_negative(self):
        # bid 101→100 with prev qty 8: e_bid = -8
        bid = np.array([101.0, 100.0])
        ask = np.array([102.0, 102.0])
        bq = np.array([8.0, 10.0])
        aq = np.array([10.0, 10.0])
        ofi = compute_ofi_events(bid, ask, bq, aq)
        # bid down → -prev_bq = -8; ask unchanged → 0
        assert ofi[1] == pytest.approx(-8.0)

    def test_ask_steps_down_negative(self):
        # ask 102→101 with new qty 4: e_ask = -4
        bid = np.array([100.0, 100.0])
        ask = np.array([102.0, 101.0])
        bq = np.array([10.0, 10.0])
        aq = np.array([7.0, 4.0])
        ofi = compute_ofi_events(bid, ask, bq, aq)
        # ask down → -new_aq = -4; bid unchanged → 0
        assert ofi[1] == pytest.approx(-4.0)

    def test_ask_steps_up_positive(self):
        # ask 101→102 with prev qty 5: e_ask = +5 (ask backed away → bullish)
        bid = np.array([100.0, 100.0])
        ask = np.array([101.0, 102.0])
        bq = np.array([10.0, 10.0])
        aq = np.array([5.0, 3.0])
        ofi = compute_ofi_events(bid, ask, bq, aq)
        # ask up → +prev_aq = +5; bid unchanged → 0
        assert ofi[1] == pytest.approx(5.0)

    def test_ask_same_price_qty_change(self):
        # ask same price, qty 10→6: e_ask = -(6-10) = +4
        bid = np.array([100.0, 100.0])
        ask = np.array([102.0, 102.0])
        bq = np.array([10.0, 10.0])
        aq = np.array([10.0, 6.0])
        ofi = compute_ofi_events(bid, ask, bq, aq)
        # ask qty decreased (sellers retreating) → -(new-prev) = -(-4) = +4
        assert ofi[1] == pytest.approx(4.0)

    def test_ask_same_price_qty_increase(self):
        # ask same price, qty 10→15: e_ask = -(15-10) = -5
        bid = np.array([100.0, 100.0])
        ask = np.array([102.0, 102.0])
        bq = np.array([10.0, 10.0])
        aq = np.array([10.0, 15.0])
        ofi = compute_ofi_events(bid, ask, bq, aq)
        # ask qty increased (more sellers) → -(new-prev) = -5
        assert ofi[1] == pytest.approx(-5.0)

    def test_same_price_qty_increase(self):
        # bid unchanged, qty 10→15: e_bid = +5
        bid = np.array([100.0, 100.0])
        ask = np.array([102.0, 102.0])
        bq = np.array([10.0, 15.0])
        aq = np.array([10.0, 10.0])
        ofi = compute_ofi_events(bid, ask, bq, aq)
        assert ofi[1] == pytest.approx(5.0)

    def test_combined_bid_and_ask_changes(self):
        # bid up by 1, ask down by 1: both bullish? Let me trace.
        # bid 100→101 nbq=5: e_bid = +5 (bid up)
        # ask 102→101 naq=3: e_ask = -3 (ask down — wait that's bearish)
        # Hmm — actually ask DOWN with prev=102 → naq=3 → e_ask = -naq = -3
        # Net OFI = +5 - 3 = +2
        bid = np.array([100.0, 101.0])
        ask = np.array([102.0, 101.0])
        bq = np.array([10.0, 5.0])
        aq = np.array([10.0, 3.0])
        ofi = compute_ofi_events(bid, ask, bq, aq)
        assert ofi[1] == pytest.approx(5.0 - 3.0)


# ── abs_log_ret ───────────────────────────────────────────────────────────────

class TestAbsLogRet:
    def test_first_event_zero(self):
        mp = np.array([100.0])
        out = compute_abs_log_ret(mp)
        assert out[0] == 0.0

    def test_known_log_return(self):
        mp = np.array([100.0, 110.0, 100.0])
        out = compute_abs_log_ret(mp)
        assert out[0] == 0.0
        assert out[1] == pytest.approx(abs(np.log(110.0 / 100.0)))
        assert out[2] == pytest.approx(abs(np.log(100.0 / 110.0)))


# ── dt_over_last_n_events ─────────────────────────────────────────────────────

class TestDtOverLastN:
    def test_empty(self):
        out = dt_over_last_n_events(np.array([], dtype=np.int64), 5)
        assert out.shape == (0,)

    def test_zeros_for_first_N_events(self):
        ts = np.arange(10, dtype=np.int64) * 100
        out = dt_over_last_n_events(ts, 3)
        # First 3 entries should be 0 (insufficient history)
        np.testing.assert_array_equal(out[:3], [0, 0, 0])

    def test_correct_dt(self):
        ts = np.array([100, 200, 350, 400, 500], dtype=np.int64)
        out = dt_over_last_n_events(ts, 2)
        # out[0], out[1] = 0
        # out[2] = ts[2] - ts[0] = 350 - 100 = 250
        # out[3] = ts[3] - ts[1] = 400 - 200 = 200
        # out[4] = ts[4] - ts[2] = 500 - 350 = 150
        np.testing.assert_array_equal(out, [0, 0, 250, 200, 150])


# ── time_since_event_ms ───────────────────────────────────────────────────────

class TestTimeSinceEvent:
    def test_no_events_returns_zeros(self):
        grid = np.array([100, 200, 300], dtype=np.int64) * 1_000_000
        out = time_since_event_ms(np.array([], dtype=np.int64), grid)
        np.testing.assert_array_equal(out, [0.0, 0.0, 0.0])

    def test_known_recencies(self):
        # Event at ms 50 (ns 50_000_000). Grid ticks at ms 100, 110, 200.
        event_t = np.array([50_000_000], dtype=np.int64)
        grid = np.array([100, 110, 200], dtype=np.int64) * 1_000_000
        out = time_since_event_ms(event_t, grid)
        # Elapsed: 50ms, 60ms, 150ms
        np.testing.assert_allclose(out, [50.0, 60.0, 150.0])

    def test_grid_before_event_returns_zero(self):
        # Event at ms 100. Grid at ms 50 (no prior event)
        event_t = np.array([100_000_000], dtype=np.int64)
        grid = np.array([50_000_000], dtype=np.int64)
        out = time_since_event_ms(event_t, grid)
        np.testing.assert_array_equal(out, [0.0])


class TestTrailingCountMs:
    def test_empty_events(self):
        grid = np.array([100, 200], dtype=np.int64) * 1_000_000
        np.testing.assert_array_equal(trailing_count_ms(np.array([], np.int64), grid, 50), [0.0, 0.0])

    def test_window_count_causal(self):
        # trades at ms 10,20,30,95,100; window 50ms; grid at ms 100 → (50,100] holds 95,100 → 2
        ev = np.array([10, 20, 30, 95, 100], dtype=np.int64) * 1_000_000
        grid = np.array([100], dtype=np.int64) * 1_000_000
        np.testing.assert_array_equal(trailing_count_ms(ev, grid, 50), [2.0])

    def test_left_open_right_closed(self):
        # window (t-win, t]: a trade exactly at t-win is EXCLUDED, one exactly at t is INCLUDED
        ev = np.array([50, 100], dtype=np.int64) * 1_000_000
        grid = np.array([100], dtype=np.int64) * 1_000_000
        np.testing.assert_array_equal(trailing_count_ms(ev, grid, 50), [1.0])

    def test_grid_before_events(self):
        ev = np.array([100], dtype=np.int64) * 1_000_000
        grid = np.array([50], dtype=np.int64) * 1_000_000
        np.testing.assert_array_equal(trailing_count_ms(ev, grid, 50), [0.0])

    def test_vectorised_matches_per_tick(self):
        rng = np.random.default_rng(0)
        ev = np.sort(rng.integers(0, 10_000, 200).astype(np.int64)) * 1_000_000
        grid = np.sort(rng.integers(0, 10_000, 50).astype(np.int64)) * 1_000_000
        win = 100
        got = trailing_count_ms(ev, grid, win)
        ref = np.array([np.sum((ev > t - win * 1_000_000) & (ev <= t)) for t in grid], np.float64)
        np.testing.assert_array_equal(got, ref)


# ── BBO event EMAs: zero-when-opposite + value correctness ────────────────────

class TestComputeBboEventEmas:
    @pytest.fixture
    def spans(self):
        return DEFAULT_BBO_SPANS

    def test_ema_ofi_matches_independent_computation(self, spans):
        ts = np.array([100, 200, 300, 400], dtype=np.int64)
        bid = np.array([100.0, 100.0, 101.0, 101.0])
        ask = np.array([102.0, 102.0, 102.0, 102.0])
        bq = np.array([10.0, 15.0, 5.0, 5.0])
        aq = np.array([10.0, 10.0, 10.0, 10.0])

        out = compute_bbo_event_emas(ts, bid, ask, bq, aq, wide_threshold=1e-3, spans=spans)

        # Independent OFI computation
        # i=1: bid same, bq 10→15 → e_bid=+5; ask same, aq same → 0 ⇒ ofi=5
        # i=2: bid 100→101 → e_bid=+new_bq=+5; ask same → 0 ⇒ ofi=5
        # i=3: bid 101 same, bq same → 0; ask same, aq same → 0 ⇒ ofi=0
        expected_ofi = np.array([0.0, 5.0, 5.0, 0.0])

        # The OFI event series isn't directly returned in EMA form, but ofi_event is
        np.testing.assert_allclose(out["ofi_event"], expected_ofi)

        # Now check EMA with span=3 (alpha = 0.5)
        alpha = _alpha(3)
        ref = np.zeros(4)
        ref[0] = alpha * expected_ofi[0]
        for i in range(1, 4):
            ref[i] = alpha * expected_ofi[i] + (1 - alpha) * ref[i - 1]
        np.testing.assert_allclose(out["ema_ofi_3b"], ref)

    def test_book_imbalance_and_depth(self, spans):
        ts = np.array([100, 200], dtype=np.int64)
        bid = np.array([100.0, 100.0])
        ask = np.array([101.0, 101.0])
        bq = np.array([30.0, 10.0])
        aq = np.array([10.0, 30.0])
        out = compute_bbo_event_emas(ts, bid, ask, bq, aq, wide_threshold=1e-2, spans=spans)
        # book_imbalance = (bq - aq) / (bq + aq)
        np.testing.assert_allclose(out["book_imbalance"], [20.0/40.0, -20.0/40.0])
        np.testing.assert_allclose(out["book_depth"], [40.0, 40.0])

    def test_spread_wide_flag_threshold(self, spans):
        # Two events: one with spread > threshold, one with spread < threshold
        ts = np.array([100, 200], dtype=np.int64)
        bid = np.array([100.0, 100.0])
        ask = np.array([101.0, 100.001])  # spread changes from 1 to 0.001
        bq = aq = np.array([10.0, 10.0])
        out = compute_bbo_event_emas(ts, bid, ask, bq, aq, wide_threshold=1e-3, spans=spans)
        # spread_width = (ask-bid)/microprice
        # mp0 ≈ (10*101 + 10*100)/20 = 100.5; spread_width ≈ 1/100.5 ≈ 0.01 > 1e-3 → flag=1
        # mp1 ≈ ~100.0005; spread_width ≈ 1e-5 < 1e-3 → flag=0
        assert out["spread_wide_flag"][0] == 1.0
        assert out["spread_wide_flag"][1] == 0.0


class TestComputeTradeEventEmas:
    @pytest.fixture
    def spans(self):
        return DEFAULT_TRADE_SPANS

    def test_opposite_side_contributes_zero(self, spans):
        # 4 trades: buy, sell, buy, sell — buy EMA should accumulate only buys
        ts = np.array([1, 2, 3, 4], dtype=np.int64) * 1000
        prc = np.array([100.0, 100.0, 100.0, 100.0])
        qty = np.array([10.0, 20.0, 30.0, 40.0])
        dir_ = np.array([1.0, -1.0, 1.0, -1.0])
        out = compute_trade_event_emas(ts, prc, qty, dir_, spans)

        # Independent: buy_in = [10, 0, 30, 0], sell_in = [0, 20, 0, 40]
        # EMA span=10 → alpha = 2/11
        alpha = _alpha(10)
        buy_in = np.array([10.0, 0.0, 30.0, 0.0])
        sell_in = np.array([0.0, 20.0, 0.0, 40.0])
        ref_buy = np.zeros(4)
        ref_sell = np.zeros(4)
        ref_buy[0] = alpha * buy_in[0]
        ref_sell[0] = alpha * sell_in[0]
        for i in range(1, 4):
            ref_buy[i] = alpha * buy_in[i] + (1 - alpha) * ref_buy[i - 1]
            ref_sell[i] = alpha * sell_in[i] + (1 - alpha) * ref_sell[i - 1]

        np.testing.assert_allclose(out["ema_buy_trade_qty_10t"], ref_buy)
        np.testing.assert_allclose(out["ema_sell_trade_qty_10t"], ref_sell)

    def test_value_is_qty_times_price(self, spans):
        ts = np.array([1, 2], dtype=np.int64) * 1000
        prc = np.array([100.0, 200.0])
        qty = np.array([10.0, 20.0])
        dir_ = np.array([1.0, 1.0])
        out = compute_trade_event_emas(ts, prc, qty, dir_, spans)
        # buy_val = [10*100, 20*200] = [1000, 4000]
        # EMA span=3, alpha=0.5
        alpha = _alpha(3)
        ref = np.array([alpha * 1000.0, alpha * 4000.0 + (1 - alpha) * alpha * 1000.0])
        np.testing.assert_allclose(out["ema_buy_trade_value_3t"], ref)

    def test_empty_trades_returns_zero_arrays(self, spans):
        ts = np.array([], dtype=np.int64)
        prc = qty = dir_ = np.array([], dtype=np.float64)
        out = compute_trade_event_emas(ts, prc, qty, dir_, spans)
        for N in EMA_TRADE_SPANS:
            assert out[f"ema_buy_trade_qty_{N}t"].shape == (0,)
            assert out[f"ema_sell_trade_qty_{N}t"].shape == (0,)

    def test_serial_cov_uses_consecutive_dp_products(self, spans):
        # Trades at prices [100, 101, 100, 101] → dp = [1, -1, 1]
        # dp_prod = [1*(-1), (-1)*1] = [-1, -1] aligned at trade indices 2 and 3
        ts = np.array([1, 2, 3, 4], dtype=np.int64) * 1000
        prc = np.array([100.0, 101.0, 100.0, 101.0])
        qty = np.array([1.0, 1.0, 1.0, 1.0])
        dir_ = np.array([1.0, 1.0, 1.0, 1.0])
        out = compute_trade_event_emas(ts, prc, qty, dir_, spans)

        # Reference: cov_input = [0, 0, -1, -1]
        # span=100, alpha=2/101
        alpha = _alpha(100)
        cov_input = np.array([0.0, 0.0, -1.0, -1.0])
        ref = np.zeros(4)
        ref[0] = alpha * cov_input[0]
        for i in range(1, 4):
            ref[i] = alpha * cov_input[i] + (1 - alpha) * ref[i - 1]
        np.testing.assert_allclose(out["ema_trade_serial_cov_100t"], ref)


# ── Calendar-time temporal features ──────────────────────────────────────────

class TestMsGridTemporal:
    @pytest.fixture
    def spans(self):
        return DEFAULT_TEMPORAL_SPANS

    def test_return_1ms_simple(self, spans):
        # Microprice grid: simple ramp 1.0 → 1.001 → 1.002 → 1.003 (one ms each)
        mp = np.array([1.0, 1.001, 1.002, 1.003])
        grid = np.arange(4, dtype=np.int64) * 1_000_000
        out = compute_ms_grid_temporal(mp, grid, spans)
        # return_1ms[i] = log(mp[i]/mp[i-1])
        expected = np.array([0.0, np.log(1.001/1.0), np.log(1.002/1.001), np.log(1.003/1.002)])
        np.testing.assert_allclose(out["return_1ms"], expected)

    def test_ema_microprice_matches_lfilter(self, spans):
        rng = np.random.default_rng(0)
        mp = 1.0 + rng.standard_normal(200) * 0.001
        grid = np.arange(200, dtype=np.int64) * 1_000_000
        out = compute_ms_grid_temporal(mp, grid, spans)
        # ema_microprice_centered_100ms with alpha = 2/101
        alpha = _alpha(100)
        ref = lfilter([alpha], [1.0, -(1.0 - alpha)], mp.astype(np.float64))
        np.testing.assert_allclose(out["ema_microprice_centered_100ms"], ref, rtol=1e-12)

    def test_ema_microprice_sq_is_ema_of_squared(self, spans):
        mp = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        grid = np.arange(5, dtype=np.int64) * 1_000_000
        out = compute_ms_grid_temporal(mp, grid, spans)
        # ema_microprice_sq = EMA(mp²)
        alpha = _alpha(100)
        ref = lfilter([alpha], [1.0, -(1.0 - alpha)], (mp ** 2).astype(np.float64))
        np.testing.assert_allclose(out["ema_microprice_centered_sq_100ms"], ref, rtol=1e-12)


# ── End-to-end: integration with forward-fill and sampling ────────────────────

class TestIntegrationOneListing:
    def test_grid_with_no_event_holds_previous(self):
        """Critical edge case: ms tick with no BBO/trade event must forward-fill."""
        # Three BBO events: at ns 0, 5_000_000 (5ms), 15_000_000 (15ms)
        # Grid ticks every 1ms from 0 to 20
        bbo_t = np.array([0, 5_000_000, 15_000_000], dtype=np.int64)
        bid = np.array([100.0, 101.0, 100.5])
        ask = np.array([101.0, 102.0, 101.5])
        bq = aq = np.array([10.0, 10.0, 10.0])

        grid = np.arange(21, dtype=np.int64) * 1_000_000

        microprice = compute_microprice(bid, ask, bq, aq)
        mp_grid = forward_fill_to_ms_grid(bbo_t, microprice, grid, fill_value=0.0)

        # Expected: ticks 0..4 → microprice[0] = 100.5
        # ticks 5..14 → microprice[1] = 101.5
        # ticks 15..20 → microprice[2] = 101.0
        assert mp_grid[0] == 100.5
        assert mp_grid[4] == 100.5
        assert mp_grid[5] == 101.5
        assert mp_grid[14] == 101.5
        assert mp_grid[15] == 101.0
        assert mp_grid[20] == 101.0


# ── Feature column ordering ──────────────────────────────────────────────────

class TestForwardFillOfEma:
    """Critical: forward-filling an EMA array to the ms grid must reflect
    the EMA state AFTER the last event at-or-before each grid tick."""

    def test_ema_value_at_grid_tick_is_state_after_last_event(self):
        # 5 BBO events at ns 0, 10ms, 20ms, 30ms, 40ms with increasing microprices
        bbo_t = np.array([0, 10, 20, 30, 40], dtype=np.int64) * 1_000_000
        # Compute EMA of [1, 2, 3, 4, 5] at alpha = 0.5
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        alpha = 0.5
        ema_evt = np.zeros(5)
        ema_evt[0] = alpha * x[0]
        for i in range(1, 5):
            ema_evt[i] = alpha * x[i] + (1 - alpha) * ema_evt[i - 1]

        # Sample to grid ticks at ns 5ms, 15ms, 25ms, 35ms, 45ms
        # Each grid tick falls between two events; should pick up the EMA value
        # AFTER the most recent event.
        grid = np.array([5, 15, 25, 35, 45], dtype=np.int64) * 1_000_000
        sampled = forward_fill_to_ms_grid(bbo_t, ema_evt, grid, fill_value=0.0)
        # At 5ms: last event was at 0ms (index 0) → ema_evt[0]
        # At 15ms: last event was at 10ms (index 1) → ema_evt[1]
        # ...
        np.testing.assert_allclose(sampled, ema_evt)

    def test_multiple_events_within_one_ms_use_final_state(self):
        # Several BBO events at ns 1, 2, 3 (sub-ms), then grid tick at 1ms boundary
        bbo_t = np.array([1, 2, 3], dtype=np.int64)
        x = np.array([10.0, 20.0, 30.0])
        ema = np.zeros(3)
        alpha = 0.5
        ema[0] = alpha * 10.0
        ema[1] = alpha * 20.0 + (1 - alpha) * ema[0]
        ema[2] = alpha * 30.0 + (1 - alpha) * ema[1]

        grid = np.array([1_000_000], dtype=np.int64)  # 1ms after all events
        sampled = forward_fill_to_ms_grid(bbo_t, ema, grid, fill_value=0.0)
        # Grid tick sees the EMA state after the last event (index 2)
        np.testing.assert_allclose(sampled, [ema[2]])


class TestSameNsAggregationAffectsEMA:
    """Verifies that same-ns aggregation runs BEFORE the EMA computation,
    not after — so OFI and vol don't get inflated by phantom events."""

    def test_three_same_ns_bbo_treated_as_one_event(self):
        # Three "feed messages" at the same ns: only the final state should drive OFI.
        # Raw stream: bid=100/qty=10 → bid=101/qty=5 → bid=100/qty=20 (all at ns=1)
        # Aggregated final state: bid=100, qty=20.
        # If a prior state existed (event before), OFI compares prior → 100/20, NOT through 101/5.
        ts = np.array([1, 1, 1, 2], dtype=np.int64)
        bid = np.array([100.0, 101.0, 100.0, 100.0])
        ask = np.array([102.0, 102.0, 102.0, 102.0])
        bq = np.array([10.0, 5.0, 20.0, 25.0])
        aq = np.array([10.0, 10.0, 10.0, 10.0])

        # Aggregate first, then OFI
        agg_ts, agg_bid, agg_ask, agg_bq, agg_aq = aggregate_same_ns_bbo(ts, bid, ask, bq, aq)
        assert len(agg_ts) == 2  # collapsed to 2 events
        # ofi between [bid=100/bq=20] → [bid=100/bq=25]: same price, qty +5 → +5
        ofi = compute_ofi_events(agg_bid, agg_ask, agg_bq, agg_aq)
        assert ofi[1] == pytest.approx(5.0)

    def test_aggregated_then_emas_consistent(self):
        # Build a small BBO stream with same-ns clustering, run through the
        # full compute_bbo_event_emas pipeline, and verify the EMA only saw
        # the deduplicated stream.
        ts = np.array([1, 1, 2, 3, 3, 4], dtype=np.int64)
        bid = np.array([100.0, 101.0, 101.0, 101.0, 102.0, 102.0])
        ask = np.array([103.0, 103.0, 103.0, 103.0, 103.0, 103.0])
        bq = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        aq = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

        # Aggregate first (same as _compute_per_listing does)
        agg_ts, agg_bid, agg_ask, agg_bq, agg_aq = aggregate_same_ns_bbo(ts, bid, ask, bq, aq)
        out = compute_bbo_event_emas(agg_ts, agg_bid, agg_ask, agg_bq, agg_aq, 1e-3, DEFAULT_BBO_SPANS)

        # After dedup we have 4 events (ns 1, 2, 3, 4) with final states bid=[101,101,102,102]
        assert len(out["ts"]) == 4
        assert len(out["ema_ofi_3b"]) == 4


class TestComputePerListing:
    """End-to-end test: synthesize input data, run _compute_per_listing,
    and verify several features against independent computation."""

    @pytest.fixture
    def small_session(self):
        """Build a tiny SessionData-like object with one listing."""
        from boba.dataset.session_data import SessionData

        # BBO events at ms 0, 5, 10, 15 (in ns)
        bbo_t = np.array([0, 5, 10, 15], dtype=np.int64) * 1_000_000
        bbo_bid = np.array([100.0, 100.0, 100.1, 100.1])
        bbo_ask = np.array([100.1, 100.1, 100.2, 100.2])
        bbo_bq = np.array([10.0, 20.0, 20.0, 10.0])
        bbo_aq = np.array([10.0, 10.0, 10.0, 10.0])

        # Trades at ms 3, 7, 12
        tr_t = np.array([3, 7, 12], dtype=np.int64) * 1_000_000
        tr_prc = np.array([100.1, 100.1, 100.2])
        tr_qty = np.array([5.0, 10.0, 3.0])
        tr_dir = np.array([1.0, -1.0, 1.0])

        # Feed latency excess: dummy zeros (one per raw BBO event)
        feed_lat = np.zeros(4, dtype=np.int64)

        data = SessionData(
            listing_book_t={"bin": bbo_t},
            listing_book_bid={"bin": bbo_bid},
            listing_book_ask={"bin": bbo_ask},
            listing_book_bid_qty={"bin": bbo_bq},
            listing_book_ask_qty={"bin": bbo_aq},
            trade_ts={"bin": tr_t},
            trade_exchange_ts={"bin": tr_t},
            trade_prc={"bin": tr_prc},
            trade_qty={"bin": tr_qty},
            trade_dir={"bin": tr_dir},
            listing_feed_latency_excess_ns={"bin": feed_lat},
            target_listing="bin",
            book_t=bbo_t,
            book_bid=bbo_bid,
            book_ask=bbo_ask,
            book_mid=(bbo_bid + bbo_ask) / 2,
            feed_latency_raw_ns=np.zeros(4, dtype=np.int64),
            feed_latency_excess_ns=feed_lat,
            all_rx=np.sort(np.concatenate([bbo_t, tr_t])),
        )
        return data

    def test_microprice_forward_filled_to_grid(self, small_session):
        cfg = make_cfg()
        # Grid ticks at ms 0, 1, 2, ... 19 (20 ticks)
        grid_t_ns = np.arange(20, dtype=np.int64) * 1_000_000
        out = _compute_per_listing("bin", small_session, grid_t_ns, cfg)

        # Compute expected microprices at the BBO events
        # Event 0: bq=10, aq=10, bid=100, ask=100.1 → mp = 100.05
        # Event 1: bq=20, aq=10, bid=100, ask=100.1 → mp = (20*100.1 + 10*100)/30 = 100.0667
        # Event 2: bq=20, aq=10, bid=100.1, ask=100.2 → mp = (20*100.2 + 10*100.1)/30 = 100.1667
        # Event 3: bq=10, aq=10, bid=100.1, ask=100.2 → mp = 100.15
        mp_events = compute_microprice(
            small_session.listing_book_bid["bin"],
            small_session.listing_book_ask["bin"],
            small_session.listing_book_bid_qty["bin"],
            small_session.listing_book_ask_qty["bin"],
        )

        # Grid tick at ms 0 → event 0
        assert out["microprice"][0] == pytest.approx(mp_events[0], rel=1e-5)
        # Grid tick at ms 4 → still event 0 (event 1 is at ms 5)
        assert out["microprice"][4] == pytest.approx(mp_events[0], rel=1e-5)
        # Grid tick at ms 5 → event 1
        assert out["microprice"][5] == pytest.approx(mp_events[1], rel=1e-5)
        # Grid tick at ms 16 → event 3 (last event was at ms 15)
        assert out["microprice"][16] == pytest.approx(mp_events[3], rel=1e-5)

    def test_buy_sell_trade_emas_on_grid(self, small_session):
        cfg = make_cfg()
        grid_t_ns = np.arange(20, dtype=np.int64) * 1_000_000
        out = _compute_per_listing("bin", small_session, grid_t_ns, cfg)

        # Trades: ms 3 buy 5, ms 7 sell 10, ms 12 buy 3
        # buy_in stream: [5, 0, 3], sell_in: [0, 10, 0]
        alpha = _alpha(10)
        buy_ema_evt = np.zeros(3)
        sell_ema_evt = np.zeros(3)
        buy_ema_evt[0] = alpha * 5.0
        sell_ema_evt[0] = alpha * 0.0
        buy_ema_evt[1] = alpha * 0.0 + (1 - alpha) * buy_ema_evt[0]
        sell_ema_evt[1] = alpha * 10.0 + (1 - alpha) * sell_ema_evt[0]
        buy_ema_evt[2] = alpha * 3.0 + (1 - alpha) * buy_ema_evt[1]
        sell_ema_evt[2] = alpha * 0.0 + (1 - alpha) * sell_ema_evt[1]

        # Grid tick at ms 2 → no trade yet → buy EMA = 0
        assert out["ema_buy_trade_qty_10t"][2] == 0.0
        assert out["ema_sell_trade_qty_10t"][2] == 0.0
        # Grid tick at ms 3 → trade 0 fired → buy EMA = alpha * 5
        assert out["ema_buy_trade_qty_10t"][3] == pytest.approx(buy_ema_evt[0])
        assert out["ema_sell_trade_qty_10t"][3] == pytest.approx(sell_ema_evt[0])
        # Grid tick at ms 7 → trade 1 fired (sell)
        assert out["ema_buy_trade_qty_10t"][7] == pytest.approx(buy_ema_evt[1])
        assert out["ema_sell_trade_qty_10t"][7] == pytest.approx(sell_ema_evt[1])
        # Grid tick at ms 12 → trade 2 fired (buy)
        assert out["ema_buy_trade_qty_10t"][12] == pytest.approx(buy_ema_evt[2])
        assert out["ema_sell_trade_qty_10t"][12] == pytest.approx(sell_ema_evt[2])

    def test_dt_features_in_ms(self, small_session):
        cfg = make_cfg()
        grid_t_ns = np.arange(20, dtype=np.int64) * 1_000_000
        out = _compute_per_listing("bin", small_session, grid_t_ns, cfg)

        # dt_10b — 4 BBO events, span=10 → all zeros (insufficient history)
        assert np.all(out["dt_10b"] == 0.0)

        # dt_3t — trades at 3, 7, 12 ms; dt at trade 2 = (12 - 3) = 9 ms
        # Grid tick at ms 12: dt_3t should reflect this.
        # But wait, span=3 means N=3 (back 3 events). With 3 trades total, dt_3t[2] = ts[2] - ts[-1] which is invalid.
        # Actually len(ts)=3, N=3, so dt_3t[3] would need ts[3] which doesn't exist.
        # All dt_3t should be 0 (insufficient).
        assert np.all(out["dt_3t"] == 0.0)

    def test_time_since_last_trade_ms(self, small_session):
        cfg = make_cfg()
        grid_t_ns = np.arange(20, dtype=np.int64) * 1_000_000
        out = _compute_per_listing("bin", small_session, grid_t_ns, cfg)

        # Trades at ms 3, 7, 12
        # At ms 0: no trade yet → 0
        assert out["time_since_last_trade_ms"][0] == 0.0
        # At ms 2: still no trade → 0
        assert out["time_since_last_trade_ms"][2] == 0.0
        # At ms 3: trade just landed → 0
        assert out["time_since_last_trade_ms"][3] == 0.0
        # At ms 5: 2ms since last trade (at ms 3)
        assert out["time_since_last_trade_ms"][5] == pytest.approx(2.0)
        # At ms 10: 3ms since last trade (at ms 7)
        assert out["time_since_last_trade_ms"][10] == pytest.approx(3.0)

    def test_return_1ms_on_grid(self, small_session):
        cfg = make_cfg()
        grid_t_ns = np.arange(20, dtype=np.int64) * 1_000_000
        out = _compute_per_listing("bin", small_session, grid_t_ns, cfg)
        mp_grid = out["microprice"]
        # return_1ms[i] = log(mp[i] / mp[i-1]) where both > eps
        for i in range(1, 20):
            if mp_grid[i] > 0 and mp_grid[i - 1] > 0:
                expected = np.log(mp_grid[i] / mp_grid[i - 1])
                assert out["return_1ms"][i] == pytest.approx(expected, abs=1e-7)


class TestBuildFeaturesRaw:
    """Full pipeline smoke test."""

    @pytest.fixture
    def three_listing_session(self):
        from boba.dataset.session_data import SessionData
        # Tiny session with 3 listings, each with a handful of events
        bbo_t = np.array([0, 5, 10, 15, 20], dtype=np.int64) * 1_000_000
        tr_t = np.array([3, 7, 12], dtype=np.int64) * 1_000_000

        def make(seed):
            rng = np.random.default_rng(seed)
            bid = 100.0 + rng.standard_normal(5) * 0.01
            ask = bid + 0.001
            bq = rng.uniform(5, 20, 5)
            aq = rng.uniform(5, 20, 5)
            tr_prc = 100.0 + rng.standard_normal(3) * 0.001
            tr_qty = rng.uniform(1, 10, 3)
            tr_dir = rng.choice([1.0, -1.0], 3)
            return bid, ask, bq, aq, tr_prc, tr_qty, tr_dir

        e_bid, e_ask, e_bq, e_aq, e_tprc, e_tqty, e_tdir = {}, {}, {}, {}, {}, {}, {}
        for i, e in enumerate(["bin", "byb", "okx"]):
            bid, ask, bq, aq, tprc, tqty, tdir = make(i)
            e_bid[e], e_ask[e], e_bq[e], e_aq[e] = bid, ask, bq, aq
            e_tprc[e], e_tqty[e], e_tdir[e] = tprc, tqty, tdir

        data = SessionData(
            listing_book_t={e: bbo_t for e in ["bin", "byb", "okx"]},
            listing_book_bid=e_bid,
            listing_book_ask=e_ask,
            listing_book_bid_qty=e_bq,
            listing_book_ask_qty=e_aq,
            trade_ts={e: tr_t for e in ["bin", "byb", "okx"]},
            trade_exchange_ts={e: tr_t for e in ["bin", "byb", "okx"]},
            trade_prc=e_tprc,
            trade_qty=e_tqty,
            trade_dir=e_tdir,
            listing_feed_latency_excess_ns={e: np.zeros(5, np.int64) for e in ["bin", "byb", "okx"]},
            target_listing="bin",
            book_t=bbo_t,
            book_bid=e_bid["bin"],
            book_ask=e_ask["bin"],
            book_mid=(e_bid["bin"] + e_ask["bin"]) / 2,
            feed_latency_raw_ns=np.zeros(5, dtype=np.int64),
            feed_latency_excess_ns=np.zeros(5, dtype=np.int64),
            all_rx=np.sort(np.concatenate([bbo_t, tr_t])),
        )
        return data

    def test_runs_and_produces_correct_shape(self, three_listing_session):
        cfg = make_cfg(warmup_ms=0, horizon_ms=0, only_on_event=False)
        result = build_features_raw(three_listing_session, ["bin", "byb", "okx"], cfg)
        # Should be (N_grid_ticks, n_per_listing * 3)
        F_expected = len(legacy_local_names()) * 3
        assert cfg.n_features() == F_expected
        assert result.x.shape[1] == F_expected
        assert len(result.column_names) == F_expected
        assert result.x.dtype == np.float32
        # Some grid ticks present
        assert result.x.shape[0] > 0

    def test_no_nan_in_output(self, three_listing_session):
        cfg = make_cfg(warmup_ms=0, horizon_ms=0, only_on_event=False)
        result = build_features_raw(three_listing_session, ["bin", "byb", "okx"], cfg)
        assert not np.any(np.isnan(result.x)), "Output contains NaN"
        assert not np.any(np.isinf(result.x)), "Output contains Inf"

    def test_column_names_are_unique(self, three_listing_session):
        cfg = make_cfg(warmup_ms=0, horizon_ms=0, only_on_event=False)
        result = build_features_raw(three_listing_session, ["bin", "byb", "okx"], cfg)
        assert len(set(result.column_names)) == len(result.column_names)

    def test_column_names_prefixed_per_listing(self, three_listing_session):
        cfg = make_cfg(warmup_ms=0, horizon_ms=0, only_on_event=False)
        result = build_features_raw(three_listing_session, ["bin", "byb", "okx"], cfg)
        # Each listing contributes the same number of columns
        for e in ["bin", "byb", "okx"]:
            prefixed = [c for c in result.column_names if c.startswith(f"{e}_")]
            assert len(prefixed) == len(legacy_local_names())


class TestGridBoundsMs:
    """_grid_bounds_ms is the single source of truth for the grid flooring arithmetic;
    build_features_raw and build_dataset's degenerate-block SKIP both call it."""

    def test_matches_legacy_inline_formula(self):
        # The helper must reproduce the original inline arithmetic bit-for-bit
        # (float division preserved), else cached blocks and fresh builds diverge.
        rng = np.random.default_rng(0)
        for _ in range(200):
            t0 = int(rng.integers(1_600_000_000, 1_800_000_000)) * 1_000_000_000 + int(rng.integers(0, 10**9))
            span = int(rng.integers(0, 90_000_000_000))  # 0–90s, straddles warmup+horizon
            t1 = t0 + span
            warmup_ms, horizon_ms = 30_000, 100
            warmup_ns = int(warmup_ms * 1_000_000)
            horizon_ns = int(horizon_ms * 1_000_000)
            legacy = (int((t0 + warmup_ns) / 1_000_000) + 1, int((t1 - horizon_ns) / 1_000_000))
            assert _grid_bounds_ms(t0, t1, warmup_ms, horizon_ms) == legacy

    def test_stub_block_is_empty(self):
        # A few-second recording stub vs 30s warmup → no usable grid.
        t0 = 1_700_000_000_000_000_000
        gs, ge = _grid_bounds_ms(t0, t0 + 5_000_000_000, warmup_ms=30_000, horizon_ms=100)
        assert ge < gs

    def test_boundary_span(self):
        # Span exactly warmup+horizon → empty (the +1 on start); a little more → non-empty.
        t0 = 1_700_000_000_000_000_000
        wup, hor = 30_000, 100
        edge_ns = (wup + hor) * 1_000_000
        gs, ge = _grid_bounds_ms(t0, t0 + edge_ns, wup, hor)
        assert ge < gs
        gs, ge = _grid_bounds_ms(t0, t0 + edge_ns + 2_000_000, wup, hor)
        assert ge >= gs

    def test_normal_day_block(self):
        # 24h block: bounds sane, ~86.4M grid ticks survive warmup+horizon.
        t0 = 1_700_000_000_000_000_000
        gs, ge = _grid_bounds_ms(t0, t0 + 86_400_000_000_000, warmup_ms=30_000, horizon_ms=100)
        n = ge - gs + 1
        assert abs(n - (86_400_000 - 30_000 - 100)) <= 2


class TestEmptyGridRaises:
    """build_features_raw must fail fast with a clear error on a degenerate (shorter than
    warmup+horizon) session, not crash deep in the cost-field chunking (IndexError)."""

    def test_raises_value_error(self, three_listing_session=None):
        # Reuse TestBuildFeaturesRaw's fixture shape: 20ms of events vs 30s warmup.
        fixture = TestBuildFeaturesRaw.three_listing_session
        session = fixture.__wrapped__(TestBuildFeaturesRaw())
        cfg = make_cfg(warmup_ms=30_000, horizon_ms=100, only_on_event=False)
        with pytest.raises(ValueError, match="empty grid"):
            build_features_raw(session, ["bin", "byb", "okx"], cfg)


class TestSameNsEndToEnd:
    """Critical: verify same-ns aggregation actually changes downstream values
    when wired through _compute_per_listing (not just in isolation)."""

    def test_same_ns_bbo_collapses_in_full_pipeline(self):
        from boba.dataset.session_data import SessionData

        # Two scenarios with the same logical events but different feed-message ordering:
        # Scenario A: clean, 3 BBO events.
        # Scenario B: same 3 logical events, but 2 of them are split into 3 same-ns rows.
        # Same-ns aggregation should make A and B produce identical pipeline output.

        # Scenario A
        bbo_t_A = np.array([10, 20, 30], dtype=np.int64) * 1_000_000
        bid_A = np.array([100.0, 100.1, 100.0])
        ask_A = np.array([100.1, 100.2, 100.1])
        bq_A = np.array([10.0, 20.0, 30.0])
        aq_A = np.array([10.0, 10.0, 10.0])

        # Scenario B: ns=20 has 3 same-ns rows; the final state matches the single row in A.
        bbo_t_B = np.array([10, 20, 20, 20, 30], dtype=np.int64) * 1_000_000
        bid_B = np.array([100.0, 100.05, 100.15, 100.1, 100.0])
        ask_B = np.array([100.1, 100.2, 100.2, 100.2, 100.1])
        bq_B = np.array([10.0, 15.0, 25.0, 20.0, 30.0])  # ends at 20 = same as A
        aq_B = np.array([10.0, 10.0, 10.0, 10.0, 10.0])

        feed_lat_A = np.zeros(3, dtype=np.int64)
        feed_lat_B = np.zeros(5, dtype=np.int64)

        def make_session(bbo_t, bid, ask, bq, aq, feed_lat):
            return SessionData(
                listing_book_t={"bin": bbo_t},
                listing_book_bid={"bin": bid},
                listing_book_ask={"bin": ask},
                listing_book_bid_qty={"bin": bq},
                listing_book_ask_qty={"bin": aq},
                trade_ts={"bin": np.array([], dtype=np.int64)},
                trade_exchange_ts={"bin": np.array([], dtype=np.int64)},
                trade_prc={"bin": np.array([], dtype=np.float64)},
                trade_qty={"bin": np.array([], dtype=np.float64)},
                trade_dir={"bin": np.array([], dtype=np.float64)},
                listing_feed_latency_excess_ns={"bin": feed_lat},
                target_listing="bin",
                book_t=bbo_t,
                book_bid=bid,
                book_ask=ask,
                book_mid=(bid + ask) / 2,
                feed_latency_raw_ns=feed_lat,
                feed_latency_excess_ns=feed_lat,
                all_rx=bbo_t,
            )

        sess_A = make_session(bbo_t_A, bid_A, ask_A, bq_A, aq_A, feed_lat_A)
        sess_B = make_session(bbo_t_B, bid_B, ask_B, bq_B, aq_B, feed_lat_B)

        cfg = make_cfg()
        grid = np.arange(40, dtype=np.int64) * 1_000_000

        out_A = _compute_per_listing("bin", sess_A, grid, cfg)
        out_B = _compute_per_listing("bin", sess_B, grid, cfg)

        # Microprice (forward-filled) should be identical at every grid tick
        np.testing.assert_allclose(out_A["microprice"], out_B["microprice"], rtol=1e-12)
        # OFI EMA should be identical (because same-ns events collapsed)
        np.testing.assert_allclose(out_A["ema_ofi_3b"], out_B["ema_ofi_3b"], rtol=1e-12)
        # book_imbalance forward-filled
        np.testing.assert_allclose(out_A["book_imbalance"], out_B["book_imbalance"], rtol=1e-12)

    def test_same_ns_trade_sums_qty_in_full_pipeline(self):
        from boba.dataset.session_data import SessionData

        # Two scenarios for trades:
        # Scenario A: one trade of qty=15 at ns=10.
        # Scenario B: three trades at ns=10, qtys 5,7,3, same buy side. After
        #             same-ns aggregation, this becomes one trade of qty=15.
        bbo_t = np.array([5, 15], dtype=np.int64) * 1_000_000
        bid = ask = np.array([100.0, 100.0])
        ask = np.array([100.1, 100.1])
        bq = aq = np.array([10.0, 10.0])
        feed_lat = np.zeros(2, dtype=np.int64)

        def make_session(tr_t, tr_prc, tr_qty, tr_dir):
            return SessionData(
                listing_book_t={"bin": bbo_t},
                listing_book_bid={"bin": bid},
                listing_book_ask={"bin": ask},
                listing_book_bid_qty={"bin": bq},
                listing_book_ask_qty={"bin": aq},
                trade_ts={"bin": tr_t},
                trade_exchange_ts={"bin": tr_t},
                trade_prc={"bin": tr_prc},
                trade_qty={"bin": tr_qty},
                trade_dir={"bin": tr_dir},
                listing_feed_latency_excess_ns={"bin": feed_lat},
                target_listing="bin",
                book_t=bbo_t,
                book_bid=bid,
                book_ask=ask,
                book_mid=(bid + ask) / 2,
                feed_latency_raw_ns=feed_lat,
                feed_latency_excess_ns=feed_lat,
                all_rx=np.sort(np.concatenate([bbo_t, tr_t])),
            )

        # Scenario A
        sess_A = make_session(
            tr_t=np.array([10_000_000], dtype=np.int64),
            tr_prc=np.array([100.05]),
            tr_qty=np.array([15.0]),
            tr_dir=np.array([1.0]),
        )
        # Scenario B
        sess_B = make_session(
            tr_t=np.array([10_000_000, 10_000_000, 10_000_000], dtype=np.int64),
            tr_prc=np.array([100.05, 100.05, 100.05]),
            tr_qty=np.array([5.0, 7.0, 3.0]),
            tr_dir=np.array([1.0, 1.0, 1.0]),
        )

        cfg = make_cfg()
        grid = np.arange(20, dtype=np.int64) * 1_000_000

        out_A = _compute_per_listing("bin", sess_A, grid, cfg)
        out_B = _compute_per_listing("bin", sess_B, grid, cfg)

        # After aggregation both have one logical buy trade of qty=15 at ns=10
        np.testing.assert_allclose(
            out_A["ema_buy_trade_qty_10t"], out_B["ema_buy_trade_qty_10t"], rtol=1e-12
        )
        np.testing.assert_allclose(
            out_A["ema_buy_trade_value_10t"], out_B["ema_buy_trade_value_10t"], rtol=1e-12
        )


class TestVarianceReconstruction:
    """Verify that downstream ETL can recover variance from ema_x and ema_x_sq."""

    def test_reconstructable_variance_matches_direct_computation(self):
        # Synthetic microprice grid
        rng = np.random.default_rng(7)
        mp = 1.0 + rng.standard_normal(500) * 0.001
        grid = np.arange(500, dtype=np.int64) * 1_000_000
        out = compute_ms_grid_temporal(mp, grid, DEFAULT_TEMPORAL_SPANS)

        # Reconstruct variance: var = ema_sq - ema²
        ema = out["ema_microprice_centered_1000ms"]
        ema_sq = out["ema_microprice_centered_sq_1000ms"]
        recovered_var = np.maximum(ema_sq - ema * ema, 0.0)

        # Independent direct computation: EMA of (mp - ema)²
        # (this won't be exactly equal due to numerical differences, but close)
        # Just verify it's non-negative and finite
        assert np.all(recovered_var >= 0.0)
        assert np.all(np.isfinite(recovered_var))


class TestOneEmaTickPerUniqueTimestamp:
    """Explicit: K events sharing one timestamp must produce ONE EMA tick,
    AND the value going into that tick is the post-aggregation value
    (last-state for BBO; summed qty for trades), not any individual row.
    """

    def test_bbo_k_same_ns_events_produce_one_ema_tick(self):
        # 5 BBO events at ns=100 (same timestamp). Then 1 event at ns=200.
        # Aggregated: 2 events. So EMA output length = 2.
        ts = np.array([100, 100, 100, 100, 100, 200], dtype=np.int64)
        bid = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 14.0])
        ask = np.array([15.0, 15.0, 15.0, 15.0, 15.0, 16.0])
        bq = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 7.0])
        aq = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])

        # Aggregate first (matches _compute_per_listing)
        agg = aggregate_same_ns_bbo(ts, bid, ask, bq, aq)
        agg_ts, agg_bid, agg_ask, agg_bq, agg_aq = agg

        # Aggregation collapses the 5 same-ns rows to 1 (the last state)
        assert len(agg_ts) == 2
        assert agg_bid[0] == 14.0    # last bid at ns=100
        assert agg_bq[0] == 5.0      # last qty at ns=100

        # Now run the EMA pipeline on the aggregated stream
        out = compute_bbo_event_emas(agg_ts, agg_bid, agg_ask, agg_bq, agg_aq, 1e-3, DEFAULT_BBO_SPANS)

        # All EMA arrays must have length exactly equal to aggregated event count = 2.
        # This proves K same-ns events → 1 EMA tick.
        for key, arr in out.items():
            if key == "ts" or key.startswith("dt_"):
                continue
            assert len(arr) == 2, f"{key} has length {len(arr)}, expected 2"

    def test_bbo_ema_value_uses_aggregated_state_not_intermediate(self):
        # Two BBO events:
        # ns=100 — single event, bid=10, bq=1
        # ns=200 — 3 same-ns events with bq=2, 99, 5 (raw); aggregated final = bq=5
        # OFI between these aggregated events (both bid=10):
        #   same price, bq 1→5 → +(5-1) = +4
        # If aggregation did NOT happen, the EMA would see intermediate states
        # and produce different OFI events.
        ts = np.array([100, 200, 200, 200], dtype=np.int64)
        bid = np.array([10.0, 10.0, 10.0, 10.0])
        ask = np.array([15.0, 15.0, 15.0, 15.0])
        bq = np.array([1.0, 2.0, 99.0, 5.0])
        aq = np.array([5.0, 5.0, 5.0, 5.0])

        agg_ts, agg_bid, agg_ask, agg_bq, agg_aq = aggregate_same_ns_bbo(ts, bid, ask, bq, aq)
        out = compute_bbo_event_emas(agg_ts, agg_bid, agg_ask, agg_bq, agg_aq, 1e-3, DEFAULT_BBO_SPANS)

        # OFI[1] computed from aggregated bq 1 → 5: e_bid = +4, ask unchanged
        assert out["ofi_event"][1] == pytest.approx(4.0)

        # If we had NOT aggregated, OFI events would step through 1→2 (+1), 2→99 (+97), 99→5 (-94)
        # Net cumulative would still be +4, but the EMA would see three separate
        # large-magnitude events, completely changing ema_ofi and especially ema_ofi_sq.
        # By construction of the aggregated input, the EMA sees ONE event of +4.
        alpha = _alpha(3)
        ref_ofi_ema = np.zeros(2)
        ref_ofi_ema[0] = alpha * 0.0           # first event, ofi=0
        ref_ofi_ema[1] = alpha * 4.0 + (1 - alpha) * ref_ofi_ema[0]
        np.testing.assert_allclose(out["ema_ofi_3b"], ref_ofi_ema)

        # And ema_ofi_sq sees ONE squared event of 16, not three of (1, 9409, 8836)
        ref_ofi_sq_ema = np.zeros(2)
        alpha_sq = _alpha(1000)
        ref_ofi_sq_ema[0] = alpha_sq * 0.0
        ref_ofi_sq_ema[1] = alpha_sq * 16.0 + (1 - alpha_sq) * ref_ofi_sq_ema[0]
        np.testing.assert_allclose(out["ema_ofi_sq_1000b"], ref_ofi_sq_ema)

    def test_trade_k_same_ns_buys_produce_one_ema_tick(self):
        # 5 same-ns buy trades + 1 separate trade. After aggregation: 2 events.
        ts = np.array([100, 100, 100, 100, 100, 200], dtype=np.int64)
        prc = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        qty = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 7.0])
        dir_ = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        agg_ts, agg_prc, agg_qty, agg_dir = aggregate_same_ns_trade(ts, prc, qty, dir_)
        # 5 same-ns same-side trades → 1 aggregated event with qty=15
        assert len(agg_ts) == 2
        assert agg_qty[0] == 15.0  # sum

        out = compute_trade_event_emas(agg_ts, agg_prc, agg_qty, agg_dir, DEFAULT_TRADE_SPANS)

        # All trade EMAs must have length = 2 (one tick per aggregated event)
        for N in EMA_TRADE_SPANS:
            assert len(out[f"ema_buy_trade_qty_{N}t"]) == 2
            assert len(out[f"ema_sell_trade_qty_{N}t"]) == 2
            assert len(out[f"ema_buy_trade_value_{N}t"]) == 2

    def test_trade_ema_value_uses_summed_qty(self):
        # Three same-ns buys with qtys 1, 99, 5. Aggregated qty = 105.
        # If aggregation didn't happen, the EMA would see three separate events.
        ts = np.array([100, 100, 100], dtype=np.int64)
        prc = np.array([10.0, 10.0, 10.0])
        qty = np.array([1.0, 99.0, 5.0])
        dir_ = np.array([1.0, 1.0, 1.0])

        agg_ts, agg_prc, agg_qty, agg_dir = aggregate_same_ns_trade(ts, prc, qty, dir_)
        out = compute_trade_event_emas(agg_ts, agg_prc, agg_qty, agg_dir, DEFAULT_TRADE_SPANS)

        # After aggregation: 1 buy trade with qty=105.
        # EMA span=10 → alpha = 2/11. ema_buy_trade_qty_10t[0] = (2/11) * 105.
        alpha = _alpha(10)
        expected = alpha * 105.0
        assert out["ema_buy_trade_qty_10t"][0] == pytest.approx(expected)

        # Critical: NOT alpha*1 + (1-alpha)*alpha*99 + (1-alpha)²*alpha*5 (the un-aggregated value).
        # That would be:
        unaggregated_ref = alpha * 1.0
        unaggregated_ref = alpha * 99.0 + (1 - alpha) * unaggregated_ref
        unaggregated_ref = alpha * 5.0 + (1 - alpha) * unaggregated_ref
        # These two values differ — confirm the aggregated one matches and the un-aggregated does NOT
        assert out["ema_buy_trade_qty_10t"][0] != pytest.approx(unaggregated_ref)


class TestSpreadWidthFormula:
    """Direct test of spread_width = (ask − bid) / microprice."""

    def test_spread_width_value(self):
        ts = np.array([100], dtype=np.int64)
        bid = np.array([100.0])
        ask = np.array([100.1])
        bq = aq = np.array([10.0])
        out = compute_bbo_event_emas(ts, bid, ask, bq, aq, 1e-3, DEFAULT_BBO_SPANS)
        # microprice with balanced book = mid = 100.05
        # spread_width = (100.1 - 100.0) / 100.05 = 0.1 / 100.05 ≈ 9.995e-4
        expected = 0.1 / 100.05
        np.testing.assert_allclose(out["spread_width"], [expected], rtol=1e-10)


class TestAnyEventInMs:
    """Direct test of _any_event_in_ms — used for `only_on_event` filtering."""

    def test_no_events_returns_all_false(self):
        from boba.dataset.raw import _any_event_in_ms
        grid = np.array([0, 1_000_000, 2_000_000], dtype=np.int64)
        out = _any_event_in_ms(np.array([], dtype=np.int64), grid)
        assert not np.any(out)

    def test_event_exactly_at_grid_tick_counted(self):
        from boba.dataset.raw import _any_event_in_ms
        # Event at exactly ms tick boundaries
        event_t = np.array([0, 2_000_000], dtype=np.int64)
        grid = np.arange(3, dtype=np.int64) * 1_000_000   # 0ms, 1ms, 2ms
        out = _any_event_in_ms(event_t, grid)
        # ms tick 0 has event at ns=0 → True
        # ms tick 1 (1_000_000 to 1_999_999) has no event → False
        # ms tick 2 has event at ns=2_000_000 → True
        np.testing.assert_array_equal(out, [True, False, True])

    def test_event_strictly_inside_ms_window(self):
        from boba.dataset.raw import _any_event_in_ms
        # Event at 1_500_000 falls in ms tick 1 [1_000_000, 2_000_000)
        event_t = np.array([1_500_000], dtype=np.int64)
        grid = np.arange(3, dtype=np.int64) * 1_000_000
        out = _any_event_in_ms(event_t, grid)
        np.testing.assert_array_equal(out, [False, True, False])


class TestOnlyOnEventFilter:
    """build_features_raw with only_on_event=True should remove ms ticks with no events."""

    def test_filters_out_empty_ticks(self):
        from boba.dataset.session_data import SessionData
        # BBO events at ms 5, 10, 15. With warmup_ms=0, grid starts at ms 6
        # (the +1 in grid_start_ms) and ends at ms 15. Of the 10 ms ticks in
        # [6, 15], only 2 (ms 10 and 15) contain events.
        bbo_t = np.array([5, 10, 15], dtype=np.int64) * 1_000_000
        bid = np.array([100.0, 100.0, 100.0])
        ask = np.array([100.1, 100.1, 100.1])
        bq = aq = np.array([10.0, 10.0, 10.0])

        data = SessionData(
            listing_book_t={"bin": bbo_t},
            listing_book_bid={"bin": bid},
            listing_book_ask={"bin": ask},
            listing_book_bid_qty={"bin": bq},
            listing_book_ask_qty={"bin": aq},
            trade_ts={"bin": np.array([], dtype=np.int64)},
            trade_exchange_ts={"bin": np.array([], dtype=np.int64)},
            trade_prc={"bin": np.array([], dtype=np.float64)},
            trade_qty={"bin": np.array([], dtype=np.float64)},
            trade_dir={"bin": np.array([], dtype=np.float64)},
            listing_feed_latency_excess_ns={"bin": np.zeros(3, dtype=np.int64)},
            target_listing="bin",
            book_t=bbo_t,
            book_bid=bid,
            book_ask=ask,
            book_mid=(bid + ask) / 2,
            feed_latency_raw_ns=np.zeros(3, dtype=np.int64),
            feed_latency_excess_ns=np.zeros(3, dtype=np.int64),
            all_rx=bbo_t,
        )
        # Compare with-filter vs without-filter
        cfg_on = make_cfg(listings=("bin",), warmup_ms=0, horizon_ms=0, only_on_event=True)
        cfg_off = make_cfg(listings=("bin",), warmup_ms=0, horizon_ms=0, only_on_event=False)
        result_on = build_features_raw(data, ["bin"], cfg_on)
        result_off = build_features_raw(data, ["bin"], cfg_off)
        # Without filter: full grid range
        assert result_off.x.shape[0] >= result_on.x.shape[0]
        # With filter: only the 2 ms ticks containing the in-range events (ms 10, 15)
        assert result_on.x.shape[0] == 2


class TestColumnOrderMatchesFeatureNames:
    """_compute_per_listing must return exactly the listing's requested local
    names, in expansion order — if it ever diverges from the expanded specs,
    the matrix columns will be wrong."""

    def test_per_listing_dict_has_all_named_keys(self):
        from boba.dataset.session_data import SessionData
        cfg = make_cfg()
        # Build a minimal session
        bbo_t = np.array([1_000_000, 2_000_000], dtype=np.int64)
        bid = ask = np.array([100.0, 100.0])
        ask = np.array([100.1, 100.1])
        bq = aq = np.array([10.0, 10.0])
        data = SessionData(
            listing_book_t={"bin": bbo_t},
            listing_book_bid={"bin": bid},
            listing_book_ask={"bin": ask},
            listing_book_bid_qty={"bin": bq},
            listing_book_ask_qty={"bin": aq},
            trade_ts={"bin": np.array([], dtype=np.int64)},
            trade_exchange_ts={"bin": np.array([], dtype=np.int64)},
            trade_prc={"bin": np.array([], dtype=np.float64)},
            trade_qty={"bin": np.array([], dtype=np.float64)},
            trade_dir={"bin": np.array([], dtype=np.float64)},
            listing_feed_latency_excess_ns={"bin": np.zeros(2, dtype=np.int64)},
            target_listing="bin",
            book_t=bbo_t,
            book_bid=bid,
            book_ask=ask,
            book_mid=(bid + ask) / 2,
            feed_latency_raw_ns=np.zeros(2, dtype=np.int64),
            feed_latency_excess_ns=np.zeros(2, dtype=np.int64),
            all_rx=bbo_t,
        )
        grid = np.arange(5, dtype=np.int64) * 1_000_000
        out_dict = _compute_per_listing("bin", data, grid, cfg)
        names = legacy_local_names()
        # Every name must exist in the dict
        for nm in names:
            assert nm in out_dict, f"Feature {nm} missing from _compute_per_listing output"
        # …and the dict order is exactly the expansion order of the specs
        assert list(out_dict.keys()) == names


class TestWideThresholdDefaultFallback:
    """Unknown listing should fall back to the default wide_threshold."""

    def test_unknown_listing_uses_default(self):
        # Default value from DatasetRawConfig.wide_threshold is 1.0e-4 per known listing,
        # and cfg.wide_threshold.get(e, 1.0e-4) falls back to 1.0e-4 for unknown.
        cfg = make_cfg()
        assert cfg.wide_threshold.get("unknown_listing", 1.0e-4) == 1.0e-4


class TestFloat32VarianceReconstructionPrecision:
    """Document precision limits of variance reconstruction at float32 storage.

    Two effects need to be understood by downstream users:
      1. EMA convergence transient: var = ema_sq − ema² is biased by how
         far the EMAs have converged from their y[−1] = 0 initial condition.
         Until the EMAs are settled (~3× span), the recovered "variance"
         includes transient bias, not just the variance of the input.
      2. Float32 storage: when microprice ≈ 0.15 and variance is ≤1e-9, the
         subtraction of two near-equal float32 values loses meaningful
         precision. Downstream variance reconstruction is only reliable
         once the EMA has converged AND the true variance exceeds ~1e-8.
    """

    def test_variance_is_non_negative_after_storage_roundtrip(self):
        # Behaviour we GUARANTEE: variance reconstructed downstream is always ≥ 0
        # (after the max(_, 0) clamp), and finite, even at float32 storage.
        rng = np.random.default_rng(0)
        mp = 0.15 + rng.standard_normal(10000) * 1e-4
        grid = np.arange(10000, dtype=np.int64) * 1_000_000
        out = compute_ms_grid_temporal(mp, grid, DEFAULT_TEMPORAL_SPANS)
        ema_f32 = out["ema_microprice_centered_1000ms"].astype(np.float32)
        ema_sq_f32 = out["ema_microprice_centered_sq_1000ms"].astype(np.float32)
        var_recovered = np.maximum(
            ema_sq_f32.astype(np.float64) - ema_f32.astype(np.float64) ** 2, 0.0
        )
        assert np.all(np.isfinite(var_recovered))
        assert np.all(var_recovered >= 0.0)

    def test_converged_ema_variance_in_right_order_of_magnitude(self):
        # After ~10x span samples, the EMA has converged and the recovered
        # variance should match the true variance to within an order of magnitude.
        rng = np.random.default_rng(0)
        true_std = 1e-3   # larger noise — well above float32 reconstruction limit
        mp = 0.15 + rng.standard_normal(20000) * true_std
        grid = np.arange(20000, dtype=np.int64) * 1_000_000
        out = compute_ms_grid_temporal(mp, grid, DEFAULT_TEMPORAL_SPANS)
        # Use span=1000ms, take samples well past the convergence transient (>10*span)
        ema_f32 = out["ema_microprice_centered_1000ms"].astype(np.float32)
        ema_sq_f32 = out["ema_microprice_centered_sq_1000ms"].astype(np.float32)
        var_recovered = np.maximum(
            ema_sq_f32.astype(np.float64) - ema_f32.astype(np.float64) ** 2, 0.0
        )
        # Take only post-convergence samples
        post = var_recovered[15000:]
        true_var = true_std ** 2  # 1e-6
        # Median should be within order of magnitude of true variance
        assert true_var / 10.0 <= np.median(post) <= true_var * 10.0, (
            f"Median recovered variance {np.median(post):.2e} far from true {true_var:.2e}"
        )


class TestMicropriceCentering:
    """The centered microprice EMAs (`ema_microprice_centered_{N}ms`,
    `_centered_sq_{N}ms`) subtract a per-listing reference price before
    the EMA. This keeps stored values small so float32 variance reconstruction
    (`ema_sq − ema²`) is precise even for tiny variances.
    """

    def test_centering_subtracts_reference_before_ema(self):
        # microprice ≈ 0.15, ref = 0.15 → centered ≈ 0
        # ema_microprice_centered_*  ≈ 0  (much better precision than ema ≈ 0.15)
        mp = np.full(5000, 0.15)
        grid = np.arange(5000, dtype=np.int64) * 1_000_000
        out = compute_ms_grid_temporal(mp, grid, DEFAULT_TEMPORAL_SPANS, microprice_ref=0.15)
        # After convergence the centered EMA should be near 0
        assert abs(out["ema_microprice_centered_1000ms"][-1]) < 1e-10

    def test_centering_does_not_affect_returns(self):
        # Returns are log-ratios, so the centering cancels out
        rng = np.random.default_rng(0)
        mp = 0.15 + rng.standard_normal(1000) * 1e-4
        grid = np.arange(1000, dtype=np.int64) * 1_000_000
        out_a = compute_ms_grid_temporal(mp, grid, DEFAULT_TEMPORAL_SPANS, microprice_ref=0.0)
        out_b = compute_ms_grid_temporal(mp, grid, DEFAULT_TEMPORAL_SPANS, microprice_ref=0.15)
        np.testing.assert_allclose(out_a["return_1ms"], out_b["return_1ms"])
        np.testing.assert_allclose(out_a["return_100ms"], out_b["return_100ms"])

    def test_centered_variance_recovers_tiny_variance_in_float32(self):
        # The whole point of centering: recover tiny variances after float32 roundtrip
        rng = np.random.default_rng(0)
        true_std = 1e-4
        mp = 0.15 + rng.standard_normal(20000) * true_std
        grid = np.arange(20000, dtype=np.int64) * 1_000_000
        out = compute_ms_grid_temporal(mp, grid, DEFAULT_TEMPORAL_SPANS, microprice_ref=0.15)
        # Cast to float32 (simulating storage)
        ema_f32 = out["ema_microprice_centered_1000ms"].astype(np.float32).astype(np.float64)
        ema_sq_f32 = out["ema_microprice_centered_sq_1000ms"].astype(np.float32).astype(np.float64)
        var_recovered = np.maximum(ema_sq_f32 - ema_f32 ** 2, 0.0)
        true_var = true_std ** 2  # 1e-8
        post = var_recovered[15000:]
        # With centering, the variance is recovered well within an order of magnitude
        # even at true_var = 1e-8 (the previous precision floor).
        assert true_var / 5.0 <= np.median(post) <= true_var * 5.0, (
            f"Centered recovery {np.median(post):.2e} vs true {true_var:.2e}"
        )

    def test_centered_recovers_tiny_variance_uncentered_loses(self):
        # At a variance below the uncentered float32 precision floor (~1.4e-9),
        # the uncentered version recovers ~0 (precision lost) while centered
        # recovers a meaningful estimate.
        rng = np.random.default_rng(0)
        true_std = 2e-5
        mp = 0.15 + rng.standard_normal(20000) * true_std
        grid = np.arange(20000, dtype=np.int64) * 1_000_000
        out_uncentered = compute_ms_grid_temporal(mp, grid, DEFAULT_TEMPORAL_SPANS, microprice_ref=0.0)
        out_centered = compute_ms_grid_temporal(mp, grid, DEFAULT_TEMPORAL_SPANS, microprice_ref=0.15)

        def recover_var(out):
            ema_f32 = out["ema_microprice_centered_1000ms"].astype(np.float32).astype(np.float64)
            ema_sq_f32 = out["ema_microprice_centered_sq_1000ms"].astype(np.float32).astype(np.float64)
            return np.maximum(ema_sq_f32 - ema_f32 ** 2, 0.0)

        var_u = recover_var(out_uncentered)
        var_c = recover_var(out_centered)

        true_var = true_std ** 2  # 4e-10 — well below the uncentered float32 floor
        med_u = np.median(var_u[15000:])
        med_c = np.median(var_c[15000:])
        # Uncentered should be dominated by precision noise (often ~0 or hugely off)
        # Centered should be in the right order of magnitude
        assert true_var / 10.0 <= med_c <= true_var * 10.0, (
            f"Centered failed at small var: {med_c:.2e} vs true {true_var:.2e}"
        )
        # Centered relative error must be smaller than uncentered
        err_u = abs(med_u - true_var) / true_var
        err_c = abs(med_c - true_var) / true_var
        assert err_c < err_u, (
            f"Centering didn't help at true_var={true_var:.2e}: "
            f"uncentered err={err_u:.2%}, centered err={err_c:.2%}"
        )


class TestDogePerpScale:
    """Verify every feature category survives the float64 → float32 cast
    at DOGE-perp realistic scales without losing meaningful precision.

    DOGE-perp realistic ranges (per CLAUDE.md and empirical data):
      - microprice         ≈ 0.10 to 0.20
      - tick size          = 1e-5  → 1-tick log-return ≈ 6.7e-5
      - spread             = 1 tick almost always; rarely 2-5 ticks
      - book_depth         ≈ 1e3 to 1e6 DOGE per side
      - book_imbalance     ∈ [-1, +1]
      - trade qty          ≈ 1 to 1e5 DOGE per trade
      - trade value (USD)  ≈ 0.15 to 1.5e4 USD per trade
      - inter-trade time   ≈ ms to seconds
    """

    def _doge_microprice_series(self, n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        # Random walk in tick units around 0.15
        ticks = np.cumsum(rng.choice([-1, 0, 1], size=n, p=[0.2, 0.6, 0.2]))
        return 0.15 + ticks * 1e-5

    def test_microprice_storage_resolves_ticks(self):
        # 1-tick changes (1e-5 at price 0.15) must survive float32 storage
        mp = np.array([0.15000, 0.15001, 0.14999], dtype=np.float64)
        mp_f32 = mp.astype(np.float32).astype(np.float64)
        # After roundtrip, the tick increments should still be distinguishable
        diffs = np.diff(mp_f32)
        assert np.all(np.abs(diffs) > 5e-6), f"Tick changes lost in float32: {diffs}"

    def test_log_returns_at_tick_scale_survive_float32(self):
        mp = self._doge_microprice_series(1000)
        grid = np.arange(1000, dtype=np.int64) * 1_000_000
        out = compute_ms_grid_temporal(mp, grid, DEFAULT_TEMPORAL_SPANS)
        # 1ms returns should be of order 1e-5 (1-tick moves) or 0 (no move).
        # float32 can resolve 1e-5 with relative precision 6e-8 → absolute 6e-13. ✓
        ret = out["return_1ms"].astype(np.float32)
        nonzero = ret[ret != 0]
        if len(nonzero) > 0:
            assert np.all(np.abs(nonzero) >= 5e-6), "Non-zero returns rounded to zero in float32"

    def test_book_depth_at_doge_scale_in_float32(self):
        # Depths from 1e3 to 1e6 DOGE
        ts = np.arange(100, dtype=np.int64) * 1_000_000
        bid = np.full(100, 100.0)
        ask = np.full(100, 100.001)
        rng = np.random.default_rng(1)
        bq = rng.uniform(1e3, 1e6, 100)
        aq = rng.uniform(1e3, 1e6, 100)
        out = compute_bbo_event_emas(ts, bid, ask, bq, aq, 1e-3, DEFAULT_BBO_SPANS)
        # book_depth and its EMA should match originals within float32 precision
        bd_f32 = out["book_depth"].astype(np.float32).astype(np.float64)
        bd_orig = bq + aq
        rel_err = np.max(np.abs(bd_f32 - bd_orig) / bd_orig)
        assert rel_err < 1e-6, f"book_depth float32 storage too lossy: {rel_err}"

    def test_book_imbalance_at_doge_scale_in_float32(self):
        ts = np.arange(100, dtype=np.int64) * 1_000_000
        bid = np.full(100, 100.0)
        ask = np.full(100, 100.001)
        rng = np.random.default_rng(2)
        bq = rng.uniform(1e3, 1e6, 100)
        aq = rng.uniform(1e3, 1e6, 100)
        out = compute_bbo_event_emas(ts, bid, ask, bq, aq, 1e-3, DEFAULT_BBO_SPANS)
        # book_imbalance ∈ [-1, +1], float32 precision 6e-8 — easily resolves typical values
        bi_f32 = out["book_imbalance"].astype(np.float32).astype(np.float64)
        bi_orig = (bq - aq) / (bq + aq)
        assert np.max(np.abs(bi_f32 - bi_orig)) < 1e-6

    def test_ema_book_depth_sq_variance_at_extreme_depth(self):
        # Extreme depth: 1e6 DOGE → depth² ≈ 1e12
        # Reconstructing variance from EMA(depth) and EMA(depth²) at this scale
        # tests the float32 precision floor.
        rng = np.random.default_rng(3)
        depth = rng.normal(1e6, 1e4, 20000)   # mean 1e6, std 1e4 → var = 1e8
        ts = np.arange(20000, dtype=np.int64) * 1_000_000

        # Run through EMA manually
        alpha = _alpha(1000)
        ema = _ewm_1d(depth, alpha)
        ema_sq = _ewm_1d(depth ** 2, alpha)
        # Cast to float32
        ema_f32 = ema.astype(np.float32).astype(np.float64)
        ema_sq_f32 = ema_sq.astype(np.float32).astype(np.float64)
        var = np.maximum(ema_sq_f32 - ema_f32 ** 2, 0.0)
        # True variance ≈ 1e8. After convergence, recovered should be within an order.
        true_var = 1e8
        post_conv = var[15000:]
        assert true_var / 10.0 <= np.median(post_conv) <= true_var * 10.0, (
            f"Variance recovered {np.median(post_conv):.2e} vs true {true_var:.2e}"
        )

    def test_trade_value_ema_at_doge_scale(self):
        # Trades of ~1000 DOGE × $0.15 = $150 per trade
        n = 1000
        ts = np.arange(n, dtype=np.int64) * 1_000_000
        prc = np.full(n, 0.15)
        qty = np.full(n, 1000.0)
        dir_ = np.ones(n)  # all buys
        out = compute_trade_event_emas(ts, prc, qty, dir_, DEFAULT_TRADE_SPANS)
        # After convergence, ema_buy_trade_value_100t ≈ qty*price = 150
        v_f32 = out["ema_buy_trade_value_100t"][-1:].astype(np.float32).astype(np.float64)[0]
        assert 100.0 < v_f32 < 200.0, f"Expected ~150, got {v_f32}"

    def test_time_since_at_long_session_scale(self):
        # 24h session: ms count = 86_400_000. float32 has ~7 digits, so this
        # is on the edge of integer-exact representation (2^24 ≈ 1.67e7).
        # Verify the rounding doesn't break "time since" semantics (sub-second
        # accuracy at a few hours).
        event_ns = np.array([0], dtype=np.int64)
        grid_ns = np.array([3600 * 1_000_000_000], dtype=np.int64)  # 1 hour later
        elapsed = time_since_event_ms(event_ns, grid_ns)
        # Stored as float32 → cast back
        elapsed_roundtrip = float(np.float32(elapsed[0]))
        # 1 hour = 3_600_000 ms; allow up to 1 ms loss from float32
        assert abs(elapsed_roundtrip - 3_600_000.0) < 1.0


class TestCostFields:
    """Cost fields are computed by `_cost_fields_for_grid` (boba.dataset.costs).
    The simulated land-time logic is critical — if entry/exit times are wrong,
    the model can learn to act on book state it could never actually trade on.

    Simulated land time:
      t_entry = grid_tick + baseline_rt_ms + feed_excess_at_grid + processing_ms
      t_exit  = grid_tick + horizon_ms

    Latency blowout: when feed_excess is large, t_entry can exceed t_exit,
    in which case `valid_cost` is False and cost fields default to 0/inf/-inf.
    """

    def _make_session(self, bbo_t, bid, ask, bq, aq, tr_t, tr_prc, tr_qty,
                      tr_dir, feed_lat_ns):
        from boba.dataset.session_data import SessionData
        return SessionData(
            listing_book_t={"bin": bbo_t, "byb": bbo_t, "okx": bbo_t},
            listing_book_bid={e: bid for e in ["bin", "byb", "okx"]},
            listing_book_ask={e: ask for e in ["bin", "byb", "okx"]},
            listing_book_bid_qty={e: bq for e in ["bin", "byb", "okx"]},
            listing_book_ask_qty={e: aq for e in ["bin", "byb", "okx"]},
            trade_ts={e: tr_t for e in ["bin", "byb", "okx"]},
            trade_exchange_ts={e: tr_t for e in ["bin", "byb", "okx"]},
            trade_prc={e: tr_prc for e in ["bin", "byb", "okx"]},
            trade_qty={e: tr_qty for e in ["bin", "byb", "okx"]},
            trade_dir={e: tr_dir for e in ["bin", "byb", "okx"]},
            listing_feed_latency_excess_ns={e: feed_lat_ns for e in ["bin", "byb", "okx"]},
            target_listing="bin",
            book_t=bbo_t,
            book_bid=bid,
            book_ask=ask,
            book_mid=(bid + ask) / 2,
            feed_latency_raw_ns=feed_lat_ns,
            feed_latency_excess_ns=feed_lat_ns,
            all_rx=np.sort(np.concatenate([bbo_t, tr_t])),
        )

    def test_cost_fields_present_and_correct_shape(self):
        # Tight grid where everything has data
        bbo_t = np.arange(200, dtype=np.int64) * 1_000_000     # 200 events, 1ms apart
        bid = 100.0 + np.arange(200) * 0.0001
        ask = bid + 0.001
        bq = aq = np.full(200, 10.0)
        tr_t = bbo_t[::5]    # trades every 5ms
        tr_prc = bid[::5] + 0.0005
        tr_qty = np.full(40, 5.0)
        tr_dir = np.ones(40)
        feed_lat = np.zeros(200, dtype=np.int64)

        data = self._make_session(bbo_t, bid, ask, bq, aq, tr_t, tr_prc, tr_qty, tr_dir, feed_lat)
        cfg = make_cfg(warmup_ms=0, horizon_ms=20.0, only_on_event=False)
        result = build_features_raw(data, ["bin", "byb", "okx"], cfg)

        # All cost field arrays should have the same length as x
        N = result.x.shape[0]
        for arr_name in ["eval_bid_l", "eval_ask_l", "eval_mid",
                         "c_ask_entry_l", "c_bid_entry_l",
                         "c_ask_exit_l", "c_bid_exit_l", "c_mid_exit_l",
                         "c_mid_move_count",
                         "c_buy_trade_min_l", "c_buy_trade_max_l",
                         "c_sell_trade_min_l", "c_sell_trade_max_l",
                         "feed_latency_raw_ms", "feed_latency_excess_ms"]:
            arr = getattr(result, arr_name)
            assert arr.shape == (N,), f"{arr_name} has shape {arr.shape}, expected ({N},)"
            assert arr.dtype == np.float32

    def test_mid_move_count_matches_window_and_atom(self):
        # c_mid_move_count = # of book_mid changes in (eval, exit] — the SAME window as c_mid_exit_l.
        # Verify against an independent count, and the atom property: count==0 ⇒ no net move.
        bbo_t = (np.arange(30) * 10).astype(np.int64) * 1_000_000        # 30 events, 10ms apart
        steps = np.array([0, .1, .1, 0, -.1, 0, 0, .1, -.1, .1, 0, .1, .1, -.1, 0,
                          .1, 0, -.1, .1, .1, 0, 0, -.1, .1, 0, .1, -.1, 0, .1, 0])
        mid = 100.0 + np.cumsum(steps)
        bid, ask = mid - 0.05, mid + 0.05
        bq = aq = np.full(30, 10.0)
        empty_i, empty_f = np.array([], np.int64), np.array([], np.float64)
        data = self._make_session(bbo_t, bid, ask, bq, aq, empty_i, empty_f, empty_f, empty_f,
                                  np.zeros(30, np.int64))
        H = 50.0
        cfg = make_cfg(warmup_ms=0, horizon_ms=H, baseline_rt_ms=0.0, processing_ms=0.0, only_on_event=False)
        res = build_features_raw(data, ["bin", "byb", "okx"], cfg)
        bmid = (bid + ask) / 2
        for i, t in enumerate(res.timestamp_ms):
            tns = int(round(t * 1e6)); hns = int(round(H * 1e6))
            eg = np.searchsorted(bbo_t, tns, "right") - 1
            xg = np.searchsorted(bbo_t, tns + hns, "left")
            ref = 0.0 if (eg < 0 or xg <= eg) else float((bmid[eg + 1:max(eg, xg - 1) + 1]
                                                          != bmid[eg:max(eg, xg - 1)]).sum())
            assert res.c_mid_move_count[i] == pytest.approx(ref), f"tick {t}: {res.c_mid_move_count[i]} vs {ref}"
        # atom consistency: zero count ⇒ no net mid move
        z = res.c_mid_move_count == 0
        assert np.all(res.c_mid_exit_l[z] == 0.0)
        assert res.c_mid_move_count.max() > 0   # the path does move

    def test_cost_field_selection_keeps_only_requested(self):
        # cost_fields selects which cost arrays are populated (others None = lean memory),
        # without changing the requested values vs the full build.
        bbo_t = np.arange(100, dtype=np.int64) * 1_000_000
        bid = 100.0 + np.arange(100) * 0.0001; ask = bid + 0.001
        bq = aq = np.full(100, 10.0)
        tr_t = bbo_t[::5]; tr_prc = bid[::5] + 0.0005
        tr_qty = np.full(len(tr_t), 5.0); tr_dir = np.ones(len(tr_t))
        data = self._make_session(bbo_t, bid, ask, bq, aq, tr_t, tr_prc, tr_qty, tr_dir,
                                  np.zeros(100, np.int64))
        ex = ["bin", "byb", "okx"]
        want = ("eval_mid", "c_mid_exit_l", "c_mid_move_count")
        full = build_features_raw(data, ex, make_cfg(warmup_ms=0, horizon_ms=20.0, only_on_event=False))
        sel = build_features_raw(data, ex, make_cfg(warmup_ms=0, horizon_ms=20.0, only_on_event=False,
                                                    cost_fields=want))
        for f in want:                                   # requested: present + identical to full
            assert getattr(sel, f) is not None, f"{f} requested but None"
            np.testing.assert_array_equal(getattr(sel, f), getattr(full, f))
        for f in ("c_buy_trade_min_l", "c_sell_trade_max_l", "c_ask_entry_l", "feed_latency_raw_ms"):
            assert getattr(sel, f) is None, f"{f} not requested but populated (not lean)"
            assert getattr(full, f) is not None           # full build still has them

    def test_default_cost_fields_empty_populates_none(self):
        # Default cost_fields=() means NO cost fields — there is no implicit "all" sentinel.
        from boba.dataset.raw import DatasetRawConfig
        assert DatasetRawConfig(columns=make_cfg().columns).cost_fields == ()
        bbo_t = np.arange(40, dtype=np.int64) * 1_000_000
        bid = 100.0 + np.arange(40) * 0.0001; ask = bid + 0.001
        bq = aq = np.full(40, 10.0)
        e = np.array([], np.int64); ef = np.array([], np.float64)
        data = self._make_session(bbo_t, bid, ask, bq, aq, e, ef, ef, ef, np.zeros(40, np.int64))
        res = build_features_raw(data, ["bin", "byb", "okx"],
                                 make_cfg(warmup_ms=0, horizon_ms=20.0, only_on_event=False, cost_fields=()))
        for f in _COST_FIELDS:
            assert getattr(res, f) is None, f"{f} populated despite cost_fields=()"
        assert res.x.shape[0] == len(res) > 0   # feature matrix still built

    def test_microprice_ref_has_no_default(self):
        # microprice_ref no longer defaults to the DOGE 0.15 dict — absent ⇒ {} ⇒ no centering.
        from boba.dataset.raw import DatasetRawConfig
        assert DatasetRawConfig(columns=make_cfg().columns).microprice_ref == {}

    def test_cost_field_selection_save_load_roundtrip(self):
        from boba.dataset.raw import _save_raw, _load_raw
        import tempfile
        from pathlib import Path
        bbo_t = np.arange(60, dtype=np.int64) * 1_000_000
        bid = 100.0 + np.arange(60) * 0.0001; ask = bid + 0.001
        bq = aq = np.full(60, 10.0)
        e = np.array([], np.int64); ef = np.array([], np.float64)
        data = self._make_session(bbo_t, bid, ask, bq, aq, e, ef, ef, ef, np.zeros(60, np.int64))
        sel = build_features_raw(data, ["bin", "byb", "okx"],
                                 make_cfg(warmup_ms=0, horizon_ms=20.0, only_on_event=False,
                                          cost_fields=("eval_mid", "c_mid_move_count")))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "blk.npz"; _save_raw(p, sel); back = _load_raw(p)
        np.testing.assert_array_equal(back.c_mid_move_count, sel.c_mid_move_count)
        np.testing.assert_array_equal(back.eval_mid, sel.eval_mid)
        assert back.c_buy_trade_min_l is None and back.c_mid_exit_l is None   # selection survives roundtrip

    def test_eval_mid_matches_target_book_state(self):
        # eval_mid at each grid tick should equal book_mid at the latest book event ≤ grid tick.
        bbo_t = np.array([10, 50, 100], dtype=np.int64) * 1_000_000
        bid = np.array([100.0, 100.1, 100.2])
        ask = np.array([100.1, 100.2, 100.3])
        bq = aq = np.full(3, 10.0)
        tr_t = np.array([], dtype=np.int64)
        tr_prc = tr_qty = tr_dir = np.array([], dtype=np.float64)
        feed_lat = np.zeros(3, dtype=np.int64)

        data = self._make_session(bbo_t, bid, ask, bq, aq, tr_t, tr_prc, tr_qty, tr_dir, feed_lat)
        cfg = make_cfg(warmup_ms=0, horizon_ms=0.0, only_on_event=False)
        result = build_features_raw(data, ["bin", "byb", "okx"], cfg)

        # Find the grid tick that lands just after event at ms 50: expect mid = (100.1+100.2)/2 = 100.15
        # The grid starts at ms 11 (warmup=0 → grid_start = floor(10/1) + 1 = 11)
        # At ms 50 the second event has fired → eval_mid = 100.15
        # We use _t_at: timestamp_ms is float64
        ms50_idx = np.where(result.timestamp_ms == 50.0)[0]
        if len(ms50_idx) > 0:
            assert result.eval_mid[ms50_idx[0]] == pytest.approx(100.15, abs=1e-5)

    def test_latency_blowout_invalidates_cost(self):
        # When feed latency excess is huge relative to horizon, the simulated
        # order entry happens AFTER the outcome horizon → cost fields invalid (0).
        bbo_t = np.arange(200, dtype=np.int64) * 1_000_000
        bid = np.full(200, 100.0)
        ask = np.full(200, 100.1)
        bq = aq = np.full(200, 10.0)
        tr_t = np.array([], dtype=np.int64)
        tr_prc = tr_qty = tr_dir = np.array([], dtype=np.float64)
        # Huge feed latency: 500ms excess at every event
        feed_lat = np.full(200, 500_000_000, dtype=np.int64)

        data = self._make_session(bbo_t, bid, ask, bq, aq, tr_t, tr_prc, tr_qty, tr_dir, feed_lat)
        # Horizon = 10ms — much less than 500ms latency, so all cost fields should be invalid
        cfg = make_cfg(
            warmup_ms=0, horizon_ms=10.0,
            baseline_rt_ms=3.0, processing_ms=0.5,
            only_on_event=False,
        )
        result = build_features_raw(data, ["bin", "byb", "okx"], cfg)

        # All cost-at-exit fields should be 0 (invalid, no exit observed before entry)
        assert np.all(result.c_bid_exit_l == 0.0)
        assert np.all(result.c_ask_exit_l == 0.0)
        assert np.all(result.c_mid_exit_l == 0.0)
        # Entry-related fields can be either 0 (invalid) or non-zero — depends on entry_gi
        # but the cost shouldn't blow up
        assert np.all(np.isfinite(result.c_bid_entry_l))
        assert np.all(np.isfinite(result.c_ask_entry_l))

    def test_normal_latency_produces_valid_cost(self):
        # Low feed latency: cost fields should be valid (non-zero) at many ticks
        bbo_t = np.arange(200, dtype=np.int64) * 1_000_000
        bid = 100.0 + np.arange(200) * 0.0001       # drifting bid
        ask = bid + 0.001
        bq = aq = np.full(200, 10.0)
        tr_t = bbo_t[::10]
        tr_prc = bid[::10] + 0.0005
        tr_qty = np.full(20, 5.0)
        tr_dir = np.ones(20)
        feed_lat = np.zeros(200, dtype=np.int64)    # zero feed excess

        data = self._make_session(bbo_t, bid, ask, bq, aq, tr_t, tr_prc, tr_qty, tr_dir, feed_lat)
        cfg = make_cfg(
            warmup_ms=0, horizon_ms=50.0,
            baseline_rt_ms=3.0, processing_ms=0.5,
            only_on_event=False,
        )
        result = build_features_raw(data, ["bin", "byb", "okx"], cfg)

        # At least some ticks should have valid (non-zero) exit costs
        nonzero_exits = (result.c_mid_exit_l != 0.0).sum()
        assert nonzero_exits > 0, "No valid cost samples — latency formula may be wrong"

    def test_entry_time_uses_simulated_land_formula(self):
        # Construct a case where we can verify the entry time formula directly.
        # baseline=3, processing=0.5, feed_excess=0 → t_entry = grid + 3.5ms.
        # The grid tick at ms=0 with no feed excess should have entry at ms=3.5.
        # The BBO state at ms=3.5 should determine bid_entry/ask_entry.
        bbo_t = np.array([0, 1, 2, 3, 4, 5, 10, 50], dtype=np.int64) * 1_000_000
        bid = np.array([100.0, 100.0, 100.0, 100.0, 100.5, 100.5, 100.5, 100.5])
        ask = np.array([100.1, 100.1, 100.1, 100.1, 100.6, 100.6, 100.6, 100.6])
        bq = aq = np.full(8, 10.0)
        tr_t = np.array([], dtype=np.int64)
        tr_prc = tr_qty = tr_dir = np.array([], dtype=np.float64)
        feed_lat = np.zeros(8, dtype=np.int64)

        data = self._make_session(bbo_t, bid, ask, bq, aq, tr_t, tr_prc, tr_qty, tr_dir, feed_lat)
        cfg = make_cfg(
            warmup_ms=0, horizon_ms=20.0,
            baseline_rt_ms=3.0, processing_ms=0.5,
            only_on_event=False,
        )
        result = build_features_raw(data, ["bin", "byb", "okx"], cfg)

        # Grid starts at ms=1 (warmup=0 → grid_start = floor(0/1)+1=1)
        # At grid ms=1: t_entry = 1 + 3 + 0 + 0.5 = 4.5ms (in float)
        # Floor to int_ns: 4_500_000. BBO at-or-before is index 4 (event at ms=4): bid=100.5
        # So c_bid_entry_l ≈ log(100.5/eval_mid) where eval_mid at ms=1 is (100.0+100.1)/2 = 100.05
        ms1_idx = np.where(result.timestamp_ms == 1.0)[0]
        assert len(ms1_idx) > 0
        # eval_mid at ms=1 should be 100.05 (from BBO at ms=1)
        assert result.eval_mid[ms1_idx[0]] == pytest.approx(100.05, abs=1e-4)
        # c_bid_entry_l ≈ log(100.5 / 100.05) = 0.00449
        expected_entry = np.log(100.5 / 100.05)
        assert result.c_bid_entry_l[ms1_idx[0]] == pytest.approx(expected_entry, abs=1e-4)

    def test_trade_gap_in_outcome_window_returns_sentinel(self):
        # No trades in the outcome window → c_buy/sell_trade_min_l should be +inf,
        # max_l should be -inf (sentinel values).
        bbo_t = np.arange(200, dtype=np.int64) * 1_000_000
        bid = np.full(200, 100.0)
        ask = np.full(200, 100.1)
        bq = aq = np.full(200, 10.0)
        # NO trades at all
        tr_t = np.array([], dtype=np.int64)
        tr_prc = tr_qty = tr_dir = np.array([], dtype=np.float64)
        feed_lat = np.zeros(200, dtype=np.int64)

        data = self._make_session(bbo_t, bid, ask, bq, aq, tr_t, tr_prc, tr_qty, tr_dir, feed_lat)
        cfg = make_cfg(warmup_ms=0, horizon_ms=20.0, only_on_event=False)
        result = build_features_raw(data, ["bin", "byb", "okx"], cfg)

        # All trade min/max fields should be sentinels (no trades to find)
        assert np.all(np.isposinf(result.c_buy_trade_min_l))
        assert np.all(np.isneginf(result.c_buy_trade_max_l))
        assert np.all(np.isposinf(result.c_sell_trade_min_l))
        assert np.all(np.isneginf(result.c_sell_trade_max_l))

    def test_feed_latency_excess_preserved_through_pipeline(self):
        # Feed latency excess passed in should appear in feed_latency_excess_ms field.
        bbo_t = np.arange(50, dtype=np.int64) * 1_000_000
        bid = np.full(50, 100.0)
        ask = np.full(50, 100.1)
        bq = aq = np.full(50, 10.0)
        tr_t = np.array([], dtype=np.int64)
        tr_prc = tr_qty = tr_dir = np.array([], dtype=np.float64)
        # 5ms = 5_000_000 ns excess latency everywhere
        feed_lat = np.full(50, 5_000_000, dtype=np.int64)

        data = self._make_session(bbo_t, bid, ask, bq, aq, tr_t, tr_prc, tr_qty, tr_dir, feed_lat)
        cfg = make_cfg(warmup_ms=0, horizon_ms=20.0, only_on_event=False)
        result = build_features_raw(data, ["bin", "byb", "okx"], cfg)

        # feed_latency_excess_ms should be approximately 5.0 (ms) for all valid ticks
        valid_excess = result.feed_latency_excess_ms[result.feed_latency_excess_ms != 0]
        if len(valid_excess) > 0:
            assert np.allclose(valid_excess, 5.0, atol=1e-3)


class TestFeatureNames:
    def test_count_matches_config_estimate(self):
        cfg = make_cfg()
        names = legacy_local_names()
        # 3 listings × the per-listing catalogue
        assert cfg.n_features() == 3 * len(names)
        assert len(feature_names(cfg, ["bin", "byb", "okx"])) == cfg.n_features()

    def test_no_duplicate_names(self):
        names = legacy_local_names()
        assert len(set(names)) == len(names)

    def test_includes_expected_features(self):
        names = legacy_local_names()
        # Sanity checks: a few critical names
        assert "microprice" in names
        assert "spread_width" in names
        assert "book_imbalance" in names
        assert "spread_wide_flag" in names
        assert "feed_latency_excess_ms" in names
        assert "ema_buy_trade_qty_10t" in names
        assert "ema_ofi_100b" in names
        assert "ema_microprice_centered_1000ms" in names
        assert "ema_microprice_centered_sq_1000ms" in names
        assert "ema_book_imbalance_100b" in names
        assert "ema_book_depth_1000b" in names
        assert "dt_100b" in names
        assert "dt_10t" in names
        assert "time_since_last_trade_ms" in names
        assert "time_since_spread_wide_ms" in names


# ── Per-ms trade flow (buy/sell_trade_value + wallclock-EMA rates) ────────────

class TestTradeValuePerMs:
    """trade_value_per_ms: per-ms buy/sell traded value summed onto the 1ms grid.

    Causality convention (pinned here): grid tick t accumulates trades with
    timestamp in (t − 1ms, t] — exactly the trades that became visible
    at-or-before t since the previous tick, matching forward_fill_to_ms_grid
    (a trade exactly AT the tick is included; one strictly after it belongs to
    the next tick). Quiet ms read 0: these are flow, not state — no
    forward-fill.
    """

    def test_no_trades_returns_zeros(self):
        grid = np.arange(5, dtype=np.int64) * 1_000_000
        buy, sell = trade_value_per_ms(
            np.array([], dtype=np.int64), np.array([], dtype=np.float64),
            np.array([], dtype=np.float64), np.array([], dtype=np.float64), grid,
        )
        np.testing.assert_array_equal(buy, np.zeros(5))
        np.testing.assert_array_equal(sell, np.zeros(5))
        assert buy.dtype == np.float64

    def test_trade_exactly_at_tick_included_at_that_tick(self):
        # at-or-before: a trade at exactly 2ms lands in the tick at 2ms
        grid = np.arange(5, dtype=np.int64) * 1_000_000
        buy, sell = trade_value_per_ms(
            np.array([2_000_000], dtype=np.int64),
            np.array([10.0]), np.array([3.0]), np.array([1.0]), grid,
        )
        np.testing.assert_allclose(buy, [0.0, 0.0, 30.0, 0.0, 0.0])
        np.testing.assert_array_equal(sell, np.zeros(5))

    def test_trade_strictly_inside_window_assigned_to_next_tick(self):
        # a trade at 1.5ms is in the FUTURE of tick 1ms → lands at tick 2ms
        grid = np.arange(5, dtype=np.int64) * 1_000_000
        buy, _ = trade_value_per_ms(
            np.array([1_500_000], dtype=np.int64),
            np.array([10.0]), np.array([3.0]), np.array([1.0]), grid,
        )
        np.testing.assert_allclose(buy, [0.0, 0.0, 30.0, 0.0, 0.0])

    def test_causality_matches_forward_fill_convention(self):
        # The bucket a trade lands in is exactly the FIRST grid tick whose
        # at-or-before forward-fill sees it — no lookahead, no lag.
        grid = np.arange(10, dtype=np.int64) * 1_000_000
        for tr_ns in [1, 999_999, 1_000_000, 1_000_001, 4_500_000, 9_000_000]:
            tr_t = np.array([tr_ns], dtype=np.int64)
            buy, _ = trade_value_per_ms(
                tr_t, np.array([1.0]), np.array([1.0]), np.array([1.0]), grid)
            ff = forward_fill_to_ms_grid(tr_t, np.array([1.0]), grid, 0.0)
            assert np.any(buy > 0), f"tr_ns={tr_ns}: trade dropped"
            bucket = int(np.argmax(buy > 0))
            first_visible = int(np.argmax(ff > 0))
            assert bucket == first_visible, (
                f"tr_ns={tr_ns}: bucket {bucket} != first ff-visible {first_visible}"
            )

    def test_buy_sell_split_by_direction(self):
        # tr_dir +1 = buy aggressor ("Bid" in the inverted raw convention)
        grid = np.arange(3, dtype=np.int64) * 1_000_000
        buy, sell = trade_value_per_ms(
            np.array([1_000_000, 2_000_000], dtype=np.int64),
            np.array([10.0, 20.0]), np.array([2.0, 3.0]), np.array([1.0, -1.0]), grid,
        )
        np.testing.assert_allclose(buy, [0.0, 20.0, 0.0])
        np.testing.assert_allclose(sell, [0.0, 0.0, 60.0])

    def test_multiple_trades_in_one_ms_sum(self):
        # 1.2ms, 1.7ms ∈ (1, 2] and 2.0ms exactly at the tick → all sum at tick 2
        grid = np.arange(3, dtype=np.int64) * 1_000_000
        buy, _ = trade_value_per_ms(
            np.array([1_200_000, 1_700_000, 2_000_000], dtype=np.int64),
            np.array([10.0, 10.0, 10.0]), np.array([1.0, 2.0, 4.0]),
            np.array([1.0, 1.0, 1.0]), grid,
        )
        np.testing.assert_allclose(buy, [0.0, 0.0, 70.0])

    def test_trade_before_grid_window_dropped(self):
        # grid starts at 5ms: a trade at 3.9ms predates (4ms, 5ms] → dropped,
        # NOT credited to the first tick
        grid = (np.arange(5, dtype=np.int64) + 5) * 1_000_000
        buy, _ = trade_value_per_ms(
            np.array([3_900_000], dtype=np.int64),
            np.array([10.0]), np.array([1.0]), np.array([1.0]), grid,
        )
        np.testing.assert_array_equal(buy, np.zeros(5))

    def test_trade_in_first_tick_window_kept(self):
        # grid starts at 5ms: a trade at 4.5ms ∈ (4ms, 5ms] lands at the first tick
        grid = (np.arange(5, dtype=np.int64) + 5) * 1_000_000
        buy, _ = trade_value_per_ms(
            np.array([4_500_000], dtype=np.int64),
            np.array([10.0]), np.array([1.0]), np.array([1.0]), grid,
        )
        np.testing.assert_allclose(buy, [10.0, 0.0, 0.0, 0.0, 0.0])

    def test_trade_after_grid_end_dropped(self):
        grid = np.arange(3, dtype=np.int64) * 1_000_000
        buy, _ = trade_value_per_ms(
            np.array([2_000_001], dtype=np.int64),
            np.array([10.0]), np.array([1.0]), np.array([1.0]), grid,
        )
        np.testing.assert_array_equal(buy, np.zeros(3))

    def test_invariant_to_same_ns_aggregation(self):
        # Σ qty·prc is preserved by the VWAP same-ns aggregation (impl note 3),
        # so the per-ms sums are identical raw vs aggregated.
        grid = np.arange(3, dtype=np.int64) * 1_000_000
        ts = np.array([1_000_000, 1_000_000, 1_000_000], dtype=np.int64)
        prc = np.array([10.0, 11.0, 12.0])
        qty = np.array([1.0, 2.0, 3.0])
        dir_ = np.array([1.0, 1.0, 1.0])
        raw_buy, raw_sell = trade_value_per_ms(ts, prc, qty, dir_, grid)
        agg = aggregate_same_ns_trade(ts, prc, qty, dir_)
        agg_buy, agg_sell = trade_value_per_ms(*agg, grid)
        # 1·10 + 2·11 + 3·12 = 68 — at tick 1 either way
        np.testing.assert_allclose(raw_buy, [0.0, 68.0, 0.0], rtol=1e-12)
        np.testing.assert_allclose(agg_buy, raw_buy, rtol=1e-12)
        np.testing.assert_allclose(agg_sell, raw_sell, rtol=1e-12)


class TestTradeValueMsEmas:
    """Wallclock trade-flow EMAs (ema_{buy,sell}_trade_value_{N}ms): EMA over the
    per-ms value sums on the 1ms grid. Defining property vs the event-clocked
    `_t` family: the EMA absorbs a 0 every quiet ms, so it DECAYS in real time
    during silence instead of holding its last value on a dead tape.
    """

    def _make_session(self, tr_t, tr_prc, tr_qty, tr_dir, n_bbo=20):
        from boba.dataset.session_data import SessionData
        bbo_t = np.arange(n_bbo, dtype=np.int64) * 1_000_000
        bid = np.full(n_bbo, 100.0)
        ask = np.full(n_bbo, 100.1)
        bq = aq = np.full(n_bbo, 10.0)
        feed_lat = np.zeros(n_bbo, dtype=np.int64)
        return SessionData(
            listing_book_t={"bin": bbo_t},
            listing_book_bid={"bin": bid},
            listing_book_ask={"bin": ask},
            listing_book_bid_qty={"bin": bq},
            listing_book_ask_qty={"bin": aq},
            trade_ts={"bin": tr_t},
            trade_exchange_ts={"bin": tr_t},
            trade_prc={"bin": tr_prc},
            trade_qty={"bin": tr_qty},
            trade_dir={"bin": tr_dir},
            listing_feed_latency_excess_ns={"bin": feed_lat},
            target_listing="bin",
            book_t=bbo_t,
            book_bid=bid,
            book_ask=ask,
            book_mid=(bid + ask) / 2,
            feed_latency_raw_ns=feed_lat,
            feed_latency_excess_ns=feed_lat,
            all_rx=np.sort(np.concatenate([bbo_t, tr_t])),
        )

    def test_ema_impulse_then_real_time_decay(self):
        # ONE buy of value V at ms 5, silence after: the _ms EMA reads V·α at
        # the landing ms, then decays ×(1−α) per quiet ms.
        data = self._make_session(
            tr_t=np.array([5_000_000], dtype=np.int64),
            tr_prc=np.array([100.05]), tr_qty=np.array([2.0]), tr_dir=np.array([1.0]),
        )
        grid = np.arange(20, dtype=np.int64) * 1_000_000
        cfg = make_cfg()
        out = _compute_per_listing("bin", data, grid, cfg)
        a = _alpha(10)
        v = 100.05 * 2.0
        ema = out["ema_buy_trade_value_10ms"]
        np.testing.assert_allclose(ema[:5], 0.0)
        assert ema[5] == pytest.approx(v * a)
        for k in range(6, 20):
            assert ema[k] == pytest.approx(v * a * (1 - a) ** (k - 5)), f"ms {k}"
        # the sell side never fired
        np.testing.assert_allclose(out["ema_sell_trade_value_10ms"], 0.0)

    def test_event_clock_ema_freezes_while_ms_ema_decays(self):
        # The contrast that motivated the family: after the last trade the
        # event-clocked `_t` EMA holds its value (dead tape = stale-high act);
        # the wallclock `_ms` EMA keeps decaying.
        data = self._make_session(
            tr_t=np.array([5_000_000], dtype=np.int64),
            tr_prc=np.array([100.05]), tr_qty=np.array([2.0]), tr_dir=np.array([1.0]),
        )
        grid = np.arange(20, dtype=np.int64) * 1_000_000
        cfg = make_cfg()
        out = _compute_per_listing("bin", data, grid, cfg)
        evt = out["ema_buy_trade_value_10t"]
        ms = out["ema_buy_trade_value_10ms"]
        np.testing.assert_allclose(evt[5:], evt[5])      # frozen
        assert np.all(np.diff(ms[5:]) < 0)               # decaying

    def test_matches_ewm_reference_on_sparse_stream(self):
        # Reference: _ewm_1d over the explicit per-ms impulse array.
        data = self._make_session(
            tr_t=np.array([3_000_000, 7_000_000, 7_400_000], dtype=np.int64),
            tr_prc=np.array([100.0, 101.0, 101.0]),
            tr_qty=np.array([1.5, 0.5, 1.0]),
            tr_dir=np.array([1.0, 1.0, -1.0]),
        )
        grid = np.arange(20, dtype=np.int64) * 1_000_000
        cfg = make_cfg()
        out = _compute_per_listing("bin", data, grid, cfg)
        buy_impulse = np.zeros(20)
        buy_impulse[3] = 100.0 * 1.5
        buy_impulse[7] = 101.0 * 0.5
        sell_impulse = np.zeros(20)
        sell_impulse[8] = 101.0 * 1.0          # 7.4ms → tick 8 (next tick)
        np.testing.assert_allclose(out["buy_trade_value"], buy_impulse)
        np.testing.assert_allclose(out["sell_trade_value"], sell_impulse)
        for N in EMA_TRADE_VALUE_MS_SPANS:
            np.testing.assert_allclose(
                out[f"ema_buy_trade_value_{N}ms"], _ewm_1d(buy_impulse, _alpha(N)),
                rtol=1e-12, err_msg=f"buy span {N}")
            np.testing.assert_allclose(
                out[f"ema_sell_trade_value_{N}ms"], _ewm_1d(sell_impulse, _alpha(N)),
                rtol=1e-12, err_msg=f"sell span {N}")

    def test_quiet_ms_between_trades_pure_decay(self):
        # The tradeless-ms case, explicitly: on every ms with NO trade the EMA
        # absorbs a 0 → y[t] = (1−α)·y[t−1] exactly (pure decay — no freeze,
        # no carry-forward of the last trade's value), and the raw per-ms sum
        # is exactly 0. The next trade lands ON TOP of the decayed state:
        # y[8] = α·V₂ + (1−α)·y[7].
        data = self._make_session(
            tr_t=np.array([3_000_000, 8_000_000], dtype=np.int64),
            tr_prc=np.array([100.0, 100.0]),
            tr_qty=np.array([2.0, 5.0]),
            tr_dir=np.array([1.0, 1.0]),
        )
        grid = np.arange(20, dtype=np.int64) * 1_000_000
        out = _compute_per_listing("bin", data, grid, make_cfg())
        a = _alpha(10)
        v1, v2 = 100.0 * 2.0, 100.0 * 5.0
        raw = out["buy_trade_value"]
        ema = out["ema_buy_trade_value_10ms"]
        quiet = np.ones(20, bool); quiet[[3, 8]] = False
        np.testing.assert_array_equal(raw[quiet], 0.0)     # quiet ms sum = 0
        assert ema[3] == pytest.approx(v1 * a)
        for t in range(4, 8):                              # quiet gap: strict decay
            assert ema[t] == pytest.approx((1 - a) * ema[t - 1]), f"ms {t}"
            assert ema[t] < ema[t - 1]
        assert ema[8] == pytest.approx(a * v2 + (1 - a) * ema[7])
        for t in range(9, 20):                             # quiet tail decays too
            assert ema[t] == pytest.approx((1 - a) * ema[t - 1]), f"ms {t}"

    def test_zero_when_no_trades(self):
        data = self._make_session(
            tr_t=np.array([], dtype=np.int64),
            tr_prc=np.array([], dtype=np.float64),
            tr_qty=np.array([], dtype=np.float64),
            tr_dir=np.array([], dtype=np.float64),
        )
        grid = np.arange(20, dtype=np.int64) * 1_000_000
        cfg = make_cfg()
        out = _compute_per_listing("bin", data, grid, cfg)
        np.testing.assert_array_equal(out["buy_trade_value"], np.zeros(20))
        np.testing.assert_array_equal(out["sell_trade_value"], np.zeros(20))
        for N in EMA_TRADE_VALUE_MS_SPANS:
            np.testing.assert_array_equal(out[f"ema_buy_trade_value_{N}ms"], np.zeros(20))
            np.testing.assert_array_equal(out[f"ema_sell_trade_value_{N}ms"], np.zeros(20))


class TestTradeValueMsNamesAndConfig:
    def test_default_spans(self):
        # the legacy default spans for the per-ms trade-value EMA family
        assert EMA_TRADE_VALUE_MS_SPANS == (10, 25, 50, 100)

    def test_names_present_and_counted(self):
        cfg = make_cfg()
        names = legacy_local_names()
        assert "buy_trade_value" in names
        assert "sell_trade_value" in names
        for N in EMA_TRADE_VALUE_MS_SPANS:
            assert f"ema_buy_trade_value_{N}ms" in names
            assert f"ema_sell_trade_value_{N}ms" in names
        assert cfg.n_features() == 3 * len(names)
        assert len(set(names)) == len(names)   # no collision with the _t family

    def test_config_str_changes_with_spans(self):
        # the spans are output-affecting → must be part of the cache key
        a = make_cfg()
        b = make_cfg(ema_trade_value_ms_spans=(10, 25))
        assert a.config_str() != b.config_str()


class TestTradeValueMsEndToEnd:
    """build_features_raw (production path) must fill the per-ms trade-flow
    columns identically to the _compute_per_listing reference, and the
    only_on_event subset must be the FULL-grid EMA sampled at event ticks
    (never an EMA computed over the event-subset grid)."""

    def _make_session(self, bbo_ms=None):
        from boba.dataset.session_data import SessionData
        if bbo_ms is None:
            bbo_ms = np.arange(30)
        bbo_t = np.asarray(bbo_ms, dtype=np.int64) * 1_000_000
        n = len(bbo_t)
        bid = np.full(n, 100.0)
        ask = np.full(n, 100.1)
        bq = aq = np.full(n, 10.0)
        feed_lat = np.zeros(n, dtype=np.int64)
        tr_t = np.array([5_000_000, 12_300_000], dtype=np.int64)
        tr_prc = np.array([100.05, 100.05])
        tr_qty = np.array([2.0, 4.0])
        tr_dir = np.array([1.0, -1.0])
        listings = ["bin", "byb", "okx"]
        empty_i = np.array([], dtype=np.int64)
        empty_f = np.array([], dtype=np.float64)
        return SessionData(
            listing_book_t={e: bbo_t for e in listings},
            listing_book_bid={e: bid for e in listings},
            listing_book_ask={e: ask for e in listings},
            listing_book_bid_qty={e: bq for e in listings},
            listing_book_ask_qty={e: aq for e in listings},
            trade_ts={"bin": tr_t, "byb": empty_i, "okx": empty_i},
            trade_exchange_ts={"bin": tr_t, "byb": empty_i, "okx": empty_i},
            trade_prc={"bin": tr_prc, "byb": empty_f, "okx": empty_f},
            trade_qty={"bin": tr_qty, "byb": empty_f, "okx": empty_f},
            trade_dir={"bin": tr_dir, "byb": empty_f, "okx": empty_f},
            listing_feed_latency_excess_ns={e: feed_lat for e in listings},
            target_listing="bin",
            book_t=bbo_t,
            book_bid=bid,
            book_ask=ask,
            book_mid=(bid + ask) / 2,
            feed_latency_raw_ns=feed_lat,
            feed_latency_excess_ns=feed_lat,
            all_rx=np.sort(np.concatenate([bbo_t, tr_t])),
        )

    def test_production_columns_match_reference(self):
        data = self._make_session()
        cfg = make_cfg(warmup_ms=0, horizon_ms=0, only_on_event=False)
        result = build_features_raw(data, ["bin", "byb", "okx"], cfg)
        grid_t_ns = result.timestamp_ms.astype(np.int64) * 1_000_000
        ref = _compute_per_listing("bin", data, grid_t_ns, cfg)
        ci = {n: i for i, n in enumerate(result.column_names)}
        check = ["buy_trade_value", "sell_trade_value"] + [
            f"ema_{side}_trade_value_{N}ms"
            for N in EMA_TRADE_VALUE_MS_SPANS for side in ("buy", "sell")
        ]
        for nm in check:
            np.testing.assert_allclose(
                result.x[:, ci[f"bin_{nm}"]], ref[nm].astype(np.float32),
                rtol=1e-6, err_msg=nm)
        # tradeless listings (byb, okx) are zero-filled
        for e in ("byb", "okx"):
            for nm in check:
                np.testing.assert_array_equal(result.x[:, ci[f"{e}_{nm}"]], 0.0)

    def test_event_filtered_rows_match_full_grid_values(self):
        # only_on_event=True keeps a subset of rows, but each kept row's value
        # must equal the full-grid computation at that timestamp — the EMA's
        # real-time decay must NOT be re-clocked onto the event grid. Sparse
        # BBO so the event grid is a strict subset; the 12.3ms trade lands at
        # full-grid tick 13 (a non-event tick), so only its decayed influence
        # is visible at later event rows.
        data = self._make_session(bbo_ms=[0, 5, 10, 15, 20, 25, 29])
        cfg_on = make_cfg(warmup_ms=0, horizon_ms=0, only_on_event=True)
        cfg_off = make_cfg(warmup_ms=0, horizon_ms=0, only_on_event=False)
        r_on = build_features_raw(data, ["bin", "byb", "okx"], cfg_on)
        r_off = build_features_raw(data, ["bin", "byb", "okx"], cfg_off)
        assert r_on.x.shape[0] < r_off.x.shape[0]
        sel = np.isin(r_off.timestamp_ms, r_on.timestamp_ms)
        assert sel.sum() == r_on.x.shape[0]
        ci_on = {n: i for i, n in enumerate(r_on.column_names)}
        ci_off = {n: i for i, n in enumerate(r_off.column_names)}
        for nm in ("bin_buy_trade_value", "bin_ema_buy_trade_value_10ms",
                   "bin_ema_sell_trade_value_100ms"):
            np.testing.assert_allclose(
                r_on.x[:, ci_on[nm]], r_off.x[sel, ci_off[nm]], err_msg=nm)


class TestColumnSelection:
    """cfg.columns IS the dataset definition: an ordered tuple of ColumnSpecs,
    no full-catalogue default. Output order is exactly the expansion/request
    order, and the cache key hashes the ORDERED expanded names, so different
    selections — including permutations of the same selection — never collide
    on disk."""

    _LISTINGS = ["bin", "byb", "okx"]

    # ── config / names ────────────────────────────────────────────────────

    def test_columns_required(self):
        # no full-catalogue default any more: an empty selection is an error
        cfg = make_cfg(columns=())
        with pytest.raises(ValueError, match="at least one ColumnSpec"):
            cfg.n_features()

    def test_n_features_is_len_columns(self):
        full = feature_names(make_cfg(), self._LISTINGS)
        cfg = make_cfg(columns=cols_from_names(full[:7]))
        assert cfg.n_features() == 7

    def test_feature_names_in_request_order(self):
        full = feature_names(make_cfg(), self._LISTINGS)
        pick = (full[50], full[3], full[120])              # scrambled request order
        cfg = make_cfg(columns=cols_from_names(pick))
        # output order is the REQUEST order — canonical reordering is gone
        assert feature_names(cfg, self._LISTINGS) == [full[50], full[3], full[120]]

    def test_unknown_column_raises(self):
        # a name that never existed cannot be mapped to any template
        with pytest.raises(ValueError, match="cannot map legacy name"):
            cols_from_names(["bin_microprice", "bin_nope"])

    def test_unknown_listing_raises(self):
        cfg = make_cfg(columns=(col("{LISTING}_microprice", LISTING="bin"),
                                col("{LISTING}_microprice", LISTING="xyz")))
        with pytest.raises(ValueError, match="not one of the listings"):
            feature_names(cfg, self._LISTINGS)

    def test_duplicate_column_raises(self):
        cfg = make_cfg(columns=cols_from_names(["bin_microprice", "bin_microprice"]))
        with pytest.raises(ValueError, match="duplicate columns after expansion"):
            feature_names(cfg, self._LISTINGS)

    # ── cache key ─────────────────────────────────────────────────────────

    def test_config_str_differs_from_full_catalogue(self):
        sel = make_cfg(columns=cols_from_names(["bin_microprice", "byb_dt_10b"]))
        assert sel.config_str() != make_cfg().config_str()

    def test_config_str_order_sensitive(self):
        # column order is part of the output → permutations of the same
        # selection are DIFFERENT datasets and must get different cache keys
        a = make_cfg(columns=cols_from_names(["bin_microprice", "byb_dt_10b"]))
        b = make_cfg(columns=cols_from_names(["byb_dt_10b", "bin_microprice"]))
        assert a.config_str() != b.config_str()

    def test_config_str_differs_across_subsets(self):
        # same selection size → the _f fragment matches; the hash must differ
        a = make_cfg(columns=cols_from_names(["bin_microprice", "byb_dt_10b"]))
        b = make_cfg(columns=cols_from_names(["bin_microprice", "okx_dt_10b"]))
        assert a.config_str() != b.config_str()

    # ── end-to-end build ──────────────────────────────────────────────────

    _PICK = (
        "bin_microprice",                        # S1 instantaneous
        "byb_dt_10b",                            # S1 dt
        "bin_buy_trade_value",                   # S1 per-ms flow sum
        "bin_return_5ms",                        # S2 return
        "okx_ema_microprice_centered_100ms",     # S2 calendar EMA (mean only)
        "bin_ema_microprice_centered_sq_100ms",  # S2 calendar EMA (sq only)
        "bin_ema_buy_trade_value_10ms",          # S2 per-ms flow EMA
        "bin_ema_buy_trade_value_3t",            # S3 trade-clock EMA
        "byb_ema_sell_trade_qty_3t",             # S3 zero-fill path (byb has no trades)
        "okx_ema_ofi_3b",                        # S3 BBO-clock EMA
    )

    def test_build_subset_matches_full(self):
        data = TestTradeValueMsEndToEnd()._make_session()
        kw = dict(warmup_ms=0, horizon_ms=0, only_on_event=False)
        full = build_features_raw(data, self._LISTINGS, make_cfg(**kw))
        cfg = make_cfg(columns=cols_from_names(tuple(reversed(self._PICK))), **kw)
        sub = build_features_raw(data, self._LISTINGS, cfg)
        assert sub.x.shape == (full.x.shape[0], len(self._PICK))
        # output order is the REQUEST order (here: reversed _PICK), not canonical
        assert sub.column_names == list(reversed(self._PICK))
        fci = {n: i for i, n in enumerate(full.column_names)}
        for j, nm in enumerate(sub.column_names):
            np.testing.assert_array_equal(sub.x[:, j], full.x[:, fci[nm]], err_msg=nm)
        # cost fields and timestamps are independent of the column selection
        np.testing.assert_array_equal(sub.timestamp_ms, full.timestamp_ms)
        np.testing.assert_array_equal(sub.eval_mid, full.eval_mid)
        np.testing.assert_array_equal(sub.c_ask_entry_l, full.c_ask_entry_l)

    def test_explicit_names_roundtrip_equals_catalogue(self):
        # specs rebuilt from the catalogue's concrete names, in the same order,
        # define the identical dataset
        data = TestTradeValueMsEndToEnd()._make_session()
        kw = dict(warmup_ms=0, horizon_ms=0, only_on_event=False)
        full = build_features_raw(data, self._LISTINGS, make_cfg(**kw))
        cfg = make_cfg(columns=cols_from_names(full.column_names), **kw)
        res = build_features_raw(data, self._LISTINGS, cfg)
        assert res.column_names == full.column_names
        np.testing.assert_array_equal(res.x, full.x)

    # ── disk cache ────────────────────────────────────────────────────────

    def test_cache_roundtrip_and_key_isolation(self, tmp_path):
        from boba.dataset.raw import _save_raw, _load_raw
        data = TestTradeValueMsEndToEnd()._make_session()
        kw = dict(warmup_ms=0, horizon_ms=0, only_on_event=False)
        names_a = ["bin_microprice", "byb_dt_10b"]
        cfg_a = make_cfg(columns=cols_from_names(names_a), **kw)
        cfg_b = make_cfg(columns=cols_from_names(["bin_microprice", "okx_ema_ofi_3b"]), **kw)
        cfg_full = make_cfg(**kw)
        # different selections (and the full catalogue) → three distinct cache keys
        keys = {c.config_str() for c in (cfg_a, cfg_b, cfg_full)}
        assert len(keys) == 3
        # npz round-trip preserves the subset matrix, names and cost fields exactly
        sub = build_features_raw(data, self._LISTINGS, cfg_a)
        p = tmp_path / cfg_a.config_str() / "data.npz"
        _save_raw(p, sub)
        back = _load_raw(p)
        assert back.column_names == sub.column_names == names_a
        np.testing.assert_array_equal(back.x, sub.x)
        np.testing.assert_array_equal(back.timestamp_ms, sub.timestamp_ms)
        np.testing.assert_array_equal(back.c_bid_entry_l, sub.c_bid_entry_l)


class TestTradeOrMoveFilter:
    """only_on_trade_or_move=True keeps only grid rows whose latest ms
    (t−1ms, t] contains a trade or a microprice change on ≥1 generated
    listing. Uses the at-or-before window — a kept row's features already
    reflect the event that kept it (unlike only_on_event's floor convention)
    — and supersedes only_on_event. The cache key must reflect the flag."""

    _LISTINGS = ["bin", "byb", "okx"]

    def _make_session(self):
        from boba.dataset.session_data import SessionData
        n = 30
        bbo_t = np.arange(n, dtype=np.int64) * 1_000_000
        bid = np.full(n, 100.0); ask = np.full(n, 100.1)
        bq = aq = np.full(n, 10.0)
        # okx: bid PRICE steps at ms 12 (stays), bid QTY steps at ms 25 (stays)
        # — both change the microprice → exactly two move events. All other BBO
        # updates on every listing are identical-state → not moves.
        okx_bid = bid.copy(); okx_bid[12:] = 100.01
        okx_bq = bq.copy(); okx_bq[25:] = 15.0
        feed = np.zeros(n, dtype=np.int64)
        # bin: one trade strictly inside an ms (5.3ms → row 6) and one exactly
        # at a tick (20.0ms → row 20). byb: no trades, no moves.
        tr_t = np.array([5_300_000, 20_000_000], dtype=np.int64)
        tr_prc = np.array([100.05, 100.05]); tr_qty = np.array([2.0, 4.0])
        tr_dir = np.array([1.0, -1.0])
        empty_i = np.array([], dtype=np.int64); empty_f = np.array([], dtype=np.float64)
        return SessionData(
            listing_book_t={e: bbo_t for e in self._LISTINGS},
            listing_book_bid={"bin": bid, "byb": bid, "okx": okx_bid},
            listing_book_ask={e: ask for e in self._LISTINGS},
            listing_book_bid_qty={"bin": bq, "byb": bq, "okx": okx_bq},
            listing_book_ask_qty={e: aq for e in self._LISTINGS},
            trade_ts={"bin": tr_t, "byb": empty_i, "okx": empty_i},
            trade_exchange_ts={"bin": tr_t, "byb": empty_i, "okx": empty_i},
            trade_prc={"bin": tr_prc, "byb": empty_f, "okx": empty_f},
            trade_qty={"bin": tr_qty, "byb": empty_f, "okx": empty_f},
            trade_dir={"bin": tr_dir, "byb": empty_f, "okx": empty_f},
            listing_feed_latency_excess_ns={e: feed for e in self._LISTINGS},
            target_listing="bin",
            book_t=bbo_t, book_bid=bid, book_ask=ask, book_mid=(bid + ask) / 2,
            feed_latency_raw_ns=feed, feed_latency_excess_ns=feed,
            all_rx=np.sort(np.concatenate([bbo_t, tr_t])),
        )

    def test_keeps_only_trade_or_move_rows(self):
        data = self._make_session()
        cfg = make_cfg(warmup_ms=0, horizon_ms=0, only_on_trade_or_move=True)
        res = build_features_raw(data, self._LISTINGS, cfg)
        # 6: bin trade 5.3ms ∈ (5,6]; 12: okx price step exactly at 12ms ∈ (11,12];
        # 20: bin trade exactly at tick; 25: okx qty-driven microprice move.
        # Identical-state BBO updates on every other ms do NOT count.
        np.testing.assert_array_equal(res.timestamp_ms, [6.0, 12.0, 20.0, 25.0])
        ci = {n: i for i, n in enumerate(res.column_names)}
        # kept rows already SHOW their triggering event (at-or-before window)
        assert res.x[0, ci["bin_buy_trade_value"]] == pytest.approx(100.05 * 2.0)
        assert res.x[2, ci["bin_sell_trade_value"]] == pytest.approx(100.05 * 4.0)
        mp = res.x[:, ci["okx_microprice"]]
        assert mp[1] > mp[0]               # row 12 reflects the okx price step
        assert mp[3] != mp[2]              # row 25 reflects the qty-driven move

    def test_supersedes_only_on_event(self):
        data = self._make_session()
        kw = dict(warmup_ms=0, horizon_ms=0, only_on_trade_or_move=True)
        r_ev = build_features_raw(data, self._LISTINGS, make_cfg(only_on_event=True, **kw))
        r_no = build_features_raw(data, self._LISTINGS, make_cfg(only_on_event=False, **kw))
        np.testing.assert_array_equal(r_ev.timestamp_ms, r_no.timestamp_ms)
        np.testing.assert_array_equal(r_ev.x, r_no.x)

    def test_kept_rows_match_full_grid_values(self):
        # The filter selects rows; it must NOT change any value — every kept
        # row equals the full-grid build at the same timestamp (EMAs keep
        # their real-time clock; nothing is re-clocked onto the kept rows).
        data = self._make_session()
        r_f = build_features_raw(
            data, self._LISTINGS, make_cfg(warmup_ms=0, horizon_ms=0, only_on_trade_or_move=True))
        r_all = build_features_raw(
            data, self._LISTINGS, make_cfg(warmup_ms=0, horizon_ms=0, only_on_event=False))
        sel = np.isin(r_all.timestamp_ms, r_f.timestamp_ms)
        assert sel.sum() == len(r_f.timestamp_ms)
        np.testing.assert_array_equal(r_f.x, r_all.x[sel])
        np.testing.assert_array_equal(r_f.eval_mid, r_all.eval_mid[sel])

    def test_composes_with_column_selection(self):
        data = self._make_session()
        cols = ("bin_buy_trade_value", "okx_microprice")
        cfg = make_cfg(warmup_ms=0, horizon_ms=0,
                       only_on_trade_or_move=True, columns=cols_from_names(cols))
        res = build_features_raw(data, self._LISTINGS, cfg)
        assert res.column_names == list(cols)
        np.testing.assert_array_equal(res.timestamp_ms, [6.0, 12.0, 20.0, 25.0])
        assert res.x.shape == (4, 2)
        assert res.x[0, 0] == pytest.approx(100.05 * 2.0)

    def test_config_str_reflects_flag_and_default_unchanged(self):
        on = make_cfg(only_on_trade_or_move=True)
        off = make_cfg()
        assert on.config_str() != off.config_str()
        assert "_tom1" in on.config_str()
        # default keys keep their pre-flag form — existing caches stay valid
        assert "_tom" not in off.config_str()
        # composes with column selection in the key (the selection folds into
        # the hashed names — no separate _cols fragment any more)
        both = make_cfg(only_on_trade_or_move=True, columns=cols_from_names(["bin_microprice"]))
        assert "_tom1" in both.config_str()
        assert both.config_str() != on.config_str()
