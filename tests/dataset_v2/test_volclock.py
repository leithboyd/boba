"""Shared trade-clock realized-vol — property tests + an independent dead-simple reference.

Written test-first (the generic builder doesn't exist yet). The reference here is the
single-threaded, non-generic ground truth; the library implementation will later be diffed
against it.

The headline invariant (the whole point of the shared clock): if two streams trace the SAME
price line — one just a down-sampling of the other that drops only no-move book updates — and
both are clocked off the BUSY stream's trades, their event-vol-EMAs are identical. Vol is a
function of the forward-filled price path over the shared clock's windows; it does not care
how many redundant points or how many of its own trades a stream has.
"""
from __future__ import annotations

import numpy as np

MS = 1_000_000  # ns per ms


# ── Reference: dead-simple, direct, single-threaded ──────────────────────────────────────────

def vol_ref(price_ts_ns, microprice, clock_ts_ns, grid_ms, N, eps=1e-30):
    """Realized-vol tick-EMA of a listing's microprice, clocked off ``clock_ts_ns`` trades.

    - Forward-fill the listing's microprice onto the 1ms grid (full resolution — every move
      counts, no sampling at trade instants).
    - Per clock interval k = (C_{k-1}, C_k], realized variance V_k = Σ squared 1ms log-returns.
    - Tick EMA over clock events: y_k = α·V_k + (1-α)·y_{k-1},  α = 2/(N+1)  (span = N clock events).
    - vol(t) = sqrt(y at the latest clock event ≤ t), forward-filled; 0 before the first clock event.
    Deliberately plain (loops, no shared code) so it can serve as the oracle.
    """
    grid_ms = np.asarray(grid_ms, dtype=np.int64)
    grid_ns = grid_ms * MS

    # forward-fill microprice to each grid tick (latest book obs at-or-before the tick)
    pi = np.searchsorted(price_ts_ns, grid_ns, side="right") - 1
    mp = np.where(pi >= 0, microprice[np.clip(pi, 0, len(microprice) - 1)], np.nan)
    valid = np.isfinite(mp) & (mp > 0)
    logmp = np.where(valid, np.log(np.maximum(mp, eps)), np.nan)

    # 1ms squared log-returns, cumulatively summed (variance over grid[a:b) = cs[b] - cs[a])
    sq = np.zeros(len(grid_ms))
    both = valid[1:] & valid[:-1]
    sq[1:] = np.where(both, (logmp[1:] - logmp[:-1]) ** 2, 0.0)
    cs = np.concatenate([[0.0], np.cumsum(sq)])

    # clock events inside the grid, as grid positions (count of grid ticks ≤ the event).
    # Filter the ms tick to the grid bound; keep the ns time for the causal forward-fill.
    tr = np.sort(np.asarray(clock_ts_ns, dtype=np.int64))
    clk_ms = tr // MS
    keep = (clk_ms >= grid_ms[0]) & (clk_ms <= grid_ms[-1])
    tr, clk_ms = tr[keep], clk_ms[keep]
    cpos = np.searchsorted(grid_ms, clk_ms, side="right")          # grid index just after each clock event

    # tick EMA over clock intervals
    alpha = 2.0 / (N + 1)
    y, prev = 0.0, 0
    yk = np.empty(len(cpos))
    for k in range(len(cpos)):
        V = cs[cpos[k]] - cs[prev]                                 # variance in (prev grid pos, C_k]
        y = alpha * V + (1.0 - alpha) * y
        yk[k] = y
        prev = cpos[k]

    # forward-fill the per-clock-event EMA to the grid (constant between clock events; 0 before
    # first). ns-causal: a grid tick sees only clock trades whose ns time is at-or-before it.
    vol = np.zeros(len(grid_ms))
    ci = np.searchsorted(tr, grid_ms * MS, side="right") - 1       # latest clock trade ≤ tick (ns)
    has = ci >= 0
    vol[has] = np.sqrt(np.maximum(yk[np.clip(ci[has], 0, len(yk) - 1)], 0.0))
    return vol


# ── Synthetic builders: one price line, two sampling densities ───────────────────────────────

def _price_line(n_changes, seed):
    """A microprice 'line': change times (ms) and the price after each change."""
    rng = np.random.default_rng(seed)
    times_ms = np.cumsum(rng.integers(1, 20, n_changes)).astype(np.int64)   # gaps 1–19 ms
    prices = 100.0 + np.cumsum(rng.standard_normal(n_changes) * 0.01)
    return times_ms, prices


def _book(times_ms, prices, redundant_per=0, seed=0):
    """Book observations tracing the line. ``redundant_per`` extra NO-MOVE updates inserted in
    each gap (same price ⇒ they don't change the forward-filled path) — i.e. the points a
    down-sampling would drop."""
    rng = np.random.default_rng(seed)
    ts, mp = [], []
    for i in range(len(times_ms)):
        ts.append(int(times_ms[i])); mp.append(float(prices[i]))
        if redundant_per and i + 1 < len(times_ms):
            gap = times_ms[i + 1] - times_ms[i]
            for off in sorted(set(rng.integers(1, max(gap, 2), redundant_per))):
                if times_ms[i] + off < times_ms[i + 1]:
                    ts.append(int(times_ms[i] + off)); mp.append(float(prices[i]))   # same price = no move
    return np.array(ts, np.int64) * MS, np.array(mp, np.float64)


def _trades(lo_ms, hi_ms, every, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(lo_ms, hi_ms, every, np.int64)
    t = t + rng.integers(0, max(every - 1, 1), len(t))      # jitter so trade times aren't on book times
    return np.sort(t[(t >= lo_ms) & (t < hi_ms)]) * MS


# ── The property ─────────────────────────────────────────────────────────────────────────────

class TestSharedClockAlignment:
    def test_downsampled_aligns_on_busy_clock(self):
        # SAME line: A is the dense stream (lots of no-move book updates), B drops them.
        times_ms, prices = _price_line(120, seed=3)
        a_ts, a_mp = _book(times_ms, prices, redundant_per=6, seed=11)   # busy: redundant 0-move updates
        b_ts, b_mp = _book(times_ms, prices, redundant_per=0)            # down-sampled: no-move updates removed
        assert len(a_ts) > 2 * len(b_ts)                                # genuinely denser
        grid = np.arange(1, int(times_ms[-1]) + 1, dtype=np.int64)
        clock = _trades(1, int(times_ms[-1]), every=7, seed=5)          # the BUSY stream's trade clock

        for N in (5, 20, 100):
            va = vol_ref(a_ts, a_mp, clock, grid, N)
            vb = vol_ref(b_ts, b_mp, clock, grid, N)                    # B's price, same (busy) clock
            np.testing.assert_allclose(va, vb, rtol=1e-9, atol=1e-12,
                                       err_msg=f"down-sampled vol diverged from dense, N={N}")

    def test_own_sparse_clock_diverges(self):
        # control: clock the down-sampled stream off ITS OWN sparse trades and it no longer lines up
        times_ms, prices = _price_line(120, seed=3)
        a_ts, a_mp = _book(times_ms, prices, redundant_per=6, seed=11)
        b_ts, b_mp = _book(times_ms, prices, redundant_per=0)
        grid = np.arange(1, int(times_ms[-1]) + 1, dtype=np.int64)
        busy = _trades(1, int(times_ms[-1]), every=7, seed=5)
        sparse = _trades(1, int(times_ms[-1]), every=70, seed=6)        # 10× fewer trades
        N = 20
        shared = vol_ref(b_ts, b_mp, busy, grid, N)
        own = vol_ref(b_ts, b_mp, sparse, grid, N)
        assert not np.allclose(shared, own, rtol=1e-2)                  # different clock ⇒ different windows


# ── Reference sanity + edges ─────────────────────────────────────────────────────────────────

class TestReferenceAndEdges:
    def test_flat_price_is_zero_vol(self):
        ts = np.arange(0, 100, 5, np.int64) * MS
        mp = np.full(len(ts), 100.0)                                    # no moves at all
        grid = np.arange(1, 100, dtype=np.int64)
        clock = _trades(1, 100, every=3, seed=1)
        np.testing.assert_array_equal(vol_ref(ts, mp, clock, grid, 10), np.zeros(len(grid)))

    def test_zero_before_first_clock_trade(self):
        ts = np.array([0, 10, 20], np.int64) * MS
        mp = np.array([100.0, 100.5, 101.0])
        grid = np.arange(1, 30, dtype=np.int64)
        clock = np.array([15], np.int64) * MS                          # one trade at ms 15
        vol = vol_ref(ts, mp, clock, grid, 5)
        assert np.all(vol[grid < 15] == 0.0)                           # nothing before the first clock event
        assert vol[grid >= 15].max() > 0.0                             # something after

    def test_hand_computed_single_window(self):
        # microprice 100 → 101 (one move at ms 10); two clock trades at ms 5 and ms 20, N=1 (α=1).
        ts = np.array([0, 10], np.int64) * MS
        mp = np.array([100.0, 101.0])
        grid = np.arange(1, 25, dtype=np.int64)
        clock = np.array([5, 20], np.int64) * MS
        vol = vol_ref(ts, mp, clock, grid, N=1)                        # α=1 ⇒ y = V (the latest interval)
        r = np.log(101.0) - np.log(100.0)
        # tick at ms ≥20: latest interval (5,20] contains the single move ⇒ y = r², vol = |r|
        np.testing.assert_allclose(vol[grid >= 20][0], abs(r), rtol=1e-12)
        # ticks in [5,20): interval (start,5] had no move ⇒ y = 0
        assert np.all(vol[(grid >= 5) & (grid < 20)] == 0.0)

    def test_trades_tied_in_same_ms(self):
        # multiple clock trades in one ms must not break the windowing
        ts = np.array([0, 10, 20], np.int64) * MS
        mp = np.array([100.0, 100.5, 100.2])
        grid = np.arange(1, 30, dtype=np.int64)
        clock = np.array([7, 7, 7, 15], np.int64) * MS                 # three trades tied at ms 7
        vol = vol_ref(ts, mp, clock, grid, 4)
        assert np.all(np.isfinite(vol)) and vol.shape == grid.shape

    def test_fewer_than_N_trades_is_stable(self):
        # only 2 clock trades but N=50 — EMA is just cold, not broken
        ts = np.arange(0, 50, 3, np.int64) * MS
        mp = 100.0 + np.cumsum(np.full(len(ts), 0.001))
        grid = np.arange(1, 50, dtype=np.int64)
        clock = np.array([10, 40], np.int64) * MS
        vol = vol_ref(ts, mp, clock, grid, N=50)
        assert np.all(np.isfinite(vol)) and np.all(vol >= 0)


# ── Library (boba.dataset_v2) vs the oracle, end-to-end ──────────────────────────────────────

def _session_same_mid(seed, lo_ms=0, hi_ms=4000):
    """A SessionData where bin and byb share the IDENTICAL mid path but trade at different rates
    (bin every 3 ms, byb every 11 ms). Lets us prove a shared clock lines two venues up."""
    from boba.dataset_v2.session_data import SessionData
    rng = np.random.default_rng(seed)
    bt = np.arange(lo_ms, hi_ms, dtype=np.int64) * MS
    n = len(bt)
    mid = 100.0 + np.cumsum(rng.choice([-1.0, 0.0, 1.0], n) * 0.001)
    bid, ask = mid - 0.01, mid + 0.01
    bf = {f: {} for f in ("listing_book_t", "listing_book_bid", "listing_book_ask",
                          "listing_book_bid_qty", "listing_book_ask_qty", "listing_feed_latency_excess_ns")}
    tf = {f: {} for f in ("trade_ts", "trade_exchange_ts", "trade_prc", "trade_qty", "trade_dir")}
    for l, every in (("bin", 3), ("byb", 11)):
        bf["listing_book_t"][l] = bt
        bf["listing_book_bid"][l] = bid.copy(); bf["listing_book_ask"][l] = ask.copy()
        bf["listing_book_bid_qty"][l] = np.full(n, 10.0); bf["listing_book_ask_qty"][l] = np.full(n, 10.0)
        bf["listing_feed_latency_excess_ns"][l] = np.zeros(n, np.int64)
        tt = bt[::every]
        tf["trade_ts"][l] = tt; tf["trade_exchange_ts"][l] = tt
        tf["trade_prc"][l] = mid[::every]; tf["trade_qty"][l] = np.full(len(tt), 1.0)
        tf["trade_dir"][l] = rng.choice([-1.0, 1.0], len(tt))
    z = np.zeros(n, np.int64)
    all_rx = np.sort(np.concatenate([bt, bt, tf["trade_ts"]["bin"], tf["trade_ts"]["byb"]]))
    return SessionData(target_listing="bin", all_rx=all_rx, book_t=bt, book_bid=bid, book_ask=ask,
                       book_mid=mid, feed_latency_raw_ns=z, feed_latency_excess_ns=z, **bf, **tf)


def _build(sd, columns, listings=("bin", "byb"), microprice_ref=None):
    from boba.dataset_v2 import DatasetRawConfig
    from boba.dataset_v2.raw import build_features_raw
    cfg = DatasetRawConfig(columns=columns, listings=tuple(listings), target_listing="bin",
                           warmup_ms=0, horizon_ms=0.0, event_mask="none",
                           wide_threshold={l: 1.0e-4 for l in listings},
                           microprice_ref=microprice_ref or {})
    return build_features_raw(sd, list(listings), cfg)


class TestLibraryVsOracle:
    def test_library_matches_raw_oracle(self):
        # MANUAL calc on the raw arrays (vol_ref, the PoC logic) must equal the dataset's column,
        # for own-clock AND foreign-clock vol, at several N.
        from boba.dataset_v2 import col
        from tests.dataset_v2.test_engine import _synth_block            # real BBO+trade SessionData
        sd = _synth_block(0, 4000 * MS, seed=7)
        out = _build(sd, (
            col("{LISTING}_vol_{N}t", LISTING=["bin"], N=[10, 50]),                    # own clock
            col("{LISTING}_vol_{N}t@{CLOCK}", LISTING=["byb"], N=[10, 50], CLOCK="trades_bin"),  # byb on bin's clock
        ))
        grid = out.timestamp_ms.astype(np.int64)
        mid = lambda l: (sd.listing_book_bid[l] + sd.listing_book_ask[l]) / 2.0
        for name, L, C, N in (("bin_vol_10t", "bin", "bin", 10),
                              ("bin_vol_50t", "bin", "bin", 50),
                              ("byb_vol_10t@trades_bin", "byb", "bin", 10),
                              ("byb_vol_50t@trades_bin", "byb", "bin", 50)):
            ref = vol_ref(sd.listing_book_t[L], mid(L), sd.trade_ts[C], grid, N)
            got = out.x[:, out.column_names.index(name)]
            np.testing.assert_allclose(got, ref.astype(np.float32), rtol=1e-5, atol=1e-9,
                                       err_msg=f"library vol {name} != raw oracle")

    def test_own_vs_foreign_clock_actually_differs(self):
        # sanity that the clock binding is wired: byb on its own slow clock (every 11ms) ≠ byb on
        # bin's faster clock (every 3ms). Same mid, different clock density ⇒ different windows.
        from boba.dataset_v2 import col
        sd = _session_same_mid(seed=2)
        out = _build(sd, (
            col("{LISTING}_vol_{N}t", LISTING=["byb"], N=[10]),                         # own (byb every 11ms)
            col("{LISTING}_vol_{N}t@{CLOCK}", LISTING=["byb"], N=[10], CLOCK="trades_bin"),     # bin clock (every 3ms)
        ))
        own = out.x[:, out.column_names.index("byb_vol_10t")]
        shared = out.x[:, out.column_names.index("byb_vol_10t@trades_bin")]
        assert not np.allclose(own, shared, rtol=1e-2)

    def test_own_clock_equals_no_clock(self):
        # the bare column (default = own trade clock) must be byte-identical to the same column
        # explicitly clocked off ITSELF (@self) — the @{CLOCK} default and the explicit own clock
        # are the same thing.
        from boba.dataset_v2 import col
        from tests.dataset_v2.test_engine import _synth_block
        sd = _synth_block(0, 4000 * MS, seed=7)
        out = _build(sd, (
            col("{LISTING}_vol_{N}t", LISTING=["bin"], N=[10, 50]),                       # bare → own clock
            col("{LISTING}_vol_{N}t@{CLOCK}", LISTING=["bin"], N=[10, 50], CLOCK="trades_bin"),   # @trades_bin == own
        ))
        for N in (10, 50):
            bare = out.x[:, out.column_names.index(f"bin_vol_{N}t")]
            self_clk = out.x[:, out.column_names.index(f"bin_vol_{N}t@trades_bin")]
            np.testing.assert_array_equal(bare, self_clk)

    def test_clock_name_must_be_trades_listing(self):
        # the clock token names its event type + listing: 'trades_<listing>'. Anything else rejected.
        import pytest
        from boba.dataset_v2 import col, expand_columns
        for bad in ("bin", "trades_xyz", "trades_"):            # no prefix / unknown listing / empty
            with pytest.raises(ValueError):
                expand_columns((col("{LISTING}_vol_{N}t@{CLOCK}", LISTING="bin", N=10, CLOCK=bad),),
                               listings=["bin", "byb"])
        exp = expand_columns((col("{LISTING}_vol_{N}t@{CLOCK}", LISTING="byb", N=100, CLOCK="trades_bin"),),
                             listings=["bin", "byb"])
        assert exp.names[0] == "byb_vol_100t@trades_bin" and exp.units[0].clock == "trades_bin"

    def test_shared_clock_aligns_venues(self):
        # the user's invariant: same mid path on two venues + the SAME clock ⇒ identical event-vol.
        from boba.dataset_v2 import col
        sd = _session_same_mid(seed=2)
        for N in (5, 20, 100):
            out = _build(sd, (col("{LISTING}_vol_{N}t@{CLOCK}", LISTING=["bin", "byb"], N=[N], CLOCK="trades_bin"),))
            a = out.x[:, out.column_names.index(f"bin_vol_{N}t@trades_bin")]
            b = out.x[:, out.column_names.index(f"byb_vol_{N}t@trades_bin")]
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9,
                                       err_msg=f"venues did not line up on the shared clock, N={N}")
        # …and on their OWN (different-rate) clocks they should NOT line up
        out = _build(sd, (col("{LISTING}_vol_{N}t", LISTING=["bin", "byb"], N=[20]),))
        assert not np.allclose(out.x[:, out.column_names.index("bin_vol_20t")],
                               out.x[:, out.column_names.index("byb_vol_20t")], rtol=1e-2)

    def test_vol_is_chunk_independent(self, tmp_path):
        # vol is event-clocked, so the engine's tail carry must warm it across block seams:
        # a chunked+carried build == the continuous build on every shared row.
        import dataclasses
        from boba.dataset_v2 import DatasetRawConfig, col
        from boba.dataset_v2.raw import build_features_raw
        from boba.dataset_v2.engine import Block, build_chunked, concat_sessions
        from boba.dataset_v2.cache import read_blocks
        from boba.dataset_v2.driver import tail_window_ns_for
        from tests.dataset_v2.test_engine import _synth_block, LISTINGS, TARGET, SPAN_MS
        blocks = [Block(f"b{i}", i * SPAN_MS, (i + 1) * SPAN_MS,
                        _synth_block(i * SPAN_MS * MS, (i + 1) * SPAN_MS * MS, seed=i)) for i in range(3)]
        cfg = DatasetRawConfig(
            columns=(col("{LISTING}_vol_{N}t", LISTING=["bin"], N=[10, 50]),
                     col("{LISTING}_vol_{N}t@{CLOCK}", LISTING=["byb"], N=[20], CLOCK="trades_bin")),
            listings=tuple(LISTINGS), target_listing=TARGET,
            warmup_ms=0, horizon_ms=50.0, event_mask="none", wide_threshold={l: 1e-4 for l in LISTINGS})
        sd = concat_sessions([b.session for b in blocks])
        names = list(build_features_raw(sd, list(LISTINGS), cfg).column_names)
        build_chunked(blocks, cfg, tmp_path, "gh", tail_window_ns=tail_window_ns_for(cfg), capacity=4000)
        v2 = read_blocks(tmp_path, "gh", ["b0", "b1", "b2"], names)
        cont = build_features_raw(sd, list(LISTINGS), dataclasses.replace(cfg, warmup_ms=0))
        idx = {int(t): i for i, t in enumerate(cont.timestamp_ms)}
        rows = [idx[int(t)] for t in v2.timestamp_ms]
        assert len(rows) > 1000
        np.testing.assert_allclose(v2.x, cont.x[rows], rtol=1e-5, atol=1e-7)


# ════════════════════════════════════════════════════════════════════════════════════════════
# Book-event quantities on the trade clock — PoC: one LEVEL (book_imbalance) + one FLOW (ofi)
# ════════════════════════════════════════════════════════════════════════════════════════════

def _ema_loop(x, N):
    """Dead-simple causal EMA y[i]=α·x[i]+(1-α)·y[i-1], y[-1]=0 (matches _ewm_1d/lfilter)."""
    a, acc = 2.0 / (N + 1), 0.0
    y = np.empty(len(x))
    for i in range(len(x)):
        acc = a * x[i] + (1.0 - a) * acc
        y[i] = acc
    return y


def _clk_to_grid(clock_ts_ns, grid_ms):
    """Clock trades sorted+clipped to the grid; returns (trade_ns, trade_ms, ci) where ci = the
    latest clock trade ≤ each grid tick."""
    grid_ms = np.asarray(grid_ms, np.int64)
    tr = np.sort(np.asarray(clock_ts_ns, np.int64))           # all clock trades (warmed by any pre-grid)
    ci = np.searchsorted(tr, grid_ms * MS, side="right") - 1  # ns-causal: clock trade ns ≤ tick ns
    return tr, tr // MS, ci


def level_ema_ref(value_ts_ns, value, clock_ts_ns, grid_ms, N, square=False):
    """LEVEL oracle: tick-EMA of `value` SAMPLED (forward-filled) at each clock trade (`_sq`
    squares the sampled value before the EMA)."""
    grid_ms = np.asarray(grid_ms, np.int64)
    tr, clk_ms, ci = _clk_to_grid(clock_ts_ns, grid_ms)
    vi = np.searchsorted(value_ts_ns, tr, side="right") - 1
    samp = np.where(vi >= 0, value[np.clip(vi, 0, len(value) - 1)], 0.0) if len(tr) else np.empty(0)
    if square:
        samp = samp * samp
    y = _ema_loop(samp, N)
    out = np.zeros(len(grid_ms)); has = ci >= 0
    if len(y):
        out[has] = y[np.clip(ci[has], 0, len(y) - 1)]
    return out


def flow_ema_ref(event_ts_ns, flow, clock_ts_ns, grid_ms, N):
    """FLOW oracle: tick-EMA of the per-clock-interval SUM of `flow` over its events."""
    grid_ms = np.asarray(grid_ms, np.int64)
    tr, clk_ms, ci = _clk_to_grid(clock_ts_ns, grid_ms)
    cs = np.concatenate([[0.0], np.cumsum(flow)])
    epos = np.searchsorted(event_ts_ns, tr, side="right")
    S, prev = np.empty(len(epos)), 0
    for k in range(len(epos)):
        S[k] = cs[epos[k]] - cs[prev]; prev = epos[k]
    y = _ema_loop(S, N)
    out = np.zeros(len(grid_ms)); has = ci >= 0
    if len(y):
        out[has] = y[np.clip(ci[has], 0, len(y) - 1)]
    return out


def _derived(sd, l):
    """The builder's per-book-event book_imbalance + ofi, recomputed from the raw session."""
    from boba.dataset_v2.raw import aggregate_same_ns_bbo, compute_ofi_events
    bt, bid, ask, bq, aq = aggregate_same_ns_bbo(
        sd.listing_book_t[l], sd.listing_book_bid[l], sd.listing_book_ask[l],
        sd.listing_book_bid_qty[l], sd.listing_book_ask_qty[l])
    bi = (bq - aq) / np.maximum(bq + aq, 1e-30)
    return bt, bi, compute_ofi_events(bid, ask, bq, aq)


def _derived_all(sd, l, mp_ref=0.0, wide_thr=1.0e-4):
    """Every per-book-event quantity the builder derives, recomputed from the raw session — the
    oracle inputs for all the trade-clock book-event families."""
    from boba.dataset_v2.raw import (aggregate_same_ns_bbo, compute_microprice,
                                     compute_ofi_events, compute_abs_log_ret)
    bt, bid, ask, bq, aq = aggregate_same_ns_bbo(
        sd.listing_book_t[l], sd.listing_book_bid[l], sd.listing_book_ask[l],
        sd.listing_book_bid_qty[l], sd.listing_book_ask_qty[l])
    eps = 1e-30
    mp = compute_microprice(bid, ask, bq, aq)
    sw = (ask - bid) / np.maximum(mp, eps)
    return {
        "bt": bt,
        "book_imbalance": (bq - aq) / np.maximum(bq + aq, eps),
        "book_depth": bq + aq,
        "spread_wide_flag": (sw > wide_thr).astype(np.float64),
        "microprice_centered": np.where(mp > eps, mp - mp_ref, 0.0),
        "ofi": compute_ofi_events(bid, ask, bq, aq),
        "abs_log_ret": compute_abs_log_ret(mp),
    }


# family → (oracle mode, _derived_all key, square)
_TC_ORACLE = {
    "ema_book_imbalance":         ("level", "book_imbalance",      False),
    "ema_book_imbalance_sq":      ("level", "book_imbalance",      True),
    "ema_book_depth":             ("level", "book_depth",          False),
    "ema_book_depth_sq":          ("level", "book_depth",          True),
    "ema_spread_wide_flag":       ("level", "spread_wide_flag",    False),
    "ema_microprice_centered":    ("level", "microprice_centered", False),
    "ema_microprice_centered_sq": ("level", "microprice_centered", True),
    "ema_ofi":                    ("flow",  "ofi",                 False),
    "ema_ofi_sq":                 ("flow",  "ofi",                 True),
    "ema_abs_log_ret":            ("flow",  "abs_log_ret",         False),
}


def _session_same_book(seed, lo_ms=0, hi_ms=4000):
    """bin & byb share the IDENTICAL book (bid/ask/qtys → identical book_imbalance and ofi) but
    trade at different rates — so a shared clock must line their trade-clock book EMAs up."""
    from boba.dataset_v2.session_data import SessionData
    rng = np.random.default_rng(seed)
    bt = np.arange(lo_ms, hi_ms, dtype=np.int64) * MS
    n = len(bt)
    mid = 100.0 + np.cumsum(rng.choice([-1.0, 0.0, 1.0], n) * 0.001)
    bid, ask = mid - 0.01, mid + 0.01
    bq = np.abs(rng.standard_normal(n)) + 1.0       # non-trivial, varying ⇒ real imbalance/ofi
    aq = np.abs(rng.standard_normal(n)) + 1.0
    bf = {f: {} for f in ("listing_book_t", "listing_book_bid", "listing_book_ask",
                          "listing_book_bid_qty", "listing_book_ask_qty", "listing_feed_latency_excess_ns")}
    tf = {f: {} for f in ("trade_ts", "trade_exchange_ts", "trade_prc", "trade_qty", "trade_dir")}
    for l, every in (("bin", 3), ("byb", 11)):
        bf["listing_book_t"][l] = bt
        bf["listing_book_bid"][l] = bid.copy(); bf["listing_book_ask"][l] = ask.copy()
        bf["listing_book_bid_qty"][l] = bq.copy(); bf["listing_book_ask_qty"][l] = aq.copy()
        bf["listing_feed_latency_excess_ns"][l] = np.zeros(n, np.int64)
        tt = bt[::every]
        tf["trade_ts"][l] = tt; tf["trade_exchange_ts"][l] = tt
        tf["trade_prc"][l] = mid[::every]; tf["trade_qty"][l] = np.full(len(tt), 1.0)
        tf["trade_dir"][l] = rng.choice([-1.0, 1.0], len(tt))
    z = np.zeros(n, np.int64)
    all_rx = np.sort(np.concatenate([bt, bt, tf["trade_ts"]["bin"], tf["trade_ts"]["byb"]]))
    return SessionData(target_listing="bin", all_rx=all_rx, book_t=bt, book_bid=bid, book_ask=ask,
                       book_mid=mid, feed_latency_raw_ns=z, feed_latency_excess_ns=z, **bf, **tf)


class TestBookEventClock:
    def test_all_families_match_raw_oracle(self):
        # EVERY trade-clock book-event family vs its independent raw-data oracle — on its OWN clock
        # AND on a FOREIGN clock (byb's quantity over bin's trades). The wiring gate for the whole
        # set: level/flow, _sq, centered microprice, own + cross-exchange clock.
        from boba.dataset_v2 import col
        from tests.dataset_v2.test_engine import _synth_block
        sd = _synth_block(0, 4000 * MS, seed=7)
        ref_mp = {"bin": 100.0, "byb": 100.0}
        N = 15
        cols = (tuple(col("{LISTING}_" + f + "_{N}t", LISTING=["bin"], N=[N]) for f in _TC_ORACLE)
                + tuple(col("{LISTING}_" + f + "_{N}t@{CLOCK}", LISTING=["byb"], N=[N], CLOCK="trades_bin")
                        for f in _TC_ORACLE))
        out = _build(sd, cols, microprice_ref=ref_mp)
        grid = out.timestamp_ms.astype(np.int64)
        d = {l: _derived_all(sd, l, mp_ref=ref_mp[l], wide_thr=1.0e-4) for l in ("bin", "byb")}

        def oracle(fam, L):                                   # clock is always bin's trades here
            mode, key, square = _TC_ORACLE[fam]
            if mode == "level":
                return level_ema_ref(d[L]["bt"], d[L][key], sd.trade_ts["bin"], grid, N, square=square)
            return flow_ema_ref(d[L]["bt"], d[L][key] ** 2 if square else d[L][key], sd.trade_ts["bin"], grid, N)

        for fam in _TC_ORACLE:
            own = out.x[:, out.column_names.index(f"bin_{fam}_{N}t")]                  # bin on its own clock
            np.testing.assert_allclose(own, oracle(fam, "bin").astype(np.float32),
                                       rtol=1e-4, atol=1e-6, err_msg=f"{fam} (own clock)")
            foreign = out.x[:, out.column_names.index(f"byb_{fam}_{N}t@trades_bin")]   # byb on bin's clock
            np.testing.assert_allclose(foreign, oracle(fam, "byb").astype(np.float32),
                                       rtol=1e-4, atol=1e-6, err_msg=f"{fam} (foreign clock)")

    def test_level_matches_raw_oracle(self):              # book_imbalance = LEVEL (sample at trades)
        from boba.dataset_v2 import col
        from tests.dataset_v2.test_engine import _synth_block
        sd = _synth_block(0, 4000 * MS, seed=7)
        out = _build(sd, (
            col("{LISTING}_ema_book_imbalance_{N}t", LISTING=["bin"], N=[10, 50]),
            col("{LISTING}_ema_book_imbalance_{N}t@{CLOCK}", LISTING=["byb"], N=[10], CLOCK="trades_bin"),
        ))
        grid = out.timestamp_ms.astype(np.int64)
        for name, L, C, N in (("bin_ema_book_imbalance_10t", "bin", "bin", 10),
                              ("bin_ema_book_imbalance_50t", "bin", "bin", 50),
                              ("byb_ema_book_imbalance_10t@trades_bin", "byb", "bin", 10)):
            bt, bi, _ = _derived(sd, L)
            ref = level_ema_ref(bt, bi, sd.trade_ts[C], grid, N)
            got = out.x[:, out.column_names.index(name)]
            np.testing.assert_allclose(got, ref.astype(np.float32), rtol=1e-5, atol=1e-7, err_msg=name)

    def test_flow_matches_raw_oracle(self):              # ofi = FLOW (sum per clock interval)
        from boba.dataset_v2 import col
        from tests.dataset_v2.test_engine import _synth_block
        sd = _synth_block(0, 4000 * MS, seed=7)
        out = _build(sd, (
            col("{LISTING}_ema_ofi_{N}t", LISTING=["bin"], N=[10, 50]),
            col("{LISTING}_ema_ofi_{N}t@{CLOCK}", LISTING=["byb"], N=[10], CLOCK="trades_bin"),
        ))
        grid = out.timestamp_ms.astype(np.int64)
        for name, L, C, N in (("bin_ema_ofi_10t", "bin", "bin", 10),
                              ("bin_ema_ofi_50t", "bin", "bin", 50),
                              ("byb_ema_ofi_10t@trades_bin", "byb", "bin", 10)):
            bt, _, ofi = _derived(sd, L)
            ref = flow_ema_ref(bt, ofi, sd.trade_ts[C], grid, N)
            got = out.x[:, out.column_names.index(name)]
            np.testing.assert_allclose(got, ref.astype(np.float32), rtol=1e-5, atol=1e-7, err_msg=name)

    def test_own_clock_equals_no_clock(self):
        from boba.dataset_v2 import col
        from tests.dataset_v2.test_engine import _synth_block
        sd = _synth_block(0, 4000 * MS, seed=7)
        out = _build(sd, (
            col("{LISTING}_ema_book_imbalance_{N}t", LISTING=["bin"], N=[20]),
            col("{LISTING}_ema_book_imbalance_{N}t@{CLOCK}", LISTING=["bin"], N=[20], CLOCK="trades_bin"),
            col("{LISTING}_ema_ofi_{N}t", LISTING=["bin"], N=[20]),
            col("{LISTING}_ema_ofi_{N}t@{CLOCK}", LISTING=["bin"], N=[20], CLOCK="trades_bin"),
        ))
        for fam in ("ema_book_imbalance", "ema_ofi"):
            bare = out.x[:, out.column_names.index(f"bin_{fam}_20t")]
            self_clk = out.x[:, out.column_names.index(f"bin_{fam}_20t@trades_bin")]
            np.testing.assert_array_equal(bare, self_clk)

    def test_shared_clock_aligns_venues(self):
        from boba.dataset_v2 import col
        sd = _session_same_book(seed=4)                  # identical book on both venues, different trade rates
        for fam in ("ema_book_imbalance", "ema_ofi"):
            for N in (5, 20, 100):
                out = _build(sd, (col("{LISTING}_" + fam + "_{N}t@{CLOCK}",
                                       LISTING=["bin", "byb"], N=[N], CLOCK="trades_bin"),))
                a = out.x[:, out.column_names.index(f"bin_{fam}_{N}t@trades_bin")]
                b = out.x[:, out.column_names.index(f"byb_{fam}_{N}t@trades_bin")]
                np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9,
                                           err_msg=f"{fam} venues not aligned on shared clock, N={N}")
            # on their OWN different-rate clocks they must NOT line up
            out = _build(sd, (col("{LISTING}_" + fam + "_{N}t", LISTING=["bin", "byb"], N=[20]),))
            assert not np.allclose(out.x[:, out.column_names.index(f"bin_{fam}_20t")],
                                   out.x[:, out.column_names.index(f"byb_{fam}_20t")], rtol=1e-2)

    def test_chunk_independent(self, tmp_path):
        import dataclasses
        from boba.dataset_v2 import DatasetRawConfig, col
        from boba.dataset_v2.raw import build_features_raw
        from boba.dataset_v2.engine import Block, build_chunked, concat_sessions
        from boba.dataset_v2.cache import read_blocks
        from boba.dataset_v2.driver import tail_window_ns_for
        from tests.dataset_v2.test_engine import _synth_block, LISTINGS, TARGET, SPAN_MS
        blocks = [Block(f"b{i}", i * SPAN_MS, (i + 1) * SPAN_MS,
                        _synth_block(i * SPAN_MS * MS, (i + 1) * SPAN_MS * MS, seed=i)) for i in range(3)]
        cfg = DatasetRawConfig(
            columns=(col("{LISTING}_ema_book_imbalance_{N}t", LISTING=["bin"], N=[10, 50]),
                     col("{LISTING}_ema_ofi_{N}t@{CLOCK}", LISTING=["byb"], N=[20], CLOCK="trades_bin")),
            listings=tuple(LISTINGS), target_listing=TARGET,
            warmup_ms=0, horizon_ms=50.0, event_mask="none", wide_threshold={l: 1e-4 for l in LISTINGS})
        sd = concat_sessions([b.session for b in blocks])
        names = list(build_features_raw(sd, list(LISTINGS), cfg).column_names)
        build_chunked(blocks, cfg, tmp_path, "gh", tail_window_ns=tail_window_ns_for(cfg), capacity=4000)
        v2 = read_blocks(tmp_path, "gh", ["b0", "b1", "b2"], names)
        cont = build_features_raw(sd, list(LISTINGS), dataclasses.replace(cfg, warmup_ms=0))
        idx = {int(t): i for i, t in enumerate(cont.timestamp_ms)}
        rows = [idx[int(t)] for t in v2.timestamp_ms]
        assert len(rows) > 1000
        np.testing.assert_allclose(v2.x, cont.x[rows], rtol=1e-5, atol=1e-6)


def _trade_q(sd, l, family):
    """A listing's per-trade quantity (oracle input for a trade-flow `_t` family) + its trade times."""
    from boba.dataset_v2.raw import aggregate_same_ns_trade
    tt, prc, qty, d = aggregate_same_ns_trade(sd.trade_ts[l], sd.trade_prc[l], sd.trade_qty[l], sd.trade_dir[l])
    is_buy = d > 0; val = qty * prc
    table = {"ema_buy_trade_qty_t": np.where(is_buy, qty, 0.0),
             "ema_sell_trade_qty_t": np.where(~is_buy, qty, 0.0),
             "ema_buy_trade_value_t": np.where(is_buy, val, 0.0),
             "ema_sell_trade_value_t": np.where(~is_buy, val, 0.0)}
    if family in table:
        return tt, table[family]
    cov = np.zeros(len(prc))                                  # ema_trade_serial_cov_t
    if len(prc) >= 3:
        dp = np.diff(prc); cov[2:] = dp[1:] * dp[:-1]
    return tt, cov


_TRADE_FLOW = ("ema_buy_trade_qty", "ema_sell_trade_qty", "ema_buy_trade_value",
               "ema_sell_trade_value", "ema_trade_serial_cov")


class TestTradeFlowClock:
    def test_own_clock_byte_identical_to_section3(self):
        # the bare own-clock column is produced by the OLD Section-3 path; @trades_self goes through
        # the new clock path. They must be byte-identical (one trade per own-clock interval).
        from boba.dataset_v2 import col
        from tests.dataset_v2.test_engine import _synth_block
        sd = _synth_block(0, 4000 * MS, seed=7)
        cols = []
        for f in _TRADE_FLOW:
            cols += [col("{LISTING}_" + f + "_{N}t", LISTING=["bin"], N=[20]),
                     col("{LISTING}_" + f + "_{N}t@{CLOCK}", LISTING=["bin"], N=[20], CLOCK="trades_bin")]
        out = _build(sd, tuple(cols))
        for f in _TRADE_FLOW:
            bare = out.x[:, out.column_names.index(f"bin_{f}_20t")]
            self_clk = out.x[:, out.column_names.index(f"bin_{f}_20t@trades_bin")]
            np.testing.assert_array_equal(bare, self_clk, err_msg=f)

    def test_foreign_clock_matches_oracle(self):
        # byb's trade flow summed over bin's trade intervals vs the raw-data oracle.
        from boba.dataset_v2 import col
        sd = _session_same_book(seed=4)                              # bin every 3ms, byb every 11ms
        out = _build(sd, tuple(col("{LISTING}_" + f + "_{N}t@{CLOCK}", LISTING=["byb"], N=[15],
                                    CLOCK="trades_bin") for f in _TRADE_FLOW))
        grid = out.timestamp_ms.astype(np.int64)
        bin_tr, _ = _trade_q(sd, "bin", "ema_buy_trade_value_t")     # bin's trade times = the clock
        for f in _TRADE_FLOW:
            tt, q = _trade_q(sd, "byb", f + "_t")
            ref = flow_ema_ref(tt, q, bin_tr, grid, 15)
            got = out.x[:, out.column_names.index(f"byb_{f}_15t@trades_bin")]
            np.testing.assert_allclose(got, ref.astype(np.float32), rtol=1e-4, atol=1e-7, err_msg=f)


# ════════════════════════════════════════════════════════════════════════════════════════════
# Real block — library vs the dead-simple oracle on actual data (arbitrary-ns trades, which the
# synthetic exact-ms blocks cannot exercise). Skipped when no DATA_DIR is configured.
# ════════════════════════════════════════════════════════════════════════════════════════════
import pytest
from boba.io import DATA_DIR, list_blocks

_ETH = ("bin_eth_usdt_p", "byb_eth_usdt_p", "okx_eth_usdt_p")


def _have_eth_blocks():
    if DATA_DIR is None:
        return False
    try:
        return all(list_blocks(l, "front_levels") for l in _ETH)
    except Exception:
        return False


@pytest.mark.skipif(not _have_eth_blocks(), reason="no real ETH-perp block data (DATA_DIR)")
class TestRealBlock:
    def test_library_matches_oracle_on_real_block(self):
        from boba.dataset_v2 import DatasetRawConfig, col
        from boba.dataset_v2.driver import _load_session
        from boba.dataset_v2.engine import slice_session
        from boba.dataset_v2.raw import (build_features_raw, _grid_bounds_ms,
                                         aggregate_same_ns_bbo, aggregate_same_ns_trade, compute_ofi_events)
        TGT = "byb_eth_usdt_p"
        base = DatasetRawConfig(columns=(col("{LISTING}_microprice", LISTING=TGT),),
                                listings=_ETH, target_listing=TGT)
        full = _load_session(base, list_blocks(TGT)[1])
        t0 = min(full.listing_book_t[l][0] for l in _ETH)
        sd = slice_session(full, t0 + 5_000_000_000, t0 + 35_000_000_000)        # 30s window
        cfg = DatasetRawConfig(columns=(
            col("{LISTING}_vol_{N}t", LISTING=[TGT], N=[50, 200]),
            col("{LISTING}_vol_{N}t@{CLOCK}", LISTING=[TGT], N=[50], CLOCK="trades_bin_eth_usdt_p"),
            col("{LISTING}_ema_book_imbalance_{N}t", LISTING=[TGT], N=[50]),
            col("{LISTING}_ema_ofi_{N}t@{CLOCK}", LISTING=[TGT], N=[50], CLOCK="trades_bin_eth_usdt_p"),
            col("{LISTING}_ema_buy_trade_value_{N}t@{CLOCK}", LISTING=[TGT], N=[50], CLOCK="trades_bin_eth_usdt_p")),
            listings=_ETH, target_listing=TGT, warmup_ms=0, horizon_ms=0.0, event_mask="trade",
            microprice_ref={l: 0.0 for l in _ETH}, wide_threshold={l: 1e-4 for l in _ETH})
        out = build_features_raw(sd, list(_ETH), cfg)
        assert len(out.timestamp_ms) > 200                                       # real rows present

        ts, te = (min(sd.listing_book_t[l][0] for l in _ETH), max(sd.listing_book_t[l][-1] for l in _ETH))
        gs, ge = _grid_bounds_ms(ts, te, 0, 0.0)
        fg = np.arange(gs, ge + 1, dtype=np.int64)
        rows = [{int(t): i for i, t in enumerate(fg)}[int(t)] for t in out.timestamp_ms]

        bt, bid, ask, bq, aq = aggregate_same_ns_bbo(
            sd.listing_book_t[TGT], sd.listing_book_bid[TGT], sd.listing_book_ask[TGT],
            sd.listing_book_bid_qty[TGT], sd.listing_book_ask_qty[TGT])
        mid = (bid + ask) / 2.0
        bi = (bq - aq) / np.maximum(bq + aq, 1e-30)
        ofi = compute_ofi_events(bid, ask, bq, aq)
        byb_t, byb_p, byb_q, byb_d = aggregate_same_ns_trade(
            sd.trade_ts[TGT], sd.trade_prc[TGT], sd.trade_qty[TGT], sd.trade_dir[TGT])
        byb_tr = byb_t
        byb_buyval = np.where(byb_d > 0, byb_q * byb_p, 0.0)                  # byb's per-trade buy value
        bin_tr = aggregate_same_ns_trade(sd.trade_ts["bin_eth_usdt_p"], sd.trade_prc["bin_eth_usdt_p"],
                                         sd.trade_qty["bin_eth_usdt_p"], sd.trade_dir["bin_eth_usdt_p"])[0]
        cases = {
            f"{TGT}_vol_50t":                                       vol_ref(bt, mid, byb_tr, fg, 50),
            f"{TGT}_vol_200t":                                      vol_ref(bt, mid, byb_tr, fg, 200),
            f"{TGT}_vol_50t@trades_bin_eth_usdt_p":                 vol_ref(bt, mid, bin_tr, fg, 50),
            f"{TGT}_ema_book_imbalance_50t":                        level_ema_ref(bt, bi, byb_tr, fg, 50),
            f"{TGT}_ema_ofi_50t@trades_bin_eth_usdt_p":             flow_ema_ref(bt, ofi, bin_tr, fg, 50),
            f"{TGT}_ema_buy_trade_value_50t@trades_bin_eth_usdt_p": flow_ema_ref(byb_t, byb_buyval, bin_tr, fg, 50),
        }
        for name, ref in cases.items():
            got = out.x[:, out.column_names.index(name)]
            np.testing.assert_allclose(got, ref[rows].astype(np.float32), rtol=1e-4, atol=1e-7, err_msg=name)
