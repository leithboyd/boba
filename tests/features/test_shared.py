"""Tests for boba.features.shared — the standalone `shared_data` precompute (and the `flow_at` primitive).

The decay clock, the event/output grid, and the per-listing mid have dead-simple independent oracles;
σ_ev and `flow_at` are checked against explicit one-event-at-a-time reference loops (no shared code with
production); the mid-policy (merged_levels vs front_levels) selection is checked directly. A real-block
test (skipif no DATA_DIR) builds raw_data from a block and asserts the clock, the yardsticks (sampled
back onto the research anchor grid), and the mid reproduce the existing `build_context` pipeline —
proving the repackaging is faithful and the migration can proceed."""
import numpy as np
import pytest

import boba.io as io
from boba.features.base import Config, FrontLevels, ListingRaw, MergedLevels, RawData, Trade
from boba.features.shared import _ffill, build_shared_data, flow_at, hold_last


# --------------------------------------------------------------------------------------------------
# independent σ_ev oracle — explicit per-event loop (the VolYardstick reference, evaluated on a grid)
# --------------------------------------------------------------------------------------------------
def _sigma_over_grid(target_mid, clock, event_ts, span):
    """σ_ev at each event_ts by a dead-simple loop: inject `a·(Δlog)²` on a target mid CHANGE, decay by
    `(1-a)` on a trade tick, read `sqrt(E/W)` after each event. No production code."""
    rx, mid = target_mid
    log_at = dict(zip(rx.tolist(), np.log(mid).tolist()))     # last wins (same-rx collapse to the last)
    ticks = set(clock.tolist())
    a = 2.0 / (span + 1.0)
    E = W = 0.0
    prev = None
    out = []
    for t in event_ts.tolist():
        if t in log_at:
            lm = log_at[t]
            if prev is not None and lm != prev:
                E += a * (lm - prev) ** 2
                W += a
            prev = lm
        if t in ticks:
            E *= (1.0 - a)
            W *= (1.0 - a)
        out.append((E / W) ** 0.5 if W > 0 else float("nan"))
    return np.array(out)


def _lambda_over_grid(target_mid, clock, event_ts, span):
    """λ_ev at each event_ts by a dead-simple loop INDEPENDENT of `_yardsticks` (no flow_at / lfilter): a
    move-count EMA `W` (W += a on a target mid CHANGE, W *= (1-a) on a trade tick) over an inter-tick-time
    EMA `dt` (on each tick, dt = (1-a)·dt + a·Δt_seconds). λ_ev = W / dt once a gap has been seen (else NaN).
    Built from the written definition 'target moves per second', not from the production code."""
    rx, mid = target_mid
    log_at = dict(zip(rx.tolist(), np.log(mid).tolist()))     # last wins (same-rx collapse to the last)
    ticks = set(clock.tolist())
    a = 2.0 / (span + 1.0)
    W = dt = 0.0
    prev = None
    last_tick = None
    out = []
    for t in event_ts.tolist():
        if t in log_at:
            lm = log_at[t]
            if prev is not None and lm != prev:
                W += a                                        # one move injected (count flow)
            prev = lm
        if t in ticks:
            gap = 0.0 if last_tick is None else (t - last_tick) / 1e9
            dt = (1.0 - a) * dt + a * gap                     # inter-tick-time EMA
            last_tick = t
            W *= (1.0 - a)                                    # decay the count flow on the tick
        out.append(W / dt if dt > 0.0 else float("nan"))
    return np.array(out)


def _ones(n):
    return np.ones(n)


# --------------------------------------------------------------------------------------------------
# flow_at — the shared sparse-flow primitive, vs an explicit per-event reference loop
# --------------------------------------------------------------------------------------------------
def test_flow_at_matches_independent_oracle():
    rng = np.random.default_rng(5)
    span = 12
    a = 2.0 / (span + 1.0)
    k = np.arange(1, 600)
    clock = (k * 10).astype(np.int64)                    # ticks at 10,20,…  (disjoint from injects/reads)
    src_rx = (k * 10 + 3).astype(np.int64)               # injections at 13,23,…
    val = rng.standard_normal(len(src_rx))
    out_ts = (k[:-1] * 10 + 7).astype(np.int64)          # reads at 17,27,…  (after the inject, before next tick)
    got = flow_at(clock, src_rx, val, out_ts, span)

    # oracle: walk every event in time order; inject E += a·val, decay E *= (1-a) on a tick, read E.
    events = sorted([(t, 0, v) for t, v in zip(src_rx.tolist(), val.tolist())]
                    + [(t, 1, 0.0) for t in clock.tolist()]
                    + [(t, 2, 0.0) for t in out_ts.tolist()])
    E = 0.0
    seen = {}
    for t, kind, v in events:
        if kind == 0:
            E += a * v
        elif kind == 1:
            E *= (1.0 - a)
        else:
            seen[t] = E
    ref = np.array([seen[t] for t in out_ts.tolist()])
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


def test_flow_at_injection_landing_exactly_on_a_tick():
    """The real-block case the disjoint-timestamp test misses: an injection at the SAME ns as a clock tick.
    flow_at must commit it into that tick's decay (inject-then-decay in rx order) — checked vs the explicit
    loop, which sorts a coincident (inject, tick) as inject-before-tick. Synthetic exact-ms grids never hit
    this; real blocks (arbitrary-ns timing) do."""
    rng = np.random.default_rng(7)
    span = 8
    a = 2.0 / (span + 1.0)
    k = np.arange(1, 500)
    clock = (k * 10).astype(np.int64)
    src_rx = clock.copy()                                # injections land EXACTLY on the ticks
    val = rng.standard_normal(len(src_rx))
    out_ts = (k[:-1] * 10 + 4).astype(np.int64)          # reads between ticks
    got = flow_at(clock, src_rx, val, out_ts, span)
    events = sorted([(t, 0, v) for t, v in zip(src_rx.tolist(), val.tolist())]
                    + [(t, 1, 0.0) for t in clock.tolist()]
                    + [(t, 2, 0.0) for t in out_ts.tolist()])
    E = 0.0
    seen = {}
    for t, kind, v in events:
        if kind == 0:
            E += a * v
        elif kind == 1:
            E *= (1.0 - a)
        else:
            seen[t] = E
    ref = np.array([seen[t] for t in out_ts.tolist()])
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------------------------------
# hold_last — the model-facing NaN guard: carry the last finite value through any MID-STREAM NaN/inf,
# keep only the LEADING warm-up NaN. A bug here is invisible to parity AND the oracles (both run on the
# RAW event-grid output, before hold_last), so it needs its own direct test.
# --------------------------------------------------------------------------------------------------
def test_hold_last_holds_mid_stream_nan_and_inf_keeps_leading_warmup():
    v = np.array([np.nan, np.nan, 1.0, np.nan, 2.0, np.inf, 3.0, -np.inf])
    np.testing.assert_array_equal(hold_last(v), [np.nan, np.nan, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])


def test_hold_last_all_nan_stays_nan_and_all_finite_unchanged():
    np.testing.assert_array_equal(hold_last(np.array([np.nan, np.nan])), [np.nan, np.nan])   # never warmed up
    np.testing.assert_array_equal(hold_last(np.array([1.0, 2.0, 3.0])), [1.0, 2.0, 3.0])     # nothing to hold
    np.testing.assert_array_equal(hold_last(np.array([5.0, np.nan, np.nan])), [5.0, 5.0, 5.0])   # trailing held


# --------------------------------------------------------------------------------------------------
# mid policy — merged_levels when the venue carries it, else front_levels
# --------------------------------------------------------------------------------------------------
def test_build_shared_data_uses_merged_levels_mid_when_policy_is_merged():
    tgt = "byb_x"
    rx_fl = np.array([10, 20, 30], np.int64)
    rx_ml = np.array([15, 25], np.int64)
    rx_tr = np.array([12, 22], np.int64)
    raw = RawData(listings={tgt: ListingRaw(
        front_levels=FrontLevels(rx_fl, rx_fl - 1, np.array([100., 101, 102]), _ones(3),
                                 np.array([100.2, 101.2, 102.2]), _ones(3)),
        trade=Trade(rx_tr, rx_tr - 1, np.array([100.1, 101.1]), np.array([1., 0.]), _ones(2)),
        merged_levels=MergedLevels(rx_ml, rx_ml - 1, rx_ml - 1, np.array([200., 201.]), np.array([200.4, 201.4])),
    )})
    shared = build_shared_data(raw, Config(tgt, (), "x", {tgt: "merged_levels"}, yardstick_span=10))
    mrx, mmid = shared.listings[tgt].mid
    np.testing.assert_array_equal(mrx, rx_ml)                                  # mid from merged_levels, NOT front_levels
    np.testing.assert_allclose(mmid, [(200 + 200.4) / 2, (201 + 201.4) / 2])
    assert set(rx_ml.tolist()) <= set(shared.event_ts.tolist())               # merged rx are in the event grid
    np.testing.assert_array_equal(shared.clock, rx_tr)                         # clock = the trade rx


def test_build_shared_data_falls_back_to_front_levels_when_no_merged():
    tgt = "byb_x"
    rx_fl = np.array([10, 20, 30], np.int64)
    raw = RawData(listings={tgt: ListingRaw(
        front_levels=FrontLevels(rx_fl, rx_fl - 1, np.array([100., 101, 102]), _ones(3),
                                 np.array([100.2, 101.2, 102.2]), _ones(3)),
        trade=Trade(np.array([12], np.int64), np.array([11], np.int64), np.array([100.1]), np.array([1.]), _ones(1)),
        merged_levels=None,                                                   # policy says merged, but none present
    )})
    shared = build_shared_data(raw, Config(tgt, (), "x", {tgt: "merged_levels"}, yardstick_span=10))
    mrx, mmid = shared.listings[tgt].mid
    np.testing.assert_array_equal(mrx, rx_fl)                                  # fell back to front_levels
    np.testing.assert_allclose(mmid, [(100 + 100.2) / 2, (101 + 101.2) / 2, (102 + 102.2) / 2])


def test_build_shared_data_synthetic_clock_grid_mid_and_sigma():
    rng = np.random.default_rng(0)
    coin = "x"
    tgt, src = "byb_x", "bin_x"
    span = 25
    n = 2500
    base = np.arange(1, n + 1) * 100
    rx_tf = (base + 0).astype(np.int64)       # target front_levels  (distinct streams -> no ties)
    rx_sf = (base + 20).astype(np.int64)      # source front_levels
    rx_tt = (base + 40).astype(np.int64)      # target trades (a decay tick)
    rx_st = (base + 60).astype(np.int64)      # source trades

    moved = rng.random(n) < 0.5
    moved[0] = True
    mid_t = 100.0 * np.exp(np.cumsum(np.where(moved, rng.standard_normal(n) * 1e-4, 0.0)))
    mid_s = 200.0 * np.exp(np.cumsum(rng.standard_normal(n) * 1e-4))

    lat = 1_000_000  # 1 ms synthetic feed latency: exchange_time = rx - lat (unused by build_shared_data)
    raw = RawData(listings={
        tgt: ListingRaw(
            front_levels=FrontLevels(rx_tf, rx_tf - lat, mid_t - 0.05, _ones(n), mid_t + 0.05, _ones(n)),
            trade=Trade(rx_tt, rx_tt - lat, mid_t, (rng.random(n) < 0.5).astype(float), _ones(n)),
        ),
        src: ListingRaw(
            front_levels=FrontLevels(rx_sf, rx_sf - lat, mid_s - 0.05, _ones(n), mid_s + 0.05, _ones(n)),
            trade=Trade(rx_st, rx_st - lat, mid_s, (rng.random(n) < 0.5).astype(float), _ones(n)),
        ),
    })
    config = Config(target_listing=tgt, other_listings=(src,), coin=coin,
                    mid_stream={tgt: "front_levels", src: "front_levels"}, yardstick_span=span)
    shared = build_shared_data(raw, config)

    # decay clock = union of every listing's trade timestamps
    np.testing.assert_array_equal(shared.clock, np.union1d(rx_tt, rx_st))
    # event grid = every timestamp with ANY event (book or trade), any listing
    np.testing.assert_array_equal(shared.event_ts, np.unique(np.concatenate([rx_tf, rx_sf, rx_tt, rx_st])))
    # per-listing mid = (bid+ask)/2 on the policy stream
    rx0, m0 = shared.listings[tgt].mid
    np.testing.assert_array_equal(rx0, rx_tf)
    np.testing.assert_allclose(m0, mid_t, rtol=1e-12)

    # σ_ev vs the independent loop oracle, wherever the oracle is warmed up
    ref = _sigma_over_grid(shared.listings[tgt].mid, shared.clock, shared.event_ts, span)
    ok = np.isfinite(ref)
    assert ok.sum() > 1000
    np.testing.assert_allclose(shared.vol_yardstick[ok], ref[ok], rtol=1e-9, atol=1e-15)
    # λ_ev vs its own independent loop oracle (moves per second), wherever the oracle is warmed up
    lam_ref = _lambda_over_grid(shared.listings[tgt].mid, shared.clock, shared.event_ts, span)
    okl = np.isfinite(lam_ref)
    assert okl.sum() > 1000
    np.testing.assert_allclose(shared.rate_yardstick[okl], lam_ref[okl], rtol=1e-9, atol=1e-15)
    assert np.all(shared.rate_yardstick[okl] >= 0.0)         # a rate: 0 in a quiet window (ticks, no moves)


# --------------------------------------------------------------------------------------------------
# real block — build raw_data off disk and reproduce the existing build_context pipeline
# --------------------------------------------------------------------------------------------------
def _raw_data_from_block(block, coin, listings, mid_stream, cap):
    """Load raw_data (front_levels / trade / merged_levels per listing) from a block, capped to rx <= cap."""
    import polars as pl

    from boba.io import load_block

    def cut(stream):
        m = stream.rx <= cap
        return type(stream)(*(c[m] for c in stream))

    def npint(col):
        return col.cast(pl.Int64).to_numpy()

    out = {}
    for l in listings:
        fl = (load_block(block, l, "front_levels")
              .select("rx_time", "exchange_time", "bid_prc", "bid_qty", "ask_prc", "ask_qty")
              .drop_nulls(["rx_time", "bid_prc", "bid_qty", "ask_prc", "ask_qty"])
              .with_columns(pl.col("exchange_time").fill_null(pl.col("rx_time"))))
        front = FrontLevels(npint(fl["rx_time"]), npint(fl["exchange_time"]), fl["bid_prc"].to_numpy(),
                            fl["bid_qty"].to_numpy(), fl["ask_prc"].to_numpy(), fl["ask_qty"].to_numpy())
        td = (load_block(block, l, "trade").select("rx_time", "exchange_time", "prc", "qty", "aggressor")
              .filter((pl.col("prc") > 0) & (pl.col("qty") > 0))
              .with_columns(pl.col("exchange_time").fill_null(pl.col("rx_time"))))
        trade = Trade(npint(td["rx_time"]), npint(td["exchange_time"]), td["prc"].to_numpy(),
                      io._trade_lifts_ask(l, td["aggressor"].to_numpy()).astype(float), td["qty"].to_numpy())
        merged = None
        if mid_stream[l] == "merged_levels":
            ml = (load_block(block, l, "merged_levels")
                  .select("rx_time", "bid_prc", "ask_prc", "bid_exchange_time", "ask_exchange_time")
                  .drop_nulls(["rx_time", "bid_prc", "ask_prc"])
                  .with_columns(pl.col("bid_exchange_time").fill_null(pl.col("rx_time")),
                                pl.col("ask_exchange_time").fill_null(pl.col("rx_time"))))
            merged = MergedLevels(npint(ml["rx_time"]), npint(ml["bid_exchange_time"]),
                                  npint(ml["ask_exchange_time"]), ml["bid_prc"].to_numpy(), ml["ask_prc"].to_numpy())
        out[l] = ListingRaw(front_levels=cut(front), trade=cut(trade),
                            merged_levels=None if merged is None else cut(merged))
    return RawData(listings=out)


@pytest.mark.skipif(getattr(io, "DATA_DIR", None) is None, reason="needs DATA_DIR (real block)")
def test_build_shared_data_matches_build_context_on_real_block():
    from boba.research.screening import build_context

    coin = "eth_usdt_p"
    target = f"byb_{coin}"
    listings = (target, f"bin_{coin}", f"okx_{coin}")
    mid_stream = {target: "merged_levels", f"bin_{coin}": "front_levels", f"okx_{coin}": "merged_levels"}

    ctx = build_context(coin=coin, target=target, sources=("bin", "okx"), hours=2, grid_ms=50)
    cap = int(ctx.merged_ts[-1])                              # the last clock tick the context loaded
    raw = _raw_data_from_block(ctx.block, coin, listings, mid_stream, cap)
    config = Config(target_listing=target, other_listings=(f"bin_{coin}", f"okx_{coin}"),
                    coin=coin, mid_stream=mid_stream, yardstick_span=ctx.yardstick_span)
    shared = build_shared_data(raw, config)

    # the decay clock is exactly the context's shared trade clock
    np.testing.assert_array_equal(shared.clock, ctx.merged_ts)

    # the yardsticks, sampled (forward-filled) from the event grid back onto the research anchors, equal
    # the context's anchor-grid yardsticks — the feature output is piecewise-constant between events, so
    # "compute on the event grid then sample" == "compute directly at the anchor".
    sig = _ffill(shared.event_ts, shared.vol_yardstick, ctx.anchor_ts)
    lam = _ffill(shared.event_ts, shared.rate_yardstick, ctx.anchor_ts)
    ok = np.isfinite(ctx.sigma_at_anchor)
    assert ok.sum() > 100
    np.testing.assert_allclose(sig[ok], ctx.sigma_at_anchor[ok], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(lam[ok], ctx.lam_at_anchor[ok], rtol=1e-6, atol=1e-12)

    # the target's derived mid matches the context's loaded mid (on the capped overlap)
    rx0, m0 = shared.listings[target].mid
    rx1, m1 = ctx._mids["byb"]
    keep = rx1 <= cap
    np.testing.assert_array_equal(rx0, rx1[keep])
    np.testing.assert_allclose(m0, m1[keep], rtol=1e-12)
