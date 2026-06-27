"""Self-contained test suite for `boba.features.rate_momentum` — `log(λ_ev_fast / λ_ev_slow)`, the target's
trade-rate acceleration (see the module docstring). ONE target-based leg, `params = (n_fast, n_slow)`, EVEN
under the tape reflection (`mirror = identity`). The AUTHORING validation trio + streaming coverage, on
synthetic data with NO shared code with production:

  - `test_vectorized_matches_oracle`     vectorized vs an INDEPENDENT λ_ev event-loop oracle (incl. a 1-span).
  - `test_even_under_reflection`         reflecting the mid leaves it unchanged (λ_ev counts sign-free moves).
  - `test_streaming_matches_vectorized`  `LiveRateMomentum` vs the sampled vectorized via `parity_check`.
  - `test_real_block_parity`             streaming-vs-vectorized on a real block (DATA_DIR-gated).
"""
import numpy as np
import pytest

import boba.io as io
from boba.features import base
import boba.features.rate_momentum  # noqa: F401  (registers the spec)
from boba.features.base import (
    Config, FrontLevels, ListingRaw, ListingShared, RawData, Series, SharedData, Trade)
from boba.features.shared import build_shared_data
from boba.research.screening import RawEventStream, ScreeningContext, parity_check

_YARD = 25
KEY = "byb"


# --------------------------------------------------------------------------------------------------
# synthetic standalone inputs (target venue only; its trades are the decay clock) — own builders
# --------------------------------------------------------------------------------------------------
def _market(seed=0, n=2600):
    rng = np.random.default_rng(seed)
    rx_b = (np.arange(1, n + 1) * 10).astype(np.int64)               # a book row every 10 ns
    mid = 100.0 * np.exp(np.cumsum(np.where(rng.random(n) < 0.6, rng.standard_normal(n) * 8e-5, 0.0)))
    rx_b[100:106] = rx_b[100]                                        # same-ts burst -> only the final mid counts
    rx_t = (np.cumsum(rng.integers(7, 19, (3 * n) // 4)) + 3).astype(np.int64)   # VARIABLE inter-tick gaps
    return rx_b, mid, rx_t


def _inputs(rx_b, mid, rx_t, coin="x"):
    tgt = f"{KEY}_{coin}"
    front = FrontLevels(rx_b, rx_b, mid - 0.01, np.ones(len(rx_b)), mid + 0.01, np.ones(len(rx_b)))
    trade = Trade(rx_t, rx_t, np.full(len(rx_t), 100.0), np.zeros(len(rx_t)), np.ones(len(rx_t)))
    raw = RawData(listings={tgt: ListingRaw(front_levels=front, trade=trade)})
    config = Config(tgt, (), coin, {tgt: "front_levels"}, yardstick_span=_YARD)
    return raw, build_shared_data(raw, config), config


def _raw_events(rx_b, mid, rx_t, coin="x") -> RawEventStream:
    tgt = f"{KEY}_{coin}"
    nb, nt = len(rx_b), len(rx_t)
    rx = np.concatenate([rx_b, rx_t])
    kind = np.concatenate([np.zeros(nb, np.int8), np.ones(nt, np.int8)])
    lid = np.zeros(nb + nt, np.int8)
    t = rx.copy()
    a = np.concatenate([mid - 0.01, np.full(nt, 100.0)])
    b = np.concatenate([mid + 0.01, np.zeros(nt)])
    c = np.ones(nb + nt)
    d = np.concatenate([np.ones(nb), np.full(nt, np.nan)])
    order = np.lexsort((kind, rx))
    return RawEventStream(rx[order], kind[order], lid[order], t[order],
                          a[order], b[order], c[order], d[order], (tgt,))


def _ctx(rx_b, mid, rx_t, anchor_ts) -> ScreeningContext:
    raw, shared, config = _inputs(rx_b, mid, rx_t)
    return ScreeningContext(
        block="syn", coin="x", target=config.target_listing, sources=(), horizon_ns=0,
        yardstick_span=_YARD, mid_stream={}, merged_ts=shared.clock, anchor_ts=anchor_ts,
        sigma_at_anchor=np.empty(0), lam_at_anchor=np.empty(0),
        price_target=np.empty(0), rate_target=np.empty(0), base=[], vol_level=np.empty(0),
        rate_level=np.empty(0), vol_regime=np.empty(0), raw_events=_raw_events(rx_b, mid, rx_t),
        raw_data=raw, shared_data=shared, config=config)


# --------------------------------------------------------------------------------------------------
# INDEPENDENT, dead-simple oracle — λ_ev by an explicit loop (move-count flow / inter-tick-time flow),
# then log(fast/slow). No production code.
# --------------------------------------------------------------------------------------------------
def _lambda_oracle(rx_b, mid, trade_ts, grid, span):
    """λ_ev at each `grid`: α=2/(span+1); W += α on a target mid CHANGE (final mid per ts) and W *= (1−α)
    on a trade tick; the inter-tick-time EMA `dt = (1−α)·dt + α·Δt_seconds` updates on each tick. λ = W/dt
    once a gap has been seen (else NaN). Read at the last event <= grid."""
    a = 2.0 / (span + 1.0)
    rx, m = np.asarray(rx_b), np.asarray(mid)
    keep = np.concatenate([rx[1:] != rx[:-1], [True]])
    rx, m = rx[keep], m[keep]
    lm = np.log(m)
    move = {}
    prev = None
    for ts, x in zip(rx.tolist(), lm.tolist()):
        if prev is not None and x != prev:
            move[ts] = 1
        prev = x
    trades = set(int(t) for t in trade_ts)
    all_ts = sorted(set(move) | trades)
    W = dt = 0.0
    last_tick = None
    ts_arr, vals = [], []
    for ts in all_ts:
        if ts in move:
            W += a
        if ts in trades:
            gap = 0.0 if last_tick is None else (ts - last_tick) / 1e9
            dt = (1.0 - a) * dt + a * gap
            last_tick = ts
            W *= (1.0 - a)
        ts_arr.append(ts); vals.append(W / dt if dt > 0.0 else np.nan)
    ts_arr, vals = np.array(ts_arr), np.array(vals)
    idx = np.searchsorted(ts_arr, grid, "right") - 1
    return np.where(idx < 0, np.nan, vals[np.clip(idx, 0, len(vals) - 1)])


def _rate_momentum_oracle(rx_b, mid, trade_ts, grid, n_fast, n_slow):
    lf = _lambda_oracle(rx_b, mid, trade_ts, grid, n_fast)
    ls = _lambda_oracle(rx_b, mid, trade_ts, grid, n_slow)
    ok = np.isfinite(lf) & np.isfinite(ls) & (lf > 0) & (ls > 0)
    return np.where(ok, np.log(np.where(ok, lf, 1.0) / np.where(ok, ls, 1.0)), np.nan)


def test_vectorized_matches_oracle():
    rx_b, mid, rx_t = _market(seed=1)
    raw, shared, config = _inputs(rx_b, mid, rx_t)
    spec = base.get("rate_momentum")
    for params in ((1, 25), (5, 50), (10, 200)):                   # incl. a 1-span fast leg
        got = spec.vectorized(raw, shared, config, params)[KEY]
        ref = _rate_momentum_oracle(rx_b, mid, rx_t, shared.event_ts, *params)
        assert not np.any(np.isinf(got))                           # never a stray inf — log(0) is guarded
        np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))   # consistent NaN both builds agree on
        ok = np.isfinite(got) & np.isfinite(ref)
        assert ok.sum() > 100
        # log AMPLIFIES the λ-EMA float-accumulation gap between the two independent builds when λ_fast is
        # tiny, so the log-ratio is accurate to ~1e-6, not 1e-9.
        np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------------------------------
# EVEN under the tape reflection: λ_ev counts mid CHANGES (sign-free) -> unchanged -> feature unchanged.
# --------------------------------------------------------------------------------------------------
def _mirror_shared(shared: SharedData, c=100.0) -> SharedData:
    listings = {l: ListingShared(mid=Series(np.asarray(ls.mid.rx), c * c / np.asarray(ls.mid.value)))
                for l, ls in shared.listings.items()}
    return SharedData(event_ts=shared.event_ts, clock=shared.clock, vol_yardstick=shared.vol_yardstick,
                      rate_yardstick=shared.rate_yardstick, listings=listings)


def test_even_under_reflection():
    rx_b, mid, rx_t = _market(seed=4)
    raw, shared, config = _inputs(rx_b, mid, rx_t)
    spec = base.get("rate_momentum")
    feat = spec.vectorized(raw, shared, config, (10, 100))[KEY]
    refl = spec.vectorized(raw, _mirror_shared(shared), config, (10, 100))[KEY]
    lhs = spec.mirror(feat)                                          # mirror = identity (EVEN)
    ok = np.isfinite(lhs) & np.isfinite(refl)
    assert ok.sum() > 100
    np.testing.assert_allclose(lhs[ok], refl[ok], rtol=1e-6, atol=1e-9)


def _span1_anchors(rx_b, mid, rx_t):
    """Union of (target mid-move timestamps that are NOT trade ticks) + interior trade ticks. At a move-only
    anchor a span=1 λ_ev is ALIVE (the fresh move's count, not yet decayed) -> span=1 is DEFINED there,
    giving the parity sweep real span=1 coverage (a trade-tick anchor decays span=1's count to 0 -> NaN)."""
    rx, m = np.asarray(rx_b), np.asarray(mid)
    keep = np.concatenate([rx[1:] != rx[:-1], [True]]); rx, m = rx[keep], m[keep]
    moved = np.concatenate([[False], np.diff(np.log(m)) != 0])
    trset = set(int(t) for t in rx_t)
    move_only = np.unique([int(t) for t in rx[moved] if int(t) not in trset])
    move_only = move_only[(move_only > rx_t[120]) & (move_only < rx_t[-120])]
    return np.unique(np.concatenate([move_only, rx_t[150:-150:5]]))


def test_streaming_matches_vectorized():
    rx_b, mid, rx_t = _market(seed=7)
    anchor_ts = _span1_anchors(rx_b, mid, rx_t)              # move-only anchors keep a span=1 λ_ev alive
    ctx = _ctx(rx_b, mid, rx_t, anchor_ts)
    # tol 1e-8 (not 1e-9): at span=1 λ_fast is tiny and the log amplifies the two builds' summation-order
    # float gap; the realistic spans match to ~1e-13. Still orders below any real bug.
    rep = parity_check(ctx, base.get("rate_momentum"), [(1, 25), (5, 50), (50, 500)], n_grid=len(anchor_ts), tol=1e-8)
    assert rep.passed, str(rep)


@pytest.mark.skipif(getattr(io, "DATA_DIR", None) is None, reason="no DATA_DIR configured")
def test_real_block_parity():
    from boba.research.screening import build_context
    ctx = build_context(hours=2)
    rep = parity_check(ctx, base.get("rate_momentum"), [(100, 1000), (1000, 10000)], tol=1e-6)
    assert rep.passed, str(rep)
