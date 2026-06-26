"""Tests for trade-flow and price-momentum registered features.

Each vectorized build is checked against an independent event-loop oracle on synthetic data; the
streaming build is checked through the generic parity driver; the declared mirror is checked via the
input-reflection commutation invariant; and a real-block parity test runs when DATA_DIR is configured.
"""
import numpy as np
import pytest

import boba.io as io
from boba.features import base
import boba.features.flow_persistence       # noqa: F401  (registers)
import boba.features.price_momentum         # noqa: F401  (registers)
import boba.features.trade_flow_imbalance   # noqa: F401  (registers)
from boba.research.screening import RawEventStream, ScreeningContext, parity_check


def _synthetic_book(seed=0, n=2200):
    rng = np.random.default_rng(seed)
    rx = (np.arange(1, n + 1) * 10).astype(np.int64)
    mid = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 8e-5))
    hs = 0.005 + 0.005 * rng.random(n)
    bid, ask = mid - hs, mid + hs
    bq, aq = rng.uniform(1.0, 100.0, n), rng.uniform(1.0, 100.0, n)
    rx[100:106] = rx[100]       # same-timestamp level burst -> final mid only
    return rx, bid, bq, ask, aq


def _synthetic_trades(seed=0, n=1800):
    rng = np.random.default_rng(seed)
    rx = (np.arange(1, n + 1) * 13 + seed % 7).astype(np.int64)
    px = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 7e-5))
    lifts = (rng.random(n) > 0.48).astype(float)
    qty = rng.uniform(0.1, 5.0, n)
    rx[50:55] = rx[50]          # same-timestamp trade burst -> summed flow
    rx[250:254] = rx[250]
    return rx, px, lifts, qty


def _raw_events(books, trades, coin="x"):
    listings = tuple(f"{ex}_{coin}" for ex in books)
    cols: dict[str, list] = {k: [] for k in "rx kind lid t a b c d".split()}

    def add(rx, kind, lid, t, a, b, c, d):
        n = len(rx)
        cols["rx"].append(rx.astype(np.int64))
        cols["kind"].append(np.full(n, kind, np.int8))
        cols["lid"].append(np.full(n, lid, np.int8))
        cols["t"].append(t.astype(np.int64))
        for k, v in (("a", a), ("b", b), ("c", c), ("d", d)):
            cols[k].append(v.astype(float))

    for lid, ex in enumerate(books):
        rx, bid, bq, ask, aq = books[ex]
        add(rx, 0, lid, rx, bid, ask, bq, aq)
    for lid, ex in enumerate(books):
        rx, px, lifts, qty = trades[ex]
        add(rx, 1, lid, rx, px, lifts, qty, np.full(len(rx), np.nan))

    C = {k: np.concatenate(v) for k, v in cols.items()}
    order = np.lexsort((C["kind"], C["rx"]))
    return RawEventStream(*(C[k][order] for k in "rx kind lid t a b c d".split()), listings)


def _move_stream(mid_stream):
    rx, mid = mid_stream
    keep = np.concatenate([rx[1:] != rx[:-1], [True]])
    rx, mid = rx[keep], mid[keep]
    dlog = np.diff(np.log(mid))
    move = dlog != 0.0
    return rx[1:][move], dlog[move]


def _kernel_sum(event_rx, values, merged_ts, anchor_ts, span):
    a = 2.0 / (span + 1.0)
    beta = 1.0 - a
    events: dict[int, float] = {}
    for ts, val in zip(event_rx, values):
        events[int(ts)] = events.get(int(ts), 0.0) + float(val)
    trades = set(int(t) for t in merged_ts)
    all_ts = sorted(set(events) | trades)
    E = 0.0
    ts_arr, Es = [], []
    for ts in all_ts:
        if ts in events:
            E += a * events[ts]
        if ts in trades:
            E *= beta
        ts_arr.append(ts)
        Es.append(E)
    out = np.full(len(anchor_ts), np.nan)
    if not ts_arr:
        return out
    ts_arr, Es = np.array(ts_arr), np.array(Es)
    idx = np.searchsorted(ts_arr, anchor_ts, "right") - 1
    ok = idx >= 0
    out[ok] = Es[idx[ok]]
    return out


def _kernel_mean(event_rx, values, merged_ts, anchor_ts, span):
    E = _kernel_sum(event_rx, values, merged_ts, anchor_ts, span)
    W = _kernel_sum(event_rx, np.ones(len(values)), merged_ts, anchor_ts, span)
    return E / np.where(W == 0.0, np.nan, W)


def _sigma(mid_stream, merged_ts, anchor_ts, span=25):
    rx, ret = _move_stream(mid_stream)
    return np.sqrt(_kernel_mean(rx, ret * ret, merged_ts, anchor_ts, span))


def _synthetic_market(seed=0):
    exes = ("byb", "bin", "okx")
    books = {ex: _synthetic_book(seed + i) for i, ex in enumerate(exes)}
    trades = {ex: _synthetic_trades(seed + 10 + i) for i, ex in enumerate(exes)}
    merged_ts = np.unique(np.concatenate([trades[ex][0] for ex in exes]))
    anchor_ts = merged_ts[150:-150:7]
    return books, trades, merged_ts, anchor_ts


def _ctx(books, trades, merged_ts, anchor_ts, target="byb_x", sources=("bin", "okx")):
    """Synthetic `ScreeningContext`; target+sources define which venues a feature fans out over."""
    exes = tuple(dict.fromkeys((target.split("_", 1)[0],) + tuple(sources)))
    mids = {ex: (b[0], 0.5 * (b[1] + b[3])) for ex, b in books.items()}
    target_ex = target.split("_", 1)[0]
    return ScreeningContext(
        block="syn", coin="x", target=target, sources=tuple(sources), horizon_ns=0,
        yardstick_span=25, mid_stream={ex: "front_levels" for ex in exes},
        merged_ts=merged_ts, anchor_ts=anchor_ts,
        tick_at_anchor=np.searchsorted(merged_ts, anchor_ts, "right") - 1,
        sigma_at_anchor=_sigma(mids[target_ex], merged_ts, anchor_ts, 25), lam_at_anchor=np.empty(0),
        price_target=np.empty(0), rate_target=np.empty(0), base=[], vol_level=np.empty(0),
        rate_level=np.empty(0), vol_regime=np.empty(0),
        raw_events=_raw_events({ex: books[ex] for ex in exes}, {ex: trades[ex] for ex in exes}),
        _mids={ex: mids[ex] for ex in exes},
        _books={ex: (books[ex][0], books[ex][1], books[ex][2], books[ex][3], books[ex][4]) for ex in exes},
        _trades={ex: trades[ex] for ex in exes},
    )


def _trade_value_ref(trades, merged_ts, anchor_ts, span):
    rx, px, lifts, qty = trades
    value = px * qty
    signed = np.where(lifts > 0.0, value, -value)
    uniq, inv = np.unique(rx, return_inverse=True)
    signed_sum = np.bincount(inv, weights=signed)
    value_sum = np.bincount(inv, weights=value)
    num = _kernel_sum(uniq, signed_sum, merged_ts, anchor_ts, span)
    den = _kernel_sum(uniq, value_sum, merged_ts, anchor_ts, span)
    return num / np.where(den == 0.0, np.nan, den)


def _price_momentum_ref(mid_stream, sigma, merged_ts, anchor_ts, span):
    rx, ret = _move_stream(mid_stream)
    return _kernel_mean(rx, ret, merged_ts, anchor_ts, span) / sigma


def _flow_persistence_ref(trades, merged_ts, anchor_ts, span):
    rx, px, lifts, qty = trades
    signs = np.where(lifts > 0.0, 1.0, -1.0)
    uniq, inv = np.unique(rx, return_inverse=True)
    eps = np.sign(np.bincount(inv, weights=signs))
    keep = eps != 0.0
    uniq, eps = uniq[keep], eps[keep]
    if len(eps) < 2:
        return np.full(len(anchor_ts), np.nan)
    return _kernel_mean(uniq[1:], eps[1:] * eps[:-1], merged_ts, anchor_ts, span)


def test_trade_flow_imbalance_vectorized_matches_oracle():
    books, trades, merged_ts, anchor_ts = _synthetic_market()
    ctx = _ctx(books, trades, merged_ts, anchor_ts)
    spec = base.get("trade_flow_imbalance")
    for n in (2, 50):
        got = spec.vectorized(ctx, n)["byb"]
        ref = _trade_value_ref(trades["byb"], merged_ts, anchor_ts, n)
        ok = np.isfinite(got) & np.isfinite(ref)
        assert ok.sum() > 100
        np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-9, atol=1e-12)


def test_price_momentum_vectorized_matches_oracle():
    books, trades, merged_ts, anchor_ts = _synthetic_market(seed=1)
    ctx = _ctx(books, trades, merged_ts, anchor_ts)
    spec = base.get("price_momentum")
    for n in (2, 50):
        got = spec.vectorized(ctx, n)["byb"]
        ref = _price_momentum_ref(ctx._mids["byb"], ctx.sigma_at_anchor, merged_ts, anchor_ts, n)
        ok = np.isfinite(got) & np.isfinite(ref)
        assert ok.sum() > 100
        np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-9, atol=1e-12)


def test_flow_persistence_vectorized_matches_oracle():
    books, trades, merged_ts, anchor_ts = _synthetic_market(seed=2)
    ctx = _ctx(books, trades, merged_ts, anchor_ts)
    spec = base.get("flow_persistence")
    for n in (2, 50):
        got = spec.vectorized(ctx, n)["byb"]
        ref = _flow_persistence_ref(trades["byb"], merged_ts, anchor_ts, n)
        ok = np.isfinite(got) & np.isfinite(ref)
        assert ok.sum() > 100
        np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-9, atol=1e-12)


def test_trade_flow_features_fan_out_over_all_exchanges():
    books, trades, merged_ts, anchor_ts = _synthetic_market(seed=3)
    ctx = _ctx(books, trades, merged_ts, anchor_ts)
    for name in ("trade_flow_imbalance", "price_momentum", "flow_persistence"):
        spec = base.get(name)
        out = spec.vectorized(ctx, 20)
        assert set(out) == {"byb", "bin", "okx"}
        for ex in ("byb", "bin", "okx"):
            sources = tuple(s for s in ("bin", "okx") if s == ex)
            solo_ctx = _ctx(books, trades, merged_ts, anchor_ts, sources=sources)
            np.testing.assert_array_equal(out[ex], spec.vectorized(solo_ctx, 20)[ex])


def test_trade_flow_imbalance_mirror_commutes_with_side_reflection():
    books, trades, merged_ts, anchor_ts = _synthetic_market(seed=4)
    ctx = _ctx(books, trades, merged_ts, anchor_ts)
    mctx = _ctx(books, trades, merged_ts, anchor_ts)
    mctx._trades = {
        ex: (rx, px, 1.0 - lifts, qty) for ex, (rx, px, lifts, qty) in ctx._trades.items()
    }
    spec = base.get("trade_flow_imbalance")
    feat = spec.vectorized(ctx, 50)
    refl = spec.vectorized(mctx, 50)
    for ex in ("byb", "bin", "okx"):
        lhs = spec.mirror(feat[ex])
        ok = np.isfinite(lhs) & np.isfinite(refl[ex])
        assert ok.sum() > 100
        np.testing.assert_allclose(lhs[ok], refl[ex][ok], rtol=1e-6, atol=1e-12)


def test_price_momentum_mirror_commutes_with_price_reflection():
    books, trades, merged_ts, anchor_ts = _synthetic_market(seed=4)
    ctx = _ctx(books, trades, merged_ts, anchor_ts)
    mctx = _ctx(books, trades, merged_ts, anchor_ts)
    mctx._mids = {ex: (rx, 100.0 * 100.0 / mid) for ex, (rx, mid) in ctx._mids.items()}
    mctx.sigma_at_anchor = ctx.sigma_at_anchor
    spec = base.get("price_momentum")
    feat = spec.vectorized(ctx, 50)
    refl = spec.vectorized(mctx, 50)
    for ex in ("byb", "bin", "okx"):
        lhs = spec.mirror(feat[ex])
        ok = np.isfinite(lhs) & np.isfinite(refl[ex])
        assert ok.sum() > 100
        np.testing.assert_allclose(lhs[ok], refl[ex][ok], rtol=1e-6, atol=1e-12)


def test_flow_persistence_mirror_commutes_with_trade_reflection():
    books, trades, merged_ts, anchor_ts = _synthetic_market(seed=4)
    ctx = _ctx(books, trades, merged_ts, anchor_ts)
    mctx = _ctx(books, trades, merged_ts, anchor_ts)
    mctx._trades = {
        ex: (rx, 100.0 * 100.0 / px, 1.0 - lifts, qty) for ex, (rx, px, lifts, qty) in ctx._trades.items()
    }
    spec = base.get("flow_persistence")
    feat = spec.vectorized(ctx, 50)
    refl = spec.vectorized(mctx, 50)
    for ex in ("byb", "bin", "okx"):
        lhs = spec.mirror(feat[ex])
        ok = np.isfinite(lhs) & np.isfinite(refl[ex])
        assert ok.sum() > 100
        np.testing.assert_allclose(lhs[ok], refl[ex][ok], rtol=1e-6, atol=1e-12)


def test_trade_flow_imbalance_synthetic_parity():
    books, trades, merged_ts, anchor_ts = _synthetic_market(seed=5)
    ctx = _ctx(books, trades, merged_ts, anchor_ts)
    spec = base.get("trade_flow_imbalance")
    rep = parity_check(ctx, spec, [2, 50], n_grid=len(ctx.anchor_ts), tol=1e-9)
    assert rep.passed, str(rep)


def test_price_momentum_synthetic_parity():
    books, trades, merged_ts, anchor_ts = _synthetic_market(seed=5)
    ctx = _ctx(books, trades, merged_ts, anchor_ts)
    spec = base.get("price_momentum")
    rep = parity_check(ctx, spec, [2, 50], n_grid=len(ctx.anchor_ts), tol=1e-9)
    assert rep.passed, str(rep)


def test_flow_persistence_synthetic_parity():
    books, trades, merged_ts, anchor_ts = _synthetic_market(seed=5)
    ctx = _ctx(books, trades, merged_ts, anchor_ts)
    spec = base.get("flow_persistence")
    rep = parity_check(ctx, spec, [2, 50], n_grid=len(ctx.anchor_ts), tol=1e-9)
    assert rep.passed, str(rep)


@pytest.mark.skipif(getattr(io, "DATA_DIR", None) is None, reason="no DATA_DIR configured")
def test_real_block_trade_flow_features_parity():
    from boba.research.screening import build_context

    ctx = build_context(hours=2)
    for name in ("trade_flow_imbalance", "price_momentum", "flow_persistence"):
        rep = parity_check(ctx, base.get(name), [2, 100], tol=1e-6)
        assert rep.passed, str(rep)
