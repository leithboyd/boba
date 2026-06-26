"""Tests for `boba.features.stoikov_premium_fast_slow`.

The vectorized build is checked against an independent event-loop oracle on synthetic books; the
streaming build is checked through the generic parity driver; the declared mirror is checked via the
book-reflection commutation invariant; and a real-block parity test runs when DATA_DIR is configured.
"""
import numpy as np
import pytest

import boba.io as io
from boba.features import base
import boba.features.stoikov_premium_fast_slow  # noqa: F401  (registers)
from boba.research.screening import RawEventStream, ScreeningContext, parity_check


def _raw_events(books, merged_ts, coin="x"):
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

    for lid, (ex, book) in enumerate(books.items()):
        rx, bid, bq, ask, aq = book
        add(rx, 0, lid, rx, bid, ask, bq, aq)

    # One trade event per shared trade timestamp is enough to drive the feature's decay clock.
    n = len(merged_ts)
    add(merged_ts, 1, 0, merged_ts, np.ones(n), np.ones(n), np.ones(n), np.full(n, np.nan))

    C = {k: np.concatenate(v) for k, v in cols.items()}
    order = np.lexsort((C["kind"], C["rx"]))
    return RawEventStream(*(C[k][order] for k in "rx kind lid t a b c d".split()), listings)


def _ctx(books, merged_ts, anchor_ts, target="byb_x", sources=(), raw=True):
    return ScreeningContext(
        block="syn", coin="x", target=target, sources=tuple(sources), horizon_ns=0,
        yardstick_span=10, mid_stream={}, merged_ts=merged_ts, anchor_ts=anchor_ts,
        tick_at_anchor=np.searchsorted(merged_ts, anchor_ts, "right") - 1,
        sigma_at_anchor=np.empty(0), lam_at_anchor=np.empty(0), price_target=np.empty(0),
        rate_target=np.empty(0), base=[], vol_level=np.empty(0), rate_level=np.empty(0),
        vol_regime=np.empty(0), raw_events=_raw_events(books, merged_ts) if raw else RawEventStream(*([np.empty(0)] * 8), ()),
        _books=dict(books),
    )


def _synthetic_book(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    rx = (np.arange(1, n + 1) * 10).astype(np.int64)
    mid = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 1e-4))
    hs = 0.005 + 0.005 * rng.random(n)
    bid, ask = mid - hs, mid + hs
    bq, aq = rng.uniform(1.0, 100.0, n), rng.uniform(1.0, 100.0, n)
    # Duplicate a few timestamps; level semantics should keep the final book state for that timestamp.
    rx[100:110] = rx[100]
    rx[100:110].sort()
    merged_ts = np.unique(np.concatenate([rx[::3], rx[1::7] + 3]))
    anchor_ts = rx[250:-250:11]
    return (rx, bid, bq, ask, aq), merged_ts, anchor_ts


def _premium(bid, bq, ask, aq):
    mid = 0.5 * (bid + ask)
    micro = (bq * ask + aq * bid) / (bq + aq)
    return (micro - mid) / mid


def _live_front_leg_oracle(book, merged_ts, anchor_ts, span):
    """Dead-simple `LiveFrontEMA` oracle for the premium level, with last book state per timestamp."""
    rx, bid, bq, ask, aq = book
    ok = (bid > 0.0) & (ask > 0.0) & (bq > 0.0) & (aq > 0.0)
    by_ts: dict[int, float] = {}
    for ts, val in zip(rx[ok], _premium(bid[ok], bq[ok], ask[ok], aq[ok])):
        by_ts[int(ts)] = float(val)

    trades = set(int(t) for t in merged_ts)
    all_ts = sorted(set(by_ts) | trades)
    a = 2.0 / (span + 1.0)
    latest = None
    ema = 0.0
    started = False
    ts_arr, vals = [], []
    for ts in all_ts:
        if ts in by_ts:
            latest = by_ts[ts]
        if ts in trades and latest is not None:
            ema = (1.0 - a) * ema + a * latest
            started = True
        vals.append((1.0 - a) * ema + a * latest if started and latest is not None else np.nan)
        ts_arr.append(ts)

    out = np.full(len(anchor_ts), np.nan)
    if not ts_arr:
        return out
    ts_arr, vals = np.array(ts_arr), np.array(vals)
    idx = np.searchsorted(ts_arr, anchor_ts, "right") - 1
    ok = idx >= 0
    out[ok] = vals[idx[ok]]
    return out


def test_stoikov_premium_vectorized_matches_oracle():
    book, merged_ts, anchor_ts = _synthetic_book()
    ctx = _ctx({"byb": book}, merged_ts, anchor_ts, raw=False)
    spec = base.get("stoikov_premium_fast_slow")
    for nf, ns in [(1, 100), (10, 500)]:
        got = spec.vectorized(ctx, (nf, ns))["byb"]
        ref = _live_front_leg_oracle(book, merged_ts, anchor_ts, nf) - _live_front_leg_oracle(book, merged_ts, anchor_ts, ns)
        ok = np.isfinite(got) & np.isfinite(ref)
        assert ok.sum() > 50
        np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-9, atol=1e-12)


def test_stoikov_premium_fans_out_over_all_exchanges():
    b_byb, merged_ts, anchor_ts = _synthetic_book(seed=3)
    b_bin, _, _ = _synthetic_book(seed=4)
    b_okx, _, _ = _synthetic_book(seed=5)
    books = {"byb": b_byb, "bin": b_bin, "okx": b_okx}
    ctx = _ctx(books, merged_ts, anchor_ts, sources=("bin", "okx"), raw=False)
    spec = base.get("stoikov_premium_fast_slow")
    out = spec.vectorized(ctx, (10, 100))
    assert set(out) == {"byb", "bin", "okx"}
    for ex, book in books.items():
        solo = spec.vectorized(_ctx({ex: book}, merged_ts, anchor_ts, target=f"{ex}_x", raw=False), (10, 100))[ex]
        np.testing.assert_array_equal(out[ex], solo)


def _mirror_book(book, c=100.0):
    """Reflect the L1 book through price level c; sizes follow their original price level."""
    rx, bid, bq, ask, aq = book
    return (rx, c * c / ask, aq, c * c / bid, bq)


def test_stoikov_premium_mirror_commutes_with_book_reflection():
    book, merged_ts, anchor_ts = _synthetic_book(seed=2)
    ctx = _ctx({"byb": book}, merged_ts, anchor_ts, raw=False)
    mctx = _ctx({"byb": _mirror_book(book)}, merged_ts, anchor_ts, raw=False)
    spec = base.get("stoikov_premium_fast_slow")
    for params in [(1, 100), (10, 500)]:
        feat = spec.vectorized(ctx, params)["byb"]
        refl = spec.vectorized(mctx, params)["byb"]
        ok = np.isfinite(feat) & np.isfinite(refl)
        assert ok.sum() > 50
        np.testing.assert_allclose(spec.mirror(feat)[ok], refl[ok], rtol=1e-6, atol=1e-12)


def test_stoikov_premium_synthetic_parity():
    b_byb, merged_ts, anchor_ts = _synthetic_book(seed=6)
    b_bin, _, _ = _synthetic_book(seed=7)
    ctx = _ctx({"byb": b_byb, "bin": b_bin}, merged_ts, anchor_ts, sources=("bin",))
    spec = base.get("stoikov_premium_fast_slow")
    rep = parity_check(ctx, spec, [(1, 100), (10, 500)], n_grid=len(anchor_ts), tol=1e-12)
    assert rep.passed, str(rep)


@pytest.mark.skipif(getattr(io, "DATA_DIR", None) is None, reason="no DATA_DIR configured")
def test_real_block_stoikov_premium_parity():
    from boba.research.screening import build_context

    ctx = build_context(hours=2)
    spec = base.get("stoikov_premium_fast_slow")
    rep = parity_check(ctx, spec, [(1, 100), (10, 500)], tol=1e-6)
    assert rep.passed, str(rep)
