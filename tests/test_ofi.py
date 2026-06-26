"""Tests for the path-sum OFI features (boba.features.ofi_fast_slow, boba.features.ofi_ema).

The vectorized build is checked against an INDEPENDENT, dead-simple oracle (an explicit event loop, no
shared code) on synthetic books; the declared mirror is checked via the commutation invariant (reflect
the book, rebuild, assert the feature negates); and a real-block parity test (skipped without DATA_DIR)
confirms the streaming build reproduces the vectorized one — the AUTHORING.md validation trio.
"""
import numpy as np
import pytest

import boba.io as io
from boba.features import base
import boba.features.ofi_ema        # noqa: F401  (registers)
import boba.features.ofi_fast_slow  # noqa: F401  (registers)
from boba.research.screening import RawEventStream, ScreeningContext


# --------------------------------------------------------------------------------------------------
# synthetic context with a byb book + trade clock
# --------------------------------------------------------------------------------------------------
def _ctx(books, merged_ts, anchor_ts, target="byb_x", sources=()):
    """`books` is {short_ex -> (rx, bid, bid_qty, ask, ask_qty)}; the feature fans out over target+sources."""
    return ScreeningContext(
        block="syn", coin="x", target=target, sources=tuple(sources), horizon_ns=0,
        yardstick_span=10, mid_stream={}, merged_ts=merged_ts, anchor_ts=anchor_ts,
        tick_at_anchor=np.empty(0), sigma_at_anchor=np.empty(0), lam_at_anchor=np.empty(0),
        price_target=np.empty(0), rate_target=np.empty(0), base=[], vol_level=np.empty(0),
        rate_level=np.empty(0), vol_regime=np.empty(0),
        raw_events=RawEventStream(*([np.empty(0)] * 8), ()), _books=dict(books))


def _synthetic_book(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    rx = (np.arange(1, n + 1) * 10).astype(np.int64)             # a book row every 10 ns
    mid = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 1e-4))
    hs = 0.01 + 0.01 * rng.random(n)
    bid, ask = mid - hs, mid + hs
    bq, aq = rng.uniform(1.0, 100.0, n), rng.uniform(1.0, 100.0, n)
    # trade clock: every 3rd book row (coincident) plus some trade-only timestamps between rows
    merged_ts = np.unique(np.concatenate([rx[::3], rx[1::7] + 3]))
    anchor_ts = rx[200:-200:9]
    return (rx, bid, bq, ask, aq), merged_ts, anchor_ts


# --------------------------------------------------------------------------------------------------
# oracle — an explicit one-event-at-a-time loop, no shared code with the production build
# --------------------------------------------------------------------------------------------------
def _ofi_leg_oracle(book, merged_ts, anchor_ts, span):
    """E/W of the path-sum OFI flow at each anchor, by a dead-simple loop: per consecutive raw row form
    the CKS increment, SUM increments sharing a timestamp, then walk all event timestamps in order
    injecting (book update) / decaying (trade), and read E/W after the last event at-or-before each anchor."""
    rx, bid, bq, ask, aq = book
    a = 2.0 / (span + 1.0)
    sums: dict[int, float] = {}
    for i in range(1, len(rx)):
        pbp, pbq, pap, paq = bid[i - 1], bq[i - 1], ask[i - 1], aq[i - 1]
        cbp, cbq, cap, caq = bid[i], bq[i], ask[i], aq[i]
        inc = ((cbq if cbp >= pbp else 0.0) - (pbq if cbp <= pbp else 0.0)
               - (caq if cap <= pap else 0.0) + (paq if cap >= pap else 0.0))
        sums[int(rx[i])] = sums.get(int(rx[i]), 0.0) + inc
    trades = set(int(t) for t in merged_ts)
    all_ts = sorted(set(sums) | trades)
    E = W = 0.0
    ts_arr, Es, Ws = [], [], []
    for ts in all_ts:
        if ts in sums:                       # inject the summed OFI (one weight-1 sample)
            E += a * sums[ts]; W += a
        if ts in trades:                     # decay once on the trade clock (inject-then-decay, like refresh)
            E *= (1.0 - a); W *= (1.0 - a)
        ts_arr.append(ts); Es.append(E); Ws.append(W)
    ts_arr, Es, Ws = np.array(ts_arr), np.array(Es), np.array(Ws)
    j = np.searchsorted(ts_arr, anchor_ts, "right") - 1          # last event at-or-before each anchor
    out = np.full(len(anchor_ts), np.nan)
    ok = j >= 0
    Ej, Wj = Es[j[ok]], Ws[j[ok]]
    out[ok] = np.where(Wj > 0.0, Ej / np.where(Wj == 0.0, np.nan, Wj), np.nan)
    return out


def test_ofi_ema_vectorized_matches_oracle():
    book, merged_ts, anchor_ts = _synthetic_book()
    ctx = _ctx({"byb": book}, merged_ts, anchor_ts)
    spec = base.get("ofi_ema")
    for n in (1, 10, 100):
        got = spec.vectorized(ctx, n)["byb"]
        ref = _ofi_leg_oracle(book, merged_ts, anchor_ts, n)
        ok = np.isfinite(got) & np.isfinite(ref)
        assert ok.sum() > 50
        np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-9, atol=1e-9)


def test_ofi_fast_slow_vectorized_matches_oracle():
    book, merged_ts, anchor_ts = _synthetic_book(seed=1)
    ctx = _ctx({"byb": book}, merged_ts, anchor_ts)
    spec = base.get("ofi_fast_slow")
    for nf, ns in [(1, 100), (10, 500)]:
        got = spec.vectorized(ctx, (nf, ns))["byb"]
        ref = _ofi_leg_oracle(book, merged_ts, anchor_ts, nf) - _ofi_leg_oracle(book, merged_ts, anchor_ts, ns)
        ok = np.isfinite(got) & np.isfinite(ref)
        assert ok.sum() > 50
        np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-9, atol=1e-9)


def test_ofi_fans_out_over_all_exchanges():
    # one independent OFI leg per venue (target + sources), each = the single-venue build on its own book
    b_byb, merged_ts, anchor_ts = _synthetic_book(seed=3)
    b_bin, _, _ = _synthetic_book(seed=4)
    b_okx, _, _ = _synthetic_book(seed=5)
    books = {"byb": b_byb, "bin": b_bin, "okx": b_okx}
    ctx = _ctx(books, merged_ts, anchor_ts, sources=("bin", "okx"))
    for name, params in [("ofi_fast_slow", (10, 100)), ("ofi_ema", 50)]:
        spec = base.get(name)
        out = spec.vectorized(ctx, params)
        assert set(out) == {"byb", "bin", "okx"}                       # one leg per exchange
        for ex, book in books.items():                                 # each leg is that venue's own-book OFI, independent
            solo = spec.vectorized(_ctx({ex: book}, merged_ts, anchor_ts, target=f"{ex}_x"), params)[ex]
            np.testing.assert_array_equal(out[ex], solo)


# --------------------------------------------------------------------------------------------------
# mirror commutation invariant — reflecting the book through byb's mid negates the OFI feature
# --------------------------------------------------------------------------------------------------
def _mirror_book(book, c=100.0):
    """Reflect the book through price level c: bid/ask SWAP and reflect (p -> c**2/p), sizes follow their
    level (bid' size = old ask size). The data-level operation a signed-flow feature's `mirror` must commute with."""
    rx, bid, bq, ask, aq = book
    return (rx, c * c / ask, aq, c * c / bid, bq)


def test_ofi_features_mirror_commute_with_book_reflection():
    book, merged_ts, anchor_ts = _synthetic_book(seed=2)
    ctx = _ctx({"byb": book}, merged_ts, anchor_ts)
    mctx = _ctx({"byb": _mirror_book(book)}, merged_ts, anchor_ts)
    for name, params in [("ofi_fast_slow", (10, 100)), ("ofi_ema", 50)]:
        spec = base.get(name)
        feat = spec.vectorized(ctx, params)["byb"]            # feature(books)
        refl = spec.vectorized(mctx, params)["byb"]           # feature(mirror_books(books))
        ok = np.isfinite(feat) & np.isfinite(refl)
        assert ok.sum() > 50
        np.testing.assert_allclose(spec.mirror(feat)[ok], refl[ok], rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------------------------------
# real-block parity (skipped without DATA_DIR) — streaming reproduces vectorized
# --------------------------------------------------------------------------------------------------
@pytest.mark.skipif(getattr(io, "DATA_DIR", None) is None, reason="no DATA_DIR configured")
def test_real_block_ofi_parity():
    from boba.research.screening import build_context, parity_check

    ctx = build_context(hours=2)
    for name, spans in [("ofi_fast_slow", [(1, 100), (10, 500)]), ("ofi_ema", [1, 100])]:
        rep = parity_check(ctx, base.get(name), spans)
        assert rep.passed, str(rep)
