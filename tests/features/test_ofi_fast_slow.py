"""Per-feature test suite for `boba.features.ofi_fast_slow` (the path-sum L1 OFI fast/slow oscillator).

Self-contained: its own synthetic `ScreeningContext` / book builders and its own INDEPENDENT,
dead-simple oracle (an explicit per-event loop sharing NO code with the production build). Covers the
AUTHORING.md validation trio plus the `span = 1` boundary:

  (a) vectorized  vs  the independent oracle on synthetic books (incl. a span=1 leg),
  (b) the mirror COMMUTATION invariant — `mirror(feature(books)) == feature(mirror_books(books))`
      against the FULL book reflection (OFI is ODD: signed flow, so the value negates),
  (c) the per-venue fan-out — one independent OFI leg per exchange, each that venue's own-book OFI,
  + span=1 (alpha=1, the most-used fast leg): finite wherever the inputs exist, consistent NaN where not,
  + a real-block parity test (streaming == vectorized), skipped when DATA_DIR is unset.
"""
import numpy as np
import pytest

import boba.io as io
from boba.features import base
import boba.features.ofi_fast_slow  # noqa: F401  (registers SPEC)
from boba.research.screening import RawEventStream, ScreeningContext


# --------------------------------------------------------------------------------------------------
# synthetic context — a byb book + a shared trade clock; the feature fans out over target + sources
# --------------------------------------------------------------------------------------------------
def _ctx(books, merged_ts, anchor_ts, target="byb_x", sources=()):
    """`books` is {short_ex -> (rx, bid, bid_qty, ask, ask_qty)}; vectorized-only, so raw_events is empty."""
    return ScreeningContext(
        block="syn", coin="x", target=target, sources=tuple(sources), horizon_ns=0,
        yardstick_span=10, mid_stream={}, merged_ts=merged_ts, anchor_ts=anchor_ts,
        tick_at_anchor=np.empty(0), sigma_at_anchor=np.empty(0), lam_at_anchor=np.empty(0),
        price_target=np.empty(0), rate_target=np.empty(0), base=[], vol_level=np.empty(0),
        rate_level=np.empty(0), vol_regime=np.empty(0),
        raw_events=RawEventStream(*([np.empty(0)] * 8), ()), _books=dict(books))


def _synthetic_book(seed=0, n=4000):
    """A 1-row-per-10ns book with random walk mid + random sizes, plus a shared trade clock (some
    coincident with book rows, some trade-only) and a warmed-up anchor grid."""
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
# oracle — an explicit one-event-at-a-time loop, NO shared code with the production build.
# Implementable from the feature's written definition alone: form the CKS L1 OFI increment for every
# consecutive raw row, SUM increments sharing a receive-timestamp into one weight-1 flow sample, then
# walk all event timestamps in order injecting (book update) / decaying once per trade (E/W EMA).
# --------------------------------------------------------------------------------------------------
def _ofi_leg_oracle(book, merged_ts, anchor_ts, span):
    rx, bid, bq, ask, aq = book
    a = 2.0 / (span + 1.0)                                       # span=1 -> a=1 (no smoothing)
    sums: dict[int, float] = {}
    for i in range(1, len(rx)):
        pbp, pbq, pap, paq = bid[i - 1], bq[i - 1], ask[i - 1], aq[i - 1]
        cbp, cbq, cap, caq = bid[i], bq[i], ask[i], aq[i]
        inc = ((cbq if cbp >= pbp else 0.0) - (pbq if cbp <= pbp else 0.0)
               - (caq if cap <= pap else 0.0) + (paq if cap >= pap else 0.0))
        sums[int(rx[i])] = sums.get(int(rx[i]), 0.0) + inc       # records at one ns are ONE flow event
    trades = set(int(t) for t in merged_ts)
    all_ts = sorted(set(sums) | trades)
    E = W = 0.0
    ts_arr, Es, Ws = [], [], []
    for ts in all_ts:
        if ts in sums:                       # inject the summed OFI as ONE weight-1 sample
            E += a * sums[ts]; W += a
        if ts in trades:                     # decay once on the shared trade clock (inject-then-decay)
            E *= (1.0 - a); W *= (1.0 - a)
        ts_arr.append(ts); Es.append(E); Ws.append(W)
    ts_arr, Es, Ws = np.array(ts_arr), np.array(Es), np.array(Ws)
    j = np.searchsorted(ts_arr, anchor_ts, "right") - 1          # last event at-or-before each anchor
    out = np.full(len(anchor_ts), np.nan)
    ok = j >= 0
    Ej, Wj = Es[j[ok]], Ws[j[ok]]
    out[ok] = np.where(Wj > 0.0, Ej / np.where(Wj == 0.0, np.nan, Wj), np.nan)
    return out


def _fast_slow_oracle(book, merged_ts, anchor_ts, n_fast, n_slow):
    return (_ofi_leg_oracle(book, merged_ts, anchor_ts, n_fast)
            - _ofi_leg_oracle(book, merged_ts, anchor_ts, n_slow))


# (a) vectorized vs the independent oracle, including a span=1 fast leg
def test_vectorized_matches_oracle():
    book, merged_ts, anchor_ts = _synthetic_book(seed=1)
    ctx = _ctx({"byb": book}, merged_ts, anchor_ts)
    spec = base.get("ofi_fast_slow")
    for nf, ns in [(1, 100), (10, 500), (1, 50)]:               # (1, *) exercises the alpha=1 fast leg
        got = spec.vectorized(ctx, (nf, ns))["byb"]
        ref = _fast_slow_oracle(book, merged_ts, anchor_ts, nf, ns)
        ok = np.isfinite(got) & np.isfinite(ref)
        assert ok.sum() > 50
        np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-9, atol=1e-9)


# span = 1 (alpha = 1, the most-used fast leg): finite wherever inputs exist, consistent NaN otherwise.
def test_span_one_is_finite_and_consistent():
    book, merged_ts, anchor_ts = _synthetic_book(seed=8)
    ctx = _ctx({"byb": book}, merged_ts, anchor_ts)
    spec = base.get("ofi_fast_slow")
    # A single leg at span=1: alpha=1 -> E/W is the latest summed-OFI sample / 1; never inf or 0/0.
    leg1 = spec.vectorized(ctx, (1, 1))["byb"]                  # fast=slow=1 -> identically 0 where defined
    ref1 = _ofi_leg_oracle(book, merged_ts, anchor_ts, 1)
    defined = np.isfinite(ref1)                                 # the oracle's notion of "inputs exist"
    assert defined.sum() > 50
    # Wherever the span=1 leg is defined the value is FINITE (no inf), and the production NaN mask is
    # exactly the oracle's (a consistent NaN where undefined — never a build-specific number / inf).
    assert np.all(np.isfinite(leg1[defined]))
    assert np.array_equal(np.isfinite(leg1), defined)
    np.testing.assert_array_equal(leg1[defined], 0.0)          # fast - slow with equal spans is 0
    # And a genuine span=1 fast leg vs a slow leg: finite wherever both legs are defined, NaN where not.
    fs = spec.vectorized(ctx, (1, 100))["byb"]
    both = np.isfinite(_ofi_leg_oracle(book, merged_ts, anchor_ts, 1)) & \
        np.isfinite(_ofi_leg_oracle(book, merged_ts, anchor_ts, 100))
    assert np.all(np.isfinite(fs[both]))
    assert np.array_equal(np.isfinite(fs), both)


# (b) mirror COMMUTATION invariant against the FULL book reflection
def _mirror_book(book, c=100.0):
    """Reflect the L1 book through price level c: bid/ask SWAP and reflect (p -> c**2/p), and each
    side's SIZE follows its original price level (bid' size = old ask size). The clock is unchanged.
    This is the data-level reflection a signed-flow feature's `mirror` must commute with (AUTHORING.md)."""
    rx, bid, bq, ask, aq = book
    return (rx, c * c / ask, aq, c * c / bid, bq)


def test_mirror_commutes_with_book_reflection():
    # OFI is ODD (signed order flow): reflecting the book swaps bid/ask so the increment negates,
    # hence SPEC.mirror == np.negative. Assert mirror(feature(books)) == feature(mirror_books(books)).
    book, merged_ts, anchor_ts = _synthetic_book(seed=2)
    ctx = _ctx({"byb": book}, merged_ts, anchor_ts)
    mctx = _ctx({"byb": _mirror_book(book)}, merged_ts, anchor_ts)
    spec = base.get("ofi_fast_slow")
    for params in [(1, 100), (10, 500)]:                       # incl. a span=1 fast leg
        feat = spec.vectorized(ctx, params)["byb"]            # feature(books)
        refl = spec.vectorized(mctx, params)["byb"]           # feature(mirror_books(books))
        ok = np.isfinite(feat) & np.isfinite(refl)
        assert ok.sum() > 50
        np.testing.assert_allclose(spec.mirror(feat)[ok], refl[ok], rtol=1e-6, atol=1e-9)


# (c) per-venue fan-out — one independent OFI leg per exchange (target + every source)
def test_fans_out_over_all_exchanges():
    b_byb, merged_ts, anchor_ts = _synthetic_book(seed=3)
    b_bin, _, _ = _synthetic_book(seed=4)
    b_okx, _, _ = _synthetic_book(seed=5)
    books = {"byb": b_byb, "bin": b_bin, "okx": b_okx}
    ctx = _ctx(books, merged_ts, anchor_ts, sources=("bin", "okx"))
    spec = base.get("ofi_fast_slow")
    out = spec.vectorized(ctx, (10, 100))
    assert set(out) == {"byb", "bin", "okx"}                   # one leg per exchange
    for ex, book in books.items():                             # each leg = that venue's own-book OFI, independent
        solo = spec.vectorized(_ctx({ex: book}, merged_ts, anchor_ts, target=f"{ex}_x"), (10, 100))[ex]
        np.testing.assert_array_equal(out[ex], solo)


# real-block parity (skipped without DATA_DIR) — the streaming build reproduces the vectorized one,
# including a span=1 fast leg in the sweep.
@pytest.mark.skipif(getattr(io, "DATA_DIR", None) is None, reason="no DATA_DIR configured")
def test_real_block_parity():
    from boba.research.screening import build_context, parity_check

    ctx = build_context(hours=2)
    spec = base.get("ofi_fast_slow")
    rep = parity_check(ctx, spec, [(1, 100), (10, 500)])       # (1, *) exercises alpha=1 online vs offline
    assert rep.passed, str(rep)
