"""Self-contained test suite for `boba.features.stoikov_premium_fast_slow`.

The feature computes the SIZE-WEIGHTED MID premium -- the leading / first-order term of Stoikov's
micro-price (the imbalance-adjusted mid, NOT the full martingale estimator):

    microprice = (bid_qty * ask_prc + ask_qty * bid_prc) / (bid_qty + ask_qty)
    prem       = (microprice - mid) / mid

then smoothed two ways with `LiveFrontEMA` and returned as `fast - slow`, per venue.
Reference: Stoikov, S. (2018) 'The Micro-Price: A High-Frequency Estimator of Future Prices',
Quantitative Finance 18(12):1959-1966 (SSRN 2970694); Gatheral, J. & Oomen, R. (size-weighted mid).

The AUTHORING.md validation trio, self-contained (own synthetic builders + own independent oracle,
no shared code with the production build):
  (a) vectorized vs. an independent dead-simple event-loop oracle on synthetic books;
  (b) the mirror COMMUTATION invariant `mirror(feature(books)) == feature(mirror_books(books))` against
      the FULL book reflection (reflect prices AND swap sides/sizes) -- ODD here, so the value negates;
  (c) the per-venue fan-out (one independent leg per exchange);
plus a `span = 1` (alpha=1) finiteness row, the synthetic streaming-vs-vectorized parity, and a
real-block parity test (skipped without DATA_DIR).
"""
import numpy as np
import pytest

import boba.io as io
from boba.features import base
import boba.features.stoikov_premium_fast_slow  # noqa: F401  (registers)
from boba.research.screening import RawEventStream, ScreeningContext, parity_check


# --------------------------------------------------------------------------------------------------
# synthetic raw-event stream + context (own builders)
# --------------------------------------------------------------------------------------------------
def _raw_events(books, merged_ts, coin="x"):
    """Raw event stream: one book row per `(rx, bid, bq, ask, aq)` plus one trade per shared ts."""
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
    """`books` is {short_ex -> (rx, bid, bid_qty, ask, ask_qty)}; fans out over target + sources."""
    return ScreeningContext(
        block="syn", coin="x", target=target, sources=tuple(sources), horizon_ns=0,
        yardstick_span=10, mid_stream={}, merged_ts=merged_ts, anchor_ts=anchor_ts,
        tick_at_anchor=np.searchsorted(merged_ts, anchor_ts, "right") - 1,
        sigma_at_anchor=np.empty(0), lam_at_anchor=np.empty(0), price_target=np.empty(0),
        rate_target=np.empty(0), base=[], vol_level=np.empty(0), rate_level=np.empty(0),
        vol_regime=np.empty(0),
        raw_events=_raw_events(books, merged_ts) if raw else RawEventStream(*([np.empty(0)] * 8), ()),
        _books=dict(books),
    )


def _synthetic_book(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    rx = (np.arange(1, n + 1) * 10).astype(np.int64)                 # a book row every 10 ns
    mid = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 1e-4))
    hs = 0.005 + 0.005 * rng.random(n)
    bid, ask = mid - hs, mid + hs
    bq, aq = rng.uniform(1.0, 100.0, n), rng.uniform(1.0, 100.0, n)
    # Duplicate a few timestamps; level semantics keep the final book state for that timestamp.
    rx[100:110] = rx[100]
    rx[100:110].sort()
    merged_ts = np.unique(np.concatenate([rx[::3], rx[1::7] + 3]))
    anchor_ts = rx[250:-250:11]
    return (rx, bid, bq, ask, aq), merged_ts, anchor_ts


# --------------------------------------------------------------------------------------------------
# independent oracle — explicit per-event loop, NO shared code with the production build
# --------------------------------------------------------------------------------------------------
def _premium(bid, bq, ask, aq):
    """Size-weighted-mid premium from a single L1 state (leading Stoikov term)."""
    mid = 0.5 * (bid + ask)
    micro = (bq * ask + aq * bid) / (bq + aq)
    return (micro - mid) / mid


def _live_front_leg_oracle(book, merged_ts, anchor_ts, span):
    """Dead-simple `LiveFrontEMA` oracle for the premium level.

    Walks every event timestamp in order: a book row refreshes the timestamp's last valid premium
    (last state wins on a duplicate ts); a trade decays/commits the EMA once. The read at each anchor
    is the live front `(1 - a) * committed + a * latest`. `span = 1` => a = 1 => the value collapses to
    the freshest premium read (no smoothing). NaN before the first commit / first valid level.
    """
    rx, bid, bq, ask, aq = book
    ok = (bid > 0.0) & (ask > 0.0) & (bq > 0.0) & (aq > 0.0)
    by_ts: dict[int, float] = {}
    for ts, val in zip(rx[ok], _premium(bid[ok], bq[ok], ask[ok], aq[ok])):
        by_ts[int(ts)] = float(val)                      # last state for a duplicated timestamp

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
    sel = idx >= 0
    out[sel] = vals[idx[sel]]
    return out


def _oracle_fast_slow(book, merged_ts, anchor_ts, nf, ns):
    return (_live_front_leg_oracle(book, merged_ts, anchor_ts, nf)
            - _live_front_leg_oracle(book, merged_ts, anchor_ts, ns))


# span sweep used by the oracle / parity tests — INCLUDES a span=1 (alpha=1) fast leg.
_SWEEP = [(1, 100), (10, 500)]


# --------------------------------------------------------------------------------------------------
# (a) vectorized vs. independent oracle
# --------------------------------------------------------------------------------------------------
def test_vectorized_matches_oracle():
    book, merged_ts, anchor_ts = _synthetic_book()
    ctx = _ctx({"byb": book}, merged_ts, anchor_ts, raw=False)
    spec = base.get("stoikov_premium_fast_slow")
    for nf, ns in _SWEEP:
        got = spec.vectorized(ctx, (nf, ns))["byb"]
        ref = _oracle_fast_slow(book, merged_ts, anchor_ts, nf, ns)
        ok = np.isfinite(got) & np.isfinite(ref)
        assert ok.sum() > 50
        np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------------------------------
# span = 1 (alpha=1, no smoothing): finite wherever inputs exist, consistent NaN where undefined.
# --------------------------------------------------------------------------------------------------
def test_span1_leg_matches_oracle_and_is_finite():
    book, merged_ts, anchor_ts = _synthetic_book(seed=11)
    ctx = _ctx({"byb": book}, merged_ts, anchor_ts, raw=False)
    spec = base.get("stoikov_premium_fast_slow")

    # The single span=1 leg = the freshest premium read at-or-before each anchor (alpha=1, no memory).
    # vectorized(1, ns) - vectorized(slow) isolates the fast leg via the oracle; check the leg directly
    # by differencing two builds that share the slow leg.
    got_fast = (spec.vectorized(ctx, (1, 100))["byb"] - spec.vectorized(ctx, (50, 100))["byb"]
                + _live_front_leg_oracle(book, merged_ts, anchor_ts, 50))
    ref_fast = _live_front_leg_oracle(book, merged_ts, anchor_ts, 1)
    ok = np.isfinite(got_fast) & np.isfinite(ref_fast)
    assert ok.sum() > 50
    np.testing.assert_allclose(got_fast[ok], ref_fast[ok], rtol=1e-9, atol=1e-12)

    # Do-rule: at span=1 the value is FINITE everywhere the oracle is defined (no inf / build-specific
    # number), and the two builds agree on WHERE it is undefined (a consistent NaN both share).
    full = spec.vectorized(ctx, (1, 100))["byb"]
    ref_full = _oracle_fast_slow(book, merged_ts, anchor_ts, 1, 100)
    assert np.array_equal(np.isfinite(full), np.isfinite(ref_full))   # same defined/undefined mask
    assert np.all(np.isfinite(full[np.isfinite(ref_full)]))           # finite where inputs exist
    assert not np.any(np.isinf(full))                                 # never inf


# --------------------------------------------------------------------------------------------------
# (c) per-venue fan-out — one independent leg per exchange (target + sources)
# --------------------------------------------------------------------------------------------------
def test_fans_out_over_all_exchanges():
    b_byb, merged_ts, anchor_ts = _synthetic_book(seed=3)
    b_bin, _, _ = _synthetic_book(seed=4)
    b_okx, _, _ = _synthetic_book(seed=5)
    books = {"byb": b_byb, "bin": b_bin, "okx": b_okx}
    ctx = _ctx(books, merged_ts, anchor_ts, sources=("bin", "okx"), raw=False)
    spec = base.get("stoikov_premium_fast_slow")
    out = spec.vectorized(ctx, (10, 100))
    assert set(out) == {"byb", "bin", "okx"}                            # one leg per exchange
    for ex, book in books.items():                                     # each leg = that venue's own-book build
        solo = spec.vectorized(_ctx({ex: book}, merged_ts, anchor_ts, target=f"{ex}_x", raw=False), (10, 100))[ex]
        np.testing.assert_array_equal(out[ex], solo)


# --------------------------------------------------------------------------------------------------
# (b) mirror commutation invariant — the FULL book reflection.
# Reflect every price through level c (p -> c**2/p) AND swap sides/sizes (bid' = reflect(ask), the
# bid keeps the OLD ask's size). The premium is ODD, so its declared mirror (np.negative) must
# reproduce the rebuild on the reflected book.
# --------------------------------------------------------------------------------------------------
def _mirror_book(book, c=100.0):
    rx, bid, bq, ask, aq = book
    return (rx, c * c / ask, aq, c * c / bid, bq)


def test_mirror_commutes_with_book_reflection():
    book, merged_ts, anchor_ts = _synthetic_book(seed=2)
    ctx = _ctx({"byb": book}, merged_ts, anchor_ts, raw=False)
    mctx = _ctx({"byb": _mirror_book(book)}, merged_ts, anchor_ts, raw=False)
    spec = base.get("stoikov_premium_fast_slow")
    for params in _SWEEP:
        feat = spec.vectorized(ctx, params)["byb"]                     # feature(books)
        refl = spec.vectorized(mctx, params)["byb"]                    # feature(mirror_books(books))
        ok = np.isfinite(feat) & np.isfinite(refl)
        assert ok.sum() > 50
        np.testing.assert_allclose(spec.mirror(feat)[ok], refl[ok], rtol=1e-6, atol=1e-12)


# --------------------------------------------------------------------------------------------------
# synthetic streaming-vs-vectorized parity (the production O(1) path reproduces the offline build)
# --------------------------------------------------------------------------------------------------
def test_synthetic_parity():
    b_byb, merged_ts, anchor_ts = _synthetic_book(seed=6)
    b_bin, _, _ = _synthetic_book(seed=7)
    ctx = _ctx({"byb": b_byb, "bin": b_bin}, merged_ts, anchor_ts, sources=("bin",))
    spec = base.get("stoikov_premium_fast_slow")
    rep = parity_check(ctx, spec, _SWEEP, n_grid=len(anchor_ts), tol=1e-12)
    assert rep.passed, str(rep)


# --------------------------------------------------------------------------------------------------
# real-block parity (skipped without DATA_DIR) — streaming reproduces vectorized on real timing
# --------------------------------------------------------------------------------------------------
@pytest.mark.skipif(getattr(io, "DATA_DIR", None) is None, reason="no DATA_DIR configured")
def test_real_block_parity():
    from boba.research.screening import build_context

    ctx = build_context(hours=2)
    spec = base.get("stoikov_premium_fast_slow")
    rep = parity_check(ctx, spec, _SWEEP, tol=1e-6)
    assert rep.passed, str(rep)
