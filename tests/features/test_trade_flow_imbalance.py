"""Dedicated self-contained tests for `boba.features.trade_flow_imbalance`.

`trade_flow_imbalance` is the signed-trade-VOLUME imbalance: per venue and receive timestamp it sums
the trade flow into `signed_volume = sum(side*qty)` and `total_volume = sum(qty)`, then reads the
trade-clock EMA ratio `EMA(signed_volume) / EMA(total_volume)` in [-1, 1] (the exponentially-weighted
fraction of recent volume that was buyer- minus seller-initiated). It fans out one independent leg per
exchange (target + every source), and is ODD under the book/tape reflection -- `qty` is a count
(invariant) while the side flips -- so `SPEC.mirror = np.negative`.

This file is self-contained (its own synthetic context + book/trade builders + an INDEPENDENT
dead-simple oracle with no shared production code) and covers the AUTHORING.md validation trio:

  (a) vectorized vs an independent per-event-loop oracle on synthetic trades (volume-weighted),
  (b) the mirror COMMUTATION invariant against the FULL book reflection (reflect prices AND flip
      sides, qty invariant) -- an ODD feature, so the value negates,
  (c) the per-venue fan-out (one leg per exchange, each = that venue's own-tape build),

plus boundedness assertions, a synthetic streaming-parity check, and a real-block parity test
(skipped without DATA_DIR). Wherever the value is DEFINED it is asserted FINITE and BOUNDED in
[-1, 1] (no spurious inf/nan); where UNDEFINED both builds agree on a consistent NaN.

span=1 handling (the AUTHORING alpha=1 Do-rule): a *pure trade-flow* EMA read after the trade-clock
decay has no committed event at alpha=1 -- inject-then-decay zeroes both the signed and total kernel
sums, so `E/W` is 0/0. The feature is therefore *genuinely undefined* at span=1, and the rule's
fallback applies: a CONSISTENT NaN that BOTH builds agree on (the gates mask it), never a spurious inf
or a build-specific number. We sweep span=1 in the independent oracle and assert exactly that
consistency (vectorized == streaming == NaN, everywhere), then run the streaming/vectorized parity on
the spans where the feature is defined (>=2).
"""
import numpy as np
import pytest

import boba.io as io
from boba.features import base
import boba.features.trade_flow_imbalance  # noqa: F401  (registers SPEC)
from boba.research.screening import RawEventStream, ScreeningContext, parity_check


# --------------------------------------------------------------------------------------------------
# synthetic market — its own book / trade / raw-event / context builders (no production helpers)
# --------------------------------------------------------------------------------------------------
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
    """Pack per-venue books (kind 0) and trades (kind 1) into the receive-ordered RawEventStream."""
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
    return ScreeningContext(
        block="syn", coin="x", target=target, sources=tuple(sources), horizon_ns=0,
        yardstick_span=25, mid_stream={ex: "front_levels" for ex in exes},
        merged_ts=merged_ts, anchor_ts=anchor_ts,
        tick_at_anchor=np.searchsorted(merged_ts, anchor_ts, "right") - 1,
        sigma_at_anchor=np.empty(0), lam_at_anchor=np.empty(0),
        price_target=np.empty(0), rate_target=np.empty(0), base=[], vol_level=np.empty(0),
        rate_level=np.empty(0), vol_regime=np.empty(0),
        raw_events=_raw_events({ex: books[ex] for ex in exes}, {ex: trades[ex] for ex in exes}),
        _mids={ex: mids[ex] for ex in exes},
        _books={ex: (books[ex][0], books[ex][1], books[ex][2], books[ex][3], books[ex][4]) for ex in exes},
        _trades={ex: trades[ex] for ex in exes},
    )


# --------------------------------------------------------------------------------------------------
# INDEPENDENT oracle — an explicit per-event loop, no shared code with the production build.
#
# Implementable from the feature's written definition alone: per receive timestamp sum the trade flow
# into signed_volume = sum(side*qty) and total_volume = sum(qty) (dropping bad prc<=0 / qty<=0 prints),
# then walk every event/trade timestamp in order maintaining two trade-clock EMA *sums* E_signed and
# E_value (inject `alpha*x` at a timestamp carrying the flow, decay by `1-alpha` once per trade
# timestamp), and read the ratio E_signed / E_value after the last event at-or-before each anchor.
# The common alpha*(1-alpha)^k weights cancel in the ratio, so this is the per-event volume mean of the
# signed fraction, bounded in [-1, 1]. VOLUME-weighted (qty, not px*qty) so the feature is exactly odd.
# --------------------------------------------------------------------------------------------------
def _kernel_sum(event_rx, values, merged_ts, anchor_ts, span):
    """Trade-clock EMA *sum* of `values` (aligned to `event_rx`) read at each anchor: inject alpha*x at
    an event timestamp, decay by (1-alpha) once per trade timestamp, last-state hold between anchors."""
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


def _volume_imbalance_ref(trades, merged_ts, anchor_ts, span):
    """EMA(signed_volume) / EMA(total_volume) in [-1, 1] by the dead-simple loop above. VOLUME-weighted."""
    rx, px, lifts, qty = trades
    ok = (px > 0.0) & (qty > 0.0) & np.isfinite(px) & np.isfinite(qty) & np.isfinite(lifts)
    rx, lifts, qty = rx[ok], lifts[ok], qty[ok]
    signed = np.where(lifts > 0.0, qty, -qty)     # side * qty: +qty buy lifts ask, -qty sell hits bid
    uniq, inv = np.unique(rx, return_inverse=True)
    signed_sum = np.bincount(inv, weights=signed)  # per-timestamp summed flow
    value_sum = np.bincount(inv, weights=qty)
    num = _kernel_sum(uniq, signed_sum, merged_ts, anchor_ts, span)
    den = _kernel_sum(uniq, value_sum, merged_ts, anchor_ts, span)
    return num / np.where(den == 0.0, np.nan, den)


# the parity/oracle span sweep, on spans where the feature is DEFINED (span=1 is handled separately,
# below, as a consistent-NaN degenerate case per the AUTHORING alpha=1 rule).
_SPANS = (2, 50)
# the full sweep including span=1 (alpha=1, the no-smoothing fast leg) — swept in the oracle and the
# consistency check that both builds agree on a NaN there.
_SPANS_WITH_1 = (1,) + _SPANS


# --------------------------------------------------------------------------------------------------
# (a) vectorized vs the independent volume-weighted oracle, including the span=1 leg
# --------------------------------------------------------------------------------------------------
def test_vectorized_matches_independent_oracle():
    books, trades, merged_ts, anchor_ts = _synthetic_market()
    ctx = _ctx(books, trades, merged_ts, anchor_ts)
    spec = base.get("trade_flow_imbalance")
    for n in _SPANS:
        got = spec.vectorized(ctx, n)["byb"]
        ref = _volume_imbalance_ref(trades["byb"], merged_ts, anchor_ts, n)
        ok = np.isfinite(got) & np.isfinite(ref)
        assert ok.sum() > 100, f"too few finite points at span={n}"
        np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-9, atol=1e-12)
        # consistent NaN where undefined: both builds NaN at exactly the same anchors
        np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))


def test_value_is_bounded_and_finite_where_defined():
    """A volume-weighted signed fraction lives in [-1, 1]; and wherever the inputs exist (the oracle is
    finite) the production value must be FINITE, never inf/nan -- the AUTHORING Do-rule."""
    books, trades, merged_ts, anchor_ts = _synthetic_market(seed=8)
    ctx = _ctx(books, trades, merged_ts, anchor_ts)
    spec = base.get("trade_flow_imbalance")
    for n in _SPANS:
        for ex in ("byb", "bin", "okx"):
            got = spec.vectorized(ctx, n)[ex]
            ref = _volume_imbalance_ref(ctx._trades[ex], merged_ts, anchor_ts, n)
            defined = np.isfinite(ref)
            assert defined.sum() > 100
            # finite wherever the oracle is defined (no spurious inf/nan)
            assert np.all(np.isfinite(got[defined])), f"non-finite value at span={n}, ex={ex}"
            # bounded in [-1, 1] (a tiny float epsilon for round-off)
            assert np.all(got[defined] >= -1.0 - 1e-9)
            assert np.all(got[defined] <= 1.0 + 1e-9)


def test_span1_is_a_consistent_nan_both_builds_agree_on():
    """span=1 => alpha=1: a pure trade-flow EMA read AFTER the trade-clock decay has no committed event
    (inject-then-decay zeroes E and W), so E/W is the genuinely-undefined 0/0. Per the AUTHORING alpha=1
    rule this must be a CONSISTENT NaN both builds agree on -- never an inf or a build-specific number.
    Assert the vectorized build and the independent oracle are NaN everywhere at span=1 (no spurious
    finite/inf value), and that the streaming build matches it (consistent NaN -> 0 comparable points)."""
    books, trades, merged_ts, anchor_ts = _synthetic_market(seed=9)
    ctx = _ctx(books, trades, merged_ts, anchor_ts)
    spec = base.get("trade_flow_imbalance")
    for ex in ("byb", "bin", "okx"):
        got = spec.vectorized(ctx, 1)[ex]
        ref = _volume_imbalance_ref(ctx._trades[ex], merged_ts, anchor_ts, 1)
        assert np.all(np.isnan(got)), "span=1 must be a consistent NaN, never a spurious finite/inf"
        assert np.all(np.isnan(ref))
    # the streaming build agrees (it too reads 0/0 after the alpha=1 decay): a consistent NaN, so the
    # parity driver finds 0 comparable points and no disagreement on any finite value.
    rep = parity_check(ctx, spec, [1], n_grid=len(ctx.anchor_ts), tol=1e-9)
    assert all(n == 0 for n in rep.n_points.values()), "span=1 streaming should also be all-NaN"
    assert all(np.isnan(d) for d in rep.max_diff.values()), "no finite disagreement at span=1"


# --------------------------------------------------------------------------------------------------
# (b) mirror COMMUTATION invariant against the FULL book/tape reflection
#     reflect every trade price px -> c**2/px AND flip every side (buy<->sell); qty is invariant.
#     Price is unused by the feature (a no-op), the side flip negates the signed flow -> ODD feature,
#     so spec.mirror(feature(books)) == feature(mirror_books(books)) exactly.
# --------------------------------------------------------------------------------------------------
def _mirror_trades(trades, c=100.0):
    rx, px, lifts, qty = trades
    return (rx, c * c / px, 1.0 - lifts, qty)   # reflect price, flip side (lifts_ask<->hits_bid), qty same


def test_mirror_commutes_with_full_book_reflection():
    books, trades, merged_ts, anchor_ts = _synthetic_market(seed=4)
    ctx = _ctx(books, trades, merged_ts, anchor_ts)
    mctx = _ctx(books, trades, merged_ts, anchor_ts)
    mctx._trades = {ex: _mirror_trades(t) for ex, t in ctx._trades.items()}  # FULL reflection: price + side
    spec = base.get("trade_flow_imbalance")
    for n in _SPANS:
        feat = spec.vectorized(ctx, n)
        refl = spec.vectorized(mctx, n)
        for ex in ("byb", "bin", "okx"):
            lhs = spec.mirror(feat[ex])               # declared closed-form reflection (np.negative)
            ok = np.isfinite(lhs) & np.isfinite(refl[ex])
            assert ok.sum() > 100, f"too few finite points at span={n}, ex={ex}"
            np.testing.assert_allclose(lhs[ok], refl[ex][ok], rtol=1e-6, atol=1e-12)


# --------------------------------------------------------------------------------------------------
# (c) per-venue fan-out — one independent leg per exchange (target + each source)
# --------------------------------------------------------------------------------------------------
def test_fans_out_one_independent_leg_per_exchange():
    books, trades, merged_ts, anchor_ts = _synthetic_market(seed=3)
    ctx = _ctx(books, trades, merged_ts, anchor_ts)
    spec = base.get("trade_flow_imbalance")
    out = spec.vectorized(ctx, 20)
    assert set(out) == {"byb", "bin", "okx"}                    # one leg per exchange
    for ex in ("byb", "bin", "okx"):
        # each leg is that venue's own-tape build, independent of which other sources are present
        sources = tuple(s for s in ("bin", "okx") if s == ex)
        solo_ctx = _ctx(books, trades, merged_ts, anchor_ts, sources=sources)
        np.testing.assert_array_equal(out[ex], spec.vectorized(solo_ctx, 20)[ex])


# --------------------------------------------------------------------------------------------------
# streaming parity on synthetic data — the O(1) build reproduces the vectorized one (span=1 included)
# --------------------------------------------------------------------------------------------------
def test_synthetic_streaming_parity():
    books, trades, merged_ts, anchor_ts = _synthetic_market(seed=5)
    ctx = _ctx(books, trades, merged_ts, anchor_ts)
    spec = base.get("trade_flow_imbalance")
    rep = parity_check(ctx, spec, list(_SPANS), n_grid=len(ctx.anchor_ts), tol=1e-9)
    assert rep.passed, str(rep)
    # every leg has real comparable points (the build is defined at these spans)
    assert all(n > 100 for n in rep.n_points.values()), str(rep)


# --------------------------------------------------------------------------------------------------
# real-block parity (skipped without DATA_DIR) — streaming reproduces vectorized on arbitrary-ns timing
# --------------------------------------------------------------------------------------------------
@pytest.mark.skipif(getattr(io, "DATA_DIR", None) is None, reason="no DATA_DIR configured")
def test_real_block_parity():
    """Streaming reproduces vectorized on a real block (arbitrary-ns event timing). Parity on the spans
    where the feature is defined (>=2); span=1 is a consistent NaN both builds agree on (asserted)."""
    from boba.research.screening import build_context

    ctx = build_context(hours=2)
    spec = base.get("trade_flow_imbalance")
    rep = parity_check(ctx, spec, [2, 100], tol=1e-6)
    assert rep.passed, str(rep)
    assert all(n > 0 for n in rep.n_points.values()), str(rep)
    # span=1 (alpha=1, the most-used fast leg): genuinely undefined for a pure trade-flow EMA read after
    # the decay -> a consistent NaN both builds agree on (0 comparable points, no finite disagreement).
    rep1 = parity_check(ctx, spec, [1], tol=1e-6)
    assert all(n == 0 for n in rep1.n_points.values()), str(rep1)
    assert all(np.isnan(d) for d in rep1.max_diff.values()), str(rep1)
    for ex in spec.keys_for(ctx, 1):
        assert np.all(np.isnan(spec.vectorized(ctx, 1)[ex])), f"span=1 should be all-NaN on real block ({ex})"
