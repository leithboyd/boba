"""Per-feature test suite for `boba.features.price_momentum`.

`price_momentum` is, per venue, a sparse-flow `KernelMeanEMA` of non-zero `Δlog(mid)` moves divided
by the target volatility yardstick `σ_ev` (see the module docstring): `EMA(Δlog mid) / σ_ev`. It fans
out over every exchange (target + sources), `params = N` (the EMA span), and is ODD under price
reflection so `SPEC.mirror = np.negative`.

This file is SELF-CONTAINED — its own synthetic `ScreeningContext` / book / trade builders and its own
INDEPENDENT, dead-simple oracle (an explicit per-event loop, sharing NO code with the production
build). It covers the AUTHORING.md validation trio plus the span=1 leg:

  (a) vectorized build vs the independent oracle on synthetic data, INCLUDING span=1;
  (b) the mirror commutation invariant `mirror(feature(books)) == feature(mirror_books(books))`
      against the full book reflection (reflect prices, swap+flip sides), to round-off;
  (c) the per-venue fan-out (one independent leg per exchange = its own-book build);
  (d) span=1 (α=1): a build where every anchor has a defined input, asserting the value is FINITE
      everywhere (never inf / build-specific), with a consistent NaN where undefined;
  (e) synthetic streaming parity, plus a real-block parity test (skipped without DATA_DIR).
"""
import numpy as np
import pytest

import boba.io as io
from boba.features import base
import boba.features.price_momentum  # noqa: F401  (registers SPEC)
from boba.research.screening import RawEventStream, ScreeningContext, parity_check


# --------------------------------------------------------------------------------------------------
# synthetic market — books (per-venue BBO) + trades (the shared trade clock), with same-timestamp
# bursts so the "collapse same-ts to the final mid" rule is exercised.
# --------------------------------------------------------------------------------------------------
def _synthetic_book(seed=0, n=2200):
    rng = np.random.default_rng(seed)
    rx = (np.arange(1, n + 1) * 10).astype(np.int64)             # a book row every 10 ns
    mid = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 8e-5))
    hs = 0.005 + 0.005 * rng.random(n)
    bid, ask = mid - hs, mid + hs
    bq, aq = rng.uniform(1.0, 100.0, n), rng.uniform(1.0, 100.0, n)
    rx[100:106] = rx[100]       # same-timestamp level burst -> only the final mid should count
    return rx, bid, bq, ask, aq


def _synthetic_trades(seed=0, n=1800):
    rng = np.random.default_rng(seed)
    rx = (np.arange(1, n + 1) * 13 + seed % 7).astype(np.int64)  # a trade every ~13 ns (distinct grid)
    px = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 7e-5))
    lifts = (rng.random(n) > 0.48).astype(float)
    qty = rng.uniform(0.1, 5.0, n)
    rx[50:55] = rx[50]
    rx[250:254] = rx[250]
    return rx, px, lifts, qty


def _raw_events(books, trades, coin="x"):
    """Pack per-venue books + trades into the integer-coded `RawEventStream` the parity driver replays."""
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
        add(rx, 0, lid, rx, bid, ask, bq, aq)            # book: (bid, ask, bid_qty, ask_qty)
    for lid, ex in enumerate(books):
        rx, px, lifts, qty = trades[ex]
        add(rx, 1, lid, rx, px, lifts, qty, np.full(len(rx), np.nan))  # trade: (px, lifts_ask, qty, nan)

    C = {k: np.concatenate(v) for k, v in cols.items()}
    order = np.lexsort((C["kind"], C["rx"]))
    return RawEventStream(*(C[k][order] for k in "rx kind lid t a b c d".split()), listings)


# --------------------------------------------------------------------------------------------------
# INDEPENDENT, dead-simple oracle — implementable from the feature's written definition alone; shares
# NO code with the production build (plain numpy + explicit per-timestamp loop, single-threaded).
# --------------------------------------------------------------------------------------------------
def _move_stream_oracle(rx, mid):
    """`(rx, mid)` level rows -> non-zero `(ts, Δlog mid)`, keeping only the final mid per timestamp."""
    rx, mid = np.asarray(rx), np.asarray(mid)
    ok = (mid > 0.0) & np.isfinite(mid)
    rx, mid = rx[ok], mid[ok]
    if len(rx) < 2:
        return rx[:0], mid[:0].astype(float)
    keep = np.concatenate([rx[1:] != rx[:-1], [True]])   # last row per timestamp
    rx, mid = rx[keep], mid[keep]
    if len(rx) < 2:
        return rx[:0], mid[:0].astype(float)
    dlog = np.diff(np.log(mid))
    move = dlog != 0.0
    return rx[1:][move], dlog[move]


def _kernel_ew(event_rx, values, trade_ts, anchor_ts, span):
    """E and W of a sparse flow by an explicit loop: α = 2/(span+1). At each event-timestamp inject
    `α·(summed value)` into E and `α` into W; at each trade-timestamp decay both by (1−α). Walk all
    timestamps in order (inject-then-decay within a shared timestamp) and read the running E, W at the
    last event at-or-before each anchor. Returns (E, W) arrays aligned to `anchor_ts`."""
    a = 2.0 / (span + 1.0)
    beta = 1.0 - a
    inj_E: dict[int, float] = {}
    inj_W: dict[int, int] = {}
    for ts, val in zip(event_rx, values):
        inj_E[int(ts)] = inj_E.get(int(ts), 0.0) + float(val)   # SUM same-timestamp records
        inj_W[int(ts)] = inj_W.get(int(ts), 0) + 1
    trades = set(int(t) for t in trade_ts)
    all_ts = sorted(set(inj_E) | trades)
    E = W = 0.0
    ts_arr, Es, Ws = [], [], []
    for ts in all_ts:
        if ts in inj_E:
            E += a * inj_E[ts]
            W += a * inj_W[ts]
        if ts in trades:
            E *= beta
            W *= beta
        ts_arr.append(ts)
        Es.append(E)
        Ws.append(W)
    E_out = np.full(len(anchor_ts), np.nan)
    W_out = np.full(len(anchor_ts), np.nan)
    if not ts_arr:
        return E_out, W_out
    ts_arr, Es, Ws = np.array(ts_arr), np.array(Es), np.array(Ws)
    idx = np.searchsorted(ts_arr, anchor_ts, "right") - 1
    ok = idx >= 0
    E_out[ok], W_out[ok] = Es[idx[ok]], Ws[idx[ok]]
    return E_out, W_out


def _kernel_mean_oracle(event_rx, values, trade_ts, anchor_ts, span):
    """Self-normalising per-event mean E/W (NaN where W == 0 — a consistent undefined)."""
    E, W = _kernel_ew(event_rx, values, trade_ts, anchor_ts, span)
    return E / np.where(W == 0.0, np.nan, W)


def _sigma_oracle(target_mid_stream, trade_ts, anchor_ts, span):
    """σ_ev oracle: sqrt of the per-event mean of squared target log-moves (the yardstick the
    streaming `LiveYardstick` builds), at the yardstick span."""
    rx, ret = _move_stream_oracle(*target_mid_stream)
    return np.sqrt(_kernel_mean_oracle(rx, ret * ret, trade_ts, anchor_ts, span))


def _price_momentum_oracle(venue_mid_stream, sigma, trade_ts, anchor_ts, span):
    """price_momentum for one venue: EMA(Δlog mid) / σ_ev, by the dead-simple kernel loop."""
    rx, ret = _move_stream_oracle(*venue_mid_stream)
    return _kernel_mean_oracle(rx, ret, trade_ts, anchor_ts, span) / sigma


# --------------------------------------------------------------------------------------------------
# synthetic ScreeningContext assembly
# --------------------------------------------------------------------------------------------------
_YARD = 25  # yardstick span used for σ_ev


def _ctx(books, trades, trade_ts, anchor_ts, target="byb_x", sources=("bin", "okx")):
    """Build a synthetic `ScreeningContext`; target + sources define the venues the feature fans over.
    `σ_ev` at the anchors is precomputed by the independent oracle, exactly as production reads it off
    the context (the feature divides by `ctx.sigma_at_anchor`)."""
    exes = tuple(dict.fromkeys((target.split("_", 1)[0],) + tuple(sources)))
    mids = {ex: (b[0], 0.5 * (b[1] + b[3])) for ex, b in books.items()}
    target_ex = target.split("_", 1)[0]
    sigma = _sigma_oracle(mids[target_ex], trade_ts, anchor_ts, _YARD)
    return ScreeningContext(
        block="syn", coin="x", target=target, sources=tuple(sources), horizon_ns=0,
        yardstick_span=_YARD, mid_stream={ex: "front_levels" for ex in exes},
        merged_ts=trade_ts, anchor_ts=anchor_ts,
        tick_at_anchor=np.searchsorted(trade_ts, anchor_ts, "right") - 1,
        sigma_at_anchor=sigma, lam_at_anchor=np.empty(0),
        price_target=np.empty(0), rate_target=np.empty(0), base=[], vol_level=np.empty(0),
        rate_level=np.empty(0), vol_regime=np.empty(0),
        raw_events=_raw_events({ex: books[ex] for ex in exes}, {ex: trades[ex] for ex in exes}),
        _mids={ex: mids[ex] for ex in exes},
        _books={ex: (books[ex][0], books[ex][1], books[ex][2], books[ex][3], books[ex][4]) for ex in exes},
        _trades={ex: trades[ex] for ex in exes},
    )


def _market(seed=0):
    """Three venues (byb target, bin/okx sources); anchors are interior trade ticks."""
    exes = ("byb", "bin", "okx")
    books = {ex: _synthetic_book(seed + i) for i, ex in enumerate(exes)}
    trades = {ex: _synthetic_trades(seed + 10 + i) for i, ex in enumerate(exes)}
    trade_ts = np.unique(np.concatenate([trades[ex][0] for ex in exes]))
    anchor_ts = trade_ts[150:-150:7]
    return books, trades, trade_ts, anchor_ts


def _move_only_anchors(books, trades, trade_ts):
    """Anchors placed ON byb move-timestamps that are NOT trade ticks. At such an anchor the move's
    sample is still alive (not yet decayed), so even at span=1 (α=1) the value is DEFINED and FINITE —
    the leg used to assert the span=1 Do-rule. (At a trade-tick anchor span=1 has just decayed E,W to
    0 -> a consistent NaN; that is the undefined case, not a bug.)"""
    mids = (books["byb"][0], 0.5 * (books["byb"][1] + books["byb"][3]))
    rx, _ = _move_stream_oracle(*mids)
    trset = set(int(t) for t in trade_ts)
    move_only = np.unique([int(t) for t in rx if int(t) not in trset])
    return move_only[(move_only > trade_ts[120]) & (move_only < trade_ts[-120])]


def _market_span1_defined(seed=0):
    """Like `_market`, but the anchor grid UNION-s the interior trade ticks with byb move-only
    timestamps, so the span=1 (α=1) leg has many DEFINED anchors (not only trade ticks, where span=1
    decays to a consistent NaN). Lets the oracle/parity sweep include span=1 with real coverage."""
    books, trades, trade_ts, trade_anchors = _market(seed)
    move_only = _move_only_anchors(books, trades, trade_ts)
    anchor_ts = np.unique(np.concatenate([move_only, trade_anchors]))
    return books, trades, trade_ts, anchor_ts


# --------------------------------------------------------------------------------------------------
# (a) vectorized vs independent oracle — including the span=1 leg
# --------------------------------------------------------------------------------------------------
def test_price_momentum_vectorized_matches_oracle():
    books, trades, trade_ts, anchor_ts = _market_span1_defined(seed=1)
    ctx = _ctx(books, trades, trade_ts, anchor_ts)
    spec = base.get("price_momentum")
    for n in (1, 2, 50):                                  # span=1 included in the sweep
        got = spec.vectorized(ctx, n)["byb"]
        ref = _price_momentum_oracle(ctx._mids["byb"], ctx.sigma_at_anchor, trade_ts, anchor_ts, n)
        ok = np.isfinite(got) & np.isfinite(ref)
        assert ok.sum() > 100                             # incl. span=1 (anchors land on live moves)
        np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-9, atol=1e-12)
        # consistent undefined: wherever the oracle is NaN, the build is NaN too (never a stray inf)
        np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
        assert not np.any(np.isinf(got))


# --------------------------------------------------------------------------------------------------
# (d) span = 1 (α = 1): FINITE wherever the inputs exist (the AUTHORING span=1 Do-rule)
# --------------------------------------------------------------------------------------------------
def test_price_momentum_span1_finite_where_defined():
    books, trades, trade_ts, _ = _market(seed=1)
    anchors = _move_only_anchors(books, trades, trade_ts)
    assert len(anchors) > 100                              # plenty of defined-input anchors to check
    ctx = _ctx(books, trades, trade_ts, anchors)
    spec = base.get("price_momentum")

    got = spec.vectorized(ctx, 1)["byb"]                  # span=1, α=1, the most-used fast leg
    ref = _price_momentum_oracle(ctx._mids["byb"], ctx.sigma_at_anchor, trade_ts, anchors, 1)
    assert np.all(np.isfinite(got))                        # every anchor has a live move -> finite, no 0/0, no inf
    assert not np.any(np.isinf(got))
    ok = np.isfinite(ref)
    assert ok.sum() == len(anchors)
    np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------------------------------
# (c) per-venue fan-out — one independent leg per exchange, each = its own-book build
# --------------------------------------------------------------------------------------------------
def test_price_momentum_fans_out_over_all_exchanges():
    books, trades, trade_ts, anchor_ts = _market(seed=3)
    ctx = _ctx(books, trades, trade_ts, anchor_ts)
    spec = base.get("price_momentum")
    out = spec.vectorized(ctx, 20)
    assert set(out) == {"byb", "bin", "okx"}              # one leg per exchange
    for ex in ("byb", "bin", "okx"):
        sources = tuple(s for s in ("bin", "okx") if s == ex)   # solo context with byb target + ex
        solo_ctx = _ctx(books, trades, trade_ts, anchor_ts, sources=sources)
        np.testing.assert_array_equal(out[ex], spec.vectorized(solo_ctx, 20)[ex])


# --------------------------------------------------------------------------------------------------
# (b) mirror commutation invariant — reflect the FULL book through a fixed price level c (byb's mid),
# rebuild, and assert `mirror(feature(books)) == feature(mirror_books(books))`. For this ODD feature
# `mirror = np.negative`. Per AUTHORING the two sides swap and each price reflects (`p -> c**2/p`) and
# the **mid reflects** with them (`log mid -> 2log c - log mid`); price_momentum reads the mid, so the
# data-level op it must commute with is the reflected mid stream. σ_ev is even -> held fixed.
# --------------------------------------------------------------------------------------------------
def _mirror_book(book, c=100.0):
    """Reflect the L1 book through price level c: the two sides SWAP and reflect (`p -> c**2/p`), and
    each side's size follows its original level (new bid size = old ask size)."""
    rx, bid, bq, ask, aq = book
    return (rx, c * c / ask, aq, c * c / bid, bq)


def _mirror_mid_stream(mid_stream, c=100.0):
    """Reflect a venue's mid stream through level c — the AUTHORING `mid reflects` row, `mid -> c**2/mid`
    (`log mid -> 2log c - log mid`). This is the reflected mid the swapped+reflected book yields, and the
    exact input price_momentum consumes (it reads the mid, not bid/ask separately)."""
    rx, mid = mid_stream
    return (rx, c * c / mid)


def test_price_momentum_mirror_commutes_with_book_reflection():
    books, trades, trade_ts, anchor_ts = _market(seed=4)
    ctx = _ctx(books, trades, trade_ts, anchor_ts)
    # full book reflection (sides swap + each price reflects); the mid it yields is the reflected mid.
    mctx = _ctx({ex: _mirror_book(book) for ex, book in books.items()}, trades, trade_ts, anchor_ts)
    mctx._mids = {ex: _mirror_mid_stream(mid) for ex, mid in ctx._mids.items()}
    mctx.sigma_at_anchor = ctx.sigma_at_anchor            # σ_ev is even -> unchanged under reflection
    spec = base.get("price_momentum")
    feat = spec.vectorized(ctx, 50)                       # feature(books)
    refl = spec.vectorized(mctx, 50)                      # feature(mirror_books(books))
    for ex in ("byb", "bin", "okx"):
        lhs = spec.mirror(feat[ex])
        ok = np.isfinite(lhs) & np.isfinite(refl[ex])
        assert ok.sum() > 100
        np.testing.assert_allclose(lhs[ok], refl[ex][ok], rtol=1e-6, atol=1e-12)


# --------------------------------------------------------------------------------------------------
# (e) synthetic streaming parity — the streaming build reproduces the vectorized one (span=1 included)
# --------------------------------------------------------------------------------------------------
def test_price_momentum_synthetic_parity():
    books, trades, trade_ts, anchor_ts = _market_span1_defined(seed=5)   # anchors give span=1 coverage
    ctx = _ctx(books, trades, trade_ts, anchor_ts)
    spec = base.get("price_momentum")
    rep = parity_check(ctx, spec, [1, 2, 50], n_grid=len(ctx.anchor_ts), tol=1e-9)
    assert rep.passed, str(rep)


# --------------------------------------------------------------------------------------------------
# real-block parity (skipped without DATA_DIR) — streaming reproduces vectorized on real ns timing
# --------------------------------------------------------------------------------------------------
@pytest.mark.skipif(getattr(io, "DATA_DIR", None) is None, reason="no DATA_DIR configured")
def test_real_block_price_momentum_parity():
    from boba.research.screening import build_context

    ctx = build_context(hours=2)
    rep = parity_check(ctx, base.get("price_momentum"), [1, 2, 100], tol=1e-6)
    assert rep.passed, str(rep)
