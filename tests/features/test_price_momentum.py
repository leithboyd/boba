"""Self-contained test suite for `boba.features.price_momentum` — a per-venue EMA of log-mid moves
in target volatility units: `EMA(Δlog mid) / σ_ev` (see the module docstring). It fans out over every
exchange (target + sources), `params = N` (the EMA span), and is ODD under price reflection so
`SPEC.mirror = np.negative`.

The feature is now a standalone transform `vectorized(raw_data, shared_data, config, N) -> {ex -> value
per shared_data.event_ts}`. The AUTHORING.md validation trio + streaming coverage, on synthetic data with
NO shared code with production:

  - `test_vectorized_matches_oracle`     vectorized vs an INDEPENDENT dead-simple event-loop oracle, on
                                         the event grid, across spans incl. span=1.
  - `test_fans_out_over_all_exchanges`   one independent leg per venue (target + each source).
  - `test_mirror_commutes_with_reflection`  the COMMUTATION invariant against the FULL book reflection.
  - `test_span1_finite_where_inputs_exist`  span=1 (α=1) is finite wherever a live move exists, NaN only
                                            during warm-up — the AUTHORING span=1 Do-rule.
  - `test_streaming_matches_vectorized`  drive `LivePriceMomentum` over a synthetic raw-event stream and
                                         assert it matches the (sampled) vectorized via `parity_check`.
  - `test_real_block_parity`             streaming-vs-vectorized parity on a real block (DATA_DIR-gated).
"""
import numpy as np
import pytest

import boba.io as io
from boba.features import base
import boba.features.price_momentum  # noqa: F401  (registers the spec)
from boba.features.base import (
    Config, FrontLevels, ListingRaw, ListingShared, RawData, Series, SharedData, Trade)
from boba.features.shared import build_shared_data
from boba.research.screening import RawEventStream, ScreeningContext, parity_check


# --------------------------------------------------------------------------------------------------
# synthetic standalone inputs (own builders — no shared code with production)
# --------------------------------------------------------------------------------------------------
def _raw_data(books, trades, coin="x") -> RawData:
    """`{short_ex -> book}` + `{short_ex -> trade}` -> RawData. Each book is `(rx, bid, bid_qty, ask,
    ask_qty)` and each trade is `(rx, px, lifts_ask, qty)`. Every listing uses a front_levels mid;
    exchange_time is set to rx (the synthetic books carry no fusion latency)."""
    listings: dict[str, ListingRaw] = {}
    for ex, (rx, bid, bq, ask, aq) in books.items():
        rx = rx.astype(np.int64)
        front = FrontLevels(rx, rx, bid, bq, ask, aq)
        trx, tpx, tlifts, tqty = trades[ex]
        trx = trx.astype(np.int64)
        trade = Trade(trx, trx, tpx.astype(float), tlifts.astype(float), tqty.astype(float))
        listings[f"{ex}_{coin}"] = ListingRaw(front_levels=front, trade=trade)
    return RawData(listings=listings)


def _inputs(books, trades, target_ex="byb", sources=(), coin="x"):
    """-> (raw_data, shared_data, config) for the standalone feature build."""
    raw = _raw_data(books, trades, coin)
    config = Config(f"{target_ex}_{coin}", tuple(f"{s}_{coin}" for s in sources), coin,
                    {f"{ex}_{coin}": "front_levels" for ex in books}, yardstick_span=_YARD)
    return raw, build_shared_data(raw, config), config


def _raw_events(books, trades, coin="x") -> RawEventStream:
    """Per-venue raw book rows + per-venue trades, for the parity driver."""
    listings = tuple(f"{ex}_{coin}" for ex in books)
    cols: dict[str, list] = {k: [] for k in "rx kind lid t a b c d".split()}

    def add(rx, kind, lid, t, a, b, c, d):
        n = len(rx)
        cols["rx"].append(rx.astype(np.int64)); cols["kind"].append(np.full(n, kind, np.int8))
        cols["lid"].append(np.full(n, lid, np.int8)); cols["t"].append(t.astype(np.int64))
        for k, v in (("a", a), ("b", b), ("c", c), ("d", d)):
            cols[k].append(v.astype(float))

    for lid, (ex, (rx, bid, bq, ask, aq)) in enumerate(books.items()):
        add(rx, 0, lid, rx, bid, ask, bq, aq)                  # BookEvent(listing, rx, exch_time, bid, ask, bid_qty, ask_qty); fixture sets exch_time == rx
    for lid, ex in enumerate(books):
        rx, px, lifts, qty = trades[ex]
        add(rx, 1, lid, rx, px, lifts, qty, np.full(len(rx), np.nan))   # TradeEvent(listing, rx, exch_time, px, lifts_ask, qty); fixture sets exch_time == rx
    C = {k: np.concatenate(v) for k, v in cols.items()}
    order = np.lexsort((C["kind"], C["rx"]))
    return RawEventStream(*(C[k][order] for k in "rx kind lid t a b c d".split()), listings)


def _ctx_for_parity(books, trades, anchor_ts, target_ex="byb", sources=()) -> ScreeningContext:
    """A minimal ScreeningContext carrying the standalone inputs + the raw-event stream + an anchor grid,
    enough for `parity_check` (streaming read at anchors vs the sampled vectorized)."""
    raw, shared, config = _inputs(books, trades, target_ex, sources)
    return ScreeningContext(
        block="syn", coin="x", target=config.target_listing, sources=config.other_listings, horizon_ns=0,
        yardstick_span=_YARD, mid_stream={}, merged_ts=shared.clock, anchor_ts=anchor_ts,
        sigma_at_anchor=np.empty(0), lam_at_anchor=np.empty(0), price_target=np.empty(0),
        rate_target=np.empty(0), base=[], vol_level=np.empty(0), rate_level=np.empty(0), vol_regime=np.empty(0),
        raw_events=_raw_events(books, trades), raw_data=raw, shared_data=shared, config=config)


# --------------------------------------------------------------------------------------------------
# synthetic market — books (per-venue BBO, with same-timestamp bursts so the "collapse same-ts to the
# final mid" rule is exercised) + trades (per venue; their union is the shared decay clock).
# --------------------------------------------------------------------------------------------------
_YARD = 25  # yardstick span used for σ_ev


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


def _market(seed=0):
    """Three venues (byb target, bin/okx sources). Returns books, trades, and the union trade clock."""
    exes = ("byb", "bin", "okx")
    books = {ex: _synthetic_book(seed + i) for i, ex in enumerate(exes)}
    trades = {ex: _synthetic_trades(seed + 10 + i) for i, ex in enumerate(exes)}
    trade_ts = np.unique(np.concatenate([trades[ex][0] for ex in exes]))
    return books, trades, trade_ts


# --------------------------------------------------------------------------------------------------
# INDEPENDENT, dead-simple oracle — implementable from the feature's written definition alone; shares
# NO code with the production build (plain numpy + explicit per-timestamp loop, single-threaded).
# --------------------------------------------------------------------------------------------------
def _book_mid(book):
    """A book's mid stream `(rx, (bid+ask)/2)`."""
    rx, bid, _bq, ask, _aq = book
    return rx, 0.5 * (bid + ask)


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


def _kernel_ew(event_rx, values, trade_ts, grid, span):
    """E and W of a sparse flow by an explicit loop: α = 2/(span+1). At each event-timestamp inject
    `α·(summed value)` into E and `α·(count)` into W; at each trade-timestamp decay both by (1−α). Walk
    all timestamps in order (inject-then-decay within a shared timestamp) and read the running E, W at the
    last event at-or-before each grid point. Returns (E, W) arrays aligned to `grid`."""
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
        ts_arr.append(ts); Es.append(E); Ws.append(W)
    E_out = np.full(len(grid), np.nan)
    W_out = np.full(len(grid), np.nan)
    if not ts_arr:
        return E_out, W_out
    ts_arr, Es, Ws = np.array(ts_arr), np.array(Es), np.array(Ws)
    idx = np.searchsorted(ts_arr, grid, "right") - 1
    ok = idx >= 0
    E_out[ok], W_out[ok] = Es[idx[ok]], Ws[idx[ok]]
    return E_out, W_out


def _kernel_mean_oracle(event_rx, values, trade_ts, grid, span):
    """Self-normalising per-event mean E/W (NaN where W == 0 — a consistent undefined)."""
    E, W = _kernel_ew(event_rx, values, trade_ts, grid, span)
    return E / np.where(W == 0.0, np.nan, W)


def _sigma_oracle(target_mid_stream, trade_ts, grid, span):
    """σ_ev oracle: sqrt of the per-event mean of squared target log-moves (the yardstick the shared
    `build_shared_data` / streaming `VolYardstick` builds), at the yardstick span."""
    rx, ret = _move_stream_oracle(*target_mid_stream)
    return np.sqrt(_kernel_mean_oracle(rx, ret * ret, trade_ts, grid, span))


def _price_momentum_oracle(venue_book, target_book, trade_ts, grid, span):
    """price_momentum for one venue: EMA(Δlog mid) / σ_ev, by the dead-simple kernel loop, on `grid`."""
    sigma = _sigma_oracle(_book_mid(target_book), trade_ts, grid, _YARD)
    rx, ret = _move_stream_oracle(*_book_mid(venue_book))
    return _kernel_mean_oracle(rx, ret, trade_ts, grid, span) / sigma


# --------------------------------------------------------------------------------------------------
# (a) vectorized vs the independent oracle, across spans incl. span=1
# --------------------------------------------------------------------------------------------------
def test_vectorized_matches_oracle():
    books, trades, trade_ts = _market(seed=1)
    raw, shared, config = _inputs(books, trades, sources=("bin", "okx"))
    spec = base.get("price_momentum")
    for n in (1, 2, 50):                                  # span=1 included in the sweep
        got = spec.vectorized(raw, shared, config, n)["byb"]
        ref = _price_momentum_oracle(books["byb"], books["byb"], trade_ts, shared.event_ts, n)
        ok = np.isfinite(got) & np.isfinite(ref)
        assert ok.sum() > 100
        np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-9, atol=1e-12)
        # consistent undefined: wherever the oracle is NaN, the build is NaN too (never a stray inf)
        np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
        assert not np.any(np.isinf(got))


# --------------------------------------------------------------------------------------------------
# (b) per-venue fan-out: one independent leg per exchange, each the single-venue own-book build
# --------------------------------------------------------------------------------------------------
def test_fans_out_over_all_exchanges():
    books, trades, trade_ts = _market(seed=3)
    raw, shared, config = _inputs(books, trades, sources=("bin", "okx"))
    spec = base.get("price_momentum")
    out = spec.vectorized(raw, shared, config, 20)
    assert set(out) == {"byb", "bin", "okx"}
    assert tuple(spec.keys_for(config, 20)) == ("byb", "bin", "okx")
    for ex in ("byb", "bin", "okx"):                          # each leg's MOVE stream is its own book
        ref = _price_momentum_oracle(books[ex], books["byb"], trade_ts, shared.event_ts, 20)
        ok = np.isfinite(out[ex]) & np.isfinite(ref)
        assert ok.sum() > 100
        np.testing.assert_allclose(out[ex][ok], ref[ok], rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------------------------------
# (c) the mirror COMMUTATION invariant against the full book reflection
# --------------------------------------------------------------------------------------------------
def _mirror_shared(shared: SharedData, c=100.0) -> SharedData:
    """Reflect every listing's mid stream through level c — the AUTHORING `mid reflects` row,
    `mid -> c**2/mid` (`log mid -> 2 log c - log mid`, so `Δlog mid -> -Δlog mid`). price_momentum reads
    the mid (not bid/ask separately), so this reflected-mid `SharedData` is the exact input it consumes
    under the book reflection. σ_ev is EVEN (squared moves are unchanged) -> the yardsticks are held."""
    listings = {l: ListingShared(mid=Series(np.asarray(ls.mid.rx), c * c / np.asarray(ls.mid.value)))
                for l, ls in shared.listings.items()}
    return SharedData(event_ts=shared.event_ts, clock=shared.clock, vol_yardstick=shared.vol_yardstick,
                      rate_yardstick=shared.rate_yardstick, listings=listings)


def test_mirror_commutes_with_reflection():
    books, trades, trade_ts = _market(seed=4)
    raw, shared, config = _inputs(books, trades, sources=("bin", "okx"))
    mshared = _mirror_shared(shared)                             # reflect the mids; σ_ev even -> held
    spec = base.get("price_momentum")
    assert spec.mirror is not None
    feat = spec.vectorized(raw, shared, config, 50)               # feature(mids)
    refl = spec.vectorized(raw, mshared, config, 50)             # feature(reflect(mids))
    for ex in ("byb", "bin", "okx"):
        lhs = spec.mirror(feat[ex])
        ok = np.isfinite(lhs) & np.isfinite(refl[ex])
        assert ok.sum() > 100
        np.testing.assert_allclose(lhs[ok], refl[ex][ok], rtol=1e-6, atol=1e-12)


# --------------------------------------------------------------------------------------------------
# span = 1 (α = 1): finite wherever a live move exists, consistent NaN only where undefined.
# Anchors placed ON byb move-timestamps that are NOT trade ticks: the move's sample is still alive
# (not yet decayed), so even at span=1 (α=1) the value is DEFINED and FINITE.
# --------------------------------------------------------------------------------------------------
def test_span1_finite_where_inputs_exist():
    books, trades, trade_ts = _market(seed=6)
    raw, shared, config = _inputs(books, trades, sources=("bin", "okx"))
    spec = base.get("price_momentum")
    got = spec.vectorized(raw, shared, config, 1)["byb"]         # span=1, α=1, the most-used fast leg
    ref = _price_momentum_oracle(books["byb"], books["byb"], trade_ts, shared.event_ts, 1)
    assert np.isfinite(got).sum() > 100                          # the span=1 leg is well-populated
    assert not np.isinf(got).any()                               # never a spurious inf at α=1
    # finite EXACTLY where the independent oracle is (a consistent NaN both builds agree on)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
    fin = np.isfinite(ref)
    np.testing.assert_allclose(got[fin], ref[fin], rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------------------------------
# streaming (LivePriceMomentum) vs vectorized — synthetic parity, including a span=1 leg
# --------------------------------------------------------------------------------------------------
def _span1_anchors(books, trade_ts):
    """Anchors that UNION interior trade ticks with byb move-only timestamps (moves that are NOT trade
    ticks). At a move-only anchor the move's sample is still alive (not yet decayed), so even span=1
    (α=1) is DEFINED there — giving the parity sweep real span=1 coverage (a trade-tick anchor decays
    span=1's E,W to 0 -> a consistent NaN, the undefined case, not a bug)."""
    rx, _ = _move_stream_oracle(*_book_mid(books["byb"]))
    trset = set(int(t) for t in trade_ts)
    move_only = np.unique([int(t) for t in rx if int(t) not in trset])
    move_only = move_only[(move_only > trade_ts[120]) & (move_only < trade_ts[-120])]
    return np.unique(np.concatenate([move_only, trade_ts[150:-150:7]]))


def test_streaming_matches_vectorized():
    books, trades, trade_ts = _market(seed=7)
    anchor_ts = _span1_anchors(books, trade_ts)                  # move-only anchors give span=1 coverage
    ctx = _ctx_for_parity(books, trades, anchor_ts, sources=("bin", "okx"))
    rep = parity_check(ctx, base.get("price_momentum"), [1, 2, 50], n_grid=len(anchor_ts), tol=1e-9)
    assert rep.passed, str(rep)


# --------------------------------------------------------------------------------------------------
# real-block parity (skipped without DATA_DIR) — streaming reproduces vectorized, incl. span=1
# --------------------------------------------------------------------------------------------------
@pytest.mark.skipif(getattr(io, "DATA_DIR", None) is None, reason="no DATA_DIR configured")
def test_real_block_parity():
    from boba.research.screening import build_context

    ctx = build_context(hours=2)
    rep = parity_check(ctx, base.get("price_momentum"), [1, 2, 100], tol=1e-6)
    assert rep.passed, str(rep)
