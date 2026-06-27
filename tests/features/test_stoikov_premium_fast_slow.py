"""Self-contained test suite for `boba.features.stoikov_premium_fast_slow`.

The feature computes the SIZE-WEIGHTED MID premium -- the leading / first-order term of Stoikov's
micro-price (the imbalance-adjusted mid, NOT the full martingale estimator):

    microprice = (bid_qty * ask_prc + ask_qty * bid_prc) / (bid_qty + ask_qty)
    prem       = (microprice - mid) / mid

then smoothed two ways with `LiveFrontEMA` and returned as `fast - slow`, per venue.
Reference: Stoikov, S. (2018) 'The Micro-Price: A High-Frequency Estimator of Future Prices',
Quantitative Finance 18(12):1959-1966 (SSRN 2970694); Gatheral, J. & Oomen, R. (size-weighted mid).

The feature is now a standalone transform `vectorized(raw_data, shared_data, config, params) -> {ex ->
value per shared_data.event_ts}`. The AUTHORING.md validation trio + streaming coverage, on synthetic
data with NO shared code with the production build:
  (a) vectorized vs. an independent dead-simple event-loop oracle on the event grid;
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
from boba.features.base import Config, FrontLevels, ListingRaw, RawData, Trade
from boba.features.shared import build_shared_data
from boba.research.screening import RawEventStream, ScreeningContext, parity_check


# --------------------------------------------------------------------------------------------------
# synthetic standalone inputs (own builders — no shared code with production)
# --------------------------------------------------------------------------------------------------
def _raw_data(books, trade_ts, coin="x") -> RawData:
    """`{short_ex -> (rx, bid, bid_qty, ask, ask_qty)}` + a shared trade clock -> RawData. The decay clock
    lives on the FIRST listing's trades (the premium needs only the book, so the trade payload is dummy);
    every listing uses a front_levels mid. exchange_time is set to rx (irrelevant to the premium)."""
    listings: dict[str, ListingRaw] = {}
    nt = len(trade_ts)
    for i, (ex, (rx, bid, bq, ask, aq)) in enumerate(books.items()):
        rx = rx.astype(np.int64)
        front = FrontLevels(rx, rx, bid, bq, ask, aq)
        if i == 0:
            t = trade_ts.astype(np.int64)
            trade = Trade(t, t, np.full(nt, 100.0), np.ones(nt), np.ones(nt))
        else:
            trade = Trade(np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0), np.empty(0), np.empty(0))
        listings[f"{ex}_{coin}"] = ListingRaw(front_levels=front, trade=trade)
    return RawData(listings=listings)


def _inputs(books, trade_ts, target_ex="byb", sources=(), coin="x"):
    """-> (raw_data, shared_data, config) for the standalone feature build."""
    raw = _raw_data(books, trade_ts, coin)
    config = Config(f"{target_ex}_{coin}", tuple(f"{s}_{coin}" for s in sources), coin,
                    {f"{ex}_{coin}": "front_levels" for ex in books}, yardstick_span=10)
    return raw, build_shared_data(raw, config), config


def _raw_events(books, trade_ts, coin="x") -> RawEventStream:
    """Per-venue raw book rows + one dummy trade per trade-timestamp (listing 0), for the parity driver."""
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
    nt = len(trade_ts)
    add(trade_ts, 1, 0, trade_ts, np.ones(nt), np.ones(nt), np.ones(nt), np.full(nt, np.nan))
    C = {k: np.concatenate(v) for k, v in cols.items()}
    order = np.lexsort((C["kind"], C["rx"]))
    return RawEventStream(*(C[k][order] for k in "rx kind lid t a b c d".split()), listings)


def _ctx_for_parity(books, trade_ts, anchor_ts, target_ex="byb", sources=()) -> ScreeningContext:
    """A minimal ScreeningContext carrying the standalone inputs + the raw-event stream + an anchor grid,
    enough for `parity_check` (streaming read at anchors vs the sampled vectorized)."""
    raw, shared, config = _inputs(books, trade_ts, target_ex, sources)
    return ScreeningContext(
        block="syn", coin="x", target=config.target_listing, sources=config.other_listings, horizon_ns=0,
        yardstick_span=10, mid_stream={}, merged_ts=shared.clock, anchor_ts=anchor_ts,
        sigma_at_anchor=np.empty(0), lam_at_anchor=np.empty(0), price_target=np.empty(0),
        rate_target=np.empty(0), base=[], vol_level=np.empty(0), rate_level=np.empty(0), vol_regime=np.empty(0),
        raw_events=_raw_events(books, trade_ts), raw_data=raw, shared_data=shared, config=config)


def _synthetic_book(seed=0, n=4000):
    """A book row every 10 ns, with a same-ts burst so the level (last-state-wins) path is exercised.
    Returns `((rx, bid, bid_qty, ask, ask_qty), trade_ts)`."""
    rng = np.random.default_rng(seed)
    rx = (np.arange(1, n + 1) * 10).astype(np.int64)                 # a book row every 10 ns
    mid = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 1e-4))
    hs = 0.005 + 0.005 * rng.random(n)
    bid, ask = mid - hs, mid + hs
    bq, aq = rng.uniform(1.0, 100.0, n), rng.uniform(1.0, 100.0, n)
    rx[100:110] = rx[100]                                            # duplicate ts -> last book state wins
    rx[100:110].sort()
    trade_ts = np.unique(np.concatenate([rx[::3], rx[1::7] + 3]))    # trades coincident + between rows
    return (rx, bid, bq, ask, aq), trade_ts


# --------------------------------------------------------------------------------------------------
# independent oracle — explicit per-event loop, NO shared code with the production build
# --------------------------------------------------------------------------------------------------
def _premium(bid, bq, ask, aq):
    """Size-weighted-mid premium from a single L1 state (leading Stoikov term)."""
    mid = 0.5 * (bid + ask)
    micro = (bq * ask + aq * bid) / (bq + aq)
    return (micro - mid) / mid


def _live_front_leg_oracle(book, trade_ts, grid, span):
    """Dead-simple `LiveFrontEMA` oracle for the premium level.

    Walks every event timestamp in order: a book row refreshes the timestamp's last valid premium
    (last state wins on a duplicate ts); a trade decays/commits the EMA once. The read at each grid
    point is the live front `(1 - a) * committed + a * latest`. `span = 1` => a = 1 => the value
    collapses to the freshest premium read (no smoothing). NaN before the first commit / first valid level.
    """
    rx, bid, bq, ask, aq = book
    ok = (bid > 0.0) & (ask > 0.0) & (bq > 0.0) & (aq > 0.0)
    by_ts: dict[int, float] = {}
    for ts, val in zip(rx[ok], _premium(bid[ok], bq[ok], ask[ok], aq[ok])):
        by_ts[int(ts)] = float(val)                      # last state for a duplicated timestamp

    trades = set(int(t) for t in trade_ts)
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

    out = np.full(len(grid), np.nan)
    if not ts_arr:
        return out
    ts_arr, vals = np.array(ts_arr), np.array(vals)
    idx = np.searchsorted(ts_arr, grid, "right") - 1
    sel = idx >= 0
    out[sel] = vals[idx[sel]]
    return out


def _oracle_fast_slow(book, trade_ts, grid, nf, ns):
    return (_live_front_leg_oracle(book, trade_ts, grid, nf)
            - _live_front_leg_oracle(book, trade_ts, grid, ns))


def _mirror_book(book, c=100.0):
    """Reflect the L1 book through price level c: the two sides SWAP and reflect (`p -> c**2/p`), each
    size following its original price level. The data-level operation a signed-premium `mirror` commutes with."""
    rx, bid, bq, ask, aq = book
    return (rx, c * c / ask, aq, c * c / bid, bq)


# span sweep used by the oracle / parity tests — INCLUDES a span=1 (alpha=1) fast leg.
_SWEEP = [(1, 100), (10, 500)]


# --------------------------------------------------------------------------------------------------
# (a) vectorized vs. independent oracle, on the event grid
# --------------------------------------------------------------------------------------------------
def test_vectorized_matches_oracle():
    book, trade_ts = _synthetic_book()
    raw, shared, config = _inputs({"byb": book}, trade_ts)
    spec = base.get("stoikov_premium_fast_slow")
    for nf, ns in _SWEEP:
        got = spec.vectorized(raw, shared, config, (nf, ns))["byb"]
        ref = _oracle_fast_slow(book, trade_ts, shared.event_ts, nf, ns)
        ok = np.isfinite(got) & np.isfinite(ref)
        assert ok.sum() > 50
        np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------------------------------
# span = 1 (alpha=1, no smoothing): finite wherever inputs exist, consistent NaN where undefined.
# --------------------------------------------------------------------------------------------------
def test_span1_leg_matches_oracle_and_is_finite():
    book, trade_ts = _synthetic_book(seed=11)
    raw, shared, config = _inputs({"byb": book}, trade_ts)
    spec = base.get("stoikov_premium_fast_slow")
    grid = shared.event_ts

    # The single span=1 leg = the freshest premium read at-or-before each event (alpha=1, no memory).
    # vectorized(1, ns) - vectorized(slow) isolates the fast leg via the oracle; check the leg directly
    # by differencing two builds that share the slow leg.
    got_fast = (spec.vectorized(raw, shared, config, (1, 100))["byb"]
                - spec.vectorized(raw, shared, config, (50, 100))["byb"]
                + _live_front_leg_oracle(book, trade_ts, grid, 50))
    ref_fast = _live_front_leg_oracle(book, trade_ts, grid, 1)
    ok = np.isfinite(got_fast) & np.isfinite(ref_fast)
    assert ok.sum() > 50
    np.testing.assert_allclose(got_fast[ok], ref_fast[ok], rtol=1e-9, atol=1e-12)

    # Do-rule: at span=1 the value is FINITE everywhere the oracle is defined (no inf / build-specific
    # number), and the two builds agree on WHERE it is undefined (a consistent NaN both share).
    full = spec.vectorized(raw, shared, config, (1, 100))["byb"]
    ref_full = _oracle_fast_slow(book, trade_ts, grid, 1, 100)
    assert np.array_equal(np.isfinite(full), np.isfinite(ref_full))   # same defined/undefined mask
    assert np.all(np.isfinite(full[np.isfinite(ref_full)]))           # finite where inputs exist
    assert not np.any(np.isinf(full))                                 # never inf


# --------------------------------------------------------------------------------------------------
# (c) per-venue fan-out — one independent leg per exchange (target + sources)
# --------------------------------------------------------------------------------------------------
def test_fans_out_over_all_exchanges():
    b_byb, trade_ts = _synthetic_book(seed=3)
    b_bin, _ = _synthetic_book(seed=4)
    b_okx, _ = _synthetic_book(seed=5)
    books = {"byb": b_byb, "bin": b_bin, "okx": b_okx}
    raw, shared, config = _inputs(books, trade_ts, sources=("bin", "okx"))
    spec = base.get("stoikov_premium_fast_slow")
    out = spec.vectorized(raw, shared, config, (10, 100))
    assert set(out) == {"byb", "bin", "okx"}                            # one leg per exchange
    assert tuple(spec.keys_for(config, (10, 100))) == ("byb", "bin", "okx")
    for ex, book in books.items():                                     # each leg = that venue's own-book build
        r1, s1, c1 = _inputs({ex: book}, trade_ts, target_ex=ex)
        np.testing.assert_array_equal(out[ex], spec.vectorized(r1, s1, c1, (10, 100))[ex])


# --------------------------------------------------------------------------------------------------
# (b) mirror commutation invariant — the FULL book reflection.
# Reflect every price through level c (p -> c**2/p) AND swap sides/sizes (bid' = reflect(ask), the
# bid keeps the OLD ask's size). The premium is ODD, so its declared mirror (np.negative) must
# reproduce the rebuild on the reflected book.
# --------------------------------------------------------------------------------------------------
def test_mirror_commutes_with_book_reflection():
    book, trade_ts = _synthetic_book(seed=2)
    raw, shared, config = _inputs({"byb": book}, trade_ts)
    mraw, mshared, mconfig = _inputs({"byb": _mirror_book(book)}, trade_ts)
    spec = base.get("stoikov_premium_fast_slow")
    assert spec.mirror is not None
    for params in _SWEEP:
        feat = spec.vectorized(raw, shared, config, params)["byb"]      # feature(books)
        refl = spec.vectorized(mraw, mshared, mconfig, params)["byb"]   # feature(mirror_books(books))
        ok = np.isfinite(feat) & np.isfinite(refl)
        assert ok.sum() > 50
        np.testing.assert_allclose(spec.mirror(feat)[ok], refl[ok], rtol=1e-6, atol=1e-12)


# --------------------------------------------------------------------------------------------------
# synthetic streaming-vs-vectorized parity (the production O(1) path reproduces the offline build)
# --------------------------------------------------------------------------------------------------
def test_synthetic_parity():
    b_byb, trade_ts = _synthetic_book(seed=6)
    b_bin, _ = _synthetic_book(seed=7)
    books = {"byb": b_byb, "bin": b_bin}
    anchor_ts = b_byb[0][250:-250:11]
    ctx = _ctx_for_parity(books, trade_ts, anchor_ts, sources=("bin",))
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
