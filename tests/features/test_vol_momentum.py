"""Self-contained test suite for `boba.features.vol_momentum` — `log(σ_ev_fast / σ_ev_slow)` per venue from
that venue's OWN mid (see the module docstring). It FANS OUT over `config.all_listings` (target + sources),
`params = (n_fast, n_slow)`, EVEN under the tape reflection (`mirror = identity`). The AUTHORING validation
trio + streaming coverage, on synthetic data with NO shared code with production:

  - `test_vectorized_matches_oracle`     each leg vs an INDEPENDENT σ_ev event-loop oracle (incl. a 1-span).
  - `test_fans_out_over_all_listings`    one leg per venue; a byb-only config (`other_listings=()`) is just byb.
  - `test_even_under_reflection`         reflecting the mids leaves it unchanged (σ_ev is even -> mirror=identity).
  - `test_streaming_matches_vectorized`  `LiveVolMomentum` vs the sampled vectorized via `parity_check` (incl. span=1).
  - `test_real_block_parity`             streaming-vs-vectorized on a real block (DATA_DIR-gated).
"""
from dataclasses import replace

import numpy as np
import pytest

import boba.io as io
from boba.features import base
import boba.features.vol_momentum  # noqa: F401  (registers the spec)
from boba.features.base import (
    Config, FrontLevels, ListingRaw, ListingShared, RawData, Series, SharedData, Trade)
from boba.features.shared import build_shared_data
from boba.research.screening import RawEventStream, ScreeningContext, parity_check

_YARD = 25


# --------------------------------------------------------------------------------------------------
# synthetic standalone inputs (per-venue books + trades; their union is the decay clock) — own builders
# --------------------------------------------------------------------------------------------------
def _raw_data(books, trades, coin="x") -> RawData:
    listings: dict[str, ListingRaw] = {}
    for ex, (rx, bid, bq, ask, aq) in books.items():
        rx = rx.astype(np.int64)
        trx, tpx, tl, tq = trades[ex]
        listings[f"{ex}_{coin}"] = ListingRaw(
            front_levels=FrontLevels(rx, rx, bid, bq, ask, aq),
            trade=Trade(trx.astype(np.int64), trx.astype(np.int64), tpx.astype(float), tl.astype(float), tq.astype(float)))
    return RawData(listings=listings)


def _inputs(books, trades, target_ex="byb", sources=(), coin="x"):
    raw = _raw_data(books, trades, coin)
    config = Config(f"{target_ex}_{coin}", tuple(f"{s}_{coin}" for s in sources), coin,
                    {f"{ex}_{coin}": "front_levels" for ex in books}, yardstick_span=_YARD)
    return raw, build_shared_data(raw, config), config


def _raw_events(books, trades, coin="x") -> RawEventStream:
    listings = tuple(f"{ex}_{coin}" for ex in books)
    cols: dict[str, list] = {k: [] for k in "rx kind lid t a b c d".split()}

    def add(rx, kind, lid, t, a, b, c, d):
        n = len(rx)
        cols["rx"].append(rx.astype(np.int64)); cols["kind"].append(np.full(n, kind, np.int8))
        cols["lid"].append(np.full(n, lid, np.int8)); cols["t"].append(t.astype(np.int64))
        for k, v in (("a", a), ("b", b), ("c", c), ("d", d)):
            cols[k].append(v.astype(float))

    for lid, (ex, (rx, bid, bq, ask, aq)) in enumerate(books.items()):
        add(rx, 0, lid, rx, bid, ask, bq, aq)                  # book: bid, ask, bid_qty, ask_qty
    for lid, ex in enumerate(books):
        rx, px, lifts, qty = trades[ex]
        add(rx, 1, lid, rx, px, lifts, qty, np.full(len(rx), np.nan))   # trade: px, lifts_ask, qty
    C = {k: np.concatenate(v) for k, v in cols.items()}
    order = np.lexsort((C["kind"], C["rx"]))
    return RawEventStream(*(C[k][order] for k in "rx kind lid t a b c d".split()), listings)


def _ctx_for_parity(books, trades, anchor_ts, target_ex="byb", sources=()) -> ScreeningContext:
    raw, shared, config = _inputs(books, trades, target_ex, sources)
    return ScreeningContext(
        block="syn", coin="x", target=config.target_listing, sources=config.other_listings, horizon_ns=0,
        yardstick_span=_YARD, mid_stream={}, merged_ts=shared.clock, anchor_ts=anchor_ts,
        sigma_at_anchor=np.empty(0), lam_at_anchor=np.empty(0), price_target=np.empty(0),
        rate_target=np.empty(0), base=[], vol_level=np.empty(0), rate_level=np.empty(0), vol_regime=np.empty(0),
        raw_events=_raw_events(books, trades), raw_data=raw, shared_data=shared, config=config)


def _book(seed, n=2600):
    rng = np.random.default_rng(seed)
    rx = (np.arange(1, n + 1) * 10).astype(np.int64)
    mid = 100.0 * np.exp(np.cumsum(np.where(rng.random(n) < 0.6, rng.standard_normal(n) * 8e-5, 0.0)))
    rx[100:106] = rx[100]                                       # same-ts burst -> only the final mid counts
    hs = 0.005 + 0.005 * rng.random(n)
    return rx, mid - hs, np.ones(n), mid + hs, np.ones(n)       # (rx, bid, bid_qty, ask, ask_qty)


def _trades(seed, n=1900):
    rng = np.random.default_rng(seed)
    rx = (np.cumsum(rng.integers(7, 19, n)) + (seed % 5)).astype(np.int64)   # variable inter-tick gaps
    return rx, np.full(n, 100.0), (rng.random(n) > 0.5).astype(float), rng.uniform(0.1, 5.0, n)


def _market(seed=0):
    """Two venues (byb target, bin source), each with its OWN moving mid + trades."""
    exes = ("byb", "bin")
    books = {ex: _book(seed + i) for i, ex in enumerate(exes)}
    trades = {ex: _trades(seed + 10 + i) for i, ex in enumerate(exes)}
    trade_ts = np.unique(np.concatenate([trades[ex][0] for ex in exes]))
    return books, trades, trade_ts


# --------------------------------------------------------------------------------------------------
# INDEPENDENT, dead-simple oracle — per-venue σ_ev by an explicit per-event loop, then log(fast/slow).
# --------------------------------------------------------------------------------------------------
def _book_mid(book):
    rx, bid, _bq, ask, _aq = book
    return rx, 0.5 * (bid + ask)


def _sigma_oracle(book, trade_ts, grid, span):
    """σ_ev of this venue's mid at each `grid` point: α=2/(span+1); inject α·(Δlog mid)² into E and α into W
    on a mid CHANGE (final mid per ts), decay both by (1−α) on a trade tick; read sqrt(E/W) at the last event."""
    a = 2.0 / (span + 1.0)
    rx, m = _book_mid(book)
    rx, m = np.asarray(rx), np.asarray(m)
    keep = np.concatenate([rx[1:] != rx[:-1], [True]])
    rx, m = rx[keep], m[keep]
    lm = np.log(m)
    inj: dict[int, float] = {}
    prev = None
    for ts, x in zip(rx.tolist(), lm.tolist()):
        if prev is not None and x != prev:
            inj[ts] = (x - prev) ** 2
        prev = x
    trades = set(int(t) for t in trade_ts)
    all_ts = sorted(set(inj) | trades)
    E = W = 0.0
    ts_arr, vals = [], []
    for ts in all_ts:
        if ts in inj:
            E += a * inj[ts]; W += a
        if ts in trades:
            E *= (1.0 - a); W *= (1.0 - a)
        ts_arr.append(ts); vals.append((E / W) ** 0.5 if W > 0 else np.nan)
    ts_arr, vals = np.array(ts_arr), np.array(vals)
    idx = np.searchsorted(ts_arr, grid, "right") - 1
    return np.where(idx < 0, np.nan, vals[np.clip(idx, 0, len(vals) - 1)])


def _vol_momentum_oracle(book, trade_ts, grid, n_fast, n_slow):
    sf = _sigma_oracle(book, trade_ts, grid, n_fast)
    ss = _sigma_oracle(book, trade_ts, grid, n_slow)
    ok = np.isfinite(sf) & np.isfinite(ss) & (sf > 0) & (ss > 0)
    return np.where(ok, np.log(np.where(ok, sf, 1.0) / np.where(ok, ss, 1.0)), np.nan)


def test_vectorized_matches_oracle():
    books, trades, trade_ts = _market(seed=1)
    raw, shared, config = _inputs(books, trades, sources=("bin",))
    spec = base.get("vol_momentum")
    for params in ((1, 25), (5, 50), (10, 200)):                   # incl. a 1-span fast leg
        out = spec.vectorized(raw, shared, config, params)
        for ex in ("byb", "bin"):                                  # EACH leg from its OWN mid
            got, ref = out[ex], _vol_momentum_oracle(books[ex], trade_ts, shared.event_ts, *params)
            assert not np.any(np.isinf(got))                       # never a stray inf — log(0) is guarded
            np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))   # consistent NaN both builds agree on
            ok = np.isfinite(got) & np.isfinite(ref)
            assert ok.sum() > 100
            # log AMPLIFIES the σ-EMA float-accumulation gap between the two independent builds when σ_fast
            # is tiny (a fast leg right before full decay), so the log-ratio is accurate to ~1e-6, not 1e-9.
            np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-6, atol=1e-9)


def test_fans_out_over_all_listings():
    books, trades, trade_ts = _market(seed=3)
    raw, shared, config = _inputs(books, trades, sources=("bin",))
    spec = base.get("vol_momentum")
    out = spec.vectorized(raw, shared, config, (10, 100))
    assert set(out) == {"byb", "bin"}                             # one leg per venue
    assert tuple(spec.keys_for(config, (10, 100))) == ("byb", "bin")
    for ex in ("byb", "bin"):
        ref = _vol_momentum_oracle(books[ex], trade_ts, shared.event_ts, 10, 100)
        ok = np.isfinite(out[ex]) & np.isfinite(ref)
        assert ok.sum() > 100
        np.testing.assert_allclose(out[ex][ok], ref[ok], rtol=1e-6, atol=1e-9)
    # restricting to the target alone = a config whose other_listings is empty (the byb leg, unchanged)
    out_tgt = spec.vectorized(raw, shared, replace(config, other_listings=()), (10, 100))
    assert set(out_tgt) == {"byb"}
    np.testing.assert_array_equal(out_tgt["byb"], out["byb"])


# --------------------------------------------------------------------------------------------------
# EVEN under the tape reflection: σ_ev is built from squared moves -> unchanged -> the feature is unchanged.
# --------------------------------------------------------------------------------------------------
def _mirror_shared(shared: SharedData, c=100.0) -> SharedData:
    listings = {l: ListingShared(mid=Series(np.asarray(ls.mid.rx), c * c / np.asarray(ls.mid.value)))
                for l, ls in shared.listings.items()}
    return SharedData(event_ts=shared.event_ts, clock=shared.clock, vol_yardstick=shared.vol_yardstick,
                      rate_yardstick=shared.rate_yardstick, listings=listings)


def test_even_under_reflection():
    books, trades, trade_ts = _market(seed=4)
    raw, shared, config = _inputs(books, trades, sources=("bin",))
    spec = base.get("vol_momentum")
    feat = spec.vectorized(raw, shared, config, (10, 100))
    refl = spec.vectorized(raw, _mirror_shared(shared), config, (10, 100))
    for ex in ("byb", "bin"):
        lhs = spec.mirror(feat[ex])                               # mirror = identity (EVEN)
        ok = np.isfinite(lhs) & np.isfinite(refl[ex])
        assert ok.sum() > 100
        np.testing.assert_allclose(lhs[ok], refl[ex][ok], rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------------------------------
# streaming vs vectorized — incl. a span=1 leg (move-only anchors keep a span=1 σ_ev alive, per venue)
# --------------------------------------------------------------------------------------------------
def _span1_anchors(books, trade_ts):
    """Union over ALL venues of (that venue's mid-move timestamps NOT on a trade tick) + interior trade
    ticks. At a venue's move-only anchor that venue's span=1 σ_ev is ALIVE -> each leg gets span=1 coverage."""
    trset = set(int(t) for t in trade_ts)
    mo: list[int] = []
    for book in books.values():
        rx, m = _book_mid(book)
        rx, m = np.asarray(rx), np.asarray(m)
        keep = np.concatenate([rx[1:] != rx[:-1], [True]]); rx, m = rx[keep], m[keep]
        moved = np.concatenate([[False], np.diff(np.log(m)) != 0])
        mo.extend(int(t) for t in rx[moved] if int(t) not in trset)
    lo, hi = trade_ts[120], trade_ts[-120]
    mo = np.unique([t for t in mo if lo < t < hi])
    return np.unique(np.concatenate([mo, trade_ts[150:-150:5]]))


def test_streaming_matches_vectorized():
    books, trades, trade_ts = _market(seed=7)
    anchor_ts = _span1_anchors(books, trade_ts)
    ctx = _ctx_for_parity(books, trades, anchor_ts, sources=("bin",))
    # realistic spans: machine-precision parity (the same KernelMeanEMA recursion drives both builds).
    rep = parity_check(ctx, base.get("vol_momentum"), [(5, 50), (50, 500)], n_grid=len(anchor_ts), tol=1e-9)
    assert rep.passed, str(rep)
    # span=1 (the α=1 path): σ_fast is a tiny spiky RMS, so the LOG amplifies the two builds' boundary/
    # summation float gap (data-dependent, here ~5e-7) — still orders below any real α=1 bug (O(1e-2)). The
    # yardstick's OWN span=1 is machine-precision-tied in test_streaming::test_yardsticks_via_parity_on_real_block.
    rep1 = parity_check(ctx, base.get("vol_momentum"), [(1, 25)], n_grid=len(anchor_ts), tol=1e-5)
    assert rep1.passed, str(rep1)


@pytest.mark.skipif(getattr(io, "DATA_DIR", None) is None, reason="no DATA_DIR configured")
def test_real_block_parity():
    from boba.research.screening import build_context
    ctx = build_context(hours=2)
    rep = parity_check(ctx, base.get("vol_momentum"), [(100, 1000), (1000, 10000)], tol=1e-6)
    assert rep.passed, str(rep)
