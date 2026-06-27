"""Self-contained test suite for `boba.features.ofi_fast_slow` — the path-sum L1 OFI fast/slow oscillator.

The feature is now a standalone transform `vectorized(raw_data, shared_data, config, (n_fast, n_slow)) ->
{ex -> value per shared_data.event_ts}`. The AUTHORING.md validation trio + streaming coverage, on
synthetic data with NO shared code with production:

  - `test_vectorized_matches_oracle`     vectorized vs an INDEPENDENT dead-simple event-loop oracle
                                         (fast leg minus slow leg), on the event grid, incl. a span=1 fast leg.
  - `test_fans_out_over_all_exchanges`   one independent OFI leg per venue (target + each source).
  - `test_mirror_commutes_with_reflection`  the COMMUTATION invariant against the FULL book reflection.
  - `test_span1_finite_where_inputs_exist`  span=1 (α=1) fast leg is finite wherever the flow exists,
                                            NaN only during warm-up — the AUTHORING span=1 Do-rule.
  - `test_streaming_matches_vectorized`  drive `LiveOFIFastSlow` over a synthetic raw-event stream and
                                         assert it matches the (sampled) vectorized via `parity_check`.
  - `test_real_block_parity`             streaming-vs-vectorized parity on a real block (DATA_DIR-gated).
"""
import numpy as np
import pytest

import boba.io as io
from boba.features import base
import boba.features.ofi_fast_slow  # noqa: F401  (registers the spec)
from boba.features.base import Config, FrontLevels, ListingRaw, RawData, Trade
from boba.features.shared import build_shared_data
from boba.research.screening import RawEventStream, ScreeningContext, parity_check


# --------------------------------------------------------------------------------------------------
# synthetic standalone inputs (own builders — no shared code with production)
# --------------------------------------------------------------------------------------------------
def _raw_data(books, trade_ts, coin="x") -> RawData:
    """`{short_ex -> (rx, bid, bid_qty, ask, ask_qty)}` + a shared trade clock -> RawData. The decay clock
    lives on the FIRST listing's trades (OFI is book-only, so the trade payload is dummy); every listing
    uses a front_levels mid. exchange_time is set to rx (irrelevant to OFI)."""
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
    """A book row every 10 ns; a same-ts burst so the path-sum (intra-ns sum) path is exercised. Returns
    `((rx, bid, bid_qty, ask, ask_qty), trade_ts)`."""
    rng = np.random.default_rng(seed)
    rx = (np.arange(1, n + 1) * 10).astype(np.int64)
    mid = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 1e-4))
    hs = 0.01 + 0.01 * rng.random(n)
    bid, ask = mid - hs, mid + hs
    bq, aq = rng.uniform(1.0, 100.0, n), rng.uniform(1.0, 100.0, n)
    rx[100:105] = rx[100]                                          # same-ts book burst -> summed increment
    rx[100:105] = np.sort(rx[100:105])
    trade_ts = np.unique(np.concatenate([rx[::3], rx[1::7] + 3]))  # trades coincident + between rows
    return (rx, bid, bq, ask, aq), trade_ts


# --------------------------------------------------------------------------------------------------
# INDEPENDENT oracle — an explicit one-event-at-a-time loop, no shared code with the production build
# --------------------------------------------------------------------------------------------------
def _ofi_leg_oracle(book, trade_ts, grid, span):
    """E/W of the path-sum OFI flow at each `grid` timestamp, dead-simple. Per consecutive raw row form the
    CKS L1 increment, SUM increments sharing a receive-ts into one flow sample, then walk all event
    timestamps in order injecting (book change) / decaying (trade), reading E/W after the last event
    at-or-before each grid point. Inject-then-decay within a timestamp. α = 2/(span+1); span=1 => α=1."""
    rx, bid, bq, ask, aq = book
    a = 2.0 / (span + 1.0)
    sums: dict[int, float] = {}
    for i in range(1, len(rx)):
        pbp, pbq, pap, paq = bid[i - 1], bq[i - 1], ask[i - 1], aq[i - 1]
        cbp, cbq, cap, caq = bid[i], bq[i], ask[i], aq[i]
        inc = ((cbq if cbp >= pbp else 0.0) - (pbq if cbp <= pbp else 0.0)
               - (caq if cap <= pap else 0.0) + (paq if cap >= pap else 0.0))
        sums[int(rx[i])] = sums.get(int(rx[i]), 0.0) + inc
    trades = set(int(t) for t in trade_ts)
    all_ts = sorted(set(sums) | trades)
    E = W = 0.0
    ts_arr, Es, Ws = [], [], []
    for ts in all_ts:
        if ts in sums:
            E += a * sums[ts]; W += a
        if ts in trades:
            E *= (1.0 - a); W *= (1.0 - a)
        ts_arr.append(ts); Es.append(E); Ws.append(W)
    out = np.full(len(grid), np.nan)
    if not ts_arr:
        return out
    ts_arr, Es, Ws = np.array(ts_arr), np.array(Es), np.array(Ws)
    j = np.searchsorted(ts_arr, grid, "right") - 1
    ok = j >= 0
    Wj = Ws[j[ok]]
    out[ok] = np.where(Wj > 0.0, Es[j[ok]] / np.where(Wj == 0.0, np.nan, Wj), np.nan)
    return out


def _fast_slow_oracle(book, trade_ts, grid, n_fast, n_slow):
    """fast − slow: the OFI E/W leg at the fast span minus that at the slow span, on the same grid."""
    return (_ofi_leg_oracle(book, trade_ts, grid, n_fast)
            - _ofi_leg_oracle(book, trade_ts, grid, n_slow))


def _mirror_book(book, c=100.0):
    """Reflect the L1 book through price level c: the two sides SWAP and reflect (`p -> c**2/p`), each
    size following its original price level. The data-level operation a signed-flow `mirror` must commute with."""
    rx, bid, bq, ask, aq = book
    return (rx, c * c / ask, aq, c * c / bid, bq)


# --------------------------------------------------------------------------------------------------
# (a) vectorized vs the independent oracle, across fast/slow pairs incl. a span=1 fast leg
# --------------------------------------------------------------------------------------------------
def test_vectorized_matches_oracle():
    book, trade_ts = _synthetic_book(seed=1)
    raw, shared, config = _inputs({"byb": book}, trade_ts)
    spec = base.get("ofi_fast_slow")
    for nf, ns in [(1, 100), (10, 500), (1, 50)]:               # (1, *) exercises the α=1 fast leg
        got = spec.vectorized(raw, shared, config, (nf, ns))["byb"]
        ref = _fast_slow_oracle(book, trade_ts, shared.event_ts, nf, ns)
        ok = np.isfinite(got) & np.isfinite(ref)
        assert ok.sum() > 50
        np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-9, atol=1e-9)
        np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))


# --------------------------------------------------------------------------------------------------
# (b) per-venue fan-out: one independent leg per exchange, each the single-venue own-book build
# --------------------------------------------------------------------------------------------------
def test_fans_out_over_all_exchanges():
    b_byb, trade_ts = _synthetic_book(seed=3)
    b_bin, _ = _synthetic_book(seed=4)
    b_okx, _ = _synthetic_book(seed=5)
    books = {"byb": b_byb, "bin": b_bin, "okx": b_okx}
    raw, shared, config = _inputs(books, trade_ts, sources=("bin", "okx"))
    spec = base.get("ofi_fast_slow")
    for params in [(1, 50), (10, 100)]:
        out = spec.vectorized(raw, shared, config, params)
        assert set(out) == {"byb", "bin", "okx"}
        assert tuple(spec.keys_for(config, params)) == ("byb", "bin", "okx")
        for ex, book in books.items():                           # each leg == that venue's own-book OFI, independent
            r1, s1, c1 = _inputs({ex: book}, trade_ts, target_ex=ex)
            np.testing.assert_array_equal(out[ex], spec.vectorized(r1, s1, c1, params)[ex])


# --------------------------------------------------------------------------------------------------
# (c) the mirror COMMUTATION invariant against the full book reflection
# --------------------------------------------------------------------------------------------------
def test_mirror_commutes_with_reflection():
    book, trade_ts = _synthetic_book(seed=2)
    raw, shared, config = _inputs({"byb": book}, trade_ts)
    mraw, mshared, mconfig = _inputs({"byb": _mirror_book(book)}, trade_ts)
    spec = base.get("ofi_fast_slow")
    assert spec.mirror is not None
    for params in [(1, 100), (10, 500)]:
        feat = spec.vectorized(raw, shared, config, params)["byb"]        # feature(books)
        refl = spec.vectorized(mraw, mshared, mconfig, params)["byb"]     # feature(mirror_books(books))
        ok = np.isfinite(feat) & np.isfinite(refl)
        assert ok.sum() > 50
        np.testing.assert_allclose(spec.mirror(feat)[ok], refl[ok], rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------------------------------
# span = 1 (α = 1, the most-used fast leg): finite wherever the flow exists, consistent NaN otherwise
# --------------------------------------------------------------------------------------------------
def test_span1_finite_where_inputs_exist():
    book, trade_ts = _synthetic_book(seed=6)
    raw, shared, config = _inputs({"byb": book}, trade_ts)
    spec = base.get("ofi_fast_slow")
    grid = shared.event_ts
    # fast = slow = 1: identically 0 where both legs are defined; NaN exactly where the span=1 leg is undefined.
    leg1 = spec.vectorized(raw, shared, config, (1, 1))["byb"]
    ref1 = _ofi_leg_oracle(book, trade_ts, grid, 1)
    defined = np.isfinite(ref1)
    assert defined.sum() > 100                                  # the span=1 leg is well-populated, not degenerate
    assert not np.isinf(leg1).any()                             # never a spurious inf at α=1
    np.testing.assert_array_equal(np.isfinite(leg1), defined)
    np.testing.assert_array_equal(leg1[defined], 0.0)          # fast − slow with equal spans is 0
    # a genuine span=1 fast leg vs a slow leg: finite EXACTLY where both legs are defined, NaN where not.
    fs = spec.vectorized(raw, shared, config, (1, 100))["byb"]
    both = np.isfinite(ref1) & np.isfinite(_ofi_leg_oracle(book, trade_ts, grid, 100))
    assert np.all(np.isfinite(fs[both]))
    np.testing.assert_array_equal(np.isfinite(fs), both)
    np.testing.assert_allclose(fs[both], _fast_slow_oracle(book, trade_ts, grid, 1, 100)[both],
                               rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------------------------------
# streaming (LiveOFIFastSlow) vs vectorized — synthetic parity, including a span=1 fast leg
# --------------------------------------------------------------------------------------------------
def test_streaming_matches_vectorized():
    b_byb, trade_ts = _synthetic_book(seed=7)
    b_bin, _ = _synthetic_book(seed=8)
    books = {"byb": b_byb, "bin": b_bin}
    anchor_ts = b_byb[0][200:-200:9]
    ctx = _ctx_for_parity(books, trade_ts, anchor_ts, sources=("bin",))
    rep = parity_check(ctx, base.get("ofi_fast_slow"), [(1, 100), (10, 500)],
                       n_grid=len(anchor_ts), tol=1e-9)
    assert rep.passed, str(rep)


# --------------------------------------------------------------------------------------------------
# real-block parity (skipped without DATA_DIR) — streaming reproduces vectorized, incl. a span=1 fast leg
# --------------------------------------------------------------------------------------------------
@pytest.mark.skipif(getattr(io, "DATA_DIR", None) is None, reason="no DATA_DIR configured")
def test_real_block_parity():
    from boba.research.screening import build_context

    ctx = build_context(hours=2)
    rep = parity_check(ctx, base.get("ofi_fast_slow"), [(1, 100), (10, 500)], tol=1e-6)
    assert rep.passed, str(rep)
