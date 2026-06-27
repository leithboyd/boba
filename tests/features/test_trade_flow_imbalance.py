"""Self-contained test suite for `boba.features.trade_flow_imbalance` -- signed trade-VOLUME imbalance.

The feature is now a standalone transform `vectorized(raw_data, shared_data, config, N) -> {ex -> value
per shared_data.event_ts}`: per venue and receive timestamp it sums the trade flow into
`signed_volume = sum(side*qty)` and `total_volume = sum(qty)`, then reads the trade-clock EMA ratio
`EMA(signed_volume) / EMA(total_volume)` in [-1, 1]. It fans out one independent leg per exchange
(target + every source), and is ODD under the book/tape reflection -- `qty` is a count (invariant) while
the side flips -- so `SPEC.mirror = np.negative`.

The AUTHORING.md validation trio + streaming coverage, on synthetic data with NO shared code with
production:

  - `test_vectorized_matches_oracle`     vectorized vs an INDEPENDENT dead-simple event-loop oracle, on
                                         the event grid, across spans where the feature is defined (>=2).
  - `test_value_is_bounded_and_finite`   a volume-weighted signed fraction lives in [-1, 1] and is finite
                                         wherever the oracle is defined (the AUTHORING Do-rule).
  - `test_fans_out_over_all_exchanges`   one independent imbalance leg per venue (target + each source).
  - `test_mirror_commutes_with_reflection`  the COMMUTATION invariant against the FULL tape reflection.
  - `test_span1_is_a_consistent_nan`     span=1 (alpha=1) is genuinely undefined for a pure trade-flow
                                         EMA read after the decay -> a consistent NaN both builds agree on.
  - `test_streaming_matches_vectorized`  drive `LiveTradeFlowImbalance` over a synthetic raw-event stream
                                         and assert it matches the (sampled) vectorized via `parity_check`.
  - `test_real_block_parity`             streaming-vs-vectorized parity on a real block (DATA_DIR-gated).

span=1 handling (the AUTHORING alpha=1 Do-rule): a *pure trade-flow* EMA read after the trade-clock
decay has no committed event at alpha=1 -- inject-then-decay zeroes both the signed and total kernel
sums, so `E/W` is 0/0. The feature is therefore genuinely undefined at span=1, and the rule's fallback
applies: a CONSISTENT NaN that BOTH builds agree on.
"""
import numpy as np
import pytest

import boba.io as io
from boba.features import base
import boba.features.trade_flow_imbalance  # noqa: F401  (registers the spec)
from boba.features.base import Config, FrontLevels, ListingRaw, RawData, Trade
from boba.features.shared import build_shared_data
from boba.research.screening import RawEventStream, ScreeningContext, parity_check


# --------------------------------------------------------------------------------------------------
# synthetic standalone inputs (own builders — no shared code with production)
# --------------------------------------------------------------------------------------------------
def _raw_data(books, trades, coin="x") -> RawData:
    """`{short_ex -> (rx,bid,bid_qty,ask,ask_qty)}` books + `{short_ex -> (rx,px,lifts,qty)}` trades ->
    RawData. Every venue carries its OWN real trade tape (the decay clock is the union of all of them);
    exchange_time is set to rx (irrelevant to a trade-flow feature)."""
    listings: dict[str, ListingRaw] = {}
    for ex, (rx, bid, bq, ask, aq) in books.items():
        rx = rx.astype(np.int64)
        front = FrontLevels(rx, rx, bid, bq, ask, aq)
        trx, px, lifts, qty = trades[ex]
        trx = trx.astype(np.int64)
        trade = Trade(trx, trx, px.astype(float), lifts.astype(float), qty.astype(float))
        listings[f"{ex}_{coin}"] = ListingRaw(front_levels=front, trade=trade)
    return RawData(listings=listings)


def _inputs(books, trades, target_ex="byb", sources=(), coin="x"):
    """-> (raw_data, shared_data, config) for the standalone feature build."""
    raw = _raw_data(books, trades, coin)
    config = Config(f"{target_ex}_{coin}", tuple(f"{s}_{coin}" for s in sources), coin,
                    {f"{ex}_{coin}": "front_levels" for ex in books}, yardstick_span=10)
    return raw, build_shared_data(raw, config), config


def _raw_events(books, trades, coin="x") -> RawEventStream:
    """Per-venue raw book rows (kind 0) + per-venue trades (kind 1), receive-ordered, for the parity driver."""
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
        trx, px, lifts, qty = trades[ex]
        add(trx, 1, lid, trx, px, lifts, qty, np.full(len(trx), np.nan))  # TradeEvent(listing, rx, exch_time, px, lifts_ask, qty); fixture sets exch_time == rx
    C = {k: np.concatenate(v) for k, v in cols.items()}
    order = np.lexsort((C["kind"], C["rx"]))
    return RawEventStream(*(C[k][order] for k in "rx kind lid t a b c d".split()), listings)


def _ctx_for_parity(books, trades, anchor_ts, target_ex="byb", sources=()) -> ScreeningContext:
    """A minimal ScreeningContext carrying the standalone inputs + the raw-event stream + an anchor grid,
    enough for `parity_check` (streaming read at anchors vs the sampled vectorized)."""
    raw, shared, config = _inputs(books, trades, target_ex, sources)
    return ScreeningContext(
        block="syn", coin="x", target=config.target_listing, sources=config.other_listings, horizon_ns=0,
        yardstick_span=10, mid_stream={}, merged_ts=shared.clock, anchor_ts=anchor_ts,
        sigma_at_anchor=np.empty(0), lam_at_anchor=np.empty(0), price_target=np.empty(0),
        rate_target=np.empty(0), base=[], vol_level=np.empty(0), rate_level=np.empty(0), vol_regime=np.empty(0),
        raw_events=_raw_events(books, trades), raw_data=raw, shared_data=shared, config=config)


def _synthetic_book(seed=0, n=2200):
    """A book row every 10 ns; a same-ts burst (final mid only). Returns `(rx, bid, bid_qty, ask, ask_qty)`."""
    rng = np.random.default_rng(seed)
    rx = (np.arange(1, n + 1) * 10).astype(np.int64)
    mid = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 8e-5))
    hs = 0.005 + 0.005 * rng.random(n)
    bid, ask = mid - hs, mid + hs
    bq, aq = rng.uniform(1.0, 100.0, n), rng.uniform(1.0, 100.0, n)
    rx[100:106] = rx[100]
    return rx, bid, bq, ask, aq


def _synthetic_trades(seed=0, n=1800):
    """A trade every ~13 ns; same-ts bursts so the intra-ns summed flow path is exercised. Returns
    `(rx, px, lifts, qty)`."""
    rng = np.random.default_rng(seed)
    rx = (np.arange(1, n + 1) * 13 + seed % 7).astype(np.int64)
    px = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 7e-5))
    lifts = (rng.random(n) > 0.48).astype(float)
    qty = rng.uniform(0.1, 5.0, n)
    rx[50:55] = rx[50]          # same-timestamp trade burst -> summed flow
    rx[250:254] = rx[250]
    return rx, px, lifts, qty


def _market(seed=0, exes=("byb", "bin", "okx")):
    """`(books, trades, anchor_ts)` for a small synthetic market — one own book + tape per venue."""
    books = {ex: _synthetic_book(seed + i) for i, ex in enumerate(exes)}
    trades = {ex: _synthetic_trades(seed + 10 + i) for i, ex in enumerate(exes)}
    merged_ts = np.unique(np.concatenate([trades[ex][0] for ex in exes]))
    anchor_ts = merged_ts[150:-150:7]
    return books, trades, anchor_ts


# --------------------------------------------------------------------------------------------------
# INDEPENDENT oracle — an explicit one-event-at-a-time loop, no shared code with the production build.
#
# Implementable from the feature's written definition alone: per receive timestamp sum the trade flow
# into signed_volume = sum(side*qty) and total_volume = sum(qty) (dropping bad prc<=0 / qty<=0 prints),
# then walk every event/trade timestamp in order maintaining two trade-clock EMA *sums* E_signed and
# E_value (inject `alpha*x` at a timestamp carrying the flow, decay by `1-alpha` once per trade
# timestamp), and read the ratio E_signed / E_value after the last event at-or-before each grid point.
# The common alpha*(1-alpha)^k weights cancel in the ratio, so this is the per-event volume mean of the
# signed fraction, bounded in [-1, 1]. VOLUME-weighted (qty, not px*qty) so the feature is exactly odd.
# --------------------------------------------------------------------------------------------------
def _kernel_sum(event_rx, values, clock_ts, grid, span):
    """Trade-clock EMA *sum* of `values` (aligned to `event_rx`) read at each grid point: inject alpha*x
    at an event timestamp, decay by (1-alpha) once per trade timestamp, last-state hold between reads."""
    a = 2.0 / (span + 1.0)
    beta = 1.0 - a
    events: dict[int, float] = {}
    for ts, val in zip(event_rx, values):
        events[int(ts)] = events.get(int(ts), 0.0) + float(val)
    trades = set(int(t) for t in clock_ts)
    all_ts = sorted(set(events) | trades)
    E = 0.0
    ts_arr, Es = [], []
    for ts in all_ts:
        if ts in events:
            E += a * events[ts]
        if ts in trades:
            E *= beta
        ts_arr.append(ts); Es.append(E)
    out = np.full(len(grid), np.nan)
    if not ts_arr:
        return out
    ts_arr, Es = np.array(ts_arr), np.array(Es)
    idx = np.searchsorted(ts_arr, grid, "right") - 1
    ok = idx >= 0
    out[ok] = Es[idx[ok]]
    return out


def _imbalance_oracle(trades, clock_ts, grid, span):
    """EMA(signed_volume) / EMA(total_volume) in [-1, 1] by the dead-simple loop above. VOLUME-weighted."""
    rx, px, lifts, qty = trades
    ok = (px > 0.0) & (qty > 0.0) & np.isfinite(px) & np.isfinite(qty) & np.isfinite(lifts)
    rx, lifts, qty = rx[ok], lifts[ok], qty[ok]
    signed = np.where(lifts > 0.0, qty, -qty)      # side * qty: +qty buy lifts ask, -qty sell hits bid
    uniq, inv = np.unique(rx, return_inverse=True)
    signed_sum = np.bincount(inv, weights=signed)  # per-timestamp summed flow
    value_sum = np.bincount(inv, weights=qty)
    num = _kernel_sum(uniq, signed_sum, clock_ts, grid, span)
    den = _kernel_sum(uniq, value_sum, clock_ts, grid, span)
    return num / np.where(den == 0.0, np.nan, den)


def _mirror_trades(trades, c=100.0):
    """Reflect every trade price `px -> c**2/px` AND flip every side (buy<->sell); qty is invariant. The
    data-level operation a signed-flow `mirror` must commute with (price is unused, the side flip negates)."""
    rx, px, lifts, qty = trades
    return (rx, c * c / px, 1.0 - lifts, qty)


def _post_warmup(shared):
    """The event timestamps at-or-after the first DECAY tick. This is a *pure trade-flow* feature: it
    injects ONLY on trade timestamps, so before the first trade tick the only event_ts are book rows
    where `flow_at` returns un-decayed warm-up garbage (documented; dropped downstream by anchor
    sampling, which sits post-warm-up). The oracle is NaN there (no trade event yet). We therefore
    compare the two builds on this post-warm-up region — exactly the region any downstream read sees."""
    return shared.event_ts >= shared.clock[0]


_SPANS = (2, 50)                 # spans where the feature is DEFINED (span=1 handled separately, below)


# --------------------------------------------------------------------------------------------------
# (a) vectorized vs the independent volume-weighted oracle, across spans where defined
# --------------------------------------------------------------------------------------------------
def test_vectorized_matches_oracle():
    books, trades, _ = _market()
    raw, shared, config = _inputs(books, trades, sources=("bin", "okx"))
    spec = base.get("trade_flow_imbalance")
    post = _post_warmup(shared)
    for n in _SPANS:
        got = spec.vectorized(raw, shared, config, n)["byb"][post]
        ref = _imbalance_oracle(trades["byb"], shared.clock, shared.event_ts, n)[post]
        ok = np.isfinite(got) & np.isfinite(ref)
        assert ok.sum() > 100, f"too few finite points at span={n}"
        np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-9, atol=1e-12)
        # consistent NaN where undefined: both builds NaN at exactly the same event timestamps
        np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))


# --------------------------------------------------------------------------------------------------
# a volume-weighted signed fraction lives in [-1, 1] and is finite wherever the inputs exist
# --------------------------------------------------------------------------------------------------
def test_value_is_bounded_and_finite_where_defined():
    books, trades, _ = _market(seed=8)
    raw, shared, config = _inputs(books, trades, sources=("bin", "okx"))
    spec = base.get("trade_flow_imbalance")
    post = _post_warmup(shared)
    for n in _SPANS:
        out = spec.vectorized(raw, shared, config, n)
        for ex in ("byb", "bin", "okx"):
            got = out[ex][post]
            ref = _imbalance_oracle(trades[ex], shared.clock, shared.event_ts, n)[post]
            defined = np.isfinite(ref)
            assert defined.sum() > 100
            assert np.all(np.isfinite(got[defined])), f"non-finite value at span={n}, ex={ex}"
            assert np.all(got[defined] >= -1.0 - 1e-9)
            assert np.all(got[defined] <= 1.0 + 1e-9)


# --------------------------------------------------------------------------------------------------
# (b) per-venue fan-out: one independent leg per exchange, each that venue's own-tape build
# --------------------------------------------------------------------------------------------------
def test_fans_out_over_all_exchanges():
    books, trades, _ = _market(seed=3)
    raw, shared, config = _inputs(books, trades, sources=("bin", "okx"))
    spec = base.get("trade_flow_imbalance")
    for params in (2, 20):
        out = spec.vectorized(raw, shared, config, params)
        assert set(out) == {"byb", "bin", "okx"}
        assert tuple(spec.keys_for(config, params)) == ("byb", "bin", "okx")
        # each leg reads ONLY its own venue's tape (given the shared clock/grid of the present set): it is
        # invariant to the target/source PARTITION. Re-pick the target -> identical legs, byte-for-byte.
        for tgt in ("bin", "okx"):
            srcs = tuple(s for s in ("byb", "bin", "okx") if s != tgt)
            r2, s2, c2 = _inputs(books, trades, target_ex=tgt, sources=srcs)
            assert np.array_equal(shared.event_ts, s2.event_ts)
            out2 = spec.vectorized(r2, s2, c2, params)
            for ex in ("byb", "bin", "okx"):
                np.testing.assert_array_equal(out[ex], out2[ex])


# --------------------------------------------------------------------------------------------------
# (c) the mirror COMMUTATION invariant against the FULL tape reflection (reflect prices AND flip sides)
# --------------------------------------------------------------------------------------------------
def test_mirror_commutes_with_reflection():
    books, trades, _ = _market(seed=4)
    mtrades = {ex: _mirror_trades(t) for ex, t in trades.items()}     # FULL reflection: price + side
    raw, shared, config = _inputs(books, trades, sources=("bin", "okx"))
    mraw, mshared, mconfig = _inputs(books, mtrades, sources=("bin", "okx"))
    spec = base.get("trade_flow_imbalance")
    assert spec.mirror is not None
    for params in _SPANS:
        feat = spec.vectorized(raw, shared, config, params)
        refl = spec.vectorized(mraw, mshared, mconfig, params)
        for ex in ("byb", "bin", "okx"):
            lhs = spec.mirror(feat[ex])                          # declared closed-form reflection (np.negative)
            ok = np.isfinite(lhs) & np.isfinite(refl[ex])
            assert ok.sum() > 100, f"too few finite points at span={params}, ex={ex}"
            np.testing.assert_allclose(lhs[ok], refl[ex][ok], rtol=1e-6, atol=1e-12)


# --------------------------------------------------------------------------------------------------
# span=1 (alpha=1): genuinely undefined for a pure trade-flow EMA read after the decay -> a consistent
# NaN both builds agree on (the AUTHORING alpha=1 Do-rule fallback), never a spurious finite/inf.
# --------------------------------------------------------------------------------------------------
def test_span1_is_a_consistent_nan_both_builds_agree_on():
    books, trades, anchor_ts = _market(seed=9)
    raw, shared, config = _inputs(books, trades, sources=("bin", "okx"))
    spec = base.get("trade_flow_imbalance")
    post = _post_warmup(shared)                         # the region any downstream read sees (post first tick)
    for ex in ("byb", "bin", "okx"):
        got = spec.vectorized(raw, shared, config, 1)[ex][post]
        ref = _imbalance_oracle(trades[ex], shared.clock, shared.event_ts, 1)[post]
        assert np.all(np.isnan(got)), "span=1 must be a consistent NaN, never a spurious finite/inf"
        assert np.all(np.isnan(ref))
    # the streaming build agrees (it too reads 0/0 after the alpha=1 decay): 0 comparable points, no diff.
    ctx = _ctx_for_parity(books, trades, anchor_ts, sources=("bin", "okx"))
    rep = parity_check(ctx, spec, [1], n_grid=len(anchor_ts), tol=1e-9)
    assert all(n == 0 for n in rep.n_points.values()), "span=1 streaming should also be all-NaN"
    assert all(np.isnan(d) for d in rep.max_diff.values()), "no finite disagreement at span=1"


# --------------------------------------------------------------------------------------------------
# streaming (LiveTradeFlowImbalance) vs vectorized — synthetic parity on the spans where defined
# --------------------------------------------------------------------------------------------------
def test_streaming_matches_vectorized():
    books, trades, anchor_ts = _market(seed=5, exes=("byb", "bin"))
    ctx = _ctx_for_parity(books, trades, anchor_ts, sources=("bin",))
    rep = parity_check(ctx, base.get("trade_flow_imbalance"), list(_SPANS), n_grid=len(anchor_ts), tol=1e-9)
    assert rep.passed, str(rep)
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
    # span=1 (alpha=1, the most-used fast leg): genuinely undefined -> a consistent NaN both builds agree on.
    rep1 = parity_check(ctx, spec, [1], tol=1e-6)
    assert all(n == 0 for n in rep1.n_points.values()), str(rep1)
    assert all(np.isnan(d) for d in rep1.max_diff.values()), str(rep1)
    # span=1 is all-NaN where any downstream read lands (the anchor grid, post-warm-up): the raw event
    # grid carries pre-first-tick warm-up garbage from `flow_at` that anchor sampling drops.
    for ex in spec.keys_for(ctx.config, 1):
        ev = spec.vectorized(ctx.raw_data, ctx.shared_data, ctx.config, 1)[ex]
        assert np.all(np.isnan(ctx.sample_to_anchor(ev))), f"span=1 should be all-NaN on real block ({ex})"
