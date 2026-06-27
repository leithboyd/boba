"""Self-contained test suite for `boba.features.flow_persistence`.

`flow_persistence` is the trade-tick EMA of consecutive per-timestamp trade-flow sign agreement
`eps_t * eps_prev` per venue (see the module docstring). It is now a standalone transform
`vectorized(raw_data, shared_data, config, N) -> {ex -> value per shared_data.event_ts}`. This file
builds the standalone inputs directly (the `_inputs` pattern) and validates against an INDEPENDENT,
dead-simple oracle (an explicit per-event loop sharing NO code with the production build), on the
event grid.

It covers the AUTHORING.md validation trio plus the span=1 boundary:
  - vectorized vs the independent oracle on synthetic data (spans incl. span=1);
  - the mirror COMMUTATION invariant -- reflecting the tape (reflect trade prices AND flip side/qty)
    and rebuilding must leave the value UNCHANGED, because `flow_persistence` is EVEN (mirror=identity);
  - per-venue fan-out -- one independent leg per exchange, each = its own single-venue build;
  - span=1 finiteness -- the value is finite wherever inputs exist, NaN (consistently) where undefined;
  - same-timestamp ties (net side == 0) are skipped and do NOT advance `eps_prev`;
  - synthetic + real-block streaming-vs-vectorized parity (the real block skipped without DATA_DIR).
"""
import numpy as np
import pytest

import boba.io as io
from boba.features import base
import boba.features.flow_persistence  # noqa: F401  (registers the spec)
from boba.features.base import Config, FrontLevels, ListingRaw, RawData, Trade
from boba.features.shared import build_shared_data
from boba.research.screening import RawEventStream, ScreeningContext, parity_check


# --------------------------------------------------------------------------------------------------
# synthetic standalone inputs (own builders — no shared code with production)
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
    rx[50:55] = rx[50]          # same-timestamp trade burst -> netted to one sign
    rx[250:254] = rx[250]
    return rx, px, lifts, qty


def _raw_data(books, trades, coin="x") -> RawData:
    """`{short_ex -> book}` + `{short_ex -> trade tuple}` -> RawData. Each venue carries its own
    front_levels (for the event grid / mid) and trade stream; exchange_time is set to rx (irrelevant
    to flow_persistence, which reads price/side/qty/rx only)."""
    listings: dict[str, ListingRaw] = {}
    for ex in books:
        rxb, bid, bq, ask, aq = books[ex]
        rxb = rxb.astype(np.int64)
        front = FrontLevels(rxb, rxb, bid, bq, ask, aq)
        rxt, px, lifts, qty = trades[ex]
        rxt = rxt.astype(np.int64)
        trade = Trade(rxt, rxt, px, lifts, qty)
        listings[f"{ex}_{coin}"] = ListingRaw(front_levels=front, trade=trade)
    return RawData(listings=listings)


def _inputs(books, trades, target_ex="byb", sources=(), coin="x"):
    """-> (raw_data, shared_data, config) for the standalone feature build."""
    raw = _raw_data(books, trades, coin)
    config = Config(f"{target_ex}_{coin}", tuple(f"{s}_{coin}" for s in sources), coin,
                    {f"{ex}_{coin}": "front_levels" for ex in books}, yardstick_span=25)
    return raw, build_shared_data(raw, config), config


def _raw_events(books, trades, coin="x") -> RawEventStream:
    """Per-venue raw book rows + raw trade rows, for the streaming parity driver."""
    listings = tuple(f"{ex}_{coin}" for ex in books)
    cols: dict[str, list] = {k: [] for k in "rx kind lid t a b c d".split()}

    def add(rx, kind, lid, t, a, b, c, d):
        n = len(rx)
        cols["rx"].append(rx.astype(np.int64)); cols["kind"].append(np.full(n, kind, np.int8))
        cols["lid"].append(np.full(n, lid, np.int8)); cols["t"].append(t.astype(np.int64))
        for k, v in (("a", a), ("b", b), ("c", c), ("d", d)):
            cols[k].append(v.astype(float))

    for lid, ex in enumerate(books):
        rx, bid, bq, ask, aq = books[ex]
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
        yardstick_span=25, mid_stream={}, merged_ts=shared.clock, anchor_ts=anchor_ts,
        sigma_at_anchor=np.empty(0), lam_at_anchor=np.empty(0), price_target=np.empty(0),
        rate_target=np.empty(0), base=[], vol_level=np.empty(0), rate_level=np.empty(0), vol_regime=np.empty(0),
        raw_events=_raw_events(books, trades), raw_data=raw, shared_data=shared, config=config)


def _synthetic_market(seed=0, target_ex="byb", sources=("bin", "okx")):
    """-> (books, trades, raw, shared, config, anchor_ts) for the target + each source venue."""
    exes = tuple(dict.fromkeys((target_ex,) + tuple(sources)))
    books = {ex: _synthetic_book(seed + i) for i, ex in enumerate(exes)}
    trades = {ex: _synthetic_trades(seed + 10 + i) for i, ex in enumerate(exes)}
    raw, shared, config = _inputs(books, trades, target_ex, sources)
    anchor_ts = shared.clock[150:-150:7]
    return books, trades, raw, shared, config, anchor_ts


# --------------------------------------------------------------------------------------------------
# INDEPENDENT, dead-simple oracle — an explicit per-event loop, NO shared code with production.
# Implementable from the feature's written definition alone:
#   net each rx-timestamp's signs (+1 lifts the ask, -1 hits the bid), take the sign -> eps_t,
#   skip exact ties (eps_t == 0), form eps_t * eps_prev against the previous NON-ZERO sign, then run a
#   sparse-flow E/W trade-tick EMA of that product (inject at the event ts, decay once per trade ts).
# --------------------------------------------------------------------------------------------------
def _eps_stream_oracle(trade):
    """`(rx, px, lifts, qty)` -> the per-timestamp non-zero sign stream `(ts, eps)`, by an explicit loop."""
    rx, px, lifts, qty = trade
    by_ts: dict[int, float] = {}
    order = np.argsort(rx, kind="stable")
    for i in order:
        if not (px[i] > 0.0 and qty[i] > 0.0 and np.isfinite(px[i])
                and np.isfinite(qty[i]) and np.isfinite(lifts[i])):
            continue
        s = 1.0 if lifts[i] > 0.0 else -1.0
        by_ts[int(rx[i])] = by_ts.get(int(rx[i]), 0.0) + s
    ts_out, eps_out = [], []
    for ts in sorted(by_ts):
        net = by_ts[ts]
        e = 1.0 if net > 0.0 else (-1.0 if net < 0.0 else 0.0)
        if e != 0.0:                                  # exact ties (net == 0) are skipped
            ts_out.append(ts)
            eps_out.append(e)
    return np.array(ts_out, dtype=np.int64), np.array(eps_out, dtype=float)


def _persistence_pairs_oracle(trade):
    """`(ts, eps_t * eps_prev)` over consecutive non-zero per-timestamp signs (eps_prev = previous one)."""
    ts, eps = _eps_stream_oracle(trade)
    if len(eps) < 2:
        return ts[:0], eps[:0]
    return ts[1:], eps[1:] * eps[:-1]


def _flow_persistence_oracle(trade, clock, grid, span):
    """E/W sparse-flow trade-tick EMA of `eps_t * eps_prev`, read at each `grid` ts, by a plain loop.
    Inject one weight-1 sample at the event ts; decay once per trade-clock tick (inject-then-decay).
    α = 2/(span+1); span=1 => α=1. Reads E/W after the last event at-or-before each grid point."""
    ev_ts, vals = _persistence_pairs_oracle(trade)
    a = 2.0 / (span + 1.0)
    beta = 1.0 - a
    by_ts: dict[int, float] = {}
    for ts, v in zip(ev_ts, vals):
        by_ts[int(ts)] = by_ts.get(int(ts), 0.0) + float(v)   # at most one pair per ts, but stay general
    trades_set = set(int(t) for t in clock)
    all_ts = sorted(set(by_ts) | trades_set)
    E = W = 0.0
    ts_arr, Es, Ws = [], [], []
    for ts in all_ts:
        if ts in by_ts:                       # inject one weight-1 sample of the summed product
            E += a * by_ts[ts]
            W += a
        if ts in trades_set:                  # decay once on the shared trade clock (inject-then-decay)
            E *= beta
            W *= beta
        ts_arr.append(ts); Es.append(E); Ws.append(W)
    out = np.full(len(grid), np.nan)
    if not ts_arr:
        return out
    ts_arr, Es, Ws = np.array(ts_arr), np.array(Es), np.array(Ws)
    j = np.searchsorted(ts_arr, grid, "right") - 1
    ok = j >= 0
    Ej, Wj = Es[j[ok]], Ws[j[ok]]
    out[ok] = np.where(Wj > 0.0, Ej / np.where(Wj == 0.0, np.nan, Wj), np.nan)
    return out


def _mirror_trades(trades, c=100.0):
    """Reflect a trade tape through price level c: price p -> c**2/p, aggressor side flips (buy<->sell),
    qty (a count) is unchanged — the data-level operation the EVEN `mirror` must commute with."""
    return {ex: (rx, c * c / px, 1.0 - lifts, qty) for ex, (rx, px, lifts, qty) in trades.items()}


# --------------------------------------------------------------------------------------------------
# (a) vectorized vs the independent oracle — span sweep including the span=1 fast leg
# --------------------------------------------------------------------------------------------------
def test_vectorized_matches_oracle():
    books, trades, raw, shared, config, _ = _synthetic_market(seed=2)
    spec = base.get("flow_persistence")
    for n in (1, 2, 50):                              # span=1 (alpha=1, no smoothing) included
        got = spec.vectorized(raw, shared, config, n)["byb"]
        ref = _flow_persistence_oracle(trades["byb"], shared.clock, shared.event_ts, n)
        ok = np.isfinite(got) & np.isfinite(ref)
        # NaN where undefined must agree on BOTH builds (no build-specific spurious finite/inf)
        np.testing.assert_array_equal(np.isfinite(got), np.isfinite(ref))
        if n == 1:
            assert ok.sum() == 0                     # span=1 collapses to a consistent NaN everywhere
        else:
            assert ok.sum() > 100
            np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------------------------------
# span=1 boundary (AUTHORING span=1 Do). At alpha=1 the EMA fully decays at every trade tick, and a
# persistence event ALWAYS lands on a trade tick (trade timestamps define the decay clock), so its one
# weight-1 sample is decayed to 0 at the same tick: E/W is a 0/0 that must resolve to a CONSISTENT NaN
# both builds agree on — never an inf or a build-specific number. We assert exactly that, and that
# wherever a finite value DOES appear (the smallest finite leg, span=2) it is finite, never inf.
# --------------------------------------------------------------------------------------------------
def test_span1_is_consistent_nan_never_inf():
    books, trades, raw, shared, config, _ = _synthetic_market(seed=7)
    spec = base.get("flow_persistence")
    for ex in ("byb", "bin", "okx"):
        got = spec.vectorized(raw, shared, config, 1)[ex]   # span = 1 => alpha = 1, the most-used fast leg
        ref = _flow_persistence_oracle(trades[ex], shared.clock, shared.event_ts, 1)
        # every persistence event coincides with a trade tick -> the value is a consistent NaN
        ev_ts, _ = _persistence_pairs_oracle(trades[ex])
        assert np.all(np.isin(ev_ts, shared.clock))  # the structural reason span=1 collapses to 0/0
        assert not np.any(np.isinf(got))             # NEVER an inf at the alpha=1 boundary
        assert np.all(np.isnan(got))                 # a consistent NaN (the 0/0 the gates mask)
        np.testing.assert_array_equal(np.isfinite(got), np.isfinite(ref))   # both builds agree

    # And the smallest leg that IS defined (span=2): finite wherever the inputs/window exist, never inf.
    for ex in ("byb", "bin", "okx"):
        got2 = spec.vectorized(raw, shared, config, 2)[ex]
        ref2 = _flow_persistence_oracle(trades[ex], shared.clock, shared.event_ts, 2)
        np.testing.assert_array_equal(np.isfinite(got2), np.isfinite(ref2))
        assert np.isfinite(got2).sum() > 50
        assert np.all(np.isfinite(got2[np.isfinite(ref2)]))
        assert not np.any(np.isinf(got2))


# --------------------------------------------------------------------------------------------------
# same-timestamp ties (net side == 0) are skipped and do NOT advance eps_prev
# --------------------------------------------------------------------------------------------------
def test_skips_exact_ties_without_advancing_prev():
    # Construct a tape with an exact tie wedged between two real signs. eps_prev must "see through"
    # the tie: the product formed after the tie pairs the post-tie sign with the PRE-tie sign.
    # timestamps: 100 -> net +1 (eps=+1); 200 -> one buy + one sell (net 0, tie, SKIPPED);
    #             300 -> net -1 (eps=-1).  The only persistence pair is eps(300)*eps(100) = -1.
    rx = np.array([100, 200, 200, 300], dtype=np.int64)
    lifts = np.array([1.0, 1.0, 0.0, 0.0])          # buy, (buy, sell -> tie), sell
    px = np.full(4, 100.0)
    qty = np.full(4, 1.0)
    one = {"byb": (rx, px, lifts, qty)}
    one_book = {"byb": _synthetic_book(0, n=400)}
    raw, shared, config = _inputs(one_book, one, target_ex="byb", sources=())
    anchor_ts = np.array([100, 200, 250, 300, 350], dtype=np.int64)

    # Oracle: exactly one persistence pair, value -1, landing at ts 300.
    ev_ts, vals = _persistence_pairs_oracle(one["byb"])
    np.testing.assert_array_equal(ev_ts, np.array([300]))
    np.testing.assert_allclose(vals, np.array([-1.0]))

    spec = base.get("flow_persistence")
    for n in (1, 2, 5):
        got = spec.vectorized(raw, shared, config, n)["byb"]
        ref = _flow_persistence_oracle(one["byb"], shared.clock, shared.event_ts, n)
        np.testing.assert_array_equal(np.isfinite(got), np.isfinite(ref))
        ok = np.isfinite(got) & np.isfinite(ref)
        np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-9, atol=1e-12)
    # With smoothing (span>=2) the lone surviving pair, eps(300)*eps(100) = -1, is the only signal:
    # at every event_ts at/after ts 300 the EMA holds exactly -1 (the tie at ts 200 injected nothing,
    # so eps_prev = the PRE-tie sign), and is NaN before any pair exists.
    got2 = spec.vectorized(raw, shared, config, 2)["byb"]
    at = shared.event_ts >= 300
    np.testing.assert_allclose(got2[at], -1.0, rtol=1e-9, atol=1e-12)
    assert np.all(np.isnan(got2[~at]))               # no pair exists before ts 300
    # At span=1 (alpha=1) the pair lands on its own trade tick and decays to a consistent NaN.
    assert np.all(np.isnan(spec.vectorized(raw, shared, config, 1)["byb"]))


# --------------------------------------------------------------------------------------------------
# (c) per-venue fan-out — one independent leg per exchange, each = its own single-venue build
# --------------------------------------------------------------------------------------------------
def _flip_sides(trade):
    """A different tape on the SAME rx stamps (flip every aggressor side) — changes the per-ts signs but
    NOT the clock/event grid. Lets us prove a leg is independent of the OTHER venues' trade contents."""
    rx, px, lifts, qty = trade
    return rx, px, 1.0 - lifts, qty


def test_fans_out_over_all_exchanges():
    books, trades, raw, shared, config, _ = _synthetic_market(seed=3)
    spec = base.get("flow_persistence")
    out = spec.vectorized(raw, shared, config, 20)
    assert set(out) == {"byb", "bin", "okx"}
    assert tuple(spec.keys_for(config, 20)) == ("byb", "bin", "okx")
    # Each leg uses ONLY that venue's own trade stream: rebuild with a DIFFERENT tape on every OTHER
    # venue (same rx stamps, flipped sides — so the shared clock/event grid are unchanged); leg `ex`
    # must stay bit-identical. (The target byb leg uses byb's flipped tape only when ex != byb.)
    for ex in ("byb", "bin", "okx"):
        scrambled = {e: (trades[e] if e == ex else _flip_sides(trades[e])) for e in ("byb", "bin", "okx")}
        r2, s2, c2 = _inputs(books, scrambled, target_ex="byb", sources=("bin", "okx"))
        np.testing.assert_array_equal(out[ex], spec.vectorized(r2, s2, c2, 20)[ex])


# --------------------------------------------------------------------------------------------------
# (b) mirror COMMUTATION invariant — reflect the FULL tape (reflect prices AND flip side/qty); for an
# EVEN feature like flow_persistence the value is UNCHANGED:  mirror(feat) == feat(mirror_books).
# --------------------------------------------------------------------------------------------------
def test_mirror_commutes_with_trade_reflection():
    books, trades, raw, shared, config, _ = _synthetic_market(seed=4)
    mtrades = _mirror_trades(trades)
    mraw, mshared, mconfig = _inputs(books, mtrades, target_ex="byb", sources=("bin", "okx"))
    spec = base.get("flow_persistence")
    assert spec.mirror is not None                   # an EVEN feature still MUST declare its reflection
    for n in (2, 50):
        feat = spec.vectorized(raw, shared, config, n)
        refl = spec.vectorized(mraw, mshared, mconfig, n)
        for ex in ("byb", "bin", "okx"):
            lhs = spec.mirror(feat[ex])              # identity for an EVEN feature
            ok = np.isfinite(lhs) & np.isfinite(refl[ex])
            assert ok.sum() > 100
            # EVEN: the value is UNCHANGED under the full tape reflection
            np.testing.assert_allclose(lhs[ok], refl[ex][ok], rtol=1e-6, atol=1e-12)
            np.testing.assert_allclose(feat[ex][ok], refl[ex][ok], rtol=1e-6, atol=1e-12)


# --------------------------------------------------------------------------------------------------
# streaming (LiveFlowPersistence) vs vectorized — synthetic parity, including a span=1 leg
# --------------------------------------------------------------------------------------------------
def test_streaming_matches_vectorized():
    books, trades, raw, shared, config, anchor_ts = _synthetic_market(seed=5)
    ctx = _ctx_for_parity(books, trades, anchor_ts, sources=("bin", "okx"))
    spec = base.get("flow_persistence")
    # The finite-valued legs carry the parity pass (span=1 is structurally all-NaN, below).
    rep = parity_check(ctx, spec, [2, 50], n_grid=len(anchor_ts), tol=1e-9)
    assert rep.passed, str(rep)
    # span=1 (alpha=1) is swept too: both builds collapse to a consistent NaN everywhere, so there is
    # nothing finite to disagree on -- assert the streaming and vectorized builds agree on that (0 pts).
    rep1 = parity_check(ctx, spec, [1], n_grid=len(anchor_ts), tol=1e-9)
    for ex in ("byb", "bin", "okx"):
        assert rep1.n_points[(1, ex)] == 0           # neither build has a finite point at span=1
        assert np.all(np.isnan(spec.vectorized(raw, shared, config, 1)[ex]))


# --------------------------------------------------------------------------------------------------
# real-block parity (skipped without DATA_DIR) — streaming reproduces vectorized, span=1 swept
# --------------------------------------------------------------------------------------------------
@pytest.mark.skipif(getattr(io, "DATA_DIR", None) is None, reason="no DATA_DIR configured")
def test_real_block_parity():
    from boba.research.screening import build_context

    ctx = build_context(hours=2)
    spec = base.get("flow_persistence")
    # Arbitrary-nanosecond event timing is exactly what the real block adds over synthetic fixtures.
    rep = parity_check(ctx, spec, [2, 100], tol=1e-6)
    assert rep.passed, str(rep)
    # span=1 swept on the real block too: a persistence event still lands on its own trade tick, so the
    # alpha=1 leg is a consistent NaN both builds agree on (no finite point to disagree on).
    rep1 = parity_check(ctx, spec, [1], tol=1e-6)
    for k in spec.keys_for(ctx.config, 1):
        assert rep1.n_points[(1, k)] == 0, str(rep1)
        assert np.all(np.isnan(spec.vectorized(ctx.raw_data, ctx.shared_data, ctx.config, 1)[k]))
