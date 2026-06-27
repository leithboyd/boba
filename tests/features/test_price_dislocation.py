"""Self-contained test suite for `boba.features.price_dislocation` — the worked-example cross-venue gap.

The feature is now a standalone transform `vectorized(raw_data, shared_data, config, (n_fast, n_slow)) ->
{source -> value per shared_data.event_ts}`. The AUTHORING.md validation trio + streaming coverage, on
synthetic data with NO shared code with production:

  (a) `test_vectorized_matches_independent_oracle`  vectorized vs an INDEPENDENT dead-simple per-tick
      event-loop oracle, on the event grid, across span pairs incl. a span=1 fast leg.
  (b) `test_fans_out_one_independent_leg_per_source`  one independent leg per foreign source.
  (c) `test_mirror_commutes_with_full_book_reflection`  the COMMUTATION invariant vs the full reflection.
  (d) `test_span_one_leg_finite_where_mids_exist_and_nan_otherwise`  span=1 (α=1) finite wherever both
      mids exist, a consistent NaN where a source has not quoted — the AUTHORING span=1 Do-rule.
  (e) `test_streaming_matches_vectorized`  drive `LiveDislocation` over a synthetic raw-event stream and
      assert it matches the (sampled) vectorized via `parity_check`.
  (f) `test_real_block_price_dislocation_parity`  streaming-vs-vectorized parity on a real block (DATA_DIR-gated).
"""
import numpy as np
import pytest

import boba.io as io
from boba.features import base
import boba.features.price_dislocation  # noqa: F401  (registers the feature)
from boba.features.base import Config, FrontLevels, ListingRaw, RawData, Trade
from boba.features.shared import _ffill, build_shared_data
from boba.research.screening import RawEventStream, ScreeningContext, parity_check


# --------------------------------------------------------------------------------------------------
# synthetic standalone inputs (own builders — no shared code with production)
# --------------------------------------------------------------------------------------------------
def _raw_data(mids, trade_ts, coin="x") -> RawData:
    """`{short_ex -> (rx, mid)}` + a shared trade clock -> RawData. Each listing's front_levels carries
    `bid = ask = mid` (so `(bid+ask)/2 == mid` and a price reflection keeps the mirror clean); the decay
    clock lives on the FIRST listing's trades (dummy payload). exchange_time is set to rx."""
    listings: dict[str, ListingRaw] = {}
    nt = len(trade_ts)
    for i, (ex, (rx, mid)) in enumerate(mids.items()):
        rx = rx.astype(np.int64)
        front = FrontLevels(rx, rx, mid.copy(), np.ones(len(rx)), mid.copy(), np.ones(len(rx)))
        if i == 0:
            t = trade_ts.astype(np.int64)
            trade = Trade(t, t, np.full(nt, 100.0), np.ones(nt), np.ones(nt))
        else:
            trade = Trade(np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0), np.empty(0), np.empty(0))
        listings[f"{ex}_{coin}"] = ListingRaw(front_levels=front, trade=trade)
    return RawData(listings=listings)


def _inputs(mids, trade_ts, target_ex="byb", sources=("bin", "okx"), coin="x"):
    """-> (raw_data, shared_data, config) for the standalone feature build."""
    raw = _raw_data(mids, trade_ts, coin)
    config = Config(f"{target_ex}_{coin}", tuple(f"{s}_{coin}" for s in sources), coin,
                    {f"{ex}_{coin}": "front_levels" for ex in mids}, yardstick_span=10)
    return raw, build_shared_data(raw, config), config


def _raw_events(mids, trade_ts, coin="x") -> RawEventStream:
    """Per-venue raw book rows (bid=ask=mid) + one dummy trade per trade-timestamp (listing 0)."""
    listings = tuple(f"{ex}_{coin}" for ex in mids)
    cols: dict[str, list] = {k: [] for k in "rx kind lid t a b c d".split()}

    def add(rx, kind, lid, t, a, b, c, d):
        n = len(rx)
        cols["rx"].append(rx.astype(np.int64)); cols["kind"].append(np.full(n, kind, np.int8))
        cols["lid"].append(np.full(n, lid, np.int8)); cols["t"].append(t.astype(np.int64))
        for k, v in (("a", a), ("b", b), ("c", c), ("d", d)):
            cols[k].append(v.astype(float))

    for lid, (ex, (rx, mid)) in enumerate(mids.items()):
        add(rx, 0, lid, rx, mid, mid, np.ones(len(rx)), np.ones(len(rx)))  # BookEvent(listing, rx, exch_time, bid, ask, bid_qty, ask_qty); fixture sets exch_time == rx
    nt = len(trade_ts)
    add(trade_ts, 1, 0, trade_ts, np.full(nt, 100.0), np.ones(nt), np.ones(nt), np.full(nt, np.nan))
    C = {k: np.concatenate(v) for k, v in cols.items()}
    order = np.lexsort((C["kind"], C["rx"]))
    return RawEventStream(*(C[k][order] for k in "rx kind lid t a b c d".split()), listings)


def _ctx_for_parity(mids, trade_ts, anchor_ts, target_ex="byb", sources=("bin", "okx")) -> ScreeningContext:
    """A minimal ScreeningContext carrying the standalone inputs + the raw-event stream + an anchor grid,
    enough for `parity_check` (streaming read at anchors vs the sampled vectorized)."""
    raw, shared, config = _inputs(mids, trade_ts, target_ex, sources)
    return ScreeningContext(
        block="syn", coin="x", target=config.target_listing, sources=config.other_listings, horizon_ns=0,
        yardstick_span=10, mid_stream={}, merged_ts=shared.clock, anchor_ts=anchor_ts,
        sigma_at_anchor=np.empty(0), lam_at_anchor=np.empty(0), price_target=np.empty(0),
        rate_target=np.empty(0), base=[], vol_level=np.empty(0), rate_level=np.empty(0), vol_regime=np.empty(0),
        raw_events=_raw_events(mids, trade_ts), raw_data=raw, shared_data=shared, config=config)


def _synthetic_mids(seed=0, n=4000, c=100.0):
    """`{ex -> (rx, mid)}` for byb + two sources, a geometric-random-walk mid each on its own rx clock,
    plus the shared trade clock and an anchor grid. All quotes start before the first trade so every mid
    is finite (the warm-up NaN path is exercised by `_late_quote_mids`)."""
    rng = np.random.default_rng(seed)
    mids = {}
    for k, ex in enumerate(("byb", "bin", "okx")):
        rx = (np.arange(0, n + 5) * 8 + k).astype(np.int64)               # a quote every ~8 ns, offset per ex
        mid = c * np.exp(np.cumsum(rng.standard_normal(n + 5) * 1e-4))
        mids[ex] = (rx, mid)
    last = min(int(rx[-1]) for rx, _ in mids.values())
    trade_ts = np.unique(np.concatenate([
        (np.arange(1, n + 1) * 8)[::3],                                   # trades coincident-ish with quotes
        (np.arange(1, n + 1) * 8)[1::7] + 2,                             # trade-only timestamps between quotes
    ])).astype(np.int64)
    trade_ts = trade_ts[trade_ts <= last]
    return mids, trade_ts


def _late_quote_mids(seed=1, n=4000, c=100.0):
    """Same, but `okx` does not quote until well after the warm-up — so its leg is a consistent NaN on the
    early event timestamps (a source that has not quoted yet) and finite once it does."""
    mids, trade_ts = _synthetic_mids(seed=seed, n=n, c=c)
    rx, mid = mids["okx"]
    start = rx[len(rx) // 2]                                              # okx silent until mid-grid
    keep = rx >= start
    mids["okx"] = (rx[keep], mid[keep])
    return mids, trade_ts


# --------------------------------------------------------------------------------------------------
# INDEPENDENT oracle — an explicit one-tick-at-a-time loop, NO shared code with the production build.
# --------------------------------------------------------------------------------------------------
def _ffill_scalar(rx, val, t):
    """The last `val` whose `rx <= t`, or None if none — a dead-simple causal forward-fill at a scalar t."""
    j = int(np.searchsorted(rx, t, "right")) - 1
    return None if j < 0 else float(val[j])


def _sigma_oracle(byb_mids, clock, event_ts, span):
    """σ_ev at each `event_ts` by a dead-simple loop: a KernelMeanEMA (E/W) of squared byb log-mid moves,
    decayed once per trade tick, injected on each real target mid CHANGE; read live (committed-per-tick +
    the partial epoch since the last tick), then sqrt(E/W). `span` = yardstick span (α = 2/(span+1)).

    One ordered walk over the union of move-timestamps and clock ticks records (E, W) at each clock tick
    (committed). Each event_ts then reads the committed E/W of its last tick plus the moves landed since."""
    byb_rx, byb_mid = byb_mids
    a = 2.0 / (span + 1.0)
    # the target-move stream: collapse same-rx mids to the last, keep only real changes.
    keep = np.concatenate([byb_rx[1:] != byb_rx[:-1], [True]])
    t_rx, t_mid = byb_rx[keep], byb_mid[keep]
    lm = np.log(t_mid)
    dlr = np.empty_like(lm); dlr[0] = 0.0; dlr[1:] = np.diff(lm)
    mv = dlr != 0.0
    moves: dict[int, float] = {}                                         # rx -> summed squared move at that rx
    for r, v in zip(t_rx[mv], dlr[mv] ** 2):
        moves[int(r)] = moves.get(int(r), 0.0) + v

    # committed (E, W) AFTER each clock tick: a single ordered walk, inject (a move) then decay (a tick).
    tickset = set(int(x) for x in clock)
    all_ts = sorted(set(moves) | tickset)
    E = W = 0.0
    E_at_tick = np.empty(len(clock)); W_at_tick = np.empty(len(clock)); j = 0
    for ts in all_ts:
        if ts in moves:
            E += a * moves[ts]; W += a
        if ts in tickset:
            E *= (1.0 - a); W *= (1.0 - a)
            E_at_tick[j] = E; W_at_tick[j] = W; j += 1

    # cumulative injected E / W over the move stream, for the partial epoch since the last tick.
    mv_rx = np.array(sorted(moves)) if moves else np.empty(0, np.int64)
    mv_e = np.array([moves[int(r)] for r in mv_rx]) if moves else np.empty(0)
    csE = np.concatenate([[0.0], np.cumsum(a * mv_e)])
    csW = np.concatenate([[0.0], np.cumsum(np.full(len(mv_rx), a))])

    out = np.full(len(event_ts), np.nan)
    for i, t in enumerate(event_ts):
        ti = int(np.searchsorted(clock, t, "right")) - 1                 # last clock tick <= t
        Ec = E_at_tick[ti] if ti >= 0 else 0.0
        Wc = W_at_tick[ti] if ti >= 0 else 0.0
        last_tick = int(clock[ti]) if ti >= 0 else None
        hi = int(np.searchsorted(mv_rx, t, "right"))                     # moves at-or-before t
        lo = int(np.searchsorted(mv_rx, last_tick, "right")) if last_tick is not None else 0
        Ep = csE[hi] - csE[lo]; Wp = csW[hi] - csW[lo]                   # partial epoch since the last tick
        E, W = Ec + Ep, Wc + Wp
        out[i] = np.sqrt(E / W) if W > 0.0 else np.nan
    return out


def _dislocation_leg_oracle(src_mids, byb_mids, clock, event_ts, span):
    """The fast/slow live-front leg's NUMERATOR EMA at `span`, by a dead-simple loop.

    Committed EMA on the trade clock of the log gap `g = log(mid_src) - log(mid_byb)` (a missing mid on
    the clock contributes a 0 gap, matching the build's `nan_to_num`), read at each event_ts as the live
    front `(1-a)*committed + a*g_fresh` where `g_fresh` is the freshest gap AT the event_ts (NaN if either
    side has not quoted there). `span == 1` (a = 1) ⇒ the live front collapses to `g_fresh`."""
    src_rx, src_mid = src_mids
    byb_rx, byb_mid = byb_mids
    a = 2.0 / (span + 1.0)

    # committed EMA over the trade clock (y[-1] = 0), one step per clock tick.
    committed_at_tick = np.empty(len(clock))
    ema = 0.0
    for i, ts in enumerate(clock):
        ms = _ffill_scalar(src_rx, src_mid, ts)
        mb = _ffill_scalar(byb_rx, byb_mid, ts)
        g = 0.0 if (ms is None or mb is None) else (np.log(ms) - np.log(mb))   # nan_to_num(., 0.0)
        ema = (1.0 - a) * ema + a * g
        committed_at_tick[i] = ema

    out = np.full(len(event_ts), np.nan)
    for i, t in enumerate(event_ts):
        ms = _ffill_scalar(src_rx, src_mid, t)
        mb = _ffill_scalar(byb_rx, byb_mid, t)
        if ms is None or mb is None:
            continue                                                          # no fresh gap -> NaN leg
        g_fresh = np.log(ms) - np.log(mb)
        tick = int(np.searchsorted(clock, t, "right")) - 1
        committed = committed_at_tick[tick] if tick >= 0 else 0.0
        out[i] = (1.0 - a) * committed + a * g_fresh
    return out


def _dislocation_oracle(src_mids, byb_mids, clock, event_ts, sigma, n_fast, n_slow):
    """The whole feature for one source: (fast leg - slow leg) / σ_ev, both live-front EMAs of the gap."""
    fast = _dislocation_leg_oracle(src_mids, byb_mids, clock, event_ts, n_fast)
    slow = _dislocation_leg_oracle(src_mids, byb_mids, clock, event_ts, n_slow)
    return (fast - slow) / sigma


SPANS = [(1, 200), (1, 100), (10, 100), (50, 500), (200, 2000)]


# --------------------------------------------------------------------------------------------------
# (a) vectorized vs the independent oracle, across span pairs incl. a span=1 fast leg
# --------------------------------------------------------------------------------------------------
def test_vectorized_matches_independent_oracle():
    mids, trade_ts = _synthetic_mids()
    raw, shared, config = _inputs(mids, trade_ts)
    spec = base.get("price_dislocation")
    sigma = _sigma_oracle(mids["byb"], shared.clock, shared.event_ts, config.yardstick_span)
    np.testing.assert_allclose(sigma, shared.vol_yardstick, rtol=1e-9, atol=1e-12,
                               equal_nan=True)                            # σ_ev oracle ties the shared yardstick
    for params in SPANS:
        got = spec.vectorized(raw, shared, config, params)
        assert set(got) == {"bin", "okx"}
        for ex in ("bin", "okx"):
            ref = _dislocation_oracle(mids[ex], mids["byb"], shared.clock, shared.event_ts, sigma, *params)
            ok = np.isfinite(got[ex]) & np.isfinite(ref)
            assert ok.sum() > 100
            np.testing.assert_allclose(got[ex][ok], ref[ok], rtol=1e-7, atol=1e-9)
            np.testing.assert_array_equal(np.isnan(got[ex]), np.isnan(ref))


# --------------------------------------------------------------------------------------------------
# (b) per-venue fan-out: one independent leg per source, each equal to that source's solo build
# --------------------------------------------------------------------------------------------------
def test_fans_out_one_independent_leg_per_source():
    """One independent leg per foreign source; each equals that source's own solo build (no cross-talk).
    A solo build omits the OTHER source's event timestamps, so we causal-sample both onto the full event
    grid (a leg is piecewise-constant between events) and assert equality there."""
    mids, trade_ts = _synthetic_mids(seed=4)
    raw, shared, config = _inputs(mids, trade_ts, sources=("bin", "okx"))
    spec = base.get("price_dislocation")
    params = (10, 100)
    out = spec.vectorized(raw, shared, config, params)
    assert set(out) == {"bin", "okx"}                                    # one leg per foreign source
    assert tuple(spec.keys_for(config, params)) == ("bin", "okx")
    for ex in ("bin", "okx"):
        solo_mids = {"byb": mids["byb"], ex: mids[ex]}
        r1, s1, c1 = _inputs(solo_mids, trade_ts, sources=(ex,))
        solo = _ffill(s1.event_ts, spec.vectorized(r1, s1, c1, params)[ex], shared.event_ts)
        ok = np.isfinite(out[ex]) & np.isfinite(solo)
        assert ok.sum() > 100
        np.testing.assert_allclose(out[ex][ok], solo[ok], rtol=1e-9, atol=1e-12)  # leg independent
        np.testing.assert_array_equal(np.isnan(out[ex]), np.isnan(solo))


# --------------------------------------------------------------------------------------------------
# (c) the mirror COMMUTATION invariant against the full book reflection
# --------------------------------------------------------------------------------------------------
def _mirror_mids(mids, c=100.0):
    """Reflect every venue's mid through the fixed price level c (`mid -> c**2/mid`, i.e.
    `log mid -> 2 ln c - log mid`). The exact data-level reflection the feature's `mirror` must commute with."""
    return {ex: (rx, c * c / mid) for ex, (rx, mid) in mids.items()}


def test_mirror_commutes_with_full_book_reflection():
    c = 100.0
    mids, trade_ts = _synthetic_mids(seed=2, c=c)
    raw, shared, config = _inputs(mids, trade_ts)
    mraw, mshared, mconfig = _inputs(_mirror_mids(mids, c), trade_ts)
    spec = base.get("price_dislocation")
    assert spec.mirror is not None
    for params in SPANS:
        feat = spec.vectorized(raw, shared, config, params)              # feature(books)
        refl = spec.vectorized(mraw, mshared, mconfig, params)           # feature(mirror_books(books))
        for ex in ("bin", "okx"):
            lhs = spec.mirror(feat[ex])                                   # mirror(feature(books))
            ok = np.isfinite(lhs) & np.isfinite(refl[ex])
            assert ok.sum() > 100
            np.testing.assert_allclose(lhs[ok], refl[ex][ok], rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------------------------------
# (d) span = 1 (α = 1): finite wherever both mids exist, a consistent NaN where a source has not quoted
# --------------------------------------------------------------------------------------------------
def test_span_one_leg_finite_where_mids_exist_and_nan_otherwise():
    """AUTHORING's span=1 Do-rule: at α=1 the live front collapses to the fresh gap, so the value must be
    FINITE wherever both mids exist, and a CONSISTENT NaN exactly where a source has not quoted yet."""
    mids, trade_ts = _late_quote_mids()
    raw, shared, config = _inputs(mids, trade_ts)
    spec = base.get("price_dislocation")

    okx_rx = mids["okx"][0]
    okx_quoted = np.array([np.searchsorted(okx_rx, t, "right") > 0 for t in shared.event_ts])
    # σ_ev is NaN during warm-up; only judge where the divide is finite/non-zero.
    sig_ok = np.isfinite(shared.vol_yardstick) & (shared.vol_yardstick > 0.0)
    assert (okx_quoted & sig_ok).sum() > 50 and ((~okx_quoted) & sig_ok).sum() > 50  # both branches exercised

    for params in [(1, 200), (1, 100)]:                                  # a span=1 fast leg
        out = spec.vectorized(raw, shared, config, params)
        # bin quotes from the start -> finite wherever σ_ev is defined (no spurious inf / nan).
        assert np.all(np.isfinite(out["bin"][sig_ok]))
        # okx: finite exactly where it has quoted (and σ_ev defined), NaN where it has not.
        assert np.all(np.isfinite(out["okx"][okx_quoted & sig_ok]))
        assert np.all(np.isnan(out["okx"][~okx_quoted]))
        assert not np.any(np.isinf(out["okx"]))


# --------------------------------------------------------------------------------------------------
# (e) streaming (LiveDislocation) vs vectorized — synthetic parity, including a span=1 fast leg
# --------------------------------------------------------------------------------------------------
def test_streaming_matches_vectorized():
    mids, trade_ts = _synthetic_mids(seed=7)
    last = min(int(rx[-1]) for rx, _ in mids.values())
    anchor_ts = np.arange(int(trade_ts[60]), last, 37, dtype=np.int64)
    ctx = _ctx_for_parity(mids, trade_ts, anchor_ts, sources=("bin", "okx"))
    rep = parity_check(ctx, base.get("price_dislocation"), [(1, 100), (10, 500)],
                       n_grid=len(anchor_ts), tol=1e-9)
    assert rep.passed, str(rep)


# --------------------------------------------------------------------------------------------------
# (f) real-block parity (skipped without DATA_DIR) — the streaming build reproduces the vectorized one.
# --------------------------------------------------------------------------------------------------
@pytest.mark.skipif(getattr(io, "DATA_DIR", None) is None, reason="no DATA_DIR configured")
def test_real_block_price_dislocation_parity():
    from boba.research.screening import build_context

    ctx = build_context(hours=2)
    spec = base.get("price_dislocation")
    rep = parity_check(ctx, spec, [(1, 100), (10, 500)], tol=1e-6)       # a span=1 fast leg in the sweep
    assert rep.passed, str(rep)
