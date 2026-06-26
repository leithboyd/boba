"""Tests for the span & head SELECTION engines (boba.research.selection).

The generic machinery (the count-conditioned price target, the in-sample IC grid, the conditional
second-span partial-IC + walk-forward join, and the per-exchange-vs-single comparison) is
exercised on SYNTHETIC families with a target of KNOWN structure, so the picks are predictable and
the tests run with no DATA_DIR. A real-block integration test (skipped without DATA_DIR) builds the
price_dislocation family over the standard sweep grid and asserts the documented in-sample span picks
(price head (10, 500); rate-head abs-feature (1, 100)) reproduce the screening refactor's ground truth.
"""
import numpy as np
import pytest

import boba.io as io
from boba.research.screening import RawEventStream, ScreeningContext, _ffill
from boba.research.selection import (
    fixed_move_targets,
    ic_grid,
    ic_scan,
    per_exchange_vs_single,
    second_span_adds,
)


# --------------------------------------------------------------------------------------------------
# helpers — synthetic contexts / families
# --------------------------------------------------------------------------------------------------
def _empty_ctx(**over) -> ScreeningContext:
    """A minimal ScreeningContext with everything empty — fields are overridden per test."""
    kw = dict(
        block="syn", coin="x", target="byb_x", sources=("aaa", "bbb"), horizon_ns=0,
        yardstick_span=10, mid_stream={}, merged_ts=np.empty(0), anchor_ts=np.empty(0),
        tick_at_anchor=np.empty(0), sigma_at_anchor=np.empty(0), lam_at_anchor=np.empty(0),
        price_target=np.empty(0), rate_target=np.empty(0), base=[], vol_level=np.empty(0),
        rate_level=np.empty(0), vol_regime=np.empty(0),
        raw_events=RawEventStream(*([np.empty(0)] * 8), ()),
    )
    kw.update(over)
    return ScreeningContext(**kw)


def _grid_family(spans_fast, spans_slow, leg_of):
    """Build `{(nf, ns) -> {source -> vec}}` for every valid (nf<ns) pair via `leg_of(src, nf, ns)`."""
    sources = ("aaa", "bbb")
    fam = {}
    for nf in spans_fast:
        for ns in spans_slow:
            if nf >= ns:
                continue
            fam[(nf, ns)] = {s: leg_of(s, nf, ns) for s in sources}
    return fam


# --------------------------------------------------------------------------------------------------
# fixed_move_targets — count-conditioned signed return / σ_ev, against a dead-simple loop oracle
# --------------------------------------------------------------------------------------------------
def _fixed_move_oracle(rx, mid, move_rx, anchor_ts, sigma, counts):
    """Independent oracle: for each anchor and count n, walk forward to the n-th target MOVE strictly
    after the anchor and take log(mid_at_that_move) - log(mid_now), / sigma. Explicit, no shared code."""
    out = {}
    move_lm = np.array([np.log(mid[np.searchsorted(rx, mrx, "right") - 1]) for mrx in move_rx])
    for n in counts:
        col = np.full(len(anchor_ts), np.nan)
        for a in range(len(anchor_ts)):
            inow = np.searchsorted(rx, anchor_ts[a], "right") - 1
            if inow < 0:
                continue
            log_now = np.log(mid[inow])
            # index into move_rx: first move strictly after anchor is searchsorted(...,"right"); n-th is +n-1
            k = np.searchsorted(move_rx, anchor_ts[a], "right") + n - 1
            if k < len(move_lm):
                col[a] = (move_lm[k] - log_now) / sigma[a]
        out[n] = col
    return out


def test_fixed_move_targets_matches_oracle():
    rng = np.random.default_rng(0)
    n_ticks = 4000
    rx = (np.arange(1, n_ticks + 1) * 10).astype(np.int64)
    # a target mid that moves on ~60% of timestamps (the rest repeat the previous mid -> not a "move")
    moves = rng.random(n_ticks) < 0.6
    moves[0] = True
    steps = np.where(moves, rng.standard_normal(n_ticks) * 1e-4, 0.0)
    mid = 100.0 * np.exp(np.cumsum(steps))
    # the move stream the context exposes: dedup-collapsed timestamps where the mid changed
    move_rx = rx[moves]
    anchor_ts = np.arange(int(rx[50]), int(rx[-50]), 70, dtype=np.int64)
    sigma = np.full(len(anchor_ts), 2e-4)
    ctx = _empty_ctx(target="byb_x", anchor_ts=anchor_ts, sigma_at_anchor=sigma,
                     _mids={"byb": (rx, mid)}, _mv_rx=move_rx)
    counts = (1, 3, 6, 9)
    got = fixed_move_targets(ctx, counts=counts)
    ref = _fixed_move_oracle(rx, mid, move_rx, anchor_ts, sigma, counts)
    assert set(got) == set(counts)
    for n in counts:
        ok = np.isfinite(ref[n])
        assert ok.sum() > 100
        np.testing.assert_allclose(got[n][ok], ref[n][ok], rtol=1e-10, atol=1e-12)
        # n=1 must be the nearest move (smallest |return|, on average), n=9 the farthest
    # monotone: a farther move accumulates a larger typical |return|
    mags = [np.nanmean(np.abs(got[n])) for n in counts]
    assert mags[0] < mags[-1]


def test_fixed_move_targets_returns_supplied_counts():
    rng = np.random.default_rng(1)
    rx = (np.arange(1, 2001) * 10).astype(np.int64)
    moves = rng.random(2000) < 0.7
    moves[0] = True
    mid = 100.0 * np.exp(np.cumsum(np.where(moves, rng.standard_normal(2000) * 1e-4, 0.0)))
    anchor_ts = np.arange(int(rx[40]), int(rx[-40]), 50, dtype=np.int64)
    ctx = _empty_ctx(anchor_ts=anchor_ts, sigma_at_anchor=np.full(len(anchor_ts), 1e-4),
                     _mids={"byb": (rx, mid)}, _mv_rx=rx[moves])
    counts = (2, 4, 8)
    got = fixed_move_targets(ctx, counts)
    assert set(got) == set(counts)                       # caller-supplied counts (no default)


# --------------------------------------------------------------------------------------------------
# ic_grid — nf>=ns cells are nan; the planted (fast, slow) cell is the in-sample best
# --------------------------------------------------------------------------------------------------
def test_ic_grid_axes_and_nan_triangle():
    rng = np.random.default_rng(2)
    n = 6000
    target = rng.standard_normal(n)
    fasts, slows = (1, 10, 50), (10, 50, 100)
    fam = _grid_family(fasts, slows, lambda s, nf, ns: rng.standard_normal(n))
    grids = ic_grid(_empty_ctx(), fam, target)
    assert set(grids) == {"aaa", "bbb"}
    for g in grids.values():
        assert g.shape == (len(fasts), len(slows))
        # nf>=ns has no family member -> stays nan
        assert np.isnan(g[1, 0])                          # (10, 10): nf==ns
        assert np.isnan(g[2, 0]) and np.isnan(g[2, 1])    # (50,10),(50,50): nf>=ns
        assert np.isfinite(g[0, 0])                        # (1, 10): valid pair scored


def test_ic_grid_picks_planted_cell():
    rng = np.random.default_rng(3)
    n = 8000
    target = rng.standard_normal(n)
    fasts, slows = (1, 10, 50), (10, 50, 100)
    best = (10, 50)                                        # the cell we plant the signal into

    def leg_of(src, nf, ns):
        if (nf, ns) == best:
            return target * 0.6 + rng.standard_normal(n)   # strong in-sample IC
        return rng.standard_normal(n) + target * 0.05      # weak everywhere else

    fam = _grid_family(fasts, slows, leg_of)
    grids = ic_grid(_empty_ctx(), fam, target)
    fi, sj = fasts.index(best[0]), slows.index(best[1])
    for src, g in grids.items():
        i, j = np.unravel_index(np.nanargmax(g), g.shape)
        assert (i, j) == (fi, sj), f"{src}: in-sample best {(i, j)} != planted {(fi, sj)}"


def test_ic_grid_parallel_matches_sequential():
    # n_jobs only changes scheduling: the parallel grid must be BIT-IDENTICAL to the sequential one
    # (each cell writes its own [i, j]; gates.ic is deterministic), incl. the nan triangle.
    rng = np.random.default_rng(10)
    n = 8000
    target = rng.standard_normal(n)
    fasts, slows = (1, 10, 50, 200), (10, 50, 100, 500)   # overlapping -> a real nan triangle
    fam = _grid_family(fasts, slows, lambda s, nf, ns: rng.standard_normal(n) + 0.1 * target)
    for magnitude in (False, True):
        seq = ic_grid(_empty_ctx(), fam, target, magnitude=magnitude, n_jobs=1)
        par = ic_grid(_empty_ctx(), fam, target, magnitude=magnitude, n_jobs=8)
        assert set(seq) == set(par)
        for src in seq:
            np.testing.assert_array_equal(seq[src], par[src])   # NaNs compared equal by position


def _scan_family(spans, leg_of):
    """Build `{N -> {source -> vec}}` for a single-span (1-D param) feature via `leg_of(src, N)`."""
    return {N: {s: leg_of(s, N) for s in ("aaa", "bbb")} for N in spans}


def test_ic_scan_picks_planted_span():
    rng = np.random.default_rng(30)
    n = 8000
    target = rng.standard_normal(n)
    spans, best = (1, 10, 50, 200), 50

    def leg_of(src, N):
        return (target * 0.6 + rng.standard_normal(n)) if N == best else (rng.standard_normal(n) + target * 0.05)

    scan = ic_scan(_empty_ctx(), _scan_family(spans, leg_of), target)
    assert set(scan) == {"aaa", "bbb"}
    for src, arr in scan.items():
        assert arr.shape == (len(spans),)
        assert sorted(spans)[int(np.nanargmax(arr))] == best


def test_ic_scan_mirror_matches_manual_and_magnitude_noop():
    import boba.research.gates as _g
    rng = np.random.default_rng(31)
    n = 8000
    target = rng.standard_normal(n)
    spans = (1, 10, 100)
    fam = _scan_family(spans, lambda s, N: rng.standard_normal(n) + 0.1 * target)
    scan = ic_scan(_empty_ctx(), fam, target, mirror=np.negative)
    vec = fam[10]["aaa"]
    manual = _g.ic(np.concatenate([vec, -vec]), np.concatenate([target, -target]))
    assert scan["aaa"][sorted(spans).index(10)] == pytest.approx(manual, abs=1e-12)
    a = ic_scan(_empty_ctx(), fam, target, magnitude=True, mirror=np.negative)
    b = ic_scan(_empty_ctx(), fam, target, magnitude=True)
    for src in a:
        np.testing.assert_array_equal(a[src], b[src])   # |·| is sign-blind: mirror is a no-op


def test_ic_grid_magnitude_scores_abs():
    rng = np.random.default_rng(4)
    n = 8000
    base = rng.standard_normal(n)
    target = np.abs(base) + 0.3 * rng.standard_normal(n)   # depends on |signal|, symmetric in sign
    fasts, slows = (1, 10), (10, 100)
    best = (10, 100)

    def leg_of(src, nf, ns):
        # sign-randomised so the SIGNED IC ~ 0 but |feature| tracks the target's magnitude structure
        sgn = rng.choice([-1.0, 1.0], n)
        if (nf, ns) == best:
            return sgn * (np.abs(base) + 0.2 * rng.standard_normal(n))
        return sgn * rng.standard_normal(n)

    fam = _grid_family(fasts, slows, leg_of)
    signed = ic_grid(_empty_ctx(), fam, target, magnitude=False)
    mag = ic_grid(_empty_ctx(), fam, target, magnitude=True)
    fi, sj = fasts.index(best[0]), slows.index(best[1])
    for src in ("aaa", "bbb"):
        assert abs(signed[src][fi, sj]) < 0.1             # signed carries ~no signal
        assert mag[src][fi, sj] > 0.3                      # |feature| does
        i, j = np.unravel_index(np.nanargmax(mag[src]), mag[src].shape)
        assert (i, j) == (fi, sj)


# --------------------------------------------------------------------------------------------------
# second_span_adds — an orthogonal alt that adds OOS is KEPT; a redundant copy is not
# --------------------------------------------------------------------------------------------------
def test_second_span_adds_keeps_orthogonal():
    rng = np.random.default_rng(5)
    n = 24000
    # target = two independent components; the chosen span carries A, an orthogonal span carries B.
    comp_a = rng.standard_normal(n)
    comp_b = rng.standard_normal(n)
    target = comp_a + comp_b + 0.3 * rng.standard_normal(n)
    fasts, slows = (1, 10, 50), (10, 50, 100)
    chosen = (1, 10)
    orth = (50, 100)

    def leg_of(src, nf, ns):
        if (nf, ns) == chosen:
            return comp_a + 0.2 * rng.standard_normal(n)
        if (nf, ns) == orth:
            return comp_b + 0.2 * rng.standard_normal(n)   # the OTHER component -> adds OOS
        return comp_a + 0.5 * rng.standard_normal(n)        # diluted copies of the chosen span

    fam = _grid_family(fasts, slows, leg_of)
    res = second_span_adds(_empty_ctx(), fam, chosen, target)
    assert set(res) == {"aaa", "bbb"}
    for src, r in res.items():
        assert r["best_alt"] == orth                       # the orthogonal span is the most-orthogonal cell
        assert r["oos_joint"] - r["oos_solo"] >= 0.01
        assert r["keep"] is True


def test_second_span_adds_rejects_redundant():
    rng = np.random.default_rng(6)
    n = 24000
    comp_a = rng.standard_normal(n)
    target = comp_a + 0.05 * rng.standard_normal(n)       # target is comp_a almost exactly
    fasts, slows = (1, 10, 50), (10, 50, 100)
    chosen = (1, 10)

    def leg_of(src, nf, ns):
        # the chosen span is a near-perfect read of comp_a; every OTHER span is a much NOISIER copy of
        # the SAME single component -> no second span carries anything the (already near-ceiling) pick lacks
        noise = 0.05 if (nf, ns) == chosen else 1.5
        return comp_a + noise * rng.standard_normal(n)

    fam = _grid_family(fasts, slows, leg_of)
    res = second_span_adds(_empty_ctx(), fam, chosen, target)
    for src, r in res.items():
        assert r["oos_joint"] - r["oos_solo"] < 0.01
        assert r["keep"] is False


def test_second_span_adds_chosen_cell_is_zero_in_screen():
    # the chosen cell's conditional partial-IC is forced to 0 -> it can never be its own "best alt"
    rng = np.random.default_rng(7)
    n = 12000
    target = rng.standard_normal(n)
    fasts, slows = (1, 10), (10, 100)
    chosen = (1, 10)
    fam = _grid_family(fasts, slows, lambda s, nf, ns: target * 0.4 + rng.standard_normal(n))
    res = second_span_adds(_empty_ctx(), fam, chosen, target)
    for r in res.values():
        assert r["best_alt"] != chosen


# --------------------------------------------------------------------------------------------------
# per_exchange_vs_single — keep every source's leg vs the single best source (no pooling / merging)
# --------------------------------------------------------------------------------------------------
def test_per_exchange_vs_single_keeps_all():
    rng = np.random.default_rng(8)
    n = 24000
    # the two sources enter the target with OPPOSITE sign -> the legs carry genuinely distinct structure
    # a learned joint weighting (per-exchange) captures but neither single leg alone can recover.
    sa = rng.standard_normal(n)
    sb = rng.standard_normal(n)
    target = sa - sb + 0.3 * rng.standard_normal(n)
    params = (10, 100)
    fam = {params: {"aaa": sa + 0.2 * rng.standard_normal(n),
                    "bbb": sb + 0.2 * rng.standard_normal(n)}}
    res = per_exchange_vs_single(_empty_ctx(), fam, params, target)
    assert set(res) == {"per_exchange", "best_single", "adds_over_single"}
    assert set(res["best_single"]) == {"ic", "source"}
    assert res["per_exchange"] > res["best_single"]["ic"] + 0.05   # keeping both clearly beats one
    assert res["adds_over_single"] is True


def test_per_exchange_vs_single_one_suffices():
    rng = np.random.default_rng(9)
    n = 24000
    signal = rng.standard_normal(n)
    target = signal + 0.3 * rng.standard_normal(n)
    params = (10, 100)
    fam = {params: {"aaa": signal + 0.2 * rng.standard_normal(n),   # carries the signal
                    "bbb": rng.standard_normal(n)}}                  # pure noise — adds nothing
    res = per_exchange_vs_single(_empty_ctx(), fam, params, target)
    assert res["best_single"]["source"] == "aaa"
    assert res["adds_over_single"] is False                          # the noise leg adds nothing over one good leg


# --------------------------------------------------------------------------------------------------
# real-block integration (skipped without DATA_DIR) — the documented span picks reproduce
# --------------------------------------------------------------------------------------------------
GRID = [(nf, ns) for nf in (1, 10, 50, 200, 500, 1000)
        for ns in (100, 500, 1000, 2000, 5000, 10000) if nf < ns]


@pytest.mark.skipif(getattr(io, "DATA_DIR", None) is None, reason="no DATA_DIR configured")
def test_real_block_span_picks():
    from boba.features import price_dislocation as pd  # noqa: F401  (registers the feature)
    from boba.features import base
    from boba.research.screening import build_context, build_family

    ctx = build_context()
    spec = base.get("price_dislocation")
    family = build_family(ctx, spec.vectorized, GRID, n_jobs=8)

    # --- price head: signed feature -> 100 ms σ_ev return; in-sample best span is (10, 500) ---
    price_grids = ic_grid(ctx, family, ctx.price_target)
    mean_price = sum(price_grids.values()) / len(price_grids)   # mean over sources (nan-aligned: same valid cells)
    fasts = sorted({p[0] for p in family}); slows = sorted({p[1] for p in family})
    i, j = np.unravel_index(np.nanargmax(mean_price), mean_price.shape)
    assert (fasts[i], slows[j]) == (10, 500), f"price-head best span {(fasts[i], slows[j])} != (10, 500)"

    # --- rate head: |feature| -> count target; in-sample best abs-feature span is (1, 100) ---
    rate_grids = ic_grid(ctx, family, ctx.rate_target, magnitude=True)
    mean_rate = sum(rate_grids.values()) / len(rate_grids)
    ri, rj = np.unravel_index(np.nanargmax(mean_rate), mean_rate.shape)
    assert (fasts[ri], slows[rj]) == (1, 100), f"rate-head abs best span {(fasts[ri], slows[rj])} != (1, 100)"


@pytest.mark.skipif(getattr(io, "DATA_DIR", None) is None, reason="no DATA_DIR configured")
def test_real_block_fixed_move_and_second_span():
    from boba.features import price_dislocation as pd  # noqa: F401
    from boba.features import base
    from boba.research.screening import build_context, build_family

    ctx = build_context()
    spec = base.get("price_dislocation")

    # fixed-move targets are finite and ordered (a farther move -> a larger typical |return|)
    counts = (1, 5, 10, 15, 20, 25, 30)
    fmt = fixed_move_targets(ctx, counts)
    assert set(fmt) == set(counts)
    for n, col in fmt.items():
        assert np.isfinite(col).sum() > 1000
    assert np.nanmean(np.abs(fmt[1])) < np.nanmean(np.abs(fmt[30]))

    # second_span_adds runs end-to-end on the real family at the documented price pick
    family = build_family(ctx, spec.vectorized, GRID, n_jobs=8)
    res = second_span_adds(ctx, family, (10, 500), ctx.price_target)
    assert set(res) == set(ctx.sources)
    for r in res.values():
        assert r["best_alt"] in GRID
        assert isinstance(r["keep"], bool)
        assert np.isfinite(r["oos_solo"]) and np.isfinite(r["oos_joint"])


# --------------------------------------------------------------------------------------------------
# mirror augmentation — feature-driven reflection (AUTHORING.md → Mirror augmentation)
# --------------------------------------------------------------------------------------------------
import boba.research.gates as _g


def test_ic_grid_mirror_matches_manual_and_magnitude_noop():
    rng = np.random.default_rng(20)
    n = 8000
    target = rng.standard_normal(n)
    fasts, slows = (1, 10, 50), (10, 50, 100)
    fam = _grid_family(fasts, slows, lambda s, nf, ns: rng.standard_normal(n) + 0.1 * target)
    gm = ic_grid(_empty_ctx(), fam, target, mirror=np.negative)
    # a known valid cell equals the hand-built reflection: ic(concat[v,-v], concat[t,-t])
    vec = fam[(1, 10)]["aaa"]
    manual = _g.ic(np.concatenate([vec, -vec]), np.concatenate([target, -target]))
    assert gm["aaa"][fasts.index(1), slows.index(10)] == pytest.approx(manual, abs=1e-12)
    # magnitude is sign-blind -> mirror is a no-op (bit-identical, incl. the nan triangle)
    a = ic_grid(_empty_ctx(), fam, target, magnitude=True, mirror=np.negative)
    b = ic_grid(_empty_ctx(), fam, target, magnitude=True)
    for src in a:
        np.testing.assert_array_equal(a[src], b[src])


def test_feature_spec_mirror_declarations():
    from boba.features.base import FeatureSpec, get
    import boba.features.price_dislocation  # noqa: F401  (registers the feature)

    assert FeatureSpec.__dataclass_fields__["mirror"].default is None       # undeclared by default
    assert get("price_dislocation").mirror is np.negative                   # odd in the gap


def test_second_span_and_per_exchange_run_with_mirror():
    rng = np.random.default_rng(21)
    n = 24000
    sa, sb = rng.standard_normal(n), rng.standard_normal(n)
    target = sa - sb + 0.3 * rng.standard_normal(n)
    fasts, slows = (1, 10), (10, 100)
    fam = _grid_family(fasts, slows, lambda s, nf, ns: (sa if s == "aaa" else sb) + 0.2 * rng.standard_normal(n))
    res = second_span_adds(_empty_ctx(), fam, (1, 10), target, mirror=np.negative)
    for r in res.values():
        assert r["best_alt"] in [(1, 10), (1, 100), (10, 100)]
        assert np.isfinite(r["oos_solo"]) and np.isfinite(r["oos_joint"])
    pc = per_exchange_vs_single(_empty_ctx(), fam, (10, 100), target, mirror=np.negative)
    assert set(pc) == {"per_exchange", "best_single", "adds_over_single"}
    assert np.isfinite(pc["per_exchange"])


# --------------------------------------------------------------------------------------------------
# mirror augmentation — the COMMUTATION INVARIANT (AUTHORING.md → Mirror augmentation): applying a
# feature's declared mirror to the feature must equal recomputing the feature on the mirror-reflected
# books:   spec.mirror(feature(books)) == feature(mirror_books(books)).  Every feature must satisfy it.
# --------------------------------------------------------------------------------------------------
def _commutation_ctx(seed=0, n_ticks=2000):
    """A synthetic ScreeningContext rich enough to build mid-based features (byb + bin + okx mids on a
    trade clock, an anchor grid, σ_ev). Quotes start before merged_ts[0] so every mid is finite."""
    rng = np.random.default_rng(seed)
    merged_ts = (np.arange(1, n_ticks + 1) * 10).astype(np.int64)
    mids = {v: ((np.arange(0, n_ticks + 5) * 8).astype(np.int64),
                100.0 * np.exp(np.cumsum(rng.standard_normal(n_ticks + 5) * 1e-4)))
            for v in ("byb", "bin", "okx")}
    anchor_ts = np.arange(int(merged_ts[50]), int(merged_ts[-1]), 30, dtype=np.int64)
    ctx = _empty_ctx(target="byb_x", sources=("bin", "okx"), merged_ts=merged_ts, anchor_ts=anchor_ts,
                     tick_at_anchor=np.searchsorted(merged_ts, anchor_ts, "right") - 1,
                     sigma_at_anchor=np.abs(rng.standard_normal(len(anchor_ts))) * 1e-4 + 1e-5, _mids=mids)
    ctx.target_logmid_on_clock = np.log(ctx.mid_on_clock("byb"))
    return ctx


def _mirror_books(ctx, c=100.0):
    """Mirror_Books: reflect every venue's mid through the fixed price level `c` (log mid -> 2 ln c - log
    mid, i.e. mid -> c**2/mid) — the reflection of the whole book through byb's mid. σ_ev (even) and the
    clock/anchors are unchanged. The exact data-level operation a feature's `mirror` must commute with."""
    m = _empty_ctx(target=ctx.target, sources=ctx.sources, merged_ts=ctx.merged_ts, anchor_ts=ctx.anchor_ts,
                   tick_at_anchor=ctx.tick_at_anchor, sigma_at_anchor=ctx.sigma_at_anchor,
                   _mids={v: (rx, c * c / mid) for v, (rx, mid) in ctx._mids.items()})
    m.target_logmid_on_clock = 2.0 * np.log(c) - ctx.target_logmid_on_clock
    return m


def test_price_dislocation_mirror_commutes_with_book_reflection():
    from boba.features import base
    import boba.features.price_dislocation  # noqa: F401  (registers)

    ctx = _commutation_ctx()
    mctx = _mirror_books(ctx)
    spec = base.get("price_dislocation")
    for params in [(1, 200), (10, 100), (50, 500), (200, 2000)]:
        feat = spec.vectorized(ctx, params)            # feature(books)
        refl = spec.vectorized(mctx, params)           # feature(mirror_books(books))
        for ex in ctx.sources:
            lhs = spec.mirror(feat[ex])                # mirror(feature(books))
            ok = np.isfinite(lhs) & np.isfinite(refl[ex])
            assert ok.sum() > 100
            # to float round-off (lfilter EMA recursion; < the 1e-6 parity floor)
            np.testing.assert_allclose(lhs[ok], refl[ex][ok], rtol=1e-6, atol=1e-9)


def test_every_registered_feature_declares_mirror():
    # The mirror augmentation is a required invariant: every feature must DEFINE its reflection.
    from boba.features import base
    import boba.features.price_dislocation  # noqa: F401  (registers)
    import boba.features.ofi_fast_slow       # noqa: F401  (registers)
    import boba.features.ofi_ema             # noqa: F401  (registers)
    import boba.features.stoikov_premium_fast_slow  # noqa: F401  (registers)

    for spec in base.all_specs():
        assert spec.mirror is not None, f"feature {spec.name!r} has no mirror augmentation defined"


def test_features_declare_param_kind():
    # param_kind drives the (1-D vs 2-D) span sweep in the shared screening/finalize notebooks.
    from boba.features import base
    from boba.features.base import ParamKind
    import boba.features.price_dislocation  # noqa: F401
    import boba.features.ofi_fast_slow       # noqa: F401
    import boba.features.ofi_ema             # noqa: F401
    import boba.features.stoikov_premium_fast_slow  # noqa: F401

    assert base.get("price_dislocation").param_kind is ParamKind.FAST_SLOW
    assert base.get("ofi_fast_slow").param_kind is ParamKind.FAST_SLOW
    assert base.get("ofi_ema").param_kind is ParamKind.SINGLE
    assert base.get("stoikov_premium_fast_slow").param_kind is ParamKind.FAST_SLOW
