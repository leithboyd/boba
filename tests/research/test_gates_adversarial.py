"""Adversarial / mutation-style suite for boba.research.gates.

This complements tests/research/test_gates.py. Its job is to find FALSE CONFIDENCE: assertions that would pass
even if the gate machinery were wrong. For every key safeguard we keep a *local broken copy* of the
relevant computation (we never touch gates.py) and assert the BROKEN variant gives the wrong answer
while the SHIPPED gate gives the right one — i.e. the safeguard has teeth.

Everything is synthetic, stochastic and market-like (sparse ~95%-zero target, autocorrelated,
heavy-tailed), seeded, with should-PASS and should-FAIL examples.

It also adds a regression test for each of the four bugs fixed just before this review:
  (1) fold ALIGNMENT          — base vs base+legs share ONE valid mask & fold boundaries (_aligned_folds)
  (2) estimate_block_len cap  — the >=30-blocks cap is PER-FOLD (min_blocks_n), not full-series
  (3) gate_a scale guard      — empty/constant deciles can't make max/min divide by zero / return nan
  (4) bootstrap degenerate    — all-tied/constant resampled blocks are dropped; output stays nan-safe
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest
from scipy.stats import ConstantInputWarning, rankdata, spearmanr

from boba.research import gates as g

SEEDS = (0, 1, 2, 3, 4)


# ==================================================================================================
# Stochastic market-like generators (independent re-implementation; no shared code with gates.py)
# ==================================================================================================

def _ar1(rng, n, rho):
    """Unit-variance AR(1) path — the persistence/memory in a stream."""
    z = np.zeros(n)
    eps = rng.standard_normal(n)
    for i in range(1, n):
        z[i] = rho * z[i - 1] + eps[i]
    return z / (z.std() + 1e-12)


def synth(rng, n, *, signal=0.0, regime_leak=0.0, zero_frac=0.93, autocorr=0.97,
          move_clustering=0.95, regime_rho=0.999):
    """A sparse, heavy-tailed, autocorrelated crypto-microstructure screening problem."""
    vol_level = np.exp(_ar1(rng, n, regime_rho) * 0.8)            # positive, log-normal-ish vol
    rate_level = _ar1(rng, n, regime_rho)                        # signed mean-reverting rate
    z = _ar1(rng, n, autocorr)                                   # latent driver
    if move_clustering > 0:
        occ = _ar1(rng, n, move_clustering)
        moves = occ > np.quantile(occ, zero_frac)
    else:
        moves = rng.random(n) > zero_frac
    tgt = np.where(moves, rng.standard_t(3, n), 0.0) + signal * z * moves
    vc = (vol_level - vol_level.mean()) / (vol_level.std() + 1e-12)
    feat = z.copy() + regime_leak * vc
    base = [np.diff(vol_level, prepend=vol_level[0]), np.diff(rate_level, prepend=rate_level[0])]
    return dict(feat=feat, tgt=tgt, z=z, moves=moves, vol_level=vol_level,
                rate_level=rate_level, base=base)


def _ordinal(v):
    """Ordinal (argsort) ranking — the WRONG ranking for a tie-heavy target (spreads ties by position)."""
    o = np.empty(len(v))
    o[np.argsort(v, kind="stable")] = np.arange(len(v))
    return o


N = 60_000


@pytest.fixture
def null_d():
    return synth(np.random.default_rng(2024), N, signal=0.0)


@pytest.fixture
def signal_d():
    return synth(np.random.default_rng(4048), N, signal=2.0)


# ==================================================================================================
# TEETH (a) — average-rank vs ordinal tie handling on the zero-heavy target.
#
# The shipped test only greps the source for "rankdata"/"argsort"; it never runs the gate with the
# wrong ranking, so it could pass even if rankdata were used wrongly. Here we run a BROKEN ordinal
# bootstrap end-to-end and prove its CI is wildly inflated by the spurious rank<->time signal that
# ordinal manufactures across the tied zeros, while the shipped (average-rank) CI stays tight.
# ==================================================================================================

def _boot_ordinal(folds, block_len, B=250, seed=0, q=(5, 95)):
    """marginal_block_bootstrap with the ONE mutation: ordinal ranking instead of average-rank."""
    rng = np.random.default_rng(seed)
    ar = np.arange(block_len)
    ests = []
    for _ in range(B):
        fold_margs = []
        for full_pred, base_pred, y in folds:
            n = len(y)
            if n <= block_len:
                idx = np.arange(n)
            else:
                starts = rng.integers(0, n - block_len + 1, int(np.ceil(n / block_len)))
                idx = (starts[:, None] + ar).ravel()[:n]
            y_ranks = _ordinal(y[idx])
            if np.std(y_ranks) == 0:
                continue
            with np.errstate(invalid="ignore"):
                f_ic = np.corrcoef(_ordinal(full_pred[idx]), y_ranks)[0, 1]
                b_ic = np.corrcoef(_ordinal(base_pred[idx]), y_ranks)[0, 1]
            if np.isfinite(f_ic) and np.isfinite(b_ic):
                fold_margs.append(f_ic - b_ic)
        if fold_margs:
            ests.append(np.mean(fold_margs))
    ests = np.asarray(ests)
    lo, hi = np.percentile(ests, list(q))
    return float(lo), float(hi), float(ests.mean())


def _persistent_null_folds(seed):
    """Folds where full_pred is highly PERSISTENT (slow time trend) but the target is a NULL zero-heavy
    series independent of it. Average-rank sees ~0; ordinal turns full_pred's time structure into a
    huge spurious rank<->time IC that does NOT cancel in the marginal (base is white noise)."""
    rng = np.random.default_rng(seed)
    folds = []
    for _ in range(5):
        n = 8000
        full = _ar1(rng, n, 0.9999)                                # persistent -> time-trend-like
        base = rng.standard_normal(n)                              # white noise
        moves = rng.random(n) > 0.95
        y = np.where(moves, rng.standard_t(3, n), 0.0)            # NULL, independent of full
        folds.append((full, base, y))
    return folds


def test_teeth_tie_handling_ordinal_inflates_ci():
    real_w, ord_w = [], []
    for s in SEEDS:
        folds = _persistent_null_folds(s)
        rl = g.marginal_block_bootstrap(folds, 500, B=250, seed=0)
        ol = _boot_ordinal(folds, 500, B=250, seed=0)
        real_w.append(rl[1] - rl[0])
        ord_w.append(ol[1] - ol[0])
    # shipped (average-rank) CI is tight; ordinal manufactures an order-of-magnitude wider CI.
    assert all(rw < 0.05 for rw in real_w), real_w
    assert all(ow > rw * 3 for ow, rw in zip(ord_w, real_w)), (ord_w, real_w)


def test_teeth_tie_handling_spearman_is_average_rank():
    # Direct, value-level proof the gate uses average-rank: on a persistent prediction vs a zero-heavy
    # target, average-rank Spearman is ~0 but ordinal manufactures a large (here negative) correlation.
    rng = np.random.default_rng(0)
    n = 20000
    y = np.where(rng.random(n) > 0.95, rng.standard_t(3, n), 0.0)
    pred = _ar1(rng, n, 0.9999)
    avg_rank = abs(spearmanr(pred, y).statistic)              # what gates.ic / wf_ic use
    ordinal = abs(np.corrcoef(_ordinal(pred), _ordinal(y))[0, 1])
    assert avg_rank < 0.05, avg_rank
    assert ordinal > 0.3, ordinal
    # and confirm the shipped wf_ic (a persistent-but-null feature) is ~0, not inflated
    assert abs(g.wf_ic([_ar1(rng, n, 0.9999)], y, embargo=200)) < 0.05


# ==================================================================================================
# TEETH (b) — the embargo actually purges the trailing train rows (leakage).
#
# Mirrors the shipped sign-flip probe but states it as an explicit mutation: embargo=0 (the mutation)
# leaks the band -> WRONG (positive) sign; embargo>band (correct) -> right (negative) sign.
# ==================================================================================================

def test_teeth_embargo_zero_leaks_band_sign():
    rng = np.random.default_rng(7)
    n = 12000
    edges = np.linspace(0, n, 7).astype(int)
    e1, e2 = edges[1], edges[2]
    feat = rng.standard_normal(n)
    y = -3.0 * feat + 0.1 * rng.standard_normal(n)               # bulk: NEGATIVE relation
    band = slice(e1 - 1500, e1)
    y[band] = 3.0 * feat[band] + 0.1 * rng.standard_normal(1500)  # last train rows: POSITIVE
    y[e1:e2] = 3.0 * feat[e1:e2] + 0.1 * rng.standard_normal(e2 - e1)  # test fold: POSITIVE

    def fold1_ic(embargo):
        for t, p in g.wf_folds([feat], y, k=6, embargo=embargo, min_rows=100):
            return spearmanr(p[t], y[t]).statistic
        return np.nan

    leaked = fold1_ic(0)        # MUTATION: no embargo -> band leaks in -> learns positive
    purged = fold1_ic(1700)     # CORRECT: band purged -> learns the bulk's negative relation
    assert leaked > 0.5, leaked
    assert purged < -0.5, purged
    assert np.sign(leaked) != np.sign(purged)  # the sign FLIP is the leakage signature


# ==================================================================================================
# TEETH (c) — per-fold-average (not pooled) AND aligned base vs base+legs.
# ==================================================================================================

def test_teeth_pooled_differs_from_per_fold_average(signal_d):
    # The shipped headline averages PER-FOLD marginals; pooling all OOS rows and taking one Spearman
    # mixes the per-fold-standardised prediction scales. They must NOT coincide on real signal.
    d = signal_d
    per_fold, folds = g.fold_marginals([d["feat"]], d["base"], d["tgt"])
    shipped = float(np.mean(per_fold))
    full_all = np.concatenate([f for f, _, _ in folds])
    base_all = np.concatenate([b for _, b, _ in folds])
    y_all = np.concatenate([y for _, _, y in folds])
    pooled = float(spearmanr(full_all, y_all).statistic - spearmanr(base_all, y_all).statistic)
    assert abs(shipped - pooled) > 1e-3, (shipped, pooled)
    # the shipped signal_ic returns the per-fold-average, NOT the pooled value
    assert g.signal_ic([d["feat"]], d["base"], d["tgt"]) == round(shipped, 3)


def test_teeth_misaligned_folds_subtract_mismatched_sets():
    # legs carry NaN across (almost) one whole test fold's window. The ALIGNED gate skips that fold for
    # BOTH models (shared valid mask) -> 4 folds each. The naive per-model approach (separate wf_ic)
    # would average base over 5 folds and full over 4 -> subtracting different fold sets -> wrong value.
    rng = np.random.default_rng(3)
    n = 30000
    edges = np.linspace(0, n, 7).astype(int)
    base = [rng.standard_normal(n), rng.standard_normal(n)]
    y = 0.3 * base[0] + rng.standard_normal(n)
    legs = [0.5 * base[0] + rng.standard_normal(n)]
    i = 3
    legs[0][edges[i]:edges[i + 1] - 50] = np.nan   # kill nearly all of fold i's test rows in the legs

    per_fold, _ = g.fold_marginals(legs, base, y)
    assert len(per_fold) == 4   # the aligned gate drops the wiped fold for BOTH models

    aligned = g.signal_ic(legs, base, y)

    # MUTATION: compute the two means over each model's OWN surviving folds (mismatched sets)
    full_ics = [spearmanr(p[t], y[t]).statistic for t, p in g.wf_folds(base + legs, y)]
    base_ics = [spearmanr(p[t], y[t]).statistic for t, p in g.wf_folds(base, y)]
    assert len(full_ics) == 4 and len(base_ics) == 5   # the fold-count MISMATCH the bug-fix prevents
    misaligned = round(float(np.mean(full_ics)) - float(np.mean(base_ics)), 3)
    assert aligned != misaligned, (aligned, misaligned)


def test_aligned_folds_share_one_mask_and_boundaries():
    # _aligned_folds must yield identical test masks for full and base, on ONE shared valid mask.
    rng = np.random.default_rng(5)
    n = 30000
    base = [rng.standard_normal(n)]
    legs = [rng.standard_normal(n)]
    y = rng.standard_normal(n)
    legs[0][10000:10200] = np.nan   # rows usable by base alone but not by full -> must be shared-out
    folds = g._aligned_folds(base, legs, y, k=6, embargo=2000)
    assert len(folds) == 5
    for test, full_pred, base_pred in folds:
        assert full_pred.shape == base_pred.shape == (n,)
        # the legs-NaN rows must never appear in any test fold (shared mask excludes them for BOTH)
        assert not test[10000:10200].any()


# ==================================================================================================
# TEETH (d) — gate_a scale guard against empty/constant deciles.
# ==================================================================================================

def _deciles_indep(level):  # independent decile assignment (no gates.py helper)
    fin = np.isfinite(level)
    cuts = np.nanpercentile(level[fin], np.arange(10, 100, 10))
    return np.digitize(level, cuts)


def test_teeth_gate_a_scale_guard_blocks_divide_by_zero():
    rng = np.random.default_rng(0)
    n = 20000
    vol = np.exp(np.cumsum(rng.standard_normal(n)) * 0.01)
    rate = rng.standard_normal(n)
    feat = rng.standard_normal(n)
    decs = _deciles_indep(vol)
    feat[decs < 3] = 0.0   # CONSTANT (std==0) inside the lowest vol deciles

    shipped = g.gate_a(feat, vol, rate)["scale"]
    assert np.isfinite(shipped), shipped     # the guard keeps it finite (drops the zero-std deciles)

    # MUTATION: unguarded max/min over ALL deciles (min picks up a zero-std decile) -> inf / nan
    band_all = [np.nanstd(feat[decs == d]) for d in range(10)]
    with np.errstate(divide="ignore", invalid="ignore"):
        unguarded = max(band_all) / min(band_all)
    assert not np.isfinite(unguarded)        # exactly the divide-by-zero the guard prevents


def test_teeth_gate_a_constant_feature_is_nan_not_crash():
    # A fully-constant feature -> EVERY decile std is 0 -> band empty -> guard returns nan (not a crash).
    rng = np.random.default_rng(1)
    n = 15000
    vol = np.exp(np.cumsum(rng.standard_normal(n)) * 0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)  # ic() on the constant feat -> nan (expected)
        ga = g.gate_a(np.zeros(n), vol, rng.standard_normal(n))
    assert np.isnan(ga["scale"])
    # the other three numbers must still be finite (or nan via ic), never raise
    assert set(ga) == {"scale", "track", "mag", "disp"}


# ==================================================================================================
# TEETH (e) — wrong (tiny) block length under-covers (CI too narrow).
# ==================================================================================================

def _persistent_signal_folds(seed):
    """Folds whose per-fold marginal is itself slow-varying (very persistent agreement). A tiny block
    destroys that dependence and shrinks the CI; the data-derived block keeps it honest."""
    rng = np.random.default_rng(seed)
    folds = []
    for _ in range(5):
        n = 6000
        common = _ar1(rng, n, 0.9995)
        full = common + 0.5 * rng.standard_normal(n)
        base = 0.5 * rng.standard_normal(n)
        moves = rng.random(n) > 0.9
        y = np.where(moves, 1.0 * common + 0.5 * rng.standard_t(3, n), 0.0)
        folds.append((full, base, y))
    return folds


def test_teeth_tiny_block_len_undercovers():
    widths_tiny, widths_proper = [], []
    for s in SEEDS:
        folds = _persistent_signal_folds(s)
        pooled = np.concatenate([y for _, _, y in folds])
        smallest = min(len(y) for _, _, y in folds)
        L = g.estimate_block_len(pooled, 50, min_blocks_n=smallest)  # data-derived, per-fold-capped
        assert L > 1
        lo_t, hi_t, _ = g.marginal_block_bootstrap(folds, 1, B=300, seed=0)   # MUTATION: block=1
        lo_p, hi_p, _ = g.marginal_block_bootstrap(folds, L, B=300, seed=0)
        widths_tiny.append(hi_t - lo_t)
        widths_proper.append(hi_p - lo_p)
    # block=1 ignores autocorrelation -> systematically NARROWER CI (under-coverage), every seed.
    assert all(p > t for p, t in zip(widths_proper, widths_tiny)), (widths_proper, widths_tiny)
    assert float(np.median(widths_proper)) > 1.2 * float(np.median(widths_tiny))


# ==================================================================================================
# REGRESSION (1) — fold alignment (also covered by the teeth test above; this pins the public API).
# ==================================================================================================

def test_regression_alignment_signal_ic_robust_to_legs_nans():
    # With legs NaN over a whole fold, signal_ic must still return a clean rounded float over the
    # surviving shared folds (no crash, no fold-count mismatch leaking into the marginal).
    rng = np.random.default_rng(9)
    n = 30000
    edges = np.linspace(0, n, 7).astype(int)
    base = [rng.standard_normal(n)]
    legs = [0.4 * base[0] + rng.standard_normal(n)]
    y = 0.4 * base[0] + 0.4 * legs[0] + rng.standard_normal(n)
    legs[0][edges[2]:edges[3] - 30] = np.nan
    val = g.signal_ic(legs, base, y)
    assert np.isfinite(val) and val == round(val, 3)


# ==================================================================================================
# REGRESSION (2) — estimate_block_len cap is PER-FOLD (min_blocks_n), not full-series.
# ==================================================================================================

def test_regression_block_len_cap_is_per_fold():
    rng = np.random.default_rng(0)
    y = _ar1(rng, 90000, 0.9995)        # enormous ACF -> wants a huge block, must be capped
    embargo = 100
    full_cap = g.estimate_block_len(y, embargo)                      # cap = len(y)//30 = 3000
    per_fold_cap = g.estimate_block_len(y, embargo, min_blocks_n=6000)  # cap = 6000//30 = 200
    assert full_cap == max(embargo, 90000 // 30)
    assert per_fold_cap == max(embargo, 6000 // 30)
    assert per_fold_cap < full_cap          # the per-fold cap is the tighter (correct) one
    # the embargo FLOOR still wins when the per-fold cap would go below it
    assert g.estimate_block_len(y, 2000, min_blocks_n=6000) == 2000


def test_regression_marginal_ci_uses_per_fold_cap():
    # marginal_ci must cap on the SMALLEST fold, not the pooled length. Verify block_len <= smallest//30
    # (when that bites) rather than the looser pooled//30 the shipped shape-test asserts.
    d = synth(np.random.default_rng(123), N, signal=0.6, autocorr=0.999)
    _, folds = g.fold_marginals([d["feat"]], d["base"], d["tgt"])
    smallest = min(len(y) for _, _, y in folds)
    pooled = np.concatenate([y for _, _, y in folds])
    expected = g.estimate_block_len(pooled, 2000, min_blocks_n=smallest)
    out = g.marginal_ci([d["feat"]], d["base"], d["tgt"], B=80, seed=0)
    assert out["block_len"] == expected
    assert out["block_len"] <= max(2000, smallest // 30)


# ==================================================================================================
# REGRESSION (3) — gate_a scale guard (the value-level regression; teeth above prove it has teeth).
# ==================================================================================================

def test_regression_gate_a_finite_when_some_deciles_constant():
    rng = np.random.default_rng(2)
    n = 20000
    vol = np.exp(np.cumsum(rng.standard_normal(n)) * 0.01)
    feat = rng.standard_normal(n)
    feat[_deciles_indep(vol) < 2] = 0.0   # a couple of constant deciles
    ga = g.gate_a(feat, vol, rng.standard_normal(n))
    assert np.isfinite(ga["scale"]) and ga["scale"] >= 1.0


# ==================================================================================================
# REGRESSION (4) — bootstrap drops degenerate (all-tied / constant) resampled blocks; nan-safe.
# ==================================================================================================

def test_regression_bootstrap_all_tied_fold_returns_nan_tuple():
    rng = np.random.default_rng(0)
    n = 4000
    folds = [(rng.standard_normal(n), rng.standard_normal(n), np.zeros(n))]  # all-tied target
    out = g.marginal_block_bootstrap(folds, block_len=500, B=50, seed=0)
    assert all(np.isnan(x) for x in out)


def test_regression_bootstrap_constant_pred_fold_returns_nan_tuple():
    rng = np.random.default_rng(1)
    n = 4000
    y = np.where(rng.random(n) > 0.9, rng.standard_t(3, n), 0.0)
    folds = [(np.ones(n), rng.standard_normal(n), y)]  # constant full_pred -> corrcoef nan -> dropped
    with np.errstate(invalid="ignore"):  # corrcoef divides by a zero stddev (then the finite guard drops it)
        out = g.marginal_block_bootstrap(folds, block_len=500, B=50, seed=0)
    assert all(np.isnan(x) for x in out)


def test_regression_bootstrap_mixed_degenerate_keeps_good_fold():
    rng = np.random.default_rng(0)
    n = 4000
    good = (rng.standard_normal(n), rng.standard_normal(n),
            np.where(rng.random(n) > 0.9, rng.standard_t(3, n), 0.0))
    degenerate = (rng.standard_normal(n), rng.standard_normal(n), np.zeros(n))
    out = g.marginal_block_bootstrap([degenerate, good], block_len=500, B=80, seed=0)
    assert all(np.isfinite(x) for x in out)   # the good fold survives; degenerate one is skipped


def test_teeth_bootstrap_guard_prevents_nan_pollution():
    # Without the std/finite guards, a mixed (degenerate + good) fold list yields a nan CI.
    def boot_unguarded(folds, block_len, B=80, seed=0, q=(5, 95)):
        rng = np.random.default_rng(seed)
        ar = np.arange(block_len)
        ests = []
        for _ in range(B):
            fm = []
            for full, base, y in folds:
                n = len(y)
                if n <= block_len:
                    idx = np.arange(n)
                else:
                    starts = rng.integers(0, n - block_len + 1, int(np.ceil(n / block_len)))
                    idx = (starts[:, None] + ar).ravel()[:n]
                yr = rankdata(y[idx], method="average")
                with np.errstate(invalid="ignore"):
                    fi = np.corrcoef(rankdata(full[idx], method="average"), yr)[0, 1]
                    bi = np.corrcoef(rankdata(base[idx], method="average"), yr)[0, 1]
                fm.append(fi - bi)   # NO guard
            ests.append(np.mean(fm))
        ests = np.asarray(ests)
        lo, hi = np.percentile(ests, list(q))
        return float(lo), float(hi), float(ests.mean())

    rng = np.random.default_rng(0)
    n = 4000
    good = (rng.standard_normal(n), rng.standard_normal(n),
            np.where(rng.random(n) > 0.9, rng.standard_t(3, n), 0.0))
    mixed = [(rng.standard_normal(n), rng.standard_normal(n), np.zeros(n)), good]
    shipped = g.marginal_block_bootstrap(mixed, 500, B=80, seed=0)
    broken = boot_unguarded(mixed, 500, B=80, seed=0)
    assert all(np.isfinite(x) for x in shipped)
    assert all(np.isnan(x) for x in broken)


# ==================================================================================================
# COVERAGE — branches the shipped suite leaves untested.
# ==================================================================================================

def test_signal_ic_by_regime_excludes_regime_missing_from_base():
    # The alpha branch reports only regimes present in BOTH the full and base accumulators
    # ("if r in base_acc"): a rare regime with <100 rows must not be emitted as a 0.0 baseline.
    d = synth(np.random.default_rng(11), 40000, signal=2.0)
    reg = (d["vol_level"] > np.median(d["vol_level"])).astype(int)
    reg[-50:] = 2   # a rare third regime with too few rows to qualify in any fold
    out = g.signal_ic_by_regime([d["feat"]], d["base"], d["tgt"], reg.astype(float))
    assert 2 not in out                     # excluded (never reached the >=100 threshold)
    assert set(out.keys()) <= {0, 1}
    assert all(v == round(v, 3) for v in out.values())


def test_signal_ic_by_regime_control_own_strat_branch():
    # The control+own+strat_var regime branch (separate code path) must return one rounded value per
    # regime, keyed by int.
    rng = np.random.default_rng(11)
    n = 40000
    sv = np.exp(np.cumsum(rng.standard_normal(n)) * 0.02) * 3.0
    f = rng.standard_normal(n) / sv
    tgt = rng.standard_normal(n) / sv
    reg = np.digitize(np.abs(np.cumsum(rng.standard_normal(n))) + 1,
                      [np.quantile(np.abs(np.cumsum(rng.standard_normal(n))) + 1, q) for q in (1 / 3, 2 / 3)])
    out = g.signal_ic_by_regime([f], [], tgt.astype(float), reg.astype(float),
                                feature_kind="control", own=True, strat_var=sv)
    assert isinstance(out, dict) and len(out) >= 2
    for k, v in out.items():
        assert isinstance(k, int) and v == round(v, 3)


def test_signal_ic_control_own_no_strat_is_plain_mean_ic():
    # control + own + strat_var=None -> mean of plain g.ic over the legs (NOT marginal, NOT stratified).
    rng = np.random.default_rng(9)
    n = 20000
    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n)
    tgt = 0.4 * f1 + 0.2 * f2 + rng.standard_normal(n)
    got = g.signal_ic([f1, f2], [], tgt, feature_kind="control", own=True)
    expected = round(float(np.mean([g.ic(f1, tgt), g.ic(f2, tgt)])), 3)
    assert got == expected
    # and it must NOT coincide with the marginal (alpha) branch over a real base (different estimator)
    base = [rng.standard_normal(n)]
    assert got != g.signal_ic([f1, f2], base, tgt, feature_kind="alpha")


def test_wf_ic_returns_nan_when_all_folds_skipped():
    rng = np.random.default_rng(0)
    feats = [rng.standard_normal(300)]
    y = rng.standard_normal(300)
    assert np.isnan(g.wf_ic(feats, y, k=6, embargo=2000))   # embargo eats the train history


def test_signal_ic_alpha_nan_when_all_folds_skipped():
    rng = np.random.default_rng(1)
    assert np.isnan(g.signal_ic([rng.standard_normal(300)], [rng.standard_normal(300)],
                                rng.standard_normal(300)))


def test_marginal_ci_empty_documented_dict_on_tiny_input():
    rng = np.random.default_rng(2)
    out = g.marginal_ci([rng.standard_normal(300)], [rng.standard_normal(300)], rng.standard_normal(300))
    assert out["nf"] == 0 and out["per_fold"] == [] and out["block_len"] == 0
    assert np.isnan(out["ci"][0]) and np.isnan(out["ci"][1]) and np.isnan(out["boot_mean"])


def test_stratified_ic_nan_when_yardstick_not_distinct():
    rng = np.random.default_rng(0)
    n = 5000
    # constant yardstick -> <2 distinct values -> cannot bin -> nan
    assert np.isnan(g.stratified_ic(rng.standard_normal(n), rng.standard_normal(n), np.ones(n)))
    # too few finite jointly -> nan
    assert np.isnan(g.stratified_ic(np.arange(40.0), np.arange(40.0), np.arange(40.0)))


def test_wf_ic_by_regime_skips_thin_regime_buckets():
    # regime buckets with < min_n rows must not appear in the output dict.
    d = synth(np.random.default_rng(3), 40000, signal=2.0)
    reg = (d["vol_level"] > np.median(d["vol_level"])).astype(int)
    reg[-50:] = 9   # rare bucket, too thin
    out = g.wf_ic_by_regime([d["feat"]], d["tgt"], reg)
    assert 9 not in out
    assert all(isinstance(k, int) for k in out)


def test_disp_gate_a_catches_nonmonotone_dispersion():
    # disp must flag a feature whose per-decile dispersion leaks the regime even when track passes.
    def metrics(rng, _s):
        d = synth(rng, N, signal=0.0)
        vol = d["vol_level"]
        sign = rng.choice([-1, 1], N)
        feat = sign * (vol - vol.min() + 0.1) * (0.5 + 0.5 * np.abs(rng.standard_normal(N)))
        ga = g.gate_a(feat, vol, d["rate_level"])
        return ga["track"], ga["disp"]
    tr = float(np.median([metrics(np.random.default_rng(s), s)[0] for s in SEEDS]))
    dp = float(np.median([metrics(np.random.default_rng(s), s)[1] for s in SEEDS]))
    assert tr < 0.05      # signed mean flat -> track passes ...
    assert dp > 0.1       # ... but disp catches the magnitude-only leak


def test_shipped_bootstrap_source_uses_rankdata_not_argsort():
    # Keep the documentary check too (cheap) — but it is NOT the only tie-handling guard now.
    src = inspect.getsource(g.marginal_block_bootstrap)
    assert "rankdata" in src and "argsort" not in src
