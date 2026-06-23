"""Regression tests that DEMONSTRATE each fixed gates.py bug with data, then pin the fix.

Each test follows demonstrate-then-fix: it reconstructs the OLD buggy logic inline and asserts it
gives the WRONG answer on crafted (market-like) data (the "red"), then asserts the shipped, fixed
function in boba.research.gates gives the right answer (the "green"). If a fix were reverted, the
green half would fail — these have teeth.

Bugs (found by the verification workflow on src/boba/research/gates.py):
  1. fold alignment   — base vs base+legs used separate valid masks -> skipped different folds -> the
                        per-fold marginal subtracted mismatched folds.
  2. block-length cap — capped on the FULL series, but blocks are resampled per fold -> only a few
                        blocks/fold -> degenerate, under-covering (too-narrow) bootstrap CI.
  3. gate_a scale     — max(band)/min(band) divided by zero on an empty/constant volatility decile.
  4. bootstrap nan    — corrcoef on an all-tied resampled block returned nan that poisoned the CI.
"""
import numpy as np
import pytest
from scipy.stats import rankdata

import boba.research.gates as g


def _market(rng, n, zero_frac=0.93, ac=40):
    """Market-like series: an autocorrelated latent process and a SPARSE (~zero_frac exact-zero),
    heavy-tailed, signed forward-return target — the structure that makes these bugs bite."""
    latent = np.convolve(rng.standard_normal(n), np.ones(ac) / ac, mode="same")
    moved = rng.random(n) > zero_frac
    target = np.where(moved, latent * rng.standard_normal(n) * 3.0, 0.0)
    return latent, target


def _center(mask):
    idx = np.where(mask)[0]
    return int(idx.mean()) if idx.size else -1


# ----------------------------------------------------------------------------------------- BUG 1
def test_bug1_separate_valid_masks_misalign_folds():
    rng = np.random.default_rng(3)
    n = 120_000
    _, target = _market(rng, n)
    target = np.convolve(target, np.ones(200) / 200, mode="same")
    control = np.convolve(rng.standard_normal(n), np.ones(50) / 50, mode="same")
    feature = 0.5 * target + rng.standard_normal(n)
    feature[20_000:40_000] = np.nan                      # feature missing in fold 1's region
    base, legs = [control], [feature]

    # RED: two independent wf_folds calls skip different folds, so zip() pairs mismatched time regions.
    old_full = list(g.wf_folds(base + legs, target))
    old_base = list(g.wf_folds(base, target))
    assert len(old_full) != len(old_base)                # base+legs drops a fold the base keeps
    old_pairs = [(_center(tf), _center(tb)) for (tf, _), (tb, _) in zip(old_full, old_base)]
    assert any(cf != cb for cf, cb in old_pairs)         # at least one pair compares different windows

    # GREEN: _aligned_folds drives both models from ONE shared valid mask + boundaries.
    aligned = g._aligned_folds(base, legs, target)
    assert len(aligned) == len(old_full)                 # the shared, consistent fold set
    per_fold, folds = g.fold_marginals(legs, base, target)
    assert len(per_fold) == len(folds) == len(aligned)   # every per-fold marginal is well-defined


# ----------------------------------------------------------------------------------------- BUG 2
def test_bug2_block_length_cap_is_per_fold_not_full_series():
    rng = np.random.default_rng(2)
    folds = []
    for _ in range(5):
        raw = np.where(rng.random(100_000) > 0.9, rng.standard_normal(100_000), 0.0)
        target = np.convolve(raw, np.ones(300) / 300, mode="same")   # long autocorrelation
        full_pred = 0.25 * target + rng.standard_normal(100_000)
        base_pred = rng.standard_normal(100_000)
        folds.append((full_pred, base_pred, target))
    pooled = np.concatenate([f[2] for f in folds])
    min_fold = min(len(f[2]) for f in folds)

    L_full_cap = g.estimate_block_len(pooled, 2000)                          # RED: cap on full pooled n
    L_per_fold = g.estimate_block_len(pooled, 2000, min_blocks_n=min_fold)   # GREEN: cap on per-fold n

    assert L_full_cap > L_per_fold
    assert min_fold // L_full_cap < 10                   # RED: only a handful of blocks per fold
    assert min_fold // L_per_fold >= 25                  # GREEN: enough blocks for a stable CI

    lo_old, hi_old, _ = g.marginal_block_bootstrap(folds, L_full_cap, B=200, seed=0)
    lo_new, hi_new, _ = g.marginal_block_bootstrap(folds, L_per_fold, B=200, seed=0)
    assert (hi_old - lo_old) <= (hi_new - lo_new)        # too-few-blocks CI is the narrower (under-covering) one


# ----------------------------------------------------------------------------------------- BUG 3
def test_bug3_gate_a_scale_survives_constant_decile():
    rng = np.random.default_rng(0)
    n = 60_000
    vol_level = np.cumsum(rng.standard_normal(n))
    rate_level = np.cumsum(rng.standard_normal(n))
    feature = rng.standard_normal(n)
    feature[g._deciles(vol_level) == 0] = 0.0            # feature constant within one vol decile

    # RED: unfiltered max(band)/min(band) divides by the zero-std decile -> inf.
    band = [np.nanstd(feature[g._deciles(vol_level) == d]) for d in range(10)]
    with np.errstate(divide="ignore", invalid="ignore"):
        old_scale = max(band) / min(band)
    assert not np.isfinite(old_scale)

    # GREEN: gate_a keeps only finite, positive band entries.
    assert np.isfinite(g.gate_a(feature, vol_level, rate_level)["scale"])


# ----------------------------------------------------------------------------------------- BUG 4
def test_bug4_bootstrap_drops_degenerate_resample():
    rng = np.random.default_rng(1)
    folds = []
    for i in range(5):
        _, target = _market(rng, 40_000)
        if i == 2:
            target = np.zeros_like(target)              # one fold all-zero -> any resample is all-tied
        full_pred = 0.3 * target + rng.standard_normal(len(target))
        base_pred = rng.standard_normal(len(target))
        folds.append((full_pred, base_pred, target))

    # RED: no guard -> corrcoef on the all-tied fold is nan -> the mean (and CI) is nan.
    def _old_bootstrap(folds, L=2000, B=150, seed=0):
        rng2 = np.random.default_rng(seed)
        ar = np.arange(L)
        ests = np.empty(B)
        for b in range(B):
            marg = []
            for full_pred, base_pred, y in folds:
                m = len(y)
                idx = (rng2.integers(0, m - L + 1, int(np.ceil(m / L)))[:, None] + ar).ravel()[:m]
                yr = rankdata(y[idx])
                marg.append(np.corrcoef(rankdata(full_pred[idx]), yr)[0, 1]
                            - np.corrcoef(rankdata(base_pred[idx]), yr)[0, 1])
            ests[b] = np.mean(marg)
        return np.percentile(ests, [5, 95])

    with np.errstate(invalid="ignore", divide="ignore"):
        old_ci = _old_bootstrap(folds)
    assert np.isnan(old_ci).all()

    # GREEN: marginal_block_bootstrap skips degenerate folds and returns a finite CI.
    lo, hi, mean = g.marginal_block_bootstrap(folds, 2000, B=150, seed=0)
    assert np.isfinite([lo, hi, mean]).all()


# ------------------------------------------------------ DOCUMENTED LIMITATION (not a bug): Gate B is linear
def test_gateb_linear_scorer_blind_to_nonlinear_signal_but_transform_recovers_it():
    """Gate B's OLS scorer is LINEAR, so a feature predictive only through a non-monotone/even transform
    (here its square) reads a marginal of ~0 — a documented scope limit, externally flagged. The mitigation
    is to feed the transform explicitly (the rate head does exactly this by feeding |feature|). DEMONSTRATE
    the blindness, then show the transform recovers the signal."""
    def _both(rng):
        n = 60_000
        feature = np.convolve(rng.standard_normal(n), np.ones(30) / 30, mode="same")
        feature = (feature - feature.mean()) / (feature.std() + 1e-12)
        moved = rng.random(n) > 0.90
        latent = feature ** 2 - 1.0                       # even in feature -> non-monotone in the signed feature
        target = np.where(moved, latent + 0.5 * rng.standard_normal(n), 0.0)
        control = [rng.standard_normal(n)]                # a do-nothing control
        blind = g.signal_ic([feature], control, target)       # signed feature: the linear scorer is blind
        recovered = g.signal_ic([feature ** 2], control, target)   # the SQUARE transform: signal recovered
        return blind, recovered

    blind = float(np.median([_both(np.random.default_rng(100 + s))[0] for s in range(5)]))
    recovered = float(np.median([_both(np.random.default_rng(100 + s))[1] for s in range(5)]))
    assert abs(blind) < 0.02, blind                       # RED: linear Gate B misses the purely-nonlinear signal
    assert recovered > 0.05, recovered                    # GREEN: feeding the transform recovers it


# ============================================ FINAL ADVERSARIAL-REVIEW findings (demonstrate-then-fix) ===
def test_review3_gate_a_track_finite_when_one_regime_coord_degenerate():
    """[review #3] gate_a track/mag used Python max(), which returns nan when the FIRST arg is nan — so a
    degenerate vol coordinate (too few finite rows -> ic=nan) poisoned track even with a fine rate IC."""
    rng = np.random.default_rng(7)
    n = 40_000
    rate_level = np.cumsum(rng.standard_normal(n))
    feature = 0.3 * rate_level / rate_level.std() + rng.standard_normal(n)   # tracks the rate coordinate
    vol_level = np.full(n, np.nan); vol_level[:50] = rng.standard_normal(50)  # <100 finite -> ic(feature,vol)=nan
    out = g.gate_a(feature, vol_level, rate_level)
    assert np.isfinite(out["track"]), out                 # GREEN: nanmax falls back to the finite coordinate
    assert np.isfinite(out["mag"]), out


def test_review6_deciles_excludes_nonfinite_rows():
    """[review #6] _deciles sent NaN regime-coordinate rows to the top decile (digitize(nan)=len(edges)),
    silently polluting decile 9. GREEN: non-finite rows get sentinel -1 and are excluded from gate_a's buckets."""
    rng = np.random.default_rng(8)
    level = rng.standard_normal(5000)
    level[::50] = np.nan
    buckets = g._deciles(level)
    assert np.all(buckets[~np.isfinite(level)] == -1)
    assert set(np.unique(buckets[np.isfinite(level)])) <= set(range(10))


def test_review4_signal_ic_by_regime_control_own_strat_skips_thin_buckets():
    """[review #4] the control+own+strat branch emitted EVERY regime bucket with no min-rows guard, leaking
    nan for a thin bucket (unlike the alpha branch / wf_ic_by_regime). GREEN: thin buckets are skipped."""
    rng = np.random.default_rng(9)
    n = 30_000
    yardstick = np.abs(np.cumsum(rng.standard_normal(n))) + 1.0
    leg = rng.standard_normal(n) / yardstick
    target = rng.standard_normal(n) / yardstick
    regime = np.zeros(n, int)
    regime[15_000:29_990] = 1
    regime[29_990:] = 2                                   # bucket 2 = 10 rows (thin)
    out = g.signal_ic_by_regime([leg], [rng.standard_normal(n)], target, regime,
                                feature_kind="control", own=True, strat_var=yardstick)
    assert all(np.isfinite(v) for v in out.values()), out
    assert 2 not in out                                   # the thin bucket is dropped, not emitted as nan


def test_review2_marginal_ci_per_fold_cap_binds_end_to_end():
    """[review #2] make the per-fold block cap actually BIND through marginal_ci (the prior test didn't —
    the embargo floor dominated). A long series + low embargo where the per-fold cap (smallest_fold//30) is
    well below both the full-series cap and the ACF*safety raw, so a regression to the full-series cap shows."""
    rng = np.random.default_rng(10)
    n = 300_000
    raw = np.where(rng.random(n) > 0.9, rng.standard_normal(n), 0.0)
    target = np.convolve(raw, np.ones(400) / 400, mode="same")   # long ACF -> safety*acf is large
    legs = [0.2 * target + rng.standard_normal(n)]
    controls = [rng.standard_normal(n)]
    out = g.marginal_ci(legs, controls, target, embargo=200, B=120, seed=0)
    # per-fold OOS ~ n/6 ~ 50k -> per-fold cap ~1666; full-series pooled ~250k -> ~8333. Correct path binds at ~1666.
    assert out["block_len"] <= 50_000 // 30 + 200, out["block_len"]   # bound by per-fold cap, NOT full-series
    assert out["nf"] >= 4
