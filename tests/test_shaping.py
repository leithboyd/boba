"""Tests for the input-shaping engine (boba.research.shaping) — step 3 of feature analysis.

The recommendation logic and the candidate transforms are exercised on SYNTHETIC arrays (no DATA_DIR
needed): a symmetric fat-tailed feature with a big spike must fall through to robust+clip, while a clean
gaussian must earn the light z-score. A real-block test ties the port to the monolith's conclusion when
DATA_DIR is set (skipped otherwise): the price_dislocation feature at the price-head span (10, 500) is
near-symmetric with a large raw spike, so the recommendation is robust+clip — exactly the §8 verdict.
"""
import numpy as np
import pytest
from scipy.stats import kurtosis, skew

import boba.io as io
from boba.research.shaping import LIGHTNESS_ORDER, ShapingReport, shaping_report


# --------------------------------------------------------------------------------------------------
# the candidate transforms, against a dead-simple independent reference (no production helpers)
# --------------------------------------------------------------------------------------------------
def _candidates_ref(f, clip):
    """The four shaping candidates by direct numpy, from the written definitions alone (no shared code)."""
    from scipy.stats import norm, rankdata
    n = len(f)
    z = (f - f.mean()) / f.std()
    med = np.median(f)
    mad = 1.4826 * np.median(np.abs(f - med))
    rz = (f - med) / mad
    return {
        "z-score": z,
        "robust+clip": np.clip(rz, -clip, clip),
        "arcsinh": np.arcsinh(rz),
        "rank-Gaussian": norm.ppf((rankdata(f, method="average") - 0.5) / n),
    }


def test_candidates_match_reference():
    rng = np.random.default_rng(7)
    f = rng.standard_t(4, 20000)
    rep = shaping_report(f, clip=4.0)
    ref = _candidates_ref(f, 4.0)
    assert set(rep.candidates) == set(ref) == set(LIGHTNESS_ORDER)
    for name in LIGHTNESS_ORDER:
        assert np.allclose(rep.candidates[name], ref[name], rtol=1e-12, atol=1e-12)
        # diagnostics are the candidate's own excess-kurt / max-abs
        assert rep.diagnostics[name]["max_abs"] == pytest.approx(np.abs(ref[name]).max())
        assert rep.diagnostics[name]["excess_kurt"] == pytest.approx(kurtosis(ref[name]))


# --------------------------------------------------------------------------------------------------
# the recommendation — fat-tailed-with-a-spike falls through to robust+clip; clean gaussian gets z-score
# --------------------------------------------------------------------------------------------------
def test_fat_tailed_with_spike_recommends_robust_clip():
    rng = np.random.default_rng(1)
    f = rng.standard_t(3, 50000)               # symmetric, fat-tailed
    f[0], f[1] = 80.0, -80.0                    # a big symmetric spike (a wild outlier)
    rep = shaping_report(f, clip=4.0, outlier_bar=5.0)

    assert isinstance(rep, ShapingReport)
    assert abs(rep.raw_skew) < 0.5             # symmetric
    assert rep.raw_max_abs > 50                # the spike dominates the raw distribution

    # z-score keeps the spike -> a large max-abs that BLOWS the bar
    assert rep.diagnostics["z-score"]["max_abs"] > 5.0
    # robust+clip caps the tail at exactly the clip
    assert rep.diagnostics["robust+clip"]["max_abs"] == pytest.approx(4.0)
    # the lightest candidate that meets the bar is robust+clip (z-score failed it)
    assert rep.recommended == "robust+clip"


def test_clean_gaussian_recommends_zscore():
    rng = np.random.default_rng(2)
    f = rng.standard_normal(50000)             # clean, near-normal, no wild outliers
    rep = shaping_report(f, clip=4.0, outlier_bar=5.0)

    assert abs(rep.raw_skew) < 0.1
    assert abs(rep.raw_excess_kurt) < 0.2
    # a clean gaussian's worst |z| sits just under the bar -> z-score already meets it
    assert rep.diagnostics["z-score"]["max_abs"] <= 5.0
    assert rep.recommended == "z-score"


# --------------------------------------------------------------------------------------------------
# edge cases — masking, degenerate input, the report's lightness order
# --------------------------------------------------------------------------------------------------
def test_masks_nonfinite_first():
    rng = np.random.default_rng(3)
    f = rng.standard_normal(10000)
    g = f.copy()
    g[::7] = np.nan
    g[3] = np.inf
    rep_f = shaping_report(f[np.isfinite(g)])   # the surviving subset, computed directly
    rep_g = shaping_report(g)                    # masking should drop exactly the non-finite rows
    assert rep_g.n_finite == int(np.isfinite(g).sum())
    assert rep_g.raw_std == pytest.approx(rep_f.raw_std)
    assert rep_g.recommended == rep_f.recommended


def test_constant_feature_is_degenerate_not_nan():
    rep = shaping_report(np.full(5000, 3.14), clip=4.0)
    # std = 0 and MAD = 0 -> z and robust legs are all zeros (no nan/inf), so the bar is met by z-score
    assert np.all(rep.candidates["z-score"] == 0.0)
    assert np.all(rep.candidates["robust+clip"] == 0.0)
    assert rep.recommended == "z-score"


def test_empty_feature_raises():
    with pytest.raises(ValueError):
        shaping_report(np.array([np.nan, np.inf, -np.inf]))


def test_str_lists_every_candidate_and_marks_recommended():
    rng = np.random.default_rng(5)
    rep = shaping_report(rng.standard_normal(2000))
    s = str(rep)
    for name in LIGHTNESS_ORDER:
        assert name in s
    assert "recommended" in s


# --------------------------------------------------------------------------------------------------
# real-block integration (skipped without DATA_DIR) — the §8 conclusion on price_dislocation
# --------------------------------------------------------------------------------------------------
@pytest.mark.skipif(getattr(io, "DATA_DIR", None) is None, reason="no DATA_DIR configured")
def test_real_block_price_dislocation_recommends_robust_clip():
    from boba.features import base
    import boba.features.price_dislocation  # noqa: F401  (registers the feature)
    from boba.research.screening import build_context, build_family

    ctx = build_context()
    spec = base.get("price_dislocation")
    legs = build_family(ctx, spec.vectorized, [(10, 500)])[(10, 500)]   # price-head span (ground truth)
    src = ctx.sources[0]
    f = legs[src]

    rep = shaping_report(f)
    # §8 conclusion: divided by σ_ev it's near-symmetric, but leaves a large outlier spike, so the
    # lightest transform that clears the "no wild outliers" bar is robust + clip.
    assert abs(rep.raw_skew) < 0.5
    assert rep.raw_max_abs > 10.0
    assert rep.diagnostics["z-score"]["max_abs"] > rep.outlier_bar
    assert rep.diagnostics["robust+clip"]["max_abs"] == pytest.approx(rep.clip)
    assert rep.recommended == "robust+clip"
