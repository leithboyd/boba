"""Extensive synthetic test-suite for boba.research.gates — the walk-forward feature-screening gates.

Everything is SYNTHETIC and STOCHASTIC (no data files). The whole suite is derived from ONE reusable
crypto-microstructure generator, ``synth_market``, that produces data resembling the real screening
target rather than clean iid toys:

  * a SPARSE forward-return target — mostly EXACT ZEROS (byb's mid usually doesn't move over a 100 ms
    window: ~90-96% zeros on a real block), the nonzero "moves" HEAVY-TAILED (Student-t) and SIGNED;
  * AUTOCORRELATION / persistence (AR(1) latent driver + autocorrelated move-occupancy), not iid;
  * a slowly-varying, mean-reverting REGIME coordinate (log-normal-ish vol_level / signed rate_level);
  * a candidate feature with tunable MARGINAL signal (``signal``) and tunable REGIME LEAKAGE
    (``regime_leak`` / ``mag_leak``).

Each gate gets BOTH a fixture that EXHIBITS its problem (must FAIL the gate) and one that PASSES.
Robustness: every RNG is seeded; genuinely-noisy stochastic assertions are taken over a median of
several seeds so a single unlucky draw can't flake.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import rankdata, spearmanr

from boba.research import gates as g

# Public API under test (re-checked by test_public_api_covered):
PUBLIC_FUNCS = [
    "ic", "wf_folds", "wf_ic", "wf_ic_by_regime", "stratified_ic",
    "signal_ic", "signal_ic_by_regime", "gate_a",
    "estimate_block_len", "fold_marginals", "marginal_block_bootstrap", "marginal_ci",
]

SEEDS = (0, 1, 2, 3, 4)


# ==================================================================================================
# THE reusable stochastic crypto-microstructure generator
# ==================================================================================================

def _ar1(rng, n, rho):
    """AR(1) latent path, unit-variance normalised — the persistence/memory in a feature stream."""
    z = np.zeros(n)
    eps = rng.standard_normal(n)
    for i in range(1, n):
        z[i] = rho * z[i - 1] + eps[i]
    return z / (z.std() + 1e-12)


def synth_market(rng, n, *, signal=0.0, regime_leak=0.0, mag_leak=0.0,
                 zero_frac=0.92, autocorr=0.97, move_clustering=0.0, regime_rho=0.999):
    """Generate one synthetic crypto-microstructure screening problem.

    Returns a dict with:
      feat       : the candidate feature (AR(1) driver + optional regime leakage)
      tgt        : sparse, mostly-zero, heavy-tailed SIGNED forward return  (the screening target)
      z          : the latent autocorrelated driver behind both feat and the signal in tgt
      moves      : boolean mask of the (sparse) non-zero target bars
      vol_level  : slowly-varying POSITIVE regime coordinate (log-normal-ish, mean-reverting vol)
      rate_level : slowly-varying SIGNED regime coordinate (mean-reverting rate)
      base       : realistic regime-momentum controls (1-step diffs of the regime coords)

    Knobs:
      signal         marginal predictive power of feat over tgt (0 = NULL, no signal)
      regime_leak    signed-mean leakage: feat's level tracks vol_level
      mag_leak       magnitude-only leakage: |feat| tracks vol_level (signed mean stays flat)
      zero_frac      fraction of EXACT-ZERO target bars (~real fwd_return sparsity)
      autocorr       AR(1) coefficient of the latent driver (persistence/memory)
      move_clustering AR-ify the move-occupancy so non-zero bars cluster in time (overlapping labels)
      regime_rho     persistence of the (mean-reverting) regime coordinates

    The regime coords are slowly-varying but MEAN-REVERTING / BOUNDED — real vol & rate levels do not
    random-walk to infinity. vol_level is multiplicative (log-normal-ish, the realistic vol model);
    rate_level is a signed mean-reverting level. A regime-invariant feature genuinely passes Gate A
    against these, whereas an unbounded random walk would spuriously correlate with a persistent feature.
    """
    # slowly-varying, mean-reverting regime coordinates
    log_vol = _ar1(rng, n, regime_rho) * 0.8
    vol_level = np.exp(log_vol)                       # positive, wide range, log-normal-ish vol
    rate_level = _ar1(rng, n, regime_rho)             # signed, mean-reverting rate

    # latent autocorrelated driver
    z = _ar1(rng, n, autocorr)

    # sparse, time-clustered move occupancy
    if move_clustering > 0:
        occ = _ar1(rng, n, move_clustering)
        thr = np.quantile(occ, zero_frac)
        moves = occ > thr
    else:
        moves = rng.random(n) > zero_frac

    # heavy-tailed signed move magnitudes + optional injected signal (only on move bars)
    raw = np.where(moves, rng.standard_t(3, n), 0.0)
    tgt = raw + signal * z * moves

    # candidate feature: latent driver + optional regime leakage
    vc = (vol_level - vol_level.mean()) / (vol_level.std() + 1e-12)
    feat = z.copy()
    feat = feat + regime_leak * vc
    if mag_leak > 0:
        # magnitude scales with |regime| but sign is independent -> signed mean stays ~flat
        feat = feat + mag_leak * np.abs(rng.standard_normal(n)) * np.abs(vc) * rng.choice([-1, 1], n)

    base = [np.diff(vol_level, prepend=vol_level[0]),
            np.diff(rate_level, prepend=rate_level[0])]
    return dict(feat=feat, tgt=tgt, z=z, moves=moves,
                vol_level=vol_level, rate_level=rate_level, base=base)


# Default size: big enough for a 6-fold (embargo 2000) walk-forward with thick folds.
N = 60_000


@pytest.fixture
def null_data():
    """Heavy-zeros + NO predictive signal — the manufacture-no-signal guard fixture."""
    rng = np.random.default_rng(12345)
    return synth_market(rng, N, signal=0.0, zero_frac=0.93, autocorr=0.97, move_clustering=0.95)


@pytest.fixture
def signal_data():
    """Heavy-zeros WITH genuine marginal signal in the moves."""
    rng = np.random.default_rng(54321)
    return synth_market(rng, N, signal=2.0, zero_frac=0.93, autocorr=0.97, move_clustering=0.95)


def _median_over_seeds(fn, seeds=SEEDS):
    return float(np.median([fn(np.random.default_rng(s), s) for s in seeds]))


def _regime_buckets(level, nb=3):
    """Balanced (equal-mass) integer regime buckets of a level coordinate — calm/mid/wild."""
    edges = np.quantile(level, np.linspace(0, 1, nb + 1)[1:-1])
    return np.digitize(level, edges)


# ==================================================================================================
# 0. API coverage + sanity
# ==================================================================================================

def test_public_api_covered():
    for name in PUBLIC_FUNCS:
        assert hasattr(g, name), f"missing public function {name}"
        assert callable(getattr(g, name))


def test_generator_shape_and_sparsity():
    rng = np.random.default_rng(0)
    d = synth_market(rng, N, signal=1.0, zero_frac=0.93, move_clustering=0.95)
    for k in ("feat", "tgt", "z", "vol_level", "rate_level"):
        assert d[k].shape == (N,)
    zf = np.mean(d["tgt"] == 0.0)
    assert 0.88 < zf < 0.97, f"target should be sparse-zero, got zero_frac={zf}"
    # nonzero moves are heavy-tailed & signed
    nz = d["tgt"][d["tgt"] != 0]
    assert nz.min() < 0 < nz.max()
    assert (d["vol_level"] >= 0).all()


# ==================================================================================================
# 1. ic — masked Spearman primitive
# ==================================================================================================

def test_ic_known_correlation_recovery():
    # rank-IC of a ~0.7-pearson pair recovers a clearly-positive, near-0.7 spearman
    def one(rng, _s):
        x = rng.standard_normal(5000)
        y = 0.7 * x + np.sqrt(1 - 0.49) * rng.standard_normal(5000)
        return g.ic(x, y)
    med = _median_over_seeds(one)
    assert 0.6 < med < 0.78, med


def test_ic_perfect_monotone_is_one():
    x = np.arange(500.0)
    assert g.ic(x, 2 * x + 1) == pytest.approx(1.0, abs=1e-9)
    assert g.ic(x, -x) == pytest.approx(-1.0, abs=1e-9)


def test_ic_nan_when_too_few_points():
    # fewer than min_n jointly-finite points -> nan
    assert np.isnan(g.ic(np.arange(50.0), np.arange(50.0), min_n=100))
    # exactly at the boundary: needs STRICTLY more than min_n
    assert np.isnan(g.ic(np.arange(100.0), np.arange(100.0), min_n=100))
    assert not np.isnan(g.ic(np.arange(101.0), np.arange(101.0), min_n=100))


def test_ic_masks_nans():
    a = np.arange(200.0)
    b = np.arange(200.0)
    a[:150] = np.nan  # only 50 finite -> below default min_n=100
    assert np.isnan(g.ic(a, b))
    a2 = np.arange(200.0)
    b2 = np.arange(200.0)
    a2[:50] = np.nan  # 150 finite, still perfectly monotone on the finite part
    assert g.ic(a2, b2) == pytest.approx(1.0, abs=1e-9)


def test_ic_null_is_near_zero(null_data):
    # a noise feature vs the sparse-zero target: marginal rank-IC ~ 0
    med = _median_over_seeds(
        lambda rng, s: g.ic(synth_market(rng, 20000, signal=0.0)["feat"],
                            synth_market(np.random.default_rng(s + 1000), 20000, signal=0.0)["tgt"]))
    assert abs(med) < 0.03, med


# ==================================================================================================
# 2. wf_folds — purged expanding-window walk-forward
# ==================================================================================================

def test_wf_folds_count_and_boundaries():
    rng = np.random.default_rng(0)
    feats = [rng.standard_normal(N)]
    y = rng.standard_normal(N)
    folds = list(g.wf_folds(feats, y, k=6, embargo=2000))
    assert len(folds) == 5  # folds i=1..k-1
    edges = np.linspace(0, N, 7).astype(int)
    for i, (t, p) in enumerate(folds, start=1):
        idx = np.where(t)[0]
        assert idx.min() == edges[i]
        assert idx.max() == edges[i + 1] - 1
        assert p.shape == (N,)


def test_wf_folds_k_controls_fold_count():
    rng = np.random.default_rng(1)
    feats = [rng.standard_normal(N)]
    y = rng.standard_normal(N)
    for k in (3, 4, 6, 10):
        assert len(list(g.wf_folds(feats, y, k=k, embargo=2000))) == k - 1


def test_wf_folds_skips_thin_folds():
    rng = np.random.default_rng(2)
    # tiny n with a big embargo eats the whole train window -> all folds skipped
    feats = [rng.standard_normal(300)]
    y = rng.standard_normal(300)
    assert list(g.wf_folds(feats, y, k=6, embargo=2000, min_rows=100)) == []
    # with embargo 0 and enough rows, folds appear again
    assert len(list(g.wf_folds([rng.standard_normal(3000)], rng.standard_normal(3000),
                               k=6, embargo=0, min_rows=100))) == 5


def test_wf_folds_embargo_removes_last_train_rows():
    # The embargo must purge exactly the last `embargo` train rows before each test block. Probe it by
    # making those rows carry the OPPOSITE relation to the rest of train, and matching the test block:
    #   bulk train  (rows [0, e1-1500))   : y = -3*feat   (negative relation)
    #   band        (rows [e1-1500, e1))  : y = +3*feat   (positive relation)  <- the last train rows
    #   fold-1 test (rows [e1, e2))       : y = +3*feat   (positive)
    # With embargo 0 the band is in train -> the fit leans POSITIVE -> fold-1 OOS IC ~ +1.
    # With embargo > 1500 the band is purged -> only the bulk's NEGATIVE relation is learned -> IC ~ -1.
    # The OOS IC sign FLIP is unambiguous evidence the trailing `embargo` rows were removed.
    rng = np.random.default_rng(7)
    n = 12000
    edges = np.linspace(0, n, 7).astype(int)
    e1, e2 = edges[1], edges[2]
    feat = rng.standard_normal(n)
    y = -3.0 * feat + 0.1 * rng.standard_normal(n)              # bulk: negative
    band = slice(e1 - 1500, e1)
    y[band] = 3.0 * feat[band] + 0.1 * rng.standard_normal(1500)        # band: positive
    y[e1:e2] = 3.0 * feat[e1:e2] + 0.1 * rng.standard_normal(e2 - e1)   # test: positive

    def fold1_test_ic(embargo):
        for t, p in g.wf_folds([feat], y, k=6, embargo=embargo, min_rows=100):
            return spearmanr(p[t], y[t]).statistic  # first fold only
        return np.nan

    assert fold1_test_ic(0) > 0.5        # band in train -> learns the positive relation
    assert fold1_test_ic(1700) < -0.5    # band purged -> learns the bulk's negative relation


def test_wf_folds_train_only_standardisation():
    # Standardisation must use TRAIN mu/sd only; a wild scale shift confined to the TEST region
    # must not corrupt the train fit (predictions stay finite & the train relation generalises).
    rng = np.random.default_rng(3)
    n = 30000
    feat = rng.standard_normal(n)
    y = 0.8 * feat + 0.3 * rng.standard_normal(n)
    feat_leak = feat.copy()
    feat_leak[25000:] *= 1e6  # giant test-only scale blow-up
    pred_clean = dict((tuple(np.where(t)[0][[0, -1]]), p) for t, p in g.wf_folds([feat], y, embargo=500))
    pred_leak = dict((tuple(np.where(t)[0][[0, -1]]), p) for t, p in g.wf_folds([feat_leak], y, embargo=500))
    # all folds whose train window is entirely before the blow-up are bit-identical
    for key in pred_clean:
        if key[0] < 25000:  # this fold's train precedes the blow-up region
            np.testing.assert_allclose(pred_clean[key][:25000], pred_leak[key][:25000], rtol=1e-9, atol=1e-9)


def test_wf_folds_causal_no_lookahead_leak():
    # A feature whose relationship to y is ANTI-stationary (sign flips every fold) cannot be learned
    # causally: train-fit on past folds predicts the WRONG sign forward -> low/negative mean OOS IC,
    # while a stationary genuinely-known feature gives high OOS IC. (Past->future only.)
    def known_minus_flip(rng, _s):
        n = 30000
        moves = rng.random(n) > 0.92
        y = np.where(moves, rng.standard_t(3, n), 0.0)
        edges = np.linspace(0, n, 7).astype(int)
        sign = np.ones(n)
        for i in range(6):
            sign[edges[i]:edges[i + 1]] = (-1) ** i
        feat_flip = sign * (y + 0.1 * rng.standard_normal(n))
        feat_known = y + 0.3 * rng.standard_normal(n)
        return g.wf_ic([feat_known], y, embargo=500), g.wf_ic([feat_flip], y, embargo=500)

    known = _median_over_seeds(lambda rng, s: known_minus_flip(rng, s)[0])
    flip = _median_over_seeds(lambda rng, s: known_minus_flip(rng, s)[1])
    assert known > 0.25, known                    # stationary known feature: real OOS power
    assert flip < known - 0.3, (flip, known)      # anti-stationary: no causal generalisation


# ==================================================================================================
# 3. wf_ic / wf_ic_by_regime
# ==================================================================================================

def test_wf_ic_null_near_zero(null_data):
    d = null_data
    assert abs(g.wf_ic([d["feat"]], d["tgt"])) < 0.03


def test_wf_ic_recovers_signal(signal_data):
    d = signal_data
    assert g.wf_ic([d["feat"]], d["tgt"]) > 0.1


def test_wf_ic_by_regime_buckets_and_signal(signal_data):
    d = signal_data
    reg = _regime_buckets(d["vol_level"])  # 3 balanced calm/mid/wild buckets
    out = g.wf_ic_by_regime([d["feat"]], d["tgt"], reg)
    assert isinstance(out, dict) and len(out) >= 2
    # signal present in every populated regime bucket
    assert all(v > 0.05 for v in out.values()), out


def test_wf_ic_by_regime_null_near_zero(null_data):
    d = null_data
    reg = _regime_buckets(d["vol_level"])
    out = g.wf_ic_by_regime([d["feat"]], d["tgt"], reg)
    assert all(abs(v) < 0.05 for v in out.values()), out


# ==================================================================================================
# 4. stratified_ic — shared-denominator decoupling
# ==================================================================================================

def test_stratified_ic_collapses_spurious_shared_denominator():
    # X = numA/Z, tgt = numB/Z with numA,numB INDEPENDENT and POSITIVE: a wide shared 1/Z makes the
    # two ratios co-move strongly (plain IC inflated), but the comovement vanishes within Z-strata.
    def one(rng, _s):
        n = 60000
        logZ = np.cumsum(rng.standard_normal(n)) * 0.02
        logZ -= logZ.mean()
        Z = np.exp(logZ) * 3.0           # wide positive multiplicative range
        numA = np.abs(rng.standard_normal(n)) + 0.5
        numB = np.abs(rng.standard_normal(n)) + 0.5
        X, tgt = numA / Z, numB / Z
        return g.ic(X, tgt), g.stratified_ic(X, tgt, Z)

    plain = _median_over_seeds(lambda rng, s: one(rng, s)[0])
    strat = _median_over_seeds(lambda rng, s: one(rng, s)[1])
    assert plain > 0.5, plain                 # plain IC badly inflated by the shared denominator
    assert abs(strat) < 0.12, strat           # stratifying by Z collapses it toward 0
    assert strat < plain - 0.4


def test_stratified_ic_preserves_genuine_within_stratum_signal():
    # A real common driver inside each Z bucket survives stratification.
    def one(rng, _s):
        n = 60000
        logZ = np.cumsum(rng.standard_normal(n)) * 0.02
        logZ -= logZ.mean()
        Z = np.exp(logZ) * 3.0
        common = rng.standard_normal(n)
        X = (rng.standard_normal(n) + 1.5 * common) / Z
        tgt = (rng.standard_normal(n) + 1.5 * common) / Z
        return g.stratified_ic(X, tgt, Z)
    strat = _median_over_seeds(one)
    assert strat > 0.3, strat


def test_stratified_ic_nan_when_no_usable_buckets():
    # All-constant feature -> std==0 in every bucket -> no contributing strata -> nan
    rng = np.random.default_rng(0)
    n = 5000
    sv = rng.standard_normal(n)
    feat = np.zeros(n)
    tgt = rng.standard_normal(n)
    assert np.isnan(g.stratified_ic(feat, tgt, sv))


def test_stratified_ic_null_near_zero(null_data):
    d = null_data
    med = _median_over_seeds(
        lambda rng, s: g.stratified_ic(synth_market(rng, 20000, signal=0.0)["feat"],
                                       synth_market(np.random.default_rng(s + 7), 20000, signal=0.0)["tgt"],
                                       synth_market(np.random.default_rng(s + 99), 20000, signal=0.0)["vol_level"]))
    assert abs(med) < 0.05, med


# ==================================================================================================
# 5. signal_ic (Gate B) — all branches
# ==================================================================================================

def test_signal_ic_null_marginal_near_zero(null_data):
    d = null_data
    assert abs(g.signal_ic([d["feat"]], d["base"], d["tgt"])) < 0.02


def test_signal_ic_recovers_marginal_signal(signal_data):
    d = signal_data
    marg = g.signal_ic([d["feat"]], d["base"], d["tgt"])
    assert marg > 0.05, marg                       # clearly above the ~0.01 floor


def test_signal_ic_alpha_branch_equals_wf_ic_difference():
    rng = np.random.default_rng(8)
    z = rng.standard_normal(N)
    base = [rng.standard_normal(N)]
    tgt = 0.3 * z + rng.standard_normal(N)
    expected = round(g.wf_ic(base + [z], tgt) - g.wf_ic(base, tgt), 3)
    assert g.signal_ic([z], base, tgt) == expected


def test_signal_ic_control_own_strat_none_equals_mean_ic():
    rng = np.random.default_rng(9)
    n = 20000
    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n)
    tgt = 0.4 * f1 + 0.2 * f2 + rng.standard_normal(n)
    got = g.signal_ic([f1, f2], [], tgt, feature_kind="control", own=True)
    expected = round(float(np.mean([g.ic(f1, tgt), g.ic(f2, tgt)])), 3)
    assert got == expected


def test_signal_ic_control_own_strat_var_uses_stratified():
    rng = np.random.default_rng(10)
    n = 30000
    sv = np.exp(np.cumsum(rng.standard_normal(n)) * 0.02) * 3.0
    f = rng.standard_normal(n) / sv
    tgt = rng.standard_normal(n) / sv
    got = g.signal_ic([f], [], tgt, feature_kind="control", own=True, strat_var=sv)
    expected = round(float(np.mean([g.stratified_ic(f, tgt, sv)])), 3)
    assert got == expected


def test_signal_ic_is_rounded_3dp(signal_data):
    d = signal_data
    val = g.signal_ic([d["feat"]], d["base"], d["tgt"])
    assert val == round(val, 3)


# ==================================================================================================
# 6. signal_ic_by_regime
# ==================================================================================================

def test_signal_ic_by_regime_signal(signal_data):
    d = signal_data
    reg = _regime_buckets(d["vol_level"])
    out = g.signal_ic_by_regime([d["feat"]], d["base"], d["tgt"], reg)
    assert isinstance(out, dict) and len(out) >= 2
    assert all(v > 0.03 for v in out.values()), out


def test_signal_ic_by_regime_null(null_data):
    d = null_data
    reg = _regime_buckets(d["vol_level"])
    out = g.signal_ic_by_regime([d["feat"]], d["base"], d["tgt"], reg)
    assert all(abs(v) < 0.05 for v in out.values()), out


def test_signal_ic_by_regime_control_own_strat_branch():
    rng = np.random.default_rng(11)
    n = 40000
    sv = np.exp(np.cumsum(rng.standard_normal(n)) * 0.02) * 3.0
    f = rng.standard_normal(n) / sv
    tgt = rng.standard_normal(n) / sv
    vol = np.abs(np.cumsum(rng.standard_normal(n))) + 1
    reg = _regime_buckets(vol)
    out = g.signal_ic_by_regime([f], [], tgt, reg, feature_kind="control", own=True, strat_var=sv)
    assert isinstance(out, dict) and len(out) >= 2
    # each bucket value is a rounded float, dict keyed by int regime label
    for k, v in out.items():
        assert isinstance(k, int)
        assert v == round(v, 3)


# ==================================================================================================
# 7. gate_a — regime invariance (FAIL must exceed the bars; PASS under all four)
# ==================================================================================================

def _gate_a_pass(d):
    ga = d
    return (ga["scale"] < 3.0 and ga["track"] < 0.05 and ga["mag"] < 0.1 and ga["disp"] < 0.1)


def test_gate_a_pass_regime_invariant():
    # regime-invariant feature: all four metrics under their bars (median over seeds)
    def metrics(rng, _s):
        d = synth_market(rng, N, signal=0.0, regime_leak=0.0, mag_leak=0.0)
        ga = g.gate_a(d["feat"], d["vol_level"], d["rate_level"])
        return ga["scale"], ga["track"], ga["mag"], ga["disp"]
    sc = _median_over_seeds(lambda rng, s: metrics(rng, s)[0])
    tr = _median_over_seeds(lambda rng, s: metrics(rng, s)[1])
    mg = _median_over_seeds(lambda rng, s: metrics(rng, s)[2])
    dp = _median_over_seeds(lambda rng, s: metrics(rng, s)[3])
    assert sc < 3.0 and tr < 0.05 and mg < 0.1 and dp < 0.1, (sc, tr, mg, dp)


def test_gate_a_fail_scale_leak():
    # std tracks the regime -> scale (max/min std across vol buckets) blows past its bar
    def one(rng, _s):
        d = synth_market(rng, N, signal=0.0)
        feat = rng.standard_normal(N) * d["vol_level"]
        return g.gate_a(feat, d["vol_level"], d["rate_level"])["scale"]
    assert _median_over_seeds(one) > 3.0


def test_gate_a_fail_track_leak():
    # signed mean tracks the regime -> track (|IC(feat, level)|) blows past its bar
    def one(rng, _s):
        d = synth_market(rng, N, signal=0.0)
        vc = (d["vol_level"] - d["vol_level"].mean()) / d["vol_level"].std()
        feat = 2.0 * vc + rng.standard_normal(N)
        return g.gate_a(feat, d["vol_level"], d["rate_level"])["track"]
    assert _median_over_seeds(one) > 0.05


def test_gate_a_fail_magnitude_only_leak():
    # MAGNITUDE-ONLY leak: |feat| tracks the regime while the signed mean stays flat.
    # This is the case that makes the |feature| checks (mag / disp) earn their keep:
    # track (signed) stays under its bar, but mag (|feat| track) and disp blow past theirs.
    def metrics(rng, _s):
        d = synth_market(rng, N, signal=0.0)
        vol = d["vol_level"]
        sign = rng.choice([-1, 1], N)
        feat = sign * (vol - vol.min() + 0.1) * (0.5 + 0.5 * np.abs(rng.standard_normal(N)))
        ga = g.gate_a(feat, vol, d["rate_level"])
        return ga["track"], ga["mag"], ga["disp"]
    tr = _median_over_seeds(lambda rng, s: metrics(rng, s)[0])
    mg = _median_over_seeds(lambda rng, s: metrics(rng, s)[1])
    dp = _median_over_seeds(lambda rng, s: metrics(rng, s)[2])
    assert tr < 0.05, tr            # signed mean is flat (track passes) ...
    assert mg > 0.1, mg             # ... but |feat| tracks the regime (mag fails) ...
    assert dp > 0.1, dp             # ... and per-decile dispersion of |feat| leaks (disp fails)


def test_gate_a_null_feature_passes(null_data):
    # a pure-noise feature on the NULL fixture must PASS gate_a (no manufactured leak)
    d = null_data
    noise = np.random.default_rng(0).standard_normal(N)
    ga = g.gate_a(noise, d["vol_level"], d["rate_level"])
    assert _gate_a_pass(ga), ga


def test_gate_a_returns_four_rounded_keys():
    rng = np.random.default_rng(1)
    d = synth_market(rng, N, signal=0.0)
    ga = g.gate_a(d["feat"], d["vol_level"], d["rate_level"])
    assert set(ga) == {"scale", "track", "mag", "disp"}
    assert ga["scale"] == round(ga["scale"], 2)
    for k in ("track", "mag", "disp"):
        assert ga[k] == round(ga[k], 3)


# ==================================================================================================
# 8. estimate_block_len + _acf_decay_lag
# ==================================================================================================

def test_acf_decay_lag_grows_with_autocorrelation():
    def lag(rng, ac):
        return g._acf_decay_lag(_ar1(rng, 30000, ac))
    white = _median_over_seeds(lambda rng, s: lag(rng, 0.0))
    mid = _median_over_seeds(lambda rng, s: lag(rng, 0.9))
    high = _median_over_seeds(lambda rng, s: lag(rng, 0.995))
    assert white <= mid <= high
    assert high > white                      # strictly longer memory => later decay lag


def test_acf_decay_lag_floor_on_short_series():
    # n < 200 returns the smallest candidate lag
    assert g._acf_decay_lag(np.random.default_rng(0).standard_normal(50)) == 50


def test_estimate_block_len_within_bounds():
    rng = np.random.default_rng(0)
    n = 60000
    embargo = 500
    # random walk: enormous ACF -> wants a huge block but is capped at n//30
    rw = np.cumsum(rng.standard_normal(n))
    L = g.estimate_block_len(rw, embargo)
    assert embargo <= L <= n // 30
    # white noise: floored at embargo
    Lw = g.estimate_block_len(rng.standard_normal(n), embargo)
    assert Lw == embargo


def test_estimate_block_len_grows_with_autocorr_length():
    def bl(rng, ac):
        return g.estimate_block_len(_ar1(rng, 90000, ac), embargo=300)
    low = _median_over_seeds(lambda rng, s: bl(rng, 0.0))
    high = _median_over_seeds(lambda rng, s: bl(rng, 0.999))
    assert high >= low
    assert high > low                       # longer memory -> longer block


# ==================================================================================================
# 9. fold_marginals
# ==================================================================================================

def test_fold_marginals_structure_and_consistency(signal_data):
    d = signal_data
    per_fold, folds = g.fold_marginals([d["feat"]], d["base"], d["tgt"])
    assert len(per_fold) == len(folds) == 5
    for pf, (full_pred, base_pred, y) in zip(per_fold, folds):
        assert full_pred.shape == base_pred.shape == y.shape
        # per_fold[i] is exactly full-marginal minus base-marginal Spearman on that fold's OOS rows
        recomputed = float(spearmanr(full_pred, y).statistic - spearmanr(base_pred, y).statistic)
        assert pf == pytest.approx(recomputed, abs=1e-9)


def test_fold_marginals_positive_on_signal(signal_data):
    d = signal_data
    per_fold, _ = g.fold_marginals([d["feat"]], d["base"], d["tgt"])
    assert np.mean(per_fold) > 0.05
    assert sum(x > 0 for x in per_fold) >= 4   # signal is broadly positive across folds


def test_fold_marginals_null_near_zero(null_data):
    d = null_data
    per_fold, _ = g.fold_marginals([d["feat"]], d["base"], d["tgt"])
    assert abs(np.mean(per_fold)) < 0.03


# ==================================================================================================
# 10. marginal_block_bootstrap — the zero-tie / rankdata regression
# ==================================================================================================

def test_bootstrap_null_brackets_zero_and_unbiased(null_data):
    d = null_data
    per_fold, folds = g.fold_marginals([d["feat"]], d["base"], d["tgt"])
    L = g.estimate_block_len(folds[0][2], 2000)
    lo, hi, mean = g.marginal_block_bootstrap(folds, L, B=300, seed=0)
    point = float(np.mean(per_fold))
    assert lo <= 0.0 <= hi                          # NULL CI brackets 0
    assert mean == pytest.approx(point, abs=0.02)   # unbiased: boot mean ~ per-fold point


def test_bootstrap_signal_excludes_zero_and_centred(signal_data):
    d = signal_data
    per_fold, folds = g.fold_marginals([d["feat"]], d["base"], d["tgt"])
    L = g.estimate_block_len(folds[0][2], 2000)
    lo, hi, mean = g.marginal_block_bootstrap(folds, L, B=300, seed=0)
    point = float(np.mean(per_fold))
    assert lo > 0.0                                 # SIGNAL CI strictly excludes 0
    assert mean == pytest.approx(point, abs=0.03)   # centred on per-fold-averaged point
    assert lo < point < hi


def test_bootstrap_deterministic_same_seed(signal_data):
    d = signal_data
    _, folds = g.fold_marginals([d["feat"]], d["base"], d["tgt"])
    L = g.estimate_block_len(folds[0][2], 2000)
    a = g.marginal_block_bootstrap(folds, L, B=200, seed=42)
    b = g.marginal_block_bootstrap(folds, L, B=200, seed=42)
    assert a == b
    c = g.marginal_block_bootstrap(folds, L, B=200, seed=43)
    assert a != c                                   # different seed -> different draw


def test_bootstrap_handles_short_fold():
    # a fold shorter than the block length is used whole (idx = arange(n)); no crash, finite output
    rng = np.random.default_rng(0)
    n = 300
    folds = [(rng.standard_normal(n), rng.standard_normal(n), np.where(rng.random(n) > 0.9, rng.standard_t(3, n), 0.0))]
    lo, hi, mean = g.marginal_block_bootstrap(folds, block_len=1000, B=50, seed=0)
    assert np.isfinite(lo) and np.isfinite(hi) and np.isfinite(mean)


def test_rankdata_required_ordinal_argsort_is_biased():
    # DOCUMENTS why rankdata is mandatory (the shipped code uses it). On a zero-heavy target with a
    # time-trending prediction, ORDINAL (argsort) ranking spreads the tied zeros by time-position,
    # manufacturing a huge spurious rank<->time IC; average-rank (rankdata) stays ~0.
    def one(rng, _s):
        n = 20000
        moves = rng.random(n) > 0.95
        y = np.where(moves, rng.standard_t(3, n), 0.0)
        pred = np.arange(n, dtype=float) + 100 * rng.standard_normal(n)   # time-trending

        def ordinal(v):
            o = np.empty(len(v))
            o[np.argsort(v, kind="stable")] = np.arange(len(v))
            return o
        r_rank = np.corrcoef(rankdata(pred), rankdata(y))[0, 1]
        r_ord = np.corrcoef(ordinal(pred), ordinal(y))[0, 1]
        return r_rank, r_ord

    r_rank = _median_over_seeds(lambda rng, s: one(rng, s)[0])
    r_ord = _median_over_seeds(lambda rng, s: one(rng, s)[1])
    assert abs(r_rank) < 0.05, r_rank          # correct (rankdata) ~ 0
    assert r_ord > 0.5, r_ord                  # ordinal manufactures a massive spurious IC
    # and PROVE the shipped bootstrap actually handles the ties (not a source grep): on a zero-heavy fold
    # set it must stay UNBIASED (bootstrap mean ~ the per-fold point estimate). An ordinal-ranking bootstrap
    # would be biased away from the point (demonstrated in test_gates_bug_regressions / the adversarial teeth tests).
    from scipy.stats import spearmanr
    rng2 = np.random.default_rng(0)
    folds = []
    for _ in range(5):
        moves = rng2.random(40000) > 0.95
        y = np.where(moves, rng2.standard_t(3, 40000), 0.0)
        full = 0.3 * y + rng2.standard_normal(40000)
        base = rng2.standard_normal(40000)
        folds.append((full, base, y))
    point = float(np.mean([spearmanr(f, yy).statistic - spearmanr(b, yy).statistic for f, b, yy in folds]))
    lo, hi, boot_mean = g.marginal_block_bootstrap(folds, 2000, B=200, seed=0)
    assert abs(boot_mean - point) < 0.02, (boot_mean, point)   # unbiased => ties handled correctly
    assert lo < point < hi


# ==================================================================================================
# 11. marginal_ci — the one-call headline-uncertainty wrapper
# ==================================================================================================

def test_marginal_ci_documented_dict_shape(signal_data):
    d = signal_data
    out = g.marginal_ci([d["feat"]], d["base"], d["tgt"], B=150, seed=0)
    assert set(out) == {"per_fold", "pos", "nf", "ci", "boot_mean", "block_len", "block_s"}
    assert isinstance(out["per_fold"], list) and len(out["per_fold"]) == out["nf"] == 5
    assert isinstance(out["ci"], tuple) and len(out["ci"]) == 2
    assert out["ci"][0] <= out["ci"][1]
    assert out["block_s"] == pytest.approx(round(out["block_len"] * 0.05, 1))
    # block length respects the estimate_block_len bounds
    assert 2000 <= out["block_len"] <= N // 30


def test_marginal_ci_pos_counts_correctly(signal_data):
    d = signal_data
    out = g.marginal_ci([d["feat"]], d["base"], d["tgt"], B=120, seed=0)
    assert out["pos"] == sum(x > 0 for x in out["per_fold"])


def test_marginal_ci_ci_brackets_per_fold_mean_on_signal(signal_data):
    d = signal_data
    out = g.marginal_ci([d["feat"]], d["base"], d["tgt"], B=300, seed=0)
    point = float(np.mean(out["per_fold"]))
    lo, hi = out["ci"]
    assert lo <= point <= hi
    assert lo > 0.0                              # genuine signal: CI excludes 0
    assert out["pos"] == 5


def test_marginal_ci_null_brackets_zero(null_data):
    d = null_data
    out = g.marginal_ci([d["feat"]], d["base"], d["tgt"], B=300, seed=0)
    lo, hi = out["ci"]
    assert lo <= 0.0 <= hi
    assert out["boot_mean"] == pytest.approx(float(np.mean(out["per_fold"])), abs=0.02)


def test_marginal_ci_deterministic(signal_data):
    d = signal_data
    a = g.marginal_ci([d["feat"]], d["base"], d["tgt"], B=150, seed=7)
    b = g.marginal_ci([d["feat"]], d["base"], d["tgt"], B=150, seed=7)
    assert a == b


# ==================================================================================================
# Mirror augmentation (reflection of the tape through byb's mid; AUTHORING.md → Mirror augmentation)
# ==================================================================================================
def test_wf_ic_mirror_matches_manual_interleave():
    # wf_ic(mirror=fn) must equal running the plain walk-forward on the dataset where each anchor is
    # interleaved with its reflection (feature cols via fn, target negated) and the embargo doubled.
    rng = np.random.default_rng(7)
    n = 12_000
    f1, f2 = rng.standard_normal(n), rng.standard_normal(n)
    y = 0.4 * f1 - 0.3 * f2 + 0.5 * rng.standard_normal(n)
    auto = g.wf_ic([f1, f2], y, embargo=500, mirror=np.negative)

    def il(a, b):
        out = np.empty(2 * len(a)); out[0::2] = a; out[1::2] = b; return out
    manual = g.wf_ic([il(f1, -f1), il(f2, -f2)], il(y, -y), embargo=1000, mirror=None)
    assert auto == pytest.approx(manual, abs=1e-12)


def test_mirror_ic_preserves_odd_kills_even():
    # The mirror-augmented Spearman measures only the ODD (direction-consistent) association:
    #   a purely odd relationship is preserved; a purely even one cannot manufacture signed signal.
    rng = np.random.default_rng(11)
    m = 60_000
    f = rng.standard_normal(m)
    cat = lambda a: np.concatenate([a, -a])
    y_odd = 0.5 * f + 0.3 * rng.standard_normal(m)
    y_even = 0.8 * np.abs(f) + 0.3 * rng.standard_normal(m)
    assert g.ic(cat(f), cat(y_odd)) == pytest.approx(g.ic(f, y_odd), abs=0.01)   # odd preserved
    assert abs(g.ic(cat(f), cat(y_even))) < 0.02                                  # even -> ~0 (no signed signal)


def test_mirror_forces_oos_fit_through_origin():
    # The symmetric pair drives the OLS intercept to ~0. Probe via wf_folds with a strong target drift:
    # the mirrored fit must not lean on a constant, so its prediction is odd in the feature sign.
    rng = np.random.default_rng(3)
    n = 12_000
    f = rng.standard_normal(n)
    y = 0.5 * f + 0.4 + 0.3 * rng.standard_normal(n)        # +0.4 directional drift
    preds = [pred for _, pred in g.wf_folds([f], y, embargo=500, mirror=np.negative)]
    assert preds, "no folds produced"
    # mirrored design has each row paired with its negation -> predictions are antisymmetric across the pair
    p = preds[0]
    assert np.allclose(p[0::2], -p[1::2], atol=1e-9)        # pred(reflection) == -pred(anchor)
