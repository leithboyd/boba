"""Walk-forward feature-screening gates — shared, tested, used by every feature notebook.

Pulled out of the per-feature builders so the gate machinery lives (and is TESTED in tests/research/test_gates*.py)
in ONE place instead of being copy-pasted into ~23 builders. This is research/analysis tooling, not part
of the production ETL path.

WHAT THE GATES ANSWER
  Gate A — regime invariance: is a feature's distribution stable across the vol/rate regime, or does it
    leak it? (`gate_a`). A feature that merely re-expresses the regime is a level in disguise, not signal.
  Gate B — predictive signal: does the feature add out-of-sample rank-IC OVER the regime-invariant momentum
    controls, walk-forward? (`signal_ic`), and does it hold up within calm/mid/wild vol (`signal_ic_by_regime`)?
  Headline uncertainty: the point IC understates its own precision under autocorrelation, so we report the
    per-fold spread and a moving-block bootstrap CI (`estimate_block_len`, `marginal_block_bootstrap`, `marginal_ci`).

EXTERNALLY VALIDATED (nothing reinvented — see tests + the verification notes):
  * rank-IC / Information Coefficient as a feature screen — Grinold & Kahn (Fundamental Law of Active Mgmt).
  * marginal/incremental IC over controls — an OUT-OF-SAMPLE NESTED-MODEL incremental-skill measure
    (forecast-encompassing flavour; Clark & McCracken). NB: it is a DIFFERENCE OF TWO rank-ICs of OLS
    predictions, *not* a partial/residualised correlation — a legitimate but distinct object (it can differ
    in magnitude from partial-Spearman; use it as a relative screen, not an effect size).
  * purged/embargoed expanding walk-forward — Lopez de Prado, Advances in Financial ML.
  * moving-block bootstrap + block length ~ ACF length / N^(1/3) — Kunsch 1989; Politis & White 2004
    (+ Patton-Politis-White 2009); fixed-length MBB + percentile CI is the standard, adequate (if not the
    most coverage-optimal) choice.
  * ratio stratification for the spurious correlation of ratios — Pearson 1897; Kronmal 1993.
  * Gate A diagnostics — Goldfeld-Quandt subgroup variance ratio (scale), monotone rank-association (track),
    PSI/covariate-shift-style binned diagnostics (disp). They check stability against the regime coordinates
    we track; they do NOT prove full distributional independence and can miss TAIL/quantile shifts.
  * sigma/lambda normalisation — volatility scaling + event-time (count) subordination (Clark 1973; Ane & Geman 2000).

CORRECTNESS NOTES (each was a bug found the hard way; tests/research/test_gates_bug_regressions.py pins them red->green):
  * Tie handling. The screening target `fwd_return/sigma_ev` is mostly EXACT ZEROS (~95% — byb's mid rarely
    moves over a 100ms window). Spearman MUST use average-rank ties (`spearmanr`/`rankdata(method="average")`);
    ordinal (argsort) ranking spreads the tied zeros across distinct ranks by time-position, manufacturing a
    spurious rank<->time signal that badly biases the IC.
  * Fold alignment. The base-only and base+legs models are fit on ONE shared valid mask and ONE set of fold
    boundaries (`_aligned_folds`); otherwise they could skip different folds and the marginal would subtract
    mismatched folds.
  * Bootstrap = the SAME estimator as the headline (per-fold-averaged marginal), resampled WITHIN each fold
    (per-fold-standardised predictions can't be pooled across folds); degenerate (all-tied/constant) resamples
    are dropped so a nan can't poison the CI.
  * Block length is derived from the data (target ACF) and the >=~30-blocks cap is on the PER-FOLD length
    (blocks are resampled within folds; the `embargo` floor can still dominate on a short fold). Too few
    blocks -> few distinct start positions -> the CI comes out too NARROW and unstable (under-coverage),
    which the cap prevents where it binds.

KNOWN SCOPE LIMITS (documented, not bugs):
  * Gate B's OLS scorer is LINEAR, so it is blind to a purely non-monotone/nonlinear incremental signal
    (a leg whose *square* predicts will read marginal ~ 0). The rate head handles this by feeding |feature|;
    for other heads, transform the feature explicitly if you expect nonlinear signal.
  * `stratified_ic` used as a STANDALONE number conditions on a causal-at-anchor yardstick computed over the
    whole series — fine for a relative screen, but it is not itself a walk-forward OOS number.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr

_AVG = "average"   # Spearman tie method — average ranks (NEVER ordinal; see module docstring)


def _interleave(a, b):
    """Interleave two equal-length arrays: a -> even rows, b -> odd rows (length 2n). Used by mirror
    augmentation to place each anchor's reflection at the SAME time index as the anchor."""
    out = np.empty(2 * len(a), float)
    out[0::2] = a; out[1::2] = b
    return out


# --------------------------------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------------------------------

def ic(x, y, min_n=100):
    """Masked Spearman rank-IC between two arrays — `nan` if fewer than `min_n` jointly-finite points.

    Mirror augmentation (AUTHORING.md → Mirror augmentation) is applied by the CALLER, by concatenating
    each input with its reflection before calling this — `ic(concat[x, mirror(x)], concat[y, -y])` — so
    this primitive stays a plain masked Spearman.
    """
    finite = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[finite], y[finite]).statistic) if finite.sum() > min_n else float("nan")


def wf_folds(features, target, k=6, embargo=2000, min_rows=100, mirror=None):
    """Purged, expanding-window walk-forward. Yields `(test_mask, oos_prediction)` per fold.

    `features` is a list of equal-length arrays (the design columns). The series is split into `k` equal
    segments; for each segment i in 1..k-1 the model trains on rows STRICTLY BEFORE it (an expanding past),
    minus an `embargo` gap that purges overlapping forward labels, standardises the design on TRAIN rows
    only, fits ordinary least squares, and predicts the segment out-of-sample. Folds with fewer than
    `min_rows` train or test rows are skipped. Causal: only past -> future. (`k` segments -> k-1 OOS folds;
    segment 0 is the initial train-only seed.)

    `mirror` (a feature's reflection callable, e.g. `np.negative`; see AUTHORING.md → Mirror augmentation)
    mirror-augments the series: each anchor is paired with its reflection — feature columns via `mirror`,
    the signed target via negation — INTERLEAVED at the same time index, so each reflection shares its
    row's fold and is purged by the same (doubled) embargo. The OLS then fits through the origin (the
    symmetric pair forces ~0 intercept) and the OOS rank-IC is direction-free.
    """
    if mirror is not None:
        features = [_interleave(np.asarray(f, float), np.asarray(mirror(f), float)) for f in features]
        t = np.asarray(target, float); target = _interleave(t, -t)
        embargo = 2 * embargo
    design = np.column_stack(features)
    n_rows = len(target)
    usable = np.isfinite(design).all(axis=1) & np.isfinite(target)
    segment_edges = np.linspace(0, n_rows, k + 1).astype(int)
    for i in range(1, k):
        test_window = np.zeros(n_rows, bool); test_window[segment_edges[i]:segment_edges[i + 1]] = True
        train_window = np.zeros(n_rows, bool); train_window[:max(0, segment_edges[i] - embargo)] = True
        train_mask, test_mask = usable & train_window, usable & test_window
        if train_mask.sum() < min_rows or test_mask.sum() < min_rows:
            continue
        train_mean = design[train_mask].mean(axis=0)
        train_std = design[train_mask].std(axis=0) + 1e-12
        standardised = np.column_stack([(design - train_mean) / train_std, np.ones(n_rows)])
        coef, *_ = np.linalg.lstsq(standardised[train_mask], target[train_mask], rcond=None)
        yield test_mask, standardised @ coef


def wf_ic(features, target, k=6, embargo=2000, mirror=None):
    """Mean out-of-sample rank-IC across the walk-forward folds (the causal OOS gate). `nan` if every fold
    was skipped (e.g. the embargo exhausts the train history on a tiny series). `mirror` (a feature
    reflection callable) mirror-augments the series — see `wf_folds`."""
    scored_target = target
    if mirror is not None:
        t = np.asarray(target, float); scored_target = _interleave(t, -t)   # the target as wf_folds saw it
    fold_ics = [spearmanr(pred[test].astype(float), scored_target[test]).statistic
                for test, pred in wf_folds(features, target, k, embargo, mirror=mirror)]
    return float(np.mean(fold_ics)) if fold_ics else float("nan")


def wf_ic_by_regime(features, target, regime, k=6, embargo=2000, min_n=100):
    """Mean OOS rank-IC WITHIN each bucket of `regime` (the regime-stable companion to `wf_ic`)."""
    bucket_ics = {}
    for test, pred in wf_folds(features, target, k, embargo):
        for bucket in np.unique(regime[test]):
            in_bucket = test & (regime == bucket)
            if in_bucket.sum() >= min_n:
                bucket_ics.setdefault(int(bucket), []).append(
                    spearmanr(pred[in_bucket], target[in_bucket]).statistic)
    return {bucket: float(np.mean(vals)) for bucket, vals in bucket_ics.items()}


def stratified_ic(feature, target, shared_yardstick, n_bins=40, min_n=100):
    """Mean within-stratum Spearman, stratifying by the shared yardstick into `n_bins` equal-mass bins.

    When feature = numA/Z and target = numB/Z share a denominator Z (`shared_yardstick`), a plain
    correlation is inflated by Z alone (Pearson's spurious correlation of ratios). Scoring WITHIN narrow
    bins of Z and mass-averaging multiplicatively decouples Z, without the over-removal a linear partial
    would cause. Returns `nan` if there isn't enough finite/distinct yardstick to bin. (`n_bins` is a fixed
    bias/variance choice: too few -> residual spurious bias, too many -> thin, noisy bins.)
    """
    finite = np.isfinite(feature) & np.isfinite(target) & np.isfinite(shared_yardstick)
    if finite.sum() <= min_n or np.unique(shared_yardstick[finite]).size < 2:
        return float("nan")
    bin_edges = np.nanquantile(shared_yardstick[finite], np.linspace(0, 1, n_bins + 1)[1:-1])
    bin_of_row = np.digitize(shared_yardstick, bin_edges)
    within_bin_ics, bin_weights = [], []
    for b in range(n_bins):
        in_bin = finite & (bin_of_row == b)
        if in_bin.sum() > min_n and np.std(feature[in_bin]) > 0 and np.std(target[in_bin]) > 0:
            r = spearmanr(feature[in_bin], target[in_bin]).statistic
            if np.isfinite(r):
                within_bin_ics.append(r); bin_weights.append(int(in_bin.sum()))
    return float(np.average(within_bin_ics, weights=bin_weights)) if within_bin_ics else float("nan")


# --------------------------------------------------------------------------------------------------
# Gate B — predictive signal
# --------------------------------------------------------------------------------------------------

def signal_ic(feature_legs, controls, target, *, feature_kind="alpha", own=False, strat_var=None,
              k=6, embargo=2000):
    """Gate B value, rounded to 3 dp.

    For an `alpha` (or a control's CROSS-venue leg) this is the MARGINAL rank-IC over the `controls`: the
    per-fold-averaged `spearman(full_pred) - spearman(base_pred)` on SHARED folds (a lead test; see
    `fold_marginals`). NB this is an out-of-sample nested-model incremental-IC, NOT a partial correlation.
    For a `control` feature's OWN-venue leg, marginal-over-own-controls is circular, so it falls back to a
    STANDALONE rank-IC: stratified by `strat_var` when the control divides by the scored target's yardstick
    (decoupling the shared denominator), else a plain mean IC. `own=True` selects the standalone branch.
    """
    if feature_kind == "control" and own:
        if strat_var is None:
            return round(float(np.mean([ic(leg, target) for leg in feature_legs])), 3)
        return round(float(np.mean([stratified_ic(leg, target, strat_var) for leg in feature_legs])), 3)
    per_fold, _ = fold_marginals(feature_legs, controls, target, k, embargo)
    return round(float(np.mean(per_fold)), 3) if per_fold else float("nan")


def signal_ic_by_regime(feature_legs, controls, target, regime, *, feature_kind="alpha", own=False,
                        strat_var=None, k=6, embargo=2000):
    """The regime-stable companion to `signal_ic` — the same marginal, restricted to each `regime` bucket.
    Buckets with fewer than 100 rows are dropped (parity with the alpha branch and `wf_ic_by_regime`).
    The alpha branch scores the full and base models on the SAME rows (`_aligned_folds`), so both see the
    same buckets; the `bucket in base_ics` filter below is a defensive guard that never excludes under that
    alignment."""
    if feature_kind == "control" and own and strat_var is not None:
        out = {}
        for bucket in np.unique(regime[np.isfinite(regime)]):
            in_bucket = regime == bucket
            if in_bucket.sum() < 100:                      # skip thin buckets (parity with the alpha branch)
                continue
            out[int(bucket)] = round(float(np.mean(
                [stratified_ic(np.where(in_bucket, leg, np.nan), target, strat_var) for leg in feature_legs])), 3)
        return out
    full_ics, base_ics = {}, {}
    for test, full_pred, base_pred in _aligned_folds(controls, feature_legs, target, k, embargo):
        for bucket in np.unique(regime[test]):
            in_bucket = test & (regime == bucket)
            if in_bucket.sum() >= 100:
                full_ics.setdefault(int(bucket), []).append(spearmanr(full_pred[in_bucket], target[in_bucket]).statistic)
                base_ics.setdefault(int(bucket), []).append(spearmanr(base_pred[in_bucket], target[in_bucket]).statistic)
    return {bucket: round(float(np.mean(full_ics[bucket])) - float(np.mean(base_ics[bucket])), 3)
            for bucket in full_ics if bucket in base_ics}


# --------------------------------------------------------------------------------------------------
# Gate A — regime invariance
# --------------------------------------------------------------------------------------------------

def _deciles(level):
    """Assign each row to one of 10 equal-mass buckets of `level` (the regime coordinate). Non-finite rows
    get the sentinel -1 so they are EXCLUDED from gate_a's per-decile loops (plain `digitize` would dump
    NaN rows into the top bucket and silently pollute it)."""
    finite = np.isfinite(level)
    buckets = np.digitize(level, np.nanpercentile(level[finite], np.arange(10, 100, 10)))
    buckets[~finite] = -1
    return buckets


def _decile_mean_dispersion(values, norm, decile_groupings):
    """Worst (over the given decile groupings) std of per-decile MEANS of `values`, divided by `norm`.
    Catches a NON-monotone leak (e.g. a U-shaped dependence on the regime) that a monotone IC misses."""
    return max(float(np.nanstd([np.nanmean(values[buckets == d] ) for d in range(10)]) / (norm + 1e-12))
               for buckets in decile_groupings)


def gate_a(feature, vol_level, rate_level):
    """The four Gate A regime-invariance numbers for one feature, each collapsed to the WORST coordinate:
      scale  — std of the feature across vol-deciles, max/min       (want < ~3)
      track  — |IC(feature, vol/rate level)|                        (monotone signed mean-track, want < ~0.05)
      mag    — |IC(|feature|, vol/rate level)|                      (the magnitude must not track either, want < ~0.1)
      disp   — per-decile-mean dispersion of feature & |feature|    (non-monotone leak, worst, want < ~0.1)
    `vol_level`/`rate_level` are the regime COORDINATE (never controls). These are stability/independence
    diagnostics against the coordinates we track — they do NOT prove full distributional independence and
    can miss TAIL/quantile shifts (use them as a screen, not a theorem).
    """
    decile_groupings = (_deciles(vol_level), _deciles(rate_level))
    # per-vol-decile spread; keep only finite, positive entries so an empty/constant decile can't make
    # max/min divide by zero or return nan.
    per_decile_std = [s for s in (np.nanstd(feature[decile_groupings[0] == d]) for d in range(10))
                      if np.isfinite(s) and s > 0]
    return dict(
        scale=round(max(per_decile_std) / min(per_decile_std), 2) if len(per_decile_std) >= 2 else float("nan"),
        track=round(float(np.nanmax([abs(ic(feature, vol_level)), abs(ic(feature, rate_level))])), 3),
        mag=round(float(np.nanmax([abs(ic(np.abs(feature), vol_level)), abs(ic(np.abs(feature), rate_level))])), 3),
        disp=round(max(_decile_mean_dispersion(feature, np.nanstd(feature), decile_groupings),
                       _decile_mean_dispersion(np.abs(feature), np.nanmean(np.abs(feature)), decile_groupings)), 3),
    )


# --------------------------------------------------------------------------------------------------
# Headline-marginal uncertainty (per-fold spread + moving-block bootstrap CI)
# --------------------------------------------------------------------------------------------------

def _acf_decay_lag(series, thr=None, lags=(50, 100, 200, 500, 1000, 2000, 5000, 10000)):
    """First lag (from `lags`) where |sample autocorrelation| falls below `thr` (default 2/sqrt(n)) — a
    cheap dependence-length proxy for sizing the bootstrap block."""
    series = np.asarray(series, float)
    series = series[np.isfinite(series)]
    n_rows = series.size
    if n_rows < 200:
        return lags[0]
    series = series - series.mean()
    total_variance = float(series @ series) or 1.0
    thr = 2.0 / np.sqrt(n_rows) if thr is None else thr
    for lag in lags:
        if lag >= n_rows:
            break
        if abs(float(series[:-lag] @ series[lag:]) / total_variance) < thr:
            return lag
    return lags[-1]


def estimate_block_len(target, embargo, safety=10, min_blocks_n=None):
    """Moving-block length (grid-anchor units): the target's ACF-decay lag x `safety`, floored at the
    walk-forward `embargo` and capped so the series keeps >= ~30 blocks — UNLESS the `embargo` floor is
    larger than that cap, in which case the floor wins and there may be fewer (the floor takes precedence).
    Data-derived, so it adapts per feature/block instead of assuming a fixed memory.

    `min_blocks_n` is the length the >=30-blocks cap is enforced against — pass the PER-FOLD OOS length
    when the block is resampled within folds (the bootstrap does), since a cap on the full series would
    still leave only a handful of blocks per fold. Too few blocks -> few distinct start positions -> the
    moving-block bootstrap CI comes out too NARROW and unstable (under-coverage), which the cap prevents
    where it binds.
    """
    cap_n = len(target) if min_blocks_n is None else min_blocks_n
    return int(np.clip(safety * _acf_decay_lag(target), embargo, max(embargo, cap_n // 30)))


def _aligned_folds(controls, feature_legs, target, k=6, embargo=2000, min_rows=100):
    """Walk-forward folds SHARED by the base-only and base+legs models — so a per-fold marginal is well
    defined. ONE valid mask (finite across controls AND legs AND target) and ONE set of fold boundaries
    drive both fits; otherwise the two models could skip *different* folds and the marginal would subtract
    mismatched folds. Yields `(test_mask, full_pred, base_pred)` per surviving fold.
    """
    base_design = np.column_stack(controls)
    full_design = np.column_stack(controls + feature_legs)
    n_rows = len(target)
    usable = np.isfinite(full_design).all(axis=1) & np.isfinite(target)   # shared: rows usable by BOTH models
    segment_edges = np.linspace(0, n_rows, k + 1).astype(int)
    folds = []
    for i in range(1, k):
        test_window = np.zeros(n_rows, bool); test_window[segment_edges[i]:segment_edges[i + 1]] = True
        train_window = np.zeros(n_rows, bool); train_window[:max(0, segment_edges[i] - embargo)] = True
        train_mask, test_mask = usable & train_window, usable & test_window
        if train_mask.sum() < min_rows or test_mask.sum() < min_rows:
            continue

        def _oos_prediction(design):                       # standardise on TRAIN only, fit OLS, predict all rows
            train_mean = design[train_mask].mean(axis=0)
            train_std = design[train_mask].std(axis=0) + 1e-12
            standardised = np.column_stack([(design - train_mean) / train_std, np.ones(n_rows)])
            coef, *_ = np.linalg.lstsq(standardised[train_mask], target[train_mask], rcond=None)
            return standardised @ coef

        folds.append((test_mask, _oos_prediction(full_design), _oos_prediction(base_design)))
    return folds


def fold_marginals(feature_legs, controls, target, k=6, embargo=2000):
    """Per-fold marginal rank-IC and the per-fold OOS arrays, for the headline uncertainty.

    Returns `(per_fold, folds)` where `per_fold[i] = spearman(full_i, target_i) - spearman(base_i, target_i)`
    (exact, average-rank Spearman) and `folds[i] = (full_pred_i, base_pred_i, target_i)` over fold i's OOS
    rows. Built from `_aligned_folds` so the base and base+legs predictions come from the SAME folds.
    """
    per_fold, folds = [], []
    for test_mask, full_pred, base_pred in _aligned_folds(controls, feature_legs, target, k, embargo):
        per_fold.append(float(spearmanr(full_pred[test_mask], target[test_mask]).statistic
                              - spearmanr(base_pred[test_mask], target[test_mask]).statistic))
        folds.append((full_pred[test_mask], base_pred[test_mask], target[test_mask]))
    return per_fold, folds


def marginal_block_bootstrap(folds, block_len, B=400, seed=0, percentiles=(5, 95)):
    """90% CI (and bootstrap mean) for the PER-FOLD-AVERAGED marginal rank-IC.

    `folds`: list of `(full_pred, base_pred, target)` OOS arrays (one per fold, each contiguous in time).
    `B`: number of bootstrap resamples. For each resample, draw contiguous blocks of `block_len` WITHIN each
    fold (with replacement), recompute that fold's marginal Spearman, and average across folds; the spread
    of those `B` averages gives the CI. Average-rank ties are required (the target is mostly tied at zero),
    and degenerate (all-tied/constant) resampled blocks are dropped so a `nan` can't poison the CI. Returns
    `(lo, hi, mean)`, or `(nan, nan, nan)` if no usable resample was produced.
    """
    rng = np.random.default_rng(seed)
    block_offsets = np.arange(block_len)
    resample_estimates = []
    for _ in range(B):
        fold_marginals_this_resample = []
        for full_pred, base_pred, target in folds:
            n_rows = len(target)
            if n_rows <= block_len:
                idx = np.arange(n_rows)
            else:
                block_starts = rng.integers(0, n_rows - block_len + 1, int(np.ceil(n_rows / block_len)))
                idx = (block_starts[:, None] + block_offsets).ravel()[:n_rows]
            target_ranks = rankdata(target[idx], method=_AVG)
            if np.std(target_ranks) == 0:                  # resample landed all-tied target -> skip this fold
                continue
            full_ic = np.corrcoef(rankdata(full_pred[idx], method=_AVG), target_ranks)[0, 1]
            base_ic = np.corrcoef(rankdata(base_pred[idx], method=_AVG), target_ranks)[0, 1]
            if np.isfinite(full_ic) and np.isfinite(base_ic):   # corrcoef is nan on a constant resampled block
                fold_marginals_this_resample.append(full_ic - base_ic)
        if fold_marginals_this_resample:
            resample_estimates.append(np.mean(fold_marginals_this_resample))
    if not resample_estimates:
        return float("nan"), float("nan"), float("nan")
    resample_estimates = np.asarray(resample_estimates)
    lo, hi = np.percentile(resample_estimates, list(percentiles))
    return float(lo), float(hi), float(resample_estimates.mean())


def marginal_ci(feature_legs, controls, target, *, embargo=2000, B=400, seed=0, k=6):
    """Headline-marginal uncertainty in one call: per-fold marginals, count positive, and the block-bootstrap
    90% CI (block length auto-derived from the target's autocorrelation, capped per fold)."""
    per_fold, folds = fold_marginals(feature_legs, controls, target, k, embargo)
    if not folds:                                          # every fold skipped (tiny / all-nan input)
        return dict(per_fold=[], pos=0, nf=0, ci=(float("nan"), float("nan")),
                    boot_mean=float("nan"), block_len=0, block_s=0.0)
    # ACF length from the POOLED OOS target (not just fold 0); >=30-blocks cap on the SMALLEST fold, since
    # the bootstrap resamples blocks WITHIN each fold.
    pooled_target = np.concatenate([fold_target for _, _, fold_target in folds])
    smallest_fold = min(len(fold_target) for _, _, fold_target in folds)
    block_len = estimate_block_len(pooled_target, embargo, min_blocks_n=smallest_fold)
    lo, hi, boot_mean = marginal_block_bootstrap(folds, block_len, B=B, seed=seed)
    return dict(per_fold=[round(x, 3) for x in per_fold], pos=sum(x > 0 for x in per_fold),
                nf=len(per_fold), ci=(round(lo, 3), round(hi, 3)), boot_mean=round(boot_mean, 3),
                block_len=block_len, block_s=round(block_len * 0.05, 1))
