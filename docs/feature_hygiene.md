# Feature hygiene methodology

How to analyse whether a candidate feature carries **regime-invariant, transferable
signal**. This is the statistical counterpart to the oracle rule in `CLAUDE.md`
(oracle = computed correctly; hygiene = a real signal). It defines a procedure only;
whether any given feature is used is a judgement informed by these analyses.

These checks are **not hard pass/fail rules** — they are a standard set of analyses
run against every feature. A feature can fail a check and still carry genuine signal
(for example, a feature can both leak the regime *and* hold real predictive
structure). We may still decide to include a feature that fails one of these checks —
but doing so demands a clear, deliberate justification, because a failed check is a
strong warning that the feature will not transfer.

## Harness

Run on a real multi-day block, 60/40 train/test:

- **Target** — forward, dimensionless: `log(next-window / trailing-baseline)`.
- **Base `[cr, cv]`** — causal momenta `cr = log(rate_recent / rate_baseline)`,
  `cv = log(vol_recent / vol_baseline)`: the reference a feature is measured against.
- **Level controls `[lv, lr]`** — `lv = log(vol_level)`, `lr = log(rate_level)`: the
  regime itself; adding them is the leak detector.
- **Score** — standardize on train, `lstsq` fit, **Spearman rank-IC on test**.
  Incremental IC of a feature `f` over a set `S` = `IC([S, f]) − IC([S])`.

In the nn_features setting: base = per-venue rate + vol momenta on the merged clock;
level controls = the absolute rate/vol levels.

## Checks

1. **Causal.** The feature uses only data at-or-before `t`; the target is strictly
   forward. Verify every window index the feature reads is `≤ t`. (This one is a hard
   requirement — a lookahead feature is invalid, not merely risky.)
2. **Marginal value.** Incremental IC over `[cr, cv]` for each target. A feature with
   no increment for any head is carrying no signal beyond the base.
3. **Leak.** Recompute the increment over `[cr, cv, lv, lr]`. If a gain over `[cr,cv]`
   collapses to ~0 once `[lv,lr]` are in the base, the feature was sorting across
   regimes rather than adding regime-invariant structure. Diagnostics: `corr(f, lv)`
   and `corr(f, lr)` (≈ 0 = clean), and `f`'s mean across deciles of the level (a
   monotone profile = the regime bleeding in). Applies to any feature regardless of
   construction — a ratio or regime-normalized feature can either remove a leak or
   inject one; the analysis decides.
4. **Stratified leak.** Cut into ~10 deciles of the level; inside each fixed-regime
   bucket, measure the increment over `[cr,cv]`. A feature that adds pooled but ~0
   within every bucket was only sorting across regimes. Catches nonlinear leaks the
   linear `[lv,lr]` control misses; the mean within-bucket gain should match the
   pooled `[cr,cv,lv,lr]`-controlled gain.
5. **Normalizer stability** (magnitude features). Compare candidate normalizers by how
   well each flattens the conditional std across level deciles: take the per-decile
   std, normalize the curve, measure its spread (CV-across-deciles; lower = flatter).
   The best normalizer can depend on the target's clock; test candidates per clock.
6. **Incremental & collinearity.** Measure whether the feature adds over the *existing
   feature set* (not just the base) and whether it is an algebraic restatement of it.
   Check pairwise correlation and whether it collapses onto a factor already present.
7. **Temporal stationarity.** Compare the regime-decile CV (instantaneous regime
   removed) to the temporal-CV (per-day / per-chunk std). High temporal-CV means the
   feature carries its own slow drift even if it is cross-sectionally invariant.
8. **OOS across a regime gap.** Score out-of-sample, ideally training and testing in
   different regimes. An in-sample gain that does not hold across the gap is a leak in
   disguise; treat single-window results as provisional.

## Reading the results

These analyses describe a feature; they do not by themselves decide its fate. Typical
readings:

| analysis result | reading |
|---|---|
| adds incrementally and clean across all checks | strong candidate to keep |
| leaks (3 / 4) | likely a regime-level proxy; include only with explicit justification, and prefer handling the regime level outcome-side |
| clean but non-incremental or collinear (6) | adds no information over the existing set |
| signal only at a finer scale / shorter horizon than the head | better suited to a finer-scale model than this head |
| cross-sectionally invariant but high temporal drift (7) | its level is non-transferable; if used, handle the drift outcome-side rather than feeding it raw |
