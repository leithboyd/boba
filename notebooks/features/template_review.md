# Review of `notebooks/features/template.ipynb`

Scope: source and saved outputs of `notebooks/features/template.ipynb`, reviewed as a methodology template and as a worked `price_dislocation` example. I did not re-run the notebook; numeric checks below use the saved notebook outputs.

Severity key:

- Critical: can lead to an unsupported shipping decision, leakage, or a wrong implementation.
- High: materially weakens the methodology or creates a serious conceptual contradiction.
- Medium: should be fixed before using the notebook as a reusable template.
- Low: wording, maintainability, consistency, or presentation polish.

## Critical Findings

### C1. The rate-head verdict is not supported by the gates

Location: sections 6, 8, and 10; cells 16, 18, 19, 30, and 32.

The notebook selects a separate rate-head span family in cell 16 using `abs(d)` versus `rate_target`, but the hygiene gates in cell 18 are run only on:

- `disloc = price_dislocation(... price_member ...)`, i.e. the price-selected spans;
- the price target `target = fwd_return / sigma_at_anchor`;
- a representative feature from `OTHERS[0]` for Gate A.

The notebook then concludes that the feature is "real and ship-able, for both heads" and that the rate head should use the sharp `fast=1, slow=100` pair. That conclusion is stronger than the analysis performed. The rate-head-selected feature is not put through Gate A, not scored walk-forward against the rate target, and not checked per exchange.

Suggested change: add a separate rate-head gate block. Build the signed features at `rate_member` spans for every exchange, run Gate A on both the signed feature and its magnitude, and run Gate B against a properly defined rate target. The univariate `abs(feature) -> count` heat-map should be labelled as a span-selection diagnostic only until this is done. Update the verdict to distinguish "price head passed gates" from "rate head pending gates."


**reply: [VALID]**

Verified against source. The `rate_member` pick is explicitly flagged as diagnostic-only and ungated: line 774 says it is `a DIAGNOSTIC-ONLY readout of where |feature| peaks against the move count; it is NOT separately put through the §5 gates (only the price feature is gated...)`. Gate A runs on `rep = disloc[OTHERS[0]]` (line 867) — a price-span signed feature — and Gate B's `signal_ic`/`stratified` paths score against `target` (the σ_ev price target), never `rate_target` for the chosen rate-head spans. Yet the §10/§6 verdict (line 909) asserts `real and ship-able, for **both heads, every exchange**` and line 911 picks `the rate head stayed sharp (fast=1, slow=100)`. The reviewer is correct that the rate-head verdict outruns the gates actually executed: no Gate A on the rate-span feature, no walk-forward of `|feature|`/signed against `rate_target`, no per-exchange rate check. This is a genuine over-claim, not a misread.

**Suggested fix:** In `build_feat_nb.py`, soften the §6 conclusion (line ~909) and §10 verdict (line ~1195) to distinguish heads: state "price head: passed Gates A and B" and "rate head: span selected by the diagnostic heat-map only; gates pending." Either add a rate-head gate block that runs Gate A on the signed feature and `|feature|` at `rate_member` spans for every exchange and Gate B (`signal_ic(..., tgt=rate_target)` with `STRAT_VAR=lam_at_anchor`), or relabel the rate-head row of the heat-map as a span-selection diagnostic and drop "ship-able for both heads" until those gates are run.

### C2. Span selection is in-sample, so the walk-forward IC is still selection-biased

Location: cell 16 and all conclusions using `price_member` / `rate_member`.

The heat maps compute raw in-sample Spearman ICs across multiple fast/slow pairs and exchanges, then `best_member()` chooses the best span using the full block. The chosen span is later re-scored with walk-forward folds on the same block. That is still data snooping: the test periods influenced the span choice before the walk-forward score was reported.

The cell comment says the in-sample sweep is "ONLY to PICK a time-scale, not to claim out-of-sample power," but if the same block is then used for the out-of-sample claim, the selected member has already been optimized on that block's future periods.

Suggested change: use nested walk-forward selection, or split blocks by purpose:

- selection block(s): choose candidate spans;
- validation block(s): estimate performance and gates;
- final holdout block(s): report the number.

At minimum, state that the reported walk-forward IC is post-selection and provisional.


**reply: [PARTIALLY-VALID]**

The mechanism the reviewer describes is real: `best_member` is `np.unravel_index(np.nanargmax(grid), grid.shape)` (line 772) over `price_grid`, each cell an **in-sample** `spearmanr(d, target)` on the full block (line 750), and the chosen `price_member` is then re-scored by the same-block walk-forward in §5. So the span pick does see the block's later periods before they are used as walk-forward test folds — that is selection leakage in the strict sense. But the reviewer overstates it as an unqualified defect: the code already discloses exactly this (lines 769-771: `each grid cell is an IN-SAMPLE spearmanr... used ONLY to PICK a time-scale, not to claim out-of-sample power. The chosen feature is then re-scored OUT-OF-SAMPLE...`), and per the orchestration context this is a single-block ALPHA worked example with a separate multi-block OOS harness in `tools/oss`. The bias is also small here (a 4x4 grid, IC dominated by structure not noise). So the finding is right in substance but the "still selection-biased / data snooping" framing is partly already-addressed disclosure.

**Suggested fix:** Add one sentence to the §6 markdown (cell after line 771) noting the reported walk-forward IC is **post-span-selection and provisional**, with held-out span selection deferred to the multi-block `tools/oss` harness. No code change to the worked example is required given its single-block scope.

### C3. A single saved 24h block cannot justify "ship-grade", "model-ready", or "ship-able"

Location: cells 12, 19, 30, and 32.

The notebook uses one block from `list_blocks(...)[0]`, yet it repeatedly uses production-strength language: "ship-grade estimate", "model-ready as-is", "ship-able", and "Keep it." The notebook itself later says there is still work to run: additivity over existing features, day-to-day stability, and out-of-sample across a market-regime change. Those pending checks directly contradict the shipping verdict.

Suggested change: downgrade the verdict to "single-block candidate passed local checks." Reserve "ship" or "model-ready" for the checklist after multi-block, day-by-day, regime-gap, existing-feature-set, and streaming-builder tests have passed.


**reply: [PARTIALLY-VALID]**

Factually grounded: the notebook uses `block = list_blocks(TARGET, "front_levels")[0]` (line 350) — one ~24h slice — and the saved verdict uses `ship-grade` (cells 12/13), `ship-able` (cell 19, line 909), and `model-ready as-is` (line 907). Meanwhile §8 (lines 1156-1159) lists still-pending checks (additivity over existing features, day-to-day stability, OOS across a regime change) and §10 (lines 1210-1216) is an unchecked `To ship:` checklist. So production-strength wording does coexist with an explicit not-yet-shipped list — a real tension the reviewer correctly flags. However, the reviewer treats this as a pure contradiction; per the orchestration context this is deliberately a single-block ALPHA *worked example* of the method, with multi-block OOS living in `tools/oss`, and the checklist already signals "not shipped." So it is overclaimed wording, not a false shipping decision.

**Suggested fix:** In `build_feat_nb.py`, replace "ship-grade estimate" / "model-ready as-is" / "real and ship-able" (lines ~632, 907, 909) with single-block-scoped language, e.g. "single-block candidate — clears every local gate" and reserve "ship"/"model-ready" for after the §10 checklist plus multi-block/regime-gap validation. Keep the `To ship:` checklist as the gating list.

### C4. The oracle is not independent enough for the claim being made

Location: section 4; cell 10.

The oracle imports and uses `KernelMeanEMA` and `LiveFrontEMA` from `boba.ema`. That is useful as a production-style integration check, but it is not a fully independent oracle for the EMA definitions. A bug in the EMA classes, live-front convention, `tick` order, or warm-up behavior could be shared by the analysis and the "oracle."

This also conflicts with the repository validation guidance in `CLAUDE.md`, which calls for a dead-simple reference with no shared production helpers.

Suggested change: split this into two checks:

- a dead-simple reference loop that uses only scalar variables and direct formulas for the feature, with no `boba.ema` imports;
- the current production-style streaming builder check, described as an integration/parity check.

Then soften the current conclusion from "both are right" to "the vectorized and streaming implementations agree."


**reply: [VALID]**

Confirmed: the §4 oracle imports and uses the production EMA classes — line 481 `from boba.ema import KernelMeanEMA, LiveFrontEMA`, and `LiveDislocation` builds its legs from `LiveFrontEMA(n_fast)` / `KernelMeanEMA(vol_span)` (lines 496-498). CLAUDE.md's validation rule is explicit that the oracle must be "**dead simple** — direct array ops / explicit loops... **no shared code** with `build_features_raw`. Plain `numpy`, no production helpers," and "no `boba.ema` imports" is exactly the spirit. So a bug shared by `LiveFrontEMA`/`KernelMeanEMA` (live-front `(1-a)*committed + a*latest`, the `E/W` warm-up, or `tick` ordering) would be invisible to this check. The reviewer is right that §4 is a vectorized-vs-streaming *parity* check, not an independent oracle. Mitigating nuance (worth noting, doesn't void the point): the EMA convention itself is separately validated against a plain one-event loop in `notebooks/03_ema_clock_validation.ipynb`, which the source repeatedly cites (lines 129, 164) — so the definitions are not wholly unchecked, but they are not re-checked *here* independently.

**Suggested fix:** In §4, add a small second reference that recomputes one gap's `(fast-slow)/sigma` with scalar Python loops and explicit `(1-a)*c + a*latest` / `E,W += a*...; E/W` formulas, **no `boba.ema` imports**, and assert it matches; relabel the existing `LiveDislocation` block as a production integration/parity check. Then change the §4 conclusion (line ~604) from "both are right" to "the vectorized and streaming production implementations agree, and both match an independent scalar reference."

### C5. The notebook contradicts itself about whether the rate head is fed signed features or `|feature|`

Location: cells 1, 2, 15, 16, 17, 18, 25, 26, and 32.

Several cells correctly say the model receives the signed feature for both heads and that `abs(feature)` is only a diagnostic. But cell 2 and cell 17 also say "the rate head is fed `|feature|`." Cell 18 then phrases the magnitude leak test as if `|feature|` is the rate-head input.

This is a serious conceptual contradiction because it changes the production feature contract. Feeding signed features lets the model learn cancellation/reinforcement across exchanges; feeding magnitudes discards sign before the model sees it.

Suggested change: use one rule everywhere:

"The production model receives the signed feature for both heads. The rate-head diagnostic also evaluates `abs(feature)` because a nonlinear model can learn magnitude from signed inputs, and because magnitude leakage would still be exploitable by the rate head."

Then update every sentence that says the rate head is fed `|feature|`.


**reply: [VALID]**

This is a real, in-source self-contradiction, not stale output. The dominant rule is stated repeatedly: line 94-95 `the model is fed the **signed** feature for *both* heads, and the rate head learns the magnitude... on its own`; line 716 `the model is fed the *signed* feature for both heads`; line 170 `never a pre-computed |feature|`. But two Gate-A passages assert the opposite as the production contract: line 177 `for **both the signed feature and its magnitude |feature|** (the rate head is fed |feature|)` and line 790 `(the rate head is fed |feature|, so a magnitude that tracks the regime leaks straight into it)`. As the reviewer notes, these are not the same claim — "fed the signed feature, learns magnitude internally" vs "fed |feature|" describe different feature contracts (the latter discards cross-exchange sign cancellation before the model). The parenthetical is trying to *motivate* why the magnitude leak matters, but it does so by misstating the input. VALID.

**Suggested fix:** In `build_feat_nb.py`, edit the two parentheticals (lines 177 and 790) to remove "the rate head is fed |feature|." Replace with the reason the magnitude check still matters under signed input, e.g.: "the rate head receives the **signed** feature and can learn its magnitude, so a magnitude that tracks the regime would leak into it." Leave lines 94-95 and 716 as the single canonical rule.

### C6. The stated two-head/subordination target model is not the same as the diagnostics

Location: cells 1, 8, 15, 16, 25, and 32.

The model description says the rate head predicts a count distribution `P(K = k)` and the price head predicts a family of price-change distributions `D_k` conditional on the move count. The diagnostics do something different:

- the price diagnostic scores a fixed 100 ms return divided by per-move `sigma_ev`, not a `k`-conditioned price distribution or per-event return;
- the rate target is `fwd_count / lambda_ev`, which is not itself a count distribution;
- the rate target omits the horizon multiplier, so it has units of seconds rather than "count relative to expected count over this horizon."

For rank correlation, multiplying by the constant horizon would not change the score. For model semantics, calibration, and distribution mixing, the distinction matters.

Suggested change: define the head targets precisely and make diagnostics match them. If the rate head is meant to predict relative count intensity, define `rate_target = fwd_count / (lambda_ev * horizon_seconds)` and explain how it is converted back into a count distribution. If the price head is conditional on count, add diagnostics conditioned on `K` or use a per-event/k-move return target.


**reply: [VALID]**

The diagnostics genuinely differ from the stated two-head/subordination model, as the reviewer claims. (1) Price head is described as a count-conditioned family `D_k` mixed by `P(K=k)` (lines 73-78), but the price diagnostic scores a fixed-horizon `target = fwd_return / sigma_at_anchor` (line 435) — not a k-conditioned or per-event return. (2) `rate_target = fwd_count / np.maximum(lam_at_anchor, 1e-9)` (line 739); since `lam_ev` is moves/sec (line 406 `lam = e_mv / ... # byb mid-moves per second`), this carries units of seconds and is not itself a count distribution. (3) The horizon multiplier is omitted — the source even concedes it: line 739 comment `this carries units of seconds; the constant horizon factor drops out of the rank correlation`, and H11 flags the same. The reviewer correctly concedes rank-IC is unaffected by the constant horizon, so for the screening number shown this is harmless; the valid part is that the *prose contract* (dimensionless "count relative to expected count over this horizon") does not match the *coded* quantity, which matters for calibration/distribution mixing if reused. VALID on the definitional mismatch.

**Suggested fix:** In §6, change `rate_target` (line 739) to `fwd_count / (lam_at_anchor * HORIZON_NS / 1e9)` so it is the dimensionless "observed vs expected count over the horizon," or keep the value but relabel it in the comment/prose as "count divided by moves-per-second; equal up to the constant horizon for rank-IC only." Separately, add a one-line note in §1 that the price diagnostic scores the marginal fixed-horizon σ-return as a screen, with the k-conditioned `D_k` family being the model target it feeds, not what the §6 IC measures.

## High Findings

### H1. Shared-denominator coupling is not ruled out for the signed price feature

Location: cells 13, 17, 18, and 21.

The feature is divided by `sigma_ev`, and the price target is also divided by `sigma_ev`. The notebook says a signed price-head alpha "shares no denominator with the sigma target" because "the signs decouple it." That is not generally guaranteed. A common positive denominator can still affect rank correlation when numerator magnitudes, skew, tails, or missingness vary with the denominator.

The self-test in cell 21 validates the control branch for a `count / lambda_ev` ratio, but it does not validate the signed-alpha case.

Suggested change: add a sensitivity line for the alpha path: report Gate B both unstratified and stratified by `sigma_ev`, and/or run a signed-null simulation where independent signed numerators are divided by the same `sigma_ev`. Replace "sign decouples it" with "we check that shared-scale coupling is negligible."


**reply: [PARTIALLY-VALID]**

The factual core is correct: the self-test cell (the `count/λ_ev` spurious/real pair, lines 931-965) exercises only the **control** branch (`FEATURE_KIND="control"`, `own=True`, `STRAT_VAR=lam`), and there is no signed-null simulation for the alpha path. And the code does state the decoupling claim as a near-certainty — `STRAT_VAR = None` is justified because "A SIGNED price-head alpha like price_dislocation shares no denominator with the σ_ev target (the signs decouple it)". That is a real, mild over-statement: a common positive denominator can still bleed into rank-IC via numerator/denominator dependence, so "the signs decouple it" is an assertion, not a measured fact (which also trips the template's own "don't assert any property you haven't measured" guard rail). Where the reviewer **overstates**: the price feature *is* put through the full Gate B marginal walk-forward and clears it at ≈0.086, and Gate A separately shows it barely tracks vol/rate level — so it is not unvalidated, just not specifically probed for shared-scale coupling. So this is a worthwhile hardening, not a defect that undermines the verdict.

**Suggested fix:** In the §6 gate cell, add one cheap diagnostic for the alpha path — print Gate B both unstratified and stratified by `sigma_at_anchor` (reuse `stratified_ic(rep, target, sigma_at_anchor)` vs the joint `signal_ic`), and soften the `STRAT_VAR` comment from "the signs decouple it" to "signed numerators make shared-scale coupling negligible — verified by the stratified-vs-unstratified line below."

### H2. Gate A is run on only one representative exchange

Location: cell 18.

`rep = disloc[OTHERS[0]]` means Gate A regime-invariance checks are run only for `bin`. The notebook then generalizes to every exchange. That is unsafe because the exchanges have different feed cadences, data policies, spreads, and stale-book behavior. The notebook itself flags feed-resolution differences between `bin`, `byb`, and `okx`.

Suggested change: run Gate A for every feature that will be fed to the model, and report either every row or the worst case by check. A "same construction" argument is not enough when the data-generating processes differ by venue.


**reply: [VALID]**

Confirmed against the code: `rep = disloc[OTHERS[0]]` (line 867) means every Gate A number (`band`/scale, `loc_v`, `loc_r`, `mag_v`, `mag_r`, `mean_disp`, `mag_disp`) is computed on **bin only** (`OTHERS = ["bin", "okx"]`), and the comment "same construction for every exchange -> one is enough for Gate A" is the exact generalization the reviewer flags. The "same construction" defense is weak precisely here, because Gate A is a property of the *output distribution*, which depends on the data-generating process, and the template itself documents that the venues differ materially: byb/okx top-of-book is stale at p90 ~100-160 ms while bin is sub-ms (line 1046). Worse, the one venue chosen (`OTHERS[0]` = bin) is the *freshest*, so Gate A is run on the venue least likely to expose a stale-book regime leak — the okx leg, which the verdict still ships, is never regime-checked. The Gate B side is already done per-exchange (lines 884-885), so extending Gate A is a small, consistent change.

**Suggested fix:** Loop Gate A over `OTHERS` instead of a single `rep`: compute the seven Gate A rows per exchange and report the worst case per check (e.g. `loc_v = max(abs(ic(disloc[ex], vol_level)) for ex in OTHERS)`), or emit one gate-row block per exchange. Drop the "one is enough for Gate A" comment.

### H3. The feed-resolution artifact is acknowledged but not tested before the verdict

Location: cell 24 and section 10.

The notebook correctly notes that cross-venue lead/lag can be a feed-cadence artifact. But this is placed after the main gates and is not part of the pass/fail evidence for the worked example. The conclusion still claims a cross-venue lead and a shippable feature without showing the cadence-matched re-measurement.

Suggested change: promote the cadence-matching test into a required gate for cross-venue features. Do not claim economic leadership or catch-up until the signal survives resampling/coarsening the foreign feed to the target venue cadence.


**reply: [PARTIALLY-VALID]**

The reviewer is right that the cadence-matched re-measurement is **prose only** — lines 1045-1050 describe "sample the foreign book only at byb update times, or coarsen it to byb's median inter-update gap," but no cell computes a cadence-matched IC, and the text sits in the §6 lifetime markdown *after* the §5 gates and *after* the worked-example verdict (lines 899-911) which already declares the feature "ship-able, for both heads, every exchange." Since byb/okx are the stale venues and the okx leg is shipped, the "foreign venue leads byb" reading genuinely isn't artifact-cleared for the worked example. Where the reviewer **understates the template**: the instruction to rule this out "before claiming a lead" *is* present and explicit, with the correct tell ("if the *stalest* venue shows the *largest* IC, suspect resolution") — so the methodology is documented, it's just not executed or wired into the pass/fail evidence. So it's a real gap between stated method and worked-example verdict, not a missing concept.

**Suggested fix:** Add a code cell after §6 that recomputes the cross-venue Gate B IC for the okx/byb leg with the foreign mid sampled only at byb update times (or coarsened to byb's median inter-update gap), and gate the verdict on it; until then, change the §10 verdict line to "ship-able for bin; okx leg pending the cadence-match check."

### H4. The gate thresholds are arbitrary but presented as hard statistical rules

Location: cells 2, 12, 17, 18, and 19.

Examples include IC floors around `0.01`, scale max/min `< ~3`, signed regime IC `< ~0.05`, magnitude regime IC `< ~0.1`, and dispersion `< ~0.1`. These may be useful project heuristics, but the notebook presents them as hard gates without calibration, confidence intervals, cost context, or multiple-testing adjustment.

Suggested change: label these as project heuristics calibrated on prior blocks, or add calibration evidence. Report uncertainty bands or block-bootstrap intervals around all IC and gate values. Avoid saying a feature "fails" solely from one threshold unless the threshold has a documented rationale.


**reply: [SUBJECTIVE]**

The thresholds the reviewer lists are real (IC floor ≳0.01, scale max/min < ~3, signed track < ~0.05, mag track < ~0.1, dispersion < ~0.1) and it's true there are no CIs, calibration curves, or multiple-testing adjustments. But the claim that they are "presented as hard statistical rules" overstates what the code says: every threshold carries a "~" approximation tilde ("want **< ~3**", "< ~0.05", "< ~0.1"), and the guard-rails open with "Hard rules, learned the hard way. Follow them unless you have a specific, written reason not to" (line 101) — i.e. they are explicitly framed as project heuristics with an escape hatch, not as derived statistical tests. "Hard gate" in this notebook means "pass/fail in the project's screen," not "statistically calibrated bound," and the reviewer reads a stronger claim than the text makes. Whether to additionally print uncertainty bands is a reasonable preference (and overlaps H5), but the framing complaint is largely a matter of taste, not a defect. No fix required — the tildes and "unless you have a written reason" already encode the heuristic nature.

### H5. No uncertainty estimates are reported despite huge autocorrelation and overlapping labels

Location: cells 8, 12, 13, 16, 18, 23, and 26.

The notebook reports ICs to three decimals, but anchors are spaced every 50 ms, labels cover 100 ms, and the feature EMAs have long memory. The effective sample size is much lower than the raw `1,706,369` anchors. Without block-bootstrap intervals, per-fold distributions, day-level variability, or Newey-West-style adjustments, it is hard to know which differences are real.

Suggested change: print per-fold ICs and confidence intervals using block bootstrap over time chunks or across blocks. For heat maps, show stability across folds/blocks, not just point estimates.


**reply: [VALID]**

Grounded and correct. The executed notebook reports `grid: 1,706,369` anchors on a 50 ms grid with 100 ms outcome windows, and the source itself acknowledges the dependence: "adjacent 100 ms outcome windows still overlap ~50%, so neighbouring samples are correlated" (line 419) and "the embargo does **not** fully decorrelate the slow EMA/yardstick memory (≈ YARDSTICK_N/trades-per-sec ≈ 140 s, longer than the ~100 s embargo)" (line 663). Despite that, a grep for `bootstrap|confidence|interval|per-fold` in the source returns nothing: `wf_ic` collapses the folds to a single `np.mean` (line 672) and the verdict quotes three decimals ("≈0.086", "0.066 / 0.085 / 0.116") with no spread, per-fold distribution, or block-bootstrap band. So the effective sample size is far below 1.7M and no uncertainty is reported — exactly the gap the reviewer names, made sharper by the fact that the notebook already knows the autocorrelation is there.

**Suggested fix:** Have `wf_ic`/`wf_ic_by_regime` also return the per-fold IC vector, print it (e.g. `folds: [...] mean±std`), and add a block-bootstrap CI over contiguous time chunks for the headline marginal IC; in the §6 heat-map cell, annotate cross-fold stability rather than only the in-sample point estimate.

### H6. The controls are narrower than the claims

Location: cells 12, 13, 17, and 18.

Gate B controls only byb rate and volatility momentum. That may be the intended base, but the notebook sometimes says the feature is proven not to be regime wearing a disguise. A cross-venue feature can also be entangled with foreign venue rate/volatility, spread state, feed staleness, or liquidity regime. Gate A checks only byb `vol_level` and `rate_level`, and only for one representative exchange.

Suggested change: either narrow the claim to "not explained by the two target-venue momentum controls" or add controls/checks for per-venue and global regime coordinates, spread/liquidity state, and feed freshness.


**reply: [PARTIALLY-VALID]** The reviewer is right that Gate B's controls are only the two byb target-venue invariant momenta (`base = [rate_momentum, vol_momentum]`, line 687) and Gate A's coordinates are only byb `vol_level`/`rate_level` (lines 872-873). But the reviewer overstates the notebook's claim: the source never says the feature is "proven not to be regime wearing a disguise." The actual prose is narrow and correctly scoped — line 704 says any IC "*on top of* these controls is genuinely new information, not the regime wearing a disguise," i.e. new *over these specific controls*, and Gate A is explicitly framed as invariance against "the regime diagnostics we track." So the defect is real only in that the notebook does not add foreign-venue/spread/liquidity controls, not that it overclaims generality. Note also the design rationale that foreign-venue *levels* are deliberately excluded as controls (lines 188-190: raw levels "are never controls") — but the reviewer's point about *foreign momenta / liquidity state* not being checked stands.

**Suggested fix:** In the §5 controls cell, add the foreign-venue invariant momenta (e.g. compute `rate_momentum`/`vol_momentum` analogues from okx/bin move streams) to `base`, or add a one-line caveat to the §5/§6 prose: "controls are the byb-venue invariant momenta only; entanglement with foreign-venue regime, spread/liquidity state, and feed freshness is not controlled here (foreign-feed freshness is treated separately in the §6.5 cadence check)."

### H7. "Raw vol/rate levels are never controls" is too absolute and partly contradictory

Location: cells 2, 12, 13, and 17.

The notebook says raw vol/rate levels are never controls because they are not valid features, but it also says a control can be a valid feature. In statistics and ML, regime levels can be valid covariates, conditioning variables, calibration inputs, or model heads even if they are not alpha features.

Suggested change: say "raw vol/rate levels are not used as alpha controls in this Gate B marginal test; they are used as regime coordinates in Gate A." If the project forbids them as model inputs, state that as a project design choice rather than a general rule.


**reply: [PARTIALLY-VALID]** The two sentences the reviewer flags are both present and do sit in tension on a careless read: line 188 "The raw vol/rate **levels are never controls** (they aren't valid features)" and line 190 "**a control can be a valid feature**." But this is not actually a contradiction — they refer to two different sets: raw *levels* (σ_ev, λ_ev) are not valid features and so are never Gate-B controls, whereas a *regime descriptor* feature (e.g. a vol/rate ratio under test) can be a valid feature and is judged standalone. The reviewer's substantive point is narrower and fair: the phrase "are never controls" is stated as a general truth when it is really this project's design choice for *this* Gate-B marginal test (the notebook even keeps levels as the Gate-A regime *coordinate*, line 184/621, so they are not categorically useless). So it is a scoping/wording defect, not a logical contradiction.

**Suggested fix:** Reword line 188 to scope the rule: replace "The raw vol/rate **levels are never controls** (they aren't valid features)" with "The raw vol/rate **levels are not used as Gate-B controls** — they aren't valid alpha features, so putting them in the marginal test just smuggles the Gate-A regime test back in; they serve only as the Gate-A regime *coordinate*."

### H8. "Regime independence" is overstated

Location: cells 2, 12, 17, 18, and 19.

Gate A checks scale stability, monotone association with two regime coordinates, and decile-mean dispersion. These are useful diagnostics, but they do not prove distributional independence from regime. They can miss tail changes, conditional dependence, interaction effects, and dependence on unobserved regimes.

Suggested change: replace "distribution must be independent of the regime" with "distribution should be stable against the regime diagnostics we track." If true independence is important, add distributional tests such as KS/energy distance by bucket, tail quantile checks, or mutual-information/HSIC-style diagnostics.


**reply: [PARTIALLY-VALID]** The reviewer correctly quotes the overclaim: line 174 ("its distribution must be **independent of the regime**") and line 787 ("is the feature's distribution **independent of the regime**"). Gate A in fact only tests three diagnostics — scale max/min across vol buckets, monotone `|IC(·, level)|`, and per-decile-mean dispersion (lines 874-892) — which, as the reviewer says, cannot establish full distributional independence (they miss conditional/tail/interaction dependence and unobserved regimes). However, the notebook is *not* naive about this: line 184 already self-corrects with "Never call a feature regime-invariant ... until **every** Gate A number says so" and the guard rails repeatedly frame these as the diagnostics "we track." So the word "independent" is loose prose, not a claimed theorem; the fix is a wording tightening, exactly as the reviewer suggests.

**Suggested fix:** Change "its distribution must be **independent of the regime**" (lines 174 and 787) to "its distribution must be **stable against the regime diagnostics we track** (scale, monotone tracking, decile-mean dispersion) — not a proof of independence."

### H9. The notebook relies on rank-IC, but the model is distributional

Location: cells 1, 12, 16, 18, 25, 26, 27, and 32.

Rank-IC is a useful feature-screening statistic, but the described model predicts distributions. The notebook does not evaluate proper scoring rules such as negative log likelihood, CRPS, Brier/log-loss for event occurrence, calibration curves, or incremental model loss. It also does not test trading utility after costs.

Suggested change: label rank-IC as a screening diagnostic. Before shipping, require model-level incremental loss/calibration and, if this is trading-facing, latency/cost-aware utility.


**reply: [PARTIALLY-VALID]** Factually correct: the model is described as distributional (the two heads predict `P(K=k)` and the family `D_k`, lines 73-78), yet every screening number is a Spearman rank-IC (`wf_ic`, `signal_ic`, the §6 heat-maps) and no proper scoring rule (NLL/CRPS/Brier), calibration, or cost-aware utility is computed. But the reviewer overstates it as a methodology gap for *this notebook*: the notebook never claims rank-IC measures distributional fit — it explicitly frames IC as a *screening/feature-hygiene* statistic, and §7 (lines 1063-1098) does inspect the actual forward-return and move-count *distributions* by feature bucket, which is a (qualitative) distributional check. Proper-scoring/model-loss evaluation is genuinely a model-training concern, arguably out of scope for a single-feature analysis template, but a one-line label would prevent the reader from treating rank-IC as a shipping-grade distributional verdict.

**Suggested fix:** Add to the §5 intro a sentence: "Rank-IC here is a **feature-screening** statistic, not a distributional score; proper scoring (NLL/CRPS, occurrence log-loss, calibration) and cost-aware utility are evaluated at the model level downstream, not in this per-feature template." Optionally cross-reference the multi-block harness in `tools/oss`.

### H10. The notebook says "done" too early

Location: cell 0 and cell 32.

Cell 0 says a feature is done when the oracle and hygiene gates pass. Cell 32 then lists additional shipping requirements: streaming builder, tests, gate results recorded, chosen spans documented, and data quirks handled. Cell 30 adds more pending checks.

Suggested change: separate "analysis done" from "shipping done." For example:

- Analysis candidate: hypothesis, definition, oracle, single-block gates.
- Research acceptance: multi-block OOS, stability, existing-feature increment, latency.
- Production acceptance: streaming builder, tests, monitoring, documented transforms.


**reply: [PARTIALLY-VALID]** The tension the reviewer cites is real: line 30 says "**A feature is 'done' when two checks pass**" (oracle + hygiene gates), while §10 (lines 1210-1216) lists a five-item "**To ship**" checklist and §8 (lines 1156-1159) lists "**Still to run**" items (additivity over existing features, day-to-day stability, OOS across a regime change). But the reviewer misreads the scope of "done": line 35-36 immediately qualifies it — "Everything **after that** decides *which part of the model* the feature feeds, and *at what time-scale*" — so "done" plainly means "done with the *analysis-correctness* stage," not "done for shipping," and the notebook keeps an explicit ship checklist. The word "done" is overloaded and the §10 verdict's "real and ship-able" (line 909) is genuinely stronger than the pending checklist supports, so a labelling split (analysis-done vs ship-done) is a fair, low-cost improvement.

**Suggested fix:** Reword line 30 to "A feature's **analysis** is done when two checks pass" and add one line under it: "That is *analysis-done*, not *ship-done* — the §10 checklist (streaming builder, tests, multi-block OOS, additivity over existing features, documented spans) gates shipping." This also softens the §10 "ship-able" verdict to "single-block analysis passed; ship checklist pending."

### H11. The rate-target normalization is dimensionally inconsistent in the prose

Location: cells 1, 15, 16, and 26.

`lambda_ev` is moves per second. `fwd_count / lambda_ev` has units of seconds. The code comment in cell 16 acknowledges the horizon constant drops out of rank correlation, but the prose describes the target as a dimensionless count measured against recent pace.

Suggested change: use `fwd_count / (lambda_ev * HORIZON_NS / 1e9)` in the notebook, or explicitly call the current value "count divided by moves-per-second; equivalent up to a constant for rank-IC only."


**reply: [PARTIALLY-VALID]**

The dimensional fact is correct: line 739 is `rate_target = fwd_count / np.maximum(lam_at_anchor, 1e-9)`, and with `lam` in moves/sec this has units of seconds. The reviewer is right that the §1 prose ("its target is `count ÷ λ_ev`") and §6 ("more or fewer moves than usual") read as dimensionless while the value is not. But the code is NOT silent about it — line 739's own comment already says "λ_ev is moves/sec over 100 ms, so this carries units of seconds; the constant horizon factor drops out of the rank correlation," and the §6 markdown at line 739-context restates it. So the defect is purely prose/code-comment dimensional sloppiness in the narrative cells, not an analysis error, and it is rank-IC-invariant. The reviewer slightly overstates by implying the code is misleading; only the heads-description prose is.

**Suggested fix:** In the §1 head description (build_feat_nb.py ~line 54), change "so its target is `count ÷ λ_ev`" to "`count ÷ (λ_ev · horizon)` — a dimensionless count-vs-expected-count (the constant horizon drops out of rank-IC, so the code omits it)", and mirror the same parenthetical in the §6 markdown so the prose matches the line-739 comment.

### H12. Input normalization is fit on the whole block

Location: cells 28, 29, and 30.

The input-shaping section computes mean/std, median/MAD, clipping diagnostics, and rank-Gaussian transforms on the full feature sample. That is fine for exploratory plots, but not for a causal evaluation or production transform. A rank-Gaussian map fitted on the full block is especially non-causal if reused for scoring.

Suggested change: state that all transforms must be fitted on training data only, then applied to validation/test/live. For walk-forward reporting, fit the normalizer inside each fold. For production, use a frozen training quantile map or a causal/rolling calibrator.


**reply: [VALID]**

The §8 cell fits every transform on the full feature sample: line 1119 `f = price_dislocation(...); f = f[np.isfinite(f)]` then line 1120-1124 compute `med`, `mad`, `f.mean()`, `f.std()`, and `norm.ppf((rankdata(f)-0.5)/len(f))` over the whole block. The rank-Gaussian map in particular is a full-block fit, which is non-causal if reused for scoring. The reviewer correctly scopes this as fine for exploratory plots but a hazard in a reusable template, and the §8 conclusion (lines 1147-1154) recommends "robust z-score followed by a clip" without any train-only caveat. This is a genuine methodology gap the template should close, and unlike the §5 gates there is no walk-forward framing here.

**Suggested fix:** Add a sentence to the §8 markdown conclusion (build_feat_nb.py ~line 1154): "These statistics (mean/std, median/MAD, the rank→Gaussian map) are fit on the whole block for the plot only. In production and for any walk-forward score, fit the normalizer on training rows alone (or inside each fold) and apply it forward — a full-block quantile map is non-causal." Optionally add a code comment at line 1120 marking the fit as illustrative-only.

### H13. The "keep all exchanges" recommendation is plausible but not demonstrated

Location: cells 2, 3, 15, 16, 19, 31, and 32.

The notebook argues that leadership rotates, so every exchange should be kept. The actual evidence shown is per-exchange IC on one block plus a joint marginal score where bin appears to capture the shared signal. There is no rolling leadership analysis, no feature importance over time, no collinearity/incremental contribution against existing cross-venue features, and section 9's table is explicitly illustrative.

Suggested change: keep the guard rail as a prior, but report rolling or block-level evidence: per-period winner, incremental IC from adding each venue, correlation among venue legs, and whether the extra venue improves model loss out of sample.


**reply: [PARTIALLY-VALID]**

The reviewer is right that the "keep all exchanges" recommendation rests on a single block plus a §9 table the notebook itself flags as illustrative (line 1171: "NOT computed for `price_dislocation`"). The §6 evidence is per-exchange in-sample IC heat-maps plus one joint marginal where the conclusion (lines 900-903) explicitly admits "jointly bin already captures the shared signal so the joint marginal = bin-alone ≈ 0.086" and keeps okx only because "leadership rotates and its standalone gain is positive" — i.e. a prior, not rolling evidence. So the demonstrated-vs-asserted gap is real. However, the reviewer understates that the template already treats keep-all as a guard-rail/prior and that §6 (lines 723-729) gives a mechanistic time-scale argument, and a single-block template is the stated scope with `tools/oss` doing multi-block OOS. So it is partially valid: the prose overstates for one block, but the recommendation is correctly framed as a default prior, not a proven result.

**Suggested fix:** In the §6 markdown (build_feat_nb.py ~line 728) or the §10 verdict, add: "On this single block bin captures the shared signal and okx's joint marginal is ~0; keeping okx is a prior (leadership rotation), confirmed only by the rolling per-period winner / incremental-IC / venue-leg correlation analysis in the multi-block harness (`tools/oss`), not by this block."

### H14. The live-front/trade-clock convention is valid project methodology, but it is overgeneralized as universal best practice

Location: cells 1, 2, 4, 5, and 7.

Event-time EMAs are standard and well understood. However, the notebook repeatedly says "every average" must be trade-clock and "never wall-clock or hard window." That is a project convention, not a universal ML/statistics best practice. Wall-clock windows are standard for latency, execution, realized volatility, monitoring, and fixed-horizon prediction problems. The repository's own raw feature spec includes wall-clock features.

Suggested change: scope the rule: "For this feature family and model design, smoothers use the shared trade-clock EMA unless a feature has a written reason to use another clock."


**reply: [PARTIALLY-VALID]**

The reviewer correctly identifies absolute universal phrasing: line 66 "like every average here, they live on the trade-tick clock — never wall-clock or a hard window"; line 124 "Do make every average a trade-tick EMA. *Every* smoother in the pipeline"; line 227 "Every smoother here is an EMA on the trade clock." These ARE stated as blanket rules. The claim that "the repository's own raw feature spec includes wall-clock features" is plausible per CLAUDE.md (it mentions a 1 ms grid and the trade-clock bug was one specific family), but I did not open docs/raw_features.md to confirm, so I won't certify that half. The valid core: the rule is correct *for this feature family and model*, and most of the cited text already scopes it via "unless you have a specific, written reason not to" (line 101) and "unless a feature has a written reason to use another clock" is the spirit. So it is partly a project-convention-stated-as-universal (a real wording defect) and partly already hedged — PARTIALLY-VALID, and largely a wording matter.

**Suggested fix:** Soften the two strongest blanket lines. At build_feat_nb.py line 66 and line 124, change "every average / *Every* smoother ... never wall-clock" to "every smoother *in this feature family* is a trade-clock EMA unless a feature has a written reason to use another clock (e.g. a latency or realized-vol measure that is intrinsically wall-clock)."

### H15. The causal forward-fill helper can use a future first value before a venue has quoted

Location: cell 6 and cell 8.

`mid_on_clock()` uses `np.clip(searchsorted(...) - 1, 0, len(mid) - 1)`, which returns the first mid even if the clock tick precedes the first quote for that venue. That is a future fill. Warm-up likely hides it in the saved run, but the template should not contain a causal helper with this edge case.

Suggested change: return `nan` when `searchsorted(...) == 0`, then mask invalid anchors until every venue has a prior mid. Apply the same rule to arbitrary-time `_mid_at` helpers where needed.


**reply: [VALID]**

The helpers do exactly what the reviewer describes. `mid_on_clock` (line 371) returns `mid[np.clip(np.searchsorted(rx, merged_ts, 'right') - 1, 0, len(mid) - 1)]`, and `_mid_at` (line 439) and the §-lifetime `_mid_at` (line 993) use the same `np.clip(..., 0, ...)` pattern. When the clock tick precedes a venue's first quote, `searchsorted - 1 == -1` is clamped to `0`, returning that venue's *first* mid — a future value. The reviewer is candid that WARMUP (anchors start at `merged_ts[50000]`, line 426) almost certainly masks it in the saved run, since all venues have quoted within the first 50k ticks. But `gap_committed` and `log_mid_byb` are still built over *all* ticks from index 0 (lines 372, 437), so the EMA legs do ingest the future-filled early values pre-warmup; only the read is protected, not the state. For a template whose central discipline is "don't peek ahead," shipping a causal helper with a silent future-fill edge case is a legitimate defect.

**Suggested fix:** In `mid_on_clock` (build_feat_nb.py line 369-371) and both `_mid_at` helpers (lines 438-439, 992-993), replace the clamped index with a nan-on-underflow read, e.g.: `idx = np.searchsorted(rx, t, 'right') - 1; out = np.where(idx < 0, np.nan, mid[np.clip(idx, 0, len(mid)-1)])`, and add a comment that anchors before every venue has quoted are invalid and masked (the finite-mask in §4/§5 already drops nan rows).

## Medium Findings

### M1. The phrase "predicts byb's mid-price 100 ms from now" is too imprecise

Location: cells 0, 1, 3, 8, and 25.

The code predicts a log return over `[t, t + 100 ms]`, normalized by a per-move volatility yardstick, and separately predicts a count/intensity target. That is not the same as directly predicting the future mid-price level.

Suggested change: use "predicts byb's 100 ms forward mid-price return and move-count intensity" when referring to the diagnostics.


**reply: [PARTIALLY-VALID]**

The reviewer is right that the literal prose is imprecise in a couple of places. The `predicts` row of the §1 table says `byb's mid-price 100 ms from now` (build_feat_nb.py line 204), and §0 says the model `forecasts one exchange's mid-price about 100 ms ahead` (line 16). The actual diagnostic target is `target = fwd_return / sigma_at_anchor` (line 435) — a sigma-normalized log return — and a separate count target. So `mid-price level` is loose. However, the reviewer overstates by implying the notebook conflates this throughout: §1's `what` row already says the model predicts *moves* and the §model section (lines 49-55) defines both heads in return/count terms precisely. This is a wording tightening, not a methodological defect.

**Suggested fix:** In build_feat_nb.py change the §1 table `predicts` cell (line 204) from `byb's mid-price 100 ms from now` to `byb's 100 ms forward mid-price return (in sigma-units) and its move-count intensity`, and reword line 16 to `forecasts one exchange's mid-price *return* about 100 ms ahead`.

### M2. The "how many times how big" explanation is useful but too simplified

Location: cell 1.

The total return is a sum of event returns, not literally a product of "how many" and "how big." The product intuition is fine for scale, but the distributional mixture needs to handle sign, count dependence, and feature interactions.

Suggested change: phrase it as "a fixed-horizon return is the sum of a random number of event returns; count controls much of the scale, while the event-return distribution controls direction and shape."


**reply: [PARTIALLY-VALID]**

The reviewer is technically correct: a fixed-horizon log return is the *sum* of per-event log returns, not literally a product of a count and a size. The §model prose does lean on a product framing — `A move over a window is just *how many* little moves happen times *how big* each one is` (line 69) and `how many x how big` (line 84). But the notebook is not actually wrong about the math it implements: the formal statement right below it is the *mixture* `distribution of the 100 ms move = Sigma_k P(K=k).D_k` (line 78), which is the correct subordination form and explicitly handles count-conditioned distributions and sign via `D_k`. So the `times` phrasing is a deliberate scale-intuition gloss, with the rigorous version adjacent. Minor polish, not a defect.

**Suggested fix:** Soften line 69 to `A move over a window is the *sum* of however many little moves happen, each of its own size — so *how many* (rate) mostly sets the scale and *how big/which way* (price) sets the shape`, keeping the existing mixture formula as the precise statement.

### M3. The subordination citations support the broad decomposition, not every implementation choice

Location: cell 1.

Clark and Ane-Geman support event-time/subordination ideas. They do not directly validate the specific `sigma_ev`, `lambda_ev`, live-front EMA, cross-venue gap, or 100 ms crypto microstructure implementation.

Suggested change: keep the citations but say they motivate the decomposition. Do not imply they validate the whole method.


**reply: [PARTIALLY-VALID]**

The claim is fair on its face. Lines 86-89 cite Clark (1973) and Ane & Geman (2000) for the subordination decomposition, and the prose says they make the split `a principled decomposition - not just a convenient one`. Those papers genuinely support event-time subordination but not the specific `sigma_ev` RMS-per-move estimator, the `lambda_ev` two-EMA rate, the live-front EMA, or the 100 ms cross-venue implementation. That said, the notebook does *not* actually claim the citations validate those implementation choices — it scopes them to `this how many x how big split is the classic subordination model` (line 85), i.e. the decomposition only. So the reviewer is correcting an overreach that the text mostly already avoids; the risk is a reader inferring more. A one-clause clarification suffices.

**Suggested fix:** Append to line 89 a clause: `(The citations motivate the count/size decomposition only; the specific sigma_ev / lambda_ev estimators, the live-front EMA, and the cross-venue gap are this notebook's own implementation choices, validated by the oracle and gates below, not by Clark/Ane-Geman.)`

### M4. Gate B's "marginal IC" is not the same as a partial rank correlation

Location: cells 13 and 18.

`wf_ic(base + features) - wf_ic(base)` fits least-squares predictions and subtracts two Spearman correlations. That is a useful incremental predictive-score heuristic, but it is not a standard partial Spearman correlation or a direct estimate of unique association. It can behave oddly when the base score is negative, when predictions are nonlinear, or when feature scaling changes.

Suggested change: name it "incremental walk-forward IC of linear predictions." Optionally add partial Spearman, residualized rank IC, or model-loss delta as companion metrics.


**reply: [VALID]**

Confirmed against the code. `signal_ic`'s alpha branch returns `wf_ic(base + leg_feats, tgt) - wf_ic(base, tgt)` (line 848), and `wf_ic` is `mean over folds of spearmanr(p[t], y[t])` where `p` is an OLS prediction `X @ coef` (lines 668-672). So Gate B's headline number is indeed a *difference of two Spearman correlations of least-squares predictions*, not a partial Spearman / residualized rank-IC. The reviewer's caveats are real: this can behave oddly when `wf_ic(base)` is negative or when scaling shifts the lstsq fit. The notebook even labels it loosely as `marginal rank-IC` (line 804) and `MARGINAL value over the controls` (line 681), which invites the partial-correlation reading. Naming it precisely costs nothing and prevents misinterpretation.

**Suggested fix:** In build_feat_nb.py rename in the prose/comments — e.g. line 671 comment to `# mean OOS rank-IC of the LINEAR PREDICTION (not a partial correlation)` and line 804 to `incremental walk-forward IC of linear predictions over the controls`. Optionally add a residualized-rank-IC companion, but the rename is the load-bearing fix.

### M5. The notebook does not check additivity over existing features

Location: cells 30 and 32.

Cell 30 says this is still to run, but cell 32 makes a keep/ship recommendation. A feature can pass all single-feature gates and still be redundant with existing features.

Suggested change: add an "existing feature set" gate before the verdict, or make the verdict explicitly conditional on that pending check.


**reply: [PARTIALLY-VALID]**

The factual observation holds: §8's `Still to run` list includes `whether the feature adds over features we already have` (line 1156), yet §10 opens `Keep it - feed the *signed* feature to both heads` (line 1195) and §6 concludes `real and ship-able` (line 909). So a keep/ship verdict is rendered before the existing-feature additivity check. But the reviewer overstates the contradiction: the verdict is already hedged as a *single-block* result with an explicit pending list, and the orchestration context notes a multi-block OOS harness lives in `tools/oss`, so additivity-over-existing-features is arguably the next harness's job, not this template's. This overlaps heavily with C3 (single-block overclaim). The fix is to make the verdict explicitly conditional, not to add a new gate to a single-block template.

**Suggested fix:** In build_feat_nb.py §10 (line 1195) preface the verdict with `**Single-block candidate — conditional on the §8 pending checks (additivity over existing features, day-to-day stability, regime-gap OOS).**` so the keep recommendation is not stated as unconditional.

### M6. The latency section is valuable but disconnected from the shipping decision

Location: cells 22, 23, 24, and 32.

The lifetime curve shows the price IC drops below half by around 100 ms in the saved output. The notebook says this is a latency budget, not a pass/fail. But the final verdict does not state the latency budget as a requirement or connect it to system latency, feed latency, decision latency, and execution latency.

Suggested change: include the measured lifetime in the final verdict: "usable only if observe-to-act latency is below X ms, after feed-cadence artifact checks."


**reply: [PARTIALLY-VALID]**

The numbers check out: the saved cell prints `price forward IC by delta(ms): 0:+0.103 ... 50:+0.063 100:+0.031` and `drops below half by delta~=100 ms`. The notebook frames this as `a latency budget, not a pass/fail` (line 1026) and says the verdict should read `predicts ~X ms ahead, needs latency < X` (line 1028). The reviewer is right that the §10 verdict (lines 1195-1216) never actually states a measured latency budget or the ~100 ms half-life — the connection the notebook itself prescribes is dropped at the verdict. But the reviewer's framing that the lifetime is `disconnected` overstates: it is a deliberate non-gate by design, and an explicit shipping requirement (`needs latency < X`) is just not carried forward. Minor, and a clean fix.

**Suggested fix:** Add a bullet to the §10 `Keep it` block (after line 1208): `- **Latency budget:** the price edge's forward IC halves by delta~=100 ms (saved run), so this is usable only if observe-to-act latency stays well inside that, after the cross-venue feed-cadence check (§lifetime).`

### M7. The distribution-shape conclusion has no printed numeric support

Location: cells 25, 26, and 27.

Cell 26 produces plots but no table of bucket means, quantiles, counts, or monotonicity statistics. Cell 27 states monotonic distribution shifts. That may be visually true, but it is hard to audit in a text review or saved notebook output.

Suggested change: print a compact table with bucket sample counts, forward return mean/median/quantiles, `E[K]`, `P(K >= 1)`, and monotonicity checks.


**reply: [VALID]**

Confirmed by inspecting the executed notebook: the §7 distribution-shape code cell (cell 26) produces only a `display_data` output (the matplotlib figure) and no stream/text output — there is no printed table of bucket means, counts, `E[K]`, `P(K>=1)`, or a monotonicity statistic. Yet the §7 conclusion asserts `both the mean move-count E[K] and P(K>=1) climb monotonically from the smallest to the largest |dislocation| decile` (lines 1095-1096). That monotonicity claim is visual-only and unauditable from saved output. The plot does compute the per-decile `E[K]` and `P(K>=1)` (lines 1083-1084), so emitting them as a table is cheap and makes the conclusion verifiable in a text review.

**Suggested fix:** In the §7 code cell (around line 1086) add a print of the per-decile arrays, e.g. after the plot: `print('|disloc| decile -> E[K], P(K>=1), n:'); [print(f'  {b}: E[K]={fwd_count[dec==b].mean():.3f}  P(K>=1)={(fwd_count[dec==b]>=1).mean():.3f}  n={int((dec==b).sum())}') for b in range(10)]`, plus a one-line monotonicity check `print('E[K] monotone increasing:', bool(np.all(np.diff([fwd_count[dec==b].mean() for b in range(10)])>=0)))`.

### M8. The transform recommendation uses a stale numeric value

Location: cells 29 and 30.

The saved output says z-score `max|.| = 27.8`, but the markdown conclusion says the z-score leaves a `~21sigma` spike and `max|.| = 21.3`.

Suggested change: update the markdown to match the saved output, or avoid hard-coded values by saying "the saved output shows a very large z-score spike."


**reply: [VALID]**

Directly confirmed against saved output. The executed §8 cell prints `z-score  excess_kurt=1.4  max|.|=27.8`, but the §8 conclusion markdown says `it still leaves a ~=21sigma spike (max|.| = 21.3)` (line 1149). The hard-coded `21.3` is stale and contradicts the actual saved `27.8` — exactly the kind of drift the reviewer flagged. The downstream `robust + clip +-4 -> max|.| = 4.0` claim does match the output, so only the z-score value is wrong. This is a clear factual error in a worked-example conclusion.

**Suggested fix:** In build_feat_nb.py line 1149 change `a ~=21sigma spike (max|.| = 21.3)` to `a ~=28sigma spike (max|.| = 27.8)` to match the saved output — or, to avoid future drift, reword to `a very large z-score spike (see the printed max|.|)` and drop the hard-coded number.

### M9. The clipping recommendation should distinguish training-time clipping from analysis clipping

Location: cell 30.

"You clip whenever you feed a network" is too broad. Clipping can help stability, but the threshold should be selected on training data and checked for information loss. Some architectures may prefer winsorization, robust scaling, learned normalization, or rank/quantile transforms.

Suggested change: say "for this saved run, robust z-score plus clipping at +/-4 is the lightest candidate shown that controls outliers; fit and validate the transform on training data."


**reply: [PARTIALLY-VALID]**

The §8 conclusion does say "so you clip whenever you feed a network" (line 1153), an over-broad absolute, and it never states the clip threshold must be fit/validated on training data only — a real gap that H12 also flags. So the substance of M9 holds. But the reviewer's suggested replacement quotes `max|·| = 21.3`; the *saved* output (cell 29) actually prints `max|·|=27.8` for the z-score, so the reviewer is propagating the same stale number M8 catches. The fix is the wording + a causal-fit caveat, not the numeric they propose.

**Suggested fix:** In the §8 conclusion md, change "so you clip whenever you feed a network" to "clip thresholds are fit and validated on training data only (never the whole block), and the architecture may instead prefer winsorisation, robust scaling, or a learned/rank transform." Use "a very large σ spike" rather than a hard-coded multiple.

### M10. Same-timestamp grouping is good, but the explanation is repeated too often

Location: cells 2, 4, 6, 9, and 10.

The rule is important, but the notebook repeats it with long explanations in several sections. This makes the template harder to adapt and obscures the feature-specific method.

Suggested change: define same-timestamp grouping once in the guard rails, then refer back to it briefly in code comments.


**reply: [PARTIALLY-VALID]**

The same-timestamp "one event, not a sequence" rule genuinely is restated at length in multiple places: the guard rails (lines 137-145, ~9 lines with the σ_ev=0.13 anecdote), the "Choosing the EMA" section (lines 271-273), §2 (line 285), the §2 code comment (line 381), and the §4 builder docstring (lines 511-512, 582). That is real repetition. But it is partly intentional doctrine-reinforcement for a rule the project treats as load-bearing (it caused a real bug), so calling it a defect is partly a preference. The reviewer's own M11/L1 cover the broader "too verbose" theme, so this is a narrower duplicate. Worth a trim, not a rewrite.

**Suggested fix:** Keep the full explanation once in the guard rails (lines 137-145); in the "Choosing the EMA" section and §2 prose, collapse to a one-line back-reference ("records sharing a timestamp are one event — see guard rails") and keep code comments to the minimal `collapse same-timestamp` note.

### M11. The notebook mixes template instructions with worked-example conclusions

Location: throughout, especially cells 0, 3, 19, 24, 30, 31, and 32.

The file is both a reusable template and a `price_dislocation` report. Some cells are generic method, some are example-specific, and some are illustrative placeholders for other feature families. That makes it easy for someone copying the notebook to keep stale conclusions or irrelevant guidance.

Suggested change: mark cells as one of: "template instruction", "example-specific result", or "illustrative only." Put all example-specific conclusions in clearly replaceable cells.


**reply: [VALID]**

The file is unambiguously both a reusable template and a `price_dislocation` report, and the two are interleaved without per-cell role tags. Concrete evidence: the §10 verdict hard-codes example-specific picks ("fast=10, slow=500", "n_fast=1 … n_slow=100", lines 1199-1201); §9's table (lines 1178-1182) is explicitly "illustrative … NOT computed for price_dislocation" yet sits adjacent to the verdict; and the top md (lines 21-24) tells the reader to "copy this notebook and change the parts specific to the feature." A copier could easily retain stale example conclusions. The reviewer is right that nothing structurally marks which cells are generic vs example-specific.

**Suggested fix:** Prefix each md/code cell's first line with an explicit tag — `**[TEMPLATE]**`, `**[EXAMPLE: price_dislocation]**`, or `**[ILLUSTRATIVE]**` — and move §9's illustrative table and the §10 worked picks into clearly tagged example blocks so the generic method reads cleanly when the example is swapped out.

### M12. The hard "do not normalize reflexively" rule is good, but the examples need clearer decision criteria

Location: cells 2, 5, 12, and 17.

The notebook says to divide out volatility/rate only when needed, but it does not give a crisp procedure for deciding when normalization is needed beyond Gate A after the fact.

Suggested change: add a decision table: raw feature, candidate normalizers, expected units, Gate A result, chosen form. This would prevent reflexive normalization while keeping the process reproducible.


**reply: [PARTIALLY-VALID]**

The notebook does state a "don't normalize reflexively" rule (guard rails line 110; §2 lines 290-293) and the principle "divide out vol/rate only when the feature carries it," but the only concrete *decision procedure* offered is Gate A run after the fact — exactly the reviewer's complaint. There is no crisp ex-ante "is this already a ratio / does it need σ_ev" checklist; the §2 prose explains the choice for this one feature but not a reusable rule. So the gap is real, though milder than stated since Gate A does give an objective post-hoc test. A small decision table would make the template reproducible.

**Suggested fix:** Add a short decision table in §2 (or the guard rails) with columns `raw feature | already comparable? (ratio/bounded) | candidate normaliser (σ_ev/λ_ev/none) | expected units | Gate A result | chosen form`, and state: normalise only if the raw form fails Gate A scale/track, and then only with the model yardstick.

### M13. The use of private `io._trade_lifts_ask` in the notebook is brittle

Location: cell 10.

The notebook calls a private helper from `boba.io`. For exploratory notebooks this may be acceptable, but a reusable template should avoid relying on private APIs unless the dependency is intentional and stable.

Suggested change: expose a public helper or include a small local, documented conversion for the oracle.


**reply: [VALID]**

Verified: §4's stream-gathering cell calls `io._trade_lifts_ask(f"{ex}_{COIN}", td["aggressor"].to_numpy())` (build_feat_nb.py line 562), and that function is private — `src/boba/io.py:225 def _trade_lifts_ask(...)` carries a leading underscore and is not exported. A reusable template depending on a private, underscore-prefixed helper is brittle exactly as the reviewer says; if `io` refactors its internal aggressor mapping the template breaks silently. The dependency is intentional (the io docstring at line 55 even points users at it) but "intended internal use" is not the same as a stable public surface.

**Suggested fix:** Promote a thin public wrapper in `boba.io` (e.g. `trade_lifts_ask = _trade_lifts_ask` or a documented `def trade_lifts_ask(...)`) and call that from the notebook, or inline a small documented venue→side conversion in the oracle cell so the template owns its own stable helper.

### M14. The "alpha" versus "control" branch is conceptually dense and easy to misuse

Location: cells 13, 17, 18, and 21.

`FEATURE_KIND`, `STRAT_VAR`, `own`, `tgt`, and the control-ratio decoupling logic are powerful but fragile. The notebook uses global state and comments like `BLOCKER-2`, `ISSUE-5`, and `LOW-1`, which look like patch history rather than stable template guidance.

Suggested change: refactor the gate logic into explicit functions or a small config object:

- `feature_role = "alpha" | "regime_descriptor"`;
- `target_kind = "price" | "rate"`;
- `shared_denominator = sigma_ev | lambda_ev | None`.

Then remove patch-history comments.


**reply: [PARTIALLY-VALID]**

The alpha/control machinery does lean on module-global state (`FEATURE_KIND`, `STRAT_VAR`, `base`) that `signal_ic` reads implicitly (lines 686-695, 840-848), and the self-test has to mutate globals via `globals().update(...)` in a try/finally (lines 952-958) — that is genuinely fragile and easy to misuse, supporting the reviewer. The patch-history comments are real: `HIGH-2` (688), `BLOCKER-2` (840), `BLOCKER-3` (836,838), `ISSUE-5` (831), `LOW-1` (843), `N1/ISSUE-4` (878). However the underlying *concept* (alpha→marginal vs control→standalone-stratified) is internally certified and sound, so the defect is presentation/structure, not logic; "conceptually dense" alone is partly subjective.

**Suggested fix:** Replace the patch-history tags (`HIGH-2`, `BLOCKER-2/3`, `ISSUE-5`, `LOW-1`, `N1/ISSUE-4`) with stable descriptive comments, and pass the role/target/yardstick into `signal_ic` as explicit arguments (`signal_ic(feats, role="alpha"|"regime_descriptor", tgt=..., strat_var=...)`) instead of reading module globals, so the self-test no longer needs `globals().update`.

### M15. The walk-forward fold implementation should report skipped folds and fold sizes

Location: cell 13.

`wf_folds()` silently skips folds with too few train/test rows. In this saved run that likely does not matter, but as a template it can hide broken inputs.

Suggested change: print train/test counts, embargo duration in seconds, number of folds scored, and per-fold ICs.


**reply: [VALID]**

Confirmed against the code: `wf_folds` does `if train.sum() < 100 or test.sum() < 100: continue` (line 665) — silently dropping folds — and neither `wf_folds`, `wf_ic`, nor the cell that calls them prints fold counts, train/test sizes, embargo seconds, or per-fold ICs. The saved output of that cell is a single line, `control-only predictive power (walk-forward): momenta 0.003`, with no fold accounting. As a *template* this can hide a degenerate input (e.g. a thin block where most folds are skipped) behind a plausible mean. The reviewer's diagnosis is accurate.

**Suggested fix:** Have `wf_folds`/`wf_ic` surface diagnostics — return or print `k`, number of folds actually scored vs skipped, min train/test row counts, embargo in seconds (`embargo / trades_per_sec`), and the per-fold IC list — and print them once in the §5 control cell so a starved walk-forward is visible.

### M16. The output precision is too high relative to the evidence

Location: cells 16, 18, 19, 21, 23, and 30.

Many conclusions quote three decimal places. Given autocorrelation, single-block selection, and no confidence intervals, three decimals can imply unjustified precision.

Suggested change: keep tables at three decimals if useful, but prose should use ranges or rounded values unless uncertainty is reported.


**reply: [PARTIALLY-VALID]**

The prose does quote 3-decimal point estimates with no uncertainty: the §6 conclusion cites "≈ 0.086", "≈ 0.048", mean-track "0.00 / 0.01", dispersion "≈ 0.02", "0.066 / 0.085 / 0.116" (lines 900-908), all from one 24h block with heavy label overlap and no CIs — so H5's concern that 3 decimals overstates precision in *prose* is fair, and M16 is its presentation echo. But the criticism is partly subjective: 3-decimal *tables* are reasonable for an IC screen, and the gate thresholds (0.01, 0.05, 0.1) are coarse enough that the displayed precision doesn't drive any decision. The real fix is uncertainty (H5), of which M16 is a weaker restatement.

**Suggested fix:** Keep the gate *table* at 3 decimals but round prose conclusions to 2 (or use ranges) and add a one-line caveat that these are single-block point estimates without confidence intervals — deferring true uncertainty bands to the multi-block `tools/oss` harness.

## Low Findings

### L1. The notebook is overly verbose for a template

Location: throughout.

The major ideas are repeated many times: two heads, yardsticks, trade clock, live front, same-timestamp events, no fixed leader, signed feature to both heads, and regime controls. The repetition makes it harder to see what must be changed for a new feature.

Suggested change: move stable doctrine into a short preamble or linked method note. Keep the notebook focused on the feature-specific hypothesis, definition, implementation, checks, and verdict.


**reply: [SUBJECTIVE]**

This is a matter of preference, not a defect. The notebook is dual-purpose by explicit design — cell 1 (lines 10-37) states "This notebook is two things at once: a method ... and a worked example," and the doctrine repetition (trade clock, live front, signed-to-both-heads, same-timestamp = one event) is deliberate: the guard rails (lines 99-191) say "Hard rules, learned the hard way," and the §4 oracle, §2 definition, and §5 gates each restate the rule they enforce on purpose so a copier cannot drop it silently. The CLAUDE.md validation doctrine actively wants the oracle/causal/timestamp rules repeated at each enforcement point. "Overly verbose" is the reviewer's taste; the author chose redundancy as a safety property. Nothing here is wrong, leaks, or misleads, so there is no defect to fix — only a verbosity-vs-safety trade-off the author has already chosen the other side of.

### L2. The tone is often too absolute

Location: cells 0, 2, 4, 9, 17, 22, 24, and 32.

Examples: "hard rules," "non-negotiable," "never," "wrong," "one true non-signal," "the plot makes the choice for you," and "ship-grade." Some absolutes are justified for causality; many are project preferences or heuristics.

Suggested change: reserve absolute language for causality, timestamp grouping, and no-lookahead rules. Use measured language for modeling choices and heuristics.


**reply: [SUBJECTIVE]**

The reviewer's own suggested fix concedes the point is preference-shaped: it says reserve absolutes "for causality, timestamp grouping, and no-lookahead rules" — and that is essentially what the notebook already does. The strongest absolutes sit exactly on those load-bearing rules: "Don't peek ahead" (line 117), "Non-negotiable" heads the oracle (line 456) and the same-timestamp rule (lines 141-145), and "never wall-clock or a hard window" is the trade-clock causality invariant the whole method rests on. "ship-grade" (line 633) is qualified in-line as "strictly past→future, as it would run live," and "the plot makes the choice for you" (line 1113) is a teaching flourish, not a claim of proof. The few looser absolutes are stylistic emphasis, not methodological overclaim. No concrete defect; this is house style.

### L3. The file uses inconsistent terms for the same idea

Location: throughout.

Examples include "time-scale" and "timescale"; "vol/rate" and "volatility/rate"; "hygiene gates," "Gate A/Gate B," and "companion"; "yardstick," "regime coordinate," and "normalizer." Most are understandable, but the mix increases cognitive load.

Suggested change: define a small glossary and standardize on one term per concept.


**reply: [PARTIALLY-VALID]**

The cited synonym pairs are mostly real but mostly harmless, and one is plain wrong. "yardstick" / "regime coordinate" / "normalizer" are *not* loose synonyms — the notebook deliberately distinguishes them: lines 620-621 make "the raw vol/rate **levels** are the regime **coordinate** for Gate A, never controls," while the yardstick (`σ_ev`, `λ_ev`) is the divisor the target shares units with. Conflating those would be a real bug; keeping them distinct is correct. The genuinely loose pairs are cosmetic: "vol/rate" vs "volatility/rate" appear interchangeably (e.g. line 114 vs line 215), and "time-scale" is used throughout while "timescale" — the reviewer's claimed variant — does **not** actually appear in the source, so that example is fabricated. So a light terminology pass is reasonable, but the reviewer overstated it and miscited at least one pair.

**Suggested fix:** in `build_feat_nb.py`, standardize the one genuinely-mixed pair — pick "vol/rate" everywhere in prose (or expand it everywhere) — and leave yardstick/coordinate/normalizer distinct as they carry different meanings. Do not introduce a "timescale" spelling; "time-scale" is already consistent.

### L4. Section numbers and references are dense

Location: cells 0, 2, 5, 7, 9, 12, 15, 17, 24, and 31.

The prose often points to multiple sections in parentheses. This is helpful in a finished report but noisy in a template.

Suggested change: use fewer inline cross-references and add a top-level checklist/table of sections instead.


**reply: [SUBJECTIVE]**

This is presentation taste. The inline cross-references ("§9 covers...", "see the Gate A / Gate B box in §6", "the §4 oracle re-checks here") are wayfinding for a dense, tightly-coupled method where the yardsticks defined in §2 are consumed in §5/§6 and the oracle in §4 enforces the §2 rules — the forward/back references are load-bearing for a reader, not noise. The reviewer's own framing ("helpful in a finished report but noisy in a template") admits it is a finished-report-vs-template preference, not an error. A top-level section index would be a nice-to-have, but nothing here is incorrect or misleading, so there is no defect to fix.

### L5. Some comments are longer than the code they explain

Location: cells 6, 8, 10, 13, 16, 18, 21, 23, and 26.

The comments contain important reasoning, but they also repeat markdown and make code cells difficult to scan.

Suggested change: move methodological explanation to markdown and keep code comments focused on non-obvious implementation details.


**reply: [SUBJECTIVE]**

"Comments longer than the code" is true in places (e.g. the `STRAT_VAR` comment at lines 688-695 dwarfs its one assignment) but that is by design, not a defect: the §4 doctrine in CLAUDE.md and the guard rails treat the *why* of each non-obvious choice as the thing most likely to be silently broken, so the dense comments encode the reasoning that keeps a copier from making a wrong-but-plausible change (e.g. why a signed alpha sets `STRAT_VAR=None`). The reviewer frames the fix as "move methodology to markdown" — but the markdown already carries the methodology; the inline comments pin it to the exact line it governs, which is the safer place for it given this notebook's whole thesis is that wrong choices fail silently. This is a readability-vs-safety preference the author has resolved deliberately. No correctness issue, so no fix.

### L6. Patch-history labels should be removed from template comments

Location: cells 13, 18, and 21.

Comments such as `HIGH-2`, `BLOCKER-2`, `BLOCKER-3`, `ISSUE-5`, `LOW-1`, and `N1/ISSUE-4` read like unresolved defect references. They distract from the reusable method.

Suggested change: replace them with stable descriptions or remove them after the fix is incorporated.


**reply: [VALID]**

The patch-history labels are really there and really read like unresolved defect IDs. Confirmed in source: `HIGH-2` (lines 688, 827), `BLOCKER-2` (lines 840, 952), `BLOCKER-3` (lines 836, 838), `ISSUE-5` (lines 831, 954), `LOW-1` (lines 843, 881), and `N1/ISSUE-4` (line 878). These are review-round artifacts (the template just went through 5 rounds of adversarial review), and in a reusable template they read as open tickets rather than stable guidance — a genuine maintainability wart, distinct from the substantive comment text they prefix. The reviewer is correct and specific here.

**Suggested fix:** in `build_feat_nb.py`, strip the bare tags and keep only the stable description, e.g. `# ISSUE-5: equal-MASS bins...` → `# equal-mass bins, finer nb tames a heavy-tailed sv`; `# BLOCKER-2: tgt defaults...` → `# tgt defaults to the price target; a rate-head control passes tgt=rate_target`; `# HIGH-2:` / `# BLOCKER-3:` / `# LOW-1:` / `# N1/ISSUE-4:` likewise. The explanatory text after each tag is good and should stay.

### L7. The illustrative section 9 can be mistaken for evidence

Location: cell 31.

The section says the table is illustrative and not computed for `price_dislocation`, which is good. But it sits near the final verdict and reinforces conclusions about pooling/per-exchange decisions.

Suggested change: move illustrative material to an appendix or replace it with a computed table for the current feature family.


**reply: [PARTIALLY-VALID]**

The risk the reviewer names is real but the notebook already mitigates it heavily. §9 (lines 1162-1190) opens with a blockquote in bold: "**The table below is an illustrative example for a poolable trade-flow feature — it is NOT computed for `price_dislocation`**," and the numbers (0.22/0.23/0.16...) are explicitly "the typical pattern." So the "can be mistaken for evidence" claim is weaker than stated — the warning is about as loud as a warning can be. What survives is purely positional: §9 sits between the gates and the §10 verdict, and the reviewer's own suggestion (move to an appendix) is a layout preference. Worth a small nudge, not a substantive fix.

**Suggested fix:** optional — in `build_feat_nb.py`, retitle §9 to "## 9. Appendix: when is per-exchange worth it? (illustrative, poolable features)" and place it after the §10 verdict so the illustrative table cannot be read as part of the worked example's evidence chain.

### L8. The exact exchange symbols should be introduced once and then used consistently

Location: cells 0, 3, 6, 9, 10, 16, 19, and 32.

The notebook alternates between exchange codes, full listings, and pair names. This is manageable for the worked example but can confuse template users.

Suggested change: include one small configuration table with `exchange`, `listing`, `stream`, and `role`, then refer to those variables.


**reply: [SUBJECTIVE]**

The notebook already does most of what the reviewer asks. Cell 1 introduces the codes once — "**byb** (Bybit), **bin** (Binance), **okx** (OKX) — and **byb is the target**" (lines 17-19) — and the config block at lines 339-349 defines `TARGET`, `OTHERS`, `COIN`, and `MID_STREAM` as the single source of truth that the rest of the code references by variable. The alternation between `byb` and the full listing `byb_eth_usdt_p` is not sloppiness: the full listing is required precisely where the oracle keys state by listing so "a perp and a spot on one exchange never collide" (line 494), so the long form carries real meaning there. A formatted exchange/listing/stream/role table would be a minor polish, but the information already exists and is consistent. Preference, not defect — no fix needed.

### L9. "byb<->okx" wording hides the signed direction

Location: cell 10 output and related prose.

The feature is signed as `log(other) - log(byb)`. The output text says `byb<->okx`, which is symmetric-looking.

Suggested change: print `okx - byb` and `bin - byb`, or explicitly state the sign convention in output labels.


**reply: [VALID]**

This is a genuine, if small, output-clarity bug. The feature is signed: line 522 computes `g = math.log(m) - lt` and line 437 `np.log(mid_on_clock(ex)) - log_mid_byb`, i.e. `log(other) − log(byb)` — a directional quantity. But the oracle print (line 598) labels it `byb<->{o}` ("byb<->okx"), and the lifetime prose (line 1050) uses the same symmetric arrow. A reader cannot tell from `byb<->okx` whether a positive value means okx above byb or below, which matters for interpreting the sign of every downstream IC. The reviewer correctly identifies that the arrow notation hides the convention the code actually uses.

**Suggested fix:** in `build_feat_nb.py`, change the oracle loop label from `f"  byb<->{o}:  ..."` (line 598) and `f"...does not reproduce the byb<->{o} feature"` (line 599) to `f"{o}-byb"` (and mirror in the §4 conclusion prose), making explicit that the gap is `log({o}) − log(byb)`. Add a one-line sign-convention note where the gap is first defined (~line 437).

### L10. Some claims are asserted before the notebook measures them

Location: cells 3, 15, 19, 24, and 32.

Examples include leadership rotation, cadence artifacts, and time-scale-specific venue differences. Some are plausible and may be known from other work, but the template should distinguish "prior belief" from "measured here."

Suggested change: mark such statements as hypotheses until the notebook prints the relevant measurement.


**reply: [PARTIALLY-VALID]**

The reviewer is right that some mechanism statements are asserted as priors, but the notebook is unusually disciplined about flagging exactly that, so the finding is half-addressed. The leadership-rotation claim is explicitly hedged as a *prior*: line 217-218 "It's tempting to call one exchange 'the leader.' Don't — leadership moves around," framed as a guard rail, and §1 even says "We don't measure the exact lag in this notebook" (line 212). The cadence-artifact claim (lines 1045-1050) is presented as a *to-do check*, not a measured result ("Confirm by re-measuring ... matched to byb's update cadence"). What the reviewer catches that is *not* yet hedged: the time-scale→venue-difference story in §6 (lines 723-729, "at short time-scales the venues genuinely differ ... at long time-scales ... redundant") is stated as fact, while §9's own cross-over table is labelled illustrative — so that specific mechanism is asserted ahead of any measurement in this notebook. (This overlaps the High-severity H13.)

**Suggested fix:** in `build_feat_nb.py` §6, soften lines 723-729 from a declarative claim to a hypothesis, e.g. prefix with "We expect (and §9 sketches, but this block does not measure) that ...", so the time-scale-dependent venue-difference mechanism is marked as a prior until the §9 sweep is actually computed for the feature.

## What Is Sound

The notebook has several strong methodological components that should be preserved:

- It starts from a falsifiable mechanism rather than blind feature mining.
- It treats causality, timestamp grouping, and no-lookahead as first-class requirements.
- It separates feature construction from input shaping.
- It uses event-time EMAs consistently and validates vectorized output against a streaming state machine.
- It uses Spearman/rank-IC as a robust screening statistic.
- It uses walk-forward evaluation rather than a random split.
- It includes regime diagnostics, latency/lifetime checks, and echo-netting.
- It explicitly warns against fixed-leader assumptions and blind pooling.

The main issue is not that the techniques are exotic or invalid. Most are standard or defensible. The problem is that the notebook often presents exploratory, single-block, post-selection diagnostics as if they were final production evidence.

## Recommended Fix Order

1. Fix the signed-versus-absolute contradiction and rate-head gate gap.
2. Replace the current shipping verdict with a provisional single-block verdict.
3. Add nested or held-out span selection before reporting walk-forward IC.
4. Add per-exchange Gate A and cadence-matched cross-venue checks.
5. Add uncertainty estimates and multi-block/regime-gap validation.
6. Split the oracle into a dead-simple independent reference plus the current streaming parity check.
7. Tighten wording: remove hard-gate overclaims, stale numbers, and repeated doctrine.
