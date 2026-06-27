# Feature screening — method (companion)

Read this once. It is the **method** that every per-feature screening notebook follows. The notebooks themselves hold only the parts that change per feature — what the feature is, its exact definition, its builder, and the results — and link back to the sections here. Nothing below changes when you clone a notebook for a new feature.

The model forecasts one exchange's mid-price about 100 ms ahead. The prediction target is fixed across every feature: **byb** (Bybit) — so "the target", "the target's mid-move", and "the target's volatility" always mean byb. Everything *else* is the feature under test and varies. Where a concrete illustration helps, the **running example** below is `price_dislocation` — a cross-venue log-price gap (the other venues are **bin** (Binance) and **okx** (OKX)) smoothed fast-vs-slow and divided by the volatility yardstick. That example is one feature among many (order-flow imbalance, queue imbalance, microprice, trade-flow, …); read every *rule* as general and the cross-venue / `price_dislocation` specifics as illustration.

How a feature's EMAs are *constructed* — which EMA class (flow vs level) and *when* to inject vs decay on the two clocks (the shared trade clock for decay; each EMA's own events for injection) — is a separate **implementation contract**: author features in `src/boba/features/` following [`AUTHORING.md`](../../src/boba/features/AUTHORING.md). This companion covers the *analysis* method (the heads, the gates, the parity check, echo-netting), not how a feature's EMAs are built.

---

## The two-heads model

A feature is only worth something if it helps the model predict, so it's worth knowing what the model does. We forecast how the target's mid-price moves over the next ~100 ms, and we split that into two simpler questions — the two **heads**:

**Price head — which way and how far?** Over the next few price-moves, what is the *signed* move (its direction *and* its size, together)? The head predicts the whole distribution of that move, in units of the target's recent **volatility** — the **volatility yardstick** `σ_ev` (the exp-weighted RMS of the target's *actual* mid-moves) — so the head's target is `price change ÷ σ_ev`.

**Rate head — how many moves?** Busy markets pack many price-moves into the window, quiet ones few. This head predicts the *count* of moves over the next 100 ms as a distribution, measured against the recent pace — the **rate yardstick** `λ_ev` — so its target is `count ÷ λ_ev`.

**What's a yardstick?** A causal, trailing estimate of the target's *volatility* (`σ_ev`) and move *rate* (`λ_ev`), from past data only. A regime gauge, nothing more. When a feature *carries* that regime and you want it gone, dividing by the yardstick is **one option** — but only when the regime enters the feature as a multiplicative **scale** (a raw price gap is literally `σ` times bigger in a wild market, so `gap/σ` is scale-free). A feature that is already dimensionless — a ratio, or a bounded imbalance — carries no such scale to divide out, and dividing anyway can *inject* the yardstick's own regime; that is why you don't normalise reflexively. Both yardsticks are EMAs **decayed on the trade clock** (`α = 2/(span+1)`) but **updated between trades** — they react to every target mid-move, so they read live at every instant. They use one fixed span, `YARDSTICK_N` (any feature may use that span too). (`σ_ev` is the exp-weighted RMS of the target's *actual* mid-moves — read as an `E/W` ratio so the many non-move trades cancel out; `λ_ev` is a ratio of two EMAs — the exp-weighted move-count `W` (the same `W` that is `σ_ev`'s denominator) ÷ the exp-weighted seconds-per-trade = the target's moves per second.) Like every average here, they live on the trade-tick clock — never wall-clock or a hard window.

**Why split into two heads?** A move over a window is just *how many* little moves happen times *how big* each one is. Pulling "how many" (rate) apart from "how big" (price) lets each head learn a steadier thing — and is why there are two yardsticks, one per head.

**Putting them back together.** The price head isn't a single distribution — it's a *family*, one per possible move-count: "if `k` moves happen, here's the spread of the total price change" (call it `D_k`). The rate head gives the probability of each count, `P(K = k)`. The 100 ms move is the two **mixed** — for every count `k`, take the price head's `k`-move distribution, weight it by the rate head's probability of exactly `k` moves, and sum:

`distribution of the 100 ms move  =  Σ_k  P(K = k) · D_k`

More moves → a wider spread, so the rate head's "how many" sets the scale and the price head's `D_k` sets the shape. This is why a feature that predicts the *count* (rate head) and one that predicts the *per-move direction* (price head) are both useful: they feed the two factors that multiply together.

**Backed by research.** This "how many × how big" split is the classic **subordination** model of asset prices (Clark, 1973, *Econometrica*; Ané & Geman, 2000, *Journal of Finance*): returns over a fixed clock-time window look messy, but become well-behaved once you condition on the *number* of events. So modelling the event count and the per-event move separately, then mixing them, is a principled decomposition — not just a convenient one.

Every feature feeds one or both heads, and borrows the matching yardstick when it needs to be made comparable across markets. Two questions recur: does a feature's *signed* value predict the move (price head), and does its *magnitude* predict how many moves come (rate head)? Those are diagnostics — the model is fed the **signed** feature for *both* heads, and the rate head learns the magnitude (and how features cancel or reinforce) on its own. And they are **marginal screens** — does the feature carry signal worth feeding — **not** measures of distributional fit: the model's actual targets are the count distribution `P(K = k)` and the count-conditioned price family `D_k` mixed above, fit downstream, not what these rank-ICs measure.

---

## Guard rails

The hard rules and guard rails — feature **construction** *and* the analysis **method** — are the enforceable contract in [`AUTHORING.md`](../../src/boba/features/AUTHORING.md), next to the feature code. The sections below explain the method that motivates them; author features by those rules.

---

## Choosing the EMA

Which EMA class to use (a sparse **flow** → `KernelMeanEMA` read as `E/W`; a forward-filled **level** → `LiveFrontEMA` read as a live front) and *when* to inject vs decay is the feature **implementation contract** — see [`src/boba/features/AUTHORING.md`](../../src/boba/features/AUTHORING.md). Author features in that package, by those rules; the parity check enforces them. (Only `KernelMeanEMA` and `LiveFrontEMA` are allowed — never `EventEMA`, a hand-rolled EMA, or `lfilter` in the streaming build.)

---

## The parity check

**Why this exists.** The feature is computed *twice, in two different worlds*: research screens it with a **vectorized offline array path**, but production runs an **online, O(1)-per-event state machine**. Those are two genuinely separate codebases. If they ever disagree, the number you screened is not the number that will trade — and the disagreement is usually a silent causal or aggregation bug (a stray forward-fill, a double-counted same-timestamp burst, a decay on the wrong clock). The parity check forces the two to agree on real data, so a passing screen is trustworthy.

**Non-negotiable.** Reproduce the feature with the production-style streaming build and confirm the two agree on real data to floating-point round-off — a **parity check**, not an independent oracle. The streaming build is an O(1) state machine you push **raw events** into — `on_book(...)` for a top-of-book update, `on_trade(...)` for a trade — and read the current feature from `value()`. State is a few scalar EMAs — no buffers, no history, independent of how long it runs.

The general design that makes online == offline (independent of the feature):
- It is fed **only raw events**, each tagged by its full source (e.g. `byb_eth_usdt_p`) and keyed by it, so two instruments never collide. It reconstructs any **derived inputs it needs (e.g. a mid) itself**, the same way the offline path does, so the live input matches the analysis input — never read a pre-derived stream.
- Events sharing a **timestamp are one event**: the driver applies them all, then calls `refresh()` **once** — which updates each EMA's live front and injects on a relevant event, then advances the clock **at most once** (decay, commit) and only if a trade landed. A book-only instant moves the inputs without advancing the clock.
- `value()` returns the live-front feature, current at the instant it's read.

If the online build reproduces the vectorized feature on real data to round-off, the two implementations compute the same feature. The check validates the **actual** `boba.ema` online classes, not a throwaway re-implementation.

*(Running-example specifics — kept in the notebook, not general: the example's state is two `LiveFrontEMA` gap legs plus `σ_ev`'s `KernelMeanEMA`; it builds each venue's mid by the project's per-venue policy — merged venues fuse trades into the book by newest-exchange-time, the book-only venue takes the latest snapshot — and injects `σ_ev` on a target mid-move.)*

---

## The hygiene gates

A correlation is an easy way to fool yourself. The gates are **two independent tests**: **(A)** is the feature **regime-invariant** — a stable distribution that doesn't leak the vol/rate state — and **(B)** does it **predict** something the market's current state doesn't already tell us? The only **controls** for Gate B are the two **regime-invariant momenta**:
- **rate momentum** — from `λ_ev` (the target's mid-move rate): is the target moving more or less often than its own recent pace?
- **vol momentum** — the same, for volatility.

The raw **levels** of vol and rate (`log σ_ev`, `log λ_ev`) are **not** controls — they aren't regime-invariant, so they're never model features; we keep them only as the regime *coordinate* for Gate A.

**Use the shared, tested gate library — do not re-implement any of this.** Every computation below has a tested home in `boba.research.gates` (see `tests/research/test_gates*.py`; externally validated against the literature and adversarially reviewed). Re-deriving a walk-forward, an IC mask, a stratified IC, or a bootstrap by hand is exactly the blind-spot the parity check guards against on the EMA side — so call the library: the masked rank-IC is `ic`; the walk-forward folds and mean OOS IC are `wf_folds` / `wf_ic`; Gate A's four numbers are `gate_a`; the Gate-B marginal is `signal_ic` (with `signal_ic_by_regime` for the calm/mid/wild companion); the within-stratum decouple is `stratified_ic`; and the per-fold spread + block-bootstrap CI is `marginal_ci`. The notebook binds the regime scaffolding (`base` controls, `FEATURE_KIND`, `STRAT_VAR`, `vol_regime`) into thin wrappers so the call sites read the same while the logic stays in one validated place.

"Predictive power" is the **rank correlation** between feature and outcome (Spearman — robust to outliers), scored **out-of-sample with a purged, expanding-window walk-forward** (`wf_ic` / `wf_folds`): each fold trains only on the *past*, leaves an **embargo gap** sized to clear the outcome windows with margin (so overlapping forward targets can't straddle the train→test boundary). Note the embargo does **not** always fully decorrelate the slow EMA/yardstick features: their memory ≈ `YARDSTICK_N / trade-rate` can *exceed* the embargo (embargo in anchors × the grid spacing in seconds), in which case the slow features stay partly correlated across the boundary — re-check this whenever you slow the spans or use a thinner-traded block. The fold scores on the *next* segment, and we average over folds. That's the causal, production-style estimate — strictly past→future, as it would run live.

Because adjacent samples are correlated (overlapping forward labels + long EMA/yardstick memory), a single point IC overstates its own precision — so for the **headline marginal** we also report the **per-fold** ICs and a **block-bootstrap 90% CI** (`marginal_ci`). **Why this matters for the verdict:** a point IC comfortably above the acceptance floor can still flip sign across folds or have a 90% CI straddling zero — i.e. an edge not reliably above noise on this block. Read the per-fold count and the CI alongside the headline, not the point estimate alone. The bootstrap resamples contiguous time blocks whose length is **auto-derived from the target's own autocorrelation-decay lag** (`estimate_block_len`: the first lag where |ACF| falls below ~`2/√n`, scaled, floored at the embargo, and capped to keep ~30 blocks per fold), so the interval respects the dependence length rather than pretending all anchors are independent. Don't re-derive that block-length heuristic — call `marginal_ci`, which wraps it.

Rank-IC is a **feature-screening** statistic, not a distributional score — proper scoring (NLL/CRPS, occurrence log-loss, calibration) and cost-aware utility are judged at the **model** level downstream.

Because the feature and target are both in σ-units, a *scale* regime-shift mostly cancels — but scale is not the *relationship*. So beside the gates we run a **companion check** (`signal_ic_by_regime`): the same marginal power computed **within calm / mid / wild volatility buckets**. **Why:** the all-regime marginal can read positive even if the feature's *entire* edge lives in one regime (say only wild markets) and is flat or negative elsewhere — an artefact of this block's regime mix that need not recur on the next block. A gain positive in all three buckets is regime-stable, not a one-regime accident.

### Gate A: regime invariance

(the feature *alone* — `gate_a` returns the four numbers.) Is the feature's distribution **stable against our regime diagnostics**, or does it *leak* it? **Control-free** checks: **scale** — its std across vol buckets (max/min, want **< ~3**); and then, for **both the signed feature *and* its magnitude `|feature|`** (the rate head receives the *signed* feature and can learn its magnitude, so a magnitude that tracks the regime would leak into it), two leak modes against **both regime coordinates** (vol *and* rate level): **tracking** — `|IC(·, level)|` ≈ 0 (the monotone test, want **< ~0.05** for the signed feature, **< ~0.1** for the magnitude), and **dispersion** — the spread of its per-decile *means* (want **< ~0.1**), which catches a *non-monotone* leak the monotone IC misses. Each closes what the others miss: scale alone passes `z + c·vol_level` (flat std, mean rides the regime); the monotone IC misses a *U-shaped* leak (`z + |rank(vol) − 0.5|`) that only dispersion catches; and a feature flat in signed mean and scale can still leak through its *magnitude* into the |·|-fed rate head, which only the magnitude checks see. The vol/rate level is only the regime *coordinate* (what we bucket/correlate against), **never** a control. Fail any one of them = a level in disguise, not a feature. (The bars above are working thresholds, not laws.)

### Gate B: predictive signal

Does it predict? `signal_ic` returns the value. Because *a control can itself be a valid feature*, "signal **over** the controls" only makes sense for a feature that **isn't** a control. An **alpha** (a candidate signal that is *not* a regime descriptor) is judged on its **marginal** rank-IC over the regime-invariant controls (the momenta — **never** the raw levels), all instances together and each on its own. **What the marginal is:** an out-of-sample, nested-model *incremental* IC — the per-fold difference between the rank-IC of a model with the feature and one with only the controls. It is a **relative screen, not a partial correlation or an effect size** (don't read it as "the feature's IC after residualising"). A working acceptance floor is ≳ 0.01. *(A **control-type** feature — a regime descriptor like σ_ev/λ_ev — is instead judged on its **standalone** signal, since marginal-over-its-own-controls is circular; only its cross-source legs stay a marginal lead test.)*

**Mechanical-coupling guard.** When a feature is a **ratio that divides by the scored target's own yardstick** (numerator `A/Z`, target `B/Z`, sharing the same `Z`), the shared `Z` inflates the IC for a purely arithmetic reason (Pearson's spurious correlation of ratios). The fix is to **stratify by that shared yardstick** and score *within* its strata (`stratified_ic`, with the stratifier set to the scored target's yardstick array — `σ_ev` for the price target, `λ_ev` for the rate target): stratifying multiplicatively **decouples** the shared denominator, where a *linear partial* would over-remove most of the genuine within-yardstick signal. **A signed numerator escapes the trap:** the inflation comes from the numerator correlating with the *strictly-positive* `1/Z` factor that the target also carries; a roughly sign-symmetric (signed) numerator barely correlates with a strictly-positive factor, so there is little for the shared `Z` to manufacture — hence a signed price-head alpha can use no stratifier (in the running example, `STRAT_VAR = None`), **verified, not assumed**, by the coupling rows in the gate table. (This control-standalone stratified IC is in-sample decoupled, not walk-forward; its out-of-sample confirmation comes from the separate multi-block harness in `tools/oss`.)

---

## The rate head

The hygiene gates above run the *price-span* feature against the σ_ev **price** target. The rate head is fed a **different-span** feature (its own sharp pick) and predicts the **count** target, so its verdict is **not** inherited from the price gates — it gets the same two-gate battery (`signal_ic(..., tgt=rate_target)`, `gate_a`), against the count target.

The rate head's signal lives in the **magnitude**: `|feature| → count` is the rate diagnostic, and the model is fed the **signed** feature and recovers `|·|` itself (a nonlinear head can). So **Gate B scores `|feature|`** — a *linear* score on the signed feature would read ≈ 0 precisely because the count relationship is symmetric, so `|feature|` is the honest proxy for what the nonlinear rate head extracts.

**Coupling guard (a different mechanism than the price head's).** The price head's signed numerator decoupled the shared `σ_ev` exactly. The magnitude path has no sign to decouple it, and here the coupling is *not* an exact shared denominator: `|feature| ∝ 1/σ_ev` while `count/λ_ev ∝ 1/λ_ev` — different denominators that nonetheless **co-move** because `σ_ev` and `λ_ev` both track the same activity regime. So add a **within-`λ_ev` stratified** line per source (`stratified_ic(..., λ_ev)`) as a robustness check that the marginal isn't a `1/λ_ev` artefact.

**Why Gate A is re-run.** Gate A is a property of the feature's **own output distribution**, and the rate head feeds a *different-span* feature — a different distribution (different scale-across-vol, different magnitude-tracking). Its regime-invariance therefore cannot be inherited from the price-span Gate A any more than its signal can; re-run `gate_a` per source on the rate-span feature.

---

## The self-test

**When it's needed.** The gates' **control branch** (`stratified_ic` decoupling a shared-denominator ratio) is shared, tested code — but a notebook only *runs* that branch when its feature **is** a regime control (`STRAT_VAR` set). Any **alpha** screen leaves the stratify-decouple branch unexercised. So when the feature under test is an alpha, add a small self-test that runs the branch deliberately — it guards against a regression in the decouple, or a yardstick distribution on this block too degenerate for the stratification to work. (For a feature that *is* a control, the notebook exercises the branch directly and the self-test is redundant.)

It builds, on the grid, a `count/λ_ev`-style target and two control *ratios that both divide by `λ_ev`*:
- **pure-spurious** — a numerator with **no** real link to the count, divided by `λ_ev`. Its **raw** Spearman against the target is large (the shared `1/λ_ev` manufactures it); **stratified** by `λ_ev` (`stratified_ic`) it collapses to ≈ 0.
- **real-signal** — a numerator that genuinely tracks the count, divided by `λ_ev`. Its raw IC is inflated by the same coupling, but the **stratified** IC keeps the genuine within-stratum signal.

If `stratified_ic` returns ≈ 0 for the spurious ratio (vs a large raw IC) and recovers the real one, the decouple does what Gate B's control branch needs.

---

## Echo-netting

A feature can be perfectly causal and still not *predict*: if its apparent edge is the price move **already underway** at the anchor, you can't capture it — by the time you observe, decide, and act, that move is gone — and a purely *contemporaneous* feature can post a positive forward IC from window overlap alone. So before trusting a forward IC, **net out the echo**: measure the feature's correlation with the **forward** return (`[anchor, anchor+H]`) *controlling for the move that already happened* (`[anchor−H, anchor]`). The **backward IC** sizes the echo; the **echo-netted** forward IC is what survives once it's partialled out — the genuinely forward-looking edge. If a big raw IC collapses once the trailing move is partialled out, the feature was mostly re-reporting the move already underway — report the **netted** number in the verdict, not the raw IC. (A near-zero netted IC alongside a large backward IC is the one true non-signal: all echo, no prediction.)

The echo-net is a small **partial rank-IC** — a relative diagnostic, not a tested gate primitive. Build it from the shared masked rank-IC (`boba.research.gates.ic`) for the three pairwise correlations and combine with the standard partial-correlation formula; or, equivalently and preferably, compute it through the gate library as a marginal IC (`signal_ic`) with the trailing-outcome leg added to the control set — that reuses the validated walk-forward/masking machinery rather than a fresh hand-rolled correlation. Either way, don't re-implement the masked rank correlation by hand.

**For cross-source features only: a freshness lead is *real edge*, not an artifact to coarsen away.** This applies when a feature compares sources with different feed latencies (the running cross-venue example; not a single-source feature). The data is recorded on a production box in the target datacenter, so each event's `rx_time` is exactly the timing you'd see live — there is **no recording/snapshot artifact** to rule out. So when a faster source's book moves before the target's reflects it, that lead is **genuine and exploitable**, and the *mechanism* (economic price-discovery vs pure latency lead-lag) is irrelevant to P&L. Do **not** coarsen the faster feed to the target's cadence — that throws the edge away. *(A freshness lead would only be fake if the recording's cadence didn't match production — e.g. a backtest on vendor snapshots; not the case here, where the recording* is *production timing. The specific feed-staleness profiles of each venue belong in the notebook, measured per block, not asserted here.)*
