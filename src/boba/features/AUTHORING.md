# Authoring a feature — rules, guard rails, and the EMA contract

This is the **enforceable contract** for feature modules in this package: the vectorized builder and the streaming class behind every `FeatureSpec` (see `base.py`). A feature is two implementations of the same maths that must agree to floating-point round-off on a real block — `boba.research.screening.parity_check` enforces it. Get a rule below wrong and the number is wrong, and either the parity check fails or a gate quietly lies.

The screening notebook's companion (`notebooks/features_v2/METHOD.md`) explains the conceptual *why* — the two-heads model, what each gate proves, the subordination backing. **This** file is the *what*: the hard rules every feature must follow, kept next to the code. If the two ever drift, this file is the source of truth for the rules.

Throughout: the prediction **target** is fixed (byb) — "the target" / "the target's mid-move" always means it. A **feature** is the thing under test and varies; many features **fan out** into one instance per source (e.g. one gap per foreign venue), and the rules below say "per source / per instance" wherever that matters.

---

## Guard rails

Hard rules, learned the hard way. Follow them unless you have a specific, written reason not to.

**Don't**
- **Don't assume a fixed leader, when a feature fans out into per-source instances.** No single source always "leads" — leadership moves around. Build every instance the same way and keep them all. (A single-source feature has no fan-out and nothing to choose here.)
- **Don't pick "the best instance" by average score** — that throws away the moments another source leads (often the most informative ones). Keep all; let the model weight them.
- **Don't merge a fanned-out feature's sources into one averaged value.** Keep every source's instance (per-exchange) so the model weights whichever leads; collapse to a single source only when one demonstrably suffices (checked single-vs-per-exchange across time-scales, a later step).
- **Don't normalize reflexively.** Divide volatility or rate out of a feature *only when it needs it* — only when the regime enters as a multiplicative scale (a raw gap is literally `σ` times bigger in a wild market, so `gap/σ` is scale-free; a ratio or bounded imbalance is already dimensionless and dividing again can *inject* the yardstick's own regime).
- **Don't invent your own vol/rate scale.** When you do normalize, use the model's yardstick (`σ_ev` or `λ_ev`, from `ScreeningContext`), so the feature shares units with the target.
- **Don't trust a correlation** until it survives the regime controls (rate and vol momenta) — else it may just be re-reporting "the market is volatile."
- **Don't ship a feature without the parity check** — the streaming build must reproduce the vectorized build on real data (`parity_check`).
- **Don't hand-roll a streaming EMA, and don't use `EventEMA`** — use only `KernelMeanEMA` (flow) or `LiveFrontEMA` (level) from `boba.ema`, and no `lfilter` in the streaming build. (See [EMA construction](#ema-construction) for the full rule and the why.)
- **Don't peek ahead.** Every value uses only data at-or-before its own timestamp (a stray forward-fill is the usual way to break this).
- **Don't over-transform for the network.** Pick the lightest reshaping that works.
- **Don't fuse the two gates.** Regime invariance (Gate A — the feature's own distribution being stable across regimes) and signal (Gate B — what it predicts over the *invariant* controls) are independent tests. The raw vol/rate **levels** are never Gate-B controls (they aren't valid features) — they are only the Gate-A regime *coordinate*. And a control can itself be a valid feature: when the feature *is* a regime descriptor, judge it on standalone signal, never "redundant" from its algebra alone.

**Do**
- **Do start with a falsifiable hypothesis** — a mechanism for why it should work, and what would prove it wrong.
- **Do build every average by the EMA contract** — which class and when to inject vs decay are specified in [EMA construction](#ema-construction) below; the parity check enforces it.
- **Do test against both heads — but feed both the *signed* feature.** Check whether the signed feature predicts *direction* (price head) and whether its *magnitude* predicts *intensity* (rate head). Those are diagnostics — the model is fed the **signed** feature to *both* heads, never a pre-computed `|feature|` (which would destroy the sign and the cancel/reinforce interaction the rate head learns).
- **Do score out-of-sample** with a purged, embargoed walk-forward (strictly past→future), the embargo sized to clear the outcome window — *because* overlapping forward labels let a train row's outcome window straddle the train/test boundary and leak the future, inflating the OOS IC. A single split is only a faster screen.
- **Do use the freshest valid price per source.**
- **Do treat a feature as a family across time-scales** and let the data assign scales to heads.
- **Do prove regime-invariance with Gate A — never assume it.** A usable feature reads the *same* in calm and wild markets: scale stable across vol buckets, and neither the signed feature nor its magnitude tracks the regime (monotonically or non-monotonically), against both the vol and rate coordinate. A raw **level** usually *is* the regime and fails — but *measure, don't assume*; a ratio/bounded/normalised form may pass. Never call a feature regime-invariant *or* not until every Gate A number says so.

---

## EMA construction

Every average in a feature — each leg, plus any yardstick it composes — is a trade-tick EMA, and the rules below are consequences of running **two independent clocks**:

- **Decay clock** — the *shared trade clock*: one tick per trade-timestamp across **all** venues (simultaneous prints count once).
- **Injection clock** — each EMA's *own* relevant-event timestamps (a target mid-move, a book update, a trade flow — whatever that EMA measures).

Every streaming primitive must keep these two **separable**. That is the whole reason only `KernelMeanEMA` and `LiveFrontEMA` are allowed: both expose **separate `tick()` (decay) and `add()` (inject)** calls. `EventEMA`'s single `step()` fuses them and is banned for direct use.

### Which EMA class (from `boba.ema`)

Classify the quantity, then pick the class — mis-classifying produces a wrong number, not an error.

- **A flow** — present on only *some* clock ticks (a target mid-move, a per-source trade flow; `σ_ev` is one): use **`KernelMeanEMA`**, read as the self-normalising `E / W`. *Why the ratio:* the flow lands on only a fraction of the shared-clock ticks, so a plain EMA is diluted by every tick that doesn't carry it (and biased during warm-up). `W` counts only the events that carry the quantity and shares the *same* decay-and-warm-up factor as `E`, so the ratio divides both out exactly — a true per-event mean.
- **A level** — defined at *every* instant (a price, a cross-source gap, a microprice): use **`LiveFrontEMA`**, read as the live front `(1 − α)·committed + α·latest` — current between trades, never frozen on the last trade. *Why no `W`:* a level has a value at every tick, so there are no missing events to normalise away; what it needs is to stay fresh between trades, which the live front gives.
- **Banned:** `EventEMA` directly; any hand-rolled scalar EMA (no `_ScalarEMA`, no `(1−α)·s + α·x` per-event loop); `scipy.signal.lfilter` **in the streaming build**. (The *vectorized* offline builder may use `lfilter` — it is not the production path.)
- Anything not cleanly a flow or a level is mis-modelled. A slope / covariance is a **ratio of flow EMAs** (several `KernelMeanEMA`s), not a bespoke class.
- All EMAs use `α = 2/(span + 1)`. `span = 1` ⇒ `α = 1` (no memory / no smoothing) — a distinct code path on both sides and the most-used fast leg; handle it deliberately.

### When to inject vs decay

These never change, whatever the feature:

- **Inject at most one sample per timestamp**, and only at a timestamp that carries the EMA's **own** relevant event. A timestamp with only irrelevant events injects nothing for that EMA.
- **Decay once per trade-timestamp**, iff a trade lands, on the shared clock. Inject and decay are independent; neither ever fires more than once per timestamp.
- **Records sharing a timestamp are ONE update, not a sequence.** Aggregate them, then register a single update. *What* value depends on the quantity:
  - a **level** takes the **last** state (the final mid / book / microprice at that timestamp);
  - a **flow** **SUMS** the records — in code, a single `KernelMeanEMA.add(summed_value)` with **weight 1** per timestamp. **Never** N `add()` calls and **never** `add(value, weight=N)`: injecting N over-weights the instant a feed stalled and dumped a burst (N× on the `E/W` mean) and corrupts the count `W` that normalises it.
- **Read the freshest value between trades** (live front); never freeze on the last trade's snapshot. But keep decay on the trade clock — push a fresh *sample* once per book-update *timestamp*, not per message (per-message would weight by quote activity).
- **A sparse flow is two trade-tick EMAs read as `E / W`** — the value `E` and its weight `W`, both decayed every trade-timestamp, each pushed a sample only at the timestamps carrying its own events. Dividing by `W` cancels the foreign-event decay and the warm-up bias exactly.

---

## The interface (`base.py`)

A feature supplies a `FeatureSpec` bundling the two builds with a standard interface, then `register()`s it; the generic engines in `boba.research.screening` drive it with no per-feature glue.

- **Vectorized builder** `(ctx, params) -> {leg_key -> vector}` — the offline/analysis path. May use `lfilter` and any array trick. Causal (each row uses only data at-or-before its anchor; NaN where undefined). Divides by the yardstick vectors on `ctx` only when the feature carries that regime as a multiplicative scale. `params` is an opaque token (`N`, or `(n_fast, n_slow)`).
- **Streaming class** (`StreamingFeature`) — the O(1) production path, built only from the allowed EMA classes:
  - `on_book` / `on_trade` mutate internal state for one raw event — they do **not** decay, inject, or read.
  - `refresh()` runs **once per receive-timestamp**, after all that timestamp's events are applied: update each level's live front and inject each flow's one sample, then advance the trade clock **at most once** (decay; commit each level) iff a trade landed.
  - `value()` returns `{leg_key -> scalar}` — the live front at the instant it is read; its keys must match the vectorized builder's dict keys for the same `params`.
  - **Compose `boba.research.screening.LiveYardstick`** rather than recomputing `σ_ev` / `λ_ev`, so yardstick parity is established once in the test suite and your feature's parity validates only the feature-specific maths.

---

## Validation

- Establish the trade-clock EMA convention against a plain one-event-at-a-time loop on a real block, the way `notebooks/03_ema_clock_validation.ipynb` does.
- Every feature's vectorized and streaming builds are tied together by `boba.research.screening.parity_check` on a real block — to float round-off. A failing parity is almost always one of the rules above broken (a stale read, a double-counted same-timestamp burst, a decay on the wrong clock, or a mis-classified flow/level).
- Beyond parity (which only proves the two builds agree), validate each feature against an **independent, dead-simple oracle** on a real block — the project's standing validation requirement. The oracle must be implementable from the feature's written definition alone, share no code with the production build, and match it to float32 tolerance.
