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
- **Don't leave `span = 1` ill-defined.** `span = 1` (α = 1, no smoothing) is a *distinct code path* and the most-used fast leg — never let it slip into a `0/0`, an `inf`, or a value that differs online vs offline. A genuinely-undefined state (warm-up / no relevant event yet) must be a **consistent NaN both builds agree on** (the gates mask it) — never a spurious `inf` or a build-specific number; everywhere the inputs exist the value must be **finite**.
- **Don't peek ahead.** Every value uses only data at-or-before its own timestamp (a stray forward-fill is the usual way to break this).
- **Don't over-transform for the network.** Pick the lightest reshaping that works.
- **Don't fuse the two gates.** Regime invariance (Gate A — the feature's own distribution being stable across regimes) and signal (Gate B — what it predicts over the *invariant* controls) are independent tests. The raw vol/rate **levels** are never Gate-B controls (they aren't valid features) — they are only the Gate-A regime *coordinate*. And a control can itself be a valid feature: when the feature *is* a regime descriptor, judge it on standalone signal, never "redundant" from its algebra alone.

**Do**
- **Do start with a falsifiable hypothesis** — a mechanism for why it should work, and what would prove it wrong.
- **Do build every average by the EMA contract** — which class and when to inject vs decay are specified in [EMA construction](#ema-construction) below; the parity check enforces it.
- **Do define and test the feature at `span = 1`.** Choose its value at the no-smoothing boundary deliberately — a level's live front collapses to the freshest read, a flow's `E/W` to the latest sample — and guard the degenerate arithmetic (`0/0`, `inf`) on **both** sides. Include a `span = 1` leg in the parity check (and an oracle row that asserts finiteness wherever inputs exist): it is the leg most likely to expose an α=1 online/offline mismatch, and the one the analysis sweeps most.
- **Do leave only WARM-UP NaN in the model input — never a mid-stream NaN.** `NaN` is not a valid model input, so once a feature has warmed up it must hand the model a number at *every* sampled point. A flow `E/W` at `span = 1` momentarily reads `0/0 = NaN` at a bare trade-tick (it fully decayed, no fresh sample yet) — a legitimate *raw* event-grid NaN both builds agree on, but it must NOT reach the model. The model-facing layer (`build_family`, and the streaming driver in production) `hold_last`s each leg — carrying the freshest real reading forward through any mid-stream NaN/inf — so only the leading warm-up NaN (before the first valid value) survives. So: the **raw** event-grid output may be NaN at a genuine no-fresh-input instant (held downstream); the **model input** is NaN/inf-free except during warm-up. Don't hand-roll this per feature — it's centralised in `boba.features.shared.hold_last`.
- **Do test against both heads — but feed both the *signed* feature.** Check whether the signed feature predicts *direction* (price head) and whether its *magnitude* predicts *intensity* (rate head). Those are diagnostics — the model is fed the **signed** feature to *both* heads, never a pre-computed `|feature|` (which would destroy the sign and the cancel/reinforce interaction the rate head learns).
- **Do score out-of-sample** with a purged, embargoed walk-forward (strictly past→future), the embargo sized to clear the outcome window — *because* overlapping forward labels let a train row's outcome window straddle the train/test boundary and leak the future, inflating the OOS IC. A single split is only a faster screen.
- **Do use the freshest valid price per source** — read the trade-fused `merged_levels` mid where the venue benefits; see [Price source](#price-source--fresh-mids-vs-raw-sizes).
- **Do treat a feature as a family across time-scales** and let the data assign scales to heads.
- **Do prove regime-invariance with Gate A — never assume it.** A usable feature reads the *same* in calm and wild markets: scale stable across vol buckets, and neither the signed feature nor its magnitude tracks the regime (monotonically or non-monotonically), against both the vol and rate coordinate. A raw **level** usually *is* the regime and fails — but *measure, don't assume*; a ratio/bounded/normalised form may pass. Never call a feature regime-invariant *or* not until every Gate A number says so.
- **Do declare the feature's mirror reflection** (`FeatureSpec.mirror`) so the signed-head IC can be mirror-augmented (direction-free). See [Mirror augmentation](#mirror-augmentation).

---

## Mirror augmentation

The market has an **up/down symmetry**: no microstructural law makes rallies behave differently from selloffs. We exploit it as a data augmentation — score the IC (and fit the OOS regression) on the real tape **and on its mirror image**. The mirror image is the whole book + tape **reflected through a fixed price level** `c` (byb's mid at the anchor): geometrically, draw a horizontal line through byb's mid and reflect every price across it.

### Reflecting a book about a fixed level

Work in log-price and reflect each log-price about `ℓ = log c`:  `log p ↦ 2ℓ − log p`. (In price space this is `p ↦ c²/p ≈ 2c − p` for the basis-point moves we deal in — the horizontal-line picture.) Apply it field by field:

| quantity | under the reflection | parity |
|---|---|---|
| any log-price | `log p ↦ 2ℓ − log p` | — |
| **best bid / best ask** | the two sides **swap and reflect**: `bid' = reflect(ask)`, `ask' = reflect(bid)` (the best buy becomes the best sell) | — |
| **spread** (`ask − bid`) | **unchanged** | even |
| **mid** | reflects: `log mid ↦ 2ℓ − log mid` | — |
| **any log-gap or log-return** (a difference of log-prices) | **negates** — the level `ℓ` cancels in the difference | odd |
| **trade price** | reflects: `log px ↦ 2ℓ − log px` | — |
| **trade aggressor / side** | **flips**: buy ↔ sell; the signed direction `d ↦ −d` (a buy lifting the ask becomes a sell hitting the bid) | odd |
| **trade size / volume** (contracts) | **unchanged** — a count, not a price | even |
| **signed order flow** (`size × direction`, OFI) | **negates** — size invariant, direction flips | odd |
| **σ_ev, λ_ev, \|gap\|, spread, any magnitude / RMS** | **unchanged** — built from squared or absolute moves | even |
| **time / the trade clock** | **unchanged** | — |

Two anchors of the picture: the reflection **preserves the spread and every size** (it is a price mirror, not a rescaling), and it **negates every signed price-difference and flips every trade side** — exactly the quantities that carry *direction*.

### A feature's reflection — `FeatureSpec.mirror`

Each feature declares how **its value** transforms, by applying the table above to its inputs (a callable `vec -> reflected vec`):

- **Odd** — built from signed differences / signed flow (a log-gap, a return, a signed imbalance, OFI): the value **negates**, so `mirror = np.negative`. Only odd features carry *signed* (price-head) signal. `price_dislocation` is odd (log-gaps).
- **Even** — built from magnitudes / sizes / spreads (a volume, a `|gap|`, a spread, σ_ev): the value is **unchanged**, so `mirror = lambda v: v`. An even feature has no signed signal by symmetry — it belongs to the rate head, scored on `|·|`, and is never mirror-augmented.
- **Undeclared** (`mirror=None`, the default): the feature may not be mirror-augmented; the engines skip it.

In practice `mirror` is the closed-form effect on the value — derive it from the table; you do **not** rebuild the feature from reflected inputs.

### The invariant — required, and tested

`FeatureSpec.mirror` is a *closed-form shortcut* for reflecting the inputs, so it is only correct if it **commutes** with the actual book reflection. This is a hard invariant every feature must satisfy:

```
mirror(feature(books))  ==  feature(mirror_books(books))
```

— applying the declared `mirror` to the feature computed on the real books equals recomputing the feature on the **mirror-reflected** books (`mirror_books` = reflect every price through a fixed level per the table above; sizes/clock unchanged). If they disagree, the declared `mirror` is wrong (or the feature has a hidden even/odd-breaking term) and the mirror-augmented IC is meaningless.

**Every feature MUST (a) declare `FeatureSpec.mirror` (never leave it `None`) and (b) ship a test of this commutation invariant** — reflect a synthetic block's books, rebuild, and assert equality to float round-off. See `tests/features/test_price_dislocation.py::test_mirror_commutes_with_full_book_reflection` for the pattern (and `test_every_registered_feature_declares_mirror`, which fails any feature that omits the declaration). This is non-negotiable, exactly like the parity check.

### What the engines do, and what it buys

For the **signed price head**, the scorers (`ic_grid`, `best_span`, `second_span_adds`, `per_exchange_vs_single`, and the OOS `gates.wf_ic`) mirror-augment by appending each anchor's reflection — feature legs via `FeatureSpec.mirror`, the signed target via negation — and scoring the union. For the walk-forward the reflection is **interleaved at the same time index** and the embargo doubled, so it shares the anchor's fold. Provable effects (verified in the tests):

- **The IC measures only the odd (direction-consistent) association.** A purely odd relationship is preserved (mirror ≈ plain); a purely even one scores ~0 (no signed signal can be manufactured); a mixed relationship keeps the odd part while the even part adds scatter and **dilutes** the IC — it is *not* cleanly subtracted out.
- **The OOS regression is forced through the origin** — the symmetric pair drives the intercept to 0, removing a directional/level bias from the fitted model.
- **No-op for the rate head** — `|·|` is sign-blind (`|−f| = |f|`, the count target is even), so the engines disable mirror whenever they score a magnitude.

**Beyond the IC — input shaping (step 2).** The same augmentation applies when *shaping* a signed feature for the network: estimate its centre/scale on `concat[f, mirror(f)]`, **not** on the raw feature. A signed feature is symmetric about 0 in the population, so any non-zero centre or skew in one block is mostly a **trend artifact** — shaping on the raw feature bakes that block's drift into the transform, and an asymmetric "fix" would break the very sign-symmetry the model relies on. Mirror-augmenting forces centre 0, skew 0, and an RMS scale. The transform is still *applied* to the raw feature in production; only its estimation uses the symmetrised data. A no-op for an even feature. (`02_finalize.ipynb` §2 does this.)

**Costs:** it *assumes* the edge is symmetric, so a genuinely **asymmetric** edge (works on rallies, not selloffs) averages away; and the reflection is deterministic, so it **adds no independent information** — use it to make the signed score direction-robust, not to claim significance.

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
- All EMAs use `α = 2/(span + 1)`. `span = 1` ⇒ `α = 1` (no memory / no smoothing) — a distinct code path on both sides and the most-used fast leg; handle it deliberately and keep it **well-defined**: finite and identical online vs offline, with any degenerate `E/W` / `0/0` resolving to a *consistent* NaN (the gates mask it), never `inf` or a build-specific number (see the `span = 1` Do/Don't in the guard rails).

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

- **Vectorized builder** `(raw_data, shared_data, config, params) -> {leg_key -> vector}` — the offline/analysis path; it imports **nothing** from `boba.research`. Each vector has length `len(shared_data.event_ts)` — ONE value per event timestamp (the feature output index; sampling onto a research eval grid happens downstream, not here). May use `lfilter` and any array trick. Causal (each value uses only data at-or-before its `event_ts`; NaN where undefined, e.g. warm-up). Divides by the yardstick vectors on `shared_data` (`vol_yardstick` / `rate_yardstick`) only when the feature carries that regime as a multiplicative scale. `params` is an opaque token (`N`, or `(n_fast, n_slow)`).
- **Streaming class** (`StreamingFeature`) — the O(1) production path, built only from the allowed EMA classes:
  - `on_book` / `on_trade` mutate internal state for one raw event — they do **not** decay, inject, or read.
  - `refresh()` runs **once per receive-timestamp**, after all that timestamp's events are applied: update each level's live front and inject each flow's one sample, then advance the trade clock **at most once** (decay; commit each level) iff a trade landed.
  - `value()` returns `{leg_key -> scalar}` — the live front at the instant it is read; its keys must match the vectorized builder's dict keys for the same `params`.
  - **Compose `boba.features.streaming.VolYardstick` / `RateYardstick`** rather than recomputing `σ_ev` / `λ_ev`, so yardstick parity is established once in the test suite and your feature's parity validates only the feature-specific maths.

---

## Price source — fresh mids vs raw sizes

A venue's raw BBO (`front_levels`) is a *slow sampler* of top-of-book on the slow feeds (byb/okx refresh only every ~10–20 ms; p90 100–160 ms). `boba.io`'s **`merged_levels`** fuses trade prints into the book — holding, per side, the **newest-by-exchange-time** price among {BBO snapshots, qualifying trades (`prc>0 & qty>0`)} — for a *fresher* top-of-book **price**. A trade updates the side it **lifts**, and that aggressor→side map is **venue-specific** (Binance *spot* inverts it — a `Bid`-aggressor is a SELL that hits the bid), so rely on the raw stream's `lifts_ask` (from `io._trade_lifts_ask`), never a hardcoded `Bid → ask`. The merge is deliberately **price-only** — it carries no `qty` (forwarding a snapshot's size is a stale-data trap) — and because the two sides fuse *independently* the fused book can momentarily **cross**, so `boba.io` **un-crosses** the stored `merged_levels` with the listing's tick (trust the fresher side, push the stale side one tick): `ask ≥ bid` ALWAYS holds in the stored stream and in `LiveMergedBook.quote`. The tick comes from `tick_sizes.toml` via `io.tick_size`, which **raises** if a fused listing has no configured tick. So choose the price source by what the feature reads:

- **Needs only prices** (a mid, a cross-venue gap, a microprice numerator): use **`merged_levels`** — *but only where the venue/instrument benefits*. The fusion is a big win for the slow feeds (byb/okx) and **hurts** an already-sub-ms feed, so it is **disallowed for bin perp** in `boba.io` (it raises; bin *spot* is slower and ~neutral, so allowed). Encode this per listing (the `mid_stream` / fuse-trades policy), never as a blanket choice.
  - **Vectorized:** just read the prepared per-listing mid `shared_data.listings[<listing>].mid` (a `(rx, value)` `Series` that `build_shared_data` derives per the `config.mid_stream` policy) — forward-fill it onto `shared_data.clock` / `shared_data.event_ts` as you need; don't re-derive it.
  - **Streaming:** **Compose `boba.features.streaming.LiveMergedBook`** (the online twin of `io._build_merged_levels`) — never hand-roll the fusion. Construct it with the per-listing fuse ticks `{listing: config.tick_size[listing]}` for the `merged_levels` listings, feed it every `on_book` / `on_trade`, and read the **un-crossed `(bid, ask)`** via `quote(listing)` (derive the mid / spread / microprice yourself). It holds the newest-by-exchange-time fuse + the tick un-cross internally, so every book-based feature shares ONE implementation and parity to `io.py`'s `merged_levels` is established once. `price_dislocation` is the worked example.
- **Needs sizes** (OFI, queue/volume imbalance, depth): `merged_levels` can't help — it has no `qty`. Read **raw `front_levels`** via `raw_data.listings[<listing>].front_levels` (a `FrontLevels` named tuple: `rx, exchange_time, bid, bid_qty, ask, ask_qty`); the sizes are snapshot-stale between refreshes, which is inherent to the data. `ofi_fast_slow` / `ofi_ema` are the worked examples.

---

## Validation

- Establish the trade-clock EMA convention against a plain one-event-at-a-time loop on a real block, the way `notebooks/03_ema_clock_validation.ipynb` does.
- Every feature's vectorized and streaming builds are tied together by `boba.research.screening.parity_check` on a real block — to float round-off. A failing parity is almost always one of the rules above broken (a stale read, a double-counted same-timestamp burst, a decay on the wrong clock, or a mis-classified flow/level).
- Beyond parity (which only proves the two builds agree), validate each feature against an **independent, dead-simple oracle** on a real block — the project's standing validation requirement. The oracle must be implementable from the feature's written definition alone, share no code with the production build, and match it to float32 tolerance.
- **Test the mirror-augmentation commutation invariant** (`mirror(feature(books)) == feature(mirror_books(books))`) — see [Mirror augmentation](#mirror-augmentation). A feature without a declared, tested `FeatureSpec.mirror` is not done.
