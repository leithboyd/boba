# Feature analysis — template

Every feature gets one analysis that follows this structure, top to bottom. Copy this
file to `notebooks/features/<feature>.ipynb` (or `.md` for a paper analysis) and fill
each section. The analysis is not done until **§4 (oracle) and §5 (hygiene gates)**
both pass — those are the two hard gates (one for *correctness*, one for *signal*); the
rest is the reasoning that earns the feature its slot and its head.

Worked example throughout (marked **Ex —**): `price_dislocation` — a cross-venue
**log-price spread** (e.g. `log(bin_mid) − log(byb_mid)`) run through a **two-EMA
oscillator** `(EMA_fast − EMA_slow)`, normalised by a shared vol yardstick `σ_ev`. A
price-head feature that *also* feeds the rate head, chosen because it exercises every
section (the level-flavour EMA, the best-fresh-price-per-listing choice, one shared
`σ_ev` for feature *and* target, head routing, a 2-parameter feature **family**, and all
the gates — including normalizer stability).

References this leans on: `CLAUDE.md` (the oracle rule), `docs/feature_hygiene.md` (the
gates), `docs/ema_event_clock.md` (the clock + EMA convention), `docs/nn_features.md`
(heads, scales, the L/T set), and the data quirks in `src/boba/io.py`.

---

## 0. Metadata

A one-glance header: what the feature is and where it plugs in.

| field | value |
|---|---|
| **name** | the column/feature id |
| **one-line** | what it measures |
| **target head** | rate / vol / direction(price) / value — which head consumes it (`docs/nn_features.md`) |
| **event clock** | merged-trade / per-venue-trade / mid-move / wall-clock |
| **scales N** | the multi-scale set (e.g. {10, 50, 100, 1000}) |
| **L / T** | listings used / target listing |
| **kind** | flow / level / ratio (drives the EMA flavour, `docs/ema_event_clock.md`) |
| **date / author** | |

**Ex —** name `price_dislocation_{listing}(N_fast, N_slow)` (keyed by **listing** — `bin_eth_usdt` and `bin_eth_usdt_p` are distinct books); one-line "two-EMA oscillator on the `listing`-vs-target log-mid spread, in shared σ_ev units"; head **both** — direction (price head, *signed*) + intensity (rate head, *|·|*); clock **merged-trade**; **family** `N_fast < N_slow` (the single-EMA case is `N_fast=1`); mids **best-fresh per listing** (bin→front_levels, byb/okx→merged_levels); L `{bin,byb,okx}_eth_usdt_p`, T `byb_eth_usdt_p` (byb is both spread reference and target); kind **level**.

---

## 1. Hypothesis & mechanism

State, in one paragraph: **what the feature predicts, the microstructure mechanism, and
the explicit falsifier.** No mechanism ⇒ no feature. Name the scale/horizon the
mechanism should live at (so the gates know where to look).

**Ex —** *Hypothesis:* when a venue richens vs byb (the log-spread sits above its running
EMA) byb **catches up**, so the basis mean-reverts and the dislocation predicts byb's
forward return *positively*. *Mechanism:* the leader (bin) leads (nb03: bin→byb ≈ 75 ms);
the slow venue's quoted mid trails, so a persistent spread gap is a catch-up signal.
*Lives at:* this is a deviation-from-a-stable-basis signal, so it **strengthens with
larger N** (a longer reference EMA) — the opposite of small-N order-flow. *Falsifier:*
~0 IC with byb's forward (σ-normalised) return at every scale, or signal only via the
regime level (a leak).

---

## 2. Definition (exact, causal, dimensionless)

The precise computation, written so a stranger could reimplement it. Must be:
- **Causal** — uses only data at-or-before `t`.
- **On a named event clock** with the EMA convention from `docs/ema_event_clock.md`
  (`α = 2/(N+1)`, gap-power decay, flow vs level flavour).
- **Dimensionless / regime-form** where possible — for prices use the **log-difference**
  (geometric, scale-invariant), and normalise by a **shared vol yardstick `σ_ev`**: the
  *same* `σ_ev` that normalises the forward target, so feature and target share σ-units.
  (`dimensionless ≠ regime-safe`; §5 still gates it.)
- **Pick the price stream per listing.** Use the freshest *valid* mid: `merged_levels`
  where it helps (byb/okx slow feeds), `front_levels` where `merged_levels` is blocked
  (bin perp — its BBO is already sub-ms; see `io.load_block`). A relative feature also
  needs a **reference** listing (here byb, the target).
- **Watch the data traps** in `io.py`: drop `prc=0,qty=0` prints; and if a feature uses
  trade *direction*, take it from `_trade_lifts_ask` (**Binance spot inverts** `Bid`/`Ask`).

**Ex —** per **listing** `ℓ` (full token — spot and perp are distinct), on the merged
clock:
- log-spread (a **level**): `sₗ = log(mid_ℓ) − log(mid_byb)`, mids forward-filled to each
  merged tick from the best-fresh stream per listing.
- two level-flavour EMAs of it: `EMA_{N_fast}[sₗ]`, `EMA_{N_slow}[sₗ]` (`α=2/(N+1)`).
- shared yardstick: `σ_ev` = trailing RMS of **byb's** per-merged-tick mid-return over
  `VOL_LOOKBACK` events (computed once; also normalises the target).
- **feature** `price_dislocation_ℓ(N_fast, N_slow) = (EMA_{N_fast}[sₗ] − EMA_{N_slow}[sₗ]) / σ_ev`
  — a 2-EMA oscillator on the basis, in σ-units. `N_fast=1` (EMA = the latest value) is
  the single-EMA special case `sₗ − EMA_{N_slow}[sₗ]` — the *noisy* high-frequency edge of
  the family. Inputs are the per-listing dislocations `{bin, okx}` vs byb.

---

## 3. Streaming construction

The production realization: an `IFeatureBuilder` that maintains the feature in **O(1)
state per (scale, listing)** from the event stream — no buffers (`docs/ema_event_clock.md`
§7–8). This is what ships; the notebook's vectorized version is just for analysis and
must match it (§4).

```
on_price(listing, e):     spread[listing] = log(mid(e)) − log(mid_byb)         # held level, no clock tick
on_trade(e):              c += 1                                                # only trades tick the clock
                          sigma.update(c, byb_tick_return)                      # shared σ_ev
                          for listing,(Nf,Ns): fast[…].decay_to(c); slow[…].decay_to(c)   # two level EMAs
emit():  for listing,(Nf,Ns):
                          dislocation = (fast[…].value(c) − slow[…].value(c)) / sigma.value(c)
```

State = **two** level-flavour `ScalarEMA`s of the spread (fast + slow) per (listing,
member) plus the single shared `σ_ev`. (Flow-type features use the flow flavour instead —
inject on the event, decay to zero — as in an order-flow feature.)

---

## 4. Oracle validation — HARD GATE (correctness)

Per `CLAUDE.md`: the production/vectorized feature must match an **independent,
dead-simple reference** to float tolerance **on a real block** (real blocks have the
arbitrary-ns, multi-venue interleaving synthetic fixtures can't exercise). Ship the
**trio**: (a) blind oracle — an explicit loop, no shared code; (b) synthetic property
tests; (c) the real-block diff (skipped when `DATA_DIR` unset). See `tests/
test_io_merged_levels.py` and the nb02/04 oracle cells for the pattern.

**Ex —** oracle: a Python loop over the merged stream maintaining the spread's EMA by
hand (`ema = decay·ema + α·spread`), `dislocation = (spread − ema)/σ_ev`, compared
bit-for-bit to the `lfilter`/`ScalarEMA` version on a real ETH-perp block. Property test:
a constant spread → dislocation → 0; the sign flips under `spread → −spread`. *Result:*
[paste max abs diff = 0].

---

## 5. Feature-hygiene gates — HARD GATE (signal)

Run **every gate** in `docs/feature_hygiene.md` and tabulate. These are analyses, not
hard pass/fail — but a feature that fails the leak or incremental gates ships only with
an explicit, written justification. Base `[cr,cv]` = causal rate+vol momenta on the
merged clock; level controls `[lv,lr]` = absolute rate/vol levels.

| # | gate | what it shows | result (Ex, illustrative) |
|---|---|---|---|
| 1 | causal | no lookahead | ✓ (clock ticks ≤ t) |
| 2 | marginal value | incremental IC over `[cr,cv]` for the head's target | +IC vs byb fwd (σ-norm) return, grows with N |
| 3 | **leak** | gain *survives* adding `[lv,lr]`; `corr(f, lv/lr)≈0` | clean (`σ_ev`-normalised ⟂ vol/rate level) |
| 4 | stratified leak | adds *within* level deciles | survives in-bucket |
| 5 | normalizer stability | feature std ~flat across vol deciles | the shared `σ_ev` holds the scale (max/min ≈ 1.4) |
| 6 | incremental & collinear | adds over the *existing* set; not a restatement | bin vs okx partly redundant (both track byb's basis) |
| 7 | temporal stationarity | low per-day drift, not just decile-flat | [per-day CV] |
| 8 | OOS across a regime gap | IC holds train→test in different regimes | [report] |

Connect to the model: this basis signal **strengthens with N** (deviation from a stable
basis), so expect value at N≈500–5000 rather than small-N. Report the IC-vs-N curve.

**Visualise how the conditional distribution shifts**, not just the IC scalar. Bucket the
anchors by the feature and show the **outcome distribution each head models**: the
forward-**return** distribution *tilting* with the signed feature (price head — a
location/skew shift on top of the no-move spike), and the **move-count** distribution
*shifting up* with `|feature|` (rate head). A shift you can see is stronger evidence than
one IC number — and it's exactly the object each head is trained to predict.

**Then visualise the feature's *own* distribution to choose its NN-input normalisation** —
a step *distinct* from the `σ_ev` regime-normalisation already baked into the definition.
Plot the feature, and compare candidate transforms (z-score, robust-scale + clip, arcsinh,
rank-Gaussian) by how Gaussian *and* bounded they make it (a QQ-plot vs N(0,1) makes the
choice visual). Pick the **lightest adequate** one: a symmetric, mild-tailed feature needs
only a z-score (or robust + clip to bound rare outliers); heavy skew/kurtosis is what
pushes you toward arcsinh or a rank transform. The plot *is* the decision.

---

## 6. Disposition (and head routing)

State the call and the evidence, using the dispositions in `docs/feature_hygiene.md`:
**keep** (which head(s), which scales) / **drop** (leaks / clean-but-empty / collinear) /
**route to a finer-scale or faster model** / **handle the level outcome-side**.

**Head routing — test against BOTH heads' targets**, not just the one the feature was
designed for. The **price head** wants *direction* (the signed σ-return); the **rate
head** wants *intensity* (the forward mid-move count = the subordination `N`). Correlate
the **signed** feature with the return and the **magnitude** `|feature|` with the count —
a feature can serve both, often at different scales.

**Ex —** **keep, in BOTH heads, as a small family** (each head picks its corner of the
`(N_fast, N_slow)` grid). *Price head (direction):* **signed**, a *smoothed* fast leg over
a slow leg (≈ `N_fast 10–50`, `N_slow 2000–5000`) — the instantaneous `N_fast=1` is worse
here, its single-tick noise dilutes direction; leading listing `bin_eth_usdt_p`. *Rate
head (intensity):* `|·|`, the *instantaneous* fast leg / faster slow leg (`N_fast=1`,
`N_slow≈100`) — intensity keys off the immediate gap, so smoothing hurts. Signed vs the
count is ~0 (symmetry), confirming the split is real. Keep a few non-redundant members per
head; pool bin/okx; both heads share the `σ_ev` yardstick.

---

## 7. Ship checklist

- [ ] streaming `IFeatureBuilder` class implemented (O(1), no buffers)
- [ ] oracle trio in `tests/` (blind oracle + synthetic + real-block diff), suite green
- [ ] hygiene gate table filled, leak/incremental/normalizer justified if failing
- [ ] head routing recorded in `docs/nn_features.md` (which head(s), signed vs `|·|`, scales); `σ_ev` lookback matches the target normalization
- [ ] data traps handled (zero-prints; best-fresh price stream per listing; aggressor sign if signed)
- [ ] (if the analysis made a surprising/contested claim) adversarially reviewed, like nb05
