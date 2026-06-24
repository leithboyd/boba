"""Generate notebooks/features/range_breakout_v2.ipynb — a feature-analysis notebook for the
EMA-based Bollinger-style range breakout, on the CURRENT template (build_feat_nb.py) methodology."""
import json
from pathlib import Path

cells = []
def md(s):   cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r"""
# Feature analysis — range breakout

This notebook is two things at once: a **method** for analysing a trading feature, and a
**worked example** of that method on one feature. The text explains what to do and why; the
code does it.

We build features for a model that forecasts one exchange's mid-price about 100 ms ahead.
Three crypto exchanges appear throughout — **byb** (Bybit), **bin** (Binance), **okx** (OKX) —
and **byb is the target**: the one we predict. ("Mid-price" = the midpoint between the best
buy and best sell quote.) The next section explains the model these features feed.

To analyse a new feature, copy this notebook and change the parts specific to the feature:
**what it is and why** (§1), **its exact definition** (§2), and **its builder** — the
`range_breakout` function in §3. The data loading, the checks, and the plots adapt on their own.

**The example** is `range_breakout`: a **Bollinger-style breakout** computed on **each venue's own
mid**, entirely from EMAs (no rolling max/min, no fixed window). Build a slow EMA band around the
mid — its centre `ema(mid,N)` and width `σ_band = √(ema(mid²,N) − ema(mid,N)²)` — and read where
the mid sits inside it as a σ-normalised **z-score** `z = (mid − ema(mid,N)) / σ_band`. A large `z`
says price has extended; whether that continues (momentum) or reverts is decided by the IC sign,
per span and venue. It is a **price-head (direction)** feature, and we check the rate head too.

**A feature's *analysis* is done when two checks pass:**
- **The parity check (§4)** — a second implementation reproduces the feature on real data, so the code really computes what we think it does.
- **The hygiene gates (§5)** — the signal is real and holds in any market, not just an echo
  of "the market is volatile right now."

Everything after that decides *which part of the model* the feature feeds, and *at what
time-scale* — and the §10 checklist (streaming builder, tests, multi-block OOS) gates **shipping**: analysis-done is not ship-done.
""")

md(r"""
## The model these features feed: two heads

A feature is only worth something if it helps the model predict, so it's worth knowing what
the model does. We forecast how byb's mid-price moves over the next ~100 ms, and we split
that into two simpler questions — the two **heads**:

**Price head — which way and how far?** Over the next few price-moves, what is the *signed*
move (its direction *and* its size, together)? The head predicts the whole distribution of
that move, in units of byb's recent **volatility** — the **volatility yardstick** `σ_ev`
(the exp-weighted RMS of byb's *actual* mid-moves) — so the head's target is
`price change ÷ σ_ev`.

**Rate head — how many moves?** Busy markets pack many price-moves into the window, quiet
ones few. This head predicts the *count* of moves over the next 100 ms as a distribution,
measured against the recent pace — the **rate yardstick** `λ_ev` — so its target is
`count ÷ λ_ev`.

**What's a yardstick?** A causal, trailing estimate of byb's *volatility* (`σ_ev`) and move
*rate* (`λ_ev`), from past data only. A regime gauge, nothing more. When a feature *carries* that regime and you want it gone, dividing
by the yardstick is **one option** for removing it (only *when applicable* — you can't always
divide the regime out of a feature). Both are EMAs **decayed on the trade clock** (`α = 2/(span+1)`)
but **updated between trades** — they react to every byb mid-move, so they read live at every
instant. They use one fixed span, `YARDSTICK_N` (any feature may use that span too). (`σ_ev` is
the exp-weighted RMS of byb's *actual* mid-moves — read as an `E/W` ratio so the many non-move
trades cancel out; `λ_ev` is a ratio of two EMAs — the exp-weighted byb move-count `W` (the same
`W` that is `σ_ev`'s denominator) ÷ the exp-weighted seconds-per-trade = byb's moves per second.) Like every average here, they live
on the trade-tick clock — never wall-clock or a hard window.

**Why split into two heads?** A move over a window is just *how many* little moves happen
times *how big* each one is. Pulling "how many" (rate) apart from "how big" (price) lets each
head learn a steadier thing — and is why there are two yardsticks, one per head.

**Putting them back together.** The price head isn't a single distribution — it's a *family*,
one per possible move-count: "if `k` moves happen, here's the spread of the total price
change" (call it `D_k`). The rate head gives the probability of each count, `P(K = k)`. The
100 ms move is the two **mixed** — for every count `k`, take the price head's `k`-move
distribution, weight it by the rate head's probability of exactly `k` moves, and sum:

`distribution of the 100 ms move  =  Σ_k  P(K = k) · D_k`

More moves → a wider spread, so the rate head's "how many" sets the scale and the price head's
`D_k` sets the shape. This is why a feature that predicts the *count* (rate head) and one that
predicts the *per-move direction* (price head) are both useful: they feed the two factors that
multiply together.

**Backed by research.** This "how many × how big" split is the classic **subordination** model
of asset prices (Clark, 1973, *Econometrica*; Ané & Geman, 2000, *Journal of Finance*): returns
over a fixed clock-time window look messy, but become well-behaved once you condition on the
*number* of events. So modelling the event count and the per-event move separately, then mixing
them, is a principled decomposition — not just a convenient one.

**Where `range_breakout` fits.** A σ-normalised breakout has a **sign** (the mid is above or
below its EMA band) — so it is, first and foremost, a **price-head** (direction) feature: a
large `z` says price has extended, and the head learns whether that extension *continues*
(momentum) or *reverts* — **the IC sign decides, per span and per venue, measured not assumed**.
Its *magnitude* `|z|` is a clean "how stretched is price" gauge, so we also check it against the
**rate head** as a diagnostic — a big extension may precede a burst of catch-up moves. Both
checks are **marginal screens** — does the feature carry signal worth feeding — **not** measures
of distributional fit: the model's actual targets are the count distribution `P(K = k)` and the
count-conditioned price family `D_k` mixed above, fit downstream, not what these rank-ICs measure.
""")

md(r"""
## Guard rails — do this, not that

Hard rules, learned the hard way. Follow them unless you have a specific, written reason not to.

**Don't**
- **Don't assume a fixed leader.** No exchange always leads — leadership moves around. Build
  the feature the same way for every exchange and keep them all.
- **Don't pick "the best exchange"** by average score — that throws away the moments another
  exchange leads (often the most informative ones). Keep all; let the model weight them.
- **Don't pool exchanges by default.** Pooling can blur the cross-exchange structure. `range_breakout`
  is a genuine **per-venue** number (each venue's own mid against its own band), so the choice is
  intrinsic — keep every venue; §9 covers the per-exchange/cross-venue structure.
- **Don't normalize reflexively.** `range_breakout` is **already σ-normalised by its own band
  width** `σ_band` — a z-score is a ratio, dimensionless and comparable across markets — so it
  may not need a yardstick division at all. We **measure** its regime-invariance (Gate A, §6) and
  only normalise if it fails (it doesn't — see §2's decision).
- **NEVER a rolling/boxcar window.** A "range" or "breakout" here is **EMA-based only** — the
  band centre and width are trade-clock EMAs of the mid and mid². *No* Donchian rolling max/min,
  *no* fixed-N lookback buffer, *no* wall-clock window. That is non-negotiable and is exactly
  what keeps the state **O(1)** and the read causal + online-bit-exact.
- **Don't trust a correlation** until it survives the regime controls (rate and vol) — else
  it may just be re-reporting "the market is volatile."
- **Don't ship a feature without the §4 parity check** — a second, production-style build reproducing it on real data.
- **Don't hand-roll a streaming EMA, and don't use `EventEMA` — use only `KernelMeanEMA` or `LiveFrontEMA`.**
  Every *online* average — the §4 production/streaming parity build, and any event-by-event EMA state anywhere
  in the notebook — MUST be exactly one of **two** `boba.ema` classes: **`KernelMeanEMA`** (the self-normalising
  `E / W` read — for a **flow**) or **`LiveFrontEMA`** (the live-front read — for a **level**). Both expose
  **separate `tick()` (decay) and `add()` (inject)** calls — that separation is the whole point: decay rides the
  shared trade clock (once per trade-timestamp) while injection rides each relevant-event timestamp, and the two
  must fire **independently** (see the inject-≠-decay rule below). **Never use `EventEMA`:** its single `step()`
  **couples decay and injection** into one call, so it *cannot* decay on the trade clock while injecting on a
  different event stream — that directly violates the inject-once / decay-once-independently rule. And **never**
  hand-roll a scalar EMA (no private `_ScalarEMA`, no `(1−α)·s + α·x` per-event loop) or reach for
  `scipy.signal.lfilter` in the streaming build. Every quantity here is either a **flow** (→ `KernelMeanEMA`) or
  a **level** (→ `LiveFrontEMA`); if you think you need anything else, you've mis-modelled it (a slope/covariance
  is a ratio of flow EMAs — several `KernelMeanEMA`s, not a bespoke class). §4 exists to validate the **actual
  production code** (`boba.ema` online vs §3's vectorized path), not a throwaway re-implementation that could
  carry the same blind spot. (The §3 *vectorized* analysis may still use `lfilter` — the offline array path, not
  online EMA state.)
- **Don't peek ahead.** Every value uses only data at-or-before its own timestamp.
- **Don't over-transform for the network.** Pick the lightest reshaping that works.

**Do**
- **Do start with a falsifiable hypothesis** — a mechanism for why it should work, and what
  would prove it wrong.
- **Do make every average a trade-tick EMA.** *Every* smoother in the pipeline — both band legs
  per venue (`ema(mid)`, `ema(mid²)`), `σ_ev`, `λ_ev` — is an exponential moving average
  **decayed once per tick of the one shared trade clock** (one tick per trade-timestamp across **all** venues; simultaneous prints count once; `α = 2/(span+1)`) yet **taking in a fresh value at most once per timestamp that carries a relevant event** (a fresh mid for its venue, a byb move — same-timestamp events as one update, never stale between trades), **never** a wall-clock
  average and **never** a hard/boxcar window. That keeps the whole state **O(1)** (one scalar
  per average, no ring buffers) and bit-identical online vs offline. Establish and validate the
  convention against a plain one-event-at-a-time loop on a real block — exactly as
  `notebooks/03_ema_clock_validation.ipynb` does, and as the §4 parity check re-checks here.
- **Do make every EMA react to relevant events as they land — never let it read a stale state.** An EMA's
  *decay* lives on the trade clock, but it must *update its value* at **each timestamp that carries a
  relevant event** (same-timestamp events aggregated to one update): each band leg refreshes its front to
  that timestamp's mid; `σ_ev` pushes a sample on a byb move. A value that ignores the timestamps since the
  last trade is **stale** — wrong, and a silent source of lag. The choice of EMA *and when it updates* is
  what makes the inter-trade read well-behaved — pick both deliberately (see the "Choosing the EMA" section).
- **Do inject at most one EMA sample per timestamp, and decay once per shared-trade-clock tick — this
  never changes.** At each timestamp an EMA injects **one** sample *iff* a relevant event for that EMA
  lands there: for a band leg, a book update on its own venue; for `σ_ev`, a byb mid-move. A timestamp with only
  **irrelevant** trades injects nothing for that EMA, yet it still
  **ticks** (decays) once *iff* a trade lands, on the one **shared** trade clock
  (one tick per trade-timestamp across **all** venues, simultaneous prints counting once). Inject and
  decay are independent and neither ever fires more than once per timestamp, whatever the feature.
  Records sharing a timestamp are **one** sample, not a sequence: same-timestamp aggregation sets only
  that sample's **value** — never the number of samples, never the number of ticks. *What* value depends on the quantity: a **level**
  read takes the *last* state (mid, microprice — the band legs are this); a **flow** **sums** the records at that timestamp
  (trade volume; `σ_ev` sums its return²). So a backed-up burst of N book updates at one
  instant is still **one** weight-1 sample — value = the final mid — **never** N
  samples. Injecting N silently overweights exactly the instants where a feed stalled and dumped a burst
  (N× weight on the band leg at one timestamp). The level-read face of the same trap: `σ_ev`
  counts squared mid-*moves*, so N simultaneous records fabricate N−1 phantom moves and inflate volatility.
- **Do always read the *freshest* value — never freeze on the last trade's.** Each band leg decays once
  per trade-timestamp (that fixes its memory in trade-count), but its *read* uses the latest mid:
  `read = (1 − α)·committed + α·current_mid`, current at every instant between trades. Keep the two clocks
  separate: refresh the *read* at each book-update **timestamp**, but keep the *decay/weighting* on the
  trade clock — push a fresh *sample* once per book-update **timestamp** (one per timestamp, not per message).
- **Do read a sparse flow as `E / W`.** `σ_ev`/`λ_ev` are byb's own mid-moves — present on only
  *some* events — so each is **two** trade-tick EMAs (`E`, `W`) read as the ratio `E / W`, which
  cancels the in-between decay and the warm-up bias. (`range_breakout`'s band legs are a *level*,
  not a sparse flow, so they use the live-front read, not `E/W`.)
- **Do test against both heads — but feed both the *signed* feature.** Check whether the signed
  `z` predicts *direction* (price head) and whether `|z|` predicts *intensity* (rate head). Those
  checks are diagnostics — in the model, feed the **signed** `z` to *both* heads. The rate head
  will learn the magnitude itself, and learn how a +breakout on one venue cancels a −breakout on
  another. Pre-taking `|·|` per feature destroys the sign and that interaction.
- **Do score out-of-sample** with a purged, embargoed walk-forward (strictly past→future) — a single split is only a faster screen.
- **Do use the freshest valid price per exchange.**
- **Do treat a feature as a family across time-scales** and let the data assign scales to heads.
- **Do prove regime-invariance with Gate A — never assume it.** A usable feature reads the *same* in calm and
  wild markets: its distribution must be **stable against the regime diagnostics we track** (not a proof of independence). Gate A's **control-free** checks (§6):
  **scale** (std stable across vol buckets, max/min < ~3); and — for **both the signed feature and its
  magnitude `|feature|`** (the rate head sees the signed feature but can learn its magnitude) — that it does **not track** the regime
  (`|IC(·, vol & rate level)| ≈ 0`, the monotone test) **nor leak non-monotonically** (per-decile-mean
  **dispersion** small, which catches a U-shaped leak the monotone IC misses). All are **hard** gates. A raw
  **level** (`σ_ev`, `λ_ev`, a mean trade size) usually *is* the regime and fails them — but *measure, don't
  assume*: a **ratio / bounded / normalised** form may pass (a z-score *should* be invariant — it divides by
  its own band width — but we measure it). Never call a feature regime-invariant *or* not — nor a level useless — until
  **every** Gate A number says so. (Corollary: don't assert *any* property — invariance, a lead/lag, "it's
  circular" — you haven't measured on a real block.)
- **Don't fuse the two gates — they're independent.** *Regime invariance* (Gate A) is the feature's own
  distribution being stable across regimes; *signal* (Gate B) is what it predicts over the **invariant**
  controls. The raw vol/rate **levels aren't Gate-B controls** (they aren't valid alpha features) — putting them in
  the signal test just smuggles the Gate-A regime test back in; they serve only as the Gate-A regime *coordinate*. And **a control can be a valid feature**: when the
  feature under test *is* a regime descriptor, marginal-over-its-own-controls is circular — judge it on its
  **standalone** signal, and never call it "redundant" from its algebra alone.
""")

md(r"""
## 1. What the feature is, and why it might work

Every feature starts with a one-line summary and a **hypothesis you could be wrong about**.
If there's no story for *why* it should predict anything, don't bother testing it.

| | |
|---|---|
| **what** | how far a venue's mid has stretched from its own EMA band, in band-σ units (a Bollinger z), over two time-scales |
| **feeds** | both heads — *direction* (price head: signed `z`) and *intensity* (rate head: `|z|`); both are fed the *signed* `z` |
| **predicts** | byb's mid-price 100 ms from now |

**The idea.** For each venue, take its mid and build a slow EMA **band** around it:
a centre `ema(mid,N)` and a width `σ_band = √(ema(mid²,N) − ema(mid,N)²)` (the EMA standard
deviation). The breakout is the σ-normalised distance of the mid from the centre:

`z = (mid − ema(mid,N)) / σ_band`

A large positive `z` means the mid has pushed well above its recent local mean *relative to how
much it normally wanders*; a large negative `z`, well below. `range_position = tanh(z)` bounds
that to (−1, +1); `breakout_magnitude = z` keeps it unbounded for the price head.

**Why it might work — and the two-sided hypothesis.** A fresh breakout can mean one of two
opposite things, and **which one is an empirical question we let the IC sign answer, per span and
per venue**:
- **Continuation (momentum):** an extension is the leading edge of a directional move — price
  broke out because information arrived, and it keeps going. Then `z` predicts byb's next move
  with the **same** sign.
- **Reversion:** an extension is an overshoot — liquidity was briefly thin, price stretched, and
  it snaps back. Then `z` predicts byb's next move with the **opposite** sign.
Short-horizon breakouts (small N) tend toward reversion (microstructure overshoot); regime
breakouts (large N) tend toward continuation (a real trend). We do **not** assume which — we
sweep N and read the IC sign off the data.

**Why a *cross-venue* leg.** If okx breaks out a beat before byb (one venue ticks first), then
okx's `z` predicts byb's catch-up. So we compute each venue's breakout against **byb's** future:
byb's own leg (does byb's breakout predict byb?) and the cross legs (does okx/bin breaking out
lead byb?). Keep all; let the model weight whoever's leading.

> **The byb-own leg is echo-prone — flagged up front.** byb's *own* breakout numerator `mid − ema(mid)`
> shares the **anchor mid** with the target (`log(mid_fwd / mid_now)` opens at that same mid). It is still
> strictly causal — no future leak — but a byb breakout `z` is mechanically large *because* byb just moved,
> so much of its raw IC is the **contemporaneous echo**, not a forward prediction. We measure this directly
> (backward vs forward IC, and the echo-netted partial IC, in the echo-netting section) and **report the
> echo-netted number**, treating the byb-own leg as suspect. The **cross legs** (okx/bin → byb) are the ones
> where "lead" is even coherent.

**What would disprove it.** No predictive power at any time-scale, or power that vanishes
once we account for how volatile the market is (meaning it was only tracking volatility) — or, for the
byb-own leg, a forward edge that disappears once the move already underway is netted out (pure echo).

> **No fixed leader.** It's tempting to call one exchange "the leader." Don't — leadership
> moves around, and any exchange can lead at any moment. So build the feature *the same way
> for every exchange and keep them all*; never hard-code one as the leader.
""")

md(r"""
## Choosing the EMA

Every smoother here is an EMA on the **trade clock**: its **decay** steps once per *trade-timestamp*
(`α = 2/(span+1)`), but its **value updates at most once per timestamp that carries the EMA's own
relevant event** (same-timestamp events as one update). If that relevant event is a book update on the
band leg's venue, the read refreshes; if a trade lands, the shared clock ticks. Choose the EMA type and
injection rule for the statistic being measured.

**Choice 1 — which EMA, and how you read it** (`boba.ema`):

- A **forward-filled level** — defined at every instant (a price, a cross-venue gap): use
  **`LiveFrontEMA`**, which reads the committed mean carried one step toward the freshest value,
  `(1 − α)·committed + α·latest` — current between trades, never frozen on the last trade.
  **`range_breakout`'s two band legs per venue** (`ema(mid)`, `ema(mid²)`) are exactly this — the mid
  has a value at every instant, so both are `LiveFrontEMA`s read on the live front.
- A **sparse flow** — present on only *some* events (byb's mid-moves): use **`KernelMeanEMA`**, the
  self-normalising `E / W`. The yardsticks `σ_ev`/`λ_ev` are this; the breakout's legs are **not**.
- They share the `tick` / `add` / `value` interface, so you A/B the two reads by swapping the class.
  (`LiveFrontEMA` composes a plain **`EventEMA`** — `step` / `value`, no ratio — for its committed part.)

**Choice 2 — *when* you push a value in** (the injection clock — a *separate* decision from the decay
clock). Decay is always once per trade-timestamp; injection is only for timestamps carrying the
quantity the EMA is measuring. The allowed patterns are:

| pattern | what it weights / measures | use for |
|---|---|---|
| commit every trade-timestamp; read refreshes each book-update timestamp | by **trade-dwell** — a value spanning more trades counts more | a level (the band legs: `ema(mid)`, `ema(mid²)`) |
| only on the feature's real event, skip irrelevant clock ticks | a **per-real-event mean** | a flow (`σ_ev`: push `r²` only on a byb move) |

Classify the timestamp for the EMA being updated. The band legs **commit** the current mid (and mid²)
once per trade-timestamp (trade-dwell weighting — the right average for a level) and **refresh** the live
front on every book update, so the read is current between trades. For `σ_ev`, a trade-only timestamp decays
the clock and injects no sample.

Between trades:

- **React at every relevant timestamp, and read the freshest value.** Decay rides the trade clock, but each
  band leg *updates* on every book update for its venue, and the read reflects the timestamps since the last
  trade — never a stale snapshot.
- **Records sharing a timestamp are ONE update, not a sequence** — apply them all, then register a single
  update (one mid, one move, one refresh).
""")

md(r"""
## 2. The exact definition

A definition has to get three things right.

**Causal** — it uses only data up to the moment it's computed; no peeking at the future. (A stray
forward-fill is the usual way to break this by accident.) The band legs read the mid *at-or-before*
each anchor; the band is built only from the past.

**Measured on the trade clock** — markets alternate busy and quiet, so we count progress in *trades*
(one tick per trade-timestamp on any exchange — simultaneous prints are one tick) rather than in clock-seconds; that keeps the feature's meaning
steady whether trading is fast or slow. Book updates between trades refresh what each EMA *reads*, but
they do **not** advance the clock — only a trade does, decaying each leg once. **All venues share this one
clock.**

**Comparable across calm and wild markets** — and here is the key choice for *this* feature: a Bollinger
**z-score divides the mid's distance from its centre by its own band width `σ_band`**, so it is *already*
σ-normalised — a dimensionless ratio that should read the same in calm and wild markets **without** a
yardstick division. The guard rail says *measure, don't assume*: we **build `z` raw** and measure its
regime-invariance with Gate A in §6. **It passes** (the scale across vol buckets is well under the 3× gate),
so **we ship it raw** — no `/σ_ev`, no `/λ_ev`. (Had it failed we would have tried dividing by the model's
yardstick and kept the invariant form; it did not need it.)

Put the three together — a slow EMA band on each venue's mid, and the σ-normalised distance of the mid
from the band centre:

**2a — The shape** (the whole idea, in one line):

`σ_band = √( ema(mid²,N) − ema(mid,N)² )`,   `z = (mid − ema(mid,N)) / σ_band`

`range_position = tanh(z)` (bounded, where-in-band), `breakout_magnitude = z` (the price-head feature).
Stop here and you have the feature — the rest is making each piece correct.

**2b — The EMAs we use, and how they update** (applying the two choices from the section above):

- **The two band legs** (`ema(mid,N)`, `ema(mid²,N)`) are a **forward-filled level** — the mid and its
  square have a value at every instant — so each is a **`LiveFrontEMA`** read as a **live front**:
  `(1 − α)·(value committed at the last trade) + α·(the current value now)`. We **commit** the current mid
  (and mid²) once per trade-timestamp (trade-dwell weighting, the right average for a level) and **refresh**
  the live front on every book update, so the read is current between trades, never frozen. The band variance
  `σ_band² = E[mid²] − E[mid]²` is the standard EMA variance identity; we clamp it at 0 against tiny negative
  round-off. (Using the *same* live mid in the numerator `mid − ema(mid)` and inside the band keeps `z`
  self-consistent and reacting the instant the mid moves.)
- **`σ_ev` and `λ_ev`** (the yardsticks, for the §5 controls and the price-head target) are a **sparse
  flow** — byb moves on only *some* events — so each is a **`KernelMeanEMA`** read as `E / W`: push
  `byb_return²` (weight `1`) only on a real byb mid-move, decay every trade, read `σ_ev = √(E/W)`. They are
  **not** used to normalise the breakout itself (it is self-normalised by `σ_band`).

Both reads obey the two between-trade rules. §3 builds exactly this; §4 re-derives it from raw events and
confirms it bit-exact (band legs) / to float32 (the derived z, where the variance subtraction cancels).

The cell below loads the data, builds the shared trade clock, and computes **both yardsticks** as `E/W`
flows on that clock, span `YARDSTICK_N`.
""")

code(r"""
import numpy as np, polars as pl
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.stats import spearmanr
from boba.io import list_blocks, load_block

COIN        = "eth_usdt_p"
TARGET      = "byb_eth_usdt_p"                    # the exchange we predict
EXCHANGES   = ["byb", "bin", "okx"]              # EVERY venue gets its OWN breakout leg; byb's own + the cross legs all predict byb
# freshest mid per exchange. front_levels = raw best-bid/ask snapshots; merged_levels = those snapshots fused
# with trades for a fresher mid. byb/okx use merged_levels; bin MUST use front_levels — merged_levels is DISALLOWED for
# bin perp in boba.io (it raises). This is a policy, not a tuning choice.
MID_STREAM  = {"bin": "front_levels", "byb": "merged_levels", "okx": "merged_levels"}
SHORT       = [50, 200, 1000]                    # SHORT band spans (short-horizon breakout) — swept on the §6 sweep
LONG        = [2000, 8000, 30000]                # LONG band spans (regime breakout) — swept too; this feature is ONE span N per leg (not a fast/slow pair)
HORIZON_NS  = 100 * 1_000_000                    # how far ahead we predict (100 ms, in nanoseconds)
YARDSTICK_N = 10000                              # the ONE span for BOTH yardsticks (σ_ev, λ_ev): an EMA on the trade-tick clock (α=2/(N+1)) — fixed, controls + targets only
block       = list_blocks(TARGET, "front_levels")[0]   # one ~24h slice of recorded data
SPANS       = SHORT + LONG                       # every band span we evaluate (one breakout leg per span, per venue)

# load each exchange's mid-price stream (rows already arrive in time order) — EACH feeds its OWN breakout band
def load_mid(ex):
    df = (load_block(block, f"{ex}_{COIN}", MID_STREAM[ex]).select("rx_time", "bid_prc", "ask_prc").drop_nulls())
    return df["rx_time"].cast(pl.Int64).to_numpy(), (df["bid_prc"].to_numpy() + df["ask_prc"].to_numpy()) / 2
mids = {ex: load_mid(ex) for ex in EXCHANGES}

# the shared trade clock: one tick per trade-TIMESTAMP across ALL venues. Simultaneous prints (one order sweeping levels) are ONE event -> ONE tick.
trade_ts = []
for ex in EXCHANGES:
    td = (load_block(block, f"{ex}_{COIN}", "trade").select("rx_time", "prc", "qty")
          .filter((pl.col("prc") > 0) & (pl.col("qty") > 0)))         # drop bad prc=qty=0 prints
    trade_ts.append(td["rx_time"].cast(pl.Int64).to_numpy())
trade_prints = np.concatenate(trade_ts)
merged_ts = np.unique(trade_prints)                               # collapse same-timestamp prints: at most one decay per timestamp
n_ticks = len(merged_ts)
print(f"shared trade clock: {n_ticks:,} ticks (timestamps) from {len(trade_prints):,} trade prints")

def mid_on_clock(ex):                              # causal: each exchange's most-recent mid at-or-before every clock tick (the COMMITTED level series)
    rx, mid = mids[ex]
    return mid[np.clip(np.searchsorted(rx, merged_ts, "right") - 1, 0, len(mid) - 1)]

# A fixed per-venue REFERENCE price, subtracted from the mid before forming the band. The breakout z is EXACTLY
# shift-invariant — (mid−c) − ema(mid−c) = mid − ema(mid), and σ_band is shift-invariant — so this changes NOTHING in
# the feature. But it is numerically essential: σ_band² = ema(mid²) − ema(mid)² subtracts two ~6.8M numbers to get a
# ~0.5 variance (a ~1e-9 relative loss at price ~2600 = the catastrophic-cancellation trap); centering by a constant
# near the price recovers full float64 precision (~1e-16). BOTH the vectorized path and the streaming build use this SAME
# REF per venue, so they stay bit-identical. The constant is each venue's first observed mid (a causal, fixed scalar).
REF = {ex: float(mids[ex][1][0]) for ex in EXCHANGES}   # fixed reference price per venue (its first mid) — band built on (mid − REF)
print("band reference prices (subtracted before the band — leaves z exactly invariant, just numerically stable):",
      {ex: round(REF[ex], 1) for ex in EXCHANGES})

# --- byb's sparse mid-move flow (for the yardsticks + the rate-head target) ---
byb_rx0, byb_mid0 = mids["byb"]                                                   # raw byb merged mid stream (front_levels + trade fusion)
keep = np.concatenate([byb_rx0[1:] != byb_rx0[:-1], [True]])                     # collapse same-TIMESTAMP rows to ONE update (the final mid): simultaneous events are not a sequence
byb_rx, byb_mid = byb_rx0[keep], byb_mid0[keep]
byb_lm = np.log(byb_mid)
byb_blr = np.empty_like(byb_lm); byb_blr[0] = 0.0; byb_blr[1:] = np.diff(byb_lm)  # byb log-return per timestamp
mv = byb_blr != 0.0                                                              # a REAL byb mid-move: ONE per timestamp where the mid changed
mv_rx, mv_r2 = byb_rx[mv], byb_blr[mv] ** 2                                       # move times + squared returns
cum_mv = np.concatenate([[0.0], np.cumsum(mv.astype(float))])                    # running count of byb mid-moves (rate-head target)
byb_dt = np.zeros(n_ticks); byb_dt[1:] = np.diff(merged_ts) / 1e9                # seconds between consecutive trades (per-trade)
def _ewma(x, span):                                                # per-trade EMA (for the seconds-per-trade leg of λ_ev)
    a = 2.0 / (span + 1.0); return lfilter([a], [1.0, -(1.0 - a)], x)
def _flow_at(anchors, val, span):           # EWMA of `val` over the byb-MOVE stream, decayed once per trade-timestamp, read AT each anchor
    a = 2.0 / (span + 1.0)
    k = np.searchsorted(merged_ts, mv_rx, "left")                               # trades strictly before each move (a same-rx trade decays it)
    ep = np.bincount(k, weights=val, minlength=n_ticks + 1)                     # per-trade-epoch sums
    x = np.zeros(n_ticks + 1); x[1:] = a * (1.0 - a) * ep[:-1]
    com = lfilter([1.0], [1.0, -(1.0 - a)], x)                                  # committed E just after each trade
    ta = np.searchsorted(merged_ts, anchors, "right") - 1                       # last trade <= anchor
    cs = np.concatenate([[0.0], np.cumsum(val)])                               # prefix sums over the move stream (the partial epoch)
    partial = cs[np.searchsorted(mv_rx, anchors, "right")] - cs[np.searchsorted(mv_rx, merged_ts[ta], "right")]
    return com[ta + 1] + a * partial
def yardsticks(anchors, span):                                     # σ_ev, λ_ev — defined AT the anchor, reacting to every byb mid-move
    e_sq = _flow_at(anchors, mv_r2, span)                          # E: exp-weighted squared byb moves
    e_mv = _flow_at(anchors, np.ones(mv_r2.size), span)           # W: exp-weighted byb-move count
    e_dt = _ewma(byb_dt, span)[np.searchsorted(merged_ts, anchors, "right") - 1]  # seconds/trade (per-trade, held flat between trades)
    sig = np.sqrt(e_sq / np.maximum(e_mv, 1e-12))                 # σ_ev: RMS byb mid-move (E/W — non-moves cancel)
    lam = e_mv / np.maximum(e_dt, 1e-12)                          # λ_ev: byb mid-moves per second
    return sig, lam
print(f"yardsticks: react to every byb mid-move; decay span {YARDSTICK_N} trades (controls + price-head target only — the breakout is self-normalised)")
""")

md(r"""
## 3. Build it (twice)

Build the feature two ways: this fast array version for analysis, and — in production — a
streaming version that does constant work per ingested event (no growing buffers). They have to agree,
which §4 checks **for every venue**.

Here we lay an evaluation grid every 50 ms (half the 100 ms horizon — plenty of samples; note
that adjacent 100 ms outcome windows still overlap ~50%, so neighbouring samples are correlated,
which is exactly why §5's walk-forward gate uses an embargo), read byb's actual move over the
next 100 ms (the thing we're predicting), and compute the breakout `z` for **each venue** at each
grid point.

**Vectorising a live-front band leg.** A `LiveFrontEMA` reads `(1−α)·committed + α·current`,
where `committed` is a plain `EventEMA` over the per-trade committed level. We build the committed
EMA over the trade-clock mid series with one `lfilter` (the same recursion the class runs), index
it at the last tick ≤ each anchor, and blend in the *fresh* mid at the anchor — exactly the class's
read. We do this for both legs (`mid`, `mid²`), form the band, and read `z`.
""")

code(r"""
# evaluation grid (causal) + forward target in shared σ-units
WARMUP = 5 * max(YARDSTICK_N, max(LONG))   # enough trades for the slowest band/yardstick to converge
anchor_ts      = np.arange(merged_ts[WARMUP], merged_ts[-1] - HORIZON_NS, 50 * 1_000_000)   # 50 ms grid, past warmup (bands + yardsticks)
tick_at_anchor = np.searchsorted(merged_ts, anchor_ts, "right") - 1                         # last trade-clock tick <= anchor
sigma_at_anchor, lam_at_anchor = yardsticks(anchor_ts, YARDSTICK_N)   # both yardsticks at each grid point (the same span-YARDSTICK_N trade-tick EMA) — for controls + price-head target
print(f"σ_ev median {np.nanmedian(sigma_at_anchor):.2e},  λ_ev median {np.nanmedian(lam_at_anchor):.2f} moves/s")

# price-head target: byb's 100 ms log-return ÷ σ_ev (regime-normalised, σ-units)
mid_now    = byb_mid[np.searchsorted(byb_rx, anchor_ts, "right") - 1]
mid_fwd    = byb_mid[np.searchsorted(byb_rx, anchor_ts + HORIZON_NS, "right") - 1]
fwd_return = np.log(mid_fwd / mid_now)
target     = fwd_return / sigma_at_anchor                          # byb's 100 ms return ÷ σ_ev — the price head's target

# --- the band legs: committed per-trade EMA of the (REF-centred) mid + its square, read on the LIVE FRONT at each anchor ---
# Centre by REF[ex] (a constant) before squaring: leaves z identical, but keeps ema(mid²) − ema(mid)² off the cancellation cliff.
mid_committed  = {ex: mid_on_clock(ex) - REF[ex] for ex in EXCHANGES}              # each venue's CENTRED mid on the trade clock (committed level)
mid2_committed = {ex: mid_committed[ex] ** 2     for ex in EXCHANGES}              # and its square
def _mid_at(ex, t):                                                # a venue's mid forward-filled to arbitrary times (the FRESH value at the anchor)
    rx, mid = mids[ex]; return mid[np.clip(np.searchsorted(rx, t, "right") - 1, 0, len(mid) - 1)]
mid_fresh  = {ex: _mid_at(ex, anchor_ts) - REF[ex] for ex in EXCHANGES}            # the CENTRED mid as of each grid anchor — every book update, never stale
mid2_fresh = {ex: mid_fresh[ex] ** 2             for ex in EXCHANGES}
def _event_ema(series, N):                                         # plain EventEMA over the committed trade-clock series (y[-1]=0): the committed part
    a = 2.0 / (N + 1.0); return lfilter([a], [1.0, -(1.0 - a)], series)
_emean_cache = {}
def live_front(committed, fresh, N):                               # the LiveFrontEMA read at every anchor: (1-α)·committed-at-last-trade + α·fresh
    key = (id(committed), N)
    if key not in _emean_cache: _emean_cache[key] = _event_ema(committed, N)
    a = 2.0 / (N + 1.0)
    return (1.0 - a) * _emean_cache[key][tick_at_anchor] + a * fresh
def range_breakout(ex, N):                                         # the feature, per venue: Bollinger z = (mid − ema(mid)) / √(ema(mid²) − ema(mid)²)
    em  = live_front(mid_committed[ex],  mid_fresh[ex],  N)        # ema(mid,N)  — band centre  (live front)
    em2 = live_front(mid2_committed[ex], mid2_fresh[ex], N)        # ema(mid²,N) — live front
    var = np.maximum(em2 - em * em, 0.0)                           # band variance (clamp tiny negative round-off at 0)
    sig_band = np.sqrt(var)
    z = (mid_fresh[ex] - em) / np.where(sig_band > 0, sig_band, np.nan)   # σ-normalised extension; nan where the band has no width
    return z
def range_position(ex, N):  return np.tanh(range_breakout(ex, N))  # bounded where-in-band read (the bounded atom)
print(f"grid: {len(anchor_ts):,} anchors")
for ex in EXCHANGES:
    z = range_breakout(ex, 1000)
    print(f"  range_breakout[{ex}](N=1000): z median {np.nanmedian(z):+.2f}, |z| 95th pct {np.nanpercentile(np.abs(z), 95):.2f}")
""")

md(r"""
## 4. Check the code is right — a parity check (a production-style streaming build)

**Non-negotiable.** Reproduce the feature with a second implementation and confirm the two
agree on real data — a **parity check**, not an independent oracle. The streaming build is the **production**
reference: an O(1) state machine you push **raw events** into — `on_book(...)` for a top-of-book update,
`on_trade(...)` for a trade — and read each venue's current breakout `z` from `value()`. Its EMA state is
**`boba.ema`'s tested `LiveFrontEMA`** — the *actual production code* — wired to a raw-event driver: two
live-front level reads per venue (`ema(mid)`, `ema(mid²)`), no buffers, no history, independent of how long it
runs. We deliberately do **not** hand-roll a parallel EMA here; the whole point is that the production
`boba.ema` recursion agrees with §3's vectorized `lfilter` path.

**The design:**
- Fed **only raw events**, each tagged by its full listing (e.g. `byb_eth_usdt_p`) and keyed by it, so a perp
  and a spot on one exchange stay separate books. It builds each mid **itself** (per `MID_STREAM`): merged
  venues (byb, okx) fuse trades into the book by newest-exchange-time; the book-only venue (bin) takes the
  latest snapshot — so the live mid matches the analysis mid.
- Events sharing a **timestamp are one event**: the driver applies them all, then calls **`refresh()` once** —
  which refreshes both band legs' live front to the current mid, then advances the clock **at most once**
  (commit both legs) and only if a trade landed. A book-only instant moves the mid without advancing the clock.
- `value()` returns the live-front `z = (mid − ema(mid)) / √(ema(mid²) − ema(mid)²)` per venue, current at the
  instant it's read.

**Coverage — the shipped spans, on the whole block.** The check runs at the **shipped** band spans
(§6 picks them per venue) over the **full** grid (not a prefix) — the small-N case is the *harder* one for
the variance identity `ema(mid²) − ema(mid)²` (less smoothing, the two terms closer), so validating the small
shipped spans is the honest test.

**Two honest tolerance levels.** The band **legs** — `ema(mid)`, `ema(mid²)`, the production *state* — are
the same recursion both ways (`boba.ema`'s `EventEMA` online vs §3's `lfilter`), so they match **bit-exact
(< 1e-9)**. The derived **`z`** divides by the **catastrophically-cancelling** variance
`ema(mid²) − ema(mid)²`, worst at the smallest shipped span (here N=50 for every venue, the §6 pick) and worst
of all at the **near-zero-width** band points where `z ≈ 0` and that subtraction is near-total; there an
unavoidable ~1e-12 leg round-off is *amplified* to ~1e-5, so we certify the derived `z` to **float32
tolerance** (a 5e-5 bar — CLAUDE.md's production-vs-oracle bar). That is intrinsic float32 behaviour on an
ill-conditioned `0/0` quantity, not a logic bug.

We feed the **whole raw stream** — every venue's book updates and trades — into **one builder per venue** and
read each `range_breakout[ex]` at every grid anchor. If the online build reproduces §3's vectorized feature on
real data, the two implementations agree.
""")

code(r"""
import boba.io as io
import math
from boba.ema import LiveFrontEMA   # forward-filled LEVEL read with a live front — exactly the band legs, src/boba/ema.py

# The spans §6 actually ships, declared UP FRONT so §4 validates THESE (not a convenient large N): the smallest
# shipped span is the HARDEST case for the variance identity ema(mid²)−ema(mid)² (less smoothing, the two terms closer),
# so validating the small shipped spans is the honest test. §6 confirms these are the |IC|-argmax picks per venue.
SHIP_SPAN = {"byb": 50, "bin": 50, "okx": 50}          # validate THESE spans — the |IC|-argmax is N=50 for ALL THREE venues (§6 confirms); the smallest span is the hardest variance-cancellation case, so this is the honest test
F32_TOL = 5e-5                                          # float32-scale bar for the DERIVED z. z divides by the catastrophically-cancelling variance ema(mid²)−ema(mid)²; at the smallest shipped span (N=50) and at near-zero-WIDTH bands (z≈0) that cancellation is near-total, so an unavoidable ~1e-12 leg round-off is AMPLIFIED to ~1e-5 there — intrinsic float32 behaviour on a 0/0-conditioned quantity, NOT a logic bug (the band LEGS are bit-exact below).

class LiveRangeBreakout:
    # Pure feature state machine for ONE target venue. z = (mid − ema(mid)) / sqrt(ema(mid²) − ema(mid)²),
    # two LiveFrontEMAs over the TARGET venue's mid and mid². on_book/on_trade only update that venue's bid/ask;
    # the driver calls refresh() ONCE per timestamp, which (a) pushes the current mid + mid² into BOTH legs' live front,
    # then (b) advances the clock at most once (commit both legs) — only if a trade landed. State is O(1), all scalar.
    # The mid is built INSIDE (never read from merged_levels): merged venues fuse trades by newest-exchange-time; bin book-only.
    def __init__(self, target, N, fuse_trades, ref):
        self.target = target
        self.ref = ref                                                     # the SAME per-venue reference price the vectorized path centres by (numerical stability only; z is shift-invariant)
        self.fuse_trades = set(fuse_trades)                                # listings whose mid folds in trades (merged); rest book-only
        self.bid = {}; self.bid_t = {}; self.ask = {}; self.ask_t = {}     # newest-by-exchange-time top-of-book, keyed by listing
        self.leg_m  = LiveFrontEMA(N)                                      # ema(mid−ref,N)  — band centre  (live-front level read)
        self.leg_m2 = LiveFrontEMA(N)                                      # ema((mid−ref)²,N) — live-front level read
        self.was_trade_present = False                                     # did a trade land this timestamp? -> exactly one commit per timestamp

    def _side(self, listing, is_ask, px, t):                              # keep this listing-side's newest-by-exchange-time price
        held_t = (self.ask_t if is_ask else self.bid_t)
        if t > held_t.get(listing, -1):
            (self.ask if is_ask else self.bid)[listing] = px; held_t[listing] = t

    def _mid(self, listing):
        b, a = self.bid.get(listing), self.ask.get(listing)
        return None if b is None or a is None else 0.5 * (b + a)

    def refresh(self):                                                    # ONE per TIMESTAMP: refresh the live front, then advance the clock AT MOST ONCE (only if a trade landed)
        traded, self.was_trade_present = self.was_trade_present, False      # consume the flag — simultaneous prints = one commit, not many
        m = self._mid(self.target)
        if m is not None:
            mc = m - self.ref                                          # CENTRE by the reference (z is invariant to this; it just keeps the variance off the cancellation cliff)
            self.leg_m.add(mc); self.leg_m2.add(mc * mc)              # both band legs react to EVERY book update — refresh the live front (decay/commit waits for the tick)
        if traded: self.tick()                                          # a trade landed -> advance the clock exactly once: commit both legs

    def on_book(self, listing, t, bid, ask):                              # BBO update -> mutate that listing's mid only (the single per-timestamp refresh is the driver's job)
        if listing in self.fuse_trades:                                  # merged-mid listing: newest-by-exchange-time
            self._side(listing, False, bid, t); self._side(listing, True, ask, t)
        else:                                                            # book-only listing (e.g. bin): latest snapshot wins
            self.bid[listing] = bid; self.ask[listing] = ask

    def on_trade(self, listing, t, px, lifts_ask):                       # trade -> fold into the mid (merged listings only) and flag the timestamp as traded
        if listing in self.fuse_trades: self._side(listing, lifts_ask, px, t)
        self.was_trade_present = True                                    # any trade this timestamp -> the clock advances once (in refresh)

    def tick(self):                                                       # advance the trade clock ONE step: commit both band legs (trade-dwell weighting)
        self.leg_m.tick(); self.leg_m2.tick()

    def value(self):                                                      # the live breakout z: (mid − ema(mid)) / sqrt(ema(mid²) − ema(mid)²), fresh at read (nan until warm / zero-width band)
        em, em2 = self.leg_m.value(), self.leg_m2.value()                  # both on the CENTRED mid
        m = self._mid(self.target)
        if m is None or not (em == em) or not (em2 == em2): return float("nan")  # em/em2 nan before first commit
        var = em2 - em * em
        if not (var > 0.0): return float("nan")
        return ((m - self.ref) - em) / math.sqrt(var)                      # centred numerator (= mid − ema(mid)); ref cancels exactly

    def legs(self):                                                       # the two live-front BAND LEGS themselves (centred ema(mid), ema(mid²)) — for the bit-exact leg check vs §3
        em, em2 = self.leg_m.value(), self.leg_m2.value()
        m = self._mid(self.target)
        if m is None or not (em == em) or not (em2 == em2): return float("nan"), float("nan")
        return em, em2

# --- gather the WHOLE raw stream (every venue's book updates + trades) over a slice; no merged_levels anywhere ---
# Each builder runs at its OWN SHIPPED span (§6's picks), validated over the FULL grid (no prefix).
N_GRID = len(anchor_ts)                            # validate over the FULL grid (every anchor, the whole block) — not a prefix
cutoff = int(anchor_ts[min(N_GRID - 1, len(anchor_ts) - 1)])    # wall-clock time of the last grid anchor
LISTINGS = [f"{ex}_{COIN}" for ex in ("byb", "okx", "bin")] # 0=byb, 1=okx, 2=bin — integer codes keep the event arrays numeric & fast
cols = {k: [] for k in "rx kind lid t a b".split()}
def add(rx, kind, lid, t, a, b):                   # stage one source's events (kept ≤ cutoff)
    m = rx <= cutoff; n = int(m.sum())
    cols["rx"].append(rx[m]); cols["kind"].append(np.full(n, kind, np.int8)); cols["lid"].append(np.full(n, lid, np.int8))
    cols["t"].append(t[m]); cols["a"].append(a[m].astype(float)); cols["b"].append(b[m].astype(float))
for lid, ex in ((0, "byb"), (1, "okx")):           # merged-mid venues: raw front_levels (bid/ask + exchange_time), they fuse trades
    fl = load_block(block, f"{ex}_{COIN}", "front_levels").select("rx_time", "exchange_time", "bid_prc", "ask_prc").drop_nulls()
    add(fl["rx_time"].cast(pl.Int64).to_numpy(), 0, lid, fl["exchange_time"].cast(pl.Int64).to_numpy(), fl["bid_prc"].to_numpy(), fl["ask_prc"].to_numpy())
brx, bmid = mids["bin"]                             # book-only venue: bin's front_levels mid (already loaded in §2); fed as bid=ask=mid
add(brx, 0, 2, brx, bmid, bmid)
for lid, ex in ((2, "bin"), (0, "byb"), (1, "okx")):   # trades from EVERY venue tick the shared clock; byb/okx also fuse their mid
    td = load_block(block, f"{ex}_{COIN}", "trade").select("rx_time", "exchange_time", "prc", "qty", "aggressor").filter((pl.col("prc") > 0) & (pl.col("qty") > 0))
    add(td["rx_time"].cast(pl.Int64).to_numpy(), 1, lid, td["exchange_time"].cast(pl.Int64).to_numpy(),
        td["prc"].to_numpy(), io._trade_lifts_ask(f"{ex}_{COIN}", td["aggressor"].to_numpy()).astype(float))
C = {k: np.concatenate(v) for k, v in cols.items()}
order = np.lexsort((C["kind"], C["rx"]))           # rx ascending; book (0) before trade (1) on ties (matches merged_levels)
rxL, kindL, lidL, tL, aL, bL = (C[k][order].tolist() for k in "rx kind lid t a b".split())
print(f"streaming {len(rxL):,} raw events (book + trades, all venues) over {N_GRID:,} grid points (FULL grid), ALL THREE venues at their SHIPPED spans...")

# --- the CALLER drives all three builders in lock-step: apply each timestamp's events, refresh() each once, READ value() at every anchor ---
fuse = {f"{ex}_{COIN}" for ex in EXCHANGES if MID_STREAM[ex] == "merged_levels"}   # byb,okx merged; bin book-only
feats = {ex: LiveRangeBreakout(f"{ex}_{COIN}", SHIP_SPAN[ex], fuse, REF[ex]) for ex in EXCHANGES}  # one builder per venue at its SHIPPED span; same REF as §3
na = min(N_GRID, len(anchor_ts))                                                   # the validated grid slice
streams = {ex: np.full(na, np.nan) for ex in EXCHANGES}
legs_m  = {ex: np.full(na, np.nan) for ex in EXCHANGES}                            # the live-front band legs themselves (for the bit-exact leg check)
legs_m2 = {ex: np.full(na, np.nan) for ex in EXCHANGES}
n = len(rxL); i = 0; ai = 0
while i < n:                                        # walk the stream, grouped by receive-timestamp
    rx = rxL[i]
    while ai < na and anchor_ts[ai] < rx:          # read every anchor whose state is settled (all events before rx applied)
        for ex in EXCHANGES:
            streams[ex][ai] = feats[ex].value()
            legs_m[ex][ai], legs_m2[ex][ai] = feats[ex].legs()
        ai += 1
    while i < n and rxL[i] == rx:                  # apply EVERY event stamped at this nanosecond (no half-applied timestamp) to ALL builders
        lst = LISTINGS[lidL[i]]
        if kindL[i] == 0:
            for ex in EXCHANGES: feats[ex].on_book(lst, tL[i], aL[i], bL[i])
        else:
            for ex in EXCHANGES: feats[ex].on_trade(lst, tL[i], aL[i], bL[i])
        i += 1
    for ex in EXCHANGES: feats[ex].refresh()       # apply the timestamp to each builder: refresh the live front, then advance the clock ONCE if any trade landed
while ai < na:                                     # trailing anchors after the last event
    for ex in EXCHANGES:
        streams[ex][ai] = feats[ex].value()
        legs_m[ex][ai], legs_m2[ex][ai] = feats[ex].legs()
    ai += 1

# --- §3's live-front band legs (centred ema(mid), ema(mid²)) at the same anchors, to compare leg-for-leg ---
def _vec_legs(ex, N):
    em  = live_front(mid_committed[ex],  mid_fresh[ex],  N)
    em2 = live_front(mid2_committed[ex], mid2_fresh[ex], N)
    return em, em2

# --- check EVERY venue at its SHIPPED span: the BAND LEGS bit-exact (1e-9), the derived z to float32 ---
# The legs ema(mid), ema(mid²) are the production state; they must be bit-for-bit (streaming EventEMA-recursion == §3).
# z divides by the CANCELLING variance ema(mid²) − ema(mid)², so an unavoidable ~1e-12 leg round-off is AMPLIFIED into
# up to ~1e-5 in z at the smallest shipped span (N=50), worst at the near-zero-width z≈0 points. We therefore certify: legs bit-exact, z float32.
print(f"one raw-event stream -> per-venue range_breakout, vs the vectorized build (SHIPPED spans, FULL grid):")
worst_leg = 0.0; worst_z = 0.0
for ex in EXCHANGES:
    N = SHIP_SPAN[ex]
    em_v, em2_v = _vec_legs(ex, N)
    bm  = np.isfinite(legs_m[ex])  & np.isfinite(em_v[:na])
    bm2 = np.isfinite(legs_m2[ex]) & np.isfinite(em2_v[:na])
    dleg = max(float(np.nanmax(np.abs(legs_m[ex][bm] - em_v[:na][bm]))),
               float(np.nanmax(np.abs(legs_m2[ex][bm2] - em2_v[:na][bm2])))); worst_leg = max(worst_leg, dleg)
    refz = range_breakout(ex, N)[:na]; bz = np.isfinite(streams[ex]) & np.isfinite(refz)
    dz = float(np.nanmax(np.abs(streams[ex][bz] - refz[bz]))); worst_z = max(worst_z, dz)
    ok_leg = "OK (bit-exact)" if dleg < 1e-9 else "FAIL"
    ok_z   = "OK (float32)"   if dz   < F32_TOL else "FAIL"
    print(f"  {ex} N={N}:  band legs max |diff| {dleg:.2e} -> {ok_leg};  derived z max |diff| {dz:.2e} -> {ok_z}  ({int(bz.sum()):,} grid points)")
    assert dleg < 1e-9, f"streaming band legs do not match §3 bit-exact for {ex}"
    assert dz   < F32_TOL, f"streaming z does not match §3 to float32 for {ex}"
print(f"§4 parity: BAND LEGS bit-exact (worst {worst_leg:.2e}), derived z float32 (worst {worst_z:.2e}) — all venues, shipped spans  OK")
""")

md(r"""
**§4 conclusion.** From a stream of **34.4 M** raw events the O(1) production builder — its EMA state is
`boba.ema`'s `LiveFrontEMA`, the actual production code — reproduces §3's vectorized `lfilter` path **for
every venue at its shipped span** (all N=50 — the §6 |IC|-argmax pick) over the **full** grid (1.665 M
anchors), certified at two honest levels:

- **Band legs — bit-exact (< 1e-9).** The production *state* is the two live-front EMAs `ema(mid)`,
  `ema(mid²)`; `boba.ema`'s online `EventEMA` recursion matches §3 to ~1e-12 (machine precision on the centred
  legs): worst leg max|diff| **3.6e-12** (the same 3.6e-12 for all three venues). This is the meaningful
  bit-exact claim, and it holds at the shipped spans.
- **Derived z — float32 (< 5e-5).** `z` divides by the **catastrophically-cancelling** variance
  `ema(mid²) − ema(mid)²`. At the smallest shipped span (N=50, the narrowest band) — and worst of all at the
  **near-zero-width** band points where `z ≈ 0` — an unavoidable ~1e-12 leg round-off is *amplified* into the
  `z` read: worst derived-z max|diff| **3.9e-8** (byb), **1.6e-6** (bin), **1.4e-5** (okx, at its `z ≈ 0`
  points). That is float32, not float64, and it is intrinsic to the cancellation — REF centering reduces the
  variance's absolute magnitude but cannot remove the relative cancellation at small N / near-zero band width.
  Both paths subtract the **same** `REF[ex]`, so `z` is exactly invariant to it.

The honest statement is "legs bit-exact, z to float32 at the shipped small span" — a worked instance of the
variance cancellation, not a logic bug. The production `boba.ema` build computes exactly what the offline
analysis did, on the one shared trade clock — so the two implementations compute the same feature. EMA-based
band only — no rolling max/min, no boxcar.
""")

md(r"""
## 5. Is the signal real? — the hygiene gates

A correlation is an easy way to fool yourself. The gates are **two independent tests** (see the
Gate A / Gate B box in §6): **(A)** is the feature **regime-invariant** — a stable distribution that
doesn't leak the vol/rate state — and **(B)** does it **predict** something the market's current state
doesn't already tell us? The only **controls** for Gate B are the two **regime-invariant momenta**:
- **rate momentum** — from `λ_ev` (byb's mid-move rate): is byb moving more or less often than its own
  recent pace?
- **vol momentum** — the same, for volatility.
The raw **levels** of vol and rate (`log σ_ev`, `log λ_ev`) are **not** controls — they aren't
regime-invariant, so they're never model features; we keep them only as the regime *coordinate* for Gate A.

"Predictive power" here is the **rank correlation** between feature and outcome (Spearman —
robust to outliers), scored **out-of-sample with a purged, expanding-window walk-forward**:
each fold trains only on the *past*, leaves an **embargo gap** sized to clear the 100 ms
outcome windows with margin (so overlapping forward targets can't straddle the train→test
boundary). Note the embargo does **not** fully decorrelate the slow EMA/yardstick features:
their memory is ≈ `YARDSTICK_N` / (trades-per-sec) — longer than the ~100 s
embargo — so re-check this if you slow the spans or use a thinner-traded block. The fold
scores on the *next* segment,
and we average over folds. That's the causal, production-style estimate — strictly past→future, as
it would run live. (A single 60/40 split is a faster screen, but it tests only one transition
and can swing ~2× on the luck of where the cut lands.)

Because adjacent samples are correlated (overlapping 100 ms labels + long EMA/yardstick memory), a single
point IC overstates its own precision — so for the **headline marginal** we also report the **per-fold** ICs
(did the gain hold in *every* OOS segment?) and a **block-bootstrap 90% CI** that resamples contiguous time
blocks sized to the EMA/yardstick memory (block length derived from the block's own trade-rate, not hard-coded),
so the interval respects the autocorrelation rather than pretending all anchors are independent.

Rank-IC is a **feature-screening** statistic, not a distributional score — proper scoring (NLL/CRPS, occurrence log-loss, calibration) and cost-aware utility are judged at the **model** level downstream, not in this per-feature template.

Because the feature is a band-σ z-score and the target is in σ-units, a *scale* regime-shift mostly cancels —
but scale is not the *relationship*. So beside the gates we run a **companion check**: the same
marginal power computed **within calm / mid / wild volatility buckets**. If the gain stays
positive in all three, the signal is regime-stable, not an artefact of one regime.

The gates that follow ask, separately: **Gate A** — is the feature's distribution stable across
volatility states (**scale**, plus neither the feature **nor its magnitude** `|feature|` tracks the regime level
(monotonically *or* non-monotonically), against **both** the vol and rate coordinate)? **Gate B** — does it *add* signal over the invariant
momentum controls, walk-forward? And the **companion** — does the Gate-B gain hold across calm / mid /
wild volatility, not just one regime?

> **Note on normalisation.** `range_breakout` is a z-score, **self-normalised by its own band width
> `σ_band`** — so it is **not** divided by `σ_ev` (nor `λ_ev`). The guard rail is *measure, don't assume*:
> Gate A's scale check decides whether the raw z ships, and it passes (see §6). The target still is
> `return ÷ σ_ev`, and the controls are still the yardsticks, so the gates are identical; we just don't
> σ-divide the feature itself.
""")

code(r"""
# --- the regime signals: the two INVARIANT momenta (the Gate B controls) + the two yardstick LEVELS
# (the regime COORDINATE for Gate A — used only to bucket/correlate against, NEVER as controls) ---
FAST_YARD = YARDSTICK_N // 10                        # a faster span (1/10 the yardstick) for the momentum controls
sig_fast, lam_fast = yardsticks(anchor_ts, FAST_YARD)
vol_level     = np.log(sigma_at_anchor)                                             # σ_ev — how volatile now
vol_momentum  = np.log(sig_fast / sigma_at_anchor)                                  # recent vol vs slower vol
rate_level    = np.log(lam_at_anchor)                                               # λ_ev = byb's mid-move rate — is byb moving more or less often than usual?
rate_momentum = np.log(lam_fast / lam_at_anchor)                                    # recent mid-move rate vs slower mid-move rate

# The gate machinery (walk-forward IC, Gate A/B, the block-bootstrap CI) lives in the shared, TESTED
# library boba.research.gates — defined once and used by every feature notebook (see tests/test_gates*.py;
# externally validated against the literature and adversarially reviewed). We import the primitives here and
# bind the regime scaffolding below into thin wrappers, so the call sites read the same while the logic
# stays in one validated place.
from boba.research import gates as _gates
from boba.research.gates import ic, wf_folds, wf_ic, wf_ic_by_regime, stratified_ic

vol_regime = np.digitize(vol_level, np.nanpercentile(vol_level[np.isfinite(vol_level)], [33, 67]))   # 0 calm, 1 mid, 2 wild
# FEATURE_KIND drives GATE B below. "alpha" = a candidate signal that is NOT a regime descriptor
# (OFI, microprice, flow, price moves, breakouts) -> judged on MARGINAL value over the controls. "control" = a
# regime descriptor (vol, rate, volume & their ratios/dynamics) -> itself a valid feature, but it
# OVERLAPS the controls, so it's judged on STANDALONE signal.
FEATURE_KIND = "alpha"                                                  # range_breakout = a signed price-extension signal, not a regime descriptor
base = [rate_momentum, vol_momentum]                                   # the ONLY controls: regime-INVARIANT momenta. Raw vol/rate LEVELS are never controls (not valid features).
# byb momenta proxy the GLOBAL regime — one instrument, cross-venue divergence is arbed away fast, so at this slow control scale all venues share one regime; foreign-venue momenta would just re-add it.
STRAT_VAR = None                                                       # mechanical-coupling guard (HIGH-2): the shared YARDSTICK to STRATIFY by. Set it to the
                                                                       # denominator of the *scored* target — the gate here always scores the σ_ev PRICE target, so a
                                                                       # control RATIO that divides by σ_ev sets STRAT_VAR = sigma_at_anchor. (The rate-head analogue,
                                                                       # for a count/λ_ev clone scored against a count/λ_ev target, is lam_at_anchor.) Gate B then scores
                                                                       # the feature WITHIN strata of that yardstick, so the shared denominator can't manufacture IC
                                                                       # (the spurious correlation of ratios) — stratifying multiplicatively DECOUPLES the shared scale,
                                                                       # where a linear partial over-removes genuine within-yardstick signal. range_breakout is a Bollinger
                                                                       # z self-normalised by its OWN band width σ_band — it does NOT divide by σ_ev (nor λ_ev) — so it
                                                                       # shares NO denominator with either target -> no shared-yardstick coupling to decouple -> None.
# Thin wrappers binding the scaffolding above to the shared gate library (call sites stay unchanged):
def signal_ic(leg_feats, *, own, tgt=None):                            # Gate B value (marginal over the controls, or standalone for a control's own leg)
    return _gates.signal_ic(leg_feats, base, target if tgt is None else tgt,
                            feature_kind=FEATURE_KIND, own=own, strat_var=STRAT_VAR)
def signal_ic_by_regime(leg_feats, *, own, tgt=None):                  # the regime-stable companion (per calm/mid/wild bucket)
    return _gates.signal_ic_by_regime(leg_feats, base, target if tgt is None else tgt, vol_regime,
                                      feature_kind=FEATURE_KIND, own=own, strat_var=STRAT_VAR)
def gate_a(feat):                                                      # Gate A on one feature vs the vol/rate regime coordinates
    return _gates.gate_a(feat, vol_level, rate_level)
def wf_marginal_ci(legs, tgt, B=400, seed=0):                          # per-fold spread + block-bootstrap 90% CI for the headline marginal
    return _gates.marginal_ci(legs, base, tgt, B=B, seed=seed)

print("control-only predictive power (walk-forward):  momenta", round(wf_ic(base, target), 3),
      " (near 0 = controls barely predict direction, so any feature gain is genuinely new)")
""")

md(r"""
**Conclusion.** On their own the controls carry essentially **no** directional signal — walk-forward rank-IC
≈ **0.002** for the momenta. That is exactly what we want here: the regime barely predicts *which way* byb
moves, so any rank-IC a feature shows *on top of* these controls is genuinely new information, not the regime
wearing a disguise. That's what makes the "added over the controls" gates below a fair test.
""")

md(r"""
## 6. Two choices: which time-scale per head, and which exchanges to keep

A feature is rarely a single number — it's a **family** across time-scales (here, every band span N —
short = a short-horizon breakout, long = a regime breakout). And the same feature can carry signal for both
heads, so we check two things:
- does the **signed** `z` predict *direction* — which way (and how far) byb moves next? **The IC sign tells
  us continuation (same sign) vs reversion (opposite sign), per span and per venue — we read it off, never
  assume it.**
- does its **magnitude** `|z|` predict *intensity* — *how many* moves byb makes next?

The magnitude check is a **diagnostic only**: the model is fed the *signed* `z` for both heads (see the
guard rails) — pre-taking `|·|` per exchange would stop the rate head learning that opposing breakouts cancel.
The two signals usually live at different time-scales, so we sweep the whole family, for **every exchange**,
against both targets. We keep *all* exchanges (never pick one); the only thing we choose is the best
time-scale, **per head**.

**How many exchanges to keep is itself time-scale-dependent.** The cross-exchange edge is an **arbitrage
lead/lag** — one venue breaking out before another — and that mispricing is closed within milliseconds. So
at **short** spans the venues genuinely differ (who broke out first) and the signal is worth keeping
**per-exchange**; at **long** spans the lead/lag is long gone, every venue carries the *same* drift, and a
**single** venue's breakout already says what the rest do. Keep all exchanges as the safe default, but what
they *add* over one shrinks as the span grows (§9 discusses that cross-over).

**The IC-max span is only half the answer.** A short span can have the *highest* in-sample IC and yet carry
no signal the model can't already get from another span — or its edge can be the move *already underway* at
the anchor (a contemporaneous echo) rather than a forward prediction. So below the headline sweep we ask two
follow-up questions: does a **second** time-scale add anything orthogonal to the chosen span (the conditional
partial-IC cell, immediately below), and how much of the raw edge is **echo** versus real forward prediction
(the echo-netting section, after the gates).

The rate-head target is the count of byb's moves over the next 100 ms, divided by its recent
move-rate `λ_ev` — i.e. "more or fewer moves than usual," using the rate yardstick.

**On IC sign.** The sweep shows the **raw signed** rank-IC for the price head, so a *negative* IC means
reversion (a high `z` precedes a byb move the *other* way) and a *positive* IC means continuation. We pick
the time-scale by `|IC|` (the strength), and carry the **sign** as part of the feature — the model sees the
signed `z` and learns the sign itself.
""")

code(r"""
# rate-head target = byb moves (on the trade clock) in the next 100 ms, divided by λ_ev (the rate yardstick — the same trade-tick EMA).
fwd_count = (cum_mv[np.searchsorted(byb_rx, anchor_ts + HORIZON_NS, "right")]
             - cum_mv[np.searchsorted(byb_rx, anchor_ts, "right")])            # byb mid-moves over the next 100 ms
rate_target = fwd_count / np.maximum(lam_at_anchor, 1e-9)   # count ÷ λ_ev ∝ "more/fewer moves than usual" (λ_ev is moves/sec over 100 ms, so this carries units of seconds; the constant horizon factor drops out of the rank correlation)

# Sweep the band-span family for EVERY exchange, BOTH heads, treating exchanges symmetrically (no assumed leader).
# Each cell is a single span N (not a fast/slow pair): SHORT spans on the left, LONG on the right.
feat_cache = {(ex, N): range_breakout(ex, N) for ex in EXCHANGES for N in SPANS}
price_grid = {ex: np.full(len(SPANS), np.nan) for ex in EXCHANGES}   # signed z -> byb's signed return (SIGN = continuation/reversion)
rate_grid  = {ex: np.full(len(SPANS), np.nan) for ex in EXCHANGES}   # |z|      -> byb's move count (diagnostic)
for ex in EXCHANGES:
    for j, N in enumerate(SPANS):
        z = feat_cache[(ex, N)]
        mp = np.isfinite(z) & np.isfinite(target)                     # MASK non-finite (a zero-width band -> nan z at small N): unmasked spearmanr propagates nan and SKIPS the true argmax
        price_grid[ex][j] = spearmanr(z[mp], target[mp]).statistic    # signed: + = continuation, − = reversion
        mr = np.isfinite(z) & np.isfinite(rate_target)
        rate_grid[ex][j]  = spearmanr(np.abs(z)[mr], rate_target[mr]).statistic  # |z|→count is a univariate DIAGNOSTIC; the model is fed signed z

fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.6))
xs = np.arange(len(SPANS)); span_lbl = [f"{N}\n{'short' if N in SHORT else 'long'}" for N in SPANS]
for ax, grids, head in [(axes[0], price_grid, "price head: signed z -> byb return  (sign = continuation +/ reversion −)"),
                        (axes[1], rate_grid,  "rate head: |z| -> byb move count")]:
    for ex in EXCHANGES:
        ax.plot(xs, [grids[ex][j] for j in range(len(SPANS))], "o-", label=ex)
    ax.axhline(0, color="0.7", lw=0.8); ax.set_xticks(xs); ax.set_xticklabels(span_lbl)
    ax.set_xlabel("band span N (trade ticks)"); ax.set_ylabel("rank-IC (in-sample)"); ax.set_title(head); ax.legend(fontsize=8)
fig.suptitle("predictive power across the band-span family — every exchange, both heads", y=1.00)
fig.tight_layout(); plt.show()

# Heat-map form too (exchange × span), the price head's SIGNED IC — to read the continuation/reversion sign at a glance.
fig, ax = plt.subplots(figsize=(8.5, 3.2))
M = np.array([[price_grid[ex][j] for j in range(len(SPANS))] for ex in EXCHANGES])
vmax = np.nanmax(np.abs(M)); im = ax.imshow(M, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
ax.set_xticks(xs); ax.set_xticklabels([str(N) for N in SPANS]); ax.set_xlabel("band span N")
ax.set_yticks(range(len(EXCHANGES))); ax.set_yticklabels(EXCHANGES); ax.set_ylabel("venue")
for r in range(len(EXCHANGES)):
    for j in range(len(SPANS)):
        if np.isfinite(M[r, j]): ax.text(j, r, f"{M[r, j]:+.3f}", ha="center", va="center", fontsize=7)
ax.set_title("price head — SIGNED rank-IC (blue = reversion −, red = continuation +)")
fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout(); plt.show()

# We do NOT pick an exchange (not "the best", not assumed). Every exchange's feature is kept; the model weights
# whichever is leading at the moment. Choosing the best time-scale PER exchange (by |IC|) is fine; choosing an exchange is not.
# NB: each cell is an IN-SAMPLE spearmanr — best_member is the |IC|-argmax over it (in-sample maximisation), used ONLY to PICK a
# span. The chosen feature is then re-scored OUT-OF-SAMPLE by the §5 walk-forward gates below, and that is the number that counts.
def best_member(arr): return int(np.nanargmax(np.abs(arr)))                        # best span for THIS exchange by |IC| (in-sample pick); sign carried separately
price_member = {ex: best_member(price_grid[ex]) for ex in EXCHANGES}              # one signed feature per exchange (price head)
rate_member  = {ex: best_member(rate_grid[ex])  for ex in EXCHANGES}              # one |feature| per exchange (rate head) — diagnostic readout
print("kept features (one per exchange, all fed to the model — none privileged):")
for ex in EXCHANGES:
    pj, rj = price_member[ex], rate_member[ex]
    sgn = "continuation (+)" if price_grid[ex][pj] >= 0 else "reversion (−)"
    print(f"  {ex}:  price head N={SPANS[pj]} ({'short' if SPANS[pj] in SHORT else 'long'}) IC={price_grid[ex][pj]:+.3f} [{sgn}]"
          f"   |  rate head N={SPANS[rj]} ({'short' if SPANS[rj] in SHORT else 'long'}) IC={rate_grid[ex][rj]:+.3f}")
""")

md(r"""
**Reading the sweep.** The **price head** (top) is where this feature lives: the signed `z` correlates
**positively** with byb's next return at every venue and span here, so the IC **sign is continuation (+)
everywhere on this block** — a high `z` precedes a byb move the *same* way (price broke out on information
and kept going); no venue/span flipped to reversion. The IC **falls monotonically as the span lengthens** —
the breakout signal is strongest in the short bands and washes out by the long ones (N≥8000 is ≈0). The
in-sample `|IC|`-argmax picks therefore land at the **shortest span for every venue — byb N=50 (+0.106),
bin N=50 (+0.175), okx N=50 (+0.131)** — bin is the strongest single leg (its sub-ms-fresh feed), with okx
next and byb's own breakout behind. The **rate head** (bottom) is the weaker, diagnostic readout
(`|z| → count`; byb/bin/okx all pick **N=50** at **+0.069 / +0.154 / +0.098**). We keep every exchange and let
the model weight whoever is leading — but remember the byb-own leg is echo-prone (its numerator shares the
anchor mid with the target), so its raw price IC is inflated by the contemporaneous move; the echo-netting
section nets that echo out.
""")

code(r"""
# === Does a SECOND time-scale ADD over the pick? — the sweep RE-SCORED conditional on the best member, PER HEAD ===
# Selection lives here in §6. We picked the best span per head by IC; now re-score the WHOLE 1-D family as
# partial-IC(cell | chosen) -- each span's IC against the head's target, CONTROLLING for the span we picked (the
# same partial-IC tool the echo-netting cell uses; control = the chosen span, not the trailing move).
# A span still LIT adds signal ORTHOGONAL to the pick; a span that COLLAPSES to ~0 is a diluted copy. The line
# is in-sample, so the keep/drop DECISION is the walk-forward joint-vs-solo OOS IC (wf_ic, imported in §5).
def _pic(f, y, c):                                                   # partial rank-IC of f with y, controlling for c
    m = np.isfinite(f) & np.isfinite(y) & np.isfinite(c)
    if m.sum() <= 100: return np.nan
    rfy = spearmanr(f[m], y[m]).statistic; rfc = spearmanr(f[m], c[m]).statistic; rcy = spearmanr(c[m], y[m]).statistic
    return (rfy - rfc * rcy) / np.sqrt(max((1.0 - rfc**2) * (1.0 - rcy**2), 1e-12))

# Per head: feat() maps a span N to its scored feature, tgt is the head's target, member is the in-sample pick.
HEADS = [("price head", lambda ex, N: range_breakout(ex, N),         target,      price_member),
         ("rate head",  lambda ex, N: np.abs(range_breakout(ex, N)), rate_target, rate_member)]
fig, axes = plt.subplots(len(HEADS), 1, figsize=(9.5, 7.6), squeeze=False)
second = {}
for row, (head, feat, tgt, member) in enumerate(HEADS):
    print(f"does a 2nd span add over the pick? — {head}: conditional partial-IC (in-sample screen) + walk-forward joint-vs-solo OOS:")
    ax = axes[row][0]
    for ex in EXCHANGES:
        ci = member[ex]; chosen = feat(ex, SPANS[ci])
        cond = np.full(len(SPANS), np.nan)
        for j, N in enumerate(SPANS):
            cond[j] = 0.0 if j == ci else _pic(feat(ex, N), tgt, chosen)   # partial-IC(span N | chosen span)
        ax.plot(range(len(SPANS)), cond, "o-", label=ex)
        bj = int(np.nanargmax(np.abs(np.where(np.arange(len(SPANS)) == ci, np.nan, cond))))   # most-orthogonal ALTERNATIVE span (in-sample screen)
        f1, f2 = chosen, feat(ex, SPANS[bj])
        solo, joint = wf_ic([f1], tgt), wf_ic([f1, f2], tgt)              # OOS: chosen alone vs the pair
        keep = bool((joint - solo) >= 0.01)                              # the OOS joint gain DECIDES
        second[(head, ex)] = bj if keep else None
        print(f"  {ex}: best alt N={SPANS[bj]} IC|best {cond[bj]:+.3f};  OOS joint {joint:+.3f} vs solo {solo:+.3f} (Δ{joint - solo:+.3f})"
              f"  ->  {'KEEP 2nd span (adds OOS)' if keep else 'one span suffices (no OOS gain)'}")
    ax.axhline(0, color="0.7", lw=0.8); ax.set_xticks(range(len(SPANS))); ax.set_xticklabels([str(N) for N in SPANS])
    ax.set_xlabel("band span N (trade ticks)"); ax.set_ylabel("partial-IC | chosen span")
    ax.set_title(f"the family RE-SCORED conditional on the best pick — {head}"); ax.legend(fontsize=8)
fig.suptitle("RE-SCORED conditional on the best pick, per head (lit = orthogonal / adds; ~0 = diluted copy of the pick)", y=1.005)
fig.tight_layout(); plt.show()
""")

md(r"""
**Does a *second* time-scale add — per head?** The IC sweep picks the best span per head; a second span is only
worth feeding if it carries signal the first doesn't. We test that by **re-scoring the whole family conditional on
the chosen pick** — each span's IC recomputed as `partial-IC(span | chosen)`: its predictive power against the
head's target **controlling for the span we already picked** (the same partial-IC tool the echo-netting cell
uses; control = the chosen span). A span that stays **lit** is **orthogonal** — it adds new signal — while
a span that **collapses to ≈ 0** is a diluted copy of the pick. Because that line is **in-sample**, the
keep/drop **decision** is the overfitting-resistant **walk-forward joint-vs-solo** IC
(`wf_ic([chosen, alt]) − wf_ic([chosen])`): keep the second span only when the out-of-sample gain clears the ~0.01
floor. We run it for **both heads** (price → signed `z` vs the σ-return target; rate → `|z|` vs the count
target). On this block both find the second span a diluted copy (OOS gain ≈ 0), so one span per head suffices.
""")

md(r"""
**Now the gates** — and they are **two independent tests**, not one. Every *marginal* number is the
**walk-forward** mean (causal, purged); the one exception is the control-standalone **stratified** IC, which is
in-sample decoupled (its out-of-sample confirmation comes from the multi-block harness in `tools/oss`).

One caveat for **both heads**: the span is chosen by the in-sample heat-map *above* over the
**whole** block, so the walk-forward IC below is **post-selection and provisional** — mildly optimistic,
since the test folds also informed the span pick. Held-out span selection is deferred to the multi-block
`tools/oss` harness; here the choice is among a small 6-span family of near-identical neighbours, so the
selection bias is expected to be small (we don't measure it on this single block).

**Gate A — regime invariance** (the feature *alone*): is the feature's distribution **stable against our regime
diagnostics**, or does it *leak* it? **Control-free** checks: **scale** — its std across vol buckets (max/min,
want **< ~3**); and then, for **both the signed feature *and* its magnitude `|feature|`** (the rate head
receives the *signed* feature and can learn its magnitude, so a magnitude that tracks the regime would leak
into it), two leak modes against **both
regime coordinates** (vol *and* rate level): **tracking** — `|IC(·, level)|` ≈ 0 (the monotone test, want
**< ~0.05** for the signed feature, **< ~0.1** for the magnitude), and **dispersion** — the spread of its
per-decile *means* (want **< ~0.1**), which catches a *non-monotone* leak the monotone IC misses. Each closes
what the others miss: scale alone passes `z + c·vol_level` (flat std, mean rides the regime); the monotone IC
misses a *U-shaped* leak that only dispersion catches; and a feature flat in signed mean and scale can still
leak through its *magnitude* into the |·|-fed rate head, which only the magnitude checks see. The vol/rate
level is only the regime *coordinate* here (what we bucket/correlate against), **never** a control. Fail any
one of them = a level in disguise, not a feature. *(A z-score self-normalised by its own band σ should have a
steady scale and near-zero tracking — but we measure, never assume — that is the §2 ship-raw decision.)*

**Gate B — predictive signal**: does it predict? Because *a control can itself be a valid feature*,
"signal **over** the controls" only makes sense for a feature that **isn't** a control. `range_breakout`
is a price-extension **alpha** (not a regime descriptor), so Gate B is its **marginal** rank-IC over the
regime-invariant controls (the momenta — **never** the raw levels), all exchanges together and each on
its own, want ≳ 0.01. *(A **control-type** feature — a regime descriptor like σ_ev/λ_ev — is instead
judged on its **standalone** signal, since marginal-over-its-own-controls is circular.)* **Mechanical-coupling
guard:** the gate scores the **σ_ev price target** (`fwd_return/σ_ev`) by default, but `range_breakout`
divides by **no** model yardstick at all — it is self-normalised by its own band σ — so it shares no
denominator with either target, and there is no spurious ratio coupling to stratify away (`STRAT_VAR = None`).
We still print the within-σ_ev stratified standalone IC below as a *confirmation* that the unstratified IC isn't
a σ_ev artefact. *Regime-stable* (companion): is Gate B still positive **within** calm, mid, and wild vol — not
a one-regime artefact? *(Note: the marginal/standalone IC may be **negative** where the IC sign is reversion —
strength is `|IC|`, and the model keeps the sign.)*
""")

code(r"""
# Gates on the set of per-exchange features — symmetric; KEEP ALL exchanges. Marginal value is checked
# JOINTLY (do the exchanges together add over the controls?) and per-exchange (does each contribute?), never by picking one.
# GATE B — predictive signal.
#   alpha (or a control's CROSS-venue leg): MARGINAL rank-IC over the base momenta — a lead test.
#   control + OWN-venue leg: STANDALONE (marginal-over-own-momenta is circular).
# signal_ic / signal_ic_by_regime / stratified_ic / ic all come from boba.research.gates (the §5 wrappers
# bind base / FEATURE_KIND / STRAT_VAR / vol_regime). range_breakout is an alpha, so all legs take the marginal branch.

brk = {ex: feat_cache[(ex, SPANS[price_member[ex]])] for ex in EXCHANGES}        # one signed z per exchange (price head)
joint = signal_ic(list(brk.values()), own=False)
strat = signal_ic_by_regime(list(brk.values()), own=False)
pci = wf_marginal_ci(list(brk.values()), target)   # H5: per-fold spread + block-bootstrap CI for the headline marginal
print(f"joint Gate B marginal {joint:.3f}  |  per-fold {pci['per_fold']} (positive in {pci['pos']}/{pci['nf']})  |  block-bootstrap 90% CI [{pci['ci'][0]}, {pci['ci'][1]}]  (block ~{pci['block_len']} anchors ~{pci['block_s']}s)")
rep = brk[EXCHANGES[0]]                                 # representative leg — for the momenta-overlap sanity print only (Gate A runs PER EXCHANGE below)
# GATE A — regime invariance (vol_level/rate_level are the regime COORDINATE, never controls), run PER EXCHANGE
# (Gate A is a property of the output DISTRIBUTION, which differs by venue — bin is sub-ms fresh, byb/okx stale):
# scale, plus the signed feature AND its magnitude |feature| (which the signed-fed rate head can learn), each checked
# two ways x two coordinates — TRACK = |IC(., vol/rate level)| (monotone drift) and DISP = per-decile-mean dispersion
# vs vol OR rate (a NON-monotone, e.g. U-shaped, leak the monotone IC misses); each row reports the WORST coordinate.
# LOW-1 sanity: an "alpha" must NOT be a hidden regime descriptor -> its overlap with the momenta is small.
print(f"FEATURE_KIND={FEATURE_KIND!r} | feature vs momenta |IC|: rate {abs(ic(rep, rate_momentum)):.3f} vol {abs(ic(rep, vol_momentum)):.3f}  (large for an 'alpha' => it's really a control)")
gate_rows = [dict(gate="B · signal", detail="all exchanges together — marginal over the invariant controls", value=joint)]
gate_rows += [dict(gate="B · signal", detail=f"{ex} alone — marginal over the controls",
                   value=signal_ic([brk[ex]], own=False)) for ex in EXCHANGES]
# σ_ev-coupling confirmation: range_breakout divides by NO model yardstick (STRAT_VAR=None) — it is self-normalised by
# its own band σ. A within-σ_ev stratified standalone IC ≈ the unstratified standalone IC confirms σ_ev isn't
# manufacturing the rank-IC (no spurious ratio corr).
gate_rows += [dict(gate="B · coupling", detail=f"{ex} — signed z standalone IC, unstratified", value=round(ic(brk[ex], target), 3)) for ex in EXCHANGES]
gate_rows += [dict(gate="B · coupling", detail=f"{ex} — signed standalone IC WITHIN σ_ev strata (≈ unstratified => σ_ev not manufacturing it)", value=round(stratified_ic(brk[ex], target, sigma_at_anchor), 3)) for ex in EXCHANGES]
for ex in EXCHANGES:                                    # GATE A PER EXCHANGE (worst coordinate per check) — every venue, since the output distribution differs by venue
    a = gate_a(brk[ex])
    gate_rows += [dict(gate=f"A · regime-inv ({ex})", detail="feature scale across vol buckets (max/min) — HARD, want < ~3", value=a["scale"]),
                  dict(gate=f"A · regime-inv ({ex})", detail="|IC(feature, vol/rate level)| — signed mean-track, worst coord; want < ~0.05", value=a["track"]),
                  dict(gate=f"A · regime-inv ({ex})", detail="|IC(|feature|, vol/rate level)| — magnitude track (the signed-fed rate head can learn it), worst coord; want < ~0.1", value=a["mag"]),
                  dict(gate=f"A · regime-inv ({ex})", detail="non-monotone leak — per-decile-mean dispersion (mean & magnitude), worst; want < ~0.1", value=a["disp"])]
gate_rows += [dict(gate="regime-stable", detail=f"signal within {nm}-vol (companion: stay same-sign)", value=strat.get(r, float("nan")))
              for r, nm in [(0, "calm"), (1, "mid"), (2, "wild")]]
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(60)
pl.DataFrame(gate_rows)
""")

md(r"""
**Conclusion.** `range_breakout` is built **raw** — a Bollinger z-score self-normalised by its own band
width — and **Gate A** decides whether raw is shippable. Run **per exchange** (Gate A is a distribution
property — venues differ), the measured **scale** wanders only **1.05–1.09×** (≪ 3) and the signed feature
barely tracks the regime (mean-track **≤ 0.01**, ≪ 0.05). **The one Gate-A sub-check that does *not* fully
pass** is the **magnitude** track the rate head can learn, at the shipped **short** span (N=50, the noisiest
band): mag-track is **byb 0.107 (right at the ~0.1 bar), bin 0.119 (over), okx 0.143 (well over — ~43% above
the bar)** — so the `|z|` magnitude-track sub-gate is **breached on bin and okx and sits on the bar for byb**,
not "just cleared." This is a short-span effect (a narrower band makes `|z|` ride the vol regime more), with
dispersion **0.069 / 0.083 / 0.098** (under, or at, the ~0.1 bar). So the *signed* feature — what **both heads
are actually fed** — is clean everywhere (scale and signed mean-track all pass), but the `|z|` the rate head
can recover weakly-to-clearly tracks the vol regime on **all** venues at N=50 (worst on okx) — flagged to
watch across blocks, and a reason the **rate head** (gated separately below) leans on its echo-netted /
within-λ_ev numbers rather than the raw `|z|→count`. Because every **scale** and **signed mean-track** number
passes, the signed feature ships **raw** (no `/σ_ev`, no `/λ_ev`), with the breached `|z|` mag-track logged as
an open Gate-A item. Within σ_ev strata the standalone IC barely moves (byb 0.106→0.107, bin 0.175→0.173, okx
0.131→0.130), so the shared σ_ev isn't manufacturing it — as expected, since this feature never divides by
σ_ev (and its overlap with the momenta is tiny: |IC| rate 0.001, vol 0.016, confirming it really is an alpha,
not a control). **Gate B** — it adds **+0.155** walk-forward rank-IC over the controls *jointly*, positive in
**5/5** OOS folds with a tight block-bootstrap 90% CI **[0.149, 0.16]**, and each venue contributes on its
own (**bin 0.166, okx 0.124, byb 0.102**) — far above the ~0.01 floor; bin is the strongest single leg. The
companion keeps the marginal positive — and *rising* — across regimes (**0.129 / 0.154 / 0.183** for calm /
mid / wild). **But the raw price IC of the byb-own leg is inflated by the contemporaneous echo** (backward
+0.300 vs forward δ=0 +0.106) — the verdict (§10) quotes the **echo-netted** number for any echo-heavy leg,
never the raw δ=0 IC. Verdict for the **price head**: the *signed* feature both heads are fed clears every
local *mechanics* gate on this block, every exchange — but Gate A does **not** fully pass, because the N=50
`|z|` magnitude-track breaches the ~0.1 bar on bin/okx (and sits on it for byb), the open item to watch across
blocks; the *forward* edge lives in the **cross legs** (bin/okx), with byb-own suspect-echo — a single-block
candidate, shipping gated by §10. (The rate head is gated separately just below — its verdict is read off the
rate-head gate table, not inherited from these price-head gates.)
""")

md(r"""
**Now the rate head — its own gates, against the count target.** The block above gated the *price-span*
feature against the σ_ev **price** target. The rate head is fed a **different-span** feature (the
`rate_member` pick) and predicts the **count** target, so its verdict is **not** inherited from the price
gates — it gets the same two-gate battery here.

The rate head's signal lives in the **magnitude**: `|z| → count` is the §6 diagnostic, and the model is fed
the **signed** `z` and recovers `|·|` itself (a nonlinear head can). So **Gate B scores `|z|`** — a *linear*
score on the signed feature would read ≈ 0 precisely because the count relationship is symmetric, so `|z|` is
the honest proxy for what the nonlinear rate head extracts. **Coupling guard:** the price head used
`STRAT_VAR=None` because the breakout divides by no model yardstick; the magnitude path has no sign, and `|z|`
is self-normalised by its own band σ (no `1/λ_ev` factor) while `count/λ_ev ∝ 1/λ_ev`, so they share no
denominator — we add a within-λ_ev stratified line per exchange to confirm the marginal isn't a `1/λ_ev`
artefact regardless. **Gate A** is re-run **per exchange** on the rate-span feature (a distribution property —
it differs by venue *and* by span).
""")

code(r"""
# === RATE-HEAD GATES ===
# The price-head gates above ran the price-SPAN feature against the σ_ev price target. The rate head is fed a
# DIFFERENT-span feature (the rate_member pick) and predicts the COUNT target, so its verdict can't be inherited
# from the price gates — it gets the SAME two-gate battery here, against the count target.
#  * The rate head's signal lives in the MAGNITUDE: |z|->count is the §6 diagnostic, and the model is fed the
#    SIGNED z and recovers |.| internally (a nonlinear head can). So Gate B scores |z| — a LINEAR score on the
#    signed feature reads ~0 because the count relationship is symmetric; |z| is the honest proxy.
#  * Coupling guard: |z| is self-normalised by its own band σ (no 1/λ_ev factor) while count/λ_ev ~ 1/λ_ev — no shared
#    denominator — but we add a within-λ_ev stratified line per exchange to confirm the marginal is not a 1/λ_ev artefact.
#  * Gate A is re-run PER EXCHANGE on the rate-span feature (a distribution property — it differs by venue AND by span).
brk_rate = {ex: feat_cache[(ex, SPANS[rate_member[ex]])] for ex in EXCHANGES}
mag_rate = {ex: np.abs(brk_rate[ex]) for ex in EXCHANGES}

joint_rate  = signal_ic(list(mag_rate.values()), own=False, tgt=rate_target)            # |z|->count, marginal over the controls, all venues together
strat_rate  = signal_ic_by_regime(list(mag_rate.values()), own=False, tgt=rate_target)  # the calm/mid/wild companion
rci = wf_marginal_ci(list(mag_rate.values()), rate_target)   # H5: per-fold spread + block-bootstrap CI for the rate-head headline marginal
print(f"joint Gate B (rate) marginal {joint_rate:.3f}  |  per-fold {rci['per_fold']} (positive in {rci['pos']}/{rci['nf']})  |  block-bootstrap 90% CI [{rci['ci'][0]}, {rci['ci'][1]}]  (block ~{rci['block_len']} anchors ~{rci['block_s']}s)")
# gate_a (worst-coordinate Gate A, used for both heads) is bound in §5.

rate_rows  = [dict(gate="B · signal (rate)", detail="all exchanges together — marginal |z|->count over the controls; want >~ 0.01", value=joint_rate)]
rate_rows += [dict(gate="B · signal (rate)", detail=f"{ex} alone — marginal |z|->count over the controls",
                   value=signal_ic([mag_rate[ex]], own=False, tgt=rate_target)) for ex in EXCHANGES]
rate_rows += [dict(gate="B · coupling (rate)", detail=f"{ex} — |z|->count WITHIN λ_ev strata (stays +ve => not a 1/λ_ev artefact)",
                   value=round(stratified_ic(mag_rate[ex], rate_target, lam_at_anchor), 3)) for ex in EXCHANGES]
for ex in EXCHANGES:
    a = gate_a(brk_rate[ex])
    rate_rows += [dict(gate=f"A · regime-inv ({ex})", detail="feature scale across vol buckets (max/min) — HARD, want < ~3", value=a["scale"]),
                  dict(gate=f"A · regime-inv ({ex})", detail="|IC(feature, vol/rate level)| — signed mean-track, worst coord; want < ~0.05", value=a["track"]),
                  dict(gate=f"A · regime-inv ({ex})", detail="|IC(|feature|, vol/rate level)| — the rate head sees |z|, worst coord; want < ~0.1", value=a["mag"]),
                  dict(gate=f"A · regime-inv ({ex})", detail="non-monotone leak — per-decile-mean dispersion (mean & magnitude), worst; want < ~0.1", value=a["disp"])]
rate_rows += [dict(gate="regime-stable (rate)", detail=f"|z|->count within {nm}-vol (companion: stay positive)", value=strat_rate.get(r, float("nan")))
              for r, nm in [(0, "calm"), (1, "mid"), (2, "wild")]]
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(70)
pl.DataFrame(rate_rows)
""")

md(r"""
**Conclusion (rate head).** The rate-span feature (all venues N=50) clears the same battery against the
**count** target. **Gate B** — the marginal `|z| → count` over the controls is **+0.086** jointly (positive in
**5/5** OOS folds, CI **[0.08, 0.092]**), **+0.087** (bin) and **+0.041** (okx) on their own well above the
~0.01 floor; **byb's own** rate leg is **+0.018** (echo-flavoured, consistent with the echo-netting read below —
kept because leadership rotates, but watch it). *Within* **λ_ev strata** the signal stays robustly positive
(**0.139** bin, **0.078** okx, **0.055** byb), so the marginal isn't a `1/λ_ev` arithmetic artefact. **Gate A**
is the same per-exchange read as the price head (the rate-span feature is also N=50): scale **1.05–1.09**
(≪ 3), signed mean-track **≤ 0.01** (≪ 0.05); but the `|z|` the rate head actually sees **exceeds the ~0.1
mag-track bar on bin and okx and sits on it for byb** (**byb 0.107 at the bar, bin 0.119 over, okx 0.143 — well
over, ~43% above**; dispersion 0.069 / 0.083 / 0.098) — so this Gate-A sub-check is **breached, not cleared**,
the short-span magnitude caveat that here bites the rate head directly (it is the head that sees `|z|`). The
**companion** is positive in every regime (**0.075 / 0.084 / 0.081** for calm / mid / wild). So the rate head
clears the **signal** gates on this block against the count target — the "**both heads, every exchange**"
signal verdict is *earned* here, not inherited; the cross legs (bin/okx) carry it, byb-own is the
weak/echo-flavoured one — but the N=50 `|z|` mag-track **breach** on bin/okx (and at-the-bar on byb) is the
open Gate-A item to watch across blocks, since it is exactly the magnitude this head consumes.
""")

md(r"""
### Self-test — does the stratify-by-yardstick decouple actually work?

The alpha worked example above never exercises the **control** branch (`FEATURE_KIND="control"`, own-leg,
`STRAT_VAR` set), so this small cell checks the one thing that branch relies on: that **stratifying by the
shared yardstick kills a purely-mechanical ratio coupling while keeping a real signal**. We build, on the
grid, a `count/λ_ev`-style rate-head target and two control *ratios that both divide by `λ_ev`* (so
`STRAT_VAR = lam_at_anchor`):
- **pure-spurious** — a numerator with **no** real link to the count, divided by `λ_ev`. Its **raw** Spearman
  against the target is large (the shared `1/λ_ev` manufactures it); **stratified** by `λ_ev` it collapses to ≈0.
- **real-signal** — a numerator that genuinely tracks the count, divided by `λ_ev`. Its raw IC is inflated by
  the same coupling, but the **stratified** IC keeps the genuine within-stratum signal.

If `signal_ic` (the real gate path) returns ≈0 for the spurious ratio (vs a large raw IC) and recovers the
real one, the decouple does what Gate B's control branch needs.
""")

code(r"""
# SELF-TEST of the control-branch decouple (does NOT touch the alpha worked example above).
rng = np.random.default_rng(0)
# a count/λ_ev-style rate-head target on the grid (the same shape rate_target has)
st_count  = (cum_mv[np.searchsorted(byb_rx, anchor_ts + HORIZON_NS, "right")]
             - cum_mv[np.searchsorted(byb_rx, anchor_ts, "right")]).astype(float)   # byb moves over the next 100 ms
st_target = st_count / np.maximum(lam_at_anchor, 1e-9)                               # count ÷ λ_ev (the scored ratio target)
st_lam    = lam_at_anchor                                                            # the shared yardstick == STRAT_VAR

# PURE-SPURIOUS control ratio: numerator independent of the count, divided by the SAME λ_ev -> shares only the denominator.
# Keep the numerator nearly CONSTANT so the shared 1/λ_ev dominates the ratio's variance -> a fake raw IC the stratify removes.
spur_num   = 1.0 + 0.25 * rng.standard_normal(len(anchor_ts))                        # constant-ish, count-independent numerator
spur_ratio = spur_num / np.maximum(st_lam, 1e-9)
# REAL-signal control ratio: numerator that genuinely predicts the count, divided by the SAME λ_ev.
real_num   = st_count + 3.0 * rng.standard_normal(len(anchor_ts))                    # tracks the count (with noise)
real_ratio = real_num / np.maximum(st_lam, 1e-9)

def _raw(f, y):
    m = np.isfinite(f) & np.isfinite(y); return float(spearmanr(f[m], y[m]).statistic)
print("control-branch decouple self-test — THROUGH signal_ic (target = count/λ_ev, STRAT_VAR = λ_ev, heavy-tailed):")
spur_raw, real_raw = _raw(spur_ratio, st_target), _raw(real_ratio, st_target)
_k, _s = FEATURE_KIND, STRAT_VAR                                          # BLOCKER-2: exercise the REAL gate path (signal_ic), not stratified_ic directly
try:
    globals().update(FEATURE_KIND="control", STRAT_VAR=st_lam)           # st_lam == lam_at_anchor is genuinely heavy-tailed (ISSUE-5 stress)
    spur_gate = signal_ic([spur_ratio], own=True, tgt=st_target)
    real_gate = signal_ic([real_ratio], own=True, tgt=st_target)
finally:
    globals().update(FEATURE_KIND=_k, STRAT_VAR=_s)
print(f"  pure-spurious:  raw Spearman {spur_raw:+.3f}  ->  signal_ic {spur_gate:+.3f}   (want raw large, gated ~0)")
print(f"  real-signal:    raw Spearman {real_raw:+.3f}  ->  signal_ic {real_gate:+.3f}   (want signal kept)")
assert abs(spur_gate) < 0.04,                      f"stratify failed to kill the spurious heavy-tailed coupling ({spur_gate:+.3f})"
assert abs(spur_raw) > 3 * abs(spur_gate) + 0.04,  f"spurious raw IC not materially larger than gated ({spur_raw:+.3f} vs {spur_gate:+.3f})"
assert abs(real_gate) > 0.05,                      f"stratify destroyed the real signal ({real_gate:+.3f})"
print("  OK — the gate's control path decouples the spurious heavy-tailed ratio and keeps the real signal.")
""")

md(r"""
## Is the edge real prediction, or an echo of the move already underway?

A feature can be perfectly causal and still not *predict*: if its apparent edge is the price move **already
underway** at the anchor, you can't capture it — by the time you observe, decide, and act, that move is gone — and
a purely *contemporaneous* feature can post a positive forward IC from window overlap alone. So before trusting a
forward IC, **net out the echo**: measure the feature's correlation with the **forward** return
(`[anchor, anchor+100 ms]`) *controlling for the move that already happened* (`[anchor−100 ms, anchor]`). The
**backward IC** sizes the echo; the **echo-netted** forward IC is what survives once it's partialled out — the
genuinely forward-looking edge. (It's the same partial-IC tool §6 uses to test a second time-scale — here the
control is the trailing move instead of the chosen span.)
""")

code(r"""
# Echo-netting: is the edge real prediction, or just re-reporting the move ALREADY underway at the anchor?
rep_ex = EXCHANGES[0]                                   # one exchange to illustrate; every feature carries its own
def _ic(feat, ret):
    v = np.isfinite(feat) & np.isfinite(ret)
    return spearmanr(feat[v], ret[v]).statistic if v.sum() > 100 else float("nan")
def _mid_at(t):                                         # byb merged mid at-or-before t (causal)
    idx = np.searchsorted(byb_rx, t, "right") - 1; return np.where(idx < 0, np.nan, byb_mid[np.clip(idx, 0, len(byb_mid) - 1)])   # nan before byb's first quote
def _ret(t0, t1): return np.log(_mid_at(t1) / _mid_at(t0))
def _partial_ic(f, y, t):                               # partial rank-IC of f with y, CONTROLLING for t
    v = np.isfinite(f) & np.isfinite(y) & np.isfinite(t)
    if v.sum() <= 100: return float("nan")
    rfy = spearmanr(f[v], y[v]).statistic; rft = spearmanr(f[v], t[v]).statistic; rty = spearmanr(t[v], y[v]).statistic
    return (rfy - rft*rty) / np.sqrt(max((1.0 - rft**2) * (1.0 - rty**2), 1e-12))

signed = range_breakout(rep_ex, SPANS[price_member[rep_ex]])   # the chosen price-head feature (signed z, this venue's pick)
_fwd0  = _ret(anchor_ts, anchor_ts + HORIZON_NS)          # forward 100 ms return — what we predict
_trail = _ret(anchor_ts - HORIZON_NS, anchor_ts)          # the move ALREADY underway at the anchor
raw_ic   = _ic(signed, _fwd0)                             # raw forward IC
back_ic  = _ic(signed, _trail)                            # backward IC — sizes the contemporaneous echo
echo_net = _partial_ic(signed, _fwd0, _trail)             # forward IC NETTED of the echo (controls for the trailing move)

fig, ax = plt.subplots(figsize=(6.4, 4.0))
labels = ["raw forward\nIC (δ=0)", "backward IC\n(echo size)", "echo-netted\nforward IC"]
vals = [raw_ic, back_ic, echo_net]
bars = ax.bar(labels, vals, color=["C0", "0.6", "C2"])
ax.axhline(0, color="0.7", lw=0.8); ax.set_ylabel("rank-IC"); ax.set_title(f"echo-netting — {rep_ex}, price head")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
fig.tight_layout(); plt.show()
print(f"raw forward IC {raw_ic:+.3f};  backward (echo) IC {back_ic:+.3f};  echo-netted forward IC {echo_net:+.3f}  (the raw->netted shortfall is echo)")
""")

md(r"""
**Net the echo out — don't report a forward IC a contemporaneous predictor could fake.** The cleanest
"is this real prediction?" number is the **echo-netted (partial) forward IC** above: the feature's rank-IC with
the *forward* return *controlling for the trailing* `[anchor−100 ms, anchor]` return. If a big raw IC collapses
once the trailing move is partialled out, the feature was mostly re-reporting the move already underway — report
the **netted** number in the verdict, not the raw IC. (A near-zero netted IC alongside a large backward IC is the
one true non-signal: all echo, no prediction.) For `range_breakout` the **byb-own** leg is the echo-prone one —
its `z` numerator `mid − ema(mid)` shares the anchor mid with the target — so its raw IC is heavily contemporaneous
and the netted number is the one to trust; the **cross legs** (bin/okx → byb) are where a surviving netted IC is a
genuine lead.

**Cross-venue legs: a freshness lead is *real edge*, not an artifact to coarsen away.** The data is recorded
on a production box in the target datacenter, so each event's `rx_time` is exactly the timing you'd see live —
there is **no recording/snapshot artifact** to rule out. So when okx's book moves before byb's reflects it
(byb/okx top-of-book is stale p90 ~100–160 ms vs bin's sub-ms), that lead is **genuine and exploitable**, and
the *mechanism* (economic price-discovery vs pure latency lead-lag) is irrelevant to P&L. Do **not** coarsen the
foreign feed to byb's cadence — that throws the edge away. *(A freshness lead would only be fake if the
recording's cadence didn't match production — e.g. a backtest on vendor snapshots; not the case here, where the
recording* is *production timing.)*
""")

md(r"""
## 7. What the prediction actually looks like

A single correlation number hides *how* the feature changes the outcome. So group the data by
the feature and look at the real distributions the two heads care about:
- **price head:** byb's next return for low / middle / high `z` values — it should tilt one way
  as `z` turns positive and the other as it turns negative (the *direction* of the tilt is the
  continuation/reversion sign §6 measured);
- **rate head:** how the number of upcoming moves grows as `|z|` grows.
""")

code(r"""
rep_ex = EXCHANGES[0]                                                          # byb's own breakout shown to illustrate the shape; the model uses every exchange
signed = feat_cache[(rep_ex, SPANS[price_member[rep_ex]])]
absmag = np.abs(feat_cache[(rep_ex, SPANS[rate_member[rep_ex]])])             # deliberately the RATE-head span (rate_member), not the price-head span
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.6))

# price head: forward σ-return distribution across signed-breakout buckets
lo, hi = np.nanpercentile(signed, [10, 90])
groups = [("strong −", signed <= lo, "C3"), ("≈ 0", (signed > lo) & (signed < hi), "0.6"),
          ("strong +", signed >= hi, "C0")]
bins = np.linspace(-8, 8, 81)
for lab, m, col in groups:
    axA.hist(np.clip(target[m & np.isfinite(target)], -8, 8), bins=bins, density=True,
             histtype="step", color=col, lw=1.7, label=f"{lab}:  mean {np.nanmean(target[m]):+.2f}")
axA.set_yscale("log"); axA.set_xlabel("forward σ-return"); axA.set_ylabel("density (log)")
axA.set_title("price head: return distribution tilts with signed breakout z"); axA.legend(fontsize=8)

# rate head: forward move-count shifts up with |z|
# raw fwd_count is shown here only for intuition; the rate head's actual target is rate_target = fwd_count / λ_ev (count ÷ λ_ev)
dec = np.digitize(absmag, np.nanpercentile(absmag, np.arange(10, 100, 10)))
axB.plot(range(10), [fwd_count[dec == b].mean() for b in range(10)], "o-", color="C3", label="mean count  E[K]")
axB.plot(range(10), [(fwd_count[dec == b] >= 1).mean() for b in range(10)], "s--", color="C1", label="P(K ≥ 1)")
axB.set_xlabel("|breakout z| decile (small → large extension)"); axB.set_ylabel("forward 100 ms")
axB.set_title("rate head: move-count distribution shifts up with |z|"); axB.legend(fontsize=8)
fig.tight_layout(); plt.show()
""")

md(r"""
**Conclusion.** The feature moves the *actual outcome distributions* the way the two heads need — not just a
single summary number (shown here for byb's own breakout; the model uses every venue). **Price head (left):**
the forward-return distribution shifts bodily with the *signed* `z` — a **positive** mean return for the
strong-positive group and a **negative** mean for the strong-negative group, with the flat ≈0 group between
(read the group means in the legend) — and the **direction** of that shift is **continuation**, exactly the
positive IC sign §6 read off the data. **Rate head (right):** the mean move-count `E[K]` and `P(K ≥ 1)` climb
with the `|z|` decile — a large extension precedes more byb moves. So both heads have something to learn here,
in the directions §1 predicted. (For byb's own breakout this tilt is partly the contemporaneous echo the
echo-netting section nets out; the cross-venue legs carry the genuine forward version of the same shape.)
""")

md(r"""
## 8. Input shaping for the network

This is a *different* step from §2. There we established the feature is *self*-normalised by its
band width (the "regime division"), so it means the same thing in any market. Here we reshape the
feature for the neural network's input — roughly centred, unit-scale, no wild outliers.

Pick the **lightest** transform that does the job. Plot the feature (left) and compare a few
candidates by how close to a clean bell curve they make it (right — points on the diagonal are
perfectly normal). A z-score is already roughly centred and unit-scale, so it usually only needs
a mild clip for the tail; `range_position = tanh(z)` is the natural **bounded** companion (it
*is* the bounded atom). The plot makes the choice for you.
""")

code(r"""
from scipy.stats import skew, kurtosis, rankdata, norm
rep_ex = EXCHANGES[0]                                                          # one exchange shown to illustrate the shape (same construction for every exchange)
f = feat_cache[(rep_ex, SPANS[price_member[rep_ex]])]; f = f[np.isfinite(f)]
med = np.median(f); mad = 1.4826 * np.median(np.abs(f - med)) + 1e-12; rz = (f - med) / mad
cand = {"z-score": (f - f.mean()) / (f.std() + 1e-12),
        "tanh (= range_position)": np.tanh(f),
        "robust + clip ±4": np.clip(rz, -4, 4),
        "rank-Gaussian": norm.ppf((rankdata(f) - 0.5) / len(f))}
print(f"feature: std={f.std():.2f}  skew={skew(f):+.2f}  excess_kurt={kurtosis(f):.1f}  (0 = normal)")
for name, v in cand.items():
    print(f"  {name:24} excess_kurt={kurtosis(v):>6.1f}   max|·|={np.abs(v).max():.2f}")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.6))
axA.hist(f, bins=120, density=True, color="C0", alpha=.85, label="feature (z)")
xg = np.linspace(np.percentile(f, 0.05), np.percentile(f, 99.95), 300)
axA.plot(xg, norm.pdf(xg, f.mean(), f.std()), "k--", lw=1, label="normal (same σ)")
axA.set_yscale("log"); axA.set_xlabel("range_breakout z (band-σ units)"); axA.set_ylabel("density (log)")
axA.set_title(f"feature distribution (excess kurt {kurtosis(f):.1f})"); axA.legend(fontsize=8)

q = norm.ppf((np.arange(1, len(f) + 1) - 0.5) / len(f)); sub = np.linspace(0, len(f) - 1, 4000).astype(int)
for name, v in cand.items():
    axB.plot(q[sub], np.sort(v)[sub], lw=1.5, label=name)
axB.plot([-5, 5], [-5, 5], "k:", lw=1, label="perfect normal")
axB.set_xlim(-5, 5); axB.set_xlabel("normal quantile"); axB.set_ylabel("normalised feature quantile")
axB.set_title("how to normalise — QQ vs N(0,1): on-diagonal & bounded wins"); axB.legend(fontsize=8)
fig.tight_layout(); plt.show()
""")

md(r"""
**Conclusion.** The printout settles it: `range_breakout` is a z-score, near-**symmetric** (skew **+0.02**)
and order-unity in scale — but its **measured std is 0.83, not exactly 1**: `σ_band =
√(ema(mid²) − ema(mid)²)` is an *exponentially-weighted population* variance, which is **biased low** (the
usual EMA downward bias) and, at the shipped small spans, **noisy** — a jumpy denominator inflates the `|z|`
tails while pulling the bulk std under 1. So "z-score" here means "self-normalised by its own band σ," **not**
"std exactly 1" — read the printed std, don't assume it. The tails are only **mildly** fat (excess kurtosis
**+0.8**), but a band z still throws the occasional large extension, so a plain z-score leaves a **max|·| =
5.95** — over the "no wild outliers" bar. The lightest transform that *meets* the bar is a robust z-score
followed by a clip (**robust + clip ±4 → max|·| = 4.0**, excess kurt +0.5), and `tanh(z) = range_position` is
the natural **bounded** alternative (**max|·| = 1.0**) — it maps the whole real line into (−1, +1), killing the
tail entirely at the cost of compressing large extensions. Feed **`range_position = tanh(z)`** where you want a
bounded input and **clipped `breakout_magnitude = z`** where you want the unbounded extension preserved; the
QQ-plot shows both sit close to the diagonal in the body. The heavier rank-Gaussian (excess kurt −0.0) would
just throw information away here.

**Still to run** (same machinery): whether the feature adds over features we already have;
whether the signal is steady day-to-day; and whether it survives out-of-sample across a
market-regime change. (The leak check *inside* volatility buckets is no longer pending —
Gate A's checks and the regime-stable companion now run it.)
""")

md(r"""
## 9. When is per-exchange worth it? (for poolable features)

`range_breakout` is computed **per venue** (each venue's own mid against its own band), and every venue's
leg predicts byb — byb's own leg plus the cross legs (okx/bin → byb). So unlike a poolable trade-flow feature
there is no single "pooled" version to compute: pooling mids *across* venues would mix instruments with
different reference levels and is not a meaningful breakout. The per-exchange structure is **intrinsic** —
keep them all. The real question for *this* feature is the **lead/lag cross-over across band spans** — when
does *another* venue's breakout add over byb's own?

> **The table below is an illustrative example for a poolable trade-flow feature — it is NOT
> computed for `range_breakout`** (whose real per-exchange numbers are the heat-maps in §6 and the
> echo-netted forward IC above). It just shows the typical pattern, predicting byb's next
> 100 ms. The time-scale column is labelled in ms / seconds only for readability — those wall-clock figures are
> just a translation of points on the trade-span N clock; the actual sweep is over trade-span N, as everywhere else.

| time-scale | pooled | best single | per-exchange | what it means |
|---|---|---|---|---|
| ≤ 50 ms | 0.22 | **0.23** | 0.22 | one exchange is enough — splitting just adds noise |
| ~500 ms | 0.11 | 0.12 | **0.16** | keep per-exchange — exchanges genuinely differ here |
| ≥ 2.5 s | 0.08 | 0.08 | 0.09 | exchanges agree — any one will do |

So: at the shortest scales the signal lives in one venue; in a **middle band** the venues genuinely differ
(who broke out first) and keeping them separate pays off; at long scales they all converge. For
`range_breakout` the mechanism is **arbitrage lead/lag** — one venue breaking out a beat before another,
closed within milliseconds — so the cross-over is exactly the §6 sweep read across the span axis: compare each
venue's IC to byb's at short vs long N. Keep all legs as the safe default, and let the model weight whichever
venue is leading at the current scale. (Longer prediction horizons push the useful band wider, so in general
sweep the horizon too.)
""")

md(r"""
## 10. The verdict, and what it takes to ship

**Lead with the honest comparison, not the headline IC.** For the **byb-own** leg the backward
(already-happened) IC is large — **+0.300** vs the forward δ=0 **+0.106** — the textbook **echo** signature:
a byb breakout `z` is mechanically large because byb's mid *just moved*, and its numerator `mid − ema(mid)`
shares the anchor mid with the target. So the raw in-sample byb-own IC (and the §6 |IC|-argmax) **overstate**
that leg's edge: the echo-netting cell shows it collapses **0.106 → 0.033** once the trailing move is partialled
out. The number that counts is the **echo-netted forward IC**, and on this block the surviving forward edge is
carried by the **cross legs** (bin/okx → byb), where a lead is even coherent (their `z` numerator does not share
byb's anchor mid), while the **byb-own** leg is suspect-echo on both heads. The IC **sign is continuation (+)**
everywhere on this block, and the |IC|-argmax span is **N=50** (the shortest band) for **every** venue, both heads.

**The mechanics are clean, and the gates mostly pass — with one Gate-A sub-check breached:**
- **Causal & parity.** §4's production-shaped streaming build — its EMA state is `boba.ema`'s `LiveFrontEMA`,
  the actual production code — matches §3's vectorized `lfilter` path at the **shipped** spans (all N=50) over
  the **full** grid, with its **band legs bit-exact** (worst **3.6e-12**) and the **derived `z` to float32**
  (worst **1.4e-5** — the variance cancellation amplifies the ~1e-12 leg round-off in `z` at the small N=50
  band, worst at the near-zero-width `z ≈ 0` points on okx; the float32 bar is CLAUDE.md's production-vs-§3
  tolerance). EMA-based band only — no rolling max/min, no boxcar.
- **Regime-invariant for the *signed* feature, RAW — but the `|z|` mag-track sub-gate is breached.** Per
  exchange, the **scale** (**1.05–1.09**, ≪ 3) and the **signed mean-track** (**≤ 0.01**, ≪ 0.05) both pass,
  so the *signed* `z` — what both heads are fed — ships **raw**, no `/σ_ev`, no `/λ_ev` (the within-σ_ev
  standalone IC confirms σ_ev isn't manufacturing the signal). **The exception:** at the shipped short span
  (N=50) the `|z|` magnitude-track **exceeds the ~0.1 hard bar on bin and okx and sits on it for byb** —
  **byb 0.107 (at the bar), bin 0.119 (over), okx 0.143 (well over, ~43% above)**; dispersion 0.069–0.098 —
  so Gate A does **not** fully pass for the magnitude the rate head can recover (a short-span effect: a narrow
  band makes `|z|` ride the vol regime). The signed feature stays clean, but this breached sub-gate is the open
  Gate-A item to watch across blocks.
- **Predictive.** Gate B joint price **+0.155** (5/5 folds, CI [0.149, 0.16]); rate **+0.086** (5/5, CI
  [0.08, 0.092]); both companions positive across calm/mid/wild.

**How to feed it, for the legs whose echo-netted edge holds up:** feed the **signed**
`breakout_magnitude = z` to both heads, every venue (let the model weight whoever genuinely leads), with
`range_position = tanh(z)` as the bounded companion; carry the §6 |IC|-argmax shipped member (N=50) per venue
(§6's conditional partial-IC found no second span adds OOS, so one span per head suffices). The IC **sign** is
continuation here — the model sees the signed `z` and learns the sign itself.

**Verdict: a single-block candidate — keep the cross legs (bin/okx), whose edge survives echo-netting; the
byb-own leg is suspect-echo on both heads.** The feature is mechanically sound, causal, parity-validated, and
regime-invariant *in its signed form* on this block — but with **one open Gate-A item**: at N=50 the `|z|`
magnitude-track **breaches the ~0.1 bar on bin (0.119) and okx (0.143) and sits on it for byb (0.107)**, so
Gate A does not fully pass for the magnitude the rate head consumes (the signed feature both heads are fed
stays clean). Its headline price edge is partly **contemporaneous echo** (heaviest on the byb-own leg, which
nets from +0.106 to ~+0.033), so ship on the **echo-netted** number, not the raw δ=0 IC. Shipping is gated by
the checklist — and resolving the `|z|` mag-track breach across blocks is part of that gate.

**To ship:**
- [ ] the streaming (constant-work-per-trade) builder, matching this analysis version — **§4 `boba.ema`
      streaming build, band legs bit-exact / derived z float32, at shipped spans, full grid**
- [ ] the parity check (§4) and tests, passing — **§4 `boba.ema` build vs §3: legs bit-exact / z float32**
- [ ] the gate results recorded — incl. the **echo-netted** forward IC (raw δ=0 vs netted of the trailing move)
- [ ] the chosen heads, the IC sign (continuation/reversion) per span, the band spans, and the RAW
      normalisation decision written down, with the yardstick spans
- [ ] the data quirks handled (bad zero-price prints; the right price source per exchange; REF centering for
      the variance cancellation)
""")

nb = {
    "cells": [
        ({"cell_type": "markdown", "id": f"c{i}", "metadata": {}, "source": s}
         if t == "markdown" else
         {"cell_type": "code", "id": f"c{i}", "metadata": {}, "execution_count": None, "outputs": [], "source": s})
        for i, (t, s) in enumerate(cells)],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
out = Path(__file__).resolve().parents[1] / "range_breakout_v2.ipynb"   # notebooks/features/range_breakout_v2.ipynb
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, "with", len(cells), "cells")
