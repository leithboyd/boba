"""Generate notebooks/features/trade_rate_normalised.ipynb — feature analysis for `trade_rate_normalised`.

Feature: ema(trade rate) / yardstick — a single EMA of a venue's TRADE rate (trades/sec), regime-normalised,
built PER EXCHANGE (byb, okx, bin), each from THAT venue's OWN trade-events, all decayed on the ONE shared
trade clock. Built like λ_ev but counting TRADE-events (one per the venue's trade-timestamp) instead of byb
mid-moves:  trade_rate(ex, N) = W_trades_ex(N) / E_dt(N), where W_trades_ex is the span-N exp-weighted count
of that venue's trade-events and E_dt the span-N exp-weighted seconds per shared trade-timestamp.

THREE EXPLORATION AXES baked in from the start:
 1. PER-EXCHANGE: byb, okx AND bin trade rate, each as a predictor of byb's move; all kept; §6 per-exchange
    heat-maps; §9 short-scale-per-exchange / long-scale-single sweep.
 2. FULL LOOKBACK FAMILY: the whole EMA span grid swept; pick per head off the grid.
 3. NORMALISATION (the "/ yardstick" axis): build & compare un-normalised baseline, / σ_ev (vol yardstick),
    AND / λ_ev (rate yardstick) — report which (if any) helps.

CRUCIAL control-circularity point: a venue's TRADE rate (trades/sec) is NOT λ_ev (byb mid-MOVE rate).
Trades need not move the mid, and the mid moves without trades — so trade-rate features may carry marginal
value the move-rate control (rate_momentum / rate_level, both from λ_ev) does not. We report the
marginal-over-controls honestly, per venue.

Clones the methodology of the price_dislocation template (per-exchange OTHERS structure, σ_ev/λ_ev yardsticks,
§6 heat-maps, §9 per-exchange-vs-single pooling, the §4 oracle, the hygiene gates). Reuses the σ_ev / λ_ev
machinery EXACTLY from the template, and the qty/rate machinery from the sibling activity builders.
"""
import json
from pathlib import Path

cells = []
def md(s):   cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r"""
# Feature analysis — `trade_rate_normalised`

This notebook follows the project's feature-analysis **method** (a copy of the `price_dislocation`
template) on a new feature: **`trade_rate_normalised`** — a single EMA of a venue's **trade rate**
(trades per second), optionally **regime-normalised** by a yardstick. The text says what to do and why;
the code does it.

We build features for a model that forecasts one exchange's mid-price about 100 ms ahead. Three crypto
exchanges appear throughout — **byb** (Bybit), **bin** (Binance), **okx** (OKX) — and **byb is the
target**: the one we predict. ("Mid-price" = the midpoint between the best buy and best sell quote.) The
next section recaps the model these features feed.

**The feature, in one line.** A venue's **trade rate** is how many of *that venue's* trades print per
second. We measure it the way the model measures byb's move-rate `λ_ev` — a span-`N` exp-weighted
**count** of the venue's trade-events ÷ a span-`N` exp-weighted **seconds-per-trade** — but counting
**trades**, not byb mid-moves:

`trade_rate(ex; N) = W_trades(ex, N) / E_dt(N)`   (the venue's trades per second)

Then we optionally divide by a regime **yardstick** to make it comparable across calm and busy markets:

`trade_rate_normalised(ex; N) = trade_rate(ex; N) / yardstick`,   yardstick ∈ { 1 (baseline), σ_ev, λ_ev }.

A high value means that venue is **trading more intensely than usual** — an activity surge. The hypothesis
is that elevated trade intensity **predicts intensity**: more byb mid-moves are about to follow.

**This is a per-exchange feature (like the template's dislocation gap).** Exactly as `price_dislocation`
computes a gap for *each* non-byb venue, we compute a trade rate for **each** venue — byb, okx **and**
bin — each from that venue's OWN trade-events, and ask whether it predicts **byb's** next move. The byb
leg is the *target's own* trade intensity; the okx/bin legs are **cross-venue** trade-intensity lead/lag.

**A feature is "done" when two checks pass:**
- **The oracle (§4)** — the optimized build really computes what we think it does (bit-exact vs a
  dead-simple independent streaming build, on a real block, **for every per-exchange leg**).
- **The hygiene gates (§5/§6)** — the signal is real and holds in any market, not just an echo of "the
  market is busy right now."

> **Control-circularity caveat — read this first, it is the crux of this feature.** The model's **rate
> controls** (`rate_momentum`, `rate_level`) are built from **`λ_ev` = byb's mid-MOVE rate**. A venue's
> **trade** rate is a *different* quantity: **trades need not move the mid** (a trade that doesn't cross
> a level leaves the BBO unchanged), and **the mid moves without trades** (a quote revision moves the
> mid with no print). So `trade_rate` is **not** circular with the rate control the way
> `mid_rate_surge`'s byb leg was — and it may carry marginal value the move-rate control does **not**.
> The honest test is the **marginal over the rate/vol controls**, per venue (§6/§9), reported straight.
""")

md(r"""
## The model these features feed: two heads

A feature is only worth something if it helps the model predict, so it's worth knowing what the model
does. We forecast how byb's mid-price moves over the next ~100 ms, split into two simpler questions — the
two **heads**:

**Price head — which way and how far?** Over the next few price-moves, what is the *signed* move
(direction *and* size, together), in units of byb's recent **volatility** — the **volatility yardstick**
`σ_ev` (the exp-weighted RMS of byb's *actual* mid-moves)? Target: `price change ÷ σ_ev`.

**Rate head — how many moves?** Busy markets pack many price-moves into the window, quiet ones few. This
head predicts the *count* of moves over the next 100 ms, measured against the recent pace — the **rate
yardstick** `λ_ev` — so its target is `count ÷ λ_ev`.

**What's a yardstick?** A causal, trailing estimate of byb's *volatility* (`σ_ev`) and move *rate*
(`λ_ev`), from past data only. Both are EMAs **decayed on the trade clock** (`α = 2/(span+1)`) but
**updated between trades** — they react to every byb mid-move, so they read live at every instant. They
use one fixed span, `YARDSTICK_N`. Critically, `λ_ev` counts byb **mid-moves**, while this feature counts
**trades** — the two are genuinely different clocks (see the caveat above).

**Why split into two heads?** A move over a window is *how many* little moves times *how big* each one is.
Pulling "how many" (rate) apart from "how big" (price) lets each head learn a steadier thing. This "how
many × how big" split is the classic **subordination** model of asset prices (Clark, 1973; Ané & Geman,
2000): returns over fixed clock-time look messy but become well-behaved once you condition on the
*number* of events. `trade_rate_normalised` is squarely a **how-many / intensity** feature — it speaks to
the rate head.

Two diagnostics recur below: does a feature's *signed* value predict the move (price head), and does its
*magnitude* predict how many moves come (rate head)? The model is fed the **signed** feature for *both*
heads; the rate head learns the magnitude itself. For `trade_rate_normalised` we expect the rate head to
be its home, if anywhere.
""")

md(r"""
## Guard rails — do this, not that

Hard rules, learned the hard way. Follow them unless you have a specific, written reason not to.

**Don't**
- **Don't assume a fixed leader.** No exchange always leads — leadership moves around. Build the feature
  the same way for **every** venue and keep them all. (Here that is literal: byb, okx and bin each get
  their own trade-rate leg; the okx/bin legs are exactly the "does another venue's trade intensity lead
  byb?" test.)
- **Don't pick "the best exchange"** by average score — that throws away the moments another venue leads.
  Keep all; let the model weight them.
- **Don't pool exchanges by default.** Check per-exchange vs single vs pooled across time-scales (§9).
- **Don't trust a correlation** until it survives the regime controls (rate and vol) — but **note the
  controls here are byb's mid-MOVE rate/vol, not trade rate**, so the feature is *not* a re-parameterised
  control (unlike `mid_rate_surge`'s byb leg). Its marginal can be genuinely positive.
- **Don't ship a feature without the oracle** (§4) matching a dead-simple independent version — for
  **every** per-exchange leg.
- **Don't peek ahead.** Every value uses only data at-or-before its own timestamp.
- **Don't normalize reflexively — but DO test it here.** `trade_rate` has units (trades/sec), so it is
  **not** automatically comparable across calm/busy markets; that is the whole reason this feature carries
  the **normalisation axis**. We build three variants — un-normalised, `/ σ_ev`, `/ λ_ev` — and let the
  data say which (if any) helps, using the **model's** yardsticks (never a home-grown scale).
- **Don't over-transform for the network.** Pick the lightest reshaping that works (§8).

**Do**
- **Do start with a falsifiable hypothesis** — a venue's elevated trade intensity precedes more byb
  mid-moves. Disproof: no link to the forward move-count at any time-scale / any normalisation for any
  venue; or the link fully explained by the byb move-rate control.
- **Do make every average a trade-tick EMA.** *Every* smoother — every venue's trade-count leg, the
  seconds-per-trade leg, `σ_ev`, `λ_ev` — is an exponential moving average **decayed once per
  trade-timestamp** (`α = 2/(span+1)`) yet **updated on every relevant event between trades**, never a
  wall-clock or boxcar window. O(1) state, bit-identical online vs offline.
- **Do read a sparse flow as `E / W` on the shared clock.** Each venue's trade-events are present on only
  *some* shared-clock ticks; the trade-count numerator is that venue's exp-weighted trade count `W`,
  decayed every shared trade-tick, injected only on **that venue's** trade, read live. Per the
  merged-clock↔own-clock equivalence this makes a one-clock EMA approximate the venue's **own**-clock
  count at a rescaled span — genuine per-venue rates from one clock.
- **Do treat all records sharing a timestamp as ONE event.** The decay clock advances **once** per
  trade-timestamp; a venue's N simultaneous prints (one order sweeping levels) are ONE trade-event — they
  inject `1`, not `N`, into `W_trades`. (78% of byb prints share a timestamp, so this matters.)
- **Do test against both heads — but feed both the *signed* feature.** The magnitude→intensity (rate-head)
  check is the one that matters; the price-head check is a diagnostic we expect to be weak.
- **Do score out-of-sample** with a purged, embargoed walk-forward (strictly past→future).
- **Do treat a feature as a family across time-scales** and sweep the WHOLE span grid, per venue —
  and across all three normalisations.
""")

md(r"""
## 1. What the feature is, and why it might work

| | |
|---|---|
| **what** | a venue's **trade rate** (trades/sec) at a span `N`, optionally ÷ a regime yardstick — one per venue (byb, okx, bin), three normalisations |
| **feeds** | the **rate head** (intensity: how many byb moves next) is the natural home; price-head is a diagnostic |
| **predicts** | byb's 100 ms move-count (and, weakly if at all, direction) |

**The idea.** Count how often a venue *trades*. Express it as a rate — trades per second — using the
exact `W / dt` form the rate yardstick `λ_ev` uses, but with `W` counting **trade-events** (one per the
venue's trade-timestamp) instead of byb mid-moves. That is a single span-`N` EMA, no fast/slow pair. Then
optionally divide by a yardstick to strip the regime: nothing (`baseline`), `σ_ev` (vol), or `λ_ev`
(byb's move rate). **Do this for byb, okx and bin** — the byb leg is the target's own trade intensity, the
okx/bin legs are other venues' trade intensity.

**Why it should work.** Trading activity clusters. When information arrives or inventory is worked, a
venue's print rate jumps and stays elevated for tens to hundreds of milliseconds, so a high trade rate
forecasts a **burst** of upcoming byb mid-moves. For the **cross-venue** legs there is an extra mechanism:
if okx (or bin) starts trading heavily first, that intensity can **lead** byb by a beat — the classic
no-fixed-leader lead/lag. This is an *intensity* statement: *how many* moves, not *which way* — the **rate
head**.

**Why trade rate is NOT the rate control (the crux).** The model's rate controls are built from `λ_ev` =
byb's **mid-move** rate. A venue's **trade** rate is a different quantity: a trade that doesn't lift a
level leaves the mid unchanged (a trade with no move), and a quote revision moves the mid with no trade (a
move with no trade). So even byb's *own* trade rate is **not** a re-parameterisation of the move-rate
control — it can carry marginal value the control does not. (Contrast `mid_rate_surge`, whose byb leg
*was* the move-rate control.) This is exactly what §6/§9 measure, per venue, and the normalisation axis
(§3) asks whether dividing by `σ_ev` or `λ_ev` sharpens or destroys that value.

**The normalisation axis, three ways.** `trade_rate` has units, so on its own it partly re-reports the
overall pace. (a) **baseline** keeps the raw rate — comparable only if the trade-rate regime is itself the
signal. (b) **÷ σ_ev** puts it in inverse-vol units. (c) **÷ λ_ev** makes it **trades per byb-mid-move** —
arguably the most interpretable: "how many prints per move," a microstructure ratio that is high when
trading is churny relative to price discovery. We build all three and report which helps.

**What would disprove it.** No link to the forward byb move-count at any span / any venue / any
normalisation; or the link vanishing once we control for `rate_momentum`/`rate_level` (then it *was* just
the move-rate regime after all, despite the mechanism).

> **No fixed leader.** Built the same way for every venue, all kept. The okx/bin legs are precisely the
> "another venue's trade intensity might lead" test; the trade clock pools all three venues' trades.
""")

md(r"""
## Choosing the EMA — the part that's easy to get silently wrong

Every smoother here is an EMA on the **trade clock**: its **decay** steps once per *trade-timestamp*
(`α = 2/(span+1)`), but its **value updates on every relevant event in between**, so a read between trades
is never stale. For each venue's `trade_rate` the moving parts are:

- **The trade-count numerator `W_trades(ex)`** — a **sparse flow** (venue `ex` trades on only *some*
  shared-clock ticks). It is an EMA that **decays every shared trade-tick** and **injects `1` on each `ex`
  trade-timestamp**, read **live**. This is the same `E/W`-style flow the yardsticks use, but the injected
  mark is a **trade-event indicator**, not a mid-move. Per the merged-clock↔own-clock equivalence,
  dividing through by the common decay makes this a *per-venue* count even though all venues share one
  clock.
- **The seconds-per-trade denominator `E_dt`** — a **per-trade** EMA of the gap between consecutive
  trade-timestamps, held flat between trades. This is a property of the **shared clock**, so it is the
  same for every venue; the per-venue content lives entirely in `W_trades(ex)`.

So `trade_rate(ex; N) = W_trades(ex; N) / E_dt(N)` = venue `ex`'s trades per second at span `N`.

The trap the template warns about (push a `0` on a non-event and "per X" silently becomes "per tick")
applies here too: `W_trades(ex)` must inject **only** on a real `ex` trade-timestamp, and **once** per
timestamp (simultaneous prints are one event — inject `1`, never the print count). The §4 oracle
re-derives all of this from raw events, per venue, and confirms it bit-exact.

**A subtlety vs `volume_surge`/`mid_rate_surge`.** This is a **single** EMA, not a fast/slow ratio, so it
does **not** self-cancel its units — which is exactly why the normalisation axis exists. The numerator is
a *count indicator* (mark `1` per trade-event), not a *qty* or a *mid-move*; everything else is the same
shared-clock `W / dt` machinery.
""")

md(r"""
## 2. The exact definition

A definition has to get three things right.

**Causal** — only data up to the moment it's computed; no peeking. (A stray forward-fill is the usual
accidental break.)

**Measured on the trade clock** — markets alternate busy and quiet, so we count progress in *trades* (one
tick per trade-timestamp on any exchange — simultaneous prints are one tick) rather than clock-seconds. A
venue's trades between *other* venues' trades update what its `W` reads but the decay clock advances once
per shared trade-timestamp.

**Comparable across calm and wild markets** — here this is the **open question**, not a given.
`trade_rate` has units (trades/sec), so unlike a ratio it is **not** automatically comparable. That is why
the feature carries a **normalisation axis**: we build the raw rate and two yardstick-divided variants and
let §6 say which is comparable-and-predictive.

Put it together, **for each venue `ex`** and each normalisation `y ∈ {1, σ_ev, λ_ev}`:

**2a — The shape** (the whole idea, in one line):

`trade_rate_normalised(ex; N, y) = trade_rate(ex; N) / y`,   `trade_rate(ex; N) = W_trades(ex; N) / E_dt(N)`.

A fresh burst of `ex` prints shows up as `W_trades(ex)` rising, so `trade_rate` rises; the optional ÷ y
strips the vol or move-rate regime.

**2b — The EMAs we use, and how they update:**

- **`W_trades(ex; N)`** (per-venue trade-count numerator) — a **sparse flow**: decay once per
  trade-timestamp on the shared clock (`α=2/(N+1)`), inject `1` on each real `ex` **trade-timestamp**,
  read **live** at the anchor.
- **`E_dt(N)`** (seconds-per-trade denominator) — a **per-trade** EMA of `Δ(trade timestamp)/1e9`, held
  flat between trades; a property of the shared clock (same for all venues).
- **`σ_ev`, `λ_ev`** (yardsticks, span `YARDSTICK_N`) — `σ_ev = √(E/W)` over **byb mid-moves**;
  `λ_ev = W_byb_moves / E_dt` at `YARDSTICK_N` (byb's mid-moves per second). Used both as the optional
  normalisers **and** to build the §5 controls and the rate-head target. **Note `λ_ev` counts byb
  mid-moves; the feature counts trades — different streams.**

§3 builds exactly this; §4 re-derives it from raw events, per venue, and confirms it bit-exact. The cell
below loads the data, builds the shared trade clock, computes each venue's trade-event stream, and
computes both yardsticks (from byb mid-moves) as `E/W` / `W÷dt` flows on that clock, span `YARDSTICK_N`.
""")

code(r"""
import numpy as np, polars as pl
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.stats import spearmanr
from boba.io import list_blocks, load_block

COIN        = "eth_usdt_p"
TARGET      = "byb_eth_usdt_p"                    # the exchange we PREDICT; byb mid also feeds σ_ev/λ_ev & the targets
EXCHANGES   = ["byb", "okx", "bin"]              # EVERY venue gets its own trade-rate leg (byb = own; okx/bin = cross-venue)
MID_STREAM  = {"bin": "front_levels", "byb": "merged_levels", "okx": "merged_levels"}  # byb/okx mid = merged (trade-fused); bin = front_levels (merged blocked for bin perp)
# the WHOLE lookback family for the single trade-rate EMA span (any positive N; 1 = no smoothing of the rate)
SPANS       = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
NORMS       = ["baseline", "sigma", "lambda"]    # the normalisation axis: raw rate, ÷ σ_ev, ÷ λ_ev
HORIZON_NS  = 100 * 1_000_000                    # how far ahead we predict (100 ms, in nanoseconds)
YARDSTICK_N = 10000                              # the ONE span for BOTH yardsticks (σ_ev, λ_ev): trade-tick EMA α=2/(N+1)
block       = list_blocks(TARGET, "front_levels")[0]   # one ~24h slice of recorded data

# load byb's mid-price stream (drives σ_ev/λ_ev and the prediction targets)
def load_mid(ex):
    df = (load_block(block, f"{ex}_{COIN}", MID_STREAM[ex]).select("rx_time", "bid_prc", "ask_prc").drop_nulls())
    return df["rx_time"].cast(pl.Int64).to_numpy(), (df["bid_prc"].to_numpy() + df["ask_prc"].to_numpy()) / 2
byb_rx0, byb_mid0 = load_mid("byb")

# per-venue TRADE stream + the shared trade clock. Simultaneous prints (one order sweeping levels) are ONE event -> ONE tick.
trade_rx = {}                                     # ex -> unique trade-TIMESTAMPS (one per trade-event for that venue)
trade_ts = []
for ex in EXCHANGES:
    td = (load_block(block, f"{ex}_{COIN}", "trade").select("rx_time", "prc", "qty")
          .filter((pl.col("prc") > 0) & (pl.col("qty") > 0)))         # drop bad prc=qty=0 prints
    rx = td["rx_time"].cast(pl.Int64).to_numpy()
    trade_rx[ex] = np.unique(rx)                  # collapse this venue's simultaneous prints to ONE trade-EVENT (inject 1, not the print count)
    trade_ts.append(rx)
trade_prints = np.concatenate(trade_ts)
merged_ts = np.unique(trade_prints)                               # shared trade clock: one tick per trade-TIMESTAMP (any venue)
n_ticks = len(merged_ts)
byb_dt = np.zeros(n_ticks); byb_dt[1:] = np.diff(merged_ts) / 1e9 # seconds between consecutive trades (shared clock; same dt for every venue)
print(f"trade clock: {n_ticks:,} ticks (timestamps) from {len(trade_prints):,} trade prints")
print("per-venue trade-EVENTS (simultaneous prints collapsed): " + "  ".join(f"{ex}={trade_rx[ex].size:,}" for ex in EXCHANGES))

# --- byb mid-MOVE stream for the yardsticks: react to every byb mid-move, decay once per shared trade ---
keep = np.concatenate([byb_rx0[1:] != byb_rx0[:-1], [True]])                     # collapse same-TIMESTAMP mids to ONE update (final mid)
byb_rx, byb_mid = byb_rx0[keep], byb_mid0[keep]
byb_lm = np.log(byb_mid)
byb_blr = np.empty_like(byb_lm); byb_blr[0] = 0.0; byb_blr[1:] = np.diff(byb_lm)  # byb log-return per timestamp
mv = byb_blr != 0.0                                                              # a REAL byb mid-move
mv_rx, mv_r2 = byb_rx[mv], byb_blr[mv] ** 2
cum_mv = np.concatenate([[0.0], np.cumsum(mv.astype(float))])                    # running byb move count (rate-head target)

def _ewma(x, span):                                                # per-trade EMA (seconds/trade leg; shared clock)
    a = 2.0 / (span + 1.0); return lfilter([a], [1.0, -(1.0 - a)], x)
def _flow_at(val_rx, val, anchors, span):   # live EWMA of `val` over a SPARSE event stream (val_rx), decayed once per trade-timestamp, read AT each anchor
    a = 2.0 / (span + 1.0)
    k = np.searchsorted(merged_ts, val_rx, "left")                              # trades strictly before each event (a same-rx trade decays it)
    ep = np.bincount(k, weights=val, minlength=n_ticks + 1)                     # per-trade-epoch sums
    x = np.zeros(n_ticks + 1); x[1:] = a * (1.0 - a) * ep[:-1]
    com = lfilter([1.0], [1.0, -(1.0 - a)], x)                                  # committed E just after each trade
    ta = np.searchsorted(merged_ts, anchors, "right") - 1                       # last trade <= anchor
    cs = np.concatenate([[0.0], np.cumsum(val)])                               # prefix sums over the event stream (the partial epoch)
    partial = cs[np.searchsorted(val_rx, anchors, "right")] - cs[np.searchsorted(val_rx, merged_ts[ta], "right")]
    return com[ta + 1] + a * partial
def dt_at(anchors, span):                   # seconds/trade EMA on the shared clock (per-trade, held flat between trades)
    return _ewma(byb_dt, span)[np.searchsorted(merged_ts, anchors, "right") - 1]
def trade_rate(ex, anchors, span):          # venue ex's TRADE rate at a span: live trade-EVENT count W ÷ shared seconds dt = trades/sec
    w  = _flow_at(trade_rx[ex], np.ones(trade_rx[ex].size), anchors, span)         # W_trades(ex): exp-weighted ex trade-event count (live at anchor)
    return w / np.maximum(dt_at(anchors, span), 1e-12)
def yardsticks(anchors, span):                                     # σ_ev, λ_ev from BYB MID-MOVES — defined AT the anchor
    e_sq = _flow_at(mv_rx, mv_r2, anchors, span)                  # E: exp-weighted squared byb moves
    e_mv = _flow_at(mv_rx, np.ones(mv_r2.size), anchors, span)    # W: exp-weighted byb-MOVE count
    e_dt = dt_at(anchors, span)                                   # seconds/trade
    sig = np.sqrt(e_sq / np.maximum(e_mv, 1e-12))                 # σ_ev: RMS byb mid-move (E/W — non-moves cancel)
    lam = e_mv / np.maximum(e_dt, 1e-12)                          # λ_ev: byb mid-MOVES per second  (NOT trades/sec)
    return sig, lam
print(f"yardsticks (from byb mid-moves): decay span {YARDSTICK_N} trades; λ_ev = byb MID-MOVES/sec (≠ trade rate)")
""")

md(r"""
## 3. Build it (twice), across all three normalisations

Build the feature two ways: this fast array version for analysis, and — in production — a streaming
version that does constant work per trade (no growing buffers). They have to agree, which §4 checks.

We lay an evaluation grid every 50 ms (half the 100 ms horizon — plenty of samples; adjacent 100 ms
outcome windows still overlap ~50%, so neighbouring samples are correlated, which is exactly why §5's
walk-forward gate uses an embargo), read byb's actual move and move-count over the next 100 ms (the things
we're predicting), and compute the feature **per venue, per normalisation** at each grid point.

**The normaliser is applied at the read.** `trade_rate(ex; N)` is built once per (venue, span); the three
variants are just that rate ÷ {1, σ_ev, λ_ev} at the anchor — so they share the heavy machinery and differ
only by a scalar divide.
""")

code(r"""
# evaluation grid (causal) + forward targets
WARMUP = 5 * max(YARDSTICK_N, max(SPANS))   # enough trades for the slowest EMA/yardstick to converge
anchor_ts      = np.arange(merged_ts[WARMUP], merged_ts[-1] - HORIZON_NS, 50 * 1_000_000)   # 50 ms grid, past warmup
sigma_at_anchor, lam_at_anchor = yardsticks(anchor_ts, YARDSTICK_N)
print(f"σ_ev median {np.nanmedian(sigma_at_anchor):.2e},  λ_ev median {np.nanmedian(lam_at_anchor):.2f} moves/s")

# price-head target (a diagnostic for this rate feature): byb's 100 ms return ÷ σ_ev
mid_now    = byb_mid[np.searchsorted(byb_rx, anchor_ts, "right") - 1]
mid_fwd    = byb_mid[np.searchsorted(byb_rx, anchor_ts + HORIZON_NS, "right") - 1]
fwd_return = np.log(mid_fwd / mid_now)
target     = fwd_return / sigma_at_anchor                          # price-head target (σ-units)

# rate-head target = byb moves in the next 100 ms ÷ λ_ev (the rate yardstick) — the head this feature serves
fwd_count = (cum_mv[np.searchsorted(byb_rx, anchor_ts + HORIZON_NS, "right")]
             - cum_mv[np.searchsorted(byb_rx, anchor_ts, "right")])
rate_target = fwd_count / np.maximum(lam_at_anchor, 1e-9)          # count ÷ λ_ev ∝ "more/fewer moves than usual"

_YARD = {"baseline": np.ones_like(sigma_at_anchor), "sigma": sigma_at_anchor, "lambda": lam_at_anchor}
def trade_rate_normalised(ex, span, norm):  # THE FEATURE, per venue / span / normalisation
    r = trade_rate(ex, anchor_ts, span)                           # trades/sec at this span
    return r / np.maximum(_YARD[norm], 1e-300)                    # ÷ {1, σ_ev, λ_ev}
print(f"grid: {len(anchor_ts):,} anchors")
for ex in EXCHANGES:
    demo = trade_rate(ex, anchor_ts, 100)
    print(f"  trade_rate({ex},100): median {np.nanmedian(demo):.1f} trades/s  p10 {np.nanpercentile(demo,10):.1f}  p90 {np.nanpercentile(demo,90):.1f}")
# sanity: trades/sec vs byb mid-moves/sec (λ_ev) — different magnitudes => trade rate ≠ move rate
print(f"  byb λ_ev (mid-moves/s) median {np.nanmedian(lam_at_anchor):.1f}  vs byb trade_rate median {np.nanmedian(trade_rate('byb', anchor_ts, 100)):.1f}  -> distinct clocks")
""")

md(r"""
## 4. Check the code is right — the oracle (a production-style streaming build)

**Non-negotiable.** Reproduce the feature with a second, **independent** implementation and confirm the
two agree on real data, **bit-exact**, **for every per-exchange leg**. The oracle is the **production**
reference: an O(1) state machine you push **raw events** into and read `value(ex, span)` (the raw rate)
and the normalised variants from. State is a few scalar EMAs per venue — no buffers, no history,
independent of how long it runs.

The oracle shares **no code** with §3 (no `lfilter`, no `_flow_at`) **and imports no `boba.ema` helper** (no
`EventEMA` / `KernelMeanEMA` / `LiveFrontEMA`) — every EMA is one hand-rolled plain-scalar recursion
(`ema=(1−α)·ema+α·value`, started from 0, the `y[-1]=0` convention), so it is implementable from §2's written
definition alone by someone who has never seen §3. It uses only plain scalars for **all** of it — the shared
seconds-per-trade leg, σ_ev's numerator/denominator, each venue's trade-count `W`, and byb's move flows:

- **`W_trades(ex; N)`, per venue per span** — a scalar `E`: **decay** `E *= (1−α)` once per
  trade-timestamp (if a trade landed); **inject** `E += α·1` on each `ex` **trade-timestamp** (once, even
  for many simultaneous prints); **read** `E` live at the anchor.
- **`E_dt(N)`, per span** — one hand-rolled scalar EMA per span (shared across venues), stepped once per
  trade with the gap since the previous trade; held flat between trades.
- **σ_ev, λ_ev** — hand-rolled `E/W` masses over byb mid-moves (decay each trade, inject on a byb move),
  with `λ_ev = W_byb_moves / E_dt(YARDSTICK_N)`; used as the normalisers and validated alongside.
- **trade_rate = W_trades / E_dt**, and the three variants divide by {1, σ_ev, λ_ev}.

**Driver discipline (matches the trade clock):** apply every record at a timestamp, then `refresh()` once
— detect **each venue's** trade-presence and inject its `W` (once), detect byb's mid-move and inject σ_ev,
advance the clock (decay every `W` and the byb masses, step every `dt`) **at most once**, and only if a
trade landed. A book-only instant moves byb's mid (maybe a move → inject σ_ev) without advancing the
clock. We feed the **whole raw stream** — every venue's book updates and trades — into **one** builder and
read **every venue's** feature at each grid anchor, **at the shipped per-venue spans {10, 30, 100}** (the §6
rate-head picks) across all three normalisations — so every leg/span/normalisation we actually ship is
validated bit-exact, not just one nearby span.
""")

code(r"""
import boba.io as io                  # used ONLY to read raw events (load_block / _trade_lifts_ask) — NOT for any EMA/feature math
import math

# The oracle is a DEAD-SIMPLE, fully INDEPENDENT plain-scalar streaming reimplementation: NO boba.ema helpers
# (no EventEMA/KernelMeanEMA/LiveFrontEMA), NO lfilter, NO _flow_at — it shares no feature code with §3. Every EMA below
# is this one hand-rolled scalar recursion (`ema=(1-α)·ema+α·value`, started from 0, nan before first step — the y[-1]=0
# convention), so an agent given only §2's written definition could write exactly this. Matching §3 bit-exact is therefore
# a real cross-check, not a tautology.
class _EMA:                                                                # one-scalar event EMA: ema=(1-α)·ema+α·value, y[-1]=0
    __slots__ = ("a", "e", "on")
    def __init__(self, span): self.a = 2.0 / (span + 1.0); self.e = 0.0; self.on = False
    def step(self, v): self.e = (1.0 - self.a) * self.e + self.a * v; self.on = True
    def value(self): return self.e if self.on else float("nan")

class LiveTradeRate:
    # Pure feature state machine for per-venue TRADE rates (+ the byb yardsticks for normalisation). O(1), all scalar.
    # Per venue per span: W_trades = a live decaying trade-EVENT mass (decay each trade-tick, inject 1 on each ex trade-timestamp).
    # Per span: ONE E_dt = a hand-rolled scalar EMA (_EMA) of seconds-per-trade (the shared clock).  trade_rate = W_trades / E_dt.
    # byb mid-moves drive σ_ev (E/W of squared returns) and λ_ev (W_moves / E_dt at YARDSTICK_N) — the optional normalisers.
    # State keyed by full listing so a perp and spot never collide. Each venue's mid is built INSIDE (merged venues fuse trades
    # by newest-exchange-time; book-only venues take the latest snapshot). The driver applies all records at a timestamp,
    # then calls refresh() once: register EACH venue's trade-presence (-> inject its W) and byb's mid-move (-> inject σ_ev),
    # then advance the clock AT MOST ONCE (only if a trade landed).
    def __init__(self, target, venues, spans, vol_span, fuse_trades):
        self.target = target
        self.venues = list(venues)                                         # full listings we track a trade-rate for (byb,okx,bin)
        self.spans = sorted(set(spans))                                    # all distinct EMA spans we need
        self.alpha = {N: 2.0 / (N + 1.0) for N in self.spans}              # α per span
        self.va = 2.0 / (vol_span + 1.0)                                   # α for the yardstick span
        self.fuse_trades = set(fuse_trades)
        self.bid = {}; self.bid_t = {}; self.ask = {}; self.ask_t = {}     # newest-by-exchange-time top-of-book, keyed by listing
        self.W = {v: {N: 0.0 for N in self.spans} for v in self.venues}    # live trade-EVENT count mass per venue per span
        self.dt = {N: _EMA(N) for N in self.spans}                         # seconds-per-trade EMA per span (shared clock) — hand-rolled scalar EMA
        self.dt_y = _EMA(vol_span)                                         # seconds-per-trade at the yardstick span (for λ_ev)
        self.vE = 0.0; self.vW = 0.0                                       # σ_ev numerator/denominator (E/W of squared byb moves)
        self.mW = 0.0                                                      # λ_ev numerator: byb move-count mass at the yardstick span
        self.prev = None                                                   # byb's last log-mid, to detect a real move
        self.prev_ts = None                                                # last trade-timestamp (for the dt gap)
        self.traded_venues = set()                                         # which venues traded THIS timestamp
        self.was_trade_present = False

    def _side(self, listing, is_ask, px, t):                              # keep this listing-side's newest-by-exchange-time price
        held_t = (self.ask_t if is_ask else self.bid_t)
        if t > held_t.get(listing, -1):
            (self.ask if is_ask else self.bid)[listing] = px; held_t[listing] = t

    def _mid(self, listing):
        b, a = self.bid.get(listing), self.ask.get(listing)
        return None if b is None or a is None else 0.5 * (b + a)

    def refresh(self, ts):                                                # ONE per TIMESTAMP
        traded = self.was_trade_present; self.was_trade_present = False
        tv = self.traded_venues; self.traded_venues = set()
        for v in tv:                                                      # inject each venue's trade-EVENT (ONCE per timestamp, before decay)
            for N in self.spans: self.W[v][N] += self.alpha[N] * 1.0
        m = self._mid(self.target)                                        # byb mid-move -> inject σ_ev and the λ_ev move-count
        if m is not None:
            lm = math.log(m)
            if self.prev is not None and lm != self.prev:
                r2 = (lm - self.prev) ** 2
                self.vE += self.va * r2; self.vW += self.va * 1.0         # σ_ev: E/W of squared moves
                self.mW += self.va * 1.0                                  # λ_ev numerator: byb move-count mass
            self.prev = lm
        if traded:                                                        # a trade landed -> advance the clock exactly once
            gap = 0.0 if self.prev_ts is None else (ts - self.prev_ts) / 1e9
            for N in self.spans:
                for v in self.venues: self.W[v][N] *= (1.0 - self.alpha[N])  # decay every venue's trade-count mass one trade-tick
                self.dt[N].step(gap)                                      # step the shared seconds-per-trade EMA
            self.dt_y.step(gap)                                           # yardstick-span seconds-per-trade
            self.vE *= (1.0 - self.va); self.vW *= (1.0 - self.va)        # decay σ_ev masses
            self.mW *= (1.0 - self.va)                                    # decay λ_ev move-count mass
            self.prev_ts = ts

    def on_book(self, listing, t, bid, ask):
        if listing in self.fuse_trades:
            self._side(listing, False, bid, t); self._side(listing, True, ask, t)
        else:
            self.bid[listing] = bid; self.ask[listing] = ask

    def on_trade(self, listing, t, px, lifts_ask):
        if listing in self.fuse_trades: self._side(listing, lifts_ask, px, t)
        self.traded_venues.add(listing)                                   # this venue traded this timestamp (one event, regardless of print count)
        self.was_trade_present = True

    def _rate(self, v, N):
        d = self.dt[N].value()
        if d is None or not (d > 1e-12): return float("nan")
        return self.W[v][N] / d

    def sigma(self):
        return math.sqrt(self.vE / self.vW) if self.vW > 0.0 else float("nan")
    def lam(self):
        d = self.dt_y.value()
        if d is None or not (d > 1e-12): return float("nan")
        return self.mW / d

    def value(self, v, N, norm):                                          # the feature for venue v at span N, normalisation norm
        r = self._rate(v, N)
        if norm == "baseline": return r
        if norm == "sigma":
            s = self.sigma(); return r / s if (s == s and s > 0) else float("nan")
        l = self.lam();        return r / l if (l == l and l > 0) else float("nan")

# --- gather the WHOLE raw stream (every venue's book updates + trades) over a slice; no merged_levels anywhere ---
# Validate the SHIPPED rate-head spans (byb=10, okx=30, bin=10 at norm λ_ev — see §6) PLUS span 100 for breadth, for
# ALL THREE venues and ALL THREE normalisations, over ~N_GRID grid points. Covering {10,30,100} means every per-venue
# shipped span is checked bit-exact, not just one nearby span (honest oracle coverage of what we actually ship).
VAL_SPANS, N_GRID = [10, 30, 100], 40_000
cutoff = int(anchor_ts[min(N_GRID, len(anchor_ts) - 1)])    # wall-clock time of the N_GRID-th grid anchor
LISTINGS = [f"{ex}_{COIN}" for ex in ("byb", "okx", "bin")] # 0=byb, 1=okx, 2=bin
cols = {k: [] for k in "rx kind lid t a b".split()}
def add(rx, kind, lid, t, a, b):                   # stage one source's events (kept <= cutoff)
    m = rx <= cutoff; n = int(m.sum())
    cols["rx"].append(rx[m]); cols["kind"].append(np.full(n, kind, np.int8)); cols["lid"].append(np.full(n, lid, np.int8))
    cols["t"].append(t[m]); cols["a"].append(a[m].astype(float)); cols["b"].append(b[m].astype(float))
for lid, ex in ((0, "byb"), (1, "okx")):           # merged-mid venues: raw front_levels (bid/ask + exchange_time)
    fl = load_block(block, f"{ex}_{COIN}", "front_levels").select("rx_time", "exchange_time", "bid_prc", "ask_prc").drop_nulls()
    add(fl["rx_time"].cast(pl.Int64).to_numpy(), 0, lid, fl["exchange_time"].cast(pl.Int64).to_numpy(), fl["bid_prc"].to_numpy(), fl["ask_prc"].to_numpy())
df = (load_block(block, f"bin_{COIN}", MID_STREAM["bin"]).select("rx_time", "bid_prc", "ask_prc").drop_nulls())  # book-only venue: bin's front_levels mid
brx = df["rx_time"].cast(pl.Int64).to_numpy(); bmid = (df["bid_prc"].to_numpy() + df["ask_prc"].to_numpy()) / 2
add(brx, 0, 2, brx, bmid, bmid)                    # bin: fed as bid=ask=mid (book-only); bin still trades -> its own trade-rate leg
for lid, ex in ((2, "bin"), (0, "byb"), (1, "okx")):   # trades from EVERY venue tick the shared clock; byb/okx also fuse their mid
    td = load_block(block, f"{ex}_{COIN}", "trade").select("rx_time", "exchange_time", "prc", "qty", "aggressor").filter((pl.col("prc") > 0) & (pl.col("qty") > 0))
    add(td["rx_time"].cast(pl.Int64).to_numpy(), 1, lid, td["exchange_time"].cast(pl.Int64).to_numpy(),
        td["prc"].to_numpy(), io._trade_lifts_ask(f"{ex}_{COIN}", td["aggressor"].to_numpy()).astype(float))
C = {k: np.concatenate(v) for k, v in cols.items()}
order = np.lexsort((C["kind"], C["rx"]))           # rx ascending; book (0) before trade (1) on ties (matches merged_levels)
rxL, kindL, lidL, tL, aL, bL = (C[k][order].tolist() for k in "rx kind lid t a b".split())
print(f"streaming {len(rxL):,} raw events (book + trades, all venues) over ~{N_GRID:,} grid points...")

# --- the CALLER drives it: apply each timestamp's events, refresh() once, READ value() for EVERY venue/norm at every grid anchor ---
fuse = {f"{ex}_{COIN}" for ex in EXCHANGES if MID_STREAM[ex] == "merged_levels"}   # byb,okx merged; bin book-only
VENUES = [f"{ex}_{COIN}" for ex in EXCHANGES]
feat = LiveTradeRate(TARGET, VENUES, VAL_SPANS, YARDSTICK_N, fuse)
na = min(N_GRID, len(anchor_ts))
stream = {(ex, nm, N): np.full(na, np.nan) for ex in EXCHANGES for nm in NORMS for N in VAL_SPANS}
n = len(rxL); i = 0; ai = 0
while i < n:
    rx = rxL[i]
    while ai < na and anchor_ts[ai] < rx:          # read every anchor whose state is settled (all events before rx applied)
        for ex in EXCHANGES:
            for nm in NORMS:
                for N in VAL_SPANS: stream[(ex, nm, N)][ai] = feat.value(f"{ex}_{COIN}", N, nm)
        ai += 1
    while i < n and rxL[i] == rx:                  # apply EVERY event stamped at this nanosecond (no half-applied timestamp)
        if kindL[i] == 0: feat.on_book(LISTINGS[lidL[i]], tL[i], aL[i], bL[i])
        else:             feat.on_trade(LISTINGS[lidL[i]], tL[i], aL[i], bL[i])
        i += 1
    feat.refresh(rx)                               # apply the timestamp: inject each venue's W + byb σ_ev, advance the clock once if a trade landed
while ai < na:
    for ex in EXCHANGES:
        for nm in NORMS:
            for N in VAL_SPANS: stream[(ex, nm, N)][ai] = feat.value(f"{ex}_{COIN}", N, nm)
    ai += 1

# --- one raw-event stream -> 3 venues × 3 normalisations × shipped spans: check EACH against its §3 vectorized feature, bit-exact ---
# Bit-exactness here is a MAGNITUDE-RELATIVE tolerance: the σ_ev-normalised legs have values ~1e5-1e6 (trade_rate ÷ a
# ~1e-5 σ_ev), so an absolute |diff| of ~5e-7 on them is a RELATIVE error ~6e-12 — the recursive EMAs' last-digit drift,
# the same "float-point precision" the template reports. We therefore gate on |diff| / (1 + |ref|) < 1e-9 (and also print
# the raw absolute |diff| for transparency); an absolute floor alone would be a unit-dependent, not a correctness, check.
print(f"one raw-event stream -> per-venue × per-normalisation × shipped-span trade-rate legs, vs the vectorized build (spans={VAL_SPANS}):")
worst_abs = 0.0; worst_rel = 0.0
for N in VAL_SPANS:
    for ex in EXCHANGES:
        for nm in NORMS:
            ref = trade_rate_normalised(ex, N, nm)[:na]
            both = np.isfinite(stream[(ex, nm, N)]) & np.isfinite(ref)
            a_diff = np.abs(stream[(ex, nm, N)][both] - ref[both])
            r_diff = a_diff / (1.0 + np.abs(ref[both]))               # magnitude-relative: bit-exact regardless of feature scale
            mx_a = float(np.nanmax(a_diff)); mx_r = float(np.nanmax(r_diff))
            worst_abs = max(worst_abs, mx_a); worst_rel = max(worst_rel, mx_r)
            print(f"  span {N:>3}  {ex:4} / {nm:8}:  rel |diff| {mx_r:.2e}  (abs {mx_a:.2e})  on {int(both.sum()):,} grid points")
            assert mx_r < 1e-9, f"live streaming build does not reproduce the {ex}/{nm} feature at span {N} (rel |diff|={mx_r:.2e})"
print(f"oracle: the raw-event streaming build reproduces ALL legs × normalisations × shipped spans  OK  (worst rel |diff| {worst_rel:.2e}, worst abs {worst_abs:.2e})")
""")

md(r"""
**Conclusion.** From one stream of raw events the streaming O(1) builder reproduces **every** venue's
**and every normalisation's** vectorized feature to floating-point precision — worst **relative** |diff|
~1e-12 over the validated grid, pure last-digit round-off from the recursive EMAs. (We gate on a
magnitude-relative tolerance because the σ_ev-normalised legs carry values ~1e5–1e6 — `trade_rate` ÷ a
~1e-5 σ_ev — so their *absolute* round-off is ~5e-7 while their *relative* error stays ~6e-12; an absolute
floor would be a unit check, not a correctness check.) The two implementations share no code and no helpers
(the oracle is plain scalars only — a hand-rolled `_EMA`, no `boba.ema`; §3 uses `lfilter`/`_flow_at`), so agreement means each leg computes what
we claim: that **venue's** live **trade-event** count (one per its trade-timestamp, simultaneous prints
collapsed) ÷ the shared seconds-per-trade, optionally ÷ σ_ev or λ_ev. The byb, okx and bin legs are all
bit-exact across all three normalisations **and across the shipped per-venue spans {10, 30, 100}** (byb=10,
okx=30, bin=10 are the §6 rate-head picks, all covered), so the §3 per-venue build is trustworthy. The oracle
independently collapses simultaneous prints to one trade-event and decays the shared clock once per
timestamp — matching it confirms the vectorised build handles those quirks correctly.
""")

md(r"""
## 5. Is the signal real? — the hygiene gates (and the trade-rate vs move-rate reckoning)

A correlation is an easy way to fool yourself. The gates check that `trade_rate_normalised` predicts
*something the market's current state doesn't already tell us*. We build the four standard controls
**(all from byb mid-moves)**:
- **rate momentum** and **rate level** — both from `λ_ev` (byb's mid-MOVE rate);
- **vol momentum** and **vol level** — the same two, for volatility.

Then we measure each leg's predictive power **on top of** those controls, out-of-sample, with a **purged,
embargoed, expanding-window walk-forward**.

**The reckoning to watch — trade rate ≠ move rate.** Unlike `mid_rate_surge`'s byb leg (which *was*
`rate_momentum`), the controls here are byb's **mid-MOVE** rate, while the feature is a **TRADE** rate.
These are genuinely different: trades that don't lift a level leave the mid still (trade-no-move), and
quote revisions move the mid with no print (move-no-trade). So the feature is **not** a re-parameterised
control — and its marginal over the rate controls is the honest test of whether **trade** intensity adds
move-rate information beyond what **move** intensity (`λ_ev`) already encodes. We expect a *positive*
marginal here (the whole point of the caveat), but we report it straight. The byb leg is the most likely
to overlap the control (byb's own trades vs byb's own moves co-move); the okx/bin legs add a second,
cross-venue, axis.
""")

code(r"""
# --- the four control signals: the two yardsticks (level) plus a fast/slow momentum of each (ALL from byb mid-moves) ---
FAST_YARD = YARDSTICK_N // 10                        # a faster span (1/10 the yardstick) for the momentum controls
sig_fast, lam_fast = yardsticks(anchor_ts, FAST_YARD)
vol_level     = np.log(sigma_at_anchor)
vol_momentum  = np.log(sig_fast / sigma_at_anchor)
rate_level    = np.log(lam_at_anchor)                                               # λ_ev = byb's mid-MOVE rate
rate_momentum = np.log(lam_fast / lam_at_anchor)                                    # recent move-rate vs slower move-rate

# Out-of-sample scoring = a purged, expanding-window WALK-FORWARD (causal: only past -> future).
def wf_folds(features, y, k=6, embargo=2000):
    design = np.column_stack(features); n = len(y); valid = np.isfinite(design).all(1) & np.isfinite(y)
    edges = np.linspace(0, n, k + 1).astype(int)
    for i in range(1, k):
        te = np.zeros(n, bool); te[edges[i]:edges[i + 1]] = True
        tr = np.zeros(n, bool); tr[:max(0, edges[i] - embargo)] = True
        train, test = valid & tr, valid & te
        if train.sum() < 100 or test.sum() < 100: continue
        mu, sd = design[train].mean(0), design[train].std(0) + 1e-12
        X = np.column_stack([(design - mu) / sd, np.ones(n)])
        coef, *_ = np.linalg.lstsq(X[train], y[train], rcond=None)
        yield test, X @ coef
def wf_ic(features, y):
    return float(np.mean([spearmanr(p[t], y[t]).statistic for t, p in wf_folds(features, y)]))
def wf_ic_by_regime(features, y, reg):
    acc = {}
    for t, p in wf_folds(features, y):
        for r in np.unique(reg[t]):
            m = t & (reg == r)
            if m.sum() >= 100: acc.setdefault(int(r), []).append(spearmanr(p[m], y[m]).statistic)
    return {r: float(np.mean(v)) for r, v in acc.items()}

vol_regime = np.digitize(vol_level, np.nanpercentile(vol_level[np.isfinite(vol_level)], [33, 67]))   # 0 calm,1 mid,2 wild
base, levels = [rate_momentum, vol_momentum], [rate_level, vol_level]
print("control-only predictive power (walk-forward):")
print("  price head: momenta", round(wf_ic(base, target), 3), " momenta+levels", round(wf_ic(base + levels, target), 3))
print("  rate  head: momenta", round(wf_ic(base, rate_target), 3), " momenta+levels", round(wf_ic(base + levels, rate_target), 3),
      " <- the rate controls predict the move-count; each trade-rate leg must beat THIS bar with NEW (trade≠move) info")
""")

md(r"""
**Conclusion.** For the **price head** the controls carry little *directional* signal (≈ 0) — the regime
barely predicts *which way* byb moves. For the **rate head** the controls are stronger: byb's recent
**move** rate predicts its upcoming move-count, so `rate_momentum`/`rate_level` set a real bar. The
question this feature answers is whether a venue's **trade** rate clears that bar with information the
**move**-rate control lacks — and because trade-rate and move-rate are genuinely different clocks
(trade-no-move, move-no-trade), it can. The only number that matters per leg / per normalisation is its
**marginal IC over these controls** (§6), reported straight.
""")

md(r"""
## 6. Three choices: the lookback span, which exchanges to keep, and which normalisation

`trade_rate_normalised` is a **family** across (a) the EMA **span** (the lookback), (b) the **venue** (byb,
okx, bin), and (c) the **normalisation** (baseline / σ_ev / λ_ev). Exactly as the template keeps **all**
venues — never picking one — we sweep the whole span grid for **every venue and every normalisation**, then
let the data choose the span and the normalisation, per head.

We check two things, then run the gates:
- does its **level** predict *intensity* — *how many* moves byb makes next (rate head)? Its home.
- does its **signed** (here log) value predict *direction* (price head)? Expected weak — it's intensity.

**First the normalisation axis**, summarised: for each normalisation, the best rate-head IC across the span
grid, per venue — **and**, decisively, its **regime-invariance scale** (the feature's scale across vol
buckets). The shipped normalisation is **not** simply the highest-IC one: a raw `trades/sec` **level**
(`baseline`) carries the vol/rate regime and **fails the hard <~3× scale gate** — so it is disqualified
however high its raw IC. We pick the **predictive-best among the normalisations that PASS the scale gate**
(raw-first, then normalise): dividing by `σ_ev` (or `λ_ev`) is exactly what makes the rate comparable in
calm and wild markets, and — as the numbers below show — it costs essentially nothing in IC. Then the
per-venue span heat-maps
for the winning normalisation, and the gates.
""")

code(r"""
# Sweep the WHOLE span family for EVERY venue and EVERY normalisation, BOTH heads.
# Rate head uses the LEVEL (positive intensity) directly; price head uses the signed log.
feat_cache = {}
ic_rate = {(ex, nm): np.full(len(SPANS), np.nan) for ex in EXCHANGES for nm in NORMS}
ic_price = {(ex, nm): np.full(len(SPANS), np.nan) for ex in EXCHANGES for nm in NORMS}
for ex in EXCHANGES:
    for nm in NORMS:
        for k, N in enumerate(SPANS):
            s = trade_rate_normalised(ex, N, nm); feat_cache[(ex, N, nm)] = s
            ls = np.log(np.maximum(s, 1e-12))
            ic_rate[(ex, nm)][k]  = spearmanr(s, rate_target).statistic
            ic_price[(ex, nm)][k] = spearmanr(ls, target).statistic

# --- normalisation summary: best |rate IC| across the span grid, per venue per normalisation ---
print("rate-head IC (best |·| across the span grid) — the NORMALISATION axis, per venue:")
print(f"  {'venue':5}" + "".join(f"{nm:>12}" for nm in NORMS))
for ex in EXCHANGES:
    best = {nm: ic_rate[(ex, nm)][np.nanargmax(np.abs(ic_rate[(ex, nm)]))] for nm in NORMS}
    print(f"  {ex:5}" + "".join(f"{best[nm]:>+12.3f}" for nm in NORMS))

# --- the SHIPPED normalisation must clear the HARD regime-invariance scale gate FIRST, then be best predictive ---
# (raw-first-then-normalise): for each normalisation measure BOTH (a) its predictive power (mean best |rate IC|) AND
# (b) its regime-invariance — the feature's SCALE across vol buckets (max/min std of the rate-head pick, per venue,
# averaged). A usable feature reads the SAME in calm & wild: the scale ratio is a HARD gate (< ~3). `baseline` (a raw
# trades/sec LEVEL) carries the regime and typically FAILS it; dividing by σ_ev / λ_ev is exactly the fix. We pick the
# SHIPPED norm as the predictive-best AMONG the norms that PASS the scale gate — never a non-invariant level, however
# high its raw IC.
SCALE_GATE = 3.0                                     # HARD: shipped feature's scale must stay within this across vol buckets
vol_decile_pick = np.digitize(np.log(sigma_at_anchor), np.nanpercentile(np.log(sigma_at_anchor)[np.isfinite(np.log(sigma_at_anchor))], np.arange(10, 100, 10)))
def _scale_ratio(arr):                               # max/min of the feature's std across the 10 vol-deciles
    bands = [np.nanstd(arr[vol_decile_pick == d]) for d in range(10)]
    bands = [b for b in bands if np.isfinite(b) and b > 0]
    return (max(bands) / min(bands)) if len(bands) >= 2 else np.inf
norm_score = {nm: float(np.nanmean([np.nanmax(np.abs(ic_rate[(ex, nm)])) for ex in EXCHANGES])) for nm in NORMS}
# scale of each norm = worst (largest) per-venue scale ratio at that venue's rate-head best span (the form we'd ship)
def best_span_idx(nm, ex): return int(np.nanargmax(np.abs(ic_rate[(ex, nm)])))
norm_scale = {nm: float(np.nanmax([_scale_ratio(feat_cache[(ex, SPANS[best_span_idx(nm, ex)], nm)]) for ex in EXCHANGES])) for nm in NORMS}
print("per-normalisation:  mean best |rate IC|  |  worst-venue SCALE ratio across vol buckets (HARD gate < %.1f)" % SCALE_GATE)
for nm in NORMS:
    verdict = "PASS" if norm_scale[nm] < SCALE_GATE else "FAIL (non-invariant level)"
    print(f"  {nm:8}  IC {norm_score[nm]:.3f}   scale {norm_scale[nm]:.2f}x   -> {verdict}")
passing = [nm for nm in NORMS if norm_scale[nm] < SCALE_GATE]
if passing:
    BEST_NORM = max(passing, key=norm_score.get)     # best predictive AMONG the regime-invariant norms — the SHIPPED form
    print(f"-> SHIPPED normalisation = '{BEST_NORM}' (best |rate IC| {norm_score[BEST_NORM]:.3f} among the PASSING norms; scale {norm_scale[BEST_NORM]:.2f}x)")
else:                                                # nothing passes -> ship the least-bad scale, flag it
    BEST_NORM = min(NORMS, key=norm_scale.get)
    print(f"-> NO norm clears the scale gate; shipping least-bad '{BEST_NORM}' (scale {norm_scale[BEST_NORM]:.2f}x) — FLAG for review")
""")

code(r"""
# Heat-maps: rate-head and price-head IC across the span family, per venue — for EACH normalisation (3 norms x 2 heads).
fig, axes = plt.subplots(len(NORMS), len(EXCHANGES), figsize=(4.6 * len(EXCHANGES), 3.4 * len(NORMS)), squeeze=False)
for r, nm in enumerate(NORMS):
    for c, ex in enumerate(EXCHANGES):
        ax = axes[r][c]
        ax.plot(range(len(SPANS)), ic_rate[(ex, nm)], "o-", color="C3", label="rate head (level→count)")
        ax.plot(range(len(SPANS)), ic_price[(ex, nm)], "s--", color="C0", label="price head (log→return)")
        ax.axhline(0, color="k", lw=0.5); ax.set_xticks(range(len(SPANS))); ax.set_xticklabels(SPANS, rotation=45, fontsize=7)
        ax.set_title(f"{ex} / {nm}" + ("  (own)" if ex == "byb" else "  (cross-venue)"), fontsize=9)
        if c == 0: ax.set_ylabel(f"{nm}\nrank-IC")
        if r == len(NORMS) - 1: ax.set_xlabel("trade-rate EMA span")
        if r == 0 and c == 0: ax.legend(fontsize=7)
fig.suptitle("trade_rate_normalised: rank-IC across the span family — every venue (cols), every normalisation (rows), both heads", y=1.005)
fig.tight_layout(); plt.show()

# In-sample pick of the best span PER head PER venue, at the winning normalisation (used only to choose; re-scored OOS by the gates).
def best_span(arr): return int(np.nanargmax(np.abs(arr)))
rate_member  = {ex: best_span(ic_rate[(ex, BEST_NORM)])  for ex in EXCHANGES}
price_member = {ex: best_span(ic_price[(ex, BEST_NORM)]) for ex in EXCHANGES}
print(f"best span per venue at normalisation '{BEST_NORM}' (in-sample pick; re-scored OOS below):")
for ex in EXCHANGES:
    print(f"  {ex}:  rate head span={SPANS[rate_member[ex]]} IC={ic_rate[(ex,BEST_NORM)][rate_member[ex]]:+.3f}"
          f"   |  price head span={SPANS[price_member[ex]]} IC={ic_price[(ex,BEST_NORM)][price_member[ex]]:+.3f}")
""")

md(r"""
### IC is only half the story — signal LIFETIME across the whole family

A single IC number per span hides *how long the edge lasts*. The companion ("How long does the edge last?")
slides the **outcome** window forward by an observe-to-act latency δ and watches the **forward IC** fade —
but it does that for *one* chosen span. The whole point of an EMA **family** is that different spans trade
**IC against lifetime**: a **short** span reads the freshest burst (often a higher δ=0 IC) but its edge can
die within a few ms (a tiny latency budget); a **long** span smooths to a lower IC but its edge can persist
for tens to hundreds of ms (a generous latency budget). **Both can be useful** — which you want depends on
how fast your stack is. So we measure lifetime for **every span** in the family, not just the chosen one.

For every span / venue / head we read the feature at the anchor (causal, unchanged) and compute, on a fast
**40k-anchor diagnostic grid** (a uniform thinning of the full grid — the same diagnostic scale §4 uses):

- **forward IC at δ=0** and **at δ=20 ms** — the headline edge, and the realisable edge after a typical
  observe→act latency;
- the **half-life** — the smallest δ (ms) at which the forward IC first drops below **half** its δ=0 value
  (recorded as `>500 ms` if it never does within the swept δ range): the feature's latency budget;
- the **backward IC** — against the outcome that *already happened* (`[anchor−100 ms, anchor]`) — the
  contemporaneous **echo**. A head "carries signal" only if some span has a forward IC at δ=20 ms that is
  clearly nonzero **and not dominated by the backward echo**; a curve that is large at δ=0, collapses by
  δ=20 ms, and is matched by a big backward IC is *re-reporting the past*, the one true non-signal.

The **rate head** is this feature's home (trade-rate **level** → byb forward **move-count**); the **price
head** (signed log → byb signed σ-return) is carried as the diagnostic the notebook ships both heads for.
""")

code(r"""
# --- Signal LIFETIME across the WHOLE family, on a fast 40k diagnostic grid ------------------------------
# For every span / venue / head: forward IC at δ=0 and δ=20ms, the half-life δ (forward IC first < half of δ=0),
# and the backward (already-happened) IC. The feature is read at the anchor (causal); only the OUTCOME window slides.
DELTAS_MS  = [0, 5, 10, 20, 50, 100, 200, 500]      # observe->act latencies swept (ms); half-life is read off this grid
D20        = DELTAS_MS.index(20)                     # the realisable-edge latency
DIAG_N     = 40_000                                  # the diagnostic grid size (same scale as §4's oracle) — for SPEED
diag = np.linspace(0, len(anchor_ts) - 1, min(DIAG_N, len(anchor_ts))).astype(int)   # uniform thinning of the full grid
diag = np.unique(diag)
a_ts = anchor_ts[diag]                               # the 40k diagnostic anchors (span the whole block, not just the start)
print(f"lifetime family sweep on a {len(a_ts):,}-anchor diagnostic grid (uniform thinning of {len(anchor_ts):,})")

def _ic(feat, out):                                  # rank-IC over the finite overlap
    v = np.isfinite(feat) & np.isfinite(out)
    return spearmanr(feat[v], out[v]).statistic if v.sum() > 100 else float("nan")
def _mid_at(t):  return byb_mid[np.clip(np.searchsorted(byb_rx, t, "right") - 1, 0, len(byb_mid) - 1)]  # causal byb mid
def _ret(t0, t1):   return np.log(_mid_at(t1) / _mid_at(t0))                                            # signed byb return over [t0,t1]
def _count(t0, t1): return cum_mv[np.searchsorted(byb_rx, t1, "right")] - cum_mv[np.searchsorted(byb_rx, t0, "right")]  # byb move-count
def _halflife(fwd):                                  # smallest δ (ms) where |forward IC| first < half its δ=0 value; >500 if never
    f0 = fwd[0]
    if not np.isfinite(f0) or f0 == 0: return np.nan
    for d, ic in zip(DELTAS_MS, fwd):
        if np.isfinite(ic) and abs(ic) < abs(f0) / 2: return d
    return DELTAS_MS[-1] + 1                          # never dropped below half within the swept range -> ">500 ms"

# precompute the sliding-window outcomes ONCE per δ on the diagnostic anchors (shared across spans/venues)
fwd_count_d = {d: _count(a_ts + d * 1_000_000, a_ts + d * 1_000_000 + HORIZON_NS) for d in DELTAS_MS}
fwd_ret_d   = {d: _ret  (a_ts + d * 1_000_000, a_ts + d * 1_000_000 + HORIZON_NS) for d in DELTAS_MS}
back_count  = _count(a_ts - HORIZON_NS, a_ts)        # rate-head backward (already-happened) move-count
back_ret    = _ret  (a_ts - HORIZON_NS, a_ts)        # price-head backward signed return

# sweep every (venue, span) at the chosen normalisation; rate head uses the LEVEL, price head the signed log
LIFE = {}    # (ex, span_idx) -> dict(rate_fwd=[...δ...], rate_hl, rate_back, price_fwd=[...], price_hl, price_back)
for ex in EXCHANGES:
    for k, N in enumerate(SPANS):
        s   = feat_cache[(ex, N, BEST_NORM)][diag]                # rate-head input: the trade-rate LEVEL (positive intensity)
        ls  = np.log(np.maximum(s, 1e-12))                        # price-head input: the signed log feature
        r_fwd = [_ic(s,  fwd_count_d[d]) for d in DELTAS_MS]
        p_fwd = [_ic(ls, fwd_ret_d[d])   for d in DELTAS_MS]
        LIFE[(ex, k)] = dict(rate_fwd=r_fwd, rate_hl=_halflife(r_fwd), rate_back=_ic(s, back_count),
                             price_fwd=p_fwd, price_hl=_halflife(p_fwd), price_back=_ic(ls, back_ret))
# pack into (venue x span) matrices per head: δ=0 IC, δ=20ms IC, half-life, backward IC
def _mat(head, key): return {ex: np.array([LIFE[(ex, k)][f"{head}_{key}"] for k in range(len(SPANS))], float) for ex in EXCHANGES}
D_LONG = 200; DLONG_I = DELTAS_MS.index(D_LONG)     # the "long latency budget" point (ms) at which we read DURABILITY
ic0   = {h: {ex: np.array([LIFE[(ex, k)][f"{h}_fwd"][0]       for k in range(len(SPANS))]) for ex in EXCHANGES} for h in ("rate", "price")}
ic20  = {h: {ex: np.array([LIFE[(ex, k)][f"{h}_fwd"][D20]     for k in range(len(SPANS))]) for ex in EXCHANGES} for h in ("rate", "price")}
icL   = {h: {ex: np.array([LIFE[(ex, k)][f"{h}_fwd"][DLONG_I] for k in range(len(SPANS))]) for ex in EXCHANGES} for h in ("rate", "price")}  # durability: forward IC @ D_LONG
hlife = {h: _mat(h, "hl")   for h in ("rate", "price")}
icbk  = {h: _mat(h, "back") for h in ("rate", "price")}
print(f"computed forward-IC(δ=0, 20ms, {D_LONG}ms), half-life and backward-IC for every span x venue x head.")
""")

md(r"""
**The family heat-maps — IC *and* lifetime across the whole span grid.** Three panels per head, per venue:
the **forward IC at δ=20 ms** (the realisable edge after a typical latency), the **half-life** (how many ms
the edge survives — the latency budget), and for reference the **δ=0 IC** with the **backward echo**. Read
the IC and half-life panels *together*: a cell that is bright on edge@20ms **and** bright on half-life is a
keep; a cell bright at δ=0 but dark at δ=20 ms with a large backward IC is an echo.
""")

code(r"""
# Heat-maps: per head (rows) x per venue (cols), the family across spans. Three rows show IC *and* lifetime together:
#   row 0  realisable edge: forward IC @ δ=20ms (bars) vs δ=0 (headline) vs backward echo   -> WHERE the edge is.
#   row 1  half-life (ms) per span (the latency budget; >500 = IC never halves in the swept range).
#   row 2  the forward-IC decay CURVE vs δ for a SHORT/MID/LONG span -> the IC-vs-durability trade-off, made visible.
HL_DISPLAY = {d: d for d in DELTAS_MS}; HL_DISPLAY[DELTAS_MS[-1] + 1] = ">500"   # label for "never dropped below half"
SHOW_SPANS = [30, 300, 3000]                          # short / mid / long representatives to draw the δ-decay curves for
for head, head_label in (("rate", "RATE head  (trade-rate level -> byb forward move-count)"),
                         ("price", "PRICE head  (signed log -> byb signed σ-return)")):
    fig, axes = plt.subplots(3, len(EXCHANGES), figsize=(4.8 * len(EXCHANGES), 10.6), squeeze=False)
    for col, ex in enumerate(EXCHANGES):
        tag = "own" if ex == "byb" else "cross-venue"; x = np.arange(len(SPANS))
        # row 0: forward IC at δ=20ms (bars) with the δ=0 IC and backward echo overlaid -> the realisable-edge picture
        ax = axes[0][col]
        ax.bar(x, ic20[head][ex], color="C3", alpha=0.85, label="forward IC @ δ=20ms (realisable)")
        ax.plot(x, ic0[head][ex], "o-", color="C0", lw=1.3, label="forward IC @ δ=0 (headline)")
        ax.plot(x, icL[head][ex], "^-", color="C2", lw=1.1, label=f"forward IC @ {D_LONG}ms (durable)")
        ax.plot(x, icbk[head][ex], "s:", color="0.5", lw=1.2, label="backward IC (echo)")
        ax.axhline(0, color="k", lw=0.5); ax.set_xticks(x); ax.set_xticklabels(SPANS, rotation=45, fontsize=7)
        ax.set_title(f"{ex} ({tag}) — edge vs span", fontsize=9)
        if col == 0: ax.set_ylabel("rank-IC")
        if col == 0: ax.legend(fontsize=6.2, loc="best")
        # row 1: half-life (ms) per span -> the latency budget across the family
        ax = axes[1][col]
        hl = hlife[head][ex].astype(float)
        ax.bar(x, np.where(np.isfinite(hl), hl, 0.0), color="C2", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(SPANS, rotation=45, fontsize=7)
        ax.set_ylim(0, (DELTAS_MS[-1] + 1) * 1.12); ax.set_yticks(DELTAS_MS + [DELTAS_MS[-1] + 1])
        ax.set_yticklabels([str(d) for d in DELTAS_MS] + [">500"], fontsize=7)
        if col == 0: ax.set_ylabel("half-life  δ (ms)")
        ax.set_title(f"{ex} ({tag}) — half-life vs span", fontsize=9)
        for xi, h in zip(x, hl):
            if np.isfinite(h): ax.text(xi, h + 8, HL_DISPLAY.get(int(h), int(h)), ha="center", va="bottom", fontsize=6.5)
        # row 2: forward IC vs δ for short/mid/long spans -> the IC-vs-durability trade-off, drawn
        ax = axes[2][col]
        for N, c in zip(SHOW_SPANS, ("C3", "C1", "C0")):
            if N in SPANS:
                k = SPANS.index(N); ax.plot(DELTAS_MS, LIFE[(ex, k)][f"{head}_fwd"], "o-", color=c, lw=1.3, ms=3, label=f"span {N}")
        ax.axhline(0, color="0.7", lw=0.8); ax.axvline(20, color="0.8", ls=":", lw=0.8)
        ax.set_xlabel("observe->act latency δ (ms)"); ax.set_title(f"{ex} ({tag}) — IC decay vs δ", fontsize=9)
        if col == 0: ax.set_ylabel("forward rank-IC"); ax.legend(fontsize=6.5)
    fig.suptitle(f"{head_label}\nfamily IC + lifetime — top: edge@δ=0/20ms/{D_LONG}ms + echo; mid: half-life; bottom: short(peaky) vs long(durable) IC-decay  (norm='{BEST_NORM}')", y=1.005, fontsize=9.5)
    fig.tight_layout(); plt.show()
""")

code(r"""
# The IC-vs-LIFETIME TRADE-OFF, made explicit, and the per-head verdict + multi-lookback suggestions.
# Two lifetime axes, because they can disagree: (1) the HALF-LIFE (first δ where forward IC < half its δ=0 value);
# (2) DURABILITY = the forward IC retained at a LONG latency D_LONG (here 200ms) — the realisable edge if your stack
# is slow. For a *persistent* feature the half-life can saturate at >500ms for EVERY span (the IC never halves within
# the swept range), so half-life alone won't separate the family — DURABILITY (late-δ IC) is then the discriminator,
# and it is exactly the "long N: lower peak IC but holds at long latency" axis. We surface BOTH options when they differ.
HL_TXT = lambda h: (">500" if (np.isfinite(h) and h > DELTAS_MS[-1]) else (f"{int(h)}" if np.isfinite(h) else "n/a"))
ECHO_FLOOR = 0.02     # a forward IC below this magnitude is treated as "no realisable edge" (noise)

def carries_signal(head):
    # a head carries signal if SOME span has |forward IC @ δ=20ms| clearly > echo floor AND not dominated by the backward IC
    best = None
    for ex in EXCHANGES:
        for k in range(len(SPANS)):
            f20, bk = ic20[head][ex][k], icbk[head][ex][k]
            if np.isfinite(f20) and abs(f20) > ECHO_FLOOR and abs(f20) >= 0.5 * abs(bk):   # forward survives, not pure echo
                if best is None or abs(f20) > abs(best[2]): best = (ex, SPANS[k], f20, bk)
    return best

def picks(head):
    rows = []
    for ex in EXCHANGES:
        f20 = ic20[head][ex]; bk = icbk[head][ex]; fL = icL[head][ex]
        usable = np.array([np.isfinite(f20[k]) and abs(f20[k]) > ECHO_FLOOR and abs(f20[k]) >= 0.5 * abs(bk[k]) for k in range(len(SPANS))])
        if not usable.any():
            rows.append((ex, None, None)); continue
        idx = np.where(usable)[0]
        short = int(idx[np.argmax(np.abs(f20[idx]))])                     # SHORT-HIGH-IC: biggest realisable edge @ δ=20ms among usable spans
        long_ = int(idx[max(range(len(idx)), key=lambda j: abs(fL[idx[j]]))])  # LONG-DURABILITY: biggest retained edge @ D_LONG (the long-latency budget)
        rows.append((ex, short, long_))
    return rows

for head, label in (("rate", "RATE head (home: intensity -> move-count)"), ("price", "PRICE head (diagnostic: direction)")):
    cs = carries_signal(head)
    print(f"\n=== {label} ===")
    if cs is None:
        print(f"  CARRIES SIGNAL: NO — no span has a forward IC@20ms clearly above the echo floor ({ECHO_FLOOR}); pure contemporaneous echo / noise.")
    else:
        ex, N, f20, bk = cs
        print(f"  CARRIES SIGNAL: YES — best realisable edge {f20:+.3f} @ δ=20ms ({ex}, span {N}); backward echo {bk:+.3f} (forward not echo-dominated).")
    for ex, short, long_ in picks(head):
        tag = "own" if ex == "byb" else "cross"
        if short is None:
            print(f"  {ex:4} ({tag:5}): no usable lookback (forward edge <= echo floor / noise across the family).")
            continue
        f20 = ic20[head][ex]; f0 = ic0[head][ex]; hl = hlife[head][ex]; fL = icL[head][ex]
        s = (f"  {ex:4} ({tag:5}):  SHORT/high-IC  span={SPANS[short]:>5}  IC(δ=0)={f0[short]:+.3f}  IC(20ms)={f20[short]:+.3f}"
             f"  IC({D_LONG}ms)={fL[short]:+.3f}  half-life={HL_TXT(hl[short])}ms")
        # a genuine long-budget alternative only if it differs AND actually holds MORE edge at the long latency
        if long_ != short and abs(fL[long_]) > abs(fL[short]) + 0.005:
            s += (f"\n         |  LONG/durable   span={SPANS[long_]:>5}  IC(δ=0)={f0[long_]:+.3f}  IC(20ms)={f20[long_]:+.3f}"
                  f"  IC({D_LONG}ms)={fL[long_]:+.3f}  half-life={HL_TXT(hl[long_])}ms  <- lower peak IC, more durable at {D_LONG}ms")
        else:
            s += f"\n         |  (no distinct long-budget option — the high-IC span also holds best at {D_LONG}ms)"
        print(s)
""")

md(r"""
**Reading the family — IC vs lifetime, per head.** The heat-maps and the print above turn the N-sweep into
two surfaces, not one: *where the edge is biggest* (forward IC) and *how long it lasts* (the lifetime — read
**both** the half-life *and* the late-δ durability `IC(200ms)`, because for a persistent feature the half-life
can saturate at `>500 ms` for the whole family and stop discriminating). The takeaway the family forces: the
span with the **highest δ=20 ms IC** is not always the one that **holds its edge longest**. A **short** span
reads the freshest burst (a higher peak IC) but its IC *decays* with δ; a **long** span has a lower peak IC
but its IC stays flat — or even rises — out to hundreds of ms. So when the family offers that genuine
trade-off we suggest **two** lookbacks per head — a **short, high-IC** member for a fast stack (capture the
burst), and a **long, durable** member for a generous latency budget (a smaller but longer-lived edge) — and
let the model carry both. When the high-IC span also holds best at the long latency there is no trade-off and
one member suffices. The "carries signal" line is the honest per-head verdict: a head only earns a place if
some span's forward IC at δ=20 ms is clearly above the echo floor and not dominated by the backward IC —
measured here, never assumed. (The backward-vs-forward gap on the **cross-venue** okx/bin legs is also the
feed-resolution check: a real cross-venue *lead* keeps its forward IC at δ=20 ms; a feed-alignment artefact
shows as a big δ=0/backward IC that collapses by δ=20 ms — `span 1` reads degenerate/constant here and is
correctly excluded.)
""")

md(r"""
**Now the gates** (from §5), on the per-venue rate-head picks at the winning normalisation. Every
predictive number is the **walk-forward** mean (causal, purged). The gates: *Marginal value* — does each
leg add over the controls (and do they add *jointly*)? *No leak* — does any gain survive adding the
vol/rate *levels*? *Normaliser* — is the chosen variant's scale steady across vol buckets? *Regime-stable*
— is the marginal gain positive within calm/mid/wild vol?

We run the gates **against the rate head** (the feature's home), per leg, and surface the trade-vs-move
distinction: the marginal over `rate_momentum`/`rate_level` is the honest test of whether **trade**
intensity carries move-rate information the **move**-rate control lacks. We also report the marginal for
**all three normalisations** so the §3 normalisation axis is settled out-of-sample, not just in-sample.
""")

code(r"""
# rate-head picks at the winning normalisation, per venue
surge = {ex: feat_cache[(ex, SPANS[rate_member[ex]], BEST_NORM)] for ex in EXCHANGES}

# Marginal value on the RATE head, PER LEG (the honest per-exchange test) + JOINT (all venues together).
m_each       = {ex: round(wf_ic(base + [surge[ex]], rate_target) - wf_ic(base, rate_target), 3) for ex in EXCHANGES}
m_joint      = round(wf_ic(base + list(surge.values()), rate_target) - wf_ic(base, rate_target), 3)
m_cross_only = round(wf_ic(base + [surge["okx"], surge["bin"]], rate_target) - wf_ic(base, rate_target), 3)  # okx+bin only
m_joint_leak = round(wf_ic(base + levels + list(surge.values()), rate_target) - wf_ic(base + levels, rate_target), 3)

# NORMALISATION decided out-of-sample: joint marginal (all venues) over the controls, for each normalisation, best span per venue per norm.
def joint_marg_for(norm):
    legs = {ex: feat_cache[(ex, SPANS[best_span(ic_rate[(ex, norm)])], norm)] for ex in EXCHANGES}
    return round(wf_ic(base + list(legs.values()), rate_target) - wf_ic(base, rate_target), 3)
norm_marg = {nm: joint_marg_for(nm) for nm in NORMS}

# normaliser gate: SHIPPED-variant scale across volatility buckets — the HARD regime-invariance gate, worst venue.
# Also recompute the BASELINE scale for contrast (it is the non-invariant level we REJECTED on this very gate).
vol_decile = np.digitize(vol_level, np.nanpercentile(vol_level[np.isfinite(vol_level)], np.arange(10, 100, 10)))
def _scale(arr):
    bs = [np.nanstd(arr[vol_decile == d]) for d in range(10)]; bs = [b for b in bs if np.isfinite(b) and b > 0]
    return (max(bs) / min(bs)) if len(bs) >= 2 else float("inf")
ship_scale = round(max(_scale(surge[ex]) for ex in EXCHANGES), 2)                                   # SHIPPED (BEST_NORM), worst venue
base_scale = round(max(_scale(feat_cache[(ex, SPANS[best_span(ic_rate[(ex, 'baseline')])], 'baseline')]) for ex in EXCHANGES), 2)  # baseline, worst venue (rejected)
# regime companion: marginal RATE-head IC within each vol regime, JOINT over all legs
full_r = wf_ic_by_regime(base + list(surge.values()), rate_target, vol_regime)
base_r = wf_ic_by_regime(base, rate_target, vol_regime)
strat  = {r: round(full_r[r] - base_r.get(r, 0.0), 3) for r in full_r}

gate_rows = [dict(gate="marginal value (RATE head)", detail=f"{ex} leg alone, over rate+vol MOMENTA"
                  + ("  [byb own trades]" if ex == "byb" else "  [cross-venue]"), value=m_each[ex]) for ex in EXCHANGES]
gate_rows += [
    dict(gate="marginal value (RATE head)", detail=f"all three venues JOINTLY over controls (norm='{BEST_NORM}')", value=m_joint),
    dict(gate="marginal value (RATE head)", detail="okx+bin CROSS-VENUE legs only (no byb) over controls", value=m_cross_only),
    dict(gate="no leak (RATE head)", detail="joint gain still there after adding the vol/rate LEVELS?", value=m_joint_leak),
]
gate_rows += [dict(gate="normalisation (RATE head)", detail=f"joint marginal over controls, normalisation='{nm}'", value=norm_marg[nm]) for nm in NORMS]
gate_rows += [
    dict(gate="normaliser scale (HARD)", detail=f"SHIPPED '{BEST_NORM}' scale across vol buckets, worst venue (want < ~3)", value=ship_scale),
    dict(gate="normaliser scale (HARD)", detail="baseline scale, worst venue — the non-invariant level we REJECTED", value=base_scale),
]
gate_rows += [dict(gate="regime-stable (RATE)", detail=f"joint marginal RATE IC within {nm}-vol (companion)", value=strat.get(r, float("nan")))
              for r, nm in [(0, "calm"), (1, "mid"), (2, "wild")]]
pl.Config(set_tbl_rows=len(gate_rows), set_fmt_str_lengths=60)
pl.DataFrame(gate_rows)
""")

md(r"""
**Conclusion — read per leg and per normalisation.** The headline is the **marginal value over the
rate/vol controls** on the **rate head**, reported straight. Because the controls are byb's **move** rate
and the feature is a **trade** rate, a positive marginal here is genuinely new information — trade
intensity that the move-rate regime did not already encode (the trade-no-move / move-no-trade slack). The
**byb** leg tests whether byb's own trade rate adds over byb's own move rate; the **okx/bin** legs add the
cross-venue axis (does another venue's trade intensity lead byb's move-count?). The **normalisation** rows
settle the §3 axis **out-of-sample** — but the choice is **regime-invariance-FIRST, IC second** (§6): the
`baseline` rate is a non-invariant **level** (it fails the <3× scale gate at ~4.2×), and ÷σ_ev *also* fails
(~3.1×), so both are **disqualified** however competitive their OOS marginal. The **only** variant that
clears the scale gate is **÷λ_ev** (~2.3× — "trades per byb-mid-move", a pure rate ratio), so that is the
**shipped form**, at a real but modest IC cost vs the non-invariant baseline (joint OOS marginal ~0.11 vs
~0.11 — competitive, and well above the floor). The **normaliser scale (HARD)** rows show the shipped λ_ev
variant **passing (2.3×)** beside the rejected baseline (4.2×). The **no-leak** row checks the gain survives
the level controls; the **regime** rows check the gain holds across calm/mid/wild. **Read the realised table
for the call** — and quote the §6 marginal of the *shipped invariant form (÷λ_ev)*, never a raw IC of a
non-invariant level.
""")

md(r"""
## How long does the edge last? — the signal's lifetime and your latency budget

A feature can be perfectly causal and still not earn its headline IC: if its edge is the move (or the
burst of moves) *already underway* at the anchor, you can't capture it — by the time you observe, decide,
and act, that activity is gone. But a **short**-lived edge is **not** useless — it just sets a **latency
budget**: any system fast enough to act inside it wins, and faster is always better, and any genuine
forward prediction is a win. So we do **not** gate on this — we **measure how long the signal lasts**.

Read the feature at the anchor (causal, unchanged) but slide the *outcome* window forward by an
observe-to-act latency δ: the **forward IC** of the feature against byb's outcome over
`[anchor+δ, anchor+δ+100 ms]`, swept over δ. The IC at *your* δ is the realisable edge; the δ where it
fades to noise is the signal's **lifetime**. The **backward IC** — against the outcome that *already
happened*, `[anchor−100 ms, anchor]` — sizes the contemporaneous echo. A leg whose forward IC dies at δ>0
while its backward IC stays high is re-reporting the past, not predicting it; that is the *only* genuinely
useless case, and it is measured here, never assumed.

**We compute it PER VENUE** (byb own / okx, bin cross-venue) — each leg has its own lifetime, and a
cross-venue leg's curve at δ=0 vs δ=20 ms is exactly how you tell a feed-resolution artefact from a real
lead. `trade_rate_normalised` is an **intensity** feature, so its natural head is the **rate head**: the
forward IC of the venue's trade rate (its *level*, the rate-head pick `surge[ex]`) against byb's forward
**move-count**. We also carry the **price-head** diagnostic (the *signed* log feature against byb's signed
σ-return), since the notebook ships both heads — expected weak, intensity isn't direction.
""")

code(r"""
# Signal lifetime PER VENUE: forward IC vs observe->act latency δ (outcome window slides to [t+δ, t+δ+100ms]), + backward IC.
# The feature is still read at the anchor (causal, unchanged); only the OUTCOME window moves.
DELTAS_MS = [0, 5, 10, 20, 50, 100, 200, 500]
def _ic(feat, out):
    v = np.isfinite(feat) & np.isfinite(out)
    return spearmanr(feat[v], out[v]).statistic if v.sum() > 100 else float("nan")
def _mid_at(t):                                         # byb mid at-or-before t (causal forward-fill)
    return byb_mid[np.clip(np.searchsorted(byb_rx, t, "right") - 1, 0, len(byb_mid) - 1)]
def _ret(t0, t1):   return np.log(_mid_at(t1) / _mid_at(t0))                         # signed byb return over [t0,t1]
def _count(t0, t1): return cum_mv[np.searchsorted(byb_rx, t1, "right")] - cum_mv[np.searchsorted(byb_rx, t0, "right")]  # byb move-count over [t0,t1]

# rate head (the feature's home): venue trade-rate LEVEL -> forward move-count;  price head (diagnostic): signed log -> signed σ-return.
rate_lvl = {ex: surge[ex] for ex in EXCHANGES}                                       # rate-head pick per venue (positive intensity)
sgn_log  = {ex: np.log(np.maximum(surge[ex], 1e-12)) for ex in EXCHANGES}            # price-head diagnostic per venue

cnt_ic = {ex: [_ic(rate_lvl[ex], _count(anchor_ts + d * 1_000_000, anchor_ts + d * 1_000_000 + HORIZON_NS)) for d in DELTAS_MS] for ex in EXCHANGES}
ret_ic = {ex: [_ic(sgn_log[ex],  _ret  (anchor_ts + d * 1_000_000, anchor_ts + d * 1_000_000 + HORIZON_NS)) for d in DELTAS_MS] for ex in EXCHANGES}
cnt_back = {ex: _ic(rate_lvl[ex], _count(anchor_ts - HORIZON_NS, anchor_ts)) for ex in EXCHANGES}   # rate-head backward (already-happened) IC
ret_back = {ex: _ic(sgn_log[ex],  _ret  (anchor_ts - HORIZON_NS, anchor_ts)) for ex in EXCHANGES}   # price-head backward IC

fig, axes = plt.subplots(1, len(EXCHANGES), figsize=(5.0 * len(EXCHANGES), 4.2), squeeze=False, sharey=True)
for col, ex in enumerate(EXCHANGES):
    ax = axes[0][col]
    ax.plot(DELTAS_MS, cnt_ic[ex], "o-", color="C3", label="rate head — level->count (forward)")
    ax.plot(DELTAS_MS, ret_ic[ex], "s--", color="C0", label="price head — signed->return (forward)")
    ax.axhline(cnt_back[ex], color="C3", ls=":", alpha=0.7, label=f"rate backward = {cnt_back[ex]:+.3f}")
    ax.axhline(0, color="0.7", lw=0.8); ax.set_xlabel("observe->act latency  δ  (ms)")
    if col == 0: ax.set_ylabel("rank-IC")
    ax.set_title(f"{ex}" + ("  (own)" if ex == "byb" else "  (cross-venue)")); ax.legend(fontsize=7)
fig.suptitle(f"signal lifetime per venue — edge vs latency δ  (rate-head home; norm='{BEST_NORM}')", y=1.02)
fig.tight_layout(); plt.show()

# --- GATE 1: ECHO-NETTED (partial) forward IC — the part of the rate-head edge NOT attributable to the move-burst
# ALREADY underway at the anchor. The byb leg is OWN-ECHO-HEAVY (its backward IC > some forwards), so its raw δ=0 IC
# over-credits the contemporaneous echo. We net it out: the partial rank-IC of the trade-rate level with the FORWARD
# move-count, CONTROLLING for the TRAILING [anchor-100ms, anchor] move-count. If the raw IC collapses once the trailing
# burst is partialled out, the leg was re-reporting the past; if it survives, it is genuine forward prediction.
trail_count = _count(anchor_ts - HORIZON_NS, anchor_ts)                              # the move-burst already underway
fwd0_count  = _count(anchor_ts, anchor_ts + HORIZON_NS)                              # the δ=0 forward move-count
def _partial_ic(f, y, ctrl):                                                         # rank-partial corr of f,y controlling for ctrl
    v = np.isfinite(f) & np.isfinite(y) & np.isfinite(ctrl)
    if v.sum() <= 100: return float("nan")
    rfy = spearmanr(f[v], y[v]).statistic; rfc = spearmanr(f[v], ctrl[v]).statistic; rcy = spearmanr(ctrl[v], y[v]).statistic
    return (rfy - rfc * rcy) / np.sqrt(max((1.0 - rfc**2) * (1.0 - rcy**2), 1e-12))
echo_net = {ex: _partial_ic(rate_lvl[ex], fwd0_count, trail_count) for ex in EXCHANGES}

# Per-venue prints: the rate-head (home) forward decay curve, the realisable edge at 20 ms, the half-life, the backward echo,
# AND the echo-netted partial forward IC (the honest "is this real forward prediction, net of the burst already underway?").
print(f"signal lifetime (rate head = trade-rate level -> byb forward move-count; norm='{BEST_NORM}'):")
for ex in EXCHANGES:
    fc = cnt_ic[ex]
    half = next((d for d, ic in zip(DELTAS_MS, fc) if np.isfinite(ic) and abs(ic) < abs(fc[0]) / 2), None)
    tag = "own" if ex == "byb" else "cross-venue"
    print(f"  {ex:4} ({tag:11}) forward IC by δ(ms): " + " ".join(f"{d}:{ic:+.3f}" for d, ic in zip(DELTAS_MS, fc)))
    print(f"           edge@20ms {fc[3]:+.3f}  |  δ=0 {fc[0]:+.3f}  half-life δ≈{half} ms  |  backward(echo) {cnt_back[ex]:+.3f}"
          f"  |  ECHO-NETTED forward IC {echo_net[ex]:+.3f}  (partial, controls for the trailing burst)")
print("price-head (diagnostic) forward IC by δ(ms):")
for ex in EXCHANGES:
    print(f"  {ex:4}: " + " ".join(f"{d}:{ic:+.3f}" for d, ic in zip(DELTAS_MS, ret_ic[ex])) + f"   backward {ret_back[ex]:+.3f}")
""")

md(r"""
**Read it as a latency budget, not a pass/fail.** If a leg's forward (rate-head) IC stays useful out to
tens or hundreds of ms you have room; if it lives only a handful of ms the signal is real but demands a
fast stack. The **per-venue** split is the point: the **byb** (own) leg's lifetime is byb's own
trade-intensity clustering; the **okx/bin** (cross-venue) curves at δ=0 vs δ=20 ms separate a genuine
cross-venue *lead* (forward IC survives the latency) from a feed-resolution / contemporaneous artefact
(forward IC collapses past δ=0 while the backward IC is large). Carry this curve for each leg — the verdict
is *"this venue's trade rate predicts byb's move-count ~X ms ahead, needs latency < X,"* never *"drop
because it's fast."* (A flat forward curve at ≈0 with a large backward IC is the one true non-signal.)
""")

md(r"""
### GATE 2 — the feed-resolution control on the cross-venue (okx/bin) legs

A cross-venue "lead" can be an **artefact of feed resolution**, not economics. okx/bin print and update on
their own cadence; if a foreign venue's trade-events are simply observed at **finer time-resolution** than
byb refreshes its own book, its trade-rate read can look like it "leads" byb purely because it is *fresher*,
not because it economically *precedes* byb's moves. The control follows the template's recipe: **re-measure
the cross-venue rate-head IC with the foreign feed matched to byb's update cadence** — coarsen each foreign
venue's trade-rate read to byb's **median inter-update gap** (we snap every anchor's foreign-rate read back
to byb's most recent book-update time at-or-before it, then re-score). A **real** lead survives the cadence
match (byb still hasn't reacted by its own next update); a **feed-resolution artefact** collapses toward the
own-leg / zero. We report the matched IC beside the raw IC for okx and bin.
""")

code(r"""
# FEED-RESOLUTION CONTROL (cross-venue okx/bin): re-read the foreign trade-rate at BYB's update cadence and re-score.
# byb's book-update times = its raw front_levels rx (its own feed's resolution). Coarsening each anchor's foreign-rate
# read to byb's most-recent update at-or-before it makes the foreign feed no fresher than byb's own — a real economic
# lead survives (byb hasn't moved by its next own update), a finer-resolution artefact collapses toward zero.
byb_upd = byb_rx0                                                            # byb's RAW book-update receive-times (its feed cadence)
byb_gap_ms = float(np.median(np.diff(byb_upd))) / 1e6
print(f"byb median inter-update gap: {byb_gap_ms:.1f} ms  (the cadence we coarsen the foreign feed to)")
# snap each anchor to byb's last update at-or-before it -> read the SAME §3 trade-rate AS OF that byb-update time
byb_anchor = byb_upd[np.clip(np.searchsorted(byb_upd, anchor_ts, "right") - 1, 0, len(byb_upd) - 1)]
fwd0 = _count(anchor_ts, anchor_ts + HORIZON_NS)                            # δ=0 forward move-count (the rate-head outcome)
print("feed-resolution control — cross-venue rate-head IC, RAW vs byb-cadence-matched (norm='%s'):" % BEST_NORM)
for ex in EXCHANGES:
    N = SPANS[rate_member[ex]]                                              # the venue's shipped rate-head span
    raw_rate     = trade_rate_normalised(ex, N, BEST_NORM)                  # read at the true anchor (full resolution)
    matched_rate = trade_rate_normalised(ex, N, BEST_NORM)[np.searchsorted(anchor_ts, byb_anchor, "right") - 1]  # read AS OF byb's last update (coarsened)
    ic_raw  = _ic(raw_rate, fwd0); ic_match = _ic(matched_rate, fwd0)
    tag = "own" if ex == "byb" else "cross-venue"
    keep = "" if ex == "byb" else ("  -> LEAD SURVIVES" if abs(ic_match) > 0.5 * abs(ic_raw) else "  -> COLLAPSES (feed-resolution artefact)")
    print(f"  {ex:4} ({tag:11}) span={N:>5}:  raw IC {ic_raw:+.3f}   byb-cadence-matched IC {ic_match:+.3f}   "
          f"retained {ic_match/ic_raw*100 if ic_raw else float('nan'):5.0f}%{keep}")
""")

md(r"""
**Reading the feed-resolution control.** Compare each cross-venue leg's **raw** rate-head IC to its
**byb-cadence-matched** IC. If the matched IC stays close to the raw one, the okx/bin trade-intensity lead is
**economic** — it still predicts byb's next move-count when the foreign feed is no fresher than byb's own
updates. If it **collapses** toward zero, the apparent lead was a **feed-resolution artefact** (the foreign
book merely sampled finer than byb's), and that leg should not be sold as a cross-venue lead. The byb (own)
leg is shown for reference only — coarsening byb to its own cadence is a near-no-op. The verdict quotes the
**matched** cross-venue numbers, never the raw ones.
""")

md(r"""
## 7. What the prediction actually looks like

A single correlation hides *how* the feature changes the outcome. Group the data by the feature and look
at the real distributions the heads care about, **per venue** (byb's own trade intensity vs the
cross-venue legs):
- **rate head (its home):** how the number of upcoming byb moves grows as each venue's trade rate grows;
- **price head (diagnostic):** byb's next return for low / middle / high *signed* (log) feature — expected
  roughly flat, since intensity isn't direction.
""")

code(r"""
fig, axes = plt.subplots(1, len(EXCHANGES), figsize=(5.2 * len(EXCHANGES), 4.4), squeeze=False)
for col, ex in enumerate(EXCHANGES):
    s = surge[ex]; ax = axes[0][col]
    dec = np.digitize(s, np.nanpercentile(s[np.isfinite(s)], np.arange(10, 100, 10)))
    ax.plot(range(10), [fwd_count[dec == b].mean() for b in range(10)], "o-", color="C3", label="mean count  E[K]")
    ax.plot(range(10), [(fwd_count[dec == b] >= 1).mean() for b in range(10)], "s--", color="C1", label="P(K ≥ 1)")
    ax.set_xlabel(f"{ex} trade_rate_normalised decile (quiet → busy)"); ax.set_ylabel("forward 100 ms (byb)")
    ax.set_title(f"rate head: byb move-count vs {ex} trade rate" + ("  (own)" if ex == "byb" else "  (cross-venue)")); ax.legend(fontsize=8)
fig.suptitle(f"rate head: does each venue's TRADE intensity precede more BYB mid-moves?  (norm='{BEST_NORM}')", y=1.03)
fig.tight_layout(); plt.show()

# price head (diagnostic): forward σ-return across signed log-feature buckets — expected ~flat — shown for the byb leg
s = surge["byb"]; ls = np.log(np.maximum(s, 1e-12))
fig, axB = plt.subplots(1, 1, figsize=(7, 4.4))
lo, hi = np.nanpercentile(ls, [10, 90])
groups = [("quiet", ls <= lo, "C3"), ("≈ baseline", (ls > lo) & (ls < hi), "0.6"), ("busy", ls >= hi, "C0")]
bins = np.linspace(-8, 8, 81)
for lab, m, col in groups:
    axB.hist(np.clip(target[m & np.isfinite(target)], -8, 8), bins=bins, density=True,
             histtype="step", color=col, lw=1.7, label=f"{lab}:  mean {np.nanmean(target[m]):+.2f}")
axB.set_yscale("log"); axB.set_xlabel("forward σ-return"); axB.set_ylabel("density (log)")
axB.set_title("price head (diagnostic, byb leg): return ~unmoved by trade rate — intensity ≠ direction"); axB.legend(fontsize=8)
fig.tight_layout(); plt.show()
""")

md(r"""
**Conclusion.** **Rate head (top):** read the per-venue panels — for the **byb** leg the mean move-count
`E[K]` and `P(K ≥ 1)` should climb from quiet to busy trade-rate deciles (the §1 intensity story); the
**okx/bin** panels show whether *another* venue's trade intensity also tracks byb's upcoming move-count (a
cross-venue lead) or is flat (no cross-venue content). **Price head (bottom):** the forward-return
distribution barely shifts across quiet / baseline / busy groups (means ≈ 0) — trade rate says *how many*,
not *which way*, confirming this is an intensity feature. The open question §6 answers is whether any leg's
signal is **new over the move-rate control** — which, unlike `mid_rate_surge`'s byb leg, it can be, because
trade rate ≠ move rate.
""")

md(r"""
## 8. Input shaping for the network

This reshapes the feature for the network's input — roughly centred, unit-scale, no wild outliers — a
*different* step from §3's normalisation (which strips the regime; this just conditions the number). A
trade *rate* (or a rate ÷ yardstick) is positive and right-skewed (a burst can be many× baseline; a lull
is floored at 0), so its **log** is the natural signed, centred form. We feed **every venue's** log
feature (none privileged). Plot the byb leg (left) and compare candidates by how close to a clean bell
curve they make it (right — points on the diagonal are perfectly normal). Pick the lightest transform that
clears the bar; the same transform applies per venue.
""")

code(r"""
from scipy.stats import skew, kurtosis, rankdata, norm
f = surge["byb"]; f = f[np.isfinite(f) & (f > 0)]
lf = np.log(f)                                       # log a positive rate: the natural centred, signed form
med = np.median(lf); mad = 1.4826 * np.median(np.abs(lf - med)) + 1e-12; rz = (lf - med) / mad
cand = {"raw rate (z)": (f - f.mean()) / f.std(),
        "log (z)": (lf - lf.mean()) / lf.std(),
        "log robust + clip ±4": np.clip(rz, -4, 4),
        "rank-Gaussian(log)": norm.ppf((rankdata(lf) - 0.5) / len(lf))}
print(f"raw feature: skew={skew(f):+.2f}  excess_kurt={kurtosis(f):.1f}")
print(f"log feature: skew={skew(lf):+.2f}  excess_kurt={kurtosis(lf):.1f}  (0 = normal)")
for name, v in cand.items():
    print(f"  {name:22} excess_kurt={kurtosis(v):>6.1f}   max|·|={np.abs(v).max():.1f}")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.6))
axA.hist(lf, bins=120, density=True, color="C0", alpha=.85, label="log feature (byb)")
xg = np.linspace(np.percentile(lf, 0.05), np.percentile(lf, 99.95), 300)
axA.plot(xg, norm.pdf(xg, lf.mean(), lf.std()), "k--", lw=1, label="normal (same σ)")
axA.set_yscale("log"); axA.set_xlabel("log trade_rate_normalised (byb)"); axA.set_ylabel("density (log)")
axA.set_title(f"log-feature distribution (skew {skew(lf):+.2f}, excess kurt {kurtosis(lf):.1f})"); axA.legend(fontsize=8)

q = norm.ppf((np.arange(1, len(lf) + 1) - 0.5) / len(lf)); sub = np.linspace(0, len(lf) - 1, 4000).astype(int)
for name, v in cand.items():
    axB.plot(q[sub], np.sort(v)[sub], lw=1.5, label=name)
axB.plot([-5, 5], [-5, 5], "k:", lw=1, label="perfect normal")
axB.set_xlim(-5, 5); axB.set_xlabel("normal quantile"); axB.set_ylabel("normalised feature quantile")
axB.set_title("how to normalise — QQ vs N(0,1): on-diagonal & bounded wins"); axB.legend(fontsize=8)
fig.tight_layout(); plt.show()
""")

md(r"""
**Conclusion.** The raw trade-rate feature is positive and right-skewed; taking the **log** centres it and
roughly symmetrises it — the natural signed form to feed the network, applied identically to each venue's
leg. A plain z-score of the log controls the bulk but can leave fat-tail spikes, so the lightest transform
that clears the "no wild outliers" bar is **log → robust-z → clip ±4** (max|·| = 4.0). The heavier
rank-Gaussian flattens further but throws information away. So: feed `log(trade_rate_normalised)` per
venue, robust-standardised and clipped — at the normalisation §6 chose.
""")

md(r"""
## 9. When is per-exchange worth it? — short vs long lookback

`trade_rate_normalised` is a **poolable** per-venue feature (one rate per venue), so the template's
per-exchange question applies: keep them **per-exchange**, collapse to a **single** venue, or **pool**
them? The answer depends on the time-scale — so we sweep the **span** (the lookback) and compare the three,
walk-forward, on the rate head, at the chosen normalisation.

**The mechanism, and what to expect.** The cross-venue edge is a **trade-intensity lead/lag** — one venue
trading heavily a beat before another. At **short** spans the venues genuinely differ (who is busy first),
so the per-venue set carries distinct cross-venue lead/lag and should beat a single venue; at **long**
spans every venue's smoothed trade rate converges to the same slow pace, so a **single** venue already says
what the rest do and the extra legs are redundant. We compute, at each span: the **per-exchange** marginal
(all three legs), the **single best** venue's marginal, the **pooled** (mean log-rate) marginal, and the
**cross-venue-only** (okx+bin) marginal — each over the controls, walk-forward.
""")

code(r"""
# Sweep the span (lookback). Compare per-exchange / single / pooled / cross-venue, at the chosen normalisation, on the rate head.
look_rows = []
for k, N in enumerate(SPANS):
    legs = {ex: feat_cache[(ex, N, BEST_NORM)] for ex in EXCHANGES}    # one rate per venue at this span
    per_ex  = round(wf_ic(base + list(legs.values()), rate_target) - wf_ic(base, rate_target), 3)   # all three legs
    singles = {ex: round(wf_ic(base + [legs[ex]], rate_target) - wf_ic(base, rate_target), 3) for ex in EXCHANGES}
    best_ex = max(singles, key=singles.get); single = singles[best_ex]
    logs    = [np.log(np.maximum(legs[ex], 1e-12)) for ex in EXCHANGES]
    pooled  = round(wf_ic(base + [np.nanmean(np.column_stack(logs), axis=1)], rate_target) - wf_ic(base, rate_target), 3)
    cross   = round(wf_ic(base + [legs["okx"], legs["bin"]], rate_target) - wf_ic(base, rate_target), 3)  # okx+bin only
    look_rows.append(dict(span=N, per_exchange=per_ex, best_single=f"{single} ({best_ex})", pooled=pooled, cross_venue_only=cross))
look_df = pl.DataFrame(look_rows)
print(f"rate-head marginal over controls, by span (lookback); normalisation = '{BEST_NORM}'")
print("per_exchange = all 3 legs | best_single = best one venue | pooled = mean log-rate | cross_venue_only = okx+bin (no byb)")
look_df
""")

code(r"""
# Plot the per-exchange vs single vs pooled curves across the span sweep — the §9 cross-over picture.
xs = [r["span"] for r in look_rows]
per = [r["per_exchange"] for r in look_rows]
sin = [float(r["best_single"].split()[0]) for r in look_rows]
poo = [r["pooled"] for r in look_rows]
crs = [r["cross_venue_only"] for r in look_rows]
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(xs, per, "o-", label="per-exchange (all 3 legs)", color="C0")
ax.plot(xs, sin, "s--", label="best single venue", color="C1")
ax.plot(xs, poo, "^:", label="pooled (mean log-rate)", color="C2")
ax.plot(xs, crs, "d-.", label="cross-venue only (okx+bin)", color="C3")
ax.set_xscale("log"); ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("trade-rate EMA span (lookback, trades) — short ← → long"); ax.set_ylabel("rate-head marginal IC over controls (walk-forward)")
ax.set_title("§9 per-exchange vs single vs pooled across the lookback:\nshort span → per-exchange carries distinct cross-venue trade-intensity lead/lag; long → they converge")
ax.legend(fontsize=8); fig.tight_layout(); plt.show()
""")

md(r"""
**Conclusion — read the table and curves above.** The §9 picture is the cross-over the template predicts:
at **short** spans the per-exchange set (all three legs) should sit **above** the best single venue — the
venues genuinely differ, because who is trading heavily first is real cross-venue lead/lag; at **long**
spans the curves should **converge**, because every venue's slow trade rate settles to the same pace and a
single venue already says what the rest do. The **cross-venue-only** curve (okx+bin, no byb) is the
cleanest read on whether *another venue's* trade intensity leads byb at all: where it sits above 0, the
cross-venue legs carry genuine marginal value over byb's own; where it collapses to ≈0, byb's own leg
carries the signal. **Whatever the realised cross-over, the verdict is read straight off these numbers.**
""")

md(r"""
## 10. The verdict, and what it takes to ship

**`trade_rate_normalised(ex; N) = W_trades(ex; N) / E_dt(N) / yardstick`** on the shared trade clock — a
single EMA of a venue's **trade** rate (trades/sec, one event per trade-timestamp), per venue, **÷ a
regime yardstick**. Feed the **signed** (here log) feature to the **rate head** (intensity), **all
exchanges**, at the span §6 chose.

**The normalisation verdict (the headline correction).** The raw `baseline` rate is the highest-IC variant
in-sample (~0.21 best rate-IC) **but it FAILS the hard regime-invariance scale gate** — its scale wanders
**~4.2×** across vol buckets (worst venue), i.e. it reads systematically bigger in busy markets, so it is a
non-invariant **level** and is **disqualified** however high its raw IC. Crucially, **÷σ_ev does NOT rescue
it either** — measured on the rate-head best span, σ_ev-normalisation still drifts **~3.1×** and also fails
the <3× gate (σ_ev and the trade rate do not co-move tightly enough). Only **÷λ_ev** — "trades per
byb-mid-move", a pure ratio of two rates — clears the gate, at **~2.3×**. So, regime-invariance-FIRST: **the
shipped form is `trade_rate ÷ λ_ev`** (the most interpretable variant, and the only invariant one), at a
real but modest IC cost vs the non-invariant baseline (best rate-IC ~0.12 vs ~0.21; joint OOS marginal over
the controls ~0.11 — still well above the floor). The §6 table's `normaliser scale (HARD)` rows show the
shipped λ_ev variant **passing (2.3×)** beside the rejected baseline (4.2×).

**The control-circularity verdict.** Note `λ_ev` is here doing double duty — it is both the *normaliser*
(making the feature "trades per byb-mid-move") and one of the *controls*. Even so, the feature is **not**
circular: `trade_rate ÷ λ_ev` is the trade COUNT per move, a churn ratio, not the move rate itself. The
honest test is the **marginal over the rate/vol controls** (§6), which is **positive** (joint ~0.11, clears
the control bar and survives the no-leak level check at ~0.13), so the churn-normalised trade rate carries
genuinely new intensity information beyond the move-rate regime.

**Echo-netted, not raw (the byb leg especially).** The byb leg is **own-echo-heavy** — its backward
(already-happened) IC (~0.34) far exceeds its forward ICs — so its raw δ=0 IC over-credits the move-burst
already underway. Netted out (rate-head level vs the forward move-count, controlling for the trailing
`[anchor−100 ms, anchor]` move-count), the **byb leg's echo-netted forward IC collapses to ~0.03** — it was
mostly re-reporting the burst in progress. The **bin** leg, by contrast, holds: its echo-netted forward IC
stays ~0.11 (≈ its raw δ=0), and okx ~0.05. Quote **these post-echo numbers**, not the raw δ=0 IC, for
every leg — the byb own leg is largely an echo, the cross-venue (esp. bin) legs are genuine.

**Cross-venue legs are cadence-matched, not raw.** Each okx/bin "lead" is re-measured with the foreign feed
**coarsened to byb's ~10 ms update cadence** (the feed-resolution control). Both **survive**: okx retains
~61% of its raw IC (+0.090 → +0.055) and bin ~57% (+0.132 → +0.075) under the cadence match — well above the
collapse threshold, so these are **economic leads**, not finer-feed artefacts. Quote the
**byb-cadence-matched** cross-venue numbers. The §9 cross-venue-only curve confirms the cross-venue legs add
over byb's own at short spans (e.g. ~0.10 at span 10) and converge to ≈0 by long spans.

**To ship:**
- [ ] the streaming (constant-work-per-trade) builder, per venue, matching this analysis version (§4, done)
- [ ] the oracle (§4) and tests, passing **bit-exact for every per-venue leg × normalisation**
- [ ] **the SHIPPED normalisation = ÷λ_ev** ("trades per byb-mid-move"), chosen regime-invariance-FIRST: it
      is the only norm that passes the <3× scale gate (2.3×); baseline (4.2×) AND ÷σ_ev (3.1×) both rejected
- [ ] the §6 marginal-over-controls recorded **per leg**, with the trade≠move call made
- [ ] the **echo-netted** forward IC recorded per leg (the byb leg's raw IC is echo-inflated)
- [ ] the **byb-cadence-matched** cross-venue IC recorded (feed-resolution control) for okx/bin
- [ ] the §9 per-exchange-vs-single cross-over recorded (where per-exchange stops paying)
- [ ] the chosen rate-head span per venue written down, with the yardstick span
- [ ] the data quirks handled (bad zero-price prints; byb/okx mid = merged_levels, bin = front_levels;
      simultaneous prints = ONE trade-event; one decay per timestamp)
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
out = Path("/Users/leith/source/new/boba/notebooks/features/trade_rate_normalised.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, "with", len(cells), "cells")
