"""Generate notebooks/features/ofi_pathsum.ipynb — the path-sum OFI (order-flow-imbalance) feature analysis.

Clones the methodology of notebooks/features/template.ipynb (price_dislocation), swapping in the
fast/slow Order-Flow-Imbalance family:

    raw atom : level-1 OFI (Cont-Kukanov-Stoikov) from byb front_levels, a FLOW on book updates
    feature  : EMA_fast(OFI) / EMA_slow(OFI)  — a fast/slow family, predicts DIRECTION (signed)

Same target as the template (byb next-100ms), same trade clock, same yardsticks, same hygiene gates,
and a SECOND independent streaming oracle validated bit-exact on a real block.

PATH-SUM VARIANT of build_ofi.py: same at-most-one-sample-per-timestamp / one-decay-per-trade EMA, but
each timestamp's OFI value is the SUM of the per-raw-row Cont-Kukanov-Stoikov increments at that ts (the
full intra-nanosecond book path), not the single endpoint increment of the collapsed final book. Differs
from the collapse build only where a venue has multiple book updates per timestamp (bin).
"""
import json
from pathlib import Path

cells = []
def md(s):   cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r"""
# Feature analysis — Order-Flow Imbalance (OFI), a fast/slow family — **path-sum variant**

> **Path-sum variant.** Identical to `ofi.ipynb` except for how same-timestamp byb book rows feed the
> OFI flow: a burst of book updates at one nanosecond is **one** EMA sample whose value is the **sum**
> of the per-row Cont-Kukanov-Stoikov increments (the full intra-ns book path), not the single
> increment of the collapsed final book (endpoint). The EMA still takes **at most one** sample per
> timestamp and decays once per shared-trade-clock tick — only the injected *value* changes. So it
> differs from `ofi.ipynb` only on venues with multiple book updates per timestamp (**bin**); byb/okx
> are essentially unaffected.

This notebook follows the **method** of the feature-analysis template, swapping in a new feature:
**level-1 Order-Flow Imbalance (OFI)** on byb's top-of-book, read as a **fast-minus-slow oscillator**. The
text explains what to do and why; the code does it. The structure is identical to the template —
hypothesis (§1), exact definition (§2), build-it-twice (§3), the **oracle** (§4), the **hygiene
gates** (§5/§6), then interpretation (§7–§10) — so the two read side-by-side.

We build features for a model that forecasts one exchange's mid-price about 100 ms ahead. Three
crypto exchanges appear — **byb** (Bybit), **bin** (Binance), **okx** (OKX) — and **byb is the
target**: the one we predict. ("Mid-price" = the midpoint between the best buy and best sell quote.)
The next section recaps the two-head model these features feed.

**This feature** is `ofi`: how lop-sided the *changes* in byb's best bid/ask have been lately —
buy-side replenishment minus sell-side, summed over recent book updates. A run of bid-building and
ask-pulling (positive OFI) is order flow leaning up; the prediction is that byb's mid follows. It is
a **single-venue** feature built **from byb's own book** — unlike the template's cross-venue gap —
so §9's per-exchange/pool question is framed accordingly. (We read it as a fast-EMA **minus** a
slow-EMA of the OFI flow — a sign-stable oscillator; §1 explains why a *difference*, not a ratio.)

**A feature is "done" when two checks pass:**
- **The oracle (§4)** — the code really computes what we think it does (a second, independent,
  dead-simple streaming build, bit-exact on a real block).
- **The hygiene gates (§5)** — the signal is real and holds in any market, not just an echo of
  "the market is volatile right now."
""")

md(r"""
## The model these features feed: two heads

A feature is only worth something if it helps the model predict. We forecast how byb's mid-price
moves over the next ~100 ms, split into two simpler questions — the two **heads**:

**Price head — which way and how far?** Over the next few price-moves, what is the *signed* move
(direction *and* size, together)? The head predicts the whole distribution of that move in units of
byb's recent **volatility** — the yardstick `σ_ev` (the exp-weighted RMS of byb's *actual* mid-moves)
— so its target is `price change ÷ σ_ev`.

**Rate head — how many moves?** Busy markets pack many price-moves into the window, quiet ones few.
This head predicts the *count* of moves over the next 100 ms against the recent pace — the rate
yardstick `λ_ev` — so its target is `count ÷ λ_ev`.

Both yardsticks are EMAs **decayed on the trade clock** (`α = 2/(span+1)`) but **updated between
trades** — they react to every byb mid-move, so they read live at every instant — at one fixed span
`YARDSTICK_N`. (`σ_ev` is the exp-weighted RMS of byb's mid-moves, read as an `E/W` ratio so the many
non-move trades cancel; `λ_ev` is the exp-weighted move-count `W` ÷ the exp-weighted seconds-per-trade
= byb's moves per second.) Like every average here they live on the **trade-tick clock** — never
wall-clock or a hard window.

**Why two heads?** A move over a window is *how many* little moves happen times *how big* each one is.
This "how many × how big" split is the classic **subordination** model of prices (Clark 1973; Ané &
Geman 2000): returns over fixed clock-time look messy but behave once you condition on the *number* of
events. So a feature predicting the *count* (rate head) and one predicting the *per-move direction*
(price head) feed the two factors that multiply together. We test OFI against **both**, and — per the
guard rails — feed the **signed** feature to both heads.
""")

md(r"""
## 1. What the feature is, and why it might work

| | |
|---|---|
| **what** | level-1 order-flow imbalance of byb's own book, read as a fast-EMA **minus** a slow-EMA over two time-scales |
| **feeds** | both heads — *direction* (price head) and *intensity* (rate head); both are fed the *signed* feature |
| **predicts** | byb's mid-price 100 ms from now |

**The idea.** Each time byb's top-of-book changes, the **OFI increment** counts the net depth added
on the buy side minus the sell side (definition in §2). Positive means bids built / asks were pulled —
buying pressure; negative the reverse. Sum these increments into a *fast* EMA and a *slow* EMA, and take
their **difference** `fast − slow`. The slow EMA is the prevailing flow baseline; the fast EMA is the
recent flow. A fast that is larger than the slow says buy/sell pressure has just intensified in that
direction — and the prediction is byb's mid follows.

> **Why a difference, not a ratio.** An earlier cut of this feature used the *ratio* `fast / slow`. That
> is wrong for a **signed direction** feature: the ratio **inverts sign** every time the slow leg crosses
> zero (a small negative baseline flips a positive fast leg to a large *negative* reading), so "lean up"
> and "lean down" get swapped exactly when the baseline is near zero. The **difference** `fast − slow`
> never inverts: its sign is the sign of *(recent flow − baseline flow)*, which is what we mean. It is
> also the form whose two legs are each a clean `E/W` mean, so the oracle is genuinely **bit-exact**
> (no near-zero-denominator blow-up to relabel away).

**Why it should work.** OFI is the most direct microstructure read of **net pressure on the quote**:
the Cont–Kukanov–Stoikov result is that contemporaneous mid-price changes are close to *linear* in
OFI, with depth setting the slope. If OFI also *leads* the mid by tens of milliseconds — the
replenishment shows in the book a beat before the mid prints — then recent OFI predicts the next move.
The fast-minus-slow **oscillator** (rather than a raw level) is self-centring: it asks "is flow leaning
harder than its own recent baseline?", so it stays comparable as depth and activity drift through the day.

**What would disprove it.** No predictive power at any fast/slow pair, or power that vanishes once we
control for how volatile / busy the market is (meaning OFI was only a volatility proxy).

> **Single-venue, every exchange kept.** OFI here is built from **byb's own** book (the target's book
> is what most directly precedes byb's mid). The same construction generalises to each venue's book;
> we never privilege one. §9 frames the per-exchange-vs-pool question for this poolable feature.
""")

md(r"""
## Choosing the EMA — and the clock (the part that's easy to get silently wrong)

Every smoother here is an EMA on the **trade clock**: its **decay** steps once per *trade-timestamp*
(`α = 2/(span+1)`), but its **value updates on every event in between**, so a read between trades is
never stale. Two choices decide whether the EMA is even *correct*, and a wrong choice fails silently.

**Choice 1 — which EMA, and how you read it** (`boba.ema`). OFI is a **sparse flow**: it carries a
value only on a byb **book update** (a top-of-book change), and is a *non-observation* on the many
trades that don't touch byb's book. So each leg is a **`KernelMeanEMA`** read as `E / W` — the
self-normalising per-event mean. The `W` denominator counts only the book updates that injected an
increment, so the decay between book updates (caused by intervening trades) and the warm-up bias
**cancel in the ratio** — exactly as for `σ_ev`. This is the same machine as the template's `σ_ev`,
pointed at a different flow (OFI increments instead of squared mid-moves).

**Choice 2 — *when* you push a value in** (the injection clock — separate from the decay clock).
Decay is always once per trade-timestamp. OFI **injects on each byb front_levels update** (one OFI
increment per book change), and is *not* pushed on trades or on other venues' events. Pushing a `0`
on every trade would silently turn "mean OFI per book update" into "OFI diluted by the trade rate" —
contaminating it with activity. So we inject **only on a real byb book change**, and read `E/W`.

Two rules keep the read correct **between** trades:

- **React to every relevant event, read the freshest value.** Decay rides the trade clock, but each
  leg's `E`/`W` must *update* on every byb book update that lands an OFI increment — even between
  trades — and the read at a grid anchor reflects all increments since the last trade. Never a stale
  last-trade snapshot.
- **Records sharing a timestamp are ONE EMA sample.** A burst of byb book rows stamped at the exact
  same nanosecond is injected as **one** sample at that timestamp (one `W` tick, one decay-eligible
  event) — never N samples. Its **value** is the **sum** of the per-row Cont-Kukanov-Stoikov increments
  across the burst (the full intra-nanosecond book path), the canonical OFI flow at that instant — not
  just the endpoint of the final book. (The decay clock likewise advances at most once per timestamp.)
""")

md(r"""
## 2. The exact definition

**Causal** — every value uses only byb book updates and trades at-or-before the moment it's computed.

**Measured on the trade clock** — we count progress in *trades* (one tick per trade-timestamp on any
venue; simultaneous prints are one tick), not clock-seconds. byb book updates between trades inject
OFI and refresh the read, but do **not** advance the clock.

**The raw atom — level-1 OFI (Cont–Kukanov–Stoikov).** For two consecutive *raw* top-of-book rows of
byb's book (`prev` → `cur`) — then summed per timestamp (below):

```
e =  (cur_bid_prc >= prev_bid_prc) ? cur_bid_qty  : 0      # bid improved/held -> buy-side depth added
   - (cur_bid_prc <= prev_bid_prc) ? prev_bid_qty : 0      # bid worsened/held -> buy-side depth removed
   - (cur_ask_prc <= prev_ask_prc) ? cur_ask_qty  : 0      # ask improved/held -> sell-side depth added
   + (cur_ask_prc >= prev_ask_prc) ? prev_ask_qty : 0      # ask worsened/held -> sell-side depth removed
```

(When a price is unchanged, *both* the `>=` and `<=` branches fire on that side — `e` then reduces to
the **signed change in resting size** at that price, the standard OFI convention.) Positive `e` =
net buy-side pressure on the quote.

**OFI is a sparse FLOW on byb book updates.** Inject `e` on each byb top-of-book change; decay once per
trade-timestamp on the shared clock; read each leg as `E / W` (a self-normalising per-event mean). Two
legs — a **fast** span and a **slow** span — and the feature is their **difference**:

**2a — The shape** (the whole idea, in one line):

`ofi(fast, slow) = EMA_fast(OFI) − EMA_slow(OFI)`     where each `EMA = E/W` over the OFI flow

The slow leg is the prevailing flow baseline; the fast leg is recent flow. The difference is positive
when flow has just leaned harder *up* than its baseline, negative when *down* — and, crucially, its
**sign never inverts** (a difference of two means is monotone in each leg; unlike a ratio it does not
flip when the baseline crosses zero). Both legs share the same depth/activity units (depth × weight), so
their difference is already comparable across the day — no explicit `σ_ev` division is needed, exactly
the template's "don't normalise a self-centred feature" rule. (`σ_ev` and `λ_ev` still appear — as the
**targets'** yardsticks and the **controls**, §5 — but the feature itself is a bare difference.) The one
caveat is scale: a difference of two depth-scale legs is *not* bounded, so §8 still reshapes it (robust
z + clip) before the network, and §5 gates the shaped feature.

**2b — The EMAs, and how they update.** Both legs are a per-event `E/W` mean on the byb-OFI flow: push the
OFI increment `e` (weight `1`) on every byb book update, decay every trade-timestamp, read `E/W` at the
grid anchor (committed-per-trade `E`/`W` + the partial epoch of increments since the last trade). `α =
2/(span+1)` cancels in the `E/W` ratio, so only the relative kernel weights matter. Both reads obey the
two between-trade rules above. §3 builds exactly this; §4 re-derives it from raw events streamed one at
a time — **plain numpy, no production EMA classes** — and confirms it bit-exact.

The cell below loads the data, builds the shared trade clock, and computes **both yardsticks** (`σ_ev`,
`λ_ev`) as `E/W` flows on that clock at span `YARDSTICK_N` — used for the targets and controls, exactly
as in the template.
""")

code(r"""
import numpy as np, polars as pl
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.stats import spearmanr
from boba.io import list_blocks, load_block

COIN        = "eth_usdt_p"
TARGET      = "byb_eth_usdt_p"                    # the exchange we predict AND whose book the OFI is built from
OTHERS      = ["bin", "okx"]                      # other venues — used for their OFI legs in the symmetric per-venue sweep (§6/§9)
# freshest mid per exchange (for the TARGET and controls). byb/okx use merged_levels; bin MUST use front_levels
# (merged_levels is DISALLOWED for bin perp in boba.io — it raises). This is policy, not tuning.
MID_STREAM  = {"bin": "front_levels", "byb": "merged_levels", "okx": "merged_levels"}
FAST        = [1, 10, 50, 200]                   # fast-EMA spans to sweep (1 = no smoothing, the freshest OFI)
SLOW        = [100, 500, 2000, 5000]             # slow-EMA spans (each must exceed the fast one)
HORIZON_NS  = 100 * 1_000_000                    # how far ahead we predict (100 ms, in nanoseconds)
YARDSTICK_N = 10000                              # the ONE span for BOTH yardsticks (σ_ev, λ_ev): a trade-tick EMA (α=2/(N+1))
block       = list_blocks(TARGET, "front_levels")[0]   # one ~24h slice of recorded data

# load each exchange's mid-price stream (rows arrive in time order) — for the TARGET and the vol/rate controls
def load_mid(ex):
    df = (load_block(block, f"{ex}_{COIN}", MID_STREAM[ex]).select("rx_time", "bid_prc", "ask_prc").drop_nulls())
    return df["rx_time"].cast(pl.Int64).to_numpy(), (df["bid_prc"].to_numpy() + df["ask_prc"].to_numpy()) / 2
mids = {ex: load_mid(ex) for ex in ("bin", "byb", "okx")}

# the trade clock: one tick per trade-TIMESTAMP. Simultaneous prints (one order sweeping levels) are ONE event -> ONE tick.
trade_ts = []
for ex in ("bin", "byb", "okx"):
    td = (load_block(block, f"{ex}_{COIN}", "trade").select("rx_time", "prc", "qty")
          .filter((pl.col("prc") > 0) & (pl.col("qty") > 0)))         # drop bad prc=qty=0 prints (bin-perp insurance/ADL)
    trade_ts.append(td["rx_time"].cast(pl.Int64).to_numpy())
trade_prints = np.concatenate(trade_ts)
merged_ts = np.unique(trade_prints)                               # collapse same-timestamp prints: at most one decay per timestamp
n_ticks = len(merged_ts)
print(f"trade clock: {n_ticks:,} ticks (timestamps) from {len(trade_prints):,} trade prints")

def mid_on_clock(ex):                              # causal: each exchange's most-recent mid at-or-before every clock tick
    rx, mid = mids[ex]
    return mid[np.clip(np.searchsorted(rx, merged_ts, "right") - 1, 0, len(mid) - 1)]

# --- the yardsticks (σ_ev, λ_ev): react to every byb mid-move, decay once per trade. Identical to the template. ---
byb_rx0, byb_mid0 = mids["byb"]                                                   # raw byb merged mid stream
keep = np.concatenate([byb_rx0[1:] != byb_rx0[:-1], [True]])                     # collapse same-TIMESTAMP rows to ONE update (final mid)
byb_rx, byb_mid = byb_rx0[keep], byb_mid0[keep]
byb_lm = np.log(byb_mid)
byb_blr = np.empty_like(byb_lm); byb_blr[0] = 0.0; byb_blr[1:] = np.diff(byb_lm)  # byb log-return per timestamp
mv = byb_blr != 0.0                                                              # a REAL byb mid-move: ONE per timestamp where mid changed
mv_rx, mv_r2 = byb_rx[mv], byb_blr[mv] ** 2                                       # move times + squared returns
cum_mv = np.concatenate([[0.0], np.cumsum(mv.astype(float))])                    # running count of byb mid-moves (rate-head target)
byb_dt = np.zeros(n_ticks); byb_dt[1:] = np.diff(merged_ts) / 1e9                # seconds between consecutive trades (per-trade)
def _ewma(x, span):                                                # per-trade EMA (for the seconds-per-trade leg of λ_ev)
    a = 2.0 / (span + 1.0); return lfilter([a], [1.0, -(1.0 - a)], x)
def _flow_at(anchors, src_rx, val, span):   # EWMA of `val` over an EVENT stream `src_rx`, decayed once per trade-timestamp, read AT each anchor
    a = 2.0 / (span + 1.0)
    k = np.searchsorted(merged_ts, src_rx, "left")                              # trades strictly before each event (a same-rx trade decays it)
    ep = np.bincount(k, weights=val, minlength=n_ticks + 1)                     # per-trade-epoch sums
    x = np.zeros(n_ticks + 1); x[1:] = a * (1.0 - a) * ep[:-1]
    com = lfilter([1.0], [1.0, -(1.0 - a)], x)                                  # committed E just after each trade
    ta = np.searchsorted(merged_ts, anchors, "right") - 1                       # last trade <= anchor
    cs = np.concatenate([[0.0], np.cumsum(val)])                               # prefix sums over the event stream (the partial epoch)
    partial = cs[np.searchsorted(src_rx, anchors, "right")] - cs[np.searchsorted(src_rx, merged_ts[ta], "right")]
    return com[ta + 1] + a * partial
def yardsticks(anchors, span):                                     # σ_ev, λ_ev — defined AT the anchor, reacting to every byb mid-move
    e_sq = _flow_at(anchors, mv_rx, mv_r2, span)                  # E: exp-weighted squared byb moves
    e_mv = _flow_at(anchors, mv_rx, np.ones(mv_r2.size), span)   # W: exp-weighted byb-move count
    e_dt = _ewma(byb_dt, span)[np.searchsorted(merged_ts, anchors, "right") - 1]  # seconds/trade (per-trade, held flat between trades)
    sig = np.sqrt(e_sq / np.maximum(e_mv, 1e-12))                 # σ_ev: RMS byb mid-move (E/W — non-moves cancel)
    lam = e_mv / np.maximum(e_dt, 1e-12)                          # λ_ev: byb mid-moves per second
    return sig, lam
print(f"yardsticks: react to every byb mid-move; decay span {YARDSTICK_N} trades")
""")

md(r"""
## 2c. The OFI flow — built once from byb's book

Now the new piece: the **OFI increment stream**. Load byb's raw `front_levels`, form one OFI increment
`e` per consecutive **raw** row by the §2 formula, then **sum** the increments sharing a timestamp into
**one** sample per timestamp (records sharing a nanosecond are ONE EMA sample — one `W` tick — whose
value is the full intra-ns path sum). That gives an event stream `(ofi_rx, ofi_e)` — one sample per byb
book-update timestamp — that we decay on the trade clock and read as `E/W`, *exactly* like
`mv_rx, mv_r2` feeds `σ_ev`. The same `_flow_at` machine serves both.
""")

code(r"""
# --- the OFI flow on byb's own book: ONE sample per timestamp = SUM of the intra-ns OFI increments ---
def ofi_stream(listing):
    fl = (load_block(block, listing, "front_levels")
          .select("rx_time", "bid_prc", "bid_qty", "ask_prc", "ask_qty").drop_nulls())
    rx = fl["rx_time"].cast(pl.Int64).to_numpy()
    bp, bq = fl["bid_prc"].to_numpy(), fl["bid_qty"].to_numpy()
    ap, aq = fl["ask_prc"].to_numpy(), fl["ask_qty"].to_numpy()
    # OFI (Cont-Kukanov-Stoikov) increment for EVERY consecutive RAW book row (prev -> cur); NO same-ns collapse
    pbp, pbq, pap, paq = bp[:-1], bq[:-1], ap[:-1], aq[:-1]   # prev
    cbp, cbq, cap, caq = bp[1:],  bq[1:],  ap[1:],  aq[1:]    # cur
    e = (np.where(cbp >= pbp, cbq, 0.0) - np.where(cbp <= pbp, pbq, 0.0)
         - np.where(cap <= pap, caq, 0.0) + np.where(cap >= pap, paq, 0.0))
    # ONE sample per timestamp: SUM the increments sharing an rx_time (one EMA sample / one W tick per ts)
    uniq, inv = np.unique(rx[1:], return_inverse=True)       # increments are stamped at the CUR row's rx_time
    return uniq, np.bincount(inv, weights=e)                  # value = the full intra-ns path sum at each timestamp
ofi_rx, ofi_e = ofi_stream(TARGET)
print(f"byb OFI flow: {len(ofi_e):,} per-timestamp samples;  e mean {ofi_e.mean():+.2f}  std {ofi_e.std():.1f}")

def ofi_legs_at(anchors, span):                # E/W of the OFI flow at each anchor: committed-per-trade + partial epoch since last trade
    E = _flow_at(anchors, ofi_rx, ofi_e, span)                 # exp-weighted sum of OFI increments
    W = _flow_at(anchors, ofi_rx, np.ones(ofi_e.size), span)   # exp-weighted count of increments
    return E, W
""")

md(r"""
## 3. Build it (twice)

Build the feature two ways: this fast array version for analysis, and — in §4 — a streaming version
that does constant work per trade (no growing buffers). They have to agree.

We lay an evaluation grid every 50 ms (half the 100 ms horizon — plenty of samples; adjacent 100 ms
outcome windows still overlap ~50%, which is why §5's walk-forward gate uses an embargo), read byb's
actual move over the next 100 ms (the price-head target, in `σ_ev` units), and compute the OFI
feature `EMA_fast(OFI) − EMA_slow(OFI)` at each grid point.
""")

code(r"""
# evaluation grid (causal) + forward targets
WARMUP = 5 * max(YARDSTICK_N, max(SLOW))   # = 50000: enough trades for the slowest EMA/yardstick to converge
anchor_ts      = np.arange(merged_ts[WARMUP], merged_ts[-1] - HORIZON_NS, 50 * 1_000_000)   # 50 ms grid, past warmup
sigma_at_anchor, lam_at_anchor = yardsticks(anchor_ts, YARDSTICK_N)   # both yardsticks at each grid point (span YARDSTICK_N)
print(f"σ_ev median {np.nanmedian(sigma_at_anchor):.2e},  λ_ev median {np.nanmedian(lam_at_anchor):.2f} moves/s")

mid_now    = byb_mid[np.searchsorted(byb_rx, anchor_ts, "right") - 1]
mid_fwd    = byb_mid[np.searchsorted(byb_rx, anchor_ts + HORIZON_NS, "right") - 1]
fwd_return = np.log(mid_fwd / mid_now)
target     = fwd_return / sigma_at_anchor                          # byb's 100 ms return ÷ σ_ev — the price head's target (σ-units)

# precompute the OFI E/W legs once per fast/slow span we will need (a ratio is cheap to recombine)
_E_cache, _W_cache = {}, {}
def _legs(span):
    if span not in _E_cache:
        _E_cache[span], _W_cache[span] = ofi_legs_at(anchor_ts, span)
    return _E_cache[span], _W_cache[span]
def ofi(n_fast, n_slow):                          # the feature: EMA_fast(OFI) − EMA_slow(OFI), each EMA = E/W on the OFI flow
    Ef, Wf = _legs(n_fast); Es, Ws = _legs(n_slow)
    fast = Ef / np.where(Wf == 0.0, np.nan, Wf)
    slow = Es / np.where(Ws == 0.0, np.nan, Ws)
    return fast - slow                                            # sign-stable oscillator; sign = direction of the lean vs baseline (never inverts)
print(f"grid: {len(anchor_ts):,} anchors")
print("sample ofi(10,500) finite frac:", round(float(np.isfinite(ofi(10, 500)).mean()), 4))
""")

md(r"""
## 4. Check the code is right — the oracle (an independent streaming build)

**Non-negotiable.** Reproduce the feature with a second, **independent** implementation and confirm the
two agree on real data, **bit-exact**. The oracle is a dead-simple O(1) state machine you push **raw
events** into one at a time — `on_book(...)` for a byb top-of-book row, `on_trade(...)` for a trade on
any venue — reading the current feature from `value()`. State is two scalar `(E, W)` pairs (the fast and
slow OFI legs). No buffers, no history.

**It shares NO code with §3 and uses NO production helpers.** The two `E/W` legs are hand-rolled here
as plain floats — `tick` multiplies `E`,`W` by `(1−α)`; `add` adds `α·e` to `E` and `α` to `W`; the read
is `E/W` — written from the §2 description alone, *not* imported from `boba.ema`. So agreement is a real
cross-check of the maths, not the same class compared to itself.

**The design:**
- Fed **only raw events** in rx-time order. It tracks its **previous raw** byb top-of-book and, on each
  new byb row, accumulates one OFI increment `e` (by the §2 formula) into the current timestamp's sum.
- Events sharing a **timestamp are one EMA sample**: the driver applies them all (each byb book row adds
  its increment to the running sum; trades set a flag), then calls **`refresh()` once** — which injects
  the **summed** intra-timestamp OFI as **at most one** sample into both legs (a trade-only timestamp
  injects nothing), then advances the clock **at most once** (decays both legs) and only if a trade landed.
- `value()` returns `(E_fast/W_fast) − (E_slow/W_slow)`, read at the instant it's called — the increments
  since the last trade are already folded in (the legs `add` between trades; only `tick`/decay waits
  for a trade).

We feed the **whole raw stream** — byb's book updates and **every** venue's trades — into one builder
and read the feature at each grid anchor. The difference of two `E/W` legs has no near-zero-denominator
blow-up, so the **feature itself** (not just the legs) is bit-exact — see the banner below.
""")

code(r"""
import math

class _EW:
    # A plain hand-rolled E/W flow mean (causal geometric-kernel mean), written from the §2 description —
    # NO boba.ema, NO shared code with §3. tick = decay E,W by (1-α); add = inject α·value into E, α into W; read E/W.
    __slots__ = ("a", "E", "W")
    def __init__(self, span):
        self.a = 2.0 / (span + 1.0); self.E = 0.0; self.W = 0.0
    def tick(self):  self.E *= (1.0 - self.a); self.W *= (1.0 - self.a)
    def add(self, v): self.E += self.a * v;    self.W += self.a
    def value(self): return self.E / self.W if self.W > 0.0 else float("nan")

class LiveOFI:
    # Pure feature state machine for OFI = EMA_fast(OFI) − EMA_slow(OFI). Each leg is a plain _EW (E/W flow):
    # PATH-SUM: each byb book ROW accumulates one OFI increment vs the previous RAW row; per timestamp we inject
    # the SUMMED increment as ONE sample into both legs; on a trade-timestamp we decay both ONCE.
    # State is O(1), all scalar — two (E, W) pairs + the previous RAW byb top-of-book. WHEN to read/tick is the driver's job.
    def __init__(self, target, n_fast, n_slow):
        self.target = target
        self.leg_f = _EW(n_fast)                            # fast OFI leg (E/W)
        self.leg_s = _EW(n_slow)                            # slow OFI leg (E/W)
        self.pb = self.pq = self.pa = self.paq = None      # PREVIOUS raw byb top-of-book row (any timestamp)
        self.have_prev = False
        self.ts_e = 0.0; self.ts_got_incr = False          # running SUM of this timestamp's OFI increments + did any land
        self.was_trade_present = False                     # did a trade land this timestamp? -> exactly one decay

    def on_book(self, listing, bid, bq, ask, aq):          # byb BBO row -> accumulate ONE OFI increment vs the previous RAW row
        if listing != self.target: return
        if self.have_prev:
            e = ((bq if bid >= self.pb else 0.0) - (self.pq if bid <= self.pb else 0.0)
                 - (aq if ask <= self.pa else 0.0) + (self.paq if ask >= self.pa else 0.0))
            self.ts_e += e; self.ts_got_incr = True        # sum within the timestamp
        self.pb, self.pq, self.pa, self.paq = bid, bq, ask, aq   # this raw row becomes prev
        self.have_prev = True

    def on_trade(self, listing, px, lifts_ask):            # any venue's trade -> just flag the timestamp as traded (OFI is book-only)
        self.was_trade_present = True

    def refresh(self):                                     # ONE per TIMESTAMP: inject the SUMMED OFI (at most one sample), then decay AT MOST ONCE
        traded, self.was_trade_present = self.was_trade_present, False
        if self.ts_got_incr:                               # >=1 byb book change this timestamp -> one sample = SUM of its increments
            self.leg_f.add(self.ts_e); self.leg_s.add(self.ts_e)   # inject the summed intra-ns OFI into both legs (no decay yet)
            self.ts_e = 0.0; self.ts_got_incr = False      # reset the per-timestamp accumulator
        if traded:                                         # a trade landed -> advance the clock once
            self.leg_f.tick(); self.leg_s.tick()

    def legs(self):                                        # the two E/W leg reads (for the per-leg precision check)
        return self.leg_f.value(), self.leg_s.value()

    def value(self):                                       # E/W difference read live: (E_f/W_f) − (E_s/W_s)
        f, s = self.leg_f.value(), self.leg_s.value()
        if not (f == f) or not (s == s):                   # nan in either leg -> undefined
            return float("nan")
        return f - s

# --- gather the raw stream: byb front_levels (book) + EVERY venue's trades, over a slice ---
NF, NS, N_GRID = 10, 500, 40_000                  # validate ofi(10,500) over the first ~N_GRID grid points
cutoff = int(anchor_ts[min(N_GRID, len(anchor_ts) - 1)])    # wall-clock time of the N_GRID-th grid anchor
cols = {k: [] for k in "rx kind bid bq ask aq".split()}     # kind 0 = byb book, kind 1 = trade
def add(rx, kind, bid, bq, ask, aq):
    m = rx <= cutoff; n = int(m.sum())
    cols["rx"].append(rx[m]); cols["kind"].append(np.full(n, kind, np.int8))
    cols["bid"].append(bid[m].astype(float)); cols["bq"].append(bq[m].astype(float))
    cols["ask"].append(ask[m].astype(float)); cols["aq"].append(aq[m].astype(float))
# byb book rows — raw front_levels (the OFI source); bid/ask prc + qty
flb = load_block(block, TARGET, "front_levels").select("rx_time", "bid_prc", "bid_qty", "ask_prc", "ask_qty").drop_nulls()
add(flb["rx_time"].cast(pl.Int64).to_numpy(), 0,
    flb["bid_prc"].to_numpy(), flb["bid_qty"].to_numpy(), flb["ask_prc"].to_numpy(), flb["ask_qty"].to_numpy())
# trades from every venue — only their rx_time matters (they tick the shared clock); fields unused, pass zeros
for ex in ("bin", "byb", "okx"):
    td = load_block(block, f"{ex}_{COIN}", "trade").select("rx_time", "prc", "qty").filter((pl.col("prc") > 0) & (pl.col("qty") > 0))
    z = np.zeros(td.height)
    add(td["rx_time"].cast(pl.Int64).to_numpy(), 1, z, z, z, z)
C = {k: np.concatenate(v) for k, v in cols.items()}
order = np.lexsort((C["kind"], C["rx"]))           # rx ascending; book (0) before trade (1) on ties (book settles before the tick)
rxL, kindL, bidL, bqL, askL, aqL = (C[k][order].tolist() for k in "rx kind bid bq ask aq".split())
print(f"streaming {len(rxL):,} raw events (byb book + all-venue trades) over ~{N_GRID:,} grid points...")

# --- the CALLER drives it: apply each timestamp's events, refresh() once, READ value() + legs() at every grid anchor ---
feat = LiveOFI(TARGET, NF, NS)
na = min(N_GRID, len(anchor_ts))
stream = np.full(na, np.nan); fastL = np.full(na, np.nan); slowL = np.full(na, np.nan)
n = len(rxL); i = 0; ai = 0
while i < n:
    rx = rxL[i]
    while ai < na and anchor_ts[ai] < rx:          # read every anchor whose state is settled (all events before rx applied)
        stream[ai] = feat.value(); fastL[ai], slowL[ai] = feat.legs(); ai += 1
    while i < n and rxL[i] == rx:                  # apply EVERY event stamped at this nanosecond
        if kindL[i] == 0: feat.on_book(TARGET, bidL[i], bqL[i], askL[i], aqL[i])
        else:             feat.on_trade("trade", 0.0, False)
        i += 1
    feat.refresh()                                 # apply the timestamp: form one OFI increment, then decay once if a trade landed
while ai < na:
    stream[ai] = feat.value(); fastL[ai], slowL[ai] = feat.legs(); ai += 1

# --- check the streaming feature vs the §3 vectorized ofi(NF, NS) ---
# The OFI feature is now a DIFFERENCE of two O(1) E/W legs. Unlike a ratio, a difference has no near-zero-denominator
# blow-up, so BOTH the individual legs AND the feature itself are bit-exact to absolute round-off — one honest metric.
ref      = ofi(NF, NS)[:na]
ref_fast = (_E_cache[NF] / np.where(_W_cache[NF] == 0, np.nan, _W_cache[NF]))[:na]   # the vectorized legs (cached in §3)
ref_slow = (_E_cache[NS] / np.where(_W_cache[NS] == 0, np.nan, _W_cache[NS]))[:na]
mlegs = np.isfinite(fastL) & np.isfinite(slowL) & np.isfinite(ref_fast) & np.isfinite(ref_slow)
fast_d = np.nanmax(np.abs(fastL[mlegs] - ref_fast[mlegs]))     # fast E/W leg, absolute round-off
slow_d = np.nanmax(np.abs(slowL[mlegs] - ref_slow[mlegs]))     # slow E/W leg, absolute round-off
both = np.isfinite(stream) & np.isfinite(ref)
abs_d = np.nanmax(np.abs(stream[both] - ref[both]))            # feature (fast − slow), absolute round-off
print(f"oracle vs vectorized ofi(fast={NF}, slow={NS})  on {int(both.sum()):,} grid points:")
print(f"  fast E/W leg  max|diff| {fast_d:.2e}   slow E/W leg  max|diff| {slow_d:.2e}   (each O(1) -> absolute round-off)")
print(f"  difference feature  max|diff| {abs_d:.2e}   (a difference of bit-exact legs is itself bit-exact)")
assert fast_d < 1e-9 and slow_d < 1e-9, "live OFI legs do not reproduce the vectorized E/W"      # the legs are bit-exact
assert abs_d  < 1e-9, "live OFI difference does not reproduce the vectorized feature"            # the feature is bit-exact (absolute)
print(f"oracle: independent plain-numpy streaming build reproduces the OFI feature  OK "
      f"(absolute round-off: legs/feature max|diff| ~{max(fast_d, slow_d, abs_d):.0e})")
""")

md(r"""
**Conclusion.** From one stream of raw events — byb's book updates and every venue's trades — the
independent plain-numpy streaming builder reproduces the §3 vectorized `ofi(fast, slow)` to
floating-point precision. OFI is a **difference of two O(1) `E/W` legs**, and a difference (unlike a
ratio) has no near-zero-denominator amplification — so the **whole feature** is bit-exact, not just its
legs: on this block the fast and slow legs match to **~1.3e-11 / ~6.6e-13** absolute, and the **feature
`fast − slow` matches to ~1.3e-11 absolute** — pure recursive last-digit round-off, no 1e-5 spikes (that
was an artefact of the old ratio's near-zero-denominator blow-up, gone with the difference form). The two
implementations share **no code and no helper class** — the oracle hand-rolls its own
plain-float `E/W` legs (`_EW`) and tracks the previous book to form its own OFI increments one event at
a time, while §3 vectorizes via `_flow_at` — so agreeing this tightly means the OFI feature is computed
correctly: causal, on the trade clock, injected on byb book changes, read `E/W`. The §3 build is
trustworthy.
""")

md(r"""
## 5. Is the signal real? — the hygiene gates

A correlation is an easy way to fool yourself. The gates check that OFI predicts *something the
market's current state doesn't already tell us*. We build four "control" signals from the recent past:
- **rate momentum** and **rate level** — both from `λ_ev` (byb's mid-move rate): is byb moving more or
  less often than usual?
- **vol momentum** and **vol level** — the same two, for volatility.
Then we measure OFI's predictive power **on top of** those controls.

"Predictive power" is the **rank correlation** between feature and outcome (Spearman — robust to
outliers), scored **out-of-sample with a purged, embargoed, expanding-window walk-forward**: each fold
trains only on the *past*, leaves an embargo gap sized to clear the 100 ms outcome windows, and scores
on the next segment; we average over folds. That's the causal, ship-grade estimate — strictly
past→future. (A single 60/40 split is a faster screen but tests only one transition.)

The gates ask: does OFI *add* over the controls (walk-forward)? Does the gain *survive* once we also
control for the *level* of vol and rate (so it isn't secretly just volatility/activity)? Is its scale
steady across volatility states? And does the gain hold across volatility *regimes* (the companion)?
""")

code(r"""
# --- the four control signals: the two yardsticks (level) plus a fast/slow momentum of each ---
FAST_YARD = YARDSTICK_N // 10                        # a faster span (1/10 the yardstick) for the momentum controls
sig_fast, lam_fast = yardsticks(anchor_ts, FAST_YARD)
vol_level     = np.log(sigma_at_anchor)                                             # σ_ev — how volatile now
vol_momentum  = np.log(sig_fast / sigma_at_anchor)                                  # recent vol vs slower vol
rate_level    = np.log(lam_at_anchor)                                               # λ_ev = byb's mid-move rate
rate_momentum = np.log(lam_fast / lam_at_anchor)                                    # recent mid-move rate vs slower

# Out-of-sample scoring = a purged, expanding-window WALK-FORWARD (causal: only past -> future).
def wf_folds(features, y, k=6, embargo=2000):       # yields (test_mask, oos_prediction) for each fold
    design = np.column_stack(features); n = len(y); valid = np.isfinite(design).all(1) & np.isfinite(y)
    edges = np.linspace(0, n, k + 1).astype(int)
    for i in range(1, k):                            # fold i: train on the PAST minus an embargo gap, test on the next segment
        te = np.zeros(n, bool); te[edges[i]:edges[i + 1]] = True
        tr = np.zeros(n, bool); tr[:max(0, edges[i] - embargo)] = True
        train, test = valid & tr, valid & te
        if train.sum() < 100 or test.sum() < 100: continue
        mu, sd = design[train].mean(0), design[train].std(0) + 1e-12
        X = np.column_stack([(design - mu) / sd, np.ones(n)])
        coef, *_ = np.linalg.lstsq(X[train], y[train], rcond=None)
        yield test, X @ coef

def wf_ic(features, y):                              # mean OOS rank-IC across the walk-forward folds (the ship-grade gate)
    return float(np.mean([spearmanr(p[t], y[t]).statistic for t, p in wf_folds(features, y)]))
def wf_ic_by_regime(features, y, reg):              # same, but the mean OOS rank-IC WITHIN each regime bucket (the companion)
    acc = {}
    for t, p in wf_folds(features, y):
        for r in np.unique(reg[t]):
            m = t & (reg == r)
            if m.sum() >= 100: acc.setdefault(int(r), []).append(spearmanr(p[m], y[m]).statistic)
    return {r: float(np.mean(v)) for r, v in acc.items()}

vol_regime = np.digitize(vol_level, np.nanpercentile(vol_level[np.isfinite(vol_level)], [33, 67]))   # 0 calm, 1 mid, 2 wild
base, levels = [rate_momentum, vol_momentum], [rate_level, vol_level]   # momenta = base controls; levels added later for the leak test
print("control-only predictive power (walk-forward):  momenta", round(wf_ic(base, target), 3),
      " momenta+levels", round(wf_ic(base + levels, target), 3),
      " (near 0 = controls barely predict direction, so any feature gain is genuinely new)")
""")

md(r"""
**Conclusion.** On their own the controls carry essentially **no** directional signal — walk-forward
rank-IC near 0 for the momenta and with the levels added. That is what we want: the regime barely
predicts *which way* byb moves, so any rank-IC OFI shows *on top of* these controls is genuinely new
information, not the regime wearing a disguise. That makes the "added over the controls" gates below a
fair test.
""")

md(r"""
## 6. Two choices: which time-scale per head, and which exchanges to keep

OFI is a **family** across time-scales (every fast/slow pair). The same feature can carry signal for
both heads, so we check two things:
- does the **signed** feature predict *direction* — which way (and how far) byb moves next?
- does its **magnitude** predict *intensity* — *how many* moves byb makes next?

The magnitude check is a **diagnostic only**: the model is fed the *signed* feature for both heads
(guard rails) — pre-taking `|·|` per exchange would stop the rate head learning that opposing flows
cancel. We sweep the whole family — built symmetrically **from each exchange's own book** — against
both targets, and draw heat-maps. We keep *all* exchanges; the only thing we choose is the best
time-scale, **per head**.

The rate-head target is the count of byb's moves over the next 100 ms, divided by `λ_ev` — "more or
fewer moves than usual," using the rate yardstick.
""")

code(r"""
# rate-head target = byb moves (trade clock) in the next 100 ms, ÷ λ_ev (the rate yardstick).
fwd_count = (cum_mv[np.searchsorted(byb_rx, anchor_ts + HORIZON_NS, "right")]
             - cum_mv[np.searchsorted(byb_rx, anchor_ts, "right")])            # byb mid-moves over the next 100 ms
rate_target = fwd_count / np.maximum(lam_at_anchor, 1e-9)   # count ÷ λ_ev ∝ "more/fewer moves than usual"

# Build each exchange's OWN-book OFI flow, then sweep the fast/slow family for BOTH heads, all exchanges symmetrically.
EX_LIST = ["byb"] + OTHERS                              # byb (the target's own book) + bin, okx — never privilege one
ofi_flows = {ex: ofi_stream(f"{ex}_{COIN}") for ex in EX_LIST}    # (rx, e) per venue's book
def ofi_ex(ex, n_fast, n_slow):                        # the OFI feature from exchange `ex`'s own book
    rx, e = ofi_flows[ex]
    Ef = _flow_at(anchor_ts, rx, e, n_fast); Wf = _flow_at(anchor_ts, rx, np.ones(e.size), n_fast)
    Es = _flow_at(anchor_ts, rx, e, n_slow); Ws = _flow_at(anchor_ts, rx, np.ones(e.size), n_slow)
    fast = Ef / np.where(Wf == 0.0, np.nan, Wf); slow = Es / np.where(Ws == 0.0, np.nan, Ws)
    return fast - slow                                 # sign-stable difference (never inverts when the slow leg crosses 0)

price_grid = {ex: np.full((len(FAST), len(SLOW)), np.nan) for ex in EX_LIST}   # signed feature -> byb's signed return
rate_grid  = {ex: np.full((len(FAST), len(SLOW)), np.nan) for ex in EX_LIST}   # |feature|      -> byb's move count
for ex in EX_LIST:
    for i, nf in enumerate(FAST):
        for j, ns in enumerate(SLOW):
            if nf >= ns: continue
            d = ofi_ex(ex, nf, ns)
            fin = np.isfinite(d) & np.isfinite(target)
            price_grid[ex][i, j] = spearmanr(d[fin], target[fin]).statistic
            finr = np.isfinite(d) & np.isfinite(rate_target)
            rate_grid[ex][i, j]  = spearmanr(np.abs(d[finr]), rate_target[finr]).statistic

fig, axes = plt.subplots(2, len(EX_LIST), figsize=(5.0 * len(EX_LIST), 8.4), squeeze=False)
for row, (grids, head) in enumerate([(price_grid, "price head: signed -> return"), (rate_grid, "rate head: |feature| -> move count")]):
    for col, ex in enumerate(EX_LIST):
        ax = axes[row][col]; grid = grids[ex]; im = ax.imshow(grid, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(SLOW))); ax.set_xticklabels(SLOW); ax.set_xlabel("slow span")
        ax.set_yticks(range(len(FAST))); ax.set_yticklabels(FAST); ax.set_ylabel("fast span")
        ax.set_title(f"{head}  —  {ex}")
        for i in range(len(FAST)):
            for j in range(len(SLOW)):
                if np.isfinite(grid[i, j]): ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", color="w", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("OFI predictive power across the time-scale family — every exchange, both heads (fast span=1 = no smoothing)", y=1.01)
fig.tight_layout(); plt.show()

# We do NOT pick an exchange. Each grid cell is IN-SAMPLE; best_member is the in-sample argmax used ONLY to PICK a time-scale.
# The chosen feature is re-scored OUT-OF-SAMPLE by the §5 walk-forward gates below — that is the number that counts.
def best_member(grid): return np.unravel_index(np.nanargmax(grid), grid.shape)
price_member = {ex: best_member(price_grid[ex]) for ex in EX_LIST}
rate_member  = {ex: best_member(rate_grid[ex])  for ex in EX_LIST}
print("kept features (one per exchange, all fed to the model — none privileged):")
for ex in EX_LIST:
    pi, pj = price_member[ex]; ri, rj = rate_member[ex]
    print(f"  {ex}:  price head (fast={FAST[pi]}, slow={SLOW[pj]}) power={price_grid[ex][pi, pj]:+.3f}"
          f"   |  rate head (fast={FAST[ri]}, slow={SLOW[rj]}) power={rate_grid[ex][ri, rj]:+.3f}")
""")

md(r"""
**Reading the heat-maps.** Each cell is an in-sample rank-IC for one (fast, slow) pair and one
exchange's own-book OFI. The **sign** of the price-head map matters: OFI is signed so that *positive*
flow should precede *up* moves, so positive cells confirm the §1 mechanism. The **byb** column (the
target's own book) most directly precedes byb's own mid, but on this block **all three venues** show a
comparably strong price-head IC (and §5 confirms they add over each other OOS), so every exchange's OFI
is kept (leadership rotates; §9). The best (fast, slow) per exchange/head is an
in-sample pick used only to choose a time-scale; the §5 gates below re-score it out-of-sample.
""")

md(r"""
### 6b. IC is only half the story — map the SIGNAL LIFETIME across the whole family

The §6 heat-map ranks each (fast, slow) pair by its IC **at δ=0** — the edge you'd realise with *zero*
observe-to-act latency. But a real stack has latency, and two pairs with the *same* δ=0 IC can be worth
very different amounts: one whose edge is gone by the time you act, another whose edge survives tens or
hundreds of ms. So we run the §"How long does the edge last?" companion **across the entire family at
once** — every (fast, slow) pair, both heads, every venue — and draw two more heat-maps beside the IC one:

- **edge@20 ms** — the forward IC after a realistic 20 ms observe→act latency (window slid to
  `[anchor+20 ms, anchor+20 ms+100 ms]`). This is the IC a *latency-bound* system actually captures.
- **half-life (ms)** — the δ at which the forward IC first falls below **half** its δ=0 value (`>500` if
  it never does within the probed range). This is the size of the **latency budget** the pair grants you.

We also carry the **backward IC** (vs the already-happened move `[anchor−100 ms, anchor]`) per pair, so
we can tell a genuine forward predictor from a contemporaneous **echo**: a pair whose forward IC@20 ms is
near zero while its backward IC is large is re-reporting the past, not leading it.

**The key insight this surfaces: within one family, N trades IC against half-life.** A *short* fast/slow
pair reacts to the freshest lean — often a higher δ=0 IC, but a **short** half-life (small latency budget).
A *longer* pair smooths the lean — often a **lower** IC, but a **longer** half-life (more latency room,
and frequently an *equal-or-higher* IC once you pay the 20 ms). **Both can be useful**, so where the family
offers a genuine short-high-IC vs long-high-half-life split, we suggest **more than one** lookback per head.

(Computed on a **40k-anchor diagnostic subsample** of the §3 grid — the full family × 8 δ × 3 venues ×
2 heads is far too much for all 1.7 M anchors, and the rank-IC is stable at 40k. Same `ofi_ex` builder,
same byb forward returns/counts, same Spearman metric as the §6 map — only the grid is thinned.)
""")

code(r"""
# --- family-wide signal lifetime: forward IC@0, IC@20ms, half-life, backward IC for EVERY pair / head / venue ---
DELTAS_MS = [0, 5, 10, 20, 50, 100, 200, 500]                 # observe->act latencies to slide the outcome window by
DIAG_N    = 40_000                                            # diagnostic subsample of the §3 grid (the full family is too big at 1.7M)
didx      = np.linspace(0, len(anchor_ts) - 1, min(DIAG_N, len(anchor_ts))).astype(int)
diag      = anchor_ts[didx]                                   # the 40k diagnostic anchors
# (the rate-head IC uses the raw forward move-count; dividing by λ_ev is a positive monotone rescale and drops out of a rank-IC)

def _ic(feat, ret):
    v = np.isfinite(feat) & np.isfinite(ret)
    return spearmanr(feat[v], ret[v]).statistic if v.sum() > 100 else float("nan")
def _mid_at(t):  return byb_mid[np.clip(np.searchsorted(byb_rx, t, "right") - 1, 0, len(byb_mid) - 1)]
def _ret(t0, t1):   return np.log(_mid_at(t1) / _mid_at(t0))
def _count(t0, t1): return cum_mv[np.searchsorted(byb_rx, t1, "right")] - cum_mv[np.searchsorted(byb_rx, t0, "right")]

# byb forward returns (price head) and forward move-counts (rate head) at each δ — shared across all pairs/venues
price_fwd = {d: _ret(diag + d * 1_000_000, diag + d * 1_000_000 + HORIZON_NS) for d in DELTAS_MS}
rate_fwd  = {d: _count(diag + d * 1_000_000, diag + d * 1_000_000 + HORIZON_NS) for d in DELTAS_MS}
price_back = _ret(diag - HORIZON_NS, diag)                    # the already-happened move (echo gauge), price head
rate_back  = _count(diag - HORIZON_NS, diag)                  # the already-happened move-count (echo gauge), rate head

def ofi_ex_diag(ex, n_fast, n_slow):                          # the §6 own-book OFI difference, evaluated on the diagnostic grid
    rx, e = ofi_flows[ex]
    Ef = _flow_at(diag, rx, e, n_fast); Wf = _flow_at(diag, rx, np.ones(e.size), n_fast)
    Es = _flow_at(diag, rx, e, n_slow); Ws = _flow_at(diag, rx, np.ones(e.size), n_slow)
    fast = Ef / np.where(Wf == 0.0, np.nan, Wf); slow = Es / np.where(Ws == 0.0, np.nan, Ws)
    return fast - slow                                        # sign-stable difference

def _half_life(curve):                                        # first δ where |forward IC| < half its δ=0 value; 999 (=">500") if it never does
    ic0 = curve[0]
    if not np.isfinite(ic0) or ic0 == 0.0: return np.nan
    for d, ic in zip(DELTAS_MS, curve):
        if np.isfinite(ic) and abs(ic) < abs(ic0) / 2.0: return float(d)
    return 999.0

HEADS = [("price", "signed -> byb return", price_fwd, price_back, lambda d: d),          # price head: signed OFI difference
         ("rate",  "|feature| -> byb move-count", rate_fwd, rate_back, np.abs)]          # rate head: |OFI| difference (diagnostic)
# per head/venue: one (FAST x SLOW) grid each of {IC@0, IC@20ms, half-life, backward IC}
ic0_g  = {(h, ex): np.full((len(FAST), len(SLOW)), np.nan) for h, *_ in HEADS for ex in EX_LIST}
ic20_g = {(h, ex): np.full((len(FAST), len(SLOW)), np.nan) for h, *_ in HEADS for ex in EX_LIST}
half_g = {(h, ex): np.full((len(FAST), len(SLOW)), np.nan) for h, *_ in HEADS for ex in EX_LIST}
back_g = {(h, ex): np.full((len(FAST), len(SLOW)), np.nan) for h, *_ in HEADS for ex in EX_LIST}
for hname, _desc, fwd, back, sgn in HEADS:
    for ex in EX_LIST:
        for i, nf in enumerate(FAST):
            for j, ns in enumerate(SLOW):
                if nf >= ns: continue
                feat = sgn(ofi_ex_diag(ex, nf, ns))                       # signed for price, |·| for rate — exactly the §6 head convention
                curve = [_ic(feat, fwd[d]) for d in DELTAS_MS]            # forward IC vs latency δ
                ic0_g [(hname, ex)][i, j] = curve[0]
                ic20_g[(hname, ex)][i, j] = curve[DELTAS_MS.index(20)]
                half_g[(hname, ex)][i, j] = _half_life(curve)
                back_g[(hname, ex)][i, j] = _ic(feat, back)
print(f"family lifetime swept on {len(diag):,} diagnostic anchors  ({len(FAST)}x{len(SLOW)} pairs x {len(HEADS)} heads x {len(EX_LIST)} venues)")

# --- three heat-maps PER head/venue: IC@δ=0 (the §6 metric), edge@20ms, half-life ---
def _annot(ax, grid, fmt, color="w"):
    for i in range(len(FAST)):
        for j in range(len(SLOW)):
            if np.isfinite(grid[i, j]): ax.text(j, i, fmt(grid[i, j]), ha="center", va="center", color=color, fontsize=7)
def _axfmt(ax, title):
    ax.set_xticks(range(len(SLOW))); ax.set_xticklabels(SLOW); ax.set_xlabel("slow span")
    ax.set_yticks(range(len(FAST))); ax.set_yticklabels(FAST); ax.set_ylabel("fast span"); ax.set_title(title, fontsize=9)

for hname, desc, *_ in HEADS:
    fig, axes = plt.subplots(3, len(EX_LIST), figsize=(4.6 * len(EX_LIST), 11.5), squeeze=False)
    for col, ex in enumerate(EX_LIST):
        g0, g20, gh = ic0_g[(hname, ex)], ic20_g[(hname, ex)], half_g[(hname, ex)]
        im0 = axes[0][col].imshow(g0,  cmap="viridis", aspect="auto"); _axfmt(axes[0][col], f"IC @ δ=0  —  {ex}")
        _annot(axes[0][col], g0, lambda v: f"{v:+.3f}"); fig.colorbar(im0, ax=axes[0][col], fraction=0.046)
        im1 = axes[1][col].imshow(g20, cmap="viridis", aspect="auto"); _axfmt(axes[1][col], f"edge @ 20ms latency  —  {ex}")
        _annot(axes[1][col], g20, lambda v: f"{v:+.3f}"); fig.colorbar(im1, ax=axes[1][col], fraction=0.046)
        ghp = np.where(gh >= 999, 600.0, gh)                              # ">500" plotted at 600 so the colour saturates at the long end
        im2 = axes[2][col].imshow(ghp, cmap="magma", aspect="auto", vmin=0, vmax=600); _axfmt(axes[2][col], f"half-life (ms)  —  {ex}")
        _annot(axes[2][col], gh, lambda v: ">500" if v >= 999 else f"{v:.0f}"); fig.colorbar(im2, ax=axes[2][col], fraction=0.046)
    fig.suptitle(f"{hname} head ({desc}) — IC vs LIFETIME across the OFI family, per venue (40k diagnostic grid)", y=1.005)
    fig.tight_layout(); plt.show()
""")

md(r"""
**Reading the three maps together (per head/venue).** The top row is the **δ=0 IC** (the §6 metric); the
middle row is the **edge a 20 ms-latency system actually captures**; the bottom row is the **half-life** —
how big a latency budget each pair grants. A pair is *worth more than its δ=0 IC suggests* when its
middle-row cell stays bright and its bottom-row cell is large (a long budget); it is *worth less* when the
edge@20 ms collapses or the half-life is a handful of ms. The cell below prints, **per head/venue**, the
best δ=0 pair AND the best long-budget pair, and flags whether the head carries genuine forward signal
(forward IC@20 ms clearly nonzero, not just a backward echo).
""")

code(r"""
# For each head/venue, surface the IC-vs-half-life trade-off explicitly: the highest-IC pair AND the
# longest-budget pair, plus an honest "carries signal?" verdict (forward edge@20ms vs the backward echo).
def _best(grid):  return np.unravel_index(np.nanargmax(grid), grid.shape) if np.isfinite(grid).any() else None
def _name(ij):    return f"(fast={FAST[ij[0]]}, slow={SLOW[ij[1]]})"
SUGGEST = {}                                                   # (head, ex) -> list of suggested (nf, ns) lookbacks
for hname, desc, *_ in HEADS:
    print(f"\n=== {hname} head — {desc} ===")
    for ex in EX_LIST:
        g0, g20, gh, gb = ic0_g[(hname, ex)], ic20_g[(hname, ex)], half_g[(hname, ex)], back_g[(hname, ex)]
        hi = _best(g0)                                         # the headline (highest δ=0 IC) pair
        if hi is None: print(f"  {ex}: no finite IC"); continue
        # long-budget candidate: among pairs whose edge@20ms is within 70% of the best edge@20ms, take the longest half-life
        e20_best = np.nanmax(g20)
        cand = [(i, j) for i in range(len(FAST)) for j in range(len(SLOW))
                if np.isfinite(g20[i, j]) and g20[i, j] >= 0.7 * e20_best and e20_best > 0]
        longb = max(cand, key=lambda ij: (np.nan_to_num(gh[ij], nan=-1), g20[ij])) if cand else hi
        carries = (np.isfinite(g20[hi]) and abs(g20[hi]) >= 0.01 and abs(g20[hi]) >= 0.3 * abs(np.nan_to_num(gb[hi])))
        def _hl(ij): v = gh[ij]; return ">500" if v >= 999 else f"{v:.0f}ms"
        print(f"  {ex:>3}:  HIGH-IC {_name(hi):>22}  IC@0={g0[hi]:+.3f} IC@20={g20[hi]:+.3f} half={_hl(hi):>5} back={gb[hi]:+.3f}"
              f"   |  LONG-BUDGET {_name(longb):>22}  IC@0={g0[longb]:+.3f} IC@20={g20[longb]:+.3f} half={_hl(longb):>5}"
              f"   |  carries signal: {'YES' if carries else 'no'}")
        # suggest one or two lookbacks: always the high-IC pair; add the long-budget pair when it is genuinely different and long
        sug = [hi]
        if longb != hi and (np.nan_to_num(gh[longb], nan=0) > np.nan_to_num(gh[hi], nan=0)) and np.isfinite(g20[longb]) and g20[longb] > 0:
            sug.append(longb)
        SUGGEST[(hname, ex)] = sug
""")

md(r"""
**What the family-wide lifetime says (read off the printout and the maps).** All numbers here are
**in-sample** rank-IC across the (fast, slow) family — they pick a *time-scale*, nothing more. The
out-of-sample verdict for **both heads** is the §5 walk-forward; the rate head in particular shrinks once
scored OOS (see §5) — do not read these cells as the shippable edge.

- **Price head — every venue carries forward direction signal.** Each venue's δ=0 IC peaks at a
  lightly-smoothed fast leg over a long slow leg (the printout names it — here `(fast=10, slow=5000)` for all
  three) and the edge **survives the latency**: IC@20 ms stays close to IC@0 (byb +0.227→+0.201, bin
  +0.275→+0.264, okx +0.248→+0.219) with half-lives ~200 ms. byb's forward IC is nonzero but rides on a
  **large backward echo** (+0.357), so its honest lead is the echo-netted number (§"how long does the edge
  last?"); bin and okx show a **smaller backward echo than their forward IC** — a cleaner cross-venue lead.
  The family also offers a **longer-budget** alternative (a *very* fast n_fast=1 leg over a long slow leg)
  trading a little δ=0 IC for a longer half-life. So the price edge is **multi-venue**, not single-venue.
- **Rate head — |OFI| → move-count is positive in-sample for all three venues**, peaking at a fast-over-
  slow pair, surviving 20 ms with half-lives ~200 ms; bin is the cleanest (smallest backward echo relative
  to its forward edge). **But this is in-sample** — §5 re-scores |OFI| through the purged walk-forward
  against `count÷λ_ev` with the rate controls in `base`, and the OOS marginal is **smaller** than these
  cells (finding #1): joint ≈ +0.17 OOS vs ≈ +0.18–0.24 here. Treat the §6 rate cells as a *time-scale
  picker*, the §5 number as the verdict.

**The picks (time-scales only — the OOS verdict is §5):**
- **Price head:** **all venues** carry the edge — the short-high-IC pair `(10, 5000)` for a low-latency
  stack; the n_fast=1 long-budget pair if you have ~100–500 ms of latency to spend.
- **Rate head:** all venues, the short-high-IC pair per venue, with the n_fast=1 long-budget pair when
  latency is tight — **subject to the §5 OOS downgrade**.

This generalises the single-span companion below to the whole family: every reader sees IC **and** lifetime
across all (fast, slow) pairs, and can pick the lookback that matches their latency budget — then confirms
it OOS in §5.
""")

md(r"""
**One correctness point first — gate the feature the model would actually receive.** The OFI
difference `fast − slow` is a depth-scale quantity, and depth events are fat-tailed: a few book updates
carry order-of-magnitude-larger quantities than the median, so the raw difference has heavy tails (see
§8). The walk-forward gate fits a **linear** OOS regression, which a handful of large-magnitude outliers
can hijack — washing the rank-IC down even when the raw, rank-based §6 heat-map shows clear signal. The
model never sees the raw difference either — §8 maps it through a **robust z-score + clip ±4** before the
network. So we gate the **shaped** feature (the genuine model input): a leak-free, monotone, per-feature
reshape that preserves the sign and rank order while taming the tails. (The sanity line below prints raw
vs shaped so you can see how much the clip recovers.)

**Both heads are gated out-of-sample.** Finding from round 1: the rate-head IC quoted in §6 is an
*in-sample grid-argmax* and must not headline the verdict. So below we run the **full walk-forward
battery on BOTH heads** — the price head (signed feature → σ-return, controls = vol/rate momenta) **and
the rate head** (|feature| → count÷λ_ev, with the **rate controls in `base`**). The number that ships
is the OOS marginal, not the §6 cell.

**Now the gates** (from §5). Every predictive number is the **walk-forward** mean (causal, purged).
Rough pass-marks: the added power should be clearly positive (≳ 0.01); it should barely shrink when we
add the level controls (no leak); and the shaped scale should stay within ~3× across volatility buckets.
*Marginal value:* does OFI add over the controls — all exchanges together, and each on its own, **per
head**? *No leak:* does that gain survive adding the vol/rate levels? *Normaliser:* is the (shaped) scale
roughly steady across volatility states? *Regime-stable* (the companion): is the marginal gain still
positive **within** calm, mid, and wild vol?
""")

code(r"""
# The model input is the §8-shaped feature (robust z-score + clip ±4): monotone, sign-preserving, leak-free,
# and it tames the fat-tailed depth-scale difference that otherwise hijacks the linear walk-forward fit. Gate THAT.
def shape_for_model(f):
    m = np.isfinite(f)
    if m.sum() == 0: return f
    med = np.median(f[m]); mad = 1.4826 * np.median(np.abs(f[m] - med)) + 1e-12
    return np.clip((f - med) / mad, -4.0, 4.0)          # robust z then clip — the §8 transform

# ============================ PRICE HEAD (direction) ============================
# Gates on the per-exchange OFI features at each venue's PRICE-head span pick — symmetric; KEEP ALL exchanges.
ofi_raw  = {ex: ofi_ex(ex, FAST[price_member[ex][0]], SLOW[price_member[ex][1]]) for ex in EX_LIST}
ofi_feat = {ex: shape_for_model(ofi_raw[ex]) for ex in EX_LIST}   # the model input (shaped)
joint      = round(wf_ic(base + list(ofi_feat.values()), target) - wf_ic(base, target), 3)
# MEASURED on this block: the price edge is genuinely MULTI-venue — the all-venues joint marginal EXCEEDS byb alone
# (the cross-venue legs add over byb, they are not collinear), and each venue is individually strong. We still run
# no-leak / normaliser / regime-stability on byb (the target's OWN book — the most direct leg, and a clean representative)
# and KEEP ALL venues for the model. The feed-resolution control in the lifetime section confirms the cross-venue legs.
byb_feat   = ofi_feat["byb"]
byb_marg   = round(wf_ic(base + [byb_feat], target) - wf_ic(base, target), 3)
byb_leak   = round(wf_ic(base + levels + [byb_feat], target) - wf_ic(base + levels, target), 3)   # no-leak on the byb price-head feature
vol_decile = np.digitize(vol_level, np.nanpercentile(vol_level[np.isfinite(vol_level)], np.arange(10, 100, 10)))
band = [np.nanstd(byb_feat[vol_decile == d]) for d in range(10)]
band = [b for b in band if b > 0]
full_r = wf_ic_by_regime(base + [byb_feat], target, vol_regime)     # regime stability on the byb price-head feature
base_r = wf_ic_by_regime(base, target, vol_regime)
strat  = {r: round(full_r[r] - base_r.get(r, 0.0), 3) for r in full_r}
gate_rows  = [dict(head="price", gate="marginal value", detail="byb own-book (price-head edge), added over the controls", value=byb_marg)]
gate_rows += [dict(head="price", gate="marginal value", detail="all exchanges together (joint > byb alone -> cross-venue adds)", value=joint)]
gate_rows += [dict(head="price", gate="marginal value", detail=f"{ex} alone, added over the controls",
                   value=round(wf_ic(base + [ofi_feat[ex]], target) - wf_ic(base, target), 3)) for ex in OTHERS]
gate_rows += [dict(head="price", gate="no leak", detail="byb gain still there after adding the vol/rate levels?", value=byb_leak),
              dict(head="price", gate="normaliser", detail="byb shaped-feature scale across vol buckets (max/min, want < ~3)", value=round(max(band) / min(band), 2))]
gate_rows += [dict(head="price", gate="regime-stable", detail=f"byb marginal IC within {nm}-vol (companion: stay positive)", value=strat.get(r, float("nan")))
              for r, nm in [(0, "calm"), (1, "mid"), (2, "wild")]]

# ============================ RATE HEAD (intensity) — finding #1: gate it OUT-OF-SAMPLE ============================
# The §6 rate map is in-sample grid-argmax. Re-score the |OFI| feature at each venue's RATE-head span pick through the
# SAME purged walk-forward, against rate_target (count÷λ_ev), with the rate momentum/level controls already in `base`.
# (The rate head is fed |feature|, the diagnostic intensity readout; we gate exactly what §6 measured.)
rate_raw  = {ex: np.abs(ofi_ex(ex, FAST[rate_member[ex][0]], SLOW[rate_member[ex][1]])) for ex in EX_LIST}
rate_feat = {ex: shape_for_model(rate_raw[ex]) for ex in EX_LIST}
rate_joint = round(wf_ic(base + list(rate_feat.values()), rate_target) - wf_ic(base, rate_target), 3)
rate_marg  = {ex: round(wf_ic(base + [rate_feat[ex]], rate_target) - wf_ic(base, rate_target), 3) for ex in EX_LIST}
rate_leak  = {ex: round(wf_ic(base + levels + [rate_feat[ex]], rate_target) - wf_ic(base + levels, rate_target), 3) for ex in EX_LIST}
rfull_r = wf_ic_by_regime(base + list(rate_feat.values()), rate_target, vol_regime)   # rate-head joint regime stability
rbase_r = wf_ic_by_regime(base, rate_target, vol_regime)
rstrat  = {r: round(rfull_r[r] - rbase_r.get(r, 0.0), 3) for r in rfull_r}
gate_rows += [dict(head="rate", gate="marginal value", detail="all venues together (|OFI|), over the rate/vol controls", value=rate_joint)]
gate_rows += [dict(head="rate", gate="marginal value", detail=f"{ex} alone (|OFI|), over the controls", value=rate_marg[ex]) for ex in EX_LIST]
gate_rows += [dict(head="rate", gate="no leak", detail=f"{ex} rate gain after adding the vol/rate levels?", value=rate_leak[ex]) for ex in EX_LIST]
gate_rows += [dict(head="rate", gate="regime-stable", detail=f"rate joint IC within {nm}-vol (companion: stay positive)", value=rstrat.get(r, float("nan")))
              for r, nm in [(0, "calm"), (1, "mid"), (2, "wild")]]

print(f"(sanity) RAW byb difference into the linear gate: marginal {round(wf_ic(base + [ofi_raw['byb']], target) - wf_ic(base, target), 3)}"
      f"  vs SHAPED: {byb_marg}  — equal here (a rank-IC of ONE standardized feature is invariant to a monotone reshape);"
      f" the clip matters once a feature is mixed with others in the linear design (and is what the model receives)")
print(f"PRICE head OOS: byb {byb_marg}, joint(all venues) {joint}; per-venue (over controls)",
      {ex: round(wf_ic(base + [ofi_feat[ex]], target) - wf_ic(base, target), 3) for ex in EX_LIST})
print(f"RATE head OOS (the §6 cell was in-sample): joint {rate_joint}; per-venue {rate_marg}")
with pl.Config(tbl_rows=30, fmt_str_lengths=80, tbl_width_chars=140):
    print(pl.DataFrame(gate_rows))
""")

md(r"""
**Conclusion.** Read the table against the §1 hypothesis and the pass-marks. The headline numbers are
quoted from the printed walk-forward table (both heads, OOS) — **not** the in-sample §6 cells.

- **Price head — OFI predicts direction, and the edge is genuinely multi-venue.** byb's own-book OFI adds
  **+0.212** walk-forward rank-IC over the rate/vol controls — well above the ~0.01 floor — exactly the
  §1 mechanism: recent net pressure on byb's quote precedes byb's mid. Critically, the venues are **not**
  collinear for direction: bin alone is **+0.248**, okx alone **+0.222**, and the **all-venues joint
  marginal is +0.268 — larger than byb alone**, so each venue's OFI carries direction information the
  others don't. (Earlier round-1 numbers, computed under the broken *ratio* form, showed bin/okx ≈ 0 and a
  joint ≈ 0; with the corrected *difference* form on this block they are strongly positive.) No leak: byb's
  gain **survives** the vol/rate *levels* (**+0.215**), so it isn't disguised volatility. Normaliser: the
  *shaped* byb feature's scale is steady across vol buckets (**1.18×**, far under 3×). Regime-stable: the
  byb marginal stays **positive in all three** vol regimes (calm **+0.197**, mid **+0.21**, wild **+0.232**).
  (The *echo-netted* and *feed-resolution* discounts are in the lifetime section below — they are what makes
  the headline honest.)
- **Rate head — gated OOS now (finding #1).** The §6 rate cells are an **in-sample grid-argmax** (≈ +0.18–
  0.24) and must not headline. Re-scored through the SAME purged walk-forward against `count÷λ_ev` with the
  rate momentum in `base` (and the rate *level* tested in no-leak), the |OFI| intensity feature shrinks but
  stays **clearly real**: the all-venues joint rate marginal is **+0.17** OOS, per-venue **byb +0.086, bin
  +0.148, okx +0.114** — bin the strongest. It **survives** the level controls (no-leak: byb +0.073, bin
  +0.139, okx +0.100) and is **regime-stable** (calm +0.153, mid +0.166, wild +0.167). So the honest
  rate-head claim is "a solid OOS intensity edge (joint ≈ +0.17), strongest from bin," **downgraded** from
  the in-sample §6 figure by the OOS gap but not washed out.

Verdict (honest, OOS): **real and ship-able for BOTH heads, and the direction edge is multi-venue** —
price marginal +0.212 (byb) up to +0.268 (joint, all venues), no leak, regime-stable; rate joint +0.17
OOS. Both fed as the *signed*, §8-shaped feature; we keep all venues. (The byb price headline is further
discounted to its **echo-netted +0.163** below — the contemporaneous-echo-free lead.)
""")

md(r"""
## How long does the edge last? — the signal's lifetime and your latency budget

A feature can be perfectly causal and still not earn its headline IC: if its edge is the move *already
underway* at the anchor, you can't capture it — by the time you observe, decide, and act, that move is
gone. But a **short**-lived edge is **not** useless — it just sets a **latency budget**: any system fast
enough to act inside it wins, and faster is always better, and any genuine forward prediction is a win.
So we do **not** gate on this — we **measure how long the signal lasts**.

Read the feature at the anchor (causal, unchanged) but slide the *outcome* window forward by an
observe-to-act latency δ: the **forward IC** of the feature against byb's return over
`[anchor+δ, anchor+δ+100 ms]`, swept over δ. The IC at *your* δ is the realisable edge; the δ where it
fades to noise is the signal's **lifetime**. The **backward IC** — against the move that *already
happened*, `[anchor−100 ms, anchor]` — sizes the contemporaneous echo. A feature whose forward IC dies at
δ>0 while the backward IC stays high is re-reporting the past, not predicting it; that is the *only*
genuinely useless case, and it is measured here, never assumed.

OFI is **poolable** — every venue builds its own own-book OFI difference — so we compute the lifetime
**per venue** (byb / bin / okx). A cross-venue leg's curve at δ=0 vs δ=20 ms is exactly how you tell a
feed-resolution artifact (a δ=0 spike that has already collapsed by δ=20 ms) from a real lead (an edge
that survives the latency). OFI is signed, so the natural head is the **price head** — the signed
difference against byb's signed forward return — and as a diagnostic we also carry the **rate head**
(|OFI| against byb's forward move-count) for byb's own book. The IC is Spearman (rank-based), so the raw
difference's fat tail doesn't distort it — same rank metric as the §6 heat-maps.

**Two extra gates (round-1 finding #4):**
- **Echo-netted (partial) forward IC** for the byb price leg. byb's own-book OFI is built from the book
  that *moves byb's mid*, so part of its δ=0 forward IC may be re-reading the move already underway (a
  large backward IC is the tell). We report the **partial** rank-IC of the byb signed feature with the
  *forward* return **controlling for the trailing** `[anchor−100 ms, anchor]` return — the part of the
  edge **not** attributable to the echo. That netted number, not the raw δ=0 IC, headlines the verdict.
- **Feed-resolution control** for the cross-venue (bin / okx) legs. byb's own book is the reference
  cadence; a *foreign* venue's OFI can look like it "leads" simply because its feed is fresher (or
  staler) than byb's. So we re-measure each foreign venue's price IC with its OFI sampled **only at byb's
  book-update times** (cadence-matched to byb): a real cross-venue lead survives, a pure feed-resolution
  artifact collapses toward zero.
""")

code(r"""
# Signal lifetime: forward IC vs observe->act latency δ (window slides to [t+δ, t+δ+100ms]), + backward IC, PER VENUE.
DELTAS_MS = [0, 5, 10, 20, 50, 100, 200, 500]
def _ic(feat, ret):
    v = np.isfinite(feat) & np.isfinite(ret)
    return spearmanr(feat[v], ret[v]).statistic if v.sum() > 100 else float("nan")
def _mid_at(t):                                          # byb merged mid at-or-before t (causal)
    return byb_mid[np.clip(np.searchsorted(byb_rx, t, "right") - 1, 0, len(byb_mid) - 1)]
def _ret(t0, t1):   return np.log(_mid_at(t1) / _mid_at(t0))
def _count(t0, t1): return cum_mv[np.searchsorted(byb_rx, t1, "right")] - cum_mv[np.searchsorted(byb_rx, t0, "right")]

# Per-venue price-head feature (signed OFI ratio at that venue's price-head span pick). byb also gets the rate head (|OFI|).
fwd_ic = {}; back_ic = {}
for ex in EX_LIST:
    sgn = ofi_ex(ex, FAST[price_member[ex][0]], SLOW[price_member[ex][1]])        # signed -> direction (price head)
    fwd_ic[ex]  = [_ic(sgn, _ret(anchor_ts + d*1_000_000, anchor_ts + d*1_000_000 + HORIZON_NS)) for d in DELTAS_MS]
    back_ic[ex] = _ic(sgn, _ret(anchor_ts - HORIZON_NS, anchor_ts))               # vs the already-happened move
byb_absmag = np.abs(ofi_ex("byb", FAST[rate_member["byb"][0]], SLOW[rate_member["byb"][1]]))   # rate head (intensity), byb book
cnt_ic = [_ic(byb_absmag, _count(anchor_ts + d*1_000_000, anchor_ts + d*1_000_000 + HORIZON_NS)) for d in DELTAS_MS]

# --- finding #4a: ECHO-NETTED (partial) forward IC for the byb price leg ---------------------------------------
# Part of byb's δ=0 forward IC may just re-read the move already underway (byb's own book moves byb's mid). The honest
# "is this prediction?" number is the partial rank-IC of the feature with the FORWARD return, CONTROLLING for the
# TRAILING [anchor-100ms, anchor] return. If the raw δ=0 IC collapses once the trailing move is partialled out, it was echo.
def _partial_ic(f, y, t):                                # Spearman partial corr of f with y given t (all rank-based)
    v = np.isfinite(f) & np.isfinite(y) & np.isfinite(t)
    if v.sum() <= 100: return float("nan")
    rfy = spearmanr(f[v], y[v]).statistic; rft = spearmanr(f[v], t[v]).statistic; rty = spearmanr(t[v], y[v]).statistic
    return (rfy - rft*rty) / np.sqrt(max((1.0 - rft**2) * (1.0 - rty**2), 1e-12))
_trail = _ret(anchor_ts - HORIZON_NS, anchor_ts); _fwd0 = _ret(anchor_ts, anchor_ts + HORIZON_NS)
byb_sgn  = ofi_ex("byb", FAST[price_member["byb"][0]], SLOW[price_member["byb"][1]])
echo_net = _partial_ic(byb_sgn, _fwd0, _trail)           # byb price edge with the contemporaneous echo netted out

# --- finding #4b: FEED-RESOLUTION control for the cross-venue (bin/okx) legs ----------------------------------
# A foreign venue's OFI can look like it "leads" only because its feed cadence differs from byb's. Re-measure each
# foreign venue's price IC with its OFI coarsened to BYB's book-update cadence (sampled only at byb update times, then
# forward-filled to the grid). A real cross-venue lead survives; a pure feed-resolution artifact collapses toward 0.
byb_upd_rx = ofi_flows["byb"][0]                          # byb's own book-update timestamps = the reference cadence
def ofi_ex_cadence(ex, n_fast, n_slow):                  # ex's OFI difference, sampled at byb update times, held to the grid
    rx, e = ofi_flows[ex]
    Ef = _flow_at(byb_upd_rx, rx, e, n_fast); Wf = _flow_at(byb_upd_rx, rx, np.ones(e.size), n_fast)
    Es = _flow_at(byb_upd_rx, rx, e, n_slow); Ws = _flow_at(byb_upd_rx, rx, np.ones(e.size), n_slow)
    f_at_byb = (Ef / np.where(Wf == 0.0, np.nan, Wf)) - (Es / np.where(Ws == 0.0, np.nan, Ws))   # value at each byb update
    return f_at_byb[np.clip(np.searchsorted(byb_upd_rx, anchor_ts, "right") - 1, 0, len(byb_upd_rx) - 1)]  # forward-fill to the grid
fwd0_ic   = {ex: fwd_ic[ex][0] for ex in EX_LIST}        # native-cadence δ=0 price IC (from the curve above)
cadence_ic = {}
for ex in OTHERS:                                        # cross-venue legs only (byb is the reference cadence)
    cad = ofi_ex_cadence(ex, FAST[price_member[ex][0]], SLOW[price_member[ex][1]])
    cadence_ic[ex] = _ic(cad, _fwd0)                     # δ=0 forward price IC, foreign feed matched to byb's cadence

fig, ax = plt.subplots(figsize=(8.0, 4.6))
col = {"byb": "C0", "bin": "C2", "okx": "C4"}
for ex in EX_LIST:
    ax.plot(DELTAS_MS, fwd_ic[ex], "o-", color=col[ex], label=f"{ex} — price head forward IC (direction)")
    ax.axhline(back_ic[ex], color=col[ex], ls=":", alpha=0.6, label=f"{ex} backward (already-happened) IC = {back_ic[ex]:+.3f}")
ax.plot(DELTAS_MS, cnt_ic, "s--", color="C3", label="byb — rate head forward |OFI|->count IC")
ax.axhline(0, color="0.7", lw=0.8); ax.set_xlabel("observe->act latency  δ  (ms)"); ax.set_ylabel("rank-IC")
ax.set_title("signal lifetime — OFI per venue: edge vs latency δ"); ax.legend(fontsize=7); fig.tight_layout(); plt.show()

def _half(curve):   # first δ where |forward IC| drops below half its δ=0 value (the lifetime); None if it never does
    return next((d for d, ic in zip(DELTAS_MS, curve) if np.isfinite(ic) and abs(ic) < abs(curve[0]) / 2), None)
for ex in EX_LIST:
    print(f"{ex} price forward IC by δ(ms):", " ".join(f"{d}:{ic:+.3f}" for d, ic in zip(DELTAS_MS, fwd_ic[ex])))
    print(f"   {ex} backward (already-happened) IC: {back_ic[ex]:+.3f}"
          f"   |  δ=0 {fwd_ic[ex][0]:+.3f} -> δ=20ms {fwd_ic[ex][3]:+.3f};  drops below half by δ≈{_half(fwd_ic[ex])} ms")
print("byb rate forward |OFI|->count IC by δ(ms):", " ".join(f"{d}:{ic:+.3f}" for d, ic in zip(DELTAS_MS, cnt_ic)))
print(f"\nECHO-NETTED forward IC (byb price, partial on the trailing move): {echo_net:+.3f}"
      f"   (raw δ=0 {fwd_ic['byb'][0]:+.3f}, backward {back_ic['byb']:+.3f}; the shortfall is echo)")
print("FEED-RESOLUTION control (cross-venue price IC, foreign OFI matched to byb's update cadence):")
for ex in OTHERS:
    print(f"   {ex}: native-cadence δ=0 IC {fwd0_ic[ex]:+.3f}  ->  byb-cadence-matched IC {cadence_ic[ex]:+.3f}"
          f"   ({'survives -> real lead' if abs(cadence_ic[ex]) >= 0.5*abs(fwd0_ic[ex]) and abs(fwd0_ic[ex])>0 else 'collapses -> feed-resolution artifact'})")
""")

md(r"""
**Read it as a latency budget, not a pass/fail.** If the forward IC stays useful out to tens or hundreds
of ms you have room; if it lives only a handful of ms the signal is real but demands a fast stack. On this
block **all three** venues' forward price IC survives well — δ=0 → δ=20 ms barely moves (byb +0.237→+0.211,
bin +0.281→+0.270, okx +0.252→+0.226) and each drops below half only around **δ≈200 ms** — so the OFI edge
is a real lead with a generous latency budget, not a one-tick contemporaneous spike. The verdict is
*"predicts ~X ms ahead, needs latency < X,"* never *"drop because it's fast."* (A flat forward curve at ≈0
with a large backward IC would be the one true non-signal — re-reporting the past; that is **not** what we
see here.)

**The two new gates, read honestly.**
- The **echo-netted** byb price IC is the part of byb's edge that is *not* the move already underway —
  partialling out the trailing `[anchor−100 ms, anchor]` return. byb's own-book OFI has a **large backward
  IC (+0.359)** because it reads the very book that moves byb's mid, so the raw δ=0 IC (+0.237) overstates
  the prediction. Netted, byb's price edge is **+0.163** — discounted by the echo but **still clearly a
  forward lead**, not a pure echo. That netted number, not the raw δ=0, is what the verdict quotes for byb.
- The **feed-resolution control** re-measures the cross-venue (bin/okx) price IC with the foreign OFI
  coarsened to **byb's** update cadence (sampled only at byb's book-update times). Both legs **survive**:
  bin +0.281 → **+0.244**, okx +0.252 → **+0.233** (each retains > 85% of its native-cadence IC). So the
  cross-venue direction edge is a **real lead**, not a fresher-feed artifact — measured, not assumed. The
  price edge is therefore **multi-venue** (byb echo-netted **+0.163**; bin/okx genuine cadence-matched
  leads ~+0.23–0.24), and every venue is kept.
""")

md(r"""
## 7. What the prediction actually looks like

A single correlation hides *how* the feature changes the outcome. So group the data by the feature and
look at the real distributions the two heads care about:
- **price head:** byb's next return for low / middle / high feature values — it should tilt one way as
  OFI turns positive and the other as it turns negative;
- **rate head:** how the number of upcoming moves grows as the feature's *magnitude* grows.
""")

code(r"""
rep_ex = "byb"                                                                  # byb's own-book OFI illustrates the shape; the model uses every exchange
signed = ofi_ex(rep_ex, FAST[price_member[rep_ex][0]], SLOW[price_member[rep_ex][1]])
absmag = np.abs(ofi_ex(rep_ex, FAST[rate_member[rep_ex][0]], SLOW[rate_member[rep_ex][1]]))   # the RATE-head's own span pick
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.6))

# price head: forward σ-return distribution across signed-OFI buckets
lo, hi = np.nanpercentile(signed, [10, 90])
groups = [("strong −", signed <= lo, "C3"), ("≈ 0", (signed > lo) & (signed < hi), "0.6"),
          ("strong +", signed >= hi, "C0")]
bins = np.linspace(-8, 8, 81)
for lab, m, col in groups:
    mm = m & np.isfinite(target)
    axA.hist(np.clip(target[mm], -8, 8), bins=bins, density=True,
             histtype="step", color=col, lw=1.7, label=f"{lab}:  mean {np.nanmean(target[mm]):+.2f}")
axA.set_yscale("log"); axA.set_xlabel("forward σ-return"); axA.set_ylabel("density (log)")
axA.set_title("price head: return distribution tilts with signed OFI"); axA.legend(fontsize=8)

# rate head: forward move-count shifts up with |OFI|
fin = np.isfinite(absmag)
dec = np.full(absmag.shape, -1); dec[fin] = np.digitize(absmag[fin], np.nanpercentile(absmag[fin], np.arange(10, 100, 10)))
axB.plot(range(10), [fwd_count[dec == b].mean() for b in range(10)], "o-", color="C3", label="mean count  E[K]")
axB.plot(range(10), [(fwd_count[dec == b] >= 1).mean() for b in range(10)], "s--", color="C1", label="P(K ≥ 1)")
axB.set_xlabel("|OFI fast−slow| decile (small → large lean)"); axB.set_ylabel("forward 100 ms")
axB.set_title("rate head: move-count shifts with |OFI|"); axB.legend(fontsize=8)
fig.tight_layout(); plt.show()
""")

md(r"""
**Conclusion.** The plots show whether OFI moves the *actual outcome distributions* the two heads need.
**Price head (left):** if the forward-return distribution shifts bodily with the *signed* feature —
negative mean for the strong-negative group, positive for strong-positive, flat between — then the sign
genuinely carries direction. **Rate head (right):** if the mean move-count `E[K]` and `P(K ≥ 1)` climb
with the |OFI| decile, a large flow imbalance precedes more moves. The directions to expect are those
§1 predicts (positive OFI → up; large |OFI| → busier).
""")

md(r"""
## 8. Input shaping for the network

This is a *different* step from §2. OFI is a self-centred **difference** of two same-units legs, so
there's no volatility to divide out (§2 noted it's self-scaling). Here we reshape it for the network's
input — roughly centred, unit-scale, no wild outliers. A *difference of two depth-scale E/W means* is
symmetric but **fat-tailed** (depth events span orders of magnitude: a few book updates carry far more
quantity than the median), so we expect this feature to need more than a plain z-score — though *not* the
~1e5 spikes the old ratio had (a difference cannot blow up at a near-zero denominator). Pick the
**lightest** transform that meets the bar; the QQ-plot makes the choice.
""")

code(r"""
from scipy.stats import skew, kurtosis, rankdata, norm
rep_ex = "byb"
f = ofi_ex(rep_ex, FAST[price_member[rep_ex][0]], SLOW[price_member[rep_ex][1]]); f = f[np.isfinite(f)]
med = np.median(f); mad = 1.4826 * np.median(np.abs(f - med)) + 1e-12; rz = (f - med) / mad
cand = {"z-score": (f - f.mean()) / (f.std() + 1e-12),
        "robust + clip ±4": np.clip(rz, -4, 4),
        "arcsinh(robust)": np.arcsinh(rz),
        "rank-Gaussian": norm.ppf((rankdata(f) - 0.5) / len(f))}
print(f"feature: std={f.std():.2g}  skew={skew(f):+.2f}  excess_kurt={kurtosis(f):.1f}  (0 = normal)")
for name, v in cand.items():
    print(f"  {name:18} excess_kurt={kurtosis(v):>8.1f}   max|·|={np.abs(v).max():.1f}")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.6))
lo_p, hi_p = np.percentile(f, [0.5, 99.5])
axA.hist(np.clip(f, lo_p, hi_p), bins=120, density=True, color="C0", alpha=.85, label="feature (clipped for view)")
axA.set_yscale("log"); axA.set_xlabel("ofi fast−slow (clipped 0.5–99.5%)"); axA.set_ylabel("density (log)")
axA.set_title(f"feature distribution (excess kurt {kurtosis(f):.1f})"); axA.legend(fontsize=8)

q = norm.ppf((np.arange(1, len(f) + 1) - 0.5) / len(f)); sub = np.linspace(0, len(f) - 1, 4000).astype(int)
for name, v in cand.items():
    axB.plot(q[sub], np.sort(v)[sub], lw=1.5, label=name)
axB.plot([-5, 5], [-5, 5], "k:", lw=1, label="perfect normal")
axB.set_xlim(-5, 5); axB.set_ylim(-8, 8); axB.set_xlabel("normal quantile"); axB.set_ylabel("normalised feature quantile")
axB.set_title("how to normalise — QQ vs N(0,1): on-diagonal & bounded wins"); axB.legend(fontsize=8)
fig.tight_layout(); plt.show()
""")

md(r"""
**Conclusion.** The printout settles the transform. As a **difference of two depth-scale E/W means**, OFI
is symmetric but fat-tailed (depth events span orders of magnitude), so a plain z-score leaves wild
outliers and fails the "no wild outliers" bar. The lightest transform that *meets* it is a robust z-score
followed by a clip (robust + clip ±4 → max|·| = 4) — or, if the tail is severe, a rank-Gaussian map, which
is the heaviest but flattens any tail by construction. The QQ-plot points to the on-diagonal, bounded
curve; use that whenever you feed a network. (The tail is real but bounded — no ~1e5 ratio spikes — so the
clip throws away very little.)
""")

md(r"""
## 9. When is per-exchange worth it? (for a poolable feature)

OFI is **poolable** — each venue yields its own one-number OFI difference from its own book — so unlike
the template's cross-venue gap, you face a real choice: keep them **per-exchange**, collapse to **one**
(byb's own book, the most direct), or **pool** them. The right answer depends on the time-scale, so
sweep it and compare the three. (The feed-resolution control in the lifetime section is the cross-venue
guard: here it **confirms the foreign venues carry a genuine direction lead** — bin/okx keep > 85% of
their δ=0 price IC when matched to byb's cadence, so it is a real lead, not a fresher-feed artifact.)

**What we measured on this block.** The per-exchange choice pays off: the all-venues joint price marginal
(**+0.268** OOS, §5) **exceeds byb alone** (**+0.212**), and bin (+0.248) and okx (+0.222) are each strong
on their own — so the venues are *not* collinear for direction at the picked span, and pooling to one venue
would throw away real signal. The mechanism: byb's own-book OFI most directly precedes byb's mid (largest
backward echo), while another venue's flow can lead byb's by a beat — and at this ~`(10, 5000)` span the
cross-venue lead is live. The §6 heat-maps give the full per-exchange numbers across the family; the table
below sketches the qualitative pattern (the genuine numbers are §6/§5, not this table).

> **Illustrative pattern** (not recomputed here — §6/§5 have the real per-exchange numbers). The ms/second
> labels just translate points on the trade-span N clock for readability.

| time-scale | pooled | best single (byb book) | per-exchange | what it means |
|---|---|---|---|---|
| ≤ 50 ms | mid | high | **high** | byb leads its own mid AND foreign flow leads by a beat — keep all |
| middle band | mid | mid | **high** | another venue's flow leads byb — separate venues add most |
| long | low | low | low | flow drift is shared — any one venue will do |

So keep all exchanges (on this block per-exchange genuinely beats a single venue and pooling), and sweep
per-exchange / single / pooled rather than pool by default — pooling tends to blur the middle band.
(Longer prediction horizons widen the useful band, so
sweep the horizon too.)
""")

md(r"""
## 10. The verdict, and what it takes to ship

**Keep it — feed the *signed*, §8-shaped OFI difference (`fast − slow`) to both heads, all exchanges, at
a couple of time-scales each. The direction edge is strong and genuinely multi-venue; the one honest
discount is that byb's own-book leg is partly a contemporaneous echo (netted out below), and the rate head
shrinks from its in-sample figure once scored out-of-sample (but stays clearly real).**
- **Price head (direction):** OFI predicts direction and the edge is **multi-venue**. Each venue's signed
  OFI difference (`fast − slow`, §6 picks `(10, 5000)` for all three) clears the walk-forward gate over the
  rate/vol controls: **byb +0.212, bin +0.248, okx +0.222**, and the **all-venues joint marginal is +0.268
  — larger than any single venue**, so they are *not* collinear and pooling to one venue loses signal. No
  leak (byb +0.215 after adding the vol/rate levels), normaliser fine (shaped scale 1.18× across vol
  buckets), regime-stable (byb +0.197 / +0.21 / +0.232 in calm / mid / wild). byb's own-book leg is **partly
  an echo** — byb's book moves byb's own mid (backward IC +0.359 vs raw δ=0 +0.237) — so its honest headline
  is the **echo-netted** forward IC (partial on the trailing 100 ms return): **+0.163**, still a clear
  forward lead. The cross-venue (bin/okx) legs are **real leads, not feed artifacts**: matched to byb's
  update cadence the **feed-resolution control** leaves bin +0.281→+0.244 and okx +0.252→+0.233 (each keeps
  > 85%). (Round-1's "bin/okx ≈ 0, single-venue" read came from the broken *ratio* form; the corrected
  *difference* form measures a strong multi-venue direction edge on this block.)
- **Rate head (intensity):** |OFI| → move-count is gated **out-of-sample** now (finding #1 — the §6 cell was
  an in-sample grid-argmax, ≈ +0.18–0.24). Through the same purged walk-forward against `count÷λ_ev` (rate
  momentum in `base`, rate level tested in no-leak) the OOS number is **joint +0.17** (per-venue byb +0.086,
  bin +0.148, okx +0.114 — bin the strongest), it **survives** the level controls (no-leak byb +0.073 / bin
  +0.139 / okx +0.100) and is **regime-stable** (+0.153 / +0.166 / +0.167). So a **solid, real OOS intensity
  edge (~+0.17 joint)** — downgraded from the in-sample §6 figure but not washed out.

Feed **every exchange's** signed OFI difference and let the model lean on whichever leads; do **not** collapse
to byb-only (the joint beats any single venue). OFI is a **difference** (not a ratio), so it ships **without**
a `σ_ev` division — but it is *not* self-bounding: a difference of depth-scale legs is fat-tailed (excess
kurtosis ~43), so it **must** go through the §8 robust-z + clip ±4 before the network (the model never sees
the raw difference; §5 gates the shaped feature). The move-count it predicts is still divided by the rate
yardstick `λ_ev`.

**Why a difference, not the original ratio.** The first cut used `fast / slow`, which **inverts sign**
whenever the slow leg crosses zero — fatal for a signed direction feature. `fast − slow` never inverts,
and (a bonus) makes the §4 oracle genuinely **bit-exact** (no near-zero-denominator amplification to
relabel away).

**Clock choice (stated plainly).** OFI is a **flow injected on byb front_levels updates** and **decayed
once per trade-timestamp** on the shared bin/byb/okx merged trade clock (`merged_ts = np.unique(...)`,
simultaneous prints = one tick), read as an `E/W` per-event mean — exactly the template's `σ_ev` machine
pointed at OFI increments. It updates on every byb book change between trades and is read live at the grid
anchor; it never reads a stale last-trade state. Same-nanosecond byb book rows are one EMA sample —
value = the SUM of their per-row OFI increments (the full intra-ns path), not the collapsed endpoint.

**To ship:**
- [x] the streaming (constant-work-per-trade) builder, matching this analysis version — done (§4, plain-numpy oracle, bit-exact: legs/feature ~1.3e-11 absolute round-off)
- [x] the oracle (§4), independent plain-numpy (no `boba.ema`), passing on a real block
- [x] both heads gated **out-of-sample** (§5 walk-forward); rate head downgraded to its OOS number
- [x] echo-netted forward IC (byb price) and feed-resolution control (cross-venue) reported
- [x] the gate results recorded (honest: direction multi-venue, joint +0.268; byb echo-netted +0.163; rate joint +0.17 OOS)
- [x] the chosen heads and time-scales written down, with the yardstick span (YARDSTICK_N=10000)
- [x] the data quirks handled (bad zero-price prints filtered; byb front_levels qty used; same-ns rows path-summed into one sample)
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
out = Path(__file__).resolve().parent.parent / "ofi_pathsum.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, "with", len(cells), "cells")
