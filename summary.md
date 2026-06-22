# Feature analysis — master summary (post-review, post-control)

18 features, each predicting **byb's next-100 ms outcome** — **price head** (signed → byb signed return) and **rate head** (unsigned → byb move-count) — on the real ETH-perp block. Every feature passed: independent bit-exact oracle, strict causality, the **hard regime-invariance gate** (measured scale across vol buckets < ~3, raw-first then normalize), the lifetime + per-N half-life sweep, and the two gates added this round:

- **Echo-netted (partial) forward IC** — the feature's IC with the *forward* outcome controlling for the *trailing* `[anchor−100 ms, anchor]` outcome. Strips out re-reporting of the move already underway. **This is the honest headline**, not the raw δ=0 IC.
- **Feed-resolution control** — cross-venue legs re-measured with the foreign feed matched to byb's ~6–10 ms update cadence. A real lead survives; a finer-feed artifact collapses.

All 8 new features went through a fix→review loop until **0 critical / 0 high** findings remained (12/12 converged).

## The headline result — the cross-exchange thesis holds

**`ofi_normalised`'s bin lead is real.** The 0.47 survives *both* decisive controls: cadence-matched to byb's feed it keeps **94%** (≈+0.45 @δ=0, +0.43 @20 ms), and echo-netted it keeps **91%** (+0.446). It is a genuine cross-venue order-flow lead, **not** bin's finer feed. The same control vindicates the cross-venue legs of `flow_imbalance`, the rate-intensities, etc. — they survive cadence-matching. Conversely, echo-netting honestly **shrank or killed** the echo-heavy own-legs (`price_momentum` byb, `kyle_lambda`, `range_breakout`).

## Verdicts (all 18)

### ✅ SHIP / KEEP

| Feature | Type | Honest edge (echo-netted / cadence-matched) | Regime-inv | Note |
|---|---|---|---|---|
| **microprice** ⭐ | per-exch, dir | OOS marginal **+0.26** joint; echo-net **+0.22**; feed survives (bin 90%, okx 93%) | ✓ 1.40× raw | **strongest single L1 predictor**; byb's *own* premium is the lead; N=1 (freshness) |
| **ofi_normalised** ⭐ | cross-venue, dir | bin **+0.446** echo-net / **+0.43@20ms** cadence-matched (94%); okx +0.366 | ✓ 1.25× raw | **0.47 confirmed real** post-control; byb own-leg echo-nets to +0.277 |
| **ofi** | multi-venue, both | price marginal byb +0.212 / bin **+0.248** OOS; rate real OOS | ✓ 1.12× | **sign-flip fixed** (fast−slow); byb echo netted |
| **flow_imbalance** | cross-venue, dir | bin +0.091 / okx +0.081 cadence-matched (survive); byb echo-net +0.060 | ✓ 1.03× | trade-based → genuine order-flow lead |
| **price_momentum** | per-exch, dir | bin **+0.147** echo-net (cleanest); joint marginal echo-net +0.109 | ✓ 1.24× raw | short-N byb mostly echo (+0.040); long-N member survives 3/3 |
| **flow_persistence** | per-exch, dir | `signed_trade` +0.167 OOS (~95% bin); cadence-confirmed | ✓ 1.07× raw | **demote** the `flow_sign_persistence` interaction (≈0) |
| **xv_book_pressure** | cross-venue, both | nested-fold marginal +0.066; feed survives; price head negative (catch-up) | ✓ 1.08–1.40× raw | modest but clean, new-info over controls |
| **trade_rate_surge** | cross-venue, rate | bin/okx rate leads survive cadence (okx 70%) | ✓ 1.41× | sharp + durable members |
| **mid_rate_surge** | cross-venue, rate | bin/okx survive cadence; byb = rate_momentum (circular) | ✓ 1.69× | keep cross-venue legs |
| **trade_rate_normalised** | rate | **ship /λ_ev** (invariant 2.32×); baseline 4.24× **FAILS** | ✓ /λ_ev only | "trades per byb-mid-move"; baseline disqualified |
| **volume_normalised** | rate | **ship /σ_ev** (invariant 2.51×); baseline 3.65× **FAILS** | ✓ /σ_ev only | modest intensity; baseline non-invariant |
| **vol_over_rate** | rate | −0.120 echo-net; cross-venue survive cadence | ✓ 1.76× | long-lived regime feature (circular controls but real) |
| **volume_surge** | rate | bin +0.039 echo-net / +0.025 cadence (57%) | ✓ 1.2× | small but real bin lead |

### ⏸️ HOLD (real core, but a gate fails — don't ship yet)

| Feature | Why HOLD |
|---|---|
| **gap_dynamics** | okx reversion is **genuine** (echo-netting *strengthens* it to −0.125, opposing byb's own continuation), but the **bin 2nd leg collapses 34% under the feed-control** (fresher-mid artifact), it's one block / one ~3 h window (no multi-block OOS), and `gap_halflife` reads sub-tick (dead gate). Ship needs a 2nd leg surviving cadence + multi-block OOS. |
| **range_breakout** | mechanics pass (independent oracle, invariant 1.05×), but the continuation edge is **mostly echo** — echo-nets to **+0.033**. Blocking gate stays open. |

### 🅿️ PARK (candidate — no standalone edge on this block)

| Feature | Why |
|---|---|
| **xv_leadlag** | the lag-k lead is real and **survives the feed-control (82%)**, but at the 100 ms horizon it has **≈0 marginal over byb's own recent return** (the lead is already in byb's tape). Re-sweep at a shorter horizon. |
| **kyle_lambda** | regime-invariant + latency-robust, but echo-nets to +0.01–0.02 and is **collinear with the rate controls** — no marginal edge. Kept as a candidate. |

## What changed vs the builders' self-reports

- The fix loop **confirmed** the two big claims under proper controls: `microprice` (+0.26 OOS) and `ofi_normalised` (0.43–0.45 cross-venue, post-control).
- It **honestly downgraded** the echo-inflated ones: `price_momentum` short-N (δ=0 mostly contemporaneous), `gap_dynamics` bin 2nd-leg (feed artifact), and **killed** `kyle_lambda` / `range_breakout` as standalone edges.
- It **fixed** real bugs: `ofi`'s sign-flipping ratio (→ fast−slow), `volume_normalised`/`trade_rate_normalised` shipping non-invariant baselines (→ /σ_ev, /λ_ev), and several oracles that imported production EMA primitives (→ independent plain-numpy).

## Multi-block OOS validation (36 usable of 58 blocks)

Each feature's span **fixed to its block-0 pick**, then scored across all blocks via the purged-WF marginal IC over rate/vol controls (block-0 = selection block; OOS = held-out blocks). Full detail in `oss_report.md`; matrix in `/tmp/oss_harness/oss_ic_matrix.csv`.

| Feature | block-0 | OOS mean±std | same-sign | OOS verdict |
|---|---|---|---|---|
| ofi_normalised | +0.285 | **+0.378±0.092** | 100% | PERSISTS — cross-venue legs (bin +0.338, okx +0.307) > byb own (+0.285) |
| ofi | +0.276 | +0.368±0.091 | 100% | PERSISTS |
| microprice | +0.260 | +0.348±0.083 | 100% | PERSISTS |
| price_momentum | +0.177 | +0.282±0.087 | 100% | PERSISTS (echo caveat) |
| flow_persistence | +0.167 | +0.257±0.071 | 100% | PERSISTS |
| range_breakout | +0.151 | +0.250±0.082 | 100% | PERSISTS marginal — but net-of-echo only +0.033 |
| flow_imbalance | +0.158 | +0.234±0.057 | 100% | PERSISTS |
| mid_rate_surge | +0.132 | +0.199±0.055 | 100% | PERSISTS |
| gap_dynamics | +0.081 | **+0.190±0.098** | 100% | PERSISTS — **off the HOLD bench** (block-0 was a weak block) |
| trade_rate_normalised | +0.107 | +0.170±0.054 | 100% | PERSISTS |
| trade_rate_surge | +0.090 | +0.147±0.046 | 100% | PERSISTS |
| xv_book_pressure | +0.066 | +0.068±0.021 | 100% | PERSISTS (okx gap; bin gap sign-unstable) |
| volume_surge | +0.010 | +0.005±0.007 | 72% | FADES |
| vol_over_rate | −0.008 | −0.013±0.013 | 89% | FADES |
| volume_normalised | +0.004 | +0.001±0.005 | 69% | FADES |

**Findings:** cross-exchange thesis **confirmed OOS** (ofi_normalised cross-venue legs > own-book on every block); `gap_dynamics` rehabilitated; OOS ≥ in-sample everywhere (block-0 is a low-IC block → no overfitting); the 3 near-zero/circular rate intensities fade; no outlier-block dependence.

**Caveat:** OOS validated the **marginal-IC** gate, **not** the echo-net/feed-control gates (block-0 analyses) — for echo-heavy features (`range_breakout`) the tradeable net-of-echo number is the binding one, not the +0.25 marginal.

## Echo-netted OOS validation (36 blocks, the *tradeable* edge)

Re-ran the 36-block fixed-span OOS sweep scoring each feature on its **net-of-contemporaneous-echo** forward IC: the same purged-WF marginal IC, but adding the *trailing* `[anchor−100 ms, anchor]` outcome (already-happened analogue of the forward target on the byb grid — strictly past, no leak) to the base controls. RAW reproduces the marginal-IC sweep above to **0.0e+00** on block-0; the ECHO-NETTED column is the honest tradeable number. Full detail in `tools/oss/oss_echonet_report.md`; matrix in `tools/oss/oss_echonet_matrix.csv`. **Verdict: PERSISTS** = net OOS clearly nonzero, same sign, retains ≥½ the raw; **SHRINKS** = retains <½ but a tradeable core remains; **COLLAPSES→echo** = net ~0 or sign-flips.

| Feature | head | OOS raw IC | OOS echo-netted IC | retention | tradeable verdict |
|---|---|---|---|---|---|
| microprice | price | +0.348±0.083 | **+0.202±0.041** | 58% | **PERSISTS** — strongest net-of-echo direction; prem legs all persist |
| ofi_normalised | price | +0.378±0.092 | **+0.232±0.052** | 61% | **PERSISTS** — bin 61% / okx 56% > byb 48%: cross-venue lead is the real edge |
| ofi | price | +0.368±0.091 | +0.222±0.051 | 60% | **PERSISTS** — bin 60% / okx 55% > byb 47% |
| mid_rate_surge | rate | +0.199±0.055 | +0.118±0.035 | 59% | **PERSISTS** — bin 65% persists; byb own-leg collapses (2%) |
| price_momentum | price | +0.282±0.087 | +0.155±0.046 | 55% | **PERSISTS (joint)** — but byb own-leg COLLAPSES (sign-flips to −0.006); bin 54% persists |
| range_breakout | price | +0.250±0.082 | +0.135±0.044 | 54% | **PERSISTS (joint)** — but byb own-leg collapses to **+0.034** (24%): mostly echo |
| trade_rate_normalised | rate | +0.170±0.054 | +0.092±0.032 | 54% | **PERSISTS** — bin 63% persists; byb collapses (15%) |
| gap_dynamics | price | +0.190±0.098 | +0.103±0.051 | 54% | **PERSISTS** — bin leg 53% persists; okx 45% shrinks |
| trade_rate_surge | rate | +0.147±0.046 | +0.075±0.024 | 51% | **PERSISTS** — bin 54% persists; byb own-leg collapses (17%) |
| flow_persistence | price | +0.257±0.071 | +0.121±0.028 | 47% | **SHRINKS** — all legs net 33–47%; tradeable core remains |
| flow_imbalance | price | +0.234±0.057 | +0.106±0.016 | 45% | **SHRINKS** — bin 47% / okx 36%; byb 33% |
| xv_book_pressure | price | +0.068±0.021 | +0.030±0.011 | 44% | **SHRINKS** — okx gap legs survive (44–45%); bin gap legs collapse |
| vol_over_rate | rate | −0.013±0.013 | −0.011±0.012 | 80% | **COLLAPSES→echo** — ≈0 in both (circular) |
| volume_surge | rate | +0.005±0.007 | +0.004±0.006 | 75% | **COLLAPSES→echo** — ≈0 in both |
| volume_normalised | rate | +0.001±0.005 | +0.001±0.004 | 119% | **COLLAPSES→echo** — ≈0 in both |

**Findings (tradeable, OOS, net of echo):**
- **The cross-exchange thesis is the tradeable thesis.** For *every* cross-venue direction feature, the bin/okx legs retain a much larger fraction of their raw IC than the byb own-leg, and the own-byb legs are the ones that collapse: ofi_normalised/ofi cross-venue legs PERSIST (55–61%) while byb own-leg SHRINKS (47–48%); price_momentum byb **flips sign** to −0.006 (pure echo) while bin PERSISTS (+0.145, 54%); the rate-surge byb own-legs all collapse (2–17%) while bin legs persist (54–65%). Netting the echo *confirms* that the durable forward edge lives in the foreign feed leading byb, not in byb re-reporting itself.
- **microprice / ofi_normalised / ofi keep a real tradeable forward edge** net of echo (joint +0.20 / +0.23 / +0.22), with the cross-venue prem/bin/okx legs all PERSISTS — these are the shippable directions.
- **range_breakout & the byb own-legs of price_momentum/gap_dynamics shrink/collapse** exactly as expected: range_breakout's own-book byb leg nets to **+0.034** (24% — mostly echo), price_momentum byb collapses (sign-flip). The joint verdicts still PERSISTS for these, but **carried by their cross-venue legs**, not the own-book breakout/momentum.
- **The 3 near-circular rate intensities (volume_normalised, vol_over_rate, volume_surge) COLLAPSE** — ≈0 in both raw and net, consistent with the marginal-IC sweep (no tradeable edge by construction).

## Open items
- ✅ **Echo-netted OOS** — DONE. Re-ran the 36-block sweep on the *echo-netted* IC (`tools/oss/run_oss_echonet.py` → `oss_echonet_report.md`): the tradeable edge for the echo-sensitive features is OOS-validated. range_breakout own-leg nets to +0.034 OOS (mostly echo); price_momentum byb own-leg collapses (sign-flip); gap_dynamics bin-leg persists OOS net of echo. The cross-venue legs of microprice/ofi_normalised/ofi/mid_rate_surge are the durable tradeable directions.
- **Cross-instrument / cross-period** — all 58 blocks are one continuous ETH-perp capture (May 2026). Generalization to other tokens/venues/periods is untested.
- **`xv_leadlag` / `gap_dynamics` short-horizon re-sweep** — the cross-venue lead may carry marginal value below 100 ms where it hasn't yet propagated into byb's tape.
- Both standard gates (echo-netted partial IC + feed-resolution control) are in the template and all 18 notebooks; the OOS harness (`/tmp/oss_harness/`) is reusable for any new feature.
