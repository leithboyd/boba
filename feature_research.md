This is a synthesis task — no code execution needed. I have all 33 proposals and the constraints. Let me produce the report directly.

# Microstructure Feature Research Report — byb ETH-perp next-100ms

**Scope reminder.** Target = byb's next-100ms signed return (price head) + mid-move count (rate head). Methodology constraints: strictly causal; trade-clock EMAs (α=2/(span+1)); EMA-span families; per-exchange; regime-invariant (vol-bucket scale < ~3); cross-venue legs read on a shared `@trades_byb` clock. Data = L1 front_levels (price+size), merged_levels (mid only), trade (rx/exch time, aggressor, prc, qty). No depth beyond L1, no funding, no spot.

User priorities: (1) cross-exchange lead/lag, (2) per-exchange breakout at short AND long.

---

## 1. Merges and drops (audit before the table)

**Merged clusters** (the 33 collapse to 18 distinct features):

- **Cross-venue lead-lag return correlation** — #4 (`xv_leadlag_corr`) and #11 (`gap_leadlag_xcov`) are the same online lagged cross-correlation ρ_k. Merged → **xv_leadlag_xcorr**. #11's explicit lag-k atom is the better construction; keep it.
- **Cross-venue book-pressure lead** — #24 (`queue_imbalance_lead`) is the cross-venue read of #22 (`queue_imbalance_signed`) and #23 (`microprice_premium`). #2 (`xv_micro_pressure`) is the price-space version of the same okx−byb microprice displacement. Merged → **xv_book_pressure_gap** (QI-gap + premium-gap legs), with the per-exchange QI/premium oscillators (#22/#23) kept as their own per-exchange entries.
- **Cross-venue gap level/derivatives** — #6, #7, #8, #9, #10 are all functions of the okx−byb microprice gap: #8 is the OU z-score (level), #7 its velocity (1st derivative), #10 its term-structure/acceleration, #9 its half-life (persistence), #6 the error-correction asymmetry (who reverts). These are genuinely orthogonal *transforms* of one atom, so I keep #7, #8, #9 and **fold #10 into #8** (it overlaps price_dislocation in raw form) and **fold #6's reversion-beta into #9** (both are "who-adjusts / persistence" gating). Merged gap family → **gap_zscore (OU), gap_velocity, gap_halflife**.
- **Cross-venue flow lead** — #1 (`xv_ofi_lead`), #3 (`xv_trade_lead_imbalance`) are leader-OFI and leader-aggressor-flow on byb clock. Distinct atoms (book-OFI vs trade-sign), both kept but grouped → **xv_ofi_lead**, **xv_flow_imbalance_lead**.
- **Cross-venue rate/breakout lead** — #5 (`xv_quote_arrival_race`) and #16 (`cross_venue_breakout_lead`) both compare a leader's per-exchange signal to byb's on the shared clock. Kept distinct (intensity-ratio vs breakout-event).
- **Momentum cluster** — #17 (`signed_return_momentum`) is the base atom; #19 (`momentum_spread`) and #21 (`micro_trend_drift`/efficiency-ratio) are downstream of it. Kept as three (one atom, two derived forms — both cheap and orthogonal: MACD vs efficiency ratio).
- **Sign-persistence cluster** — #18 (`trade_sign_autocorr`) and #30 (`aggressor_run_persistence`) are the same sign serial-cov + signed-trade EMA. Merged → **flow_sign_persistence**.
- **OFI momentum** — #20 is downstream of existing `ofi`/`ofi_sq` atoms + one small sign-cov atom; kept but flagged as low-novelty.
- **Range/breakout cluster** — #12 (`range_position`), #13 (`breakout_magnitude`), #15 (`new_extreme_run`) all share one new runmax/runmin atom pair. Kept as three (position / magnitude / frequency — genuinely different reads of the same extremes).

**Dropped:**
- **#10 gap_momentum** — author-flagged overlap with the already-built `price_dislocation` (fast−slow gap oscillator). Its only novelty (σ_gap normaliser, acceleration ladder) is folded into #8.
- **#14 vol_baseline_ratio** — overlaps already-built `vol_surge`/`mid_rate_surge` on the same channel. Only the *squeeze-flag* (sustained low-vol state + release) survives; demoted to a one-line note under book-shape, not a table row.
- **#27 book_depth_asymmetry_persistence** — author-flagged low; the imbalance-persistence half duplicates #22's normalisation, the depth-surge half is a thin volume_surge analogue. Note only.

No proposal needs missing data — **all 18 survivors are buildable from L1+trades.** Genuinely-unavailable channels (spot basis, funding, full-depth queue dynamics) are appendixed as *new ideas the user did not propose*, since nothing in the 33 actually required them.

---

## 2. Prioritised table

Priority key: **P1** = build first (high cross-exchange or breakout value, clearly invariant, tradable latency); **P2** = solid, build after; **P3** = speculative / low-novelty / weak edge.

### Theme A — Cross-exchange lead/lag (user priority #1)

| Feature | X/per | Horizon | Mechanism (1 line) | Construction (atom → invariant EMA family) | Data | Pri |
|---|---|---|---|---|---|---|
| **xv_leadlag_xcorr** (#4+#11) | cross | short→med | Lagged return cross-corr ρ_k identifies *who leads byb and by how many ms* (Hayashi-Yoshida / Hasbrouck). | New atom `ema_cross_leadret_cov_{k}_{N} = EMA(Δlogmp_src(t−k)·Δlogmp_byb(t))` on `@trades_byb`, k∈{1,5,20}ms; invariant ρ_k = γ_k / √(ret_sq_src·ret_sq_byb) ∈[−1,1]. src∈{okx,bin}. | L1 mid + byb clock | **P1** |
| **xv_book_pressure_gap** (#24+#2+#22x+#23x) | cross | both | Leader's queue-imbalance / micro-price premium skews *before* byb prints — earliest cross-venue channel (book moves before price). | `okx_byb_qi_gap = ema_book_imbalance_{N}t@trades_byb(okx) − (byb)`; premium-gap = same on (microprice−mid)/mid. Differences of bounded quantities → invariant. Short span = lead, long = structural bias. | L1 price+size all venues | **P1** |
| **xv_ofi_lead** (#1) | cross | short | Leader's L1 OFI is the dominant linear predictor of *its own* next mid (CKS); transmitted to byb via arbitrage re-quoting. | Reclock `ema_ofi_okx_{Nshort}t@trades_byb / √(ema_ofi_sq_okx_{Nlong}t@trades_byb)` (existing ofi_normalised, foreign venue, byb clock). | L1 price+size | **P1** |
| **xv_flow_imbalance_lead** (#3) | cross | short | Informed aggressor flow hits liquid venue first; bounded signed-flow imbalance on okx/bin leads byb direction (cross-venue VPIN). | `(ema_buy_val_okx − ema_sell_val_okx)/(ema_buy+ema_sell)` on `@trades_byb` ∈[−1,1]. | trades all venues | **P2** |
| **xv_breakout_lead** (#16) | cross | short | Leader pierces its *own* Donchian channel before byb → byb follows (price-discovery on discrete break events; sparser than the dislocation oscillator). | `brk_norm[okx]@trades_byb − brk_norm[byb]` using #13's σ_ev-normalised pierce on shared clock. | L1 mid + σ_ev | **P2** |
| **xv_rate_race** (#5) | cross | short | Venue firing faster right now is leading discovery; rate-ratio gates the rate head, leader's sign directs it. | `(dt_byb_{N}m / dt_okx_{N}m)@trades_byb` (ratio of move-rates, dimensionless) × sign(okx short return). | L1 mid + trades | **P3** |

### Theme B — Cross-venue relative-value (gap dynamics)

| Feature | X/per | Horizon | Mechanism | Construction | Data | Pri |
|---|---|---|---|---|---|---|
| **gap_zscore (OU)** (#8, +#10 folded) | cross | both | okx−byb log-mid gap is a mean-reverting cointegration residual; z tells if byb is the *dislocated* leg about to snap back. | z = (g − g_{Nlong})/σ_gap, σ_gap=√(ratio_sq_{Nlong}−g²); g=log(mp_byb/mp_consensus). Unitless by construction. Mostly downstream of existing ratio/ratio_sq atoms. | L1 mid | **P1** |
| **gap_velocity** (#7) | cross | short | 1st derivative of the gap: which way / how fast it is *opening now* — directional snap signal orthogonal to the gap level. | v = (g_fast(t)−g_fast(t−τ))/τ, τ∈{5,20,100}ms; invariant z = v·τ/σ_gap. Needs a short-span gap EMA atom. | L1 mid | **P2** |
| **gap_halflife** (#9, +#6 folded) | cross | gate | OU reversion speed κ / who-reverts — gates whether a gap means snap-back (fast) or lead-lag continuation (slow/near-unit-root). | New `ema_gap_serial_cov_{N} = EMA(Δg_t·Δg_{t−1})` / EMA(Δg²) → lag-1 autocorr → half-life. Mirror of Roll serial-cov machinery on gap increments. | L1 mid | **P2** |

### Theme C — Per-exchange breakout (user priority #2)

| Feature | X/per | Horizon | Mechanism | Construction | Data | Pri |
|---|---|---|---|---|---|---|
| **range_position** (#12) | both | both | Donchian / %K: band-edge = breakout (continuation), mid-band = noise. Bounded → intrinsically invariant. | New atoms `runmax_mp_{N}t`,`runmin_mp_{N}t` via leaky EWMA-extreme (α=2/(N+1)); pos=(mp−L)/(H−L)∈[0,1]. Cross leg on `@trades_byb`. | L1 mid | **P1** |
| **breakout_magnitude** (#13) | both | both | ATR-normalised pierce beyond prior extreme = momentum ignition; sign + size of a *fresh* new high. | From runmax/runmin: signed pierce / (σ_ev·mp). Short=fast breaks, long=regime breaks. | L1 mid + σ_ev | **P1** |
| **new_extreme_run** (#15) | both | both | Frequency of new-extreme prints = trend persistence (Donchian run-stats); staleness clock for consolidation. | EMA of new_high/new_low flag from runmax/runmin ∈[0,1]; signed = ema_new_high − ema_new_low. | L1 mid | **P2** |

### Theme D — Per-exchange momentum (directional)

| Feature | X/per | Horizon | Mechanism | Construction | Data | Pri |
|---|---|---|---|---|---|---|
| **signed_return_momentum** (#17) | both | both | Short-lag positive return autocorr (Lo-MacKinlay / informed drift) — the missing *signed* trade-clock momentum primitive. | New atom `ema_signed_logret_{N}t` = tick-EMA of per-interval Σ signed Δlog mp; invariant = /(σ_ev·k) (drift t-stat). | L1 mid + clock | **P1** |
| **trend_efficiency_ratio** (#21) | both | both | Net displacement / path length (Kaufman ER / variance-ratio): trending vs chopping; bounded [−1,1], directional. | `ema_signed_logret_{N}t / ema_abs_log_ret_{N}t` (denominator already built). No yardstick needed. | L1 mid + clock | **P1** |
| **momentum_spread** (#19) | both | both | Trade-clock MACD: fast−slow signed-return EMA = acceleration/breakout vs exhaustion. | `ema_signed_logret_{Nfast} − ema_signed_logret_{Nslow}`, /(σ_ev·k). Downstream of #17. | L1 mid + clock | **P2** |

### Theme E — Book-shape (state, not flow)

| Feature | X/per | Horizon | Mechanism | Construction | Data | Pri |
|---|---|---|---|---|---|---|
| **microprice_premium** (#23) | both | short | Stoikov micro-price premium (MP−mid)/mid = E[next mid move]; folds QI *and* half-spread into one signed predictor. | New atom premium on grid; trade-clock EMA family; breakout = premium_now − ema_premium_long. Bounded by ±halfspread/mid → invariant. | L1 price+size | **P1** |
| **queue_imbalance_dev** (#22) | both | short | Size dislocation oscillator: how lopsided the touch is *now* vs its norm (price_dislocation applied to size). | `book_imbalance_now − ema_book_imbalance_{long}t` (level atom exists; the signed deviation is new). Bounded. | L1 size | **P2** |
| **spread_state** (#26) | both | both | Widening spread = MM pulling quotes ahead of info → rate head up; gates price head actionability (Roll). | `spread_ratio = spread_now / ema_spread_long`; `spread_surge = ema_spread_short/ema_spread_long` (mirrors vol_surge). | L1 price | **P2** |
| **depth_depletion** (#25) | both | short | Same-price queue depletion/replenishment (queue-reactive, Huang-Lehalle-Rosenbaum) — info OFI misses at fixed price. | New atom: Σ same-price Δbid−Δask per interval / ema_book_depth (existing). | L1 price+size | **P3** |
| **signed_ofi_momentum** (#20) | both | both | Short-vs-long OFI persistence as directional book pressure. | Mostly downstream of existing ofi/ofi_sq; optional new ofi-sign serial-cov atom. Low novelty vs built ofi_normalised. | L1 price+size | **P3** |

### Theme F — Toxicity / impact

| Feature | X/per | Horizon | Mechanism | Construction | Data | Pri |
|---|---|---|---|---|---|---|
| **kyle_lambda** (#28) | both | both | Regression slope of mid-move per signed flow = adverse-selection / fragility (Kyle 1985); rising λ → larger continuing moves. | New atoms `ema_signed_flow_ret_cov_{N}t`, `ema_signed_flow_sq_{N}t`; λ = cov/(var+ε); cross-venue compare λ_okx vs λ_byb. | L1 mid + trades | **P1** |
| **flow_sign_persistence** (#18+#30) | both | both | Aggressor-sign long-memory (Lillo-Farmer / order-splitting): runs predict same-sign next trade → byb continuation. | `ema_sign_serial_cov_{N}t = EMA(ε_t·ε_{t−1})`∈[−1,1] (regime gauge) + `ema_trade_sign_{N}t`∈[−1,1] (signed direction). No vol norm needed. | trades | **P1** |
| **sweep_intensity** (#29) | both | short | Sweep (print walks ticks / outsized vs touch) = impatient informed taker → continuation; detect on okx/bin first. | New atoms `ema_ticks_crossed_{N}t` (=|prc−mid|/spread_before), `ema_qty_over_touch_{N}t`; both ratio-normalised. Cross leg @trades_byb. | L1 + trades | **P2** |
| **impact_share** (#31) | both | short | Glosten-Harris effective/realised decomposition: rising adverse-selection share → post-trade mid keeps drifting = byb's next move. | Causal across-trade mid revision: `ema_trade_impact / ema_eff_spread`∈~[0,1]. | L1 mid + trades | **P2** |
| **trade_to_quote_ratio** (#32) | both | short→med | Trades per book update = liquidity drained faster than refreshed → toxic, move-heavy regime. | New `ema_book_update_count_{N}t`; TQR = 1/(count+ε); fast/slow surge. Counts only. | event times | **P3** |
| **size_concentration** (#33) | both | med→long | Rising large-print share = informed/block flow (VPIN size intuition). | New `ema_trade_qty_sq_{N}t`; concentration = qty_sq/(qty²+ε); signed by buy/sell split. | trades | **P3** |

---

## 3. TOP picks to build next (weighted to cross-exchange + breakout)

1. **xv_book_pressure_gap** (#24) — *cross-exchange, earliest-firing.* Book pressure (QI / micro-price premium) on okx/bin moves *before* any byb price print; the bounded gap on the shared clock is the cleanest hit on priority #1. Already-invariant.
2. **xv_leadlag_xcorr** (#11/#4) — *cross-exchange, directly answers "who leads byb and by how many ms."* Nothing built measures the lead-lag relationship itself; bounded ρ_k is regime-invariant by construction.
3. **gap_zscore (OU)** (#8) — *cross-exchange relative value, mostly free.* Built almost entirely from existing ratio/ratio_sq atoms; gives the mean-reversion (snap-back) read that price_dislocation's momentum-style oscillator does not. Pair with **gap_halflife** as the continue-vs-revert gate.
4. **range_position + breakout_magnitude** (#12/#13) — *the breakout pair (priority #2), one shared atom.* Bounded position + σ_ev-normalised pierce cover short *and* long via the span family; runmax/runmin atom unlocks #15 for free.
5. **signed_return_momentum + trend_efficiency_ratio** (#17/#21) — *per-exchange directional breakout at both horizons.* The signed trade-clock momentum atom is a genuine gap in the catalogue; the efficiency ratio reuses the already-built `ema_abs_log_ret` denominator and is intrinsically invariant.
6. **microprice_premium** (#23) — *strongest single L1 next-mid predictor (Stoikov).* Cheap, theory-backed, and the cross-venue premium-gap feeds straight into pick #1.
7. **kyle_lambda** (#28) — *the missing impact slope.* Built set has flow and spread proxies but nothing regresses return on signed flow; cross-venue λ comparison flags where information arrives first.
8. **flow_sign_persistence** (#18/#30) — *cheapest high-value toxicity add.* Trades-only, intrinsically bounded ([−1,1]), and orthogonal to the price-based Roll serial-cov already built.

**Why this set:** five of eight are cross-exchange or have a cross-venue leg; three are pure breakout/momentum at both horizons; all are bounded-or-yardstick-normalised (expected vol-bucket scale < 3); none exceed L1+trades; and four reuse existing atoms (low build cost). Picks 1–2 are the highest-conviction cross-venue bets; 4–5 directly serve breakout priority #2.

---

## 4. Speculative / honesty flags

- **xv_rate_race (#5)** and **trade_to_quote_ratio (#32)** are plausible but weak — pure event-intensity ratios with no price content alone; only useful as gates/interactions. Build last.
- **signed_ofi_momentum (#20)** and **depth_depletion (#25)** overlap the already-built `ofi_normalised`; novelty is marginal. Build only if #28/#23 underperform.
- **gap_velocity (#7)** is theoretically clean but τ-differencing of a short EMA is noise-prone at ms scale; validate the oracle carefully on a real block before trusting it.
- The **squeeze-flag** (from #14) and **imbalance-persistence/depth-surge** (from #27) are demoted to notes — they are thin re-skins of `vol_surge`/`volume_surge`. Not worth their own columns unless a downstream model specifically wants a sustained-state flag.

---

## 5. Appendix — needs new data (none of the 33 require this; these are gaps the survivors expose)

Every one of the 33 proposals is buildable from current L1+trades. The following are *adjacent features the proposals gesture at but cannot reach* with current data — flagged so the user can decide whether to source them:

- **Spot–perp basis (needs spot books).** The cross-venue relative-value theme (gap_zscore, xv_book_pressure_gap) currently tethers perp↔perp. A spot-anchored basis would give a stronger no-arbitrage cointegration mean and a cleaner "which leg is dislocated" read. *Needs spot front_levels.*
- **Funding-rate pressure (needs funding).** Perp mids drift toward funding-implied fair value; a funding-distance term would condition the OU mean and explain slow byb drift the gap features attribute to noise. *Needs funding stream.*
- **Full-depth queue dynamics (needs L2+).** sweep_intensity (#29) and depth_depletion (#25) are truncated to the first tick / top-of-book. Real sweep depth, iceberg detection, and book-slope (cumulative-depth Kyle λ) need L2. *Needs full depth.* The L1 versions are the dominant-signal approximation and worth building now; L2 would sharpen, not replace, them.

No survivor was dropped for missing data — the data envelope (L1 + trades, perp-only) is sufficient for all 18 prioritised features.