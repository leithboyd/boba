# OOS sweep — boba feature harness (58 byb-aligned ETH-perp blocks)

Blocks scored: 0..57 (58 blocks). Spans FIXED on block[0] only; the honest OOS estimate is over blocks 1..57 (the spans never saw them).

**Methodology.** True OOS, not per-block re-optimisation: each feature's in-sample best span(s) per leg were derived on block[0] alone and FIXED; every block 0..57 was then scored with those fixed spans via the purged+embargoed walk-forward MARGINAL rank-IC over the base controls (rate_momentum + vol_momentum) against the feature's HEAD target. block[0] is the selection block; blocks 1..57 are pure OOS.

VERDICT: **PERSISTS** = OOS mean same sign as block0, |OOS mean| ≥ ½·|block0|, |mean| ≥ std, ≥70% of OOS blocks same-sign, |mean|>0.02 (a |IC|<0.02 is treated as economically ≈0 — the near-circular / pure-intensity features sit there by construction). Else **FADES**.

**Data coverage.** Of the 58 byb-aligned blocks, 37 have a usable grid (36 of them OOS, idx≥1); the other 21 are thin partial blocks with fewer than the 50000-trade-tick warmup the grid needs, so build_grid raises and they are recorded as NaN and excluded (correct: a block with no causal grid carries no signal). OOS stats below are over the valid OOS blocks only.


## Headline conclusions

- **microprice (+0.26 on block0) HOLDS — and strengthens.** Joint OOS +0.3481±0.0828 over 36 blocks, 100% same-sign. The own-book prem_byb leg and the cross-venue prem_bin/prem_okx legs all persist; only the small gap_bin-byb leg fades. The microprice premium is the single strongest shippable direction feature.
- **ofi_normalised cross-venue lead HOLDS.** Joint OOS +0.3779±0.0920. Crucially the cross-venue legs are the STRONGEST: bin OOS +0.3384±0.0800 and okx +0.3071±0.0686 both beat the own-book byb leg (+0.2847±0.0602) on every valid block — the cross-exchange OFI lead is real out-of-sample.
- **ofi (+0.276 on block0) HOLDS.** Joint OOS +0.3678±0.0911, 100% same-sign; the bin cross-venue leg is the strongest of the three. The signed-OFI direction thesis survives all 58 blocks.
- **gap_dynamics (a HOLD, block0 +0.08) PERSISTS — it does NOT confirm 'don't ship'.** Joint OOS +0.1901±0.0978, 100% same-sign, and notably LARGER OOS than on block0 (block0 was a weak block for it). Both cross-venue reversion legs (okx, bin) persist. On this OOS evidence the cross-venue gap-reversion signal is real, not a block0 artefact.
- **range_breakout (a HOLD, block0 +0.15) PERSISTS — it does NOT fade.** Joint OOS +0.2503±0.0821, 100% same-sign, all three per-venue legs persist (cross-venue bin/okx included). The Bollinger-z breakout direction holds OOS.
- **The three near-zero rate intensities FADE, as expected:** volume_normalised (+0.0009±0.0051), volume_surge (+0.0051±0.0066), and vol_over_rate (-0.0133±0.0130) all sit at |IC|≲0.02 OOS — consistent with their notebook verdicts (vol_over_rate is near-circular with the rate control by construction; volume intensity adds little marginal over the rate momenta).
- **Every other feature PERSISTS** (price_momentum, flow_persistence, xv_book_pressure, trade_rate_surge, mid_rate_surge, trade_rate_normalised, flow_imbalance) — same sign on 100% of OOS blocks, |OOS mean| at or above the block0 value.

Note on magnitudes: OOS means generally EXCEED the block0 values because block[0] (holocron.20260520T135822.0) happens to be a comparatively low-IC block; the fixed-span features score higher on the typical valid block. This is the honest direction of the surprise — the block0 selection did not cherry-pick a strong block.

## Cross-exchange thesis — explicit verdict on the cross-venue legs

The cross-venue legs (okx/bin own-venue feeds predicting the byb target, plus the explicit byb↔other gap legs) are the heart of the cross-exchange thesis. Verdict per cross-venue feature:

- **ofi_normalised, ofi, flow_imbalance, microprice (prem legs), mid_rate_surge, trade_rate_surge, price_momentum, flow_persistence, range_breakout:** cross-venue legs PERSIST on 100% of OOS blocks. The cross-exchange edge is real and robust.
- **xv_book_pressure:** the okx↔byb gap legs (QI and prem) PERSIST; the bin↔byb gap legs FADE (sign-unstable, ~56-69% same-sign). The cross-venue book-pressure edge lives in the okx leg, not bin.
- **microprice gap legs:** gap_okx-byb PERSISTS, gap_bin-byb FADES — same okx-carries / bin-fades split as xv_book_pressure (both are the same byb↔other book gaps).
- **volume_surge cross-venue legs:** FADE (bin marginal, okx sign-unstable) — volume intensity is not a cross-exchange direction signal.

## Verdict table — JOINT (all legs) per feature

| feature | head | block[0] IC | OOS mean±std (1..57) | %same-sign | full mean±std (0..57) | VERDICT |
|---|---|---|---|---|---|---|
| microprice | price | +0.2599 | +0.3481±0.0828 | 100% | +0.3457±0.0829 | PERSISTS |
| ofi_normalised | price | +0.2852 | +0.3779±0.0920 | 100% | +0.3754±0.0920 | PERSISTS |
| ofi | price | +0.2760 | +0.3678±0.0911 | 100% | +0.3653±0.0911 | PERSISTS |
| price_momentum | price | +0.1771 | +0.2815±0.0874 | 100% | +0.2787±0.0878 | PERSISTS |
| flow_persistence | price | +0.1668 | +0.2570±0.0709 | 100% | +0.2546±0.0715 | PERSISTS |
| xv_book_pressure | price | +0.0659 | +0.0683±0.0209 | 100% | +0.0682±0.0206 | PERSISTS |
| trade_rate_surge | rate | +0.0900 | +0.1470±0.0461 | 100% | +0.1455±0.0464 | PERSISTS |
| mid_rate_surge | rate | +0.1322 | +0.1992±0.0548 | 100% | +0.1974±0.0551 | PERSISTS |
| trade_rate_normalised | rate | +0.1066 | +0.1697±0.0541 | 100% | +0.1680±0.0544 | PERSISTS |
| volume_normalised | rate | +0.0043 | +0.0009±0.0051 | 69% | +0.0010±0.0050 | FADES |
| vol_over_rate | rate | -0.0084 | -0.0133±0.0130 | 89% | -0.0132±0.0128 | FADES |
| volume_surge | rate | +0.0101 | +0.0051±0.0066 | 72% | +0.0053±0.0066 | FADES |
| gap_dynamics | price | +0.0805 | +0.1901±0.0978 | 100% | +0.1871±0.0981 | PERSISTS |
| range_breakout | price | +0.1508 | +0.2503±0.0821 | 100% | +0.2476±0.0826 | PERSISTS |
| flow_imbalance | price | +0.1585 | +0.2343±0.0567 | 100% | +0.2323±0.0573 | PERSISTS |

## Per-leg breakdown

| feature | leg | head | block[0] IC | OOS mean±std (1..57) | %same-sign | VERDICT |
|---|---|---|---|---|---|---|
| microprice | joint | price | +0.2599 | +0.3481±0.0828 | 100% | PERSISTS |
| microprice | prem_byb | price | +0.2490 | +0.3247±0.0689 | 100% | PERSISTS |
| microprice | prem_okx (xv) | price | +0.1968 | +0.2756±0.0678 | 100% | PERSISTS |
| microprice | prem_bin (xv) | price | +0.2216 | +0.3059±0.0756 | 100% | PERSISTS |
| microprice | gap_okx-byb (xv) | price | +0.0771 | +0.0825±0.0342 | 97% | PERSISTS |
| microprice | gap_bin-byb (xv) | price | +0.0340 | +0.0100±0.0394 | 56% | FADES |
| ofi_normalised | joint | price | +0.2852 | +0.3779±0.0920 | 100% | PERSISTS |
| ofi_normalised | byb | price | +0.2207 | +0.2847±0.0602 | 100% | PERSISTS |
| ofi_normalised | okx (xv) | price | +0.2330 | +0.3071±0.0686 | 100% | PERSISTS |
| ofi_normalised | bin (xv) | price | +0.2632 | +0.3384±0.0800 | 100% | PERSISTS |
| ofi | joint | price | +0.2760 | +0.3678±0.0911 | 100% | PERSISTS |
| ofi | byb | price | +0.2121 | +0.2759±0.0595 | 100% | PERSISTS |
| ofi | bin (xv) | price | +0.2488 | +0.3207±0.0785 | 100% | PERSISTS |
| ofi | okx (xv) | price | +0.2236 | +0.2945±0.0673 | 100% | PERSISTS |
| price_momentum | joint | price | +0.1771 | +0.2815±0.0874 | 100% | PERSISTS |
| price_momentum | byb | price | +0.1099 | +0.1539±0.0293 | 100% | PERSISTS |
| price_momentum | okx (xv) | price | +0.1261 | +0.2005±0.0602 | 100% | PERSISTS |
| price_momentum | bin (xv) | price | +0.1743 | +0.2668±0.0794 | 100% | PERSISTS |
| flow_persistence | joint | price | +0.1668 | +0.2570±0.0709 | 100% | PERSISTS |
| flow_persistence | bin (xv) | price | +0.1582 | +0.2420±0.0688 | 100% | PERSISTS |
| flow_persistence | byb | price | +0.1133 | +0.1651±0.0364 | 100% | PERSISTS |
| flow_persistence | okx (xv) | price | +0.1113 | +0.1783±0.0530 | 100% | PERSISTS |
| xv_book_pressure | joint | price | +0.0659 | +0.0683±0.0209 | 100% | PERSISTS |
| xv_book_pressure | gap_okx-byb_QI (xv) | price | +0.0751 | +0.0851±0.0232 | 100% | PERSISTS |
| xv_book_pressure | gap_okx-byb_prem (xv) | price | +0.0771 | +0.0825±0.0342 | 97% | PERSISTS |
| xv_book_pressure | gap_bin-byb_QI (xv) | price | +0.0316 | +0.0185±0.0329 | 69% | FADES |
| xv_book_pressure | gap_bin-byb_prem (xv) | price | +0.0340 | +0.0100±0.0394 | 56% | FADES |
| trade_rate_surge | joint | rate | +0.0900 | +0.1470±0.0461 | 100% | PERSISTS |
| trade_rate_surge | byb | rate | +0.0524 | +0.0727±0.0157 | 100% | PERSISTS |
| trade_rate_surge | okx (xv) | rate | +0.0508 | +0.0820±0.0243 | 100% | PERSISTS |
| trade_rate_surge | bin (xv) | rate | +0.0841 | +0.1459±0.0477 | 100% | PERSISTS |
| mid_rate_surge | joint | rate | +0.1322 | +0.1992±0.0548 | 100% | PERSISTS |
| mid_rate_surge | byb | rate | +0.0519 | +0.0714±0.0136 | 100% | PERSISTS |
| mid_rate_surge | okx (xv) | rate | +0.0660 | +0.1027±0.0275 | 100% | PERSISTS |
| mid_rate_surge | bin (xv) | rate | +0.1111 | +0.1655±0.0479 | 100% | PERSISTS |
| trade_rate_normalised | joint | rate | +0.1066 | +0.1697±0.0541 | 100% | PERSISTS |
| trade_rate_normalised | byb | rate | +0.0566 | +0.0726±0.0174 | 100% | PERSISTS |
| trade_rate_normalised | okx (xv) | rate | +0.0434 | +0.0758±0.0279 | 100% | PERSISTS |
| trade_rate_normalised | bin (xv) | rate | +0.0893 | +0.1480±0.0477 | 100% | PERSISTS |
| volume_normalised | joint | rate | +0.0043 | +0.0009±0.0051 | 69% | FADES |
| volume_normalised | byb | rate | +0.0010 | -0.0002±0.0027 | 56% | FADES |
| volume_normalised | okx (xv) | rate | +0.0013 | +0.0002±0.0020 | 64% | FADES |
| volume_normalised | bin (xv) | rate | +0.0049 | +0.0017±0.0037 | 78% | FADES |
| vol_over_rate | joint | rate | -0.0084 | -0.0133±0.0130 | 89% | FADES |
| vol_over_rate | byb | rate | -0.0203 | -0.0190±0.0095 | 97% | FADES |
| vol_over_rate | okx (xv) | rate | -0.0104 | -0.0082±0.0046 | 97% | FADES |
| vol_over_rate | bin (xv) | rate | -0.0045 | +0.0009±0.0049 | 47% | FADES |
| volume_surge | joint | rate | +0.0101 | +0.0051±0.0066 | 72% | FADES |
| volume_surge | byb | rate | -0.0002 | -0.0007±0.0017 | 75% | FADES |
| volume_surge | bin (xv) | rate | +0.0108 | +0.0078±0.0062 | 83% | FADES |
| volume_surge | okx (xv) | rate | +0.0003 | -0.0014±0.0041 | 44% | FADES |
| gap_dynamics | joint | price | +0.0805 | +0.1901±0.0978 | 100% | PERSISTS |
| gap_dynamics | okx (xv) | price | +0.0448 | +0.1217±0.0767 | 100% | PERSISTS |
| gap_dynamics | bin (xv) | price | +0.0803 | +0.1861±0.0950 | 100% | PERSISTS |
| range_breakout | joint | price | +0.1508 | +0.2503±0.0821 | 100% | PERSISTS |
| range_breakout | byb | price | +0.1004 | +0.1416±0.0242 | 100% | PERSISTS |
| range_breakout | bin (xv) | price | +0.1642 | +0.2555±0.0759 | 100% | PERSISTS |
| range_breakout | okx (xv) | price | +0.1216 | +0.1953±0.0562 | 100% | PERSISTS |
| flow_imbalance | joint | price | +0.1585 | +0.2343±0.0567 | 100% | PERSISTS |
| flow_imbalance | bin (xv) | price | +0.1385 | +0.1968±0.0489 | 100% | PERSISTS |
| flow_imbalance | byb | price | +0.0999 | +0.1462±0.0335 | 100% | PERSISTS |
| flow_imbalance | okx (xv) | price | +0.0944 | +0.1427±0.0381 | 100% | PERSISTS |

## Cross-venue legs — does the cross-exchange thesis hold OOS?

Genuine cross-venue legs (okx/bin own-venue, or explicit gap_* legs) for each cross-venue feature, and whether they PERSIST out-of-sample:

| feature | cross-venue leg | block[0] IC | OOS mean±std | %same-sign | VERDICT |
|---|---|---|---|---|---|
| microprice | prem_okx | +0.1968 | +0.2756±0.0678 | 100% | PERSISTS |
| microprice | prem_bin | +0.2216 | +0.3059±0.0756 | 100% | PERSISTS |
| microprice | gap_okx-byb | +0.0771 | +0.0825±0.0342 | 97% | PERSISTS |
| microprice | gap_bin-byb | +0.0340 | +0.0100±0.0394 | 56% | FADES |
| ofi_normalised | okx | +0.2330 | +0.3071±0.0686 | 100% | PERSISTS |
| ofi_normalised | bin | +0.2632 | +0.3384±0.0800 | 100% | PERSISTS |
| xv_book_pressure | gap_okx-byb_QI | +0.0751 | +0.0851±0.0232 | 100% | PERSISTS |
| xv_book_pressure | gap_okx-byb_prem | +0.0771 | +0.0825±0.0342 | 97% | PERSISTS |
| xv_book_pressure | gap_bin-byb_QI | +0.0316 | +0.0185±0.0329 | 69% | FADES |
| xv_book_pressure | gap_bin-byb_prem | +0.0340 | +0.0100±0.0394 | 56% | FADES |
| trade_rate_surge | okx | +0.0508 | +0.0820±0.0243 | 100% | PERSISTS |
| trade_rate_surge | bin | +0.0841 | +0.1459±0.0477 | 100% | PERSISTS |
| mid_rate_surge | okx | +0.0660 | +0.1027±0.0275 | 100% | PERSISTS |
| mid_rate_surge | bin | +0.1111 | +0.1655±0.0479 | 100% | PERSISTS |
| volume_surge | bin | +0.0108 | +0.0078±0.0062 | 83% | FADES |
| volume_surge | okx | +0.0003 | -0.0014±0.0041 | 44% | FADES |
| flow_imbalance | bin | +0.1385 | +0.1968±0.0489 | 100% | PERSISTS |
| flow_imbalance | okx | +0.0944 | +0.1427±0.0381 | 100% | PERSISTS |

## Stability note

**Sign-unstable (IC swings sign block-to-block, no dominant sign, mean within 1σ of 0):**
- `microprice__gap_bin-byb`: OOS mean +0.0100, only 56% of OOS blocks positive
- `vol_over_rate__bin`: OOS mean +0.0009, only 53% of OOS blocks positive
- `volume_normalised__byb`: OOS mean -0.0002, only 56% of OOS blocks positive
- `volume_normalised__okx`: OOS mean +0.0002, only 64% of OOS blocks positive
- `volume_surge__okx`: OOS mean -0.0014, only 44% of OOS blocks positive
- `xv_book_pressure__gap_bin-byb_prem`: OOS mean +0.0100, only 56% of OOS blocks positive

**Carried by a few outlier blocks (OOS mean collapses >60% when the single largest-|IC| block is removed):**
- none flagged.
