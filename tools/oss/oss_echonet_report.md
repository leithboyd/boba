# Echo-netted OOS sweep — tradeable (net-of-contemporaneous-echo) edge

Blocks scored: 37 cached blocks (idx 0..57; 36 are OOS, idx>=1). Spans FROZEN on block[0] (fixed_spans_block0.json) — the same fixed-span OOS discipline as Part 1; the spans never saw blocks 1..57.

**What this measures.** Part 1's marginal IC (over base controls [rate_momentum, vol_momentum]) includes the *contemporaneous echo*: a feature can post a high forward IC just by re-reporting the move already underway at the anchor. Here every feature is re-scored TWICE per block via the same purged+embargoed walk-forward marginal rank-IC against its HEAD forward target:

- **RAW**: base = [rate_momentum, vol_momentum] (reproduces Part 1 exactly — block[0] RAW joint matches oss_ic_matrix.csv to 0.0e+00).
- **ECHO-NETTED**: base = [rate_momentum, vol_momentum, **trailing_outcome**], where trailing_outcome is the already-happened analogue of the forward target on the same byb grid over the STRICTLY-PAST window [anchor-100ms, anchor]: PRICE head = log(mid@anchor / mid@(anchor-100ms)) / sigma_ev; RATE head = byb mid-move count over (anchor-100ms, anchor] / lambda_ev. No forward leak — the window ends AT the anchor.

The ECHO-NETTED IC is the honest **tradeable** forward edge: what the feature adds ABOVE simply knowing the contemporaneous move.

**Tradeable verdict.** **PERSISTS** = net OOS mean clearly nonzero (|mean|>0.02), same sign as the raw OOS mean, and retains >=half the raw OOS mean (a real forward edge survives netting). **SHRINKS** = net still nonzero & same sign but retains <half the raw (a tradeable core remains, but much of the raw IC was echo). **COLLAPSES->echo** = net ~0 (|mean|<=0.02) or sign-flips: the raw IC was mostly/entirely contemporaneous echo.


## Headline conclusions (tradeable, OOS, net of echo)

- **Keep a real tradeable forward edge after netting (PERSISTS):** microprice, ofi_normalised, ofi, price_momentum, trade_rate_surge, mid_rate_surge, trade_rate_normalised, gap_dynamics, range_breakout.
- **Shrink (a tradeable core survives, but much of the raw IC was echo) (SHRINKS):** flow_persistence, xv_book_pressure, flow_imbalance.
- **Collapse to mostly echo (COLLAPSES):** volume_normalised, vol_over_rate, volume_surge.
- **microprice** is the strongest tradeable direction feature net of echo: joint raw +0.3481±0.0828 -> net +0.2022±0.0408 (retention 58%). The premium leads the byb mid, not just echoes it.
- **ofi_normalised** cross-venue lead holds net of echo: joint raw +0.3779±0.0920 -> net +0.2321±0.0517 (retention 61%); the bin/okx cross-venue legs survive (see the cross-venue table).
- **range_breakout** sheds most of its edge to echo — esp. the own-book byb leg (raw +0.1416 -> net +0.0338): the Bollinger-z breakout largely re-reports the move already in progress.
- **own-book momentum legs collapse:** price_momentum byb (raw +0.1539 -> net -0.0064) and the byb own-leg of gap_dynamics-style features are mostly contemporaneous echo, as expected (own-venue trailing return ~= own-venue forward echo).

## Verdict table — JOINT (all legs) per feature

| feature | head | OOS raw IC (mean±std) | OOS echo-netted IC (mean±std) | retention (net/raw) | TRADEABLE verdict |
|---|---|---|---|---|---|
| microprice | price | +0.3481±0.0828 | +0.2022±0.0408 | 58% | PERSISTS |
| ofi_normalised | price | +0.3779±0.0920 | +0.2321±0.0517 | 61% | PERSISTS |
| ofi | price | +0.3678±0.0911 | +0.2220±0.0510 | 60% | PERSISTS |
| price_momentum | price | +0.2815±0.0874 | +0.1546±0.0457 | 55% | PERSISTS |
| flow_persistence | price | +0.2570±0.0709 | +0.1210±0.0281 | 47% | SHRINKS |
| xv_book_pressure | price | +0.0683±0.0209 | +0.0302±0.0105 | 44% | SHRINKS |
| trade_rate_surge | rate | +0.1470±0.0461 | +0.0745±0.0241 | 51% | PERSISTS |
| mid_rate_surge | rate | +0.1992±0.0548 | +0.1183±0.0353 | 59% | PERSISTS |
| trade_rate_normalised | rate | +0.1697±0.0541 | +0.0919±0.0321 | 54% | PERSISTS |
| volume_normalised | rate | +0.0009±0.0051 | +0.0011±0.0038 | 119% | COLLAPSES->echo |
| vol_over_rate | rate | -0.0133±0.0130 | -0.0107±0.0118 | 80% | COLLAPSES->echo |
| volume_surge | rate | +0.0051±0.0066 | +0.0038±0.0057 | 75% | COLLAPSES->echo |
| gap_dynamics | price | +0.1901±0.0978 | +0.1027±0.0507 | 54% | PERSISTS |
| range_breakout | price | +0.2503±0.0821 | +0.1346±0.0440 | 54% | PERSISTS |
| flow_imbalance | price | +0.2343±0.0567 | +0.1064±0.0164 | 45% | SHRINKS |

## Per-leg breakdown (raw -> echo-netted, OOS)

| feature | leg | head | OOS raw IC | OOS echo-netted IC | retention | verdict |
|---|---|---|---|---|---|---|
| microprice | joint | price | +0.3481±0.0828 | +0.2022±0.0408 | 58% | PERSISTS |
| microprice | prem_byb | price | +0.3247±0.0689 | +0.1755±0.0241 | 54% | PERSISTS |
| microprice | prem_okx (xv) | price | +0.2756±0.0678 | +0.1447±0.0259 | 52% | PERSISTS |
| microprice | prem_bin (xv) | price | +0.3059±0.0756 | +0.1725±0.0334 | 56% | PERSISTS |
| microprice | gap_okx-byb (xv) | price | +0.0825±0.0342 | +0.0365±0.0147 | 44% | SHRINKS |
| microprice | gap_bin-byb (xv) | price | +0.0100±0.0394 | +0.0044±0.0133 | 44% | COLLAPSES->echo |
| ofi_normalised | joint | price | +0.3779±0.0920 | +0.2321±0.0517 | 61% | PERSISTS |
| ofi_normalised | byb | price | +0.2847±0.0602 | +0.1377±0.0219 | 48% | SHRINKS |
| ofi_normalised | okx (xv) | price | +0.3071±0.0686 | +0.1715±0.0307 | 56% | PERSISTS |
| ofi_normalised | bin (xv) | price | +0.3384±0.0800 | +0.2058±0.0413 | 61% | PERSISTS |
| ofi | joint | price | +0.3678±0.0911 | +0.2220±0.0510 | 60% | PERSISTS |
| ofi | byb | price | +0.2759±0.0595 | +0.1295±0.0212 | 47% | SHRINKS |
| ofi | bin (xv) | price | +0.3207±0.0785 | +0.1914±0.0404 | 60% | PERSISTS |
| ofi | okx (xv) | price | +0.2945±0.0673 | +0.1610±0.0303 | 55% | PERSISTS |
| price_momentum | joint | price | +0.2815±0.0874 | +0.1546±0.0457 | 55% | PERSISTS |
| price_momentum | byb | price | +0.1539±0.0293 | -0.0064±0.0190 | -4% | COLLAPSES->echo |
| price_momentum | okx (xv) | price | +0.2005±0.0602 | +0.0870±0.0216 | 43% | SHRINKS |
| price_momentum | bin (xv) | price | +0.2668±0.0794 | +0.1453±0.0387 | 54% | PERSISTS |
| flow_persistence | joint | price | +0.2570±0.0709 | +0.1210±0.0281 | 47% | SHRINKS |
| flow_persistence | bin (xv) | price | +0.2420±0.0688 | +0.1140±0.0263 | 47% | SHRINKS |
| flow_persistence | byb | price | +0.1651±0.0364 | +0.0540±0.0092 | 33% | SHRINKS |
| flow_persistence | okx (xv) | price | +0.1783±0.0530 | +0.0645±0.0146 | 36% | SHRINKS |
| xv_book_pressure | joint | price | +0.0683±0.0209 | +0.0302±0.0105 | 44% | SHRINKS |
| xv_book_pressure | gap_okx-byb_QI (xv) | price | +0.0851±0.0232 | +0.0381±0.0096 | 45% | SHRINKS |
| xv_book_pressure | gap_okx-byb_prem (xv) | price | +0.0825±0.0342 | +0.0365±0.0147 | 44% | SHRINKS |
| xv_book_pressure | gap_bin-byb_QI (xv) | price | +0.0185±0.0329 | +0.0072±0.0115 | 39% | COLLAPSES->echo |
| xv_book_pressure | gap_bin-byb_prem (xv) | price | +0.0100±0.0394 | +0.0044±0.0133 | 44% | COLLAPSES->echo |
| trade_rate_surge | joint | rate | +0.1470±0.0461 | +0.0745±0.0241 | 51% | PERSISTS |
| trade_rate_surge | byb | rate | +0.0727±0.0157 | +0.0122±0.0063 | 17% | COLLAPSES->echo |
| trade_rate_surge | okx (xv) | rate | +0.0820±0.0243 | +0.0269±0.0079 | 33% | SHRINKS |
| trade_rate_surge | bin (xv) | rate | +0.1459±0.0477 | +0.0794±0.0259 | 54% | PERSISTS |
| mid_rate_surge | joint | rate | +0.1992±0.0548 | +0.1183±0.0353 | 59% | PERSISTS |
| mid_rate_surge | byb | rate | +0.0714±0.0136 | +0.0017±0.0057 | 2% | COLLAPSES->echo |
| mid_rate_surge | okx (xv) | rate | +0.1027±0.0275 | +0.0463±0.0118 | 45% | SHRINKS |
| mid_rate_surge | bin (xv) | rate | +0.1655±0.0479 | +0.1074±0.0325 | 65% | PERSISTS |
| trade_rate_normalised | joint | rate | +0.1697±0.0541 | +0.0919±0.0321 | 54% | PERSISTS |
| trade_rate_normalised | byb | rate | +0.0726±0.0174 | +0.0111±0.0071 | 15% | COLLAPSES->echo |
| trade_rate_normalised | okx (xv) | rate | +0.0758±0.0279 | +0.0228±0.0115 | 30% | SHRINKS |
| trade_rate_normalised | bin (xv) | rate | +0.1480±0.0477 | +0.0935±0.0322 | 63% | PERSISTS |
| volume_normalised | joint | rate | +0.0009±0.0051 | +0.0011±0.0038 | 119% | COLLAPSES->echo |
| volume_normalised | byb | rate | -0.0002±0.0027 | -0.0000±0.0023 | 14% | COLLAPSES->echo |
| volume_normalised | okx (xv) | rate | +0.0002±0.0020 | +0.0003±0.0017 | 128% | COLLAPSES->echo |
| volume_normalised | bin (xv) | rate | +0.0017±0.0037 | +0.0018±0.0028 | 109% | COLLAPSES->echo |
| vol_over_rate | joint | rate | -0.0133±0.0130 | -0.0107±0.0118 | 80% | COLLAPSES->echo |
| vol_over_rate | byb | rate | -0.0190±0.0095 | -0.0144±0.0068 | 76% | COLLAPSES->echo |
| vol_over_rate | okx (xv) | rate | -0.0082±0.0046 | -0.0037±0.0032 | 45% | COLLAPSES->echo |
| vol_over_rate | bin (xv) | rate | +0.0009±0.0049 | +0.0052±0.0048 | 597% | COLLAPSES->echo |
| volume_surge | joint | rate | +0.0051±0.0066 | +0.0038±0.0057 | 75% | COLLAPSES->echo |
| volume_surge | byb | rate | -0.0007±0.0017 | -0.0008±0.0009 | 120% | COLLAPSES->echo |
| volume_surge | bin (xv) | rate | +0.0078±0.0062 | +0.0075±0.0050 | 96% | COLLAPSES->echo |
| volume_surge | okx (xv) | rate | -0.0014±0.0041 | -0.0024±0.0030 | 167% | COLLAPSES->echo |
| gap_dynamics | joint | price | +0.1901±0.0978 | +0.1027±0.0507 | 54% | PERSISTS |
| gap_dynamics | okx (xv) | price | +0.1217±0.0767 | +0.0547±0.0326 | 45% | SHRINKS |
| gap_dynamics | bin (xv) | price | +0.1861±0.0950 | +0.0993±0.0483 | 53% | PERSISTS |
| range_breakout | joint | price | +0.2503±0.0821 | +0.1346±0.0440 | 54% | PERSISTS |
| range_breakout | byb | price | +0.1416±0.0242 | +0.0338±0.0107 | 24% | SHRINKS |
| range_breakout | bin (xv) | price | +0.2555±0.0759 | +0.1250±0.0333 | 49% | SHRINKS |
| range_breakout | okx (xv) | price | +0.1953±0.0562 | +0.0768±0.0173 | 39% | SHRINKS |
| flow_imbalance | joint | price | +0.2343±0.0567 | +0.1064±0.0164 | 45% | SHRINKS |
| flow_imbalance | bin (xv) | price | +0.1968±0.0489 | +0.0917±0.0136 | 47% | SHRINKS |
| flow_imbalance | byb | price | +0.1462±0.0335 | +0.0483±0.0075 | 33% | SHRINKS |
| flow_imbalance | okx (xv) | price | +0.1427±0.0381 | +0.0518±0.0092 | 36% | SHRINKS |

## Cross-venue legs — does the cross-exchange lead survive netting OOS?

For each cross-venue feature, the genuine cross-venue legs (okx/bin own-venue feeds, or explicit gap_* legs) and whether the lead survives net of the contemporaneous echo. A cross-venue leg that PERSISTS net of echo is a true cross-exchange LEAD (the other venue moved first), not a re-report of byb's own move.

| feature | cross-venue leg | OOS raw IC | OOS echo-netted IC | retention | verdict |
|---|---|---|---|---|---|
| microprice | prem_okx | +0.2756±0.0678 | +0.1447±0.0259 | 52% | PERSISTS |
| microprice | prem_bin | +0.3059±0.0756 | +0.1725±0.0334 | 56% | PERSISTS |
| microprice | gap_okx-byb | +0.0825±0.0342 | +0.0365±0.0147 | 44% | SHRINKS |
| microprice | gap_bin-byb | +0.0100±0.0394 | +0.0044±0.0133 | 44% | COLLAPSES->echo |
| ofi_normalised | okx | +0.3071±0.0686 | +0.1715±0.0307 | 56% | PERSISTS |
| ofi_normalised | bin | +0.3384±0.0800 | +0.2058±0.0413 | 61% | PERSISTS |
| ofi | bin | +0.3207±0.0785 | +0.1914±0.0404 | 60% | PERSISTS |
| ofi | okx | +0.2945±0.0673 | +0.1610±0.0303 | 55% | PERSISTS |
| price_momentum | okx | +0.2005±0.0602 | +0.0870±0.0216 | 43% | SHRINKS |
| price_momentum | bin | +0.2668±0.0794 | +0.1453±0.0387 | 54% | PERSISTS |
| flow_persistence | bin | +0.2420±0.0688 | +0.1140±0.0263 | 47% | SHRINKS |
| flow_persistence | okx | +0.1783±0.0530 | +0.0645±0.0146 | 36% | SHRINKS |
| xv_book_pressure | gap_okx-byb_QI | +0.0851±0.0232 | +0.0381±0.0096 | 45% | SHRINKS |
| xv_book_pressure | gap_okx-byb_prem | +0.0825±0.0342 | +0.0365±0.0147 | 44% | SHRINKS |
| xv_book_pressure | gap_bin-byb_QI | +0.0185±0.0329 | +0.0072±0.0115 | 39% | COLLAPSES->echo |
| xv_book_pressure | gap_bin-byb_prem | +0.0100±0.0394 | +0.0044±0.0133 | 44% | COLLAPSES->echo |
| trade_rate_surge | okx | +0.0820±0.0243 | +0.0269±0.0079 | 33% | SHRINKS |
| trade_rate_surge | bin | +0.1459±0.0477 | +0.0794±0.0259 | 54% | PERSISTS |
| mid_rate_surge | okx | +0.1027±0.0275 | +0.0463±0.0118 | 45% | SHRINKS |
| mid_rate_surge | bin | +0.1655±0.0479 | +0.1074±0.0325 | 65% | PERSISTS |
| volume_surge | bin | +0.0078±0.0062 | +0.0075±0.0050 | 96% | COLLAPSES->echo |
| volume_surge | okx | -0.0014±0.0041 | -0.0024±0.0030 | 167% | COLLAPSES->echo |
| gap_dynamics | okx | +0.1217±0.0767 | +0.0547±0.0326 | 45% | SHRINKS |
| gap_dynamics | bin | +0.1861±0.0950 | +0.0993±0.0483 | 53% | PERSISTS |
| range_breakout | bin | +0.2555±0.0759 | +0.1250±0.0333 | 49% | SHRINKS |
| range_breakout | okx | +0.1953±0.0562 | +0.0768±0.0173 | 39% | SHRINKS |
| flow_imbalance | bin | +0.1968±0.0489 | +0.0917±0.0136 | 47% | SHRINKS |
| flow_imbalance | okx | +0.1427±0.0381 | +0.0518±0.0092 | 36% | SHRINKS |
