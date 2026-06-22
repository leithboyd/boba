# OSS harness — feature-module `compute()` contract

The OSS harness re-measures each feature's marginal value across all 58 ETH-perp
blocks (purged walk-forward) instead of one, reusing the exact scaffold the
single-block notebooks (`notebooks/features/template.ipynb`,
`.../flow_imbalance.ipynb`) build for block[0].

A **feature module** plugs into `oss_core.run_oss` by exposing two things:

```python
NAME = "flow_imbalance"                 # str — the feature's name (used as the results key)
def compute(arrays, grid) -> dict       # {venue: feature_array_on_grid}
```

`oss_features/flow_imbalance.py` is the reference module — copy its shape.

---

## Inputs

### `arrays` — `oss_core.BlockArrays`
Per-venue raw event arrays for one block. Dicts keyed by short venue code
(`"bin"`, `"byb"`, `"okx"`). All timestamps are **int64 ns**; prices/qtys float64.

| field | type | meaning |
|---|---|---|
| `fl_rx[ex]` | int64[] | front_levels receive timestamps (rx_time), sorted |
| `fl_bid_prc[ex]`, `fl_bid_qty[ex]` | float64[] | best-bid price / qty per snapshot |
| `fl_ask_prc[ex]`, `fl_ask_qty[ex]` | float64[] | best-ask price / qty per snapshot |
| `tr_rx[ex]` | int64[] | trade receive timestamps (prc>0 & qty>0 only) |
| `tr_prc[ex]`, `tr_qty[ex]` | float64[] | trade price / qty |
| `tr_sign[ex]` | float64[] | **+1** if the trade lifts the ask (buy), **−1** if it hits the bid (sell) — venue-specific aggressor via `io._trade_lifts_ask` |
| `byb_rx`, `byb_mid` | int64[], float64[] | byb **merged** mid stream (front_levels fused with trades, newest-by-exchange-time), same-timestamp rows collapsed to ONE final mid |
| `merged_ts` | int64[] | the shared **trade clock**: `np.unique` of all-venue trade rx (one tick per trade-timestamp; simultaneous prints = one tick) |

For a venue mid other than byb's, call `oss_core.mid_stream(arrays, ex)` (byb/okx →
`merged_levels`, bin → `front_levels`, per the boba.io policy).

### `grid` — `oss_core.Grid`
The causal evaluation scaffold. All `*_target` / control / yardstick arrays have
length `len(grid.anchor_ts)` (≈1.7M on block[0]) and are aligned to the anchors.

| field | meaning |
|---|---|
| `anchor_ts` | int64[] — grid anchor timestamps (ns), 50 ms grid, past a 50000-tick warmup, leaving a 100 ms forward window |
| `tick_at_anchor` | int64[] — **index into `merged_ts`** of the last trade-clock tick at-or-before each anchor (the causal read index) |
| `merged_ts` | the trade clock (same array as `arrays.merged_ts`) |
| `sigma_ev`, `lambda_ev` | byb's vol RMS-per-move and move-rate (moves/sec) at each anchor, span `YARDSTICK_N=10000` |
| `price_target` | byb 100 ms log return ÷ σ_ev (price head) |
| `rate_target` | byb 100 ms mid-move count ÷ λ_ev (rate head) |
| `controls` | dict: `rate_level`, `rate_momentum`, `vol_level`, `vol_momentum` (all causal, on the grid) |

The harness `base` controls (what the §5 marginal IC adds over) are
**`[rate_momentum, vol_momentum]`** — the two momenta. (`*_level` are the extra
controls used only for the no-leak gate.)

---

## Return shape

`compute(arrays, grid)` returns a **dict `{venue: feature_array}`**:

- one entry per venue the feature covers (`flow_imbalance` returns all three: `bin`,
  `byb`, `okx`);
- each value is a 1-D float array of length **`len(grid.anchor_ts)`** — one value per
  anchor, **aligned index-for-index** with `grid.anchor_ts` / the targets / the
  controls;
- the array is the **SIGNED** feature (never `|·|`). The harness feeds the signed
  feature to both heads; the rate head learns the magnitude itself.
- `np.nan` is allowed where the feature is undefined (e.g. before a venue's first
  trade); the walk-forward marginal IC masks non-finite rows.

`run_oss` then computes, per block, the **joint** marginal IC (all the venue arrays
added together over the controls) and the **per-venue** marginal IC (each alone).

---

## The on-grid causal-read convention

Every feature is computed on the venue's own native events, then **read at each grid
anchor using only data at-or-before that anchor** — no peeking. Two standard patterns:

1. **Trade-clock EMA (sparse flow, e.g. flow_imbalance).** Build the feature as an
   `lfilter` recursion over the per-tick injected mass on `merged_ts` (decay once per
   trade-timestamp, `α = 2/(span+1)`), then gather the committed value at
   **`grid.tick_at_anchor`**:
   ```python
   ratio = E / W                      # one value per clock tick (merged_ts)
   feature_on_grid = ratio[grid.tick_at_anchor]   # causal read at each anchor
   ```
   This is piecewise-constant between the venue's own trades — exactly what
   `tick_at_anchor` (last tick ≤ anchor) returns.

2. **Forward-filled level (e.g. a price/gap, a live-front EMA).** Compute the level on
   `merged_ts` (committed legs) and additionally read the **fresh** value as of the
   anchor (`np.searchsorted(rx, anchor_ts, "right") - 1`), combining them as the
   notebook's `LiveFrontEMA` does: `(1−α)·committed[tick_at_anchor] + α·fresh`.

**Rules (from the notebook guard rails):**
- Same-timestamp prints are **one event** — sum their mass into a single injection;
  the clock advances **once** per timestamp.
- Sparse flows are read as `E/W` (two trade-clock EMAs) so non-events cancel — never
  push a 0 on a non-event.
- Decay rides the **trade clock**; the read rides the freshest event. Never freeze on
  a stale last-trade value for a level.

---

## Span selection (the reported number)

`flow_imbalance.compute(arrays, grid)` defaults to the **in-sample best span per
venue** against the head target (notebook §6: `best_spans`), which on block[0] lands
on span 5 and reproduces the reported **joint +0.158 / bin +0.138**. The selection is
in-sample only; the chosen feature is then scored **out-of-sample** by the harness's
purged+embargoed walk-forward marginal IC. Pass `compute(arrays, grid, span=N)` to
force a single fixed span (the oracle uses `span=100`).

A sibling module may instead return a single fixed-construction array per venue if it
has no span family — the only hard requirement is the `{venue: array_on_grid}` shape
above.

---

## Wiring it up

```python
import oss_core as core
from oss_features import flow_imbalance as fi

results = core.run_oss([0, 1, 2], [fi], head="price")
# results["flow_imbalance"][block_idx] = {"joint": float, "per_venue": {venue: float}}
```

Each block's arrays + grid are computed once and cached as a compressed npz under
`cache/` (≈236 MB/block), so re-runs over many blocks are cheap.
```
```

## Validation (block[0], reproduced by `validate.py`)

1. flow_imbalance PRICE-head marginal IC over controls: **joint +0.1585** (target
   +0.158), **bin +0.1385** (target +0.138) — within 0.0005.
2. grid anchors **1,706,369** (~1.7M); targets 100% finite; σ_ev, λ_ev > 0.
3. `run_oss([0,1,2], [flow_imbalance])` runs end-to-end → joint **+0.158 / +0.162 /
   +0.184**.
Plus a bit-exact oracle (independent KernelMean streaming loop vs the production E/W
EMA, worst |diff| ≈ 9e-16 over 40k anchors per venue on the real block).
