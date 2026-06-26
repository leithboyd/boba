"""Generate notebooks/features_v2/02_finalize.ipynb — step 2 of feature analysis (finalize), for ANY
feature: it passed screening; now pick the time-scale per head, shape the input, decide single-vs-per-
exchange, and ship.

ONE notebook handles both param shapes via `FeatureSpec.param_kind` (set the `FEATURE` knob and it adapts):
a FAST_SLOW feature (e.g. price_dislocation) sweeps a 2-D fast/slow grid (`ic_grid`, heatmaps); a SINGLE
feature (e.g. ofi_ema) sweeps one span (`ic_scan`, lines). The 1-D/2-D difference is encapsulated in a few
helpers in the setup cell; the analysis cells are kind-agnostic. All the implementation lives in shared,
tested code (boba.research.selection + boba.research.shaping); this notebook is just wiring + a checklist."""
import json
from pathlib import Path

cells = []
def md(s):   cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r"""
# Finalize a feature for the model

The feature **cleared screening** ([`01_screening.ipynb`](01_screening.ipynb)) — computed correctly and
carrying real signal. This is **step 2 — finalize it for the model**: pick the time-scale per head, shape
the feature for the network, decide single-vs-per-exchange, and run the ship checklist.

**One notebook, any feature.** Set `FEATURE` below; the notebook adapts to its `FeatureSpec.param_kind` —
a **fast/slow** feature (`FAST_SLOW`, e.g. `price_dislocation`) sweeps a 2-D grid (heatmaps); a
**single-span** feature (`SINGLE`, e.g. `ofi_ema`) sweeps one span (lines). The 1-D/2-D difference lives in
the helpers in the setup cell; everything below is kind-agnostic.

Wiring only — the engines are shared, tested code: span/head selection (`boba.research.selection`), input
shaping (`boba.research.shaping`). The **method** is in [`METHOD.md`](METHOD.md).

We evaluate on an **event-gated grid** (`build_context(grid_ms=1, active_only=True, hours=2)`): a regular
1 ms grid, keeping only the ms windows that carried an event (book update or trade) from *any* exchange,
over the first 2 h of the block. `grid_ms` sets the resolution; `hours` caps how much of the block is read.
""")

code(r"""
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1)       # every pl.DataFrame: show ALL rows and columns,
pl.Config.set_fmt_str_lengths(1000); pl.Config.set_tbl_width_chars(10_000)   # never truncate strings or by width
from boba.features import base
from boba.features.base import ParamKind
import boba.features.price_dislocation, boba.features.ofi_ema   # register the example features (FAST_SLOW / SINGLE)
from boba.research.screening import build_context, build_family, best_span
from boba.research.selection import fixed_move_targets, ic_grid, ic_scan, per_exchange_vs_single
from boba.research.shaping import shaping_report

FEATURE = "price_dislocation"                                 # <- the feature to finalize; e.g. "ofi_ema" (the notebook adapts)
ctx  = build_context(grid_ms=1, active_only=True, hours=24)   # 1 ms grid, keep only ms windows with an event (any venue); first 24 h
spec = base.get(FEATURE)
IS_2D = spec.param_kind == ParamKind.FAST_SLOW
if IS_2D:                                                   # fast/slow grid: params = (n_fast, n_slow)
    FAST = [1, 10, 50, 200, 500, 1000]; SLOW = [100, 500, 1000, 2000, 5000, 10000]
    GRID = [(nf, ns) for nf in FAST for ns in SLOW if nf < ns]
else:                                                       # single-span family: params = N
    SPANS = sorted([1, 10, 50, 100, 500, 1000, 2000, 5000, 10000]); GRID = SPANS
COUNTS = (1, 3, 5)                                         # fixed mid-move-count horizons for the price-head sweep (§1)
KEYS = spec.keys_for(ctx, GRID[0])                         # the feature's leg keys (per-exchange) — works for any feature
family = build_family(ctx, spec.vectorized, GRID, n_jobs=18)                  # {params: {leg: vector}}
print(f"{FEATURE}  ({spec.param_kind.value})   block {ctx.block}   {len(GRID)} spans   legs {KEYS}")
print(f"{len(ctx.anchor_ts):,} examples (anchors) on the event-gated grid   from {len(ctx.merged_ts):,} trade ticks")

# --- the 1-D/2-D difference lives in these helpers; the analysis cells below are kind-agnostic ---
def show_grids(grids, title):                              # 2-D heatmaps (fast x slow)
    fig, axes = plt.subplots(1, len(grids), figsize=(5.4 * len(grids), 4.0), squeeze=False)
    for ax, (leg, g) in zip(axes[0], grids.items()):
        im = ax.imshow(g, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(SLOW))); ax.set_xticklabels(SLOW); ax.set_xlabel("slow span")
        ax.set_yticks(range(len(FAST))); ax.set_yticklabels(FAST); ax.set_ylabel("fast span")
        ax.set_title(f"{title} — {leg}")
        for i in range(len(FAST)):
            for j in range(len(SLOW)):
                if np.isfinite(g[i, j]): ax.text(j, i, f"{g[i, j]:.3f}", ha="center", va="center", color="w", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); plt.show()

def show_scan(scans, title):                               # 1-D lines (IC vs span)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for leg, arr in scans.items():
        ax.plot(range(len(SPANS)), arr, "o-", label=leg)
    ax.set_xticks(range(len(SPANS))); ax.set_xticklabels(SPANS, rotation=45); ax.set_xlabel("EMA span N")
    ax.axhline(0, color="0.7", lw=0.8); ax.set_ylabel("rank-IC"); ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout(); plt.show()

def ic_breakdown(target, magnitude=False, mirror=None):    # 2-D grid (ic_grid) or 1-D scan (ic_scan), by param_kind
    fn = ic_grid if IS_2D else ic_scan
    return fn(ctx, family, target, magnitude=magnitude, n_jobs=18, mirror=mirror)

def show_breakdown(res, title):
    (show_grids if IS_2D else show_scan)(res, title)

def best_param(res, leg):                                  # (param token, IC) at this leg's in-sample argmax
    a = res[leg]
    if IS_2D:
        i, j = np.unravel_index(np.nanargmax(a), a.shape); return (FAST[i], SLOW[j]), float(a[i, j])
    i = int(np.nanargmax(a)); return SPANS[i], float(a[i])
""")

md(r"""
## 1. Which time-scale per head? — the IC breakdown

Where the signal lives across the span family, per leg (a fast/slow heatmap for a `FAST_SLOW` feature, a
line vs `N` for a `SINGLE` feature):
- **price head, count-conditioned (the `D_k` family)** — the signed feature vs the signed `n`-move
  return, and `|feature|` vs `|n-move return|`, for the `fixed_move_targets`;
- **rate head** — `|feature|` vs the move count.

The price breakdowns score against a *range of fixed mid-move-count* targets (not the 100 ms wall-clock
return). The signed price breakdowns are **mirror-augmented** (the tape reflected through byb's mid via the
feature's own `spec.mirror`; the signed target negates) so the IC is direction-free — see
[`AUTHORING.md`](../../src/boba/features/AUTHORING.md) → Mirror augmentation. (Held-out span *selection*
across blocks is the OOS harness; these in-sample breakdowns pick a representative scale.)
""")

code(r"""
# the IC sweep — count-conditioned price breakdowns over the move-counts + the rate breakdown. No 100 ms
# target. fixed_move_targets / ic_grid / ic_scan are the shared, tested engines (boba.research.selection);
# ic_breakdown / show_breakdown dispatch on param_kind (set up above).
fmt = fixed_move_targets(ctx, COUNTS)                                        # {n: signed n-move return / σ_ev}
signed = {n: ic_breakdown(t, mirror=spec.mirror)     for n, t in fmt.items()}   # signed feature -> signed n-move return (mirror-augmented)
magnit = {n: ic_breakdown(np.abs(t), magnitude=True) for n, t in fmt.items()}   # |feature| -> |n-move return| (sign-blind: no mirror)
rate_res = ic_breakdown(ctx.rate_target, magnitude=True)                        # rate head: |feature| -> move count

for n in fmt:                                                                # all signed breakdowns first ...
    show_breakdown(signed[n], f"price head n={n}: signed -> signed {n}-move return")
for n in fmt:                                                                # ... then all magnitude breakdowns ...
    show_breakdown(magnit[n], f"price head n={n}: |feature| -> |{n}-move return|")
show_breakdown(rate_res, "rate head: |feature| -> move count")               # ... then the rate breakdown
""")

code(r"""
# best span PER LEG per diagnostic — each leg's OWN in-sample argmax (no leg is privileged; a representative
# scale, not an OOS claim). One row per diagnostic; span + its rank-IC per leg. The legs are the feature's
# own keys (`spec.keys_for`) and the span is its param token, so this works for any feature and param_kind.
diagnostics = ([(f"price n={n} signed",    signed[n]) for n in fmt]
             + [(f"price n={n} |feature|", magnit[n]) for n in fmt]
             + [("rate |feature|->count",  rate_res)])
rows = []
for name, res in diagnostics:
    row = {"diagnostic": name}
    for leg in KEYS:
        span, icv = best_param(res, leg)
        row[f"{leg} span"] = str(span); row[f"{leg} IC"] = round(icv, 3)
    rows.append(row)
pl.DataFrame(rows)
""")

code(r"""
# Pick the price-head TARGET from the §1 sweep itself: the fixed-move-count whose signed IC is strongest
# (leg-averaged, at each count's best span). Then use THAT count-conditioned target for the span pick and
# §3 below — NO 100 ms wall-clock target anywhere, so the whole notebook is consistent with the §1 breakdown.
def _peak_ic(res):                                          # a breakdown's leg-averaged best-span IC
    return float(np.nanmean([np.nanmax(res[leg]) for leg in KEYS]))
best_n = max(fmt, key=lambda n: _peak_ic(signed[n]))        # the strongest move-count horizon (price_dislocation: n=5)
price_target = fmt[best_n]                                  # the count-conditioned price-head target (mirror-augmented below)

price_span = best_span(ctx, family, price_target, mirror=spec.mirror)   # span maximising mean rank-IC over the legs
rate_span  = best_span(ctx, family, ctx.rate_target, score_magnitude=True)
print(f"price head: best move-count n={best_n} (peak IC {_peak_ic(signed[best_n]):+.3f})   span {price_span}")
print(f"rate head:  span {rate_span}   (rate target = move count, not a price target)")
""")

md(r"""
## 2. Input shaping for the network

Reshape the feature for the network input — centred, unit-scale, no wild outliers — with the **lightest**
transform that clears the bar. For a **signed** feature we shape the **mirror-augmented** data
(`concat[f, spec.mirror(f)]`): the centre and skew then reflect the feature's up/down symmetry rather than
*this* block's trend — a trending market would otherwise fake a skew and bias the centre, baking the block's
drift into the transform. The transform is still *applied* to the raw feature in production; this only fixes
how its centre/scale are *estimated* (a no-op for an even feature). Same symmetry principle as the §1 IC.
""")

code(r"""
src = KEYS[0]                                                                 # one leg shown; same recipe for each
f = family[price_span][src]
# mirror-augment the shaping data: for a SIGNED feature concat[f, -f] symmetrises it (mean 0, skew 0,
# RMS scale) so the centre/skew reflect the feature's symmetry, not the block's trend. No-op for an even feature.
f_shape = np.concatenate([f, spec.mirror(f)]) if spec.mirror is not None else f
rep = shaping_report(f_shape)
print(rep)
print("recommended transform:", rep.recommended)

ff = f_shape[np.isfinite(f_shape)]
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.4))
axA.hist(ff, bins=120, density=True, color="C0", alpha=.85)
axA.set_yscale("log"); axA.set_xlabel("feature"); axA.set_ylabel("density (log)")
axA.set_title(f"distribution — skew {rep.raw_skew:+.2f}, excess kurt {rep.raw_excess_kurt:.1f}")
q = np.sort(np.random.default_rng(0).standard_normal(len(ff)))                # reference normal quantiles
sub = np.linspace(0, len(ff) - 1, 4000).astype(int)
for name, v in rep.candidates.items():
    axB.plot(q[sub], np.sort(v)[sub], lw=1.4, label=name)
axB.plot([-5, 5], [-5, 5], "k:", lw=1, label="perfect normal")
axB.set_xlim(-5, 5); axB.set_xlabel("normal quantile"); axB.set_ylabel("transformed quantile")
axB.set_title("how to normalise — QQ vs N(0,1)"); axB.legend(fontsize=8)
fig.tight_layout(); plt.show()
""")

md(r"""
## 3. Single exchange or per-exchange?

For a feature that fans out into one leg per venue, does keeping **every** leg add over the single best one,
out-of-sample (against the price-head target chosen above — the best `n`-move horizon, mirror-augmented)?
Keep all (per-exchange) if they genuinely differ; collapse to one if not. We never merge the legs into one
averaged value.
""")

code(r"""
pc = per_exchange_vs_single(ctx, family, price_span, price_target, mirror=spec.mirror)   # the chosen n-move target, not 100 ms
print(f"per-exchange (all legs jointly) {pc['per_exchange']:+.3f}   "
      f"best single ({pc['best_single']['source']}) {pc['best_single']['ic']:+.3f}   "
      f"-> {'keep per-exchange' if pc['adds_over_single'] else 'single exchange suffices'}")
""")

md(r"""
## 4. Ship checklist

- [ ] the streaming (constant-work-per-event) builder, matching the analysis — parity-checked in screening
- [ ] the tests, passing — the feature's parity/oracle/mirror tests + `tests/test_selection.py`, `tests/test_shaping.py`
- [ ] the gate results recorded (with any failures justified) — screening verdict
- [ ] the chosen heads and time-scales written down, with the yardstick spans (`YARDSTICK_N`)
- [ ] the data quirks handled (the right price source per exchange) — in `build_context`

**The finalised recipe:** feed the **signed** feature to **both heads**, **every leg**, at the spans picked
above (price head + rate head), shaped with the recommended transform. Then validate out-of-sample across
blocks (the `tools/oss` harness) before shipping.
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
out = Path(__file__).resolve().parents[1] / "02_finalize.ipynb"   # notebooks/features_v2/02_finalize.ipynb
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out, "with", len(cells), "cells")
