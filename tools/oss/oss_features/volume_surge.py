"""volume_surge — a feature module for the OSS harness (sibling of flow_imbalance).

Definition (verbatim from notebooks/features/volume_surge.ipynb / build_volume_surge.py §2-§3):
each venue's recent traded VOLUME relative to its own slower baseline — a fast/slow ratio of two
self-normalising E/W EMAs of qty, per venue (byb/bin/okx), on the SHARED trade clock:

    volume_surge(venue, n_fast, n_slow) = (E/W)_fast(qty) / (E/W)_slow(qty)

where each (E/W)_N(qty) is the exp-weighted mean qty per that-venue trade at span N:
  - inject Σ qty of that venue's prints AT each venue-trade-timestamp (simultaneous prints summed
    into ONE event — searchsorted maps each unique venue timestamp to its clock tick),
  - decay once per SHARED trade-timestamp (α = 2/(N+1)),
  - read E = exp-weighted Σ qty, W = exp-weighted count -> E/W = mean qty per venue-trade.
The ratio fast/slow is **> 1** when the venue is trading heavier than its slow baseline (a surge),
**< 1** when it has gone quiet relative to baseline.

On-grid read (the INTERFACE convention, pattern 1 — trade-clock EMA, committed-only):
each E/W leg is piecewise-constant between THAT venue's trades, so the committed value is read at
`grid.tick_at_anchor` (the last shared-clock tick at-or-before each anchor). No live front is used
for a sparse flow — exactly the notebook's `_ew_flow_at`, which reads
`E[searchsorted(merged_ts, anchors, "right") - 1] == E[grid.tick_at_anchor]`.

HEAD — "rate". volume_surge is primarily a rate-head (intensity) feature: a burst of trading volume
is a busy-market signal that should precede more byb mid-moves. The reported §6 number is the
RATE-head marginal IC, with the per-venue best (fast,slow) span chosen IN-SAMPLE against
grid.rate_target (notebook §6 `rate_member`).

NORMALISATION — none. The feature is a RATIO of two EMAs of the same quantity, so the absolute size
units cancel; it is already comparable across regimes (the guard rail: don't normalise a ratio
reflexively — §2 / §5 of the notebook). It is shipped raw (the SIGNED value fed to the model is the
raw ratio; the notebook's rate-head gate scores the raw level, and only the price-head DIAGNOSTIC
centres it as `surge - 1`).
"""
import numpy as np
from scipy.signal import lfilter
from scipy.stats import spearmanr

NAME = "volume_surge"
HEAD = "rate"                      # intensity feature: scored vs grid.rate_target
EXCHANGES = ["byb", "bin", "okx"]  # per-venue: each venue's OWN traded volume vs its OWN baseline

# The fast/slow span family the notebook sweeps (EMA memory in trades). A "span" per leg is a
# (fast, slow) PAIR; the reported §6 number uses the IN-SAMPLE best pair per venue against the
# rate-head target. span=1 (α=1) is DEGENERATE for an E/W flow (fully decays to 0 each tick), so
# the smallest useful fast span is 2.
FAST = [2, 10, 50, 200]
SLOW = [100, 500, 2000, 5000]
PAIRS = [(nf, ns) for nf in FAST for ns in SLOW if nf < ns]   # valid (fast,slow) members


def _qty_per_ts(arrays, ex):
    """Per-venue total qty per trade-TIMESTAMP: simultaneous prints summed into ONE event.
    Returns (unique_rx, summed_qty) — exactly the notebook's qty_ts[ex]."""
    rx, qty = arrays.tr_rx[ex], arrays.tr_qty[ex]
    u, inv = np.unique(rx, return_inverse=True)
    return u, np.bincount(inv, weights=qty)


def _ew_flow(arrays, val_rx, val, N):
    """Committed E of a KernelMeanEMA flow on the SHARED clock at span N, one value per clock tick.
    add(a*val) at each venue-trade tick (BEFORE that tick's decay), then E_t = (1-a)*(E_{t-1}+inj_t)
    -> lfilter on (1-a)*inj. Returns E over merged_ts (flat between this venue's trades).
    Verbatim from the notebook `_ew_flow_at` (minus the final grid gather)."""
    merged_ts = arrays.merged_ts
    n_ticks = len(merged_ts)
    a = 2.0 / (N + 1.0)
    kt = np.searchsorted(merged_ts, val_rx, "left")          # shared-clock tick index of each venue-trade timestamp
    inj = np.zeros(n_ticks)
    np.add.at(inj, kt, a * val)                              # injection a*val at that tick (events sit AT a tick)
    return lfilter([1.0], [1.0, -(1.0 - a)], (1.0 - a) * inj)


def surge(arrays, grid, ex, n_fast, n_slow):
    """The volume-surge ratio for one venue at (n_fast, n_slow), read committed at each anchor's last
    clock tick (causal, piecewise-constant between this venue's trades). Array on the anchor grid.
    SIGNED (raw ratio), nan where W==0 (the venue has not yet traded)."""
    u, q = _qty_per_ts(arrays, ex)
    ones = np.ones_like(q)
    ef = _ew_flow(arrays, u, q, n_fast)
    wf = _ew_flow(arrays, u, ones, n_fast)
    es = _ew_flow(arrays, u, q, n_slow)
    ws = _ew_flow(arrays, u, ones, n_slow)
    fast = ef / np.where(wf > 0.0, wf, np.nan)               # E/W_fast: mean qty per venue-trade (fast); nan until traded
    slow = es / np.where(ws > 0.0, ws, np.nan)               # E/W_slow
    ratio = fast / slow                                      # > 1 = trading heavier than baseline
    return ratio[grid.tick_at_anchor]                        # committed value as of the last clock tick <= anchor


def best_spans(arrays, grid, head=HEAD):
    """The notebook §6 pick: per venue, the IN-SAMPLE best (fast,slow) pair by strongest |Spearman IC|
    against the head target — `rate_member` (rate head, the feature's home) / `price_member` (price).
    In-sample only; the chosen feature is re-scored OUT-OF-SAMPLE by the harness. Returns {venue: (fast,slow)}."""
    target = grid.rate_target if head == "rate" else grid.price_target
    out = {}
    for ex in EXCHANGES:
        best_pair, best_abs = None, -np.inf
        for nf, ns in PAIRS:
            d = surge(arrays, grid, ex, nf, ns)
            sc = spearmanr(d, target).statistic
            if np.isfinite(sc) and abs(sc) > best_abs:       # strongest |IC| cell (sign carried through)
                best_abs, best_pair = abs(sc), (nf, ns)
        out[ex] = best_pair
    return out


def compute(arrays, grid, spans=None):
    """The module contract: return {venue: feature_array_on_grid} for volume_surge.

    arrays — BlockArrays (per-venue trades + the shared trade clock merged_ts).
    grid   — Grid (anchor_ts, tick_at_anchor, merged_ts, rate_target/price_target).
    spans  — None (default) -> per-venue IN-SAMPLE best (fast,slow) pair against the HEAD target
             (notebook §6 `rate_member`, the reported number); or {venue: (fast,slow)} -> force that
             fixed pair per venue (the harness uses this to FIX block[0]'s pick for the OOS run).
             A scalar/2-tuple may also be passed to force one pair for every venue.

    Returns one SIGNED array per venue (the raw ratio — no σ/λ normalisation, a ratio is already
    unit-free), length len(grid.anchor_ts), read causally at every anchor, nan before that venue's
    first trade."""
    if spans is None:
        spans = best_spans(arrays, grid, head=HEAD)
    elif isinstance(spans, dict):
        spans = dict(spans)
    else:
        spans = {ex: tuple(spans) for ex in EXCHANGES}       # one (fast,slow) pair forced for every venue
    return {ex: surge(arrays, grid, ex, spans[ex][0], spans[ex][1]) for ex in EXCHANGES}
