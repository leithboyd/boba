"""microprice — the micro-price-premium feature module for the OSS harness.

Definition (verbatim from notebooks/features/microprice.ipynb / build_microprice.py §2-§3):
the Stoikov micro-price premium per venue, plus the cross-venue premium gap, as a
LiveFrontEMA (a forward-filled LEVEL read with a live front) on the SHARED trade clock.

Per venue's RAW front_levels book (it needs the L1 SIZES; merged_levels is price-only):

    micro = (bid_qty·ask_prc + ask_qty·bid_prc) / (bid_qty + ask_qty)   # cross-weighted
    mid   = (bid_prc + ask_prc) / 2
    prem  = (micro − mid) / mid                                         # bounded by ±halfspread/mid

Two feature families, built RAW (NO σ_ev/λ_ev division — §5 measured the premium
regime-invariant: worst scale ≈1.4× across vol buckets, far under the 3× bar):

  1. per-venue smoothed premium  ema(prem_ex, N)
  2. cross-venue premium gap     ema(prem_other − prem_byb, N)   (other ∈ {okx, bin})

Each leg is a forward-filled LEVEL -> LiveFrontEMA read at each anchor:

    leg(N) = (1−α)·ema(committed, N)[tick_at_anchor] + α·fresh        α = 2/(N+1)

where `committed` is the premium/gap forward-filled onto merged_ts (committed once per
trade-timestamp) and `fresh` is the premium/gap as of the anchor itself (read at every
book update, never frozen on the last trade). N=1 sets α=1 -> the leg collapses to the
fresh premium (the instantaneous, identity read — the span the §6 sweep lands on).

Legs shipped (notebook §6, byb is target + also carries a premium):
  prem_byb, prem_okx, prem_bin, gap_okx-byb, gap_bin-byb.

HEAD = "price": scored vs grid.price_target (a direction feature). The reported
block[0] number is the IN-SAMPLE best span per leg (§6, argmax |Spearman| vs the price
target); on block[0] that is span 1 (the freshest read) for every leg — the §6 map is
monotonic in N. See INTERFACE.md for the on-grid causal-read contract.
"""
import numpy as np
from scipy.signal import lfilter
from scipy.stats import spearmanr

NAME = "microprice"
HEAD = "price"

# venue roster (notebook §2): byb is the target AND carries its own premium.
VENUES = ["byb", "okx", "bin"]
OTHERS = ["okx", "bin"]                       # the venues whose premium GAP vs byb ships
# the EMA span family the notebook sweeps; the reported number is the IN-SAMPLE best
# span per leg vs the price head (§6), which on block[0] lands on span 1 for every leg.
NSPANS = [1, 5, 20, 50, 200, 1000, 5000]
DEFAULT_SPAN = 1                              # the shipped identity span (used if a leg is asked for a fixed span by name)


def _prem_stream(arrays, ex):
    """Each venue's micro-price premium per RAW front_levels row, from the five L1
    columns (bid/ask price AND size). Rows with a non-positive size are dropped — both
    sizes are needed for the cross-weighting (notebook §2). Returns (rx, prem) rx-sorted."""
    rx = arrays.fl_rx[ex]
    bp, bq = arrays.fl_bid_prc[ex], arrays.fl_bid_qty[ex]
    ap, aq = arrays.fl_ask_prc[ex], arrays.fl_ask_qty[ex]
    keep = (bq > 0.0) & (aq > 0.0)                          # need both touch sizes for the weighting
    rx, bp, bq, ap, aq = rx[keep], bp[keep], bq[keep], ap[keep], aq[keep]
    mid = 0.5 * (bp + ap)
    micro = (bq * ap + aq * bp) / (bq + aq)                 # Stoikov micro-price: heavy bid -> pulled toward the ASK
    return rx, (micro - mid) / mid                          # prem = (micro − mid)/mid, bounded by ±halfspread/mid


def _prem_committed(rx, prem, merged_ts):
    """The premium forward-filled onto the trade clock: each venue's most-recent premium
    at-or-before every clock tick (committed leg, causal, piecewise-constant)."""
    idx = np.clip(np.searchsorted(rx, merged_ts, "right") - 1, 0, len(prem) - 1)
    return prem[idx]


def _prem_at(rx, prem, t):
    """The premium forward-filled to arbitrary times — the FRESH read as of each anchor
    (every book update since the last trade, never frozen on the last trade)."""
    idx = np.clip(np.searchsorted(rx, t, "right") - 1, 0, len(prem) - 1)
    return prem[idx]


def _ema(x, N):
    """Trade-clock EMA, α = 2/(N+1). N=1 -> α=1 -> the identity (all weight on the latest tick)."""
    if N == 1:
        return x
    a = 2.0 / (N + 1.0)
    return lfilter([a], [1.0, -(1.0 - a)], x)


def _live_front(committed, fresh, tick_at_anchor, N):
    """LiveFrontEMA read at the grid: (1−α)·(committed EMA at the last trade tick) + α·fresh.
    N=1 collapses to `fresh` (the instantaneous premium/gap)."""
    a = 2.0 / (N + 1.0)
    return (1.0 - a) * _ema(committed, N)[tick_at_anchor] + a * fresh


def _legs_at_span(arrays, grid, N):
    """Build every shipped leg at span N, read on the anchor grid. Returns a dict keyed by
    the notebook leg labels: prem_<venue> and gap_<other>-byb."""
    merged_ts = grid.merged_ts
    tick = grid.tick_at_anchor
    anchors = grid.anchor_ts

    streams = {ex: _prem_stream(arrays, ex) for ex in VENUES}                   # (rx, prem) per venue
    committed = {ex: _prem_committed(rx, p, merged_ts) for ex, (rx, p) in streams.items()}
    fresh = {ex: _prem_at(rx, p, anchors) for ex, (rx, p) in streams.items()}   # premium AS OF each anchor (fresh)

    out = {}
    for ex in VENUES:                                                           # (1) per-venue smoothed premium
        out[f"prem_{ex}"] = _live_front(committed[ex], fresh[ex], tick, N)
    for o in OTHERS:                                                            # (2) cross-venue premium gap (other − byb)
        gap_committed = committed[o] - committed["byb"]
        gap_fresh = fresh[o] - fresh["byb"]
        out[f"gap_{o}-byb"] = _live_front(gap_committed, gap_fresh, tick, N)
    return out


def _leg_labels():
    return [f"prem_{ex}" for ex in VENUES] + [f"gap_{o}-byb" for o in OTHERS]


def best_spans(arrays, grid, head=None):
    """The notebook §6 pick: per leg, the IN-SAMPLE best span vs the head target, chosen by
    the STRENGTH of the predictive power, argmax |Spearman| (the SIGN is the feature's own
    direction, which the model learns — a strongly-negative gap leg is a strong predictor).
    In-sample only — the chosen leg is re-scored OUT-OF-SAMPLE by the harness. Returns
    {leg: span}."""
    target = grid.price_target if (head or HEAD) == "price" else grid.rate_target
    per_span = {N: _legs_at_span(arrays, grid, N) for N in NSPANS}              # build each leg once per span
    out = {}
    for lab in _leg_labels():
        scores = [spearmanr(per_span[N][lab], target).statistic for N in NSPANS]
        out[lab] = NSPANS[int(np.nanargmax(np.abs(scores)))]                    # strongest |IC| span (in-sample pick)
    return out


def compute(arrays, grid, spans=None, head=None):
    """The module contract: return {leg: feature_array_on_grid} for the micro-price premium.

    arrays — BlockArrays (per-venue front_levels book + the shared trade clock).
    grid   — Grid (anchor_ts, tick_at_anchor, merged_ts, price/rate targets).
    spans  — None (default) -> per-leg in-sample best span (notebook §6, the reported
             number); or a {leg: N} dict -> force those fixed spans (the harness uses this
             to FIX block[0]'s pick for the OOS run). A leg absent from the dict falls back
             to DEFAULT_SPAN.

    Returns one SIGNED array per leg (prem_<venue>, gap_<other>-byb), length
    len(grid.anchor_ts), index-aligned with the anchors, read causally at every anchor,
    NaN-where-undefined (the targets/controls are finite on the grid, so NaNs only arise
    before a venue's first book row), shipped RAW (no σ_ev/λ_ev division — §5 measured the
    premium regime-invariant). The model is fed the signed feature for both heads."""
    if spans is None:
        spans = best_spans(arrays, grid, head=head)
    labels = _leg_labels()
    # group legs by their chosen span so each span's legs are built in one pass
    by_span = {}
    for lab in labels:
        by_span.setdefault(int(spans.get(lab, DEFAULT_SPAN)), []).append(lab)
    out = {}
    for N, labs in by_span.items():
        built = _legs_at_span(arrays, grid, N)
        for lab in labs:
            out[lab] = built[lab]
    return {lab: out[lab] for lab in labels}                                    # preserve the notebook leg order
