"""xv_book_pressure — cross-exchange book-pressure GAP feature module for the OSS harness.

Definition (verbatim from notebooks/features/xv_book_pressure.ipynb / build_xv_book_pressure.py
§2/§3): per other venue o ∈ {okx, bin} and per bounded top-of-book atom A ∈ {QI, prem}, the
cross-venue gap A(o) − A(byb), smoothed on byb's shared trade clock as a LiveFrontEMA.

The two bounded atoms, read from each venue's front_levels book (bid/ask price AND size — the
atoms need resting quantity, which merged_levels omits):

    QI(ex)   = (bid_qty − ask_qty) / (bid_qty + ask_qty)                       ∈ [−1, 1]
    micro    = (bid_qty·ask_prc + ask_qty·bid_prc) / (bid_qty + ask_qty)       (size-weighted fair px)
    prem(ex) = (micro − mid) / mid,   mid = (bid_prc + ask_prc)/2              (tiny dimensionless)

The feature per (source o, atom A, span N):

    feature[o, A, N] = LiveFrontEMA_N( A(o) − A(byb) )
                     = (1 − α)·committed[tick_at_anchor] + α·fresh_gap_at_anchor,   α = 2/(N+1)

  - committed leg: the cross-venue gap sampled on the SHARED trade clock (merged_ts), decayed
    once per trade-timestamp (lfilter, α = 2/(N+1), y[-1]=0). N=1 ⇒ α=1 ⇒ committed = the gap
    itself (no smoothing) and the live front is all-fresh — the shipped raw path.
  - fresh leg: the same gap as of each anchor (each venue's most-recent book at-or-before the
    anchor — every book update, never stale), read on grid.anchor_ts.

NORMALISATION: ships RAW. §5's scale gate measured the gap regime-invariant (a difference of
BOUNDED atoms: QI ∈ [−1,1], prem a tiny ratio), ≈1.1–1.4× across vol buckets — well under the
~3× hard gate — so NO /σ_ev, NO /λ_ev. The TARGET is divided by the yardsticks; the feature is not.

SIGNED feature, fed to both heads (the rate head learns the magnitude itself).

Legs (the four members the notebook ships, symmetric, no privileged source/atom):
    gap_okx-byb_QI, gap_okx-byb_prem, gap_bin-byb_QI, gap_bin-byb_prem

See INTERFACE.md for the {leg: array_on_grid} contract.
"""
import numpy as np
from scipy.signal import lfilter
from scipy.stats import spearmanr

NAME = "xv_book_pressure"
HEAD = "price"                     # a DIRECTION feature: scored vs grid.price_target

TARGET_VENUE = "byb"
SOURCES = ["okx", "bin"]           # the OTHER venues; each one's atom gap vs byb is a feature (no leader)
ATOMS = ["QI", "prem"]             # the two bounded book-pressure atoms
# the trade-span family the notebook sweeps (EMA memory in trades); 1 = no smoothing (the freshest gap).
SPANS = [1, 5, 20, 100, 500, 2000, 8000]
DEFAULT_SPAN = 100                 # a single fixed span (used only by the bit-exact oracle check)

# the leg key the harness ships, one per (source, atom) member.
MEMBERS = [(o, A) for o in SOURCES for A in ATOMS]


def _leg(o, A):
    return f"gap_{o}-byb_{A}"


def _book(arrays, ex):
    """The venue's top-of-book event stream (rx, bid_prc, bid_qty, ask_prc, ask_qty), with
    same-TIMESTAMP rows collapsed to the final book at that ns (one event per timestamp)."""
    rx = arrays.fl_rx[ex]
    bp, bq = arrays.fl_bid_prc[ex], arrays.fl_bid_qty[ex]
    ap, aq = arrays.fl_ask_prc[ex], arrays.fl_ask_qty[ex]
    keep = np.concatenate([rx[1:] != rx[:-1], [True]])   # last row at each distinct ns wins
    return rx[keep], bp[keep], bq[keep], ap[keep], aq[keep]


def _atoms_of(arrays, ex):
    """The two bounded atoms (QI, prem) from a venue's book event stream. Returns (rx, {atom: vals})."""
    rx, bp, bq, ap, aq = _book(arrays, ex)
    tot = bq + aq
    qi = (bq - aq) / tot                                 # queue imbalance ∈ [−1, 1]
    mid = 0.5 * (bp + ap)
    micro = (bq * ap + aq * bp) / tot                    # size-weighted fair price (leans to the heavier side)
    prem = (micro - mid) / mid                           # micro-price premium (dimensionless, tiny)
    return rx, {"QI": qi, "prem": prem}


def _atom_on(rx, vals, t):
    """The venue's most-recent atom value at-or-before times t (causal forward-fill)."""
    return vals[np.clip(np.searchsorted(rx, t, "right") - 1, 0, len(vals) - 1)]


def _gaps(arrays, grid):
    """Per (source, atom): the committed gap on the trade clock and the fresh gap at each anchor.
    Returns (gap_committed, gap_fresh) dicts keyed by (o, A)."""
    merged_ts = arrays.merged_ts
    anchor_ts = grid.anchor_ts
    rxv, valv = {}, {}
    for ex in [TARGET_VENUE] + SOURCES:
        rxv[ex], valv[ex] = _atoms_of(arrays, ex)
    gap_committed, gap_fresh = {}, {}
    for o in SOURCES:
        for A in ATOMS:
            byb_c = _atom_on(rxv["byb"], valv["byb"][A], merged_ts)
            byb_f = _atom_on(rxv["byb"], valv["byb"][A], anchor_ts)
            gap_committed[(o, A)] = _atom_on(rxv[o], valv[o][A], merged_ts) - byb_c
            gap_fresh[(o, A)] = _atom_on(rxv[o], valv[o][A], anchor_ts) - byb_f
    return gap_committed, gap_fresh


def _ema_commit(g, N):
    """Committed per-trade EMA of the gap on the trade clock (α = 2/(N+1), y[-1]=0).
    N=1 ⇒ α=1 ⇒ all weight on the latest tick (no smoothing)."""
    if N == 1:
        return g
    a = 2.0 / (N + 1.0)
    return lfilter([a], [1.0, -(1.0 - a)], g)


def book_pressure(gap_committed, gap_fresh, grid, o, A, N):
    """The LiveFrontEMA read of the (o, A) atom gap at span N, on the anchor grid:
    (1 − α)·committed-at-last-trade + α·fresh-gap-at-anchor — current between trades, never stale."""
    a = 2.0 / (N + 1.0)
    committed_at_anchor = _ema_commit(gap_committed[(o, A)], N)[grid.tick_at_anchor]
    return (1.0 - a) * committed_at_anchor + a * gap_fresh[(o, A)]


def best_spans(arrays, grid, head="price"):
    """The notebook §6 pick: per member (source, atom), the IN-SAMPLE best span by |IC| against
    the head target (Spearman). |IC| — NOT signed argmax — because the model is fed the SIGNED
    feature, so a consistently-signed predictor is equally useful at either sign (this gap predicts
    byb direction with a stable NEGATIVE sign, so plain argmax would wrongly pick the weakest span).
    Returns {leg: span}."""
    target = grid.price_target if head == "price" else grid.rate_target
    gap_committed, gap_fresh = _gaps(arrays, grid)
    out = {}
    for o, A in MEMBERS:
        scores = []
        for N in SPANS:
            d = book_pressure(gap_committed, gap_fresh, grid, o, A, N)
            fin = np.isfinite(d) & np.isfinite(target)
            if head == "rate":
                d = np.abs(d)
            scores.append(abs(spearmanr(d[fin], target[fin]).statistic) if fin.any() else np.nan)
        j = int(np.nanargmax(scores)) if np.isfinite(scores).any() else 0
        out[_leg(o, A)] = SPANS[j]
    return out


def compute(arrays, grid, spans=None, head=None):
    """The module contract: return {leg: feature_array_on_grid} for xv_book_pressure.

    arrays — BlockArrays (per-venue front_levels book + the shared trade clock).
    grid   — Grid (anchor_ts, tick_at_anchor, merged_ts, targets, controls).
    spans  — None (default) -> per-leg IN-SAMPLE best span (notebook §6, reproduces the headline);
             or a {leg: N} dict -> force that fixed span per leg (the harness uses this to FIX
             block[0]'s pick for the OOS run). An int forces that span for every leg.

    Returns one SIGNED, RAW array per leg (the four source×atom gaps), length len(grid.anchor_ts),
    read causally at every anchor (never |·|; ships raw — no /σ_ev, no /λ_ev)."""
    if head is None:
        head = HEAD
    gap_committed, gap_fresh = _gaps(arrays, grid)
    if spans is None:
        chosen = best_spans(arrays, grid, head=head)
    elif isinstance(spans, dict):
        chosen = {_leg(o, A): spans.get(_leg(o, A), DEFAULT_SPAN) for o, A in MEMBERS}
    else:
        chosen = {_leg(o, A): int(spans) for o, A in MEMBERS}
    return {_leg(o, A): book_pressure(gap_committed, gap_fresh, grid, o, A, chosen[_leg(o, A)])
            for o, A in MEMBERS}
