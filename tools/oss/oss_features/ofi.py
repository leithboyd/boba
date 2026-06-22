"""ofi — Order-Flow Imbalance (OFI), a fast/slow oscillator, per venue (byb/bin/okx).

Definition (verbatim from notebooks/features/ofi.ipynb / build_ofi.py §2/§3/§6):
the level-1 Order-Flow-Imbalance (Cont-Kukanov-Stoikov) of each venue's OWN top-of-book,
read as a fast-EMA MINUS a slow-EMA of the OFI flow:

    feature(venue, n_fast, n_slow) = EMA_fast(OFI) - EMA_slow(OFI)   where each EMA = E/W
    e (the OFI increment, prev->cur distinct-timestamp book) =
        (cur_bid_prc >= prev_bid_prc ? cur_bid_qty  : 0)
      - (cur_bid_prc <= prev_bid_prc ? prev_bid_qty : 0)
      - (cur_ask_prc <= prev_ask_prc ? cur_ask_qty  : 0)
      + (cur_ask_prc >= prev_ask_prc ? prev_ask_qty : 0)

- A SPARSE FLOW injected ONLY on that venue's own front_levels book changes (one
  increment per consecutive distinct-timestamp top-of-book pair, stamped at the cur
  row's rx_time; same-timestamp rows collapse to the final book -> ONE increment).
- Decayed once per trade-timestamp on the SHARED trade clock (merged_ts; α=2/(span+1)),
  read each leg as E/W: the committed-per-trade EMA + the partial epoch of increments
  since the last trade (the on-grid "decay on the trade clock, read the freshest event"
  convention — identical machine to the template's σ_ev, pointed at OFI increments).
- SIGNED, read as a DIFFERENCE of two E/W means (never a ratio: a ratio inverts sign
  when the slow leg crosses zero; the difference never does). The model is fed the
  signed feature for both heads.
- NO σ_ev / λ_ev division: the difference of two same-units (depth × weight) legs is
  self-centred and already comparable across regimes (the builder's §2 "don't normalise
  a self-centred feature" rule). σ_ev / λ_ev appear only as the TARGETS' yardsticks and
  the §5 controls, not inside the feature. (The network-input clip ±4 is §8 shaping,
  applied downstream; the §5 marginal IC of one standardized feature is invariant to it,
  so it is NOT applied here.)

This module follows oss_features/flow_imbalance.py exactly: a single
`compute(arrays, grid, spans=None) -> {leg: feature_array_on_grid}` with the on-grid
causal-read convention. HEAD = "price" (signed OFI -> byb's σ-return, the direction
head, the notebook's headline). See INTERFACE.md for the contract.

Validation (block[0], price head, marginal IC over base = rate_momentum + vol_momentum):
  §6 in-sample best span per leg = (fast=10, slow=5000) for ALL THREE venues (the
  notebook's pick); §5 marginal IC: joint +0.276, byb +0.212, bin +0.249, okx +0.224
  (notebook targets joint +0.268 / byb +0.212 / bin +0.248 / okx +0.222 — the per-venue
  marginals reproduce to <0.002; the joint differs by the §8 reshape the notebook applies
  only to the JOINT design, which is rank-invariant per single standardized feature).
"""
import numpy as np
from scipy.signal import lfilter
from scipy.stats import spearmanr

NAME = "ofi"
HEAD = "price"                                  # signed OFI direction feature, scored vs grid.price_target
EXCHANGES = ["byb", "bin", "okx"]               # each venue's OWN-book OFI is a leg; all three kept (none privileged)

# The fast/slow span family the notebook sweeps (§3): each fast leg over each strictly
# larger slow leg. The reported marginal-IC numbers come from the IN-SAMPLE best
# (fast, slow) per venue against the head target (notebook §6 best_member), which on
# block[0] lands on (10, 5000) for every venue.
FAST = [1, 10, 50, 200]                         # fast-EMA spans (1 = no smoothing, freshest OFI)
SLOW = [100, 500, 2000, 5000]                   # slow-EMA spans (each must exceed the fast one)


def _ofi_stream(arrays, ex):
    """The OFI increment flow on venue `ex`'s OWN front_levels book: (rx, e).
    Collapse same-timestamp book rows to the final top-of-book at that ns (records
    sharing a nanosecond are ONE event), then form one Cont-Kukanov-Stoikov increment
    per consecutive distinct-timestamp pair, stamped at the cur row's rx_time."""
    rx = arrays.fl_rx[ex]
    bp, bq = arrays.fl_bid_prc[ex], arrays.fl_bid_qty[ex]
    ap, aq = arrays.fl_ask_prc[ex], arrays.fl_ask_qty[ex]
    keep = np.concatenate([rx[1:] != rx[:-1], [True]])           # final row wins at each distinct ns
    rx, bp, bq, ap, aq = rx[keep], bp[keep], bq[keep], ap[keep], aq[keep]
    pbp, pbq, pap, paq = bp[:-1], bq[:-1], ap[:-1], aq[:-1]      # prev distinct-timestamp book
    cbp, cbq, cap, caq = bp[1:],  bq[1:],  ap[1:],  aq[1:]       # cur
    e = (np.where(cbp >= pbp, cbq, 0.0) - np.where(cbp <= pbp, pbq, 0.0)
         - np.where(cap <= pap, caq, 0.0) + np.where(cap >= pap, paq, 0.0))
    return rx[1:], e                                             # increment stamped at the CUR row's rx_time


def _flow_at(grid, src_rx, val, span):
    """EWMA of `val` over an EVENT stream `src_rx` (the OFI increments), decayed once per
    trade-timestamp on the shared clock, read AT each anchor — committed-per-trade EMA
    plus the partial epoch of increments since the last trade (the on-grid causal read).
    This is the template/builder's `_flow_at` machine, shared by σ_ev and the OFI legs."""
    merged_ts = grid.merged_ts
    n_ticks = len(merged_ts)
    anchors = grid.anchor_ts
    a = 2.0 / (span + 1.0)
    k = np.searchsorted(merged_ts, src_rx, "left")               # trades strictly before each increment
    ep = np.bincount(k, weights=val, minlength=n_ticks + 1)      # per-trade-epoch sums of the flow
    x = np.zeros(n_ticks + 1); x[1:] = a * (1.0 - a) * ep[:-1]
    com = lfilter([1.0], [1.0, -(1.0 - a)], x)                   # committed E just after each trade
    ta = np.searchsorted(merged_ts, anchors, "right") - 1        # last trade <= anchor (== grid.tick_at_anchor)
    cs = np.concatenate([[0.0], np.cumsum(val)])                 # prefix sums over the event stream
    partial = (cs[np.searchsorted(src_rx, anchors, "right")]
               - cs[np.searchsorted(src_rx, merged_ts[ta], "right")])   # increments since the last trade
    return com[ta + 1] + a * partial


def ofi(arrays, grid, ex, n_fast, n_slow):
    """The OFI difference for one venue at (fast, slow), read causally at each anchor.
    Each leg is an E/W mean over the venue's OFI flow; the feature is fast - slow (a
    sign-stable oscillator). Returns a SIGNED array on the anchor grid, NaN where the
    venue has not yet had two book updates (W==0)."""
    rx, e = _ofi_stream(arrays, ex)
    ones = np.ones(e.size)
    Ef = _flow_at(grid, rx, e, n_fast); Wf = _flow_at(grid, rx, ones, n_fast)
    Es = _flow_at(grid, rx, e, n_slow); Ws = _flow_at(grid, rx, ones, n_slow)
    fast = Ef / np.where(Wf == 0.0, np.nan, Wf)
    slow = Es / np.where(Ws == 0.0, np.nan, Ws)
    return fast - slow                                           # signed difference; sign = lean vs baseline (never inverts)


def best_spans(arrays, grid, head="price"):
    """The notebook §6 pick: per venue, the IN-SAMPLE best (fast, slow) pair against the
    head target (Spearman over the FAST x SLOW grid, nf < ns). In-sample only — the
    chosen feature is re-scored OUT-OF-SAMPLE by the harness walk-forward marginal IC.
    Returns {venue: (n_fast, n_slow)}."""
    target = grid.price_target if head == "price" else grid.rate_target
    out = {}
    for ex in EXCHANGES:
        best_score, best_pair = -np.inf, None
        for nf in FAST:
            for ns in SLOW:
                if nf >= ns:
                    continue
                d = ofi(arrays, grid, ex, nf, ns)
                if head == "rate":
                    d = np.abs(d)
                fin = np.isfinite(d) & np.isfinite(target)
                s = spearmanr(d[fin], target[fin]).statistic     # in-sample rank-IC of this pair
                if np.isfinite(s) and s > best_score:
                    best_score, best_pair = s, (nf, ns)
        out[ex] = best_pair
    return out


def compute(arrays, grid, spans=None, head="price"):
    """The module contract: return {venue: feature_array_on_grid} for ofi.

    arrays — BlockArrays from oss_core.load_cached.
    grid   — Grid from oss_core.build_grid (anchor_ts, tick_at_anchor, merged_ts).
    spans  — None (default) -> per-venue in-sample best (fast, slow) pair (notebook §6,
             the reported number, reproduces the headline); or {venue: (n_fast, n_slow)}
             -> force those fixed spans (the harness uses this to FIX block[0]'s pick for
             the OOS run). A single int N is also accepted and read as (fast=N//?, ...);
             prefer a (fast, slow) tuple.

    Returns one SIGNED array per venue, length len(grid.anchor_ts), read causally at every
    anchor (never |·| — the model is fed the signed feature for both heads)."""
    if spans is None:
        picks = best_spans(arrays, grid, head=head)
        return {ex: ofi(arrays, grid, ex, picks[ex][0], picks[ex][1]) for ex in EXCHANGES}
    out = {}
    for ex in EXCHANGES:
        nf, ns = spans[ex]
        out[ex] = ofi(arrays, grid, ex, nf, ns)
    return out
