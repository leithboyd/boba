"""ofi_normalised — per-venue order-flow-imbalance EMA feature module for the OSS harness.

Definition (verbatim from notebooks/features/ofi_normalised.ipynb / build_ofi_normalised.py
§2/§3): the level-1 Cont–Kukanov–Stoikov order-flow imbalance (OFI) of each venue's
OWN top-of-book, smoothed by a single trade-clock E/W KernelMean EMA, read as the SAME
reading the notebook ships per leg (one of: raw / σ_ev-normalised / λ_ev-normalised —
picked in-sample off the §6 grid, exactly like flow_imbalance picks its span):

    OFI_e[ex] = (cur_bid_qty if cur_bid_prc>=prev_bid_prc else 0)        # bid side
              - (prev_bid_qty if cur_bid_prc<=prev_bid_prc else 0)
              - (cur_ask_qty if cur_ask_prc<=prev_ask_prc else 0)        # ask side
              + (prev_ask_qty if cur_ask_prc>=prev_ask_prc else 0)
    ema(OFI_ex) = EMA(OFI_e) / EMA(1)                                    # E/W per-book-update mean OFI
    feature(ex) = ema(OFI_ex) / yardstick                               # yardstick ∈ {1, σ_ev, λ_ev}

- OFI is computed from venue `ex`'s OWN front_levels (the only stream carrying bid_qty /
  ask_qty); same-`rx_time` snapshot bursts are collapsed to ONE row (the final top-of-book)
  before forming consecutive-pair OFI — same-timestamp prints are one event.
- Injected ONLY on that venue's own book updates (weight 1 → W counts updates), decayed
  once per trade-timestamp on the SHARED clock (α = 2/(N+1)). E/W self-normalises so the
  in-between foreign-event decay cancels (per-book-update mean, not per-trade).
- Read committed at the last clock tick at-or-before each anchor (causal, piecewise-constant
  between this venue's book updates).
- Three readings of the SAME EMA are swept and the per-leg best is picked IN-SAMPLE (notebook
  §6 `best_cell_for`): un-normalised `ema(OFI)` (a size), `/σ_ev`, `/λ_ev`. The σ_ev/λ_ev are
  byb's regime yardsticks (the SAME the targets use), read at each anchor from the grid.

Per the notebook §10 verdict, the in-sample winner per leg on block[0] is the un-normalised
reading (`none`), span 10, on every venue — "don't normalise reflexively." The selection is
in-sample only; the chosen reading is then re-scored OUT-OF-SAMPLE by the harness's purged,
embargoed walk-forward marginal IC. See INTERFACE.md for the on-grid causal-read contract.
"""
import numpy as np
from scipy.signal import lfilter
from scipy.stats import spearmanr

NAME = "ofi_normalised"
HEAD = "price"                                  # direction feature: scored vs grid.price_target
EXCHANGES = ["byb", "okx", "bin"]              # byb = own-book leg; okx / bin = cross-venue lead/lag legs
# The OFI-EMA trade-span family the notebook sweeps (EMA memory in trades). The reported
# §6 marginal-IC numbers come from the IN-SAMPLE best (variant, span) per venue against the
# price head, which on block[0] lands on the un-normalised reading at span 10 for every venue.
SPANS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
# The three readings of the same OFI EMA the notebook §6 compares (and picks per leg).
VARIANTS = ["none", "sigma", "lambda"]


def _flow_at(merged_ts, n_ticks, anchors, val, rx_inj, span):
    """EWMA of `val` over an event stream `rx_inj` (committed E), decayed once per trade-
    timestamp on the SHARED clock, read AT each anchor — the §2 `_flow_at` machine (verbatim
    from the template). Returns the committed-E read at each anchor; pair two of these (the
    OFI flow and an all-ones flow) to form the E/W KernelMean ratio."""
    a = 2.0 / (span + 1.0)
    k = np.searchsorted(merged_ts, rx_inj, "left")                     # trades strictly before each event
    ep = np.bincount(k, weights=val, minlength=n_ticks + 1)            # per-trade-epoch sums
    x = np.zeros(n_ticks + 1)
    x[1:] = a * (1.0 - a) * ep[:-1]
    com = lfilter([1.0], [1.0, -(1.0 - a)], x)                        # committed E just after each trade
    ta = np.searchsorted(merged_ts, anchors, "right") - 1            # last trade <= anchor
    cs = np.concatenate([[0.0], np.cumsum(val)])                      # prefix sums over the event stream
    partial = cs[np.searchsorted(rx_inj, anchors, "right")] - cs[np.searchsorted(rx_inj, merged_ts[ta], "right")]
    return com[ta + 1] + a * partial


def _ofi_events(arrays, ex):
    """The OFI flow for venue `ex`: (rx, e) per book update on its OWN front_levels.

    Collapse same-`rx_time` snapshot bursts to ONE row (the final top-of-book) — a same-instant
    burst is one update, not a sequence of phantom moves — then form the level-1
    Cont–Kukanov–Stoikov `e` on each consecutive top-of-book pair (prev -> cur), stamped at
    cur's rx_time. The first row has no prev -> dropped."""
    frx0 = arrays.fl_rx[ex]
    bp0, bq0 = arrays.fl_bid_prc[ex], arrays.fl_bid_qty[ex]
    ap0, aq0 = arrays.fl_ask_prc[ex], arrays.fl_ask_qty[ex]
    fl_keep = np.concatenate([frx0[1:] != frx0[:-1], [True]])         # last row per rx_time = ONE event per timestamp
    frx, bp, bq, ap, aq = frx0[fl_keep], bp0[fl_keep], bq0[fl_keep], ap0[fl_keep], aq0[fl_keep]
    ofi_rx = frx[1:]                                                  # OFI for cur row is stamped at cur's rx_time
    ofi_e = (np.where(bp[1:] >= bp[:-1], bq[1:], 0.0) - np.where(bp[1:] <= bp[:-1], bq[:-1], 0.0)
             - np.where(ap[1:] <= ap[:-1], aq[1:], 0.0) + np.where(ap[1:] >= ap[:-1], aq[:-1], 0.0))
    return ofi_rx, ofi_e


def ema_ofi(arrays, grid, ex, N):
    """ema(OFI_ex) = E/W: the KernelMean EMA over venue ex's OFI flow at span N, decayed on the
    SHARED trade clock, read at each anchor (causal, piecewise-constant). Returns an array on
    the anchor grid (the per-book-update mean OFI — un-normalised, a raw size)."""
    merged_ts = arrays.merged_ts
    n_ticks = len(merged_ts)
    anchors = grid.anchor_ts
    ofi_rx, ofi_e = _ofi_events(arrays, ex)
    E = _flow_at(merged_ts, n_ticks, anchors, ofi_e, ofi_rx, N)                  # exp-weighted OFI
    W = _flow_at(merged_ts, n_ticks, anchors, np.ones_like(ofi_e), ofi_rx, N)    # exp-weighted book-update count
    return E / np.maximum(W, 1e-12)                                              # per-book-update mean OFI


def _apply_variant(raw, grid, variant):
    """Apply one of the three §3 readings to the un-normalised ema(OFI): `none` (raw size),
    `/σ_ev` (vol yardstick), `/λ_ev` (rate yardstick). σ_ev/λ_ev are byb's regime yardsticks,
    read at each anchor from the grid (the SAME the targets use, for every venue's leg)."""
    if variant == "none":
        return raw
    if variant == "sigma":
        return raw / grid.sigma_ev
    if variant == "lambda":
        return raw / grid.lambda_ev
    raise ValueError(f"unknown variant {variant!r}")


def reading(arrays, grid, ex, N, variant):
    """One venue's feature at (span N, variant): ema(OFI_ex) ÷ yardstick, on the anchor grid.
    SIGNED (never |·|) and NaN-where-undefined (before the venue's first book pair)."""
    raw = ema_ofi(arrays, grid, ex, N)
    raw = np.where(np.isfinite(raw), raw, np.nan)            # nan before this venue has a book pair
    return _apply_variant(raw, grid, variant)


def best_reading(arrays, grid, ex, fixed_span=None):
    """The notebook §6 `best_cell_for` pick for one venue: the (variant, span) with the
    strongest IN-SAMPLE |Spearman IC| against the price-head target, swept over the three
    variants × the span family (or, if `fixed_span` is given, over the three variants at that
    one span — used to FIX block[0]'s span for the OOS run while still shipping the same per-leg
    reading the notebook chose). Returns (variant, span, feature_array, in_sample_ic)."""
    target = grid.price_target
    spans = [fixed_span] if fixed_span is not None else SPANS
    best = None
    for N in spans:
        raw = ema_ofi(arrays, grid, ex, N)
        raw = np.where(np.isfinite(raw), raw, np.nan)
        for variant in VARIANTS:
            d = _apply_variant(raw, grid, variant)
            ok = np.isfinite(d) & np.isfinite(target)
            if ok.sum() <= 100:
                continue
            ic = spearmanr(d[ok], target[ok]).statistic
            if best is None or abs(ic) > abs(best[3]):
                best = (variant, N, d, ic)
    return best


def compute(arrays, grid, spans=None):
    """The module contract: return {leg: feature_array_on_grid} for ofi_normalised.

    arrays — BlockArrays from oss_core (per-venue front_levels + trades + the shared trade clock).
    grid   — Grid from oss_core (anchor_ts, tick_at_anchor, merged_ts, sigma_ev, lambda_ev, targets).
    spans  — None (default) -> per-leg in-sample best (variant, span) off the §6 grid (the
             reported number); or {leg: N} -> fix that leg's span to N (block[0]'s OOS pick),
             still selecting the per-leg reading (variant) in-sample at that fixed span the way
             the notebook ships it.

    Returns one SIGNED array per leg (byb / okx / bin), length len(grid.anchor_ts), read causally
    at every anchor, NaN-where-undefined — never |·| (the model is fed the signed feature for both
    heads; the rate head learns the magnitude itself)."""
    out = {}
    for ex in EXCHANGES:
        fixed = None if spans is None else spans.get(ex)
        variant, N, d, ic = best_reading(arrays, grid, ex, fixed_span=fixed)
        out[ex] = d
    return out


def best_spans(arrays, grid):
    """Convenience: the per-leg in-sample best (variant, span) the notebook §6 picks on this
    block — {leg: (variant, span, in_sample_ic)}. The harness fixes these spans for the OOS run."""
    out = {}
    for ex in EXCHANGES:
        variant, N, d, ic = best_reading(arrays, grid, ex)
        out[ex] = (variant, N, float(ic))
    return out
