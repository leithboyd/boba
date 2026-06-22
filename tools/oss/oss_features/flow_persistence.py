"""flow_persistence — the shipped aggressor-sign directional atom for the OSS harness.

Definition (verbatim from notebooks/features/flow_persistence.ipynb / build_flow_persistence.py
§2-§3, §5, §10): the per-venue, bounded, SIGNED `signed_trade` atom — the net direction of
recent aggressive flow on byb's shared trade clock:

    signed_trade(venue, N) = EMA(ε_t) = E[ε] / E[1]          ∈ [-1, 1]
    ε_t = +1 if the trade lifts the ask (buy), -1 if it hits the bid (sell)  (io._trade_lifts_ask)

The notebook analyses a *pair* of atoms — `signed_trade` (directional) and
`flow_sign_persistence = EMA(ε_t·ε_{t-1})` (a regime gauge / interaction). The §10 verdict
SHIPS `signed_trade` and DEMOTES `flow_sign_persistence` (its Lillo-Farmer interaction
`signed_trade·persistence` measured -0.003 OOS — adds nothing). So the harness module exposes
`signed_trade`, one leg per venue (`bin`/`byb`/`okx`) — these are genuine per-exchange numbers,
NOT cross-venue gaps, so there are no "gap_*" legs.

Construction (the §3 vectorized build, exactly):
- Per venue, collapse its same-timestamp prints to ONE event: `ε = sign(Σ ε over that instant's
  prints)` (a burst sweeping levels is one event); DROP timestamps whose net is exactly 0 (a rare
  tie — no net side). The decay clock advances once per shared-clock trade tick.
- It is a SPARSE flow, read as the self-normalising `E/W` KernelMean EMA: inject `(ε, weight 1)`
  ONLY on the venue's own trade-timestamps; `E = EMA(ε)`, `W = EMA(1)`; read `E/W` = the
  count-weighted mean sign ∈ [-1, 1]. The many other-venue ticks decay E and W together and
  cancel in the ratio (the merged-clock ↔ own-clock equivalence). NaN until W>0 (venue has traded).
- Read committed at the last clock tick at-or-before each anchor (causal, piecewise-constant
  between this venue's trades — no live front: there is no level to forward-fill).
- NO σ/λ-division: it is a bounded [-1, 1] ratio already invariant across regimes (the §5 HARD
  regime-invariance gate measured a 1.07× scale ratio across vol buckets — raw is the right form).

Span selection (notebook §6 `price_member[ex] = argmax(st_price[ex])`): per venue, the IN-SAMPLE
best span N by Spearman IC of the SIGNED `signed_trade` against the PRICE-head target. On block[0]
this reproduces the §5/§10 headline: JOINT marginal IC +0.167 over the controls, bin alone +0.158.

HEAD: "price" — `signed_trade` is the directional atom, scored vs grid.price_target.
"""
import numpy as np
from scipy.signal import lfilter
from scipy.stats import spearmanr

NAME = "flow_persistence"
HEAD = "price"                       # directional atom -> scored vs grid.price_target
EXCHANGES = ["bin", "byb", "okx"]    # one signed_trade leg per venue (genuine per-exchange, not gaps)
# the trade-span family the notebook sweeps (EMA memory in trades). The reported headline comes from
# the IN-SAMPLE best span per venue against the PRICE head (§6 price_member), which on block[0] lands
# on the short end (N=20) for the venues that carry the signal.
SPANS = [5, 20, 100, 500, 2000, 8000]


def _venue_marks(arrays, ex):
    """Per-venue per-timestamp aggressor-sign marks on the shared clock (§3 _venue_marks):
    collapse this venue's same-timestamp prints to ONE event ε = sign(Σ ε over the instant's
    prints); DROP exact net-zero timestamps; map each surviving timestamp to its shared-clock
    tick. Returns (k, eps): tick index and per-timestamp sign ∈ {-1, +1}."""
    rx, sign = arrays.tr_rx[ex], arrays.tr_sign[ex]
    urx, inv = np.unique(rx, return_inverse=True)
    net = np.bincount(inv, weights=sign, minlength=len(urx))     # Σ ε over each timestamp's prints
    eps = np.sign(net)                                           # per-timestamp sign ∈ {-1,0,+1}
    nz = eps != 0.0                                              # drop net-zero timestamps (no net side)
    urx, eps = urx[nz], eps[nz]
    k = np.searchsorted(arrays.merged_ts, urx, "left")           # shared-clock tick of each timestamp (exact match)
    return k, eps


def signed_trade(arrays, grid, ex, N):
    """`signed_trade` for one venue at span N: the E/W KernelMean EMA of the aggressor sign on the
    shared clock (inject (ε, weight 1) on the venue's own trade-timestamps, decay once per clock
    tick), read committed at each anchor's last clock tick (causal). Array on the anchor grid; NaN
    until this venue has traded (W>0)."""
    a = 2.0 / (N + 1.0)
    n_ticks = len(arrays.merged_ts)
    k, eps = _venue_marks(arrays, ex)
    e_inj = np.bincount(k, weights=eps,             minlength=n_ticks)   # Σ ε   per clock tick (0 where this venue didn't trade)
    w_inj = np.bincount(k, weights=np.ones_like(eps), minlength=n_ticks) # Σ 1   per clock tick
    E = lfilter([a], [1.0, -(1.0 - a)], e_inj)
    W = lfilter([a], [1.0, -(1.0 - a)], w_inj)
    ratio = E / np.where(W > 0.0, W, np.nan)                            # E/W; nan until this venue has traded
    return ratio[grid.tick_at_anchor]                                  # value as of the last clock tick <= anchor


def best_spans(arrays, grid, head=HEAD):
    """The notebook §6 pick (`price_member`): per venue, the IN-SAMPLE best span by Spearman IC of
    the SIGNED `signed_trade` against the head target. In-sample only — the chosen feature is then
    re-scored OUT-OF-SAMPLE by the harness's walk-forward marginal IC. Returns {venue: span}."""
    target = grid.price_target if head == "price" else grid.rate_target
    out = {}
    for ex in EXCHANGES:
        scores = []
        for N in SPANS:
            d = signed_trade(arrays, grid, ex, N)
            if head == "rate":
                d = np.abs(d)                                          # rate head scores the magnitude
            scores.append(spearmanr(d, target).statistic)
        out[ex] = SPANS[int(np.nanargmax(scores))]
    return out


def compute(arrays, grid, spans=None, head=HEAD):
    """The module contract: return {leg: signed_trade_array_on_grid} for flow_persistence.

    arrays — BlockArrays from oss_core.load_cached.
    grid   — Grid (provides anchor_ts, tick_at_anchor, merged_ts, price_target).
    spans  — None (default) -> per-venue IN-SAMPLE best span (notebook §6 price_member, the
             reported number); or {leg: N} -> that fixed span per leg (the harness uses this to
             FIX block[0]'s pick for the OOS run; legs absent from the dict fall back to best).

    Returns one SIGNED array per venue, length len(grid.anchor_ts), read causally at every anchor
    (never |·| — the model is fed the signed feature for both heads). No σ/λ-division (bounded ratio)."""
    if spans is None:
        spans = best_spans(arrays, grid, head=head)
        return {ex: signed_trade(arrays, grid, ex, spans[ex]) for ex in EXCHANGES}
    chosen = best_spans(arrays, grid, head=head) if any(ex not in spans for ex in EXCHANGES) else {}
    return {ex: signed_trade(arrays, grid, ex, int(spans.get(ex, chosen.get(ex))))
            for ex in EXCHANGES}
