"""trade_rate_normalised — OSS harness feature module.

Definition (verbatim from notebooks/features/trade_rate_normalised.ipynb /
build_trade_rate_normalised.py §2/§3/§6): a venue's TRADE rate (trades per second)
at span N, regime-normalised by byb's move-rate yardstick λ_ev — one leg per venue
(byb/okx/bin), each built from THAT venue's own trade-events, all decayed on the ONE
shared trade clock:

    trade_rate(ex; N)            = W_trades(ex; N) / E_dt(N)          (trades / sec)
    trade_rate_normalised(ex; N) = trade_rate(ex; N) / λ_ev          (the SHIPPED form)

where
  - W_trades(ex; N) is a SPARSE-FLOW trade-EVENT count EMA on the shared clock:
    decay once per shared trade-timestamp (α = 2/(N+1)), inject 1 on each of THIS
    venue's trade-timestamps (simultaneous prints collapse to ONE event), read
    committed at the last clock tick at-or-before each anchor;
  - E_dt(N) is a per-trade EMA of seconds-between-consecutive-shared-trades (a
    property of the shared clock, identical for every venue);
  - λ_ev is byb's mid-MOVE rate yardstick (grid.lambda_ev) — NOT a trade rate, so
    the ÷λ_ev divide turns the feature into "trades per byb-mid-move", a pure rate
    ratio (the only normalisation that clears §6's hard regime-invariance scale gate;
    baseline and ÷σ_ev are non-invariant LEVELS and are rejected there despite higher
    raw IC).

This is the feature's HOME on the RATE head (an intensity feature — how many byb
mid-moves come next), so HEAD = "rate" and the in-sample best-span selection scores
against grid.rate_target (notebook §6). The returned array is the SIGNED feature (the
trade-rate level itself — naturally non-negative, never |·|-folded); the rate head
learns its magnitude. NaN where λ_ev / E_dt are undefined (venue not yet traded).

The committed read at grid.tick_at_anchor is exact (not an approximation of the
notebook's live read): for the trade-count W and the seconds-per-trade dt, every
injection sits on a clock tick, so the live partial-epoch term between the last tick
and the anchor is identically zero — committed[tick_at_anchor] == the §3 live read.

See INTERFACE.md for the contract; flow_imbalance.py is the structural template.
"""
import numpy as np
from scipy.signal import lfilter
from scipy.stats import spearmanr

NAME = "trade_rate_normalised"
HEAD = "rate"                                   # intensity feature -> scored vs grid.rate_target
EXCHANGES = ["byb", "okx", "bin"]               # every venue gets its own trade-rate leg (no cross-venue gap leg)
# the whole lookback family the notebook sweeps (EMA memory in trades; 1 = no smoothing)
SPANS = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]


def _dt_committed(arrays, span):
    """Per-trade EMA (α = 2/(span+1)) of seconds between consecutive shared-clock
    trades — the seconds-per-trade denominator E_dt, committed at each clock tick.
    A property of the shared clock, so the same for every venue."""
    a = 2.0 / (span + 1.0)
    merged_ts = arrays.merged_ts
    byb_dt = np.zeros(len(merged_ts))
    byb_dt[1:] = np.diff(merged_ts) / 1e9                       # seconds between consecutive shared trades
    return lfilter([a], [1.0, -(1.0 - a)], byb_dt)


def _w_committed(arrays, ex, span):
    """The per-venue trade-EVENT count EMA W_trades(ex; N), committed at each clock
    tick: decay once per shared trade-timestamp, inject 1 on each of THIS venue's
    trade-timestamps (simultaneous prints summed into ONE event of mass 1)."""
    a = 2.0 / (span + 1.0)
    merged_ts = arrays.merged_ts
    n_ticks = len(merged_ts)
    trade_rx = np.unique(arrays.tr_rx[ex])                      # this venue's trade-EVENT timestamps (one per timestamp)
    k = np.searchsorted(merged_ts, trade_rx, "left")           # clock-tick index of each trade-event (exact match)
    inj = np.bincount(k, weights=np.ones(trade_rx.size), minlength=n_ticks)
    return lfilter([a], [1.0, -(1.0 - a)], inj)                # committed trade-count mass just after each tick


def trade_rate_normalised(arrays, grid, ex, N):
    """The SHIPPED feature for one venue at span N: trades/sec ÷ λ_ev, read committed
    at each anchor's last clock tick (causal). Returns an array on the anchor grid."""
    tick = grid.tick_at_anchor
    w = _w_committed(arrays, ex, N)[tick]                      # exp-weighted ex trade-event count, live at anchor
    dt = _dt_committed(arrays, N)[tick]                        # exp-weighted seconds-per-trade (shared clock)
    rate = w / np.where(dt > 1e-12, dt, np.nan)               # trades / sec
    lam = grid.lambda_ev                                       # byb mid-MOVE rate yardstick (the SHIPPED normaliser)
    return rate / np.where(lam > 0.0, lam, np.nan)            # trades per byb-mid-move (÷λ_ev), nan where undefined


def best_spans(arrays, grid):
    """Notebook §6 pick: per venue, the IN-SAMPLE best span across the family against
    the RATE-head target (the feature's home), argmax|Spearman|. In-sample only — the
    chosen feature is re-scored OUT-OF-SAMPLE by the harness's walk-forward marginal
    IC. Returns {venue: span}."""
    target = grid.rate_target
    out = {}
    for ex in EXCHANGES:
        scores = [spearmanr(trade_rate_normalised(arrays, grid, ex, N), target).statistic
                  for N in SPANS]
        out[ex] = SPANS[int(np.nanargmax(np.abs(scores)))]
    return out


def compute(arrays, grid, spans=None):
    """The module contract: return {leg: feature_array_on_grid} for trade_rate_normalised.

    arrays — BlockArrays (per-venue trade events + shared trade clock).
    grid   — Grid (anchor_ts, tick_at_anchor, merged_ts, lambda_ev, rate_target).
    spans  — None (default) -> per-leg IN-SAMPLE best span (notebook §6, rate head — the
             reported number); or a {leg: N} dict -> that fixed span per leg (the harness
             passes block[0]'s pick to FIX the OOS run). A missing/None entry falls back
             to that leg's in-sample best span.

    Returns one SIGNED array per venue leg (byb/okx/bin), length len(grid.anchor_ts),
    read causally at every anchor — the trade-rate level ÷ λ_ev (never |·|)."""
    if spans is None:
        chosen = best_spans(arrays, grid)
    else:
        defaults = None
        chosen = {}
        for ex in EXCHANGES:
            N = spans.get(ex) if isinstance(spans, dict) else spans
            if N is None:
                if defaults is None:
                    defaults = best_spans(arrays, grid)
                N = defaults[ex]
            chosen[ex] = N
    return {ex: trade_rate_normalised(arrays, grid, ex, chosen[ex]) for ex in EXCHANGES}
