"""mid_rate_surge — a feature module for the OSS harness.

Definition (verbatim from notebooks/features/mid_rate_surge.ipynb / build_mid_rate_surge.py
§2/§3): the normalisation-free fast/slow ratio of a venue's MID-UPDATE RATE, built per
venue (byb/okx/bin), each from THAT venue's OWN mid-moves, all decayed once per
trade-timestamp on the ONE shared trade clock:

    mid_rate_surge(ex; fast, slow) = rate_ex(fast) / rate_ex(slow)
    rate_ex(N) = W_ex(N) / dt(N)        = venue ex's mid-moves per second at span N
      W_ex(N)  = live EWMA of ex's mid-MOVE count (inject 1 per real ex mid-move,
                 decay once per trade-timestamp on the shared clock, read LIVE at anchor)
      dt(N)    = per-trade EWMA of seconds between consecutive trade-timestamps
                 (a property of the SHARED clock — same for every venue)

This is an INTENSITY (how-many) feature — its home is the RATE head (`grid.rate_target`),
so HEAD = "rate" and the harness scores the SIGNED surge against the move-count target
(the rate head learns the magnitude itself). It is a dimensionless ratio of two rates, so
it ships with NO σ_ev / λ_ev division (the §2 / §8 guard-rail: don't normalise a ratio).

byb's mid-update rate IS λ_ev, so the byb leg is circular with `rate_momentum`; the
okx/bin legs are OTHER venues' quoting tempo (the genuine cross-venue lead/lag test). All
three legs ship per the no-fixed-leader rule.

Span family: a fast×slow grid (notebook §6). spans=None -> per-leg in-sample best
(fast, slow) against the rate-head target (the notebook §6 `rate_member` pick, the
reported headline); spans={leg:(fast, slow)} -> force a fixed pair per leg (the harness
fixes block[0]'s pick for the OOS run). See INTERFACE.md for the on-grid contract.
"""
import numpy as np
from scipy.signal import lfilter
from scipy.stats import spearmanr

import oss_core as core

NAME = "mid_rate_surge"
HEAD = "rate"                                  # intensity feature -> scored vs grid.rate_target
EXCHANGES = ["byb", "okx", "bin"]              # every venue gets its own surge leg (byb circular; okx/bin cross-venue)

# the (fast, slow) family the notebook §6 sweeps (EMA memory in trades); a member is valid only if fast < slow
FAST = [1, 3, 10, 30, 100, 300]
SLOW = [30, 100, 300, 1000, 3000, 10000]


def _move_stream(arrays, ex):
    """Venue ex's mid-MOVE timestamps on the shared clock. Take the venue's freshest-mid
    stream (per the MID_STREAM policy via core.mid_stream), collapse same-timestamp rows to
    ONE final mid (one event), and keep only the timestamps where the mid actually CHANGED
    (a real mid-move). Returns mv_rx (int64 ns of each move)."""
    rx0, mid0 = core.mid_stream(arrays, ex)
    keep = np.concatenate([rx0[1:] != rx0[:-1], [True]])        # collapse same-timestamp rows to the final mid
    rx, mid = rx0[keep], mid0[keep]
    lm = np.log(mid)
    blr = np.empty_like(lm)
    blr[0] = 0.0
    blr[1:] = np.diff(lm)                                       # this venue's log-return per timestamp
    mv = blr != 0.0                                            # a REAL mid-move: one per timestamp where the mid changed
    return rx[mv]


def _byb_dt(arrays):
    """Seconds between consecutive trade-timestamps on the shared clock (same dt for every
    venue) — the denominator's raw signal."""
    merged_ts = arrays.merged_ts
    dt = np.zeros(len(merged_ts))
    dt[1:] = np.diff(merged_ts) / 1e9
    return dt


def _ewma(x, span):
    """Per-trade EMA (α = 2/(span+1)) — the seconds-per-trade denominator leg."""
    a = 2.0 / (span + 1.0)
    return lfilter([a], [1.0, -(1.0 - a)], x)


def _flow_at(merged_ts, mv_rx, val, anchors, span):
    """LIVE EWMA of `val` over a venue's mid-move stream, decayed once per trade-timestamp
    on the shared clock, read AT each anchor (committed value just after the last trade
    PLUS the fresh partial epoch since that trade). Verbatim from the template / oss_core
    yardstick machinery — no peeking."""
    n_ticks = len(merged_ts)
    a = 2.0 / (span + 1.0)
    k = np.searchsorted(merged_ts, mv_rx, "left")                              # trades strictly before each move
    ep = np.bincount(k, weights=val, minlength=n_ticks + 1)
    x = np.zeros(n_ticks + 1)
    x[1:] = a * (1.0 - a) * ep[:-1]
    com = lfilter([1.0], [1.0, -(1.0 - a)], x)                                # committed E just after each trade
    ta = np.searchsorted(merged_ts, anchors, "right") - 1                     # last trade <= anchor
    cs = np.concatenate([[0.0], np.cumsum(val)])
    partial = (cs[np.searchsorted(mv_rx, anchors, "right")]
               - cs[np.searchsorted(mv_rx, merged_ts[ta], "right")])
    return com[ta + 1] + a * partial


def _mid_rate(arrays, grid, mv_rx, byb_dt, span):
    """Venue's mid-update RATE at a span: live move-count W ÷ shared seconds-per-trade dt =
    moves/sec, read at each anchor."""
    merged_ts = arrays.merged_ts
    anchors = grid.anchor_ts
    w = _flow_at(merged_ts, mv_rx, np.ones(mv_rx.size), anchors, span)         # W_ex: exp-weighted ex-move count (live)
    dt = _ewma(byb_dt, span)[np.searchsorted(merged_ts, anchors, "right") - 1]  # seconds/trade (shared clock)
    return w / np.maximum(dt, 1e-12)


def surge(arrays, grid, ex, n_fast, n_slow, _cache=None):
    """The feature for one venue at one (fast, slow) pair, read on the anchor grid:
    rate_ex(fast) / rate_ex(slow). >1 = that venue is repricing faster than its baseline."""
    if _cache is None:
        _cache = {}
    mv_rx = _cache.get(("mv", ex))
    if mv_rx is None:
        mv_rx = _move_stream(arrays, ex)
        _cache[("mv", ex)] = mv_rx
    byb_dt = _cache.get("dt")
    if byb_dt is None:
        byb_dt = _byb_dt(arrays)
        _cache["dt"] = byb_dt
    rf = _cache.get(("rate", ex, n_fast))
    if rf is None:
        rf = _mid_rate(arrays, grid, mv_rx, byb_dt, n_fast)
        _cache[("rate", ex, n_fast)] = rf
    rs = _cache.get(("rate", ex, n_slow))
    if rs is None:
        rs = _mid_rate(arrays, grid, mv_rx, byb_dt, n_slow)
        _cache[("rate", ex, n_slow)] = rs
    return rf / np.maximum(rs, 1e-12)


def best_spans(arrays, grid, head="rate"):
    """The notebook §6 `rate_member` pick: per venue, the IN-SAMPLE best (fast, slow) member
    against the head target (Spearman), over the WHOLE fast×slow family (fast < slow). The
    rate head scores the surge LEVEL (the §6 `rate_grid`); the price head scores the signed
    log-surge (`price_grid`). In-sample only — the chosen member is re-scored OOS by the
    harness. Returns {venue: (fast, slow)}."""
    target = grid.rate_target if head == "rate" else grid.price_target
    out = {}
    for ex in EXCHANGES:
        cache = {}
        best, best_abs = None, -np.inf
        for nf in FAST:
            for ns in SLOW:
                if nf >= ns:
                    continue
                s = surge(arrays, grid, ex, nf, ns, cache)
                if head == "rate":
                    f = s                                          # rate head: surge LEVEL -> move count
                else:
                    f = np.log(np.maximum(s, 1e-12))               # price head: signed log-surge -> return
                v = np.isfinite(f) & np.isfinite(target)
                if v.sum() <= 100:
                    continue
                score = spearmanr(f[v], target[v]).statistic
                if np.isfinite(score) and abs(score) > best_abs:
                    best_abs, best = abs(score), (nf, ns)
        out[ex] = best
    return out


def compute(arrays, grid, spans=None):
    """The module contract: return {venue: feature_array_on_grid} for mid_rate_surge.

    arrays — BlockArrays from oss_core.load_cached.
    grid   — Grid from oss_core.build_grid (anchor_ts, tick_at_anchor, merged_ts, rate_target).
    spans  — None (default) -> per-venue in-sample best (fast, slow) member against the
             RATE-head target (notebook §6 `rate_member`, the reported headline);
             or {venue: (fast, slow)} -> that fixed pair per venue (the harness fixes
             block[0]'s pick for the OOS run; a bare int N is accepted as (N, ·)? no —
             pass the (fast, slow) tuple).

    Returns one SIGNED array per venue, length len(grid.anchor_ts), read causally at every
    anchor. NaN before a venue's mid-rate is defined. No σ/λ division (it is a dimensionless
    ratio of two rates — the §2/§8 guard-rail)."""
    if spans is None:
        spans = best_spans(arrays, grid, head=HEAD)
    out = {}
    for ex in EXCHANGES:
        nf, ns = spans[ex]
        out[ex] = surge(arrays, grid, ex, nf, ns)
    return out
