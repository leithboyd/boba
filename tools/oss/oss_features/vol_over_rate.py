"""vol_over_rate — a feature module for the OSS harness.

Definition (verbatim from notebooks/features/vol_over_rate.ipynb / build_vol_over_rate.py §2-§3):
each venue's volatility-per-move yardstick divided by its move-rate yardstick,

    feature(venue, vol_span, rate_span) = σ_ev(venue, vol_span) / λ_ev(venue, rate_span)

computed PER VENUE (byb/okx/bin) from THAT venue's own mid-moves, decayed on the SHARED
trade clock, read at each grid anchor. Units: seconds — high = big-and-slow moves, low =
small-and-fast moves. An UNSIGNED intensity coordinate.

Construction (per venue, exactly the §2 machinery — bit-identical to oss_core's byb
yardsticks at span 10000, verified):
  - σ_ev = √(E/W), a KernelMean E/W flow over the venue's squared mid-moves r²: inject r²
    (weight 1) only on a REAL mid-move of THIS venue, decay once per trade-timestamp on the
    shared clock. Read with the live partial epoch so it is current between trades.
  - λ_ev = W / E[Δt]: the same exp-weighted move-count W (at the rate span) divided by an
    EventEMA of seconds-per-trade on the SHARED clock (byb_dt = diff(merged_ts)). The
    seconds-per-trade leg is shared across venues (the common clock), per the notebook.
  - vol_span and rate_span are swept INDEPENDENTLY (2-D family). When vol_span == rate_span
    the two W's coincide; with independent spans they are two EMAs of the same move stream.

HEAD = "rate": the notebook is emphatic this is an unsigned intensity feature whose natural
head is the rate head — scored vs grid.rate_target. The price head is ≈0 by construction
(no sign), so the §6 cell pick and the headline marginal IC are taken against the RATE head.

NORMALISATION: NONE. σ_ev/λ_ev is already a ratio of two EMAs (units: seconds), a usable
regime coordinate as-is — the notebook explicitly forbids a further σ/λ division ("dividing
it again by σ_ev would just hand back 1/λ_ev"). So compute ships the raw ratio.

Span selection (spans=None): the §6 pick — per venue, the IN-SAMPLE best (vol_span, rate_span)
cell off the full 2-D grid by |standalone rank-IC| against the head target (the notebook's
best_cell = argmax|IC|). Returns the chosen cells; the OOS harness fixes block[0]'s pick by
passing them back as `spans`.

SIGNED: σ_ev/λ_ev is a ratio of positives, so the value is non-negative by construction; it
is shipped as-is (the model is fed exactly this value — never an extra |·| or sign flip).
NaN where a venue has not yet moved (W <= 0) or before the first trade (E[Δt] undefined).
"""
import numpy as np
from scipy.signal import lfilter
from scipy.stats import spearmanr

import oss_core as core

NAME = "vol_over_rate"
HEAD = "rate"                                  # unsigned intensity feature -> scored vs grid.rate_target
EXCHANGES = ["byb", "okx", "bin"]              # byb = own/circular leg; okx/bin = cross-venue legs

# The FULL 2-D lookback family the notebook §6 sweeps: σ_ev span × λ_ev span, independently.
VOL_SPANS  = [2000, 5000, 10000, 20000, 40000]
RATE_SPANS = [2000, 5000, 10000, 20000, 40000]


def _move_stream(rx0, mid0):
    """Collapse same-timestamp rows to ONE mid, take the per-timestamp log-return, keep REAL
    moves only. Returns (mv_rx, mv_r2): the timestamps of this venue's mid-moves and their r²."""
    keep = np.concatenate([rx0[1:] != rx0[:-1], [True]])          # same-TIMESTAMP rows -> one update (final mid)
    rx, mid = rx0[keep], mid0[keep]
    lm = np.log(mid)
    blr = np.empty_like(lm)
    blr[0] = 0.0
    blr[1:] = np.diff(lm)                                         # log-return per timestamp
    mv = blr != 0.0                                              # a REAL mid-move (one per timestamp where the mid changed)
    return rx[mv], blr[mv] ** 2


class _VenueYard:
    """Per-venue σ_ev / λ_ev machinery on the shared trade clock (the §2 cell, keyed by venue).
    Injects on THIS venue's mid-moves; decays once per trade-timestamp on merged_ts; the
    seconds-per-trade leg byb_dt is the SHARED clock's. Read with the live partial epoch."""

    def __init__(self, arrays):
        self.merged_ts = arrays.merged_ts
        self.n_ticks = len(self.merged_ts)
        # shared seconds-per-trade on the common clock (λ_ev denominator), independent of venue
        self.byb_dt = np.zeros(self.n_ticks)
        self.byb_dt[1:] = np.diff(self.merged_ts) / 1e9
        self._mv = {}
        for ex in EXCHANGES:
            rx, mid = core.mid_stream(arrays, ex)
            self._mv[ex] = _move_stream(rx, mid)

    def _ewma(self, x, span):
        a = 2.0 / (span + 1.0)
        return lfilter([a], [1.0, -(1.0 - a)], x)

    def _flow_at(self, ex, anchors, val, span):
        """EWMA of `val` over venue ex's MOVE stream, decayed once per trade-timestamp on the
        shared clock, read AT each anchor with the live partial epoch (current between trades)."""
        merged_ts, n_ticks = self.merged_ts, self.n_ticks
        mv_rx, _ = self._mv[ex]
        a = 2.0 / (span + 1.0)
        k = np.searchsorted(merged_ts, mv_rx, "left")            # trades strictly before each move
        ep = np.bincount(k, weights=val, minlength=n_ticks + 1)
        x = np.zeros(n_ticks + 1)
        x[1:] = a * (1.0 - a) * ep[:-1]
        com = lfilter([1.0], [1.0, -(1.0 - a)], x)               # committed E just after each trade
        ta = np.searchsorted(merged_ts, anchors, "right") - 1    # last trade <= anchor
        cs = np.concatenate([[0.0], np.cumsum(val)])
        partial = (cs[np.searchsorted(mv_rx, anchors, "right")]
                   - cs[np.searchsorted(mv_rx, merged_ts[ta], "right")])
        return com[ta + 1] + a * partial

    def yardsticks(self, ex, anchors, vol_span, rate_span):
        """σ_ev (span=vol_span), λ_ev (span=rate_span) for venue ex — independent spans."""
        mv_rx, mv_r2 = self._mv[ex]
        ones = np.ones(mv_r2.size)
        e_sq = self._flow_at(ex, anchors, mv_r2, vol_span)        # E: exp-wt squared moves (σ_ev's span)
        e_mv_v = self._flow_at(ex, anchors, ones, vol_span)       # W at σ_ev span (σ_ev denominator)
        e_mv_r = self._flow_at(ex, anchors, ones, rate_span)      # W at λ_ev span (λ_ev numerator)
        e_dt = self._ewma(self.byb_dt, rate_span)[
            np.searchsorted(self.merged_ts, anchors, "right") - 1]  # shared seconds/trade
        sig = np.sqrt(np.where(e_mv_v > 0.0, e_sq / np.maximum(e_mv_v, 1e-300), np.nan))
        lam = e_mv_r / np.maximum(e_dt, 1e-12)
        return sig, lam, e_mv_v

    def feature(self, ex, anchors, vol_span, rate_span):
        """THE FEATURE for venue ex: σ_ev(vol_span) / λ_ev(rate_span), read at each anchor.
        NaN until this venue has moved (W <= 0) and before the first trade (E[Δt] <= 0)."""
        sig, lam, w = self.yardsticks(ex, anchors, vol_span, rate_span)
        out = sig / np.maximum(lam, 1e-12)
        out = np.where((w > 0.0) & np.isfinite(out), out, np.nan)   # undefined before this venue moves
        return out


def _as_cell(spec):
    """Map a fixed-span spec to a (vol_span, rate_span) cell. An int N means the shared cell
    (vol=rate=N); a (vol, rate) pair is used as-is (the §6 2-D pick round-trips exactly)."""
    if isinstance(spec, (tuple, list, np.ndarray)):
        return int(spec[0]), int(spec[1])
    n = int(spec)
    return n, n


def best_spans(arrays, grid, head=HEAD):
    """The notebook §6 pick: per venue, the IN-SAMPLE best (vol_span, rate_span) cell off the
    full 2-D grid by |standalone rank-IC| against the head target (best_cell = argmax|IC|).
    Returns {venue: (vol_span, rate_span)}. In-sample only; re-scored OOS by the harness."""
    target = grid.rate_target if head == "rate" else grid.price_target
    yard = _VenueYard(arrays)
    out = {}
    for ex in EXCHANGES:
        gridIC = np.full((len(VOL_SPANS), len(RATE_SPANS)), np.nan)
        for i, vs in enumerate(VOL_SPANS):
            for j, rs in enumerate(RATE_SPANS):
                d = yard.feature(ex, grid.anchor_ts, vs, rs)
                m = np.isfinite(d) & np.isfinite(target)
                if m.sum() > 100:
                    gridIC[i, j] = spearmanr(d[m], target[m]).statistic
        i, j = np.unravel_index(np.nanargmax(np.abs(gridIC)), gridIC.shape)
        out[ex] = (VOL_SPANS[i], RATE_SPANS[j])
    return out


def compute(arrays, grid, spans=None, head=HEAD):
    """The module contract: return {venue: feature_array_on_grid} for vol_over_rate.

    spans=None -> per-venue in-sample best (vol_span, rate_span) cell off the §6 2-D grid
                  (the reported headline number).
    spans={leg: N} or {leg: (vol, rate)} -> force that fixed cell per leg (the OOS harness
                  fixes block[0]'s pick this way; an int means vol=rate=N).

    Each value is the SIGNED feature (a non-negative ratio, shipped as-is — never |·|), length
    len(grid.anchor_ts), read causally at every anchor, NaN where undefined."""
    yard = _VenueYard(arrays)
    if spans is None:
        spans = best_spans(arrays, grid, head=head)
    out = {}
    for ex in EXCHANGES:
        vs, rs = _as_cell(spans[ex])
        out[ex] = yard.feature(ex, grid.anchor_ts, vs, rs)
    return out
