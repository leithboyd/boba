"""trade_rate_surge — a feature module for the OSS harness (sibling of volume_surge).

Definition (verbatim from notebooks/features/trade_rate_surge.ipynb / build_trade_rate_surge.py §2-§3):
each venue's recent TRADE RATE (trades per second) relative to its own slower baseline — a fast/slow
ratio of two trade-rate EMAs, per venue (byb/okx/bin), on the SHARED trade clock:

    trade_rate_surge(venue, n_fast, n_slow) = trade_rate(venue, n_fast) / trade_rate(venue, n_slow)

where

    trade_rate(venue, N) = T_venue(N) / dt(N)        # that venue's trades per second at span N

  - T_venue(N): a SPARSE FLOW = exp-weighted count of that venue's trade-TIMESTAMPS. Inject 1 AT each
    of that venue's unique trade timestamps (simultaneous prints are ONE event — searchsorted maps
    each unique venue timestamp to its shared-clock tick), decay once per SHARED trade-timestamp
    (α = 2/(N+1)). The injection lands AT a tick that also decays it (the "a same-rx trade decays its
    own injection" convention → searchsorted(..., "left")).
  - dt(N): a PER-TRADE EMA of seconds-per-trade (Δ(trade timestamp)/1e9) on the shared clock — a
    property of the shared clock, SAME for every venue. The per-venue content lives entirely in T.

The ratio fast/slow is **> 1** when the venue is trading faster than its slow baseline (a tempo surge),
**< 1** when it has gone quiet. The common dt factor cancels in the ratio, but it is carried in each
trade_rate exactly as the notebook builds it (so each leg's read matches the §3 vectorized build).

On-grid read (the INTERFACE convention, pattern 1 — trade-clock EMA, committed-only):
each trade_rate leg is piecewise-constant between THAT venue's trades, so the committed value is read
at `grid.tick_at_anchor` (the last shared-clock tick at-or-before each anchor). The notebook §3 build
reads the flow with a LIVE `com[ta+1] + a*partial`, but the `partial` term is the sum over THIS
venue's trade timestamps STRICTLY BETWEEN the last clock tick ≤ anchor and the anchor — and a venue's
trade timestamps ARE clock ticks, so there are never any (verified 0 for every anchor/venue on the
real block). The live read therefore reduces EXACTLY to the committed read at grid.tick_at_anchor.

HEAD — "rate". trade_rate_surge is an INTENSITY (how-many) feature: a venue's trade-tempo surge should
precede more byb mid-moves. The reported §6 number is the RATE-head marginal IC, with the per-venue
best (fast,slow) pair chosen IN-SAMPLE against grid.rate_target (notebook §6 `rate_member`, which picks
the strongest |Spearman IC| of the surge LEVEL vs rate_target).

NORMALISATION — none. The feature is a RATIO of two trade RATES (same units), so the absolute pace and
the shared dt both cancel; it is already dimensionless and comparable across calm/busy regimes (the
guard rail: don't normalise a ratio reflexively — §2 / §5 of the notebook). It is shipped RAW (the
SIGNED value fed to the model is the raw ratio; the price-head DIAGNOSTIC centres it as log-surge, but
the rate-head gate — this feature's home — scores the raw surge level).
"""
import numpy as np
from scipy.signal import lfilter
from scipy.stats import spearmanr

NAME = "trade_rate_surge"
HEAD = "rate"                      # intensity feature: scored vs grid.rate_target
EXCHANGES = ["byb", "okx", "bin"]  # per-venue: each venue's OWN trade tempo vs its OWN baseline

# The fast/slow span family the notebook sweeps (EMA memory in trades). A "span" per leg is a
# (fast, slow) PAIR; the reported §6 number uses the IN-SAMPLE best pair per venue against the
# rate-head target. fast=1 (α=1) is allowed here (the notebook sweeps it): for the trade-count flow
# T it means "no smoothing of the rate", and for dt it is a per-trade EMA at span 1.
FAST = [1, 3, 10, 30, 100, 300]
SLOW = [30, 100, 300, 1000, 3000, 10000]
PAIRS = [(nf, ns) for nf in FAST for ns in SLOW if nf < ns]   # valid (fast,slow) members (nf < ns)


def _ewma_dt(arrays, N):
    """The per-trade seconds-per-trade EMA (α = 2/(N+1)) on the SHARED clock — the same dt for every
    venue. byb_dt[i] = (merged_ts[i] - merged_ts[i-1]) / 1e9, dt[0]=0; lfilter is the per-trade EMA
    held flat between trades. Verbatim from the notebook's byb_dt + _ewma."""
    merged_ts = arrays.merged_ts
    n_ticks = len(merged_ts)
    a = 2.0 / (N + 1.0)
    byb_dt = np.zeros(n_ticks)
    byb_dt[1:] = np.diff(merged_ts) / 1e9
    return lfilter([a], [1.0, -(1.0 - a)], byb_dt)            # one value per clock tick


def _T_flow(arrays, ex, N):
    """Committed T of the per-venue trade-TIMESTAMP-count flow on the SHARED clock at span N — one
    value per clock tick, flat between this venue's trades. Inject a*1 AT each of this venue's unique
    trade timestamps (searchsorted ..., "left" → the tick decays its own injection), then
    T_t = (1-a)*(T_{t-1} + inj_t). Equivalent to the notebook `_flow_at`'s committed `com[ta+1]`
    (with the verified-zero live `partial` dropped), built with no shared code with that helper."""
    merged_ts = arrays.merged_ts
    n_ticks = len(merged_ts)
    a = 2.0 / (N + 1.0)
    u = np.unique(arrays.tr_rx[ex])                           # this venue's unique trade timestamps (simultaneous prints = ONE event)
    kt = np.searchsorted(merged_ts, u, "left")               # shared-clock tick index of each (events sit AT a tick)
    inj = np.zeros(n_ticks)
    np.add.at(inj, kt, a * 1.0)                              # inject a*1 (one trade-timestamp) at that tick
    return lfilter([1.0], [1.0, -(1.0 - a)], (1.0 - a) * inj)


def _trade_rate(arrays, ex, N):
    """trade_rate(ex, N) = T_ex(N) / dt(N) over the shared clock — venue ex's trades/sec at span N
    (one value per clock tick). nan where dt is not yet positive (matching the notebook's
    np.maximum(dt, 1e-12) guard reading effectively 0 before the clock warms)."""
    T = _T_flow(arrays, ex, N)
    dt = _ewma_dt(arrays, N)
    return T / np.maximum(dt, 1e-12)


def surge(arrays, grid, ex, n_fast, n_slow):
    """The trade-rate-surge ratio for one venue at (n_fast, n_slow), read committed at each anchor's
    last clock tick (causal, piecewise-constant between this venue's trades). Array on the anchor grid.
    SIGNED (raw ratio); nan before this venue has traded (slow rate == 0)."""
    rf = _trade_rate(arrays, ex, n_fast)                     # fast trade rate (trades/sec) per clock tick
    rs = _trade_rate(arrays, ex, n_slow)                     # slow trade rate (baseline tempo)
    ratio = rf / np.where(rs > 0.0, rs, np.nan)              # > 1 = trading faster than baseline; nan until traded
    return ratio[grid.tick_at_anchor]                        # committed value as of the last clock tick <= anchor


def best_spans(arrays, grid, head=HEAD):
    """The notebook §6 pick: per venue, the IN-SAMPLE best (fast,slow) pair by strongest |Spearman IC|
    of the surge LEVEL against the head target — `rate_member` (rate head, the feature's home) /
    `price_member` (price head, the log-surge diagnostic). In-sample only; the chosen feature is
    re-scored OUT-OF-SAMPLE by the harness. Returns {venue: (fast,slow)}.

    NB the notebook scores the rate head on the surge LEVEL (s) and the price head on log(s); only the
    sign of the ranking can differ (a monotone transform of finite values), so the |IC| argmax — hence
    the chosen pair — is identical either way. We score the level for both, matching §6's rate_member."""
    target = grid.rate_target if head == "rate" else grid.price_target
    out = {}
    for ex in EXCHANGES:
        best_pair, best_abs = None, -np.inf
        for nf, ns in PAIRS:
            d = surge(arrays, grid, ex, nf, ns)
            sc = spearmanr(d, target).statistic
            if np.isfinite(sc) and abs(sc) > best_abs:       # strongest |IC| cell (notebook np.nanargmax(np.abs(grid)))
                best_abs, best_pair = abs(sc), (nf, ns)
        out[ex] = best_pair
    return out


def compute(arrays, grid, spans=None):
    """The module contract: return {venue: feature_array_on_grid} for trade_rate_surge.

    arrays — BlockArrays (per-venue trades + the shared trade clock merged_ts).
    grid   — Grid (anchor_ts, tick_at_anchor, merged_ts, rate_target/price_target).
    spans  — None (default) -> per-venue IN-SAMPLE best (fast,slow) pair against the HEAD target
             (notebook §6 `rate_member`, the reported number); or {venue: (fast,slow)} -> force that
             fixed pair per venue (the harness uses this to FIX block[0]'s pick for the OOS run).
             A 2-tuple may also be passed to force one pair for every venue.

    Returns one SIGNED array per venue (the raw surge ratio — no σ/λ normalisation, a ratio of two
    trade rates is already unit-free), length len(grid.anchor_ts), read causally at every anchor,
    nan before that venue's first trade."""
    if spans is None:
        spans = best_spans(arrays, grid, head=HEAD)
    elif isinstance(spans, dict):
        spans = dict(spans)
    else:
        spans = {ex: tuple(spans) for ex in EXCHANGES}       # one (fast,slow) pair forced for every venue
    return {ex: surge(arrays, grid, ex, spans[ex][0], spans[ex][1]) for ex in EXCHANGES}
