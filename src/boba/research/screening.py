"""Generic screening engines — the feature-agnostic scaffolding pulled out of the notebooks.

A feature supplies a `FeatureSpec` (`boba.features.base`); these functions do everything else, the
same way for every feature. Pulling the scaffolding here (a) removes the duplicated code from every
feature notebook, (b) gives the parity driver and the gate orchestration a single TESTED home, and
(c) means a bug fixed here propagates to every feature notebook automatically.

  Step 0  build_context(...)  -> ScreeningContext   (targets, yardsticks, grid, controls, raw stream)
  Step 1  the FeatureSpec contract (boba.features.base) + parity_check (generic driver)
  Step 2  build_family(...)   -> the cached vectorized family (parity subset OR the full sweep)
  Step 3  run_gates(...)      -> the full Gate A/B/coupling/companion table for one head

Inner statistics are always `boba.research.gates` — never re-implemented here.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from boba.ema import KernelMeanEMA
from boba.features.base import BookEvent, FeatureSpec, Params, TradeEvent, VectorizedBuilder


# --------------------------------------------------------------------------------------------------
# The shared yardstick — its own streaming part, parity-tested once. Feature streaming classes
# COMPOSE this rather than recomputing σ_ev, so a feature's parity validates only its own maths.
# --------------------------------------------------------------------------------------------------
class LiveYardstick:
    """Streaming σ_ev (composes `boba.ema.KernelMeanEMA`): the online twin of `ScreeningContext.
    sigma_at_anchor`. It decays on the shared trade clock and injects a squared return on each
    *real* target mid-move. (λ_ev is a vectorized context quantity used for the rate target; it is
    not streamed by features, so it lives only on the context.)
    """

    __slots__ = ("vol", "_prev")

    def __init__(self, span: int):
        self.vol = KernelMeanEMA(span)      # E/W mean of squared target moves -> sqrt(E/W) = σ_ev
        self._prev: Optional[float] = None  # last target log-mid, to detect a real move

    def on_target_logmid(self, log_mid: Optional[float]) -> None:
        """Feed the current target log-mid (or None before it has quoted). Injects `(Δlog)**2`
        iff the mid actually moved — a flow on the byb-move stream. Does NOT decay."""
        if log_mid is None:
            return
        if self._prev is not None and log_mid != self._prev:
            self.vol.add((log_mid - self._prev) ** 2)
        self._prev = log_mid

    def tick(self) -> None:
        """Advance the shared trade clock one step (decay E and W)."""
        self.vol.tick()

    def sigma(self) -> float:
        """σ_ev = sqrt(E/W) — RMS mid-move per move, live (nan during warm-up)."""
        return self.vol.value() ** 0.5


def _ffill(rx: np.ndarray, val: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Causal forward-fill: `val` of the last `rx <= t[i]`; NaN before the first `rx` (no wrap)."""
    idx = np.searchsorted(rx, t, "right") - 1
    return np.where(idx < 0, np.nan, val[np.clip(idx, 0, len(val) - 1)])


def _active_grid(start_ns: int, end_ns: int, step_ns: int, event_ts: np.ndarray) -> np.ndarray:
    """Uniform `step_ns` grid over `[start, end)`, keeping only ticks whose preceding window `(t-step, t]`
    contains at least one event (from `event_ts` — a book update or trade on ANY exchange). A regular
    cadence that skips dead time: an empty window produces no anchor (we never make an example where
    nothing happened), while anchors stay on a clean grid rather than jittered onto event timestamps."""
    grid = np.arange(start_ns, end_ns, step_ns)
    if len(grid) == 0 or len(event_ts) == 0:
        return grid[:0]
    hi = np.searchsorted(event_ts, grid, "right")          # events at-or-before each tick
    lo = np.searchsorted(event_ts, grid - step_ns, "right")  # events at-or-before the window start
    return grid[hi > lo]                                   # keep ticks with >= 1 event in (t-step, t]


# --------------------------------------------------------------------------------------------------
# Step 0 — the prepared context.
# --------------------------------------------------------------------------------------------------
@dataclass
class RawEventStream:
    """The block's book + trade events, all sources, in receive-time order, integer-coded — what the
    streaming parity driver replays. `lid` indexes `listings`; `kind` is 0=book, 1=trade. Four payload
    columns `(a, b, c, d)` carry every event prop: a book event is `(bid, ask, bid_qty, ask_qty)`, a
    trade is `(px, lifts_ask, qty, nan)`. The driver packs them into a `BookEvent` / `TradeEvent` per
    event, so adding a future prop is a new column here, never a signature change. Feature-agnostic."""

    rx: np.ndarray
    kind: np.ndarray
    lid: np.ndarray
    t: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    listings: tuple[str, ...]


@dataclass
class ScreeningContext:
    """Step 0 output — shared, read-only, by every later step. Built from ONE real block by
    `build_context()`. Holds the shared trade clock, the causal grid, both yardstick vectors, both
    heads' targets, the Gate-B controls + Gate-A regime coordinate, and the prepared raw-event
    stream. A feature derives its own inputs from the building-block accessors at the bottom."""

    # --- config ---
    block: str
    coin: str
    target: str                       # the prediction target listing (fixed across features: byb)
    sources: tuple[str, ...]          # the other sources a feature may fan out over
    horizon_ns: int
    yardstick_span: int
    mid_stream: dict[str, str]        # per-source mid policy (merged_levels = fuse trades; else book-only)

    # --- shared trade clock + causal evaluation grid ---
    merged_ts: np.ndarray             # unique trade-timestamps — the shared decay clock
    anchor_ts: np.ndarray             # the evaluation grid (past warm-up)
    tick_at_anchor: np.ndarray        # last trade-clock tick <= each anchor

    # --- yardstick vectors on the grid (σ_ev, λ_ev) at span = yardstick_span ---
    sigma_at_anchor: np.ndarray
    lam_at_anchor: np.ndarray

    # --- both heads' targets ---
    price_target: np.ndarray          # fwd_return / σ_ev
    rate_target: np.ndarray           # fwd_count / λ_ev

    # --- Gate-B controls + Gate-A regime coordinate (the levels are NEVER controls) ---
    base: list[np.ndarray]            # [rate_momentum, vol_momentum] — the only Gate-B controls
    vol_level: np.ndarray
    rate_level: np.ndarray
    vol_regime: np.ndarray            # 0/1/2 = calm/mid/wild, for the companion

    # --- prepared raw-event stream for the streaming parity driver ---
    raw_events: RawEventStream

    # --- building blocks a feature derives its inputs from (kept here so build_context loads once) ---
    _mids: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict, repr=False)
    _books: dict[str, tuple] = field(default_factory=dict, repr=False)  # raw front_levels per ex: (rx, bid, bid_qty, ask, ask_qty)
    _trades: dict[str, tuple] = field(default_factory=dict, repr=False)  # raw trades per ex: (rx, px, lifts_ask, qty)
    target_logmid_on_clock: np.ndarray = field(default=None, repr=False)  # log target mid at each tick
    _mv_rx: np.ndarray = field(default=None, repr=False)                  # target move timestamps
    _mv_r2: np.ndarray = field(default=None, repr=False)                  # squared target moves
    _clock_dt: np.ndarray = field(default=None, repr=False)              # seconds between trade ticks

    # --- accessors a feature builds on ---
    def mid_on_clock(self, source: str) -> np.ndarray:
        """Causal mid of `source` at each trade-clock tick (NaN before its first quote)."""
        rx, mid = self._mids[source]
        return _ffill(rx, mid, self.merged_ts)

    def mid_at_anchor(self, source: str) -> np.ndarray:
        """Causal mid of `source` at each grid anchor — the freshest value, never stale."""
        rx, mid = self._mids[source]
        return _ffill(rx, mid, self.anchor_ts)

    def _flow_at(self, anchors: np.ndarray, val: np.ndarray, span: float, src_rx: np.ndarray = None) -> np.ndarray:
        """EWMA of `val` over an EVENT stream `src_rx` (default the target-MOVE stream `_mv_rx`), decayed
        once per trade-timestamp, read AT each anchor (committed-per-trade EMA + the partial epoch since
        the last trade). `val` is aligned to `src_rx`. Reused by any sparse flow (σ_ev, OFI, …)."""
        from scipy.signal import lfilter

        a = 2.0 / (span + 1.0)
        mv_rx = self._mv_rx if src_rx is None else src_rx
        n_ticks = len(self.merged_ts)
        k = np.searchsorted(self.merged_ts, mv_rx, "left")
        ep = np.bincount(k, weights=val, minlength=n_ticks + 1)
        x = np.zeros(n_ticks + 1)
        x[1:] = a * (1.0 - a) * ep[:-1]
        com = lfilter([1.0], [1.0, -(1.0 - a)], x)
        ta = np.searchsorted(self.merged_ts, anchors, "right") - 1
        cs = np.concatenate([[0.0], np.cumsum(val)])
        partial = cs[np.searchsorted(mv_rx, anchors, "right")] - cs[np.searchsorted(mv_rx, self.merged_ts[ta], "right")]
        return com[ta + 1] + a * partial

    def yardsticks_at(self, anchors: np.ndarray, span: float) -> tuple[np.ndarray, np.ndarray]:
        """`(σ_ev, λ_ev)` at arbitrary anchors / span — for the momentum controls and alt spans."""
        from scipy.signal import lfilter

        a = 2.0 / (span + 1.0)
        e_sq = self._flow_at(anchors, self._mv_r2, span)
        e_mv = self._flow_at(anchors, np.ones(self._mv_r2.size), span)
        e_dt = lfilter([a], [1.0, -(1.0 - a)], self._clock_dt)[np.searchsorted(self.merged_ts, anchors, "right") - 1]
        sig = np.sqrt(e_sq / np.maximum(e_mv, 1e-12))
        lam = e_mv / np.maximum(e_dt, 1e-12)
        return sig, lam


def build_context(
    block: Optional[str] = None,
    *,
    coin: str = "eth_usdt_p",
    target: str = "byb_eth_usdt_p",
    sources: tuple[str, ...] = ("bin", "okx"),
    mid_stream: Optional[dict[str, str]] = None,
    horizon_ns: int = 100 * 1_000_000,
    yardstick_span: int = 10_000,
    grid_ms: int = 50,
    active_only: bool = False,
    hours: Optional[float] = None,
    warmup_spans: int = 5,
    max_feature_span: int = 10_000,
    raw_stream_cutoff_grid: int = 250_000,
) -> ScreeningContext:
    """Step 0. Load one block, build the shared trade clock and the causal grid, compute both
    yardsticks and both heads' targets and the regime controls/coordinate, and prepare the raw-event
    stream (bounded to `raw_stream_cutoff_grid` anchors for the parity driver). The SINGLE place data
    is loaded. `parity_check`'s `n_grid` must be <= `raw_stream_cutoff_grid`.

    Evaluation grid: a uniform `grid_ms`-spaced wall-clock grid. With `active_only=True` it keeps only the
    ticks whose preceding `grid_ms` window carried an event (book update or trade) from ANY exchange — a
    regular cadence that samples activity and skips dead time, so a short-lived, event-coincident feature
    state is actually sampled (a 50 ms grid catches a ~ms-lived state only ~10% of the time) without
    flooding the IC with empty-window examples. Lower `grid_ms` (e.g. 1) for finer resolution.

    `hours` limits how much of the block to load — only the first `hours` hours of events (None = the whole
    block). Use it to iterate quickly, or to keep a fine (`grid_ms=1`, `active_only`) grid to a tractable size."""
    import polars as pl

    import boba.io as io
    from boba.io import list_blocks, load_block

    mid_stream = mid_stream or {"bin": "front_levels", "byb": "merged_levels", "okx": "merged_levels"}
    target_ex = target.split("_", 1)[0]
    all_ex = (target_ex,) + tuple(sources)
    if block is None:
        block = list_blocks(target, "front_levels")[0]

    def load_mid(ex):
        df = load_block(block, f"{ex}_{coin}", mid_stream[ex]).select("rx_time", "bid_prc", "ask_prc").drop_nulls()
        return (df["rx_time"].cast(pl.Int64).to_numpy(),
                (df["bid_prc"].to_numpy() + df["ask_prc"].to_numpy()) / 2.0)

    mids = {ex: load_mid(ex) for ex in all_ex}

    def load_book(ex):                                   # raw front_levels WITH sizes (snapshot qty) — for OFI etc.
        # SAME row set as this venue's mid load: merged venues fuse by exch_time (require it), book-only
        # venues don't — so byb/okx match the existing front_levels load and bin matches load_mid exactly.
        subset = ["rx_time", "bid_prc", "ask_prc"] + (["exchange_time"] if mid_stream[ex] == "merged_levels" else [])
        df = (load_block(block, f"{ex}_{coin}", "front_levels")
              .select("rx_time", "exchange_time", "bid_prc", "bid_qty", "ask_prc", "ask_qty").drop_nulls(subset))
        return (df["rx_time"].cast(pl.Int64).to_numpy(), df["exchange_time"].cast(pl.Int64).to_numpy(),
                df["bid_prc"].to_numpy(), df["bid_qty"].to_numpy(), df["ask_prc"].to_numpy(), df["ask_qty"].to_numpy())

    books_raw = {ex: load_book(ex) for ex in all_ex}     # (rx, exch_time, bid, bid_qty, ask, ask_qty) per venue

    # shared trade clock: one tick per trade-TIMESTAMP across all venues (simultaneous prints = one tick)
    trade_ts = []
    for ex in all_ex:
        td = (load_block(block, f"{ex}_{coin}", "trade").select("rx_time", "prc", "qty")
              .filter((pl.col("prc") > 0) & (pl.col("qty") > 0)))
        trade_ts.append(td["rx_time"].cast(pl.Int64).to_numpy())

    if hours is not None:                                # read only the first `hours` of the block
        first = [a[0] for a in [mids[e][0] for e in all_ex] + trade_ts if len(a)]
        cutoff = min(first) + int(hours * 3600 * 1_000_000_000)
        mids = {e: (rx[rx <= cutoff], m[rx <= cutoff]) for e, (rx, m) in mids.items()}
        books_raw = {e: tuple(col[b[0] <= cutoff] for col in b) for e, b in books_raw.items()}
        trade_ts = [a[a <= cutoff] for a in trade_ts]

    # books a feature consumes (drop exchange_time, which only the raw-event stream below needs)
    books = {e: (rx, bid, bq, ask, aq) for e, (rx, _et, bid, bq, ask, aq) in books_raw.items()}

    def load_trade_flow(ex):                             # full raw trades WITH side — for trade-flow features.
        td = (load_block(block, f"{ex}_{coin}", "trade")
              .select("rx_time", "prc", "qty", "aggressor")
              .filter((pl.col("prc") > 0) & (pl.col("qty") > 0)))
        rx = td["rx_time"].cast(pl.Int64).to_numpy()
        out = (rx, td["prc"].to_numpy(),
               io._trade_lifts_ask(f"{ex}_{coin}", td["aggressor"].to_numpy()).astype(float),
               td["qty"].to_numpy())
        if hours is not None:
            return tuple(col[rx <= cutoff] for col in out)
        return out

    trades = {ex: load_trade_flow(ex) for ex in all_ex}

    merged_ts = np.unique(np.concatenate(trade_ts))
    n_ticks = len(merged_ts)
    event_ts = np.unique(np.concatenate([mids[e][0] for e in all_ex] + trade_ts))   # book + trade, any venue

    log_mid_target = np.log(_ffill(*mids[target_ex], merged_ts))

    # target move stream (for the yardsticks + the count target)
    t_rx0, t_mid0 = mids[target_ex]
    keep = np.concatenate([t_rx0[1:] != t_rx0[:-1], [True]])
    t_rx, t_mid = t_rx0[keep], t_mid0[keep]
    t_lm = np.log(t_mid)
    blr = np.empty_like(t_lm); blr[0] = 0.0; blr[1:] = np.diff(t_lm)
    mv = blr != 0.0
    mv_rx, mv_r2 = t_rx[mv], blr[mv] ** 2
    cum_mv = np.concatenate([[0.0], np.cumsum(mv.astype(float))])
    clock_dt = np.zeros(n_ticks); clock_dt[1:] = np.diff(merged_ts) / 1e9

    ctx = ScreeningContext(
        block=block, coin=coin, target=target, sources=tuple(sources), horizon_ns=horizon_ns,
        yardstick_span=yardstick_span, mid_stream=mid_stream,
        merged_ts=merged_ts, anchor_ts=np.empty(0), tick_at_anchor=np.empty(0),
        sigma_at_anchor=np.empty(0), lam_at_anchor=np.empty(0), price_target=np.empty(0), rate_target=np.empty(0),
        base=[], vol_level=np.empty(0), rate_level=np.empty(0), vol_regime=np.empty(0),
        raw_events=RawEventStream(*([np.empty(0)] * 8), ()), _mids=mids, _books=books, _trades=trades,
        target_logmid_on_clock=log_mid_target, _mv_rx=mv_rx, _mv_r2=mv_r2, _clock_dt=clock_dt,
    )

    # causal grid past warm-up
    warmup = warmup_spans * max(yardstick_span, max_feature_span)
    if n_ticks <= warmup:
        raise ValueError(f"block too thin: {n_ticks:,} trade ticks <= WARMUP {warmup:,}")
    end_ns = merged_ts[-1] - horizon_ns                  # last anchor that keeps a full forward window
    step_ns = grid_ms * 1_000_000
    if active_only:                                      # regular grid, but only windows carrying an event
        anchor_ts = _active_grid(merged_ts[warmup], end_ns, step_ns, event_ts)
    else:
        anchor_ts = np.arange(merged_ts[warmup], end_ns, step_ns)
    if len(anchor_ts) == 0:
        raise ValueError("empty anchor grid — block too thin for the chosen grid and horizon")
    ctx.anchor_ts = anchor_ts
    ctx.tick_at_anchor = np.searchsorted(merged_ts, anchor_ts, "right") - 1
    ctx.sigma_at_anchor, ctx.lam_at_anchor = ctx.yardsticks_at(anchor_ts, yardstick_span)

    # price target: byb 100ms forward return / σ_ev (guarded forward-fill — no wrap)
    inow = np.searchsorted(t_rx, anchor_ts, "right") - 1
    ifwd = np.searchsorted(t_rx, anchor_ts + horizon_ns, "right") - 1
    mid_now = np.where(inow < 0, np.nan, t_mid[np.clip(inow, 0, len(t_mid) - 1)])
    mid_fwd = np.where(ifwd < 0, np.nan, t_mid[np.clip(ifwd, 0, len(t_mid) - 1)])
    ctx.price_target = np.log(mid_fwd / mid_now) / ctx.sigma_at_anchor

    # rate target: byb move count over the horizon / λ_ev
    fwd_count = (cum_mv[np.searchsorted(t_rx, anchor_ts + horizon_ns, "right")]
                 - cum_mv[np.searchsorted(t_rx, anchor_ts, "right")])
    ctx.rate_target = fwd_count / np.maximum(ctx.lam_at_anchor, 1e-9)

    # Gate-B controls (regime-invariant momenta) + Gate-A regime coordinate (the levels)
    fast_yard = yardstick_span // 10
    sig_fast, lam_fast = ctx.yardsticks_at(anchor_ts, fast_yard)
    ctx.vol_level = np.log(ctx.sigma_at_anchor)
    ctx.rate_level = np.log(ctx.lam_at_anchor)
    vol_momentum = np.log(sig_fast / ctx.sigma_at_anchor)
    rate_momentum = np.log(lam_fast / ctx.lam_at_anchor)
    ctx.base = [rate_momentum, vol_momentum]
    finite = np.isfinite(ctx.vol_level)
    ctx.vol_regime = np.digitize(ctx.vol_level, np.nanpercentile(ctx.vol_level[finite], [33, 67]))

    # prepared raw-event stream (bounded), integer-coded, receive-time order, book-before-trade on ties
    listings = tuple(f"{ex}_{coin}" for ex in all_ex)        # index = lid
    cutoff = int(anchor_ts[min(raw_stream_cutoff_grid, len(anchor_ts) - 1)])
    cols: dict[str, list] = {k: [] for k in "rx kind lid t a b c d".split()}

    def add(rx, kind, lid, t, a, b, c, d):                  # book: (bid, ask, bid_qty, ask_qty); trade: (px, lifts_ask, qty, nan)
        m = rx <= cutoff; n = int(m.sum())
        cols["rx"].append(rx[m]); cols["kind"].append(np.full(n, kind, np.int8)); cols["lid"].append(np.full(n, lid, np.int8))
        cols["t"].append(t[m])
        for k, v in (("a", a), ("b", b), ("c", c), ("d", d)):
            cols[k].append(v[m].astype(float))

    for lid, ex in enumerate(all_ex):                       # every venue's REAL front_levels book (bid/ask + sizes);
        rx, et, bid, bq, ask, aq = books_raw[ex]            # merged venues fuse trades into it; book-only take the snapshot.
        add(rx, 0, lid, et, bid, ask, bq, aq)              # same array as `_books` -> OFI vectorized/streaming see one stream
    for lid, ex in enumerate(all_ex):                       # trades from every venue tick the clock; merged venues fuse
        td = (load_block(block, f"{ex}_{coin}", "trade")
              .select("rx_time", "exchange_time", "prc", "qty", "aggressor")
              .filter((pl.col("prc") > 0) & (pl.col("qty") > 0)))
        rx = td["rx_time"].cast(pl.Int64).to_numpy()
        add(rx, 1, lid, td["exchange_time"].cast(pl.Int64).to_numpy(), td["prc"].to_numpy(),
            io._trade_lifts_ask(f"{ex}_{coin}", td["aggressor"].to_numpy()).astype(float),
            td["qty"].to_numpy(), np.full(len(rx), np.nan))

    C = {k: np.concatenate(v) for k, v in cols.items()}
    order = np.lexsort((C["kind"], C["rx"]))                # rx asc; book(0) before trade(1) on ties
    ctx.raw_events = RawEventStream(*(C[k][order] for k in "rx kind lid t a b c d".split()), listings)
    return ctx


# --------------------------------------------------------------------------------------------------
# Step 2 — build / cache the vectorized family (one builder, two call sites).
# --------------------------------------------------------------------------------------------------
def build_family(
    ctx: ScreeningContext,
    vectorized: VectorizedBuilder,
    params_list: list[Params],
    n_jobs: int = 1,
) -> dict[Params, dict[str, np.ndarray]]:
    """Build the vectorized feature for each `params` token (parallel when `n_jobs > 1`), keyed by the
    OPAQUE token: `{params -> {leg_key -> vector}}`. Same function for the parity subset and the sweep."""
    if n_jobs <= 1:
        return {p: vectorized(ctx, p) for p in params_list}
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        return dict(zip(params_list, pool.map(lambda p: vectorized(ctx, p), params_list)))


# --------------------------------------------------------------------------------------------------
# Step 1's payoff — the completely generic parity driver.
# --------------------------------------------------------------------------------------------------
@dataclass
class ParityReport:
    tol: float
    max_diff: dict[tuple[Params, str], float]
    n_points: dict[tuple[Params, str], int]

    @property
    def passed(self) -> bool:
        return all(d <= self.tol for d in self.max_diff.values())

    def __str__(self) -> str:
        lines = [f"parity {'OK' if self.passed else 'FAILED'} (tol {self.tol:.0e}):"]
        for (p, k), d in self.max_diff.items():
            lines.append(f"  params={p!r:>12} leg={k:<10} max|diff| {d:.2e}  on {self.n_points[(p, k)]:,} pts")
        return "\n".join(lines)


def parity_check(
    ctx: ScreeningContext,
    spec: FeatureSpec,
    params_list: list[Params],
    n_grid: int = 200_000,
    tol: float = 1e-9,
) -> ParityReport:
    """Drive `spec.make_streaming(ctx, params)` for every `params` through `ctx.raw_events` in ONE
    pass — apply each timestamp's events to every builder, `refresh()` once, read `value()` at each
    anchor — then compare leg-by-leg to `spec.vectorized(ctx, params)` to `tol`. Generic: the same
    driver validates every feature."""
    anchor = ctx.anchor_ts
    na = min(n_grid, len(anchor))
    feats = {p: spec.make_streaming(ctx, p) for p in params_list}
    for p, f in feats.items():                              # fail fast on a key mismatch
        assert tuple(f.keys) == tuple(spec.keys_for(ctx, p)), f"key mismatch for params {p!r}"
    streams = {p: {k: np.full(na, np.nan) for k in f.keys} for p, f in feats.items()}

    ev = ctx.raw_events
    rxL = ev.rx.tolist(); kindL = ev.kind.tolist(); lidL = ev.lid.tolist()
    tL = ev.t.tolist(); aL = ev.a.tolist(); bL = ev.b.tolist(); cL = ev.c.tolist(); dL = ev.d.tolist()
    listings = ev.listings
    n = len(rxL); i = 0; ai = 0

    def read(idx):
        for p, f in feats.items():
            v = f.value()
            s = streams[p]
            for k in f.keys:
                s[k][idx] = v[k]

    while i < n and ai < na:
        rx = rxL[i]
        while ai < na and anchor[ai] < rx:
            read(ai); ai += 1
        while i < n and rxL[i] == rx:
            listing = listings[lidL[i]]
            if kindL[i] == 0:
                bev = BookEvent(listing, tL[i], aL[i], bL[i], cL[i], dL[i])   # build once, share across features
                for f in feats.values():
                    f.on_book(bev)
            else:
                tev = TradeEvent(listing, tL[i], aL[i], bL[i], cL[i])
                for f in feats.values():
                    f.on_trade(tev)
            i += 1
        for f in feats.values():
            f.refresh()
    while ai < na:                                          # trailing anchors after the last event
        read(ai); ai += 1

    ref = {p: spec.vectorized(ctx, p) for p in params_list}
    max_diff: dict = {}; n_points: dict = {}
    for p in params_list:
        for k in feats[p].keys:
            s = streams[p][k]; r = ref[p][k][:na]
            both = np.isfinite(s) & np.isfinite(r)
            max_diff[(p, k)] = float(np.nanmax(np.abs(s[both] - r[both]))) if both.any() else float("nan")
            n_points[(p, k)] = int(both.sum())
    return ParityReport(tol=tol, max_diff=max_diff, n_points=n_points)


# --------------------------------------------------------------------------------------------------
# Step 3 — the gates, one call per head.
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class HeadConfig:
    """Which target / yardstick / kind a gate run scores against. Two per feature: price + rate.

    score_magnitude    -- the rate head scores `|feature|` (the count relationship is symmetric); the
                          price head scores the signed feature.
    strat_var          -- the Gate-B coupling-guard stratifier for a CONTROL feature (the scored
                          target's yardstick); `None` for an alpha (signal_ic ignores it then).
    coupling_yardstick -- the yardstick for the coupling diagnostic ROW (σ_ev for price, λ_ev for rate).
    """

    name: str                          # "price" | "rate"
    target: np.ndarray
    score_magnitude: bool
    strat_var: Optional[np.ndarray]
    coupling_yardstick: np.ndarray
    feature_kind: str = "alpha"

    @staticmethod
    def price(ctx: "ScreeningContext", feature_kind: str = "alpha",
              target: Optional[np.ndarray] = None) -> "HeadConfig":
        """The price head. `target` overrides the default 100 ms `ctx.price_target` — pass a
        count-conditioned `fixed_move_targets[n]` (also σ_ev-divided, so the σ_ev coupling yardstick still
        applies) to gate against the move-count horizon the IC sweep chose, instead of the wall-clock return."""
        return HeadConfig("price", ctx.price_target if target is None else target,
                          False, None, ctx.sigma_at_anchor, feature_kind)

    @staticmethod
    def rate(ctx: "ScreeningContext", feature_kind: str = "alpha") -> "HeadConfig":
        return HeadConfig("rate", ctx.rate_target, True, None, ctx.lam_at_anchor, feature_kind)


@dataclass
class GateReport:
    head: str
    rows: list[dict]
    passed: bool

    def to_polars(self):
        import polars as pl
        return pl.DataFrame(self.rows)

    def __str__(self) -> str:
        head = f"gates [{self.head}] {'PASS' if self.passed else 'FAIL'}"
        return head + "\n" + "\n".join(f"  {r['gate']:<24} {r['detail'][:60]:<60} {r['value']}" for r in self.rows)


def run_gates(
    legs: dict[str, np.ndarray],
    ctx: ScreeningContext,
    head: HeadConfig,
    *,
    signal_floor: float = 0.01,
) -> GateReport:
    """Step 3. Score the SET of per-source legs against one head: the Gate-B marginal (joint AND per
    source) over `ctx.base`, the coupling rows (within-yardstick stratified IC), Gate A per source vs
    the regime coordinate, and the calm/mid/wild companion. All inner statistics come from
    `boba.research.gates`; this is the shared ORCHESTRATION. Call once per head."""
    from boba.research import gates as g

    keys = list(legs)
    score = {k: (np.abs(legs[k]) if head.score_magnitude else legs[k]) for k in keys}
    score_list = [score[k] for k in keys]
    kw = dict(feature_kind=head.feature_kind, own=False, strat_var=head.strat_var)

    joint = g.signal_ic(score_list, ctx.base, head.target, **kw)
    rows = [dict(gate=f"B · signal ({head.name})", detail="all sources together — marginal over the controls", value=joint)]
    rows += [dict(gate=f"B · signal ({head.name})", detail=f"{k} alone — marginal over the controls",
                  value=g.signal_ic([score[k]], ctx.base, head.target, **kw)) for k in keys]
    # coupling diagnostic: within-yardstick stratified IC (a shared denominator can't manufacture it)
    rows += [dict(gate=f"B · coupling ({head.name})", detail=f"{k} — score WITHIN yardstick strata",
                  value=g.stratified_ic(score[k], head.target, head.coupling_yardstick)) for k in keys]

    gate_a_ok = True
    for k in keys:
        a = g.gate_a(legs[k], ctx.vol_level, ctx.rate_level)
        rows += [dict(gate=f"A · regime-inv ({k})", detail="scale across vol buckets (max/min); want < ~3", value=a["scale"]),
                 dict(gate=f"A · regime-inv ({k})", detail="|IC(feature, vol/rate level)| signed-track; want < ~0.05", value=a["track"]),
                 dict(gate=f"A · regime-inv ({k})", detail="|IC(|feature|, vol/rate level)| mag-track; want < ~0.1", value=a["mag"]),
                 dict(gate=f"A · regime-inv ({k})", detail="per-decile-mean dispersion; want < ~0.1", value=a["disp"])]
        gate_a_ok &= (a["scale"] < 3.0 and a["track"] < 0.05 and a["mag"] < 0.1 and a["disp"] < 0.1)

    comp = g.signal_ic_by_regime(score_list, ctx.base, head.target, ctx.vol_regime, **kw)
    rows += [dict(gate=f"regime-stable ({head.name})", detail=f"signal within {nm}-vol (stay positive)",
                  value=comp.get(r, float("nan"))) for r, nm in [(0, "calm"), (1, "mid"), (2, "wild")]]

    passed = bool(np.isfinite(joint) and joint >= signal_floor and gate_a_ok)
    return GateReport(head=head.name, rows=rows, passed=passed)


# --------------------------------------------------------------------------------------------------
# Small shared diagnostics the screening notebook wires up (span pick + echo-netting).
# --------------------------------------------------------------------------------------------------
def best_span(
    ctx: ScreeningContext,
    family: dict[Params, dict[str, np.ndarray]],
    target: np.ndarray,
    *,
    score_magnitude: bool = False,
    mirror=None,
) -> Params:
    """In-sample span pick: the `params` maximising the mean (over legs) rank-IC against `target`
    (`|leg|` when `score_magnitude`). Used only to give a feature its best shot at the gate — span
    SELECTION proper is a later step, not screening.

    `mirror` (a feature reflection callable, e.g. `FeatureSpec.mirror`) mirror-augments the SIGNED pick —
    `ic(concat[leg, mirror(leg)], concat[target, -target])` — so the pick is direction-free. Ignored when
    `score_magnitude` (|·| is sign-blind). See AUTHORING.md → Mirror augmentation."""
    from boba.research import gates as g

    use_mirror = mirror is not None and not score_magnitude
    aug_target = np.concatenate([target, -target]) if use_mirror else target

    def _ic(v):
        score = np.abs(v) if score_magnitude else v
        return g.ic(np.concatenate([score, mirror(score)]), aug_target) if use_mirror else g.ic(score, target)

    best, best_score = None, -np.inf
    for p, legs in family.items():
        s = float(np.nanmean([_ic(v) for v in legs.values()]))
        if np.isfinite(s) and s > best_score:
            best, best_score = p, s
    return best


def _partial_ic(f: np.ndarray, y: np.ndarray, t: np.ndarray) -> float:
    """Partial rank-IC of `f` with `y` controlling for `t` — built from the tested masked IC (`gates.ic`)
    on the common-finite subset, combined with the standard partial-correlation formula."""
    from boba.research import gates as g

    v = np.isfinite(f) & np.isfinite(y) & np.isfinite(t)
    if v.sum() <= 100:
        return float("nan")
    rfy, rft, rty = g.ic(f[v], y[v]), g.ic(f[v], t[v]), g.ic(t[v], y[v])
    return (rfy - rft * rty) / math.sqrt(max((1.0 - rft ** 2) * (1.0 - rty ** 2), 1e-12))


def echo_netted_ic(ctx: ScreeningContext, leg: np.ndarray) -> dict[str, float]:
    """Is a leg's forward IC real prediction or an echo of the move already underway? Returns the raw
    forward IC, the backward IC (the echo size), and the echo-netted forward IC (the partial,
    controlling for the trailing `[anchor-H, anchor]` move). See METHOD.md → Echo-netting."""
    target_ex = ctx.target.split("_", 1)[0]
    rx, mid = ctx._mids[target_ex]

    def ret(t0, t1):
        return np.log(_ffill(rx, mid, t1) / _ffill(rx, mid, t0))

    from boba.research import gates as g

    fwd = ret(ctx.anchor_ts, ctx.anchor_ts + ctx.horizon_ns)
    trail = ret(ctx.anchor_ts - ctx.horizon_ns, ctx.anchor_ts)
    return dict(raw=g.ic(leg, fwd), backward=g.ic(leg, trail), netted=_partial_ic(leg, fwd, trail))
