"""Raw-atom feature dataset.

Implements the feature catalogue specified in docs/raw_features.md as a (N, F)
float32 matrix on a 1-millisecond grid. Columns are selected explicitly — a
dataset is an ordered tuple of template+args specs (:class:`boba.dataset_v2.columns.ColumnSpec`);
see :class:`DatasetRawConfig`.

The philosophy is *raw atoms*: each stored feature is a single primitive
transformation of the input event streams. Compositions (deviations, Z-scores,
Bollinger bands, variance ratios, etc.) are constructed downstream from these
atoms — not pre-computed and stored.

Pipeline:

    Phase 1 — Per-listing:
              (a) same-nanosecond aggregation of BBO and trade streams
              (b) event-clock EMAs (Section 3) via scipy.signal.lfilter
              (c) instantaneous BBO state + dt_{N}b + time-since features
              (d) forward-filled microprice on the 1ms grid
              (e) ms-grid EMAs (Section 2) via scipy.signal.lfilter
              All three listings run in parallel (ThreadPoolExecutor).

    Phase 2 — Assemble the (N, F) feature matrix in parallel grid chunks
              (default n_workers=18).

EMAs are computed in float64 internally and cast to float32 on output. Note
that variance reconstruction downstream (`ema_x_sq − ema_x²`) has two known
limitations the user should understand:

  1. EMA convergence transient: until ~3× span elapses since session start,
     the EMAs have not converged from their y[−1]=0 initial condition, and
     the recovered "variance" is biased by the transient (not just the
     variance of the input).

  2. Float32 storage precision: when x has large magnitude (microprice ≈ 0.15)
     and true variance is small (≤ 1e-9), the difference `ema_x_sq − ema_x²`
     between two near-equal float32 values loses meaningful significance.
     Reliable variance reconstruction requires true variance ≳ 1e-8 at
     microprice scale. If you need tighter precision, store these specific
     columns as float64, or re-compute the EMAs in your downstream pipeline.

For DOGE-perp these limits matter only at extremely short spans (100ms) on
near-constant prices. At 1s+ spans across typical market activity, the
reconstruction is accurate to within a factor of ~2.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import polars as pl

from boba.dataset_v2.columns import ColumnSpec, expand_columns, listing_spans, _clock_listing
from boba.dataset_v2.costs import OUTCOME_MS, CostConfig, _build_cost_tables, _cost_fields_for_grid
from boba.io import DATA_DIR, list_blocks, load_block
from boba.dataset_v2.session_data import SessionData, build_session_data


# ── Constants ─────────────────────────────────────────────────────────────────

_CACHE_VERSION = "v1"   # bump when the feature layout changes to invalidate old caches


def _silent(*args, **kwargs) -> None:
    del args, kwargs


# ── Config ────────────────────────────────────────────────────────────────────

# Allowed values for DatasetRawConfig.event_mask (which events define the output grid rows).
EVENT_MASK_OPTS = ("none", "book", "trade", "both", "trade_or_move")


@dataclass(frozen=True)
class DatasetRawConfig:
    """Dataset definition. The ONLY way to select features is ``columns`` — an
    ordered tuple of :class:`boba.dataset_v2.columns.ColumnSpec` (template + args; the
    template catalogue lives in docs/raw_features.md / boba.dataset_v2.columns.TEMPLATES).
    Output column order is exactly the expansion order, so permutations of the
    same selection are different datasets (and different cache keys)."""
    # Columns — required; defines the feature matrix.
    columns: tuple[ColumnSpec, ...]

    # Book listings to load. Every listing referenced by `columns` must be a
    # member; all listings contribute to the event grid / row filter even when
    # no column references them. The target listing provides the book for the
    # entry/exit cost fields.
    listings: tuple[str, ...] = (
        "bin_doge_usdt_p", "bin_doge_usdt",
        "byb_doge_usdt_p", "byb_doge_usdt",
        "okx_doge_usdt_p", "okx_doge_usdt",
    )
    target_listing: str = "bin_doge_usdt_p"

    # Wide-spread threshold per listing (linear, units of spread_width) — feeds
    # spread_wide_flag and its derived columns. Default tuned for DOGE-perp
    # (~1 tick / mid ≈ 0.7e-4 → threshold at 1.5 ticks); listings absent from
    # the dict fall back to 1.0e-4. Add entries for any custom listing set.
    wide_threshold: dict[str, float] = field(
        default_factory=lambda: {k: 1.0e-4 for k in (
            "bin", "byb", "okx",
            "bin_doge_usdt_p", "bin_doge_usdt", "byb_doge_usdt_p",
            "byb_doge_usdt", "okx_doge_usdt_p", "okx_doge_usdt",
        )}
    )

    # Reference price for centered microprice EMAs (per listing). The EMA
    # features `ema_microprice_centered_{N}ms` and `_sq_{N}ms` are computed on
    # (microprice − microprice_ref), so that variance reconstruction downstream
    # (`ema_sq − ema²`) is numerically stable at float32 storage. The
    # `microprice` raw column itself remains absolute (not centered). No default:
    # supply a per-listing value (≈ the listing's typical microprice). Listings
    # absent from the dict fall back to 0.0 (NO centering), so if you select the
    # centered families, set this for every listing or the variance-stabilization
    # is silently off (and the wrong constant defeats it for non-DOGE prices).
    microprice_ref: dict[str, float] = field(default_factory=dict)

    # Grid + horizon
    warmup_ms: int = 500
    horizon_ms: float = float(OUTCOME_MS)
    baseline_rt_ms: float = 3.0          # baseline round-trip for "order land" time
    processing_ms: float = 0.5           # processing overhead added to entry time
    # Cost fields to compute/keep — an explicit selection, exactly like `columns`: list the
    # cost-field names you want (the catalogue is `boba.dataset_v2.raw._COST_FIELDS`). The default
    # () computes none; there is no "all" sentinel. Selecting only what you need keeps the build
    # lean in memory (and skips the heavy trade min/max sparse tables when no extreme is listed).
    cost_fields: tuple[str, ...] = ()

    # Which events define the output grid rows (sparsifies the dense 1 ms grid). One of
    # ``EVENT_MASK_OPTS``:
    #   "none"          — every ms (dense; explodes row count on 24 h blocks)
    #   "book"          — ms containing a BBO/book update on any listing
    #   "trade"         — ms containing a trade on any listing
    #   "both"          — book OR trade (default)      [floor: the ms CONTAINING the event,
    #                                                    so its effect shows at the next kept row]
    #   "trade_or_move" — a trade or a microprice-CHANGING book move (price- or qty-driven;
    #                     size-only updates don't count) [ceil (t−1ms, t]: the kept row already
    #                     reflects the event that kept it]
    # Part of the grid identity → folded into config_str / grid_hash.
    event_mask: str = "both"

    # ── Performance / memory (does NOT affect dataset output, and is NEVER in config_str /
    #    cache_key — see plan_chunks in boba.dataset_v2.chunks) ───────────────────────────
    # These only steer HOW chunks are processed (parallelism + peak memory), not WHAT is
    # produced. Continuous-carry semantics make a block's output independent of how it was
    # chunked, so two machines with very different budgets build byte-identical caches.
    n_workers: int = 18                  # thread pool size for parallel within-chunk work (≈ cores)
    # Peak-memory ceiling for one chunk's compute. The planner coalesces small blocks up to
    # this budget and SPLITS oversized blocks below it, so the same build runs on a 128 GB box
    # (big chunks, default) or a 16 GB laptop (small chunks) — only throughput differs.
    mem_budget_gb: float = 96.0

    def __post_init__(self):
        if self.event_mask not in EVENT_MASK_OPTS:
            raise ValueError(f"event_mask must be one of {EVENT_MASK_OPTS}, got {self.event_mask!r}")

    def expanded(self):
        """Expand ``columns`` to concrete names/units. Structural validation
        only — listing membership is checked at the call site against the
        relevant vocabulary (build_features_raw → its ``listings`` argument,
        the block builders → ``cfg.listings``)."""
        return expand_columns(self.columns)

    def n_features(self) -> int:
        return len(self.expanded().names)

    def config_str(self) -> str:
        """Compact, deterministic encoding of the output-affecting feature/grid
        params (``listings`` and ``target_listing`` are folded in by
        :meth:`cache_key`, which is what the block cache uses). The expanded column names (ORDERED — column order is
        part of the output) and the per-listing value knobs fold into a short
        hash; scalar knobs stay readable. `n_workers` is excluded (it does not
        affect output)."""
        import hashlib
        exp = self.expanded()
        payload = repr((
            exp.names,
            sorted(self.wide_threshold.items()),
            sorted(self.microprice_ref.items()),
            tuple(sorted(self.cost_fields)),
        ))
        h = hashlib.sha1(payload.encode()).hexdigest()[:8]
        return (
            f"wup{self.warmup_ms}"
            f"_brt{self.baseline_rt_ms:.1f}_proc{self.processing_ms:.1f}"
            f"_h{self.horizon_ms:.0f}"
            f"_em{self.event_mask}"
            f"_f{len(exp.names)}"
            f"_{h}"
        )

    def grid_hash(self) -> str:
        """v2 per-column cache **directory** key: the grid identity — everything
        output-affecting EXCEPT the per-column selection (``columns``), the cost-field
        selection (``cost_fields``), and warmup/perf knobs (which never affect output, since
        continuous-carry makes a block chunk-independent). Columns/cost are carried by
        filenames instead. See docs/v2_dataset_design.md §6."""
        import hashlib
        payload = repr((
            tuple(sorted(self.listings)), self.target_listing,
            f"{self.horizon_ms:.3f}", f"{self.baseline_rt_ms:.3f}", f"{self.processing_ms:.3f}",
            self.event_mask,
            sorted(self.wide_threshold.items()), sorted(self.microprice_ref.items()),
        ))
        return "v2_" + hashlib.sha1(payload.encode()).hexdigest()[:12]

    def cache_key(self) -> str:
        """Per-block cache-key fragment: sorted-listings + target hashed, plus
        config_str — a different listing set, target, or column/knob config
        never collides."""
        import hashlib
        h = hashlib.sha1(repr((tuple(sorted(self.listings)), self.target_listing)).encode()).hexdigest()[:8]
        return f"{_CACHE_VERSION}_listings{len(self.listings)}_{h}_config_{self.config_str()}"


# ── Core helpers ──────────────────────────────────────────────────────────────

def _alpha(span: int) -> float:
    """Standard EMA span→alpha conversion (pandas / ta-lib convention)."""
    return 2.0 / (span + 1)


def _ewm_1d(x: np.ndarray, alpha: float) -> np.ndarray:
    """Causal EWM: y[i] = alpha*x[i] + (1-alpha)*y[i-1], y[-1]=0.

    Uses scipy.signal.lfilter — O(N) in C, releases the GIL.
    Always computed in float64; caller may cast.
    """
    from scipy.signal import lfilter
    if len(x) == 0:
        return np.empty(0, np.float64)
    return lfilter([alpha], [1.0, -(1.0 - alpha)], x.astype(np.float64))


def _grid_idx(event_t: np.ndarray, grid_t: np.ndarray) -> np.ndarray:
    """For each grid tick, return index of the last event at-or-before it. -1 if none."""
    return np.searchsorted(event_t, grid_t, side="right").astype(np.int64) - 1


def _any_event_in_ms(event_t: np.ndarray, grid_t_ns: np.ndarray) -> np.ndarray:
    """Bool array: True where ≥1 event falls in [grid_t_ns[i], grid_t_ns[i] + 1ms)."""
    next_t = grid_t_ns + 1_000_000
    lo = np.searchsorted(event_t, grid_t_ns, side="left")
    hi = np.searchsorted(event_t, next_t, side="left")
    return hi > lo


def trade_value_per_ms(
    tr_t: np.ndarray,
    tr_prc: np.ndarray,
    tr_qty: np.ndarray,
    tr_dir: np.ndarray,
    grid_t_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-ms buy/sell traded value (Σ qty·prc) summed onto the 1ms grid.

    Grid tick t accumulates the trades with timestamp in (t − 1ms, t] — exactly
    the trades that became visible at-or-before t since the previous tick,
    matching the forward_fill_to_ms_grid convention (a trade exactly AT the
    tick is included; one strictly after it belongs to the next tick). Ticks
    with no trade read 0: these are flow, not state, so no forward-fill.
    Buy/sell split by tr_dir (+1 = buy aggressor). Invariant to the same-ns
    VWAP aggregation, which preserves qty·prc per (ts, side) group.

    Returns (buy_value, sell_value), float64, each len(grid_t_ns).
    """
    n = len(grid_t_ns)
    if len(tr_t) == 0 or n == 0:
        return np.zeros(n, np.float64), np.zeros(n, np.float64)
    # First grid tick ≥ trade time = the tick whose (t − 1ms, t] window holds it
    idx = np.searchsorted(grid_t_ns, tr_t, side="left")
    in_grid = idx < n
    # Window check: drop trades >1ms before their assigned tick (before the
    # grid window, or inside a grid gap).
    in_window = np.zeros(len(tr_t), bool)
    in_window[in_grid] = (grid_t_ns[idx[in_grid]] - tr_t[in_grid]) < 1_000_000
    value = tr_qty.astype(np.float64) * tr_prc.astype(np.float64)
    buy_sel = in_window & (tr_dir > 0)
    sell_sel = in_window & ~(tr_dir > 0)
    buy = np.bincount(idx[buy_sel], weights=value[buy_sel], minlength=n)
    sell = np.bincount(idx[sell_sel], weights=value[sell_sel], minlength=n)
    return buy, sell


# ── Same-ns aggregation ───────────────────────────────────────────────────────

def aggregate_same_ns_bbo(
    ts: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    bid_qty: np.ndarray,
    ask_qty: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse same-ns BBO bursts to the final state at each timestamp.

    Reasoning: same-ns updates almost always come from one matching-engine
    event producing multiple feed messages. Keeping only the last state
    avoids inflating the BBO event clock and gives correct OFI/vol semantics.

    Returns deduplicated (ts, bid, ask, bid_qty, ask_qty), all length M ≤ N.
    """
    if len(ts) == 0:
        return ts, bid, ask, bid_qty, ask_qty
    # mask[i] = True at the last row of each ts-group
    keep = np.empty(len(ts), bool)
    keep[:-1] = np.diff(ts) != 0
    keep[-1] = True
    return ts[keep], bid[keep], ask[keep], bid_qty[keep], ask_qty[keep]


def aggregate_same_ns_trade(
    ts: np.ndarray,
    prc: np.ndarray,
    qty: np.ndarray,
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sum same-(ts, side) trade rows into single events.

    Same-ns trades are sweep fills from one aggressor; collapse to one event
    per (ts, side) with summed qty and notional-VWAP price.
    Rare: same-ts trades on opposite sides are kept separate (different events).
    """
    if len(ts) == 0:
        return ts, prc, qty, direction

    # Group key combines ts and side so opposite-side same-ts rows stay split
    side_int = (direction > 0).astype(np.int64)
    # Since ts is already sorted ascending, group changes when (ts, side) changes
    same_group = np.zeros(len(ts), bool)
    same_group[1:] = (ts[1:] == ts[:-1]) & (side_int[1:] == side_int[:-1])
    # group_id increments at boundaries
    group_id = np.cumsum(~same_group) - 1
    n_groups = int(group_id[-1]) + 1

    # Aggregate
    qty_sum = np.bincount(group_id, weights=qty, minlength=n_groups)
    notional_sum = np.bincount(group_id, weights=qty * prc, minlength=n_groups)
    # VWAP price within the group (zero-safe — qty_sum should be > 0 by construction)
    vwap = np.where(qty_sum > 0, notional_sum / np.maximum(qty_sum, 1e-30), prc[0])

    # Pick the first row of each group as the "representative" for ts and dir
    first_idx = np.concatenate(([0], np.where(~same_group[1:])[0] + 1))
    return ts[first_idx], vwap, qty_sum, direction[first_idx]


# ── Forward-fill onto the ms grid ─────────────────────────────────────────────

def forward_fill_to_ms_grid(
    event_t: np.ndarray,
    values: np.ndarray,
    grid_t_ns: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """For each grid tick, return values[i] where i is the index of the last
    event at-or-before the tick. Returns fill_value where no prior event exists.
    """
    if len(event_t) == 0:
        return np.full(len(grid_t_ns), fill_value, dtype=values.dtype)
    gi = _grid_idx(event_t, grid_t_ns)
    valid = gi >= 0
    gi_c = np.maximum(gi, 0)
    out = np.where(valid, values[gi_c], fill_value)
    return out.astype(values.dtype)


# ── dt_{N} computations ──────────────────────────────────────────────────────

def dt_over_last_n_events(ts: np.ndarray, N: int) -> np.ndarray:
    """For each event i, wall-clock time elapsed between events i-N and i.

    Returns 0 for the first N events (insufficient history).
    Output length = len(ts), dtype = same as ts.
    """
    out = np.zeros(len(ts), dtype=ts.dtype)
    if N <= 0 or len(ts) <= N:
        return out
    out[N:] = ts[N:] - ts[:-N]
    return out


def trailing_count_ms(event_t_ns: np.ndarray, grid_t_ns: np.ndarray, n_ms: int) -> np.ndarray:
    """At each grid tick t, the count of events in the trailing wall-clock window (t - n_ms, t].

    Causal (only events at or before t). Returns float64, length = len(grid_t_ns). Empty events → 0.
    """
    if len(event_t_ns) == 0:
        return np.zeros(len(grid_t_ns), dtype=np.float64)
    win_ns = int(n_ms) * 1_000_000
    hi = np.searchsorted(event_t_ns, grid_t_ns, side="right")
    lo = np.searchsorted(event_t_ns, grid_t_ns - win_ns, side="right")
    return (hi - lo).astype(np.float64)


def time_since_event_ms(event_t_ns: np.ndarray, grid_t_ns: np.ndarray) -> np.ndarray:
    """At each grid tick, ms elapsed since the most recent event (float32).

    Convention: at ticks with no prior event, return 0.0 (model treats as "fresh").
    """
    if len(event_t_ns) == 0:
        return np.zeros(len(grid_t_ns), dtype=np.float32)
    gi = _grid_idx(event_t_ns, grid_t_ns)
    valid = gi >= 0
    gi_c = np.maximum(gi, 0)
    elapsed_ns = grid_t_ns - event_t_ns[gi_c]
    elapsed_ms = elapsed_ns.astype(np.float64) / 1e6
    return np.where(valid, elapsed_ms, 0.0).astype(np.float32)


# ── Microprice + OFI events ───────────────────────────────────────────────────

def compute_microprice(
    bid: np.ndarray, ask: np.ndarray,
    bid_qty: np.ndarray, ask_qty: np.ndarray,
) -> np.ndarray:
    """Stoikov microprice: size-weighted mid pulled toward the thinner side.

    microprice = (bid_qty · ask + ask_qty · bid) / (bid_qty + ask_qty)
    """
    eps = 1e-30
    total = np.maximum(bid_qty + ask_qty, eps)
    return (bid_qty * ask + ask_qty * bid) / total


def compute_ofi_events(
    bid: np.ndarray, ask: np.ndarray,
    bid_qty: np.ndarray, ask_qty: np.ndarray,
) -> np.ndarray:
    """Cont-Kukanov order flow imbalance, signed per-event flow.

    Per consecutive (i-1 → i):
      e_bid = +bq[i]            if bid[i] > bid[i-1]   (bid stepped up)
              -bq[i-1]           if bid[i] < bid[i-1]   (bid backed away)
              +(bq[i] - bq[i-1]) if bid[i] = bid[i-1]   (qty change at same level)
      e_ask = -aq[i]            if ask[i] < ask[i-1]
              +aq[i-1]           if ask[i] > ask[i-1]
              -(aq[i] - aq[i-1]) if ask[i] = ask[i-1]
      ofi[i] = e_bid + e_ask

    Output length matches input; ofi[0] = 0 (no prior state).
    """
    N = len(bid)
    out = np.zeros(N, dtype=np.float64)
    if N < 2:
        return out
    pb, pa, pbq, paq = bid[:-1], ask[:-1], bid_qty[:-1], ask_qty[:-1]
    nb, na, nbq, naq = bid[1:],  ask[1:],  bid_qty[1:],  ask_qty[1:]

    e_bid = np.where(nb > pb,  nbq,
            np.where(nb < pb, -pbq, nbq - pbq))
    e_ask = np.where(na < pa, -naq,
            np.where(na > pa,  paq, -(naq - paq)))
    out[1:] = e_bid + e_ask
    return out


def compute_abs_log_ret(microprice: np.ndarray) -> np.ndarray:
    """|log(microprice_i / microprice_{i-1})| per BBO event. First entry = 0."""
    N = len(microprice)
    out = np.zeros(N, dtype=np.float64)
    if N < 2:
        return out
    eps = 1e-30
    safe = (microprice[:-1] > eps) & (microprice[1:] > eps)
    log_ret = np.where(
        safe,
        np.log(np.maximum(microprice[1:], eps) / np.maximum(microprice[:-1], eps)),
        0.0,
    )
    out[1:] = np.abs(log_ret)
    return out


# ── Per-listing computation: tick-clock EMAs (Section 3) ─────────────────────

def compute_bbo_event_emas(
    ts: np.ndarray,
    bid: np.ndarray, ask: np.ndarray,
    bid_qty: np.ndarray, ask_qty: np.ndarray,
    wide_threshold: float,
    spans: Mapping[str, Sequence[int]],
) -> dict[str, np.ndarray]:
    """All BBO-event-clock arrays for the requested ``spans`` — a mapping of
    template family → span list, e.g. {"ema_ofi_b": [3, 10], "dt_b": [100]}
    (families: see boba.dataset_v2.columns.TEMPLATES).

    Returns a dict keyed by feature name, each array length M (post-aggregation
    BBO event count). Each entry [i] = state after BBO event i. The raw
    event-level arrays (ts, microprice, …) are always included.
    """
    eps = 1e-30
    total = np.maximum(bid_qty + ask_qty, eps)
    book_imbalance = ((bid_qty - ask_qty) / total).astype(np.float64)
    book_depth = (bid_qty + ask_qty).astype(np.float64)
    microprice = compute_microprice(bid, ask, bid_qty, ask_qty).astype(np.float64)
    spread_width = ((ask - bid) / np.maximum(microprice, eps)).astype(np.float64)
    spread_wide_flag = (spread_width > wide_threshold).astype(np.float64)

    ofi_events = compute_ofi_events(bid, ask, bid_qty, ask_qty)
    abs_log_ret = compute_abs_log_ret(microprice)

    out: dict[str, np.ndarray] = {
        # Raw arrays kept for grid sampling
        "ts": ts,
        "microprice": microprice,
        "spread_width": spread_width,
        "book_depth": book_depth,
        "book_imbalance": book_imbalance,
        "spread_wide_flag": spread_wide_flag,
        "ofi_event": ofi_events,
    }

    for N in spans.get("ema_ofi_b", ()):
        out[f"ema_ofi_{N}b"] = _ewm_1d(ofi_events, _alpha(N))
    for N in spans.get("ema_ofi_sq_b", ()):
        out[f"ema_ofi_sq_{N}b"] = _ewm_1d(ofi_events ** 2, _alpha(N))
    for N in spans.get("ema_abs_log_ret_b", ()):
        out[f"ema_abs_log_ret_{N}b"] = _ewm_1d(abs_log_ret, _alpha(N))
    for N in spans.get("ema_book_imbalance_b", ()):
        out[f"ema_book_imbalance_{N}b"] = _ewm_1d(book_imbalance, _alpha(N))
    for N in spans.get("ema_book_imbalance_sq_b", ()):
        out[f"ema_book_imbalance_sq_{N}b"] = _ewm_1d(book_imbalance ** 2, _alpha(N))
    for N in spans.get("ema_book_depth_b", ()):
        out[f"ema_book_depth_{N}b"] = _ewm_1d(book_depth, _alpha(N))
    for N in spans.get("ema_book_depth_sq_b", ()):
        out[f"ema_book_depth_sq_{N}b"] = _ewm_1d(book_depth ** 2, _alpha(N))
    for N in spans.get("ema_spread_wide_flag_b", ()):
        out[f"ema_spread_wide_flag_{N}b"] = _ewm_1d(spread_wide_flag, _alpha(N))
    for N in spans.get("dt_b", ()):
        out[f"dt_{N}b"] = dt_over_last_n_events(ts, N)

    return out


def compute_trade_event_emas(
    ts: np.ndarray,
    prc: np.ndarray,
    qty: np.ndarray,
    direction: np.ndarray,
    spans: Mapping[str, Sequence[int]],
) -> dict[str, np.ndarray]:
    """Trade-event-clock arrays for the requested ``spans`` (families:
    "ema_buy_trade_qty_t", "ema_sell_trade_qty_t", "ema_buy_trade_value_t",
    "ema_sell_trade_value_t", "ema_trade_serial_cov_t", "dt_t").

    Splits each trade into buy-only or sell-only streams (opposite side gets 0)
    and runs EMAs over each. Also computes Roll's serial covariance EMAs from
    consecutive price differences.

    Returns dict keyed by feature name, each array length M (post-aggregation
    trade count). Entry [i] = state after trade event i.
    """
    out: dict[str, np.ndarray] = {"ts": ts}

    _STREAM_FAMS = (
        ("ema_buy_trade_qty_t",    "ema_buy_trade_qty_{N}t"),
        ("ema_sell_trade_qty_t",   "ema_sell_trade_qty_{N}t"),
        ("ema_buy_trade_value_t",  "ema_buy_trade_value_{N}t"),
        ("ema_sell_trade_value_t", "ema_sell_trade_value_{N}t"),
    )

    if len(ts) == 0:
        # Empty session — return zero-length arrays for all requested keys
        z = np.empty(0, np.float64)
        for fam, fmt in _STREAM_FAMS:
            for N in spans.get(fam, ()):
                out[fmt.format(N=N)] = z
        for N in spans.get("ema_trade_serial_cov_t", ()):
            out[f"ema_trade_serial_cov_{N}t"] = z
        for N in spans.get("dt_t", ()):
            out[f"dt_{N}t"] = np.empty(0, ts.dtype)
        return out

    is_buy = direction > 0
    qty_f = qty.astype(np.float64)
    prc_f = prc.astype(np.float64)
    value = qty_f * prc_f

    streams = {
        "ema_buy_trade_qty_t":    np.where(is_buy, qty_f, 0.0),
        "ema_sell_trade_qty_t":   np.where(~is_buy, qty_f, 0.0),
        "ema_buy_trade_value_t":  np.where(is_buy, value, 0.0),
        "ema_sell_trade_value_t": np.where(~is_buy, value, 0.0),
    }
    for fam, fmt in _STREAM_FAMS:
        for N in spans.get(fam, ()):
            out[fmt.format(N=N)] = _ewm_1d(streams[fam], _alpha(N))

    # Roll's serial covariance: EMA(Δp_t · Δp_{t-1})
    # Aligned with trade indices: cov[i] uses dp[i-1] and dp[i]; first two entries = 0.
    cov_spans = list(spans.get("ema_trade_serial_cov_t", ()))
    if cov_spans:
        cov_input = np.zeros(len(prc_f), np.float64)
        if len(prc_f) >= 3:
            dp = np.diff(prc_f)                       # length N-1
            dp_prod = dp[1:] * dp[:-1]                # length N-2; aligned at trade index 2..N-1
            cov_input[2:] = dp_prod
        for N in cov_spans:
            out[f"ema_trade_serial_cov_{N}t"] = _ewm_1d(cov_input, _alpha(N))

    for N in spans.get("dt_t", ()):
        out[f"dt_{N}t"] = dt_over_last_n_events(ts, N)

    return out


# ── Per-listing computation: ms-grid EMAs (Section 2) ────────────────────────

def compute_ms_grid_temporal(
    microprice_grid: np.ndarray,
    grid_t_ns: np.ndarray,
    spans: Mapping[str, Sequence[int]],
    microprice_ref: float = 0.0,
) -> dict[str, np.ndarray]:
    """Calendar-time features on the ms grid for the requested ``spans``
    (families: "return_ms", "ema_microprice_centered_ms",
    "ema_microprice_centered_sq_ms").

    `microprice_grid` is the forward-filled microprice per ms tick.
    `microprice_ref` is subtracted before computing the centered EMA family.
    Returns dict of feature arrays each of length N_grid.

    The centered EMAs (`ema_microprice_centered_{N}ms` and `_sq_{N}ms`)
    are computed on (microprice − microprice_ref). With small-magnitude
    centered values (~0.05 for DOGE-perp), the float32 storage roundtrip
    `ema_sq − ema²` retains useful variance precision.

    Returns use raw microprice (centering cancels in log ratios).
    """
    out: dict[str, np.ndarray] = {}
    eps = 1e-30
    safe = microprice_grid > eps

    # Returns: log(microprice_t / microprice_{t-Nms}) — uses raw microprice
    ret_spans = list(spans.get("return_ms", ()))
    if ret_spans:
        log_mp = np.where(safe, np.log(np.maximum(microprice_grid, eps)), 0.0)
        for Nms in ret_spans:
            ret = np.zeros_like(log_mp)
            if len(log_mp) > Nms:
                ret[Nms:] = log_mp[Nms:] - log_mp[:-Nms]
                ret_valid = safe[Nms:] & safe[:-Nms]
                ret[Nms:] = np.where(ret_valid, ret[Nms:], 0.0)
            out[f"return_{Nms}ms"] = ret

    # Centered EMAs — float64 throughout. Centering by microprice_ref
    # keeps ema² small and preserves variance precision in float32 storage.
    # Where microprice_grid == 0 (no event yet), keep the centered value 0
    # too — so the EMA doesn't pick up a phantom -ref signal during warmup.
    emac = list(spans.get("ema_microprice_centered_ms", ()))
    emasq = list(spans.get("ema_microprice_centered_sq_ms", ()))
    if emac or emasq:
        mp_centered = np.where(safe, microprice_grid - microprice_ref, 0.0)
        mp_centered_sq = mp_centered * mp_centered if emasq else None
        for Nms in sorted({*emac, *emasq}):
            a = _alpha(Nms)
            if Nms in emac:
                out[f"ema_microprice_centered_{Nms}ms"] = _ewm_1d(mp_centered, a)
            if Nms in emasq:
                out[f"ema_microprice_centered_sq_{Nms}ms"] = _ewm_1d(mp_centered_sq, a)

    # Return realized-variance EMA — EWMA of squared 1ms log-microprice returns
    # (RiskMetrics RV rate). Differences before squaring ⇒ drift-immune; the
    # downstream normalizer is r = sqrt(ema_microprice_return_sq)·1e4.
    retsq = list(spans.get("ema_microprice_return_sq_ms", ()))
    if retsq:
        log_mp = np.where(safe, np.log(np.maximum(microprice_grid, eps)), 0.0)
        ret1 = np.zeros_like(log_mp)
        ret1[1:] = np.where(safe[1:] & safe[:-1], log_mp[1:] - log_mp[:-1], 0.0)
        ret1_sq = ret1 * ret1
        for Nms in retsq:
            out[f"ema_microprice_return_sq_{Nms}ms"] = _ewm_1d(ret1_sq, _alpha(Nms))

    return out


# ── Feature columns ───────────────────────────────────────────────────────────

def feature_names(cfg: DatasetRawConfig, listings: list[str]) -> list[str]:
    """Expanded column names, in spec/expansion order. Validates that every
    listing referenced by ``cfg.columns`` is one of ``listings`` and that the
    expansion has no duplicate names."""
    return list(expand_columns(cfg.columns, listings=list(listings)).names)


# ── Per-listing shared state (event-level, kept across column shards) ───────

@dataclass
class ListingSharedState:
    """Event-level arrays for one listing. Stays in RAM across column shards.

    All BBO and trade arrays are POST same-ns aggregation. The forward-fill
    to the (event-filtered) grid is done per-column on demand to bound memory.
    """
    # Aggregated BBO event stream
    bbo_t:               np.ndarray   # rx timestamps (int64 ns)
    bbo_bid:             np.ndarray
    bbo_ask:             np.ndarray
    bbo_bq:              np.ndarray
    bbo_aq:              np.ndarray
    # Event-level derived BBO arrays
    microprice:          np.ndarray   # float64
    spread_width:        np.ndarray
    book_depth:          np.ndarray
    book_imbalance:      np.ndarray
    spread_wide_flag:    np.ndarray
    ofi_events:          np.ndarray
    abs_log_ret:         np.ndarray
    # Aggregated trade event stream
    tr_t:                np.ndarray
    tr_prc:              np.ndarray
    tr_qty:              np.ndarray
    tr_dir:              np.ndarray


def _build_shared_state(
    e: str, data: SessionData, cfg: DatasetRawConfig,
) -> ListingSharedState:
    """Aggregate same-ns events and compute event-level derived arrays.
    Output is ~150–500 MB per listing — cheap to keep in RAM."""
    bbo_t, bbo_bid, bbo_ask, bbo_bq, bbo_aq = aggregate_same_ns_bbo(
        data.listing_book_t[e],
        data.listing_book_bid[e],
        data.listing_book_ask[e],
        data.listing_book_bid_qty[e],
        data.listing_book_ask_qty[e],
    )
    tr_t, tr_prc, tr_qty, tr_dir = aggregate_same_ns_trade(
        data.trade_ts[e],
        data.trade_prc[e],
        data.trade_qty[e],
        data.trade_dir[e],
    )

    eps = 1e-30
    total = np.maximum(bbo_bq + bbo_aq, eps)
    book_imbalance = ((bbo_bq - bbo_aq) / total).astype(np.float64)
    book_depth = (bbo_bq + bbo_aq).astype(np.float64)
    microprice = compute_microprice(bbo_bid, bbo_ask, bbo_bq, bbo_aq).astype(np.float64)
    spread_width = ((bbo_ask - bbo_bid) / np.maximum(microprice, eps)).astype(np.float64)
    wide_thr = cfg.wide_threshold.get(e, 1.0e-4)
    spread_wide_flag = (spread_width > wide_thr).astype(np.float64)
    ofi_events = compute_ofi_events(bbo_bid, bbo_ask, bbo_bq, bbo_aq)
    abs_log_ret = compute_abs_log_ret(microprice)

    return ListingSharedState(
        bbo_t=bbo_t, bbo_bid=bbo_bid, bbo_ask=bbo_ask, bbo_bq=bbo_bq, bbo_aq=bbo_aq,
        microprice=microprice, spread_width=spread_width, book_depth=book_depth,
        book_imbalance=book_imbalance, spread_wide_flag=spread_wide_flag,
        ofi_events=ofi_events, abs_log_ret=abs_log_ret,
        tr_t=tr_t, tr_prc=tr_prc, tr_qty=tr_qty, tr_dir=tr_dir,
    )


def _fill_section1_columns(
    e: str, ss: ListingSharedState, data: SessionData,
    grid_t_ns_evt: np.ndarray, spans_e: Mapping[str, Sequence[int]],
    x_out: np.ndarray, col_idx: dict[str, int],
) -> None:
    """Section 1 + dt + time-since + feed_latency columns, written to x_out.
    Each forward-fill produces a per-event-grid array (~80 MB at typical sizes).

    ``col_idx`` maps this listing's selected per-listing names → absolute x_out
    column; names absent from it are SKIPPED entirely (not computed)."""
    if not col_idx:
        return

    def _put(name: str, arr: np.ndarray) -> None:
        x_out[:, col_idx[name]] = arr.astype(np.float32)

    # Instantaneous BBO state — forward-fill from aggregated BBO events
    for nm, arr in (
        ("microprice",       ss.microprice),
        ("spread_width",     ss.spread_width),
        ("book_depth",       ss.book_depth),
        ("book_imbalance",   ss.book_imbalance),
        ("spread_wide_flag", ss.spread_wide_flag),
    ):
        if nm in col_idx:
            _put(nm, forward_fill_to_ms_grid(ss.bbo_t, arr, grid_t_ns_evt, 0.0))

    # dt_{N}b — event-level dt array, forward-filled to grid (ns → ms)
    for N in spans_e.get("dt_b", ()):
        nm = f"dt_{N}b"
        if nm not in col_idx:
            continue
        dt_evt_ns = dt_over_last_n_events(ss.bbo_t, N).astype(np.float64) / 1e6
        _put(nm, forward_fill_to_ms_grid(ss.bbo_t, dt_evt_ns, grid_t_ns_evt, 0.0))

    # dt_{N}t — same for trades
    for N in spans_e.get("dt_t", ()):
        nm = f"dt_{N}t"
        if nm not in col_idx:
            continue
        if len(ss.tr_t) > 0:
            dt_evt_ns = dt_over_last_n_events(ss.tr_t, N).astype(np.float64) / 1e6
            _put(nm, forward_fill_to_ms_grid(ss.tr_t, dt_evt_ns, grid_t_ns_evt, 0.0))
        else:
            _put(nm, np.zeros(len(grid_t_ns_evt), np.float64))

    # dt_{N}m — dt over the last N MID moves (book_mid = (bid+ask)/2 changes), forward-filled to grid
    dt_m_spans = spans_e.get("dt_m", ())
    if dt_m_spans:
        mid = (ss.bbo_bid + ss.bbo_ask) / 2.0
        mvmask = np.zeros(len(ss.bbo_t), bool)
        if len(ss.bbo_t) > 1:
            mvmask[1:] = np.diff(mid) != 0.0
        mid_t = ss.bbo_t[mvmask]
        for N in dt_m_spans:
            nm = f"dt_{N}m"
            if nm not in col_idx:
                continue
            if len(mid_t) > 0:
                dt_evt = dt_over_last_n_events(mid_t, N).astype(np.float64) / 1e6
                _put(nm, forward_fill_to_ms_grid(mid_t, dt_evt, grid_t_ns_evt, 0.0))
            else:
                _put(nm, np.zeros(len(grid_t_ns_evt), np.float64))

    # trade_count_{N}ms — trailing wall-clock trade count over (t - N ms, t] (causal; fine lead-lag)
    for N in spans_e.get("trade_count_ms", ()):
        nm = f"trade_count_{N}ms"
        if nm in col_idx:
            _put(nm, trailing_count_ms(ss.tr_t, grid_t_ns_evt, N))

    # Time-since-last-trade
    if "time_since_last_trade_ms" in col_idx:
        _put("time_since_last_trade_ms", time_since_event_ms(ss.tr_t, grid_t_ns_evt).astype(np.float64))

    # Time-since-spread-wide — events where spread_wide_flag == 1
    if "time_since_spread_wide_ms" in col_idx:
        if len(ss.bbo_t) > 0:
            wide_idx = np.where(ss.spread_wide_flag > 0.5)[0]
            wide_event_t = ss.bbo_t[wide_idx] if len(wide_idx) > 0 else np.empty(0, ss.bbo_t.dtype)
        else:
            wide_event_t = np.empty(0, np.int64)
        _put("time_since_spread_wide_ms", time_since_event_ms(wide_event_t, grid_t_ns_evt).astype(np.float64))

    # Feed latency — forward-fill from RAW (un-aggregated) BBO timestamps
    if "feed_latency_excess_ms" in col_idx:
        feed_lat_ms = data.listing_feed_latency_excess_ns[e].astype(np.float64) / 1e6
        _put("feed_latency_excess_ms",
             forward_fill_to_ms_grid(data.listing_book_t[e], feed_lat_ms, grid_t_ns_evt, 0.0))


def _fill_section2_calendar(
    ss: ListingSharedState, grid_t_ns: np.ndarray,
    event_mask: np.ndarray, spans_e: Mapping[str, Sequence[int]],
    mp_ref: float,
    x_out: np.ndarray, col_idx: dict[str, int],
) -> None:
    """Section 2 — calendar-time features. The microprice grid (full ms grid,
    ~1.1 GB at 274M ticks) is materialized briefly per listing for the EMA
    passes, then freed. Each EMA / return is subset to event_mask before
    storage.

    ``col_idx`` maps this listing's selected per-listing names → absolute x_out
    column; unselected names are skipped, and the full-grid microprice is not
    materialized at all when nothing from this section is selected."""
    def _put(name: str, arr: np.ndarray) -> None:
        x_out[:, col_idx[name]] = arr.astype(np.float32)

    ret_spans = list(spans_e.get("return_ms", ()))
    emac_spans = list(spans_e.get("ema_microprice_centered_ms", ()))
    emasq_spans = list(spans_e.get("ema_microprice_centered_sq_ms", ()))
    retsq_spans = list(spans_e.get("ema_microprice_return_sq_ms", ()))
    if not (ret_spans or emac_spans or emasq_spans or retsq_spans):
        return

    # Forward-fill microprice to FULL ms grid (this is the costly array)
    microprice_full = forward_fill_to_ms_grid(
        ss.bbo_t, ss.microprice, grid_t_ns, fill_value=0.0,
    )
    eps = 1e-30
    safe = microprice_full > eps

    # Returns
    if ret_spans:
        log_mp_full = np.where(safe, np.log(np.maximum(microprice_full, eps)), 0.0)
        for Nms in ret_spans:
            ret_full = np.zeros_like(log_mp_full)
            if len(log_mp_full) > Nms:
                diff = log_mp_full[Nms:] - log_mp_full[:-Nms]
                valid = safe[Nms:] & safe[:-Nms]
                ret_full[Nms:] = np.where(valid, diff, 0.0)
            _put(f"return_{Nms}ms", ret_full[event_mask])
            del ret_full
        del log_mp_full

    # Centered calendar-time EMAs
    if emac_spans or emasq_spans:
        mp_centered = np.where(safe, microprice_full - mp_ref, 0.0)
        mp_centered_sq = mp_centered * mp_centered if emasq_spans else None
        for Nms in sorted({*emac_spans, *emasq_spans}):
            a = _alpha(Nms)
            if Nms in emac_spans:
                ema_c = _ewm_1d(mp_centered, a)
                _put(f"ema_microprice_centered_{Nms}ms", ema_c[event_mask])
                del ema_c
            if Nms in emasq_spans:
                ema_csq = _ewm_1d(mp_centered_sq, a)
                _put(f"ema_microprice_centered_sq_{Nms}ms", ema_csq[event_mask])
                del ema_csq
        del mp_centered, mp_centered_sq

    # Return realized-variance EMA (RV rate) — drift-immune, time-clock.
    if retsq_spans:
        log_mp_full = np.where(safe, np.log(np.maximum(microprice_full, eps)), 0.0)
        ret1 = np.zeros_like(log_mp_full)
        ret1[1:] = np.where(safe[1:] & safe[:-1], log_mp_full[1:] - log_mp_full[:-1], 0.0)
        ret1_sq = ret1 * ret1
        for Nms in retsq_spans:
            _put(f"ema_microprice_return_sq_{Nms}ms", _ewm_1d(ret1_sq, _alpha(Nms))[event_mask])
        del log_mp_full, ret1, ret1_sq

    del microprice_full


def _fill_trade_value_ms_columns(
    ss: ListingSharedState, grid_t_ns: np.ndarray,
    event_mask: np.ndarray, spans_e: Mapping[str, Sequence[int]],
    x_out: np.ndarray, col_idx: dict[str, int],
) -> None:
    """Per-ms trade-flow columns: raw buy/sell value sums + wallclock EMAs.

    The sums are scattered onto the FULL ms grid (quiet ms = 0) and the EMAs
    run over that full grid — so they absorb a 0 every quiet ms and decay in
    real time during silence (the point of the `_ms` family vs the
    event-clocked `_t` EMAs) — then everything is subset to event_mask for
    storage. Holds two full-grid float64 arrays (+1 per EMA pass): same
    footprint class as _fill_section2_calendar.

    ``col_idx`` maps this listing's selected per-listing names → absolute x_out
    column; the full-grid scatter is skipped when none of this family is selected."""
    def _put(name: str, arr: np.ndarray) -> None:
        x_out[:, col_idx[name]] = arr.astype(np.float32)

    buy_spans = list(spans_e.get("ema_buy_trade_value_ms", ()))
    sell_spans = list(spans_e.get("ema_sell_trade_value_ms", ()))
    tv_names = ["buy_trade_value", "sell_trade_value"] + [
        f"ema_buy_trade_value_{N}ms" for N in buy_spans
    ] + [f"ema_sell_trade_value_{N}ms" for N in sell_spans]
    if not any(n in col_idx for n in tv_names):
        return

    buy_full, sell_full = trade_value_per_ms(
        ss.tr_t, ss.tr_prc, ss.tr_qty, ss.tr_dir, grid_t_ns,
    )
    if "buy_trade_value" in col_idx:
        _put("buy_trade_value", buy_full[event_mask])
    if "sell_trade_value" in col_idx:
        _put("sell_trade_value", sell_full[event_mask])
    for N in buy_spans:
        if f"ema_buy_trade_value_{N}ms" in col_idx:
            ema_b = _ewm_1d(buy_full, _alpha(N))
            _put(f"ema_buy_trade_value_{N}ms", ema_b[event_mask])
            del ema_b
    for N in sell_spans:
        if f"ema_sell_trade_value_{N}ms" in col_idx:
            ema_s = _ewm_1d(sell_full, _alpha(N))
            _put(f"ema_sell_trade_value_{N}ms", ema_s[event_mask])
            del ema_s
    del buy_full, sell_full


# ── Single-call reference path (used by tests) ────────────────────────────────

def _compute_per_listing(
    e: str,
    data: SessionData,
    grid_t_ns: np.ndarray,
    cfg: DatasetRawConfig,
) -> dict[str, np.ndarray]:
    """All requested ms-grid feature arrays for one listing.

    Returns a dict keyed by the listing's LOCAL feature names (no {LISTING}_
    prefix), covering exactly the per-listing columns ``cfg`` requests for
    ``e``, in expansion order. Arrays have length N_grid; values computed in
    float64 (cast to float32 happens at storage in the production path). This
    is the slow reference path used by the tests to check the parallel
    production path against.
    """
    exp = expand_columns(cfg.columns)
    units_e = [u for u in exp.units if u.listing == e]
    requested = [u.local_name for u in units_e]
    req = set(requested)
    spans = listing_spans(exp.units, e)

    # ── Aggregate same-ns BBO and trades ──────────────────────────────────────
    bbo_t, bbo_bid, bbo_ask, bbo_bq, bbo_aq = aggregate_same_ns_bbo(
        data.listing_book_t[e],
        data.listing_book_bid[e],
        data.listing_book_ask[e],
        data.listing_book_bid_qty[e],
        data.listing_book_ask_qty[e],
    )
    tr_t, tr_prc, tr_qty, tr_dir = aggregate_same_ns_trade(
        data.trade_ts[e],
        data.trade_prc[e],
        data.trade_qty[e],
        data.trade_dir[e],
    )
    wide_thr = cfg.wide_threshold.get(e, 1.0e-4)

    # ── Tick-clock EMAs (Section 3) ───────────────────────────────────────────
    bbo_evt = compute_bbo_event_emas(bbo_t, bbo_bid, bbo_ask, bbo_bq, bbo_aq, wide_thr, spans)
    trd_evt = compute_trade_event_emas(tr_t, tr_prc, tr_qty, tr_dir, spans)

    out: dict[str, np.ndarray] = {}

    # ── Instantaneous BBO state, forward-filled to the grid ───────────────────
    microprice_grid = forward_fill_to_ms_grid(bbo_t, bbo_evt["microprice"], grid_t_ns, fill_value=0.0)
    if "microprice" in req:
        out["microprice"] = microprice_grid
    for nm in ("spread_width", "book_depth", "book_imbalance", "spread_wide_flag"):
        if nm in req:
            out[nm] = forward_fill_to_ms_grid(bbo_t, bbo_evt[nm], grid_t_ns, fill_value=0.0)

    # ── dt features: sample event-clock arrays to ms grid ─────────────────────
    for N in spans.get("dt_b", ()):
        dt_ms = bbo_evt[f"dt_{N}b"].astype(np.float64) / 1e6
        out[f"dt_{N}b"] = forward_fill_to_ms_grid(bbo_t, dt_ms, grid_t_ns, fill_value=0.0)
    for N in spans.get("dt_t", ()):
        if len(tr_t) > 0:
            dt_ms = trd_evt[f"dt_{N}t"].astype(np.float64) / 1e6
        else:
            dt_ms = np.empty(0, np.float64)
        out[f"dt_{N}t"] = forward_fill_to_ms_grid(tr_t, dt_ms, grid_t_ns, fill_value=0.0)

    # ── Time-since-event + feed latency ───────────────────────────────────────
    if "time_since_last_trade_ms" in req:
        out["time_since_last_trade_ms"] = time_since_event_ms(tr_t, grid_t_ns).astype(np.float64)
    if "time_since_spread_wide_ms" in req:
        if len(bbo_t) > 0:
            wide_idx = np.where(bbo_evt["spread_wide_flag"] > 0.5)[0]
            wide_event_t = bbo_t[wide_idx] if len(wide_idx) > 0 else np.empty(0, bbo_t.dtype)
        else:
            wide_event_t = np.empty(0, np.int64)
        out["time_since_spread_wide_ms"] = time_since_event_ms(wide_event_t, grid_t_ns).astype(np.float64)
    if "feed_latency_excess_ms" in req:
        feed_lat_ms = data.listing_feed_latency_excess_ns[e].astype(np.float64) / 1e6
        out["feed_latency_excess_ms"] = forward_fill_to_ms_grid(
            data.listing_book_t[e], feed_lat_ms, grid_t_ns, fill_value=0.0)

    # ── Calendar-time EMAs (Section 2) ────────────────────────────────────────
    mp_ref = cfg.microprice_ref.get(e, 0.0)
    out.update(compute_ms_grid_temporal(microprice_grid, grid_t_ns, spans, microprice_ref=mp_ref))

    # ── Per-ms trade-flow sums + wallclock EMAs ───────────────────────────────
    buy_val_ms, sell_val_ms = trade_value_per_ms(tr_t, tr_prc, tr_qty, tr_dir, grid_t_ns)
    if "buy_trade_value" in req:
        out["buy_trade_value"] = buy_val_ms
    if "sell_trade_value" in req:
        out["sell_trade_value"] = sell_val_ms
    for N in spans.get("ema_buy_trade_value_ms", ()):
        out[f"ema_buy_trade_value_{N}ms"] = _ewm_1d(buy_val_ms, _alpha(N))
    for N in spans.get("ema_sell_trade_value_ms", ()):
        out[f"ema_sell_trade_value_{N}ms"] = _ewm_1d(sell_val_ms, _alpha(N))

    # ── Sample Section 3 tick-clock EMAs to ms grid ───────────────────────────
    base_keys = {"ts", "microprice", "spread_width", "book_depth",
                 "book_imbalance", "spread_wide_flag", "ofi_event"}
    for key, arr in bbo_evt.items():
        if key in base_keys or key.startswith("dt_"):
            continue
        out[key] = forward_fill_to_ms_grid(bbo_t, arr, grid_t_ns, fill_value=0.0)
    if len(tr_t) > 0:
        for key, arr in trd_evt.items():
            if key == "ts" or key.startswith("dt_"):
                continue
            out[key] = forward_fill_to_ms_grid(tr_t, arr, grid_t_ns, fill_value=0.0)
    else:
        # No trades → zero-fill all requested trade EMA features
        for fam, fmt in (
            ("ema_buy_trade_qty_t",    "ema_buy_trade_qty_{N}t"),
            ("ema_sell_trade_qty_t",   "ema_sell_trade_qty_{N}t"),
            ("ema_buy_trade_value_t",  "ema_buy_trade_value_{N}t"),
            ("ema_sell_trade_value_t", "ema_sell_trade_value_{N}t"),
            ("ema_trade_serial_cov_t", "ema_trade_serial_cov_{N}t"),
        ):
            for N in spans.get(fam, ()):
                out[fmt.format(N=N)] = np.zeros(len(grid_t_ns), np.float64)

    # Exactly the requested local names, in expansion order
    return {nm: out[nm] for nm in requested}


# ── Main build entry point ────────────────────────────────────────────────────

@dataclass
class SampleArraysRaw:
    """Output of the raw-atom builder.

    Features:
      x:               (N, F) float32 feature matrix, columns in cfg.columns expansion order
      timestamp_ms:    (N,)  float64 grid tick time in ms
      column_names:    list of column names matching x columns

    Entry-time book state (target listing):
      eval_bid_l:      (N,) float32 — log(bid / mid) at model fire time
      eval_ask_l:      (N,) float32 — log(ask / mid)
      eval_mid:        (N,) float32 — raw mid at fire time

    Cost fields (log-ratios vs eval_mid, all float32):
      c_ask_entry_l, c_bid_entry_l                     — cost at order landing
      c_ask_exit_l, c_bid_exit_l, c_mid_exit_l         — cost at horizon
      c_buy_trade_min_l, c_buy_trade_max_l             — outcome-window trade extremes
      c_sell_trade_min_l, c_sell_trade_max_l

    Operational:
      feed_latency_raw_ms:    (N,) float32
      feed_latency_excess_ms: (N,) float32
    """
    x: np.ndarray
    timestamp_ms: np.ndarray
    column_names: list[str]
    # Cost fields are OPTIONAL: only those listed in DatasetRawConfig.cost_fields are populated;
    # unrequested ones stay None (the default cost_fields=() populates none) — lean memory.
    # Entry-time book state
    eval_bid_l: Optional[np.ndarray] = None
    eval_ask_l: Optional[np.ndarray] = None
    eval_mid: Optional[np.ndarray] = None
    # Cost at entry/exit
    c_ask_entry_l: Optional[np.ndarray] = None
    c_bid_entry_l: Optional[np.ndarray] = None
    c_ask_exit_l: Optional[np.ndarray] = None
    c_bid_exit_l: Optional[np.ndarray] = None
    c_mid_exit_l: Optional[np.ndarray] = None
    # Forward-window mid-move count over (eval, exit] — rate/count-head target (count==0 ⇒ no move)
    c_mid_move_count: Optional[np.ndarray] = None
    # Outcome window trade extremes
    c_buy_trade_min_l: Optional[np.ndarray] = None
    c_buy_trade_max_l: Optional[np.ndarray] = None
    c_sell_trade_min_l: Optional[np.ndarray] = None
    c_sell_trade_max_l: Optional[np.ndarray] = None
    # Operational
    feed_latency_raw_ms: Optional[np.ndarray] = None
    feed_latency_excess_ms: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.x)


def _cost_cfg(cfg: DatasetRawConfig) -> CostConfig:
    """Bridge to the shared cost-field machinery. The cost code only reads
    `baseline_rt_ms`, `processing_ms`, and `horizon_ms`, so we construct a
    minimal CostConfig with the matching values."""
    return CostConfig(
        baseline_rt_ms=cfg.baseline_rt_ms,
        processing_ms=cfg.processing_ms,
        horizon_ms=cfg.horizon_ms,
        cost_fields=cfg.cost_fields,
    )


def _grid_bounds_ms(t_start_ns: int, t_end_ns: int, warmup_ms: int, horizon_ms: float) -> tuple[int, int]:
    """Inclusive [start, end] ms bounds of the usable grid; empty iff end < start.

    Single source of truth for the grid flooring arithmetic — build_features_raw and
    build_dataset's degenerate-block pre-check must agree exactly, so both call this.
    Arithmetic is verbatim from the original inline version (float division preserved)."""
    warmup_ns = int(warmup_ms * 1_000_000)
    horizon_ns = int(horizon_ms * 1_000_000)
    grid_start_ms = int((t_start_ns + warmup_ns) / 1_000_000) + 1
    grid_end_ms = int((t_end_ns - horizon_ns) / 1_000_000)
    return grid_start_ms, grid_end_ms


def _trade_flow_input(ss: ListingSharedState, family: str) -> np.ndarray:
    """Per-trade quantity for a trade-flow ``_t`` family — identical to the Section-3 streams, so
    a FLOW summed over the listing's OWN trades (one trade per own-clock interval) reproduces the
    bare own-clock column exactly."""
    is_buy = ss.tr_dir > 0
    qty = ss.tr_qty.astype(np.float64)
    val = qty * ss.tr_prc.astype(np.float64)
    if family == "ema_buy_trade_qty_t":    return np.where(is_buy, qty, 0.0)
    if family == "ema_sell_trade_qty_t":   return np.where(~is_buy, qty, 0.0)
    if family == "ema_buy_trade_value_t":  return np.where(is_buy, val, 0.0)
    if family == "ema_sell_trade_value_t": return np.where(~is_buy, val, 0.0)
    cov = np.zeros(len(ss.tr_prc), np.float64)                 # ema_trade_serial_cov_t
    if len(ss.tr_prc) >= 3:
        dp = np.diff(ss.tr_prc.astype(np.float64))
        cov[2:] = dp[1:] * dp[:-1]
    return cov


def build_features_raw(
    data: SessionData,
    listings: list[str],
    cfg: DatasetRawConfig,
    log=_silent,
    n_workers: Optional[int] = None,
) -> SampleArraysRaw:
    """Build the full (N, F) raw-atom dataset + cost fields.

    Memory-bounded pipeline:
      1. Build shared event-level state per listing (~150–500 MB / listing).
      2. Determine event mask on full ms grid; collapse to event-only grid
         (typically 10–20× smaller).
      3. Allocate output x_out sized to the event grid (~11 GB at 274M-tick
         session with ~13M event ticks × 225 cols × float32).
      4. Fill columns in groups, freeing intermediates per group:
            Section 1 (instantaneous, dt, time_since, latency)
            Section 3 (tick-clock EMAs)
            Section 2 (calendar-time EMAs — needs full-grid microprice
                       temporarily, ~1.1 GB per listing, freed after use)
            Cost fields (chunked)
      5. Save / return.

    Peak memory at 274M-tick / 13M event-tick scale: ~25 GB.
    """
    if n_workers is None:
        n_workers = cfg.n_workers
    # Expand + validate the requested columns up front (fail fast, before any
    # heavy work): every referenced listing must be one of `listings`.
    exp = expand_columns(cfg.columns, listings=list(listings))
    cols = list(exp.names)
    t0 = time.perf_counter()
    def _t() -> str: return f"[{time.perf_counter() - t0:5.1f}s]"

    # ── Construct full ms grid (not materialized except as integer array) ────
    t_start_ns = min(data.listing_book_t[e][0] for e in listings)
    t_end_ns = max(data.listing_book_t[e][-1] for e in listings)
    grid_start_ms, grid_end_ms = _grid_bounds_ms(t_start_ns, t_end_ns, cfg.warmup_ms, cfg.horizon_ms)
    if grid_end_ms < grid_start_ms:
        raise ValueError(
            f"empty grid: block span {(t_end_ns - t_start_ns) / 1e9:.1f}s ≤ warmup+horizon "
            f"({(cfg.warmup_ms + cfg.horizon_ms) / 1e3:.1f}s) — nothing to build")
    grid_ms = np.arange(grid_start_ms, grid_end_ms + 1, dtype=np.int64)
    grid_t_ns = grid_ms * 1_000_000
    N_full = len(grid_t_ns)
    log(f"{_t()} full grid {N_full:,} ticks")

    # ── Build shared state per listing (event-level) in parallel ────────────
    def _shared(e: str):
        t1 = time.perf_counter()
        ss = _build_shared_state(e, data, cfg)
        return e, ss, time.perf_counter() - t1

    with ThreadPoolExecutor(max_workers=len(listings)) as pool:
        results = list(pool.map(_shared, listings))
    shared: dict[str, ListingSharedState] = {e: ss for e, ss, _ in results}
    for e, _, dt in sorted(results):
        log(f"{_t()}   {e} shared state ready ({dt:.1f}s) — "
            f"{len(shared[e].bbo_t):,} bbo / {len(shared[e].tr_t):,} trades")

    # ── Determine event mask (FULL grid) ─────────────────────────────────────
    # Fast path: convert event ns timestamps to ms tick indices directly, instead
    # of doing searchsorted-style range queries on the 274M-tick grid which
    # allocate ~4 GB of int64 intermediates per call.
    log(f"{_t()} computing event mask…")
    t1 = time.perf_counter()
    em = cfg.event_mask
    if em == "none":
        event_mask = np.ones(N_full, bool)
    elif em == "trade_or_move":
        # Keep tick t iff a trade or a microprice-changing BBO event falls in (t−1ms, t] on any
        # listing — the at-or-before window (ceil), so the kept row already reflects the event.
        # Move events = aggregated BBO events with abs_log_ret > 0 (exact 0.0 when unchanged).
        event_mask = np.zeros(N_full, bool)
        for e in sorted(listings):
            ss = shared[e]
            move_t = ss.bbo_t[ss.abs_log_ret > 0]
            for ev_t in (move_t, ss.tr_t):
                if len(ev_t) == 0:
                    continue
                ms_ceil = (ev_t + 999_999) // 1_000_000   # tick whose (t−1, t] holds the event
                in_range = (ms_ceil >= grid_start_ms) & (ms_ceil <= grid_end_ms)
                if in_range.any():
                    event_mask[ms_ceil[in_range] - grid_start_ms] = True
            log(f"{_t()}   {e} mask updated ({len(move_t):,} move + {len(ss.tr_t):,} trade events)")
    else:
        # "book" | "trade" | "both" — floor: the ms CONTAINING a book update and/or a trade.
        event_mask = np.zeros(N_full, bool)
        for e in sorted(listings):
            srcs = []
            if em in ("book", "both"):
                srcs.append((data.listing_book_t[e] // 1_000_000).astype(np.int64))
            if em in ("trade", "both"):
                tt = data.trade_ts[e]
                srcs.append((tt // 1_000_000).astype(np.int64) if len(tt) > 0 else np.empty(0, np.int64))
            for ms_arr in srcs:
                if len(ms_arr) == 0:
                    continue
                in_range = (ms_arr >= grid_start_ms) & (ms_arr <= grid_end_ms)
                if in_range.any():
                    event_mask[ms_arr[in_range] - grid_start_ms] = True
            log(f"{_t()}   {e} mask updated (event_mask={em})")
    grid_t_ns_evt = grid_t_ns[event_mask]
    grid_ms_evt = grid_ms[event_mask]
    N_out = len(grid_t_ns_evt)
    if N_out == 0:
        raise ValueError(
            f"empty event grid: no qualifying events inside the usable grid "
            f"[{grid_start_ms}, {grid_end_ms}] ms — nothing to build")
    log(f"{_t()} event mask done ({time.perf_counter() - t1:.1f}s)  "
        f"{N_out:,} ticks ({100.0 * N_out / max(N_full, 1):.1f}% of full)")

    # ── Allocate output matrix (sized to event grid) ─────────────────────────
    F = len(cols)
    x_out = np.empty((N_out, F), dtype=np.float32)
    log(f"{_t()} x_out allocated  shape={x_out.shape}  {x_out.nbytes / 1e9:.1f} GB")

    sorted_listings = sorted(listings)
    # Column positions are name-driven: each expanded unit knows its listing and
    # local (unprefixed) name, so the selection packs into x_out in expansion
    # order. Per listing: local name → absolute x_out column, and per family →
    # requested span list.
    col_pos = {n: i for i, n in enumerate(cols)}
    col_idx_by_listing: dict[str, dict[str, int]] = {e: {} for e in sorted_listings}
    for u in exp.units:
        if u.listing is not None:
            col_idx_by_listing[u.listing][u.local_name] = col_pos[u.name]
    spans_by_listing = {e: listing_spans(exp.units, e) for e in sorted_listings}

    # ── Group A: Section 1 — parallel across listings (3 threads) ───────────
    log(f"{_t()} filling Section 1 columns (parallel across listings)…")
    def _s1(e: str):
        t1 = time.perf_counter()
        _fill_section1_columns(e, shared[e], data, grid_t_ns_evt, spans_by_listing[e], x_out, col_idx_by_listing[e])
        return e, time.perf_counter() - t1
    with ThreadPoolExecutor(max_workers=len(listings)) as pool:
        for e, dt in sorted(pool.map(_s1, sorted_listings)):
            log(f"{_t()}   {e} S1 done ({dt:.1f}s)")

    # ── Group B: Section 3 — flat task list of all (listing, EMA) units ─────
    # Each task: compute one EMA on its input, forward-fill to event grid,
    # write to x_out column. Submitted to n_workers thread pool.
    log(f"{_t()} filling Section 3 tick-clock EMAs (flat parallel tasks)…")

    def _make_section3_tasks() -> list[tuple]:
        """Return list of (label, fn) tuples — each fn writes one column to x_out."""
        tasks: list[tuple[str, callable]] = []

        for e in sorted_listings:
            ss = shared[e]
            col_idx = col_idx_by_listing[e]
            spans_e = spans_by_listing[e]
            has_trades = len(ss.tr_t) > 0

            # Pre-compute trade input arrays once per listing (cheap, ~3M × 8 = 25 MB each)
            if has_trades:
                is_buy = ss.tr_dir > 0
                qty_f = ss.tr_qty.astype(np.float64)
                prc_f = ss.tr_prc.astype(np.float64)
                value_f = qty_f * prc_f
                buy_qty = np.where(is_buy, qty_f, 0.0)
                sell_qty = np.where(~is_buy, qty_f, 0.0)
                buy_val = np.where(is_buy, value_f, 0.0)
                sell_val = np.where(~is_buy, value_f, 0.0)

                # Roll's serial covariance input
                if len(prc_f) >= 3:
                    dp = np.diff(prc_f)
                    dp_prod = dp[1:] * dp[:-1]
                    cov_input = np.zeros(len(prc_f), np.float64)
                    cov_input[2:] = dp_prod
                else:
                    cov_input = np.zeros(len(ss.tr_t), np.float64)

                trade_inputs = [
                    (buy_qty,  "ema_buy_trade_qty_{N}t",   spans_e.get("ema_buy_trade_qty_t", ())),
                    (sell_qty, "ema_sell_trade_qty_{N}t",  spans_e.get("ema_sell_trade_qty_t", ())),
                    (buy_val,  "ema_buy_trade_value_{N}t", spans_e.get("ema_buy_trade_value_t", ())),
                    (sell_val, "ema_sell_trade_value_{N}t",spans_e.get("ema_sell_trade_value_t", ())),
                    (cov_input,"ema_trade_serial_cov_{N}t",spans_e.get("ema_trade_serial_cov_t", ())),
                ]
            else:
                trade_inputs = []

            # BBO event-level inputs (pre-square once where needed)
            ofi_sq = ss.ofi_events ** 2 if spans_e.get("ema_ofi_sq_b") else None
            bi_sq = ss.book_imbalance ** 2 if spans_e.get("ema_book_imbalance_sq_b") else None
            bd_sq = ss.book_depth ** 2 if spans_e.get("ema_book_depth_sq_b") else None

            bbo_inputs = [
                (ss.ofi_events,         "ema_ofi_{N}b",                spans_e.get("ema_ofi_b", ())),
                (ofi_sq,                "ema_ofi_sq_{N}b",             spans_e.get("ema_ofi_sq_b", ())),
                (ss.abs_log_ret,        "ema_abs_log_ret_{N}b",        spans_e.get("ema_abs_log_ret_b", ())),
                (ss.book_imbalance,     "ema_book_imbalance_{N}b",     spans_e.get("ema_book_imbalance_b", ())),
                (bi_sq,                 "ema_book_imbalance_sq_{N}b",  spans_e.get("ema_book_imbalance_sq_b", ())),
                (ss.book_depth,         "ema_book_depth_{N}b",         spans_e.get("ema_book_depth_b", ())),
                (bd_sq,                 "ema_book_depth_sq_{N}b",      spans_e.get("ema_book_depth_sq_b", ())),
                (ss.spread_wide_flag,   "ema_spread_wide_flag_{N}b",   spans_e.get("ema_spread_wide_flag_b", ())),
            ]

            # Closures capture e, input_arr, name_fmt, N, grid_t_ns_evt.
            # Each task: EMA → forward-fill → write to x_out[:, ci]
            # Unselected columns never become tasks (the EMA is not computed).
            for input_arr, name_fmt, spans in trade_inputs:
                if input_arr is None or len(input_arr) == 0:
                    continue
                for N in spans:
                    col_name = name_fmt.format(N=N)
                    if col_name not in col_idx:
                        continue
                    ci = col_idx[col_name]
                    t_ev = ss.tr_t
                    def make_task(input_arr=input_arr, N=N, ci=ci, t_ev=t_ev):
                        def run():
                            ema_evt = _ewm_1d(input_arr, _alpha(N))
                            x_out[:, ci] = forward_fill_to_ms_grid(
                                t_ev, ema_evt, grid_t_ns_evt, 0.0,
                            ).astype(np.float32)
                        return run
                    tasks.append((col_name, make_task()))

            for input_arr, name_fmt, spans in bbo_inputs:
                if input_arr is None:
                    continue
                for N in spans:
                    col_name = name_fmt.format(N=N)
                    if col_name not in col_idx:
                        continue
                    ci = col_idx[col_name]
                    t_ev = ss.bbo_t
                    def make_task(input_arr=input_arr, N=N, ci=ci, t_ev=t_ev):
                        def run():
                            ema_evt = _ewm_1d(input_arr, _alpha(N))
                            x_out[:, ci] = forward_fill_to_ms_grid(
                                t_ev, ema_evt, grid_t_ns_evt, 0.0,
                            ).astype(np.float32)
                        return run
                    tasks.append((col_name, make_task()))

            # Fill zeros for missing trade EMAs (when listing has no trades)
            if not has_trades:
                zero = np.zeros(N_out, np.float32)
                for fam, fmt in (
                    ("ema_buy_trade_qty_t",    "ema_buy_trade_qty_{N}t"),
                    ("ema_sell_trade_qty_t",   "ema_sell_trade_qty_{N}t"),
                    ("ema_buy_trade_value_t",  "ema_buy_trade_value_{N}t"),
                    ("ema_sell_trade_value_t", "ema_sell_trade_value_{N}t"),
                    ("ema_trade_serial_cov_t", "ema_trade_serial_cov_{N}t"),
                ):
                    for N in spans_e.get(fam, ()):
                        nm = fmt.format(N=N)
                        if nm in col_idx:
                            x_out[:, col_idx[nm]] = zero

        return tasks

    s3_tasks = _make_section3_tasks()
    n_tasks = len(s3_tasks)
    log(f"{_t()}   submitting {n_tasks} EMA tasks to {n_workers}-thread pool")
    t_s3 = time.perf_counter()
    completed = 0
    log_every = max(1, n_tasks // 10)  # ~10 progress updates
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(nf[1]): nf[0] for nf in s3_tasks}
        from concurrent.futures import as_completed
        for fut in as_completed(futs):
            fut.result()  # raise on error
            completed += 1
            if completed % log_every == 0 or completed == n_tasks:
                log(f"{_t()}   S3 progress  {completed}/{n_tasks} EMAs done")
    log(f"{_t()}   Section 3 done ({time.perf_counter() - t_s3:.1f}s)")

    # ── Group C: Section 2 — parallel across listings ───────────────────────
    # Each listing materializes the full-grid microprice (~1.1 GB) briefly.
    # Parallel across 3 listings = ~3.3 GB peak — fine. The per-ms trade-flow
    # columns ride along here: they too need full-grid arrays (EMA over the
    # quiet-ms zeros) subset to event_mask.
    log(f"{_t()} filling Section 2 calendar EMAs + per-ms trade flow (parallel across listings)…")
    def _s2(e: str):
        t1 = time.perf_counter()
        _fill_section2_calendar(shared[e], grid_t_ns, event_mask, spans_by_listing[e],
                                cfg.microprice_ref.get(e, 0.0), x_out, col_idx_by_listing[e])
        _fill_trade_value_ms_columns(shared[e], grid_t_ns, event_mask, spans_by_listing[e],
                                     x_out, col_idx_by_listing[e])
        return e, time.perf_counter() - t1
    with ThreadPoolExecutor(max_workers=len(listings)) as pool:
        for e, dt in sorted(pool.map(_s2, sorted_listings)):
            log(f"{_t()}   {e} S2 done ({dt:.1f}s)")

    # ── Group E: Cross-listing log-microprice ratio EMAs ─────────────────────
    # Per requested listing pair: time-EMA of log(mp_l1 / mp_l2) and its square, on the
    # full ms grid (reusing the Section-2 forward-fill + _ewm_1d machinery), sampled to the
    # event grid. Mean → cross-venue spread level; variance = sq − mean² → spread volatility.
    # Memory: holds one full-grid log-microprice per listing (~0.7 GB each at 24h) during this step.
    cross_units = [u for u in exp.units if u.pair is not None]
    if cross_units:
        # Requested (pair, span) units by NAME — only the pairs/spans asked for
        # are computed (ratio_sq only if a _sq column wants it; a listing's
        # full-grid log-microprice only if it appears in a requested pair).
        by_pair: dict[tuple[str, str], dict[int, list[Optional[int]]]] = {}
        for u in cross_units:
            slot = 0 if u.family == "cross_ratio_ms" else 1
            by_pair.setdefault(u.pair, {}).setdefault(u.n, [None, None])[slot] = col_pos[u.name]
        pair_cols: list[tuple[str, str, list[tuple[int, Optional[int], Optional[int]]]]] = [
            (l1, l2, [(N, cm_cs[0], cm_cs[1]) for N, cm_cs in sorted(nmap.items())])
            for (l1, l2), nmap in by_pair.items()
        ]
        if pair_cols:
            log(f"{_t()} filling cross-listing log-ratio EMAs ({len(pair_cols)} pairs)…")
            t_ce = time.perf_counter(); eps = 1e-30
            needed_listings = sorted({l for l1, l2, _ in pair_cols for l in (l1, l2)})
            log_mp: dict[str, np.ndarray] = {}; valid_mp: dict[str, np.ndarray] = {}
            for e in needed_listings:
                mp = forward_fill_to_ms_grid(shared[e].bbo_t, shared[e].microprice, grid_t_ns, fill_value=0.0)
                v = mp > eps
                log_mp[e] = np.where(v, np.log(np.maximum(mp, eps)), 0.0)  # 0 where no event yet
                valid_mp[e] = v
                del mp
            for l1, l2, spans in pair_cols:
                both = valid_mp[l1] & valid_mp[l2]
                ratio = np.where(both, log_mp[l1] - log_mp[l2], 0.0)       # log(mp_l1 / mp_l2)
                ratio_sq = ratio * ratio if any(cs is not None for _, _, cs in spans) else None
                for N, cm, cs in spans:
                    a = _alpha(N)
                    if cm is not None:
                        x_out[:, cm] = _ewm_1d(ratio, a)[event_mask].astype(np.float32)
                    if cs is not None:
                        x_out[:, cs] = _ewm_1d(ratio_sq, a)[event_mask].astype(np.float32)
                del ratio, ratio_sq, both
            del log_mp, valid_mp
            log(f"{_t()}   cross-listing done ({time.perf_counter() - t_ce:.1f}s)")

    # ── Group F: realized vol on a (possibly foreign) trade clock ─────────────
    # vol_{N}t = sqrt( tick-EMA over the clock's trade events of the per-interval realized
    # variance of LISTING's MID ). The clock only sets the event windows (its trades' positions
    # on the grid); the mid path is full-resolution, so a clock trade just bundles the squared
    # 1ms mid-returns since the previous clock trade into one EMA step. With a foreign @{CLOCK},
    # every listing decays per the SAME trades, so a busy and a quiet venue's vol line up. This
    # is the exact vectorisation of the oracle in tests/test_dataset_v2_volclock.py:
    #   cs = cumsum(sq mid-returns); V_k = cs[clockpos_k] − cs[clockpos_{k-1}]; y = ewm(V); vol = √y.
    vol_units = [u for u in exp.units if u.family == "vol_t"]
    if vol_units:
        log(f"{_t()} filling vol_t (trade-clock realized vol)…")
        t_vol = time.perf_counter(); eps = 1e-30
        by_listing: dict[str, list] = {}
        for u in vol_units:
            by_listing.setdefault(u.listing, []).append(u)
        for L, lunits in by_listing.items():
            ss = shared[L]
            mid = (ss.bbo_bid.astype(np.float64) + ss.bbo_ask.astype(np.float64)) / 2.0
            mid_full = forward_fill_to_ms_grid(ss.bbo_t, mid, grid_t_ns, fill_value=0.0)
            safe = mid_full > eps
            logm = np.where(safe, np.log(np.maximum(mid_full, eps)), 0.0)
            sq = np.zeros(N_full, np.float64)
            sq[1:] = np.where(safe[1:] & safe[:-1], (logm[1:] - logm[:-1]) ** 2, 0.0)
            cs = np.empty(N_full + 1, np.float64); cs[0] = 0.0
            np.cumsum(sq, out=cs[1:])                                       # cs[i] = Σ sq[:i]
            del mid_full, logm, sq, safe
            by_clock: dict[str, list] = {}                                  # group L's vol cols by clock listing
            for u in lunits:
                by_clock.setdefault(_clock_listing(u.clock) or L, []).append(u)   # 'trades_<C>' → C; None → own
            for C, cunits in by_clock.items():
                clk = shared[C].tr_t                                        # clock trade times (ns)
                clk_ms = clk // 1_000_000
                m = (clk_ms >= grid_start_ms) & (clk_ms <= grid_end_ms)
                clk, clk_ms = clk[m], clk_ms[m]
                cpos = np.searchsorted(grid_ms, clk_ms, side="right")      # grid idx just after each clock trade
                Vk = np.empty(len(cpos), np.float64)
                if len(cpos):
                    Vk[0] = cs[cpos[0]]
                    Vk[1:] = cs[cpos[1:]] - cs[cpos[:-1]]                   # realized var per clock interval
                ci = np.searchsorted(clk, grid_t_ns, side="right") - 1      # latest clock trade ≤ each tick (ns-causal)
                has = ci >= 0
                for u in cunits:
                    yk = _ewm_1d(Vk, _alpha(u.n)) if len(Vk) else Vk        # tick-EMA over clock events
                    vol_full = np.zeros(N_full, np.float64)
                    if len(yk):
                        vol_full[has] = np.sqrt(np.maximum(yk[ci[has]], 0.0))
                    x_out[:, col_pos[u.name]] = vol_full[event_mask].astype(np.float32)
                    del vol_full
            del cs
        log(f"{_t()}   vol_t done ({time.perf_counter() - t_vol:.1f}s, {len(vol_units)} cols)")

    # ── Group G: book-event quantities on a (possibly foreign) trade clock ────
    # Two semantics, both tick-EMA'd over the clock's trade events (span N clock trades):
    #   LEVEL: the state forward-filled to each clock trade (book_imbalance/depth/microprice…).
    #   FLOW : the per-clock-interval SUM of the book-event quantity (ofi/abs_log_ret are flow).
    # `_sq` squares the sampled level / per-event flow first; centered microprice subtracts mp_ref.
    # The clock positions (trade times) are shared across families for each (listing, clock).
    # family → (mode, ListingSharedState attr, square, center_microprice)
    _TC_FAMILIES = {
        "ema_book_imbalance_t":         ("level", "book_imbalance",   False, False),
        "ema_book_imbalance_sq_t":      ("level", "book_imbalance",   True,  False),
        "ema_book_depth_t":             ("level", "book_depth",       False, False),
        "ema_book_depth_sq_t":          ("level", "book_depth",       True,  False),
        "ema_spread_wide_flag_t":       ("level", "spread_wide_flag", False, False),
        "ema_microprice_centered_t":    ("level", "microprice",       False, True),
        "ema_microprice_centered_sq_t": ("level", "microprice",       True,  True),
        "ema_ofi_t":                    ("flow",  "ofi_events",        False, False),
        "ema_ofi_sq_t":                 ("flow",  "ofi_events",        True,  False),
        "ema_abs_log_ret_t":            ("flow",  "abs_log_ret",       False, False),
    }
    # Trade-flow families — FLOW over the listing's OWN trade events. Own clock ⇒ one trade per
    # clock interval ⇒ identical to the Section-3 column, so only the @{CLOCK} (foreign) units
    # come here; the bare own-clock ones stay in Section 3 untouched.
    _TRADE_FLOW_FAMS = ("ema_buy_trade_qty_t", "ema_sell_trade_qty_t", "ema_buy_trade_value_t",
                        "ema_sell_trade_value_t", "ema_trade_serial_cov_t")
    tc_units = [u for u in exp.units if u.family in _TC_FAMILIES
                or (u.family in _TRADE_FLOW_FAMS and u.clock is not None)]
    if tc_units:
        log(f"{_t()} filling trade-clock event EMAs…")
        t_tc = time.perf_counter(); eps = 1e-30
        by_listing_tc: dict[str, list] = {}
        for u in tc_units:
            by_listing_tc.setdefault(u.listing, []).append(u)
        for L, lunits in by_listing_tc.items():
            ss = shared[L]
            mp_ref = cfg.microprice_ref.get(L, 0.0)
            by_clock: dict[str, list] = {}
            for u in lunits:
                by_clock.setdefault(_clock_listing(u.clock) or L, []).append(u)
            for C, cunits in by_clock.items():
                tr = shared[C].tr_t                                         # ALL clock trades (warmed by any
                ci = np.searchsorted(tr, grid_t_ns, side="right") - 1       # pre-grid; ns-causal forward-fill)
                has = ci >= 0
                epos_book = np.searchsorted(ss.bbo_t, tr, side="right")     # # book events ≤ each clock trade
                epos_trd = np.searchsorted(ss.tr_t, tr, side="right")       # # of L's trades ≤ each clock trade
                cs_cache: dict[tuple, np.ndarray] = {}

                def _interval_sum(c, epos):                                 # Σ over (prev clock, this clock]
                    s = np.empty(len(epos), np.float64)
                    if len(epos):
                        s[0] = c[epos[0]]; s[1:] = c[epos[1:]] - c[epos[:-1]]
                    return s

                for u in cunits:
                    if u.family in _TRADE_FLOW_FAMS:                        # FLOW over the listing's trades
                        key = ("trade", u.family)
                        if key not in cs_cache:
                            q = _trade_flow_input(ss, u.family)
                            c = np.empty(len(q) + 1, np.float64); c[0] = 0.0
                            np.cumsum(q, out=c[1:]); cs_cache[key] = c
                        inp = _interval_sum(cs_cache[key], epos_trd)
                    else:
                        mode, attr, square, center = _TC_FAMILIES[u.family]
                        src = getattr(ss, attr).astype(np.float64)
                        if center:                                          # centered microprice (mp − ref)
                            src = np.where(src > eps, src - mp_ref, 0.0)
                        if mode == "level":                                 # sample the state at each trade…
                            inp = (np.where(epos_book > 0, src[np.clip(epos_book - 1, 0, len(src) - 1)], 0.0)
                                   if len(tr) else np.empty(0))
                            if square:
                                inp = inp * inp                             # …squared after sampling
                        else:                                              # FLOW over book events
                            key = ("book", attr, square)
                            if key not in cs_cache:
                                f = src * src if square else src
                                c = np.empty(len(f) + 1, np.float64); c[0] = 0.0
                                np.cumsum(f, out=c[1:]); cs_cache[key] = c
                            inp = _interval_sum(cs_cache[key], epos_book)
                    yk = _ewm_1d(inp, _alpha(u.n)) if len(inp) else inp     # tick-EMA over clock events
                    col_full = np.zeros(N_full, np.float64)
                    if len(yk):
                        col_full[has] = yk[ci[has]]
                    x_out[:, col_pos[u.name]] = col_full[event_mask].astype(np.float32)
                    del col_full
        log(f"{_t()}   trade-clock event EMAs done ({time.perf_counter() - t_tc:.1f}s, {len(tc_units)} cols)")

    # ── Group D: Cost fields ─────────────────────────────────────────────────
    cost_cfg = _cost_cfg(cfg)
    t1 = time.perf_counter()
    _need_ext = any(
        n in cfg.cost_fields for n in ("c_buy_trade_min_l", "c_buy_trade_max_l",
                                       "c_sell_trade_min_l", "c_sell_trade_max_l"))
    cost_tables = _build_cost_tables(data, need_extremes=_need_ext)
    log(f"{_t()} cost tables ready ({time.perf_counter() - t1:.1f}s)")

    n_w = min(n_workers, max(1, N_out))
    chunks = [c for c in np.array_split(grid_t_ns_evt, n_w) if len(c) > 0]
    log(f"{_t()} cost fields: {len(chunks)} chunks of ~{len(chunks[0]):,} ticks across {n_workers} workers")

    def _cost_chunk(chunk: np.ndarray) -> dict[str, np.ndarray]:
        return _cost_fields_for_grid(data, cost_tables, cost_cfg, chunk)

    t_cf = time.perf_counter()
    chunk_outs: list[dict[str, np.ndarray]] = [None] * len(chunks)  # type: ignore[list-item]
    completed = 0
    log_every = max(1, len(chunks) // 5)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_cost_chunk, c): i for i, c in enumerate(chunks)}
        from concurrent.futures import as_completed
        for fut in as_completed(futs):
            chunk_outs[futs[fut]] = fut.result()
            completed += 1
            if completed % log_every == 0 or completed == len(chunks):
                log(f"{_t()}   cost chunks  {completed}/{len(chunks)} done")

    cost_concat: dict[str, np.ndarray] = {}
    if chunk_outs:
        for key in chunk_outs[0].keys():
            cost_concat[key] = np.concatenate([co[key] for co in chunk_outs])
    log(f"{_t()} cost fields done ({time.perf_counter() - t_cf:.1f}s)")

    log(f"{_t()} done: x_out shape {x_out.shape}")

    return SampleArraysRaw(
        x=x_out,
        timestamp_ms=grid_ms_evt.astype(np.float64),
        column_names=cols,
        # attach only the requested cost fields (others stay None — lean memory)
        **{k: cost_concat[k].astype(np.float32) for k in _COST_FIELDS if k in cost_concat},
    )


# ── Save / load (npz) ──────────────────────────────────────────────────────────

# Always-present array fields; cost fields are optional (only the requested ones are present).
# column_names handled separately since it is a list[str].
_CORE_FIELDS = ("x", "timestamp_ms")
_COST_FIELDS = (
    "eval_bid_l", "eval_ask_l", "eval_mid",
    "c_ask_entry_l", "c_bid_entry_l", "c_ask_exit_l", "c_bid_exit_l", "c_mid_exit_l",
    "c_mid_move_count",
    "c_buy_trade_min_l", "c_buy_trade_max_l", "c_sell_trade_min_l", "c_sell_trade_max_l",
    "feed_latency_raw_ms", "feed_latency_excess_ms",
)


def _present_cost_fields(s: SampleArraysRaw) -> tuple[str, ...]:
    return tuple(f for f in _COST_FIELDS if getattr(s, f) is not None)


def _save_raw(path: Path, s: SampleArraysRaw) -> None:
    """Save a SampleArraysRaw to a single npz (only the populated cost fields)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _CORE_FIELDS + _present_cost_fields(s)
    np.savez(str(path), column_names=np.array(s.column_names),
             **{f: getattr(s, f) for f in fields})


def _load_raw(path: Path) -> SampleArraysRaw:
    """Reconstruct a SampleArraysRaw from a npz written by :func:`_save_raw`."""
    with np.load(str(path), allow_pickle=False) as d:
        present = [f for f in (_CORE_FIELDS + _COST_FIELDS) if f in d.files]
        kwargs = {f: d[f] for f in present}
        kwargs["column_names"] = [str(c) for c in d["column_names"]]
    return SampleArraysRaw(**kwargs)


# ════════════════════════════════════════════════════════════════════════════════
# Block-based, listing-keyed builder (DATA_DIR)
# ════════════════════════════════════════════════════════════════════════════════

_DEFAULT_BLOCK_CACHE_DIR = Path("artifacts") / "dataset_raw_cache"


def _concat_raw(parts: list[SampleArraysRaw]) -> SampleArraysRaw:
    """Concatenate per-block SampleArraysRaw into one (column_names must match)."""
    assert parts, "no blocks to concat"
    cols = parts[0].column_names
    for p in parts:
        assert p.column_names == cols, "column-name mismatch across blocks"
    fields = _CORE_FIELDS + _present_cost_fields(parts[0])
    kw = {f: np.concatenate([getattr(p, f) for p in parts]) for f in fields}
    return SampleArraysRaw(column_names=cols, **kw)


def _block_missing_listings(block: str, listings: list[str]) -> list[str]:
    """Requested listings missing front_levels OR trade in this block (empty = complete)."""
    missing: list[str] = []
    for lst in listings:
        for t in ("front_levels", "trade"):
            if not list(DATA_DIR.glob(f"{block}.{lst}.{t}.parquet")):
                missing.append(f"{lst}.{t}")
    return missing


def _block_min_rx_ns(block: str, listing: str) -> int:
    """Min rx_time (ns) of a block's listing front_levels — cheap lazy scan for date filtering."""
    files = sorted(DATA_DIR.glob(f"{block}.{listing}.front_levels.parquet"))
    return int(pl.scan_parquet(files).select(pl.col("rx_time").cast(pl.Int64).min()).collect().item())


def _block_rx_span_ns(block: str, listings: list[str]) -> tuple[int, int]:
    """(min first, max last) rx_time (ns) over the listings' front_levels — the t_start/t_end
    build_features_raw derives its grid from (listing_book_t is rx_time). Conservative for the
    degenerate-block pre-check: raw parquet min/max span ⊇ post-drop_nulls span, so a block
    skipped here could never have built; a residual empty (nulls eating the whole warmup)
    still fails fast inside build_features_raw."""
    lo, hi = [], []
    for lst in listings:
        files = sorted(DATA_DIR.glob(f"{block}.{lst}.front_levels.parquet"))
        mm = pl.scan_parquet(files).select(
            pl.col("rx_time").cast(pl.Int64).min().alias("lo"),
            pl.col("rx_time").cast(pl.Int64).max().alias("hi"),
        ).collect()
        lo.append(int(mm["lo"][0])); hi.append(int(mm["hi"][0]))
    return min(lo), max(hi)


def build_block(
    block:     str,
    cfg:       DatasetRawConfig,
    cache_dir: Optional[Path]     = _DEFAULT_BLOCK_CACHE_DIR,
    verbose:   bool               = True,
) -> SampleArraysRaw:
    """Build (or cache-load) the raw-atom dataset for ONE block over ``cfg.listings``.

    Each block (~24h) is treated as a self-contained mini-session: ``cfg.warmup_ms`` is
    discarded at the start (EMA convergence) and ``cfg.horizon_ms`` at the end (no future
    for the forward outcome) — negligible at 24h. Asserts the block has all listings."""
    def _log(*a, **k):
        print(*a, **k); sys.stdout.flush()
    log = _log if verbose else _silent
    listings = list(cfg.listings)

    missing = _block_missing_listings(block, listings)
    if missing:
        raise FileNotFoundError(f"block {block} missing requested listings: {missing}")

    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"{block}.{cfg.cache_key()}.npz"
        if path.exists():
            log(f"  cache hit  {path.name}")
            return _load_raw(path)

    fl = {e: load_block(block, e, "front_levels") for e in listings}
    td = {e: load_block(block, e, "trade").filter((pl.col("prc") > 0) & (pl.col("qty") > 0)) for e in listings}
    for e in listings:
        log(f"    {e:18} bbo {len(fl[e]):>10,}  trades {len(td[e]):>10,}")
    data = build_session_data(fl, td, listings, cfg.target_listing)
    s = build_features_raw(data, listings, cfg, log=log)
    if path is not None:
        _save_raw(path, s)
        log(f"  cached → {path.name}  (x={s.x.shape}, {s.x.nbytes / 1e9:.1f} GB)")
    return s


def build_dataset(
    cfg:        DatasetRawConfig,
    date_start: Optional[str]      = None,   # "YYYY-MM-DD" inclusive (UTC block start)
    date_end:   Optional[str]      = None,   # "YYYY-MM-DD" inclusive
    cache_dir:  Optional[Path]     = _DEFAULT_BLOCK_CACHE_DIR,
    verbose:    bool               = True,
    concat:     bool               = True,   # False → build+cache each block & return block ids (RAM-safe for big ranges)
) -> "SampleArraysRaw | list[str]":
    """Build the dataset over ``cfg.listings``, restricted to blocks whose data falls
    in [date_start, date_end]. SKIPS (warns) any block missing a requested listing, builds
    each remaining block (per-block cache).

    ``concat=True``  → concatenate all blocks into one SampleArraysRaw (only feasible for a
                       small range — holds everything in RAM).
    ``concat=False`` → build + cache each block one at a time, discarding it from RAM after
                       caching; returns the list of block ids. Use this to warm the cache over
                       a large range without the ~13GB/block concat blowing up memory."""
    import datetime as _dt
    def _log(*a, **k):
        print(*a, **k); sys.stdout.flush()
    log = _log if verbose else _silent
    if DATA_DIR is None:
        raise RuntimeError("data_dir not configured in settings.local.toml")
    lo = _dt.date.fromisoformat(date_start) if date_start else None
    hi = _dt.date.fromisoformat(date_end) if date_end else None
    listings = list(cfg.listings)

    blocks = list_blocks(cfg.target_listing, "front_levels")
    kept: list[str] = []
    for b in blocks:
        miss = _block_missing_listings(b, listings)
        if miss:
            log(f"  SKIP {b}: missing {miss}")
            continue
        bday = _dt.datetime.utcfromtimestamp(_block_min_rx_ns(b, cfg.target_listing) / 1e9).date()
        if (lo and bday < lo) or (hi and bday > hi):
            continue
        t_lo, t_hi = _block_rx_span_ns(b, listings)
        gs, ge = _grid_bounds_ms(t_lo, t_hi, cfg.warmup_ms, cfg.horizon_ms)
        if ge < gs:
            log(f"  SKIP {b}: span {(t_hi - t_lo) / 1e9:.1f}s ≤ warmup+horizon "
                f"({(cfg.warmup_ms + cfg.horizon_ms) / 1e3:.1f}s) — empty grid")
            continue
        kept.append(b)
    log(f"building {len(kept)}/{len(blocks)} blocks  listings={listings}  target={cfg.target_listing}")

    if not concat:
        log(f"building+caching {len(kept)} blocks (concat=False, RAM-safe — one block at a time)…")
        for i, b in enumerate(kept, 1):
            log(f"── [{i}/{len(kept)}] block {b} ──")
            build_block(b, cfg, cache_dir, verbose)   # builds + caches; result discarded → frees RAM
        log(f"done — built+cached {len(kept)} blocks (per-block .npz under {cache_dir})")
        return kept

    parts: list[SampleArraysRaw] = []
    for b in kept:
        log(f"── block {b} ──")
        parts.append(build_block(b, cfg, cache_dir, verbose))
    s = _concat_raw(parts)
    log(f"done — {len(s):,} samples  x={s.x.shape}  from {len(parts)} blocks")
    return s


