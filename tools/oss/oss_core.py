"""oss_core — shared core of the multi-block out-of-sample (OOS) harness.

Computes, FOR ANY BLOCK, the same scaffold the single-block feature notebooks
(notebooks/features/template.ipynb, .../flow_imbalance.ipynb) build for block[0],
so a feature's marginal value can be re-measured across all 58 blocks (purged
walk-forward) instead of one.

Every convention here is lifted verbatim from the validated template builders
(/tmp/build_feat_nb.py and /tmp/build_flow_imbalance.py) — the trade clock, the
byb merged mid, the σ_ev / λ_ev yardsticks, the 50 ms anchor grid, the price /
rate targets, the four rate/vol controls, and the §5 purged+embargoed
walk-forward marginal rank-IC. Do not "improve" them — bit-compatibility with the
notebooks is the whole point (validation target: flow_imbalance PRICE-head joint
marginal IC ≈ +0.158 on block[0]).

API
---
- load_block_arrays(block_idx)  -> BlockArrays  (per-venue event arrays + byb merged mid + trade clock)
- build_grid(arrays)            -> Grid         (anchors, price/rate targets, σ_ev/λ_ev, 4 controls)
- marginal_ic(feature, target, controls) -> float  (purged+embargoed walk-forward marginal rank-IC)
- ic(feature, target)           -> float         (plain Spearman rank-IC)
- load_cached(block_idx)        -> (BlockArrays, Grid)  (npz-cached arrays+grid; computed once per block)
- run_oss(block_indices, feature_modules) -> results dict (per block: marginal IC per feature module)

Per-venue / per-block conventions (must match the notebooks):
- COIN = "eth_usdt_p"; TARGET = byb; EXCHANGES = [bin, byb, okx].
- mid stream: byb/okx = merged_levels, bin = front_levels (merged_levels is
  DISALLOWED for bin perp in boba.io — it raises).
- trade clock = np.unique of all-venue trade rx timestamps (prc>0 & qty>0);
  one tick per trade-TIMESTAMP (simultaneous prints = one tick).
- HORIZON = 100 ms; grid every 50 ms past a warmup; YARDSTICK_N = 10000.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import polars as pl
from scipy.signal import lfilter
from scipy.stats import spearmanr

import boba.io as io
from boba.io import list_blocks, load_block

# ── fixed conventions (verbatim from the template builders) ──────────────────
COIN        = "eth_usdt_p"
TARGET      = "byb_eth_usdt_p"
EXCHANGES   = ["bin", "byb", "okx"]                 # each venue's flow is a feature; byb is also the target
MID_STREAM  = {"bin": "front_levels", "byb": "merged_levels", "okx": "merged_levels"}
HORIZON_NS  = 100 * 1_000_000                       # 100 ms forecast horizon, in ns
YARDSTICK_N = 10000                                 # the ONE span for BOTH yardsticks (σ_ev, λ_ev)
GRID_STEP_NS = 50 * 1_000_000                       # 50 ms anchor grid (half the horizon)
# WARMUP must clear the slowest EMA/yardstick. The template uses
# 5*max(YARDSTICK_N, max(SLOW|SPANS)); the largest span any feature module uses
# is 8000 (flow_imbalance), well under YARDSTICK_N, so 5*YARDSTICK_N covers it.
WARMUP      = 5 * YARDSTICK_N                        # = 50000 trade ticks
CACHE_DIR   = os.environ.get("OSS_CACHE_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


# ── block arrays ─────────────────────────────────────────────────────────────
@dataclass
class BlockArrays:
    """Per-venue event arrays + the byb merged mid stream + the merged trade clock.

    All arrays are int64 (ns timestamps) or float64. Per-venue dicts are keyed by
    the short exchange code ("bin"/"byb"/"okx").

    front_levels (per venue): rx, bid_prc, bid_qty, ask_prc, ask_qty.
    trades       (per venue): rx, prc, qty, sign  (sign = +1 lifts ask / buy, -1 hits bid / sell).
    byb merged mid:           byb_rx, byb_mid  (same-timestamp rows collapsed to the final mid).
    merged trade clock:       merged_ts  (np.unique of all-venue trade rx; one tick per timestamp).
    """
    block: str
    block_idx: int
    fl_rx: dict = field(default_factory=dict)
    fl_bid_prc: dict = field(default_factory=dict)
    fl_bid_qty: dict = field(default_factory=dict)
    fl_ask_prc: dict = field(default_factory=dict)
    fl_ask_qty: dict = field(default_factory=dict)
    tr_rx: dict = field(default_factory=dict)
    tr_prc: dict = field(default_factory=dict)
    tr_qty: dict = field(default_factory=dict)
    tr_sign: dict = field(default_factory=dict)
    byb_rx: np.ndarray = None
    byb_mid: np.ndarray = None
    merged_ts: np.ndarray = None


def load_block_arrays(block_idx: int) -> BlockArrays:
    """Load every per-venue event array + the byb merged mid + the trade clock for
    one block (indexed into list_blocks(TARGET, "front_levels"))."""
    blocks = list_blocks(TARGET, "front_levels")
    block = blocks[block_idx]
    A = BlockArrays(block=block, block_idx=block_idx)

    # front_levels per venue (raw best bid/ask snapshots + qty)
    for ex in EXCHANGES:
        fl = (load_block(block, f"{ex}_{COIN}", "front_levels")
              .select("rx_time", "bid_prc", "bid_qty", "ask_prc", "ask_qty").drop_nulls())
        A.fl_rx[ex]      = fl["rx_time"].cast(pl.Int64).to_numpy()
        A.fl_bid_prc[ex] = fl["bid_prc"].to_numpy().astype(np.float64)
        A.fl_bid_qty[ex] = fl["bid_qty"].to_numpy().astype(np.float64)
        A.fl_ask_prc[ex] = fl["ask_prc"].to_numpy().astype(np.float64)
        A.fl_ask_qty[ex] = fl["ask_qty"].to_numpy().astype(np.float64)

    # trades per venue: rx, prc, qty, sign (+1 lifts ask, -1 hits bid; venue-specific aggressor)
    for ex in EXCHANGES:
        td = (load_block(block, f"{ex}_{COIN}", "trade")
              .select("rx_time", "prc", "qty", "aggressor")
              .filter((pl.col("prc") > 0) & (pl.col("qty") > 0)))           # drop bad zero prints
        A.tr_rx[ex]   = td["rx_time"].cast(pl.Int64).to_numpy()
        A.tr_prc[ex]  = td["prc"].to_numpy().astype(np.float64)
        A.tr_qty[ex]  = td["qty"].to_numpy().astype(np.float64)
        A.tr_sign[ex] = np.where(io._trade_lifts_ask(f"{ex}_{COIN}", td["aggressor"].to_numpy()),
                                 1.0, -1.0)

    # the shared trade clock: one tick per trade-TIMESTAMP across ALL venues
    trade_prints = np.concatenate([A.tr_rx[ex] for ex in EXCHANGES])
    A.merged_ts = np.unique(trade_prints)

    # byb merged mid stream (front_levels fused with trades, newest-by-exchange-time), for the
    # target + yardsticks. Same-timestamp rows collapsed to ONE update (the final mid).
    bm = (load_block(block, TARGET, MID_STREAM["byb"]).select("rx_time", "bid_prc", "ask_prc").drop_nulls())
    byb_rx0 = bm["rx_time"].cast(pl.Int64).to_numpy()
    byb_mid0 = (bm["bid_prc"].to_numpy() + bm["ask_prc"].to_numpy()) / 2.0
    keep = np.concatenate([byb_rx0[1:] != byb_rx0[:-1], [True]])
    A.byb_rx, A.byb_mid = byb_rx0[keep], byb_mid0[keep]
    return A


def mid_stream(arrays: BlockArrays, ex: str):
    """The freshest-mid stream for a venue per MID_STREAM policy: (rx, mid).
    byb/okx use merged_levels; bin uses front_levels. byb is taken from the already
    same-timestamp-collapsed byb merged mid; okx/bin are loaded here. Used by feature
    modules that need a venue mid; the core itself only needs byb."""
    if ex == "byb":
        return arrays.byb_rx, arrays.byb_mid
    if MID_STREAM[ex] == "front_levels":
        rx = arrays.fl_rx[ex]
        mid = (arrays.fl_bid_prc[ex] + arrays.fl_ask_prc[ex]) / 2.0
        return rx, mid
    df = (load_block(arrays.block, f"{ex}_{COIN}", "merged_levels")
          .select("rx_time", "bid_prc", "ask_prc").drop_nulls())
    rx = df["rx_time"].cast(pl.Int64).to_numpy()
    mid = (df["bid_prc"].to_numpy() + df["ask_prc"].to_numpy()) / 2.0
    return rx, mid


# ── grid + yardsticks + targets + controls ───────────────────────────────────
@dataclass
class Grid:
    """The causal evaluation scaffold on the 50 ms anchor grid (all arrays length
    n_anchors), plus the trade-clock fields a feature module needs to place itself
    on the same grid.

    Anchors / clock:
      anchor_ts        — grid anchor timestamps (ns), causal, past WARMUP.
      tick_at_anchor   — last trade-clock tick (index into merged_ts) at-or-before each anchor.
      merged_ts        — the trade clock (one tick per trade-timestamp).
    Yardsticks (at each anchor, span YARDSTICK_N):
      sigma_ev, lambda_ev — byb's vol RMS-per-move and move-rate (per second).
    Targets:
      price_target     — byb 100 ms log return ÷ σ_ev (price head).
      rate_target      — byb 100 ms mid-move count ÷ λ_ev (rate head).
    Controls (all causal, on the anchor grid):
      controls = {rate_level, rate_momentum, vol_level, vol_momentum}.
    """
    block: str
    block_idx: int
    anchor_ts: np.ndarray
    tick_at_anchor: np.ndarray
    merged_ts: np.ndarray
    sigma_ev: np.ndarray
    lambda_ev: np.ndarray
    price_target: np.ndarray
    rate_target: np.ndarray
    controls: dict


def _ewma(x, span):
    """Per-trade EMA (α = 2/(span+1)), used for the seconds-per-trade leg of λ_ev."""
    a = 2.0 / (span + 1.0)
    return lfilter([a], [1.0, -(1.0 - a)], x)


def _make_yardstick_fns(arrays: BlockArrays):
    """Build the σ_ev / λ_ev machinery exactly as the template's §2 cell: react to
    every byb merged-mid move, decay once per trade timestamp, read AT each anchor.
    Returns (yardsticks(anchors, span), cum_mv, byb_rx, byb_mid)."""
    merged_ts = arrays.merged_ts
    n_ticks = len(merged_ts)
    byb_rx, byb_mid = arrays.byb_rx, arrays.byb_mid
    byb_lm = np.log(byb_mid)
    byb_blr = np.empty_like(byb_lm)
    byb_blr[0] = 0.0
    byb_blr[1:] = np.diff(byb_lm)                              # byb log-return per timestamp
    mv = byb_blr != 0.0                                        # a REAL byb mid-move (one per timestamp)
    mv_rx, mv_r2 = byb_rx[mv], byb_blr[mv] ** 2
    cum_mv = np.concatenate([[0.0], np.cumsum(mv.astype(float))])   # running byb mid-move count (rate target)
    byb_dt = np.zeros(n_ticks)
    byb_dt[1:] = np.diff(merged_ts) / 1e9                      # seconds between consecutive trades

    def _flow_at(anchors, val, span):
        """EWMA of `val` over the byb-MOVE stream, decayed once per trade-timestamp, read AT each anchor."""
        a = 2.0 / (span + 1.0)
        k = np.searchsorted(merged_ts, mv_rx, "left")          # trades strictly before each move
        ep = np.bincount(k, weights=val, minlength=n_ticks + 1)
        x = np.zeros(n_ticks + 1)
        x[1:] = a * (1.0 - a) * ep[:-1]
        com = lfilter([1.0], [1.0, -(1.0 - a)], x)             # committed E just after each trade
        ta = np.searchsorted(merged_ts, anchors, "right") - 1  # last trade <= anchor
        cs = np.concatenate([[0.0], np.cumsum(val)])
        partial = cs[np.searchsorted(mv_rx, anchors, "right")] - cs[np.searchsorted(mv_rx, merged_ts[ta], "right")]
        return com[ta + 1] + a * partial

    def yardsticks(anchors, span):
        e_sq = _flow_at(anchors, mv_r2, span)                  # E: exp-weighted squared byb moves
        e_mv = _flow_at(anchors, np.ones(mv_r2.size), span)   # W: exp-weighted byb-move count
        e_dt = _ewma(byb_dt, span)[np.searchsorted(merged_ts, anchors, "right") - 1]
        sig = np.sqrt(e_sq / np.maximum(e_mv, 1e-12))         # σ_ev: RMS byb mid-move (E/W)
        lam = e_mv / np.maximum(e_dt, 1e-12)                  # λ_ev: byb mid-moves per second
        return sig, lam

    return yardsticks, cum_mv, byb_rx, byb_mid


def build_grid(arrays: BlockArrays) -> Grid:
    """Build the causal anchor grid, the yardsticks, the price/rate targets, and the
    four rate/vol controls — exactly the §2/§3/§5 scaffold the notebooks build for
    block[0], for ANY block."""
    merged_ts = arrays.merged_ts
    yardsticks, cum_mv, byb_rx, byb_mid = _make_yardstick_fns(arrays)

    # evaluation grid: 50 ms, past warmup, leaving room for the 100 ms forward window
    anchor_ts = np.arange(merged_ts[WARMUP], merged_ts[-1] - HORIZON_NS, GRID_STEP_NS)
    tick_at_anchor = np.searchsorted(merged_ts, anchor_ts, "right") - 1
    sigma_at_anchor, lam_at_anchor = yardsticks(anchor_ts, YARDSTICK_N)

    # price-head target: byb's 100 ms log return ÷ σ_ev
    mid_now = byb_mid[np.searchsorted(byb_rx, anchor_ts, "right") - 1]
    mid_fwd = byb_mid[np.searchsorted(byb_rx, anchor_ts + HORIZON_NS, "right") - 1]
    fwd_return = np.log(mid_fwd / mid_now)
    price_target = fwd_return / sigma_at_anchor

    # rate-head target: byb mid-moves over the next 100 ms ÷ λ_ev
    fwd_count = (cum_mv[np.searchsorted(byb_rx, anchor_ts + HORIZON_NS, "right")]
                 - cum_mv[np.searchsorted(byb_rx, anchor_ts, "right")])
    rate_target = fwd_count / np.maximum(lam_at_anchor, 1e-9)

    # the four controls: yardstick levels + fast/slow momenta (template §5)
    FAST_YARD = YARDSTICK_N // 10
    sig_fast, lam_fast = yardsticks(anchor_ts, FAST_YARD)
    controls = {
        "vol_level":     np.log(sigma_at_anchor),
        "vol_momentum":  np.log(sig_fast / sigma_at_anchor),
        "rate_level":    np.log(lam_at_anchor),
        "rate_momentum": np.log(lam_fast / lam_at_anchor),
    }

    return Grid(
        block=arrays.block, block_idx=arrays.block_idx,
        anchor_ts=anchor_ts, tick_at_anchor=tick_at_anchor, merged_ts=merged_ts,
        sigma_ev=sigma_at_anchor, lambda_ev=lam_at_anchor,
        price_target=price_target, rate_target=rate_target, controls=controls,
    )


# ── marginal IC — the §5 purged, embargoed walk-forward ──────────────────────
def _wf_folds(features, y, k=6, embargo=2000):
    """Purged, expanding-window walk-forward (verbatim from §5): yields (test_mask,
    oos_prediction) per fold. Fold i trains on the past minus an embargo gap, tests
    on the next segment."""
    design = np.column_stack(features)
    n = len(y)
    valid = np.isfinite(design).all(1) & np.isfinite(y)
    edges = np.linspace(0, n, k + 1).astype(int)
    for i in range(1, k):
        te = np.zeros(n, bool); te[edges[i]:edges[i + 1]] = True
        tr = np.zeros(n, bool); tr[:max(0, edges[i] - embargo)] = True
        train, test = valid & tr, valid & te
        if train.sum() < 100 or test.sum() < 100:
            continue
        mu, sd = design[train].mean(0), design[train].std(0) + 1e-12
        X = np.column_stack([(design - mu) / sd, np.ones(n)])
        coef, *_ = np.linalg.lstsq(X[train], y[train], rcond=None)
        yield test, X @ coef


def _wf_ic(features, y):
    """Mean OOS rank-IC across the walk-forward folds (the ship-grade gate)."""
    scores = [spearmanr(p[t], y[t]).statistic for t, p in _wf_folds(features, y)]
    return float(np.mean(scores)) if scores else float("nan")


def marginal_ic(feature_values, target, controls):
    """The purged+embargoed walk-forward MARGINAL rank-IC of feature-over-controls:
    wf_ic(controls + feature) - wf_ic(controls), exactly as the notebook §5 gate.

    feature_values — one feature array, or a list of feature arrays (jointly added).
    target         — the head target (e.g. grid.price_target).
    controls       — the grid.controls dict, or a list of control arrays. The notebook
                     baseline is the two MOMENTA (rate_momentum, vol_momentum); pass
                     that subset to reproduce the notebook number exactly (the helper
                     marginal_ic_price does this)."""
    if isinstance(controls, dict):
        ctrl = [controls["rate_momentum"], controls["vol_momentum"]]   # notebook `base`
    else:
        ctrl = list(controls)
    feats = list(feature_values) if isinstance(feature_values, (list, tuple)) else [feature_values]
    return _wf_ic(ctrl + feats, target) - _wf_ic(ctrl, target)


def marginal_ic_price(feature_values, grid: Grid):
    """Convenience: marginal IC of a feature (or list of features) over the notebook
    `base` controls against the PRICE-head target. This is the exact number §5
    reports (joint over all venues, or per venue)."""
    return marginal_ic(feature_values, grid.price_target,
                       [grid.controls["rate_momentum"], grid.controls["vol_momentum"]])


def ic(feature_values, target):
    """Plain Spearman rank-IC on the finite overlap (the in-sample diagnostic)."""
    f = np.asarray(feature_values, dtype=float)
    y = np.asarray(target, dtype=float)
    v = np.isfinite(f) & np.isfinite(y)
    if v.sum() <= 100:
        return float("nan")
    return float(spearmanr(f[v], y[v]).statistic)


# ── per-block cache (npz) ────────────────────────────────────────────────────
def _cache_path(block_idx: int) -> str:
    return os.path.join(CACHE_DIR, f"block_{block_idx:03d}.npz")


def _save_cache(block_idx: int, arrays: BlockArrays, grid: Grid):
    os.makedirs(CACHE_DIR, exist_ok=True)
    d = {"__block__": np.array(arrays.block), "__block_idx__": np.array(block_idx)}
    # per-venue event arrays
    for ex in EXCHANGES:
        d[f"fl_rx__{ex}"]      = arrays.fl_rx[ex]
        d[f"fl_bid_prc__{ex}"] = arrays.fl_bid_prc[ex]
        d[f"fl_bid_qty__{ex}"] = arrays.fl_bid_qty[ex]
        d[f"fl_ask_prc__{ex}"] = arrays.fl_ask_prc[ex]
        d[f"fl_ask_qty__{ex}"] = arrays.fl_ask_qty[ex]
        d[f"tr_rx__{ex}"]      = arrays.tr_rx[ex]
        d[f"tr_prc__{ex}"]     = arrays.tr_prc[ex]
        d[f"tr_qty__{ex}"]     = arrays.tr_qty[ex]
        d[f"tr_sign__{ex}"]    = arrays.tr_sign[ex]
    d["byb_rx"] = arrays.byb_rx
    d["byb_mid"] = arrays.byb_mid
    d["merged_ts"] = arrays.merged_ts
    # grid
    d["g_anchor_ts"]      = grid.anchor_ts
    d["g_tick_at_anchor"] = grid.tick_at_anchor
    d["g_sigma_ev"]       = grid.sigma_ev
    d["g_lambda_ev"]      = grid.lambda_ev
    d["g_price_target"]   = grid.price_target
    d["g_rate_target"]    = grid.rate_target
    for k, v in grid.controls.items():
        d[f"g_ctrl__{k}"] = v
    np.savez_compressed(_cache_path(block_idx), **d)   # compressed: per-block ~85GB->~few GB across 58 blocks


def _load_cache(block_idx: int):
    z = np.load(_cache_path(block_idx), allow_pickle=True)
    block = str(z["__block__"])
    A = BlockArrays(block=block, block_idx=int(z["__block_idx__"]))
    for ex in EXCHANGES:
        A.fl_rx[ex]      = z[f"fl_rx__{ex}"]
        A.fl_bid_prc[ex] = z[f"fl_bid_prc__{ex}"]
        A.fl_bid_qty[ex] = z[f"fl_bid_qty__{ex}"]
        A.fl_ask_prc[ex] = z[f"fl_ask_prc__{ex}"]
        A.fl_ask_qty[ex] = z[f"fl_ask_qty__{ex}"]
        A.tr_rx[ex]   = z[f"tr_rx__{ex}"]
        A.tr_prc[ex]  = z[f"tr_prc__{ex}"]
        A.tr_qty[ex]  = z[f"tr_qty__{ex}"]
        A.tr_sign[ex] = z[f"tr_sign__{ex}"]
    A.byb_rx = z["byb_rx"]; A.byb_mid = z["byb_mid"]; A.merged_ts = z["merged_ts"]
    controls = {k[len("g_ctrl__"):]: z[k] for k in z.files if k.startswith("g_ctrl__")}
    G = Grid(
        block=block, block_idx=A.block_idx,
        anchor_ts=z["g_anchor_ts"], tick_at_anchor=z["g_tick_at_anchor"], merged_ts=A.merged_ts,
        sigma_ev=z["g_sigma_ev"], lambda_ev=z["g_lambda_ev"],
        price_target=z["g_price_target"], rate_target=z["g_rate_target"], controls=controls,
    )
    return A, G


def load_cached(block_idx: int, rebuild: bool = False):
    """Return (BlockArrays, Grid) for a block, computing them once and caching the
    result as an npz under cache/. Subsequent calls load the cache."""
    if not rebuild and os.path.exists(_cache_path(block_idx)):
        try:
            return _load_cache(block_idx)
        except Exception:
            pass   # corrupt/stale cache -> rebuild
    A = load_block_arrays(block_idx)
    G = build_grid(A)
    _save_cache(block_idx, A, G)
    return A, G


# ── run_oss skeleton ─────────────────────────────────────────────────────────
def run_oss(block_indices, feature_modules, head: str = "price", verbose: bool = True):
    """Per block: load/cache the arrays+grid, call each feature module's
    compute(arrays, grid) -> {venue: feature_on_grid}, and collect the per-block
    purged+embargoed walk-forward marginal IC of the feature over the controls.

    feature_modules — list of modules (or objects) exposing:
        NAME (str) and compute(arrays, grid) -> {venue: np.ndarray on the anchor grid}.
    head — "price" (target = price_target) or "rate" (target = rate_target).

    Returns: {module_name: {block_idx: {"joint": float, "per_venue": {venue: float}}}}.
    The joint number is all the module's venue-features added together over the
    controls (the §5 joint marginal); per_venue is each venue alone."""
    results = {}
    for mod in feature_modules:
        name = getattr(mod, "NAME", getattr(mod, "__name__", repr(mod)))
        results[name] = {}

    for bi in block_indices:
        A, G = load_cached(bi)
        target = G.price_target if head == "price" else G.rate_target
        base = [G.controls["rate_momentum"], G.controls["vol_momentum"]]
        for mod in feature_modules:
            name = getattr(mod, "NAME", getattr(mod, "__name__", repr(mod)))
            feats = mod.compute(A, G)                                  # {venue: feature_on_grid}
            venue_arrs = list(feats.values())
            joint = marginal_ic(venue_arrs, target, base)
            per_venue = {v: marginal_ic(feats[v], target, base) for v in feats}
            results[name][bi] = {"joint": joint, "per_venue": per_venue}
            if verbose:
                pv = "  ".join(f"{v}={per_venue[v]:+.3f}" for v in feats)
                print(f"  block[{bi}] {A.block}  {name}  joint={joint:+.3f}  ({pv})")
    return results
