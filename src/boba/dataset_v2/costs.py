"""Entry/exit cost fields: book state at order-landing time and at the outcome
horizon, plus outcome-window trade extremes — evaluated on a grid of fire times."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from boba.dataset_v2.session_data import SessionData


OUTCOME_MS = 200    # target horizon (ms)


@dataclass(frozen=True)
class CostConfig:
    """Minimal config the cost-field machinery reads. Any config exposing these
    three fields (e.g. dataset_raw's DatasetRawConfig) is accepted (duck-typed)
    by :func:`_cost_fields_for_grid`."""
    baseline_rt_ms: float = 3.0
    processing_ms:  float = 0.5
    horizon_ms:     float = float(OUTCOME_MS)
    cost_fields:    tuple = ()     # explicit selection — only these cost fields are computed + returned (() = none)


# ── Sparse-table range min/max (for vectorised trade extremes) ────────────────

def _build_sparse_table(arr: np.ndarray, use_min: bool) -> np.ndarray:
    """Build sparse table for O(1) range min (use_min=True) or max queries."""
    N = len(arr)
    if N == 0:
        return arr.reshape(1, 0)
    LOG = max(1, int(np.log2(N)) + 2)
    table = np.empty((LOG, N), dtype=arr.dtype)
    table[0] = arr
    fn = np.minimum if use_min else np.maximum
    for k in range(1, LOG):
        half = 1 << (k - 1)
        end  = N - (1 << k) + 1
        if end <= 0:
            table[k] = table[k - 1]
            break
        table[k, :end] = fn(table[k - 1, :end], table[k - 1, half : half + end])
        if end < N:
            table[k, end:] = table[k - 1, end:]
    return table


def _rmq_vec(
    table:     np.ndarray,
    lo:        np.ndarray,   # inclusive start indices
    hi:        np.ndarray,   # exclusive end indices
    empty_val: float,        # value to return for empty range (lo >= hi)
) -> np.ndarray:
    """Vectorised range min/max query for all (lo[i], hi[i]) pairs."""
    has   = hi > lo
    N_tbl = table.shape[1]
    safe_len = np.maximum(hi - lo, 1).astype(np.float64)
    k    = np.floor(np.log2(safe_len)).astype(np.int64)
    k    = np.clip(k, 0, table.shape[0] - 1)
    sz   = np.int64(1) << k
    a_idx = np.clip(lo,      0, N_tbl - 1)
    b_idx = np.clip(hi - sz, 0, N_tbl - 1)
    fn = np.minimum if np.isposinf(empty_val) else np.maximum
    return np.where(has, fn(table[k, a_idx], table[k, b_idx]), empty_val)


# ── Cost field computation ────────────────────────────────────────────────────

def _build_cost_tables(data: SessionData, need_extremes: bool = True) -> dict:
    """Build shared read-only tables for cost field computation.

    Called once per session; the returned dict is shared across all parallel
    workers (numpy array reads are thread-safe). ``need_extremes=False`` skips the
    O(M log M) trade min/max sparse tables (the heavy ones) when no trade-extreme
    cost field is requested — keeps the build lean.
    """
    eps = 1e-10
    book_mid = data.book_mid
    M = len(book_mid)

    # ── Book mid prefix cumsums ───────────────────────────────────────────────
    lr = np.zeros(M, np.float64)
    if M > 1:
        lr[1:] = np.log(np.maximum(book_mid[1:], eps) / np.maximum(book_mid[:-1], eps))
    lr_up = np.maximum(lr, 0.0)
    lr_dn = np.minimum(lr, 0.0)
    cs_up     = np.concatenate([[0.0], np.cumsum(lr_up)])
    cs_dn     = np.concatenate([[0.0], np.cumsum(lr_dn)])
    cs_cnt_up = np.concatenate([[0.0], np.cumsum((lr > 0).astype(np.float64))])
    cs_cnt_dn = np.concatenate([[0.0], np.cumsum((lr < 0).astype(np.float64))])

    # ── Trade prefix cumsums and sparse tables ────────────────────────────────
    tgt_prc = data.trade_prc[data.target_listing]
    tgt_qty = data.trade_qty[data.target_listing]
    tgt_dir = data.trade_dir[data.target_listing]

    is_buy  = tgt_dir > 0
    is_sell = tgt_dir < 0

    buy_cnt_cs  = np.concatenate([[0],   np.cumsum(is_buy.astype(np.int64))])
    sell_cnt_cs = np.concatenate([[0],   np.cumsum(is_sell.astype(np.int64))])
    buy_not_cs  = np.concatenate([[0.0], np.cumsum(np.where(is_buy,  tgt_prc * tgt_qty, 0.0))])
    sell_not_cs = np.concatenate([[0.0], np.cumsum(np.where(is_sell, tgt_prc * tgt_qty, 0.0))])

    if need_extremes:
        prc32 = tgt_prc.astype(np.float32)
        sell_min_tbl = _build_sparse_table(np.where(is_sell, prc32, np.float32( np.inf)), use_min=True)
        sell_max_tbl = _build_sparse_table(np.where(is_sell, prc32, np.float32(-np.inf)), use_min=False)
        buy_min_tbl  = _build_sparse_table(np.where(is_buy,  prc32, np.float32( np.inf)), use_min=True)
        buy_max_tbl  = _build_sparse_table(np.where(is_buy,  prc32, np.float32(-np.inf)), use_min=False)
    else:
        sell_min_tbl = sell_max_tbl = buy_min_tbl = buy_max_tbl = None

    return dict(
        # book mid movement
        cs_up=cs_up, cs_dn=cs_dn, cs_cnt_up=cs_cnt_up, cs_cnt_dn=cs_cnt_dn,
        # trade counts / notionals
        buy_cnt_cs=buy_cnt_cs, sell_cnt_cs=sell_cnt_cs,
        buy_not_cs=buy_not_cs, sell_not_cs=sell_not_cs,
        # trade min/max sparse tables
        sell_min_tbl=sell_min_tbl, sell_max_tbl=sell_max_tbl,
        buy_min_tbl=buy_min_tbl,   buy_max_tbl=buy_max_tbl,
    )


def _cost_fields_for_grid(
    data:      SessionData,
    tables:    dict,
    cfg:       CostConfig,
    grid_t_ns: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute cost fields for one grid slice using pre-built tables."""
    N   = len(grid_t_ns)
    eps = 1e-10
    want = getattr(cfg, "cost_fields", ()) or ()   # explicit selection; () = no cost fields
    need_trade = any(
        n in want for n in ("c_buy_trade_min_l", "c_buy_trade_max_l",
                            "c_sell_trade_min_l", "c_sell_trade_max_l"))

    baseline_ns   = int(cfg.baseline_rt_ms  * 1_000_000)
    processing_ns = int(cfg.processing_ms   * 1_000_000)
    horizon_ns    = int(cfg.horizon_ms      * 1_000_000)

    book_t   = data.book_t
    book_bid = data.book_bid
    book_ask = data.book_ask
    book_mid = data.book_mid
    feed_exc = data.feed_latency_excess_ns
    feed_raw = data.feed_latency_raw_ns

    eval_gi   = np.searchsorted(book_t, grid_t_ns, side="right") - 1
    valid     = eval_gi >= 0
    eval_gi_c = np.maximum(eval_gi, 0)

    eval_mid = np.where(valid, book_mid[eval_gi_c], 0.0)
    eval_bid = np.where(valid, book_bid[eval_gi_c], 0.0)
    eval_ask = np.where(valid, book_ask[eval_gi_c], 0.0)
    feed_exc_at_grid = np.where(valid, feed_exc[eval_gi_c].astype(np.int64), 0)
    feed_raw_at_grid = np.where(valid, feed_raw[eval_gi_c].astype(np.int64), 0)

    t_entry_ns = grid_t_ns + baseline_ns + feed_exc_at_grid + processing_ns
    t_exit_ns  = grid_t_ns + horizon_ns

    entry_gi = np.searchsorted(book_t, t_entry_ns, side="right") - 1
    exit_gi  = np.searchsorted(book_t, t_exit_ns,  side="left")

    valid_entry = (entry_gi >= 0) & valid
    valid_cost  = valid_entry & (exit_gi > entry_gi)

    entry_gi_c   = np.maximum(entry_gi, 0)
    exit_snap_gi = np.maximum(entry_gi_c, np.maximum(exit_gi, 0) - 1)

    bid_entry = np.where(valid_entry, book_bid[entry_gi_c], 0.0)
    ask_entry = np.where(valid_entry, book_ask[entry_gi_c], 0.0)
    bid_exit  = np.where(valid_cost,  book_bid[exit_snap_gi], 0.0)
    ask_exit  = np.where(valid_cost,  book_ask[exit_snap_gi], 0.0)
    mid_exit  = np.where(valid_cost,  book_mid[exit_snap_gi], 0.0)

    safe_mid = np.maximum(eval_mid, eps)

    def _log_ratio(num, denom, mask):
        return np.where(mask, np.log(np.maximum(num, eps) / denom), 0.0)

    c_bid_entry_l = _log_ratio(bid_entry, safe_mid, valid_entry & (bid_entry > 0))
    c_ask_entry_l = _log_ratio(ask_entry, safe_mid, valid_entry & (ask_entry > 0))
    c_bid_exit_l  = _log_ratio(bid_exit,  safe_mid, valid_cost  & (bid_exit  > 0))
    c_ask_exit_l  = _log_ratio(ask_exit,  safe_mid, valid_cost  & (ask_exit  > 0))
    c_mid_exit_l  = _log_ratio(mid_exit,  safe_mid, valid_cost  & (mid_exit  > 0))
    eval_bid_l    = _log_ratio(eval_bid,  safe_mid, valid        & (eval_bid  > 0))
    eval_ask_l    = _log_ratio(eval_ask,  safe_mid, valid        & (eval_ask  > 0))

    # ── Mid movement: query pre-built prefix cumsums ──────────────────────────
    cs_up     = tables["cs_up"];     cs_dn     = tables["cs_dn"]
    cs_cnt_up = tables["cs_cnt_up"]; cs_cnt_dn = tables["cs_cnt_dn"]

    vc_idx = np.where(valid_cost)[0]
    lo_v   = entry_gi_c[vc_idx]
    hi_v   = exit_snap_gi[vc_idx]
    has_r  = hi_v > lo_v

    c_mid_up_cnt = np.zeros(N, np.float32)
    c_mid_dn_cnt = np.zeros(N, np.float32)
    c_mid_up_l   = np.zeros(N, np.float32)
    c_mid_dn_l   = np.zeros(N, np.float32)

    if has_r.any():
        idx_r = vc_idx[has_r]; lo_r = lo_v[has_r]; hi_r = hi_v[has_r]
        c_mid_up_cnt[idx_r] = (cs_cnt_up[hi_r + 1] - cs_cnt_up[lo_r + 1]).astype(np.float32)
        c_mid_dn_cnt[idx_r] = (cs_cnt_dn[hi_r + 1] - cs_cnt_dn[lo_r + 1]).astype(np.float32)
        c_mid_up_l[idx_r]   = (cs_up[hi_r + 1]     - cs_up[lo_r + 1]).astype(np.float32)
        c_mid_dn_l[idx_r]   = -(cs_dn[hi_r + 1]    - cs_dn[lo_r + 1]).astype(np.float32)

    # ── Forward-window mid-move COUNT over (eval, exit] ───────────────────────
    # Total mid moves (up+down) in the SAME window as c_mid_exit_l (the move target): from the
    # fire-time book (eval) to the horizon (exit), NOT entry-delayed. So count==0 ⇔ no net move,
    # i.e. it is the subordinator/rate-head target whose zero bin is the no-move atom.
    cs_cnt_tot = cs_cnt_up + cs_cnt_dn
    c_mid_move_count = np.zeros(N, np.float32)
    fwd_exit = np.maximum(eval_gi_c, np.maximum(exit_gi, 0) - 1)
    vf = np.where(valid & (exit_gi > eval_gi))[0]
    if len(vf):
        c_mid_move_count[vf] = (cs_cnt_tot[fwd_exit[vf] + 1] - cs_cnt_tot[eval_gi_c[vf] + 1]).astype(np.float32)

    # ── Trade extremes: query pre-built sparse tables ─────────────────────────
    tgt_ts = data.trade_ts[data.target_listing]
    ti_lo  = np.searchsorted(tgt_ts, t_entry_ns, side="left")
    ti_hi  = np.searchsorted(tgt_ts, t_exit_ns,  side="left")

    c_sell_min_l = np.full(N,  np.inf,  np.float32)
    c_sell_max_l = np.full(N, -np.inf,  np.float32)
    c_buy_min_l  = np.full(N,  np.inf,  np.float32)
    c_buy_max_l  = np.full(N, -np.inf,  np.float32)
    c_buy_cnt    = np.zeros(N, np.float32)
    c_sell_cnt   = np.zeros(N, np.float32)
    c_buy_val_l  = np.full(N, -np.inf,  np.float32)
    c_sell_val_l = np.full(N, -np.inf,  np.float32)

    has_trd = valid_cost & (ti_hi > ti_lo)
    ht_idx  = np.where(has_trd)[0]

    if need_trade and len(ht_idx) > 0 and len(tgt_ts) > 0:
        lo_t = ti_lo[ht_idx]; hi_t = ti_hi[ht_idx]; sm = safe_mid[ht_idx]

        bcs = tables["buy_cnt_cs"];  scs = tables["sell_cnt_cs"]
        bnc = tables["buy_not_cs"];  snc = tables["sell_not_cs"]

        b_cnt = (bcs[hi_t] - bcs[lo_t]).astype(np.float32)
        s_cnt = (scs[hi_t] - scs[lo_t]).astype(np.float32)
        bv    = bnc[hi_t] - bnc[lo_t]
        sv    = snc[hi_t] - snc[lo_t]

        c_buy_cnt[ht_idx]    = b_cnt
        c_sell_cnt[ht_idx]   = s_cnt
        c_buy_val_l[ht_idx]  = np.where(bv > 0, np.log(np.maximum(bv, 1e-300)).astype(np.float32), -np.inf)
        c_sell_val_l[ht_idx] = np.where(sv > 0, np.log(np.maximum(sv, 1e-300)).astype(np.float32), -np.inf)

        has_buy  = b_cnt > 0; has_sell = s_cnt > 0

        raw = _rmq_vec(tables["sell_min_tbl"], lo_t, hi_t,  np.inf)
        c_sell_min_l[ht_idx] = np.where(has_sell, np.log(np.maximum(raw, eps) / np.maximum(sm, eps)).astype(np.float32),  np.inf)

        raw = _rmq_vec(tables["sell_max_tbl"], lo_t, hi_t, -np.inf)
        c_sell_max_l[ht_idx] = np.where(has_sell, np.log(np.maximum(raw, eps) / np.maximum(sm, eps)).astype(np.float32), -np.inf)

        raw = _rmq_vec(tables["buy_min_tbl"],  lo_t, hi_t,  np.inf)
        c_buy_min_l[ht_idx]  = np.where(has_buy,  np.log(np.maximum(raw, eps) / np.maximum(sm, eps)).astype(np.float32),  np.inf)

        raw = _rmq_vec(tables["buy_max_tbl"],  lo_t, hi_t, -np.inf)
        c_buy_max_l[ht_idx]  = np.where(has_buy,  np.log(np.maximum(raw, eps) / np.maximum(sm, eps)).astype(np.float32), -np.inf)

    out = dict(
        eval_bid_l             = eval_bid_l.astype(np.float32),
        eval_ask_l             = eval_ask_l.astype(np.float32),
        eval_mid               = eval_mid.astype(np.float32),
        feed_latency_raw_ms    = (feed_raw_at_grid / 1e6).astype(np.float32),
        feed_latency_excess_ms = (feed_exc_at_grid / 1e6).astype(np.float32),
        c_bid_entry_l=c_bid_entry_l.astype(np.float32), c_ask_entry_l=c_ask_entry_l.astype(np.float32),
        c_bid_exit_l=c_bid_exit_l.astype(np.float32),   c_ask_exit_l=c_ask_exit_l.astype(np.float32),
        c_mid_exit_l=c_mid_exit_l.astype(np.float32),
        c_sell_trade_min_l=c_sell_min_l, c_sell_trade_max_l=c_sell_max_l,
        c_buy_trade_min_l=c_buy_min_l,   c_buy_trade_max_l=c_buy_max_l,
        c_buy_trade_count=c_buy_cnt,      c_sell_trade_count=c_sell_cnt,
        c_buy_trade_value_l=c_buy_val_l,  c_sell_trade_value_l=c_sell_val_l,
        c_mid_up_count=c_mid_up_cnt, c_mid_down_count=c_mid_dn_cnt,
        c_mid_up_l=c_mid_up_l,       c_mid_down_l=c_mid_dn_l,
        c_mid_move_count=c_mid_move_count,
    )
    return {k: v for k, v in out.items() if k in want}
