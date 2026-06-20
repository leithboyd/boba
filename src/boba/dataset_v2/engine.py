"""Sequential chunk-processing engine (docs/v2_dataset_design.md §3–§5).

Blocks are processed in dataset order, driven by the memory-aware planner in chunks.py.
Continuity comes from carrying the previous chunk's **event tail** in the checkpoint and
prepending it to the next chunk, building with ``warmup_ms=0`` over the combined range, then
slicing the output back to each block's rows. The tail re-warms *every* family uniformly
(calendar EMAs over the tail grid, event EMAs over the tail events, windows from their last-N
data, ``time_since`` from the last event) without touching the compute. A tail of ≥ ~K·span
reproduces the un-chunked continuous build bit-for-bit.

Two chunk shapes (from chunks.plan_chunks), handled uniformly:
- **coalesce** — several whole blocks in one pass (the 128 GB default);
- **split** — one block's sub-range, when a block exceeds the memory budget (scale-down). A
  split block's sub-ranges are buffered and written once the block is complete; each
  non-final sub-range is built with ``horizon_ms`` of forward data so the block's grid is
  reproduced exactly (no internal holes ⇒ a block is byte-identical however it was chunked).
"""
from __future__ import annotations

import dataclasses
from collections import defaultdict
from pathlib import Path

import numpy as np

from boba.dataset_v2.cache import write_block
from boba.dataset_v2.chunks import chunk_capacity_ticks, plan_chunks
from boba.dataset_v2.raw import DatasetRawConfig, build_features_raw, SampleArraysRaw, _COST_FIELDS
from boba.dataset_v2.session_data import SessionData

MS = 1_000_000          # ns per ms

_BOOK_FIELDS = ("listing_book_t", "listing_book_bid", "listing_book_ask",
                "listing_book_bid_qty", "listing_book_ask_qty",
                "listing_feed_latency_excess_ns")
_TRADE_FIELDS = ("trade_ts", "trade_exchange_ts", "trade_prc", "trade_qty", "trade_dir")
_TARGET_FIELDS = ("book_t", "book_bid", "book_ask", "book_mid",
                  "feed_latency_raw_ns", "feed_latency_excess_ns")


# ── SessionData ops ────────────────────────────────────────────────────────────────────────

def concat_sessions(sessions: list[SessionData]) -> SessionData:
    """Concatenate time-ordered sessions (tail + chunk segments) into one. Each is internally
    sorted and earlier in the list ⇒ earlier in time, so plain concatenation stays sorted."""
    if len(sessions) == 1:
        return sessions[0]
    f0 = sessions[0]
    listings = list(f0.listing_book_t.keys())
    out: dict = {}
    for field in _BOOK_FIELDS + _TRADE_FIELDS:
        out[field] = {l: np.concatenate([getattr(s, field)[l] for s in sessions]) for l in listings}
    for field in _TARGET_FIELDS:
        out[field] = np.concatenate([getattr(s, field) for s in sessions])
    all_rx = np.sort(np.concatenate([s.all_rx for s in sessions]))
    return SessionData(target_listing=f0.target_listing, all_rx=all_rx, **out)


def slice_session(s: SessionData, lo_ns: int, hi_ns: int) -> SessionData:
    """The events with ``lo_ns ≤ t < hi_ns`` as a standalone SessionData."""
    out: dict = {f: {} for f in _BOOK_FIELDS + _TRADE_FIELDS}
    for l in s.listing_book_t.keys():
        bt = s.listing_book_t[l]
        a, b = np.searchsorted(bt, [lo_ns, hi_ns], "left")
        for f in _BOOK_FIELDS:
            out[f][l] = getattr(s, f)[l][a:b]
        tt = s.trade_ts[l]
        c, d = np.searchsorted(tt, [lo_ns, hi_ns], "left")
        for f in _TRADE_FIELDS:
            out[f][l] = getattr(s, f)[l][c:d]
    a, b = np.searchsorted(s.book_t, [lo_ns, hi_ns], "left")
    for f in _TARGET_FIELDS:
        out[f] = getattr(s, f)[a:b]
    p, q = np.searchsorted(s.all_rx, [lo_ns, hi_ns], "left")
    return SessionData(target_listing=s.target_listing, all_rx=s.all_rx[p:q], **out)


def _session_last_ns(s: SessionData) -> int:
    return max(
        max((a[-1] for a in s.listing_book_t.values() if len(a)), default=0),
        max((a[-1] for a in s.trade_ts.values() if len(a)), default=0),
        int(s.book_t[-1]) if len(s.book_t) else 0,
    )


def tail_of(s: SessionData, window_ns: int, upto_ns: int | None = None) -> SessionData:
    """Events in ``[upto − window, upto)`` — the checkpoint that seeds the next chunk.
    ``upto`` is the next chunk's data start (the carry boundary), so the tail never overlaps
    it; it defaults to the last event."""
    if upto_ns is None:
        upto_ns = _session_last_ns(s) + 1
    return slice_session(s, upto_ns - int(window_ns), upto_ns)


# ── Sample ops ──────────────────────────────────────────────────────────────────────────────

def _slice_sample(s: SampleArraysRaw, lo_ms: float, hi_ms: float) -> SampleArraysRaw:
    m = (s.timestamp_ms >= lo_ms) & (s.timestamp_ms < hi_ms)
    kw = {f: getattr(s, f)[m] for f in _COST_FIELDS if getattr(s, f) is not None}
    return SampleArraysRaw(x=s.x[m], timestamp_ms=s.timestamp_ms[m],
                           column_names=list(s.column_names), **kw)


def _concat_samples(parts: list[SampleArraysRaw]) -> SampleArraysRaw:
    if len(parts) == 1:
        return parts[0]
    kw = {f: np.concatenate([getattr(p, f) for p in parts])
          for f in _COST_FIELDS if getattr(parts[0], f) is not None}
    return SampleArraysRaw(
        x=np.vstack([p.x for p in parts]),
        timestamp_ms=np.concatenate([p.timestamp_ms for p in parts]),
        column_names=list(parts[0].column_names), **kw)


# ── Engine ──────────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class Block:
    id: str
    start_ms: int                       # inclusive grid start (block owns [start_ms, end_ms))
    end_ms: int                         # exclusive
    session: SessionData | None = None  # in-memory data; if None, ``load(id)`` fetches it per chunk


def build_chunked(blocks: list[Block], cfg: DatasetRawConfig, root: Path, grid_hash: str,
                  tail_window_ns: int, capacity: int | None = None, load=None,
                  log=lambda *a: None) -> list[str]:
    """Process ``blocks`` (ordered, contiguous) chunk-by-chunk carrying the event tail, and
    write each block's per-column cache. ``capacity`` is the max chunk size (ms-span proxy for
    event-ticks); ``None`` derives it from ``cfg.mem_budget_gb`` — so a smaller budget produces
    smaller (and split) chunks, byte-identical output, lower peak memory.

    Each block owns ``[start_ms, end_ms − horizon_ms)`` — the block-local outcome trim (§3),
    which (with the forward data on split sub-ranges) makes a block chunk-independent.
    """
    listings = list(cfg.listings)
    horizon = int(round(cfg.horizon_ms))
    cfg0 = dataclasses.replace(cfg, warmup_ms=0)        # full grid; warmth comes from the tail
    by_id = {b.id: b for b in blocks}
    # wall-clock gap before each block (∞ for the dataset origin) — the reader uses it to size
    # the per-span warmup trim (a gap re-warms only features whose memory it exceeds).
    gap_by_id = {b.id: (float("inf") if j == 0 else float(b.start_ms - blocks[j - 1].end_ms))
                 for j, b in enumerate(blocks)}
    if capacity is None:
        capacity = chunk_capacity_ticks(cfg.mem_budget_gb, cfg.n_features())
    chunks = plan_chunks([(b.id, b.end_ms - b.start_ms) for b in blocks], capacity)

    written: list[str] = []
    buffers: dict[str, list[SampleArraysRaw]] = defaultdict(list)
    tail: SessionData | None = None

    for ci, chunk in enumerate(chunks):
        # Resolve each segment to (block, output range, data range) in ms.
        segs = []
        for seg in chunk.segments:
            b = by_id[seg.block]
            lo_ms = b.start_ms + seg.start
            hi_ms = b.start_ms + seg.end
            block_end = (seg.end == b.end_ms - b.start_ms)
            if block_end:
                out_hi, data_hi = b.end_ms - horizon, b.end_ms
            else:
                # forward data so the grid reaches hi_ms (no internal hole); tail will stop at hi
                out_hi, data_hi = hi_ms, min(b.end_ms, hi_ms + horizon)
            segs.append((b, lo_ms, hi_ms, out_hi, data_hi, block_end))

        # load each segment's block on demand (one chunk's data in memory at a time) and slice
        seg_sessions = [slice_session(b.session if b.session is not None else load(b.id),
                                      lo * MS, dhi * MS)
                        for (b, lo, _hi, _ohi, dhi, _be) in segs]
        sd = concat_sessions(([tail] if tail is not None else []) + seg_sessions)
        sample = build_features_raw(sd, listings, cfg0)

        for (b, lo_ms, _hi, out_hi, _dhi, block_end) in segs:
            buffers[b.id].append(_slice_sample(sample, lo_ms, out_hi))
            if block_end:
                write_block(root, grid_hash, b.id, _concat_samples(buffers.pop(b.id)),
                            gap_before_ms=gap_by_id[b.id])
                written.append(b.id)

        carry_ms = segs[-1][2]                          # last segment's hi_ms = next chunk's start
        tail = tail_of(sd, tail_window_ns, upto_ns=carry_ms * MS)
        log(f"chunk {ci}: {len(chunk.segments)} seg(s), {sample.x.shape[0]} grid rows")
    return written


# ── Checkpoint persistence (resumable / incremental "append a block" builds) ─────────────────

def save_checkpoint(path: Path, sd: SessionData) -> None:
    """Persist the carried tail so a later run can resume / append without recomputing prior
    blocks — load it and pass as the first session of the next build."""
    arrs: dict = {"__target__": np.array(sd.target_listing), "all_rx": sd.all_rx}
    for f in _BOOK_FIELDS + _TRADE_FIELDS:
        for l, a in getattr(sd, f).items():
            arrs[f"{f}::{l}"] = a
    for f in _TARGET_FIELDS:
        arrs[f] = getattr(sd, f)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **arrs)


def load_checkpoint(path: Path) -> SessionData:
    with np.load(str(path), allow_pickle=False) as z:
        listings = sorted({k.split("::", 1)[1] for k in z.files if "::" in k})
        bt = {f: {l: z[f"{f}::{l}"] for l in listings} for f in _BOOK_FIELDS + _TRADE_FIELDS}
        target = {f: z[f] for f in _TARGET_FIELDS}
        return SessionData(target_listing=str(z["__target__"]), all_rx=z["all_rx"], **bt, **target)
