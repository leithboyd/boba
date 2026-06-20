"""Per-column block cache for the v2 builder (docs/v2_dataset_design.md §6, §6.1, §11.1).

Each block is stored as a directory of individually-named column arrays rather than one
monolithic ``(N, F)`` matrix, so adding a feature writes one small ``.npy`` and reuses the
rest. The price is that **cross-column alignment is no longer free** — it becomes an
enforced invariant (every column + cost array == the block's grid length), checked on both
write and read so a truncated/shifted file fails loudly instead of silently corrupting.

    {root}/{grid_hash}/{block}/
        _grid.npz                 # timestamp_ms (the row index)
        {column_name}.npy         # one per feature column, named verbatim
        cost/{field}.npy          # one per populated cost field
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from boba.dataset_v2.raw import SampleArraysRaw, _COST_FIELDS, _present_cost_fields


class AlignmentError(ValueError):
    """A cached array's length doesn't match its block's grid — the one bug per-column
    storage can introduce that a monolithic matrix could not."""


def block_dir(root: Path, grid_hash: str, block: str) -> Path:
    return Path(root) / grid_hash / block


def write_block(root: Path, grid_hash: str, block: str, s: SampleArraysRaw,
                gap_before_ms: float = float("inf")) -> Path:
    """Write one block's columns + cost fields + grid index. Asserts every array is the same
    length as the grid before writing — a mismatch is a slicing bug, not a file to persist.

    ``gap_before_ms`` is the wall-clock gap from the previous block (``inf`` for the dataset
    origin — no prior data, so its EMAs began cold from y=0). The reader uses it to decide,
    per feature span, how much warmup to trim (§4.1): a gap re-warms only the features whose
    memory it exceeds."""
    n = len(s.timestamp_ms)
    if s.x.shape != (n, len(s.column_names)):
        raise AlignmentError(
            f"{block}: x{ s.x.shape} inconsistent with grid N={n} / "
            f"{len(s.column_names)} columns")
    cost = _present_cost_fields(s)
    for f in cost:
        if len(getattr(s, f)) != n:
            raise AlignmentError(f"{block}: cost field {f!r} len {len(getattr(s, f))} != grid N={n}")

    d = block_dir(root, grid_hash, block)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "_grid.npz", timestamp_ms=s.timestamp_ms,
             gap_before_ms=np.array(float(gap_before_ms)))
    for i, name in enumerate(s.column_names):
        np.save(d / f"{name}.npy", np.ascontiguousarray(s.x[:, i]))
    if cost:
        (d / "cost").mkdir(exist_ok=True)
        for f in cost:
            np.save(d / "cost" / f"{f}.npy", getattr(s, f))
    return d


def _load_grid(d: Path) -> np.ndarray:
    with np.load(d / "_grid.npz") as z:
        return z["timestamp_ms"]


def read_block(root: Path, grid_hash: str, block: str,
               columns: list[str], cost_fields: tuple[str, ...] = ()) -> SampleArraysRaw:
    """Read one block's requested columns + cost fields, length-checked against its grid.

    Loads only the files asked for (the per-column win). Every loaded array must match the
    block's grid length or :class:`AlignmentError` is raised — anchoring each column to the
    grid is what keeps the assembled matrix aligned across columns."""
    d = block_dir(root, grid_hash, block)
    t = _load_grid(d)
    n = len(t)
    cols = []
    for name in columns:
        a = np.load(d / f"{name}.npy")
        if len(a) != n:
            raise AlignmentError(f"{block}/{name}: len {len(a)} != grid N={n}")
        cols.append(a)
    x = np.column_stack(cols) if cols else np.empty((n, 0), np.float32)
    kw = {}
    for f in cost_fields:
        a = np.load(d / "cost" / f"{f}.npy")
        if len(a) != n:
            raise AlignmentError(f"{block}/cost/{f}: len {len(a)} != grid N={n}")
        kw[f] = a
    return SampleArraysRaw(x=x.astype(np.float32, copy=False), timestamp_ms=t,
                           column_names=list(columns), **kw)


_SPAN_RE = re.compile(r"_(\d+)(?:ms|b|t|m)(?:@\S+)?$")


def _span_ms(name: str) -> int:
    """Span of a column from its name (ms; event-clock `_b`/`_t` spans use ms as a proxy).
    Instantaneous columns → 0. Tolerates a trailing ``@{CLOCK}`` qualifier (vol on a foreign clock)."""
    m = _SPAN_RE.search(name)
    return int(m.group(1)) if m else 0


def _block_gap_before_ms(root: Path, grid_hash: str, block: str) -> float:
    with np.load(block_dir(root, grid_hash, block) / "_grid.npz") as z:
        return float(z["gap_before_ms"]) if "gap_before_ms" in z.files else 0.0


def gap_trim_ms(span_ms: int, gap_before_ms: float, k: int = 20, tau: float = 1.0) -> int:
    """Warmup rows to drop from a block's start for ONE feature of span ``N`` after a wall-clock
    gap ``G``. A warmed EMA is resilient to gaps shorter than its memory (``G < τ·N`` → 0); a gap
    that exceeds it (``G ≥ τ·N``, including the cold origin's ``G=∞``) forces a full re-warm
    (``k·N``). So short spans re-warm on small gaps (sensitive), long spans ride over them
    (resilient), and the cold first block is just the ``G=∞`` case."""
    if span_ms <= 0:
        return 0
    return int(k * span_ms) if gap_before_ms >= tau * span_ms else 0


def block_trim_ms(columns: list[str], gap_before_ms: float, k: int = 20, tau: float = 1.0) -> int:
    """Per-block warmup trim = max over the read columns of their per-span gap trim."""
    return max((gap_trim_ms(_span_ms(c), gap_before_ms, k, tau) for c in columns), default=0)


def warmup_trim_ms(columns: list[str], k: int = 20) -> int:
    """The cold-origin (``G=∞``) trim = ``k × the longest selected span`` — a convenience."""
    return block_trim_ms(columns, float("inf"), k)


def read_blocks(root: Path, grid_hash: str, blocks: list[str],
                columns: list[str], cost_fields: tuple[str, ...] = (),
                trim_warmup: bool = False, k: int = 20, tau: float = 1.0) -> SampleArraysRaw:
    """Assemble a contiguous block range into one dataset by concatenating per-block arrays
    in block order (§6.1). Iterate blocks in ONE order for every column — that uniform order
    is what keeps columns aligned. Seamless joins (v2 carries EMA state across boundaries).

    ``trim_warmup`` drops each block's start by its **gap-aware** warmup (``block_trim_ms``):
    the cold origin (gap ``∞``) loses ~``k·max_span``; a mid-dataset block loses only the spans
    its preceding gap exceeds (contiguous blocks have gap ``0`` → nothing). Off by default so
    raw / alignment reads see every row."""
    if not blocks:
        raise ValueError("no blocks to read")
    parts = []
    for b in blocks:
        p = read_block(root, grid_hash, b, columns, cost_fields)
        if trim_warmup and len(p.timestamp_ms):
            trim = block_trim_ms(columns, _block_gap_before_ms(root, grid_hash, b), k, tau)
            if trim > 0:
                keep = p.timestamp_ms >= p.timestamp_ms[0] + trim
                p = SampleArraysRaw(x=p.x[keep], timestamp_ms=p.timestamp_ms[keep],
                                    column_names=list(columns),
                                    **{f: getattr(p, f)[keep] for f in cost_fields})
        parts.append(p)
    t = np.concatenate([p.timestamp_ms for p in parts])
    # per-column concat across blocks, then stack — equivalent to vstacking the x's, but keeps
    # the explicit per-column length check at each block (read_block already enforced it).
    x = (np.vstack([p.x for p in parts]) if columns else np.empty((len(t), 0), np.float32))
    if x.shape[0] != len(t):
        raise AlignmentError(f"assembled x rows {x.shape[0]} != concatenated grid {len(t)}")
    kw = {f: np.concatenate([getattr(p, f) for p in parts]) for f in cost_fields}
    return SampleArraysRaw(x=x, timestamp_ms=t, column_names=list(columns), **kw)


def read_dataset(root: Path, grid_hash: str, blocks: list[str],
                 columns: list[str], cost_fields: tuple[str, ...] = (),
                 k: int = 20, tau: float = 1.0) -> SampleArraysRaw:
    """``read_blocks`` with the gap-aware warmup trim on — the training/inference default."""
    return read_blocks(root, grid_hash, blocks, columns, cost_fields, trim_warmup=True, k=k, tau=tau)


def cached_columns(root: Path, grid_hash: str, block: str) -> list[str]:
    """The feature columns currently cached for a block (lets a build skip/compute only the
    missing ones)."""
    d = block_dir(root, grid_hash, block)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.npy"))
