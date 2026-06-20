# Dataset v2 — sequential streaming feature builder

**Status:** Draft / proposed. The config-level cleanups in §7 are already landed; the
sequential model (§3–§5) and per-column cache (§6) are the work this doc specifies.

---

## 1. Motivation

Two problems with the current per-block builder (`boba/dataset/raw.py`), both rooted in
treating each ~24 h block as an independent mini-session:

**Warmup is wrong for long averages.** Each block resets every EMA to `y[-1]=0` at its
start (`_ewm_1d`), and the single global `warmup_ms` just trims the grid start
(`_grid_bounds_ms`). Consequences:
- Event-clock EMAs (`_b`/`_t`) get *some* burn-in (they're computed over the full event
  history then sampled to the grid), but only as much as the events in the warmup window.
- Calendar EMAs (`_ms`) get **zero** burn-in — they're computed on the post-warmup grid
  from `y[-1]=0`, so the `~3·span` transient sits in the output and `warmup_ms` can't
  remove it (raising it just moves the start and the transient reappears).
- Worst of all: every block boundary, a long EMA snaps back toward 0 and re-climbs — a
  **sawtooth with a ~24 h period** injected into exactly the slowest, most trend-like
  features. (Documented today as impl-note 5 in `raw_features.md`.)

**The cache blows up disk.** A block is cached as one monolithic `(N, F)` npz keyed by
the ordered column set (`config_str` → `cache_key`). Adding *one* column changes the hash,
so every block recomputes and writes a brand-new full matrix; the old matrices are
orphaned (never pruned). Iterating on features ⇒ repeated full-size copies ⇒ out of disk.

## 2. Goals

1. **Continuous features across blocks** — block boundaries invisible; no reset sawtooth;
   long EMAs are exactly the global continuous average.
2. **Per-column cache** — adding a column computes and stores *that column*, reusing the
   rest. No monolithic matrix, no orphaned full copies.
3. **No warmup footgun** — warmup is automatic, never a hand-set number that can be too
   small and silently emit cold rows.

## 3. Core model — one sequential streaming pass

Process blocks in **dataset order**; each block initializes its recursive state (EMAs,
`time_since_*`, dt windows) from the **previous block's end-state** (a checkpoint). Within
a block, parallelism is unchanged (columns × listings × cost chunks, as today).

Because state flows forward, **warmup is automatic** — every block after the first starts
already converged. The only genuinely cold region is the **very first block** of the whole
dataset (nothing precedes it); flag/discard it once. `warmup_ms` disappears entirely.

The state carried is tiny — a few scalars per recursive feature per listing — so it is
cheap to store and carry. Seeding is a one-liner: `_ewm_1d` runs through
`scipy.signal.lfilter`, which takes an initial-condition argument `zi`; "init from the
previous block" = pass the carried scalar as `zi` instead of `0`.

### One cross-block edge: the checkpoint

The only new cross-block dependency is **backward** — the carried **checkpoint** seeds each
block's causal features. The **forward** side (the cost/outcome window looking `horizon_ms`
ahead) stays exactly as v1: each block trims its last `horizon_ms` **block-locally**
(`_grid_bounds_ms`). We do **not** read the next block to fill it. The lost sliver is
`~horizon_ms` (e.g. 200 ms) at each ~24 h boundary — meaningless noise to a large training
run, not worth a cross-block read.

So only the **first block** of the dataset is genuinely cold — it starts every EMA from
`y=0` (no checkpoint precedes it). Generation just emits those cold-start values; trimming
the transient is a downstream concern (§4.1), not a build path. Every block (including the
last) trims its `horizon_ms` tail uniformly, so there's no special end-of-dataset case
either.

## 4. Checkpoint

The checkpoint is just the **recursive feature state** needed to continue each feature
across the boundary — a few scalars per recursive feature per listing:

- **Event-clock EMAs** (`_b`, `_t`): last EMA value.
- **Calendar EMAs** (`_ms`): last EMA value + last held input (microprice), so the next
  block forward-fills correctly from the boundary.
- **`time_since_*`**: last event timestamp of the relevant kind (`time_since_last_trade_ms`,
  `time_since_spread_wide_ms`).
- **event-count windows** (`dt_b`, `dt_t`, `dt_m`): the last `N` event timestamps of that
  type (BBO events / trades / mid-moves).
- **ms windows** (`return_ms`, `trade_count_ms`): the last `N` ms of input — the microprice
  tail for `return`, the trade times for `trade_count`.

The window tails are the only non-scalar state, and `N` can be large (e.g. 30k), so store
them minimally:
- **Dedup by source, not column** — all `dt_b` columns share one BBO-time tail; store the
  last `max_N` timestamps **once per (event-type, listing)**, sized to the largest selected
  `N` for that type. So `dt_1000b` + `dt_30000b` share a single 30k buffer.
- **Derive `max_N` from the selected columns** — carry only what the biggest window needs;
  nothing for an unselected family.
- **Sparse for the ms-windows** — `return`/`trade_count` store the *events* in the last
  `max_N` ms (bounded by activity), not a dense 1 ms grid.
- **Compact + compressed** — int64 ns (delta-encodable), npz-compressed. At `N=30k` that's
  ~240 KB per event-type per listing (~4 MB for the 6-listing checkpoint) — negligible next
  to the block's multi-GB matrix. A ring buffer of size `max_N` is the natural in-compute
  structure; the checkpoint is its contents serialized in order.

Stored as a small `_checkpoint.npz` per block (§6). On a cache miss for block *k*, load
*k−1*'s checkpoint (O(1)); only re-chain if it's missing. **Appending** a new block to a
growing archive is therefore O(1): load last checkpoint, compute, save.

### 4.1 Gaps — generation carries state; the *read* trims per-span by resilience

Generation does nothing special at a gap: it passes the checkpoint state forward, and
continuity *is* that carried state.

- For `time_since_*` and `dt`, the gap is already implicit in the carried timestamps — a
  large gap simply makes `time_since` large and stretches a `dt` window, which is correct
  with zero gap logic.
- For EMAs, the carried value just continues; we don't decay it over the gap. The build
  records one number per block — `gap_before_ms` (the wall-clock gap to the previous block,
  `∞` for the dataset origin) — and otherwise stores nothing gap-related.

**Trimming is a read-time decision, and it's per-span by resilience.** A warmed EMA of span
`N` is resilient to a gap `G` *shorter than its memory*; a gap that exceeds it forces a
re-warm. So the trim for a feature is

```
gap_trim(N, G) = k·N   if  G ≥ τ·N   else   0          (τ default 1)
```

This unifies the two cases the cold-vs-gap distinction was really about:
- **First block** is the `G = ∞` case → every span re-warms → `k·max_span` (the hard trim).
- **Mid-dataset gap** → only spans the gap *exceeds* re-warm. **Short EMAs are sensitive**
  (a modest gap exceeds their short memory), **long EMAs ride over it** (the gap is a small
  fraction of their window). For small spans the cold and gap trims coincide; for large
  spans they differ sharply — hard-trimming `k·N` after a 0.15 s seam would throw away
  minutes of good warm data.

`read_blocks(trim_warmup=True)` / `read_dataset` apply this per block from its stored
`gap_before_ms`; contiguous blocks (`G=0`) trim nothing. Off by default so raw/alignment
reads see every row.

## 5. Chunk planner (performance)

**Decouple the compute unit from the cache unit:** compute over a *chunk*, cache per
*block*.

`plan_chunks(blocks, mem_cap) -> list[list[block]]` walks blocks in order and greedily
coalesces **adjacent** blocks into a chunk until the chunk reaches ~one normal block's
memory budget. Build shared state + EMAs once over the chunk, then **slice the output back
into per-block cache files** and carry the checkpoint to the next chunk.

- **No-op for normal blocks** — a full-size block is its own chunk. Only small/fragmented
  blocks (partial blocks, low-data periods) get batched, exactly where per-block fixed
  overhead hurt. Zero downside.
- **Memory stays flat** — the cap bounds peak at ~one block's worth regardless of how many
  small blocks coalesce.
- **Slightly *more* correct** — inside a chunk the EMA flows continuously with no
  intermediate checkpoint; checkpoints are only needed at chunk boundaries.

**Determinism rule:** a block's cached bytes must not depend on how it was chunked. This
holds by construction — features are causal (they only need the carried checkpoint), and
each block trims its own `horizon_ms` tail block-locally regardless of chunk membership. So
block A is byte-identical whether computed alone or coalesced with B; the planner is a pure
compute optimization with no effect on output.

## 6. Cache design

Per-column, content-addressed, keyed by a **grid hash** that excludes the column selection.

```
artifacts/dataset_v2/{grid_hash}/{block}/
    _grid.npz                         # row index for this block: timestamp_ms, N
    _checkpoint.npz                   # end-state to seed the next block
    bin_doge_usdt_p_microprice.npy    # one file per feature column, named verbatim
    bin_doge_usdt_p_ema_ofi_10b.npy
    …
    cost/eval_mid.npy                 # one file per requested cost field
    cost/c_mid_move_count.npy
```

**`grid_hash`** = everything output-affecting **except** the per-column selection
(`columns`) and the cost-field selection (`cost_fields`), since those are carried by
filenames. Concretely: `listings`, `target_listing`, `horizon_ms`, `only_on_event`,
`only_on_trade_or_move`, `wide_threshold`, `microprice_ref`, `baseline_rt_ms`,
`processing_ms`. (i.e. today's `config_str` minus the column-name list, minus
`cost_fields`, **minus `warmup_ms`** — which no longer exists.)

**What is *not* hashed:**
- The derived warmup amount — there isn't one; state carries.
- The compute mode — there's one model (sequential); see §9 for the deferred alternative.
- Anything gap-related — generation ignores gaps; nothing gap-derived is stored or hashed.
- Read-time policy (transient/gap trimming) — applied downstream, never touches the cache.

The one thing folded into `_CACHE_VERSION` is the **model itself** (continuous-carry
semantics) — bumped once on adoption, and again only if the carry/checkpoint math changes.

**Why per-column wins on disk:** a column's values are determined by its name + the grid
identity, so each unique column is stored **once** per grid identity regardless of how many
configs reference it. Adding a feature writes one `~N·4`-byte `.npy`, not a whole `N·F`
matrix; orphans shrink from a matrix to a single column. The builder already guarantees a
subset build is byte-identical to those columns of a larger build (a tested invariant), so
the per-column slices are safe to share.

**Filenames are the literal column names** (`^[a-z0-9_]+$`, well under the 255-byte limit)
— `ls` the dir to see exactly what's cached, no reverse lookup. Cost fields live in a
`cost/` subdir to stay visually separate.

### 6.1 Read path

Assembling a dataset over a block range = concatenate the per-block arrays in block order:

```python
cols = [np.concatenate([load(f"{gh}/{b}/{c}.npy")       for b in blocks]) for c in selected]
t    =  np.concatenate([load(f"{gh}/{b}/_grid.npz")["timestamp_ms"] for b in blocks])
x    =  np.column_stack(cols)
```

- **Column selection is free** — load only the `.npy` you asked for; the rest never touches
  memory.
- **Joins are seamless** — because EMAs carry across boundaries, the concatenated column is
  the true continuous feature (no v1 sawtooth at the block seams).
- **One order, applied uniformly** — iterate blocks in a single deterministic order for
  *every* column, and assert equal lengths before stacking (§11.1). That uniform ordering is
  what keeps columns aligned.

**Eager vs. lazy.** Eager `np.concatenate` is the simple path — fine for small ranges, but
blocks exist to bound memory, so a multi-month `(N_total, F)` float32 can be enormous. The
per-column `.npy` layout supports the alternatives directly:
- **memmap** each file (`np.load(..., mmap_mode='r')`) and present a virtual concatenation;
- **stream** block-by-block, or index by `(block, offset)` for shuffled training batches —
  no full materialization.

**Trims apply here, at assembly — never in the cache.** The build emits raw continuous
values; the read path is where policy lands:
- drop the **cold first-block** warmup region (the one genuine transient);
- optionally **mask post-gap rows**, with the gap derived from the concatenated timestamps
  (`t[i] − t[i−1]`), per the consumer's tolerance and the spans it actually uses.

Different consumers trim differently from the same cached bytes (§4.1). `_grid.npz` supplies
the timestamps for the concat and gap derivation; `_checkpoint.npz` is a build/index artifact
and is **not** part of the read output.

## 7. Already-landed config changes (context)

These shipped ahead of the v2 builder and inform the grid identity above:

- **`cost_fields` is an explicit selection** (`tuple`, default `()` = none) — no implicit
  "all" sentinel. Folded into `config_str`; `make_cfg` re-supplies the full list for
  legacy tests.
- **`microprice_ref` has no default** (`field(default_factory=dict)`) — the DOGE-specific
  `0.15` is gone; absent listings fall back to `0.0` (no centering). `make_cfg` re-supplies
  the legacy `0.15` so tests are unchanged.
- **`listings` is the single name** for the per-venue set — `build_features_raw` /
  `feature_names` / `make_cfg` take `listings`; `SessionData` exposes `listing_book_*` /
  `target_listing`. `exchange` now means *only* the matching-engine clock
  (`exchange_time`, `trade_exchange_ts`).

## 8. Robustness / edge cases

| Edge | Behavior (generation) |
|---|---|
| First block of dataset | Cold (no checkpoint) — emit cold-start values; the transient is
  trimmed downstream (§4.1), not during the build. |
| Every block tail | Trim the last `horizon_ms` block-locally (as v1) — the per-boundary
  loss is negligible noise; no cross-block forward read. |
| Gaps | Ignored by generation; carried in the state (§4.1). Any trim is downstream. |
| Missing checkpoint | Re-chain from the nearest existing checkpoint forward. |
| Corrupt/bad block | State flows forward, so it poisons downstream — the main
  fault-isolation cost vs. the old block-independent model. Resumable from the last good
  checkpoint. |

## 9. Deferred / open

- **Cross-block parallelism.** All-sequential gives up embarrassingly-parallel cold
  backfill (block *k* needs *k−1*'s checkpoint). On a single machine this is ~modest
  (a block already saturates cores/memory). If a large cold backfill ever dominates: EMAs
  are linear recurrences, so they parallelize across blocks via a prefix-scan (compute each
  block's `(decay=(1−α)^M, zero-start output)` in parallel → cheap O(#blocks) scan for
  start-states → apply). Exactly sequential-correct. Defer until it bites.
- **Per-column lookback mode.** A block-independent "read a bounded `K·span` tail" mode is
  the alternative for short spans; it would re-introduce a per-column `mode ∈
  {lookback, sequential}` tagged into the cache filename (so changing it re-keys only that
  column). Not built — the extension point if cluster-scale backfill parallelism becomes a
  hard requirement.
- **Soft-EMA window variants** (to be added later). Each hard window has an EMA-span-`N`
  analog with O(1) scalar state: `dt → N·ema(inter-arrival)`, `trade_count → N·ema(per-ms
  count)`, `return → log mp − ema(log mp)` (deviation-from-EMA; the one that's a genuinely
  different feature, since `return` is a fixed lag not a window). Smoother, edge-free, and
  collapses the window checkpoints to scalars — but they are *different features*, so it's a
  feature-design choice, not a free swap. The exact windows above stay as-is until then.

## 10. Migration

- **Additive, not in-place.** v2 lands as a **separate builder/module** and cache dir
  (`artifacts/dataset_v2/`); the existing v1 builder (`boba/dataset/raw.py`) and its cache
  are left **untouched**. Nothing in the current path changes while v2 is built and
  validated side-by-side.
- Bump `_CACHE_VERSION` on adoption — the continuous model changes values vs. per-block
  reset, so old caches must not collide. They become orphaned; prune them.
- Add a prune utility (delete cache dirs whose `grid_hash` / version is stale) — independent
  of v2, also fixes today's orphan-accumulation disk pressure.
- The tested invariants (`subset == full` bit-identity, column order = expansion order,
  causality) carry over; fixtures must feed a checkpoint seed instead of assuming a cold
  block start.

## 11. Validation against v1 (reference)

v1 (`boba.dataset`) is kept untouched precisely so v2 can be diffed against it on the same
templates + data. **Always compare by timestamp join, never by row index** — v2's grid
differs from v1's (no `warmup_ms` trim ⇒ extra leading rows), so align on `timestamp_ms` and
compare the overlap.

### 11.1 Alignment — pin it with the v1 oracle

v1 stored one `(N, F)` matrix, so columns were aligned for free. v2 stores each column
separately, so a bug can pair a column's values with the **wrong rows** — no length error,
no crash, silent corruption. The v1 oracle catches exactly this, *if the comparison is set
up right*:

- **Per column, join on timestamp.** Pair each v2 column with its `_grid.npz` timestamps,
  pair v1's same-named column with its timestamps, **inner-join on `timestamp_ms`** (v1's
  rows are a subset — v2 has the extra warmup prefix), and compare. A column whose values are
  shifted relative to its grid mismatches v1 even though both share the timestamp axis — so
  the join *catches and localizes* it.
- **Match columns by name, never by position.** Compare v2 `microprice` to v1 `microprice`;
  never zip two column lists by index (a reordering would mis-pair everything). Separately
  assert the assembled `x`'s column order == `cfg.columns` expansion order (the existing
  invariant).
- **The instantaneous columns are the alignment probe.** `microprice`, `spread_width`,
  `book_depth`, `book_imbalance`, `spread_wide_flag`, `feed_latency_excess_ms`,
  `{buy,sell}_trade_value` have **no model difference**, so they must be **bit-exact** on the
  join. A mismatch there is pure value↔timestamp mispairing — an alignment bug, full stop
  (no EMA-tolerance ambiguity to hide behind).
- **One probe covers the whole block.** Every column in a block is sliced on the *same*
  grid, so if the instantaneous probe is exactly aligned, all columns are — the stateful
  columns then only validate the model (within the §11.2 tolerance), not alignment.

Cheap structural backstops, independent of v1: assert every column/cost array length == the
block's grid `N` at write and again on read before stacking — a truncated file would
otherwise shift the matrix silently.

### 11.2 Value ladder (on timestamp-joined rows)

1. **Per-block math is a faithful copy.** Run v2's single-block path on **v1's grid**
   (warmup-trimmed, cold, no checkpoint) → **bit-for-bit equal on every column**. Validates
   the copied math in isolation, before any continuity code — the strongest, cheapest check.
2. **Instantaneous features always match** to float precision (`microprice`, `spread_width`,
   …): no lookback, no carried state, so independent of block start. A mismatch ⇒ a
   plumbing/alignment bug, not a model difference.
3. **Short-span EMAs mostly agree.** v2 carries state while v1 resets to `y=0`; the gap at
   `r` rows past a boundary is `~(1−α)^r · |EMA_at_boundary|`, gone after `~K·N` rows — tiny
   vs. a ~24 h block. Disagreement is concentrated in the post-boundary warmup zone.
4. **Long-span EMAs diverge — by design.** v1's sawtooth vs. v2's continuity; divergence
   growing with span and clustered at boundaries is the **signal v2 works**, not a
   regression.

## 12. Implementation status

Built under `src/boba/dataset_v2/` (a fork of v1, which stays untouched as the oracle). 34
v2 tests, all green.

- **`chunks.py`** — `mem_budget_gb` → capacity; `plan_chunks` coalesces small blocks /
  splits oversized ones. Perf/memory config proven out of `config_str`/`cache_key`.
- **`cache.py`** — per-column write/read with the alignment invariant enforced both sides
  (`AlignmentError`); `read_blocks` concatenates a range.
- **`engine.py`** — sequential chunk loop with carried-tail checkpoint, coalesce **and split**
  execution, on-demand block loading (`load`), and checkpoint `save`/`load`. Core gate:
  chunked+carry == the un-chunked continuous build, exactly (incl. splits and cost fields).
- **`raw.py`** — `grid_hash()` (cache dir key = grid identity, excl. columns/cost/warmup/perf);
  `event_mask` (`none`/`book`/`trade`/`both`/`trade_or_move`); `mem_budget_gb`/`n_workers`.
- **`driver.py`** — `build_dataset_v2` (loader-agnostic, derives `grid_hash` + tail window)
  and `build_from_blocks` (the `boba.io` integration glue, exercised against real data).
- **Read-time gap-aware warmup trim** — the engine stores `gap_before_ms` per block (`∞` for
  the origin); `read_blocks(trim_warmup=True)` / `read_dataset` drop each block's start by
  `block_trim_ms = max over read columns of gap_trim(N, G)` where `gap_trim(N,G)=k·N if G≥τ·N
  else 0` (§4.1). So the cold origin loses `k·max_span`, a gap re-warms only the spans it
  exceeds (short EMAs sensitive, long EMAs resilient), and contiguous blocks trim nothing.

Splits need **no io work**: a block always fits in RAM (its event arrays are far smaller than
the `(rows × F)` output matrix), so the loader fetches the whole block and the engine slices
it in memory and computes only the sub-range. Splits therefore exist purely for *compute*
memory — a big block's `~2.2×output` peak can exceed a small `mem_budget` even when the block
loads fine — and work today with whole-block loading.

Remaining / refinements (consumer-side, not generation): read-time memmap/streaming for very
large ranges (§6.1). (The gap-aware warmup trim — cold origin *and* mid-dataset gaps — is
done, §4.1; the one approximation left is that event-clock spans use ms as a proxy, so a
sparse venue could under-trim there.)
