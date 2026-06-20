# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

This project uses [pixi](https://pixi.sh) for environment and dependency management (Python 3.12, conda-forge channel). Dependencies live in `pixi.toml`; package metadata lives in `pyproject.toml` (hatchling build backend). The `boba` package is installed editable into the pixi environment, so imports work without setting PYTHONPATH.

## Commands

- `pixi install` — create/update the environment
- `pixi run test` — run the test suite (pytest)
- `pixi run pytest tests/test_dataset_raw.py::TestAlpha -q` — run a single test class/test
- `pixi add <package>` — add a conda-forge dependency
- `pixi run python ...` — run anything inside the environment

## Dependency-age policy (≥ 60 days)

`[workspace] exclude-newer = "60d"` in `pixi.toml` forbids resolving any conda
package build younger than ~2 months. It's a *rolling* cutoff recomputed from
"now" at every solve (`pixi lock` / `pixi update` / `pixi add`) and enforced
natively by pixi against each package's conda-build timestamp — soak time that
keeps freshly-published, not-yet-shaken-out releases (and a wider supply-chain
window) out of the lock. Older-than-60d is always fine.

Consequence: `pixi add`/`update` will pick the newest build that is already ≥60
days old, not the latest. To intentionally adopt something newer, lower/remove
`exclude-newer` (or pin the package) and re-lock. Applies to conda only; the lone
PyPI entry is the editable local `boba` (a path dep, exempt). Add the PyPI-side
equivalent if a real PyPI dependency is ever introduced.

## What this project does

ETL for ML over crypto market microstructure data: builds a raw-atom feature matrix
(N rows × F float32 features on a 1 ms grid) from per-listing BBO + trade parquet
streams across bin/byb/okx, spot + perp. The feature templates are specified in
`docs/raw_features.md` — each stored column is a single primitive transformation
("raw atom"); compositions (Z-scores, variances, deviations) are derived downstream.

## Architecture

- `src/boba/settings.py` — loads `settings.toml` overlaid by `settings.local.toml`
  (gitignored, per-machine). `data_dir` must be set locally to load real data;
  the tests are fully synthetic and need no data or settings.
- `src/boba/io.py` — block/listing parquet loaders. Data is organised in ~24h
  "blocks" (`holocron.{ts}.{idx}` filename prefix) per listing token
  (e.g. `bin_doge_usdt_p`). `DATA_DIR` is `None` when unconfigured; loaders raise.
- `src/boba/dataset/` — the dataset package; public API re-exported from its
  `__init__` (`from boba.dataset import DatasetRawConfig, col, build_dataset, …`):
  - `session_data.py` — `SessionData`: per-listing numpy event arrays (BBO,
    trades, feed latency) plus the target listing's book; built from polars
    frames by `build_session_data`.
  - `costs.py` — entry/exit cost fields evaluated on the grid (book state at
    order-landing time and at the outcome horizon, outcome-window trade extremes).
  - `columns.py` — the column-spec API. `TEMPLATES` is the registry (the
    template strings in docs/raw_features.md); `ColumnSpec(template, args)` binds
    placeholders (list values fan out as a cross product staged in placeholder
    order); `col()` is single-mapping sugar; `expand_columns` validates and yields
    concrete `ColumnUnit`s the builder dispatches on.
  - `raw.py` — the feature builder. `DatasetRawConfig` REQUIRES `columns` (an
    ordered ColumnSpec tuple — the only way to select features; any positive N
    is valid per template) plus `listings`/`target_listing` and value knobs
    (`wide_threshold`, `microprice_ref` — keyed by listing token, with fallbacks
    for absent listings). Anything output-affecting must be folded into
    `config_str()`, which hashes the ORDERED expanded names — permuted
    selections are distinct datasets. `build_features_raw` is the core (N, F)
    builder; `build_block` / `build_dataset` wrap it with per-block npz caching
    under `artifacts/`. `_compute_per_exchange` is the slow single-call reference
    path that tests check the parallel production path against.

Key invariants the tests enforce: features are causal (each row uses only data
at-or-before its grid tick), EMAs follow the `alpha = 2/(span+1)` convention with
`y[-1] = 0` initial condition, column order is exactly the expansion order of
`cfg.columns`, and a subset selection must produce bit-identical values to the
same columns of a larger build. `tests/helpers.py` rebuilds the legacy full
catalogue as specs (`make_cfg`) — byte-identical names/order to the old repo.

## Validation requirement — every field, vs a blind simple oracle, on a real block

**Hard requirement for every feature column in this dataset:** its value must be
validated against an *independent, dead-simple reference implementation* ("oracle"),
and the comparison must be run on a **real data block** (not only synthetic) — real
blocks have arbitrary-nanosecond event timing that synthetic exact-ms fixtures cannot
exercise (this is exactly what caught the ms-vs-ns causal forward-fill bug in the
trade-clock features).

The point is that the **complex, optimized production code is validated against very
simple code**. To keep the oracle a genuine check and not a copy of the same bugs:

- The oracle must be **dead simple** — direct array ops / explicit loops, single-threaded,
  **no shared code** with `build_features_raw`. Plain `numpy`, no production helpers.
- The oracle should be implementable from a **written description of the feature alone**,
  by someone (or an agent) who has **not seen** the production implementation. If a fresh
  agent given only the column's spec would write the same oracle, it's independent enough.
- Production (parallel, chunked, cached) output must match the oracle to float32 tolerance
  on the real block.

See `tests/test_dataset_v2_volclock.py` for the pattern: `vol_ref` / `level_ema_ref` /
`flow_ema_ref` are the blind oracles; `TestRealBlock` runs the diff on a real ETH-perp block
(skipped when `DATA_DIR` is unset). Any new column lands with the same trio: oracle +
synthetic property tests + a real-block diff.
