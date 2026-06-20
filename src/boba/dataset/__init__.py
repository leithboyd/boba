"""Raw-atom feature dataset pipeline.

Public API re-exported here; internals live in the submodules:
  columns      — template registry + ColumnSpec/args expansion
  raw          — the (N, F) feature builder + block-cached dataset builders
  session_data — SessionData container built from per-listing parquet frames
  costs        — entry/exit cost fields evaluated on the grid
"""
from boba.dataset.columns import (
    TEMPLATES,
    ColumnSpec,
    ColumnUnit,
    ExpandedColumns,
    col,
    expand_columns,
    listing_spans,
)
from boba.dataset.costs import OUTCOME_MS, CostConfig
from boba.dataset.raw import (
    DatasetRawConfig,
    SampleArraysRaw,
    build_block,
    build_dataset,
    build_features_raw,
    feature_names,
)
from boba.dataset.session_data import SessionData, build_session_data, build_target_book

__all__ = [
    "TEMPLATES", "ColumnSpec", "ColumnUnit", "ExpandedColumns", "col",
    "expand_columns", "listing_spans",
    "OUTCOME_MS", "CostConfig",
    "DatasetRawConfig", "SampleArraysRaw",
    "build_block", "build_dataset", "build_features_raw", "feature_names",
    "SessionData", "build_session_data", "build_target_book",
]
