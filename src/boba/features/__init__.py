"""Per-feature implementations for the screening pipeline.

Each feature supplies a `FeatureSpec` (see `base.py`) bundling a vectorized builder and a
streaming factory with a standard interface, then `register()`s it. The generic engines in
`boba.research.screening` (parity, family cache, gates) drive any registered feature with no
per-feature glue.

NOTE: contract stubs — `base.py` defines the interface; concrete features land here once the
shared engines are extracted from `notebooks/features_v2`.
"""
