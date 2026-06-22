# Feature-analysis notebook builders

Each `build_<name>.py` generates `notebooks/features/<name>.ipynb` — a self-documenting analysis of one feature against byb's next-100 ms outcome (oracle, hygiene gates, the hard regime-invariance gate, signal-lifetime + per-N half-life sweep, echo-netted partial-IC + feed-resolution gates, per-exchange/cross-venue legs).

- `build_feat_nb.py` is the **template** (worked example: `price_dislocation`); every other builder is cloned from it with its feature swapped in.
- Regenerate a notebook: `pixi run python notebooks/features/builders/build_<name>.py`, then execute it in the `notebook` pixi env (`pixi run -e notebook jupyter nbconvert --to notebook --execute --inplace notebooks/features/<name>.ipynb`).
- The OOS harness in `tools/oss/` re-validates these features across all 58 blocks.
