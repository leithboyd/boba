# notebooks

Scratch space for investigating the raw parquet data and running experiments.

**Convention:** read the raw streams directly through `boba.io` (`list_blocks`, `load_block`)
— do **not** reach for the dataset / feature-builder classes here. These notebooks are for
looking at the data as it lands on disk, not at derived features.

## Running

The `boba` package is installed editable in the pixi env, so imports work without any path
setup. Jupyter + matplotlib live in a dedicated `notebook` environment (kept out of the
default test/CI env):

```sh
pixi install -e notebook    # one-time
pixi run -e notebook lab    # launch JupyterLab
```

`data_dir` must be set in `settings.local.toml` to load real blocks.

## Notebooks

- `01_explore_raw.ipynb` — load BBO / trades / funding for one block and eyeball them.
