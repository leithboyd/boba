"""Project settings loading.

Reads `settings.toml` (committed) overlaid by `settings.local.toml` (gitignored).
Both live at the project root.
"""
import tomllib
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[2]


def _load() -> dict:
    settings: dict = {}
    for fname in ("settings.toml", "settings.local.toml"):
        path = _PROJECT_ROOT / fname
        if path.exists():
            with path.open("rb") as f:
                settings.update(tomllib.load(f))
    return settings


SETTINGS      = _load()
PROJECT_ROOT  = _PROJECT_ROOT
