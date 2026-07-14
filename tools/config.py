"""Small helpers for stage TOML configuration."""
from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace


def load_toml(path: str | Path) -> dict:
    """Load a TOML file into a plain dictionary."""
    with Path(path).open("rb") as config_file:
        return tomllib.load(config_file)


def section(cfg: dict, name: str, defaults: dict) -> SimpleNamespace:
    """Return one config section layered over its default values."""
    values = dict(defaults)
    values.update(cfg.get(name, {}))
    return SimpleNamespace(**values)
