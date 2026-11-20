"""The one configuration loader.

``configs/bvr.toml`` holds every default once. A delta file names ``base = "configs/bvr.toml"``
(or another delta, by its path from the repository root) and lists only what it changes.
:func:`load` lays a delta over its base table by table and raises :class:`ConfigError`, before
any compute, for a key or a table the base does not have, a value of another type than the
base's, a file without a base, and a base cycle. Code reads ``config["chamfer"]["sat_m"]``: no
default lives outside the base file.

A table named ``weights`` is the one place where a delta may add a key: a term that a stage's
table does not name is off in that stage, and a name that is no term of the stage raises when
its problem is built.
"""
from __future__ import annotations

import difflib
import tomllib
from collections.abc import Iterable
from pathlib import Path

from bvr.paths import REPO_ROOT

#: The file every config chain ends at.
BASE = REPO_ROOT / "configs" / "bvr.toml"


class ConfigError(Exception):
    """A config file that cannot be laid over the defaults."""


def load(path: str | Path | None = None) -> dict:
    """Return the resolved configuration: the defaults with the delta at ``path`` laid over them.

    :param path: a delta file; ``None`` gives the defaults alone.
    """
    return _load(BASE if path is None else Path(path), ())


def select(config: dict, sections: Iterable[str]) -> dict:
    """Return the tables of ``config`` named by dotted paths, nested as they are in ``config``.

    ``select(config, ["chamfer", "stages.root"])`` is
    ``{"chamfer": {...}, "stages": {"root": {...}}}``: what one stage reads, and nothing else.
    """
    selected: dict = {}
    for section in sections:
        *parents, last = section.split(".")
        source, target = config, selected
        for name in parents:
            source, target = source[name], target.setdefault(name, {})
        target[last] = source[last]
    return selected


def _load(path: Path, chain: tuple[Path, ...]) -> dict:
    path = path.resolve()
    if path in chain:
        raise ConfigError("config base cycle: " + " -> ".join(str(p) for p in (*chain, path)))
    try:
        with path.open("rb") as file:
            tables = tomllib.load(file)
    except FileNotFoundError:
        raise ConfigError(f"no such config file: {path}") from None
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path}: {error}") from error
    base = tables.pop("base", None)
    if path == BASE.resolve():
        return tables
    if not isinstance(base, str):
        raise ConfigError(f'{path} names no base; a delta starts with base = "configs/bvr.toml"')
    return _merge(_load(REPO_ROOT / base, (*chain, path)), tables, path, ())


def _merge(base: dict, delta: dict, source: Path, table: tuple[str, ...]) -> dict:
    """``delta`` laid over ``base``: tables merge key by key, any other value replaces whole."""
    merged = dict(base)
    for key, value in delta.items():
        name = ".".join((*table, key))
        if table[-1:] == ("weights",):
            if not _same_type(0.0, value):
                raise ConfigError(f"{source}: {name} is {_kind(value)}, a weight is a number")
            merged[key] = value
        elif key not in base:
            near = difflib.get_close_matches(key, base, n=1)
            hint = f"; the nearest is {'.'.join((*table, near[0]))}" if near else ""
            raise ConfigError(f"{source}: the base has no {'table' if isinstance(value, dict) else 'key'} {name}{hint}")
        elif not _same_type(base[key], value):
            raise ConfigError(f"{source}: {name} is {_kind(value)}, the base has {_kind(base[key])}")
        else:
            merged[key] = _merge(base[key], value, source, (*table, key)) if isinstance(value, dict) else value
    return merged


def _same_type(base: object, value: object) -> bool:
    """Does ``value`` have the base value's type? An integer may stand where the base has a float."""
    if isinstance(base, float):
        return isinstance(value, int | float) and not isinstance(value, bool)
    return type(value) is type(base)


def _kind(value: object) -> str:
    return "a table" if isinstance(value, dict) else type(value).__name__
