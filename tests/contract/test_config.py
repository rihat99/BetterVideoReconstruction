"""The config loader refuses what would otherwise run with a default: a misspelled key above all."""
from __future__ import annotations

from pathlib import Path

import pytest

from bvr import config
from bvr.config import ConfigError
from bvr.run import STEPS


def delta(tmp_path: Path, text: str, name: str = "delta.toml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_a_delta_changes_only_what_it_names(tmp_path: Path) -> None:
    base = config.load()
    merged = config.load(delta(tmp_path, 'base = "configs/bvr.toml"\n[chamfer]\nsat_m = 0.5\n'))
    assert merged["chamfer"]["sat_m"] == 0.5
    merged["chamfer"]["sat_m"] = base["chamfer"]["sat_m"]
    assert merged == base


@pytest.mark.parametrize(("text", "message"), [
    ("[chamfer]\nsat_n = 0.5\n", "no key chamfer.sat_n; the nearest is chamfer.sat_m"),
    ("[chamfre]\nsat_m = 0.5\n", "no table chamfre; the nearest is chamfer"),
    ("[stages.kinematics_1.iterations]\npolsih = 3\n", "no key stages.kinematics_1.iterations.polsih"),
    ("[sam3]\noverlay = 1\n", "sam3.overlay is int, the base has bool"),
    ("[sam3]\ntop_k = 2.5\n", "sam3.top_k is float, the base has int"),
    ("chamfer = 1\n", "chamfer is int, the base has a table"),
    ("[stages.root.weights]\nchamfer = true\n", "a weight is a number"),
])
def test_a_delta_the_base_cannot_take_raises(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        config.load(delta(tmp_path, 'base = "configs/bvr.toml"\n' + text))


def test_a_weight_table_may_name_a_term_the_base_leaves_off(tmp_path: Path) -> None:
    merged = config.load(delta(tmp_path, 'base = "configs/bvr.toml"\n[stages.kinematics_2.weights]\nchamfer = 2.5\n'))
    assert merged["stages"]["kinematics_2"]["weights"]["chamfer"] == 2.5


def test_a_float_key_takes_an_integer(tmp_path: Path) -> None:
    merged = config.load(delta(tmp_path, 'base = "configs/bvr.toml"\n[chamfer]\nsat_m = 1\n'))
    assert merged["chamfer"]["sat_m"] == 1


def test_a_file_without_a_base_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="names no base"):
        config.load(delta(tmp_path, "[chamfer]\nsat_m = 0.5\n"))


def test_a_base_cycle_raises(tmp_path: Path) -> None:
    first = delta(tmp_path, f'base = "{tmp_path / "second.toml"}"\n', "first.toml")
    delta(tmp_path, f'base = "{first}"\n', "second.toml")
    with pytest.raises(ConfigError, match="base cycle"):
        config.load(first)


def test_every_step_reads_tables_the_base_has() -> None:
    base = config.load()
    for step in STEPS:
        assert config.select(base, step.sections), step.name
