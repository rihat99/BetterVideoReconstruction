"""Stage TOML configuration: the one loader and the human-optimization config.

:func:`load_config` reads any stage TOML. A file may name a ``base`` (a path relative to
the repository root); the base is loaded first, recursively, and the file's own tables are
merged over it key by key — so a dataset config holds only the knobs it changes. Arrays
replace the base's whole (a loss-weight column ``[root, kinematics_1, kinematics_2,
kindyn_1]`` is one value, never merged element by element). ``section`` layers one table
over its defaults.

The human-optimization TOML stores each loss weight as such a length-4 array under
``[loss_weights.*]`` — the single source of truth for every weight;
:func:`load_human_config` flattens them into one ``{term: weight}`` dict per stage
(the shape the losses read), so the runner just picks a stage's dict per
optimization phase. Per-stage ``iters`` and the shared knob sections
(``[optimizer]``, ``[chamfer]``, …) pass through unchanged.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from bvr.paths import REPO_ROOT


def _merge(base: dict, override: dict) -> dict:
    """``override`` laid over ``base``, table by table; any other value replaces whole."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict:
    """Load a TOML file into a plain dictionary, its ``base`` (if any) merged underneath."""
    with Path(path).open("rb") as config_file:
        raw = tomllib.load(config_file)
    base = raw.pop("base", None)
    if base is None:
        return raw
    return _merge(load_config(REPO_ROOT / base), raw)


def section(cfg: dict, name: str, defaults: dict) -> SimpleNamespace:
    """Return one config section layered over its default values."""
    values = dict(defaults)
    values.update(cfg.get(name, {}))
    return SimpleNamespace(**values)


STAGE_ORDER = ("root", "kinematics_1", "kinematics_2", "kindyn_1")

#: Reference frame rate the loss weights are tuned at.
FPS_REF = 30.0

#: Difference order k of every per-frame-difference loss term.
_DIFF_ORDER = {
    "radial_velocity": 1, "ann_velocity": 1,
    "force_smooth_mag": 1, "force_smooth_dir": 1, "torque_smooth": 1,
    "accel_root": 2, "accel_joint": 2,
    "force_accel_mag": 2, "force_accel_dir": 2, "torque_accel": 2,
    "jerk_root": 3, "jerk_joint": 3,
    "force_jerk_mag": 3, "force_jerk_dir": 3,
    "snap_root": 4, "snap_joint": 4,
}


def fps_scaled_weights(weights: dict[str, float], fps: float) -> dict[str, float]:
    """Scale the per-frame-difference weights by ``(fps/30)^2k`` above 30 fps.

    Every difference term folds a constant ``fps^k`` into its weight (k = the
    difference order), so a weight tuned at 30 fps restrains the k-th *physical*
    derivative ``(fps/30)^2k`` times weaker on a higher-fps clip — while the
    per-frame pose noise the terms exist to damp does not shrink at all (measured
    as the kindyn shake on 60 fps scenes). Multiplying by ``(fps/FPS_REF)^2k``
    keeps one weight meaning one thing at every frame rate. Clamped to identity at
    ``fps <= 30`` so every clip at or below the tuning reference keeps its exact
    shipped behaviour.
    """
    s = max(float(fps), FPS_REF) / FPS_REF
    if s == 1.0:
        return dict(weights)
    return {term: w * s ** (2 * _DIFF_ORDER[term]) if term in _DIFF_ORDER else w
            for term, w in weights.items()}


@dataclass(frozen=True)
class HumanOptimConfig:
    run: tuple[str, ...]                    # default stage chain
    weights: dict[str, dict[str, float]]    # stage -> {loss term: weight}
    iters: dict[str, dict[str, int]]        # stage -> {phase: iters}
    knobs: dict[str, dict]                  # shared sections: optimizer, chamfer, penetration, …
    stage_opts: dict[str, dict]             # per-stage extras, e.g. {"kindyn_1": {"bodies": "kinematic_1"}}
    device: str                             # default device for the kinematic stages

    def weights_for(self, stage: str) -> dict[str, float]:
        """The ``{loss term: weight}`` column for ``stage``."""
        return self.weights[stage]


def load_human_config(path: str | Path) -> HumanOptimConfig:
    """Load a human-optimization TOML into a :class:`HumanOptimConfig`."""
    raw = load_config(path)

    weights: dict[str, dict[str, float]] = {s: {} for s in STAGE_ORDER}
    for group, table in raw.get("loss_weights", {}).items():
        for term, arr in table.items():
            if not isinstance(arr, list) or len(arr) != len(STAGE_ORDER):
                raise ValueError(
                    f"[loss_weights.{group}].{term} must be a {len(STAGE_ORDER)}-element array "
                    f"[{', '.join(STAGE_ORDER)}]; got {arr!r}")
            for stage, value in zip(STAGE_ORDER, arr):
                weights[stage][term] = float(value)

    stages = raw.get("stages", {})
    iters = {s: {k: int(v) for k, v in stages.get(s, {}).get("iters", {}).items()}
             for s in STAGE_ORDER}
    stage_opts = {s: {k: v for k, v in stages.get(s, {}).items() if k != "iters"}
                  for s in STAGE_ORDER}

    knob_sections = ("optimizer", "chamfer", "keypoints", "reproj_init", "penetration",
                     "occlusion", "contact_gate", "auto_contact", "ann_attraction",
                     "ann_velocity", "ann_acquisition", "contacts", "dynamics_optimizer",
                     "friction", "torque")
    knobs = {name: dict(raw[name]) for name in knob_sections if name in raw}

    return HumanOptimConfig(
        run=tuple(stages.get("run", ["root", "kinematics_1"])),
        weights=weights,
        iters=iters,
        knobs=knobs,
        stage_opts=stage_opts,
        device=raw.get("runtime", {}).get("device", "cuda"),
    )
