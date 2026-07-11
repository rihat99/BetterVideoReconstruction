"""Parse ``configs/human_optimization.toml`` into per-stage weights + shared knobs.

The grouped ``[loss_weights.*]`` tables store each loss weight as a length-4 array
``[root, kinematics_1, kinematics_2, dynamics]`` — the single source of truth for
every weight. :func:`load_config` flattens them into one ``{term: weight}`` dict
per stage (the shape the losses read), so the runner just picks a stage's dict per
optimization phase. Per-stage ``iters`` and the shared operational knob sections
(``[optimizer]``, ``[chamfer]``, …) pass through unchanged.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

STAGE_ORDER = ("root", "kinematics_1", "kinematics_2", "dynamics")


@dataclass(frozen=True)
class HumanOptimConfig:
    run: tuple[str, ...]                    # default stage chain
    weights: dict[str, dict[str, float]]    # stage -> {loss term: weight}
    iters: dict[str, dict[str, int]]        # stage -> {phase: iters}
    knobs: dict[str, dict]                  # shared sections: optimizer, chamfer, penetration, …
    stage_opts: dict[str, dict]             # per-stage extras, e.g. {"dynamics": {"device": "cpu"}}
    device: str                             # default device for the kinematic stages

    def weights_for(self, stage: str) -> dict[str, float]:
        """The ``{loss term: weight}`` column for ``stage``."""
        return self.weights[stage]


def load_config(path: str | Path) -> HumanOptimConfig:
    """Load the single human-optimization TOML into a :class:`HumanOptimConfig`."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)

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
                     "contact_gate", "auto_contact", "ann_attraction",
                     "ann_repulsion", "contacts", "contact", "inertia", "dynamics_optimizer")
    knobs = {name: dict(raw[name]) for name in knob_sections if name in raw}

    return HumanOptimConfig(
        run=tuple(stages.get("run", ["root", "kinematics_1"])),
        weights=weights,
        iters=iters,
        knobs=knobs,
        stage_opts=stage_opts,
        device=raw.get("runtime", {}).get("device", "cuda"),
    )
