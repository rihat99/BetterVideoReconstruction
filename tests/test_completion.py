from __future__ import annotations

import numpy as np
import torch

from tools.human_optim.completion import (
    COMPLETION_DEFAULTS,
    DynamicsContext,
    complete_person,
)

# complete_person's only heavy dependency is the static-support force solve
# (SMPL-X model + RNEA); every test here stubs solve_static_support so the
# gate/force/union bookkeeping can be exercised on tiny synthetic inputs with no
# GPU, model, or scene data. The catalog only needs the six group representatives.
_JOINT_NAMES = [
    "pelvis", "left_wrist", "right_wrist", "left_foot", "right_foot",
    "left_ankle", "right_ankle",
]
_GROUPS = ("left_hand", "right_hand", "left_foot", "right_foot", "left_ankle", "right_ankle")
_N = 6                                    # frames
_J = len(_JOINT_NAMES)


def _ctx() -> DynamicsContext:
    # weights/dyn_opt/friction are only read inside the (stubbed) solve.
    return DynamicsContext(groups=_GROUPS, weights={}, dyn_opt={}, friction=None)


def _make_inputs() -> dict:
    """Neutral inputs: nothing detected, every joint far, every joint still, all valid."""
    return {
        "detector": np.zeros((_N, _J), bool),
        "dist": np.full((_N, _J), 100.0, np.float32),   # cm — far ⇒ gate off
        "drift": np.zeros((_N, _J), np.float32),        # m — still
        "valid": np.ones(_N, bool),
    }


def _run(monkeypatch, solve, inp: dict) -> dict:
    monkeypatch.setattr("tools.human_optim.completion.solve_static_support", solve)
    return complete_person(
        object(), torch.zeros(1), torch.zeros(1), inp["valid"], _JOINT_NAMES,
        detector_contact=inp["detector"], joint_distance_cm=inp["dist"],
        gate_drift=inp["drift"], gravity_world=np.array([0.0, 1.0, 0.0], np.float32),
        fps=30.0, cone_pts=None, cone_normals_t=None,
        ctx=_ctx(), knobs=dict(COMPLETION_DEFAULTS), device="cpu")


def _no_force_solve():
    """Stub solve: a plausible body mass but zero force everywhere (force never fires)."""
    def solve(mid, betas, q, valid, gravity_world, fps, *, group_dist_cm, groups, **_):
        return np.zeros((len(valid), len(groups), 3), np.float32), 70.0, 1.0, 0.5
    return solve


def _force_on(col: int, magnitude: float, *, mass: float):
    """Stub solve: allocate ``magnitude`` newtons to one group's contact point."""
    def solve(mid, betas, q, valid, gravity_world, fps, *, group_dist_cm, groups, **_):
        forces = np.zeros((len(valid), len(groups), 3), np.float32)
        forces[:, col, 2] = magnitude                   # |f| = magnitude along +z
        return forces, float(mass), 1.0, 0.5
    return solve


def _union(comp: dict, detector: np.ndarray) -> np.ndarray:
    """Reconstruct the stage's completed labels (detector ∪ gate ∪ force spans)."""
    completed = detector.copy()
    for c, j in enumerate(comp["rep_idx"]):
        completed[:, j] |= comp["gate_span"][:, c] | comp["force_span"][:, c]
    return completed


def _source_bitmask(comp: dict, detector: np.ndarray) -> np.ndarray:
    """Reconstruct the stage's contact_source bitmask (1 detector / 2 gate / 4 force)."""
    src = np.where(detector, np.int8(1), np.int8(0))
    for c, j in enumerate(comp["rep_idx"]):
        src[comp["gate_span"][:, c], j] |= 2
        src[comp["force_span"][:, c], j] |= 4
    return src


def test_completion_is_add_only(monkeypatch):
    inp = _make_inputs()
    inp["detector"][0:3, 0] = True                       # a detector contact on a non-rep joint
    inp["dist"][:, 1] = 10.0                             # left hand near …
    inp["drift"][:, 1] = 0.01                            # … and still ⇒ gate fires
    comp = _run(monkeypatch, _no_force_solve(), inp)
    completed = _union(comp, inp["detector"])
    assert completed[inp["detector"]].all()              # every detector label survives (superset)
    assert completed.sum() > inp["detector"].sum()       # and completion did add something


def test_completion_additions_only_on_representative_joints(monkeypatch):
    inp = _make_inputs()
    inp["dist"][:, 1] = 10.0
    inp["drift"][:, 1] = 0.01                            # left-hand gate
    comp = _run(monkeypatch, _force_on(2, 100.0, mass=70.0), inp)   # left-foot force
    completed = _union(comp, inp["detector"])
    additions = completed & ~inp["detector"]
    rep = list(comp["rep_idx"])
    non_rep = [j for j in range(_J) if j not in rep]
    assert not additions[:, non_rep].any()               # nothing lands off the six group joints
    assert additions[:, rep[0]].any()                    # left-hand gate added on the wrist
    assert additions[:, rep[2]].any()                    # left-foot force added on the foot


def test_invalid_frames_get_no_labels_and_zero_force(monkeypatch):
    inp = _make_inputs()
    inp["dist"][:, 1] = 10.0
    inp["drift"][:, 1] = 0.01                            # gate would otherwise fire every frame
    inp["valid"][_N - 1] = False
    comp = _run(monkeypatch, _force_on(2, 100.0, mass=70.0), inp)
    assert not comp["gate_span"][_N - 1].any()
    assert not comp["force_span"][_N - 1].any()
    assert np.all(comp["limb_force_n"][_N - 1] == 0.0)


def test_solve_failure_falls_back_to_gates_only(monkeypatch):
    inp = _make_inputs()
    inp["dist"][:, 1] = 10.0
    inp["drift"][:, 1] = 0.01                            # gate fires

    def boom(*_a, **_k):
        raise FloatingPointError("non-finite forces")

    comp = _run(monkeypatch, boom, inp)
    assert not np.isfinite(comp["body_weight_n"])        # inf sentinel ⇒ force can never fire
    assert not comp["force_span"].any()
    assert np.all(comp["limb_force_n"] == 0.0)
    assert comp["gate_span"][:, 0].any()                 # gate spans still valid after the fallback


def test_contact_source_bitmask_matches_labels(monkeypatch):
    inp = _make_inputs()
    inp["detector"][0:3, 0] = True                       # detector on a non-rep joint
    inp["dist"][:, 1] = 10.0
    inp["drift"][:, 1] = 0.01                            # left-hand gate
    comp = _run(monkeypatch, _force_on(2, 100.0, mass=70.0), inp)   # left-foot force
    completed = _union(comp, inp["detector"])
    src = _source_bitmask(comp, inp["detector"])
    assert np.array_equal(completed, src != 0)           # the stage's own label-provenance invariant
    assert np.all((src & 1).astype(bool)[inp["detector"]])   # detector bit set where the detector fired
