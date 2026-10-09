"""What the force rigs share in the viewer: measured forces (``sensor_forces.npz``) as arrows.

The peter wall boards (``datasets/peter/sensor_forces.py``), the Parkour plates
(``datasets/parkour/sensor_forces.py``) and the OpenCap plates
(``datasets/opencap/sensor_forces.py``) all write ``sensor_forces.npz`` in the schema
``bvr.stages.measure_contacts`` reads: ``reaction_world (N, C, 3)`` per channel in
``limbs`` order, the reaction sense (force on the person), plus ``cop_world`` where the sensor
knows where the force acts. :func:`measured_forces` draws them beside the solved forces as
:class:`~bvr.viewer.loading.ForceArrows`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from bvr.body import Body
from bvr.viewer.loading import BodySource, ClipData, ForceArrows, fit_frames

#: Measured channels below this many newtons are not drawn (the sensors never read exactly zero).
SENSOR_FLOOR_N = 10.0
_SENSOR_RGB = (1.0, 0.59, 0.12)                                   # measured forces: orange
_LIMB_JOINT = {"left_hand": "left_wrist", "right_hand": "right_wrist",
               "left_foot": "left_foot", "right_foot": "right_foot"}


def load_sensor_forces(pred: Path) -> dict | None:
    """Measured forces (``sensor_forces.npz``) in the world frame, in the reaction sense (force on the person).

    The peter rigs write a wall's load-cell channels (each a corner's reaction, limb-dominated but
    mixed); the climb_wall_3 ground truth keeps a per-limb board reading at the clip root; the force
    plates (Parkour, OpenCap) also know where the force acts (``cop_world``, NaN where the plate saw
    nothing).
    """
    f = next((c for c in (pred / "human_optim" / "sensor_forces.npz", pred / "sensor_forces.npz")
              if c.exists()), None)
    if f is None:
        return None
    dd = np.load(f, allow_pickle=True)
    return {"rw": dd["reaction_world"].astype(np.float32),                     # (N, C, 3) per channel
            "limbs": [str(x) for x in dd["limbs"]],                              # channel -> limb
            "cop": dd["cop_world"].astype(np.float32) if "cop_world" in dd.files else None}


def forces_source(data: ClipData) -> tuple[str, BodySource] | None:
    """The clip's first kinematics+dynamics body source whose first person is drawn, or ``None``."""
    name = next((n for n, src in data.bodies.items() if src.forces is not None), None)
    if name is None or not data.bodies[name].valid[0].any():
        return None
    return name, data.bodies[name]


def measured_forces(pred: Path, data: ClipData, body: Body | None) -> list[ForceArrows]:
    """The clip's measured forces, orange, on the first person of its first kinematics+dynamics source."""
    sensors, found = load_sensor_forces(pred), forces_source(data)
    if sensors is None or found is None or body is None:
        return []
    name, src = found
    with torch.no_grad():
        joints = body.shaped(torch.as_tensor(src.betas[0])).forward(torch.as_tensor(src.q[0])).joints.cpu().numpy()
    origins, vectors, valid = sensor_arrows(sensors, joints, body.joint_names)
    return [ForceArrows("sensor_forces", "sensor forces (measured)", origins, vectors, _SENSOR_RGB,
                        valid & src.valid[0][:, None], source=name)]


def sensor_arrows(sensors: dict, joints: np.ndarray,
                  joint_names) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measured forces as ``(origins, vectors, valid)``, ``(T, C, 3)`` / ``(T, C, 3)`` newtons / ``(T, C)``.

    A plate's arrow starts at its centre of pressure on the floor; a wall cell's at the joint of its
    limb on the body ``joints (T, J, 3)``.
    """
    T = len(joints)
    names = list(joint_names)
    vectors = fit_frames(sensors["rw"], T, axis=0)
    origins = np.zeros_like(vectors)
    for c, limb in enumerate(sensors["limbs"]):
        if _LIMB_JOINT.get(limb) in names:
            origins[:, c] = joints[:, names.index(_LIMB_JOINT[limb])]
    if sensors["cop"] is not None:
        cop = fit_frames(sensors["cop"], T, axis=0)
        on_plate = np.isfinite(cop).all(-1)
        origins[on_plate] = cop[on_plate]
    known = np.array([_LIMB_JOINT.get(limb) in names for limb in sensors["limbs"]])
    valid = (np.linalg.norm(vectors, axis=-1) > SENSOR_FLOOR_N) & known
    return origins, vectors, valid
