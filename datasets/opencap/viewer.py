"""The OpenCap viewer layer: our solved foot forces in the plates' format (``estimated_forces.npz``).

``export_opencap_mot.py`` collapses each foot's solved forces to one force at one centre of
pressure in the dataset's OpenSim ground frame and keeps the ``R``/``t`` back to our world;
:func:`plate_forces` draws them blue beside the solved (and the plates' measured) arrows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from bvr.viewer.loading import ClipData, ForceArrows, fit_frames
from datasets._rigs import SENSOR_FLOOR_N, forces_source

_PLATE_RGB = (0.16, 0.43, 1.0)                                    # ours in the plates' format: blue


def load_plate_estimate(pred: Path) -> dict | None:
    """Our solved foot forces in the force plates' format (``estimated_forces.npz``), rotated into our world.

    ``export_opencap_mot.py`` collapses each foot's solved forces to one force at one centre
    of pressure in the dataset's OpenSim ground frame; the file carries the ``R``/``t`` back.
    """
    f = pred / "human_optim" / "estimated_forces.npz"
    if not f.exists():
        return None
    dd = np.load(f, allow_pickle=True)
    R = np.asarray(dd["R_world_from_mocap"], np.float64)
    t = np.asarray(dd["t_world_from_mocap"], np.float64)
    return {"rw": (dd["force"].astype(np.float64) @ R.T).astype(np.float32),        # (N, 2, 3)
            "cop": (dd["cop"].astype(np.float64) @ R.T + t).astype(np.float32),     # (N, 2, 3)
            "limbs": [str(x) for x in dd["limbs"]]}


def plate_forces(pred: Path, data: ClipData) -> list[ForceArrows]:
    """Our plate-format forces, blue, on the first person of the clip's first kinematics+dynamics source."""
    plates, found = load_plate_estimate(pred), forces_source(data)
    if plates is None or found is None:
        return []
    name, src = found
    origins, vectors, valid = plate_arrows(plates, data.n_frames)
    return [ForceArrows("plate_forces", "estimated forces (plate format)", origins, vectors, _PLATE_RGB,
                        valid & src.valid[0][:, None], source=name)]


def plate_arrows(plates: dict, n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Our forces in the plates' format at our centre of pressure, as :func:`sensor_arrows`."""
    vectors, cop = fit_frames(plates["rw"], n_frames, axis=0), fit_frames(plates["cop"], n_frames, axis=0)
    valid = (np.linalg.norm(vectors, axis=-1) > SENSOR_FLOOR_N) & np.isfinite(cop).all(-1)
    return np.nan_to_num(cop), vectors, valid
