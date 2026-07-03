"""Mass / inertia estimation for SMPL body parts.

Given a SMPL T-pose mesh (6890 vertices) and the 24 SMPL joint positions, this
module returns per-body ``(mass, com, inertia)`` tuples suitable for feeding
into ``build_kinematic_tree_model`` (or the higher-level ``make_smpl_like_model``)
via the generic ``mass_per_body`` / ``com_per_body`` / ``inertia_per_body``
kwargs. The face segmentation and density tables ship as data files under
``tools/robot_motion/data/``.

Conventions (matching ``better_robot.spatial.inertia``)::

    masses[j]  : body j mass in kg.
    coms[j]    : offset from body j's joint origin to its COM (m), in body j's
                 link-frame axes. In the SMPL T-pose those axes align with the
                 mesh axes.
    inertias[j]: 3x3 symmetric rotational inertia at the COM (kg·m²), in body
                 j's link-frame axes.

The caller does NOT pre-apply the parallel-axis theorem — BetterRobot's
``Inertia._to_6x6`` does that when unpacking ``Model.body_inertias``.
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import cast

import numpy as np
import torch
import trimesh


_DATA_PKG = "tools.robot_motion.data"
_DENSITIES_FILE = "smpl_body_densities.json"
_SEGMENTATION_FILE = "smpl_3d_segmentation.npy"

_N_SMPL_JOINTS = 24
_N_SMPL_VERTS = 6890


def _load_densities() -> dict[str, dict[int, float]]:
    raw = json.loads(files(_DATA_PKG).joinpath(_DENSITIES_FILE).read_text())
    return {system: {int(k): float(v) for k, v in table.items()} for system, table in raw.items()}


#: ``{system_name -> {joint_index -> density_kg_per_m3}}``. Five systems:
#: ``"Naive"``, ``"Dempster"``, ``"Chandler"``, ``"Clauser"``, ``"Custom"``.
DENSITIES: dict[str, dict[int, float]] = _load_densities()


def list_density_systems() -> tuple[str, ...]:
    """Return the available density-system names."""
    return tuple(DENSITIES.keys())


def load_smpl_face_segmentation() -> list[np.ndarray]:
    """Return the per-joint face-index arrays (length 24) of the SMPL mesh.

    The raw ``.npy`` stores ``body_faces`` as ``dict[int, ndarray]`` keyed by
    SMPL joint index (0..23). This wrapper returns it as a positionally-indexed
    list so callers can iterate with ``for j, faces in enumerate(...)``.
    """
    path = files(_DATA_PKG).joinpath(_SEGMENTATION_FILE)
    with path.open("rb") as f:
        payload = np.load(f, allow_pickle=True).item()
    faces_by_joint = payload["body_faces"]
    if len(faces_by_joint) != _N_SMPL_JOINTS:
        raise RuntimeError(
            f"Expected {_N_SMPL_JOINTS} SMPL body-face groups, "
            f"got {len(faces_by_joint)}"
        )
    return [np.asarray(faces_by_joint[j], dtype=np.int64) for j in range(_N_SMPL_JOINTS)]


def compute_smpl_body_inertia(
    vertices: torch.Tensor,
    joints: torch.Tensor,
    *,
    density_system: str = "Custom",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-body mass, COM offset, and rotational inertia for a SMPL pose.

    Parameters
    ----------
    vertices
        ``(6890, 3)`` SMPL T-pose vertex positions.
    joints
        ``(24, 3)`` SMPL T-pose joint positions (same frame as ``vertices``).
    density_system
        One of :func:`list_density_systems` — defaults to ``"Custom"``.

    Returns
    -------
    masses : ``(24,)``  kg
    coms   : ``(24, 3)`` m, offset from each joint origin to its COM
    inertias : ``(24, 3, 3)`` kg·m², symmetric, **at the COM**

    Notes
    -----
    Single-pose (un-batched). The shipped face segmentation indexes into a
    ``(6890 + 24, 3)`` concatenation of mesh vertices followed by joint
    positions, so this function stacks them internally before slicing per-body
    submeshes.

    Each submesh is built with ``trimesh``, ``fix_normals()`` orients outward
    (required for positive signed volume), density is set from the lookup
    table, and ``mesh.mass`` / ``mesh.moment_inertia`` / ``mesh.center_mass``
    are read directly.
    """
    if density_system not in DENSITIES:
        raise ValueError(
            f"Unknown density system {density_system!r}; "
            f"choose from {list_density_systems()}"
        )
    if vertices.shape != (_N_SMPL_VERTS, 3):
        raise ValueError(
            f"vertices must have shape ({_N_SMPL_VERTS}, 3); got {tuple(vertices.shape)}"
        )
    if joints.shape != (_N_SMPL_JOINTS, 3):
        raise ValueError(
            f"joints must have shape ({_N_SMPL_JOINTS}, 3); got {tuple(joints.shape)}"
        )

    faces_by_joint = load_smpl_face_segmentation()
    density_table = DENSITIES[density_system]

    verts_np = torch.cat([vertices, joints], dim=0).detach().cpu().numpy().astype(np.float64)
    joints_np = joints.detach().cpu().numpy().astype(np.float64)

    masses = torch.zeros(_N_SMPL_JOINTS, dtype=dtype)
    coms = torch.zeros(_N_SMPL_JOINTS, 3, dtype=dtype)
    inertias = torch.zeros(_N_SMPL_JOINTS, 3, 3, dtype=dtype)

    for j in range(_N_SMPL_JOINTS):
        mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_by_joint[j], process=False)
        mesh.fix_normals()
        mesh.density = density_table[j]

        mass = float(mesh.mass)
        com = np.asarray(mesh.center_mass, dtype=np.float64) - joints_np[j]
        inertia = np.asarray(mesh.moment_inertia, dtype=np.float64)

        masses[j] = mass
        coms[j] = torch.from_numpy(com.astype(np.float32)).to(dtype=dtype)
        inertias[j] = torch.from_numpy(inertia.astype(np.float32)).to(dtype=dtype)

    return masses, coms, inertias


def assert_human_mass_in_range(
    masses: torch.Tensor, *, low: float = 40.0, high: float = 120.0
) -> None:
    """Sanity check: assert the summed per-body mass falls inside ``[low, high]`` kg."""
    total = float(cast(torch.Tensor, masses).sum())
    if not (low <= total <= high):
        raise AssertionError(
            f"total human mass {total:.2f} kg outside expected range [{low}, {high}] kg"
        )
