"""Per-body mass / COM / inertia for the 52-joint SMPL-X-mid body.

The SMPL-X dynamics stage (``scripts/stages/optimize_dynamics_smplx.py``) needs a
BetterRobot ``Model`` carrying *real* inertial tensors so RNEA produces meaningful
joint torques and the contact-force balance is physical. SMPL-X-mid ships **no
part segmentation** (52 joints, 10475 verts). What
:class:`~better_human.smpl.SMPLXMid` *does* expose is its skinning matrix
``lbs_weights (V, 52)`` and ``forward_shape``,
so we derive the per-body geometry the only way the model itself defines it:

  * **vertex → joint** by ``lbs_weights.argmax`` (each vertex assigned to the joint
    that skins it most — every SMPL-X-mid joint owns ≥21 verts, so no part is empty),
  * each part's **convex hull** (a watertight solid via ``trimesh``) gives a closed
    mesh whose ``mass`` / ``center_mass`` / ``moment_inertia`` we read directly.

Convex hulls slightly over-count volume in concave regions and overlap a little
between neighbours, so the **distribution** is anatomically reasonable but the
absolute scale is not trusted — we **renormalize all part masses to a target total**
(:data:`DEFAULT_TOTAL_MASS_KG`, ~an adult climber) and scale the inertias by the
same factor (inertia ∝ mass at fixed geometry). The global body ``scale`` the
kinematic stage fit is folded in by scaling the rest mesh + joints about the pelvis
*before* hull fitting, so mass ∝ scale³ and inertia ∝ scale⁵ come out automatically
and the model is metrically consistent with the posed (scaled) body.

Conventions match ``better_robot.io.builders``::

    masses[j]   : body j mass (kg)
    coms[j]     : offset joint-origin → COM (m), body-frame axes (≡ rest-mesh axes)
    inertias[j] : 3×3 symmetric rotational inertia **at the COM** (kg·m²)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from .model import _parents_from_model

if TYPE_CHECKING:
    from better_human.smpl import SMPLXMid
    from better_robot.data_model.model import Model

#: Total body mass (kg) all per-part masses are renormalized to. The convex-hull
#: distribution is kept; only the absolute scale is pinned here.
DEFAULT_TOTAL_MASS_KG = 70.0

#: Arbitrary working density (kg/m³); cancels under the total-mass renormalization.
_WORKING_DENSITY = 1000.0


def compute_smplx_body_inertia(
    vertices: torch.Tensor,
    joints: torch.Tensor,
    lbs_weights: torch.Tensor,
    *,
    total_mass_kg: float = DEFAULT_TOTAL_MASS_KG,
    density: float = _WORKING_DENSITY,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-body ``(masses (J,), coms (J, 3), inertias (J, 3, 3))`` for a posed shape.

    Parameters
    ----------
    vertices
        ``(V, 3)`` rest-pose vertex positions (already shape- and scale-adjusted).
    joints
        ``(J, 3)`` rest-pose joint positions in the same frame as ``vertices``.
    lbs_weights
        ``(V, J)`` skinning matrix; ``argmax`` over ``J`` assigns each vertex to a body.
    total_mass_kg
        Sum the returned masses are renormalized to.
    density
        Working density for the hulls; cancels in the renormalization.
    """
    import trimesh

    V, J = lbs_weights.shape
    v2j = lbs_weights.argmax(dim=1).cpu().numpy()
    verts_np = vertices.detach().cpu().numpy().astype(np.float64)
    joints_np = joints.detach().cpu().numpy().astype(np.float64)

    masses = torch.zeros(J, dtype=dtype)
    coms = torch.zeros(J, 3, dtype=dtype)
    inertias = torch.zeros(J, 3, 3, dtype=dtype)

    for j in range(J):
        pts = verts_np[v2j == j]
        m = c = inert = None
        if pts.shape[0] >= 4:
            try:
                hull = trimesh.Trimesh(vertices=pts, process=False).convex_hull
                if float(hull.volume) > 1e-9:
                    hull.density = density
                    m = float(hull.mass)
                    c = np.asarray(hull.center_mass, dtype=np.float64)
                    inert = np.asarray(hull.moment_inertia, dtype=np.float64)
            except Exception:
                m = None
        if m is None:  # degenerate part (slivered hull / too few verts): point-mass sphere
            centroid = pts.mean(0) if pts.shape[0] else joints_np[j]
            r = 0.02
            m = density * (4.0 / 3.0) * np.pi * r**3
            c = centroid
            inert = (2.0 / 5.0) * m * r**2 * np.eye(3)

        masses[j] = m
        coms[j] = torch.from_numpy((c - joints_np[j]).astype(np.float32)).to(dtype)
        inertias[j] = torch.from_numpy(inert.astype(np.float32)).to(dtype)

    # Renormalize the absolute scale to the target total; keep the distribution
    # and the geometry (COMs unchanged, inertia ∝ mass at fixed shape).
    raw_total = float(masses.sum())
    factor = (float(total_mass_kg) / raw_total) if raw_total > 1e-6 else 1.0
    masses = masses * factor
    inertias = inertias * factor
    return masses, coms, inertias


def make_smplx_inertial_model(
    smplx_mid: "SMPLXMid",
    betas: torch.Tensor,
    scale: float = 1.0,
    *,
    total_mass_kg: float = DEFAULT_TOTAL_MASS_KG,
    density: float = _WORKING_DENSITY,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> "Model":
    """Build a BetterRobot ``Model`` for the SMPL-X-mid body **with real inertia**.

    The kinematic counterpart :func:`tools.smplx_robot.make_smplx_robot_model` uses
    ``mass_per_body=1.0`` (it only measures smoothness); this one carries the
    convex-hull inertia so RNEA torques and the contact-force balance are physical.

    The body's global ``scale`` (about the pelvis) is applied to the rest mesh +
    joints before fitting, so the bone offsets, masses and inertias are all scaled
    consistently and ``forward_kinematics(model, remap_q_bh_to_model(q))`` reproduces
    the scaled, posed kinematic body.
    """
    from better_robot.io.builders import build_kinematic_tree_model

    betas_b = betas.view(1, -1).to(
        dtype=smplx_mid.vertices_template.dtype,
        device=smplx_mid.vertices_template.device,
    )
    with torch.no_grad():
        verts, joints = smplx_mid.forward_shape(betas_b)  # (1, V, 3), (1, J, 3)
    verts, joints = verts[0], joints[0]

    s = float(scale)
    if s != 1.0:  # uniform scale about the pelvis (joint 0), same as the posed body
        root = joints[:1].clone()
        verts = root + s * (verts - root)
        joints = root + s * (joints - root)

    masses, coms, inertias = compute_smplx_body_inertia(
        verts, joints, smplx_mid.lbs_weights,
        total_mass_kg=total_mass_kg, density=density, dtype=dtype,
    )

    parents = _parents_from_model(smplx_mid)
    offsets = torch.zeros_like(joints)
    for child in range(1, smplx_mid.num_joints):
        offsets[child] = joints[child] - joints[parents[child]]

    return build_kinematic_tree_model(
        name="smplx_mid_inertial",
        joint_names=tuple(smplx_mid.joint_names),
        parents=tuple(parents),
        translations=offsets.to(dtype=dtype),
        root_kind="free_flyer",
        child_kind="spherical",
        mass_per_body=masses.tolist(),
        com_per_body=coms.to(dtype=dtype),
        inertia_per_body=inertias.to(dtype=dtype),
        device=device,
        dtype=dtype,
    )
