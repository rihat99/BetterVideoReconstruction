"""Stage-2 (dynamics) helpers: inertial model rebuild + v / a from q(t).

Stage 2 (``scripts/optimize_dynamics.py``) freezes the kinematic trajectory
from Stage 1 and solves for per-contact forces via RNEA. Two pieces of
shared logic live here:

* :func:`rebuild_inertial_model` — reconstruct the BetterRobot model from a
  motion ``.npz`` *with* real per-body masses / COMs / inertias. Stage 1
  uses uniform ``mass = 1.0`` because kinematic residuals don't need mass;
  for dynamics we have to round-trip the actual inertial tensors.
* :func:`velocity_acceleration_from_trajectory` — tangent-space central
  finite differences of ``q(t)`` to produce generalised velocity and
  acceleration arrays suitable for feeding to RNEA. Mirrors the
  ``AccelerationResidual`` math (see
  ``better_robot.residuals.acceleration``).
"""
from __future__ import annotations

import numpy as np
import torch


def rebuild_inertial_model(
    npz_data: dict,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
):
    """Build a BR model from a motion ``.npz`` using real inertial tensors.

    Parameters
    ----------
    npz_data
        Dict-like view over a motion ``.npz`` (keys as in
        ``docs/motion_npz_schema.md``). Must contain at least ``joint_names``,
        ``parents``, ``joint_offsets``, ``mass_per_body``. ``com_per_body``
        and ``inertia_per_body`` are optional — when missing the function
        falls back to recomputing them from ``betas_mean`` via
        :func:`compute_smpl_body_inertia`.
    dtype, device
        Passed through to the builder.
    """
    from better_robot.io.builders import JOINT_NAMES, build_kinematic_tree_model

    saved_names = tuple(str(x) for x in npz_data["joint_names"])
    saved_parents = tuple(int(p) for p in np.asarray(npz_data["parents"]))
    if set(saved_names) != set(JOINT_NAMES):
        raise ValueError(
            "saved joint_names do not form the SMPL-24 body set "
            f"(expected {sorted(set(JOINT_NAMES))}, got {sorted(set(saved_names))})"
        )
    offsets = torch.from_numpy(np.asarray(npz_data["joint_offsets"])).to(
        dtype=dtype, device=device
    )
    mass = np.asarray(npz_data["mass_per_body"], dtype=np.float32)

    if "com_per_body" in npz_data and "inertia_per_body" in npz_data:
        coms = torch.from_numpy(
            np.asarray(npz_data["com_per_body"], dtype=np.float32)
        ).to(dtype=dtype, device=device)
        inertias = torch.from_numpy(
            np.asarray(npz_data["inertia_per_body"], dtype=np.float32)
        ).to(dtype=dtype, device=device)
    else:
        # Legacy .npz from before the Stage 0 augment: recompute from betas.
        coms, inertias = _recompute_coms_inertias_from_betas(
            npz_data, saved_names, dtype=dtype, device=device
        )

    return build_kinematic_tree_model(
        name="smpl_motion_body",
        joint_names=saved_names,
        parents=saved_parents,
        translations=offsets,
        root_kind="free_flyer",
        child_kind="spherical",
        mass_per_body=mass.tolist(),
        com_per_body=coms,
        inertia_per_body=inertias,
        device=device,
        dtype=dtype,
    )


def _recompute_coms_inertias_from_betas(
    npz_data: dict,
    dfs_names: tuple[str, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Regenerate DFS-ordered COM / inertia tensors from ``betas_mean``.

    Used only when the motion ``.npz`` predates the Stage 0 augment that
    persists ``com_per_body`` / ``inertia_per_body``.
    """
    if "betas_mean" not in npz_data:
        raise KeyError(
            "motion .npz has neither com_per_body/inertia_per_body nor betas_mean "
            "— rerun build_robot_motion to regenerate the file"
        )
    from pathlib import Path

    from better_human.smpl.model.smpl import SMPL
    from better_robot.io.builders import JOINT_NAMES

    from .smpl_inertia import compute_smpl_body_inertia

    default_model = (
        Path(__file__).resolve().parents[3]
        / "better_human"
        / "models"
        / "smpl"
        / "SMPL_NEUTRAL.npz"
    )
    smpl = SMPL(str(default_model), gender="neutral", num_betas=10).to(device)
    smpl.eval()
    betas = torch.from_numpy(np.asarray(npz_data["betas_mean"], dtype=np.float32)).to(
        dtype=dtype, device=device
    )
    with torch.no_grad():
        verts, joints = smpl.forward_shape(betas.unsqueeze(0))
    _, coms_smpl, inertias_smpl = compute_smpl_body_inertia(
        verts[0].cpu(), joints[0].cpu(), density_system="Custom"
    )
    smpl_idx = {name: k for k, name in enumerate(JOINT_NAMES)}
    dfs_from_smpl = [smpl_idx[n] for n in dfs_names]
    coms = coms_smpl.to(dtype=dtype, device=device)[dfs_from_smpl]
    inertias = inertias_smpl.to(dtype=dtype, device=device)[dfs_from_smpl]
    return coms, inertias


def velocity_acceleration_from_trajectory(
    model,
    q: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Central-difference generalised velocity and acceleration from ``q(t)``.

    Parameters
    ----------
    model
        BR ``Model`` — provides ``difference`` (tangent subtraction) and ``nv``.
    q
        ``(T, nq)`` configuration trajectory.
    dt
        Frame interval, seconds.

    Returns
    -------
    v : ``(T, nv)`` tangent-space velocity. ``v[0] = v[T-1] = 0``.
    a : ``(T, nv)`` tangent-space acceleration. ``a[0] = a[T-1] = 0``.

    Notes
    -----
    Acceleration uses the same finite-difference as
    ``AccelerationResidual``:

        a[t] = (difference(q[t], q[t+1]) − difference(q[t−1], q[t])) / dt²

    for interior ``t``; boundary values are pinned to zero so Stage 2's
    RNEA call is still well-defined at the endpoints (quasi-static).
    """
    T = q.shape[0]
    if T < 3:
        raise ValueError(f"need at least 3 frames for central differences, got T={T}")

    # diff_fwd[t] = difference(q[t], q[t+1]), length T-1
    diff_fwd = model.difference(q[:-1], q[1:])

    v = torch.zeros(T, model.nv, dtype=q.dtype, device=q.device)
    # central velocity on interior frames:
    # v[t] ≈ (diff_back[t] + diff_fwd[t]) / 2 / dt
    #       = difference(q[t-1], q[t+1]) / (2·dt)   (not true on a manifold;
    # using the sum of consecutive tangent steps preserves chain-rule behaviour
    # through model.difference and matches the AccelerationResidual idiom.)
    v[1:-1] = 0.5 * (diff_fwd[:-1] + diff_fwd[1:]) / dt

    a = torch.zeros(T, model.nv, dtype=q.dtype, device=q.device)
    a[1:-1] = (diff_fwd[1:] - diff_fwd[:-1]) / (dt * dt)
    return v, a
