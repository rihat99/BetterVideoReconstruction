"""Contact forces on BEDLAM2 ground truth: the ``kindyn_1`` force solve at a frozen pose.

BEDLAM2 gives us the motion (mocap, exact) and ``scripts/label_bedlam2.py`` gives us
which contact frames are on the floor. What neither states is the force each contact
carries. ``kindyn_1`` (:mod:`bvr.human_optim.kindyn`) already answers that question
for estimated bodies, in two parts: a force-only solve at the starting pose, then a
joint pose-and-force polish. Here the pose is ground truth, so the polish would only
pull it away from the truth — the first part alone is the whole stage.

That solve is :meth:`~bvr.human_optim.kindyn.DynamicsSpec.warmstart`: an L-BFGS
over the per-contact-frame forces, every force held inside a 45° friction pyramid
about its frame's contact normal, with the ``kindyn_1`` column of
``configs/human_optimization.toml`` as the objective — root wrench residual, force
size and smoothness, strength-weighted joint torques. The one addition is the
``force_edge`` term (the OpenCap work), which penalises the force on the first and
last frame of every contact interval so a load ramps in and out instead of stepping;
its speed scale lets a genuine landing step while a walking heel strike may not.

With the pose frozen the residual is linear in the forces, so the solve converges in
a couple of hundred L-BFGS steps and one scene takes seconds — what a dataset of
eighteen thousand needs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from bvr.config import HumanOptimConfig
from bvr.human_optim.kindyn import build_dynamics_spec
from bvr.human_optim.qmath import normalize_q

FORCE_METHOD = "bedlam2-kindyn-frozen-v1"


def person_forces(body, betas: np.ndarray, q: np.ndarray, contact: np.ndarray, *,
                  gravity_world: np.ndarray, fps: float, device,
                  weights: dict, dyn_opt: dict, torque_table: dict,
                  half_angle_deg: float, edge_speed_m_s: float) -> dict:
    """Solve one person's contact forces at their ground-truth pose.

    :param betas: ``(10,)`` body shape; the RNEA inertia is baked from it.
    :param q: ``(N, 211)`` world-frame trajectory.
    :param contact: ``(N, F)`` bool, which contact frames touch on which frame, in
        ``body.contact_frame_names`` order.
    :returns: the arrays :meth:`DynamicsSpec.extract` packages — ``frame_forces
        (N, F, 3)`` world newtons, per-joint sums, torques and the root residual.
    """
    shape = torch.as_tensor(betas, dtype=torch.float32, device=device)
    q_world = torch.as_tensor(q, dtype=torch.float32, device=device)
    labels = torch.as_tensor(np.where(contact, 1, -1), dtype=torch.int8, device=device)
    spec = build_dynamics_spec(
        body, shape, labels, np.ones(len(q), bool), gravity_world,
        weights=weights, dyn_opt=dyn_opt, fps=fps, device=str(device),
        q_world=q_world, cones=True, half_angle_deg=half_angle_deg,
        torque_table=torque_table, edge_speed_scale=edge_speed_m_s)
    qn = normalize_q(q_world, body.num_joints)
    f_free, gen_u = spec.warmstart(qn)
    return spec.extract(qn, f_free, gen_u, body)


def scene_forces(gt_dir: Path, *, body, device, config: HumanOptimConfig,
                 weight_overrides: dict | None = None, edge_speed_m_s: float = 0.0) -> dict:
    """Solve every person in one BEDLAM2 scene.

    :param gt_dir: the scene's ``features/gt/<aa>/<bb>/<scene_id>`` directory, holding
        ``smplx.npz``, ``gravity.npz`` and the ``contacts.npz`` the labels step wrote.
    :param config: the human-optimization TOML; its ``kindyn_1`` dynamics weights,
        ``[torque.multipliers]``, ``[friction]`` and ``[dynamics_optimizer]`` are used.
    :param weight_overrides: ``{loss term: weight}`` laid over the ``kindyn_1`` column.
    :returns: the arrays :func:`save_forces` stores, keyed by npz field name.
    """
    meta = np.load(gt_dir / "smplx.npz")
    grav = np.load(gt_dir / "gravity.npz")
    labels = np.load(gt_dir / "contacts.npz")
    names = list(body.contact_frame_names)
    label_names = [str(n) for n in labels["contact_frame_names"]]
    if label_names != names:
        raise ValueError(f"{gt_dir / 'contacts.npz'} labels {label_names} but the body has "
                         f"contact frames {names}")
    if not np.array_equal(labels["object_ids"], meta["object_ids"]):
        raise ValueError(f"{gt_dir / 'contacts.npz'} labels other people than smplx.npz")

    weights = {**config.weights_for("kindyn_1"), **(weight_overrides or {})}
    knobs = config.knobs
    half_angle = float(knobs.get("friction", {}).get("half_angle_deg", 45.0))
    gravity_world = np.asarray(grav["gravity_world"], np.float32)
    fps = float(meta["fps"])
    q, betas, contact = meta["q"], meta["betas"], np.asarray(labels["frame_contact"], bool)
    people, frames = q.shape[:2]

    stacked: dict[str, list] = {}
    for person in range(people):
        solved = person_forces(
            body, betas[person], q[person], contact[person], gravity_world=gravity_world,
            fps=fps, device=device, weights=weights, dyn_opt=knobs["dynamics_optimizer"],
            torque_table=knobs["torque"]["multipliers"], half_angle_deg=half_angle,
            edge_speed_m_s=edge_speed_m_s)
        for key, value in solved.items():
            stacked.setdefault(key, []).append(value)
    diagnostics = np.stack([np.stack(stacked[k]) for k in (
        "base_wrench_before", "base_wrench_warm", "base_wrench_after",
        "cone_violation_max_n")], axis=1).astype(np.float32)

    # The people the mask step found no usable frame for stay solved (their motion
    # is still ground truth) but flagged, exactly as the bodies are.
    valid = np.ones((people, frames), bool)
    boxes = gt_dir / "bboxes.npz"
    if boxes.is_file():
        valid &= np.asarray(np.load(boxes)["person_valid"], bool)[:, None]

    return {
        "force_method": np.str_(FORCE_METHOD),
        "frame_forces": np.stack(stacked["frame_forces"]),              # (P, N, F, 3) world N
        "contact_forces_world": np.stack(stacked["force_world"]),       # (P, N, J, 3) per joint
        "joints_world": np.stack(stacked["joints_world"]),              # (P, N, J, 3)
        "torque_mag": np.stack(stacked["torque_mag"]),                  # (P, N, J)
        "joint_torques": np.stack(stacked["joint_torques"]),            # (P, N, nv)
        "base_wrench": np.stack(stacked["base_wrench"]),                # (P, N, 6) root-local
        "base_wrench_diagnostics": diagnostics,                         # (P, 4) before/warm/after/cone
        "total_mass": np.asarray(stacked["total_mass"], np.float32),    # (P,) kg
        "frame_contact": contact,                                       # (P, N, F) the labels used
        "contact_frame_names": np.array(names),
        "contact_frame_parents": np.array(body.contact_frame_parents, dtype=np.int64),
        # Body fields the viewer's force sources carry, so this file reads like kindyn_1.npz.
        "q": q, "betas": betas, "valid_mask": valid,
        "joint_names": meta["joint_names"], "parents": meta["parents"],
        "object_ids": np.asarray(meta["object_ids"], np.int32),
        "num_people": np.int32(people), "num_frames": np.int32(frames),
        "fps": np.float32(fps),
        "gravity_world": gravity_world,
        "cone_half_angle_deg": np.float32(half_angle),
        "edge_speed_m_s": np.float32(edge_speed_m_s),
        "weights": np.str_(json.dumps(
            {k: v for k, v in weights.items()
             if v and k.split("_")[0] in ("base", "force", "torque")}, sort_keys=True)),
    }


def save_forces(path: Path, fields: dict) -> None:
    """Write ``gt/forces.npz`` — the force fields ``kindyn_1.npz`` carries."""
    np.savez_compressed(path, **fields)
