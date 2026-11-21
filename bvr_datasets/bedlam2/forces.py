"""Contact forces on BEDLAM2 ground truth: the ``kindyn_1`` force solve at a frozen pose.

BEDLAM2 gives us the motion (mocap, exact) and ``bvr_datasets/bedlam2/label_bedlam2.py`` gives us
which contact frames are on the floor. What neither states is the force each contact
carries. ``kindyn_1`` (:mod:`bvr.stages.solve_dynamics`) already answers that question
for estimated bodies, in two phases: the forces at the held starting pose, then pose and
forces together. Here the pose is ground truth, so the second phase would only pull it
away from the truth — the first phase alone is the whole solve.

That phase is an L-BFGS over the per-contact-frame forces, every force held inside a 45°
friction pyramid about its frame's contact normal, with the dynamics terms of
``[stages.kindyn_1.weights]`` as the objective (:mod:`bvr.terms.dynamics`) — root wrench
residual, force size and smoothness, strength-weighted joint torques. The one addition is
the ``force_edge`` term (the OpenCap work), which prices the force on the first and last
frame of every contact interval so a load ramps in and out instead of stepping; its speed
scale lets a genuine landing step while a walking heel strike may not.
"""

from __future__ import annotations

import json
from pathlib import Path

import better_human as bh
import numpy as np
import torch
from better_robot import optim, terms

from bvr import body
from bvr.terms import dynamics, motion, objective

FORCE_METHOD = "bedlam2-kindyn-frozen-v2"


def edge_weights(held: torch.Tensor, height: torch.Tensor, *, fps: float, speed_m_s: float) -> torch.Tensor:
    """Where ``force_edge`` prices a force, ``(P, T, F)``: the first and last frame of each contact interval.

    ``held (P, T, F)`` bool is where a contact frame touches; ``height (P, T)`` the root's height in metres. A
    switch is scaled by ``1 / (1 + (v / speed_m_s)^2)``, ``v`` the body's vertical speed across it — downward at
    an onset, upward at a release. A force step at a contact switch is the impulse that changes the whole body's
    momentum, so the allowed step grows with the body's speed: a drop-jump landing at 2.5 m/s may step to a large
    force, a walking heel strike with the body descending at 0.2 m/s may not. The body's speed, not the foot's: a
    swinging foot arrives at 1 m/s in ordinary walking, the same as a soft landing.
    """
    held = held.float()
    onset, release = torch.zeros_like(held), torch.zeros_like(held)
    onset[:, 1:] = held[:, 1:] * (1.0 - held[:, :-1])
    release[:, :-1] = held[:, :-1] * (1.0 - held[:, 1:])
    rate = torch.zeros_like(height)
    rate[:, 1:] = (height[:, 1:] - height[:, :-1]) * fps
    rise = torch.zeros_like(rate)
    rise[:, :-1] = torch.relu(rate[:, 1:])
    speed = onset * torch.relu(-rate)[..., None] + release * rise[..., None]
    return (onset + release).clamp(max=1.0) / (1.0 + (speed / speed_m_s) ** 2)


def scene_forces(gt_dir: Path, *, human: bh.HumanModel, config: dict, edge_weight: float,
                 edge_speed_m_s: float) -> dict:
    """Solve every person in one BEDLAM2 scene, the people as one batch.

    :param gt_dir: the scene's ``features/gt/<aa>/<bb>/<scene_id>`` directory, holding
        ``smplx.npz``, ``gravity.npz`` and the ``contacts.npz`` the labels step wrote.
    :param human: the pipeline's body (:func:`bvr.body.load`), on the device the solve runs on.
    :param config: the resolved ``configs/bvr.toml``; its ``[stages.kindyn_1]`` dynamics weights and
        forces phase, ``[torque_multipliers]``, ``[friction]`` and ``[optimizer]`` are used.
    :param edge_weight: the weight of ``force_edge``; ``edge_speed_m_s`` its speed scale.
    :returns: the arrays :func:`save_forces` stores, keyed by npz field name.
    """
    device = human.spec.device
    meta = np.load(gt_dir / "smplx.npz")
    gravity = np.load(gt_dir / "gravity.npz")
    labels = np.load(gt_dir / "contacts.npz")
    names = list(human.contact_frames.names)
    label_names = [str(n) for n in labels["contact_frame_names"]]
    if label_names != names:
        raise ValueError(f"{gt_dir / 'contacts.npz'} labels {label_names} but the body has "
                         f"contact frames {names}")
    if not np.array_equal(labels["object_ids"], meta["object_ids"]):
        raise ValueError(f"{gt_dir / 'contacts.npz'} labels other people than smplx.npz")

    options = config["stages"]["kindyn_1"]
    half_angle = float(config["friction"]["half_angle_deg"])
    down = np.asarray(gravity["gravity_world"], np.float32)
    fps = float(meta["fps"])
    q = torch.as_tensor(meta["q"], device=device)
    held = torch.as_tensor(np.asarray(labels["frame_contact"], bool), device=device)
    people, frames = q.shape[:2]
    tracked = torch.ones(people, frames, dtype=torch.bool, device=device)
    time = motion.Time(fps)
    identity = dynamics.under_gravity(body.bake(human, torch.as_tensor(meta["betas"], device=device)),
                                      torch.as_tensor(down, device=device))
    model = identity.human.model

    # The pyramids stand where the pose puts the skin.
    state = objective.posed(identity, q)
    edges = dynamics.cone_edges(body.frame_normals(body.contact_frames(state)), half_angle) * held[..., None, None]
    up = -torch.nn.functional.normalize(torch.as_tensor(down, device=device), dim=0)
    edge = edge_weights(held, state.joint_translations[..., 0, :] @ up, fps=fps, speed_m_s=edge_speed_m_s)
    weight = dynamics.weight(identity)
    rows = dynamics.counted_rows(time, tracked, held)
    scale = 0.5 / (rows.sum(dim=1).clamp_min(1) * weight.square())

    problem = optim.Problem({"q": q, "activation": q.new_zeros(*held.shape, 4)}, time=("q",))
    problem.cached(terms.RobotManifold(model).project, name="pose", reads=("q",))
    problem.cached(bh.terms.human_state(identity, q="pose"), name="body")
    problem.cached(motion.velocity(model, time), name="velocity")
    problem.cached(time.difference, name="acceleration", reads=("velocity",))
    dynamics.register(problem, identity, edges=edges, held=held, tracked=tracked, time=time,
                      multipliers=config["torque_multipliers"])
    problem.scalar("force_edge", lambda forces: scale * (edge[..., None] * forces.square()).flatten(1).sum(dim=1),
                   reads=("forces",), weight=0.0)
    weights = {**{name: options["weights"][name] for name in dynamics.TERMS}, "force_edge": edge_weight}
    result = optim.run(problem, [optim.Phase(
        optim.LBFGS(history=options["forces_history"], line_search=config["optimizer"]["lbfgs_line_search"],
                    gradient_tolerance=options["gradient_tolerance"]),
        options["iterations"]["forces"], name="forces", weights=weights,
        free={"q": torch.zeros(model.nq, dtype=torch.bool, device=device)})])

    with torch.no_grad():
        forces = dynamics.cone_forces(result.x["activation"], edges, weight)
        velocity = motion.velocity(model, time)(state.q)
        torques = dynamics.torques(state, velocity, time.difference(velocity), forces, time=time)
    root = torques[..., :6] * rows[..., None]
    sizes = torch.cat((torques[..., :6].norm(dim=-1, keepdim=True),
                       torques[..., 6:].unflatten(-1, (-1, 3)).norm(dim=-1)), dim=-1) * rows[..., None]
    parents = torch.as_tensor(human.contact_frames.parent_joints, device=device)
    joints = len(human.spec.joint_names)

    # The people the mask step found no usable frame for stay solved (their motion
    # is still ground truth) but flagged, exactly as the bodies are.
    valid = np.ones((people, frames), bool)
    boxes = gt_dir / "bboxes.npz"
    if boxes.is_file():
        valid &= np.asarray(np.load(boxes)["person_valid"], bool)[:, None]

    def numpy(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().numpy()

    return {
        "force_method": np.str_(FORCE_METHOD),
        "frame_forces": numpy(forces),                                                     # (P, N, F, 3) world N
        "contact_forces_world": numpy(forces.new_zeros(people, frames, joints, 3).index_add(2, parents, forces)),
        "joints_world": numpy(state.joint_translations),                                   # (P, N, J, 3)
        "torque_mag": numpy(sizes),                                                        # (P, N, J)
        "joint_torques": numpy(torques),                                                   # (P, N, nv)
        "base_wrench": numpy(root),                                                        # (P, N, 6) root frame
        "total_mass": numpy(weight / dynamics.GRAVITY),                                    # (P,) kg
        "frame_contact": numpy(held),                                                      # (P, N, F) labels used
        "contact_frame_names": np.array(names),
        "contact_frame_parents": numpy(parents),
        # Body fields the viewer's force sources carry, so this file reads like kindyn_1.npz.
        "q": meta["q"], "betas": meta["betas"], "valid_mask": valid,
        "joint_names": meta["joint_names"], "parents": meta["parents"],
        "object_ids": np.asarray(meta["object_ids"], np.int32),
        "num_people": np.int32(people), "num_frames": np.int32(frames),
        "fps": np.float32(fps),
        "gravity_world": down,
        "cone_half_angle_deg": np.float32(half_angle),
        "edge_speed_m_s": np.float32(edge_speed_m_s),
        "weights": np.str_(json.dumps({name: value for name, value in weights.items() if value}, sort_keys=True)),
    }


def save_forces(path: Path, fields: dict) -> None:
    """Write ``gt/forces.npz`` — the force fields ``kindyn_1.npz`` carries."""
    np.savez_compressed(path, **fields)
