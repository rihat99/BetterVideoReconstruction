"""Step ``kindyn_1``: fit every person's trajectory and contact forces together.

The steps before it fit the motion first and read contacts from it. Here the pose and the forces
at the labelled contact frames are solved as one problem, so that the physics can reshape the
motion: what the forces cannot explain of the root's six rows is a residual that pulls on the pose
too (:mod:`bvr.terms.dynamics`). The problem is the kinematic one of ``kinematics_2`` without the
wall patches, with one more variable, the force activations, and the dynamics terms. Two solves
of L-BFGS::

    forces  the pose is held; the forces find what the motion needs
    joint   pose and forces move together under one line search, with the scene

The first is there because forces start at zero, where the root residual says the body is in free
fall and its gradient on the pose means nothing. At a held pose inverse dynamics runs once and
the torques are affine in the forces (:func:`bvr.terms.dynamics.balance`), so that solve is a
problem of its own with the forces as its one variable. The people of a clip are one batch. A
person without contact labels has no dynamics term and is fitted kinematically, with no forces;
a person without a tracked frame or without keypoints is written as they came.

The file is a body file (``bvr.body.save``) with the solved forces: ``frame_forces`` at the contact
frames and ``contact_forces_world`` summed onto their parent joints (N, world), ``joint_torques``
(inverse dynamics, the root rows first), ``base_wrench`` (the root residual, in the root's frame)
and ``force_confidence``. The residual and the torque sizes are zero on the frames whose inverse
dynamics do not count.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

import better_human as bh
import numpy as np
import torch
from better_robot import optim

from bvr import body, files, fit, paths, run
from bvr.contacts import confidence
from bvr.terms import dynamics, motion, objective

log = logging.getLogger(__name__)


def load(stage: run.Stage) -> Callable[[paths.Clip], None]:
    human = body.load(stage.device)
    return lambda clip: kindyn_1(clip, human, stage.config)


def kindyn_1(clip: paths.Clip, human: bh.HumanModel, config: dict) -> None:
    """Fit the bodies of ``kinematic_2.npz`` and their contact forces together and write ``kindyn_1.npz``."""
    options, device = config["stages"]["kindyn_1"], human.spec.device
    start = files.load_npz(clip.find(paths.KINEMATIC_2), "bvr-kinematics")
    contacts = files.load_npz(clip.find(paths.CONTACTS_2), "bvr-contacts")
    down = files.load_npz(clip.find(paths.GRAVITY), "bvr-gravity")["gravity_world"]
    if list(contacts["contact_frame_names"]) != list(human.contact_frames.names):
        raise ValueError(f"{clip.find(paths.CONTACTS_2)} names other contact frames than the body; rerun contacts_2")
    evidence = fit.Evidence(clip, human, config, start)
    tensor, time, people = evidence.tensor, evidence.time, evidence.people

    q, tracked = tensor(start["q"]), tensor(start["valid_mask"].astype(bool))
    identity = dynamics.under_gravity(body.bake(human, tensor(start["betas"])), tensor(down))
    listed = {object_id: row for row, object_id in enumerate(contacts["object_ids"])}
    held = torch.zeros(*tracked.shape, len(human.contact_frames.names), dtype=torch.bool, device=device)
    for person in people:
        if (row := listed.get(start["object_ids"][person])) is not None:
            held[person] = tensor(contacts["frame_contact"][row]) & tracked[person, :, None]
    # The pyramids stand where the starting pose puts the skin.
    skin = body.frame_normals(body.contact_frames(objective.posed(identity, q)))
    edges = dynamics.cone_edges(skin, config["friction"]["half_angle_deg"]) * held[..., None, None]
    activation = torch.zeros(*held.shape, 4, device=device)

    if len(people):
        fitted = dynamics.under_gravity(body.bake(human, tensor(start["betas"])[people]), tensor(down))
        physics = {"edges": edges[people], "held": held[people], "tracked": evidence.tracked, "time": time,
                   "multipliers": config["torque_multipliers"]}
        weights, iterations = options["weights"], options["iterations"]
        forces_alone = optim.Problem({"activation": activation[people]})
        dynamics.register(forces_alone, fitted, **physics, fixed=objective.posed(fitted, q[people]))
        warm = optim.run(forces_alone, [fit.lbfgs(
            "forces", iterations["forces"], config, history=options["forces_history"],
            gradient_tolerance=options["gradient_tolerance"],
            weights={name: weights[name] for name in dynamics.TERMS})]).x["activation"]
        problem = evidence.problem(fitted, q[people], config["contact_attraction"], variables={"activation": warm})
        dynamics.register(problem, fitted, **physics)
        result = optim.run(problem.with_(data={"labels": held[people]}), [
            fit.lbfgs("joint", iterations["joint"], config, weights=weights, on_start=evidence.scene_refresh(fitted))])
        q[people], activation[people] = objective.posed(fitted, result.x["q"]).q, result.x["activation"]

    state = objective.posed(identity, q)
    velocity = motion.velocity(human.model, time)(state.q)
    weight = dynamics.weight(identity)
    forces = dynamics.cone_forces(activation, edges, weight)
    torques = dynamics.torques(state, velocity, time.difference(velocity), forces, time=time)
    counted = dynamics.stencil_rows(time, tracked) & held.any(dim=(1, 2))[:, None]
    root = torques[..., :6] * counted[..., None]
    # One size per joint: the root's six rows, then three rows for each of the other joints.
    sizes = torch.cat((torques[..., :6].norm(dim=-1, keepdim=True),
                       torques[..., 6:].unflatten(-1, (-1, 3)).norm(dim=-1)), dim=-1) * counted[..., None]
    parents = tensor(np.array(human.contact_frames.parent_joints))
    joints = len(human.spec.joint_names)
    trust = confidence.force_confidence(root.cpu().numpy(), forces.cpu().numpy(), tracked.cpu().numpy(),
                                        weight.cpu().numpy())
    clip_trust = np.array([values[rows].mean() if rows.any() else np.nan
                           for values, rows in zip(trust, tracked.cpu().numpy())], np.float32)
    residual = (root[..., :3].norm(dim=-1) / weight[:, None]).cpu().numpy()
    log.info("root residual force, median over counted frames (body weights): %s; force confidence: %s",
             "/".join(f"{np.median(values[rows]) if rows.any() else np.nan:.3f}"
                      for values, rows in zip(residual, counted.cpu().numpy())),
             "/".join(f"{value:.3f}" for value in clip_trust))

    body.save(
        clip.path(paths.KINDYN_1), "bvr-kindyn", identity, q,
        **{key: start[key] for key in ("object_ids", "valid_mask", "fps", "video_width", "video_height", "drift_pre",
                                       "auto_labels")},
        frame_forces=forces.cpu().numpy(), frame_contact=held.cpu().numpy(),
        contact_frame_parents=parents.cpu().numpy(),
        contact_forces_world=forces.new_zeros(*tracked.shape, joints, 3).index_add(2, parents, forces).cpu().numpy(),
        joint_contact=confidence.fold_contacts(held.cpu().numpy(), parents.cpu().numpy(), joints),
        joint_torques=torques.cpu().numpy(), torque_mag=sizes.cpu().numpy(), base_wrench=root.cpu().numpy(),
        joints_world=state.joint_translations.cpu().numpy(), total_mass=(weight / dynamics.GRAVITY).cpu().numpy(),
        gravity_world=down.astype(np.float32), force_confidence=trust, force_confidence_clip=clip_trust)


if __name__ == "__main__":
    run.main(load)
