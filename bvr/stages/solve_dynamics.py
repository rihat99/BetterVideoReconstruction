"""Step ``kindyn_1``: fit every person's trajectory and contact forces together.

The steps before it fit the motion first and read contacts from it. Here the pose and the forces
at the labelled contact frames are solved as one problem, so that the physics can reshape the
motion: what the forces cannot explain of the root's six rows is a residual that pulls on the pose
too (:mod:`bvr.terms.dynamics`). The problem is the kinematic one of ``kinematics_2`` without the
wall patches, with one more variable, the force activations, and the dynamics terms. Two phases
of L-BFGS, which moves pose and forces together under a line search::

    forces  the pose is held; the forces find what the motion needs
    joint   pose and forces, with the scene

The first phase is there because forces start at zero, where the root residual says the body is
in free fall and its gradient on the pose means nothing. The people of a clip are one batch. A
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

from bvr import body, camera, files, paths, run
from bvr.contacts import confidence
from bvr.geometry.normals import load_scene_cloud
from bvr.terms import dynamics, motion, objective, scene

log = logging.getLogger(__name__)


def load(stage: run.Stage) -> Callable[[paths.Clip], None]:
    human = body.load(stage.device)
    return lambda clip: kindyn_1(clip, human, stage.config)


def kindyn_1(clip: paths.Clip, human: bh.HumanModel, config: dict) -> None:
    """Fit the bodies of ``kinematic_2.npz`` and their contact forces together and write ``kindyn_1.npz``."""
    options, penetration, device = config["stages"]["kindyn_1"], config["penetration"], human.spec.device

    def tensor(array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, device=device)

    start = files.load_npz(clip.find(paths.KINEMATIC_2), "bvr-kinematics")
    contacts = files.load_npz(clip.find(paths.CONTACTS_2), "bvr-contacts")
    transform = files.load_npz(clip.find(paths.TRANSFORM), "bvr-transform")
    keypoints = files.load_npz(clip.find(paths.SAPIENS_POSE), "bvr-keypoints")
    down = files.load_npz(clip.find(paths.GRAVITY), "bvr-gravity")["gravity_world"]
    if list(contacts["contact_frame_names"]) != list(human.contact_frames.names):
        raise ValueError(f"{clip.find(paths.CONTACTS_2)} names other contact frames than the body; rerun contacts_2")
    cameras = tensor(transform["extrinsics"]), tensor(transform["intrinsics_px_orig"])
    time = motion.Time(float(transform["fps"]))
    cloud = scene.Scene(*load_scene_cloud(
        files.load_npz(clip.find(paths.SCENE), "bvr-scene"), camera.centers(cameras[0]),
        conf_drop_q=penetration["conf_drop_q"], k=penetration["normal_k"], device=device))

    q, tracked = tensor(start["q"]), tensor(start["valid_mask"].astype(bool))
    identity = dynamics.under_gravity(body.bake(human, tensor(start["betas"])), tensor(down))
    # The people the step fits, as rows of the body file, of the keypoint file and of the contact file.
    people = [person for person, object_id in enumerate(start["object_ids"])
              if tracked[person].any() and object_id in keypoints["object_ids"]]
    seen = [list(keypoints["object_ids"]).index(start["object_ids"][person]) for person in people]
    listed = {object_id: row for row, object_id in enumerate(contacts["object_ids"])}
    held = torch.zeros(*tracked.shape, len(human.contact_frames.names), dtype=torch.bool, device=device)
    for person in people:
        if (row := listed.get(start["object_ids"][person])) is not None:
            held[person] = tensor(contacts["frame_contact"][row]) & tracked[person, :, None]
    # The pyramids stand where the starting pose puts the skin.
    skin = body.frame_normals(body.contact_frames(objective.posed(identity, q)))
    edges = dynamics.cone_edges(skin, config["friction"]["half_angle_deg"]) * held[..., None, None]
    activation = torch.zeros(*held.shape, 4, device=device)

    if people:
        fitted = dynamics.under_gravity(body.bake(human, tensor(start["betas"])[people]), tensor(down))
        problem = objective.kinematics_problem(
            fitted, q[people], keypoints=tensor(keypoints["keypoints"][seen]),
            scores=tensor(keypoints["keypoint_scores"][seen]), tracked=tracked[people], cameras=cameras, time=time,
            attraction=config["contact_attraction"], config=config, variables={"activation": activation[people]})
        dynamics.register(problem, fitted, edges=edges[people], held=held[people], tracked=tracked[people], time=time,
                          multipliers=config["torque_multipliers"])

        def look(problem: optim.Problem, x: dict[str, torch.Tensor]) -> tuple[dict, dict]:
            reach = objective.reach(objective.posed(fitted, x["q"]))
            return x, {"scene": scene.candidates(cloud, reach, tracked[people], prune_margin_m=penetration[
                "prune_margin_m"], max_candidates=penetration["max_candidates"])}

        weights, iterations, search = options["weights"], options["iterations"], config["optimizer"]
        result = optim.run(problem.with_(data={"labels": held[people]}), [
            _lbfgs("forces", iterations["warm_start"], optim.LBFGS(
                history=options["warm_start_history"], line_search=search["lbfgs_line_search"],
                gradient_tolerance=options["gradient_tolerance"]),
                weights={name: weights[name] for name in dynamics.TERMS},
                free={"q": torch.zeros(human.model.nq, dtype=torch.bool, device=device)}),
            _lbfgs("joint", iterations["joint"], optim.LBFGS(
                history=search["lbfgs_history"], line_search=search["lbfgs_line_search"]),
                weights=weights, on_start=look)])
        q[people], activation[people] = objective.posed(fitted, result.x["q"]).q, result.x["activation"]

    state = objective.posed(identity, q)
    velocity = motion.velocity(human.model, time)(state.q)
    forces = dynamics.cone_forces(activation, edges)
    torques = dynamics.torques(state, velocity, time.difference(velocity), forces, time=time)
    counted = dynamics.stencil_rows(time, tracked) & held.any(dim=(1, 2))[:, None]
    root = torques[..., :6] * counted[..., None]
    # One size per joint: the root's six rows, then three rows for each of the other joints.
    sizes = torch.cat((torques[..., :6].norm(dim=-1, keepdim=True), torques[..., 6:].unflatten(-1, (-1, 3)).norm(dim=-1)),
                      dim=-1) * counted[..., None]
    parents = tensor(np.array(human.contact_frames.parent_joints))
    joints = len(human.spec.joint_names)
    weight = dynamics.weight(identity)
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


def _lbfgs(name: str, iterations: int, solver: optim.LBFGS, **phase) -> optim.Phase:
    """A phase of BetterRobot's L-BFGS, which keeps one memory and one line search per person. It ends early once
    no person's search finds a step, and logs each person's loss and terms at every quarter and at its end."""
    def on_step(problem: optim.Problem, state: optim.State) -> None:
        step = state.iteration - 1
        if step % max(1, iterations // 4) == 0 or step == iterations - 1 or bool(state.done.all()):
            values = {term: loss / problem.weights[term] for term, loss in state.extra["terms"].items()}
            log.info("%-6s %4d  loss %s  %s", name, step, _numbers(state.loss),
                     "  ".join(f"{term} {_numbers(value)}" for term, value in values.items()))
    return optim.Phase(solver, iterations, name=name, on_step=on_step, **phase)


def _numbers(values: torch.Tensor) -> str:
    return "/".join(f"{value:.4g}" for value in values.tolist())


if __name__ == "__main__":
    run.main(load)
