"""The climb_wall_3 ground truth: the human steps of the pipeline on a trial that three calibrated cameras see.

The steps and their files are the pipeline's (``human_optim/kinematic_1.npz``, ``contacts_1.npz``,
``kinematic_2.npz``, ``kindyn_1.npz``), and so are the problem, its terms and the solvers. What differs:

- the image evidence is every camera's: its Sapiens keypoints (``views_keypoints``) and its own SAM 3D Body
  estimate projected into it (``views_estimates``), through the calibrated cameras. A camera's share of a weight
  is in proportion to ``1 / sigma^2`` of its keypoints (``view_sigma_px``): the front pair agrees to a few pixels,
  the side camera to about twenty;
- the climber's shape is one per climber, fitted to their height and mass (``bvr_datasets/peter/gt/shape.py``);
- the contact labels are the boards': a limb is in contact while its board measures a force
  (``bvr_datasets/_rigs.py``). ``kinematics_1`` holds every contact frame of a loaded limb still, ``contacts_1``
  puts the label on the limb's frames at the wall, and ``kinematics_2`` and ``kindyn_1`` run on those labels;
- nothing of the climber is behind the wall's plane (``wall_penetration``), and in ``kindyn_1`` each limb's summed
  force is pulled to what its board measured (``board_force``): the boards shape the pose through the dynamics.

A trial's tree holds ``cameras.npz`` (the three cameras; the world is ``cam_left``'s frame; camera ``k``'s frame
``t + frame_offset[k]`` shows the instant of ``cam_left``'s frame ``t``), ``views/<camera>/sam3d/smplx_params.npz``
and ``views/<camera>/sapiens/pose.npz``, ``geometry/scene.npz`` and ``geometry/wall.npz``, ``geocalib/gravity.npz``
and ``sensor_forces.npz``. A run reads the trees as its upstream and writes ``<trial>/human_optim/`` under its own
root. It names a trial by a video, so ``--source`` is a folder of links ``<trial>.mp4``::

    uv run python -m bvr_datasets.peter.gt.fit_multiview --source <videos> --save-dir <run> --upstream <trees> \\
        --config bvr_datasets/peter/gt/gt.toml

``--steps`` picks some of the four steps; the other options are those of a stage
(``python -m bvr.stages.refine_kinematics --help``) without ``--step``.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import better_human as bh
import numpy as np
import torch
from better_robot import optim

from bvr import body, camera, files, fit, paths, run
from bvr.geometry.normals import load_scene_cloud
from bvr.stages import refine_kinematics, solve_dynamics
from bvr.stages.refine_kinematics import BODY, ROOT, first
from bvr.terms import Term, dynamics, motion, objective, scene, visual
from bvr_datasets import _rigs
from bvr_datasets.peter.chain import measured_contacts

#: The cameras of a trial, in the order of every array of ``cameras.npz``.
VIEWS = ("cam_left", "cam_right", "cam_3")
#: The steps of the chain, in order.
STEPS = ("kinematics_1", "contacts_1", "kinematics_2", "kindyn_1")
#: Files of a trial's tree that the pipeline's tree does not have.
CAMERAS, WALL, SENSORS = "cameras.npz", "geometry/wall.npz", "sensor_forces.npz"
#: The terms that read the scene; the phases that place the body run before it is looked up.
SCENE_TERMS = ("penetration", "contact_attraction", "contact_acquisition")
#: The start pose's shift to the cameras' mean pelvis is smoothed over this window (s).
SHIFT_WINDOW_S = 0.5


@dataclass(frozen=True)
class View:
    """What one camera saw of the climber, on ``cam_left``'s clock, ``T`` frames."""

    cameras: camera.Cameras     # (extrinsics (T, 4, 4), intrinsics (T, 3, 3)): the same camera on every frame
    keypoints: torch.Tensor     # (1, T, 308, 2) Sapiens keypoints in pixels
    scores: torch.Tensor        # (1, T, 308)
    tracked: torch.Tensor       # (1, T) bool
    estimate: bh.HumanState     # the camera's own SAM 3D Body estimate, in the world


class Trial:
    """A trial's evidence on the device, and the problem it poses.

    ``views`` are its cameras, ``tracked (1, T)`` the frames any of them tracks the climber on, ``time`` and
    ``scene`` as a clip's. ``keypoints`` and ``estimates`` are the image terms of all cameras and ``wall`` the
    wall's.
    """

    def __init__(self, clip: paths.Clip, human: bh.HumanModel, config: dict) -> None:
        self.clip, self.human, self.config, device = clip, human, config, human.spec.device
        self.tensor = lambda array: torch.as_tensor(array, device=device)
        with np.load(clip.find(CAMERAS)) as stored:
            cameras = dict(stored)
        frames = int(cameras["num_frames"]) - int(cameras["frame_offset"].max())
        self.time = motion.Time(float(cameras["fps"]))
        self.image_size = self.tensor(cameras["image_size"][0])
        self.views = [self._view(name, slice(offset, offset + frames), self.tensor(extrinsics), self.tensor(intrinsics))
                      for name, offset, extrinsics, intrinsics in zip(
                          VIEWS, cameras["frame_offset"], cameras["extrinsics"], cameras["K"])]
        self.tracked = torch.stack([view.tracked for view in self.views]).any(dim=0)
        centers = torch.cat([camera.centers(view.cameras[0][:1]) for view in self.views])
        self.scene = scene.Scene(*load_scene_cloud(
            files.load_npz(clip.find(paths.SCENE), "bvr-scene"), centers,
            conf_drop_q=config["penetration"]["conf_drop_q"], k=config["penetration"]["normal_k"], device=device))

        precision = 1.0 / np.square(np.maximum(cameras["view_sigma_px"], 1.0))
        shares, fps = (precision / precision.sum()).tolist(), self.time.fps
        self.keypoints = _shared(shares, [visual.keypoint_reprojection(
            human, view.keypoints, view.scores, view.tracked, view.cameras, fps, **config["keypoints"])
            for view in self.views])
        self.estimates = _shared(shares, [visual.start_reprojection(
            view.estimate, view.tracked, view.cameras, fps, **config["reprojection_start"]) for view in self.views])
        with np.load(clip.find(WALL)) as wall:
            self.wall = wall_penetration(human, self.tensor(wall["normal"]), self.tensor(wall["point"]),
                                         config["chamfer"]["vertex_subsample"])

    def _view(self, name: str, rows: slice, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> View:
        """The camera ``name`` over its frames ``rows``, which show the instants of ``cam_left``'s frames."""
        fitted = files.load_npz(self.clip.find(f"views/{name}/{paths.SMPLX_PARAMS}"), "bvr-smplx")
        detected = files.load_npz(self.clip.find(f"views/{name}/{paths.SAPIENS_POSE}"), "bvr-keypoints")
        betas = fitted["betas"][:1, rows]
        pose = {field: fitted[field][:1, rows] for field in refine_kinematics.POSE_FIELDS}
        # A tracked frame whose converted body is not finite is no observation.
        tracked = fitted["valid_mask"][:1, rows].astype(bool) & np.all(
            [np.isfinite(values).all(axis=-1) for values in (betas, *pose.values())], axis=0)
        betas = np.where(tracked[..., None], betas, 0.0).sum(axis=1) / max(tracked.sum(), 1)
        identity, q, _ = bh.from_classic(self.human, bh.families.SMPLXClassic(
            betas=self.tensor(betas.astype(np.float32))[:, None],
            **{field: self.tensor(np.nan_to_num(refine_kinematics.held(values, tracked)))
               for field, values in pose.items()}))
        cameras = extrinsics.expand(tracked.shape[1], 4, 4), intrinsics.expand(tracked.shape[1], 3, 3)
        return View(cameras, self.tensor(detected["keypoints"][:1, rows]),
                    self.tensor(detected["keypoint_scores"][:1, rows]), self.tensor(tracked),
                    objective.posed(identity, body.to_world(q, cameras[0])))

    def start(self) -> torch.Tensor:
        """Where a fit starts, ``(1, T, 211)``: ``cam_left``'s estimate, its pelvis at the mean of the cameras'.

        The shift is smoothed over :data:`SHIFT_WINDOW_S`, so that the estimates' disagreement from frame to frame
        does not become jitter of the start.
        """
        pelvis = torch.stack([view.estimate.joint_translations[..., 0, :] for view in self.views])
        seen = torch.stack([view.tracked for view in self.views])[..., None]
        shift = ((pelvis * seen).sum(dim=0) / seen.sum(dim=0).clamp_min(1) - pelvis[0]) * self.tracked[..., None]
        apart = float(shift.norm(dim=-1)[self.tracked].median())
        if apart > 1.0:  # three estimates of one person disagree by decimetres
            raise ValueError(f"{self.clip.name}: the cameras' estimates are {apart:.1f} m apart; they track "
                             "different people")
        shift = visual.smooth(shift[:, :, None], self.tracked[..., None], window_s=SHIFT_WINDOW_S,
                              fps=self.time.fps)[:, :, 0]
        q = self.views[0].estimate.q
        return torch.cat((q[..., :3] + shift, q[..., 3:]), dim=-1)

    def loaded(self) -> torch.Tensor:
        """Where a contact frame's limb is loaded by its board, ``(1, T, F)`` bool, on tracked frames."""
        with np.load(self.clip.find(SENSORS)) as sensors:
            limb_on, _ = _rigs.limb_contact(dict(sensors), self.tracked.shape[1])
        limbs = [_rigs.LIMBS.index(_rigs.limb_of_frame(name)) for name in self.human.contact_frames.names]
        return self.tensor(limb_on[:, limbs])[None] & self.tracked[..., None]

    def problem(self, identity: bh.Identity, start: torch.Tensor, **more) -> optim.Problem:
        """The kinematic problem of the climber from ``start (1, T, 211)``, with the cameras' terms and the wall's.

        The terms of one camera that the problem also holds read ``cam_left``; the weight tables leave them off.
        ``more`` is ``variables=`` of :func:`bvr.terms.objective.kinematics_problem`.
        """
        left = self.views[0]
        problem = objective.kinematics_problem(
            identity, start, keypoints=left.keypoints, scores=left.scores, tracked=self.tracked,
            cameras=left.cameras, time=self.time, attraction=self.config["contact_attraction"], config=self.config,
            **more)
        problem.scalar("views_keypoints", self.keypoints, weight=0.0)
        problem.scalar("views_estimates", self.estimates, weight=0.0)
        problem.scalar("wall_penetration", self.wall, weight=0.0)
        return problem

    def look(self, identity: bh.Identity, labels: torch.Tensor | None = None):
        """A phase's ``on_start``: the scene around the standing body and, given ``labels``, each one's wall patch
        as ``cam_left`` sees it."""
        prune = {key: self.config["penetration"][key] for key in ("prune_margin_m", "max_candidates")}
        patch = {key: self.config["contact_acquisition"][key] for key in ("radius_px", "patch_k", "conf_thresh")}

        def on_start(problem: optim.Problem, x: dict[str, torch.Tensor]) -> tuple[dict, dict]:
            state = objective.posed(identity, x["q"])
            data = {"scene": scene.candidates(self.scene, objective.reach(state), self.tracked, **prune)}
            if labels is not None:
                data["patches"] = scene.patches(self.scene, body.contact_frames(state), labels,
                                                self.views[0].cameras, self.image_size, **patch)
            return x, data
        return on_start


def kinematics_1(clip: paths.Clip, human: bh.HumanModel, config: dict) -> None:
    """Fit the climber's trajectory to the three cameras and write ``kinematic_1.npz``."""
    trial = Trial(clip, human, config)
    with np.load(shape_file(clip)) as shape:
        identity = body.bake(human, trial.tensor(shape["betas"])[None])
    width, height = trial.image_size.tolist()
    undetected = (*trial.tracked.shape, len(human.contact_frames.names))
    body.save(clip.path(paths.KINEMATIC_1), "bvr-kinematics", identity, placed(trial, identity),
              object_ids=np.zeros(1, np.int32), valid_mask=trial.tracked.cpu().numpy(),
              fps=np.float32(trial.time.fps), video_width=np.int32(width), video_height=np.int32(height),
              drift_pre=np.full(undetected, np.nan, np.float32), auto_labels=np.full(undetected, -1, np.int8))


def placed(trial: Trial, identity: bh.Identity) -> torch.Tensor:
    """The fit of ``kinematics_1``: the trajectory ``(1, T, 211)`` of ``identity`` that the three cameras see.

    The phases are the pipeline's. The boards' labels are there from the start, so loaded limbs are held still as
    soon as all joints move.
    """
    config, human = trial.config, trial.human
    options, root = config["stages"]["kinematics_1"], config["stages"]["root"]
    start = refine_kinematics.smooth_fingers(
        trial.start(), trial.tracked.cpu().numpy(), options["finger_smooth_s"] * trial.time.fps)
    # A trial has no depth points, so its problem has no chamfer term for the tables to weight.
    weights, placing = ({name: weight for name, weight in table.items() if name != "chamfer"}
                        for table in (options["weights"], root["weights"]))
    free = first(BODY, human) if options["freeze_fingers"] else None
    iterations, look = options["iterations"], trial.look(identity)
    result = optim.run(trial.problem(identity, start).with_(data={"labels": trial.loaded()}), [
        fit.adam("root", root["iterations"], config, weights=placing, free=first(ROOT, human)),
        fit.adam("full", iterations["full"], config, weights=weights | dict.fromkeys(SCENE_TERMS, 0.0), free=free),
        fit.adam("refine", iterations["refine"], config, weights=weights, free=free, on_start=look),
        fit.lbfgs("polish", iterations["polish"], config, weights=weights, free=free, on_start=look),
    ])
    return objective.posed(identity, result.x["q"]).q


def kinematics_2(clip: paths.Clip, human: bh.HumanModel, config: dict) -> None:
    """Seat the body of ``kinematic_1.npz`` on the contacts of ``contacts_1.npz`` and write ``kinematic_2.npz``."""
    trial, options = Trial(clip, human, config), config["stages"]["kinematics_2"]
    start = files.load_npz(clip.find(paths.KINEMATIC_1), "bvr-kinematics")
    labels = trial.tensor(_contacts(clip, human)["frame_contact"]) & trial.tracked[..., None]
    identity = body.bake(human, trial.tensor(start["betas"]))
    look = trial.look(identity, labels)
    result = optim.run(trial.problem(identity, trial.tensor(start["q"])).with_(data={"labels": labels}), [
        fit.adam(name, options["iterations"][name], config, weights=options["weights"], free=free, on_start=look)
        for name, free in (("root", first(ROOT, human)), ("full", None), ("refine", None))])
    body.save(clip.path(paths.KINEMATIC_2), "bvr-kinematics", identity, objective.posed(identity, result.x["q"]).q,
              **{key: start[key] for key in solve_dynamics.HEADER})


def kindyn_1(clip: paths.Clip, human: bh.HumanModel, config: dict) -> None:
    """Fit the body of ``kinematic_2.npz`` and the forces at the boards' contacts together; write ``kindyn_1.npz``.

    The two solves are the pipeline's (``bvr.stages.solve_dynamics``); the labels are those of ``contacts_1``,
    which the boards decided.
    """
    trial, options = Trial(clip, human, config), config["stages"]["kindyn_1"]
    tensor, time = trial.tensor, trial.time
    start = files.load_npz(clip.find(paths.KINEMATIC_2), "bvr-kinematics")
    contacts = _contacts(clip, human)
    down = files.load_npz(clip.find(paths.GRAVITY), "bvr-gravity")["gravity_world"]
    q, held = tensor(start["q"]), tensor(contacts["frame_contact"]) & trial.tracked[..., None]
    identity = dynamics.under_gravity(body.bake(human, tensor(start["betas"])), tensor(down))
    skin = body.frame_normals(body.contact_frames(objective.posed(identity, q)))
    edges = dynamics.cone_edges(skin, config["friction"]["half_angle_deg"]) * held[..., None, None]
    physics = {"edges": edges, "held": held, "tracked": trial.tracked, "time": time,
               "multipliers": config["torque_multipliers"]}
    weights, iterations = options["weights"], options["iterations"]

    forces_alone = optim.Problem({"activation": torch.zeros(*held.shape, 4, device=q.device)})
    dynamics.register(forces_alone, identity, **physics, fixed=objective.posed(identity, q))
    warm = optim.run(forces_alone, [fit.lbfgs(
        "forces", iterations["forces"], config, history=options["forces_history"],
        gradient_tolerance=options["gradient_tolerance"],
        weights={name: weights[name] for name in dynamics.TERMS})]).x["activation"]
    problem = trial.problem(identity, q, variables={"activation": warm})
    dynamics.register(problem, identity, **physics)
    counted = dynamics.counted_rows(time, trial.tracked, held)
    problem.scalar("board_force", board_force(human, tensor(contacts["limb_force_world"]), counted,
                                              dynamics.weight(identity)), weight=0.0, reads=("forces",))
    result = optim.run(problem.with_(data={"labels": held}), [
        fit.lbfgs("joint", iterations["joint"], config, weights=weights, on_start=trial.look(identity, held))])
    solve_dynamics.save(clip, start, identity, objective.posed(identity, result.x["q"]).q, result.x["activation"],
                        edges, held, down, time)


def wall_penetration(human: bh.HumanModel, normal: torch.Tensor, point: torch.Tensor, vertices: int) -> Term:
    """Term: nothing of the body is behind the wall, in m^2: the squared depth behind its plane, summed over the
    points of a frame and averaged over the frames.

    The plane passes through ``point (3,)`` with ``normal (3,)`` out of the wall; the holds stand in front of it,
    so a gripping finger costs nothing. The points are ``vertices`` of the mesh, evenly drawn, and the contact
    frames. Every buried point pays its own depth: a hand 3 cm inside is not diluted by the points that are fine.
    """
    count = human.spec.num_vertices
    sample = (torch.arange(vertices, device=normal.device) * (count / vertices)).long().clamp_max(count - 1).unique()

    def loss(body: bh.HumanState, frames: torch.Tensor) -> torch.Tensor:
        points = torch.cat((body.points(sample), frames[..., :3]), dim=-2)
        return torch.relu((point - points) @ normal).square().sum(dim=-1).mean(dim=1)
    return loss


def board_force(human: bh.HumanModel, measured: torch.Tensor, counted: torch.Tensor, weight: torch.Tensor) -> Term:
    """Term: each limb's summed force against its board's, scaled like the root residual.

    ``measured (1, T, 4, 3)`` are the boards' forces on the limbs in newtons (``bvr_datasets._rigs.LIMBS``), NaN
    where a board gave none; ``counted (1, T)`` the frames whose inverse dynamics count and ``weight (1,)`` the
    body weight in newtons. The term reads ``forces (1, T, F, 3)``.
    """
    limbs = [_rigs.LIMBS.index(_rigs.limb_of_frame(name)) for name in human.contact_frames.names]
    member = torch.nn.functional.one_hot(torch.tensor(limbs, device=measured.device), len(_rigs.LIMBS)).to(measured)
    scale = 0.5 / (counted.sum(dim=1).clamp_min(1) * weight.square())
    counted = counted & measured.isfinite().all(dim=(2, 3))
    target = torch.nan_to_num(measured)

    def loss(forces: torch.Tensor) -> torch.Tensor:
        error = torch.einsum("ptfd,fl->ptld", forces, member) - target
        return scale * (error.square().sum(dim=(2, 3)) * counted).sum(dim=1)
    return loss


def shape_file(clip: paths.Clip) -> Path:
    """The frozen shape of the trial's climber (``betas (10,)``): this run's ``_shapes/<A|B>.npz``, else the
    upstream's. A trial is named after its climber, ``a_...`` or ``b_...``."""
    name = f"_shapes/{clip.name[0].upper()}.npz"
    own = clip.directory.parent / name
    return own if own.exists() or clip.upstream is None else clip.upstream.parent / name


def chain(arguments: Sequence[str]) -> int:
    """Run the steps of the ground truth on their trials; return the process exit code."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--steps", default=",".join(STEPS))
    options, rest = parser.parse_known_args(arguments)
    steps = options.steps.split(",")
    if unknown := sorted(set(steps) - set(STEPS)):
        parser.error(f"unknown step(s) {', '.join(unknown)}; the steps are {', '.join(STEPS)}")

    def load(stage: run.Stage) -> Callable[[paths.Clip], None]:
        human = _rigs.human(stage.device)
        if stage.step == "contacts_1":
            return lambda clip: measured_contacts(clip, human, paths.KINEMATIC_1, paths.CONTACTS_1, SENSORS)
        process = {"kinematics_1": kinematics_1, "kinematics_2": kinematics_2, "kindyn_1": kindyn_1}[stage.step]
        return lambda clip: process(clip, human, stage.config)

    modules = {step.name: step.module for step in run.STEPS}
    return max([run.stage(modules[step], load, [*rest, "--step", step]) for step in steps], default=0)


def _shared(shares: Sequence[float], terms: Sequence[Term]) -> Term:
    """Term: the cameras' terms of one kind as one, each with its share of the weight."""
    return lambda body: sum(share * term(body) for share, term in zip(shares, terms))


def _contacts(clip: paths.Clip, human: bh.HumanModel) -> dict[str, np.ndarray]:
    """``contacts_1.npz`` of the trial, which must name the body's contact frames."""
    contacts = files.load_npz(clip.find(paths.CONTACTS_1), "bvr-contacts")
    if list(contacts["contact_frame_names"]) != list(human.contact_frames.names):
        raise ValueError(f"{clip.find(paths.CONTACTS_1)} names other contact frames than the body; rerun contacts_1")
    return contacts


if __name__ == "__main__":
    sys.exit(chain(sys.argv[1:]))
