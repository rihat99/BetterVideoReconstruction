"""Steps ``kinematics_1`` and ``kinematics_2``: fit every person's body trajectory to the video.

``kinematics_1`` starts from the scale stage's pre-fit bodies (the converter's, where there is no
pre-fit) and fits them to the keypoints, the depth points and the scene, in four phases::

    root    the root pose alone finds its place
    full    all joints, still without the scene
            contacts are detected once on this body, before any contact term has acted, and frozen
    refine  the scene keeps contact frames out of it and holds the detected contacts on it
    polish  L-BFGS

``kinematics_2`` starts from that result and from the labels of ``contacts_1``. In each of its
phases (root, full, refine) a labelled contact frame is pulled to the wall the camera sees at its
own pixel, which seats a body that hovered. The people of a clip are one batch: one problem, one
loss per person. A person without a tracked frame or without keypoints is written as they came.

Both steps write the same body file (``bvr.body.save``). ``drift_pre`` and ``auto_labels`` are what
``kinematics_1`` detected: how far each contact frame moved across the stillness window before
contact was enforced, and the labels it froze (1, or -1 for none). ``kinematics_2`` hands both on
unchanged, because a drift measured after contact terms held a frame still would confirm the
labels that held it.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable

import better_human as bh
import numpy as np
import torch
from better_robot import optim

from bvr import body, camera, files, paths, run
from bvr.geometry.normals import load_scene_cloud
from bvr.human_optim.contacts import detect_frame_contacts, joint_drift
from bvr.terms import motion, objective, scene, visual

log = logging.getLogger(__name__)

POSE_FIELDS = ("global_orient", "transl", "body_pose", "left_hand_pose", "right_hand_pose")
#: Coordinates of ``q``: the root pose comes first, then the body joints; the fingers are the rest.
ROOT, BODY = 7, 7 + 4 * (body.BODY_JOINTS - 1)


def load(stage: run.Stage) -> Callable[[paths.Clip], None]:
    human = body.load(stage.device)
    step = {"kinematics_1": kinematics_1, "kinematics_2": kinematics_2}[stage.step]
    return lambda clip: step(clip, human, stage.config)


def kinematics_1(clip: paths.Clip, human: bh.HumanModel, config: dict) -> None:
    """Fit the pre-fit (or converter) bodies of ``clip`` to its video and write ``kinematic_1.npz``."""
    prefit = clip.find(paths.PREFIT)
    start = (files.load_npz(prefit, "bvr-prefit") if prefit.exists()
             else files.load_npz(clip.find(paths.SMPLX_PARAMS), "bvr-smplx"))
    evidence = Evidence(clip, human, config, start)
    options, tensor = config["stages"]["kinematics_1"], evidence.tensor
    tracked = start["valid_mask"].astype(bool)
    # One shape per person, the mean over their tracked frames; an untracked frame holds the nearest tracked pose.
    betas = np.where(tracked[..., None], start["betas"], 0.0).sum(axis=1) / np.maximum(tracked.sum(axis=1), 1)[:, None]
    betas = tensor(betas.astype(np.float32))
    identity, q, _ = bh.from_classic(human, bh.families.SMPLXClassic(betas=betas[:, None], **{
        name: tensor(np.nan_to_num(_held(start[name], tracked))) for name in POSE_FIELDS}))
    q = body.to_world(q, evidence.cameras[0])
    q = _smooth_fingers(q, tracked, options["finger_smooth_s"] * evidence.time.fps)
    drift = torch.full((*tracked.shape, len(human.contact_frames.names)), torch.nan, device=q.device)
    labels = torch.zeros_like(drift, dtype=torch.bool)

    if len(evidence.people):
        fitted, own = body.bake(human, betas[evidence.people]), q[evidence.people]
        problem = evidence.problem(fitted, own, config["auto_contact"], cloud=evidence.depth_points(fitted, own))
        iterations, weights, root = options["iterations"], options["weights"], config["stages"]["root"]
        free = _first(BODY, human) if options["freeze_fingers"] else None
        placed = optim.run(problem, [
            _adam("root", root["iterations"], config, weights=root["weights"], free=_first(ROOT, human)),
            _adam("full", iterations["full"], config, free=free,
                  weights=weights | dict.fromkeys(objective.CONTACT_TERMS, 0.0)),
        ])
        labels[evidence.people], drift[evidence.people] = _detect_contacts(
            objective.posed(fitted, placed.x["q"]), evidence, config)
        look = evidence.scene_refresh(fitted)
        result = optim.run(problem.with_(data={"labels": labels[evidence.people]}), [
            _adam("refine", iterations["refine"], config, weights=weights, free=free, on_start=look),
            _lbfgs("polish", iterations["polish"], config, weights=weights, free=free, on_start=look),
        ], x=placed.x)
        q[evidence.people] = objective.posed(fitted, result.x["q"]).q

    body.save(clip.path(paths.KINEMATIC_1), "bvr-kinematics", identity, q, **evidence.header(start),
              drift_pre=drift.cpu().numpy(), auto_labels=np.where(labels.cpu().numpy(), 1, -1).astype(np.int8))


def kinematics_2(clip: paths.Clip, human: bh.HumanModel, config: dict) -> None:
    """Seat the bodies of ``kinematic_1.npz`` on the contacts of ``contacts_1.npz`` and write ``kinematic_2.npz``."""
    start = files.load_npz(clip.find(paths.KINEMATIC_1), "bvr-kinematics")
    contacts = files.load_npz(clip.find(paths.CONTACTS_1), "bvr-contacts")
    if list(contacts["contact_frame_names"]) != list(human.contact_frames.names):
        raise ValueError(f"{clip.find(paths.CONTACTS_1)} names other contact frames than the body; rerun contacts_1")
    evidence = Evidence(clip, human, config, start)
    options, tensor = config["stages"]["kinematics_2"], evidence.tensor
    betas, q = tensor(start["betas"]), tensor(start["q"])

    if len(evidence.people):
        # A person the contact readout does not list has no labels and is fitted without them.
        listed = {person: row for row, person in enumerate(contacts["object_ids"])}
        nowhere = np.zeros(contacts["frame_contact"].shape[1:], bool)
        labels = tensor(np.stack([contacts["frame_contact"][listed[person]] if person in listed else nowhere
                                  for person in evidence.object_ids])) & evidence.tracked[..., None]
        fitted = body.bake(human, betas[evidence.people])
        problem = evidence.problem(fitted, q[evidence.people], config["contact_attraction"])
        look = evidence.scene_refresh(fitted, labels)
        result = optim.run(problem.with_(data={"labels": labels}), [
            _adam(name, options["iterations"][name], config, weights=options["weights"], free=free, on_start=look)
            for name, free in (("root", _first(ROOT, human)), ("full", None), ("refine", None))])
        q[evidence.people] = objective.posed(fitted, result.x["q"]).q

    body.save(clip.path(paths.KINEMATIC_2), "bvr-kinematics", body.bake(human, betas), q, **evidence.header(start),
              drift_pre=start["drift_pre"], auto_labels=start["auto_labels"])


class Evidence:
    """What a clip's files say about the people a step can fit, on the device.

    ``people`` are the rows of the start file with a tracked frame and Sapiens keypoints, and ``tracked (P, T)``
    their tracked frames; ``cameras`` and ``time`` are the clip's, ``scene`` its static cloud.
    """

    def __init__(self, clip: paths.Clip, human: bh.HumanModel, config: dict, start: dict[str, np.ndarray]) -> None:
        self.clip, self.human, self.config, device = clip, human, config, human.spec.device
        self.tensor = lambda array: torch.as_tensor(array, device=device)
        transform = files.load_npz(clip.find(paths.TRANSFORM), "bvr-transform")
        self.cameras = self.tensor(transform["extrinsics"]), self.tensor(transform["intrinsics_px_orig"])
        self.time = motion.Time(float(transform["fps"]))
        self.image_size = self.tensor([int(transform["image_width"]), int(transform["image_height"])])
        self.scene = scene.Scene(*load_scene_cloud(
            files.load_npz(clip.find(paths.SCENE), "bvr-scene"), camera.centers(self.cameras[0]),
            conf_drop_q=config["penetration"]["conf_drop_q"], k=config["penetration"]["normal_k"], device=device))
        self.keypoints = files.load_npz(clip.find(paths.SAPIENS_POSE), "bvr-keypoints")
        tracked = start["valid_mask"].astype(bool)
        self.people = np.flatnonzero(tracked.any(axis=1) & np.isin(start["object_ids"], self.keypoints["object_ids"]))
        self.object_ids, self.tracked = start["object_ids"][self.people], self.tensor(tracked[self.people])

    def depth_points(self, identity: bh.Identity, start: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Each person's depth points for the chamfer, ``(P, T, M, 3)`` and their mask, sampled around ``start``."""
        points = files.load_npz(self.clip.find(paths.HUMAN), "bvr-human-points")
        pixels = camera.project(objective.posed(identity, start).joint_translations, *self.cameras)
        sampling = {key: self.config["chamfer"][key] for key in ("conf_drop_q", "max_points", "limb_quota")}
        clouds = []
        for person, start_pixels, seen in zip(self.object_ids, pixels, self.tracked.cpu().numpy()):
            own = points["obj_id"] == person
            clouds.append(visual.human_points(
                self.human, points["points"][own], points["confidence"][own], points["frame_id"][own],
                start_pixels, seen, self.cameras, **sampling))
        return tuple(self.tensor(np.stack(part)) for part in zip(*clouds))

    def problem(self, identity: bh.Identity, start: torch.Tensor, attraction: dict, cloud=None) -> optim.Problem:
        """The kinematic problem of the people, whose bodies are ``identity`` and start at ``start (P, T, 211)``."""
        rows = [list(self.keypoints["object_ids"]).index(person) for person in self.object_ids]
        return objective.kinematics_problem(
            identity, start, keypoints=self.tensor(self.keypoints["keypoints"][rows]),
            scores=self.tensor(self.keypoints["keypoint_scores"][rows]), tracked=self.tracked, cameras=self.cameras,
            time=self.time, attraction=attraction, config=self.config, cloud=cloud)

    def scene_refresh(self, identity: bh.Identity, labels: torch.Tensor | None = None):
        """A phase's ``on_start``: the scene candidates around the standing body and, given ``labels``, the patches."""
        prune = {key: self.config["penetration"][key] for key in ("prune_margin_m", "max_candidates")}
        patch = {key: self.config["contact_acquisition"][key] for key in ("radius_px", "patch_k", "conf_thresh")}

        def on_start(problem: optim.Problem, x: dict[str, torch.Tensor]) -> tuple[dict, dict]:
            state = objective.posed(identity, x["q"])
            data = {"scene": scene.candidates(self.scene, objective.reach(state), self.tracked, **prune)}
            if labels is not None:
                data["patches"] = scene.patches(
                    self.scene, body.contact_frames(state), labels, self.cameras, self.image_size, **patch)
                log.info("contact patches: %d of %d labelled contacts resolved",
                         len(data["patches"].person), int(labels.sum()))
            return x, data
        return on_start

    def header(self, start: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """The arrays every body file of the clip carries beside the bodies."""
        width, height = self.image_size.tolist()
        return {"object_ids": start["object_ids"], "valid_mask": start["valid_mask"],
                "fps": np.float32(self.time.fps), "video_width": np.int32(width), "video_height": np.int32(height)}


def _detect_contacts(state: bh.HumanState, evidence: Evidence, config: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Where each person's contact frames touch the scene, ``(P, T, F)`` bool, and their drift ``(P, T, F)`` in m.

    A frame touches where it is near the scene and still. The drift is the distance a frame covers across the
    stillness window; it is measured here, on the body no contact term has touched, and never again.
    """
    gate, auto, fps = config["contact_gate"], config["auto_contact"], evidence.time.fps
    names = evidence.human.contact_frames.names
    labels, drifts = [], []
    for frames, seen in zip(body.contact_frames(state), evidence.tracked):
        drift = joint_drift(frames[..., :3], fps=fps, window_s=gate["window_s"], valid_mask=seen)
        found = detect_frame_contacts(
            frames[..., :3], body.frame_normals(frames), drift, *evidence.scene, frame_names=names,
            close_thresh_m=auto["detect_thresh_m"], exit_thresh_m=auto["detect_exit_m"],
            min_frames=math.ceil(gate["detect_min_s"] * fps), still_m=gate["still_m"],
            still_override_m=gate["still_override_m"], max_dist_m=auto["max_dist_m"],
            conf_thresh=auto["conf_thresh"], prune_margin_m=auto["prune_margin_m"])
        labels.append(found.contact & seen[:, None])
        drifts.append(drift)
    labels = torch.stack(labels)
    log.info("detected contacts: %s", ", ".join(
        f"{name} {count}" for name, count in zip(names, labels.sum(dim=(0, 1)).tolist()) if count))
    return labels, torch.stack(drifts)


def _adam(name: str, iterations: int, config: dict, **phase) -> optim.Phase:
    """A phase of Adam at the config's learning rate, cut by its decay at one half and at three quarters."""
    rate, decay = config["optimizer"]["learning_rate"], config["optimizer"]["learning_rate_decay"]
    solver = optim.TorchOptim(
        lambda parameters: torch.optim.Adam(parameters, lr=rate),
        scheduler=lambda optimizer: torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [iterations // 2, 3 * iterations // 4], gamma=decay))
    return optim.Phase(solver, iterations, name=name, on_step=_report(name, iterations), stop_every=0, **phase)


def _lbfgs(name: str, iterations: int, config: dict, **phase) -> optim.Phase:
    """A phase of BetterRobot's L-BFGS, which keeps one memory and one line search per person; it ends early once
    no person's search finds a step."""
    solver = optim.LBFGS(history=config["optimizer"]["lbfgs_history"],
                         line_search=config["optimizer"]["lbfgs_line_search"])
    return optim.Phase(solver, iterations, name=name, on_step=_report(name, iterations), **phase)


def _first(coordinates: int, human: bh.HumanModel) -> dict[str, torch.Tensor]:
    """A phase's ``free``: the first ``coordinates`` of every frame's ``q`` move, the rest stay."""
    return {"q": torch.arange(human.model.nq, device=human.spec.device) < coordinates}


def _report(name: str, iterations: int) -> Callable[[optim.Problem, optim.State], None]:
    """A phase's ``on_step``: log each person's loss and terms at every quarter of the phase and at its end."""
    def on_step(problem: optim.Problem, state: optim.State) -> None:
        step = state.iteration - 1
        if step % max(1, iterations // 4) and step != iterations - 1:
            return
        values = {term: loss / problem.weights[term] for term, loss in state.extra["terms"].items()}
        log.info("%-6s %4d  loss %s  %s", name, step, _numbers(state.loss),
                 "  ".join(f"{term} {_numbers(value)}" for term, value in values.items()))
    return on_step


def _numbers(values: torch.Tensor) -> str:
    return "/".join(f"{value:.4g}" for value in values.tolist())


def _held(values: np.ndarray, tracked: np.ndarray) -> np.ndarray:
    """``values (P, N, ...)`` with each untracked frame replaced by its person's nearest tracked frame."""
    frames = np.arange(tracked.shape[1])
    distance = np.abs(frames[:, None] - np.where(tracked[:, None, :], frames, np.inf))
    nearest = distance.argmin(axis=-1).reshape(*tracked.shape, *[1] * (values.ndim - 2))
    return np.take_along_axis(values, nearest, axis=1)


def _smooth_fingers(q: torch.Tensor, tracked: np.ndarray, sigma: float) -> torch.Tensor:
    """``q (P, T, 211)`` with the finger quaternions smoothed over time by a Gaussian of ``sigma`` frames.

    The fingers are not fitted: they keep the converter's hand pose, and this removes its frame-to-frame jitter.
    Each run of tracked frames is filtered alone, with its ends repeated. A quaternion and its negative are one
    rotation, so a run is first brought to one hemisphere; the filtered components are normalised again.
    """
    if sigma <= 0:
        return q
    q = q.clone()
    for person, seen in enumerate(tracked):
        edges = np.flatnonzero(np.diff(seen.astype(np.int8), prepend=0, append=0))
        for first, last in zip(edges[::2], edges[1::2]):
            if last - first < 3:
                continue
            radius = min(round(3.0 * sigma), last - first - 1)
            kernel = torch.exp(-0.5 * (torch.arange(-radius, radius + 1, device=q.device) / sigma).square())
            fingers = q[person, first:last, BODY:].unflatten(-1, (-1, 4))
            flips = torch.where((fingers[1:] * fingers[:-1]).sum(dim=-1) < 0, -1.0, 1.0).cumprod(dim=0)
            signals = torch.cat((fingers[:1], fingers[1:] * flips[..., None])).flatten(1).T[:, None]
            padded = torch.nn.functional.pad(signals, (radius, radius), mode="replicate")
            smooth = torch.nn.functional.conv1d(padded, (kernel / kernel.sum())[None, None])[:, 0].T
            smooth = torch.nn.functional.normalize(smooth.unflatten(-1, (-1, 4)), dim=-1)
            q[person, first:last, BODY:] = smooth.flatten(1)
    return q


if __name__ == "__main__":
    run.main(load)
