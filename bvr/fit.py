"""What the fitting steps share: a clip's evidence on the device, and the phases that solve its problem.

``kinematics_1``, ``kinematics_2`` and ``kindyn_1`` fit the same people to the same files of a clip: its
cameras, its static scene and its Sapiens keypoints. :class:`Evidence` reads them once, names the people a
step can fit and builds their problem (:func:`bvr.terms.objective.kinematics_problem`). :func:`adam` and
:func:`lbfgs` are the two kinds of phase a step runs; each logs every person's loss and terms as it goes.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

import better_human as bh
import numpy as np
import torch
from better_robot import optim

from bvr import body, camera, files, paths
from bvr.geometry.normals import load_scene_cloud
from bvr.terms import motion, objective, scene, visual

log = logging.getLogger(__name__)


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

    def problem(self, identity: bh.Identity, start: torch.Tensor, attraction: dict, **more) -> optim.Problem:
        """The kinematic problem of the people, whose bodies are ``identity`` and start at ``start (P, T, 211)``.

        ``more`` is ``cloud=`` or ``variables=`` of :func:`bvr.terms.objective.kinematics_problem`.
        """
        rows = [list(self.keypoints["object_ids"]).index(person) for person in self.object_ids]
        return objective.kinematics_problem(
            identity, start, keypoints=self.tensor(self.keypoints["keypoints"][rows]),
            scores=self.tensor(self.keypoints["keypoint_scores"][rows]), tracked=self.tracked, cameras=self.cameras,
            time=self.time, attraction=attraction, config=self.config, **more)

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


def adam(name: str, iterations: int, config: dict, **phase) -> optim.Phase:
    """A phase of Adam at the config's learning rate, cut by its decay at one half and at three quarters."""
    rate, decay = config["optimizer"]["learning_rate"], config["optimizer"]["learning_rate_decay"]
    solver = optim.TorchOptim(
        lambda parameters: torch.optim.Adam(parameters, lr=rate),
        scheduler=lambda optimizer: torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [iterations // 2, 3 * iterations // 4], gamma=decay))
    return optim.Phase(solver, iterations, name=name, on_step=_report(name, iterations), stop_every=0, **phase)


def lbfgs(name: str, iterations: int, config: dict, *, history: int | None = None,
          gradient_tolerance: float | None = None, **phase) -> optim.Phase:
    """A phase of BetterRobot's L-BFGS, which keeps one memory and one line search per person.

    It ends early once no person's search finds a step. ``history`` defaults to ``[optimizer].lbfgs_history``
    and ``gradient_tolerance`` to the solver's own.
    """
    search = config["optimizer"]
    solver = optim.LBFGS(history=history or search["lbfgs_history"], line_search=search["lbfgs_line_search"],
                         gradient_tolerance=gradient_tolerance)
    return optim.Phase(solver, iterations, name=name, on_step=_report(name, iterations), **phase)


def _report(name: str, iterations: int) -> Callable[[optim.Problem, optim.State], None]:
    """A phase's ``on_step``: log each person's loss and terms at every quarter of the phase and at its end."""
    def on_step(problem: optim.Problem, state: optim.State) -> None:
        step = state.iteration - 1
        if step % max(1, iterations // 4) == 0 or step == iterations - 1 or bool(state.done.all()):
            values = {term: loss / problem.weights[term] for term, loss in state.extra["terms"].items()}
            numbers = "  ".join(f"{term} {_numbers(value)}" for term, value in values.items())
            log.info("%-6s %4d  loss %s  %s", name, step, _numbers(state.loss), numbers)
    return on_step


def _numbers(values: torch.Tensor) -> str:
    return "/".join(f"{value:.4g}" for value in values.tolist())
