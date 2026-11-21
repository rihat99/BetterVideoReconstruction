"""One body shape per climber of climb_wall_3, from their measured height and mass and the three cameras.

The ground-truth steps (``bvr_datasets/peter/gt/fit_multiview.py``) fit a climber with one frozen shape. Here it
is found. A trial is first fitted at the shape the converter gave it (the fit of ``kinematics_1``). Then pose and
shape move together against the keypoints of every camera, the climber's height (the shaped body at rest), their
mass (the mass of the body the dynamics will move) and, for the directions those leave free, the converter's
shape. A climber's shape is the mean over the trials given, written to ``<run>/_shapes/<A|B>.npz``::

    uv run python -m bvr_datasets.peter.gt.shape --source <videos> --save-dir <run> --upstream <trees> \\
        --config bvr_datasets/peter/gt/gt.toml --clips a_00_01_part3,a_16_01_part2,a_32_01_part3

``--source``, ``--save-dir`` and ``--upstream`` are those of ``fit_multiview``, which then reads the shapes.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import better_human as bh
import numpy as np
import torch
from better_robot import optim, terms

from bvr import body, config as configuration, fit, paths
from bvr.stages.refine_kinematics import BODY, first
from bvr.terms import dynamics, visual
from bvr_datasets import _rigs
from bvr_datasets.peter.gt.fit_multiview import Trial, placed

log = logging.getLogger(__name__)

#: The two climbers, by the first letter of a trial's name: height (m) and mass (kg), as measured.
CLIMBERS = {"a": (1.80, 72.5), "b": (1.60, 48.0)}
#: What a squared metre of height and a squared kilogram of mass cost (1 cm costs 10, 1 kg costs 1), and a squared
#: unit of distance from the converter's shape.
WEIGHTS = {"shape_height": 1e5, "shape_mass": 1.0, "shape_prior": 0.1}


def height(identity: bh.Identity) -> torch.Tensor:
    """The height ``(P,)`` in metres of bodies (:func:`bvr.body.bake`) at rest: the vertical extent of the mesh."""
    vertical = identity.rest_vertices[:, 0, :, 1]
    return vertical.amax(dim=-1) - vertical.amin(dim=-1)


def mass(identity: bh.Identity) -> torch.Tensor:
    """The mass ``(P,)`` in kg of bodies: that of the parts the dynamics move."""
    return dynamics.weight(identity) / dynamics.GRAVITY


def fitted(trial: Trial) -> torch.Tensor:
    """The shape ``(10,)`` of the trial's climber."""
    human, config = trial.human, trial.config
    tall, heavy = CLIMBERS[trial.clip.name[0]]
    seen = torch.stack([view.tracked.sum() for view in trial.views])
    converted = torch.stack([view.estimate.identity.values["betas"][0, 0] for view in trial.views])
    converted = (converted * seen[:, None]).sum(dim=0, keepdim=True) / seen.sum()
    start = placed(trial, body.bake(human, converted))

    problem = optim.Problem({"q": start, "betas": converted}, time=("q",))
    problem.cached(terms.RobotManifold(human.model).project, name="pose", reads=("q",))
    problem.cached(lambda betas, pose: body.bake(human, betas).forward(pose), name="body")
    problem.scalar("views_keypoints", trial.keypoints)
    problem.scalar("pose_prior", visual.pose_prior(start))
    problem.scalar("orientation_prior", visual.orientation_prior(start))
    problem.scalar("shape_height", lambda body: (height(body.identity) - tall).square())
    problem.scalar("shape_mass", lambda body: (mass(body.identity) - heavy).square())
    problem.scalar("shape_prior", lambda betas: (betas - converted).square().sum(dim=-1))

    options = config["stages"]["kinematics_1"]
    weights = WEIGHTS | {name: options["weights"][name]
                         for name in ("views_keypoints", "pose_prior", "orientation_prior")}
    free = first(BODY, human) if options["freeze_fingers"] else None
    result = optim.run(problem, [
        fit.adam("shape", options["iterations"]["refine"], config, weights=weights, free=free),
        fit.lbfgs("polish", options["iterations"]["polish"], config, weights=weights, free=free)])
    return result.x["betas"][0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="a folder of links <trial>.mp4")
    parser.add_argument("--save-dir", type=Path, required=True, help="the run root; shapes go to <save-dir>/_shapes/")
    parser.add_argument("--upstream", type=Path, help="the trials' trees")
    parser.add_argument("--config", type=Path, required=True, help="bvr_datasets/peter/gt/gt.toml")
    parser.add_argument("--clips", required=True, help="comma-separated trials; each climber's shape is their mean")
    options = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    config = configuration.load(options.config)
    torch.set_num_threads(config["run"]["threads"])
    human = _rigs.human(config["run"]["device"])
    clips = {clip.name: clip for clip in paths.clips(options.source, options.save_dir, options.upstream)}
    names = options.clips.split(",")
    files = {climber: options.save_dir / "_shapes" / f"{climber.upper()}.npz"
             for climber in sorted({name[0] for name in names})}
    if kept := [str(file) for file in files.values() if file.exists()]:
        parser.error(f"{', '.join(kept)} exist; a shape is fitted once (move the file away to fit it again)")
    shapes: dict[str, dict[str, torch.Tensor]] = {climber: {} for climber in files}
    for name in names:
        log.info("shape: %s", name)
        shapes[name[0]][name] = fitted(Trial(clips[name], human, config))
    for climber, trials in shapes.items():
        betas = torch.stack(list(trials.values())).mean(dim=0)
        identity = body.bake(human, betas[None])
        written = files[climber]
        written.parent.mkdir(parents=True, exist_ok=True)
        np.savez(written, betas=betas.cpu().numpy(), clips=np.array(list(trials)),
                 height_m=np.float32(height(identity).item()), mass_kg=np.float32(mass(identity).item()))
        log.info("%s: height %.3f m, mass %.1f kg from %d trial(s) -> %s", climber.upper(),
                 height(identity).item(), mass(identity).item(), len(trials), written)


if __name__ == "__main__":
    main()
