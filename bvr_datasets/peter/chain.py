"""The human steps of a rig run, on a body with only the rig's contact frames; with ``--measured``, on the rig's labels.

A rig run is two commands over the same videos and run root. The steps up to gravity are the pipeline's own, with
the dataset's config delta; the human steps run here, on the body of ``bvr_datasets._rigs.human`` (the hands and
feet for the peter rigs, Parkour and OpenCap)::

    uv run python scripts/pipeline.py --source <videos> --save-dir <run> --config <delta>
    uv run python -m bvr_datasets.peter.chain --source <videos> --save-dir <run> --config <delta>

With ``--measured`` the two contact steps read the labels off the rig's sensors (``<clip>/human_optim/
sensor_forces.npz``, written by the dataset's ``sensor_forces.py``) instead of estimating them, and the steps after
them run on those labels. Which limb is loaded, and when, is the sensor's; which contact frames of that limb carry
the label is read on the body: those within 5 cm of the scene for a third of the loaded span, else the nearest,
flagged ``frame_rescued``. ``kinematics_1`` never sees a label, so it is not run again: the measured run goes to a
second root and reads it from the first::

    uv run python -m bvr_datasets.peter.chain --measured --source <videos> --save-dir <run 2> --upstream <run> \\
        --config <delta>

The forces are never supervised by the sensors: they stay a prediction to score against them. The other options are
those of a stage (``python -m bvr.stages.<module> --help``) without ``--step``; ``--steps`` picks some human steps.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Collection, Sequence

import better_human as bh
import numpy as np
import torch
from scipy.spatial import cKDTree

from bvr import body, files, paths, run
from bvr.stages import estimate_contacts, refine_kinematics, solve_dynamics
from bvr.terms import objective
from bvr_datasets import _rigs

log = logging.getLogger(__name__)

#: The human steps of the chain, in order.
STEPS = ("kinematics_1", "contacts_1", "kinematics_2", "contacts_2", "kindyn_1")
#: The body file each contact step reads, and the file it writes.
CONTACT_FILES = {"contacts_1": (paths.KINEMATIC_1, paths.CONTACTS_1), "contacts_2": (paths.KINEMATIC_2, paths.CONTACTS_2)}
LABEL_METHOD = ("boards-v3: measured |F| with hysteresis; per loaded span, on the limb's frames within 5 cm of the "
                "scene for >= 1/3 of the span, else its nearest")


def chain(arguments: Sequence[str], frames: Collection[str] = _rigs.LIMB_FRAMES) -> int:
    """Run the human steps with a body of only the contact frames ``frames``; return the process exit code."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--measured", action="store_true")
    parser.add_argument("--steps", default=",".join(STEPS))
    options, rest = parser.parse_known_args(arguments)
    steps = [step for step in options.steps.split(",") if not (options.measured and step == "kinematics_1")]
    if unknown := sorted(set(steps) - set(STEPS)):
        parser.error(f"unknown step(s) {', '.join(unknown)}; the human steps are {', '.join(STEPS)}")

    def load(stage: run.Stage):
        human = _rigs.human(stage.device, frames)
        if stage.step in CONTACT_FILES and options.measured:
            return lambda clip: measured_contacts(clip, human, *CONTACT_FILES[stage.step])
        if stage.step in CONTACT_FILES:
            return lambda clip: estimate_contacts.contacts(clip, human, stage.config, *CONTACT_FILES[stage.step])
        process = {"kinematics_1": refine_kinematics.kinematics_1, "kinematics_2": refine_kinematics.kinematics_2,
                   "kindyn_1": solve_dynamics.kindyn_1}[stage.step]
        return lambda clip: process(clip, human, stage.config)

    modules = {step.name: step.module for step in run.STEPS}
    return max([run.stage(modules[step], load, [*rest, "--step", step]) for step in steps], default=0)


def measured_contacts(clip: paths.Clip, human: bh.HumanModel, bodies: str, written: str) -> None:
    """Read the contact labels of the body file ``bodies`` off the clip's sensors and write them to ``written``.

    The file has the contacts schema the kinematic and dynamic steps read, plus what the sensors said.
    """
    fitted = files.load_npz(clip.find(bodies), "bvr-kinematics")
    names = list(human.contact_frames.names)
    if len(fitted["object_ids"]) != 1:
        raise ValueError(f"{clip.name} tracks {len(fitted['object_ids'])} people; a rig measures one and cannot "
                         "say which")
    if list(fitted["contact_frame_names"]) != names:
        raise ValueError(f"{clip.find(bodies)} has other contact frames than the body")
    if unowned := [name for name in names if _rigs.limb_of_frame(name) is None]:
        raise ValueError(f"no sensor covers the contact frames {unowned}")
    device = human.spec.device
    with torch.no_grad():
        state = objective.posed(body.bake(human, torch.as_tensor(fitted["betas"], device=device)),
                                torch.as_tensor(fitted["q"], device=device))
        positions = body.contact_frames(state)[0, ..., :3].cpu().numpy()
    scene = files.load_npz(clip.find(paths.SCENE), "bvr-scene")["points"]
    distance = cKDTree(scene).query(positions.reshape(-1, 3), workers=-1)[0].reshape(positions.shape[:2])
    with np.load(clip.find(_rigs.SENSOR_FORCES)) as sensors:
        limb_on, force = _rigs.limb_contact(dict(sensors), len(positions))
    contact, rescued = _rigs.frame_contact(limb_on, fitted["valid_mask"][0].astype(bool), distance, names)
    owners = np.array([_rigs.limb_of_frame(name) for name in names])
    for limb, loaded in zip(_rigs.LIMBS, limb_on.T):
        log.info("%s: %d loaded frames, %d of them off the scene", limb, loaded.sum(),
                 rescued[:, owners == limb].any(axis=1).sum())
    files.save_npz(
        clip.path(written), "bvr-contacts",
        contact_label_schema=np.int32(estimate_contacts.LABEL_SCHEMA), contact_label_method=np.array(LABEL_METHOD),
        frame_contact=contact[None], frame_rescued=rescued[None], frame_scene_dist_m=distance[None].astype(np.float32),
        limb_names=np.array(_rigs.LIMBS), limb_contact=limb_on[None], limb_force_world=force[None],
        contact_frame_names=np.array(names), frame_parent_joint=np.array(human.contact_frames.parent_joints),
        object_ids=fitted["object_ids"], valid_mask=fitted["valid_mask"], num_people=np.int32(1),
        num_frames=np.int32(len(positions)), fps=np.float32(fitted["fps"]))


if __name__ == "__main__":
    sys.exit(chain(sys.argv[1:]))
