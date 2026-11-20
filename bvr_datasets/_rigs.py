"""What the three force rigs share: their limbs, a body with only the contact frames a rig measures, and contact labels
read off the rig.

The peter climbing boards, the Parkour plates and bar, and the OpenCap floor plates measure one force per limb. Each
dataset writes it beside its run as ``<clip>/human_optim/sensor_forces.npz``: ``reaction_world (N, 4, 3)`` in
:data:`LIMBS` order, the force on the person, NaN where the rig did not watch that limb, and ``limbs``.

A rig measures the hands and feet only, so contact found on a knee or the torso would route force where nothing
measures it. The rig runs therefore use a body whose contact frames are the rig's (:func:`human`) in every
human step: detection, labels and forces (``bvr_datasets/peter/chain.py`` runs them).
"""
from __future__ import annotations

from collections.abc import Collection, Sequence

import better_human as bh
import numpy as np
import torch

from bvr import body
from bvr.contacts import detector

#: The rig's channels: every ``sensor_forces.npz`` holds them in this order.
LIMBS = ("left_hand", "right_hand", "left_foot", "right_foot")
#: Contact frames of a hand and of a foot, by name without the side.
HAND_FRAMES = ("palm", "palm_distal", "finger", "thumb")
FOOT_FRAMES = ("heel", "ball_inner", "ball_outer", "big_toe", "toe_above")
#: The 18 contact frames of the hands and feet.
LIMB_FRAMES = tuple(f"{frame}_{side}" for frame in HAND_FRAMES + FOOT_FRAMES for side in ("left", "right"))
#: Where a dataset puts its measured forces, relative to a clip's tree.
SENSOR_FORCES = "human_optim/sensor_forces.npz"

#: A limb is in contact from this measured force on (N) until it drops below the second (hysteresis), and its
#: contact frames within this distance of the scene (m) for a third of a loaded span carry the label.
ENTER_N, EXIT_N, NEAR_M = 50.0, 30.0, 0.05


def limb_of_frame(name: str) -> str | None:
    """``palm_distal_left`` -> ``left_hand``; ``big_toe_right`` -> ``right_foot``; any other frame -> ``None``."""
    stem, _, side = name.rpartition("_")
    return f"{side}_hand" if stem in HAND_FRAMES else f"{side}_foot" if stem in FOOT_FRAMES else None


def limb_members(joint_names: Sequence[str]) -> list[list[int]]:
    """Joint columns per limb, in :data:`LIMBS` order: a hand is the wrist and its fingers, a foot the foot and ankle.

    A name the list does not hold is skipped, so the same rule folds the body's 52 joints and a shorter list.
    """
    parts = {"hand": ("wrist", "index", "middle", "pinky", "ring", "thumb"), "foot": ("foot", "ankle")}
    return [[column for column, name in enumerate(joint_names)
             if name.startswith(f"{side}_") and any(part in name for part in parts[limb])]
            for side, limb in (name.split("_") for name in LIMBS)]


def human(device: torch.device | str, frames: Collection[str] = LIMB_FRAMES) -> bh.HumanModel:
    """The pipeline's body (``bvr.body.load``) with only the contact frames ``frames``, in the shipped order."""
    shipped = bh.io.shipped_frame_set("smplx_contacts")
    return bh.HumanModel.from_spec(body.load(device).spec,
                                   contact_frames={name: frame for name, frame in shipped.items() if name in frames})


def limb_contact(sensor_forces: dict[str, np.ndarray], frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-limb contact ``(N, 4)`` bool and the measured forces ``(N, 4, 3)`` of the clip's first ``frames`` frames.

    A sample the rig did not take reads as no force, so as no contact.
    """
    if tuple(str(limb) for limb in sensor_forces["limbs"]) != LIMBS:
        raise ValueError(f"sensor channels are {list(sensor_forces['limbs'])}, expected {list(LIMBS)}")
    force = np.asarray(sensor_forces["reaction_world"], np.float32)[:frames]
    if len(force) != frames:
        raise ValueError(f"the sensors cover {len(force)} frames, the body {frames}")
    size = np.nan_to_num(np.linalg.norm(force, axis=-1))
    return detector.hysteresis(size >= ENTER_N, size >= EXIT_N), force


def frame_contact(limb_on: np.ndarray, valid: np.ndarray, distance: np.ndarray,
                  names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Spread each loaded limb's measured contact onto its own contact frames: ``(contact, rescued)``, ``(N, F)``.

    The sensor says the limb is loaded; the body says which of its frames is on the scene. The label holds for the
    whole loaded span on the frames within :data:`NEAR_M` of the scene for at least a third of it, because a set that
    changed from frame to frame made the terms that seat and hold a contact chase a moving target. A span with no
    such frame falls back to its nearest one and is flagged ``rescued``: there the reconstruction put the limb off
    the scene a sensor says it was on.

    :param limb_on: ``(N, 4)`` measured contact per limb.
    :param valid: ``(N,)`` frames where the body is tracked.
    :param distance: ``(N, F)`` distance of each contact frame to the scene (m).
    :param names: the ``F`` contact-frame names.
    """
    owners = [limb_of_frame(name) for name in names]
    contact, rescued = np.zeros(distance.shape, bool), np.zeros(distance.shape, bool)
    for limb, loaded in zip(LIMBS, limb_on.T):
        columns = [column for column, owner in enumerate(owners) if owner == limb]
        for start, stop in detector.runs(loaded & valid) if columns else ():
            near = [column for column in columns if (distance[start:stop, column] <= NEAR_M).mean() >= 1 / 3]
            if not near:
                near = [min(columns, key=lambda column: np.median(distance[start:stop, column]))]
                rescued[start:stop, near] = True
            contact[start:stop, near] = True
    return contact, rescued
