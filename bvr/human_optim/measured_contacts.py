"""Contact labels MEASURED by a force rig instead of estimated from the video.

A rig with one force sensor per limb — the peter climbing boards, a lab's force plates —
measures contact: a limb touches while its sensor reads a force. The dataset's own script
writes ``sensor_forces.npz`` (``reaction_world (N, 4, 3)`` in :data:`LIMBS` order, the
reaction sense: force on the person). The sensors say *which limb* and *when*. They
cannot say which part of a hand or foot is touching, and the optimization puts its contact points on the finer contact
*frames* (palm, thumb, heel, big toe, ...). This module closes that gap in two steps:
:func:`limb_contact` reads the measured force into a per-limb on/off series, and
:func:`frame_contact` picks the frames of a loaded limb that the body actually holds
against the wall.

Shared by the single-view stage (``bvr/stages/measure_contacts.py``) and the peter
three-camera ground-truth chain (``datasets/peter/gt/fit_multiview.py``) so the two readings
cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bvr.human_optim.contacts import FOOT_FRAMES, HAND_FRAMES

#: Sensor channel order — every rig writes ``sensor_forces.npz`` in this order.
LIMBS = ["left_hand", "right_hand", "left_foot", "right_foot"]

#: A loaded limb's frame must sit within ``near_m`` of the wall for at least this
#: fraction of the span to carry the label (see :func:`frame_contact`).
_NEAR_FRACTION = 1.0 / 3.0


def limb_of_frame(name: str) -> str | None:
    """``palm_distal_left`` → ``left_hand``; ``big_toe_right`` → ``right_foot``; else ``None``."""
    base, side = name.rsplit("_", 1)
    if base in HAND_FRAMES:
        return f"{side}_hand"
    if base in FOOT_FRAMES:
        return f"{side}_foot"
    return None


def hysteresis(force_n: np.ndarray, enter: float, exit_: float) -> np.ndarray:
    """``(N,)`` bool contact from a force magnitude series with enter/exit thresholds."""
    on = np.zeros(len(force_n), bool)
    state = False
    for t, f in enumerate(force_n):
        state = f >= enter if not state else f >= exit_
        on[t] = state
    return on


def spans(on: np.ndarray) -> list[tuple[int, int]]:
    """``[start, end)`` of every run of True in a bool series."""
    idx = np.flatnonzero(np.diff(np.r_[False, on, False].astype(np.int8)))
    return list(zip(idx[::2].tolist(), idx[1::2].tolist()))


def limb_contact(sensor_path: str | Path, num_frames: int, enter_n: float,
                 exit_n: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-limb measured contact from ``sensor_forces.npz``.

    :param sensor_path: the clip's ``sensor_forces.npz`` (channels in :data:`LIMBS` order).
    :param num_frames: video frames of the body trajectory; the boards must cover them.
    :param enter_n: a limb is in contact from this measured ``|F|`` in newtons ...
    :param exit_n: ... until it drops below this (hysteresis, so a wobbling grip holds).
    :returns: ``(limb_on (N, 4) bool, force_world (N, 4, 3))``. A non-finite board
        sample reads as zero force, i.e. as no contact.
    """
    sensor = np.load(Path(sensor_path), allow_pickle=True)
    channels = [str(x) for x in sensor["limbs"]]
    if channels != LIMBS:
        raise ValueError(f"sensor channels are {channels}, expected {LIMBS}")
    force_world = np.asarray(sensor["reaction_world"], np.float32)[:num_frames]
    if force_world.shape[0] != num_frames:
        raise ValueError(f"the boards cover {force_world.shape[0]} frames, the body {num_frames}")
    force_n = np.linalg.norm(force_world, axis=-1)
    ok = np.isfinite(force_n)
    if not ok.all():
        print(f"  {int((~ok).any(1).sum())} frames with a non-finite board sample — no contact there")
    limb_on = np.stack([hysteresis(np.where(ok[:, i], force_n[:, i], 0.0), enter_n, exit_n)
                        for i in range(len(LIMBS))], 1)
    return limb_on, force_world


def frame_contact(limb_on: np.ndarray, valid: np.ndarray, scene_distance_m: np.ndarray,
                  frame_names: list[str], near_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Spread each loaded limb's measured contact onto its own contact frames.

    The board says the limb is loaded; the body says which of its frames is against the
    wall. The label holds for the WHOLE loaded span on the frames that are near the wall
    for at least a third of it — a frame set that changed from frame to frame made the
    downstream seating and stillness terms chase a moving target. A span with no such
    frame falls back to its nearest one and is flagged ``rescued``, which is how a
    reconstruction that put the limb well off the wall shows up.

    :param limb_on: ``(N, 4)`` measured per-limb contact, in :data:`LIMBS` order.
    :param valid: ``(N,)`` frames where the body trajectory is valid.
    :param scene_distance_m: ``(N, F)`` distance from each contact frame to the scene cloud.
    :param frame_names: the body's ``F`` contact-frame names.
    :param near_m: how close to the wall a frame counts as touching.
    :returns: ``(contact (N, F) bool, rescued (N, F) bool)``.
    """
    frame_limb = [limb_of_frame(name) for name in frame_names]
    contact = np.zeros((len(limb_on), len(frame_names)), bool)
    rescued = np.zeros_like(contact)
    for i, limb in enumerate(LIMBS):
        cols = [c for c, owner in enumerate(frame_limb) if owner == limb]
        if not cols:
            continue
        for start, end in spans(limb_on[:, i] & valid):
            near_fraction = (scene_distance_m[start:end][:, cols] <= near_m).mean(0)
            near = [c for c, f in zip(cols, near_fraction) if f >= _NEAR_FRACTION]
            if not near:
                near = [min(cols, key=lambda c: np.median(scene_distance_m[start:end, c]))]
                rescued[start:end, near] = True
            contact[start:end, near] = True
    return contact, rescued
