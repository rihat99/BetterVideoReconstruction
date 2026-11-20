"""From two witnesses to one set of labels: the rules of the contact readout, for one person.

The geometric witness (:mod:`bvr.contacts.detector`) needs a contact frame to be near the scene.
Where the reconstruction puts the body in the wrong place that never happens, and a real contact
would be lost. The force witness (:mod:`bvr.contacts.witness`) answers from the other side: a frame
that has to carry load is in contact wherever it sits. The readout is their union, made of spans.

The solved load, in body weights, is read three times. It adds labels where a frame carries more
than a threshold. It confirms a press: a frame that presses into the surface while it moves counts
only where it also carries load, which tells a planted hand from one sweeping along the wall. And
it picks the side of the toes: each frame's force lies in a cone about its own skin normal, so the
side that carried the load is the side whose orientation explains the motion.

Where the solve explained nothing (a rope holds the climber, the trajectory is broken) it is
``mute``: its silence is not evidence, and a press counts without it.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import torch

from bvr.contacts import detector


class Labels(NamedTuple):
    """One person's labels; bool, all ``(T, F)``."""

    contact: np.ndarray       # the readout
    geometric: np.ndarray     # the geometric witness alone
    loaded: np.ndarray        # the force witness alone


def read(signals: detector.Signals, drift: torch.Tensor, points: torch.Tensor, load: np.ndarray, far: np.ndarray,
         mute: np.ndarray, tracked: np.ndarray, *, names: Sequence[str], fps: float, contacts: dict, gate: dict,
         force: dict) -> Labels:
    """Label one person's contact frames from both witnesses.

    :param signals: the contact frames against the scene, as the detector reads them.
    :param drift: ``(T, F)`` metres, measured before any contact term acted on the body.
    :param points: ``(T, F, 3)`` world positions of the contact frames.
    :param load: ``(T, F)`` solved force magnitude in body weights.
    :param far: ``(T, F)`` distance to the scene as the force witness searched it (m); not finite without evidence.
    :param mute: ``(T,)`` bool, the frames of the clip where the solve explained nothing.
    :param tracked: ``(T,)`` bool.
    :param contacts: ``[contacts]``, with ``gate`` = ``[contact_gate]`` and ``force`` = ``[contact_force]``.
    """
    shortest, raw = math.ceil(contacts["min_span_s"] * fps), math.ceil(gate["detect_min_s"] * fps)
    seen = tracked[:, None]
    knees = [column for column, name in enumerate(names) if name.startswith("knee")]
    heels = [column for column, name in enumerate(names) if name.startswith("heel")]

    found = detector.detect(
        signals, drift, names=names, close_thresh_m=contacts["close_thresh_m"],
        exit_thresh_m=contacts["exit_thresh_m"], min_frames=raw, still_m=gate["still_m"],
        still_override_m=gate["still_override_m"], max_dist_m=contacts["max_dist_m"],
        conf_thresh=contacts["conf_thresh"], confirmed=(load >= force["confirm_bw"]) | mute[:, None])
    # The merge fills one column at a time, so it could put a toe back where the other side of the foot won.
    geometric = detector.merge_spans(
        found.contact, points.cpu().numpy(), found.guarded, merge_dist_m=contacts["merge_dist_m"],
        max_gap_frames=round(contacts["merge_max_gap_s"] * fps), gap_sdf_m=contacts["merge_sdf_m"]) & ~found.suppressed
    # A knee rides near the wall, still, for most of a climb without touching: neither witness labels one.
    geometric[:, knees] = False

    # Within the detector's trust radius load alone labels: a fast foot plant that carries weight is a contact.
    # Beyond it geometry is blind and the load needs stillness, at a looser bar than the detector's because a
    # badly reconstructed body jitters: real far support holds still, the scraps a bad flight parks on limbs do not.
    # A distance or a drift that was not measured is no evidence: it reads as far, and as moving.
    moving = ~(drift.cpu().numpy() <= force["far_still_x"] * gate["still_m"])
    priced = np.where(~(far <= contacts["max_dist_m"]) & moving, 0.0, load)
    # Within a foot the split of load is not determined (a flat foot can stand on its ball alone), so a small
    # solved heel load says nothing: the heel is labelled by geometry only.
    priced[:, heels + knees] = 0.0
    loaded = detector.hysteresis(priced > force["theta_bw"], priced > force["theta_hold_bw"]) & seen
    loaded = one_toe_side(detector.min_span(loaded, raw), priced, names, raw)

    # Each witness drops its own short spans, but the union can leave a hole of a frame or two between a span of
    # one and a span of the other. One number rules both ways: a contact lasts ``shortest`` frames, and so does a
    # release. A span that presses is evidence however brief.
    pressed = found.pressed & ~found.suppressed
    contact = detector.close_gaps((geometric | loaded) & seen, shortest) & seen
    contact = detector.min_span(contact, shortest, pressed) & seen
    # The witnesses each keep one side of a foot's toes, but not always the same one.
    return Labels(one_toe_side(contact, load, names, raw, pressed), geometric & seen, loaded)


def one_toe_side(labels: np.ndarray, load: np.ndarray, names: Sequence[str], min_frames: int,
                 keep: np.ndarray | None = None) -> np.ndarray:
    """Give each foot one side of its toes per contact: the side the solved force loaded.

    A contact of a foot is one unbroken run of frames with any of its toe or heel frames labelled. Over the run
    the load through the underside frames is summed against the load through the top of the toes, and the larger
    takes the whole run (a tie keeps the underside), so the side cannot change while the foot stays on a hold.
    Within the winning side each frame keeps its own label; frames of the clip where only the losing side was
    labelled go to the winning side's most loaded frame, so "this foot touches" is unchanged. Spans left shorter
    than ``min_frames`` are dropped, unless ``keep`` marks a frame of them.

    The force's vertical sign does not decide this (a toe hooked overhead holds the body up, like a stance), and
    neither does the scene normal at the nearest point (on small holds that is the wall behind the hold).
    """
    labels, sides = labels.copy(), detector.toe_sides(names)
    for under, top in sides:
        for start, stop in detector.runs(labels[:, under + top].any(axis=1)):
            run = slice(start, stop)
            standing = load[run][:, under].sum() >= load[run][:, top].sum()
            wins, loses = (under, top) if standing else (top, under)
            labels[run, loses] = False
            most_loaded = wins[int(load[run][:, wins].sum(axis=0).argmax())]
            labels[run, most_loaded] |= ~labels[run][:, wins].any(axis=1)
    toes = sorted(column for side in sides for part in side for column in part)
    labels[:, toes] = detector.min_span(labels[:, toes], min_frames, None if keep is None else keep[:, toes])
    return labels
