"""The geometric witness of contact: a contact frame touches where it is near the scene and still.

Two things are measured for every frame of the clip and contact frame. :func:`scene_signals` finds
the nearest scene point: the offset from it along its normal, the distance to it and its
confidence. :func:`drift` is the net distance a contact frame covers across a short window. Speed
at one instant cannot tell a gripping hand from a slow pass along the wall, because the fit's
jitter of a few millimetres per frame is a speed of 0.1 to 0.3 m/s; over a third of a second the
jitter cancels and a grip nets under 2 cm, while a slow pass adds up to 3 to 7 cm.

:func:`detect` turns the two into labels. The offset is trusted only next to a confident scene
point. A frame that is not still cannot touch, unless it presses into the surface. A foot touches
with one side of its toes. A contact is a span: it starts close, lasts while the frame stays
nearly as close, and has a shortest length. The skin's own normal decides nothing but the toe
side: scene normals come from a reconstructed cloud and skin normals from a fitted pose, and a
gate on "does the skin face the wall" threw away real contacts.

The drift is the caller's measurement, because which motion is honest depends on the stage: it has
to be the motion of a body that no contact term has acted on, or the optimiser's own damping would
vote for its labels.
"""
from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import NamedTuple

import numpy as np
import torch

from bvr.terms.scene import Scene

#: A foot's contact frames by side: it stands on its underside or hooks the top of its toes under a hold.
_UNDERSIDE, _TOP = ("ball_inner", "ball_outer", "big_toe", "heel"), ("toe_above",)


class Signals(NamedTuple):
    """Each contact frame against its nearest scene point, for every frame of a clip; all ``(T, F)``."""

    signed: torch.Tensor      # offset along the point's normal (m), negative behind the surface
    nearest: torch.Tensor     # distance to the point (m)
    confidence: torch.Tensor  # the point's confidence
    facing: torch.Tensor      # how squarely the skin faces the surface: 1 straight into it, -1 away from it


class Detection(NamedTuple):
    """The labels of one person and what they were cut from; numpy, all ``(T, F)``."""

    contact: np.ndarray       # bool: the label
    guarded: np.ndarray       # the signed distance where it can be trusted, infinite elsewhere (m)
    suppressed: np.ndarray    # bool: ruled out because the other side of the foot's toes touches
    pressed: np.ndarray       # bool: pressing into the surface, which outranks stillness and span length


@torch.no_grad()
def scene_signals(points: torch.Tensor, normals: torch.Tensor, scene: Scene, *, prune_margin_m: float) -> Signals:
    """The nearest scene point of each contact frame, frame by frame.

    ``points (T, F, 3)`` and ``normals (T, F, 3)`` are the contact frames' world positions and outward skin
    normals. Per frame of the clip the scene is cut to the box around the contact frames grown by
    ``prune_margin_m``, which must be at least the distance the caller trusts a match within. A frame of the clip
    whose box holds no scene point has no evidence: infinite distances, zero confidence, facing -1.
    """
    signed = points.new_full(points.shape[:2], torch.inf)
    nearest, confidence, facing = signed.clone(), torch.zeros_like(signed), -torch.ones_like(signed)
    for frame, (point, normal) in enumerate(zip(points, normals)):
        low, high = point.amin(dim=0) - prune_margin_m, point.amax(dim=0) + prune_margin_m
        inside = ((scene.points >= low) & (scene.points <= high)).all(dim=-1)
        if inside.any():
            near = scene.points[inside]
            nearest[frame], index = torch.cdist(point, near).min(dim=1)
            surface = scene.normals[inside][index]
            signed[frame] = ((point - near[index]) * surface).sum(dim=-1)
            confidence[frame] = scene.confidence[inside][index]
            facing[frame] = -(normal * surface).sum(dim=-1)
    return Signals(signed, nearest, confidence, facing)


def drift(points: torch.Tensor, tracked: torch.Tensor, *, fps: float, window_s: float) -> torch.Tensor:
    """The net distance ``(T, F)`` in metres that points ``(T, F, 3)`` cover across a window centred on each frame.

    The window is ``window_s`` wide. At the ends of the clip it is cut short and the distance scaled up to the
    full width, so that a fast point there does not read as still. Where the frame or an end of its window is
    not ``tracked (T,)`` the drift is NaN: nothing was measured, and every reader takes NaN as "not still".
    """
    half = max(1, round(window_s * fps / 2))
    frame = torch.arange(len(points), device=points.device)
    first, last = (frame - half).clamp_min(0), (frame + half).clamp_max(len(points) - 1)
    covered = (points[last] - points[first]).norm(dim=-1) * (2 * half / (last - first).clamp_min(1))[:, None]
    return covered.masked_fill(~(tracked & tracked[first] & tracked[last])[:, None], torch.nan)


def detect(signals: Signals, drift: torch.Tensor, *, names: Sequence[str], close_thresh_m: float,
           exit_thresh_m: float, min_frames: int, still_m: float, still_override_m: float, max_dist_m: float,
           conf_thresh: float, confirmed: np.ndarray | None = None) -> Detection:
    """Label one person's contact frames: near the scene and still.

    :param drift: ``(T, F)`` metres, measured before any contact term acted on the body.
    :param names: the contact frames' names, which say where the toes are.
    :param close_thresh_m: a span starts where the guarded signed distance is below this,
    :param exit_thresh_m: and lasts while it is below this.
    :param min_frames: spans shorter than this are dropped, unless a frame of them presses.
    :param still_m: a frame is still while its drift is under twice this.
    :param still_override_m: a frame closer than this presses into the surface and counts however it moves: a
        sprinting foot plant is a contact that no window can call still.
    :param max_dist_m: a match is a surface only within this distance,
    :param conf_thresh: and above this confidence: a hole in the cloud reads as unknown, not as no wall.
    :param confirmed: ``(T, F)`` bool, where a second witness confirms a press (the readout's solved load). A hand
        sweeping along the wall presses as close as a hand planted on it; without this every press counts.
    """
    signed, nearest, confidence, facing, drift = (value.cpu().numpy() for value in (*signals, drift))
    guarded = np.where((nearest < max_dist_m) & (confidence > conf_thresh), signed, np.inf)
    pressed = (guarded < still_override_m) & np.isfinite(drift)
    if confirmed is not None:
        pressed &= confirmed
    # A frame that neither is still nor presses looks infinitely far away, so it starts and holds no span.
    gated = np.where((drift < 2.0 * still_m) | pressed, guarded, np.inf)
    gated, suppressed = _one_toe_side(gated, facing, toe_sides(names), exit_thresh_m)
    contact = min_span(hysteresis(gated < close_thresh_m, gated < exit_thresh_m), min_frames, keep=pressed)
    return Detection(contact, guarded, suppressed, pressed)


def toe_sides(names: Sequence[str]) -> list[tuple[list[int], list[int]]]:
    """Per foot, the columns of its underside frames and of the top of its toes, found by name."""
    column = {name: index for index, name in enumerate(names)}
    return [([column[f"{stem}_{side}"] for stem in _UNDERSIDE], [column[f"{stem}_{side}"] for stem in _TOP])
            for side in ("left", "right")]


def _one_toe_side(gated: np.ndarray, facing: np.ndarray, sides: list[tuple[list[int], list[int]]],
                  exit_thresh_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep one side of each foot's toes per frame; return the gated distance and what was ruled out.

    The underside and the top of the toes sit centimetres apart through the foot, so distance, confidence and
    stillness accept or reject them together. Where both sides of a foot could hold a span (below
    ``exit_thresh_m``), the side whose best frame faces the surface more squarely stays and the other is sent to
    infinity; a tie keeps the underside.
    """
    gated, suppressed = gated.copy(), np.zeros(gated.shape, bool)
    for under, top in sides:
        on_under, on_top = gated[:, under] < exit_thresh_m, gated[:, top] < exit_thresh_m
        both = on_under.any(axis=1) & on_top.any(axis=1)
        under_wins = (np.where(on_under, facing[:, under], -np.inf).max(axis=1)
                      >= np.where(on_top, facing[:, top], -np.inf).max(axis=1))
        for columns, candidates, lost in ((top, on_top, both & under_wins), (under, on_under, both & ~under_wins)):
            ruled_out = candidates & lost[:, None]
            gated[:, columns] = np.where(ruled_out, np.inf, gated[:, columns])
            suppressed[:, columns] |= ruled_out
    return gated, suppressed


def hysteresis(enter: np.ndarray, hold: np.ndarray) -> np.ndarray:
    """Spans ``(T, F)`` bool that start where ``enter`` holds and last while ``hold`` does.

    With one threshold a signal that hovers at the line switches every other frame; the looser ``hold`` prices
    the switch.
    """
    spans, active = np.zeros_like(enter), np.zeros(enter.shape[1], bool)
    for frame in range(len(enter)):
        active = (active & hold[frame]) | enter[frame]
        spans[frame] = active
    return spans


def runs(column: np.ndarray) -> list[tuple[int, int]]:
    """The runs of ``True`` in a bool array ``(T,)``, as ``(start, stop)`` with ``stop`` excluded."""
    edges = np.flatnonzero(np.diff(column.astype(np.int8), prepend=0, append=0))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def min_span(labels: np.ndarray, min_frames: int, keep: np.ndarray | None = None) -> np.ndarray:
    """``labels (T, F)`` without the spans shorter than ``min_frames``; a span with a ``keep`` frame stays.

    ``keep (T, F)`` marks evidence that is not flicker however brief: a frame pressing into the surface.
    """
    labels = labels.copy()
    for column in range(labels.shape[1]):
        for start, stop in runs(labels[:, column]):
            if stop - start < min_frames and not (keep is not None and keep[start:stop, column].any()):
                labels[start:stop, column] = False
    return labels


def close_gaps(labels: np.ndarray, min_frames: int) -> np.ndarray:
    """``labels (T, F)`` with the gaps shorter than ``min_frames`` filled: a release lasts as long as a contact.

    No grip is let go and taken again within a frame or two. A gap at the start or the end of the clip has no
    contact on its far side and stays.
    """
    labels = labels.copy()
    for column in range(labels.shape[1]):
        for start, stop in runs(~labels[:, column]):
            if 0 < start and stop < len(labels) and stop - start < min_frames:
                labels[start:stop, column] = True
    return labels


def merge_spans(contact: np.ndarray, points: np.ndarray, guarded: np.ndarray, *, merge_dist_m: float,
                max_gap_frames: int, gap_sdf_m: float) -> np.ndarray:
    """``contact (T, F)`` with a short gap between two spans filled where the contact frame never left.

    A gap of at most ``max_gap_frames`` is detection flicker if the frame ends within ``merge_dist_m`` of where
    it left (``points (T, F, 3)``) and its guarded distance stayed under ``gap_sdf_m`` throughout. The second
    test tells flicker from a hand that lets go, wanders and takes the same hold again.
    """
    merged = contact.copy()
    for column in range(contact.shape[1]):
        for (_, stop), (start, _) in pairwise(runs(contact[:, column])):
            last = stop - 1
            if (start - last <= max_gap_frames
                    and np.linalg.norm(points[start, column] - points[last, column]) <= merge_dist_m
                    and guarded[last:start + 1, column].max() < gap_sdf_m):
                merged[stop:start, column] = True
    return merged
