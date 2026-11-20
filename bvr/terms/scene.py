"""The scene a body touches: the points near it, the signed distance to them, and the contact terms.

The scene is the clip's fused static cloud: points, outward unit normals and a confidence each
(:class:`Scene`). Contact lives on the body's contact frames, ``frames (P, T, F, 7)``. A fit reads
the scene through values that are fixed while a phase runs and built again from the body's pose
when the next one starts: the :class:`Candidates` near each person in each frame, and for
``kinematics_2`` the :class:`Patches` of wall that labelled frames are pulled to. The signed
distance of a frame, :class:`Distance`, is its offset from the nearest candidate along that
candidate's normal: positive in front of the surface, negative behind it.

``labels (P, T, F)`` say where a contact frame touches the scene. Every term returns one loss per
person, ``(P,)``. Nearest neighbours carry no gradient: a term moves a frame, never its match.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch
from better_robot import optim

from bvr import camera
from bvr.camera import Cameras
from bvr.terms import Term

#: Entries of a pairwise table per chunk of rows, for the two searches that run once per phase.
_PAIRS = 4_000_000


class Scene(NamedTuple):
    """The static cloud of a clip."""

    points: torch.Tensor      # (S, 3) world
    normals: torch.Tensor     # (S, 3) unit, towards the cameras that saw the point
    confidence: torch.Tensor  # (S,)


class Candidates(NamedTuple):
    """The scene points near each person in each frame, padded to one width ``M``."""

    points: torch.Tensor      # (P, T, M, 3)
    normals: torch.Tensor     # (P, T, M, 3)
    confidence: torch.Tensor  # (P, T, M)
    counted: torch.Tensor     # (P, T, M) bool: a real candidate, not padding


class Distance(NamedTuple):
    """Each contact frame against its nearest candidate."""

    signed: torch.Tensor      # (P, T, F) offset along the candidate's normal (m); the gradient reaches the frame
    nearest: torch.Tensor     # (P, T, F) distance to the candidate (m); infinite where a frame has none
    confidence: torch.Tensor  # (P, T, F) the candidate's confidence

    def near(self, conf_thresh: float, max_dist_m: float) -> torch.Tensor:
        """Where the match is a surface to trust, ``(P, T, F)`` bool: a confident candidate within ``max_dist_m``."""
        return (self.confidence > conf_thresh) & (self.nearest < max_dist_m)


class Patches(NamedTuple):
    """The patch of scene each resolved contact instance is pulled to; ``I`` instances, padded to ``k`` points."""

    person: torch.Tensor      # (I,) long
    frame: torch.Tensor       # (I,) long
    contact: torch.Tensor     # (I,) long: the contact frame
    points: torch.Tensor      # (I, k, 3)
    counted: torch.Tensor     # (I, k) bool: a real patch point, not padding


@torch.no_grad()
def candidates(scene: Scene, body: torch.Tensor, tracked: torch.Tensor, *, prune_margin_m: float,
               max_candidates: int) -> Candidates:
    """The scene points a body can reach in each frame.

    Per person and frame: the points inside the box around ``body (P, T, K, 3)`` (joints and contact frames) grown
    by ``prune_margin_m``, and of those the ``max_candidates`` nearest to the box's centre. The margin covers the
    skin around the joints and the body's motion during a phase. A frame where the person is not ``tracked (P, T)``
    has none, so no scene term acts there.
    """
    low = (body.amin(dim=-2) - prune_margin_m).flatten(0, 1)
    high = (body.amax(dim=-2) + prune_margin_m).flatten(0, 1)
    found = []
    for rows in torch.arange(len(low), device=low.device).split(max(1, _PAIRS // len(scene.points))):
        inside = ((scene.points >= low[rows, None]) & (scene.points <= high[rows, None])).all(dim=-1)
        offset = scene.points - (low[rows, None] + high[rows, None]) / 2
        found.append(offset.square().sum(dim=-1).masked_fill(~inside, torch.inf).topk(
            min(max_candidates, len(scene.points)), dim=1, largest=False))
    distance, index = (torch.cat(part).unflatten(0, tracked.shape) for part in zip(*found))
    counted = distance.isfinite() & tracked[..., None]
    return Candidates(scene.points[index], scene.normals[index], scene.confidence[index], counted)


def signed_distance(chunk: int) -> Callable[[torch.Tensor, Candidates], Distance]:
    """Cached value: the :class:`Distance` of ``frames (P, T, F, 7)`` to the candidates, ``chunk`` rows at a time."""
    def value(frames: torch.Tensor, scene: Candidates) -> Distance:
        points = frames[..., :3].flatten(0, 1)
        near, normals, confidence, counted = (field.flatten(0, 1) for field in scene)
        parts = []
        for start in range(0, len(points), chunk):
            rows = slice(start, start + chunk)
            with torch.no_grad():
                table = torch.cdist(points[rows], near[rows]).masked_fill(~counted[rows, None, :], torch.inf)
                nearest, index = table.min(dim=2)
            pick = index[..., None].expand(-1, -1, 3)
            offset = points[rows] - near[rows].gather(1, pick)
            parts.append(((offset * normals[rows].gather(1, pick)).sum(dim=-1), nearest,
                          confidence[rows].gather(1, index)))
        return Distance(*(torch.cat(part).unflatten(0, frames.shape[:2]) for part in zip(*parts)))
    return value


def penetration(*, conf_thresh: float, max_dist_m: float, max_pen_m: float, margin_m: float,
                huber_m: float) -> Term:
    """Term: contact frames stay out of the scene, the mean of ``Huber(relu(-s - margin_m), huber_m)`` in m^2.

    ``s`` is the signed distance. Nothing is charged down to ``margin_m`` of penetration (soft tissue, depth
    error); beyond it the cost grows as a square, then linearly from ``huber_m`` on. The mean is over the frames
    with a trusted match that are not deeper than ``max_pen_m``: deeper is a match behind an occluder.
    """
    kernel = optim.Huber(delta=huber_m)

    def loss(distance: Distance) -> torch.Tensor:
        active = distance.near(conf_thresh, max_dist_m) & (distance.signed.detach() > -max_pen_m)
        cost = kernel.rho(torch.relu(-distance.signed - margin_m).square())
        return (cost * active).sum(dim=(1, 2)) / active.sum(dim=(1, 2)).clamp_min(1)
    return loss


def attraction(*, conf_thresh: float, max_dist_m: float, band_m: float) -> Term:
    """Term: labelled contact frames come onto the scene, the mean of the signed distance ``s`` in metres.

    It acts on a labelled frame with a trusted match that hovers within the band, ``0 < s < band_m``. Its gradient
    is the surface normal, so the frame moves straight in until ``s`` is 0, where :func:`penetration` takes over.
    """
    def loss(distance: Distance, labels: torch.Tensor) -> torch.Tensor:
        gap = distance.signed.detach()
        active = distance.near(conf_thresh, max_dist_m) & (gap > 0) & (gap < band_m) & labels
        return (distance.signed * active).sum(dim=(1, 2)) / active.sum(dim=(1, 2)).clamp_min(1)
    return loss


@torch.no_grad()
def patches(scene: Scene, frames: torch.Tensor, labels: torch.Tensor, cameras: Cameras, image_size: torch.Tensor, *,
            radius_px: float, patch_k: int, conf_thresh: float) -> Patches:
    """For each labelled contact frame, the scene that the camera sees where the frame appears.

    A label says the frame touches something, and what it touches is the surface visible at its own pixel. The
    patch of an instance (a person, a frame, a contact frame) is the confident scene points in front of the camera
    that project within ``radius_px`` of the frame's pixel; of those, the ``patch_k`` nearest to the frame in
    space. An instance is unresolved, and left out, when the frame is behind the camera or more than ``radius_px``
    off the image ``image_size (2,)`` (width, height), or when its patch is empty.
    """
    extrinsics, intrinsics = cameras
    person, frame, contact = labels.nonzero(as_tuple=True)
    point = frames[person, frame, contact, :3]
    confident = scene.confidence > conf_thresh
    found = []
    for rows in torch.arange(len(point), device=point.device).split(max(1, _PAIRS // len(scene.points))):
        view = extrinsics[frame[rows]], intrinsics[frame[rows]]
        pixel = camera.project(point[rows, None], *view)
        on_image = (camera.to_camera(point[rows, None], view[0])[..., 2] > 0) & (
            (pixel >= -radius_px) & (pixel <= image_size + radius_px)).all(dim=-1)
        seen = (camera.to_camera(scene.points, view[0])[..., 2] > camera.NEAR) & confident
        around = (camera.project(scene.points, *view) - pixel).norm(dim=-1) < radius_px
        distance = (scene.points - point[rows, None]).norm(dim=-1).masked_fill(~(seen & around & on_image), torch.inf)
        found.append(distance.topk(min(patch_k, len(scene.points)), dim=1, largest=False))
    distance, index = (torch.cat(part) for part in zip(*found))
    resolved = distance.isfinite().any(dim=1)
    return Patches(person[resolved], frame[resolved], contact[resolved], scene.points[index[resolved]],
                   distance[resolved].isfinite())


def acquisition(*, delta_m: float) -> Term:
    """Term: each resolved instance's contact frame comes to its patch, the mean of a pseudo-Huber in m^2.

    With ``d`` the distance from the frame to the nearest point of its patch, the cost is
    ``delta_m^2 (sqrt(1 + (d / delta_m)^2) - 1)``: a square near the patch and a steady, bounded pull far from it.
    """
    def loss(frames: torch.Tensor, patches: Patches) -> torch.Tensor:
        point = frames[patches.person, patches.frame, patches.contact, :3]
        squared = (patches.points - point[:, None]).square().sum(dim=-1).masked_fill(~patches.counted, torch.inf)
        cost = delta_m**2 * (torch.sqrt(1.0 + squared.amin(dim=1) / delta_m**2) - 1.0)
        people = frames.shape[0]
        instances = torch.bincount(patches.person, minlength=people).clamp_min(1)
        return cost.new_zeros(people).index_add(0, patches.person, cost) / instances
    return loss
