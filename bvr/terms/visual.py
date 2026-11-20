"""What the video says about a body: keypoints, the start pose's own projection, depth points, and the priors.

Each ``Term`` here is built once from a clip's evidence and returns a function of cached values of
the fit (``body``, the ``HumanState`` of the trajectory, or ``pose``, the trajectory itself) that
gives one loss per person, ``(P,)``. Evidence carries people first and frames second,
``(P, T, ...)``; ``tracked (P, T)`` says on which frames a person is tracked, and an untracked frame
carries no evidence. ``cameras`` is the pair ``(extrinsics (T, 4, 4), intrinsics (T, 3, 3))``.

The two reprojection terms are robust: a squared pixel error ``r2`` costs ``r2 / (c^2 + r2)``, which
grows like a square near 0 and levels off at 1, so a wrong detection cannot drag the pose.
"""
from __future__ import annotations

import better_human as bh
import numpy as np
import torch
from better_robot import optim
from scipy.signal import savgol_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from bvr import camera
from bvr.body import BODY_JOINTS
from bvr.camera import Cameras
from bvr.geometry.depth_smear import padded_frame_weights
from bvr.terms import Term

# Sapiens keypoints (Goliath-308 indices) and the body points they name.
KEYPOINT_JOINTS = {
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8, "left_wrist": 62, "right_wrist": 41,
    "left_hip": 9, "right_hip": 10, "left_knee": 11, "right_knee": 12, "left_ankle": 13, "right_ankle": 14,
}
KEYPOINT_LANDMARKS = {
    "nose": 0, "leye": 1, "reye": 2, "lear": 3, "rear": 4,
    "LBigToe": 15, "LSmallToe": 16, "LHeel": 17, "RBigToe": 18, "RSmallToe": 19, "RHeel": 20,
}
KEYPOINT_FINGERTIPS = {
    "lthumb": 42, "lindex": 46, "lmiddle": 50, "lring": 54, "lpinky": 58,
    "rthumb": 21, "rindex": 25, "rmiddle": 29, "rring": 33, "rpinky": 37,
}

# The limbs whose depth points a uniform sample starves; a joint belongs to at most one of them.
_HAND = ("wrist", "index", "middle", "pinky", "ring", "thumb")
_FOOT = ("ankle", "foot")
_LIMBS = (("left", _HAND), ("right", _HAND), ("left", _FOOT), ("right", _FOOT))


def keypoint_reprojection(human: bh.HumanModel, keypoints: torch.Tensor, scores: torch.Tensor, tracked: torch.Tensor,
                          cameras: Cameras, fps: float, *, score_thresh: float, include_hands: bool,
                          gm_scale_px: float, smooth_window_s: float) -> Term:
    """Term: Sapiens keypoints against the joints and landmarks they name, ``sum s g / sum s`` with ``s`` the score.

    ``keypoints (P, T, 308, 2)`` are pixels and ``scores (P, T, 308)`` their detection scores; a score under
    ``score_thresh`` counts as 0. The tracks are smoothed over ``smooth_window_s`` first, so the detector's
    frame-to-frame noise is not written into the motion.
    """
    spec = human.spec
    landmarks = KEYPOINT_LANDMARKS | (KEYPOINT_FINGERTIPS if include_hands else {})
    joints = torch.tensor([spec.joint_id(name) for name in KEYPOINT_JOINTS], device=spec.device)
    vertices = spec.landmarks.vertex_ids[[spec.landmarks.names.index(name) for name in landmarks]]
    named = torch.tensor([*KEYPOINT_JOINTS.values(), *landmarks.values()], device=spec.device)
    weight = torch.nan_to_num(scores[..., named]) * tracked[..., None]
    weight = torch.where(weight >= score_thresh, weight, 0.0)
    target = smooth(torch.nan_to_num(keypoints[..., named, :]), weight > 0, window_s=smooth_window_s, fps=fps)

    def loss(body: bh.HumanState) -> torch.Tensor:
        points = torch.cat((body.joint_translations[..., joints, :], body.points(vertices)), dim=-2)
        error = (camera.project(points, *cameras) - target).square().sum(dim=-1)
        return (weight * _saturating(error, gm_scale_px)).sum(dim=(1, 2)) / weight.sum(dim=(1, 2)).clamp_min(1.0)
    return loss


def start_reprojection(start: bh.HumanState, tracked: torch.Tensor, cameras: Cameras, fps: float, *,
                       gm_scale_px: float, smooth_window_s: float) -> Term:
    """Term: the 22 body joints against where the start pose projects them, a mean over tracked frames.

    The start pose reprojects well, so this holds the body on the image while the depth terms move it along the
    viewing rays. It is no observation of its own: it cannot correct the start pose, only keep its pixels.
    """
    counted = tracked[..., None].expand(-1, -1, BODY_JOINTS)
    target = camera.project(start.joint_translations[..., :BODY_JOINTS, :], *cameras)
    target = smooth(target, counted, window_s=smooth_window_s, fps=fps)

    def loss(body: bh.HumanState) -> torch.Tensor:
        pixels = camera.project(body.joint_translations[..., :BODY_JOINTS, :], *cameras)
        cost = _saturating((pixels - target).square().sum(dim=-1), gm_scale_px)
        return (cost * counted).sum(dim=(1, 2)) / counted.sum(dim=(1, 2)).clamp_min(1)
    return loss


def chamfer(human: bh.HumanModel, cloud: torch.Tensor, counted: torch.Tensor, cameras: Cameras, *, sat_m: float,
            vertex_subsample: int, emphasis_lo: float, emphasis_hi: float, emphasis_radius: float,
            chunk: int) -> Term:
    """Term: the body's visible surface against a person's depth points, both ways, in metres.

    Per frame: the weighted mean over visible sampled vertices of the distance to the nearest point of
    ``cloud (P, T, M, 3)``, plus the mean over the points of the distance to the nearest visible vertex. Only the
    points with ``counted (P, T, M)`` are real, the rest is padding (:func:`human_points`). A distance ``d``
    counts as ``sat_m tanh(d / sat_m)``, so a far outlier stops pulling. The vertex weights favour hands and feet
    (:func:`vertex_emphasis`), whose depth is the least reliable, and a frame whose points smear along the view
    ray weighs less. A vertex is visible if its normal faces the camera; nearest neighbours and visibility carry
    no gradient.
    """
    spec, (extrinsics, _) = human.spec, cameras
    sample = (torch.arange(vertex_subsample, device=spec.device) * (spec.num_vertices / vertex_subsample)).long()
    sample = torch.unique(sample.clamp_max(spec.num_vertices - 1))
    emphasis = vertex_emphasis(human, low=emphasis_lo, high=emphasis_hi, radius=emphasis_radius)[sample]
    people, frames = counted.shape[:2]
    frame_weight = padded_frame_weights(
        cloud.flatten(0, 1), counted.flatten(0, 1), camera.centers(extrinsics).repeat(people, 1))
    points, real = cloud.flatten(0, 1), counted.flatten(0, 1)

    @torch.no_grad()
    def visible(body: bh.HumanState) -> torch.Tensor:
        mesh = body.vertices
        normals = _vertex_normals(mesh, spec.faces)[..., sample, :] @ extrinsics[:, :3, :3].mT
        return ((normals * camera.to_camera(mesh[..., sample, :], extrinsics)).sum(dim=-1) < 0).flatten(0, 1)

    def loss(body: bh.HumanState) -> torch.Tensor:
        vertices, facing = body.points(sample).flatten(0, 1), visible(body)
        rows = []
        for start in range(0, people * frames, chunk):
            v, p = vertices[start:start + chunk], points[start:start + chunk]
            seen, there = facing[start:start + chunk], real[start:start + chunk]
            with torch.no_grad():
                distance = torch.cdist(v, p)
                to_cloud = distance.masked_fill(~there[:, None, :], torch.inf).argmin(dim=2)
                to_body = distance.masked_fill(~seen[:, :, None], torch.inf).argmin(dim=1)
            outward = _saturated((v - _take(p, to_cloud)).norm(dim=-1), sat_m) * seen * emphasis
            inward = _saturated((p - _take(v, to_body)).norm(dim=-1), sat_m) * there
            rows.append(outward.sum(dim=1) / (seen * emphasis).sum(dim=1).clamp_min(1e-6)
                        + inward.sum(dim=1) / there.sum(dim=1).clamp_min(1))
        weight = (frame_weight * (facing.any(dim=1) & real.any(dim=1))).view(people, frames)
        return (torch.cat(rows).view(people, frames) * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1e-6)
    return loss


def pose_prior(start: torch.Tensor) -> Term:
    """Term: the body joints near their start rotations, the mean of ``1 - <q, q_start>^2`` over the 51 joints.

    ``start (P, T, 211)`` is the start trajectory. The 30 fingers add nothing: where they are fitted, the start
    hand pose is what is no longer trusted. The root is :func:`orientation_prior`'s.
    """
    def loss(pose: torch.Tensor) -> torch.Tensor:
        aligned = (_joint_quaternions(pose) * _joint_quaternions(start)).sum(dim=-1)
        return (1.0 - aligned[..., :BODY_JOINTS - 1].square()).sum(dim=(1, 2)) / aligned[0].numel()
    return loss


def orientation_prior(start: torch.Tensor) -> Term:
    """Term: the root near its start orientation, the mean of ``1 - <q, q_start>^2`` over the frames."""
    def loss(pose: torch.Tensor) -> torch.Tensor:
        return (1.0 - (pose[..., 3:7] * start[..., 3:7]).sum(dim=-1).square()).mean(dim=1)
    return loss


def human_points(human: bh.HumanModel, points: np.ndarray, confidence: np.ndarray, frame_ids: np.ndarray,
                 start_pixels: np.ndarray, tracked: np.ndarray, cameras: Cameras, *, conf_drop_q: float,
                 max_points: int, limb_quota: int) -> tuple[np.ndarray, np.ndarray]:
    """One person's depth points per frame, hands and feet kept: ``(T, M, 3)`` and the mask ``(T, M)`` of real points.

    ``points (n, 3)`` are the person's fused world points, ``confidence (n,)`` their depth confidence and
    ``frame_ids (n,)`` the frame of each. A point belongs to the limb of the nearest start joint in the image,
    ``start_pixels (T, J, 2)``: a depth streak moves a point along its ray, which keeps its pixel. Within each limb
    and within the rest, the lowest ``conf_drop_q`` of a frame's points is dropped (the confidence has no fixed
    scale, and one gate over the whole frame would empty the limbs first). Then each hand and foot keeps up to
    ``limb_quota`` points and the rest fills ``max_points``, by uniform strides. An untracked frame keeps none.
    """
    limb_of_joint = _limbs(human.spec.joint_names)
    order = np.argsort(frame_ids, kind="stable")
    points, confidence, frame_ids = points[order], confidence[order], frame_ids[order]
    extrinsics, intrinsics = (matrix.cpu()[frame_ids] for matrix in cameras)
    pixels = camera.project(torch.from_numpy(points)[:, None], extrinsics, intrinsics)[:, 0].numpy()
    bounds = np.searchsorted(frame_ids, np.arange(len(tracked) + 1))
    cloud = np.zeros((len(tracked), max(max_points, len(_LIMBS) * limb_quota), 3), np.float32)
    counted = np.zeros(cloud.shape[:2], bool)
    for frame in np.flatnonzero(tracked & (bounds[1:] > bounds[:-1])):
        span = slice(bounds[frame], bounds[frame + 1])
        gaps = start_pixels[frame] - pixels[span, None]
        limb, trust = limb_of_joint[np.square(gaps).sum(axis=-1).argmin(axis=1)], confidence[span]
        keep = np.ones(len(limb), bool)
        for group in np.unique(limb):
            keep &= (limb != group) | (trust >= np.quantile(trust[limb == group], conf_drop_q))
        chosen = [_strided(np.flatnonzero(keep & (limb == group)), limb_quota) for group in range(1, len(_LIMBS) + 1)]
        rest = _strided(np.flatnonzero(keep & (limb == 0)), max(0, max_points - sum(map(len, chosen))))
        kept = np.sort(np.concatenate((*chosen, rest)))
        cloud[frame, :len(kept)], counted[frame, :len(kept)] = points[span][kept], True
    return cloud, counted


def vertex_emphasis(human: bh.HumanModel, *, low: float, high: float, radius: float) -> torch.Tensor:
    """A weight per vertex ``(V,)``: ``high`` on the hands and feet, ``low`` at ``radius`` metres from them and beyond.

    The distance is along the surface of the template mesh, to the nearest vertex skinned mostly to a hand or foot
    joint. Between the two ends the weight moves geometrically, so halfway it is ``sqrt(low high)``.
    """
    spec = human.spec
    vertices, faces = spec.template_vertices.double().cpu().numpy(), spec.faces.cpu().numpy()
    tail, head = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).T
    lengths = np.linalg.norm(vertices[tail] - vertices[head], axis=1)
    graph = csr_matrix((lengths, (tail, head)), shape=(len(vertices), len(vertices)))
    on_limb = _limbs(spec.joint_names)[human.skin_matrix.argmax(dim=1).cpu().numpy()] > 0
    distance = dijkstra(graph, directed=False, indices=np.flatnonzero(on_limb), min_only=True)
    weight = low * (high / low) ** np.clip(1.0 - distance / radius, 0.0, 1.0)
    return torch.as_tensor(weight, dtype=spec.dtype, device=spec.device)


def smooth(tracks: torch.Tensor, counted: torch.Tensor, *, window_s: float, fps: float) -> torch.Tensor:
    """Pixel tracks ``(P, T, K, 2)`` smoothed over about ``window_s`` seconds, within each run of counted frames.

    A quadratic Savitzky-Golay fit, so accelerating motion survives and frame-to-frame noise does not. It never
    crosses a frame where ``counted (P, T, K)`` is false; a run too short to fit stays as it is, and so does
    everything when ``window_s`` is 0.
    """
    window, order = round(window_s * fps) | 1, 2
    values, runs = tracks.cpu().numpy().copy(), counted.cpu().numpy()
    for person, point in np.ndindex(runs.shape[0], runs.shape[2]):
        edges = np.flatnonzero(np.diff(runs[person, :, point].astype(np.int8), prepend=0, append=0))
        for first, last in zip(edges[::2], edges[1::2]):
            width = min(window, (last - first - 1) | 1)
            if width > order:
                values[person, first:last, point] = savgol_filter(
                    values[person, first:last, point], width, order, axis=0, mode="interp")
    return torch.as_tensor(values, device=tracks.device)


def _saturating(squared: torch.Tensor, scale: float) -> torch.Tensor:
    """Geman-McClure in [0, 1): ``squared / (scale^2 + squared)``."""
    return optim.GemanMcClure(c=scale).rho(squared) * (2.0 / scale**2)


def _saturated(distance: torch.Tensor, limit: float) -> torch.Tensor:
    return limit * torch.tanh(distance / limit)


def _take(points: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``points (B, N, 3)`` at ``index (B, M)``, ``(B, M, 3)``."""
    return points.gather(1, index[..., None].expand(-1, -1, 3))


def _joint_quaternions(pose: torch.Tensor) -> torch.Tensor:
    """The 51 joint quaternions ``(..., 51, 4)`` of a trajectory ``(..., 211)``."""
    return pose[..., 7:].unflatten(-1, (-1, 4))


def _vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Unit normals ``(..., V, 3)`` of mesh vertices ``(..., V, 3)``: the area-weighted mean of the face normals."""
    a, b, c = (vertices[..., faces[:, corner], :] for corner in range(3))
    normals = torch.zeros_like(vertices).index_add_(
        -2, faces.reshape(-1), torch.linalg.cross(b - a, c - a).repeat_interleave(3, dim=-2))
    return torch.nn.functional.normalize(normals, dim=-1)


def _limbs(joint_names: tuple[str, ...]) -> np.ndarray:
    """The limb of each joint ``(J,)``: 1 to 4 for the hands and feet of :data:`_LIMBS`, 0 for the rest."""
    limbs = np.zeros(len(joint_names), int)
    for limb, (side, parts) in enumerate(_LIMBS, start=1):
        limbs[[name.startswith(side) and any(part in name for part in parts) for name in joint_names]] = limb
    return limbs


def _strided(indices: np.ndarray, count: int) -> np.ndarray:
    """At most ``count`` of ``indices``, evenly spread."""
    if len(indices) <= count:
        return indices
    return indices[np.linspace(0, len(indices) - 1, count).astype(int)]
