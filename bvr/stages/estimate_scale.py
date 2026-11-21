"""Step ``scale``: the metric scale of the fused world, read off the bodies, and the bodies seated on their depth.

VGGT's world is right up to one factor ``s`` per clip. The SMPL-X bodies of the converter have metric size and
sit in the same cameras, so over a person's pixels the body's depth and the fused cloud's depth differ by ``s``.
Depths are compared as ``log z_body - log z_cloud``, so a factor too large and one too small weigh alike.

1. The ruler. Per person and tracked frame, the converter's body with that frame's own shape: each vertex that
   faces the camera looks up the cloud point at its pixel, the least confident ``conf_drop_q`` of those drop, and
   the confidence-weighted median of the log ratios is the frame's reading, the summed confidence its weight. A
   frame with fewer than ``min_inlier_verts`` vertices on the cloud has none. The readings of all people pool into
   one ``s``: frames beyond ``mad_k`` MADs of the weighted median drop, the rest are averaged
   (``bvr.geometry.human_scale``).
2. The pre-fit (``prefit``). The converter's bodies sit a little off their depth, and the offset is in the ruler's
   ``s``. So the bodies of the people with keypoints are fitted to their depth points and keypoints, and ``s`` is
   read again from the fit, for up to ``rounds`` rounds. In the first round each person's ``log s`` is a variable
   of the fit: depth points cannot tell a nearer body from a smaller world, so the keypoints set the depth and the
   scale follows. Then each depth point's nearest visible vertex votes its log ratio, weighted by the robust kernel,
   and these readings pool as the ruler's do; the later rounds hold every person at the pooled ``s``. The rounds
   stop once ``s`` moves less than 0.1 %. A person's shape is the median of their ``betas`` and stays fixed (the
   body's size is the ruler); the fingers stay too.
3. The metric world: points, camera translations and the voxel size times ``s``.

A frame whose human points smear along the view ray (``bvr.geometry.depth_smear``) weighs less in every reading
and in the fit. The median smear over the clip is stored for the quality flag of the contacts steps.

The pre-fit objective, one loss per person, weighted by ``[scale.weights]``:

- ``chamfer``: every depth point near its nearest visible vertex, Geman-McClure at ``gm_chamfer_m``. One-sided:
  the cloud sees only the front, and pulling hidden vertices onto it folds limbs. It is measured in the cloud's
  units, the body and the kernel divided by ``s``: in metres the cloud's noise grows with ``s``, and the term
  would shrink the world whatever the evidence;
- ``reprojection``: the Sapiens body and face keypoints, Geman-McClure at ``gm_reproj_px``;
- ``pose``, ``orientation``: the joints and the root near the converter's rotations;
- ``translation``: the root near the converter's position, in m^2; it holds the depth where the image says little;
- ``start_velocity``: the joints move as the converter's do, per reference step (``bvr.terms.motion``).

Writes, under the clip's directory (the files of ``geometry/fused/``, in metres)::

    geometry/transform.npz   bvr-transform: extrinsics in metres, scale = s, metric = true
    geometry/scene.npz       bvr-scene: points and voxel_size in metres
    geometry/human.npz       bvr-human-points: points in metres
    geometry/scale.npz       bvr-scale: scale, scale_ruler, log_scale, scale_mad, n_frames_used, n_people, method
                             ("prefit-dual" or "human-log-ratio"); per_frame_scale, per_frame_weight (P, T) of the
                             last reading; frame_indices, object_ids; smear_median, the median depth smear of the
                             human points over people and frames (``bvr.geometry.quality.smear_stats``)
    geometry/prefit.npz      bvr-prefit: the classic SMPL-X fields of sam3d/smplx_params.npz in the camera frames,
                             the fitted people's tracked frames replaced by the fit; valid_mask, object_ids, scale
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial

import better_human as bh
import numpy as np
import torch
from better_robot import optim, terms

from bvr import body, files, paths, run
from bvr.geometry.depth_smear import alongray_smear, smear_weight
from bvr.geometry.human_scale import combine_log_scales, weighted_median
from bvr.geometry.intrinsics import denormalize_K
from bvr.geometry.normals import vertex_normals
from bvr.geometry.quality import smear_stats
from bvr.terms import Term, motion, objective, visual

log = logging.getLogger(__name__)

POSE_FIELDS = ("global_orient", "transl", "body_pose", "left_hand_pose", "right_hand_pose")
#: Rows per nearest-neighbour search: it bounds the search's memory, not its result.
CHUNK = 16


def load(stage: run.Stage) -> Callable[[paths.Clip], None]:
    human = body.load(stage.device)
    return lambda clip: estimate_scale(clip, human, stage.config["scale"])


def estimate_scale(clip: paths.Clip, human: bh.HumanModel, options: dict) -> None:
    """Measure the scale of ``clip``'s fused world and write the metric files and the pre-fit bodies."""
    fused = clip.find(paths.FUSED)
    transform = files.load_npz(fused / "transform.npz", "bvr-transform")
    points = files.load_npz(fused / "human.npz", "bvr-human-points")
    smplx = files.load_npz(clip.find(paths.SMPLX_PARAMS), "bvr-smplx")
    tracked = smplx["valid_mask"].astype(bool)
    clouds = Clouds(points, smplx["object_ids"], transform["extrinsics"])

    readings = _ruler(human, smplx, tracked, clouds, transform, options)
    ruler = combine_log_scales(*(reading.ravel() for reading in readings), mad_k=options["mad_k"])
    pooled, fitted = ruler, None
    if options["prefit"]:
        keypoints = files.load_npz(clip.find(paths.SAPIENS_POSE), "bvr-keypoints")
        people = np.flatnonzero(tracked.any(axis=1) & np.isin(smplx["object_ids"], keypoints["object_ids"]))
        if len(people):
            pooled, readings, fitted = _prefit(human, smplx, keypoints, tracked, people, clouds, transform, ruler,
                                               options)
        else:
            log.warning("no person with a tracked frame and keypoints: the pre-fit keeps the converter's bodies")
    scale = pooled["scale"]
    # A ratio of extents, so it reads the same before and after the scale.
    smear, _ = smear_stats(points["points"], points["frame_id"], points["obj_id"], transform["extrinsics"])
    log.info("scale %.4f (ruler %.4f), MAD %.4f, %d of %d frames; median depth smear %.2f", scale, ruler["scale"],
             pooled["mad"], pooled["n_used"], int(np.isfinite(readings[0]).sum()), smear)

    scene = files.load_npz(fused / "scene.npz", "bvr-scene")
    extrinsics = transform["extrinsics"].copy()
    extrinsics[:, :3, 3] *= scale
    files.save_npz(clip.path(paths.TRANSFORM), "bvr-transform", **transform | {
        "extrinsics": extrinsics, "scale": np.float32(scale), "metric": np.bool_(True)})
    files.save_npz(clip.path(paths.SCENE), "bvr-scene", **scene | {
        "points": scene["points"] * np.float32(scale), "voxel_size": np.float32(scene["voxel_size"] * scale)})
    files.save_npz(clip.path(paths.HUMAN), "bvr-human-points", **points | {
        "points": points["points"] * np.float32(scale)})
    files.save_npz(
        clip.path(paths.SCALE), "bvr-scale", scale=np.float32(scale), scale_ruler=np.float32(ruler["scale"]),
        log_scale=np.float32(pooled["log_scale"]), scale_mad=np.float32(pooled["mad"]),
        n_frames_used=np.int32(pooled["n_used"]), n_people=np.int32(len(tracked)),
        method=np.array("prefit-dual" if options["prefit"] else "human-log-ratio"),
        per_frame_scale=np.exp(readings[0]).astype(np.float32), per_frame_weight=readings[1].astype(np.float32),
        frame_indices=transform["frame_indices"].astype(np.int32), object_ids=smplx["object_ids"],
        smear_median=np.float32(smear))
    if options["prefit"]:
        _save_prefit(clip.path(paths.PREFIT), smplx, tracked, fitted, scale)


class Clouds:
    """The fused human points of a clip by person and frame, each frame's in its own camera's frame."""

    def __init__(self, points: dict[str, np.ndarray], object_ids: np.ndarray, extrinsics: np.ndarray) -> None:
        self.points, self.confidence, self.extrinsics = points["points"], points["confidence"], extrinsics
        key = points["obj_id"].astype(np.int64) * len(extrinsics) + points["frame_id"]
        self.order = np.argsort(key, kind="stable")
        first = object_ids.astype(np.int64)[:, None] * len(extrinsics) + np.arange(len(extrinsics))
        self.bounds = np.searchsorted(key[self.order], np.stack((first, first + 1)))  # (2, P, T)

    def __call__(self, person: int, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """The points ``(n, 3)`` of body-file row ``person`` in frame ``frame``, and their confidence."""
        own = self.order[self.bounds[0, person, frame]:self.bounds[1, person, frame]]
        rotation, translation = self.extrinsics[frame, :3, :3], self.extrinsics[frame, :3, 3]
        return self.points[own] @ rotation.T + translation, self.confidence[own]


def _ruler(human: bh.HumanModel, smplx: dict, tracked: np.ndarray, clouds: Clouds, transform: dict,
           options: dict) -> tuple[np.ndarray, np.ndarray]:
    """The ruler's log-scale and weight per person and frame, ``(P, T)`` each, ``nan`` and 0 where there is none."""
    size = int(transform["vggt_height"]), int(transform["vggt_width"])
    intrinsics = denormalize_K(transform["intrinsics"], *size)  # the pixels the cloud was back-projected from
    logs, weights = np.full(tracked.shape, np.nan), np.zeros(tracked.shape)
    for person, seen in enumerate(tracked):
        frames = np.flatnonzero(seen)
        if not len(frames):
            continue
        with torch.no_grad():
            identity, q, _ = bh.from_classic(human, bh.families.SMPLXClassic(**{
                name: torch.as_tensor(smplx[name][person, frames], device=human.spec.device)
                for name in ("betas", *POSE_FIELDS)}))
            mesh = identity.forward(q).vertices
            facing = (vertex_normals(mesh, human.spec.faces) * mesh).sum(dim=-1) < 0  # the camera is the origin
        for frame, vertices, front in zip(frames, mesh.cpu().numpy(), facing.cpu().numpy()):
            cloud, trust = clouds(person, frame)
            if len(cloud) >= options["min_inlier_verts"]:
                logs[person, frame], weights[person, frame] = _reading(
                    vertices[front], cloud, trust, intrinsics[frame], size, options)
    return logs, weights


def _reading(vertices: np.ndarray, cloud: np.ndarray, trust: np.ndarray, intrinsic: np.ndarray,
             size: tuple[int, int], options: dict) -> tuple[float, float]:
    """One frame's log-scale and weight: front ``vertices (V, 3)`` against the ``cloud (n, 3)`` point at their pixel.

    Both are in the camera's frame; ``trust (n,)`` is the cloud's confidence and ``intrinsic (3, 3)`` its pixels.
    """
    def pixels(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        depth = xyz[:, 2]
        safe = np.where(depth > 0, depth, 1.0)
        u = np.rint(intrinsic[0, 0] * xyz[:, 0] / safe + intrinsic[0, 2]).astype(np.int64)
        v = np.rint(intrinsic[1, 1] * xyz[:, 1] / safe + intrinsic[1, 2]).astype(np.int64)
        return v, u, (depth > 0) & (u >= 0) & (u < size[1]) & (v >= 0) & (v < size[0])

    v, u, inside = pixels(cloud)
    smear = smear_weight(alongray_smear(cloud[inside]))
    depth_map, trust_map = np.full(size, np.nan, np.float32), np.zeros(size, np.float32)
    depth_map[v[inside], u[inside]], trust_map[v[inside], u[inside]] = cloud[inside, 2], trust[inside]
    v, u, inside = pixels(vertices)
    body_depth, cloud_depth = vertices[inside, 2], depth_map[v[inside], u[inside]]
    trust = trust_map[v[inside], u[inside]]
    hit = np.isfinite(cloud_depth) & (cloud_depth > 0)
    if hit.sum() >= options["min_inlier_verts"]:
        kept = hit & (trust >= np.quantile(trust[hit], options["conf_drop_q"]))
        if kept.sum() >= options["min_inlier_verts"]:
            ratios = np.log(body_depth[kept]) - np.log(cloud_depth[kept])
            return weighted_median(ratios, trust[kept]), trust[kept].sum() * smear
    return np.nan, 0.0


def _prefit(human: bh.HumanModel, smplx: dict, keypoints: dict, tracked: np.ndarray, people: np.ndarray,
            clouds: Clouds, transform: dict, ruler: dict,
            options: dict) -> tuple[dict, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, bh.Identity, torch.Tensor]]:
    """Fit ``people`` (rows of the body file) and the scale together, round by round.

    Returns the last pooled reading, the readings ``(P, T)`` behind it, and the fit: ``people``, their bodies and their
    trajectories ``q (P, T, 211)`` in the camera frames.
    """
    device = human.spec.device
    tensor = partial(torch.as_tensor, device=device)
    seen = tracked[people]
    betas = np.stack([np.median(smplx["betas"][person, seen[row]], axis=0) for row, person in enumerate(people)])
    identity, start, _ = bh.from_classic(human, bh.families.SMPLXClassic(betas=tensor(betas)[:, None], **{
        name: tensor(np.where(seen[..., None], np.nan_to_num(smplx[name][people]), 0.0))
        for name in POSE_FIELDS}))
    cloud, counted, smear = (tensor(array) for array in _depth_points(clouds, people, seen, options))
    sample = torch.arange(options["vertex_subsample"], device=device) * (human.spec.num_vertices
                                                                          / options["vertex_subsample"])
    sample = torch.unique(sample.long().clamp_max(human.spec.num_vertices - 1))
    sapiens = [list(keypoints["object_ids"]).index(person) for person in smplx["object_ids"][people]]
    cameras = (torch.eye(4, device=device).expand(tracked.shape[1], 4, 4), tensor(transform["intrinsics_px_orig"]))
    time, seen = motion.Time(float(transform["fps"])), tensor(seen)

    problem = optim.Problem({"q": start, "log_scale": start.new_full((len(people), 1), ruler["log_scale"])},
                            data={"visible": None}, time=("q",))
    problem.cached(terms.RobotManifold(human.model).project, name="pose", reads=("q",))
    problem.cached(bh.terms.human_state(identity, q="pose"), name="body")
    term = partial(problem.scalar, weight=0.0)
    term("chamfer", _chamfer(cloud, counted, smear, sample, gm_scale_m=options["gm_chamfer_m"]))
    term("reprojection", visual.keypoint_reprojection(
        human, tensor(keypoints["keypoints"][sapiens]), tensor(keypoints["keypoint_scores"][sapiens]),
        seen, cameras, time.fps, score_thresh=options["score_thresh"], include_hands=False,
        gm_scale_px=options["gm_reproj_px"], smooth_window_s=0.0))
    term("pose", visual.pose_prior(start))
    term("orientation", visual.orientation_prior(start))
    term("translation", _translation(start, seen))
    term("start_velocity", _start_velocity(objective.posed(identity, start).joint_translations, seen, time))

    fingers = torch.arange(human.model.nq, device=device) >= 7 + 4 * (body.BODY_JOINTS - 1)
    x, previous = problem.variables, ruler
    for round in range(1, options["rounds"] + 1):
        held = {"log_scale": torch.zeros(1, dtype=torch.bool, device=device)} if round > 1 else {}
        result = optim.run(problem, [optim.Phase(
            optim.TorchOptim(lambda parameters: torch.optim.Adam(parameters, lr=options["lr"])), options["iters"],
            name=f"round {round}", weights=options["weights"], free={"q": ~fingers} | held, stop_every=0,
            on_start=lambda problem, x: (x, {"visible": _facing(objective.posed(identity, x["q"]), sample)}))], x=x)
        readings = _measure(identity, result.x, cloud, counted, smear, sample, gm_scale_m=options["gm_chamfer_m"])
        pooled = combine_log_scales(*(reading.ravel() for reading in readings), mad_k=options["mad_k"])
        log.info("round %d: s %.4f -> %.4f; %s", round, previous["scale"], pooled["scale"], "  ".join(
            f"{name} {'/'.join(f'{value:.4g}' for value in (loss / options['weights'][name]).tolist())}"
            for name, loss in result.state.extra["terms"].items()))
        x = {"q": result.x["q"], "log_scale": torch.full_like(result.x["log_scale"], pooled["log_scale"])}
        if abs(pooled["log_scale"] - previous["log_scale"]) < 1e-3:
            break
        previous = pooled
    logs, weights = np.full(tracked.shape, np.nan), np.zeros(tracked.shape)
    logs[people], weights[people] = readings
    return pooled, (logs, weights), (people, identity, terms.RobotManifold(human.model).project(x["q"]))


def _save_prefit(path, smplx: dict, tracked: np.ndarray, fitted: tuple[np.ndarray, bh.Identity, torch.Tensor] | None,
                 scale: float) -> None:
    """Write the converter's classic fields with the fitted people's tracked frames replaced by their fit."""
    fields = {name: smplx[name].astype(np.float32) for name in ("betas", *POSE_FIELDS)}
    if fitted is not None:
        people, identity, q = fitted
        classic = bh.to_classic(identity, q)
        fit = {name: getattr(classic, name).cpu().numpy() for name in POSE_FIELDS}
        fit["betas"] = identity.values["betas"].expand(*q.shape[:2], -1).cpu().numpy()
        for row, person in enumerate(people):
            for name, values in fit.items():
                fields[name][person, tracked[person]] = values[row, tracked[person]]
    files.save_npz(path, "bvr-prefit", **fields, scale=np.float32(scale), **{
        name: smplx[name] for name in ("valid_mask", "object_ids", "num_people", "num_frames", "video_height",
                                       "video_width")})


def _depth_points(clouds: Clouds, people: np.ndarray, tracked: np.ndarray,
                  options: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The fit's depth points ``(P, T, M, 3)`` in the camera frames, the mask ``(P, T, M)`` of real ones, and each
    frame's smear weight ``(P, T)``: per frame the points above the ``conf_drop_q`` quantile of confidence, at most
    ``max_points`` of them at even strides."""
    most = options["max_points"]
    cloud = np.zeros((*tracked.shape, most, 3), np.float32)
    counted, smear = np.zeros(cloud.shape[:3], bool), np.ones(tracked.shape, np.float32)
    for row, person in enumerate(people):
        for frame in np.flatnonzero(tracked[row]):
            points, trust = clouds(person, frame)
            if len(points):
                points = points[trust >= np.quantile(trust, options["conf_drop_q"])]
            points = points[(np.arange(most) * (len(points) / most)).astype(np.int64)] if len(points) > most else points
            cloud[row, frame, :len(points)], counted[row, frame, :len(points)] = points, True
            smear[row, frame] = smear_weight(alongray_smear(points))
    return cloud, counted, smear


def _chamfer(cloud: torch.Tensor, counted: torch.Tensor, smear: torch.Tensor, sample: torch.Tensor, *,
             gm_scale_m: float) -> Term:
    """Term: each depth point near its nearest visible vertex, ``d^2 / (c^2 + d^2)`` with ``c = gm_scale_m / s``.

    Distances are in the cloud's units (the body divided by its person's ``s``). Per frame the mean over the points,
    per person the mean over the frames with a point and a visible vertex, weighted by ``smear (P, T)``.
    """
    people, frames = counted.shape[:2]
    points, real = cloud.flatten(0, 1), counted.flatten(0, 1)

    def loss(body: bh.HumanState, log_scale: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        scale = log_scale.exp()[..., None, None]  # (P, 1, 1, 1)
        vertices, seen = (body.points(sample) / scale).flatten(0, 1), visible.flatten(0, 1)
        matched = vertices.gather(1, _nearest(points, vertices.detach(), seen)[..., None].expand(-1, -1, 3))
        squared = (points - matched).square().sum(dim=-1)
        kernel = (gm_scale_m / scale.detach()).square().expand(-1, frames, 1, 1).reshape(-1, 1)
        rows = (squared / (kernel + squared) * real).sum(dim=1) / real.sum(dim=1).clamp_min(1)
        weight = smear * (seen.any(dim=1) & real.any(dim=1)).view(people, frames)
        return (rows.view(people, frames) * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1e-6)
    return loss


def _translation(start: torch.Tensor, tracked: torch.Tensor) -> Term:
    """Term: the root near its start position, the mean of ``|t - t_start|^2`` over the tracked frames, in m^2."""
    def loss(pose: torch.Tensor) -> torch.Tensor:
        squared = (pose[..., :3] - start[..., :3]).square().sum(dim=-1)
        return (squared * tracked).sum(dim=1) / tracked.sum(dim=1).clamp_min(1)
    return loss


def _start_velocity(start: torch.Tensor, tracked: torch.Tensor, time: motion.Time) -> Term:
    """Term: the joints move as they do at the start, the mean of ``|D p - D p_start|^2`` over joints and counted rows.

    ``start (P, T, J, 3)`` are the start's joints and ``D`` is ``time.difference``, so the unit is m^2 per reference
    step squared. A real motion costs nothing; a jitter the start did not have does.
    """
    rows, reference = time.rows(tracked, 1), time.difference(start)

    def loss(body: bh.HumanState) -> torch.Tensor:
        error = (time.difference(body.joint_translations) - reference).square().sum(dim=-1)
        return (error * rows[..., None]).sum(dim=(1, 2)) / (rows.sum(dim=1) * error.shape[-1]).clamp_min(1)
    return loss


@torch.no_grad()
def _measure(identity: bh.Identity, x: dict[str, torch.Tensor], cloud: torch.Tensor, counted: torch.Tensor,
             smear: torch.Tensor, sample: torch.Tensor, *, gm_scale_m: float) -> tuple[np.ndarray, np.ndarray]:
    """The log-scale and weight each frame reads through the fit's own matches, ``(P, T)`` each.

    At the fitted scale each depth point votes ``log z_vertex - log z_point`` of its nearest visible vertex, with the
    Geman-McClure weight ``(1 + (d / gm_scale_m)^2)^-2`` of their distance in metres. A frame reads the weighted
    median of its votes, with their summed weight times its smear weight; a frame whose votes weigh nothing has none.
    """
    state = objective.posed(identity, x["q"])
    vertices, seen = state.points(sample).flatten(0, 1), _facing(state, sample).flatten(0, 1)
    points = (cloud * x["log_scale"].exp()[..., None, None]).flatten(0, 1)
    matched = vertices.gather(1, _nearest(points, vertices, seen)[..., None].expand(-1, -1, 3))
    depth, matched_depth = cloud.flatten(0, 1)[..., 2], matched[..., 2]
    real = counted.flatten(0, 1) & (depth > 0) & (matched_depth > 0)
    weight = (1 + ((matched - points).norm(dim=-1) / gm_scale_m).square()).pow(-2) * real
    ratio = torch.where(real, matched_depth.clamp_min(1e-9).log() - depth.clamp_min(1e-9).log(), 0.0)
    logs, weights = np.full(len(points), np.nan), np.zeros(len(points))
    for row, (votes, strengths) in enumerate(zip(ratio.cpu().numpy(), weight.cpu().numpy())):
        if strengths.sum() > 1e-6:
            logs[row], weights[row] = weighted_median(votes, strengths), strengths.sum()
    shape = counted.shape[:2]
    return logs.reshape(shape), weights.reshape(shape) * smear.cpu().numpy()


def _facing(state: bh.HumanState, sample: torch.Tensor) -> torch.Tensor:
    """Which vertices of ``sample`` face the camera, ``(..., Ns)``, for bodies in the camera's frame."""
    mesh = state.vertices
    normals = vertex_normals(mesh, state.identity.human.spec.faces)[..., sample, :]
    return (normals * mesh[..., sample, :]).sum(dim=-1) < 0


@torch.no_grad()
def _nearest(points: torch.Tensor, vertices: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    """The index of each point's nearest visible vertex ``(R, M)``, for ``points (R, M, 3)``, ``vertices (R, N, 3)``."""
    found = []
    for start in range(0, len(points), CHUNK):
        distance = torch.cdist(points[start:start + CHUNK], vertices[start:start + CHUNK])
        found.append(distance.masked_fill_(~visible[start:start + CHUNK, None], torch.inf).min(dim=-1).indices)
    return torch.cat(found)


if __name__ == "__main__":
    run.main(load)
