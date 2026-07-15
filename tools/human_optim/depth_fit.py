"""Per-joint fit of the posed body to the human depth cloud.

The diagnostic compares cloud and mesh camera depth along matching image rays.
For every frame, posed mesh vertices form a coarse pixel-cell z-buffer; each
front-surface cell retains the owning joint from ``v2j``. Reprojected human-cloud
points match the nearest occupied cell in a 3×3 neighbourhood, and the signed
residual ``z_cloud - z_mesh`` is reduced to a robust per-joint median.

Trust uses ``log(z_cloud / z_mesh)`` rather than absolute metres. With equivalent
scales on the six validation clips, log depth preserved the high low-trust rate
on the three near pathological clips (34–39%) while reducing the far clean clip
from 13% to 1%; physical gaps alone over-penalized distant people. Missing data
has trust 1 because only 51–67% of all joint-frames had enough points: absence of
a local witness must leave confidence unchanged rather than erase motion evidence.
"""

from __future__ import annotations

import torch

from .camera import project, transform_points


DEPTH_FIT_DEFAULTS = {
    "cell_px": 8,
    "min_points": 5,
    "trust_scale_log": 0.05,
}
NO_DATA_TRUST = 1.0


@torch.no_grad()
def _frame_matches(
    mesh_cam: torch.Tensor,         # (V, 3)
    mesh_px: torch.Tensor,          # (V, 2)
    cloud_cam: torch.Tensor,        # (M, 3)
    cloud_px: torch.Tensor,         # (M, 2)
    v2j: torch.Tensor,              # (V,)
    cell_px: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Matched ``(residual_m, log_depth_ratio, owner_joint)`` for one frame."""
    mesh_valid = (mesh_cam[:, 2] > 0.0) & torch.isfinite(mesh_cam).all(-1)
    mesh_valid &= torch.isfinite(mesh_px).all(-1)
    cloud_valid = (cloud_cam[:, 2] > 0.0) & torch.isfinite(cloud_cam).all(-1)
    cloud_valid &= torch.isfinite(cloud_px).all(-1)
    if not bool(mesh_valid.any()) or not bool(cloud_valid.any()):
        empty = mesh_cam.new_empty(0)
        return empty, empty, v2j.new_empty(0)

    mz = mesh_cam[mesh_valid, 2]
    mpx = mesh_px[mesh_valid]
    mowner = v2j[mesh_valid]
    cz = cloud_cam[cloud_valid, 2]
    cpx = cloud_px[cloud_valid]
    mc = torch.floor(mpx / float(cell_px)).to(torch.int64)
    cc = torch.floor(cpx / float(cell_px)).to(torch.int64)

    lo = torch.minimum(mc.amin(0), cc.amin(0) - 1)
    hi = torch.maximum(mc.amax(0), cc.amax(0) + 1)
    width = hi[0] - lo[0] + 1

    def cell_key(cell: torch.Tensor) -> torch.Tensor:
        shifted = cell - lo
        return shifted[..., 1] * width + shifted[..., 0]

    keys, inverse = torch.unique(cell_key(mc), sorted=True, return_inverse=True)
    num_cells = keys.numel()
    cell_z = torch.full((num_cells,), float("inf"), dtype=mz.dtype, device=mz.device)
    cell_z.scatter_reduce_(0, inverse, mz, reduce="amin", include_self=True)
    vertex_index = torch.arange(mz.numel(), device=mz.device)
    front_candidate = torch.where(
        mz == cell_z[inverse], vertex_index, torch.full_like(vertex_index, mz.numel()))
    front = torch.full(
        (num_cells,), mz.numel(), dtype=torch.long, device=mz.device)
    front.scatter_reduce_(0, inverse, front_candidate, reduce="amin", include_self=True)
    cell_uv = mpx[front]
    cell_owner = mowner[front]

    offsets = torch.tensor(
        [[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0],
         [1, 0], [-1, 1], [0, 1], [1, 1]],
        dtype=torch.int64, device=mc.device)
    query = cell_key(cc[:, None, :] + offsets[None])
    position = torch.searchsorted(keys, query)
    safe_position = position.clamp_max(num_cells - 1)
    hit = (position < num_cells) & (keys[safe_position] == query)
    distance_sq = ((cpx[:, None, :] - cell_uv[safe_position]) ** 2).sum(-1)
    distance_sq = torch.where(hit, distance_sq, torch.full_like(distance_sq, float("inf")))
    # A front vertex only vouches for pixels within cell_px of its own
    # projection. Without this cap, a thin foreground limb owning one cell can
    # claim cloud pixels up to two cells away and book their (torso/background)
    # depth difference as its own gap — metre-scale false spikes on clean clips.
    distance_sq = torch.where(distance_sq <= float(cell_px) ** 2, distance_sq,
                              torch.full_like(distance_sq, float("inf")))
    best = distance_sq.argmin(1)
    matched = torch.isfinite(distance_sq.gather(1, best[:, None])[:, 0])
    selected = safe_position.gather(1, best[:, None])[:, 0]
    mesh_z = cell_z[selected[matched]]
    cloud_z = cz[matched]
    return cloud_z - mesh_z, torch.log(cloud_z / mesh_z), cell_owner[selected[matched]]


@torch.no_grad()
def _joint_depth_statistics(
    vertices_world: torch.Tensor,       # (T, V, 3) posed world-frame vertices
    cloud_points_world: torch.Tensor,   # (M, 3) human cloud for all people/frames
    cloud_frame_id: torch.Tensor,       # (M,) frame index
    cloud_obj_id: torch.Tensor,         # (M,) tracked object ID
    object_id: int,
    extrinsics: torch.Tensor,           # (T, 4, 4) world→camera
    intrinsics: torch.Tensor,           # (T, 3, 3) original-resolution pixel intrinsics
    v2j: torch.Tensor,                  # (V,) vertex→joint
    num_joints: int,
    *,
    cell_px: int = 8,
    min_points: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-joint ``(gap_m, log_depth_ratio, matched_points)``."""
    if cell_px <= 0 or min_points <= 0:
        raise ValueError("cell_px and min_points must be positive")
    T = vertices_world.shape[0]
    dtype, device = vertices_world.dtype, vertices_world.device
    gap_m = torch.full((T, num_joints), float("nan"), dtype=dtype, device=device)
    log_ratio = torch.full_like(gap_m, float("nan"))
    points = torch.zeros((T, num_joints), dtype=dtype, device=device)

    person = cloud_obj_id == int(object_id)
    cloud = cloud_points_world[person]
    frame_id = cloud_frame_id[person].long()
    if frame_id.numel() and bool((frame_id[1:] < frame_id[:-1]).any()):
        order = frame_id.argsort()
        frame_id, cloud = frame_id[order], cloud[order]
    frame_counts = torch.bincount(frame_id, minlength=T)
    frame_starts = frame_counts.cumsum(0) - frame_counts

    mesh_cam = transform_points(vertices_world, extrinsics)
    mesh_px = project(mesh_cam, intrinsics)
    for t in range(T):
        count = int(frame_counts[t])
        if count == 0:
            continue
        start = int(frame_starts[t])
        cloud_world = cloud[start:start + count][None]
        cloud_cam = transform_points(cloud_world, extrinsics[t:t + 1])
        cloud_px = project(cloud_cam, intrinsics[t:t + 1])
        residual, log_residual, owner = _frame_matches(
            mesh_cam[t], mesh_px[t], cloud_cam[0], cloud_px[0], v2j, cell_px)
        for j in range(num_joints):
            owned = owner == j
            values = residual[owned]
            points[t, j] = values.numel()
            if values.numel() >= min_points:
                gap_m[t, j] = values.median()
                log_ratio[t, j] = log_residual[owned].median()

    return gap_m, log_ratio, points


@torch.no_grad()
def joint_depth_fit(
    vertices_world: torch.Tensor,       # (T, V, 3) posed world-frame vertices
    cloud_points_world: torch.Tensor,   # (M, 3) human cloud for all people/frames
    cloud_frame_id: torch.Tensor,       # (M,) frame index
    cloud_obj_id: torch.Tensor,         # (M,) tracked object ID
    object_id: int,
    extrinsics: torch.Tensor,           # (T, 4, 4) world→camera
    intrinsics: torch.Tensor,           # (T, 3, 3) original-resolution pixel intrinsics
    v2j: torch.Tensor,                  # (V,) vertex→joint
    num_joints: int,
    *,
    cell_px: int = 8,
    min_points: int = 5,
    trust_scale_log: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(gap_cm, matched_points, trust)``, each float32 ``(T, J)``.

    ``gap_cm`` is the signed median camera-depth residual, positive when the
    cloud lies beyond the fitted mesh. It is NaN where fewer than ``min_points``
    matched. ``matched_points`` remains the raw count, and ``trust`` is
    ``exp(-(median(log(z_cloud / z_mesh)) / trust_scale_log)²)`` with
    :data:`NO_DATA_TRUST` at NaN gaps.
    """
    if not (0.0 < trust_scale_log < float("inf")):
        raise ValueError("trust_scale_log must be a positive finite number")
    gap_m, log_ratio, points = _joint_depth_statistics(
        vertices_world, cloud_points_world, cloud_frame_id, cloud_obj_id,
        object_id, extrinsics, intrinsics, v2j, num_joints,
        cell_px=cell_px, min_points=min_points)

    trust = torch.exp(-torch.square(log_ratio / float(trust_scale_log)))
    trust = torch.where(
        torch.isfinite(gap_m), trust, torch.full_like(trust, NO_DATA_TRUST))
    return gap_m * 100.0, points, trust
