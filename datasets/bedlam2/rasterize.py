"""Z-buffer rasterisation of posed meshes into a per-pixel instance map.

One job: given several meshes already projected into one image, say which mesh
owns each pixel. That is what turns a set of SMPL-X bodies into segmentation
masks — self-occlusion and occlusion *between* people both fall out of the depth
test, so no visibility reasoning is needed anywhere else.

The rasteriser is a plain scanline z-buffer written in torch, because neither
``nvdiffrast`` nor ``pytorch3d`` is installed and the job is small: a body is
20 908 triangles, each a few pixels across. Every triangle emits the integer
sample points inside its pixel bounding box, a barycentric test keeps the ones it
actually covers, and one ``scatter_reduce`` keeps the nearest depth per pixel.
No gradients — this is a measurement, never something to optimize through.
"""
from __future__ import annotations

import torch

#: Depth (metres) a vertex must clear to be rasterised. A triangle reaching closer
#: than this projects to arbitrarily large pixel coordinates, which both wrecks the
#: float32 edge functions and makes its pixel bounding box the whole image.
NEAR_M = 0.05

#: Widest pixel bounding box a single triangle may have, as a multiple of the image
#: diagonal. Anything wider is a sliver from a vertex grazing the near plane; it is
#: dropped, and :func:`instance_map` reports how often that happened.
MAX_SPAN_DIAG = 4.0

#: Candidate sample points held in flight at once. Each costs ~40 B of working
#: tensors, so this is ~300 MB — triangles are chunked to stay under it.
MAX_SAMPLES = 8_000_000


def project(points_world: torch.Tensor, extrinsic: torch.Tensor,
            intrinsic_px: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """World points ``(..., 3)`` → pixel coordinates ``(..., 2)`` and depth ``(...)``.

    ``extrinsic`` is camera-from-world ``(4, 4)`` and ``intrinsic_px`` a pixel-unit
    OpenCV ``K (3, 3)``, so a point on the optical axis lands on ``(cx, cy)`` and
    pixel ``(col, row)`` is sampled at exactly that continuous coordinate.
    """
    camera = points_world @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    depth = camera[..., 2]
    uv = camera[..., :2] / depth.clamp_min(1e-6).unsqueeze(-1)
    uv = uv * torch.stack((intrinsic_px[0, 0], intrinsic_px[1, 1]))
    return uv + torch.stack((intrinsic_px[0, 2], intrinsic_px[1, 2])), depth


def instance_map(verts_px: torch.Tensor, depth: torch.Tensor, faces: torch.Tensor,
                 *, height: int, width: int, near: float = NEAR_M,
                 max_samples: int = MAX_SAMPLES) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Rasterise ``P`` instances of one mesh topology into an instance map.

    :param verts_px: ``(P, V, 2)`` projected pixel coordinates, one row per instance.
    :param depth: ``(P, V)`` camera-space depth in metres, matching ``verts_px``.
    :param faces: ``(T, 3)`` vertex indices, shared by every instance.
    :param height: image height in pixels.
    :param width: image width in pixels.
    :param near: vertices closer than this are dropped with their triangles.
    :returns: ``(ids (H, W) uint8, zbuf (P, H, W) float32, dropped)`` — ``ids`` is 0
        for background and ``p + 1`` for whichever instance is nearest, ``zbuf`` is
        each instance's **own** depth buffer (``inf`` where it has no surface, so
        ``isfinite`` is its silhouette before anyone occludes it), and ``dropped``
        counts triangles skipped as slivers.
    """
    device = verts_px.device
    P = verts_px.shape[0]
    zbuf = torch.full((P * height * width,), float("inf"), dtype=torch.float32, device=device)
    if P == 0:
        return (torch.zeros((height, width), dtype=torch.uint8, device=device),
                zbuf.view(P, height, width), 0)

    corner = verts_px[:, faces]                       # (P, T, 3, 2)
    zcorner = depth[:, faces]                         # (P, T, 3)
    corner = corner.reshape(-1, 3, 2)
    zcorner = zcorner.reshape(-1, 3)
    owner = torch.arange(P, device=device).repeat_interleave(faces.shape[0])

    lo = corner.amin(dim=1)                           # (M, 2) continuous bbox
    hi = corner.amax(dim=1)
    span = (hi - lo).amax(dim=1)
    x0 = lo[:, 0].ceil().clamp_(0, width - 1).long()
    x1 = hi[:, 0].floor().clamp_(0, width - 1).long()
    y0 = lo[:, 1].ceil().clamp_(0, height - 1).long()
    y1 = hi[:, 1].floor().clamp_(0, height - 1).long()
    diag = float((height ** 2 + width ** 2) ** 0.5)
    sliver = span > MAX_SPAN_DIAG * diag
    live = ((zcorner > near).all(dim=1) & ~sliver
            & (x1 >= x0) & (y1 >= y0)
            & (hi[:, 0] >= 0) & (lo[:, 0] <= width - 1)
            & (hi[:, 1] >= 0) & (lo[:, 1] <= height - 1))
    dropped = int((sliver & (zcorner > near).all(dim=1)).sum())

    keep = torch.nonzero(live, as_tuple=True)[0]
    corner, zcorner, owner = corner[keep], zcorner[keep], owner[keep]
    x0, x1, y0, y1 = x0[keep], x1[keep], y0[keep], y1[keep]
    boxw, boxh = x1 - x0 + 1, y1 - y0 + 1
    count = boxw * boxh
    if not len(count):
        # Nothing survives when every body is out of shot — 5.3% of BEDLAM2's
        # person-frames are, because the renderer gives crowd members ground truth
        # whether or not the camera can see them. An empty image is the right answer.
        return (torch.zeros((height, width), dtype=torch.uint8, device=device),
                zbuf.view(P, height, width), dropped)

    # Triangles are chunked so the flattened candidate list stays bounded: a body
    # whose feet graze the lens can otherwise ask for billions of samples at once.
    edge = torch.cumsum(count, 0)
    chunk_at = torch.searchsorted(edge, torch.arange(0, int(edge[-1]) + max_samples,
                                                     max_samples, device=device))
    bounds = torch.unique(torch.cat([chunk_at, torch.tensor([len(count)], device=device)]))
    start = 0
    for stop in bounds.tolist():
        if stop <= start:
            continue
        sl = slice(start, stop)
        start = stop
        n = count[sl]
        tri = torch.repeat_interleave(torch.arange(n.shape[0], device=device), n)
        within = torch.arange(int(n.sum()), device=device) - (torch.cumsum(n, 0) - n)[tri]
        w = boxw[sl][tri]
        px = (x0[sl][tri] + within % w).to(torch.float32)
        py = (y0[sl][tri] + torch.div(within, w, rounding_mode="floor")).to(torch.float32)

        a, b, c = (corner[sl][tri, i] for i in range(3))
        # Edge functions about the sample point: positive on one winding, negative on
        # the other, and their sum is twice the signed triangle area.
        e0 = (b[:, 0] - a[:, 0]) * (py - a[:, 1]) - (b[:, 1] - a[:, 1]) * (px - a[:, 0])
        e1 = (c[:, 0] - b[:, 0]) * (py - b[:, 1]) - (c[:, 1] - b[:, 1]) * (px - b[:, 0])
        e2 = (a[:, 0] - c[:, 0]) * (py - c[:, 1]) - (a[:, 1] - c[:, 1]) * (px - c[:, 0])
        area = e0 + e1 + e2
        inside = (((e0 >= 0) & (e1 >= 0) & (e2 >= 0))
                  | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))) & (area != 0)
        if not bool(inside.any()):
            continue

        tri, px, py = tri[inside], px[inside], py[inside]
        e0, e1, e2, area = e0[inside], e1[inside], e2[inside], area[inside]
        zt = zcorner[sl][tri]
        # Screen-space barycentric interpolation is exact in 1/z, never in z.
        inv = (e1 / zt[:, 0] + e2 / zt[:, 1] + e0 / zt[:, 2]) / area
        flat = (owner[sl][tri] * height + py.long()) * width + px.long()
        zbuf.scatter_reduce_(0, flat, 1.0 / inv.clamp_min(1e-9), reduce="amin")

    stack = zbuf.view(P, height, width)
    best, who = stack.min(dim=0)
    ids = torch.where(torch.isfinite(best), (who + 1).to(torch.uint8),
                      torch.zeros((), dtype=torch.uint8, device=device))
    return ids, stack, dropped


__all__ = ["project", "instance_map", "NEAR_M"]
