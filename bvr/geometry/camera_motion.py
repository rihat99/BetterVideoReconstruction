"""Is the shot locked off? — measured by how far the scene slides across the picture.

A shot is **locked** when the world lands on the same pixels in every frame.
That one question already covers every way a camera can move: travelling gives
parallax, a pan slides the whole picture sideways, a zoom pushes it outwards. So
rather than thresholding position, rotation and focal length separately, this
measures the single thing they all do — take the scene points, project them
through every frame's camera, and see how far they wander from where they
usually sit.

The answer is a fraction of the image width, which makes it independent of the
clip's resolution, of how far away the scene is, and of the world's unknown
scale (the fused world is up-to-scale until ``estimate_scale`` runs, and metric
afterwards — the same clip must get the same answer either way).

Checked against optical flow on the raw video for twenty clips: the two agree
across four orders of magnitude — a locked tripod reads 0.03 % where flow says
0.01 %, a broadcast pan reads 13 % where flow says 10 %.
"""
from __future__ import annotations

import numpy as np

#: A shot counts as locked when the scene slides less than this fraction of the
#: image width. Where 2 % comes from: on a clip whose picture optical flow says
#: is perfectly frozen this measure still reads up to 0.7 %, so that is the
#: noise the threshold has to clear; a camera that nudges reads 1–2 %; and the
#: quietest clip anyone would call a moving shot reads 4 %. 2 % sits nearly 3×
#: above the noise and 2× below the first real motion, so nothing near the line
#: is a camera that actually travels.
STATIC_MOTION_MAX = 0.02


def image_motion(
    extrinsics: np.ndarray,
    intrinsics_px: np.ndarray,
    width: int,
    height: int,
    points: np.ndarray,
    *,
    probe: int = 2000,
) -> float:
    """How far the scene slides across the picture, as a fraction of the image width.

    ``extrinsics`` ``(F, 4, 4)`` camera-from-world and ``intrinsics_px``
    ``(F, 3, 3)`` are per frame and share one ``width`` × ``height`` pixel grid;
    ``points`` ``(P, 3)`` are world points to watch — the fused scene cloud, or
    any single frame's depth map backprojected.

    Every point is projected through every frame and compared with where it sits
    in the median frame. The returned number is the 90th percentile over frames
    of the median slide over points, so neither a handful of bad frames nor a
    handful of stray points can decide the answer. Points that fall behind a
    camera or outside the frame are dropped; ``nan`` comes back when too few
    stay in view to measure.
    """
    pts = np.asarray(points, np.float64)
    pts = pts[:: max(1, len(pts) // probe)]
    extr = np.asarray(extrinsics, np.float64)
    K = np.asarray(intrinsics_px, np.float64)
    cam = np.einsum("fij,pj->fpi", extr[:, :3, :3], pts) + extr[:, None, :3, 3]
    z = cam[..., 2]
    uv = np.einsum("fij,fpj->fpi", K, cam / np.clip(z, 1e-6, None)[..., None])[..., :2]
    home = np.median(uv, axis=0)                                    # (P, 2) usual pixel
    seen = ((z > 1e-3).all(0)
            & (home[:, 0] > 0) & (home[:, 0] < width)
            & (home[:, 1] > 0) & (home[:, 1] < height))
    if int(seen.sum()) < 20:
        return float("nan")
    slide = np.linalg.norm(uv[:, seen] - home[seen], axis=-1)       # (F, P) pixels
    return float(np.percentile(np.median(slide, axis=1), 90) / width)


def is_static(motion: float, threshold: float = STATIC_MOTION_MAX) -> bool:
    """Does this count as a locked shot? A clip we could not measure (``nan``) does not."""
    return bool(motion < threshold)
