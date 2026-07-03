"""Small pure geometry + colour helpers shared across the viewer package.

Numpy-only (``viser.transforms`` is imported lazily inside the one helper that
needs it), so this is the viewer's dependency-free base layer, imported by both
:mod:`tools.viewer.loading` and :mod:`tools.viewer.scene`.
"""
from __future__ import annotations

import numpy as np

# Per-person body colours (RGB 0-255), matching the stage overlay palette so a
# person keeps one colour across the body/SMPL-X overlays and this viewer.
_BODY_COLORS = (
    (166, 189, 219),  # light blue
    (242, 153, 102),  # salmon
    (153, 217, 140),  # light green
    (204, 166, 224),  # lilac
    (242, 217, 115),  # gold
)

# Viridis control points (RGB) for colouring the camera trajectory by time.
_VIRIDIS = np.array([(68, 1, 84), (59, 82, 139), (33, 145, 140),
                     (94, 201, 98), (253, 231, 37)], np.float64)


def _camera_world_pose(extrinsic: np.ndarray):
    """cam-from-world OpenCV ``[R|t]`` → ``(world position, world-from-cam wxyz)``."""
    import viser.transforms as vt

    R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    return -R.T @ t, vt.SO3.from_matrix(R.T).wxyz


def _unit(v) -> np.ndarray:
    """Normalize a vector; return it unchanged if its norm is ~0."""
    v = np.asarray(v, np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _time_color(t: float) -> tuple[int, int, int]:
    """Sample the viridis ramp at ``t`` ∈ [0, 1] → an ``(r, g, b)`` uint tuple."""
    x = float(np.clip(t, 0.0, 1.0)) * (len(_VIRIDIS) - 1)
    i = min(int(x), len(_VIRIDIS) - 2)
    c = _VIRIDIS[i] * (1.0 - (x - i)) + _VIRIDIS[i + 1] * (x - i)
    return int(c[0]), int(c[1]), int(c[2])


def _wxyz_shortest_arc_z(d: np.ndarray) -> tuple[float, float, float, float]:
    """Shortest-arc quaternion (wxyz) rotating the unit cylinder's +z axis onto ``d``."""
    d = np.asarray(d, np.float64)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    d = d / n
    c = float(d[2])  # cos angle = z·d
    if c > 0.999999:
        return (1.0, 0.0, 0.0, 0.0)
    if c < -0.999999:
        return (0.0, 1.0, 0.0, 0.0)  # antiparallel → 180° about x
    v = np.array([-d[1], d[0], 0.0])  # z × d
    s = np.sqrt((1.0 + c) * 2.0)
    q = np.array([s * 0.5, v[0] / s, v[1] / s, v[2] / s])
    q /= np.linalg.norm(q)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _top4_skin_weights(lbs_w: np.ndarray) -> np.ndarray:
    """Top-4 per-vertex skin weights renormalized to sum 1 (``V, J``).

    viser keeps only each vertex's top-4 bone weights and does **not**
    renormalize; SMPL-X spreads weight over ~8 bones, so the dropped tail would
    leave blend-region vertices summing to <1 and — because we bake metre-scale
    world translation into the bones — fling them out as spikes. Pre-restricting
    to the top 4 and renormalizing matches what viser uses.
    """
    rows = np.arange(lbs_w.shape[0])[:, None]
    keep = np.argsort(lbs_w, axis=1)[:, -4:]
    w = np.zeros_like(lbs_w)
    w[rows, keep] = lbs_w[rows, keep]
    return (w / w.sum(1, keepdims=True)).astype(np.float32)


def _background_image(top=(30, 32, 36), bottom=(52, 55, 61), height: int = 256) -> np.ndarray:
    """A dark, faintly-graded grey backdrop (darker top → lighter bottom).

    Rendered full-canvas via :meth:`viser.SceneApi.set_background_image`; the gentle
    vertical gradient reads as a studio backdrop rather than a flat fill.
    """
    t = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    col = (1.0 - t) * np.array(top, np.float32) + t * np.array(bottom, np.float32)
    return np.repeat(col[:, None, :], 2, axis=1).astype(np.uint8)   # (H, 2, 3)
