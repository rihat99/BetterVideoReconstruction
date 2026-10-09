"""Clean the raw climb_wall_3 GT cloud: drop low-confidence points and flying pixels.

Four passes on ``scene.npz`` (per set, written by ``reconstruct_vggt.py``):

1. confidence — per view, drop the least confident ``CONF_QUANTILE`` of the
   VGGT depth pixels;
2. depth edges — drop pixels on a depth discontinuity (the "flying" points that
   bridge a foreground object and the wall behind it), with the same utils3d
   edge test the fuse stage uses (``bvr.geometry.clean.clean_depth``);
3. side camera inside the front pair's view — ``cam_3`` looks along the wall at
   a grazing angle, and its depth there is smeared over a metre; wherever a
   point of it projects into ``cam_left`` or ``cam_right`` those cameras see the
   same surface far better, so the side camera only keeps what they cannot see;
4. isolated points — drop points whose mean distance to their ``KNN`` nearest
   neighbours exceeds the median by more than ``STD_RATIO`` standard deviations.

Writes ``scene_clean.npz`` (same keys as ``scene.npz``, subset of rows) and
``scene_clean.ply``; the raw cloud is left as is.

    uv run --no-sync python scripts/peter_gt/clean_scene.py [SET ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts/peter_gt"))
from reconstruct_vggt import GT_ROOT, SETS, VIEW_NAMES, content_window, write_ply  # noqa: E402
from bvr.geometry.clean import clean_depth  # noqa: E402

CONF_QUANTILE = 0.4    # per view, the least confident 40 % goes
EDGE_RTOL = 0.04       # depth-edge relative tolerance (fuse stage default)
KNN = 20               # neighbours for the isolated-point test
STD_RATIO = 1.0        # mean neighbour distance > median + STD_RATIO * std -> isolated
SIDE_VIEW = 2          # index of cam_3 in view_names


def pixel_masks(set_dir: Path, scene: np.lib.npyio.NpzFile) -> np.ndarray:
    """Per-point keep flag from the confidence and depth-edge tests, in scene row order."""
    cam = np.load(set_dir / "vggt" / "camera.npz")
    vh, vw = int(cam["vggt_height"]), int(cam["vggt_width"])
    resolution, mode = int(cam["image_resolution"]), str(cam["mode"])
    keep_rows = []
    for k, name in enumerate(VIEW_NAMES):
        w0, h0 = (int(x) for x in scene["image_size"][k])
        top, left, th, tw = content_window(w0, h0, vh, vw, resolution, mode)
        z = np.load(set_dir / "vggt" / "depth" / name / "frame_000000.npz")
        depth = np.asarray(z["depth"], np.float64)[top:top + th, left:left + tw]
        conf = np.asarray(z["depth_conf"], np.float32)[top:top + th, left:left + tw]
        valid = np.isfinite(depth) & (depth > 0)
        n_view = int((scene["view_id"] == k).sum())
        assert valid.sum() == n_view, f"{name}: {valid.sum()} depth pixels vs {n_view} scene points"
        keep = clean_depth(depth, edge_rtol=EDGE_RTOL)
        keep &= conf >= np.quantile(conf[valid], CONF_QUANTILE)
        n_edge = int((valid & ~clean_depth(depth, edge_rtol=EDGE_RTOL)).sum())
        print(f"  {name:9s} {n_view:7d} px: {n_edge:6d} on depth edges, "
              f"{int((valid & ~keep).sum()) - n_edge:6d} more below the confidence quantile")
        keep_rows.append(keep[valid])
    return np.concatenate(keep_rows)


def inside_front_view(scene: np.lib.npyio.NpzFile, points: np.ndarray) -> np.ndarray:
    """True where a point projects inside cam_left's or cam_right's image with positive depth."""
    inside = np.zeros(len(points), bool)
    for k in range(SIDE_VIEW):
        e, K = scene["extrinsics"][k], scene["K"][k]
        w, h = (int(x) for x in scene["image_size"][k])
        cam = points @ e[:3, :3].T + e[:3, 3]
        uv = (cam @ K.T)[:, :2] / np.where(cam[:, 2:] > 0, cam[:, 2:], np.inf)
        inside |= (cam[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return inside


def isolated(points: np.ndarray) -> np.ndarray:
    """True for points whose mean KNN distance is an outlier (statistical outlier removal)."""
    dist, _ = cKDTree(points).query(points, k=KNN + 1, workers=-1)
    mean = dist[:, 1:].mean(1)
    return mean > np.median(mean) + STD_RATIO * mean.std()


def clean(name: str) -> None:
    set_dir = GT_ROOT / name
    scene = np.load(set_dir / "scene.npz")
    print(f"{name}:")
    keep = pixel_masks(set_dir, scene)
    pts = scene["points"].astype(np.float64)
    side = scene["view_id"] == SIDE_VIEW
    redundant = side & inside_front_view(scene, pts)
    print(f"  cam_3 inside the front pair's view: {int((redundant & keep).sum())} dropped")
    keep &= ~redundant
    far = np.zeros(len(pts), bool)
    far[keep] = isolated(pts[keep])
    keep &= ~far
    print(f"  isolated: {int(far.sum())}; kept {int(keep.sum())} of {len(pts)} ({100 * keep.mean():.1f} %)")
    out = {key: scene[key] for key in scene.files}
    for key in ("points", "colors", "view_id", "conf"):
        out[key] = scene[key][keep]
    np.savez(set_dir / "scene_clean.npz", **out)
    write_ply(set_dir / "scene_clean.ply", out["points"], out["colors"])


def main() -> None:
    for name in sys.argv[1:] or SETS:
        clean(name)


if __name__ == "__main__":
    main()
