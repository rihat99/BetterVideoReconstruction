"""Rebuild the climb_wall_3 wall cloud in the CHESSBOARD-CALIBRATED camera frame.

``scene_clean.npz`` is VGGT geometry in VGGT's *own* camera poses, pulled onto the
calibration by a similarity fit of the three camera centres. That fit leaves the
cameras 25-190 mm out of place, and the human body is fitted through the
calibrated cameras, so body and wall disagree by up to 10-20 cm. This script
throws VGGT's poses and intrinsics away and keeps only its depth maps:

1. the three empty-wall frames are **undistorted** with the calibration, exactly
   as ``prepare_human.py`` undistorts the clip videos, and VGGT runs on those;
2. every depth map is backprojected with the **calibrated** intrinsics and placed
   with the **calibrated** camera-from-world pose, so the cameras are right by
   construction;
3. VGGT depth is up to one scale per view, and that scale is measured against the
   calibration: SIFT points matched between ``cam_left`` and ``cam_right``,
   filtered by the calibrated epipolar geometry and triangulated, give a metric
   depth at a few hundred pixels of each front view; the median of
   (triangulated depth / VGGT depth) is the view's scale. ``cam_3`` sees almost
   nothing the front pair sees at a usable angle and matches nowhere, so its
   scale comes from the front pair's finished metric cloud instead: render that
   cloud into cam_3 and take the same median ratio. (Used on cam_right, where
   the SIFT route also works, the two routes agreed to 0.03-0.5 %.)
4. the same cleaning as ``clean_scene.py`` runs on the result.

Sets 16b / 16c have no calibrated cam_3 (it was bumped between recordings), so
they get the front pair only.

Writes per set ``scene_calib.npz`` (the keys ``scene_clean.npz`` has, plus the
per-view scales), ``scene_calib.ply``, and ``check_calib/`` — ``<view>.png`` and
``<view>_old.png``, where the *other* views' points are drawn over this view's
undistorted frame (holds have to land on holds), plus ``report.json``.

    uv run --no-sync python scripts/peter_gt/rebuild_scene_calibrated.py
    uv run --no-sync python scripts/peter_gt/rebuild_scene_calibrated.py --sets 00 32
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts/peter_gt"))

from clean_scene import (  # noqa: E402
    CONF_QUANTILE,
    EDGE_RTOL,
    SIDE_VIEW,
    inside_front_view,
    isolated,
    pixel_masks,
)
from reconstruct_vggt import (  # noqa: E402
    CALIB_REF,
    CONFIG,
    GT_ROOT,
    SETS,
    STAGE,
    VIEW_NAMES,
    centres,
    content_window,
    write_ply,
)
from bvr.geometry.backproject import backproject  # noqa: E402
from bvr.geometry.clean import clean_depth  # noqa: E402
from bvr.paths import PETER  # noqa: E402

HUMAN_ROOT = PETER / "climb_wall_3_gt/human"
#: One prepared clip per inclination — its gravity says which way is down.
GRAVITY_CLIP = {"00": "b_00_02_part2", "16": "b_16_01_part3", "32": "b_32_02_part2"}

SIFT_CONTRAST = 0.005    # low, so the wall's screw holes and hold edges all count
RATIO = 0.8              # Lowe ratio test
EPIPOLAR_PX = 1.0        # Sampson distance to the calibrated epipolar line
MAD_TRIM = 3.0           # robust cut on the depth-ratio samples
MIN_ANCHORS = 20         # fewer than this and the view's scale is not trustworthy
MIN_HITS = 3             # front-cloud points needed in a cam_3 pixel to use it
PLANE_TOL = 0.03         # RANSAC inlier band for the wall plane, metres
PLANE_TRIALS = 2000


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

def load_calibration(name: str) -> dict:
    """Calibrated cameras of a set; 16b / 16c keep only the front pair."""
    z = np.load(GT_ROOT / CALIB_REF.get(name, name) / "calibration.npz", allow_pickle=True)
    n = 2 if name in CALIB_REF else 3
    return {"K": np.asarray(z["K"], np.float64)[:n],
            "dist": np.asarray(z["dist"], np.float64)[:n],
            "extrinsics": np.asarray(z["extrinsics"], np.float64)[:n],
            "image_size": np.asarray(z["image_size"], np.int32)[:n],
            "n_views": n}


def undistort(img: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Remove lens distortion, keeping the same K — what prepare_human does to the clips."""
    h, w = img.shape[:2]
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_16SC2)
    return cv2.remap(img, map1, map2, cv2.INTER_LINEAR)


def run_vggt(names: list[str], work: Path) -> None:
    """Undistort every set's frames into ``work`` and run one VGGT stage call over them."""
    entries = []
    for name in names:
        calib = load_calibration(name)
        out = work / "frames" / name
        out.mkdir(parents=True, exist_ok=True)
        for k in range(calib["n_views"]):
            img = cv2.imread(str(GT_ROOT / name / "frames" / f"{VIEW_NAMES[k]}.png"))
            cv2.imwrite(str(out / f"{VIEW_NAMES[k]}.png"), undistort(img, calib["K"][k], calib["dist"][k]))
        views = [str(out / f"{VIEW_NAMES[k]}.png") for k in range(calib["n_views"])]
        entries.append({"source": views[0], "sources": views, "out_dir": str(work / "vggt" / name)})
    manifest = work / "manifest.json"
    manifest.write_text(json.dumps(entries, indent=2))
    subprocess.run([sys.executable, str(STAGE), "--source", str(manifest),
                    "--save-dir", str(work / "vggt"), "--config", str(CONFIG)], cwd=_REPO, check=True)


def grid_K(K: np.ndarray, shape: tuple[int, int], size_wh: tuple[int, int]) -> np.ndarray:
    """The calibrated K rewritten in the pixels of VGGT's (smaller) depth grid."""
    th, tw = shape
    w0, h0 = size_wh
    sx, sy = tw / w0, th / h0
    return np.array([[K[0, 0] * sx, 0.0, (K[0, 2] + 0.5) * sx - 0.5],
                     [0.0, K[1, 1] * sy, (K[1, 2] + 0.5) * sy - 0.5],
                     [0.0, 0.0, 1.0]])


def load_views(name: str, work: Path, calib: dict) -> list[dict]:
    """Per view: the undistorted frame, VGGT's depth + confidence on their own grid."""
    cam = np.load(work / "vggt" / name / "vggt" / "camera.npz", allow_pickle=True)
    vh, vw = int(cam["vggt_height"]), int(cam["vggt_width"])
    resolution, mode = int(cam["image_resolution"]), str(cam["mode"])
    views = []
    for k in range(calib["n_views"]):
        image = cv2.imread(str(work / "frames" / name / f"{VIEW_NAMES[k]}.png"))
        h0, w0 = image.shape[:2]
        top, left, th, tw = content_window(w0, h0, vh, vw, resolution, mode)
        z = np.load(work / "vggt" / name / "vggt" / "depth" / VIEW_NAMES[k] / "frame_000000.npz")
        depth = np.asarray(z["depth"], np.float64)[top:top + th, left:left + tw]
        conf = np.asarray(z["depth_conf"], np.float32)[top:top + th, left:left + tw]
        views.append({"name": VIEW_NAMES[k], "image": image, "depth": depth, "conf": conf,
                      "colors": cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA),
                      "K": calib["K"][k], "extr": calib["extrinsics"][k], "size": (w0, h0),
                      "K_grid": grid_K(calib["K"][k], (th, tw), (w0, h0))})
    return views


# --------------------------------------------------------------------------- #
# Depth scale: triangulated SIFT points for the front pair, the front cloud for cam_3
# --------------------------------------------------------------------------- #

def sample_depth(view: dict, px: np.ndarray) -> np.ndarray:
    """VGGT depth at full-resolution pixel coordinates (0 where there is none)."""
    th, tw = view["depth"].shape
    w0, h0 = view["size"]
    u = ((px[:, 0] + 0.5) * tw / w0 - 0.5).astype(np.float32).reshape(-1, 1)
    v = ((px[:, 1] + 0.5) * th / h0 - 0.5).astype(np.float32).reshape(-1, 1)
    return cv2.remap(view["depth"].astype(np.float32), u, v, cv2.INTER_LINEAR, borderValue=0).ravel()


def triangulated_points(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SIFT points of two views, kept only if they satisfy the calibrated epipolar
    geometry, triangulated with the calibrated cameras: the pixels in each view and
    the metric world point. These are the only metric measurements in this script.
    """
    sift = cv2.SIFT_create(contrastThreshold=SIFT_CONTRAST)
    kp_a, de_a = sift.detectAndCompute(cv2.cvtColor(a["image"], cv2.COLOR_BGR2GRAY), None)
    kp_b, de_b = sift.detectAndCompute(cv2.cvtColor(b["image"], cv2.COLOR_BGR2GRAY), None)
    matches = cv2.BFMatcher().knnMatch(de_a, de_b, k=2)
    good = [m for m, second in matches if m.distance < RATIO * second.distance]
    px_a = np.array([kp_a[m.queryIdx].pt for m in good], np.float64)
    px_b = np.array([kp_b[m.trainIdx].pt for m in good], np.float64)

    R = b["extr"][:3, :3] @ a["extr"][:3, :3].T
    t = b["extr"][:3, 3] - R @ a["extr"][:3, 3]
    skew = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    F = np.linalg.inv(b["K"]).T @ skew @ R @ np.linalg.inv(a["K"])
    ha, hb = np.c_[px_a, np.ones(len(px_a))], np.c_[px_b, np.ones(len(px_b))]
    line_b, line_a = (F @ ha.T).T, (F.T @ hb.T).T
    sampson = np.abs(np.sum(hb * line_b, 1)) / np.sqrt(
        line_b[:, 0] ** 2 + line_b[:, 1] ** 2 + line_a[:, 0] ** 2 + line_a[:, 1] ** 2)
    px_a, px_b = px_a[sampson < EPIPOLAR_PX], px_b[sampson < EPIPOLAR_PX]

    P_a = np.ascontiguousarray(a["K"] @ a["extr"][:3])
    P_b = np.ascontiguousarray(b["K"] @ b["extr"][:3])
    hom = cv2.triangulatePoints(P_a, P_b, np.ascontiguousarray(px_a.T), np.ascontiguousarray(px_b.T))
    world = (hom[:3] / hom[3]).T
    z_a = (world @ a["extr"][:3, :3].T + a["extr"][:3, 3])[:, 2]
    z_b = (world @ b["extr"][:3, :3].T + b["extr"][:3, 3])[:, 2]
    keep = (z_a > 0.5) & (z_b > 0.5) & (z_a < 25.0) & (z_b < 25.0)
    return px_a[keep], px_b[keep], world[keep]


def robust_scale(z_true: np.ndarray, z_vggt: np.ndarray) -> dict:
    """Median of (metric depth / VGGT depth) after a MAD cut, with its spread."""
    ok = z_vggt > 0
    ratio = z_true[ok] / z_vggt[ok]
    med = np.median(ratio)
    mad = np.median(np.abs(ratio - med))
    inlier = np.abs(ratio - med) < MAD_TRIM * mad + 1e-9
    scale = float(np.median(ratio[inlier]))
    residual = (z_true[ok][inlier] - scale * z_vggt[ok][inlier])
    return {"scale": scale, "n": int(ok.sum()), "n_inlier": int(inlier.sum()),
            "rel_mad_pct": float(mad / med * 100),
            "residual_rms_mm": float(np.sqrt((residual ** 2).mean()) * 1e3)}


def scale_from_cloud(points: np.ndarray, view: dict) -> dict:
    """cam_3's scale: compare its depth with the nearest front-pair surface per pixel."""
    th, tw = view["depth"].shape
    cam = points @ view["extr"][:3, :3].T + view["extr"][:3, 3]
    cam = cam[cam[:, 2] > 0.3]
    uv = cam @ view["K_grid"].T
    uv = uv[:, :2] / uv[:, 2:]
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    inside = (u >= 0) & (u < tw) & (v >= 0) & (v < th)
    flat = v[inside] * tw + u[inside]
    nearest = np.full(th * tw, np.inf)
    np.minimum.at(nearest, flat, cam[inside][:, 2])
    hits = np.bincount(flat, minlength=th * tw)
    good = (np.isfinite(nearest) & (hits >= MIN_HITS)).reshape(th, tw) & depth_mask(view)
    out = robust_scale(nearest.reshape(th, tw)[good], view["depth"][good])
    out["source"] = "front-pair cloud rendered into the view"
    return out


def depth_mask(view: dict) -> np.ndarray:
    """Pixels worth keeping: valid depth, off a depth edge, confidence above the quantile."""
    keep = clean_depth(view["depth"], edge_rtol=EDGE_RTOL)
    valid = np.isfinite(view["depth"]) & (view["depth"] > 0)
    return keep & (view["conf"] >= np.quantile(view["conf"][valid], CONF_QUANTILE))


def view_scales(views: list[dict]) -> tuple[np.ndarray, dict, np.ndarray]:
    """One depth scale per view, the numbers behind each, and the metric anchor points."""
    px_l, px_r, anchors = triangulated_points(views[0], views[1])
    report = {}
    scales = []
    for view, px in ((views[0], px_l), (views[1], px_r)):
        z_true = (anchors @ view["extr"][:3, :3].T + view["extr"][:3, 3])[:, 2]
        fit = robust_scale(z_true, sample_depth(view, px))
        fit["source"] = "SIFT points triangulated through the calibrated front pair"
        assert fit["n_inlier"] >= MIN_ANCHORS, f"{view['name']}: only {fit['n_inlier']} anchors"
        report[view["name"]] = fit
        scales.append(fit["scale"])
    if len(views) > SIDE_VIEW:
        front = np.concatenate([cloud_of(v, s) for v, s in zip(views[:2], scales)])
        fit = scale_from_cloud(front, views[SIDE_VIEW])
        report[views[SIDE_VIEW]["name"]] = fit
        scales.append(fit["scale"])
    return np.array(scales), report, anchors


# --------------------------------------------------------------------------- #
# The cloud
# --------------------------------------------------------------------------- #

def cloud_of(view: dict, scale: float, mask: np.ndarray | None = None) -> np.ndarray:
    """World points of one view: VGGT depth x scale, calibrated rays, calibrated pose."""
    mask = depth_mask(view) if mask is None else mask
    return backproject(torch.from_numpy(view["depth"] * scale), torch.from_numpy(view["K_grid"]),
                       torch.from_numpy(view["extr"]), mask=torch.from_numpy(mask)).numpy()


def build_cloud(views: list[dict], scales: np.ndarray) -> dict:
    """Every view's points in one array, before the whole-cloud cleaning passes."""
    pts, cols, vid, conf = [], [], [], []
    for k, view in enumerate(views):
        mask = depth_mask(view)
        pts.append(cloud_of(view, scales[k], mask))
        cols.append(view["colors"][mask][:, ::-1])          # BGR -> RGB
        vid.append(np.full(int(mask.sum()), k, np.int8))
        conf.append(view["conf"][mask])
    return {"points": np.concatenate(pts), "colors": np.concatenate(cols).astype(np.uint8),
            "view_id": np.concatenate(vid), "conf": np.concatenate(conf).astype(np.float16)}


def clean(cloud: dict, calib: dict) -> np.ndarray:
    """The last two ``clean_scene`` passes: cam_3 where the front pair already looks, and strays."""
    keep = np.ones(len(cloud["points"]), bool)
    if calib["n_views"] > SIDE_VIEW:
        keep &= ~((cloud["view_id"] == SIDE_VIEW) & inside_front_view(calib, cloud["points"]))
    far = np.zeros(len(keep), bool)
    far[keep] = isolated(cloud["points"][keep])
    return keep & ~far


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

def render_overlay(points: np.ndarray, image: np.ndarray, K: np.ndarray, extr: np.ndarray,
                   path: Path) -> None:
    """Half-transparent depth-coloured splat of the cloud over the undistorted frame.

    The caller leaves out the points this very camera contributed — those land on
    themselves whatever the pose is, so only the *other* views test the placement.
    """
    h, w = image.shape[:2]
    cam = points @ extr[:3, :3].T + extr[:3, 3]
    cam = cam[cam[:, 2] > 0.05]
    uv = cam @ K.T
    uv = uv[:, :2] / uv[:, 2:]
    zbuf = np.full(h * w, np.inf)
    for du in (0, 1):
        for dv in (0, 1):
            u = np.round(uv[:, 0]).astype(int) + du
            v = np.round(uv[:, 1]).astype(int) + dv
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            np.minimum.at(zbuf, v[inside] * w + u[inside], cam[inside][:, 2])
    drawn = np.isfinite(zbuf)
    lo, hi = np.percentile(zbuf[drawn], [2, 98])
    shade = np.clip((zbuf - lo) / max(hi - lo, 1e-6), 0, 1)
    colour = cv2.applyColorMap((255 * (1 - shade)).astype(np.uint8).reshape(h, w), cv2.COLORMAP_TURBO)
    out = image.copy()
    drawn = drawn.reshape(h, w)
    out[drawn] = (0.5 * image[drawn] + 0.5 * colour[drawn]).astype(np.uint8)
    cv2.imwrite(str(path), out)


def cross_view(points: np.ndarray, view_id: np.ndarray, calib: dict) -> dict:
    """Nearest-neighbour distance from each view's points to those of a view that also sees them."""
    n = calib["n_views"]
    trees = {j: cKDTree(points[view_id == j]) for j in range(n)}
    out = {}
    for i in range(n):
        src = points[view_id == i]
        for j in range(n):
            if i == j:
                continue
            extr, K = calib["extrinsics"][j], calib["K"][j]
            w, h = (int(x) for x in calib["image_size"][j])
            cam = src @ extr[:3, :3].T + extr[:3, 3]
            front = cam[:, 2] > 1e-3
            uv = cam[front] @ K.T
            uv = uv[:, :2] / uv[:, 2:]
            seen = src[front][(uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)]
            if len(seen) < 100:
                continue
            d, _ = trees[j].query(seen, k=1, workers=-1)
            out[f"{VIEW_NAMES[i]}->{VIEW_NAMES[j]}"] = {
                "n": int(len(d)), "median_mm": float(np.median(d) * 1e3),
                "p90_mm": float(np.percentile(d, 90) * 1e3)}
    return out


def wall_plane(points: np.ndarray, calib: dict, gravity: np.ndarray) -> dict:
    """The nearest big plane facing cam_left — the climbing board — by RANSAC."""
    rng = np.random.default_rng(0)
    sample = points[rng.choice(len(points), min(len(points), 40000), replace=False)]
    test = sample[rng.choice(len(sample), min(len(sample), 6000), replace=False)]
    tri = sample[rng.integers(0, len(sample), (PLANE_TRIALS, 3))]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    offset = np.sum(normal * tri[:, 0], 1)
    eye = centres(calib["extrinsics"])[0]
    axis = calib["extrinsics"][0][:3, :3][2]                     # cam_left's viewing direction
    flip = np.sign(np.sum(normal * (eye - tri[:, 0]), 1))        # point the normal at cam_left
    normal, offset = normal * flip[:, None], offset * flip
    reach = normal @ eye - offset                                # how far cam_left is from the plane
    usable = (np.abs(normal @ axis) > 0.6) & (reach > 1.5) & (reach < 6.0)
    votes = np.where(usable, (np.abs(test @ normal.T - offset) < PLANE_TOL).sum(0), -1)
    best = int(np.argmax(votes))
    inlier = np.abs(points @ normal[best] - offset[best]) < PLANE_TOL
    mean = points[inlier].mean(0)
    centred = points[inlier] - mean
    axes = np.linalg.eigh(centred.T @ centred)[1][:, 0]          # smallest-spread direction
    n = axes * np.sign(np.dot(axes, normal[best]))
    return {"n_inliers": int(inlier.sum()), "inlier_fraction": float(inlier.mean()),
            "normal": n.tolist(),
            "overhang_deg": float(90.0 - np.degrees(np.arccos(np.clip(abs(n @ gravity), -1, 1)))),
            "camera_distance_m": {VIEW_NAMES[k]: float(n @ (c - mean))
                                  for k, c in enumerate(centres(calib["extrinsics"]))}}


def old_before_cleaning(name: str) -> tuple[np.ndarray, np.ndarray]:
    """The old cloud with only its per-pixel masks applied, so it compares like for like."""
    scene = np.load(GT_ROOT / name / "scene.npz")
    keep = pixel_masks(GT_ROOT / name, scene)
    return np.asarray(scene["points"], np.float64)[keep], np.asarray(scene["view_id"])[keep]


def displacement(new: np.ndarray, old: np.ndarray) -> dict:
    """How far the cloud moved: nearest-neighbour distance, both directions."""
    d_new, _ = cKDTree(old).query(new, k=1, workers=-1)
    d_old, _ = cKDTree(new).query(old, k=1, workers=-1)
    return {"new_to_old_median_mm": float(np.median(d_new) * 1e3),
            "old_to_new_median_mm": float(np.median(d_old) * 1e3),
            "new_to_old_p90_mm": float(np.percentile(d_new, 90) * 1e3)}


def anchor_distance(points: np.ndarray, anchors: np.ndarray) -> dict:
    """Distance from each metric anchor point to the nearest point of a cloud."""
    d, _ = cKDTree(points).query(anchors, k=1, workers=-1)
    return {"n": int(len(d)), "median_mm": float(np.median(d) * 1e3),
            "p90_mm": float(np.percentile(d, 90) * 1e3)}


def checks(name: str, views: list[dict], calib: dict, raw: dict, keep: np.ndarray,
           anchors: np.ndarray) -> dict:
    """Overlays + the numbers, new cloud against the old one, both through the calibrated cameras."""
    out_dir = GT_ROOT / name / "check_calib"
    out_dir.mkdir(exist_ok=True)
    new_pts, new_vid = raw["points"][keep], raw["view_id"][keep]
    old = np.load(GT_ROOT / name / "scene_clean.npz")
    old_pts, old_vid = np.asarray(old["points"], np.float64), np.asarray(old["view_id"])
    for k, view in enumerate(views):
        render_overlay(new_pts[new_vid != k], view["image"], view["K"], view["extr"],
                       out_dir / f"{view['name']}.png")
        render_overlay(old_pts[old_vid != k], view["image"], view["K"], view["extr"],
                       out_dir / f"{view['name']}_old.png")
    gravity = np.load(HUMAN_ROOT / GRAVITY_CLIP[name[:2]] / "geocalib" / "gravity.npz")["gravity_world"]
    old_raw_pts, old_raw_vid = old_before_cleaning(name)
    return {
        "anchor_distance_new": anchor_distance(new_pts, anchors),
        "anchor_distance_old": anchor_distance(old_pts, anchors),
        "cross_view_new": cross_view(raw["points"], raw["view_id"], calib),
        "cross_view_old": cross_view(old_raw_pts, old_raw_vid, calib),
        "wall_plane_new": wall_plane(new_pts[new_vid < SIDE_VIEW], calib, gravity),
        "wall_plane_old": wall_plane(old_pts[old_vid < SIDE_VIEW], calib, gravity),
        "gravity_world": np.asarray(gravity, np.float64).tolist(),
        "displacement_old_to_new": displacement(new_pts, old_pts),
    }


# --------------------------------------------------------------------------- #

def rebuild(name: str, work: Path) -> None:
    calib = load_calibration(name)
    views = load_views(name, work, calib)
    scales, scale_report, anchors = view_scales(views)
    raw = build_cloud(views, scales)
    keep = clean(raw, calib)
    points = raw["points"][keep].astype(np.float32)

    set_dir = GT_ROOT / name
    np.savez(set_dir / "scene_calib.npz", points=points, colors=raw["colors"][keep],
             view_id=raw["view_id"][keep], conf=raw["conf"][keep],
             extrinsics=calib["extrinsics"].astype(np.float32), K=calib["K"].astype(np.float32),
             image_size=calib["image_size"], scale=np.float64(np.median(scales)),
             view_scale=scales.astype(np.float64),
             scale_source=np.str_("calibrated-anchors"), metric=np.bool_(True),
             view_names=np.array(VIEW_NAMES[:calib["n_views"]]), cam3_input=np.str_("upright"))
    write_ply(set_dir / "scene_calib.ply", points, raw["colors"][keep])

    report = {"set": name, "views": VIEW_NAMES[:calib["n_views"]],
              "depth_scale": scale_report,
              "points": {"before_cleaning": int(len(keep)), "kept": int(keep.sum())},
              "points_per_view": {VIEW_NAMES[k]: int((raw["view_id"][keep] == k).sum())
                                  for k in range(calib["n_views"])},
              **checks(name, views, calib, raw, keep, anchors)}
    (set_dir / "check_calib" / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\n=== {name} -> {set_dir/'scene_calib.npz'}")
    print(json.dumps(report, indent=2, default=float))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sets", nargs="+", default=SETS, choices=SETS, help="camera sets to rebuild")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="rebuild_scene_") as tmp:
        work = Path(tmp)
        run_vggt(args.sets, work)
        for name in args.sets:
            rebuild(name, work)


if __name__ == "__main__":
    main()
