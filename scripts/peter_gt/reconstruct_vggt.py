"""Metric point cloud + camera check for the climb_wall_3 ground-truth sets.

One VGGT-Omega forward per set over its three static views (``cam_left`` first, so
the reconstruction world *is* cam_left), then the three depth maps are
backprojected into one coloured cloud. VGGT geometry is up-to-scale, so the metric
factor comes from the chessboard calibration: a similarity fit (Umeyama, with
scale) of the three VGGT camera centres onto the calibrated ones. Sets without a
calibration of their own (16b, 16c — cam_3 was bumped, cam_left/cam_right did not
move) take the scale from the calibrated left-right baseline of set 16 instead.

Writes per set, next to ``frames/``: ``vggt/`` (stage output), ``scene.npz``,
``scene.ply`` and ``comparison.json`` (VGGT vs calibration, also printed).

``cam_3`` is stored upright (portrait 1080x1920) while the other two views are
landscape. VGGT's loader resizes every image to ~1024 patches and then *pads the
batch to a common canvas with white*, so a mixed-aspect forward runs at 688x688
with 44 % white border per view. ``--cam3 raw`` feeds cam_3 in its original
landscape orientation instead (no padding at all, 384x688 for all three) and
rotates the predicted camera back by the 90 deg roll about the optical axis; that
mapping of K and the extrinsics is exact. The two variants need their own VGGT
forward — never reuse one's ``vggt/`` for the other.

    uv run --no-sync python scripts/peter_gt/reconstruct_vggt.py
    uv run --no-sync python scripts/peter_gt/reconstruct_vggt.py --sets 00 --cam3 raw --subdir cam3_raw
    uv run --no-sync python scripts/peter_gt/reconstruct_vggt.py --no-vggt   # re-do only the cloud
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from bvr.geometry.backproject import backproject  # noqa: E402
from bvr.geometry.intrinsics import denormalize_K  # noqa: E402
from bvr.paths import PETER  # noqa: E402

GT_ROOT = PETER / "climb_wall_3_gt/scene"
CONFIG = _REPO / "scripts/peter_gt/vggt_gt.toml"
SETS = ["00", "16", "16b", "16c", "32"]
VIEW_NAMES = ["cam_left", "cam_right", "cam_3"]
CALIB_REF = {"16b": "16", "16c": "16"}      # set -> set whose calibration.npz applies
PATCH = 16                                   # VGGT patch size (load_and_preprocess_images)
RAY_GRID = 9                                 # grid side for the ray-alignment rotation fit
CENTRAL_FRAC = 0.6                           # image fraction that fit covers
# 90 deg roll about the optical axis: raw-landscape cam_3 -> the stored upright frame.
R_ROLL = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------- #
# VGGT stage
# --------------------------------------------------------------------------- #

def raw_cam3_frame(set_dir: Path) -> Path:
    """The landscape cam_3 frame, undoing the ROTATE_90_CLOCKWISE of the stored one."""
    out = set_dir / "frames_raw" / "cam_3.png"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(set_dir / "frames" / "cam_3.png"))
        cv2.imwrite(str(out), cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
    return out


def out_dir_of(name: str, subdir: str) -> Path:
    return GT_ROOT / name / subdir if subdir else GT_ROOT / name


def run_vggt(names: list[str], cam3: str, subdir: str) -> None:
    """One stage call over every set — one model load, one forward per set."""
    entries = []
    for name in names:
        set_dir = GT_ROOT / name
        cam3_path = raw_cam3_frame(set_dir) if cam3 == "raw" else set_dir / "frames" / "cam_3.png"
        entries.append({
            "source": str(set_dir / "frames" / "cam_left.png"),
            "sources": [str(set_dir / "frames" / f"{v}.png") for v in VIEW_NAMES[:2]] + [str(cam3_path)],
            "out_dir": str(out_dir_of(name, subdir)),
        })
    manifest = GT_ROOT / f"_vggt_manifest_{cam3}.json"
    manifest.write_text(json.dumps(entries, indent=2))
    cmd = [sys.executable, "-m", "bvr.stages.estimate_camera_vggt", "--source", str(manifest),
           "--save-dir", str(GT_ROOT), "--config", str(CONFIG)]
    print("[vggt] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=_REPO, check=True)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def content_window(w0: int, h0: int, vh: int, vw: int,
                   resolution: int, mode: str) -> tuple[int, int, int, int]:
    """Where the resized view sits inside VGGT's (padded) canvas: (top, left, h, w).

    ``resolution`` / ``mode`` come from the saved camera.npz, so this cannot drift
    away from the settings the forward actually used.
    """
    from vggt_omega.utils.load_fn import _balanced_target_shape, _max_size_target_shape

    shape_fn = {"balanced": _balanced_target_shape, "max_size": _max_size_target_shape}
    assert mode in shape_fn, f"unknown VGGT sizing mode {mode!r}"
    assert 0.5 <= h0 / w0 <= 2.0, "VGGT would centre-crop this aspect ratio"
    th, tw = shape_fn[mode](h0 / w0, resolution, PATCH)
    assert th <= vh and tw <= vw, f"content {tw}x{th} does not fit the {vw}x{vh} canvas"
    return (vh - th) // 2, (vw - tw) // 2, th, tw


def roll_cam3_to_upright(K: np.ndarray, extr: np.ndarray, raw_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Raw-landscape cam_3 camera -> the stored upright frame (image rotated 90 deg CW)."""
    K_u = np.eye(3)
    K_u[0, 0], K_u[1, 1] = K[1, 1], K[0, 0]
    K_u[0, 2], K_u[1, 2] = raw_h - 1 - K[1, 2], K[0, 2]
    extr_u = extr.copy()
    extr_u[:3, :4] = R_ROLL @ extr[:3, :4]
    return K_u, extr_u


def load_views(out_dir: Path, frame_paths: list[Path]) -> list[dict]:
    """Per view: VGGT-canvas depth/conf/K/extrinsics + the stored-frame K, extrinsics, size."""
    cam = np.load(out_dir / "vggt" / "camera.npz")
    vh, vw = int(cam["vggt_height"]), int(cam["vggt_width"])
    resolution, mode = int(cam["image_resolution"]), str(cam["mode"])
    extr = np.asarray(cam["extrinsics"], np.float64)
    # VGGT's first extrinsic is only *approximately* identity (~0.5 mm, 0.04 deg here).
    # Rebase so the world is exactly cam_left, which is what every consumer assumes.
    extr = extr @ np.linalg.inv(extr[0])
    K_canvas = np.asarray(denormalize_K(np.asarray(cam["intrinsics"]), vh, vw), np.float64)
    assert [str(v) for v in cam["view_names"]] == VIEW_NAMES, "unexpected view order"

    views = []
    for k, name in enumerate(VIEW_NAMES):
        img = cv2.imread(str(frame_paths[k]))
        h0, w0 = img.shape[:2]
        top, left, th, tw = content_window(w0, h0, vh, vw, resolution, mode)
        canvas = np.zeros((vh, vw, 3), np.uint8)
        canvas[top:top + th, left:left + tw] = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        z = np.load(out_dir / "vggt" / "depth" / name / "frame_000000.npz")
        depth = np.asarray(z["depth"], np.float64)
        conf = np.asarray(z["depth_conf"], np.float32)

        # The white padding is not image content — everything else is kept.
        keep = np.zeros((vh, vw), bool)
        keep[top:top + th, left:left + tw] = True
        keep &= np.isfinite(depth) & (depth > 0)

        # K in pixels of the STORED frame: undo the pad offset, then the resize.
        # The resize maps pixel *centres*, u_stored = (u_resized + 0.5) * sx - 0.5,
        # so the principal point picks up that half-pixel shift (1.4 px here).
        sx, sy = w0 / tw, h0 / th
        K_stored = np.array(
            [[K_canvas[k][0, 0] * sx, 0.0, (K_canvas[k][0, 2] - left + 0.5) * sx - 0.5],
             [0.0, K_canvas[k][1, 1] * sy, (K_canvas[k][1, 2] - top + 0.5) * sy - 0.5],
             [0.0, 0.0, 1.0]])
        views.append({"name": name, "depth": depth, "conf": conf, "keep": keep,
                      "K_canvas": K_canvas[k], "extr": extr[k], "colors": canvas,
                      "K_stored": K_stored, "extr_stored": extr[k], "size": (w0, h0)})
    return views


def build_cloud(views: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backproject every view into the cam_left frame -> points, colors, view_id, conf."""
    pts, cols, vid, conf = [], [], [], []
    for k, v in enumerate(views):
        p = backproject(torch.from_numpy(v["depth"]), torch.from_numpy(v["K_canvas"]),
                        torch.from_numpy(v["extr"]), mask=torch.from_numpy(v["keep"])).numpy()
        pts.append(p)
        cols.append(v["colors"][v["keep"]][:, ::-1])            # BGR -> RGB
        vid.append(np.full(len(p), k, np.int8))
        conf.append(v["conf"][v["keep"]])
    return (np.concatenate(pts), np.concatenate(cols).astype(np.uint8),
            np.concatenate(vid), np.concatenate(conf).astype(np.float16))


# --------------------------------------------------------------------------- #
# Scale + comparison against the chessboard calibration
# --------------------------------------------------------------------------- #

def centres(extr: np.ndarray) -> np.ndarray:
    """Camera centres in world coordinates from camera-from-world matrices."""
    return np.stack([-e[:3, :3].T @ e[:3, 3] for e in extr])


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity ``dst ~= s R src + t`` (Umeyama 1991, with scale)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(b.T @ a / len(src))
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    R = u @ d @ vt
    s = float((sig * np.diag(d)).sum() / (a * a).sum() * len(src))
    return s, R, dst.mean(0) - s * R @ mu_s


def rot_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(a @ b.T) - 1.0) / 2.0, -1.0, 1.0))))


def ray_align_rotation(K_v: np.ndarray, K_c: np.ndarray, dist_c: np.ndarray,
                       size: tuple[int, int]) -> np.ndarray:
    """Rotation carrying VGGT's pixel rays onto the calibrated ones.

    A wrong principal point tilts every ray, which then shows up as a "rotation
    error" even when the camera is pointing correctly. Fitting that tilt out on a
    grid over the central image separates the two: the leftover rotation is the
    part the intrinsics cannot explain.
    """
    w, h = size
    lo, hi = (1 - CENTRAL_FRAC) / 2, (1 + CENTRAL_FRAC) / 2
    uu, vv = np.meshgrid(np.linspace(lo * w, hi * w, RAY_GRID),
                         np.linspace(lo * h, hi * h, RAY_GRID))
    px = np.stack([uu.ravel(), vv.ravel()], 1)

    d_v = (np.linalg.inv(K_v) @ np.c_[px, np.ones(len(px))].T).T
    d_v /= np.linalg.norm(d_v, axis=1, keepdims=True)
    xy = cv2.undistortPoints(px.reshape(-1, 1, 2), K_c, dist_c).reshape(-1, 2)
    d_c = np.c_[xy, np.ones(len(xy))]
    d_c /= np.linalg.norm(d_c, axis=1, keepdims=True)

    u, _, vt = np.linalg.svd(d_c.T @ d_v)
    return u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt


def compare(views: list[dict], calib: dict, n_cams: int) -> dict:
    """VGGT cameras vs the calibrated ones. ``n_cams=2`` judges the stereo pair only."""
    v_extr = np.stack([v["extr_stored"] for v in views])[:n_cams]
    c_extr = np.asarray(calib["extrinsics"], np.float64)[:n_cams]
    K_c, dist_c = np.asarray(calib["K"], np.float64), np.asarray(calib["dist"], np.float64)
    v_c, c_c = centres(v_extr), centres(c_extr)
    v_base = float(np.linalg.norm(v_c[1] - v_c[0]))
    c_base = float(np.linalg.norm(c_c[1] - c_c[0]))

    # Rotation of every VGGT camera after its rays are matched to the calibrated ones.
    align = [ray_align_rotation(views[k]["K_stored"], K_c[k], dist_c[k], views[k]["size"])
             for k in range(n_cams)]
    v_rot = [v_extr[k][:3, :3] for k in range(n_cams)]
    a_rot = [align[k] @ v_rot[k] for k in range(n_cams)]

    def rel(rots: list[np.ndarray], j: int) -> np.ndarray:
        return rots[j] @ rots[0].T

    out: dict = {
        "baseline_scale_m_per_unit": c_base / v_base,
        "baseline_calibrated_m": c_base,
        "rot_error_deg": {
            "raw": {"left_right": rot_error_deg(rel(v_rot, 1), rel([e[:3, :3] for e in c_extr], 1))},
            "ray_aligned": {"left_right": rot_error_deg(rel(a_rot, 1),
                                                        rel([e[:3, :3] for e in c_extr], 1))},
            "ray_alignment_deg": {VIEW_NAMES[k]: rot_error_deg(align[k], np.eye(3))
                                  for k in range(n_cams)},
        },
        "intrinsics": {},
    }
    for k in range(n_cams):
        Kv, Kc = views[k]["K_stored"], K_c[k]
        w0, h0 = views[k]["size"]
        out["intrinsics"][VIEW_NAMES[k]] = {
            "stored_size_wh": [w0, h0],
            "fx_px": {"vggt": Kv[0, 0], "calib": Kc[0, 0], "ratio": Kv[0, 0] / Kc[0, 0]},
            "fy_px": {"vggt": Kv[1, 1], "calib": Kc[1, 1], "ratio": Kv[1, 1] / Kc[1, 1]},
            "cx_px": {"vggt": Kv[0, 2], "calib": Kc[0, 2], "offset": Kc[0, 2] - Kv[0, 2]},
            "cy_px": {"vggt": Kv[1, 2], "calib": Kc[1, 2], "offset": Kc[1, 2] - Kv[1, 2]},
        }
    if n_cams < 3:
        return out

    c_rot = [e[:3, :3] for e in c_extr]
    out["rot_error_deg"]["raw"]["left_cam3"] = rot_error_deg(rel(v_rot, 2), rel(c_rot, 2))
    out["rot_error_deg"]["ray_aligned"]["left_cam3"] = rot_error_deg(rel(a_rot, 2), rel(c_rot, 2))

    s, R, t = umeyama(v_c, c_c)
    resid = np.linalg.norm((s * (R @ v_c.T).T + t) - c_c, axis=1)
    out["umeyama_scale_m_per_unit"] = s
    out["umeyama_residual_mm"] = {n: float(r * 1e3) for n, r in zip(VIEW_NAMES, resid)}
    out["umeyama_rotation_deg"] = rot_error_deg(R, np.eye(3))
    out["umeyama_translation_mm"] = float(np.linalg.norm(t) * 1e3)
    out["baseline_vggt_scaled_m"] = v_base * s
    out["camera_distance_m"] = {
        "calibrated": {"left_right": c_base, "left_cam3": float(np.linalg.norm(c_c[2] - c_c[0]))},
        "vggt_scaled": {"left_right": v_base * s,
                        "left_cam3": float(np.linalg.norm(v_c[2] - v_c[0]) * s)},
    }
    return out


def direct_centre_error_mm(views: list[dict], calib: dict, scale: float, n_cams: int) -> dict:
    """Saved metric VGGT centres vs calibrated ones — no fit, just the applied scale.

    This is the error the delivered ``scene.npz`` actually carries: the scene is
    only scaled, never rotated or shifted onto the calibration.
    """
    v_c = centres(np.stack([v["extr_stored"] for v in views])[:n_cams]) * scale
    c_c = centres(np.asarray(calib["extrinsics"], np.float64)[:n_cams])
    return {VIEW_NAMES[k]: float(np.linalg.norm(v_c[k] - c_c[k]) * 1e3) for k in range(n_cams)}


def cross_view_agreement(views: list[dict], pts: np.ndarray, vid: np.ndarray) -> dict:
    """How far each view's cloud sits from the cloud of a view that also sees it.

    For every ordered pair, the source points kept are those that project inside the
    target image with positive depth (so non-overlapping field of view cannot
    pollute the number); the value is their nearest-neighbour distance to the target
    view's own points, in millimetres.
    """
    trees = {j: cKDTree(pts[vid == j]) for j in range(len(views))}
    out = {}
    for i in range(len(views)):
        src = pts[vid == i]
        for j in range(len(views)):
            if i == j:
                continue
            E, K = views[j]["extr_metric"], views[j]["K_stored"]
            w, h = views[j]["size"]
            X = (E[:3, :3] @ src.T).T + E[:3, 3]
            front = X[:, 2] > 1e-3
            uv = (K @ X[front].T).T
            uv = uv[:, :2] / uv[:, 2:]
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            seen = src[front][inside]
            if len(seen) < 100:
                continue
            d, _ = trees[j].query(seen, k=1)
            out[f"{VIEW_NAMES[i]}->{VIEW_NAMES[j]}"] = {
                "n": int(len(d)), "median_mm": float(np.median(d) * 1e3),
                "p90_mm": float(np.percentile(d, 90) * 1e3)}
    return out


def sibling_agreement(pts: np.ndarray, vid: np.ndarray, ref_scene: Path) -> dict:
    """cam_left+cam_right cloud vs the same two views of the reference set.

    Those two cameras never moved and their frames are the same image, so whatever
    distance is left is VGGT run-to-run variation plus the scale convention.
    """
    z = np.load(ref_scene)
    ref = z["points"][z["view_id"] < 2].astype(np.float64)
    d, _ = cKDTree(ref).query(pts[vid < 2], k=1)
    return {"n": int(len(d)), "median_mm": float(np.median(d) * 1e3),
            "p90_mm": float(np.percentile(d, 90) * 1e3)}


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def write_ply(path: Path, pts: np.ndarray, cols: np.ndarray) -> None:
    """Binary little-endian PLY with xyz float32 + rgb uchar."""
    rec = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                    ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["red"], rec["green"], rec["blue"] = cols[:, 0], cols[:, 1], cols[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(pts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    with path.open("wb") as f:
        f.write(header.encode())
        f.write(rec.tobytes())


def reconstruct(name: str, cam3: str, subdir: str) -> None:
    set_dir, out_dir = GT_ROOT / name, out_dir_of(name, subdir)
    frames = [set_dir / "frames" / f"{v}.png" for v in VIEW_NAMES]
    if cam3 == "raw":
        frames[2] = raw_cam3_frame(set_dir)

    views = load_views(out_dir, frames)
    if cam3 == "raw":
        v = views[2]
        v["K_stored"], v["extr_stored"] = roll_cam3_to_upright(v["K_stored"], v["extr"], v["size"][1])
        v["size"] = (v["size"][1], v["size"][0])
    pts, cols, vid, conf = build_cloud(views)

    calib_path = GT_ROOT / name / "calibration.npz"
    ref = CALIB_REF.get(name)
    ref_path = GT_ROOT / ref / "calibration.npz" if ref else None
    calib, n_cams = None, 0
    if calib_path.exists():
        calib, n_cams = dict(np.load(calib_path, allow_pickle=True)), 3
        cmp_ = compare(views, calib, n_cams)
        scale, source = cmp_["umeyama_scale_m_per_unit"], f"umeyama-3cam:{name}"
    elif ref_path is not None and ref_path.exists():
        calib, n_cams = dict(np.load(ref_path, allow_pickle=True)), 2
        cmp_ = {"reference_set": ref,
                "note": "cam_3 was bumped, so only cam_left/cam_right of set "
                        f"{ref} apply; scale from their calibrated baseline",
                **compare(views, calib, n_cams)}
        scale, source = cmp_["baseline_scale_m_per_unit"], f"baseline-left-right:{ref}"
    else:
        cmp_ = {"note": "no calibration available"}
        scale, source = 1.0, "none (up to scale)"

    pts = (pts * scale).astype(np.float64)
    extr = np.stack([v["extr_stored"] for v in views])
    extr[:, :3, 3] *= scale
    for k, v in enumerate(views):
        v["extr_metric"] = extr[k]
    K = np.stack([v["K_stored"] for v in views])
    size = np.array([v["size"] for v in views], np.int32)

    if calib is not None:
        cmp_["direct_centre_error_mm"] = direct_centre_error_mm(views, calib, scale, n_cams)
        cmp_["intrinsics_source"] = str(calib.get("intrinsics_source", "unknown"))
    cmp_["cross_view_agreement"] = cross_view_agreement(views, pts, vid)
    ref_scene = out_dir_of(ref, subdir) / "scene.npz" if ref else None
    if ref_scene is not None and ref_scene.exists():
        cmp_["sibling_agreement"] = {"vs_set": ref, **sibling_agreement(pts, vid, ref_scene)}

    np.savez(out_dir / "scene.npz", points=pts.astype(np.float32), colors=cols,
             view_id=vid, conf=conf,
             extrinsics=extr.astype(np.float32), K=K.astype(np.float32), image_size=size,
             scale=np.float64(scale), scale_source=np.str_(source),
             metric=np.bool_(calib is not None), view_names=np.array(VIEW_NAMES),
             cam3_input=np.str_(cam3))
    write_ply(out_dir / "scene.ply", pts.astype(np.float32), cols)

    lo, hi = pts.min(0), pts.max(0)
    cmp_["scale_m_per_unit"] = scale
    cmp_["scale_source"] = source
    cmp_["metric"] = calib is not None
    cmp_["cam3_input"] = cam3
    cmp_["points_per_view"] = {n: int((vid == k).sum()) for k, n in enumerate(VIEW_NAMES)}
    cmp_["bbox_m"] = {"min": lo.tolist(), "max": hi.tolist(), "extent": (hi - lo).tolist()}
    (out_dir / "comparison.json").write_text(json.dumps(cmp_, indent=2, default=float))

    print(f"\n=== {name}  (cam_3 {cam3}) -> {out_dir}")
    print(json.dumps(cmp_, indent=2, default=float))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sets", nargs="+", default=SETS, choices=SETS, help="camera sets to process")
    ap.add_argument("--cam3", choices=["upright", "raw"], default="upright",
                    help="feed cam_3 upright (portrait, padded batch) or raw landscape")
    ap.add_argument("--subdir", default="", help="write under <set>/<subdir>/ instead of <set>/")
    ap.add_argument("--no-vggt", action="store_true", help="reuse the existing vggt/ output")
    args = ap.parse_args()

    if not args.no_vggt:
        run_vggt(args.sets, args.cam3, args.subdir)
    for name in args.sets:
        reconstruct(name, args.cam3, args.subdir)


if __name__ == "__main__":
    main()
