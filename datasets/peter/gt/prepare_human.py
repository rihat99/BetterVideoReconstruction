"""Build the per-clip working tree for the climb_wall_3 human ground truth.

For every selected clip of ``peter/climb_wall_3/`` this writes
``peter/climb_wall_3_gt/human/<clip>/``:

* ``views/<view>/video.mp4`` — the clip's three cameras, **undistorted** with the
  chessboard calibration and, for ``cam_3``, turned upright (the camera was mounted
  90 degrees off; the people detectors need people standing up). Every view is cut
  to the same frame count.
* ``views/<view>/vggt/camera.npz`` — the calibrated pinhole camera in the layout
  the SAM-3D-Body stage reads (normalized intrinsics per frame), so the standard
  stage CLIs run on each view unchanged.
* ``cameras.npz`` — the three calibrated cameras (camera-from-world, world =
  cam_left, metres) the multi-view fit uses.
* ``geometry/scene.npz`` + ``geometry/transform.npz`` — the set's cleaned
  human-free scene cloud with outward normals, plus a cam_left transform, in the
  layout the contact and viewer tools expect.
* ``geocalib/gravity.npz`` — the inclination's gravity (copied from the
  single-view run: same world, the cam_left camera frame).
* ``sensor_forces.npz`` — the four boards' measured forces per video frame,
  rotated into the world (``datasets/peter/peter_sensor_forces.py``'s per-inclination
  board-to-world map).

The camera set (cam_3 pose) follows the recording, see
:mod:`datasets.peter.gt.extract_frames`. Sets without their own calibration (16b,
16c: cam_3 was bumped) take cam_3's pose from the scaled VGGT solution and its
intrinsics from set 16.

    uv run --no-sync python datasets/peter/gt/prepare_human.py [--clips a_00_01_part3,...]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch


from datasets.peter.peter_sensor_forces import R_CW3_WALL_TO_WORLD, per_frame
from bvr.geometry.intrinsics import normalize_K
from bvr.geometry.normals import estimate_point_normals
from bvr.paths import REPO_ROOT
from datasets.peter.paths import PETER

from datasets.peter.gt.extract_frames import SETS

CLIPS_DIR = PETER / "climb_wall_3"
SINGLE_OUT = PETER / "out_climb_wall_3_single"
GT_SCENE = PETER / "climb_wall_3_gt/scene"
HUMAN_ROOT = PETER / "climb_wall_3_gt/human"
VIEW_NAMES = ["cam_left", "cam_right", "cam_3"]
CALIB_REF = {"16b": "16", "16c": "16"}
FFMPEG = next((REPO_ROOT / ".venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries").glob("ffmpeg-*"))
FPS = 30.0
#: The six first clips: both climbers at every inclination, mid-climb parts, and
#: only 16-degree recordings whose cam_3 pose the calibration video saw (set 16).
DEFAULT_CLIPS = ["a_00_01_part3", "b_00_02_part2", "a_16_01_part2",
                 "b_16_01_part3", "a_32_01_part3", "b_32_02_part2"]


def camera_set(clip: str) -> str:
    """The camera set (cam_3 pose) of a clip, from its recording name."""
    recording = clip.rsplit("_part", 1)[0]
    for name, spec in SETS.items():
        if recording in spec["recordings"]:
            return name
    raise KeyError(f"{clip}: recording {recording} is in no camera set")


def load_cameras(set_name: str) -> dict:
    """Calibrated K / dist / camera-from-world per view; cam_3 from VGGT when uncalibrated."""
    calib = np.load(GT_SCENE / CALIB_REF.get(set_name, set_name) / "calibration.npz", allow_pickle=True)
    assert [str(v) for v in calib["view_names"]] == VIEW_NAMES
    extr = np.asarray(calib["extrinsics"], np.float64).copy()
    cam3_source = "chessboard"
    if set_name in CALIB_REF:
        scene = np.load(GT_SCENE / set_name / "scene.npz")
        extr[2] = np.asarray(scene["extrinsics"][2], np.float64)
        cam3_source = f"vggt ({str(scene['scale_source'])})"
        print(f"  cam_3 pose of set {set_name} is NOT calibrated — using the scaled VGGT pose")
    return {"K": np.asarray(calib["K"], np.float64), "dist": np.asarray(calib["dist"], np.float64),
            "extrinsics": extr, "image_size": np.asarray(calib["image_size"], int),
            "cam3_pose_source": cam3_source}


def count_frames(video: Path) -> int:
    cap = cv2.VideoCapture(str(video))
    n = 0
    while cap.read()[0]:
        n += 1
    return n


def write_video(src: Path, dst: Path, n_frames: int, K: np.ndarray, dist: np.ndarray,
                size_wh: tuple[int, int], rotate: bool) -> None:
    """Decode ``src``, rotate (cam_3) + undistort every frame, encode ``n_frames`` of them."""
    w, h = size_wh
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_16SC2)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(FFMPEG), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "fast", "-crf", "12", "-pix_fmt", "yuv420p", str(dst)]
    cap = cv2.VideoCapture(str(src))
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as enc:
        for _ in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"{src}: ran out of frames before {n_frames}")
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            assert frame.shape[:2] == (h, w), (frame.shape, size_wh)
            enc.stdin.write(cv2.remap(frame, map1, map2, cv2.INTER_LINEAR).tobytes())
        enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed on {dst}")
    got = count_frames(dst)
    if got != n_frames:
        raise RuntimeError(f"{dst}: wrote {got} frames, expected {n_frames}")


def write_view_camera(view_dir: Path, K: np.ndarray, extr: np.ndarray, size_wh: tuple[int, int],
                      n_frames: int) -> None:
    """The per-view camera in the ``vggt/camera.npz`` layout SAM-3D-Body reads."""
    w, h = size_wh
    idx = np.arange(n_frames, dtype=np.int32)
    (view_dir / "vggt").mkdir(parents=True, exist_ok=True)
    np.savez(view_dir / "vggt" / "camera.npz",
             extrinsics=np.repeat(extr[None].astype(np.float32), n_frames, 0),
             intrinsics=np.repeat(normalize_K(K, h, w)[None].astype(np.float32), n_frames, 0),
             intrinsics_px=np.repeat(K[None].astype(np.float32), n_frames, 0),
             frame_indices=idx, view_ids=np.zeros(n_frames, np.int32),
             image_height=np.int32(h), image_width=np.int32(w), num_frames=np.int32(n_frames),
             scale=np.str_("metric"), source=np.str_("chessboard calibration (prepare_human.py)"))


def wall_plane(points: np.ndarray, normals: np.ndarray, camera: np.ndarray,
               tol_m: float = 0.015, iters: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """The climbing wall as a plane: RANSAC over the cloud points that face the camera.

    Returns ``(normal, point, inlier_fraction)`` with the normal pointing toward ``camera``
    (out of the wall). Only points whose own normal is within 50 deg of the direction to the
    camera vote, so the floor and the side walls cannot win; the plane is refined by a
    least-squares fit to its inliers.
    """
    rng = np.random.default_rng(seed)
    to_cam = camera[None] - points
    to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True) + 1e-9
    facing = (normals * to_cam).sum(1) > np.cos(np.radians(50.0))
    cand = points[facing] if facing.sum() > 1000 else points
    sample = cand[rng.choice(len(cand), min(len(cand), 60000), replace=False)]
    best_n, best_p, best_count = None, None, -1
    for _ in range(iters):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        count = int((np.abs((sample - a) @ n) < tol_m).sum())
        if count > best_count:
            best_n, best_p, best_count = n, a, count
    inl = sample[np.abs((sample - best_p) @ best_n) < tol_m]
    centre = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - centre, full_matrices=False)
    normal = vt[2]
    if (camera - centre) @ normal < 0:
        normal = -normal
    frac = float((np.abs((points - centre) @ normal) < tol_m).mean())
    return normal, centre, frac


def write_scene(clip_dir: Path, set_name: str, cams: dict, n_frames: int) -> None:
    """The set's wall cloud + a cam_left transform in the pipeline's layout.

    ``scene_calib.npz`` (the cloud rebuilt in the calibrated cameras' frame) when the set
    has one, else the older ``scene_clean.npz`` (VGGT's own poses, similarity-aligned —
    cm to dm off the calibrated cameras at the wall).
    """
    src = GT_SCENE / set_name / "scene_calib.npz"
    if not src.exists():
        src = GT_SCENE / set_name / "scene_clean.npz"
        print(f"  WARNING: {set_name} has no scene_calib.npz — using the VGGT-frame scene_clean.npz")
    z = np.load(src)
    pts = np.asarray(z["points"], np.float32)
    # Normals are oriented toward the nearest camera, so they must come from the CALIBRATED rig
    # (the pose saved before the per-clip cam_3 PnP): every clip of a set then gets the same
    # normals and the same wall plane, and a badly placed cam_3 cannot flip the wall.
    rig = cams["extrinsics_before_pnp"] if "extrinsics_before_pnp" in cams else cams["extrinsics"]
    centres = np.stack([-e[:3, :3].T @ e[:3, 3] for e in rig]).astype(np.float32)
    normals = estimate_point_normals(torch.from_numpy(pts), torch.from_numpy(centres), k=16).numpy()
    geo = clip_dir / "geometry"
    geo.mkdir(parents=True, exist_ok=True)
    np.savez(geo / "scene.npz", points=pts, colors=np.asarray(z["colors"]),
             confidence=np.asarray(z["conf"], np.float32), normals=normals,
             view_id=np.asarray(z["view_id"]), metric=np.bool_(True), scale=np.float32(1.0),
             source=np.str_(str(src.relative_to(PETER.parent))), camera_set=np.str_(set_name))
    normal, point, inliers = wall_plane(pts, normals, centres[0])
    np.savez(geo / "wall.npz", normal=normal.astype(np.float32), point=point.astype(np.float32),
             inlier_frac=np.float32(inliers), source=np.str_(str(src.relative_to(PETER.parent))),
             note=np.str_("the climbing wall's plane: normal points out of the wall toward the cameras; "
                          "holds stand in front of it, nothing of the climber can be behind it"))
    print(f"  wall plane: normal {normal.round(3)}, {inliers:.0%} of the cloud within 1.5 cm")
    K_left = cams["K"][0].astype(np.float32)
    w, h = (int(x) for x in cams["image_size"][0])
    np.savez(geo / "transform.npz",
             scale=np.float32(1.0), metric=np.bool_(True), static_camera=np.bool_(True),
             extrinsics=np.repeat(np.eye(4, dtype=np.float32)[None], n_frames, 0),
             intrinsics_px_orig=np.repeat(K_left[None], n_frames, 0),
             intrinsics=np.repeat(normalize_K(K_left, h, w)[None].astype(np.float32), n_frames, 0),
             frame_indices=np.arange(n_frames, dtype=np.int32),
             image_height=np.int32(h), image_width=np.int32(w), fps=np.float32(FPS),
             scale_method=np.str_("chessboard calibration (metres)"))


def write_sensor_forces(clip: str, clip_dir: Path, n_frames: int) -> None:
    """Board forces per video frame, world frame (reaction on the climber) + board frame."""
    z = np.load(CLIPS_DIR / clip / "forces.npz", allow_pickle=True)
    incl = clip.split("_")[1]
    board = per_frame(np.asarray(z["forces"], np.float32),
                      float(z["fps"]) / float(z["video_fps"]), n_frames)          # (N, 4, 3) board frame
    rotation = R_CW3_WALL_TO_WORLD[incl]
    reaction = -(board @ rotation.T)
    np.savez(clip_dir / "sensor_forces.npz",
             reaction_world=reaction.astype(np.float32), total_world=reaction.sum(1).astype(np.float32),
             board_frame=board.astype(np.float32), limbs=np.asarray(z["limbs"]),
             R_board_to_world=rotation, wall_decline_deg=np.float32(z["wall_decline_deg"]),
             video_fps=np.float32(z["video_fps"]), source=z["source"],
             note=np.str_("reaction_world = force ON the climber, world = cam_left camera frame; "
                          "board_frame = the logged per-limb force in the board frame "
                          "(+x up the wall, +y right, +z out toward the climber)"))


def prepare(clip: str, skip_existing: bool, scene_only: bool = False) -> None:
    clip_dir = HUMAN_ROOT / clip
    if scene_only:       # a rebuilt wall cloud: refresh geometry/ only (cameras.npz keeps its frame offsets)
        cams = np.load(clip_dir / "cameras.npz", allow_pickle=True)
        write_scene(clip_dir, str(cams["camera_set"]), cams, int(cams["num_frames"]))
        print(f"{clip}: scene refreshed")
        return
    if skip_existing and (clip_dir / "clip.json").exists():      # written last = the clip is complete
        print(f"{clip}: prepared, skipped")
        return
    set_name = camera_set(clip)
    print(f"{clip}: camera set {set_name}")
    cams = load_cameras(set_name)
    src_dir = CLIPS_DIR / clip
    forces = np.load(src_dir / "forces.npz")
    assert float(forces["video_fps"]) == FPS, (clip, float(forces["video_fps"]))
    n_frames = min(int(forces["n_frames"]),
                   *(count_frames(src_dir / f"{v}.mp4") for v in VIEW_NAMES))
    for k, view in enumerate(VIEW_NAMES):
        size = (int(cams["image_size"][k][0]), int(cams["image_size"][k][1]))
        view_dir = clip_dir / "views" / view
        video = view_dir / "video.mp4"
        if not (skip_existing and video.exists() and count_frames(video) == n_frames):
            write_video(src_dir / f"{view}.mp4", video, n_frames, cams["K"][k], cams["dist"][k],
                        size, rotate=(view == "cam_3"))
        write_view_camera(view_dir, cams["K"][k], cams["extrinsics"][k], size, n_frames)
        print(f"  {view}: {n_frames} frames {size[0]}x{size[1]} undistorted -> {video}")
    np.savez(clip_dir / "cameras.npz", view_names=np.array(VIEW_NAMES), K=cams["K"].astype(np.float32),
             extrinsics=cams["extrinsics"].astype(np.float32), image_size=cams["image_size"],
             dist_removed=cams["dist"].astype(np.float32), camera_set=np.str_(set_name),
             # per-view lag behind cam_left, in frames (0 = unknown / none); the fit shifts each view by it
             frame_offset=np.zeros(3, np.int32),
             cam3_pose_source=np.str_(cams["cam3_pose_source"]), num_frames=np.int32(n_frames),
             fps=np.float32(FPS), world=np.str_("cam_left camera frame, metres"))
    write_scene(clip_dir, set_name, cams, n_frames)
    gravity = SINGLE_OUT / clip / "geocalib" / "gravity.npz"
    (clip_dir / "geocalib").mkdir(exist_ok=True)
    shutil.copy2(gravity, clip_dir / "geocalib" / "gravity.npz")
    write_sensor_forces(clip, clip_dir, n_frames)
    (clip_dir / "clip.json").write_text(json.dumps({
        "clip": clip, "camera_set": set_name, "num_frames": n_frames,
        "cam3_pose_source": cams["cam3_pose_source"],
        "source_videos": str(src_dir.relative_to(PETER.parent)), "gravity": str(gravity.relative_to(PETER.parent)),
    }, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", default=",".join(DEFAULT_CLIPS), help="comma-separated clip names")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--scene-only", action="store_true", help="only rewrite geometry/ from the set's wall cloud")
    args = ap.parse_args()
    for clip in [c.strip() for c in args.clips.split(",") if c.strip()]:
        prepare(clip, args.skip_existing, scene_only=args.scene_only)


if __name__ == "__main__":
    main()
