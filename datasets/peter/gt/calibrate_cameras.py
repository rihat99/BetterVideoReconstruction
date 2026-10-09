"""Classical chessboard calibration of the three climb_wall_3 cameras.

A person walks around with a 9x6-inner-corner, 80 mm chessboard in front of the
three frame-synchronised cameras, once per wall inclination (sets ``00``,
``16``, ``32``). This finds the board, solves one set of intrinsics per
physical camera, and the two relative poses per set for the pairs
(cam_left, cam_right) and (cam_left, cam_3). World = cam_left.

``cam_3`` hangs 90 degrees rolled, so its frames are turned upright before
detection -- the same convention as the stored frames in ``<set>/frames/`` --
and its intrinsics therefore describe the stored 1080x1920 image.

Three things are worth knowing about how this is solved.

**Intrinsics are pooled over all three calibration videos.** It is the same
three cameras every time; only the rig layout moved between sessions. Fitting
each video on its own leaves the focal length badly determined (cam_right came
out with a 7 % gap between fx and fy on one video alone), so the boards from
all three videos are fitted together and the one result is written into every
set. The extrinsics stay per set, because the cameras really did move between
sessions. The per-video fits are still reported next to the pooled one: if the
lenses had been refocused between sessions the pooled error would sit outside
the per-video range.

**The frame offset between two cameras is measured, not assumed.** The videos
were started together, but a camera can lag by a frame. For each pair the
stereo fit is run at offsets -2..+2 and the offset with the lowest reprojection
error wins.

**The corner ordering has to mean the same corner in every camera.** Each pair
is solved with the second camera's corners as they come and with the list
reversed (the 180-degree turn of the board), and the lower error wins.

A frame only joins a pair if the board moved slower than MAX_SPEED_PX between
that frame and the next one in both views, so a residual sync error cannot turn
into a large correspondence error.

Writes ``peter/climb_wall_3_gt/scene/<set>/calibration.{npz,json}``. It also
triangulates the board through each pair: recovering the 80 mm pitch is a
*consistency* check on the conventions and on the size of the extrinsics -- it
is not an independent check of absolute scale, since the 80 mm is what put the
metre into the fit in the first place.

    uv run --no-sync python datasets/peter/gt/calibrate_cameras.py
"""
from __future__ import annotations

import os

# OpenCV's solvers spawn one thread per core here (224), which makes them
# slower and hogs a shared machine. Has to be set before cv2 is imported.
for _var in ("OMP_NUM_THREADS", "OPENCV_FOR_THREADS_NUM", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np


from datasets.peter.paths import PETER

ORIG = PETER / "climb_wall_3_orig"
GT = PETER / "climb_wall_3_gt/scene"

CAMS = ["cam_left", "cam_right", "cam_3"]
REFERENCE = "cam_left"  # world, and the first camera of both pairs
PAIRS = [("cam_left", "cam_right"), ("cam_left", "cam_3")]
SETS = ["00", "16", "32"]
PATTERN = (9, 6)  # inner corners, columns x rows
SQUARE_M = 0.08
N_CORNERS = PATTERN[0] * PATTERN[1]
STRIDE = 5  # a pair is anchored on every 5th frame of the calibration video
OFFSETS = tuple(range(-4, 9))  # candidate frame lags of the second camera
MAX_VIEWS = 120  # views per video handed to a solver (more gets very slow)
SCAN_VIEWS = 60  # views used while ranking the frame offsets
FLIP_VIEWS = 30  # views used to decide the corner ordering (0.4 px vs 40 px, so a few do)
MAX_SPEED_PX = 3.0  # board motion between neighbouring frames, in pixels
N_TRIANGULATE = 10  # boards used for the consistency check
SB_FLAGS = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
DIST_FLAGS = cv2.CALIB_FIX_K3  # k1 k2 p1 p2 free; k3 only overfits these lenses
ITERS = 1000
CRITERIA = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, ITERS, 1e-9)
HALF_CRITERIA = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, ITERS // 2, 1e-9)


def object_points() -> np.ndarray:
    """The board corners in board coordinates (z = 0), in metres."""
    grid = np.zeros((N_CORNERS, 3), np.float32)
    grid[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_M
    return grid


def image_size(cam: str) -> tuple[int, int]:
    """Width, height of the stored frame -- cam_3 is the upright one."""
    return (1080, 1920) if cam == "cam_3" else (1920, 1080)


def detect(job: tuple[str, str]) -> tuple[str, list[int], np.ndarray, int]:
    """Find the board in one calibration video.

    The reference camera anchors the pairs, so it is only looked at on every
    STRIDE-th frame and the one after it (which the speed gate needs). Every
    frame of the other cameras is looked at, because the offset search needs
    their whole -2..+2 neighbourhood.
    """
    calib_dir, cam = job
    every_frame = cam != REFERENCE
    cap = cv2.VideoCapture(str(ORIG / "calibration" / calib_dir / f"{cam}.mkv"))
    frames: list[int] = []
    corners: list[np.ndarray] = []
    index = looked = 0
    while cap.grab():
        if every_frame or index % STRIDE in (0, 1):
            looked += 1
            ok, frame = cap.retrieve()
            if ok:
                if cam == "cam_3":
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found, pts = cv2.findChessboardCornersSB(gray, PATTERN, flags=SB_FLAGS)
                if found:
                    frames.append(index)
                    corners.append(pts.reshape(-1, 2))
        index += 1
    cap.release()
    return cam, frames, np.asarray(corners, np.float32).reshape(-1, N_CORNERS, 2), looked


def spread(n: int, k: int) -> np.ndarray:
    """At most k indices, spread evenly over range(n)."""
    return np.unique(np.linspace(0, n - 1, min(k, n)).round().astype(int))


def moved(a: np.ndarray, b: np.ndarray) -> float:
    """Mean corner displacement between two frames, in pixels."""
    return float(np.linalg.norm(b - a, axis=1).mean())


def pair_views(seen_a: dict[int, np.ndarray], seen_b: dict[int, np.ndarray],
               offset: int) -> tuple[np.ndarray, np.ndarray]:
    """Corner pairs where both cameras saw a slow-moving board, at one offset."""
    rows_a, rows_b = [], []
    for i in sorted(seen_a):
        j = i + offset
        if i % STRIDE or i + 1 not in seen_a or j not in seen_b or j + 1 not in seen_b:
            continue
        if moved(seen_a[i], seen_a[i + 1]) > MAX_SPEED_PX:
            continue
        if moved(seen_b[j], seen_b[j + 1]) > MAX_SPEED_PX:
            continue
        rows_a.append(seen_a[i])
        rows_b.append(seen_b[j])
    return (np.asarray(rows_a, np.float32).reshape(-1, N_CORNERS, 2),
            np.asarray(rows_b, np.float32).reshape(-1, N_CORNERS, 2))


def fit_intrinsics(job: tuple[str, str, np.ndarray, bool]) -> tuple:
    """One calibrateCamera run. Takes a label so it can go through a pool."""
    label, cam, views, half = job
    rms, K, dist, _, _ = cv2.calibrateCamera(
        [object_points()] * len(views), [v.reshape(-1, 1, 2) for v in views],
        image_size(cam), None, None, flags=DIST_FLAGS,
        criteria=HALF_CRITERIA if half else CRITERIA,
    )
    return label, float(rms), K, dist.ravel()[:5], len(views)


def stereo_fit(pts_a: np.ndarray, pts_b: np.ndarray, cal_a: tuple[np.ndarray, np.ndarray],
               cal_b: tuple[np.ndarray, np.ndarray], size: tuple[int, int]):
    """Pose of camera b in camera a's frame."""
    rms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        [object_points()] * len(pts_a),
        list(pts_a.reshape(-1, N_CORNERS, 1, 2)), list(pts_b.reshape(-1, N_CORNERS, 1, 2)),
        cal_a[0], cal_a[1], cal_b[0], cal_b[1], size,
        flags=cv2.CALIB_FIX_INTRINSIC, criteria=CRITERIA,
    )
    return float(rms), R, T.reshape(3)


def order_is_flipped(pts_a: np.ndarray, pts_b: np.ndarray, cal_a: tuple[np.ndarray, np.ndarray],
                     cal_b: tuple[np.ndarray, np.ndarray], size: tuple[int, int]) -> bool:
    """Is the second camera's corner list the 180-degree turn of the first's?

    The wrong ordering misses by tens of pixels, so a handful of views settles it.
    """
    k = spread(len(pts_a), FLIP_VIEWS)
    straight = stereo_fit(pts_a[k], pts_b[k], cal_a, cal_b, size)[0]
    reversed_ = stereo_fit(pts_a[k], pts_b[k][:, ::-1], cal_a, cal_b, size)[0]
    return reversed_ < straight


def triangulate_board(pts_a: np.ndarray, pts_b: np.ndarray, cal_a: tuple[np.ndarray, np.ndarray],
                      cal_b: tuple[np.ndarray, np.ndarray], R: np.ndarray, T: np.ndarray) -> dict:
    """Consistency check: the square pitch and the board flatness, in metres.

    Not a check of absolute scale -- the 80 mm square is what set the metre.
    """
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, T.reshape(3, 1)])
    pitches, flats, depths = [], [], []
    for i in spread(len(pts_a), N_TRIANGULATE):
        na = cv2.undistortPoints(pts_a[i].reshape(-1, 1, 2), cal_a[0], cal_a[1]).reshape(-1, 2)
        nb = cv2.undistortPoints(pts_b[i].reshape(-1, 1, 2), cal_b[0], cal_b[1]).reshape(-1, 2)
        hom = cv2.triangulatePoints(P1, P2, na.T.astype(np.float64), nb.T.astype(np.float64))
        xyz = (hom[:3] / hom[3]).T
        grid = xyz.reshape(PATTERN[1], PATTERN[0], 3)
        pitches.append(np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel())
        pitches.append(np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel())
        flats.append(np.linalg.svd(xyz - xyz.mean(0), compute_uv=False)[2] / np.sqrt(N_CORNERS))
        depths.append(xyz[:, 2].mean())
    pitch = np.concatenate(pitches)
    return {
        "pitch_mean_mm": float(pitch.mean()) * 1e3,
        "pitch_std_mm": float(pitch.std()) * 1e3,
        "planarity_rms_mm": float(np.mean(flats)) * 1e3,
        "board_depth_m": float(np.mean(depths)),
        "n_boards": len(depths),
    }


def angles(R: np.ndarray) -> tuple[float, float]:
    """Angle between this camera's optical axis and cam_left's, and its roll about that axis."""
    axis = R[2]  # the camera's viewing direction, written in cam_left coordinates
    up = -R[1]  # and its up direction
    axis_deg = float(np.degrees(np.arccos(np.clip(axis[2], -1.0, 1.0))))
    up_left = np.array([0.0, -1.0, 0.0])
    ref = up_left - np.dot(up_left, axis) * axis  # cam_left's up, made perpendicular to this axis
    roll_deg = float(np.degrees(np.arctan2(np.dot(np.cross(ref, up), axis), np.dot(ref, up))))
    return axis_deg, roll_deg


def run_detection(calib_dirs: dict[str, str]) -> dict[tuple[str, str], dict[int, np.ndarray]]:
    jobs = [(calib_dirs[s], cam) for s in SETS for cam in CAMS]
    started = time.time()
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        results = list(pool.map(detect, jobs))
    seen = {}
    for s in SETS:
        for cam in CAMS:
            _, frames, corners, looked = results.pop(0)
            seen[(s, cam)] = dict(zip(frames, corners))
            print(f"  {s} {cam:9s} board found in {len(frames):5d} of {looked} frames looked at", flush=True)
    print(f"  detection took {time.time() - started:.0f}s", flush=True)
    return seen


def run_intrinsics(seen: dict) -> tuple[dict, dict]:
    """One pooled fit per camera, plus the per-video fits and a half-iteration run."""
    jobs = []
    for cam in CAMS:
        pooled = []
        for s in SETS:
            views = [v for i, v in sorted(seen[(s, cam)].items()) if i % STRIDE == 0]
            picked = [views[i] for i in spread(len(views), MAX_VIEWS)]
            pooled += picked
            jobs.append((f"video {s}", cam, np.asarray(picked), False))
        jobs.append((f"pooled {ITERS}", cam, np.asarray(pooled), False))
        jobs.append((f"pooled {ITERS // 2}", cam, np.asarray(pooled), True))
    started = time.time()
    with ProcessPoolExecutor(max_workers=8) as pool:
        out = {(label, cam): (rms, K, dist, n) for (label, rms, K, dist, n), (_, cam, _, _)
               in zip(pool.map(fit_intrinsics, jobs), jobs)}
    print(f"  intrinsic fits took {time.time() - started:.0f}s", flush=True)

    intr, report = {}, {}
    for cam in CAMS:
        rms, K, dist, n = out[(f"pooled {ITERS}", cam)]
        half = out[(f"pooled {ITERS // 2}", cam)][1]
        per_video = {s: out[(f"video {s}", cam)] for s in SETS}
        intr[cam] = (rms, K, dist, n)
        lo = min(v[0] for v in per_video.values())
        hi = max(v[0] for v in per_video.values())
        report[cam] = {
            "pooled": {"rms_px": rms, "n_views": n, "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                       "cx": float(K[0, 2]), "cy": float(K[1, 2]), "dist": [float(x) for x in dist]},
            "per_video": {s: {"rms_px": v[0], "n_views": v[3], "fx": float(v[1][0, 0]),
                              "fy": float(v[1][1, 1]), "cx": float(v[1][0, 2]), "cy": float(v[1][1, 2])}
                          for s, v in per_video.items()},
            "pooled_rms_within_per_video_range": bool(lo <= rms <= hi),
            "half_iteration_K_change": {"fx": float(K[0, 0] - half[0, 0]), "fy": float(K[1, 1] - half[1, 1]),
                                        "cx": float(K[0, 2] - half[0, 2]), "cy": float(K[1, 2] - half[1, 2])},
        }
        print(f"  {cam:9s} pooled n={n:3d} rms={rms:.4f}  fx={K[0,0]:8.2f} fy={K[1,1]:8.2f} "
              f"cx={K[0,2]:8.2f} cy={K[1,2]:8.2f} fx/fy-1={K[0,0]/K[1,1]-1:+.4f}", flush=True)
        print(f"            dist={np.array2string(dist, precision=4)}", flush=True)
        for s in SETS:
            v = per_video[s]
            print(f"            video {s}: n={v[3]:3d} rms={v[0]:.4f} fx={v[1][0,0]:8.2f} fy={v[1][1,1]:8.2f} "
                  f"cx={v[1][0,2]:8.2f} cy={v[1][1,2]:8.2f}", flush=True)
        flag = "" if lo <= rms <= hi else "   <-- OUTSIDE, lenses may have been refocused"
        print(f"            pooled rms {rms:.4f} vs per-video range {lo:.4f}..{hi:.4f}{flag}", flush=True)
        print(f"            convergence, K({ITERS}) - K({ITERS // 2}): dfx={K[0,0]-half[0,0]:+.4f} "
              f"dfy={K[1,1]-half[1,1]:+.4f} dcx={K[0,2]-half[0,2]:+.4f} dcy={K[1,2]-half[1,2]:+.4f}", flush=True)
    return intr, report


def run_set(name: str, calib_dir: str, seen: dict, intr: dict, intr_report: dict) -> None:
    print(f"\n=== set {name}  ({calib_dir}) ===", flush=True)
    extrinsics = np.tile(np.eye(4), (3, 1, 1))
    rms_stereo, n_stereo, offsets, pair_info, checks = [], [], [], {}, {}

    for a, b in PAIRS:
        cal_a = (intr[a][1], intr[a][2])
        cal_b = (intr[b][1], intr[b][2])
        size = image_size(a)

        scan = {}
        for off in OFFSETS:
            pa, pb = pair_views(seen[(name, a)], seen[(name, b)], off)
            if len(pa) < 10:
                scan[off] = (float("inf"), len(pa))
                continue
            k = spread(len(pa), SCAN_VIEWS)
            scan[off] = (stereo_fit(pa[k], pb[k], cal_a, cal_b, size)[0], len(pa))
        best_off = min(OFFSETS, key=lambda o: scan[o][0])
        at_edge = best_off in (OFFSETS[0], OFFSETS[-1])
        row = "  ".join(f"{o:+d}:{scan[o][0]:6.3f}({scan[o][1]})" for o in OFFSETS)
        print(f"  {a}->{b} offset scan (rms px, n avail, {SCAN_VIEWS} views used): {row}", flush=True)
        print(f"      chosen offset {best_off:+d}"
              f"{'   <-- AT THE EDGE OF THE SEARCH, widen OFFSETS' if at_edge else ''}", flush=True)

        pa, pb = pair_views(seen[(name, a)], seen[(name, b)], best_off)
        k = spread(len(pa), MAX_VIEWS)
        pa, pb = pa[k], pb[k]
        flipped = order_is_flipped(pa, pb, cal_a, cal_b, size)
        used_b = pb[:, ::-1] if flipped else pb  # the self-check must see what the fit saw
        rms, R, T = stereo_fit(pa, used_b, cal_a, cal_b, size)

        extrinsics[CAMS.index(b), :3, :3] = R
        extrinsics[CAMS.index(b), :3, 3] = T
        rms_stereo.append(rms)
        n_stereo.append(len(pa))
        offsets.append(best_off)
        axis_deg, roll_deg = angles(R)
        check = triangulate_board(pa, used_b, cal_a, cal_b, R, T)
        pair_info[f"{a}->{b}"] = {
            "rms_px": rms, "n_views": len(pa), "frame_offset": best_off,
            "offset_scan_rms_px": {str(o): (None if scan[o][0] == float("inf") else scan[o][0]) for o in OFFSETS},
            "offset_scan_n_pairs": {str(o): scan[o][1] for o in OFFSETS},
            "corner_order_flipped": bool(flipped), "offset_at_search_edge": at_edge,
            "baseline_m": float(np.linalg.norm(T)),
            "optical_axis_angle_deg": axis_deg, "roll_deg": roll_deg,
        }
        checks[f"{a}->{b}"] = check
        print(f"  stereo {a}->{b}: rms={rms:.3f}px n={len(pa)} offset={best_off:+d} flipped={flipped}", flush=True)
        print(f"      baseline {np.linalg.norm(T) * 100:.2f} cm, optical axes {axis_deg:.2f} deg apart, "
              f"roll {roll_deg:+.2f} deg", flush=True)
        print(f"      consistency check on {check['n_boards']} boards at {check['board_depth_m']:.2f} m: pitch "
              f"{check['pitch_mean_mm']:.2f} +- {check['pitch_std_mm']:.2f} mm (asserted 80.00), "
              f"planarity rms {check['planarity_rms_mm']:.2f} mm", flush=True)

    out = GT / name
    np.savez(
        out / "calibration.npz",
        view_names=np.array(CAMS),
        K=np.stack([intr[c][1] for c in CAMS]).astype(np.float64),
        dist=np.stack([intr[c][2] for c in CAMS]).astype(np.float64),
        image_size=np.array([image_size(c) for c in CAMS], np.int32),
        extrinsics=extrinsics.astype(np.float64),
        rms_intrinsic=np.array([intr[c][0] for c in CAMS]),
        n_intrinsic=np.array([intr[c][3] for c in CAMS], np.int32),
        rms_stereo=np.array(rms_stereo),
        n_stereo=np.array(n_stereo, np.int32),
        frame_offset=np.array(offsets, np.int32),
        intrinsics_source="pooled:" + "+".join(SETS),
        pattern=np.array(PATTERN, np.int32),
        square_m=np.float64(SQUARE_M),
    )
    (out / "calibration.json").write_text(json.dumps({
        "set": name,
        "calibration_video": calib_dir,
        "world": "cam_left",
        "intrinsics_source": "pooled:" + "+".join(SETS),
        "pattern_inner_corners": list(PATTERN),
        "square_m": SQUARE_M,
        "frame_stride": STRIDE,
        "max_board_speed_px_per_frame": MAX_SPEED_PX,
        "solver_iterations": ITERS,
        "cameras": {cam: {"image_size_wh": list(image_size(cam)), **intr_report[cam]} for cam in CAMS},
        "pairs": pair_info,
        "consistency_check_triangulated_board": checks,
    }, indent=2) + "\n")
    print(f"  wrote {out / 'calibration.npz'} and {out / 'calibration.json'}", flush=True)


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()   # --help only; no knobs
    calib_dirs = {s: json.loads((GT / s / "frames/frames.json").read_text())["calibration"] for s in SETS}
    print("=== detection ===", flush=True)
    seen = run_detection(calib_dirs)
    print("\n=== intrinsics, pooled over " + "+".join(SETS) + " ===", flush=True)
    intr, intr_report = run_intrinsics(seen)
    for name in SETS:
        run_set(name, calib_dirs[name], seen, intr, intr_report)


if __name__ == "__main__":
    main()
