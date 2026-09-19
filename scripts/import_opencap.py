"""Import OpenCap LabValidation clips (Uhlrich et al., PLOS Comp Biol 2023) into
``data/LabValidation_withVideos/processed/``.

Source tree: ``/home/rikhat.akizhanov/better/data/LabValidation_withVideos/subject*``.
10 subjects x 17 trials, filmed by five static iPhones (720x1280 portrait, 60 fps)
while a Vicon rig (100 Hz markers) and two force plates (2000 Hz) recorded the
ground truth. We import 12 clips: subjects 2/3/4 x ``walking1, squats1, STS1, DJ1``,
camera ``Cam1`` (a 45 deg view — the camera the paper itself reports).

Output layout (one folder per clip)::

    processed/<subject>_<trial>_<cam>/  video.mp4  forces.npz  markers.npz
                                       meta.json  marker_overlay.mp4

``video.mp4`` is the shipped ``<trial>_syncdWithMocap.avi`` (already trimmed to the
mocap window) re-encoded to h264 at its native 60 fps — no resampling, so **row i
of every array is video frame i** and ``t = i / 60`` seconds in both the marker and
the force files, which both start at ``t = 0``.

``forces.npz`` (OpenSim ground frame, Y up, floor at y ~ 0)::

    force        (N, 4, 3) float32  N, reaction ON the person, [left_hand, right_hand,
                                    left_foot, right_foot]; hands are always zero
    cop          (N, 4, 3) float32  centre of pressure, m; NaN where nothing is loaded
    free_torque  (N, 4, 3) float32  free moment about the COP, N*m; NaN likewise
    captured     (N, 4)    bool     False where the plates cannot see the load
    fps / mocap_fps / force_fps     60 / 100 / 2000
    limbs, n_frames, frame, source, mass_kg, height_m, note

Each row is the mean of the 2000 Hz samples in the window CENTRED on the frame's
timestamp, ``[i/60 - 1/120, i/60 + 1/120)``, so the value belongs to the instant the
frame was taken rather than to the 8 ms after it. Resampling averages the **wrench**
— the force and the moment about the ground origin, ``M = p x F + T`` — and only then
recovers the COP and the free torque from that average
(:func:`~scripts.opencap_plates.cop_free_torque`). Averaging the COP directly would be
wrong: ``mean(p x F) != mean(p) x mean(F)``, and on a drop-jump landing the two differ
by ~9 N*m. Samples below :data:`LOADED_N` contribute zero moment (an unloaded plate
holds its last COP instead of resetting, and its true moment is ~0 because its force
is), but they do contribute their force.

``captured`` says the plates could SEE that limb, which is not the same as their
reading zero:

* hands are measured "no force" — nobody touches anything with their hands in these
  four trials — so they are zero with ``captured=True``;
* a foot reading under :data:`LOADED_N` is taken as measured "no force" too. In
  the walking trials every unloaded foot frame is a swing frame (min marker speed
  0.28 m/s), and in the sit-to-stand trials the feet rest ON the plates while the
  chair carries the weight, so the near-zero reading is a real measurement. (A
  stillness rule that masked any still, unloaded foot was tried 2026-09-16: it
  fired only on the seated STS phases and hid the solve's phantom foot load there.)
* in the drop jump the subject starts on a box that no plate sees, so both feet are
  ``captured=False`` until the first loaded sample;
* in the static trials (standing still on both feet) a plate reading nothing is a foot
  standing off the plates, ``captured=False`` on that foot — and where the other plate
  carries the whole body weight both feet stood on it and the file's per-foot split is
  invented (CoP between the feet), so both feet are ``captured=False``. Subjects 2, 9
  and 10;
* a video frame with no 2000 Hz sample in its window is ``captured=False`` on every
  limb.

In the sit-to-stand trials the CHAIR is unmeasured — nothing is masked for it, see
``note``.

``markers.npz``: ``markers (N, 51, 3)`` m in the same frame, linearly resampled from
the 100 Hz TRC onto the video frames, plus ``names (51,)`` and ``fps``.

``meta.json`` carries the camera chain (K, distortion, both transforms and the
composed ``*_cam_from_mocap``) that puts the mocap world into the camera that shot
the clip. Two things about that chain are NOT what the shipped files suggest, both
established by projecting the markers back onto the video (``marker_overlay.mp4``):

* ``cameraIntrinsicsExtrinsics.pickle`` stores ``imageSize`` as ``[1280, 720]``,
  but ``(cx, cy) = (366.5, 638.5)`` is the centre of the **portrait** 720x1280
  frame — K and the 5-term OpenCV distortion apply to the video as it is stored,
  with no rotation. ``translation`` is in mm.
* every ``mocapToVideoTransform.yaml`` ships a rotation that does not agree with
  the camera extrinsics (see :data:`R_VIDEO_FROM_MOCAP`); only its translation is
  used.

Usage::

    uv run python scripts/import_opencap.py                 # the 12 default clips
    uv run python scripts/import_opencap.py --subjects 2 --trials DJ1
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from imageio_ffmpeg import get_ffmpeg_exe, read_frames

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from opencap_plates import LOADED_N, cop_free_torque, origin_moment  # noqa: E402

SRC = Path("/home/rikhat.akizhanov/better/data/LabValidation_withVideos")
DST = SRC / "processed"

SUBJECTS = (2, 3, 4)
TRIALS = ("walking1", "squats1", "STS1", "DJ1")
CAMERA = "Cam1"

LIMBS = ("left_hand", "right_hand", "left_foot", "right_foot")
FPS = 60.0            # video, and the rate every imported array is sampled at
MOCAP_FPS = 100.0     # TRC markers
FORCE_FPS = 2000.0    # force plates
OVERLAY_FRAMES = 120  # frames of marker_overlay.mp4 (covers the drop-jump landing)
#: A trial longer than this is imported as consecutive parts of at most this many frames
#: (``<trial>p1``, ``<trial>p2``, ...): VGGT holds every frame's tokens at once and a
#: 900-frame clip already fills a 47 GB GPU, while the body stages need a camera for every
#: tracked frame, so a subsampled camera track is not an option. 8 of the 153 Cam1 trials
#: (the STSweakLegs ones and subject5's squats1, up to 1312 frames) are split.
MAX_FRAMES = 900


# Mocap (OpenSim ground, Y up) -> the world the camera extrinsics live in.
#
# The shipped R_fromMocap_toVideo cannot be used: Session1's matrix is rank 2
# (rows 1 and 3 are identical, det 0) in EVERY subject, and Session0's, while a
# proper rotation, maps mocap up to a horizontal axis of the extrinsics frame.
# The extrinsics frame is unambiguous — all five camera centres sit at
# y = -0.81..-0.94 m while spanning 6 m in x, so its Y axis is vertical (down).
# The rotation below was found by projecting the markers into all five cameras
# (brute force over the 24 axis-aligned rotations, both sessions, subject 2):
# it is the only one that lands the markers on the person, puts the five tripods
# 1.20-1.42 m above the mocap floor, and lays a mocap-floor grid onto the lab's
# floor tiles. Same matrix for Session0 and Session1.
R_VIDEO_FROM_MOCAP = np.array([[0, 0, -1],
                               [0, -1, 0],
                               [-1, 0, 0]], dtype=float)


def _session_dir(subject: int, trial: str, camera: str) -> Path:
    """The Session0/Session1 folder that holds this trial's videos."""
    for session in (0, 1):
        path = SRC / f"subject{subject}" / "VideoData" / f"Session{session}"
        if (path / camera / trial).is_dir():
            return path
    raise FileNotFoundError(f"subject{subject}/{trial}: no session has {camera}/{trial}")


def subject_trials(subject: int, camera: str) -> list[str]:
    """Every trial of ``subject`` that has a synced video AND a force file, sorted.

    Trial numbering differs per subject (subject2 has walkingTS4 but no walkingTS3), so
    ``--trials all`` discovers them instead of assuming a fixed list.
    """
    root = SRC / f"subject{subject}"
    found = []
    for session in (0, 1):
        cam = root / "VideoData" / f"Session{session}" / camera
        if not cam.is_dir():
            continue
        for d in cam.iterdir():
            if (d / f"{d.name}_syncdWithMocap.avi").is_file() and \
                    (root / "ForceData" / f"{d.name}_forces.mot").is_file():
                found.append(d.name)
    return sorted(set(found))


def _load_trc(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Marker names, times (s) and positions (T, M, 3) in metres."""
    lines = path.read_text().splitlines()
    names = [n for n in lines[3].split("\t")[2:] if n.strip()]
    rows = np.array([[float(v) for v in line.split("\t") if v.strip()]
                     for line in lines[6:] if line.strip()])
    xyz = rows[:, 2:2 + 3 * len(names)].reshape(len(rows), len(names), 3) / 1000.0
    return names, rows[:, 1], xyz


def _load_mot(path: Path) -> tuple[list[str], np.ndarray]:
    """Column names and data of an OpenSim ``.mot`` file."""
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "endheader") + 1
    columns = [c for c in lines[start].split("\t") if c.strip()]
    rows = np.array([[float(v) for v in line.split("\t") if v.strip()]
                     for line in lines[start + 1:] if line.strip()])
    return columns, rows


def _camera(session: Path, camera: str) -> dict:
    """Intrinsics + extrinsics of one camera (translation converted to metres)."""
    with open(session / camera / "cameraIntrinsicsExtrinsics.pickle", "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    return {"K": np.asarray(raw["intrinsicMat"], float),
            "dist": np.asarray(raw["distortion"], float).ravel(),
            "R": np.asarray(raw["rotation"], float),
            "t": np.asarray(raw["translation"], float).ravel() / 1000.0}


def _video_from_mocap(session: Path) -> tuple[np.ndarray, np.ndarray]:
    """``x_video = R (x_mocap - p)`` with ``p`` the yaml's mocap-frame offset."""
    doc = yaml.safe_load((session / "mocapToVideoTransform.yaml").read_text())
    p = np.asarray(doc["position_fromMocapOrigin_toVideoOrigin"], float)
    return R_VIDEO_FROM_MOCAP, -R_VIDEO_FROM_MOCAP @ p


def _encode_video(src: Path, dst: Path) -> None:
    """Re-encode the syncd .avi to h264, keeping every frame and the 60 fps rate."""
    subprocess.run([get_ffmpeg_exe(), "-y", "-v", "error", "-i", str(src),
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "-an", str(dst)], check=True)


def _encode_part(src: Path, dst: Path, start: int, end: int) -> None:
    """Re-encode frames ``[start, end)`` of the whole-trial mp4 (same codec settings)."""
    subprocess.run([get_ffmpeg_exe(), "-y", "-v", "error", "-i", str(src),
                    "-vf", f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS",
                    "-r", str(int(FPS)), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "-an", str(dst)], check=True)


def part_ranges(n_frames: int, max_frames: int = MAX_FRAMES) -> list[tuple[int, int]]:
    """Equal consecutive ``(start, end)`` frame ranges of at most ``max_frames`` each."""
    parts = -(-n_frames // max_frames)
    edges = np.linspace(0, n_frames, parts + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def _frame_count(path: Path) -> int:
    """Frames in ``path``, decoded twice — the mpeg4 .avi containers this dataset
    ships under-report their length, and every row index depends on this number."""
    reader = read_frames(str(path), pix_fmt="rgb24")
    next(reader)
    decoded = sum(1 for _ in reader)
    capture = cv2.VideoCapture(str(path))
    seen = 0
    while capture.read()[0]:
        seen += 1
    capture.release()
    if seen != decoded:
        raise RuntimeError(f"{path}: ffmpeg decodes {decoded} frames, OpenCV {seen}")
    return decoded


def _resample_markers(times: np.ndarray, xyz: np.ndarray, n_frames: int,
                      offset_s: float = 0.0) -> np.ndarray:
    """Markers at the video frames: linear interpolation of the 100 Hz TRC.

    Video frame ``i`` is mocap time ``i / FPS + offset_s`` (see ``sync_opencap.py``).
    """
    want = np.arange(n_frames) / FPS + offset_s
    flat = xyz.reshape(len(times), -1)
    out = np.stack([np.interp(want, times, flat[:, c]) for c in range(flat.shape[1])], 1)
    return out.reshape(n_frames, *xyz.shape[1:]).astype(np.float32)


def _resample_forces(columns: list[str], rows: np.ndarray, n_frames: int,
                     trial: str, offset_s: float = 0.0,
                     mass_kg: float = 0.0) -> tuple[np.ndarray, ...]:
    """Per-video-frame plate readings in :data:`LIMBS` order (hands stay zero).

    The bins are centred on the frame timestamps ``i / FPS + offset_s`` and the WRENCH
    is what gets averaged; see the module docstring for both.
    """
    centres = np.arange(n_frames) / FPS + offset_s
    edges = np.searchsorted(rows[:, 0], np.r_[centres - 0.5 / FPS, centres[-1] + 0.5 / FPS])
    empty = edges[:-1] == edges[1:]
    force = np.zeros((n_frames, 4, 3), np.float32)
    cop = np.full((n_frames, 4, 3), np.nan, np.float32)
    torque = np.full((n_frames, 4, 3), np.nan, np.float32)
    captured = np.ones((n_frames, 4), bool)
    captured[empty] = False                       # no plate sample belongs to this frame

    loaded_any = np.zeros(len(rows), bool)
    for limb, side in ((2, "L"), (3, "R")):
        pick = lambda kind: [columns.index(f"{side}_ground_{kind}{a}") for a in "xyz"]  # noqa: E731
        vec, pos, mom = rows[:, pick("force_v")], rows[:, pick("force_p")], rows[:, pick("torque_")]
        loaded = vec[:, 1] > LOADED_N
        loaded_any |= loaded
        # An unloaded plate's COP is stale, so it contributes no moment; its force is ~0,
        # which is what its true moment about the origin would be worth anyway.
        moment = np.where(loaded[:, None], origin_moment(pos, vec, mom), 0.0)
        y0 = float(np.median(pos[loaded, 1])) if loaded.any() else 0.0
        mean_force = np.zeros((n_frames, 3))
        mean_moment = np.zeros((n_frames, 3))
        for i in np.flatnonzero(~empty):
            span = slice(edges[i], edges[i + 1])
            mean_force[i] = vec[span].mean(axis=0)
            mean_moment[i] = moment[span].mean(axis=0)
        frame_cop, frame_torque, defined = cop_free_torque(mean_force, mean_moment, y0)
        force[:, limb] = mean_force
        cop[defined, limb] = frame_cop[defined]
        torque[defined, limb] = frame_torque[defined]

    if trial.startswith("DJ"):
        # The drop jump starts on a box no plate sees; the feet are only measured
        # from the landing on.
        if not loaded_any.any():
            raise SystemExit(f"{trial}: no plate sample is loaded — the landing is missing")
        first = int(np.floor((rows[np.argmax(loaded_any), 0] - offset_s) * FPS))
        captured[:max(first, 0), 2:] = False
    if trial.startswith("static"):
        # Standing still on both feet, yet a plate reads nothing: that foot is standing
        # off the plates (subject9), or both feet share one plate and the file reports
        # the whole weight on one foot with the CoP between the feet (subjects 2, 10) —
        # then neither foot's share is measured.
        vertical = force[:, 2:, 1]
        silent = vertical < LOADED_N
        shared = silent.any(1) & (vertical.sum(1) > 0.8 * mass_kg * 9.81)
        captured[:, 2:] &= ~silent
        captured[shared, 2:] = False
    return force, cop, torque, captured


def _project(points: np.ndarray, cam: dict, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Mocap points (M, 3) -> pixels (M, 2) through mocap -> video -> camera -> K."""
    uv, _ = cv2.projectPoints((points @ R.T + t).reshape(-1, 1, 3),
                              cv2.Rodrigues(cam["R"])[0], cam["t"].reshape(3, 1),
                              cam["K"], cam["dist"])
    return uv.reshape(-1, 2)


def _overlay(video: Path, out_path: Path, markers: np.ndarray, cam: dict,
             R: np.ndarray, t: np.ndarray) -> float:
    """Draw the projected markers on the first frames; return the fraction of all
    (frame, marker) projections that land inside the image."""
    reader = read_frames(str(video), pix_fmt="bgr24")
    width, height = next(reader)["size"]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (width, height))
    inside = []
    for i, raw in enumerate(reader):
        if i >= len(markers):
            break
        uv = _project(markers[i], cam, R, t)
        inside.append(((uv[:, 0] >= 0) & (uv[:, 0] < width)
                       & (uv[:, 1] >= 0) & (uv[:, 1] < height)).mean())
        if i < OVERLAY_FRAMES:
            frame = np.frombuffer(raw, np.uint8).reshape(height, width, 3).copy()
            for u, v in uv:
                cv2.circle(frame, (int(round(u)), int(round(v))), 3, (0, 0, 255), -1)
            writer.write(frame)
    writer.release()
    return float(np.mean(inside))


def _sync_offset_s(clip_dir: Path) -> float:
    """The clip's measured video-to-mocap offset in seconds (``sync_opencap.py``), else 0."""
    sync = clip_dir / "sync.json"
    if not sync.exists():
        return 0.0
    return float(json.loads(sync.read_text())["mocap_offset_frames"]) / FPS


def _import_clip(subject: int, trial: str, camera: str, *, resample_only: bool = False) -> None:
    session = _session_dir(subject, trial, camera)
    avi = session / camera / trial / f"{trial}_syncdWithMocap.avi"
    # The whole trial is encoded once and counted strictly (the .avi's own timestamps are
    # irregular, so parts are cut from this constant-rate mp4, never from the .avi).
    whole = DST / "_whole" / f"subject{subject}_{trial}_{camera.lower()}.mp4"
    whole.parent.mkdir(parents=True, exist_ok=True)
    if not whole.exists():
        _encode_video(avi, whole)
    n_total = _frame_count(whole)

    meta_src = yaml.safe_load((SRC / f"subject{subject}" / "sessionMetadata.yaml").read_text())
    names, times, xyz = _load_trc(SRC / f"subject{subject}" / "MarkerData" / "Mocap" / f"{trial}.trc")
    columns, rows = _load_mot(SRC / f"subject{subject}" / "ForceData" / f"{trial}_forces.mot")

    ranges = part_ranges(n_total)
    for k, (start, end) in enumerate(ranges):
        tag = trial if len(ranges) == 1 else f"{trial}p{k + 1}"
        # Each part carries its own measured offset; the trial's arrays are resampled per
        # part at that offset (cheap) and sliced.
        offset_s = _sync_offset_s(DST / f"subject{subject}_{tag}_{camera.lower()}")
        markers_all = _resample_markers(times, xyz, n_total, offset_s)
        force_all, cop_all, torque_all, captured_all = _resample_forces(
            columns, rows, n_total, trial, offset_s, meta_src["mass_kg"])
        _write_clip(subject, trial, tag, camera, session, whole, (start, end), meta_src, names,
                    markers_all[start:end], force_all[start:end], cop_all[start:end],
                    torque_all[start:end], captured_all[start:end], whole_trial=len(ranges) == 1,
                    offset_s=offset_s, resample_only=resample_only)


def _write_clip(subject: int, trial: str, tag: str, camera: str, session: Path, whole: Path,
                frames: tuple[int, int], meta_src: dict, names: list[str], markers: np.ndarray,
                force: np.ndarray, cop: np.ndarray, torque: np.ndarray,
                captured: np.ndarray, *, whole_trial: bool, offset_s: float = 0.0,
                resample_only: bool = False) -> None:
    """Write one clip folder for trial frames ``[start, end)`` (the whole trial, or a part).

    ``resample_only`` rewrites the plate and marker arrays (and the sync fields of
    ``meta.json``) of an existing clip and leaves its video and overlay alone.
    """
    start, end = frames
    name = f"subject{subject}_{tag}_{camera.lower()}"
    clip_dir = DST / name
    clip_dir.mkdir(parents=True, exist_ok=True)

    video = clip_dir / "video.mp4"
    if resample_only:
        n_frames = int(json.loads((clip_dir / "meta.json").read_text())["n_frames"])
    else:
        if not video.exists():
            if whole_trial:
                shutil.copyfile(whole, video)
            else:
                _encode_part(whole, video, start, end)
        n_frames = _frame_count(video)
    if n_frames != end - start:
        raise RuntimeError(f"{name}: video has {n_frames} frames, expected {end - start}")

    note = ("the chair is not instrumented — the seat reaction is missing from the "
            "plates while the subject is seated" if trial.startswith("STS") else "")
    np.savez(clip_dir / "forces.npz",
             force=force, cop=cop, free_torque=torque, captured=captured,
             fps=np.float32(FPS), mocap_fps=np.float32(MOCAP_FPS),
             force_fps=np.float32(FORCE_FPS), limbs=np.array(LIMBS),
             n_frames=np.int64(n_frames), frame="opensim_ground_y_up",
             source=f"opencap/subject{subject}/{trial}",
             mocap_offset_s=np.float32(offset_s),
             mass_kg=np.float32(meta_src["mass_kg"]),
             height_m=np.float32(meta_src["height_m"]), note=note)
    np.savez(clip_dir / "markers.npz", markers=markers, names=np.array(names),
             fps=np.float32(FPS), mocap_offset_s=np.float32(offset_s))

    sync_fields = {
        "mocap_offset_s": offset_s,
        "sync_note": ("row i of forces.npz / markers.npz is mocap time i / fps + "
                      "mocap_offset_s, the offset measured by scripts/sync_opencap.py "
                      "(sync.json); the shipped 'synced' videos are off by up to 0.7 s on "
                      "the walking trials"),
    }
    if resample_only:
        meta = json.loads((clip_dir / "meta.json").read_text())
        meta.update(sync_fields)
        (clip_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        feet = captured[:, 2:]
        print(f"  {name}: resampled at {offset_s * FPS:+.2f} frames, foot-frames captured "
              f"{feet.mean():.2f}, peak |F| {np.linalg.norm(force, axis=2).max():.0f} N")
        return

    cam = _camera(session, camera)
    R_vm, t_vm = _video_from_mocap(session)
    R_cm = cam["R"] @ R_vm
    t_cm = cam["R"] @ t_vm + cam["t"]
    in_image = _overlay(video, clip_dir / "marker_overlay.mp4", markers, cam, R_vm, t_vm)

    (clip_dir / "meta.json").write_text(json.dumps({
        **sync_fields,
        "subject": f"subject{subject}", "trial": trial,
        "session": session.name, "camera": camera,
        "n_frames": n_frames, "fps": FPS, "width": 720, "height": 1280,
        "frame_offset": start,
        "part_note": ("frame i of this clip is frame frame_offset + i of the trial; long "
                      "trials are imported in parts (see MAX_FRAMES)" if start or tag != trial
                      else ""),
        "mass_kg": meta_src["mass_kg"], "height_m": meta_src["height_m"],
        "sex": meta_src["sex"],
        "K": cam["K"].tolist(), "dist": cam["dist"].tolist(),
        "R_cam_from_video": cam["R"].tolist(), "t_cam_from_video": cam["t"].tolist(),
        "R_video_from_mocap": R_vm.tolist(), "t_video_from_mocap": t_vm.tolist(),
        "R_cam_from_mocap": R_cm.tolist(), "t_cam_from_mocap": t_cm.tolist(),
        "reprojection_check": {"markers_in_image_frac": round(in_image, 4),
                               "overlay": "marker_overlay.mp4"},
        "reprojection_note":
            "Verified by projecting the TRC markers onto the syncd video (all five "
            "cameras, both sessions): x_cam = R_cam_from_video @ x_video + "
            "t_cam_from_video (pickle 'translation' is mm), x_video = "
            "R_video_from_mocap @ x_mocap + t_video_from_mocap with "
            "t_video_from_mocap = -R_video_from_mocap @ "
            "position_fromMocapOrigin_toVideoOrigin, and K + the 5-term OpenCV "
            "distortion applied to the 720x1280 portrait frame as stored (no "
            "rotation; the pickle's imageSize [1280, 720] is (H, W)). Video frame i "
            "is mocap time i/60 s. R_video_from_mocap is NOT the yaml's rotation — "
            "see R_VIDEO_FROM_MOCAP in scripts/import_opencap.py. The dataset's own "
            "translation leaves a residual in-floor-plane offset of ~0.15-0.3 m; "
            "it is kept as shipped so the force comparison stays fit-free.",
    }, indent=2))

    feet = captured[:, 2:]
    print(f"  {name}: {n_frames} frames, foot-frames captured {feet.mean():.2f} "
          f"(both feet {feet.all(axis=1).mean():.2f}), "
          f"peak |F| {np.linalg.norm(force, axis=2).max():.0f} N, "
          f"markers in image {in_image:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subjects", default=",".join(str(s) for s in SUBJECTS),
                        help="comma-separated subject numbers")
    parser.add_argument("--trials", default=",".join(TRIALS),
                        help="comma-separated trial names, or 'all' for every trial the subject "
                             "has a synced video and a force file for")
    parser.add_argument("--camera", default=CAMERA, help="camera folder name")
    parser.add_argument("--skip-existing", action="store_true",
                        help="leave clips that already have forces.npz alone")
    parser.add_argument("--resample-only", action="store_true",
                        help="rewrite forces.npz / markers.npz of already imported clips at the "
                             "offsets in their sync.json (scripts/sync_opencap.py); no video work")
    args = parser.parse_args()

    subjects = [int(s) for s in args.subjects.split(",") if s.strip()]
    plan = [(subject, trial) for subject in subjects
            for trial in (subject_trials(subject, args.camera) if args.trials == "all"
                          else [t.strip() for t in args.trials.split(",") if t.strip()])]
    if args.skip_existing:
        plan = [(s, t) for s, t in plan
                if not any((DST / f"subject{s}_{t}{p}_{args.camera.lower()}" / "forces.npz").exists()
                           for p in ("", "p1"))]

    print(f"[import_opencap] {len(plan)} clip(s) → {DST}")
    for subject, trial in plan:
        _import_clip(subject, trial, args.camera, resample_only=args.resample_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
