"""Measure each imported OpenCap clip's video-to-mocap time offset, from the video alone.

The dataset ships ``<trial>_syncdWithMocap.avi`` as if video frame ``i`` were mocap time
``i / 60``. Measured against the lab's own markers, that holds for the drop jumps and
most squats but NOT for the walking trials: their videos run up to 40 frames (0.67 s)
behind the mocap, by a different amount in every trial. Any force comparison that
trusts the file name then scores our forces against plate readings from a different
moment of the gait cycle.

The offset is measured without touching a force: the mocap markers are projected into
the clip's camera through the dataset's own calibration and compared with the 2D
keypoints Sapiens detected on the video, at every candidate shift. At each shift the
median 2D offset is removed first — the calibration's translation residual, and a
walking subject turns a position offset into an apparent time offset — so only the
shape of the motion decides. The minimum is refined to a fraction of a frame by a
parabola through its neighbours. Where the motion is too small to time (a static
trial), the curve is flat and the offset stays at zero.

``--joints <run>`` measures against our own reconstruction instead of the keypoints:
the 3D joints of ``out/<clip>/<run>/kindyn_1.npz`` against the hip-centre, knee, ankle
and shoulder markers, translation-free, in metres. It exists because the keypoint scan
is nearly flat on the side-view walking clips of subjects 4, 7 and 8 (the best shift
is 17-20 px, barely below the shipped alignment) and lands up to 30 frames off there,
while the 3D curve keeps a sharp minimum. Walking is periodic, so an alias one stride
away (about 58 frames) scores almost as well; the joint scan therefore searches
``JOINT_SHIFTS`` only — every walking offset in this dataset is positive, the other
activities sit within a few frames of zero. A clip flat over that range is rescanned
over the full ±``MAX_SHIFT`` window (a static trial stays flat and keeps zero).

Writes ``processed/<clip>/sync.json``::

    mocap_offset_frames   video frame i is mocap time (i + this) / 60  (this clip's frames)
    px_at_best / px_at_zero   median 2D error after / before the shift, pixels
    halves                    the same offset measured on each half of the clip alone —
                              equal halves mean one constant shift, not a rate drift
    uncertain                 the halves disagree by more than HALVES_TOLERANCE frames
                              (a keypoint failure or a subject partly out of frame)

``datasets/opencap/import_opencap.py --resample-only`` then re-reads the plate files at the
corrected times.

Usage::

    uv run --no-sync python datasets/opencap/sync_opencap.py            # every imported clip
    uv run --no-sync python datasets/opencap/sync_opencap.py --clips subject2_walking1_cam1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


from datasets.opencap.import_opencap import DST, FPS, SRC, _load_trc, _resample_markers

#: Sapiens (Goliath-308) keypoint index ↔ mocap marker, for the joints both can see.
PAIRS = ((5, "L_Shoulder"), (6, "R_Shoulder"), (9, "L_HJC"), (10, "R_HJC"),
         (11, "L_knee"), (12, "r_knee"), (13, "L_ankle"), (14, "r_ankle"))

#: Candidate shifts, video frames either way (1 s at 60 fps).
MAX_SHIFT = 60

#: Keypoints under this Sapiens score are not trusted for timing.
MIN_SCORE = 0.5

#: A curve shallower than this (pixels between the best shift and zero) is flat: the
#: motion is too small to time, and the shipped alignment is kept.
MIN_DEPTH_PX = 2.0

#: The two halves of a clip must agree on the offset to within this many frames.
HALVES_TOLERANCE = 5.0

#: Our joint ↔ mocap marker pairs for ``--joints`` (the marker is the joint centre).
JOINT_PAIRS = (("left_hip", "L_HJC"), ("right_hip", "R_HJC"), ("left_knee", "L_knee"),
               ("right_knee", "r_knee"), ("left_ankle", "L_ankle"), ("right_ankle", "r_ankle"),
               ("left_shoulder", "L_Shoulder"), ("right_shoulder", "R_Shoulder"))

#: Candidate mocap offsets (frames) for the joint scan, see the module docstring.
JOINT_SHIFTS = range(-15, 61)

#: A joint curve shallower than this (metres) is flat: the shipped alignment is kept.
MIN_DEPTH_M = 0.01


def error_curve(projected: np.ndarray, keypoints: np.ndarray, scores: np.ndarray,
                max_shift: int = MAX_SHIFT) -> dict[int, float]:
    """Median translation-free 2D error at each shift ``s`` (``proj[i]`` vs ``kp[i + s]``).

    :param projected: ``(N, J, 2)`` mocap markers in pixels at ``t = i / 60``.
    :param keypoints: ``(N, J, 2)`` detected keypoints, same joint order.
    :param scores: ``(N, J)`` detection scores.
    """
    n = len(projected)
    curve = {}
    for s in range(-max_shift, max_shift + 1):
        i = np.arange(max(0, -s), min(n, n - s))
        if len(i) < n // 2:
            continue
        ok = scores[i + s] > MIN_SCORE
        residual = (projected[i] - keypoints[i + s])[ok]
        if len(residual) < 10:
            continue
        residual = residual - np.median(residual, 0)
        curve[s] = float(np.median(np.linalg.norm(residual, axis=1)))
    return curve


def best_shift(curve: dict[int, float]) -> tuple[float, float]:
    """``(shift, error)`` at the minimum, refined to a fraction of a frame."""
    s0 = min(curve, key=curve.get)
    if s0 - 1 in curve and s0 + 1 in curve:
        left, mid, right = curve[s0 - 1], curve[s0], curve[s0 + 1]
        denominator = left - 2 * mid + right
        if denominator > 0:
            s0 = s0 + 0.5 * (left - right) / denominator
    return float(s0), float(curve[round(s0)])


def measure(clip: Path) -> dict:
    """The sync record for one imported clip folder."""
    meta = json.loads((clip / "meta.json").read_text())
    pose = np.load(DST / "out" / clip.name / "sapiens" / "pose.npz")
    names, times, xyz = _load_trc(SRC / meta["subject"] / "MarkerData" / "Mocap"
                                  / f"{meta['trial']}.trc")
    start = int(meta.get("frame_offset", 0))
    n = min(int(meta["n_frames"]), pose["keypoints"].shape[1])
    markers = _resample_markers(times, xyz, start + n)[start:]
    picked = markers[:, [names.index(m) for _, m in PAIRS]].astype(np.float64)
    uv, _ = cv2.projectPoints(picked.reshape(-1, 1, 3),
                              cv2.Rodrigues(np.array(meta["R_cam_from_mocap"]))[0],
                              np.array(meta["t_cam_from_mocap"], float).reshape(3, 1),
                              np.array(meta["K"], float), np.array(meta["dist"], float))
    projected = uv.reshape(n, len(PAIRS), 2)
    keypoints = pose["keypoints"][0, :n][:, [k for k, _ in PAIRS]]
    scores = pose["keypoint_scores"][0, :n][:, [k for k, _ in PAIRS]]

    curve = error_curve(projected, keypoints, scores)
    shift, at_best = best_shift(curve)
    at_zero = curve.get(0, float("nan"))
    flat = not (at_zero - at_best >= MIN_DEPTH_PX)
    halves = []
    for sl in (slice(0, n // 2), slice(n // 2, n)):
        half = error_curve(projected[sl], keypoints[sl], scores[sl])
        halves.append(round(-best_shift(half)[0], 2) if half else None)
    uncertain = (not flat) and (None in halves or abs(halves[0] - halves[1]) > HALVES_TOLERANCE)
    return {
        "mocap_offset_frames": 0.0 if flat else round(-shift, 2),
        "px_at_best": round(at_best, 1), "px_at_zero": round(at_zero, 1),
        "flat": flat, "halves": halves, "uncertain": uncertain,
        "method": "mocap markers projected through meta.json vs Sapiens keypoints, "
                  "translation-free median 2D error over ±%d frames" % MAX_SHIFT,
    }


def joint_curve(ours: np.ndarray, markers: np.ndarray, shifts=JOINT_SHIFTS) -> dict[int, float]:
    """Median translation-free 3D error at each mocap offset ``s`` (frames).

    :param ours: ``(N, J, 3)`` our joints at video frame ``i``, world metres.
    :param markers: ``(N + 2·MAX_SHIFT, J, 3)`` the markers in the same world, row ``k``
        at mocap time ``(k − MAX_SHIFT) / 60``, so offset ``s`` reads rows ``MAX_SHIFT + s``.
    """
    n = len(ours)
    curve = {}
    for s in shifts:
        residual = ours - markers[MAX_SHIFT + s: MAX_SHIFT + s + n]
        residual = residual - np.nanmedian(residual.reshape(-1, 3), 0)
        curve[s] = float(np.nanmedian(np.linalg.norm(residual, axis=-1)))
    return curve


def measure_joints(clip: Path, run: str) -> dict:
    """The sync record for one clip from our reconstructed joints (``--joints``)."""
    from datasets.opencap.opencap_sensor_forces import world_from_mocap

    meta = json.loads((clip / "meta.json").read_text())
    body = np.load(DST / "out" / clip.name / run / "kindyn_1.npz", allow_pickle=True)
    joint_names = list(body["joint_names"])
    ours = body["joints_world"][0][:, [joint_names.index(a) for a, _ in JOINT_PAIRS]]
    ours = np.where(body["valid_mask"][0].astype(bool)[:, None, None], ours, np.nan)
    n = len(ours)
    names, times, xyz = _load_trc(SRC / meta["subject"] / "MarkerData" / "Mocap"
                                  / f"{meta['trial']}.trc")
    start = int(meta.get("frame_offset", 0))
    markers = _resample_markers(times, xyz, start + n + 2 * MAX_SHIFT,
                                offset_s=-MAX_SHIFT / FPS)[start:]
    rotation, translation = world_from_mocap(clip)
    markers = markers[:, [names.index(m) for _, m in JOINT_PAIRS]] @ rotation.T + translation

    curve = joint_curve(ours, markers)
    offset, at_best = best_shift(curve)
    at_zero = curve[0]
    flat = not (at_zero - at_best >= MIN_DEPTH_M)
    if flat:
        # Nothing in the preferred range: a static trial (truly flat), or a walking clip
        # whose one clear minimum lies before it. Take the full window's minimum if it has one.
        curve = joint_curve(ours, markers, range(-MAX_SHIFT, MAX_SHIFT + 1))
        offset, at_best = best_shift(curve)
        flat = not (at_zero - at_best >= MIN_DEPTH_M)
    halves = []
    for sl in (slice(0, n // 2), slice(n // 2, n)):
        half = joint_curve(ours[sl], markers[sl.start: sl.stop + 2 * MAX_SHIFT])
        halves.append(round(best_shift(half)[0], 2))
    uncertain = (not flat) and abs(halves[0] - halves[1]) > HALVES_TOLERANCE
    return {
        "mocap_offset_frames": 0.0 if flat else round(offset, 2),
        "m_at_best": round(at_best, 4), "m_at_zero": round(at_zero, 4),
        "flat": flat, "halves": halves, "uncertain": uncertain,
        "method": f"our {run} joints vs mocap joint-centre markers, translation-free "
                  f"median 3D error over offsets {JOINT_SHIFTS.start}..{JOINT_SHIFTS.stop - 1}",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", default="", help="comma-separated clip names (default: all)")
    ap.add_argument("--joints", default="", metavar="RUN",
                    help="measure with our reconstructed joints of out/<clip>/RUN/kindyn_1.npz "
                         "instead of the Sapiens keypoints (see the module docstring)")
    args = ap.parse_args(argv)
    clips = ([DST / c for c in args.clips.split(",") if c] if args.clips else
             sorted(c for c in DST.iterdir() if (c / "meta.json").exists()))
    for clip in clips:
        record = measure_joints(clip, args.joints) if args.joints else measure(clip)
        (clip / "sync.json").write_text(json.dumps(record, indent=2))
        error = (f"{record['m_at_zero'] * 100:5.1f} -> {record['m_at_best'] * 100:5.1f} cm"
                 if args.joints else f"px {record['px_at_zero']:5.1f} -> {record['px_at_best']:5.1f}")
        print(f"{clip.name:32s} offset {record['mocap_offset_frames']:+6.2f} fr  {error}  "
              f"halves {record['halves']}{'  (flat, kept 0)' if record['flat'] else ''}"
              f"{'  UNCERTAIN' if record['uncertain'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
