"""Import the ETH multi-view force recordings (the "peter" dataset) into ``peter/``.

Source tree: ``/data3/rikhat.akizhanov/climb_forces/data/eth``. Sessions ``1`` and
``2`` are campus-board recordings (two synced views + force plates in a ``.mat``);
session ``4`` is a climbing wall (three views + force ``.csv``). Calibration
sequences are skipped everywhere.

Output layout (one folder per clip):

    peter/
      campus_board/<clip>/   cam_left.mp4  cam_right.mp4  forces.npz  contacts.npz
      climb_wall/trial_NN/   cam_left.mp4  cam_right.mp4  cam_3.mp4  forces.csv
      _force_checks/<clip>.mp4        # written by --force-checks

Campus clips are trimmed to the annotated ``[frame_start, frame_end)`` from the
per-sequence ``config.yaml`` and split into equal parts when longer than 500
frames (``<seq>_partN``). ``forces.npz`` holds the matching slice of the
calibrated plate readings at the native sensor rate:

    forces      (T, 4) float32  Newtons, columns [left_hand, right_hand, left_foot, right_foot]
    fps         ()              force sample rate (1000 Hz)
    limbs       (4,) str        the column names above
    video_fps   ()              30.0
    n_frames    ()              frames in the clip videos
    frame_range (2,)            [start, end) in the source video
    source      ()              "eth/<session>/<sequence>"

Session 1 has hands on the plates only (left = plates 1+3, right = plates 0+2,
feet stay zero); session 2 has one plate per limb (3=LH, 2=RH, 1=LF, 0=RF).
Video/force sync follows the original pipeline: the first sample where plate 3
exceeds 0.5 N is matched to the annotated ``force_indicator_frame``. That first
crossing is a deliberate indicator tap and can be tiny (in ``2/02_gianna_hanging``
just a 1.2 N crosstalk blip, verified against the video events), so the 0.5 N
threshold must stay this sensitive. One recording deviates physically (see
:data:`SEQUENCE_OVERRIDES`): in ``2/02_gianna_hanging`` she hangs from the
LOWER instrumented rung, so her hands load plates 1/0 and the feet stay free.

``contacts.npz`` is derived from ``forces.npz`` the way the original pipeline
did (``ETHData.get_contacts``): each limb's force is linearly resampled to one
value per video frame, then thresholded at 50 N:

    contacts    (n_frames, 4) bool  same limb columns as forces
    fps         ()                  30.0 (one row per video frame)
    limbs, threshold, source        provenance

Climbing-wall trials are converted full length (video stream copied, no
re-encode) and the matching ``trial_NN.csv`` is copied as ``forces.csv`` — no
trim config or force schema exists for session 4 yet.
"""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy.io
import yaml
from imageio_ffmpeg import get_ffmpeg_exe

_REPO_ROOT = Path(__file__).resolve().parent.parent

MAX_FRAMES = 500
CONTACT_THRESHOLD_N = 50.0
LIMBS = ["left_hand", "right_hand", "left_foot", "right_foot"]
# Calibrated plate channel -> limb column, per session (see module docstring).
PLATES_TO_LIMBS = {"1": [[1, 3], [0, 2], [], []], "2": [[3], [2], [1], [0]]}
SYNC_CHANNEL = 3
# Recordings whose physical setup deviates from the session scheme.
SEQUENCE_OVERRIDES = {
    # Hangs by the hands from the lower rung (plates 1/0), feet in the air.
    "2/02_gianna_hanging": {"limb_channels": [[1], [0], [], []]},
}
CAMPUS_SESSIONS = ("1", "2")
WALL_SESSION = "4"
WALL_VIEWS = ("cam_left", "cam_right", "cam_3")


def run_ffmpeg(args: list[str]) -> None:
    """Run the bundled ffmpeg binary, raising with its stderr tail on failure."""
    proc = subprocess.run(
        [get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[-500:]}")


def trim_to_mp4(src: Path, dst: Path, frame_start: int, frame_end: int, fps: float) -> None:
    """Re-encode ``src`` keeping decoded frames ``[frame_start, frame_end)``.

    The OBS mkvs declare no stream framerate, so timestamps are rebuilt as exact
    CFR (``setpts=N/fps/TB``) — without this ffmpeg assumes 25 fps and drops frames.
    """
    vf = f"trim=start_frame={frame_start}:end_frame={frame_end},setpts=N/{fps}/TB"
    run_ffmpeg(["-i", str(src), "-vf", vf, "-r", str(fps), "-an", "-c:v", "libx264",
                "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(dst)])


def copy_to_mp4(src: Path, dst: Path) -> None:
    """Repackage the h264 stream of ``src`` into an mp4 without re-encoding."""
    run_ffmpeg(["-i", str(src), "-map", "0:v:0", "-c:v", "copy", "-an", str(dst)])


def count_frames(video: Path) -> int:
    """Count frames by decoding (container metadata over-reports on these mkvs)."""
    cap = cv2.VideoCapture(str(video))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def find_view(seq_dir: Path, view: str) -> Path:
    """Return the single ``<view>_*.mkv`` inside ``seq_dir``."""
    matches = sorted(seq_dir.glob(f"{view}_*.mkv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {view}_*.mkv in {seq_dir}, found {len(matches)}")
    return matches[0]


def load_campus_forces(mat_path: Path, session: str) -> np.ndarray:
    """Load calibrated plate readings as ``(T, 4)`` raw channels."""
    mat = scipy.io.loadmat(str(mat_path))
    if session == "1":
        return np.ascontiguousarray(mat["data_calib"].swapaxes(0, 1), dtype=np.float32)
    readings = []
    tempo = mat["tempo"]
    for i in range(tempo.shape[1]):
        force = tempo[0, i]["CalibratedSensorReading"][0, 0]
        if force.shape != (4, 1):
            force = np.zeros((4, 1))
        readings.append(force[:, 0])
    return np.stack(readings).astype(np.float32)


def plates_to_limbs(raw: np.ndarray, limb_channels: list[list[int]]) -> np.ndarray:
    """Map raw plate channels ``(T, 4)`` to limb columns ``(T, 4)``."""
    out = np.zeros_like(raw)
    for column, channels in enumerate(limb_channels):
        if channels:
            out[:, column] = raw[:, channels].sum(axis=1)
    return out


def split_bounds(frame_start: int, frame_end: int) -> list[tuple[int, int]]:
    """Split ``[frame_start, frame_end)`` into equal parts of at most MAX_FRAMES."""
    n_frames = frame_end - frame_start
    n_parts = math.ceil(n_frames / MAX_FRAMES)
    edges = [frame_start + round(i * n_frames / n_parts) for i in range(n_parts + 1)]
    return list(zip(edges[:-1], edges[1:]))


def import_campus_sequence(source: Path, session: str, seq: str, out_root: Path,
                           skip_existing: bool) -> list[str]:
    """Import one campus-board sequence; returns the clip names written."""
    seq_dir = source / session / "videos" / seq
    config = yaml.safe_load((source / session / "data" / seq / "config.yaml").read_text())
    video_fps = config["fps"]
    force_fps = config["force_sensor_fps"]

    override = SEQUENCE_OVERRIDES.get(f"{session}/{seq}", {})
    raw = load_campus_forces(seq_dir / f"{seq}.mat", session)
    forces = plates_to_limbs(raw, override.get("limb_channels", PLATES_TO_LIMBS[session]))
    # Sync exactly like the original pipeline (lib/contact.py): the first sample
    # where plate 3 crosses 0.5 N (the indicator tap) happened at the annotated
    # force_indicator_frame.
    sensor_impact = int(np.argmax(raw[:, SYNC_CHANNEL] > 0.5))
    lag = sensor_impact - config["force_indicator_frame"] / video_fps * force_fps

    clips = []
    parts = split_bounds(config["frame_start"], config["frame_end"])
    for part_index, (f0, f1) in enumerate(parts):
        clip = seq if len(parts) == 1 else f"{seq}_part{part_index + 1}"
        clip_dir = out_root / "campus_board" / clip
        done = [clip_dir / "cam_left.mp4", clip_dir / "cam_right.mp4", clip_dir / "forces.npz"]
        if skip_existing and all(p.exists() for p in done):
            clips.append(clip)
            continue
        clip_dir.mkdir(parents=True, exist_ok=True)

        for view in ("cam_left", "cam_right"):
            trim_to_mp4(find_view(seq_dir, view), clip_dir / f"{view}.mp4", f0, f1, video_fps)
            written = count_frames(clip_dir / f"{view}.mp4")
            if written != f1 - f0:
                raise RuntimeError(f"{clip}/{view}: wrote {written} frames, expected {f1 - f0}")

        s0 = int(f0 / video_fps * force_fps + lag)
        s1 = int(f1 / video_fps * force_fps + lag)
        if s0 < 0 or s1 > forces.shape[0]:
            raise RuntimeError(f"{clip}: force slice [{s0}, {s1}) outside 0..{forces.shape[0]}")
        np.savez(
            clip_dir / "forces.npz",
            forces=forces[s0:s1],
            fps=np.float32(force_fps),
            limbs=np.array(LIMBS),
            video_fps=np.float32(video_fps),
            n_frames=np.int64(f1 - f0),
            frame_range=np.array([f0, f1], dtype=np.int64),
            source=np.str_(f"eth/{session}/{seq}"),
        )
        clips.append(clip)
    return clips


def import_wall_trial(source: Path, trial_dir: Path, out_root: Path,
                      skip_existing: bool) -> str:
    """Import one climbing-wall trial full length; returns the clip name."""
    number = int(trial_dir.name.removeprefix("trial_"))
    clip = f"trial_{number:02d}"
    clip_dir = out_root / "climb_wall" / clip
    done = [clip_dir / f"{v}.mp4" for v in WALL_VIEWS] + [clip_dir / "forces.csv"]
    if not (skip_existing and all(p.exists() for p in done)):
        clip_dir.mkdir(parents=True, exist_ok=True)
        for view in WALL_VIEWS:
            copy_to_mp4(find_view(trial_dir, view), clip_dir / f"{view}.mp4")
        shutil.copyfile(source / WALL_SESSION / "data" / f"{clip}.csv", clip_dir / "forces.csv")
    return clip


def write_contacts(clip_dir: Path) -> None:
    """Derive ``contacts.npz`` from ``clip_dir/forces.npz``.

    Mirrors the original ``ETHData``: per limb, linearly resample the force to
    one value per video frame, then threshold at :data:`CONTACT_THRESHOLD_N`.
    """
    data = np.load(clip_dir / "forces.npz")
    forces, n_frames = data["forces"], int(data["n_frames"])
    frame_time = np.linspace(0.0, 1.0, n_frames)
    sample_time = np.linspace(0.0, 1.0, forces.shape[0])
    per_frame = np.stack([np.interp(frame_time, sample_time, forces[:, c])
                          for c in range(forces.shape[1])], axis=1)
    np.savez(
        clip_dir / "contacts.npz",
        contacts=per_frame > CONTACT_THRESHOLD_N,
        fps=data["video_fps"],
        limbs=data["limbs"],
        threshold=np.float32(CONTACT_THRESHOLD_N),
        source=data["source"],
    )


def render_force_check(out_root: Path, clip: str) -> Path:
    """Render <clip video | force curves + moving cursor> side by side for review."""
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clip_dir = out_root / "campus_board" / clip
    data = np.load(clip_dir / "forces.npz")
    forces, force_fps, video_fps = data["forces"], float(data["fps"]), float(data["video_fps"])
    time_axis = np.arange(forces.shape[0]) / force_fps

    panel_w, panel_h = 640, 540
    fig, ax = plt.subplots(figsize=(panel_w / 100, panel_h / 100), dpi=100)
    for column, limb in enumerate(LIMBS):
        ax.plot(time_axis, forces[:, column], label=limb, linewidth=1.0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("force [N]")
    ax.set_title(clip)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    panel = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    to_pixel = lambda t: int(round(ax.transData.transform((t, 0))[0]))  # noqa: E731
    plt.close(fig)

    check_dir = out_root / "_force_checks"
    check_dir.mkdir(exist_ok=True)
    out_path = check_dir / f"{clip}.mp4"
    cap = cv2.VideoCapture(str(clip_dir / "cam_left.mp4"))
    writer = imageio.get_writer(out_path, fps=video_fps, codec="libx264", quality=7)
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (960, panel_h))[..., ::-1]  # BGR -> RGB
        cursor = panel.copy()
        x = np.clip(to_pixel(frame_index / video_fps), 0, panel_w - 1)
        cursor[:, x] = (255, 0, 0)
        writer.append_data(np.hstack([frame, cursor]))
        frame_index += 1
    cap.release()
    writer.close()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path,
                        default=Path("/data3/rikhat.akizhanov/climb_forces/data/eth"))
    parser.add_argument("--dest", type=Path, default=_REPO_ROOT / "peter")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip clips whose outputs already exist")
    parser.add_argument("--force-checks", nargs="*", default=[], metavar="CLIP",
                        help="campus_board clip names to render force-check videos for")
    args = parser.parse_args()

    failures = []
    for session in CAMPUS_SESSIONS:
        for seq_dir in sorted((args.source / session / "videos").iterdir()):
            if not seq_dir.is_dir() or "calibration" in seq_dir.name:
                continue
            try:
                clips = import_campus_sequence(args.source, session, seq_dir.name,
                                               args.dest, args.skip_existing)
                print(f"[campus_board] {session}/{seq_dir.name} -> {', '.join(clips)}")
            except Exception as exc:  # noqa: BLE001 — one bad sequence must not kill the batch
                failures.append(f"{session}/{seq_dir.name}: {exc}")
                print(f"[campus_board] {session}/{seq_dir.name} FAILED: {exc}", file=sys.stderr)

    for clip_dir in sorted((args.dest / "campus_board").iterdir()):
        if (clip_dir / "forces.npz").exists():
            write_contacts(clip_dir)

    for trial_dir in sorted((args.source / WALL_SESSION / "videos").iterdir()):
        if not trial_dir.is_dir() or "calibration" in trial_dir.name:
            continue
        try:
            clip = import_wall_trial(args.source, trial_dir, args.dest, args.skip_existing)
            print(f"[climb_wall] {trial_dir.name} -> {clip}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"4/{trial_dir.name}: {exc}")
            print(f"[climb_wall] {trial_dir.name} FAILED: {exc}", file=sys.stderr)

    for clip in args.force_checks:
        print(f"[force_check] {clip} -> {render_force_check(args.dest, clip)}")

    if failures:
        print(f"\n{len(failures)} item(s) failed:\n  " + "\n  ".join(failures), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
