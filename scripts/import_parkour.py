"""Import the LAAS Parkour dataset (Li et al. CVPR'19) into ``data/Parkour-dataset/processed/``.

Source tree: ``/home/rikhat.akizhanov/better/data/Parkour-dataset`` (the
``zongmianli/Parkour-dataset`` repo + its pre-extracted ``frames/``). 28 clips of
four parkour actions (kong vault ``kv``, muscle-up ``mu``, pull-up ``pu``,
safety vault ``sv``), one subject each, static camera, 720x576. Ground truth is
Vicon MoCap + force plates: 16 3D joints and four 6D contact wrenches (feet on
floor plates, hands on an instrumented bar).

Output layout (one folder per clip):

    data/Parkour-dataset/processed/<clip>/  video.mp4  forces.npz  contacts.npz  joints_3d.npz

Sync: the dataset bakes a 3-frame video/MoCap offset — the shipped videos start
3 frames late and the last 3 GT rows are dropped, so after import **row i of
every array is video frame i**. The shipped mp4 containers claim 10 fps but the
true rate is 25 fps (= the GT rate), so ``video.mp4`` is re-encoded from the
lossless PNG frames at 25 fps — the container rate is what sets dt downstream.

``forces.npz`` (all arrays row-aligned with the video):

    forces    (N, 4, 6) float32  [fx fy fz mx my mz] per limb. Newtons / N*m.
                                 Reaction on the person, world axes (+z up).
                                 Moments: feet about the ankle joint, hands
                                 about the bar sensor frame (rotated only).
    captured  (N, 4)   bool      True where the rig actually measured this limb
                                 (False spans: the foot landed off the plates —
                                 the person may still be in contact there)
    fps       ()                 25.0 (force rows are at video rate)
    limbs     (4,) str           [left_hand, right_hand, left_foot, right_foot]
    n_frames  ()                 frames in video.mp4
    source    ()                 "parkour/<clip>"

``contacts.npz``: ``contacts (N, 4) bool`` = linear-force norm > 50 N (the
peter threshold; the extractor already zeroed anything under 30 N), plus the
same ``captured`` / ``fps`` / ``limbs`` / ``threshold`` / ``source`` keys.

``joints_3d.npz``: ``joints (N, 16, 3) float32`` metres in the MoCap world
(+z up) + ``joint_names`` + ``fps``. Kept for later pose scoring.

Usage::

    uv run python scripts/import_parkour.py            # all 28 clips
    uv run python scripts/import_parkour.py --clips kv01_PKFC,mu02_PKLP
"""
from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.paths import PARKOUR  # noqa: E402

SRC = PARKOUR
DST = SRC / "processed"

# Parkour stores (l_ankle, r_ankle, l_fingers, r_fingers); we keep the peter
# column order so every forces.npz in this repo reads the same way.
LIMBS = ("left_hand", "right_hand", "left_foot", "right_foot")
PARKOUR_TO_LIMBS = [2, 3, 0, 1]
CONTACT_THRESHOLD_N = 50.0
FPS = 25.0


def _load_gt() -> tuple[dict, dict]:
    gt = SRC / "gt_motion_forces"
    with open(gt / "contact_forces_local.pkl", "rb") as f:
        forces = pickle.load(f, encoding="latin1")
    with open(gt / "joint_3d_positions.pkl", "rb") as f:
        joints = pickle.load(f, encoding="latin1")
    return forces, joints


def _encode_video(frames_dir: Path, out_path: Path) -> None:
    """Re-encode the clip's PNG frames at the true 25 fps rate."""
    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-framerate", str(int(FPS)),
           "-i", str(frames_dir / "%04d.png"), "-c:v", "libx264", "-crf", "18",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)


def _import_clip(name: str, force_gt: dict, joint_gt: dict) -> None:
    frames_dir = SRC / "frames" / name
    n_video = len(list(frames_dir.glob("*.png")))
    n_gt = force_gt["contact_forces_local"].shape[0]
    if n_video != n_gt - 3:
        raise ValueError(f"{name}: {n_video} frames vs {n_gt} GT rows (expected GT-3)")

    clip_dir = DST / name
    clip_dir.mkdir(parents=True, exist_ok=True)

    video_path = clip_dir / "video.mp4"
    if not video_path.exists():
        _encode_video(frames_dir, video_path)

    # GT rows [0:n_video]: the authors pair video frame i with MoCap row i.
    wrench = force_gt["contact_forces_local"][:n_video, PARKOUR_TO_LIMBS].astype(np.float32)
    captured = force_gt["mask_uncaptured_forces"][:n_video, PARKOUR_TO_LIMBS] == 0

    np.savez(clip_dir / "forces.npz",
             forces=wrench, captured=captured,
             fps=np.float32(FPS), limbs=np.array(LIMBS),
             n_frames=np.int64(n_video), source=f"parkour/{name}")

    contacts = np.linalg.norm(wrench[:, :, :3], axis=2) > CONTACT_THRESHOLD_N
    np.savez(clip_dir / "contacts.npz",
             contacts=contacts, captured=captured,
             fps=np.float32(FPS), limbs=np.array(LIMBS),
             threshold=np.float32(CONTACT_THRESHOLD_N), source=f"parkour/{name}")

    np.savez(clip_dir / "joints_3d.npz",
             joints=joint_gt["joint_3d_positions"][:n_video].astype(np.float32),
             joint_names=np.array(joint_gt["joint_names"]),
             fps=np.float32(FPS))

    on = contacts[captured.all(axis=1)].mean(axis=0).round(2) if captured.all(axis=1).any() else "-"
    print(f"  {name}: {n_video} frames, contact rate {on}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips", default=None,
                        help="comma-separated clip names (default: all)")
    args = parser.parse_args()

    force_gt, joint_gt = _load_gt()
    names = sorted(force_gt.keys())
    if args.clips:
        wanted = {c.strip() for c in args.clips.split(",") if c.strip()}
        missing = wanted - set(names)
        if missing:
            raise SystemExit(f"unknown clip(s): {sorted(missing)}")
        names = [n for n in names if n in wanted]

    print(f"[import_parkour] {len(names)} clip(s) → {DST}")
    for name in names:
        _import_clip(name, force_gt[name], joint_gt[name])
    return 0


if __name__ == "__main__":
    sys.exit(main())
