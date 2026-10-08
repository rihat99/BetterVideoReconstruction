"""Shape the climb_wall_3_gt clips like a ClimbingVideos corpus so the published-model
runners can be pointed at them unchanged.

The five baseline runners (``other/*/run_climbing_testset.py``) all read one root laid
out as ``<root>/results/test_scenes.txt``, ``<root>/frames/<aa>/<bb>/<clip>/%06d.jpg``
and ``<root>/features/<stage>/<aa>/<bb>/<clip>/``. This builds exactly that shape for
the 68 climb_wall_3 clips, over the ORIGINAL (distorted) ``cam_left.mp4`` — the same
video our own single-view run used, so the models and our pipeline see identical pixels.

Features are symlinks into ``peter/out_climb_wall_3_single/<clip>/``; only the JPEG
frames are copied, because that is the one thing the runners cannot get from a video.

Usage:
    uv run --no-sync python scripts/peter_gt/prepare_baselines.py [--skip-existing] [--jobs 8]
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.paths import PETER  # noqa: E402

CLIPS = PETER / "climb_wall_3"                       # original, distorted videos
GT = PETER / "climb_wall_3_gt" / "human"             # multi-view reference fits
SINGLE = PETER / "out_climb_wall_3_single"           # our single-view run on the same video
FARM = PETER / "climb_wall_3_gt" / "others" / "_farm"
VIEW, FPS = "cam_left", 30.0
STAGES = ("sam3", "sapiens", "sam3d", "vggt", "geometry", "geocalib", "human_optim")


def sub(clip: str) -> str:
    """The ``<aa>/<bb>/<scene>`` fan-out every runner derives from a scene id."""
    return f"{clip[:2]}/{clip[2:4]}/{clip}"


def clips() -> list[str]:
    return sorted(d.name for d in GT.iterdir() if d.is_dir() and not d.name.startswith("_"))


def extract(clip: str, skip_existing: bool) -> tuple[str, int]:
    out = FARM / "frames" / sub(clip)
    src = CLIPS / clip / f"{VIEW}.mp4"
    n_want = int(np.load(GT / clip / "cameras.npz", allow_pickle=True)["num_frames"])
    if skip_existing and out.is_dir() and len(list(out.glob("*.jpg"))) >= n_want:
        return clip, -1
    out.mkdir(parents=True, exist_ok=True)
    cap, idx = cv2.VideoCapture(str(src)), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out / f"{idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        idx += 1
    cap.release()
    return clip, idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    names = clips()
    (FARM / "results").mkdir(parents=True, exist_ok=True)
    rows = [f"{c}\t{sub(c)}.mp4\t{FPS}" for c in names]
    (FARM / "results" / "test_scenes.txt").write_text("\n".join(rows) + "\n")

    for clip in names:
        for stage in STAGES:
            src = SINGLE / clip / stage
            if not src.is_dir():
                continue
            dst = FARM / "features" / stage / sub(clip)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)
        video = FARM / "videos" / f"{sub(clip)}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        if video.is_symlink() or video.exists():
            video.unlink()
        video.symlink_to(CLIPS / clip / f"{VIEW}.mp4")

    with ProcessPoolExecutor(args.jobs) as pool:
        for clip, n in pool.map(extract, names, [args.skip_existing] * len(names)):
            print(f"{clip}: {'frames present' if n < 0 else f'{n} frames'}", flush=True)
    print(f"farm ready: {FARM} ({len(names)} clips)")


if __name__ == "__main__":
    main()
