"""Shape a dataset's clips like a ClimbingVideos corpus for the published-model runners.

The baseline runners (``../other/*/run_climbing_testset.py``: DECO, BSTRO, GVHMR,
PromptHMR, InteractVLM) all read one root laid out as ``<root>/results/test_scenes.txt``
(``<clip>\\t<aa>/<bb>/<clip>.mp4\\t<fps>``), ``<root>/frames/<aa>/<bb>/<clip>/%06d.jpg``
and ``<root>/features/<stage>/<aa>/<bb>/<clip>/``, with ``<aa>/<bb>`` the
``scene_db.shard`` fan-out. This builds that root so the runners can be pointed at it
unchanged. The JPEG frames (quality 95, frames ``0..n-1`` with ``n`` the processed clip's
frame count) are the one thing the runners cannot get from a video, so they are decoded;
the rest depends on the dataset:

* ``parkour`` — every processed Parkour clip, for InteractVLM: the tracker boxes are
  re-saved as ``features/sam3/.../bboxes.npz`` holding only the two arrays the runner
  loads without ``allow_pickle``. Farm: ``Parkour-dataset/processed/others/_farm``.
* ``climb_wall_3`` — every climb_wall_3_gt clip, over the ORIGINAL (distorted)
  ``cam_left.mp4``: the same video our own single-view run used, so the models and our
  pipeline see identical pixels. Features are symlinks into
  ``peter/out_climb_wall_3_single/<clip>/<stage>`` and the video a symlink under
  ``videos/``. Farm: ``peter/climb_wall_3_gt/others/_farm``, which
  ``datasets/peter/gt/run_baselines.sh`` runs the models on.

Usage:
    uv run --no-sync python datasets/methods/baseline_farm.py --dataset climb_wall_3 \\
        [--skip-existing] [--jobs 8]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


from datasets.climbingvideos.scene_db import shard
from datasets.parkour.paths import PARKOUR
from datasets.peter.paths import PETER

PARKOUR_ROOT = (PARKOUR / "processed").resolve()
CW3_VIDEOS = PETER / "climb_wall_3"                  # original, distorted videos
CW3_GT = PETER / "climb_wall_3_gt" / "human"         # multi-view reference fits
CW3_SINGLE = PETER / "out_climb_wall_3_single"       # our single-view run on the same video
CW3_VIEW, CW3_FPS = "cam_left", 30.0
CW3_STAGES = ("sam3", "sapiens", "sam3d", "vggt", "geometry", "geocalib", "human_optim")
FARMS = {"parkour": PARKOUR_ROOT / "others" / "_farm",
         "climb_wall_3": PETER / "climb_wall_3_gt" / "others" / "_farm"}
JPEG_QUALITY = 95


def sub(clip: str) -> str:
    """``<aa>/<bb>/<clip>``, the fan-out every runner derives from a scene id."""
    return "/".join((*shard(clip), clip))


def parkour_clips() -> list[tuple[str, Path, int, str]]:
    """``(clip, video, frame count, fps text)`` per processed Parkour clip."""
    rows = []
    for run in sorted((PARKOUR_ROOT / "out").iterdir()):
        if run.is_dir() and not run.name.startswith("_"):
            geometry = np.load(run / "geometry" / "transform.npz")
            rows.append((run.name, PARKOUR_ROOT / run.name / "video.mp4",
                         len(geometry["frame_indices"]), f"{float(geometry['fps']):.6f}"))
    return rows


def climb_wall_3_clips() -> list[tuple[str, Path, int, str]]:
    """``(clip, video, frame count, fps text)`` per climb_wall_3 clip with a reference fit."""
    rows = []
    for gt in sorted(CW3_GT.iterdir()):
        if gt.is_dir() and not gt.name.startswith("_"):
            n_frames = int(np.load(gt / "cameras.npz", allow_pickle=True)["num_frames"])
            rows.append((gt.name, CW3_VIDEOS / gt.name / f"{CW3_VIEW}.mp4", n_frames,
                         str(CW3_FPS)))
    return rows


def parkour_features(farm: Path, clip: str) -> None:
    """Re-save the tracker boxes as a pickle-free ``frame_indices`` / ``bboxes`` pair."""
    src = np.load(PARKOUR_ROOT / "out" / clip / "sam3" / "bboxes.npz", allow_pickle=True)
    out = farm / "features" / "sam3" / sub(clip)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "bboxes.npz",
             frame_indices=np.asarray(src["frame_indices"], np.int64),
             bboxes=np.asarray(src["bboxes"], np.int64))


def climb_wall_3_features(farm: Path, clip: str) -> None:
    """Symlink our single-view run's stage dirs and the video into the farm."""
    links = {farm / "features" / stage / sub(clip): CW3_SINGLE / clip / stage
             for stage in CW3_STAGES if (CW3_SINGLE / clip / stage).is_dir()}
    links[farm / "videos" / f"{sub(clip)}.mp4"] = CW3_VIDEOS / clip / f"{CW3_VIEW}.mp4"
    for link, target in links.items():
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)


CLIPS = {"parkour": parkour_clips, "climb_wall_3": climb_wall_3_clips}
FEATURES = {"parkour": parkour_features, "climb_wall_3": climb_wall_3_features}


def extract(video: Path, out: Path, n_frames: int, skip_existing: bool) -> int:
    """Decode frames ``0..n_frames-1`` into ``out/%06d.jpg``; -1 if they were already there."""
    if skip_existing and out.is_dir() and len(list(out.glob("*.jpg"))) >= n_frames:
        return -1
    out.mkdir(parents=True, exist_ok=True)
    cap, written = cv2.VideoCapture(str(video)), 0
    while written < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out / f"{written:06d}.jpg"), frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        written += 1
    cap.release()
    if written != n_frames:
        raise SystemExit(f"{video}: decoded {written} frames, the clip has {n_frames}")
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=sorted(FARMS))
    ap.add_argument("--clips", default=None,
                    help="comma-separated clip names (default: all); builds only these and "
                         "leaves the farm's results/test_scenes.txt untouched")
    ap.add_argument("--skip-existing", action="store_true",
                    help="keep a clip's frames when its JPEGs are already there")
    ap.add_argument("--jobs", type=int, default=8, help="parallel frame decoders")
    ap.add_argument("--out", type=Path, default=None,
                    help="farm root (default: the dataset's others/_farm)")
    args = ap.parse_args(argv)

    farm = args.out or FARMS[args.dataset]
    clips = CLIPS[args.dataset]()
    if args.clips:
        wanted = set(args.clips.split(","))
        clips = [row for row in clips if row[0] in wanted]
    scenes = farm / "results" / "test_scenes.txt"
    if args.clips:
        print(f"--clips given: {scenes} left untouched")
    else:
        scenes.parent.mkdir(parents=True, exist_ok=True)
        scenes.write_text("".join(f"{name}\t{sub(name)}.mp4\t{fps}\n" for name, _, _, fps in clips))
    for name, *_ in clips:
        FEATURES[args.dataset](farm, name)

    with ProcessPoolExecutor(args.jobs) as pool:
        counts = pool.map(extract, [video for _, video, _, _ in clips],
                          [farm / "frames" / sub(name) for name, *_ in clips],
                          [n_frames for _, _, n_frames, _ in clips],
                          [args.skip_existing] * len(clips))
        for (name, *_), count in zip(clips, counts):
            print(f"{name}: {'frames present' if count < 0 else f'{count} frames'}", flush=True)
    print(f"farm ready: {farm} ({len(clips)} clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
