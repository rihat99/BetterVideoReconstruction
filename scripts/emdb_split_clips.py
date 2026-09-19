"""Split every EMDB sequence into ~500-frame clips the pipeline can batch over.

EMDB ships 81 sequences of up to 4084 frames. The pipeline works on clips, so
each sequence is cut into ``ceil(n_frames / 500)`` pieces of as-equal-as-possible
length — no piece is longer than 500 frames and none is a short stub. The clips
are encoded from ``images/``, the raw capture, NOT from the sequence's
``*_video.mp4``: that file is a downscaled GT-overlay visualisation.

Layout written under the dataset root::

    clips/index.csv          one row per clip (frame range, EMDB subset, ...)
    clips/<Pk>/<seq>_<i>.mp4 the clips, grouped by participant

``P0..P9`` (the official release) are left untouched. Point the pipeline at one
participant at a time so the precompute stages load their models once per run::

    uv run python scripts/pipeline.py --source data/EMDB/clips/P0 \
        --save-dir data/EMDB/out/P0 --skip-existing

Usage:
    uv run python scripts/emdb_split_clips.py --root /path/to/data/EMDB
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg

FPS = 30
# Matches the clips made by hand earlier: 1440x1920 h264 at ~34 Mb/s.
CRF = 18
PRESET = "medium"


def windows(n_frames: int, clip_frames: int) -> list[tuple[int, int]]:
    """``(start, count)`` per clip, splitting ``n_frames`` into equal-ish pieces.

    The piece count is ``ceil(n_frames / clip_frames)``; the remainder is spread
    over the last pieces, one frame each, so 1183 frames at 500 give
    ``394, 394, 395`` rather than ``500, 500, 183``.
    """
    n_clips = max(1, math.ceil(n_frames / clip_frames))
    base, rem = divmod(n_frames, n_clips)
    counts = [base + (i >= n_clips - rem) for i in range(n_clips)]
    starts = [sum(counts[:i]) for i in range(n_clips)]
    return list(zip(starts, counts))


def encode(images: Path, start: int, count: int, dest: Path) -> None:
    """Encode ``count`` frames from ``images`` starting at ``start`` into ``dest``.

    Writes a temporary file and renames it, so an interrupted run never leaves a
    truncated clip that a later run would skip.
    """
    tmp = dest.with_suffix(".tmp.mp4")
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error",
         "-framerate", str(FPS), "-start_number", str(start),
         "-i", str(images / "%05d.jpg"), "-frames:v", str(count),
         "-c:v", "libx264", "-crf", str(CRF), "-preset", PRESET,
         "-pix_fmt", "yuv420p", "-threads", "12", str(tmp)],
        check=True, capture_output=True,
    )
    tmp.rename(dest)


def sequences(root: Path) -> list[tuple[Path, dict]]:
    """Every EMDB sequence directory with its annotation dict, in release order."""
    out = []
    for pkl in sorted(root.glob("P?/*/*_data.pkl")):
        with open(pkl, "rb") as f:
            out.append((pkl.parent, pickle.load(f)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="EMDB dataset root")
    parser.add_argument("--clip-frames", type=int, default=500, help="maximum frames per clip")
    parser.add_argument("--workers", type=int, default=16, help="parallel ffmpeg encodes")
    parser.add_argument("--dry-run", action="store_true", help="plan only, encode nothing")
    args = parser.parse_args()

    jobs, rows = [], []
    for seq_dir, meta in sequences(args.root):
        participant = seq_dir.parent.name
        dest_dir = args.root / "clips" / participant
        for i, (start, count) in enumerate(windows(meta["n_frames"], args.clip_frames)):
            dest = dest_dir / f"{seq_dir.name}_{i}.mp4"
            good = int(meta["good_frames_mask"][start:start + count].sum())
            rows.append({"clip": dest.stem, "participant": participant,
                         "sequence": seq_dir.name, "start": start, "end": start + count,
                         "n_frames": count, "good_frames": good,
                         "emdb1": int(meta["emdb1"]), "emdb2": int(meta["emdb2"]),
                         "gender": meta["gender"], "fps": FPS})
            if not dest.exists():
                jobs.append((seq_dir / "images", start, count, dest))

    print(f"{len(rows)} clips from {len(sequences(args.root))} sequences; "
          f"{len(jobs)} to encode, {len(rows) - len(jobs)} already present")
    if args.dry_run:
        return

    for participant in {row["participant"] for row in rows}:
        (args.root / "clips" / participant).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as pool:
        for done, _ in enumerate(pool.map(lambda job: encode(*job), jobs), 1):
            if done % 10 == 0 or done == len(jobs):
                print(f"  encoded {done}/{len(jobs)}", flush=True)

    index = args.root / "clips" / "index.csv"
    with open(index, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {index}")


if __name__ == "__main__":
    main()
