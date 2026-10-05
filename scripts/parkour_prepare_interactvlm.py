"""Build a ClimbingVideos-shaped farm over the Parkour clips for InteractVLM.

``../other/InteractVLM/run_climbing_testset.py`` reads a corpus root holding
``results/test_scenes.txt``, ``frames/<aa>/<bb>/<clip>/%06d.jpg`` and
``features/sam3/<aa>/<bb>/<clip>/bboxes.npz``. This writes that root for the 28
Parkour clips: the frames are decoded from each clip's ``video.mp4`` (JPEG 95,
frames ``0..n-1`` with ``n`` the processed corpus's frame count) and the tracker
boxes are re-saved with the two arrays the runner loads without ``allow_pickle``.

Run with the BVR environment::

    uv run --no-sync python scripts/parkour_prepare_interactvlm.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
PARKOUR = (_REPO.parent / "data" / "Parkour-dataset" / "processed").resolve()
FARM = PARKOUR / "others" / "_farm"
JPEG_QUALITY = 95


def clips() -> list[str]:
    return sorted(d.name for d in (PARKOUR / "out").iterdir()
                  if d.is_dir() and not d.name.startswith("_"))


def subdir(clip: str) -> str:
    return f"{clip[:2]}/{clip[2:4]}/{clip}"


def write_frames(clip: str, n_frames: int) -> int:
    """Decode ``video.mp4`` into the farm's frame tree; returns the number written."""
    out = FARM / "frames" / subdir(clip)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(PARKOUR / clip / "video.mp4"))
    written = 0
    while written < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out / f"{written:06d}.jpg"), frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        written += 1
    cap.release()
    return written


def write_boxes(clip: str) -> None:
    """Re-save the tracker boxes as a pickle-free ``frame_indices`` / ``bboxes`` pair."""
    src = np.load(PARKOUR / "out" / clip / "sam3" / "bboxes.npz", allow_pickle=True)
    out = FARM / "features" / "sam3" / subdir(clip)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "bboxes.npz",
             frame_indices=np.asarray(src["frame_indices"], np.int64),
             bboxes=np.asarray(src["bboxes"], np.int64))


def main() -> None:
    rows = []
    for clip in clips():
        geometry = np.load(PARKOUR / "out" / clip / "geometry" / "transform.npz")
        n_frames = len(geometry["frame_indices"])
        written = write_frames(clip, n_frames)
        assert written == n_frames, (clip, written, n_frames)
        write_boxes(clip)
        rows.append(f"{clip}\t{subdir(clip)}.mp4\t{float(geometry['fps']):.6f}")
        print(f"{clip}: {written} frames, fps {float(geometry['fps']):.3f}", flush=True)
    (FARM / "results").mkdir(parents=True, exist_ok=True)
    (FARM / "results" / "test_scenes.txt").write_text("\n".join(rows) + "\n")
    print(f"{len(rows)} clips -> {FARM}")


if __name__ == "__main__":
    main()
