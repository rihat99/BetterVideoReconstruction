"""Resolve a CLI ``--source`` argument to a list of frame paths.

Accepts three input shapes:

  * single image file (``.jpg`` / ``.jpeg`` / ``.png`` / ``.bmp`` / ``.webp``)
    — copied into ``tmp_dir/single_image/`` as a single frame
  * folder of frames — listed and sorted numerically by stem
  * video file (``.mp4`` / ``.mov`` / ``.avi`` / ``.mkv`` / ``.webm``) —
    decoded with OpenCV into ``tmp_dir/frames/`` as ``0.jpg, 1.jpg, ...``

Returns ``(frame_paths, src_fps)``. ``src_fps`` is ``None`` for the
single-image and folder cases (no native frame rate); callers should treat
that as "no temporal context" and either skip video output or fall back to
a default ``--fps``.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_source(source: Path | str) -> bool:
    """True iff ``source`` is a single image file (not a folder or video)."""
    p = Path(source)
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def frame_stem(idx: int, source: Path | str, n_frames: int) -> str:
    """Filename stem for frame ``idx``.

    Single-image source → ``source.stem`` (e.g. ``ellen2``).
    Folder / video / multi-frame → ``frame_<idx:06d>``.

    Used by precompute scripts when writing per-frame outputs so single-image
    inputs avoid the noisy ``frame_000000`` naming.
    """
    if n_frames == 1 and is_image_source(source):
        return Path(source).stem
    return f"frame_{idx:06d}"


def find_frame_file(
    dir_path: Path,
    idx: int,
    source: Path | str | None = None,
    suffix: str = ".png",
) -> Path | None:
    """Locate a per-frame file under ``dir_path``, probing image- and video-style names.

    Returns the first existing match, or ``None``. Used by readers (e.g.
    ``estimate_body_sam3d``) to stay back-compatible with both naming styles.
    """
    candidates: list[str] = []
    if source is not None and is_image_source(source):
        candidates.append(Path(source).stem)
    candidates.append(f"frame_{idx:06d}")
    for stem in candidates:
        p = dir_path / f"{stem}{suffix}"
        if p.exists():
            return p
    return None


def _list_frame_paths(folder: Path) -> list[Path]:
    paths = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    try:
        paths.sort(key=lambda p: int(p.stem))
    except ValueError:
        paths.sort()
    if not paths:
        raise RuntimeError(f"no image frames found in {folder}")
    return paths


def _video_to_frames(video_path: Path, out_dir: Path) -> tuple[list[Path], float | None]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"{idx}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        idx += 1
    cap.release()
    if idx == 0:
        raise RuntimeError(f"no frames decoded from {video_path}")
    return [out_dir / f"{i}.jpg" for i in range(idx)], (fps if fps > 0 else None)


def resolve_source(source: Path, tmp_dir: Path) -> tuple[list[Path], float | None]:
    """Return ``(frame_paths, src_fps)`` for an image / folder / video source."""
    suffix = source.suffix.lower()
    if source.is_file() and suffix in IMAGE_EXTS:
        dest_dir = tmp_dir / "single_image"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"frame_000000{suffix}"
        shutil.copy(source, dest)
        return [dest], None
    if source.is_dir():
        return _list_frame_paths(source), None
    if suffix in VIDEO_EXTS:
        return _video_to_frames(source, tmp_dir / "frames")
    raise ValueError(
        f"source must be an image, folder of frames, or video file: {source}"
    )
