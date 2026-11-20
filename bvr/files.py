"""The files of a clip's tree: named formats, writes that are whole or absent, video frames, masks.

A stage that is killed must never leave a half-written output, because the next stage would read
it. So an npz is written under a temporary name and renamed (:func:`save_npz`), and a folder is
written as ``<name>.staging`` and renamed (:func:`staged_directory`). Every npz names its format
and version; :func:`load_npz` is the one reader, of these files and of the trees written before
formats were named.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

#: Format name of each npz of the stage table, with the version this code writes.
FORMATS = {
    "bvr-bboxes": 1,        # sam3/bboxes.npz
    "bvr-camera": 1,        # vggt/camera.npz
    "bvr-keypoints": 1,     # sapiens/pose.npz
    "bvr-mhr": 1,           # sam3d/params.npz
    "bvr-smplx": 1,         # sam3d/smplx_params.npz
    "bvr-transform": 1,     # geometry/transform.npz, geometry/fused/transform.npz
    "bvr-scene": 1,         # geometry/scene.npz, geometry/fused/scene.npz
    "bvr-human-points": 1,  # geometry/human.npz, geometry/fused/human.npz
    "bvr-scale": 1,         # geometry/scale.npz
    "bvr-prefit": 1,        # geometry/prefit.npz
    "bvr-gravity": 1,       # geocalib/gravity.npz
    "bvr-kinematics": 1,    # human_optim/kinematic_<n>.npz
    "bvr-contacts": 1,      # human_optim/contacts_<n>.npz
    "bvr-kindyn": 1,        # human_optim/kindyn_<n>.npz
}

#: Contact frames that files written before the rename call by another name.
_LEGACY_FRAME_NAMES = {"head": "head_top"}


def save_npz(path: Path, format: str, **arrays: np.ndarray) -> None:
    """Write ``arrays`` to ``path`` with the format's name and version; the file is whole or absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as file:
        np.savez(file, format=format, format_version=FORMATS[format], **arrays)
    os.replace(temporary, path)


def load_npz(path: Path, format: str) -> dict[str, np.ndarray]:
    """Read every array of an npz of the stage table.

    A file that names another format, or a version newer than this code writes, raises. A file
    without a format was written before formats were named: it is read as it is, and its contact
    frame ``head`` is returned as ``head_top``.
    """
    # The bounding-box file of old trees holds object arrays, which numpy reads only with pickle.
    with np.load(path, allow_pickle=True) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if "format" in arrays:
        found, version = str(arrays.pop("format")), int(arrays.pop("format_version"))
        if found != format or version > FORMATS[format]:
            raise ValueError(f"{path} is {found} version {version}; expected {format} up to version {FORMATS[format]}")
    elif "contact_frame_names" in arrays:
        arrays["contact_frame_names"] = np.array(
            [_LEGACY_FRAME_NAMES.get(str(name), str(name)) for name in arrays["contact_frame_names"]])
    return arrays


@contextmanager
def staged_directory(path: Path) -> Iterator[Path]:
    """Yield ``<path>.staging`` to write a folder into; it becomes ``path`` when the block ends.

    ``path`` must not exist: ``bvr.run`` moves a step's previous outputs to the trash before the
    stage starts. After an error the staging folder stays, and goes to the trash the same way.
    """
    staging = path.with_name(path.name + ".staging")
    staging.mkdir(parents=True)
    yield staging
    staging.rename(path)


def frames(video: Path) -> Iterator[np.ndarray]:
    """Yield the frames of ``video`` in decoding order, BGR ``(H, W, 3)`` uint8.

    Frame k of every output of a clip is the k-th frame yielded here.
    """
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise OSError(f"cannot open video {video}")
    try:
        while True:
            decoded, frame = capture.read()
            if not decoded:
                return
            yield frame
    finally:
        capture.release()


def frame_rate(video: Path) -> float:
    """The frame rate of ``video`` in frames per second."""
    capture = cv2.VideoCapture(str(video))
    rate = capture.get(cv2.CAP_PROP_FPS) if capture.isOpened() else 0.0
    capture.release()
    if not rate > 0.0:
        raise OSError(f"cannot read the frame rate of {video}")
    return float(rate)


def extract_frames(video: Path, directory: Path) -> list[Path]:
    """Decode ``video`` into ``directory`` as ``0.jpg``, ``1.jpg``, ... and return the paths.

    The model stages read these files, so the JPEG quality is part of their results.
    """
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, frame in enumerate(frames(video)):
        paths.append(directory / f"{index}.jpg")
        cv2.imwrite(str(paths[-1]), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not paths:
        raise OSError(f"no frame decoded from {video}")
    return paths


def read_mask(path: Path) -> np.ndarray:
    """Read a mask image as a boolean ``(H, W)`` array."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"no mask image at {path}")
    return (image[..., 0] if image.ndim == 3 else image) > 0


def write_mask(path: Path, mask: np.ndarray) -> None:
    """Write a boolean ``(H, W)`` mask as a 0 and 255 PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(bool).astype(np.uint8) * 255)
