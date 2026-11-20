"""Where things are: the repository, the data root, and every file of a clip's output tree.

Data lives outside the repository under one root, ``$BVR_DATA_ROOT`` (default ``../data``). A run
writes one directory per clip, ``<save-dir>/<clip>/``, and every stage reads and writes the files
named here, relative to that directory. With an upstream root a stage reads what its own run does
not have from ``<upstream>/<clip>/`` (:meth:`Clip.find`); outputs always go to the run's own tree.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("BVR_DATA_ROOT", REPO_ROOT.parent / "data"))
EXAMPLES = DATA_ROOT / "examples"

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm")

# The output tree of a clip, relative to its directory: files, and folders where a stage writes many.
SAM3 = "sam3"                                   # <oid:02d>/frame_*.png, bboxes.npz, mask_overlay.mp4
SAM3_BBOXES = "sam3/bboxes.npz"
VGGT = "vggt"                                   # camera.npz, depth/
VGGT_CAMERA = "vggt/camera.npz"
VGGT_DEPTH = "vggt/depth"
SAPIENS = "sapiens"
SAPIENS_POSE = "sapiens/pose.npz"
SAM3D_PARAMS = "sam3d/params.npz"
SAM3D_OVERLAY = "sam3d/overlay.mp4"
SMPLX_PARAMS = "sam3d/smplx_params.npz"
FUSED = "geometry/fused"                        # the up-to-scale transform.npz, scene.npz, human.npz
TRANSFORM = "geometry/transform.npz"            # metric from here on
SCENE = "geometry/scene.npz"
HUMAN = "geometry/human.npz"
SCALE = "geometry/scale.npz"
PREFIT = "geometry/prefit.npz"
GEOCALIB = "geocalib"
GRAVITY = "geocalib/gravity.npz"
KINEMATIC_1 = "human_optim/kinematic_1.npz"
CONTACTS_1 = "human_optim/contacts_1.npz"
KINEMATIC_2 = "human_optim/kinematic_2.npz"
CONTACTS_2 = "human_optim/contacts_2.npz"
KINDYN_1 = "human_optim/kindyn_1.npz"


@dataclass(frozen=True)
class Clip:
    """One clip of a run: its video, its output directory, and the tree it may read inputs from."""

    name: str
    video: Path
    directory: Path
    upstream: Path | None = None

    def path(self, relative: str) -> Path:
        """Where this run writes ``relative``."""
        return self.directory / relative

    def find(self, relative: str) -> Path:
        """Where to read ``relative``: this run's own file, else the upstream run's."""
        own = self.directory / relative
        if self.upstream is None or own.exists():
            return own
        return self.upstream / relative

    def manifest(self, step: str) -> Path:
        """The manifest of ``step``, written last."""
        return self.directory / "manifests" / f"{step}.json"

    def log(self, step: str) -> Path:
        """The debug log of ``step``."""
        return self.directory / "logs" / f"{step}.log"

    def trash(self, label: str) -> Path:
        """Where a rerun moves what it replaces: ``<save-dir>/_trash/<clip>/<label>/``."""
        return self.directory.parent / "_trash" / self.name / label


def clips(source: Path, save_dir: Path, upstream: Path | None = None) -> list[Clip]:
    """The clips of ``source``: one video, or every video of a folder, each named by its stem.

    :param source: a video file or a folder of video files.
    :param save_dir: the run root; a clip's tree is ``<save_dir>/<stem>/``.
    :param upstream: another run root that inputs may be read from.
    """
    if source.is_dir():
        videos = sorted(path for path in source.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES)
    elif source.suffix.lower() in VIDEO_SUFFIXES and source.is_file():
        videos = [source]
    else:
        videos = []
    if not videos:
        raise FileNotFoundError(f"no video at {source} (a video is one of {', '.join(VIDEO_SUFFIXES)})")
    return [Clip(video.stem, video, save_dir / video.stem, upstream and upstream / video.stem)
            for video in videos]


def mhr_to_smplx_mapping() -> Path:
    """The barycentric map from MHR's mesh to SMPL-X's (``triangle_ids``, ``baryc_coords``)."""
    return REPO_ROOT / "third_party/MHR/tools/mhr_smpl_conversion/assets/mhr2smplx_mapping.npz"
