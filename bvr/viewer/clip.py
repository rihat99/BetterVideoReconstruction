"""Read one clip of a run into numpy arrays: what the viewer draws.

The viewer reads a clip the way a stage does: each file from the clip's own tree, else from the
upstream run's (``paths.Clip.find``), so a run that wrote a few steps shows on top of the run it
continued. Every npz goes through ``files.load_npz``, the one reader of new and old trees, which
returns the old contact frame name ``head`` as ``head_top``. :func:`read` takes the scene, the
cameras and gravity; :func:`read_bodies` takes the people of one body source, when it is chosen.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import better_human as bh
import cv2
import numpy as np
import torch

from bvr import body, files, paths

log = logging.getLogger(__name__)

#: The body sources a clip may hold, in sidebar order: name -> (file, format, the file of its contact labels).
SOURCES = {
    "converter": (paths.SMPLX_PARAMS, "bvr-smplx", None),
    "prefit": (paths.PREFIT, "bvr-prefit", None),
    "kinematics_1": (paths.KINEMATIC_1, "bvr-kinematics", paths.CONTACTS_1),
    "kinematics_2": (paths.KINEMATIC_2, "bvr-kinematics", paths.CONTACTS_2),
    "kindyn_1": (paths.KINDYN_1, "bvr-kindyn", paths.KINDYN_1),
}
#: Share of the least confident scene points left out: the share the contact stages leave out of their cloud.
SCENE_CONFIDENCE_DROP = 0.1


@dataclass(frozen=True)
class ClipData:
    """The parts of a clip that do not depend on the body source. ``T`` frames, world units in metres."""

    name: str
    fps: float                       #: playback rate of one frame
    extrinsics: np.ndarray           #: (T, 4, 4) world to camera (x right, y down, z forward)
    intrinsics: np.ndarray           #: (T, 3, 3) camera to pixels of the video frame
    size: tuple[int, int]            #: video frame (width, height) in pixels
    scene: tuple[np.ndarray, np.ndarray] | None         #: points (N, 3), colours (N, 3) uint8; None without the file
    #: per frame: depth points (N_t, 3), colours (N_t, 3) uint8; None without the file
    human: tuple[list[np.ndarray], list[np.ndarray]] | None
    gravity: np.ndarray | None       #: (3,) unit vector down, in the world
    gravity_source: str              #: "geocalib", "ground" or "fallback_down"
    ground: tuple[np.ndarray, np.ndarray, float] | None  #: the fitted floor: centroid (3,), up normal (3,), extent
    sources: tuple[str, ...]         #: the body sources the clip holds, a subset of SOURCES in its order
    written: tuple[str, ...]         #: those of them that the clip's own tree holds, not the upstream run's
    video: VideoFrames


@dataclass(frozen=True)
class Bodies:
    """The people of one body source. ``P`` people, ``T`` frames, ``F`` contact frames."""

    q: np.ndarray                    #: (P, T, 211) world poses; an untracked frame holds the nearest tracked one
    betas: np.ndarray                #: (P, 10)
    tracked: np.ndarray              #: (P, T) bool
    object_ids: np.ndarray           #: (P,)
    contact: np.ndarray | None       #: (P, T, F) bool, in the order of the body's contact frames
    forces: np.ndarray | None        #: (P, T, F, 3) world contact forces in newtons
    torques: np.ndarray | None       #: (P, T, 159) generalised forces; the first six are the root's residual wrench


def read(clip: paths.Clip, *, human_max_points: int) -> ClipData:
    """Read ``clip``; ``human_max_points`` caps the depth points per frame and person (0 keeps all)."""
    transform = files.load_npz(clip.find(paths.TRANSFORM), "bvr-transform")
    frames = transform["frame_indices"].astype(np.int64)
    span = float(frames[-1] - frames[0])
    fps = float(transform["fps"]) * ((len(frames) - 1) / span if span > 0 else 1.0)  # a subsampled clip plays slower

    # A tree without depth (a synthetic scene with known bodies and cameras) has neither cloud.
    scene = human = None
    if clip.find(paths.SCENE).exists():
        stored = files.load_npz(clip.find(paths.SCENE), "bvr-scene")
        kept = stored["confidence"] >= np.quantile(stored["confidence"], SCENE_CONFIDENCE_DROP)
        scene = stored["points"][kept], stored["colors"][kept]
    if clip.find(paths.HUMAN).exists():
        human = _per_frame(files.load_npz(clip.find(paths.HUMAN), "bvr-human-points"), len(frames), human_max_points)

    gravity, source, ground = None, "", None
    if clip.find(paths.GRAVITY).exists():
        stored = files.load_npz(clip.find(paths.GRAVITY), "bvr-gravity")
        gravity, source = stored["gravity_world"] / np.linalg.norm(stored["gravity_world"]), str(stored["source"])
        if stored["ground_accepted"]:
            ground = stored["ground_centroid"], stored["ground_normal"], float(stored["ground_extent"])

    return ClipData(
        name=clip.name, fps=fps, extrinsics=transform["extrinsics"], intrinsics=transform["intrinsics_px_orig"],
        size=(int(transform["image_width"]), int(transform["image_height"])),
        scene=scene, human=human, gravity=gravity, gravity_source=source,
        ground=ground, sources=tuple(name for name, (file, _, _) in SOURCES.items() if clip.find(file).exists()),
        written=tuple(name for name, (file, _, _) in SOURCES.items() if clip.path(file).exists()),
        video=VideoFrames(clip.video, frames))


def read_bodies(clip: paths.Clip, source: str, human: bh.HumanModel, extrinsics: np.ndarray) -> Bodies:
    """The people of ``source`` (a key of :data:`SOURCES`) in the world; a converter or pre-fit body, which is in
    the camera frame, moves to the world through ``extrinsics (T, 4, 4)``.

    Contact labels are read from the tree the body was read from, so a body never wears the labels of another run.
    """
    file, format, labels = SOURCES[source]
    path = clip.find(file)
    stored = files.load_npz(path, format)
    tracked = stored["valid_mask"].astype(bool)
    if "q" in stored:
        q, betas = stored["q"], stored["betas"]
    else:   # classic parameters per frame, in the camera: one shape per person, the mean over the tracked frames
        betas = (np.where(tracked[..., None], stored["betas"], 0.0).sum(1) / np.maximum(tracked.sum(1), 1)[:, None])
        fields = ("global_orient", "transl", "body_pose", "left_hand_pose", "right_hand_pose")
        with torch.no_grad():
            _, q, _ = bh.from_classic(human, bh.families.SMPLXClassic(
                betas=torch.as_tensor(betas, dtype=torch.float32)[:, None],
                **{name: torch.as_tensor(np.nan_to_num(stored[name])) for name in fields}))
            q = body.to_world(q, torch.as_tensor(extrinsics)).numpy()

    labelled = path.parents[1] / labels if labels else None
    contact = forces = None
    if labelled is not None and labelled.exists():
        labels_file = stored if labelled == path else files.load_npz(labelled, "bvr-contacts")
        if (list(labels_file["contact_frame_names"]) == list(human.contact_frames.names)
                and np.array_equal(labels_file["object_ids"], stored["object_ids"])):
            contact, forces = labels_file["frame_contact"].astype(bool), labels_file.get("frame_forces")
        else:
            log.warning("%s: other people or contact frames than its body; drawn without labels", labelled)
    return Bodies(q=_held(np.asarray(q, np.float32), tracked), betas=np.asarray(betas, np.float32), tracked=tracked,
                  object_ids=stored["object_ids"], contact=contact, forces=forces, torques=stored.get("joint_torques"))


def _held(q: np.ndarray, tracked: np.ndarray) -> np.ndarray:
    """``q (P, T, nq)`` with each untracked frame replaced by its person's nearest tracked frame: a stored untracked
    frame may hold zero quaternions, which pose to NaN."""
    steps = np.arange(tracked.shape[1])
    distance = np.abs(steps[:, None] - np.where(tracked[:, None, :], steps, np.inf))
    return np.take_along_axis(q, distance.argmin(axis=-1)[..., None], axis=1)


def _per_frame(human: dict[str, np.ndarray], frames: int, cap: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The human depth points as one array per frame, at most ``cap`` per frame and person, drawn at random."""
    frame, person = human["frame_id"].astype(np.int64), human["obj_id"].astype(np.int64)
    order = np.random.default_rng(0).permutation(len(frame))
    if cap:
        group = (frame * (person.max(initial=0) + 1) + person)[order]
        order = order[np.argsort(group, kind="stable")]
        group = np.sort(group, kind="stable")
        rank = np.arange(len(group)) - np.searchsorted(group, group)    # place of each point in its group
        order = order[rank < cap]
    order = order[np.argsort(frame[order], kind="stable")]
    cuts = np.cumsum(np.bincount(frame[order], minlength=frames))[:-1]
    return np.split(human["points"][order], cuts), np.split(human["colors"][order], cuts)


class VideoFrames:
    """``step -> (h, w, 3)`` uint8 RGB frames of the clip's video, decoded on demand, the longer side at most
    ``max_side`` pixels; ``frames (T,)`` is the video frame each step shows.

    Playback reads forward, so the decoder skips ahead; a jump back or far ahead seeks. Every decoded frame is kept,
    and a frame past the end of the video is black.
    """

    def __init__(self, video: Path, frames: np.ndarray, *, max_side: int = 480) -> None:
        self.video, self.frames, self.max_side = video, frames, max_side
        self._capture: cv2.VideoCapture | None = None
        self._next = 0                                       # video frame the next read returns
        self._kept: dict[int, np.ndarray] = {}

    def __call__(self, step: int) -> np.ndarray:
        index = int(self.frames[min(step, len(self.frames) - 1)])
        if index not in self._kept:
            if self._capture is None:
                self._capture = cv2.VideoCapture(str(self.video))
            if not self._next <= index <= self._next + 60:
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                self._next = index
            while self._next < index:
                self._capture.grab()
                self._next += 1
            decoded, image = self._capture.read()
            self._next = index + 1
            if not decoded:
                return np.zeros((2, 2, 3), np.uint8)
            scale = min(1.0, self.max_side / max(image.shape[:2]))
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            self._kept[index] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self._kept[index]
