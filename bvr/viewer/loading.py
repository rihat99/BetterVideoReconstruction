"""Read one ``out/<stem>/`` clip into plain numpy (no viser): everything the viewer draws.

:func:`load_clip_data` reads the fused scene, the per-frame human depth points, the cameras,
gravity and the ground plane, every SMPL-X body source as a world-frame ``q`` per person (with
the contact labels and forces measured on it), and finds the source video.
:mod:`bvr.viewer.layers` turns the result into BetterRobot layers. A dataset's own reader builds
on these functions and adds its measured forces as :class:`ForceArrows` (``ClipData.extras``);
``scripts/view_3d.py`` plugs the dataset readers in.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from better_robot.lie import se3

from bvr.body import Body
from bvr.paths import EXAMPLES

# The contact-label schema this viewer reads: contact lives on the body-mounted contact frames
# (``frame_contact``), not on joints. An older artifact is ignored.
_CONTACT_SCHEMA = 2
# Confidence quantile dropped from the displayed scene cloud, so what you see is the cloud the
# optimizer and estimate_contacts run their SDF against (``conf_drop_q`` in the human-optimization
# config).
_SCENE_CONF_DROP_Q = 0.1
# SMPL-X hand joints: the skeleton draws them small and the per-joint forces fold them into the wrist.
FINGER_TOKENS = ("index", "middle", "pinky", "ring", "thumb")
_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


@dataclass
class BodySource:
    """One body source: a world-frame pose per person, and what was measured on it."""

    q: np.ndarray                        #: (P, T, 211) world frame; invalid frames hold a valid pose
    betas: np.ndarray                    #: (P, 10) one identity per person
    valid: np.ndarray                    #: (P, T) bool, the frames each person is drawn on
    object_ids: np.ndarray               #: (P,)
    stored_frames: int                   #: frames in the source file, before fitting to the clip
    contact: np.ndarray | None = None    #: (P, T, F) bool per contact frame, in the body's frame order
    forces: dict | None = None           #: force fields of a kinematics+dynamics body (:func:`load_forces`)


@dataclass
class ForceArrows:
    """Per-frame force arrows a dataset reader adds to a clip (measured forces), drawn like the solved ones."""

    name: str
    kind: str                            #: sidebar checkbox: a ``layers.DYNAMICS`` key, or the dataset's own label (added to the dynamics folder, off)
    origins: np.ndarray                  #: (T, C, 3) world frame, metres
    vectors: np.ndarray                  #: (T, C, 3) newtons
    color: tuple                         #: RGB in [0, 1]
    valid: np.ndarray                    #: (T, C) bool, the arrows drawn per frame
    source: str | None = None            #: the body source they are shown with; ``None``: always


@dataclass
class ClipData:
    """One clip as numpy, cached by the viewer."""

    name: str
    n_frames: int
    fps: float | None                    #: real-time playback rate of one frame step
    extr: np.ndarray                     #: (T, 4, 4) cam-from-world (OpenCV)
    fov_y: np.ndarray                    #: (T,) vertical field of view, radians
    aspect: float
    scene_points: np.ndarray             #: (N, 3)
    scene_colors: np.ndarray             #: (N, 3) uint8
    scene_normals: np.ndarray | None     #: (N, 3)
    scene_voxel: float                   #: merge voxel size in metres (sizes the normal arrows)
    human_points: np.ndarray             #: (M, 3) every frame's human depth points
    human_colors: np.ndarray             #: (M, 3) uint8
    human_frame: np.ndarray              #: (M,) frame of each human point
    gravity: dict | None                 #: see :func:`load_gravity`
    bodies: dict[str, BodySource]        #: body source name -> poses, in sidebar order
    views: dict | None = None            #: calibrated rig cameras, see :func:`_load_view_cameras`
    videos: dict = field(default_factory=dict)   #: view name -> :class:`VideoFrames`
    extras: list = field(default_factory=list)   #: :class:`ForceArrows` a dataset reader added


def load_gravity(pred: Path) -> dict | None:
    """World-frame gravity from ``geocalib/gravity.npz`` (unit down + source flag).

    When the gravity stage fitted and accepted a ground plane, ``ground`` carries it (centroid,
    up-normal, data extent, angle to the prior down) so the viewer draws the plane the pipeline
    used; a rejected fit leaves ``ground`` ``None`` and puts the reason in ``ground_reject``.
    """
    f = pred / "geocalib" / "gravity.npz"
    if not f.exists():
        return None
    gd = np.load(f)
    g = np.asarray(gd["gravity_world"], np.float32)
    source = str(gd["source"]) if "source" in gd.files else "geocalib"
    ground = None
    if "ground_accepted" in gd.files and bool(gd["ground_accepted"]):
        ground = {"centroid": np.asarray(gd["ground_centroid"], np.float64),
                  "normal": np.asarray(gd["ground_normal"], np.float64),
                  "extent": float(gd["ground_extent"]),
                  "angle_deg": float(gd["ground_angle_deg"])}
    return {"g": g / (float(np.linalg.norm(g)) + 1e-9), "is_fallback": source == "fallback_down",
            "source": source,
            "spread": (float(gd["inlier_spread_median_deg"])
                       if "inlier_spread_median_deg" in gd.files else float("nan")),
            "ground": ground,
            "ground_reject": str(gd["ground_reject"]) if "ground_reject" in gd.files else ""}


def _load_view_cameras(pred: Path) -> dict | None:
    """The calibrated static cameras of a multi-view clip (``cameras.npz``), else ``None``.

    Written by the climb_wall_3 ground-truth chain: fixed, calibrated cameras in the metric world
    the geometry uses. ``offset`` is how far each view lags ``cam_left``, whose clock the timeline
    follows: view ``k``'s video frame ``t + offset[k]`` is the same instant as frame ``t``.
    """
    f = pred / "cameras.npz"
    if not f.exists():
        return None
    cd = np.load(f, allow_pickle=True)
    K = np.asarray(cd["K"], np.float64)                     # (V, 3, 3) pixel intrinsics
    wh = np.asarray(cd["image_size"], np.int64)             # (V, 2) width, height
    return {
        "names": [str(x) for x in cd["view_names"]],
        "extr": np.asarray(cd["extrinsics"], np.float64),   # (V, 4, 4) cam-from-world
        "fov_y": 2.0 * np.arctan(wh[:, 1] / (2.0 * K[:, 1, 1])),
        "aspect": wh[:, 0] / wh[:, 1],
        "offset": (np.asarray(cd["frame_offset"], np.int64) if "frame_offset" in cd.files
                   else np.zeros(len(wh), np.int64)),
    }


def fov_y_rad(transform) -> np.ndarray:
    """Per-frame vertical field of view (radians) from ``geometry/transform.npz``.

    VGGT records it (``fov_y_deg``); a calibrated rig saves only the pixel intrinsics, so there it
    is derived from the focal length and the image height.
    """
    if "fov_y_deg" in transform.files:
        return np.radians(transform["fov_y_deg"].astype(np.float64))
    fy = transform["intrinsics_px_orig"][:, 1, 1].astype(np.float64)
    return 2.0 * np.arctan(float(transform["image_height"]) / (2.0 * fy))


def camera_poses(extr: np.ndarray) -> np.ndarray:
    """cam-from-world OpenCV ``(T, 4, 4)`` -> camera-to-world poses ``(T, 7)`` ``[x, y, z, qx, qy, qz, qw]``."""
    return se3.inverse(se3.from_matrix(torch.as_tensor(extr, dtype=torch.float64))).numpy().astype(np.float32)


def _expand_frame_columns(arr: np.ndarray, names: list[str],
                          body_names: list[str]) -> np.ndarray | None:
    """Scatter per-contact-frame columns ``arr (P, N, F_sub, ...)`` into the body's frame order.

    A stage config may restrict detection to a subset of the body's contact frames (the peter rigs
    run hands and feet only); the viewer poses the full set, so a subset artifact is expanded by
    name, uncovered frames zero/False. ``None`` if a name is unknown to the body.
    """
    pos = {n: i for i, n in enumerate(body_names)}
    if any(n not in pos for n in names):
        return None
    full = np.zeros(arr.shape[:2] + (len(body_names),) + arr.shape[3:], arr.dtype)
    full[:, :, [pos[n] for n in names]] = arr
    return full


def fit_frames(arr: np.ndarray, n_frames: int, axis: int = 1) -> np.ndarray:
    """Truncate or zero-pad the frame ``axis`` of ``arr`` to ``n_frames`` (a multi-view fit is shorter than a clip)."""
    width = [(0, 0)] * arr.ndim
    width[axis] = (0, max(n_frames - arr.shape[axis], 0))
    return np.pad(np.take(arr, np.arange(min(n_frames, arr.shape[axis])), axis=axis), width)


def _hold_valid(q: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``q (P, T, nq)`` with every invalid frame replaced by the next valid one (the last at the end).

    Invalid frames are hidden, but forward kinematics still runs on them, and a stored invalid
    frame can hold all-zero quaternions, which pose to NaN.
    """
    q = q.copy()
    for p in range(len(q)):
        idx = np.flatnonzero(valid[p])
        if idx.size:
            src = idx[np.minimum(np.searchsorted(idx, np.arange(q.shape[1])), idx.size - 1)]
            q[p] = q[p, src]
    return q


def body_source(path: Path, n_frames: int, *, valid: np.ndarray | None = None) -> BodySource:
    """A world-frame body npz (``human_optim/kinematic_*.npz``, ``kindyn_*.npz``, or a dataset's ground truth in that schema).

    ``valid (P, N)`` replaces the file's own ``valid_mask`` (a dataset that stores visibility there,
    not whether a pose exists, passes its own).
    """
    d = np.load(path, allow_pickle=True)
    q = np.asarray(d["q"], np.float32)                                    # (P, N, 211)
    valid = np.asarray(d["valid_mask"] if valid is None else valid, bool)
    oids = (np.asarray(d["object_ids"], np.int32) if "object_ids" in d.files
            else np.arange(len(q), dtype=np.int32))
    valid = fit_frames(valid, n_frames)
    return BodySource(q=_hold_valid(fit_frames(q, n_frames), valid),
                      betas=np.asarray(d["betas"], np.float32).reshape(len(q), -1),
                      valid=valid, object_ids=oids, stored_frames=q.shape[1])


def classic_source(path: Path, body: Body, extr: np.ndarray) -> BodySource:
    """A camera-frame classic SMPL-X param file (``sam3d/smplx_params.npz``, ``geometry/prefit.npz``).

    Each person's per-frame ``betas`` are averaged over the valid frames into one identity, the
    params are packed into ``q`` with :meth:`Body.from_classic` (which adds the hand mean) and the
    per-frame ``world_from_cam`` (inverse VGGT extrinsic) is composed onto the free-flyer root, the way
    the kinematic stage builds its starting pose (the root joint sits at the origin, so this moves the
    whole body).
    """
    sx = np.load(path)
    keys = ("body_pose", "transl", "global_orient", "left_hand_pose", "right_hand_pose")
    P, N = sx["betas"].shape[:2]
    T = min(N, len(extr))
    valid = np.asarray(sx["valid_mask"], bool)[:, :T]
    q = np.zeros((P, T, body.nq), np.float32)
    betas = np.zeros((P, sx["betas"].shape[-1]), np.float32)
    world_from_cam = torch.as_tensor(camera_poses(extr[:T]))                # (T, 7)
    with torch.no_grad():
        for p in range(P):
            idx = np.flatnonzero(valid[p])
            if not idx.size:
                continue
            betas[p] = sx["betas"][p][idx].mean(0)
            # One identity for every frame: the same q as a per-frame betas batch, without a bake per frame.
            q_cam = body.from_classic(torch.as_tensor(betas[p]), *(torch.as_tensor(sx[k][p][idx]) for k in keys))
            q_cam[:, :7] = se3.compose(world_from_cam[idx], q_cam[:, :7])
            q[p, idx] = q_cam.numpy()
    oids = (np.asarray(sx["object_ids"], np.int32) if "object_ids" in sx.files
            else np.arange(P, dtype=np.int32))
    valid = fit_frames(valid, len(extr))
    return BodySource(q=_hold_valid(fit_frames(q, len(extr)), valid), betas=betas,
                      valid=valid, object_ids=oids, stored_frames=N)


def contact_labels(path: Path, body: Body, src: BodySource, *, artifact: bool = True) -> np.ndarray | None:
    """``frame_contact (P, T, F)`` in the body's frame order, from a contact artifact or a self-labelled body.

    ``artifact`` (``contacts_*.npz``, or a dataset's ground-truth labels in that schema) must carry the current label
    schema; a kinematics+dynamics body writes ``frame_contact`` next to ``q`` itself (``False``).
    The labels must be shaped (people, frames, names) like the body file and the file's own frame
    names (a subset of the body's contact frames is expanded by name). Anything older, of another
    shape, for other people, or for unknown frames is ignored with a warning: the body still
    draws, only its contact layer is missing.
    """
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    if "frame_contact" not in d.files:
        return None
    schema = int(np.asarray(d["contact_label_schema"]).item()) if "contact_label_schema" in d.files else -1
    oids = np.asarray(d["object_ids"], np.int32)
    contact = np.asarray(d["frame_contact"], bool)
    names = [str(x) for x in np.asarray(d["contact_frame_names"]).tolist()]
    expected = (len(src.object_ids), src.stored_frames, len(names))
    if contact.shape != expected:
        warnings.warn(f"ignoring incompatible contact artifact {path}: frame_contact {contact.shape}, "
                      f"body (people, frames, names) {expected}; rerun estimate_contacts",
                      RuntimeWarning, stacklevel=2)
        return None
    body_names = list(body.contact_frame_names)
    if names != body_names:
        contact = _expand_frame_columns(contact, names, body_names)
    if (artifact and schema != _CONTACT_SCHEMA) or contact is None or not np.array_equal(oids, src.object_ids):
        warnings.warn(f"ignoring contact labels {path}: schema {schema} (viewer reads {_CONTACT_SCHEMA}), "
                      f"other people or unknown frames; rerun estimate_contacts", RuntimeWarning, stacklevel=2)
        return None
    return fit_frames(contact, src.valid.shape[1])


def load_forces(path: Path, body: Body, n_frames: int) -> dict | None:
    """Forces and torques of a kinematics+dynamics body (``kindyn_1.npz``, or a dataset's ground-truth forces in that schema).

    ``joint_forces (P, T, J, 3)`` are the per-joint contact forces with each hand's finger forces
    summed onto its wrist (one arrow per gripping hand); ``frame_forces (P, T, F, 3)`` are the same
    forces per contact frame, in the body's frame order; ``torque (P, T, J)`` with ``torque_ref`` its
    95th percentile over valid frames (the colour scale); ``base_wrench (P, T, 6)`` is the root's
    residual force | torque in the root-local frame.
    """
    if not path.exists():
        return None
    dd = np.load(path, allow_pickle=True)
    if "torque_mag" not in dd.files:                  # a kinematic body: no forces
        return None
    torque = dd["torque_mag"].astype(np.float32)
    on = torque[dd["valid_mask"].astype(bool)]
    ref = float(np.percentile(on, 95)) if on.size else 1.0
    names = [str(n) for n in dd["joint_names"]]
    fw = dd["contact_forces_world"].astype(np.float32)
    for side in ("left", "right"):
        fingers = [i for i, n in enumerate(names)
                   if n.startswith(f"{side}_") and any(t in n for t in FINGER_TOKENS)]
        if fingers and f"{side}_wrist" in names:
            fw[:, :, names.index(f"{side}_wrist")] += fw[:, :, fingers].sum(axis=2)
            fw[:, :, fingers] = 0.0
    ff = None
    if "frame_forces" in dd.files:
        ff = dd["frame_forces"].astype(np.float32)
        ff_names = [str(x) for x in dd["contact_frame_names"]]
        if ff_names != list(body.contact_frame_names):
            ff = _expand_frame_columns(ff, ff_names, list(body.contact_frame_names))
    fit = lambda a: None if a is None else fit_frames(a, n_frames)  # noqa: E731
    return {"joint_forces": fit(fw), "frame_forces": fit(ff), "torque": fit(torque),
            "torque_ref": ref if ref > 1e-6 else 1.0,
            "base_wrench": fit(dd["base_wrench"].astype(np.float32)) if "base_wrench" in dd.files else None}


def _subsample_human_mask(fid: np.ndarray, obj: np.ndarray, max_per_frame: int) -> np.ndarray:
    """Boolean keep-mask capping human points to ``max_per_frame`` per (frame, climber).

    The human depth cloud is dense (~27k points per frame on some clips); a random cap per
    (frame, climber) keeps the visible shape at a fraction of the bytes, seeded for a stable draw.
    """
    rng = np.random.default_rng(0)
    keep = np.ones(len(fid), bool)
    key = fid.astype(np.int64) * (int(obj.max()) + 1) + obj.astype(np.int64)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    bounds = np.flatnonzero(np.diff(ks)) + 1
    for s, e in zip(np.concatenate(([0], bounds)), np.concatenate((bounds, [len(ks)]))):
        if e - s > max_per_frame:
            keep[rng.choice(order[s:e], (e - s) - max_per_frame, replace=False)] = False
    return keep


class VideoFrames:
    """Lazy ``step -> (h, w, 3)`` uint8 RGB frames of a video, for a BetterRobot ``ImagePanel``.

    ``frame_ids (T,)`` maps a step to its source frame (a uniformly subsampled clip, or a view's
    lag); a frame past the end of the video is black. Frames are decoded with cv2, the decoder the
    pipeline used, forward and in order while playing; a jump back or far ahead seeks (up to a
    second on a long-GOP 1080p clip), so every decoded frame is kept for the next visit. The longer
    side is downscaled to ``max_dim`` for a light sidebar pane.
    """

    def __init__(self, path: Path, frame_ids: np.ndarray, max_dim: int = 400) -> None:
        self.path, self.frame_ids, self.max_dim = path, np.asarray(frame_ids, np.int64), max_dim
        self._cap = None
        self._pos = 0              # source frame the next read returns
        self._shape = (2, 2, 3)
        self._kept: dict[int, np.ndarray] = {}

    def __call__(self, step: int) -> np.ndarray:
        import cv2

        if self._cap is None:
            self._cap = cv2.VideoCapture(str(self.path))
        index = int(self.frame_ids[min(step, len(self.frame_ids) - 1)])
        if index in self._kept:
            return self._kept[index]
        if not self._pos <= index <= self._pos + 60:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._pos = index
        while self._pos < index:
            self._cap.grab()
            self._pos += 1
        ok, bgr = self._cap.read()
        self._pos = index + 1
        if not ok:
            return np.zeros(self._shape, np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        s = self.max_dim / max(h, w)
        if s < 1.0:
            rgb = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        self._shape = rgb.shape
        self._kept[index] = rgb
        return rgb

    def close(self) -> None:
        """Release the decoder (the decoded frames stay); the next call reopens it."""
        if self._cap is not None:
            self._cap.release()
            self._cap, self._pos = None, 0


def _manifest_video(pred: Path) -> Path | None:
    """Source video the batch pipeline recorded for this clip (``_batch_manifest.jsonl``), if it still exists.

    ``scripts/pipeline.py`` appends one row per processed clip next to the clip's output tree, with
    the absolute ``source`` path, so a batch stored away from the repo still finds its footage.
    The last row wins (reruns append).
    """
    manifest = pred.parent / "_batch_manifest.jsonl"
    if not manifest.exists():
        return None
    found = None
    for line in manifest.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (row.get("stem") == pred.name or Path(row.get("out_dir", "")).name == pred.name) and row.get("source"):
            found = Path(row["source"])
    return found if found is not None and found.is_file() and found.suffix.lower() in _VIDEO_EXTS else None


def find_video(pred: Path, video_dir: Path | None) -> Path | None:
    """``<video_dir>/<clip-stem>.<ext>`` (default ``../data/examples/data/``), else the batch manifest's."""
    vdir = video_dir if video_dir is not None else EXAMPLES / "data"
    return next((vdir / f"{pred.name}{ext}" for ext in _VIDEO_EXTS if (vdir / f"{pred.name}{ext}").exists()),
                None) or _manifest_video(pred)


def playback_fps(native_fps: float, frame_ids: np.ndarray) -> float | None:
    """Real-time rate of one frame step: the native fps, scaled when a long clip was uniformly subsampled."""
    span = float(frame_ids[-1] - frame_ids[0]) if len(frame_ids) > 1 else 0.0
    if native_fps <= 0.0:
        return None
    return native_fps * (len(frame_ids) - 1) / span if span > 0.0 else native_fps


def _video_native_fps(path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps


def load_clip_data(pred: Path, *, body: Body | None, human_max_points: int = 0, video: bool = True,
                   video_dir: Path | None = None) -> ClipData:
    """Load one ``out/<stem>/`` clip: geometry, cameras, gravity, video and (with a ``body``) every body source."""
    geo = pred / "geometry"
    scene = np.load(geo / "scene.npz")
    transform = np.load(geo / "transform.npz")
    if "metric" in transform.files and not bool(transform["metric"]):
        print("[viewer] WARNING: geometry is UP-TO-SCALE (run estimate_scale): "
              "metric bodies will not align with the scene")
    extr = transform["extrinsics"]
    T = len(extr)

    # Show the scene cloud the optimizer and estimate_contacts run against: drop the lowest-confidence
    # quantile of voxels (raw VGGT confidence has no fixed scale, hence a quantile).
    sp, sco = scene["points"], scene["colors"]
    sn = scene["normals"] if "normals" in scene.files else None
    if "confidence" in scene.files and len(scene["confidence"]) == len(sp) and len(sp):
        keep = scene["confidence"] >= np.quantile(scene["confidence"], _SCENE_CONF_DROP_Q)
        sp, sco, sn = sp[keep], sco[keep], None if sn is None else sn[keep]

    # Per-frame human depth points; absent on clips fused from calibrated cameras (climb_wall_3).
    hp, hc = np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    hf, ho = np.zeros(0, np.int32), np.zeros(0, np.int32)
    if (geo / "human.npz").exists():
        human = np.load(geo / "human.npz")
        hp, hc, hf, ho = human["points"], human["colors"], human["frame_id"], human["obj_id"]
    if human_max_points and len(hp):
        keep = _subsample_human_mask(hf, ho, human_max_points)
        hp, hc, hf = hp[keep], hc[keep], hf[keep]

    # Body sources in sidebar order: the raw converter output, the scale stage's pre-fit, then the
    # refined bodies, each with the contact labels measured on it.
    bodies: dict[str, BodySource] = {}
    if body is not None:
        for name, rel in (("original", "sam3d/smplx_params.npz"), ("prefit", "geometry/prefit.npz")):
            if (pred / rel).exists():
                bodies[name] = classic_source(pred / rel, body, extr)
        for name, fn, labels in (("kinematics", "kinematic_1.npz", "contacts_1.npz"),
                                 ("kinematics_2", "kinematic_2.npz", "contacts_2.npz"),
                                 ("kindyn_1", "kindyn_1.npz", None)):
            path = pred / "human_optim" / fn
            if not path.exists():
                continue
            src = bodies[name] = body_source(path, T)
            src.contact = (contact_labels(path, body, src, artifact=False) if labels is None
                           else contact_labels(pred / "human_optim" / labels, body, src))
            src.forces = load_forces(path, body, T)

    # Video pane: a calibrated rig plays each view's own footage, shifted by its lag; every other
    # clip has the one source video, picked frame by frame the way VGGT sampled it.
    views = _load_view_cameras(pred)
    frame_ids = (transform["frame_indices"].astype(np.int64)[:T] if "frame_indices" in transform.files
                 else np.arange(T))
    videos: dict[str, VideoFrames] = {}
    fps = None
    if video and views is not None:
        for name, off in zip(views["names"], views["offset"]):
            path = pred / "views" / name / "video.mp4"
            if path.exists():
                videos[name] = VideoFrames(path, np.arange(T) + int(off))
    elif video:
        path = find_video(pred, video_dir)
        if path is not None:
            videos["video"] = VideoFrames(path, frame_ids)
            fps = playback_fps(_video_native_fps(path), frame_ids)
    if fps is None and "fps" in transform.files:
        fps = playback_fps(float(transform["fps"]), frame_ids)
    return ClipData(
        name=pred.name, n_frames=T, fps=fps, extr=extr, fov_y=fov_y_rad(transform),
        aspect=float(transform["image_width"]) / float(transform["image_height"]),
        scene_points=sp, scene_colors=sco, scene_normals=sn,
        scene_voxel=float(scene["voxel_size"]) if "voxel_size" in scene.files else 0.0,
        human_points=hp, human_colors=hc, human_frame=hf,
        gravity=load_gravity(pred), bodies=bodies, views=views, videos=videos)
