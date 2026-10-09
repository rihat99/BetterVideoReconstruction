"""The BEDLAM2 viewer reader: a scene's ``gt/`` ground truth stands in for the whole stage chain.

``scripts/view_3d.py`` reads a directory holding :data:`BEDLAM_MARKER` with :func:`load_bedlam_data`
(``--bedlam`` browses the whole DB through :func:`run_scene_db`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from bvr.body import Body
from bvr.viewer.loading import (
    ClipData,
    VideoFrames,
    body_source,
    camera_poses,
    contact_labels,
    find_video,
    fov_y_rad,
    load_forces,
    playback_fps,
)
from bvr.config import load_config
from datasets.bedlam2.paths import BEDLAM2_OUR
from datasets.climbingvideos.viewer import select_scenes, serve_scene_farm

#: The body-source name of a BEDLAM2 scene: its motion is the dataset's ground truth.
BEDLAM_BODY = "ground truth"
#: What makes a directory a BEDLAM2 scene (``import_bedlam2`` writes it).
BEDLAM_MARKER = Path("gt") / "smplx.npz"


def _bedlam_gravity(gt: Path, centers: np.ndarray, pelvis: list) -> dict:
    """World-frame gravity + the floor BEDLAM2 placed the people on.

    The floor is exact and unbounded, so all that is left is where to draw a patch of it and how
    big. BEDLAM2's stored centroid is the world origin at floor height, and an Unreal level can sit
    hundreds of metres from its origin, so the patch is re-centred under the action (the cameras and
    every pelvis track, ``None`` for a person never posed) at the stored height. ``extent`` is the
    floor-patch diameter the gravity stage would have measured, and the viewer draws a grid ten
    times that, so report the diameter that clears the action by a quarter of its span on every
    side, with an 8 m floor under a scene that barely moves.
    """
    gd = np.load(gt / "gravity.npz")
    plane = np.asarray(gd["ground_centroid"], np.float64)
    normal = np.asarray(gd["ground_normal"], np.float64)
    action = np.concatenate([centers, *[p for p in pelvis if p is not None]]).astype(np.float64)
    centre = action.mean(0)
    centre -= normal * ((centre - plane) @ normal)       # slide it onto the real floor
    flat = action - centre
    flat -= np.outer(flat @ normal, normal)              # distance along the floor only
    radius = float(np.linalg.norm(flat, axis=1).max()) if len(flat) else 0.0
    g = np.asarray(gd["gravity_world"], np.float32)
    return {"g": g / (float(np.linalg.norm(g)) + 1e-9), "is_fallback": False,
            "source": str(gd["source"]), "spread": float("nan"), "ground_reject": "",
            "ground": {"centroid": centre, "normal": normal, "extent": max(0.25 * radius, 0.8),
                       "angle_deg": float(gd["ground_angle_deg"])}}


def load_bedlam_data(pred: Path, *, body: Body, video: bool, video_dir: Path | None) -> ClipData:
    """One BEDLAM2 scene from its ``gt/`` directory: cameras, gravity + floor, and the ground-truth bodies.

    ``gt/`` holds ``camera.npz`` (the per-frame camera, in ``geometry/transform.npz``'s schema),
    ``gravity.npz`` and ``smplx.npz`` (every person's world-frame ``q`` and ``betas``). Where
    ``datasets/bedlam2/label_bedlam2.py`` has run it also holds ``bboxes.npz`` (who is usable),
    ``contacts.npz`` (which skin points are on the floor) and ``forces.npz`` (the forces solved at
    the ground-truth pose). The environments ship no geometry, so there is no scene cloud.
    """
    gt = pred / "gt"
    transform = np.load(gt / "camera.npz")
    extr = transform["extrinsics"]
    T = len(extr)
    # BEDLAM2 puts visibility in ``valid_mask``, not whether a pose exists: pose every frame, so a
    # person who walks out of shot keeps their place, and drop only the people ``label_bedlam2``
    # found no usable frame for.
    valid = np.ones(np.load(gt / "smplx.npz")["q"].shape[:2], bool)
    if (gt / "bboxes.npz").exists():
        valid &= np.load(gt / "bboxes.npz")["person_valid"][:, None]
    src = body_source(gt / "smplx.npz", T, valid=valid)
    src.contact = contact_labels(gt / "contacts.npz", body, src)
    src.forces = load_forces(gt / "forces.npz", body, T)
    pelvis = [src.q[p, src.valid[p], :3] if src.valid[p].any() else None for p in range(len(src.q))]

    videos = {}
    path = find_video(pred, video_dir) if video else None
    if path is not None:
        videos["video"] = VideoFrames(path, np.arange(T))
    fps = playback_fps(float(transform["fps"]), np.arange(T))
    empty = np.zeros((0, 3), np.float32)
    return ClipData(
        name=pred.name, n_frames=T, fps=fps, extr=extr, fov_y=fov_y_rad(transform),
        aspect=float(transform["image_width"]) / float(transform["image_height"]),
        scene_points=empty, scene_colors=empty.astype(np.uint8), scene_normals=None, scene_voxel=0.0,
        human_points=empty, human_colors=empty.astype(np.uint8), human_frame=np.zeros(0, np.int32),
        gravity=_bedlam_gravity(gt, camera_poses(extr)[:, :3], pelvis),
        bodies={BEDLAM_BODY: src}, videos=videos)


#: Which scenes to browse: the ``[viewer]`` table (data_root, scene_ids, sample, limit, seed).
BEDLAM_CONFIG = Path(__file__).resolve().parent / "configs" / "bedlam.toml"


def run_scene_db(view, config_path: Path = BEDLAM_CONFIG) -> None:
    """Browse the BEDLAM2 ground truth: every scene of the DB under ``[viewer].data_root``
    (default ``BEDLAM2_our``), narrowed by its scene_ids / sample / limit knobs."""
    knobs = load_config(config_path).get("viewer", {})
    data_root = Path(knobs["data_root"]) if knobs.get("data_root") else BEDLAM2_OUR
    rows = select_scenes(data_root, categories=(), rope_supported=None,
                         scene_ids=knobs.get("scene_ids") or None,
                         sample=int(knobs["sample"]) if knobs.get("sample", "") != "" else None,
                         limit=int(knobs["limit"]) if knobs.get("limit", "") != "" else None,
                         seed=int(knobs.get("seed", 0)))
    serve_scene_farm(view, data_root, rows, stages=("gt",), needs=("gt", "smplx.npz"))
