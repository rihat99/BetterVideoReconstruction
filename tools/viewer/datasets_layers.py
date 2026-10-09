"""Viewer readers that belong to one dataset, kept apart from the generic ``out/<stem>/`` viewer.

:func:`read_clip` is the reader ``scripts/view_3d.py`` hands the app in place of ``load_clip_data``:

* BEDLAM2 — a scene's ``gt/`` ground truth stands in for the whole stage chain
  (:func:`load_bedlam_data`, ``--bedlam``).
* Measured forces — the peter rigs' wall sensors and the OpenCap force plates
  (``sensor_forces.npz``, :func:`load_sensor_forces`) and our OpenCap forces in the plates'
  format (``estimated_forces.npz``, :func:`load_plate_estimate`), added to the clip as
  :class:`~tools.viewer.loading.ForceArrows` beside the solved ones (:func:`measured_forces`).
* The ClimbingVideos / BEDLAM2 scene DB — :func:`run_scene_db` aliases the selected scenes into an
  ``out/<stem>/``-shaped symlink farm and serves it (``--scenes``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tools.body import Body

from .loading import (
    ClipData,
    ForceArrows,
    VideoFrames,
    body_source,
    camera_poses,
    contact_labels,
    find_video,
    fit_frames,
    fov_y_rad,
    load_clip_data,
    load_forces,
    playback_fps,
)

# Measured channels below this many newtons are not drawn (the sensors never read exactly zero).
_SENSOR_FLOOR_N = 10.0
_SENSOR_RGB, _PLATE_RGB = (1.0, 0.59, 0.12), (0.16, 0.43, 1.0)     # measured orange, ours blue
_LIMB_JOINT = {"left_hand": "left_wrist", "right_hand": "right_wrist",
               "left_foot": "left_foot", "right_foot": "right_foot"}
#: The body-source name of a BEDLAM2 scene: its motion is the dataset's ground truth.
BEDLAM_BODY = "ground truth"
#: What makes a directory a BEDLAM2 scene (``import_bedlam2`` writes it).
BEDLAM_MARKER = Path("gt") / "smplx.npz"


def read_clip(pred: Path, *, body: Body | None, **opts) -> ClipData:
    """``load_clip_data`` with the datasets plugged in: a BEDLAM2 scene reads its ``gt/``, and any
    other clip gets its measured forces. ``opts`` are ``load_clip_data``'s."""
    if (pred / BEDLAM_MARKER).exists():
        return load_bedlam_data(pred, body=body, video=opts["video"], video_dir=opts["video_dir"])
    data = load_clip_data(pred, body=body, **opts)
    data.extras += measured_forces(pred, data, body)
    return data


def load_sensor_forces(pred: Path) -> dict | None:
    """Measured forces (``sensor_forces.npz``) in the world frame, in the reaction sense (force on the person).

    ``scripts/peter_sensor_forces.py`` writes a wall rig's load-cell channels (each a corner's
    reaction, limb-dominated but mixed); the climb_wall_3 ground truth keeps a per-limb board reading
    at the clip root; ``scripts/opencap_sensor_forces.py`` writes force plates, which also know where
    the force acts (``cop_world``, NaN where the plate saw nothing).
    """
    f = next((c for c in (pred / "human_optim" / "sensor_forces.npz", pred / "sensor_forces.npz")
              if c.exists()), None)
    if f is None:
        return None
    dd = np.load(f, allow_pickle=True)
    return {"rw": dd["reaction_world"].astype(np.float32),                     # (N, C, 3) per channel
            "limbs": [str(x) for x in dd["limbs"]],                              # channel -> limb
            "cop": dd["cop_world"].astype(np.float32) if "cop_world" in dd.files else None}


def load_plate_estimate(pred: Path) -> dict | None:
    """Our solved foot forces in the force plates' format (``estimated_forces.npz``), rotated into our world.

    ``scripts/export_opencap_mot.py`` collapses each foot's solved forces to one force at one centre
    of pressure in the dataset's OpenSim ground frame; the file carries the ``R``/``t`` back.
    """
    f = pred / "human_optim" / "estimated_forces.npz"
    if not f.exists():
        return None
    dd = np.load(f, allow_pickle=True)
    R = np.asarray(dd["R_world_from_mocap"], np.float64)
    t = np.asarray(dd["t_world_from_mocap"], np.float64)
    return {"rw": (dd["force"].astype(np.float64) @ R.T).astype(np.float32),        # (N, 2, 3)
            "cop": (dd["cop"].astype(np.float64) @ R.T + t).astype(np.float32),     # (N, 2, 3)
            "limbs": [str(x) for x in dd["limbs"]]}


def measured_forces(pred: Path, data: ClipData, body: Body | None) -> list[ForceArrows]:
    """The clip's measured (orange) and plate-format (blue) force arrows, shown with its first
    kinematics+dynamics body source and drawn on that source's first person."""
    name = next((n for n, src in data.bodies.items() if src.forces is not None), None)
    sensors, plates = load_sensor_forces(pred), load_plate_estimate(pred)
    if name is None or not data.bodies[name].valid[0].any() or (sensors is None and plates is None):
        return []
    src = data.bodies[name]
    with torch.no_grad():
        joints = body.shaped(torch.as_tensor(src.betas[0])).forward(torch.as_tensor(src.q[0])).joints.cpu().numpy()
    out = []
    for kind, color, arrows in (("sensor", _SENSOR_RGB, sensor_arrows(sensors, joints, body.joint_names)),
                                ("plate", _PLATE_RGB, plate_arrows(plates, data.n_frames))):
        if arrows is not None:
            origins, vectors, valid = arrows
            out.append(ForceArrows(f"{kind}_forces", kind, origins, vectors, color, valid & src.valid[0][:, None],
                                   source=name))
    return out


def sensor_arrows(sensors: dict | None, joints: np.ndarray,
                  joint_names) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Measured forces as ``(origins, vectors, valid)``, ``(T, C, 3)`` / ``(T, C, 3)`` newtons / ``(T, C)``.

    A plate's arrow starts at its centre of pressure on the floor; a wall cell's at the joint of its
    limb on the body ``joints (T, J, 3)``.
    """
    if sensors is None:
        return None
    T = len(joints)
    names = list(joint_names)
    vectors = fit_frames(sensors["rw"], T, axis=0)
    origins = np.zeros_like(vectors)
    for c, limb in enumerate(sensors["limbs"]):
        if _LIMB_JOINT.get(limb) in names:
            origins[:, c] = joints[:, names.index(_LIMB_JOINT[limb])]
    if sensors["cop"] is not None:
        cop = fit_frames(sensors["cop"], T, axis=0)
        on_plate = np.isfinite(cop).all(-1)
        origins[on_plate] = cop[on_plate]
    known = np.array([_LIMB_JOINT.get(limb) in names for limb in sensors["limbs"]])
    valid = (np.linalg.norm(vectors, axis=-1) > _SENSOR_FLOOR_N) & known
    return origins, vectors, valid


def plate_arrows(plates: dict | None, n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Our forces in the plates' format at our centre of pressure, as :func:`sensor_arrows`."""
    if plates is None:
        return None
    vectors, cop = fit_frames(plates["rw"], n_frames, axis=0), fit_frames(plates["cop"], n_frames, axis=0)
    valid = (np.linalg.norm(vectors, axis=-1) > _SENSOR_FLOOR_N) & np.isfinite(cop).all(-1)
    return np.nan_to_num(cop), vectors, valid


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
    ``scripts/label_bedlam2.py`` has run it also holds ``bboxes.npz`` (who is usable),
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


# --- the scene DB (``--scenes`` / ``--bedlam``) -------------------------------------------------

#: The stage feature dirs the loaders read, each aliased as ``<farm>/<stage>``.
_FARM_STAGES = ("geometry", "sam3d", "vggt", "human_optim", "geocalib")


def _link_scene_farm(paths, row, farm_root: Path, *, stages: tuple[str, ...],
                     needs: tuple[str, str]) -> bool:
    """Alias one DB scene's stage-first feature dirs into an ``out/<stem>/``-shaped symlink farm.

    Builds ``<farm_root>/<scene_id>/<stage>`` links (skipping stages never produced) plus the source
    clip as ``<farm_root>/<scene_id><ext>`` for the video pane. ``needs`` is the ``(stage, file)``
    the viewer cannot draw without; a scene missing it is skipped.
    """
    sid = row["scene_id"]
    if not (paths.feature_dir(needs[0], sid) / needs[1]).exists():
        return False
    farm = farm_root / sid
    farm.mkdir(parents=True, exist_ok=True)
    for stage in stages:
        src, link = paths.feature_dir(stage, sid), farm / stage
        if src.exists() and not link.exists():
            link.symlink_to(src)
    clip = paths.clip_path(row)
    vlink = farm_root / f"{sid}{clip.suffix}"
    if clip.exists() and not vlink.exists():
        vlink.symlink_to(clip)
    return True


def run_scene_db(args, view) -> None:
    """Browse the ClimbingVideos (``args.scenes == "climbing"``) or BEDLAM2 (``"bedlam"``) scene DB.

    Selects the target scenes, aliases them into a throwaway symlink farm and serves it through
    ``view(root, video_dir=...)``, the same entry point a local ``out/`` tree uses.
    """
    import atexit
    import shutil
    import tempfile

    from scripts import scene_db

    bedlam = args.scenes == "bedlam"
    data_root = args.data_root or (scene_db.BEDLAM_DATA_ROOT if bedlam else scene_db.DEFAULT_DATA_ROOT)
    categories = () if bedlam else scene_db.parse_int_csv(args.categories)
    rope = None if bedlam or args.rope_supported < 0 else args.rope_supported
    stages, needs = (("gt",), ("gt", "smplx.npz")) if bedlam else (_FARM_STAGES, ("geometry", "scene.npz"))
    paths = scene_db.ScenePaths(data_root)
    with paths.connect() as conn:
        rows = scene_db.select_targets(conn, categories=categories, rope_supported=rope)
    rows = scene_db.resolve_rows(rows, scenes=args.scene_ids, sample=args.sample, limit=args.limit,
                                 seed=args.seed)
    if not rows:
        print("[view_3d] no target scenes match the selection", flush=True)
        return
    farm_root = Path(tempfile.mkdtemp(prefix="bvr_view_"))
    atexit.register(shutil.rmtree, farm_root, ignore_errors=True)   # symlinks only; targets untouched
    n = sum(_link_scene_farm(paths, row, farm_root, stages=stages, needs=needs) for row in rows)
    if not n:
        print(f"[view_3d] none of {len(rows)} target scene(s) have {needs[0]}/{needs[1]} under {data_root}",
              flush=True)
        return
    print(f"[view_3d] prepared {n}/{len(rows)} scene(s) from {data_root}", flush=True)
    view(farm_root, video_dir=farm_root)
