"""Import BEDLAM2 into the ClimbingVideos on-disk layout.

BEDLAM2 ships no per-frame label file: the ground truth is *composed* from three
sources — ``be_seq.csv`` (which body stands where, per sequence), the render
database (per-image camera pose + intrinsics) and the AMASS-format motion npz
(SMPL-X pose over time). This script does that composition once and lays the
result out the way the repo's corpus tools expect.

Source (read-only)::

    <SRC>/b2_imagedata_download/<job>_mp4.tar                    video, <job>/mp4/seq_NNNNNN.mp4
    <SRC>/b2_imagedata_download/<job>_gt_..._meta_csv.tar.gz     <job>/be_seq.csv
    <SRC>/bedlam2_render_database-20250318/bedlam2.sqlite        all 8,048,411 gt_camera rows
    <SRC>/b2_motions_npz_training/motions_npz_training/*.npz     9,952 SMPL-X motions

Output::

    <DST>/scenes/scenes.db                              sqlite, ClimbingVideos schema v8
    <DST>/scenes/<aa>/<bb>/<scene_id>.mp4               the untouched render
    <DST>/frames/<aa>/<bb>/<scene_id>/NNNNNN.jpg        0-based, 6-digit, JPEG q95
    <DST>/features/gt/<aa>/<bb>/<scene_id>/camera.npz   extrinsics + intrinsics per frame
    <DST>/features/gt/<aa>/<bb>/<scene_id>/gravity.npz  world down vector + ground plane
    <DST>/features/gt/<aa>/<bb>/<scene_id>/smplx.npz    BetterHuman q (P,N,211) per person
    <DST>/_meta/<job>/be_seq.csv                        provenance copy of the source table
    <DST>/_plan.json                                    the selected scene table

``scene_id`` is ``<2-char base36 render-job index><6-digit sequence>`` (e.g.
``0a000002``), so ``aa = scene_id[:2]`` is the render job and the tree has 59
first-level directories. The readable job name lives in the DB's ``video_id``.

Selection — only sequences whose ground truth is complete:

* every body in ``be_seq.csv`` has a released motion npz (BEDLAM2 withholds
  2,123 of 12,075 animations to form its test split, and those people are still
  visible in frame, so a partially-labelled sequence would contain unannotated humans);
* not an egocentric ``vcam_avp`` render job;
* not one of the four ``upperbody`` close-up jobs;
* not flagged ``testset`` in the render database.

18,076 sequences / 5,493,090 frames / 26,178 people survive.

Three things the ground truth does *not* give you. BEDLAM2 ships one
``ground_height_world`` per sequence, so ``gravity.npz``'s floor is wrong for
anyone standing on a step or platform. Measured over all 26,178 people: the body's
lowest point sits a median 7.6 mm *below* the plane (shoe thickness -- the mesh is
naked), exactly **one** person never comes within 5 cm of it, and 333 (1.27%) spend
most of their clip more than 2 cm above it. An earlier version of this line claimed
5% of people never come within 5 cm; that is wrong by three orders of magnitude and
no definition of the gap reproduces it. The classic
``body_pose`` arrays are AMASS axis-angle copied verbatim, including rotation
vectors past 2*pi, so they do not match ``body.to_classic(betas, q)`` elementwise
— ``q`` is the canonical form. And no *stage* in this repo reads ``features/gt``;
it is a ground-truth tree for training, not a drop-in for the corpus chain. The
viewer does read it: ``scripts/view_3d.py --bedlam``.

World frame: Unreal's world, mapped to OpenCV axes by ``unreal2cv2`` and metres.
Up is ``-y``, so gravity is ``+y`` — the same convention the geocalib stage uses.
``camera.npz:extrinsics`` is camera-from-world, and the bodies are in that world.

Shape: BEDLAM2 renders through the *locked-head* SMPL-X model with 16 betas; the
pipeline body is the standard model with 10. Going from one to the other costs
0.7 mm mean / 1.8 mm p95 vertex error on the real BEDLAM2 shapes (0.1 px mean
reprojection, and the >5 mm tail is head and face only), so the importer
truncates and keeps the original 16 in ``betas_bedlam2``.

Visibility: BEDLAM2 places crowd members outside the camera frustum and gives
them ground truth anyway — 370,939 of 7,016,224 person-frames (5.3%) have no
body pixels at all, and 1,100 of 26,178 people (4.2%) are never visible in a
single frame of their clip. So
``valid_mask`` is *measured*, not assumed: ``visible_fraction`` is the share of
the body's 10,475 vertices inside the image that frame, and ``valid_mask`` is
``visible_fraction > 0``. Threshold it as you like; occlusion is still not
modelled, so a visible body may still be fully behind something.

Hands: BEDLAM2 renders with ``flat_hand_mean=True`` (absolute hand pose), while
``Body.from_classic`` adds the model's hand mean, so the mean is subtracted on
the way in. Skipping that costs 6.5 mm mean / 93 mm max on the hands.

Usage::

    uv run python scripts/import_bedlam2.py --step plan
    uv run python scripts/import_bedlam2.py --step scenes   --workers 12
    uv run python scripts/import_bedlam2.py --step features --workers 48
    uv run python scripts/import_bedlam2.py --step frames   --workers 64
    uv run python scripts/import_bedlam2.py --step db
    uv run python scripts/import_bedlam2.py --step verify   --sample 12   # writes _verify/*.png

``vlm_category`` carries the environment (1..16, mapped in ``schema_meta``), not a
climbing class, so a corpus config over this root wants ``categories = []``;
``scene_db.select_targets(conn, categories=(), rope_supported=0)`` returns all 18,076.
    uv run python scripts/import_bedlam2.py --step all
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.paths import DATA_ROOT  # noqa: E402

SRC = DATA_ROOT / "BEDLAM2"
DST = DATA_ROOT / "BEDLAM2_our"

IMAGE_DIR = SRC / "b2_imagedata_download"
RENDER_DB = SRC / "bedlam2_render_database-20250318" / "bedlam2.sqlite"
MOTION_DIR = SRC / "b2_motions_npz_training" / "motions_npz_training"
LOCKED_HEAD = Path("/home/rikhat.akizhanov/better/BetterHuman/models/smplx_lockedhead/SMPLX_NEUTRAL.npz")

FPS = 30.0
JPEG_QUALITY = 95
NUM_BETAS = 10
SCHEMA_VERSION = "8"

#: Unreal renders every job through a 36.0 x 20.25 mm sensor; a portrait job
#: rotates the camera instead of the sensor, so the axes swap on the way out.
SENSOR_LANDSCAPE = (36.0, 20.25)
SENSOR_PORTRAIT = (20.25, 36.0)
SIZE_LANDSCAPE = (1280, 720)
SIZE_PORTRAIT = (720, 1280)

#: Render jobs shot from a head-mounted virtual camera — excluded by request.
EGOCENTRIC_MARK = "vcam_avp"
#: Extreme upper-body / head close-ups where the body is mostly out of frame.
UPPERBODY_JOBS = frozenset({
    "20250211_1_250_ai1105_upperbody",
    "20250212_1_250_yakohama_upperbody",
    "20250214_1_250_chemicalplant_upperbody",
    "20250218_2-3_250_middleeast_upperbody",
})

#: A shot counts as locked only if position, orientation and focal length are all
#: constant; BEDLAM2's "static location" jobs still carry 1-3 cm of handheld shake.
STATIC_TRANSLATION_M = 0.01
STATIC_ROTATION_DEG = 0.1
STATIC_FOCAL_MM = 0.01

#: Frames posed at once when measuring visibility; a full-length clip of vertices
#: would be ~240 MB per worker.
VISIBILITY_CHUNK = 256

_BASE36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def job_code(job_index: int) -> str:
    """Two base36 characters identifying a render job (66 jobs fit in 36**2)."""
    return _BASE36[job_index // 36] + _BASE36[job_index % 36]


def make_scene_id(job_index: int, seq_index: int) -> str:
    """``<job code><seq:06d>`` — the primary key and the sharding source."""
    return f"{job_code(job_index)}{seq_index:06d}"


def shard(scene_id: str) -> tuple[str, str]:
    """``scene_id`` → its two-level shard, matching ``scripts/scene_db.py``."""
    return scene_id[:2], scene_id[2:4]


def unreal2cv2(points: np.ndarray) -> np.ndarray:
    """Unreal centimetres ``(...,3)`` → OpenCV-axis metres: ``(x,y,z) -> (y,-z,x)/100``."""
    return np.roll(points, 2, axis=-1) * np.array([1.0, -1.0, 1.0]) / 100.0


def axis_rotation(axis: int, degrees: np.ndarray | float) -> Rotation:
    """Rotation(s) about one world axis — the unambiguous form of ``from_euler``."""
    angles = np.atleast_1d(np.asarray(degrees, dtype=np.float64))
    rotvec = np.zeros((angles.size, 3))
    rotvec[:, axis] = np.radians(angles)
    rotation = Rotation.from_rotvec(rotvec)
    return rotation[0] if np.ndim(degrees) == 0 else rotation


def parse_comment(comment: str) -> dict[str, str]:
    """``be_seq.csv`` packs its metadata as ``key=value;key=value`` in one column."""
    return dict(kv.split("=", 1) for kv in comment.split(";") if "=" in kv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_npz(path: Path, **arrays) -> None:
    """``np.savez`` through a temporary file, so a crash never leaves a readable stub."""
    tmp = path.with_name(path.name + ".part.npz")
    np.savez(tmp, **arrays)
    tmp.rename(path)


def open_render_db() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{RENDER_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# step: plan
# ---------------------------------------------------------------------------

def extract_be_seq(job: str) -> Path:
    """Copy ``<job>/be_seq.csv`` out of the ground-truth tarball (it is member 0)."""
    out = DST / "_meta" / job / "be_seq.csv"
    if out.is_file():
        return out
    tar_path = IMAGE_DIR / f"{job}_gt_centersubframe_exr_meta_csv.tar.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if member.name.endswith("/be_seq.csv"):
                source = tar.extractfile(member)
                if source is None:
                    raise ValueError(f"{tar_path}: be_seq.csv is not a regular file")
                tmp = out.with_suffix(".part")
                tmp.write_bytes(source.read())
                tmp.rename(out)
                return out
    raise ValueError(f"{tar_path}: no be_seq.csv member")


def read_be_seq(path: Path) -> list[dict]:
    """Parse ``be_seq.csv`` into one record per sequence, bodies attached."""
    sequences: list[dict] = []
    current: dict | None = None
    for row in csv.DictReader(open(path)):
        if row["Type"] == "Group":
            meta = parse_comment(row["Comment"])
            current = {
                "seq": meta["sequence_name"],
                "n_frames": int(meta["frames"]),
                "ground_height_world": float(meta.get("ground_height_world", "nan")),
                "hdri": meta.get("hdri", ""),
                "bodies": [],
            }
            sequences.append(current)
        elif row["Type"] == "Body" and current is not None:
            meta = parse_comment(row["Comment"])
            current["bodies"].append({
                "name": row["Body"],
                "x": float(row["X"]), "y": float(row["Y"]), "z": float(row["Z"]),
                "yaw": float(row["Yaw"]),
                "start_frame": int(meta["start_frame"]),
            })
    return sequences


def build_plan() -> list[dict]:
    """Select the fully-labelled, non-egocentric, non-close-up training sequences."""
    released = {p.stem for p in MOTION_DIR.glob("*.npz")}
    conn = open_render_db()
    jobs = [dict(row) for row in conn.execute(
        "SELECT name, environment, batch, camera_movement, num_sequences, num_images "
        "FROM renderjob ORDER BY name")]
    db_sequences = {
        (row["renderjob"], row["sequence_id"]): dict(row)
        for row in conn.execute(
            "SELECT renderjob, sequence_id, num_images, num_bodies, testset, affected FROM sequence")
    }
    conn.close()

    environments = sorted({job["environment"] for job in jobs})
    env_code = {name: index + 1 for index, name in enumerate(environments)}

    plan: list[dict] = []
    skipped = {"egocentric": 0, "upperbody": 0, "testset": 0, "incomplete": 0}
    for job_index, job in enumerate(jobs):
        name = job["name"]
        if EGOCENTRIC_MARK in job["camera_movement"]:
            skipped["egocentric"] += job["num_sequences"]
            continue
        if name in UPPERBODY_JOBS:
            skipped["upperbody"] += job["num_sequences"]
            continue
        portrait = "portrait mode" in job["camera_movement"]
        for record in read_be_seq(extract_be_seq(name)):
            seq_index = int(record["seq"].rsplit("_", 1)[1])
            db_row = db_sequences[(name, seq_index)]
            if db_row["testset"]:
                skipped["testset"] += 1
                continue
            if not record["bodies"] or any(b["name"] not in released for b in record["bodies"]):
                skipped["incomplete"] += 1
                continue
            if record["n_frames"] != db_row["num_images"]:
                raise ValueError(f"{name}/{record['seq']}: be_seq frames "
                                 f"{record['n_frames']} != db num_images {db_row['num_images']}")
            plan.append({
                "scene_id": make_scene_id(job_index, seq_index),
                "job": name,
                "job_index": job_index,
                "seq": record["seq"],
                "seq_index": seq_index,
                "n_frames": record["n_frames"],
                "portrait": portrait,
                "environment": job["environment"],
                "environment_code": env_code[job["environment"]],
                "batch": job["batch"],
                "camera_movement": job["camera_movement"],
                "ground_height_world": record["ground_height_world"],
                "hdri": record["hdri"],
                "bodies": record["bodies"],
            })

    DST.mkdir(parents=True, exist_ok=True)
    payload = {"environments": env_code, "scenes": plan}
    (DST / "_plan.json").write_text(json.dumps(payload))
    frames = sum(s["n_frames"] for s in plan)
    people = sum(len(s["bodies"]) for s in plan)
    print(f"[plan] {len(plan):,} sequences  {frames:,} frames  {people:,} people  "
          f"{frames / FPS / 3600:.1f} h", flush=True)
    print(f"[plan] skipped: {skipped}", flush=True)
    return plan


def load_plan() -> list[dict]:
    payload = json.loads((DST / "_plan.json").read_text())
    return payload["scenes"]


def load_environment_codes() -> dict[str, int]:
    return json.loads((DST / "_plan.json").read_text())["environments"]


# ---------------------------------------------------------------------------
# step: scenes
# ---------------------------------------------------------------------------

def scene_mp4(scene_id: str) -> Path:
    aa, bb = shard(scene_id)
    return DST / "scenes" / aa / bb / f"{scene_id}.mp4"


def extract_job_videos(job: str, scenes: list[dict]) -> tuple[int, list[str]]:
    """Stream one ``<job>_mp4.tar`` and write out the selected sequences."""
    wanted = {f"{job}/mp4/{s['seq']}.mp4": s for s in scenes if not scene_mp4(s["scene_id"]).is_file()}
    if not wanted:
        return 0, []
    failures: list[str] = []
    written = 0
    with tarfile.open(IMAGE_DIR / f"{job}_mp4.tar", "r|") as tar:
        for member in tar:
            scene = wanted.pop(member.name, None)
            if scene is None:
                continue
            out = scene_mp4(scene["scene_id"])
            out.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                failures.append(f"{member.name}: not a regular file")
                continue
            tmp = out.with_suffix(".part")
            with open(tmp, "wb") as handle:
                while chunk := source.read(1 << 22):
                    handle.write(chunk)
            tmp.rename(out)
            written += 1
            if not wanted:
                break
    for name in wanted:
        failures.append(f"{name}: missing from tar")
    return written, failures


def run_scenes(plan: list[dict], workers: int) -> None:
    by_job: dict[str, list[dict]] = {}
    for scene in plan:
        by_job.setdefault(scene["job"], []).append(scene)
    total, failures = 0, []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(extract_job_videos, job, scenes): job for job, scenes in by_job.items()}
        for done, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                written, job_failures = future.result()
            except Exception as exc:  # noqa: BLE001 - one bad tar must not kill the batch
                failures.append(f"{job}: {exc}")
                continue
            total += written
            failures.extend(job_failures)
            print(f"  [{done}/{len(futures)}] {job}: +{written}", flush=True)
    print(f"[scenes] extracted {total:,} mp4; {len(failures)} failure(s)", flush=True)
    for line in failures[:20]:
        print(f"  FAIL {line}", flush=True)


# ---------------------------------------------------------------------------
# step: features
# ---------------------------------------------------------------------------

def feature_dir(scene_id: str) -> Path:
    aa, bb = shard(scene_id)
    return DST / "features" / "gt" / aa / bb / scene_id


def camera_geometry(rows: list[sqlite3.Row], portrait: bool) -> dict[str, np.ndarray]:
    """Per-frame camera-from-world extrinsics and pixel intrinsics, OpenCV axes."""
    width, height = SIZE_PORTRAIT if portrait else SIZE_LANDSCAPE
    sensor_w, sensor_h = SENSOR_PORTRAIT if portrait else SENSOR_LANDSCAPE
    position = np.array([[row["x"], row["y"], row["z"]] for row in rows], dtype=np.float64)
    yaw = np.array([row["yaw"] for row in rows])
    pitch = np.array([row["pitch"] for row in rows])
    roll = np.array([row["roll"] for row in rows])
    focal = np.array([row["focal_length"] for row in rows])

    roll_image = -roll + 90.0 if portrait else -roll
    rotation = (axis_rotation(2, roll_image) * axis_rotation(0, -pitch)
                * axis_rotation(1, -yaw))
    centre = unreal2cv2(position)

    n_frames = len(rows)
    extrinsics = np.tile(np.eye(4, dtype=np.float64), (n_frames, 1, 1))
    extrinsics[:, :3, :3] = rotation.as_matrix()
    extrinsics[:, :3, 3] = -rotation.apply(centre)

    intrinsics = np.tile(np.eye(3, dtype=np.float64), (n_frames, 1, 1))
    intrinsics[:, 0, 0] = focal / sensor_w * width
    intrinsics[:, 1, 1] = focal / sensor_h * height
    intrinsics[:, 0, 2] = width / 2.0
    intrinsics[:, 1, 2] = height / 2.0
    return {
        "extrinsics": extrinsics, "intrinsics_px": intrinsics, "centre": centre,
        "focal": focal, "width": width, "height": height,
        "sensor": (sensor_w, sensor_h), "rotation": rotation,
    }


def write_camera_npz(path: Path, geometry: dict, rows: list[sqlite3.Row], portrait: bool) -> None:
    width, height = geometry["width"], geometry["height"]
    intrinsics_px = geometry["intrinsics_px"]
    n_frames = len(rows)
    normalised = intrinsics_px.copy()
    normalised[:, 0, :] /= width
    normalised[:, 1, :] /= height
    fov_x = np.degrees(2.0 * np.arctan(width / (2.0 * intrinsics_px[:, 0, 0])))
    fov_y = np.degrees(2.0 * np.arctan(height / (2.0 * intrinsics_px[:, 1, 1])))

    centre = geometry["centre"]
    travel = float(np.linalg.norm(centre - centre[0], axis=1).max())
    spin = float(np.degrees((geometry["rotation"][0].inv() * geometry["rotation"]).magnitude().max()))
    focal_range = float(geometry["focal"].max() - geometry["focal"].min())
    locked = (travel < STATIC_TRANSLATION_M and spin < STATIC_ROTATION_DEG
              and focal_range < STATIC_FOCAL_MM)

    save_npz(
        path,
        extrinsics=geometry["extrinsics"].astype(np.float32),
        intrinsics=normalised.astype(np.float32),
        intrinsics_px=intrinsics_px.astype(np.float32),
        intrinsics_px_orig=intrinsics_px.astype(np.float32),
        fov_x_deg=fov_x.astype(np.float32), fov_y_deg=fov_y.astype(np.float32),
        frame_indices=np.arange(n_frames, dtype=np.int32),
        focal_length_mm=geometry["focal"].astype(np.float32),
        sensor_width_mm=np.float32(geometry["sensor"][0]),
        sensor_height_mm=np.float32(geometry["sensor"][1]),
        hfov_deg=np.array([row["hfov"] for row in rows], dtype=np.float32),
        height_above_ground=np.array([row["height_above_ground"] for row in rows],
                                     dtype=np.float32) / 100.0,
        image_width=np.int32(width), image_height=np.int32(height),
        num_frames=np.int32(n_frames), fps=np.float32(FPS),
        metric=np.bool_(True), scale=np.float32(1.0),
        portrait=np.bool_(portrait),
        camera_travel_m=np.float32(travel),
        camera_rotation_deg=np.float32(spin),
        focal_range_mm=np.float32(focal_range),
        static_camera=np.bool_(locked),
        scale_method=np.array("bedlam2 ground truth", dtype="U32"),
        world=np.array("unreal_cv2", dtype="U16"),
        source=np.array("bedlam2", dtype="U16"),
    )


def write_gravity_npz(path: Path, ground_height_world: float) -> None:
    """Unreal +z is up, and ``unreal2cv2`` sends it to -y, so gravity is +y."""
    ground_y = -ground_height_world / 100.0
    save_npz(
        path,
        gravity_world=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        up_world=np.array([0.0, -1.0, 0.0], dtype=np.float32),
        ground_normal=np.array([0.0, -1.0, 0.0], dtype=np.float32),
        ground_centroid=np.array([0.0, ground_y, 0.0], dtype=np.float32),
        ground_height=np.float32(ground_y),
        ground_accepted=np.bool_(True), reliable=np.bool_(True),
        ground_angle_deg=np.float32(0.0),
        source=np.array("bedlam2_gt", dtype="U16"),
    )


def joint_parents(body) -> list[int]:
    """Parent index per joint, root as ``-1`` — the layout ``stages.py`` stores."""
    return [-1] + [int(body.parent_tree[0, joint]) for joint in range(1, body.num_joints)]

_BODY = None
_HAND_MEAN: tuple[np.ndarray, np.ndarray] | None = None


def get_body():
    """One BetterHuman body per worker process."""
    global _BODY, _HAND_MEAN
    if _BODY is None:
        import torch

        torch.set_num_threads(1)
        sys.path.insert(0, str(_REPO_ROOT))
        from tools.body import load_body

        _BODY = load_body(device="cpu")
        model = np.load(LOCKED_HEAD, allow_pickle=True)
        _HAND_MEAN = (model["hands_meanl"].astype(np.float64),
                      model["hands_meanr"].astype(np.float64))
    return _BODY, _HAND_MEAN


@lru_cache(maxsize=256)
def load_motion(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(MOTION_DIR / f"{name}.npz")
    return (data["poses"].astype(np.float64), data["trans"].astype(np.float64),
            data["betas"][:16].astype(np.float64))


@lru_cache(maxsize=4096)
def pelvis_offset(betas_key: bytes) -> np.ndarray:
    """``q[0:3] - transl`` for a shape — the rest pelvis BetterHuman poses about."""
    import torch

    body, _ = get_body()
    betas = torch.from_numpy(np.frombuffer(betas_key, dtype=np.float64).copy()).float()[None]
    zero = lambda width: torch.zeros(1, width)
    q = body.from_classic(betas, zero(63), zero(3), zero(3), zero(45), zero(45))
    return q[0, :3].detach().numpy().astype(np.float64)


def visible_fraction(body, betas: np.ndarray, q: np.ndarray, geometry: dict) -> np.ndarray:
    """Fraction of the body's vertices inside the image, per frame ``(N,)``.

    BEDLAM2 places crowd members outside the camera frustum and labels them all
    the same, so this is what separates a usable person-frame from one where the
    body is simply not in the picture. Vertices, not joints: a body can be in
    frame with every joint outside it.
    """
    import torch

    extrinsics = geometry["extrinsics"]
    intrinsics = geometry["intrinsics_px"]
    width, height = geometry["width"], geometry["height"]
    shape = torch.from_numpy(betas).float()[None]
    fraction = np.empty(len(q), dtype=np.float32)
    for start in range(0, len(q), VISIBILITY_CHUNK):
        stop = min(start + VISIBILITY_CHUNK, len(q))
        with torch.no_grad():
            vertices = body.forward(shape, torch.from_numpy(q[start:stop])).vertices.numpy()
        rotation = extrinsics[start:stop, :3, :3]
        camera = np.einsum("fij,fvj->fvi", rotation, vertices.astype(np.float64))
        camera += extrinsics[start:stop, None, :3, 3]
        depth = camera[..., 2]
        safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
        block = intrinsics[start:stop]
        column = camera[..., 0] / safe * block[:, None, 0, 0] + block[:, None, 0, 2]
        row = camera[..., 1] / safe * block[:, None, 1, 1] + block[:, None, 1, 2]
        inside = ((depth > 0) & (column >= 0) & (column < width)
                  & (row >= 0) & (row < height))
        fraction[start:stop] = inside.mean(axis=1)
    return fraction


def write_smplx_npz(path: Path, scene: dict, geometry: dict) -> None:
    """Compose every body's world-frame SMPL-X trajectory and pack it as ``q``."""
    import torch

    width, height = geometry["width"], geometry["height"]
    body, hand_mean = get_body()
    mean_left, mean_right = hand_mean
    n_frames = scene["n_frames"]
    people = scene["bodies"]

    q_all = np.zeros((len(people), n_frames, 211), dtype=np.float32)
    visible = np.zeros((len(people), n_frames), dtype=np.float32)
    betas10 = np.zeros((len(people), NUM_BETAS), dtype=np.float32)
    betas16 = np.zeros((len(people), 16), dtype=np.float32)
    classic = {name: np.zeros((len(people), n_frames, width_), dtype=np.float32)
               for name, width_ in (("global_orient", 3), ("transl", 3), ("body_pose", 63),
                                    ("left_hand_pose", 45), ("right_hand_pose", 45),
                                    ("jaw_pose", 3), ("leye_pose", 3), ("reye_pose", 3))}

    for index, person in enumerate(people):
        poses, trans, betas = load_motion(person["name"])
        start = person["start_frame"]
        poses = poses[start:start + n_frames]
        trans = trans[start:start + n_frames]
        if len(poses) != n_frames:
            raise ValueError(f"{scene['scene_id']}/{person['name']}: motion has "
                             f"{len(poses)} frames from {start}, need {n_frames}")

        shape10 = np.ascontiguousarray(betas[:NUM_BETAS])
        pelvis = pelvis_offset(shape10.tobytes())
        placement = (axis_rotation(1, person["yaw"] - 90.0)
                     * Rotation.from_matrix(np.diag([1.0, -1.0, -1.0])))
        origin = unreal2cv2(np.array([person["x"], person["y"], person["z"]]))

        global_orient = (placement * Rotation.from_rotvec(poses[:, :3])).as_rotvec()
        transl = placement.apply(pelvis + trans) - pelvis + origin
        left_hand = poses[:, 75:120] - mean_left
        right_hand = poses[:, 120:165] - mean_right

        as_tensor = lambda array: torch.from_numpy(np.ascontiguousarray(array)).float()
        # betas stay (1, 10) and broadcast: baking one shape instead of one per
        # frame is ~165x faster and gives a bit-identical q.
        q = body.from_classic(as_tensor(shape10)[None],
                              as_tensor(poses[:, 3:66]), as_tensor(transl),
                              as_tensor(global_orient), as_tensor(left_hand),
                              as_tensor(right_hand))
        q_all[index] = q.detach().numpy()
        visible[index] = visible_fraction(body, shape10.astype(np.float32), q_all[index], geometry)
        betas10[index] = shape10
        betas16[index] = betas
        classic["global_orient"][index] = global_orient
        classic["transl"][index] = transl
        classic["body_pose"][index] = poses[:, 3:66]
        classic["left_hand_pose"][index] = left_hand
        classic["right_hand_pose"][index] = right_hand
        classic["jaw_pose"][index] = poses[:, 66:69]
        classic["leye_pose"][index] = poses[:, 69:72]
        classic["reye_pose"][index] = poses[:, 72:75]

    save_npz(
        path,
        q=q_all, betas=betas10, betas_bedlam2=betas16,
        scale=np.ones(len(people), dtype=np.float32),
        expression=np.zeros((len(people), n_frames, 10), dtype=np.float32),
        joint_names=np.array(body.joint_names, dtype="U16"),
        parents=np.array(joint_parents(body), dtype=np.int32),
        object_ids=np.arange(len(people), dtype=np.int32),
        valid_mask=visible > 0.0, visible_fraction=visible,
        num_people=np.int32(len(people)), num_frames=np.int32(n_frames),
        video_width=np.int32(width), video_height=np.int32(height),
        fps=np.float32(FPS), frame=np.array("world", dtype="U8"),
        model_type=np.array("smplx_mid", dtype="U16"), num_betas=np.int32(NUM_BETAS),
        animation_names=np.array([p["name"] for p in people]),
        start_frames=np.array([p["start_frame"] for p in people], dtype=np.int32),
        source=np.array("bedlam2", dtype="U16"),
        **classic,
    )


def build_features(scene: dict) -> str:
    out = feature_dir(scene["scene_id"])
    out.mkdir(parents=True, exist_ok=True)
    conn = open_render_db()
    rows = conn.execute(
        "SELECT x, y, z, yaw, pitch, roll, focal_length, hfov, portrait_orientation, "
        "height_above_ground FROM gt_camera WHERE renderjob = ? AND sequence = ? ORDER BY image_id",
        (scene["job"], scene["seq_index"])).fetchall()
    conn.close()
    if len(rows) != scene["n_frames"]:
        raise ValueError(f"{scene['scene_id']}: {len(rows)} gt_camera rows, "
                         f"{scene['n_frames']} frames")
    portrait = bool(rows[0]["portrait_orientation"])
    geometry = camera_geometry(rows, portrait)
    write_camera_npz(out / "camera.npz", geometry, rows, portrait)
    write_gravity_npz(out / "gravity.npz", scene["ground_height_world"])
    write_smplx_npz(out / "smplx.npz", scene, geometry)
    return scene["scene_id"]


def _features_worker(scene: dict) -> tuple[str, str]:
    try:
        return build_features(scene), ""
    except Exception as exc:  # noqa: BLE001 - one bad sequence must not kill the batch
        return scene["scene_id"], f"{type(exc).__name__}: {exc}"


def run_features(plan: list[dict], workers: int, skip_existing: bool) -> None:
    todo = [s for s in plan
            if not skip_existing or not (feature_dir(s["scene_id"]) / "smplx.npz").is_file()]
    print(f"[features] {len(todo):,} of {len(plan):,} scenes to build", flush=True)
    failures: list[str] = []
    with ProcessPoolExecutor(workers) as pool:
        for done, (scene_id, error) in enumerate(
                pool.map(_features_worker, todo, chunksize=8), 1):
            if error:
                failures.append(f"{scene_id}: {error}")
            if done % 500 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}  ({len(failures)} failed)", flush=True)
    print(f"[features] done; {len(failures)} failure(s)", flush=True)
    for line in failures[:20]:
        print(f"  FAIL {line}", flush=True)


# ---------------------------------------------------------------------------
# step: frames
# ---------------------------------------------------------------------------

def frame_dir(scene_id: str) -> Path:
    aa, bb = shard(scene_id)
    return DST / "frames" / aa / bb / scene_id


def _frames_worker(scene: dict) -> tuple[str, int, str]:
    scene_id, expected = scene["scene_id"], scene["n_frames"]
    out = frame_dir(scene_id)
    try:
        if out.is_dir() and sum(1 for _ in out.glob("*.jpg")) == expected:
            return scene_id, 0, ""
        out.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(scene_mp4(scene_id)))
        written = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if written >= expected:
                capture.release()
                raise ValueError(f"more than {expected} decodable frames")
            if not cv2.imwrite(str(out / f"{written:06d}.jpg"), image,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
                raise ValueError(f"could not write frame {written}")
            written += 1
        capture.release()
        if written != expected:
            raise ValueError(f"decoded {written} frames, expected {expected}")
    except Exception as exc:  # noqa: BLE001 - one bad clip must not kill the batch
        return scene_id, 0, f"{type(exc).__name__}: {exc}"
    return scene_id, written, ""


def run_frames(plan: list[dict], workers: int) -> None:
    print(f"[frames] {len(plan):,} scenes, {sum(s['n_frames'] for s in plan):,} frames",
          flush=True)
    total, failures = 0, []
    with ProcessPoolExecutor(workers) as pool:
        for done, (scene_id, written, error) in enumerate(
                pool.map(_frames_worker, plan, chunksize=4), 1):
            total += written
            if error:
                failures.append(f"{scene_id}: {error}")
            if done % 500 == 0 or done == len(plan):
                print(f"  {done}/{len(plan)}  {total:,} frames written  "
                      f"({len(failures)} failed)", flush=True)
    print(f"[frames] wrote {total:,} jpg; {len(failures)} failure(s)", flush=True)
    for line in failures[:20]:
        print(f"  FAIL {line}", flush=True)


# ---------------------------------------------------------------------------
# step: db
# ---------------------------------------------------------------------------

_SCENES_SQL = """
CREATE TABLE IF NOT EXISTS scenes (
    scene_id      TEXT PRIMARY KEY,
    video_id      TEXT NOT NULL,
    source        TEXT,
    idx           INTEGER NOT NULL,
    start_sec     REAL,
    end_sec       REAL,
    duration_sec  REAL,
    n_frames      INTEGER,
    sub_split     INTEGER DEFAULT 0,
    width         INTEGER,
    height        INTEGER,
    fps           REAL,
    filesize      INTEGER,
    file_sha256   TEXT,
    rel_path      TEXT,
    created_at    TEXT,
    passed_filter INTEGER DEFAULT 0,
    yolo_processed INTEGER DEFAULT 0,
    yolo_at       TEXT,
    filter_reason TEXT,
    good_frac     REAL,
    max_person_height_frac REAL,
    vlm_processed INTEGER DEFAULT 0,
    vlm_at        TEXT,
    vlm_passed    INTEGER,
    vlm_reason    TEXT,
    vlm_category  INTEGER,
    vlm_rope_supported INTEGER,
    human_selected INTEGER DEFAULT 0,
    dataset_split TEXT,
    viewed        INTEGER DEFAULT 0,
    clip_deleted  INTEGER DEFAULT 0,
    static_camera INTEGER
);
CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS bedlam2 (
    scene_id        TEXT PRIMARY KEY,
    render_job      TEXT NOT NULL,
    sequence_index  INTEGER NOT NULL,
    environment     TEXT,
    batch           TEXT,
    camera_movement TEXT,
    portrait        INTEGER,
    n_bodies        INTEGER,
    ground_height_m REAL,
    hdri            TEXT,
    camera_travel_m REAL,
    focal_min_mm    REAL,
    focal_max_mm    REAL
);
CREATE INDEX IF NOT EXISTS idx_scenes_video  ON scenes(video_id);
CREATE INDEX IF NOT EXISTS idx_scenes_source ON scenes(source);
CREATE INDEX IF NOT EXISTS idx_scenes_yolo   ON scenes(yolo_processed);
CREATE INDEX IF NOT EXISTS idx_scenes_vlm    ON scenes(vlm_processed);
CREATE INDEX IF NOT EXISTS idx_scenes_human  ON scenes(human_selected);
CREATE INDEX IF NOT EXISTS idx_scenes_split  ON scenes(dataset_split);
CREATE INDEX IF NOT EXISTS idx_scenes_viewed ON scenes(viewed);
CREATE INDEX IF NOT EXISTS idx_bedlam2_job   ON bedlam2(render_job);
CREATE INDEX IF NOT EXISTS idx_bedlam2_env   ON bedlam2(environment);
"""


def _db_row(scene: dict, now: str) -> tuple[tuple, tuple] | None:
    scene_id = scene["scene_id"]
    clip = scene_mp4(scene_id)
    if not clip.is_file():
        return None
    camera = np.load(feature_dir(scene_id) / "camera.npz")
    width, height = int(camera["image_width"]), int(camera["image_height"])
    n_frames = scene["n_frames"]
    aa, bb = shard(scene_id)
    scenes_row = (
        scene_id, scene["job"], "bedlam2", scene["seq_index"],
        0.0, n_frames / FPS, n_frames / FPS, n_frames, 0,
        width, height, FPS, clip.stat().st_size, sha256_file(clip),
        f"{aa}/{bb}/{scene_id}.mp4", now, 1,
        0, None, "", None, None,
        1, now, 1, None, scene["environment_code"], 0,
        1, "train", 0, 0, int(camera["static_camera"]),
    )
    bedlam_row = (
        scene_id, scene["job"], scene["seq_index"], scene["environment"], scene["batch"],
        scene["camera_movement"], int(scene["portrait"]), len(scene["bodies"]),
        float(-scene["ground_height_world"] / 100.0), scene["hdri"],
        float(camera["camera_travel_m"]),
        float(camera["focal_length_mm"].min()), float(camera["focal_length_mm"].max()),
    )
    return scenes_row, bedlam_row


def run_db(plan: list[dict], workers: int) -> None:
    db_path = DST / "scenes" / "scenes.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(_SCENES_SQL)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    scenes_rows, bedlam_rows, missing = [], [], 0
    with ThreadPoolExecutor(workers) as pool:
        for result in pool.map(lambda s: _db_row(s, now), plan):
            if result is None:
                missing += 1
                continue
            scenes_rows.append(result[0])
            bedlam_rows.append(result[1])
    conn.executemany(
        "INSERT OR REPLACE INTO scenes VALUES (" + ",".join("?" * 33) + ")", scenes_rows)
    conn.executemany(
        "INSERT OR REPLACE INTO bedlam2 VALUES (" + ",".join("?" * 13) + ")", bedlam_rows)
    environments = load_environment_codes()
    conn.executemany("INSERT OR REPLACE INTO schema_meta VALUES (?, ?)", [
        ("version", SCHEMA_VERSION),
        ("dataset", "bedlam2"),
        ("vlm_category_map", json.dumps(environments)),
    ])
    conn.commit()
    conn.close()
    print(f"[db] wrote {len(scenes_rows):,} rows to {db_path}"
          f"{f'; {missing} scene(s) had no mp4' if missing else ''}", flush=True)


# ---------------------------------------------------------------------------
# step: verify
# ---------------------------------------------------------------------------

def run_verify(plan: list[dict], sample: int, seed: int) -> None:
    """Reproject the stored bodies onto real frames and report the pixel error."""
    import torch

    body, _ = get_body()
    rng = np.random.default_rng(seed)
    picks = [plan[i] for i in rng.choice(len(plan), size=min(sample, len(plan)), replace=False)]
    out_dir = DST / "_verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    for scene in picks:
        scene_id = scene["scene_id"]
        camera = np.load(feature_dir(scene_id) / "camera.npz")
        smplx = np.load(feature_dir(scene_id) / "smplx.npz")
        frame = int(rng.integers(0, scene["n_frames"]))
        capture = cv2.VideoCapture(str(scene_mp4(scene_id)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
        capture.release()
        if not ok:
            print(f"  {scene_id}: frame {frame} unreadable", flush=True)
            continue
        height, width = image.shape[:2]
        if (width, height) != (int(camera["image_width"]), int(camera["image_height"])):
            print(f"  {scene_id}: SIZE MISMATCH video {width}x{height} vs "
                  f"camera.npz {int(camera['image_width'])}x{int(camera['image_height'])}", flush=True)
        extrinsic = camera["extrinsics"][frame].astype(np.float64)
        intrinsic = camera["intrinsics_px"][frame].astype(np.float64)
        overlay = image.copy()
        inside, depths = [], []
        for person in range(int(smplx["num_people"])):
            betas = torch.from_numpy(smplx["betas"][person]).float()[None]
            q = torch.from_numpy(smplx["q"][person, frame]).float()[None]
            vertices = body.forward(betas, q).vertices[0].detach().numpy().astype(np.float64)
            in_camera = vertices @ extrinsic[:3, :3].T + extrinsic[:3, 3]
            depths.append(float(np.median(in_camera[:, 2])))
            uv = (intrinsic @ (in_camera / in_camera[:, 2:3]).T).T[:, :2]
            visible = ((uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height))
            inside.append(float(visible.mean()))
            for point in uv[::12].astype(int):
                if 0 <= point[0] < width and 0 <= point[1] < height:
                    cv2.circle(overlay, tuple(point), 1, (0, 255, 0), -1)
        path = out_dir / f"{scene_id}_{frame:06d}.png"
        cv2.imwrite(str(path), cv2.addWeighted(overlay, 0.75, image, 0.25, 0))
        stats.append((scene_id, scene["job"], frame, inside, depths))
        print(f"  {scene_id} {scene['job'][:34]:<34} f{frame:<5} "
              f"people={len(inside)} in-frame={[f'{v:.2f}' for v in inside]} "
              f"depth={[f'{d:.1f}m' for d in depths]}", flush=True)
    coverage = [v for _, _, _, values, _ in stats for v in values]
    if coverage:
        print(f"[verify] {len(stats)} scenes, {len(coverage)} bodies; vertices inside frame: "
              f"mean {np.mean(coverage):.3f} median {np.median(coverage):.3f} "
              f"min {np.min(coverage):.3f}", flush=True)
    print(f"[verify] overlays in {out_dir}", flush=True)


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--step", default="all",
                        choices=("plan", "scenes", "features", "frames", "db", "verify", "all"))
    parser.add_argument("--workers", type=int, default=0, help="0 picks a per-step default")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N scenes")
    parser.add_argument("--scenes", default=None,
                        help="comma-separated scene_ids, or <render job>[/<seq>] prefixes")
    parser.add_argument("--sample", type=int, default=12, help="scenes to check in --step verify")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true",
                        help="features: leave scenes that already have smplx.npz alone")
    args = parser.parse_args()

    steps = ("plan", "scenes", "features", "frames", "db") if args.step == "all" else (args.step,)
    defaults = {"scenes": 12, "features": 48, "frames": 64, "db": 32}

    plan = build_plan() if "plan" in steps else load_plan()
    if args.scenes:
        wanted = [token.strip() for token in args.scenes.split(",") if token.strip()]
        plan = [s for s in plan if s["scene_id"] in wanted or s["job"] in wanted
                or f"{s['job']}/{s['seq']}" in wanted]
        print(f"[scenes] {len(plan)} selected", flush=True)
    if args.limit is not None:
        plan = plan[:args.limit]
        print(f"[limit] {len(plan)} scene(s)", flush=True)

    for step in steps:
        workers = args.workers or defaults.get(step, 8)
        if step == "scenes":
            run_scenes(plan, workers)
        elif step == "features":
            run_features(plan, workers, args.skip_existing)
        elif step == "frames":
            run_frames(plan, workers)
        elif step == "db":
            run_db(plan, workers)
        elif step == "verify":
            run_verify(plan, args.sample, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
