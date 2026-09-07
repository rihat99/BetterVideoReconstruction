#!/usr/bin/env python3
"""Run one pipeline stage over the ClimbingVideos scene corpus."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import tomllib
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[1]
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import scene_db  # noqa: E402  (scripts/ on path)
from scene_db import ScenePaths, select_targets  # noqa: E402

_STAGES_DIR = _REPO_ROOT / "scripts" / "stages"
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "corpus.toml"
_CONTACTS_CLI = _STAGES_DIR / "estimate_contacts.py"
# Pinned copy of scripts/stages/estimate_contacts.py's schema/method (kept literal
# so the driver stays torch-free); update together with that stage.
_CONTACT_LABEL_SCHEMA = 2
_CONTACT_LABEL_METHOD = "frames-v4"
# The modules the contacts stage actually executes: a code fix in any of them must
# invalidate --skip-existing even when the artifact schema is unchanged.
_IMPLEMENTATION_INPUTS = (
    _CONTACTS_CLI,
    _REPO_ROOT / "tools" / "human_optim" / "contacts.py",
    _REPO_ROOT / "tools" / "human_optim" / "forces.py",
    _REPO_ROOT / "tools" / "human_optim" / "friction.py",
    _REPO_ROOT / "tools" / "human_optim" / "qmath.py",
    _REPO_ROOT / "tools" / "body.py",
    _REPO_ROOT / "tools" / "geometry" / "normals.py",
    _REPO_ROOT / "tools" / "geometry" / "quality.py",
    _REPO_ROOT / "tools" / "geometry" / "depth_smear.py",
)

InvocationMode = Literal[
    "manifest_subprocess", "farm_inprocess", "farm_subprocess_per_scene"
]


@dataclass(frozen=True)
class StageSpec:
    """The stage-specific hooks used by the shared orchestrator and worker."""

    mode: InvocationMode
    cli: Path | None
    out_stage: str
    out_check: str
    manifest_dependencies: tuple[str, ...] = ()
    farm_inputs: tuple[str, ...] = ()
    config_sections: tuple[str, ...] = ()
    output_config_key: str | None = None
    overlay_artifact: str | None = None
    capture_and_move: bool = False


STAGES: dict[str, StageSpec] = {
    "sam3": StageSpec(
        mode="manifest_subprocess",
        cli=_STAGES_DIR / "track_sam3.py",
        out_stage="sam3",
        out_check="bboxes.npz",
        config_sections=("sam3",),
        output_config_key="masks_subdir",
        overlay_artifact="mask_overlay.mp4",
    ),
    "vggt": StageSpec(
        mode="manifest_subprocess",
        cli=_STAGES_DIR / "estimate_camera_vggt.py",
        out_stage="vggt",
        out_check="camera.npz",
        config_sections=("vggt",),
        output_config_key="subdir",
    ),
    "sapiens": StageSpec(
        mode="manifest_subprocess",
        cli=_STAGES_DIR / "estimate_sapiens.py",
        out_stage="sapiens",
        out_check="pose.npz",
        manifest_dependencies=("bboxes",),
        config_sections=("sapiens",),
        output_config_key="subdir",
        overlay_artifact="overlay/pose.mp4",
    ),
    "sam3d": StageSpec(
        mode="manifest_subprocess",
        cli=_STAGES_DIR / "estimate_body_sam3d.py",
        out_stage="sam3d",
        out_check="params.npz",
        manifest_dependencies=("bboxes", "camera"),
        config_sections=("sam3d",),
        output_config_key="subdir",
        overlay_artifact="overlay/pose.mp4",
    ),
    "smplx": StageSpec(
        mode="manifest_subprocess",
        cli=_STAGES_DIR / "convert_mhr_to_smplx.py",
        out_stage="sam3d",
        out_check="smplx_params.npz",
        manifest_dependencies=("params",),
        config_sections=("smplx",),
        output_config_key="subdir",
        overlay_artifact="overlay/pose.mp4",
    ),
    "geometry": StageSpec(
        mode="farm_inprocess",
        cli=None,
        out_stage="geometry",
        out_check="scale.npz",  # scale runs last in the fuse→scale chain
        farm_inputs=("vggt", "sam3d", "sam3", "sapiens"),  # sapiens: the scale pre-fit reads it
        config_sections=("scale", "fuse"),
    ),
    "gravity": StageSpec(
        mode="farm_inprocess",
        cli=None,
        out_stage="geocalib",
        out_check="gravity.npz",
        farm_inputs=("geometry",),  # transform.npz extrinsics + scene.npz floor fit
        config_sections=("gravity",),
    ),
    "human_optim": StageSpec(
        mode="farm_inprocess",
        cli=None,
        out_stage="human_optim",
        out_check="kinematic_1.npz",
        farm_inputs=("sam3d", "sapiens", "geometry"),
        config_sections=("human_optimization",),
    ),
    "kinematics_2": StageSpec(
        mode="farm_inprocess",
        cli=None,
        out_stage="human_optim",
        out_check="kinematic_2.npz",
        farm_inputs=("sapiens", "geometry"),  # kinematic_1 + contacts_1 come via the human_optim out link
        config_sections=("human_optimization",),
    ),
    "kindyn": StageSpec(
        mode="farm_inprocess",
        cli=None,
        out_stage="human_optim",
        out_check="kindyn_1.npz",
        # kinematic_* + contacts_* come via the human_optim out link; geocalib is the
        # optional gravity input (a dangling link falls back to world-down, like the
        # contacts farm).
        farm_inputs=("sapiens", "geometry", "geocalib"),
        config_sections=("human_optimization",),
    ),
    "contacts": StageSpec(
        mode="farm_subprocess_per_scene",
        cli=_CONTACTS_CLI,
        out_stage="human_optim",
        out_check="contacts_{tag}.npz",
        farm_inputs=("human_optim", "geometry"),
        capture_and_move=True,
    ),
}

# farm_inprocess human-optimization stages → the --stages argument they dispatch.
_HUMAN_STAGE_ARGS = {
    "human_optim": "root,kinematics_1",
    "kinematics_2": "kinematics_2",
    "kindyn": "kindyn_1",
}


@dataclass(frozen=True)
class CorpusConfig:
    """Resolved values from ``configs/corpus.toml``."""

    source_path: Path
    data_root: Path
    categories: tuple[int, ...]
    rope_supported: int | None
    seed: int
    devices: tuple[int, ...]
    skip_existing: bool
    out_root: Path | None
    debug_overlay_dir: Path | None
    pipeline_config: Path
    human_optim_config: Path
    overlay: dict[str, bool]
    min_presence: float
    discard_failed: bool
    text: str
    iou_thresh: float
    ground_min_area_frac: float
    confidence: float
    part_length: int
    bodies: str


def _repo_path(value: str | Path) -> Path:
    """Resolve a configured path relative to the repository root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (_REPO_ROOT / path).resolve()


def load_config(path: Path) -> CorpusConfig:
    """Load and validate the corpus-driver portions of a TOML config."""
    path = path.resolve()
    with path.open("rb") as config_file:
        raw = tomllib.load(config_file)
    corpus = raw.get("corpus", {})
    track = raw.get("track_sam3", {})
    contacts = raw.get("contacts_run", {})
    rope_value = int(corpus.get("rope_supported", scene_db.DEFAULT_ROPE_SUPPORTED))
    out_root = str(corpus.get("out_root", "")).strip()
    debug_dir = str(corpus.get("debug_overlay_dir", "")).strip()
    devices = tuple(int(value) for value in corpus.get("devices", [1, 2, 3]))
    if not devices:
        raise ValueError("[corpus].devices must contain at least one GPU id")
    bodies = str(contacts.get("bodies", "kinematic_1"))
    if bodies not in {"kinematic_1", "kinematic_2"}:
        raise ValueError("[contacts_run].bodies must be kinematic_1 or kinematic_2")
    part_length = int(track.get("part_length", 200))
    if part_length < 2:
        raise ValueError("[track_sam3].part_length must be at least 2")
    return CorpusConfig(
        source_path=path,
        data_root=_repo_path(corpus.get("data_root", scene_db.DEFAULT_DATA_ROOT)),
        categories=tuple(int(value) for value in corpus.get("categories", [1, 2])),
        rope_supported=None if rope_value < 0 else rope_value,
        seed=int(corpus.get("seed", 0)),
        devices=devices,
        skip_existing=bool(corpus.get("skip_existing", False)),
        out_root=_repo_path(out_root) if out_root else None,
        debug_overlay_dir=_repo_path(debug_dir) if debug_dir else None,
        pipeline_config=_repo_path(corpus.get("pipeline_config", "configs/pipeline.toml")),
        human_optim_config=_repo_path(
            corpus.get("human_optim_config", "configs/human_optimization.toml")
        ),
        overlay={key: bool(value) for key, value in raw.get("overlay", {}).items()},
        min_presence=float(track.get("min_presence", 0.9)),
        discard_failed=bool(track.get("discard_failed", False)),
        text=str(track.get("text", "person")),
        iou_thresh=float(track.get("iou_thresh", 0.2)),
        ground_min_area_frac=float(track.get("ground_min_area_frac", 0.002)),
        confidence=float(track.get("confidence", 0.5)),
        part_length=part_length,
        bodies=bodies,
    )


# ------------------------- first-frame bbox prompt -------------------------


def first_frame_prompts(
    paths: ScenePaths, scene_id: str, *,
    text: str = "person", iou_thresh: float = 0.2, min_area_frac: float = 0.002,
) -> list[dict]:
    """SAM 3 ``select_bbox`` seed prompts from the YOLO human-of-interest track.

    Reads ``features/tracking/<scene>.npz`` (CSR-packed per-human boxes
    ``[frame, x1, y1, x2, y2, conf]``) and returns one prompt per *matched*
    human, taking that human's earliest tracked frame (98% start at frame 0; the
    rest use their first appearance). Each prompt is a ``select_bbox`` entry:
    SAM 3 text-grounds ``text`` on the seed frame and keeps the detection whose
    mask best matches this box (IoU ≥ ``iou_thresh``), then tracks that clean
    person mask — the box only *selects* the person. Returns ``[]`` when there
    is no usable track.
    """
    npz = paths.tracking_npz(scene_id)
    if not npz.is_file():
        return []
    d = np.load(npz, allow_pickle=True)
    matched = d["human_matched"]
    offsets = d["human_box_offsets"]
    boxes = d["boxes"]

    prompts: list[dict] = []
    for i in range(len(matched)):
        if not bool(matched[i]):
            continue
        sl = boxes[offsets[i]:offsets[i + 1]]
        if sl.shape[0] == 0:
            continue
        row = sl[int(sl[:, 0].argmin())]          # earliest frame for this human
        x1, y1, x2, y2 = (float(v) for v in row[1:5])
        if x2 - x1 < 2.0 or y2 - y1 < 2.0:        # skip degenerate boxes
            continue
        prompts.append({
            "select_bbox": [x1, y1, x2, y2],
            "text": text,
            "label": "person",
            "iou_thresh": iou_thresh,
            "min_area_frac": min_area_frac,
        })
    return prompts


def scene_presence(paths: ScenePaths, scene_id: str) -> tuple[float, int, int] | None:
    """``(primary_present_frac, n_obj, n_frames)`` from a scene's ``bboxes.npz``.

    ``primary_present_frac`` is the fraction of frames where object 0 (the primary
    tracked subject) has a non-empty mask — the tracking-quality signal. Returns
    ``None`` when the scene produced no output (a hard failure).
    """
    npz = paths.sam3_dir(scene_id) / "bboxes.npz"
    if not npz.is_file():
        return None
    d = np.load(npz, allow_pickle=True)
    bpo = d["bboxes_per_obj"].astype(np.float32)          # (n_obj, N, 4) xyxy
    b = bpo[0]
    area = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return float((area > 0).mean()), int(bpo.shape[0]), int(bpo.shape[1])


def discard_failures(paths: ScenePaths, rows: list[sqlite3.Row],
                     min_presence: float) -> list[dict]:
    """Unmark ``human_selected = 0`` for scenes whose tracking failed.

    A scene fails when it produced no output, or its primary-subject mask is
    present in fewer than ``min_presence`` of frames (SAM 3 lost the climber —
    e.g. the fade-in hard case). Records ``filter_reason`` for auditability /
    reversal and returns the discard records.
    """
    discards: list[dict] = []
    for row in rows:
        sid = row["scene_id"]
        res = scene_presence(paths, sid)
        if res is None:
            discards.append({"scene_id": sid, "present": 0.0,
                             "n_frames": int(row["n_frames"]), "reason": "no_output"})
        elif res[0] < min_presence:
            discards.append({"scene_id": sid, "present": round(res[0], 4),
                             "n_frames": res[2], "reason": "low_presence"})

    if discards:
        conn = paths.connect_rw()
        try:
            for d in discards:
                conn.execute(
                    "UPDATE scenes SET human_selected = 0, filter_reason = ? WHERE scene_id = ?",
                    (f"sam3_track_failed(present={d['present']:.2f})", d["scene_id"]),
                )
            conn.commit()
        finally:
            conn.close()
    return discards


# -------------------------- contacts provenance ---------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_fresh(out: Path, body: Path, geom: Path, config: Path) -> bool:
    """A resumable output is reusable only when its complete provenance matches."""
    scene, transform = geom / "scene.npz", geom / "transform.npz"
    inputs = [body, scene, transform, config, *_IMPLEMENTATION_INPUTS]
    human = geom / "human.npz"
    if human.is_file():  # optional for the stage (its smear reads NaN when absent)
        inputs.append(human)
    if not out.is_file() or any(not path.is_file() for path in inputs):
        return False
    if out.stat().st_mtime < max(path.stat().st_mtime for path in inputs):
        return False
    try:
        with np.load(out, allow_pickle=False) as saved:
            return (
                int(saved["contact_label_schema"]) == _CONTACT_LABEL_SCHEMA
                and str(saved["contact_label_method"]) == _CONTACT_LABEL_METHOD
                and str(saved["source_body"]) == body.name
                and str(saved["source_body_sha256"]) == _sha256(body)
                and str(saved["source_scene_sha256"]) == _sha256(scene)
                and str(saved["source_transform_sha256"]) == _sha256(transform)
            )
    except (KeyError, OSError, TypeError, ValueError, zipfile.BadZipFile):
        return False


# -------------------------- stage input/output ----------------------------


def _dependency_paths(paths: ScenePaths, stage: str, scene_id: str) -> dict[str, Path]:
    """Resolve the explicit dependencies for a manifest-mode stage."""
    if stage == "sapiens":
        return {"bboxes": paths.sam3_bboxes(scene_id)}
    if stage == "sam3d":
        return {"bboxes": paths.sam3_bboxes(scene_id), "camera": paths.vggt_camera(scene_id)}
    if stage == "smplx":
        return {"params": paths.feature_dir("sam3d", scene_id) / "params.npz"}
    return {}


def _out_path(
    paths: ScenePaths, out_paths: ScenePaths, spec: StageSpec, scene_id: str, tag: str
) -> Path:
    """Resolve the artifact that identifies one scene's output."""
    owner = out_paths if spec.capture_and_move else paths
    return owner.feature_dir(spec.out_stage, scene_id) / spec.out_check.format(tag=tag)


def _contacts_inputs(paths: ScenePaths, scene_id: str, bodies: str) -> tuple[Path, Path]:
    """Return the body file and geometry directory consumed by contacts."""
    body = paths.feature_dir("human_optim", scene_id) / f"{bodies}.npz"
    return body, paths.feature_dir("geometry", scene_id)


def _geometry_metric(geom_dir: Path) -> bool:
    """True when the fused world has been metricized (a re-fuse resets the flag).

    A truncated npz (a sweep killed mid-write) reads as *not done*, so the next
    sweep regenerates it instead of crashing the shard at the same scene forever."""
    tr_path = geom_dir / "transform.npz"
    if not tr_path.is_file():
        return False
    try:
        with np.load(tr_path) as tr:
            if "metric" in tr.files and not bool(tr["metric"]):
                return False
        scale_path = geom_dir / "scale.npz"
        if scale_path.is_file():  # the pre-fit ran → its bodies must exist too
            with np.load(scale_path) as sc:
                if "method" in sc.files and str(sc["method"]) == "prefit-dual":
                    return (geom_dir / "prefit.npz").is_file()
    except (KeyError, OSError, ValueError, zipfile.BadZipFile):
        return False
    return True


def _scene_fresh(
    paths: ScenePaths,
    out_paths: ScenePaths,
    spec: StageSpec,
    scene_id: str,
    config: CorpusConfig,
) -> bool:
    """Apply a stage's exists or provenance freshness policy."""
    tag = config.bodies.removeprefix("kinematic_")
    out = _out_path(paths, out_paths, spec, scene_id, tag)
    if spec.capture_and_move:
        body, geom = _contacts_inputs(paths, scene_id, config.bodies)
        return _is_fresh(out, body, geom, config.human_optim_config)
    if not out.is_file():
        return False
    if spec.out_check == "scale.npz":  # a stale scale.npz must not mask a re-fused world
        return _geometry_metric(out.parent)
    return True


def _manifest_entry(
    paths: ScenePaths, stage: str, row: sqlite3.Row, config: CorpusConfig, *, report: bool
) -> dict[str, Any] | None:
    """Assemble one manifest entry, reporting and skipping missing dependencies."""
    spec = STAGES[stage]
    sid = row["scene_id"]
    if stage == "sam3":
        prompts = first_frame_prompts(
            paths,
            sid,
            text=config.text,
            iou_thresh=config.iou_thresh,
            min_area_frac=config.ground_min_area_frac,
        )
        if not prompts:
            if report:
                print(f"  [skip] {sid}: no matched YOLO track / usable first-frame bbox")
            return None
        return {
            "source": str(paths.clip_path(row)),
            "out_dir": str(paths.feature_dir(spec.out_stage, sid)),
            "prompts": prompts,
        }

    dependencies = _dependency_paths(paths, stage, sid)
    missing = [path for path in dependencies.values() if not path.is_file()]
    if missing:
        if report:
            print(f"  [skip] {sid}: missing " + ", ".join(str(path) for path in missing))
        return None
    entry: dict[str, Any] = {
        "source": str(paths.clip_path(row)),
        "out_dir": str(paths.feature_dir(spec.out_stage, sid)),
    }
    for key, path in dependencies.items():
        if key != "params":
            entry[key] = str(path)
    return entry


def _farm_entry(paths: ScenePaths, spec: StageSpec, row: sqlite3.Row) -> dict[str, Any]:
    """Assemble one write-through symlink-farm specification."""
    sid = row["scene_id"]
    entry: dict[str, Any] = {
        "scene_id": sid,
        "clip": str(paths.clip_path(row)),
        "output": str(paths.feature_dir(spec.out_stage, sid)),
    }
    entry["inputs"] = {
        stage: str(paths.feature_dir(stage, sid)) for stage in spec.farm_inputs
    }
    return entry


def _contacts_entry(
    paths: ScenePaths, out_paths: ScenePaths, row: sqlite3.Row, config: CorpusConfig
) -> dict[str, Any]:
    """Assemble one capture-and-move contact farm specification."""
    sid = row["scene_id"]
    body, geom = _contacts_inputs(paths, sid, config.bodies)
    tag = config.bodies.removeprefix("kinematic_")
    return {
        "scene_id": sid,
        "clip": str(paths.clip_path(row)),
        "body": str(body),
        "geometry": str(geom),
        "geocalib": str(paths.feature_dir("geocalib", sid)),  # optional gravity input
        "output": str(_out_path(paths, out_paths, STAGES["contacts"], sid, tag)),
    }


# -------------------------- resolved stage config -------------------------


def _toml_key(value: str) -> str:
    """Encode a TOML key conservatively."""
    if value.replace("_", "a").replace("-", "a").isalnum():
        return value
    return json.dumps(value)


def _toml_value(value: Any) -> str:
    """Encode the scalar/list types used by the repository configs."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"cannot encode TOML value {value!r}")


def _dump_toml(data: dict[str, Any]) -> str:
    """Serialize the ordinary nested dictionaries produced by ``tomllib``."""
    lines: list[str] = []

    def emit(table: dict[str, Any], prefix: tuple[str, ...]) -> None:
        scalars = [(key, value) for key, value in table.items() if not isinstance(value, dict)]
        children = [(key, value) for key, value in table.items() if isinstance(value, dict)]
        if prefix:
            if lines and lines[-1] != "":
                lines.append("")
            lines.append("[" + ".".join(_toml_key(key) for key in prefix) + "]")
        for key, value in scalars:
            lines.append(f"{_toml_key(key)} = {_toml_value(value)}")
        for key, value in children:
            emit(value, (*prefix, key))

    emit(data, ())
    return "\n".join(lines) + "\n"


def _stage_config_data(stage: str, config: CorpusConfig) -> dict[str, Any]:
    """Copy authoritative stage knobs and apply only corpus execution overrides."""
    spec = STAGES[stage]
    base_path = (
        config.human_optim_config
        if stage in {"human_optim", "kinematics_2", "kindyn", "contacts"}
        else config.pipeline_config
    )
    with base_path.open("rb") as config_file:
        base = tomllib.load(config_file)
    resolved: dict[str, Any] = {"runtime": dict(base.get("runtime", {}))}
    resolved["runtime"]["device"] = "cuda"

    if stage in {"human_optim", "kinematics_2", "kindyn"}:
        resolved = base
        resolved.setdefault("runtime", {})["device"] = "cuda"
        if stage == "human_optim":
            resolved.setdefault("stages", {}).setdefault("kinematics_1", {})["overlay"] = bool(
                config.overlay.get(stage, False)
            )
        return resolved
    if stage == "contacts":
        resolved = base
        resolved.setdefault("runtime", {})["device"] = "cuda"
        resolved.setdefault("contacts", {})["overlay"] = bool(
            config.overlay.get(stage, False)
        )
        return resolved

    for section_name in spec.config_sections:
        resolved[section_name] = dict(base.get(section_name, {}))
    section = resolved[spec.config_sections[0]]
    section["skip_existing"] = config.skip_existing
    if spec.output_config_key is not None:
        section[spec.output_config_key] = "."
    if stage in config.overlay:
        section["overlay"] = config.overlay[stage]
    if stage == "sam3":
        section.update(
            text=config.text,
            confidence=config.confidence,
            part_length=config.part_length,
        )
    elif stage == "vggt":
        section["save_depth"] = True
    elif stage == "sapiens":
        section["tasks"] = "pose"
    elif stage == "sam3d":
        section["inference_type"] = "full"
    elif stage == "geometry":
        for section_name in ("scale", "fuse"):
            resolved[section_name]["skip_existing"] = config.skip_existing
    return resolved


def _write_stage_config(stage: str, config: CorpusConfig, path: Path) -> None:
    """Write the per-shard config consumed by a stage CLI or in-process stage."""
    path.write_text(_dump_toml(_stage_config_data(stage, config)))


# ------------------------------ worker ------------------------------------


def _build_farm(entry: dict[str, Any], root: Path, spec: StageSpec) -> Path:
    """Create a write-through feature symlink farm for one scene."""
    farm = root / entry["scene_id"]
    farm.mkdir(parents=True, exist_ok=True)
    for name in spec.farm_inputs:
        link = farm / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(entry["inputs"][name])
    output = Path(entry["output"])
    output.mkdir(parents=True, exist_ok=True)
    link = farm / spec.out_stage
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(output)
    return farm


def _drop_farm(farm: Path, spec: StageSpec) -> None:
    """Unlink only the farm links, never their feature-tree targets."""
    for name in (*spec.farm_inputs, spec.out_stage):
        link = farm / name
        if link.is_symlink() or link.exists():
            link.unlink()
    if farm.is_dir():
        farm.rmdir()


def _manifest_args(spec: StageSpec, manifest: Path, save_dir: Path, config: Path) -> list[str]:
    """Build a manifest stage's path-only CLI arguments."""
    assert spec.cli is not None
    return [
        sys.executable,
        str(spec.cli),
        "--source",
        str(manifest),
        "--save-dir",
        str(save_dir),
        "--config",
        str(config),
    ]


def _run_manifest_worker(spec: StageSpec, manifest: Path, config_path: Path) -> int:
    """Run one manifest stage subprocess for a GPU shard."""
    entries = json.loads(manifest.read_text())
    if not entries:
        return 0
    return subprocess.run(
        _manifest_args(spec, manifest, manifest.parent, config_path), check=False
    ).returncode


def _run_farm_worker(
    stage: str, spec: StageSpec, manifest: Path, config_path: Path, skip_existing: bool
) -> int:
    """Run a geometry or human-optimization shard in-process, isolating scene failures."""
    sys.path.insert(0, str(_STAGES_DIR))
    import torch  # noqa: E402  (worker-only; the orchestrator stays torch-free)

    if stage == "geometry":
        import estimate_scale as scale_stage  # noqa: E402
        import fuse_geometry as fuse_stage  # noqa: E402
    elif stage == "gravity":
        import estimate_gravity_geocalib as gravity_stage  # noqa: E402
    else:
        import human_optimization as human_stage  # noqa: E402

    entries = json.loads(manifest.read_text())
    farms_root = manifest.parent / "farms"
    farms_root.mkdir(parents=True, exist_ok=True)
    ok = 0
    for index, entry in enumerate(entries, 1):
        sid = entry["scene_id"]
        out = Path(entry["output"]) / spec.out_check
        done = out.is_file() and (stage != "geometry" or _geometry_metric(out.parent))
        if skip_existing and done:
            print(f"[{index}/{len(entries)}] {sid}: skip ({spec.out_check} exists)", flush=True)
            ok += 1
            continue
        started = time.monotonic()
        farm = _build_farm(entry, farms_root, spec)
        try:
            if stage == "geometry":
                # fuse builds the up-to-scale clouds; scale then metricizes them in place.
                fuse_rc = fuse_stage.main(
                    [
                        "--source",
                        entry["clip"],
                        "--predictions",
                        str(farm),
                        "--config",
                        str(config_path),
                    ]
                )
                scale_rc = scale_stage.main(
                    ["--predictions", str(farm), "--config", str(config_path)]
                )
                if fuse_rc or scale_rc:
                    raise RuntimeError(f"stage return codes: fuse={fuse_rc}, scale={scale_rc}")
            elif stage == "gravity":
                rc = gravity_stage.main(
                    [
                        "--source",
                        entry["clip"],
                        "--predictions",
                        str(farm),
                        "--config",
                        str(config_path),
                    ]
                )
                if rc:
                    raise RuntimeError(f"gravity return code: {rc}")
            else:
                rc = human_stage.main(
                    [
                        "--predictions",
                        str(farm),
                        "--stages",
                        _HUMAN_STAGE_ARGS[stage],
                        "--config",
                        str(config_path),
                    ]
                )
                if rc:
                    raise RuntimeError(f"human_optimization return code: {rc}")
            if out.is_file():
                ok += 1
            print(f"[{index}/{len(entries)}] {sid}: done in {time.monotonic() - started:.1f}s",
                  flush=True)
        # SystemExit included: the in-process stage entry points raise it for exactly
        # the per-scene failures a sweep must survive (missing/stale inputs, non-metric
        # geometry) — it must not kill the shard and strand every scene behind it.
        except (Exception, SystemExit):  # noqa: BLE001 — a bad scene must not abort its shard
            print(f"[{index}/{len(entries)}] {sid}: FAILED after "
                  f"{time.monotonic() - started:.1f}s", flush=True)
            traceback.print_exc()
        finally:
            _drop_farm(farm, spec)
            # Release the allocator cache so co-located shards hold only their
            # CURRENT scene's working set, not the peak of the biggest scene seen.
            gc.collect()
            torch.cuda.empty_cache()
    print(f"[worker] {ok}/{len(entries)} scene(s) have {spec.out_stage}/{spec.out_check}")
    return 0 if ok == len(entries) else 1


def _run_contacts_worker(
    manifest: Path,
    config_path: Path,
    provenance_config: Path,
    bodies: str,
    skip_existing: bool,
) -> int:
    """Run contacts in one subprocess per scene and capture only its output artifact."""
    entries = json.loads(manifest.read_text())
    tag = bodies.removeprefix("kinematic_")
    counts = {"done": 0, "skip": 0, "fail": 0}
    for entry in entries:
        sid = entry["scene_id"]
        body = Path(entry["body"])
        geom = Path(entry["geometry"])
        out = Path(entry["output"])
        try:
            if (not body.is_file() or not (geom / "scene.npz").is_file()
                    or not (geom / "transform.npz").is_file()):
                print(f"  [skip] {sid}: missing body/geometry inputs")
                counts["skip"] += 1
                continue
            if skip_existing and _is_fresh(out, body, geom, provenance_config):
                counts["skip"] += 1
                continue
            if skip_existing and out.is_file():
                print(f"  [stale] {sid}: regenerating contacts whose provenance no longer matches")
            with tempfile.TemporaryDirectory(prefix=f"contacts_{sid}_") as temporary:
                farm = Path(temporary)
                (farm / "human_optim").mkdir()
                (farm / "human_optim" / f"{bodies}.npz").symlink_to(body.resolve())
                (farm / "geometry").symlink_to(geom.resolve())
                geocalib = Path(entry["geocalib"]) if "geocalib" in entry else None
                if geocalib is not None and geocalib.is_dir():  # optional gravity input
                    (farm / "geocalib").symlink_to(geocalib.resolve())
                cmd = [
                    sys.executable,
                    str(_CONTACTS_CLI),
                    "--source",
                    entry["clip"],
                    "--predictions",
                    str(farm),
                    "--bodies",
                    bodies,
                    "--config",
                    str(config_path),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                produced = farm / "human_optim" / f"contacts_{tag}.npz"
                if result.returncode != 0 or not produced.is_file():
                    print(f"  [FAIL] {sid} (rc={result.returncode})\n"
                          f"{result.stderr.strip()[-400:]}")
                    counts["fail"] += 1
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(produced), str(out))
            counts["done"] += 1
        except Exception:  # noqa: BLE001 — a bad scene must not abort its shard
            counts["fail"] += 1
            print(f"  [FAIL] {sid}: unexpected contact worker error")
            traceback.print_exc()
    print(f"[worker] {counts}")
    return 0 if counts["fail"] == 0 else 1


def run_worker(
    stage: str,
    manifest: Path,
    gpu: int,
    config_path: Path,
    provenance_config: Path,
    bodies: str,
    skip_existing: bool,
) -> None:
    """Generic GPU worker dispatching one of the three invocation modes."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTHONUNBUFFERED"] = "1"
    spec = STAGES[stage]
    if spec.mode == "manifest_subprocess":
        code = _run_manifest_worker(spec, manifest, config_path)
    elif spec.mode == "farm_inprocess":
        code = _run_farm_worker(stage, spec, manifest, config_path, skip_existing)
    else:
        code = _run_contacts_worker(
            manifest, config_path, provenance_config, bodies, skip_existing
        )
    raise SystemExit(code)


# --------------------------- reporting/hooks ------------------------------


def _input_paths(
    paths: ScenePaths, stage: str, spec: StageSpec, row: sqlite3.Row, config: CorpusConfig
) -> dict[str, Path]:
    """Resolve every input path shown by a dry-run."""
    sid = row["scene_id"]
    if spec.mode == "manifest_subprocess":
        result = _dependency_paths(paths, stage, sid)
        if stage == "sam3":
            result["tracking"] = paths.tracking_npz(sid)
        return result
    if stage == "contacts":
        body, geom = _contacts_inputs(paths, sid, config.bodies)
        return {
            config.bodies: body,
            "scene": geom / "scene.npz",
            "transform": geom / "transform.npz",
        }
    return {
        name: paths.feature_dir(name, sid) for name in spec.farm_inputs
    }


def _dry_run(
    paths: ScenePaths,
    out_paths: ScenePaths,
    stage: str,
    rows: list[sqlite3.Row],
    config: CorpusConfig,
) -> None:
    """Print selected scenes, all resolved paths, and would-run state without writes."""
    spec = STAGES[stage]
    tag = config.bodies.removeprefix("kinematic_")
    fresh_count = runnable = 0
    for row in rows:
        sid = row["scene_id"]
        fresh = _scene_fresh(paths, out_paths, spec, sid, config)
        inputs = _input_paths(paths, stage, spec, row, config)
        usable = True
        detail = ""
        if spec.mode == "manifest_subprocess":
            entry = _manifest_entry(paths, stage, row, config, report=False)
            usable = entry is not None
            if stage == "sam3" and entry is not None:
                boxes = ", ".join(
                    f"[{','.join(f'{value:.0f}' for value in prompt['select_bbox'])}]"
                    for prompt in entry["prompts"]
                )
                detail = f" prompts={len(entry['prompts'])} boxes={boxes}"
        elif stage == "contacts":
            usable = all(path.is_file() for path in inputs.values())
        would_run = usable and not (config.skip_existing and fresh)
        fresh_count += int(fresh)
        runnable += int(would_run)
        state = "fresh" if fresh else "pending"
        print(
            f"  {sid:24s} {row['width']}x{row['height']} {row['n_frames']}f "
            f"[{state}] would-run={'yes' if would_run else 'no'}{detail}"
        )
        print(f"      source={paths.clip_path(row)}")
        for name, path in inputs.items():
            print(f"      input.{name}={path} ({'OK' if path.exists() else 'MISSING'})")
        print(f"      output={_out_path(paths, out_paths, spec, sid, tag)}")
    print(
        f"[dry-run stage={stage}] selected={len(rows)} would-run={runnable} "
        f"fresh={fresh_count}"
    )


def _gather_overlays(
    paths: ScenePaths,
    spec: StageSpec,
    rows: list[sqlite3.Row],
    destination: Path,
) -> int:
    """Gather per-scene QA overlays as scene-id symlinks."""
    assert spec.overlay_artifact is not None
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in rows:
        sid = row["scene_id"]
        source = paths.feature_dir(spec.out_stage, sid) / spec.overlay_artifact
        if not source.is_file():
            continue
        link = destination / f"{sid}.mp4"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(source.resolve())
        count += 1
    return count


def _post_sam3(
    paths: ScenePaths,
    rows: list[sqlite3.Row],
    config: CorpusConfig,
    workers_clean: bool,
) -> None:
    """Gather SAM3 overlays and conditionally perform guarded QA discard write-back."""
    spec = STAGES["sam3"]
    if config.overlay.get("sam3", True):
        destination = config.debug_overlay_dir or (
            paths.features_root / "sam3" / "_debug_overlays"
        )
        count = _gather_overlays(paths, spec, rows, destination)
        print(f"[overlays] gathered {count}/{len(rows)} overlay(s) → {destination}")

    if config.discard_failed and not workers_clean:
        print("\n[discard] SKIPPED — a worker exited non-zero. Re-run with "
              "skip_existing=true to finish, then the discard pass will run.")
    elif config.discard_failed:
        discards = discard_failures(paths, rows, config.min_presence)
        sam3_root = paths.features_root / "sam3"
        sam3_root.mkdir(parents=True, exist_ok=True)
        (sam3_root / "_discarded.json").write_text(json.dumps(discards, indent=2))
        if discards:
            discarded_ids = {record["scene_id"] for record in discards}
            discarded_rows = [row for row in rows if row["scene_id"] in discarded_ids]
            _gather_overlays(
                paths, spec, discarded_rows, sam3_root / "_discarded_overlays"
            )
        print(f"\n[discard] {len(discards)}/{len(rows)} scene(s) discarded "
              f"(presence < {config.min_presence}); human_selected → 0, filter_reason recorded")
        for record in sorted(discards, key=lambda value: value["present"]):
            print(f"   {record['scene_id']:24s} present={record['present']:.2f}  "
                  f"({record['reason']})")


# ---------------------------- orchestrator --------------------------------


def _assemble_shard(
    paths: ScenePaths,
    out_paths: ScenePaths,
    stage: str,
    rows: list[sqlite3.Row],
    config: CorpusConfig,
) -> list[dict[str, Any]]:
    """Run the selected stage's input-assembler hook for one stride shard."""
    spec = STAGES[stage]
    entries: list[dict[str, Any]] = []
    for row in rows:
        if spec.mode == "manifest_subprocess":
            entry = _manifest_entry(paths, stage, row, config, report=True)
            if entry is not None:
                entries.append(entry)
        elif stage == "contacts":
            entries.append(_contacts_entry(paths, out_paths, row, config))
        else:
            entries.append(_farm_entry(paths, spec, row))
    return entries


def _run_orchestrator(
    stage: str, rows: list[sqlite3.Row], paths: ScenePaths, config: CorpusConfig
) -> int:
    """Write shard plans, launch workers, and apply the shared completion report."""
    spec = STAGES[stage]
    out_paths = ScenePaths(config.out_root) if config.out_root else paths
    work_paths = out_paths if spec.capture_and_move else paths
    work_dir = work_paths.features_root / spec.out_stage / "_runs" / stage
    work_dir.mkdir(parents=True, exist_ok=True)

    context = multiprocessing.get_context("spawn")
    workers: list[tuple[int, int, multiprocessing.Process]] = []
    for shard_index, gpu in enumerate(config.devices):
        shard_rows = scene_db.stride_shard(rows, shard_index, len(config.devices))
        entries = _assemble_shard(paths, out_paths, stage, shard_rows, config)
        if not entries:
            print(f"[shard {shard_index}/{len(config.devices)} gpu={gpu}] no scenes — skipping")
            continue
        shard_dir = work_dir / f"shard_{shard_index}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        manifest = shard_dir / "manifest.json"
        manifest.write_text(json.dumps(entries, indent=2))
        stage_config = shard_dir / "stage.toml"
        _write_stage_config(stage, config, stage_config)
        process = context.Process(
            target=run_worker,
            args=(
                stage,
                manifest,
                gpu,
                stage_config,
                config.human_optim_config,
                config.bodies,
                config.skip_existing,
            ),
        )
        process.start()
        print(
            f"[shard {shard_index}/{len(config.devices)} gpu={gpu}] "
            f"{len(entries)} scene(s) → {manifest.name}"
        )
        workers.append((shard_index, gpu, process))

    codes: list[int] = []
    for shard_index, gpu, process in workers:
        process.join()
        code = process.exitcode if process.exitcode is not None else 1
        codes.append(code)
        print(f"[shard {shard_index}/{len(config.devices)} gpu={gpu}] exit code {code}")
    workers_clean = all(code == 0 for code in codes)

    if stage == "sam3":
        _post_sam3(paths, rows, config, workers_clean)
    elif spec.overlay_artifact is not None and config.overlay.get(stage, False):
        destination = config.debug_overlay_dir or (
            paths.features_root / spec.out_stage / "_debug_overlays"
        )
        count = _gather_overlays(paths, spec, rows, destination)
        print(f"[overlays] gathered {count}/{len(rows)} overlay(s) → {destination}")

    fresh_ids = [
        row["scene_id"]
        for row in rows
        if _scene_fresh(paths, out_paths, spec, row["scene_id"], config)
    ]
    noun = "fresh" if stage == "contacts" else "complete"
    print(
        f"\n[corpus stage={stage}] {len(fresh_ids)}/{len(rows)} scene(s) {noun}: "
        f"{spec.out_stage}/{spec.out_check.format(tag=config.bodies.removeprefix('kinematic_'))}"
    )
    missing = [row["scene_id"] for row in rows if row["scene_id"] not in set(fresh_ids)]
    if missing:
        print(f"[corpus stage={stage}] {len(missing)} scene(s) missing or stale:")
        for sid in missing:
            print(f"   {sid}")

    # The legacy SAM3 driver treated scenes without usable prompts as reported skips;
    # all later stages require every selected artifact to be complete/fresh.
    outputs_ok = stage == "sam3" or not missing
    return 0 if workers_clean and outputs_ok else 1


def _escape_scene_args(argv: list[str]) -> tuple[list[str], str]:
    """Let ``--scenes`` accept corpus ids beginning with ``-`` under argparse."""
    marker = "__CORPUS_SCENE__"
    known = {"--stage", "--sample", "--limit", "--config", "--dry-run", "-h", "--help"}
    result: list[str] = []
    in_scenes = False
    for token in argv:
        if token == "--scenes":
            in_scenes = True
            result.append(token)
            continue
        if token.startswith("--scenes="):
            in_scenes = True
            result.extend(("--scenes", marker + token.split("=", 1)[1]))
            continue
        is_known = token in known or any(token.startswith(f"{flag}=") for flag in known)
        if in_scenes and is_known:
            in_scenes = False
        result.append(marker + token if in_scenes else token)
    return result, marker


def main(argv: list[str] | None = None) -> int:
    """Parse corpus selection, then dry-run or execute one configured stage."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=tuple(STAGES))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--scenes", nargs="+", metavar="ID")
    selection.add_argument("--sample", type=int, metavar="N")
    selection.add_argument("--limit", type=int, metavar="N")
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    escaped_argv, scene_marker = _escape_scene_args(raw_argv)
    args = parser.parse_args(escaped_argv)
    if args.scenes is not None:
        args.scenes = [value.removeprefix(scene_marker) for value in args.scenes]
    if args.sample is not None and args.sample < 0:
        parser.error("--sample must be non-negative")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")

    try:
        config = load_config(args.config)
    except (OSError, TypeError, ValueError, tomllib.TOMLDecodeError) as error:
        parser.error(f"invalid corpus config: {error}")
    paths = ScenePaths(config.data_root)
    if not paths.scenes_db.is_file():
        print(f"ERROR: scenes.db not found at {paths.scenes_db}", file=sys.stderr)
        return 1
    with paths.connect() as connection:
        rows = select_targets(
            connection,
            categories=config.categories,
            rope_supported=config.rope_supported,
        )
    rows = scene_db.resolve_rows(
        rows,
        scenes=args.scenes,
        sample=args.sample,
        limit=args.limit,
        seed=config.seed,
    )
    print(
        f"[corpus stage={args.stage}] target scenes: {len(rows)} "
        f"(categories={config.categories}, rope_supported={config.rope_supported})"
    )
    if not rows:
        return 0
    if args.dry_run:
        out_paths = ScenePaths(config.out_root) if config.out_root else paths
        _dry_run(paths, out_paths, args.stage, rows, config)
        return 0
    return _run_orchestrator(args.stage, rows, paths, config)


if __name__ == "__main__":
    raise SystemExit(main())
