"""Export the auto-labeled ClimbingVideos corpus as a per-joint contact training set.

Reads the pipeline features for every selected + auto-labeled scene and writes a
self-contained dataset under ``<out>/`` so training never touches the pipeline again:

    <out>/
      dataset_info.json                      # provenance + schema + joint names
      train/  index.json  <scene>/frames/ masks/<obj>/ labels.npz
      test/   index.json  <scene>/frames/ masks/<obj>/ inputs.npz  contacts.npz

Frames are extracted from each trimmed scene clip in cv2-sequential order (the same
order every per-frame array — bboxes, masks, contacts, intrinsics — was indexed in),
so frame ``k`` on disk lines up with row ``k`` of every ``(…, N, …)`` array.

Contacts are per **22 SMPL-X body joints** (0–21): the 30 finger joints are OR-folded
into the two wrists (``left_wrist`` 20, ``right_wrist`` 21). A merged hand's confidence
is the strongest touch when it is in contact (maximum over the touching parts) and the
weakest member when it is free (minimum over all 16 parts): the folded free label is an
AND, so one occluded/unknown finger makes the whole-hand free label uncertain. The 3D
viewer uses the same fold.

Train scenes carry the 22-joint auto-labels + confidence. Test scenes carry the
SAM-3D-Body inputs (bbox / mask / intrinsics); their contact labels come from the manual
annotation (``--fill-test``): the 14-joint tri-state annotation mapped onto the 22 body
joints as ``joint_contact_22`` (bool) + an ``annotated_22`` score mask. Only those 14
joints can be scored: unlabeled (-1) values, ignored people, and the other eight SMPL-X
body joints remain outside the score mask.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import scene_db  # noqa: E402  (sibling module in scripts/)

# 52 SMPLXMid joints = 22 body (0–21) + 30 fingers (22–51). The two hands fold the
# wrist + its 15 fingers together; every other output joint is one body joint.
LEFT_HAND_GROUP = [20] + list(range(22, 37))   # left_wrist + left index/middle/pinky/ring/thumb ×3
RIGHT_HAND_GROUP = [21] + list(range(37, 52))  # right_wrist + right fingers
HAND_JOINTS = ((20, LEFT_HAND_GROUP), (21, RIGHT_HAND_GROUP))
N_JOINTS_22 = 22
N_JOINTS_52 = 52
CONTACT_CONFIDENCE_SCHEMA = 4
CONTACT_CONFIDENCE_METHOD = "motion-only-joint-label-v1"


def merge_contacts_52_to_22(
    jc52: np.ndarray, conf52: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fold 52-joint contacts + confidence into the 22 SMPL-X body joints.

    Confidence follows the folded label's logic: strongest touching vote for the
    contact OR, weakest member for the no-contact AND.

    :param jc52: ``(P, N, 52)`` bool contact labels.
    :param conf52: ``(P, N, 52)`` float32 label confidence.
    :returns: ``(jc22 (P,N,22) bool, conf22 (P,N,22) float32)``.
    """
    jc52 = np.asarray(jc52, bool)
    conf52 = np.asarray(conf52, np.float32)
    if jc52.ndim != 3 or jc52.shape[-1] != N_JOINTS_52:
        raise ValueError(f"joint_contact must have shape (P,N,52), got {jc52.shape}")
    if conf52.shape != jc52.shape:
        raise ValueError(
            f"label_confidence shape {conf52.shape} does not match contacts {jc52.shape}"
        )
    if not np.isfinite(conf52).all() or bool(((conf52 < 0.0) | (conf52 > 1.0)).any()):
        raise ValueError("label_confidence must be finite and within [0, 1]")

    jc22 = jc52[..., :N_JOINTS_22].copy()
    conf22 = conf52[..., :N_JOINTS_22].copy()
    for wrist, group in HAND_JOINTS:
        sub_lbl = jc52[..., group]                       # (P, N, 16)
        sub_conf = conf52[..., group].astype(np.float32)
        lbl = sub_lbl.any(axis=-1)                       # (P, N)
        conf_touch = np.where(sub_lbl, sub_conf, -np.inf).max(axis=-1)  # strongest touching vote
        conf_free = sub_conf.min(axis=-1)                              # weakest member of the free AND
        jc22[..., wrist] = lbl
        conf22[..., wrist] = np.where(lbl, conf_touch, conf_free)
    return jc22, conf22


def _expected_names(n: int, suffix: str) -> set[str]:
    return {f"{k:06d}{suffix}" for k in range(n)}


def _validate_indexed_files(out_dir: Path, n: int, suffix: str, what: str) -> None:
    """Require exactly ``000000<suffix> ... (n-1)<suffix>`` in ``out_dir``."""
    expected = _expected_names(n, suffix)
    actual = {p.name for p in out_dir.glob(f"*{suffix}") if p.is_file()}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"{what}: incomplete/stale export: missing={missing[:5]} "
            f"extra={extra[:5]} (expected {n}, found {len(actual)})"
        )


def _extract_frames(
    clip: Path,
    out_dir: Path,
    n_expected: int,
    quality: int,
    *,
    skip_existing: bool = False,
) -> int:
    """Decode ``clip`` sequentially, resuming missing JPEGs when requested."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        cap.release()
        raise OSError(f"cannot open video {clip}")
    k = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        dst = out_dir / f"{k:06d}.jpg"
        if not skip_existing or not dst.is_file():
            Image.fromarray(bgr[..., ::-1]).save(dst, quality=quality)
        k += 1
    cap.release()
    if k != n_expected:
        raise ValueError(f"{clip}: decoded {k} frames, arrays expect {n_expected}")
    _validate_indexed_files(out_dir, n_expected, ".jpg", f"{clip.name} frames")
    return k


def _copy_masks(
    sam3_dir: Path,
    out_dir: Path,
    obj_ids: list[int],
    n: int,
    *,
    skip_existing: bool = False,
) -> None:
    """Copy SAM-3 masks, resuming missing PNGs and validating the exact tree."""
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_dirs = {f"{oid:02d}" for oid in obj_ids}
    actual_dirs = {p.name for p in out_dir.iterdir() if p.is_dir()}
    extra_dirs = sorted(actual_dirs - expected_dirs)
    if extra_dirs:
        raise ValueError(f"stale mask person directories in {out_dir}: {extra_dirs[:5]}")
    for oid in obj_ids:
        src = sam3_dir / f"{oid:02d}"
        dst = out_dir / f"{oid:02d}"
        dst.mkdir(parents=True, exist_ok=True)
        for k in range(n):
            p = src / f"frame_{k:06d}.png"
            if not p.is_file():
                raise FileNotFoundError(f"missing SAM-3 mask {p}")
            target = dst / f"{k:06d}.png"
            if not skip_existing or not target.is_file():
                shutil.copyfile(p, target)
        _validate_indexed_files(dst, n, ".png", f"object {oid} masks")


def _aligned_object_ids(a: np.ndarray, b: np.ndarray, sid: str) -> list[int]:
    """Assert two sources agree on person order and return the id list."""
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.tolist() != b.tolist():
        raise ValueError(f"{sid}: object_ids mismatch {a.tolist()} vs {b.tolist()}")
    out = [int(x) for x in a]
    if len(out) != len(set(out)):
        raise ValueError(f"{sid}: duplicate object_ids {out}")
    return out


def _require_shape(sid: str, name: str, value: np.ndarray, shape: tuple[int, ...]) -> None:
    actual = np.asarray(value).shape
    if actual != shape:
        raise ValueError(f"{sid}: {name} has shape {actual}, expected {shape}")


def _require_sequential_frames(sid: str, name: str, data, n: int) -> None:
    """The exporter decodes the full clip, so feature rows must be sequential 0..N-1."""
    if "frame_indices" not in data.files:
        return
    fi = np.asarray(data["frame_indices"], np.int64)
    if fi.shape != (n,) or not np.array_equal(fi, np.arange(n, dtype=np.int64)):
        raise ValueError(
            f"{sid}: {name}.frame_indices is not sequential 0..{n - 1}; "
            "full-clip frame export would be misaligned"
        )


def _write_pending_test_contacts(
    path: Path,
    *,
    sid: str,
    obj_ids: list[int],
    n: int,
    joint_names: list[str],
    valid: np.ndarray,
    skip_existing: bool,
) -> bool:
    """Write a pending test-label placeholder, preserving completed manual labels.

    :returns: whether ``path`` already contains validated, completed labels.
    """
    if skip_existing and path.is_file():
        with np.load(path, allow_pickle=False) as existing:
            if "pending" not in existing.files:
                raise ValueError(f"{sid}: existing test contacts has no pending state")
            pending = np.asarray(existing["pending"])
            if pending.shape != ():
                raise ValueError(f"{sid}: existing contacts pending must be scalar")
            if not bool(pending.item()):
                p = len(obj_ids)
                _require_shape(
                    sid, "existing joint_contact_22", existing["joint_contact_22"],
                    (p, n, N_JOINTS_22),
                )
                _require_shape(
                    sid, "existing annotated_22", existing["annotated_22"],
                    (p, n, N_JOINTS_22),
                )
                _require_shape(sid, "existing object_ids", existing["object_ids"], (p,))
                _require_shape(sid, "existing valid_mask", existing["valid_mask"], (p, n))
                _require_shape(sid, "existing ignored", existing["ignored"], (p,))
                _require_sequential_frames(sid, "existing test contacts", existing, n)
                if np.asarray(existing["object_ids"], np.int64).tolist() != obj_ids:
                    raise ValueError(f"{sid}: completed test-label object_ids are stale")
                if [str(x) for x in existing["joint_names_22"]] != joint_names:
                    raise ValueError(f"{sid}: completed test-label joint order is stale")
                if not np.array_equal(np.asarray(existing["valid_mask"], bool), valid):
                    raise ValueError(f"{sid}: completed test-label valid_mask is stale")
                return True

    np.savez_compressed(
        path,
        joint_contact_22=np.zeros((len(obj_ids), n, N_JOINTS_22), bool),
        annotated_22=np.zeros((len(obj_ids), n, N_JOINTS_22), bool),
        joint_names_22=np.asarray(joint_names),
        object_ids=np.asarray(obj_ids, np.int32),
        frame_indices=np.arange(n, dtype=np.int32),
        valid_mask=valid,
        ignored=np.zeros(len(obj_ids), bool),
        pending=np.bool_(True),
    )
    return False


def export_scene(task: dict) -> dict:
    """Export one scene (frames + masks + labels/inputs). Returns an index summary."""
    sid, split = task["sid"], task["split"]
    feat, out_scene = Path(task["feat"]), Path(task["out_scene"])
    aa, bb = sid[:2], sid[2:4]
    hd = feat / "human_optim" / aa / bb / sid
    sam3 = feat / "sam3" / aa / bb / sid
    geom = feat / "geometry" / aa / bb / sid

    contacts = np.load(hd / "contacts_1.npz", allow_pickle=True)
    boxes = np.load(sam3 / "bboxes.npz", allow_pickle=True)
    transform = np.load(geom / "transform.npz", allow_pickle=True)

    n = int(contacts["num_frames"])
    obj_ids = _aligned_object_ids(contacts["object_ids"], boxes["object_ids"], sid)
    p = len(obj_ids)
    joint_names = [str(x) for x in contacts["joint_names"]][:N_JOINTS_22]
    valid = np.asarray(contacts["valid_mask"], bool)                 # (P, N)
    bbox = np.asarray(boxes["bboxes_per_obj"], np.int32)             # (P, N, 4) xyxy px
    intr = np.asarray(transform["intrinsics_px_orig"], np.float32)   # (N, 3, 3) orig res
    _require_shape(sid, "valid_mask", valid, (p, n))
    _require_shape(sid, "bboxes_per_obj", bbox, (p, n, 4))
    _require_shape(sid, "intrinsics_px_orig", intr, (n, 3, 3))
    if len(joint_names) != N_JOINTS_22 or joint_names[20:] != ["left_wrist", "right_wrist"]:
        raise ValueError(f"{sid}: unexpected SMPL-X joint order: {joint_names}")
    _require_sequential_frames(sid, "sam3/bboxes", boxes, n)
    _require_sequential_frames(sid, "geometry/transform", transform, n)
    _require_sequential_frames(sid, "human_optim/contacts", contacts, n)

    _extract_frames(
        Path(task["clip"]), out_scene / "frames", n, task["quality"],
        skip_existing=task["skip_existing"],
    )
    _copy_masks(
        sam3, out_scene / "masks", obj_ids, n,
        skip_existing=task["skip_existing"],
    )

    common = dict(
        object_ids=np.asarray(obj_ids, np.int32),
        frame_indices=np.arange(n, dtype=np.int32),
        bbox=bbox,
        intrinsics=intr,
        valid_mask=valid,
        joint_names_22=np.asarray(joint_names),
        video_height=np.int32(contacts["video_height"]),
        video_width=np.int32(contacts["video_width"]),
        fps=np.float32(contacts["fps"]),
    )

    test_labels_ready = False
    if split == "train":
        if "confidence_schema" not in contacts.files:
            raise ValueError(f"{sid}: contacts_1 has no confidence_schema; regenerate it")
        confidence_schema = int(np.asarray(contacts["confidence_schema"]).item())
        if confidence_schema != CONTACT_CONFIDENCE_SCHEMA:
            raise ValueError(
                f"{sid}: contacts_1 confidence_schema={confidence_schema}, expected "
                f"{CONTACT_CONFIDENCE_SCHEMA}; regenerate it before export")
        confidence_method = str(np.asarray(contacts["confidence_method"]).item())
        if confidence_method != CONTACT_CONFIDENCE_METHOD:
            raise ValueError(
                f"{sid}: contacts_1 confidence_method={confidence_method!r}, expected "
                f"{CONTACT_CONFIDENCE_METHOD!r}; regenerate it before export")
        _require_shape(sid, "joint_contact", contacts["joint_contact"], (p, n, N_JOINTS_52))
        _require_shape(
            sid, "label_confidence", contacts["label_confidence"], (p, n, N_JOINTS_52)
        )
        confidence_52 = np.asarray(contacts["label_confidence"], np.float32)
        if (not np.isfinite(confidence_52).all()
                or bool(((confidence_52 < 0.0) | (confidence_52 > 1.0)).any())):
            raise ValueError(f"{sid}: label_confidence must be finite and in [0,1]")
        source_files = (
            (hd / "kinematic_1.npz", "source_body_sha256"),
            (geom / "scene.npz", "source_scene_sha256"),
            (geom / "transform.npz", "source_transform_sha256"),
        )
        for source_path, hash_key in source_files:
            with source_path.open("rb") as source_file:
                current_hash = hashlib.file_digest(source_file, "sha256").hexdigest()
            if hash_key not in contacts.files or str(np.asarray(contacts[hash_key]).item()) != current_hash:
                raise ValueError(
                    f"{sid}: contacts_1 is stale for {source_path.name}; regenerate it before export")
        jc22, conf22 = merge_contacts_52_to_22(
            np.asarray(contacts["joint_contact"], bool),
            confidence_52,
        )
        np.savez_compressed(
            out_scene / "labels.npz",
            joint_contact_22=jc22, contact_conf_22=conf22,
            confidence_schema=np.int32(confidence_schema),
            confidence_method=np.str_(confidence_method), **common,
        )
    else:  # test: inputs now, contacts pending (manual annotation)
        np.savez_compressed(out_scene / "inputs.npz", **common)
        test_labels_ready = _write_pending_test_contacts(
            out_scene / "contacts.npz", sid=sid, obj_ids=obj_ids, n=n,
            joint_names=joint_names, valid=valid,
            skip_existing=task["skip_existing"],
        )

    return dict(
        scene_id=sid, n_frames=n, n_people=len(obj_ids), object_ids=obj_ids,
        valid_frames=[int(v) for v in valid.sum(1)],
        height=int(contacts["video_height"]), width=int(contacts["video_width"]),
        fps=float(contacts["fps"]),
        labels_ready=(split == "train" or test_labels_ready),
    )


def _safe_export(task: dict) -> dict:
    """Top-level (picklable) worker: run :func:`export_scene`, capturing any failure."""
    try:
        return export_scene(task)
    except Exception as e:  # tolerate per-scene failure, keep the batch going
        traceback.print_exc()
        return {"scene_id": task["sid"], "split": task["split"], "error": str(e)}


# BetterContactAnnotator's 14 manual-annotation joints (schema v2) → SMPL-X 22-body-joint
# index. Maps by name; hands land on the wrist (fingers are already OR-folded into it in the
# 22-joint space). "foot" is the big-toe joint (10/11), kept distinct from the ankle (7/8).
# The 8 SMPL-X joints absent from the annotation — pelvis, spine1/2/3, neck, both collars,
# head — remain unannotated, never invented as negative labels. Annotation on disk is
# (P, 14, N) int8 tri-state (-1 unlabeled / 0 free / 1 contact), person-matched by object_id.
ANNOTATION_JOINTS = (
    "left_hand", "right_hand", "left_foot", "right_foot", "left_ankle", "right_ankle",
    "left_knee", "right_knee", "left_elbow", "right_elbow", "left_shoulder",
    "right_shoulder", "left_hip", "right_hip",
)
ANNOTATION_TO_SMPLX22 = {
    "left_hand": 20, "right_hand": 21, "left_foot": 10, "right_foot": 11,
    "left_ankle": 7, "right_ankle": 8, "left_knee": 4, "right_knee": 5,
    "left_elbow": 18, "right_elbow": 19, "left_shoulder": 16, "right_shoulder": 17,
    "left_hip": 1, "right_hip": 2,
}


def _test_label_mapping() -> dict:
    """The 14-joint manual annotation → 22 SMPL-X body-joint map used to fill test labels."""
    obs = sorted(ANNOTATION_TO_SMPLX22.values())
    return {
        "annotation_joints": list(ANNOTATION_JOINTS),
        "annotation_to_smplx22": ANNOTATION_TO_SMPLX22,
        "observable_joint_indices": obs,
        "unannotated_joint_indices": [i for i in range(N_JOINTS_22) if i not in obs],
        "annotation_source": "features/annotation/<aa>/<bb>/<sid>/annotation.npz "
                             "(BetterContactAnnotator schema v2)",
        "annotation_layout": "(P, 14, N) int8 tri-state (-1 unlabeled / 0 no-contact / 1 contact); "
                            "person-matched by object_id; transpose to (P, N, 22) on fill",
        "note": "hands fold onto the wrist (20/21); the 8 joints absent from the manual "
                "schema (pelvis, spine1/2/3, neck, collars, head) stay outside annotated_22",
    }


def map_annotation_to_22(
    ann_contacts: np.ndarray, ann_names: list[str], ann_oids: list[int],
    data_oids: list[int], n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a manual annotation onto the 22 SMPL-X body joints.

    :param ann_contacts: ``(P_ann, 14, N)`` int8 tri-state (-1 unlabeled / 0 free / 1 contact).
    :param ann_names: the annotation's 14 joint names (mapped by name, not position).
    :param ann_oids: the annotation's per-person track ids.
    :param data_oids: the dataset scene's per-person track ids (output person order).
    :param n_frames: the dataset scene's frame count.
    :returns: ``(joint_contact_22 (P,N,22) bool, annotated_22 (P,N,22) bool)`` — the contact
        label and the score mask (True only where a manual label exists). A person not present
        in the annotation, an unlabeled (-1) joint-frame, and the 8 SMPL-X joints absent from
        the manual schema are left unannotated.
    """
    ann_contacts = np.asarray(ann_contacts)
    if ann_contacts.ndim != 3:
        raise ValueError(f"manual contacts must have shape (P,J,N), got {ann_contacts.shape}")
    if ann_contacts.shape != (len(ann_oids), len(ann_names), n_frames):
        raise ValueError(
            "manual contact shape does not match object_ids/joint_names/frame count: "
            f"{ann_contacts.shape} vs ({len(ann_oids)}, {len(ann_names)}, {n_frames})"
        )
    if len(set(ann_oids)) != len(ann_oids) or len(set(data_oids)) != len(data_oids):
        raise ValueError("manual and dataset object_ids must each be unique")
    if len(set(ann_names)) != len(ann_names):
        raise ValueError("manual joint_names must be unique")
    missing_names = sorted(set(ANNOTATION_JOINTS) - set(ann_names))
    if missing_names:
        raise ValueError(f"manual annotation is missing joints: {missing_names}")
    if not np.isin(ann_contacts, (-1, 0, 1)).all():
        raise ValueError("manual contacts must be tri-state values -1/0/1")

    pd = len(data_oids)
    jc = np.zeros((pd, n_frames, N_JOINTS_22), bool)
    annotated = np.zeros((pd, n_frames, N_JOINTS_22), bool)
    ann_col = {nm: i for i, nm in enumerate(ann_names)}
    oid_to_p = {int(o): i for i, o in enumerate(ann_oids)}
    for p, oid in enumerate(data_oids):
        if int(oid) not in oid_to_p:                       # person never annotated
            continue
        col = ann_contacts[oid_to_p[int(oid)]]             # (14, N_ann)
        for name, sidx in ANNOTATION_TO_SMPLX22.items():
            v = col[ann_col[name]]                          # (N,) int8
            jc[p, :, sidx] = v == 1
            annotated[p, :, sidx] = v != -1
    return jc, annotated


def ignored_mask_for_data(
    ann_oids: list[int], data_oids: list[int], ignored: np.ndarray,
) -> np.ndarray:
    """Map the annotator's ignored-person flags into dataset person order by object id."""
    ignored = np.asarray(ignored, bool)
    if ignored.shape != (len(ann_oids),):
        raise ValueError(
            f"ignored shape {ignored.shape} does not match {len(ann_oids)} annotation people"
        )
    if len(set(ann_oids)) != len(ann_oids) or len(set(data_oids)) != len(data_oids):
        raise ValueError("manual and dataset object_ids must each be unique")
    by_oid = {int(oid): bool(flag) for oid, flag in zip(ann_oids, ignored, strict=True)}
    return np.asarray([by_oid.get(int(oid), False) for oid in data_oids], bool)


def fill_test_labels(out_root: Path, feat: Path, test_sids: list[str]) -> list[dict]:
    """Fill each test scene's ``contacts.npz`` from its manual ``annotation.npz``."""
    summary = []
    for sid in test_sids:
        out_scene = out_root / "test" / sid
        ip = out_scene / "inputs.npz"
        aa, bb = sid[:2], sid[2:4]
        ap = feat / "annotation" / aa / bb / sid / "annotation.npz"
        if not ip.is_file():
            print(f"  ! {sid}: no inputs.npz — run the export first"); continue
        if not ap.is_file():
            print(f"  ! {sid}: MISSING annotation.npz"); summary.append(dict(scene_id=sid, missing=True)); continue
        di = np.load(ip, allow_pickle=True)
        ann = np.load(ap, allow_pickle=True)
        data_oids = [int(x) for x in di["object_ids"]]
        n = int(di["frame_indices"].shape[0])
        ann_oids = [int(x) for x in ann["object_ids"]]
        if int(ann["annotation_version"]) < 2:
            raise ValueError(f"{sid}: expected manual annotation schema v2")
        if int(ann["num_frames"]) != n:
            raise ValueError(
                f"{sid}: annotation has {int(ann['num_frames'])} frames, dataset has {n}"
            )
        jc, annotated = map_annotation_to_22(
            np.asarray(ann["contacts"]), [str(x) for x in ann["joint_names"]],
            ann_oids, data_oids, n,
        )
        ignored = ignored_mask_for_data(ann_oids, data_oids, ann["ignored"])
        annotated[ignored] = False
        jc[ignored] = False
        np.savez_compressed(
            out_scene / "contacts.npz",
            joint_contact_22=jc, annotated_22=annotated,
            joint_names_22=di["joint_names_22"], object_ids=di["object_ids"],
            frame_indices=di["frame_indices"], valid_mask=di["valid_mask"],
            ignored=ignored,
            annotation_version=np.int32(ann["annotation_version"]),
            pending=np.bool_(False),
        )
        eligible = ~ignored
        frac = float(annotated[eligible].mean()) if eligible.any() else 0.0
        contact_rate = float(jc[annotated].mean()) if annotated.any() else 0.0
        summary.append(dict(scene_id=sid, annotated_frac=frac, contact_rate=contact_rate,
                            ignored_people=int(ignored.sum())))
        print(f"  {sid}: annotated {frac:5.0%} of joint-frames, contact {contact_rate:4.0%}")
    return summary


def _finalize_test_index(out_root: Path, summ: list[dict]) -> None:
    """Mark filled test scenes labels-ready in ``test/index.json`` + ``dataset_info.json``."""
    by_sid = {s["scene_id"]: s for s in summ}
    ipath = out_root / "test" / "index.json"
    if ipath.is_file():
        with ipath.open() as f:
            idx = json.load(f)
        for sc in idx["scenes"]:
            s = by_sid.get(sc["scene_id"])
            if s and not s.get("missing"):
                sc["labels_ready"] = True
                sc["annotated_frac"] = round(s["annotated_frac"], 4)
                sc["contact_rate"] = round(s["contact_rate"], 4)
        with ipath.open("w") as f:
            json.dump(idx, f, indent=2)
    dpath = out_root / "dataset_info.json"
    if dpath.is_file():
        with dpath.open() as f:
            info = json.load(f)
        filled = [s for s in summ if not s.get("missing")]
        info["schema_version"] = 2
        info.setdefault("labels", {})["test"] = (
            "manual annotation (14-joint, mapped to 22 SMPL-X body joints); "
            "joint_contact_22 (bool) + annotated_22 (score mask)")
        info["test_labels"] = dict(
            n_filled=len(filled),
            mean_annotated_frac=round(float(np.mean([s["annotated_frac"] for s in filled])), 4) if filled else 0.0,
            ignored_people=int(sum(s.get("ignored_people", 0) for s in filled)),
            note="score only where annotated_22 is True; -1 values, ignored people, and "
                 "the 8 joints absent from the manual schema are excluded",
        )
        info["test_label_mapping"] = _test_label_mapping()
        with dpath.open("w") as f:
            json.dump(info, f, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("/data3/rikhat.akizhanov/datasets/ClimbingVideos_v1"))
    ap.add_argument("--data-root", type=Path, default=scene_db.DEFAULT_DATA_ROOT)
    ap.add_argument("--splits", default="train,test", help="which splits to export")
    ap.add_argument("--scenes", default=None, help="comma-separated scene_ids (smoke test)")
    ap.add_argument("--limit", type=int, default=None, help="cap scenes per split")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--quality", type=int, default=95, help="frame JPEG quality")
    ap.add_argument("--skip-existing", action="store_true",
                    help="resume missing frames/masks and validate existing files")
    ap.add_argument("--fill-test", action="store_true",
                    help="only fill test contacts.npz from manual annotations (no frame export)")
    args = ap.parse_args()

    want_splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not want_splits or len(want_splits) != len(set(want_splits)):
        ap.error("--splits must contain one or both unique values: train,test")
    invalid_splits = sorted(set(want_splits) - {"train", "test"})
    if invalid_splits:
        ap.error(f"invalid --splits values: {invalid_splits}")
    if args.jobs < 1:
        ap.error("--jobs must be at least 1")
    if not 1 <= args.quality <= 100:
        ap.error("--quality must be within 1..100")
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be at least 1")
    if args.out.resolve() == args.data_root.resolve() or args.out.resolve().is_relative_to(
        args.data_root.resolve()
    ):
        ap.error("--out must be outside --data-root so export cannot overwrite source data")

    only = {s.strip() for s in args.scenes.split(",") if s.strip()} if args.scenes else None
    paths = scene_db.ScenePaths(args.data_root)
    feat = args.data_root / "features"

    # Corpus rows + their train/test split from the DB.
    with paths.connect() as conn:
        rows = scene_db.select_targets(
            conn, categories=scene_db.DEFAULT_CATEGORIES,
            rope_supported=scene_db.DEFAULT_ROPE_SUPPORTED,
        )
        split_of = {r["scene_id"]: r["dataset_split"] for r in
                    conn.execute("SELECT scene_id, dataset_split FROM scenes").fetchall()}

    target_ids = {r["scene_id"] for r in rows}
    unknown = sorted((only or set()) - target_ids)
    if unknown:
        ap.error(f"--scenes not in the curated target set: {unknown}")

    if args.fill_test:
        test_sids = sorted(r["scene_id"] for r in rows if split_of.get(r["scene_id"]) == "test"
                           and (not only or r["scene_id"] in only))
        print(f"filling test labels: {len(test_sids)} scenes")
        summ = fill_test_labels(args.out, feat, test_sids)
        _finalize_test_index(args.out, summ)
        ready = [s for s in summ if not s.get("missing")]
        print(f"\nDone. filled {len(ready)}/{len(test_sids)} test scenes"
              + (f"  MISSING={len(test_sids)-len(ready)}" if len(ready) < len(test_sids) else ""))
        return 0 if len(ready) == len(test_sids) else 1

    tasks: list[dict] = []
    for split in want_splits:
        sids = [r["scene_id"] for r in rows if split_of.get(r["scene_id"]) == split]
        if only:
            sids = [s for s in sids if s in only]
        if args.limit is not None:
            sids = sids[:args.limit]
        for r in rows:
            sid = r["scene_id"]
            if sid not in sids:
                continue
            aa, bb = sid[:2], sid[2:4]
            tasks.append(dict(
                sid=sid, split=split,
                feat=str(feat),
                clip=str(paths.clip_path(r)),
                out_scene=str(args.out / split / sid),
                quality=args.quality, skip_existing=args.skip_existing,
            ))
        print(f"{split}: {len(sids)} scenes")

    if not tasks:
        ap.error("selection produced no scenes for the requested split(s)")

    args.out.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, list[dict]] = {s: [] for s in want_splits}
    failed: list[tuple[str, str]] = []

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_safe_export, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                t = futs[fut]
                res = fut.result()
                if "error" in res:
                    failed.append((t["sid"], res["error"]))
                else:
                    summaries[t["split"]].append(res)
                print(f"[{i}/{len(tasks)}] {t['split']}/{t['sid']}"
                      + (" FAILED" if "error" in res else ""))
    else:
        for i, t in enumerate(tasks, 1):
            res = _safe_export(t)
            if "error" in res:
                failed.append((t["sid"], res["error"]))
            else:
                summaries[t["split"]].append(res)
            print(f"[{i}/{len(tasks)}] {t['split']}/{t['sid']}"
                  + (" FAILED" if "error" in res else ""))

    # Per-split index + a top-level info file.
    joint_names = None
    for split in want_splits:
        items = sorted(summaries[split], key=lambda d: d["scene_id"])
        if items and joint_names is None:
            first = args.out / split / items[0]["scene_id"]
            lbl = first / ("labels.npz" if split == "train" else "inputs.npz")
            joint_names = [str(x) for x in np.load(lbl, allow_pickle=True)["joint_names_22"]]
        (args.out / split).mkdir(parents=True, exist_ok=True)
        with open(args.out / split / "index.json", "w") as f:
            json.dump({"split": split, "n_scenes": len(items),
                       "n_frame_items": int(sum(sum(it["valid_frames"]) for it in items)),
                       "scenes": items}, f, indent=2)

    info = dict(
        schema_version=2, name=args.out.name,
        source_data_root=str(args.data_root),
        corpus_filter="human_selected=1 AND vlm_category IN (1,2) AND vlm_rope_supported=0",
        n_scenes={s: len(summaries[s]) for s in want_splits},
        joint_set="smplx_body_22", joint_names_22=joint_names,
        n_joints=N_JOINTS_22, confidence_schema=CONTACT_CONFIDENCE_SCHEMA,
        confidence_method=CONTACT_CONFIDENCE_METHOD,
        hand_merge="fingers OR-folded into the wrist; conf = max over touching parts when "
                   "in contact, else min over all 16 parts (free is an AND; matches the 3D viewer)",
        intrinsics="per-frame 3x3 pixels at original resolution (geometry/transform.npz "
                   "intrinsics_px_orig)",
        bbox="xyxy absolute pixels, original resolution (SAM-3 tracker bboxes_per_obj)",
        frames="full-frame RGB JPEG, one per video frame, cv2-sequential decode order",
        masks="per-person binary PNG (0/255), original resolution",
        labels={"train": "auto (contacts_1, 22-joint)", "test": "manual annotation — PENDING"},
        test_label_mapping=_test_label_mapping(),
        failed=[{"scene_id": s, "error": e} for s, e in failed],
    )
    with open(args.out / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nDone. " + " ".join(f"{s}={len(summaries[s])}" for s in want_splits)
          + (f"  FAILED={len(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
