"""Compare two BVR run roots clip by clip and pooled: pose, root, contacts, forces, confidence.

    uv run python scripts/compare_runs.py <root_a> <root_b> [--clips a,b] [--dir-a human_optim]
        [--dir-b human_optim] [--annotations <dir of <clip>.npz>] [--model <SMPLX_NEUTRAL.npz>] [--json out.json]

Reads ``<root>/<clip>/<dir>/{kinematic_1,contacts_1,kinematic_2,contacts_2,kindyn_1}.npz`` on both
sides (default: every clip present in both roots), so it compares a run of today's code with a run
of any earlier code. It reads the files with numpy, not through ``bvr.files``, and joint positions come
from a small SMPL-X forward kinematics read straight from the model file
(``$BETTERHUMAN_MODELS_DIR/smplx/SMPLX_NEUTRAL.npz``), checked against ``kindyn_1.npz``'s
``joints_world``. People are matched by ``object_ids``,
contact frames by name (``head`` read as ``head_top``), frames over ``valid_mask`` on both sides.

Per body file: body-joint (22) position error in mm, all 52 joints, pelvis mm, root orientation
degrees, and each side's mean root acceleration (m/s^2, over neighbouring frames at every frame rate: a
smoothness check).
Per contact labelling (``frame_contact`` of contacts_1, contacts_2, kindyn_1): cell agreement,
F1 between the two label sets, and each side's F1 against the manual annotations where they
exist (the fold of ``bvr_datasets/examples/score_frame_contacts.py``: hands, feet/ankles, knees; pooled
TP/FP/FN).
kindyn_1 forces in body weights (each side divided by its own ``total_mass * 9.81``): per-frame
force vector error on entries loaded on either side, total support |sum f| per frame, root
residual |base_wrench[:3]|, and force confidence. Support and residual are each side's own, over the frames where
its inverse dynamics count by today's rule (``bvr.terms.dynamics.counted_rows`` of its ``valid_mask`` and labels);
a person without contact labels has none. Pooled rows concatenate every clip's entries.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from bvr import paths
from bvr.terms import dynamics, motion

GRAVITY = 9.81                      # bvr kindyn's gravity magnitude
BODY_FILES = ("kinematic_1", "kinematic_2", "kindyn_1")
LABEL_FILES = ("contacts_1", "contacts_2", "kindyn_1")
SMPLX_52 = [*range(22), *range(25, 55)]      # SMPL-X joints without jaw and eyes = BVR's 52
FRAME_ALIASES = {"head": "head_top"}         # contact frame renamed in BetterHuman dev

# score_frame_contacts.py's fold: annotated coarse joint <- OR of these contact frames.
_HANDS = {"left_hand": ("palm_left", "finger_left", "palm_distal_left", "thumb_left"),
          "right_hand": ("palm_right", "finger_right", "palm_distal_right", "thumb_right")}
_KNEES = {"left_knee": ("knee_left",), "right_knee": ("knee_right",)}
_TOES_L = ("toe_under_left", "toe_above_left", "ball_inner_left", "ball_outer_left", "big_toe_left")
_TOES_R = ("toe_under_right", "toe_above_right", "ball_inner_right", "ball_outer_right", "big_toe_right")
FOLD_10 = {**_HANDS, "left_foot": (*_TOES_L, "heel_left"), "right_foot": (*_TOES_R, "heel_right"), **_KNEES}
FOLD_14 = {**_HANDS, "left_foot": _TOES_L, "right_foot": _TOES_R,
           "left_ankle": ("heel_left",), "right_ankle": ("heel_right",), **_KNEES}
SCHEMA_10 = {"left_hand", "right_hand", "left_foot", "right_foot", "left_knee", "right_knee",
             "left_hip", "right_hip", "left_elbow", "right_elbow"}
SCHEMA_14 = SCHEMA_10 | {"left_ankle", "right_ankle", "left_shoulder", "right_shoulder"}


class Skeleton:
    """SMPL-X rest joints from betas, and forward kinematics of BVR's ``q (..., 211)``."""

    def __init__(self, model_path: Path):
        with np.load(model_path) as model:
            regressor = model["J_regressor"][SMPLX_52].astype(np.float64)              # (52, V)
            self.rest = regressor @ model["v_template"].astype(np.float64)             # (52, 3)
            self.shape = np.einsum("jv,vck->jck", regressor, model["shapedirs"][:, :, :10])  # (52,3,10)

    def joints(self, q: np.ndarray, betas: np.ndarray, parents: np.ndarray) -> np.ndarray:
        """World joint positions ``(P, N, 52, 3)`` of ``q (P, N, 211)`` with ``betas (P, 10)``.

        ``q`` = pelvis position, root quaternion, then 51 parent-local quaternions, all
        quaternions scalar-last (BetterRobot's layout).
        """
        rest = self.rest[None] + np.einsum("jck,pk->pjc", self.shape, betas.astype(np.float64))  # (P,52,3)
        quats = np.concatenate([q[..., 3:7], q[..., 7:]], axis=-1).reshape(*q.shape[:2], 52, 4)
        rot = _quat_to_matrix(quats.astype(np.float64))                                # (P,N,52,3,3)
        world_r = np.empty_like(rot)
        world_p = np.empty((*q.shape[:2], 52, 3))
        world_r[:, :, 0], world_p[:, :, 0] = rot[:, :, 0], q[..., :3]
        for j in range(1, 52):
            parent = int(parents[j])
            offset = rest[:, j] - rest[:, parent]                                      # (P, 3)
            world_r[:, :, j] = world_r[:, :, parent] @ rot[:, :, j]
            world_p[:, :, j] = world_p[:, :, parent] + np.einsum("pnab,pb->pna", world_r[:, :, parent], offset)
        return world_p


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-12)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1
                    ).reshape(*q.shape[:-1], 3, 3)


def _quat_angle_deg(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Rotation angle between quaternions ``(..., 4)``, sign-invariant, degrees."""
    qa = qa / np.linalg.norm(qa, axis=-1, keepdims=True).clip(1e-12)
    qb = qb / np.linalg.norm(qb, axis=-1, keepdims=True).clip(1e-12)
    sign = np.where((qa * qb).sum(-1, keepdims=True) < 0, -1.0, 1.0)
    return np.degrees(4.0 * np.arctan2(np.linalg.norm(qa - sign * qb, axis=-1),
                                       np.linalg.norm(qa + sign * qb, axis=-1)))


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, np.float64).ravel()
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def _f1(tp: int, fp: int, fn: int) -> float | None:
    return None if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _people(a: dict, b: dict) -> tuple[list[int], list[int]]:
    ids_a, ids_b = [int(o) for o in a["object_ids"]], [int(o) for o in b["object_ids"]]
    common = [o for o in ids_a if o in ids_b]
    return [ids_a.index(o) for o in common], [ids_b.index(o) for o in common]


def _frame_names(d: dict, key: str = "contact_frame_names") -> list[str]:
    return [FRAME_ALIASES.get(str(n), str(n)) for n in d[key]]


def _root_acc(joints: np.ndarray, valid: np.ndarray, fps: float) -> np.ndarray:
    """|second difference| of the pelvis, m/s^2, over triples valid throughout. ``joints (P,N,52,3)``."""
    pelvis = joints[:, :, 0]
    acc = np.linalg.norm(pelvis[:, 2:] - 2 * pelvis[:, 1:-1] + pelvis[:, :-2], axis=-1) * fps * fps
    ok = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
    return acc[ok]


def compare_body(a: dict, b: dict, skeleton: Skeleton) -> tuple[dict, dict]:
    """Pose differences of one body file; returns (record, raw arrays for pooling)."""
    ia, ib = _people(a, b)
    if a["q"].shape[1] != b["q"].shape[1]:
        return {"error": f"frame count {a['q'].shape[1]} vs {b['q'].shape[1]}"}, {}
    names_a, names_b = [str(n) for n in a["joint_names"]], [str(n) for n in b["joint_names"]]
    if names_a != names_b:
        return {"error": "joint names differ"}, {}
    valid = a["valid_mask"][ia].astype(bool) & b["valid_mask"][ib].astype(bool)          # (P', N)
    ja = skeleton.joints(a["q"][ia], a["betas"][ia], a["parents"])
    jb = skeleton.joints(b["q"][ib], b["betas"][ib], b["parents"])
    err = np.linalg.norm(ja - jb, axis=-1)[valid] * 1e3                                    # (M, 52) mm
    root = _quat_angle_deg(a["q"][ia][..., 3:7].astype(np.float64), b["q"][ib][..., 3:7].astype(np.float64))[valid]
    raw = {"body_mm": err[:, :22], "all_mm": err, "pelvis_mm": err[:, 0], "root_deg": root,
           "acc_a": _root_acc(ja, valid, float(a["fps"])), "acc_b": _root_acc(jb, valid, float(b["fps"]))}
    record = {"people": len(ia), "frames": int(valid.sum()), "fps_a": float(a["fps"]), "fps_b": float(b["fps"]),
              **{k: _stats(v) for k, v in raw.items()}}
    if "joints_world" in a:        # the FK check: our joints against the file's own
        check = np.linalg.norm(ja - a["joints_world"][ia].astype(np.float64), axis=-1)[valid] * 1e3
        record["fk_check_a_mm"] = _stats(check)
    return record, raw


def _fold(annotation: dict) -> dict[str, tuple[str, ...]]:
    names = {str(n) for n in annotation["joint_names"]}
    if names == SCHEMA_10:
        return FOLD_10
    if names == SCHEMA_14:
        return FOLD_14
    raise ValueError(f"unrecognized annotation schema {sorted(names)}")


def score_against(d: dict, annotation: dict) -> dict[str, int]:
    """TP/FP/FN/TN of ``frame_contact`` folded to the annotated joints (score_frame_contacts.py rules)."""
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    contact = d["frame_contact"].astype(bool)                                      # (P, N, F)
    valid = d["valid_mask"].astype(bool)
    column = {n: i for i, n in enumerate(_frame_names(d))}
    ann_names = [str(n) for n in annotation["joint_names"]]
    labels_all = np.transpose(annotation["contacts"].astype(np.int8), (0, 2, 1))   # (P_ann, N, J)
    ann_ids = [int(o) for o in annotation["object_ids"]]
    if labels_all.shape[1] != contact.shape[1]:
        raise ValueError(f"frame count {contact.shape[1]} vs annotation {labels_all.shape[1]}")
    for person, object_id in enumerate(int(o) for o in d["object_ids"]):
        if object_id not in ann_ids or bool(annotation["ignored"][ann_ids.index(object_id)]):
            continue
        labels = labels_all[ann_ids.index(object_id)]
        for joint, frames in _fold(annotation).items():
            cols = [column[f] for f in frames if f in column]
            predicted = contact[person][:, cols].any(axis=1)
            truth = labels[:, ann_names.index(joint)]
            keep = (truth >= 0) & valid[person]
            p, t = predicted[keep], truth[keep] == 1
            counts["tp"] += int((p & t).sum())
            counts["fp"] += int((p & ~t).sum())
            counts["fn"] += int((~p & t).sum())
            counts["tn"] += int((~p & ~t).sum())
    return counts


def compare_labels(a: dict, b: dict, annotation: dict | None) -> tuple[dict, dict]:
    ia, ib = _people(a, b)
    if a["frame_contact"].shape[1] != b["frame_contact"].shape[1]:
        return {"error": "frame count differs"}, {}
    names_a, names_b = _frame_names(a), _frame_names(b)
    common = [n for n in names_a if n in names_b]
    ca, cb = [names_a.index(n) for n in common], [names_b.index(n) for n in common]
    valid = a["valid_mask"][ia].astype(bool) & b["valid_mask"][ib].astype(bool)
    la = a["frame_contact"][ia][:, :, ca].astype(bool)[valid]                         # (M, F)
    lb = b["frame_contact"][ib][:, :, cb].astype(bool)[valid]
    tp, fp, fn = int((la & lb).sum()), int((la & ~lb).sum()), int((~la & lb).sum())
    raw = {"cells": la.size, "same": int((la == lb).sum()), "ab": (tp, fp, fn)}
    record = {"frames": int(valid.sum()), "agree": float((la == lb).mean()) if la.size else None,
              "f1_ab": _f1(tp, fp, fn), "positives_a": int(la.sum()), "positives_b": int(lb.sum()),
              "frames_only_in_a": [n for n in names_a if n not in names_b],
              "frames_only_in_b": [n for n in names_b if n not in names_a]}
    if "frame_label_confidence" in a and "frame_label_confidence" in b:
        conf = np.abs(a["frame_label_confidence"][ia][:, :, ca] - b["frame_label_confidence"][ib][:, :, cb])[valid]
        raw["conf_diff"] = conf
        record["label_confidence_diff"] = _stats(conf)
    if annotation is not None:
        for side, d in (("a", a), ("b", b)):
            counts = score_against(d, annotation)
            raw[f"ann_{side}"] = counts
            record[f"f1_ann_{side}"] = _f1(counts["tp"], counts["fp"], counts["fn"])
    return record, raw


def _counted(d: dict) -> np.ndarray:
    """The frames where the inverse dynamics of a kindyn_1 file count, ``(P, T)`` bool (kindyn_1's own rule)."""
    return dynamics.counted_rows(motion.Time(float(d["fps"])), torch.as_tensor(d["valid_mask"].astype(bool)),
                                 torch.as_tensor(d["frame_contact"].astype(bool))).numpy()


def compare_forces(a: dict, b: dict) -> tuple[dict, dict]:
    ia, ib = _people(a, b)
    names_a, names_b = _frame_names(a), _frame_names(b)
    common = [n for n in names_a if n in names_b]
    ca, cb = [names_a.index(n) for n in common], [names_b.index(n) for n in common]
    valid = a["valid_mask"][ia].astype(bool) & b["valid_mask"][ib].astype(bool)
    counted_a, counted_b = _counted(a)[ia][valid], _counted(b)[ib][valid]               # (M,)
    bw_a = (a["total_mass"][ia].astype(np.float64) * GRAVITY)[:, None, None, None]
    bw_b = (b["total_mass"][ib].astype(np.float64) * GRAVITY)[:, None, None, None]
    fa = (a["frame_forces"][ia][:, :, ca] / bw_a)[valid]                               # (M, F, 3) in BW
    fb = (b["frame_forces"][ib][:, :, cb] / bw_b)[valid]
    loaded = (np.linalg.norm(fa, axis=-1) > 1e-6) | (np.linalg.norm(fb, axis=-1) > 1e-6)
    support_a, support_b = np.linalg.norm(fa.sum(1), axis=-1), np.linalg.norm(fb.sum(1), axis=-1)
    resid_a = (np.linalg.norm(a["base_wrench"][ia][..., :3], axis=-1) / bw_a[..., 0, 0])[valid]
    resid_b = (np.linalg.norm(b["base_wrench"][ib][..., :3], axis=-1) / bw_b[..., 0, 0])[valid]
    raw = {"force_err_bw": np.linalg.norm(fa - fb, axis=-1)[loaded],
           "support_diff_bw": np.abs(support_a - support_b),
           "support_a": support_a[counted_a], "support_b": support_b[counted_b],
           "residual_a": resid_a[counted_a], "residual_b": resid_b[counted_b],
           "fconf_diff": np.abs(a["force_confidence"][ia] - b["force_confidence"][ib])[valid],
           "fconf_clip_a": a["force_confidence_clip"][ia], "fconf_clip_b": b["force_confidence_clip"][ib]}
    record = {"mass_a_kg": [float(m) for m in a["total_mass"][ia]],
              "mass_b_kg": [float(m) for m in b["total_mass"][ib]],
              **{k: _stats(v) for k, v in raw.items() if k not in ("fconf_clip_a", "fconf_clip_b")},
              "fconf_clip_a": [float(c) for c in raw["fconf_clip_a"]],
              "fconf_clip_b": [float(c) for c in raw["fconf_clip_b"]]}
    return record, raw


def _pool_stats(raws: list[dict], key: str) -> dict:
    parts = [r[key] for r in raws if key in r]
    return _stats(np.concatenate([np.ravel(p) for p in parts])) if parts else _stats(np.zeros(0))


def _pool_counts(raws: list[dict], key: str) -> dict | None:
    parts = [r[key] for r in raws if key in r]
    if not parts:
        return None
    total = {k: sum(p[k] for p in parts) for k in ("tp", "fp", "fn", "tn")}
    return {**total, "f1": _f1(total["tp"], total["fp"], total["fn"])}


def _fmt(value, spec: str = ".2f") -> str:
    return "-" if value is None else format(value, spec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root_a", type=Path)
    parser.add_argument("root_b", type=Path)
    parser.add_argument("--clips", default=None, help="comma-separated; default: every clip in both roots")
    parser.add_argument("--dir-a", default="human_optim")
    parser.add_argument("--dir-b", default="human_optim")
    parser.add_argument("--annotations", type=Path, help="default: <data root>/examples/data")
    parser.add_argument("--model", type=Path, help="default: $BETTERHUMAN_MODELS_DIR/smplx/SMPLX_NEUTRAL.npz")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if args.clips:
        clips = args.clips.split(",")
    else:
        clips = sorted(p.name for p in args.root_a.iterdir()
                       if (p / args.dir_a).is_dir() and (args.root_b / p.name / args.dir_b).is_dir())
    skeleton = Skeleton(args.model or Path(os.environ["BETTERHUMAN_MODELS_DIR"]) / "smplx" / "SMPLX_NEUTRAL.npz")
    report: dict = {"root_a": str(args.root_a), "root_b": str(args.root_b), "dir_a": args.dir_a,
                    "dir_b": args.dir_b, "clips": {}, "pooled": {}}
    raws: dict[str, list[dict]] = {}
    annotations = args.annotations or paths.data_root() / "examples" / "data"
    for clip in clips:
        ann_path = annotations / f"{clip}.npz"
        annotation = _load(ann_path)
        files = {name: (_load(args.root_a / clip / args.dir_a / f"{name}.npz"),
                        _load(args.root_b / clip / args.dir_b / f"{name}.npz"))
                 for name in ("kinematic_1", "contacts_1", "kinematic_2", "contacts_2", "kindyn_1")}
        out: dict = {}
        for name, (a, b) in files.items():
            if a is None or b is None:
                out[name] = {"missing": [s for s, d in (("a", a), ("b", b)) if d is None]}
                continue
            out[name] = {}
            if name in BODY_FILES:
                out[name]["pose"], raw = compare_body(a, b, skeleton)
                raws.setdefault(f"{name}.pose", []).append(raw)
            if name in LABEL_FILES:
                out[name]["labels"], raw = compare_labels(a, b, annotation)
                raws.setdefault(f"{name}.labels", []).append(raw)
            if name == "kindyn_1":
                out[name]["forces"], raw = compare_forces(a, b)
                raws.setdefault("kindyn_1.forces", []).append(raw)
        report["clips"][clip] = out

    for key, items in raws.items():
        items = [r for r in items if r]
        if key.endswith(".pose"):
            report["pooled"][key] = {k: _pool_stats(items, k) for k in
                                     ("body_mm", "all_mm", "pelvis_mm", "root_deg", "acc_a", "acc_b")}
        elif key.endswith(".labels"):
            cells, same = sum(r["cells"] for r in items), sum(r["same"] for r in items)
            tp, fp, fn = (sum(r["ab"][i] for r in items) for i in range(3))
            report["pooled"][key] = {"agree": same / cells if cells else None, "f1_ab": _f1(tp, fp, fn),
                                     "ann_a": _pool_counts(items, "ann_a"), "ann_b": _pool_counts(items, "ann_b"),
                                     "label_confidence_diff": _pool_stats(items, "conf_diff")}
        else:
            report["pooled"][key] = {k: _pool_stats(items, k) for k in
                                     ("force_err_bw", "support_diff_bw", "support_a", "support_b",
                                      "residual_a", "residual_b", "fconf_diff", "fconf_clip_a", "fconf_clip_b")}

    print(f"A = {args.root_a}/<clip>/{args.dir_a}   B = {args.root_b}/<clip>/{args.dir_b}")
    print(f"{'clip':20s} {'file':12s} pose: body mm mean/p95/max | pelvis mm | root deg | acc A/B m/s2 ;"
          " labels: agree F1ab F1ann A/B ; forces: err %BW med/p95 | support A/B BW | conf A/B")
    for clip, out in [*report["clips"].items(), ("POOLED", None)]:
        for name in ("kinematic_1", "contacts_1", "kinematic_2", "contacts_2", "kindyn_1"):
            rec = out[name] if out else {
                "pose": report["pooled"].get(f"{name}.pose"), "labels": report["pooled"].get(f"{name}.labels"),
                "forces": report["pooled"].get(f"{name}.forces")}
            if "missing" in rec:
                print(f"{clip:20s} {name:12s} missing on {rec['missing']}")
                continue
            parts = []
            pose = rec.get("pose")
            if pose and "error" not in pose:
                parts.append(f"body {_fmt(pose['body_mm']['mean'])}/{_fmt(pose['body_mm']['p95'], '.1f')}/"
                             f"{_fmt(pose['body_mm']['max'], '.0f')} | pelvis {_fmt(pose['pelvis_mm']['mean'])} | "
                             f"root {_fmt(pose['root_deg']['mean'], '.3f')} | acc {_fmt(pose['acc_a']['mean'], '.2f')}/"
                             f"{_fmt(pose['acc_b']['mean'], '.2f')}")
            elif pose:
                parts.append(f"pose ERROR {pose['error']}")
            labels = rec.get("labels")
            if labels and "error" not in labels:
                if out:
                    ann = f"{_fmt(labels.get('f1_ann_a'), '.3f')}/{_fmt(labels.get('f1_ann_b'), '.3f')}"
                else:
                    ann = (f"{_fmt((labels['ann_a'] or {}).get('f1'), '.3f')}/"
                           f"{_fmt((labels['ann_b'] or {}).get('f1'), '.3f')}")
                parts.append(f"agree {_fmt(labels['agree'], '.4f')} F1ab {_fmt(labels['f1_ab'], '.3f')} F1ann {ann}")
            elif labels:
                parts.append(f"labels ERROR {labels['error']}")
            forces = rec.get("forces")
            if forces:
                error = forces["force_err_bw"] if forces["force_err_bw"]["n"] else {"median": None, "p95": None}
                percent = [None if value is None else 100 * value for value in (error["median"], error["p95"])]
                trust = [forces[side]["mean"] if out is None else np.mean(forces[side])
                         for side in ("fconf_clip_a", "fconf_clip_b")]
                parts.append(f"err {_fmt(percent[0], '.2f')}/{_fmt(percent[1], '.1f')} %BW | "
                             f"support {_fmt(forces['support_a']['mean'], '.3f')}/"
                             f"{_fmt(forces['support_b']['mean'], '.3f')} | "
                             f"conf {_fmt(trust[0], '.3f')}/{_fmt(trust[1], '.3f')}")
            print(f"{clip:20s} {name:12s} " + " ; ".join(parts))
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
