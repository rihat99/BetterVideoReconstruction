"""Compare two human-chain output trees clip by clip (numpy only).

    python compare_runs.py <out_root_A> <out_root_B> --clips a,b,c [--json out.json]
                           [--dir-a human_optim] [--dir-b human_optim] [--verbose]

Reads ``<root>/<clip>/<dir>/<file>.npz`` on both sides for each of kinematic_1,
kinematic_2, kindyn_1, contacts_1, contacts_2 and reports, for every key, the max-abs and
mean-abs difference over the frames valid on both sides (``valid_mask (P,N)``).
Keys with a known meaning are compared in their own units:

- ``q (P,N,211)``: ``q[...,:3]`` pelvis position -> distance in mm; ``q[...,3:7]`` root
  quaternion and ``q[...,7:]`` the 51 parent-local joint quaternions (stride 4, joint
  ``k`` of ``q[...,7:]`` is ``joint_names[k+1]``) -> rotation angle in degrees,
  sign-invariant (|dot|), after normalising each quaternion.
- axis-angle keys (``global_orient``, ``body_pose``, ``*_hand_pose``, ``jaw_pose``,
  ``leye_pose``, ``reye_pose``) -> rotation angle per 3-vector, degrees.
- position keys (``transl``, ``joints_world``) -> per-point distance, mm.
- force keys (``frame_force``, ``frame_forces``, ``contact_forces_world``) -> per-vector
  distance in N and %BW over entries loaded on either side (|f| > 1e-6 N); BW is
  ``body_weight_n`` (contacts) or ``total_mass * 9.81`` (kindyn), per person, from A.
- bool keys -> agreement fraction and the F1 overlap of the two positive sets.
- other integer keys -> agreement fraction. Strings and 0-d values -> equal or not.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FILES = ("kinematic_1", "kinematic_2", "kindyn_1", "contacts_1", "contacts_2")
AXIS_ANGLE_KEYS = {"global_orient", "body_pose", "left_hand_pose", "right_hand_pose",
                   "jaw_pose", "leye_pose", "reye_pose"}
POSITION_KEYS = {"transl", "joints_world"}
FORCE_KEYS = {"frame_force", "frame_forces", "contact_forces_world"}
GRAVITY = 9.81  # tools/human_optim/kindyn.py _GRAVITY_MAG


def _quat_angle_deg(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Rotation angle between unit-normalised quaternions ``(..., 4)``, sign-invariant, degrees."""
    qa = qa / np.linalg.norm(qa, axis=-1, keepdims=True).clip(1e-12)
    qb = qb / np.linalg.norm(qb, axis=-1, keepdims=True).clip(1e-12)
    sign = np.where((qa * qb).sum(-1, keepdims=True) < 0, -1.0, 1.0)
    # 4·atan2(|a-b|,|a+b|) is the stable form of 2·acos(|a·b|) near zero.
    angle = 4.0 * np.arctan2(np.linalg.norm(qa - sign * qb, axis=-1),
                             np.linalg.norm(qa + sign * qb, axis=-1))
    return np.degrees(angle)


def _axis_angle_to_quat(v: np.ndarray) -> np.ndarray:
    """Axis-angle ``(..., 3)`` -> scalar-first quaternion ``(..., 4)``."""
    theta = np.linalg.norm(v, axis=-1, keepdims=True)
    half_sinc = 0.5 * np.sinc(theta / (2.0 * np.pi))       # sin(theta/2)/theta, safe at 0
    return np.concatenate([np.cos(theta / 2.0), v * half_sinc], axis=-1)


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, np.float64).ravel()
    finite = x[np.isfinite(x)]
    pct = np.percentile(finite, [50, 95, 99]) if finite.size else np.zeros(3)
    return {"n": int(x.size), "nonfinite": int(x.size - finite.size),
            "max": float(finite.max()) if finite.size else 0.0,
            "mean": float(finite.mean()) if finite.size else 0.0,
            "median": float(pct[0]), "p95": float(pct[1]), "p99": float(pct[2])}


def _f1(a: np.ndarray, b: np.ndarray) -> float | None:
    both = int((a & b).sum())
    total = int(a.sum() + b.sum())
    return None if total == 0 else 2.0 * both / total


def _frame_mask(da, db, shape) -> np.ndarray | None:
    """Frames valid on both sides, if the key leads with ``(P, N)``."""
    if "valid_mask" not in da or "valid_mask" not in db:
        return None
    vm = da["valid_mask"].astype(bool) & db["valid_mask"].astype(bool)
    return vm if tuple(shape[:2]) == vm.shape else None


def _body_weight(da) -> np.ndarray | None:
    if "body_weight_n" in da:
        return np.asarray(da["body_weight_n"], np.float64)
    if "total_mass" in da:
        return np.asarray(da["total_mass"], np.float64) * GRAVITY
    return None


def compare_key(key: str, a: np.ndarray, b: np.ndarray, da, db) -> dict:
    """One key's difference record."""
    if a.dtype.kind in "USO" or a.ndim == 0 or b.ndim == 0:
        same = a.shape == b.shape and bool(np.all(a == b))
        if not same and a.dtype.kind == "f" and a.ndim == 0:
            return {"kind": "scalar", "max": float(abs(float(a) - float(b))), "a": float(a), "b": float(b)}
        return {"kind": "meta", "equal": same}
    if a.shape != b.shape:
        return {"kind": "shape_mismatch", "a": list(a.shape), "b": list(b.shape)}
    mask = _frame_mask(da, db, a.shape)
    if mask is not None:
        person = np.broadcast_to(np.arange(a.shape[0])[:, None], mask.shape)[mask]  # (M,)
        a, b = a[mask], b[mask]
    else:
        person = None

    if a.dtype == bool:
        return {"kind": "bool", "agree": float((a == b).mean()) if a.size else 1.0,
                "n_diff": int((a != b).sum()), "pos_a": int(a.sum()), "pos_b": int(b.sum()),
                "f1": _f1(a, b)}
    if a.dtype.kind in "iu":
        return {"kind": "int", "agree": float((a == b).mean()) if a.size else 1.0,
                "n_diff": int((a != b).sum()), **_stats(np.abs(a.astype(np.int64) - b.astype(np.int64)))}

    a, b = a.astype(np.float64), b.astype(np.float64)
    if key == "q":
        names = [str(n) for n in da["joint_names"]][1:] if "joint_names" in da else None
        t = np.linalg.norm(a[..., :3] - b[..., :3], axis=-1) * 1e3
        root = _quat_angle_deg(a[..., 3:7], b[..., 3:7])
        joints = _quat_angle_deg(a[..., 7:].reshape(*a.shape[:-1], -1, 4),
                                 b[..., 7:].reshape(*b.shape[:-1], -1, 4))   # (..., 51)
        per_joint = joints.reshape(-1, joints.shape[-1]).max(0)
        worst = int(per_joint.argmax())
        return {"kind": "q", "pelvis_mm": _stats(t), "root_deg": _stats(root),
                "joint_deg": _stats(joints),
                "worst_joint": names[worst] if names else worst,
                "worst_joint_deg": float(per_joint[worst])}
    if key in AXIS_ANGLE_KEYS and a.shape[-1] % 3 == 0:
        aa = a.reshape(*a.shape[:-1], -1, 3)
        bb = b.reshape(*b.shape[:-1], -1, 3)
        return {"kind": "rot_deg", **_stats(_quat_angle_deg(_axis_angle_to_quat(aa),
                                                              _axis_angle_to_quat(bb)))}
    if key in POSITION_KEYS and a.shape[-1] == 3:
        return {"kind": "pos_mm", **_stats(np.linalg.norm(a - b, axis=-1) * 1e3)}
    if key in FORCE_KEYS and a.shape[-1] == 3:
        d = np.linalg.norm(a - b, axis=-1)                          # (..., F)
        loaded = (np.linalg.norm(a, axis=-1) > 1e-6) | (np.linalg.norm(b, axis=-1) > 1e-6)
        rec = {"kind": "force", "n_loaded": int(loaded.sum()), "N": _stats(d[loaded])}
        bw = _body_weight(da)
        if bw is not None and person is not None:
            bw_e = np.broadcast_to(bw[person].reshape(-1, *([1] * (d.ndim - 1))), d.shape)
            rec["pct_bw"] = _stats(100.0 * d[loaded] / bw_e[loaded])
        return rec
    if key == "base_wrench_diagnostics":            # (P,4) before/warm/after/coneviol
        after_a, after_b = a[:, 2], b[:, 2]
        return {"kind": "float", **_float_diff(a, b),
                "after_rel": float(np.max(np.abs(after_a - after_b) / np.abs(after_a).clip(1e-9)))}
    return {"kind": "float", **_float_diff(a, b)}


def _float_diff(a: np.ndarray, b: np.ndarray) -> dict:
    """|a-b| stats over entries finite on both sides; NaN in the same place counts as equal."""
    both_nan = np.isnan(a) & np.isnan(b)
    one_nan = np.isnan(a) ^ np.isnan(b)
    return {**_stats(np.abs(a - b)[~(both_nan | one_nan)]),
            "nan_both": int(both_nan.sum()), "nan_mismatch": int(one_nan.sum())}


def compare_file(path_a: Path, path_b: Path) -> dict:
    if not path_a.exists() or not path_b.exists():
        return {"missing": [str(p) for p in (path_a, path_b) if not p.exists()]}
    da, db = np.load(path_a, allow_pickle=True), np.load(path_b, allow_pickle=True)
    keys = {}
    for key in sorted(set(da.files) | set(db.files)):
        if key not in da.files or key not in db.files:
            keys[key] = {"kind": "only_in_" + ("a" if key in da.files else "b")}
            continue
        keys[key] = compare_key(key, da[key], db[key], da, db)
    return {"keys": keys}


def headline(name: str, rec: dict) -> str:
    """One compact line per file."""
    if "missing" in rec:
        return "MISSING " + ", ".join(rec["missing"])
    k = rec["keys"]
    parts = []
    if "q" in k and k["q"]["kind"] == "q":
        q = k["q"]
        parts.append(f"pelvis {q['pelvis_mm']['mean']:.2f}/{q['pelvis_mm']['max']:.1f} mm  "
                     f"root {q['root_deg']['mean']:.3f}/{q['root_deg']['max']:.2f} deg  "
                     f"joints {q['joint_deg']['mean']:.3f}/{q['joint_deg']['max']:.2f} deg "
                     f"(worst {q['worst_joint']})")
    if "joints_world" in k and k["joints_world"]["kind"] == "pos_mm":
        j = k["joints_world"]
        parts.append(f"jpos {j['mean']:.2f}/p99 {j['p99']:.1f}/{j['max']:.1f} mm")
    for fk in ("frame_force", "frame_forces"):
        if fk in k and k[fk]["kind"] == "force" and "pct_bw" in k[fk]:
            f = k[fk]["pct_bw"]
            parts.append(f"{fk} med {f['median']:.2f} mean {f['mean']:.2f} p95 {f['p95']:.1f} "
                         f"max {f['max']:.1f} %BW")
    if "base_wrench_diagnostics" in k and "after_rel" in k["base_wrench_diagnostics"]:
        parts.append(f"bw_after rel {100 * k['base_wrench_diagnostics']['after_rel']:.1f}%")
    if "frame_contact" in k and k["frame_contact"]["kind"] == "bool":
        c = k["frame_contact"]
        f1 = "-" if c["f1"] is None else f"{c['f1']:.4f}"
        parts.append(f"frame_contact agree {c['agree']:.4f} F1 {f1} (pos {c['pos_a']}/{c['pos_b']})")
    if "auto_labels" in k and k["auto_labels"]["kind"] == "int" and name.startswith("kinematic"):
        parts.append(f"auto_labels agree {k['auto_labels']['agree']:.4f}")
    n_same = sum(1 for r in k.values() if _identical(r))
    parts.append(f"[{n_same}/{len(k)} keys identical]")
    return "  ".join(parts)


def _identical(r: dict) -> bool:
    kind = r.get("kind")
    if kind == "meta":
        return r["equal"]
    if kind in ("bool", "int"):
        return r["n_diff"] == 0
    if kind == "q":
        return r["pelvis_mm"]["max"] == 0 and r["root_deg"]["max"] == 0 and r["joint_deg"]["max"] == 0
    if kind == "force":
        return r["N"]["max"] == 0
    if kind in ("float", "rot_deg", "pos_mm", "scalar"):
        return r["max"] == 0 and r.get("nonfinite", 0) == 0 and r.get("nan_mismatch", 0) == 0
    return False


def _verbose_row(key: str, r: dict) -> str:
    kind = r.get("kind")
    if kind in ("float", "rot_deg", "pos_mm"):
        return (f"{key:28s} {kind:8s} max {r['max']:.4g}  mean {r['mean']:.4g}  "
                f"nonfinite {r['nonfinite']}  nan_mismatch {r.get('nan_mismatch', 0)}")
    if kind in ("bool", "int"):
        f1 = r.get("f1")
        f1s = "" if f1 is None else f"  F1 {f1:.4f}"
        return f"{key:28s} {kind:8s} agree {r['agree']:.4f}  n_diff {r['n_diff']}{f1s}"
    if kind == "force":
        s = f"{key:28s} force    N med {r['N']['median']:.3g} mean {r['N']['mean']:.3g} max {r['N']['max']:.3g}"
        if "pct_bw" in r:
            s += f" | %BW med {r['pct_bw']['median']:.3g} max {r['pct_bw']['max']:.3g}"
        return s
    if kind == "q":
        return f"{key:28s} q        see headline"
    return f"{key:28s} {kind:8s} {json.dumps({k: v for k, v in r.items() if k != 'kind'})}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root_a", type=Path)
    parser.add_argument("root_b", type=Path)
    parser.add_argument("--clips", required=True, help="comma-separated clip names")
    parser.add_argument("--dir-a", default="human_optim", help="run folder under <root_a>/<clip>/")
    parser.add_argument("--dir-b", default="human_optim", help="run folder under <root_b>/<clip>/")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true", help="also print every differing key")
    args = parser.parse_args()

    report: dict = {"root_a": str(args.root_a), "root_b": str(args.root_b),
                    "dir_a": args.dir_a, "dir_b": args.dir_b, "clips": {}}
    for clip in args.clips.split(","):
        report["clips"][clip] = {}
        print(f"== {clip}  (A={args.dir_a}, B={args.dir_b}; mean/max)")
        for name in FILES:
            rec = compare_file(args.root_a / clip / args.dir_a / f"{name}.npz",
                               args.root_b / clip / args.dir_b / f"{name}.npz")
            report["clips"][clip][name] = rec
            print(f"  {name:12s} {headline(name, rec)}")
            if args.verbose and "keys" in rec:
                for key, r in rec["keys"].items():
                    if not _identical(r):
                        print(f"      {_verbose_row(key, r)}")
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
