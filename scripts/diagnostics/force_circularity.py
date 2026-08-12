"""Does the force witness survive on a body no contact term ever touched?

The bodies the main evaluation uses were refined by ``kinematics_1`` with contact
attraction and contact velocity damping switched on. Those terms are driven by the
geometric labels, so the body has already been nudged toward the very contacts the
readout is then asked to find. That is a circularity worth measuring, not assuming
away.

This control removes it. It reads a **pure-visual** body — ``kinematics_1`` re-run
with the kin1 ``ann_attraction`` weight at zero, which also disables label detection
and de-registers the velocity damper, so no contact term acted at any point — and
runs both readouts on it: the geometric detector, and the force witness at the frozen
recipe. If force still finds contacts there, its signal is not an echo of the labels
that shaped the body.

**Stillness on this body.** The usual rule is that stillness must come from the
persisted ``drift_pre``, never re-measured on an enforced body. Here the pure-visual
run never populated ``drift_pre`` (no detection ran) — and it does not need to: on a
body nothing was ever enforced on, drift measured from the final pose *is*
pre-enforcement drift. That is why this script measures it directly and the shipped
stage still refuses to.

Usage::

    uv run --no-sync python scripts/diagnostics/force_circularity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from score_frame_contacts import _GROUPS, _pool, metrics  # noqa: E402

from tools.body import frame_normals, load_body  # noqa: E402
from tools.config import load_toml  # noqa: E402
from tools.geometry.normals import load_scene_cloud  # noqa: E402
from tools.human_optim.camera import world_from_cam  # noqa: E402
from tools.human_optim.contacts import (  # noqa: E402
    detect_frame_contacts, joint_drift, merge_contact_spans,
)
from tools.human_optim.forces import ForceKnobs, force_labels, solve_person_forces  # noqa: E402

sys.path.insert(0, str(_REPO_ROOT / "scripts" / "diagnostics"))
from force_recovery import _FORCE_PRUNE_MARGIN_M, score_labels  # noqa: E402,F401

_FARM = _REPO_ROOT / "out" / "_expK_purevis"
_CONFIG = _REPO_ROOT / "configs" / "human_optimization.toml"

#: The frozen shipped recipe, mirroring estimate_contacts._FORCE_INTERNALS.
_KNOBS = ForceKnobs(cone_axis="frame", use_motion=True, relative_gap=True,
                    candidate_max_dist_m=float("inf"), torque_length_m=0.2,
                    lambda_tau=0.0, gap_dstar_m=0.05, stillness_vstar_mps=0.2)
_THETA_BW = 0.05


def readouts(stem: str, device: str = "cuda") -> tuple[dict, dict, dict]:
    """Score the gates-only, force-only and union readouts on one pure-visual body."""
    pred = _FARM / stem
    kin = dict(np.load(pred / "human_optim" / "kinematic_1.npz", allow_pickle=True))
    transform = dict(np.load(pred / "geometry" / "transform.npz", allow_pickle=True))
    scene_npz = dict(np.load(pred / "geometry" / "scene.npz", allow_pickle=True))
    config = load_toml(_CONFIG)
    cc, cg = config["contacts"], config["contact_gate"]

    gravity_path = pred / "geocalib" / "gravity.npz"
    gravity = (np.asarray(np.load(gravity_path)["gravity_world"], np.float32)
               if gravity_path.exists() else np.array([0.0, 1.0, 0.0], np.float32))
    extrinsics = torch.as_tensor(transform["extrinsics"], dtype=torch.float32, device=device)
    scene_pts, scene_normals, scene_conf = load_scene_cloud(
        scene_npz, world_from_cam(extrinsics)[:, :3, 3],
        conf_drop_q=float(cc["conf_drop_q"]), k=int(cc["normal_k"]), device=device)

    body = load_body(device=device)
    frame_names = list(body.contact_frame_names)
    knee_columns = [i for i, n in enumerate(frame_names) if n.startswith("knee")]
    fps = float(transform.get("fps", kin.get("fps", 30.0)))
    valid_mask = np.asarray(kin["valid_mask"], bool)

    gates_all, force_all, union_all = [], [], []
    for person in range(int(kin["num_people"])):
        betas = torch.as_tensor(kin["betas"][person], dtype=torch.float32, device=device)
        q = torch.as_tensor(kin["q"][person], dtype=torch.float32, device=device)
        out = body.forward(betas[None].expand(q.shape[0], -1),
                           torch.as_tensor(kin["q"][person], dtype=torch.float32, device=device))
        posed = out.frames_world
        points, normals = posed[..., :3], frame_normals(posed)
        valid = valid_mask[person]

        # Legitimately pre-enforcement: no contact term ever acted on this body.
        drift = joint_drift(points, fps=fps, window_s=float(cg["window_s"]),
                            valid_mask=torch.as_tensor(valid, device=device))

        detected = detect_frame_contacts(
            points, normals, drift, scene_pts, scene_normals, scene_conf,
            frame_names=frame_names,
            close_thresh_m=float(cc["close_thresh_m"]), exit_thresh_m=float(cc["exit_thresh_m"]),
            min_frames=int(cc["min_frames"]), still_m=float(cg["still_m"]),
            max_dist_m=float(cc["max_dist_m"]), conf_thresh=float(cc["conf_thresh"]),
            prune_margin_m=float(cc["prune_margin_m"]))
        gates = (merge_contact_spans(
            detected.contact, points, detected.guarded_sdf,
            merge_dist_m=float(cc["merge_dist_m"]), max_gap_s=float(cc["merge_max_gap_s"]),
            gap_sdf_m=float(cc["merge_sdf_m"]), fps=fps) & ~detected.toe_suppressed
        ).cpu().numpy()
        gates[~valid] = False

        solution = solve_person_forces(
            body, betas, q, valid, drift_pre=drift, drift_window_s=float(cg["window_s"]),
            gravity_world=gravity, fps=fps, scene_pts=scene_pts,
            scene_normals=scene_normals, scene_conf=scene_conf, knobs=_KNOBS)
        force, _ = force_labels(solution.magnitude, frame_names=frame_names, valid=valid,
                                threshold_n=_THETA_BW * solution.body_weight_n,
                                min_frames=int(cc["min_frames"]))
        force[:, knee_columns] = False
        gates_all.append(gates)
        force_all.append(force)
        union_all.append(gates | force)

    meta = dict(frame_names=frame_names, valid_mask=valid_mask,
                object_ids=[int(o) for o in kin["object_ids"]])
    return (score_labels(stem, gates_all, meta), score_labels(stem, force_all, meta),
            score_labels(stem, union_all, meta))


def main() -> int:
    stems = sorted(p.name for p in _FARM.iterdir()
                   if (p / "human_optim" / "kinematic_1.npz").exists())
    print("CIRCULARITY CONTROL — readouts on PURE-VISUAL bodies (no contact term ever ran)")
    print("=" * 84)
    header = f"{'scene':22s} {'readout':12s} {'TP':>6s} {'FP':>6s} {'FN':>6s} " \
             f"{'prec':>7s} {'rec':>7s} {'F1':>7s}"
    print(header)
    pooled: dict[str, list] = {}
    for stem in stems:
        for name, joints in zip(("gates", "force", "force|gates"), readouts(stem)):
            pooled.setdefault(name, []).append(joints)
            m = metrics(_pool(list(joints.values())))
            show = lambda v: "      -" if v is None else f"{v:7.3f}"  # noqa: E731
            print(f"{stem:22s} {name:12s} {m['tp']:6d} {m['fp']:6d} {m['fn']:6d} "
                  f"{show(m['precision'])}{show(m['recall'])}{show(m['f1'])}")
    print("-" * len(header))
    knees = set(_GROUPS["knees"])
    for name, rows in pooled.items():
        m = metrics(_pool([c for j in rows for c in j.values()]))
        nk = metrics(_pool([c for j in rows for n, c in j.items() if n not in knees]))
        print(f"{'POOLED':22s} {name:12s} {m['tp']:6d} {m['fp']:6d} {m['fn']:6d} "
              f"{m['precision']:7.3f}{m['recall']:7.3f}{m['f1']:7.3f}   "
              f"(no-knee F1 {nk['f1']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
