"""Scene-contact estimation for the optimized SMPL-X bodies.

Reads the refined world-frame SMPL-X trajectory (``human_optim/kinematic_1.npz``)
and the static scene cloud (``geometry/scene.npz``) and marks, for every tracked
person, which **joints** are in contact with the scene:

  1. pose each person's :class:`SMPLXMid` mesh per frame (re-applying the stored
     global scale about the pelvis, exactly as the kinematic stage / viewers do),
  2. signed-distance each body vertex to the scene surface (the same point-cloud
     SDF the kinematic stage's penetration loss uses) → per-vertex contact
     (inside the scene, or just outside it within ``close_thresh_m``),
  3. lift to **per-joint** contact by nearest-joint assignment — a joint is in
     contact if any vertex closest to it is (vertex→joint by rest-pose 3D distance),
  4. keep a joint-frame only if the joint is also **still** — its net displacement
     over a ~1/3 s window is under ``2·still_m`` (a joint sliding along the wall is a
     pass, not a grip) — then temporal hysteresis (enter/exit) + a min-duration.

The stillness gate reads the body's persisted pre-enforcement drift (``drift_pre``)
when present, and a **span-merge** bridges short flicker gaps behind a
re-grip guard (the bridged frames flagged in ``merged``). Each joint-frame then gets
a **label confidence** in [0,1] (:mod:`tools.human_optim.confidence`) — for BOTH the
contact and no-contact classes. One class-independent visibility factor multiplies
continuous surface or motion evidence. Fully occluded or
out-of-frame joints are zero-confidence; approaching no-contact joints retain strong
motion evidence even near the wall, while stationary no-contact joints are untrusted
regardless of their reconstructed wall distance.

Renders ``contacts_1_overlay.mp4``: the projected SMPL-X skeleton, joints **red**
in contact (dot size ∝ confidence) / **green** otherwise.

Run from CLI::

    .venv/bin/python scripts/stages/estimate_contacts.py \\
        --source video.mp4 --predictions out/video/ --config configs/human_optimization.toml

Reads:  human_optim/kinematic_1.npz, geometry/scene.npz, geometry/transform.npz
Writes: human_optim/contacts_1.npz          (joint_contact (P,N,J) bool + signed/Euclidean
                                             scene-distance diagnostics + merged + auditable
                                             visibility/evidence terms + label_confidence)
        human_optim/contacts_1_overlay.mp4  ([contacts].overlay; default on)
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BETTER_HUMAN_SRC = _REPO_ROOT.parent / "better_human" / "src"
for _p in (str(_BETTER_HUMAN_SRC), str(_REPO_ROOT)):
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.config import load_toml  # noqa: E402
from tools.geometry.normals import load_scene_cloud  # noqa: E402
from tools.human_optim import (  # noqa: E402
    contact_from_sdf, drift_gated_contact, joint_drift, joint_guarded_min_sdf,
    joint_occlusion_counts, lift_joint_contact, lift_joint_min_distance,
    merge_contact_spans, project, scene_sdf,
    transform_points, vertex_to_joint, world_from_cam,
)
from tools.human_optim import paths  # noqa: E402
from tools.human_optim.confidence import (  # noqa: E402
    CONTACT_CONFIDENCE_DEFAULTS, confidence_config, joint_label_confidence,
    occlusion_group_ids)
from tools.human_optim.qmath import normalize_q  # noqa: E402
from tools.smplx_robot import load_smplx_mid  # noqa: E402

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "human_optimization.toml"
_DEFAULT_SMPLX_MODEL = (
    _REPO_ROOT.parent / "better_human" / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
)


def _load_config(path: Path) -> dict:
    """Operational knobs for the contact stage (no loss weights here)."""
    cfg = load_toml(path) if path.exists() else {}
    c = cfg.setdefault("contacts", {})
    c.setdefault("close_thresh_m", 0.02)   # enter: "just outside" distance counted as contact
    c.setdefault("exit_thresh_m", 0.035)   # stay: looser hysteresis exit (holds a span, kills boundary flicker)
    c.setdefault("min_frames", 3)          # drop contact spans shorter than this many frames
    c.setdefault("max_dist_m", 0.10)       # deep-occluder guard (reject spurious deep penetration)
    c.setdefault("conf_thresh", 0.1)       # residual absolute floor (VGGT conf ≥~1 ⇒ non-binding)
    c.setdefault("conf_drop_q", 0.1)       # quantile confidence gate: drop the lowest fraction of scene voxels
    c.setdefault("normal_k", 16)           # kNN for PCA scene-normal estimation
    c.setdefault("prune_margin_m", 0.10)   # per-frame scene-AABB margin for the SDF nearest-neighbour
    c.setdefault("merge_dist_m", 0.03)     # span-merge: bridge a flicker gap the joint barely moved across
    c.setdefault("merge_max_gap_s", 2.0)   # …only gaps shorter than this
    c.setdefault("merge_sdf_m", 0.08)      # …re-grip guard: it stayed within this of the wall the whole gap
    c.setdefault("overlay", True)           # render the contact skeleton overlay
    g = cfg.setdefault("contact_gate", {})
    g.setdefault("window_s", 0.35)         # drift window (s) for the stillness gate
    g.setdefault("still_m", 0.025)         # net drift (m) at which stillness = 0.5 (gate kills frames >= 2·still_m)
    # Per-joint label confidence (readout-only; NOT registered in config.py). Defaults +
    # semantics live in tools/human_optim/confidence.py, shared with the scorer's --reliability.
    k = cfg.setdefault("contact_confidence", {})
    for key, val in CONTACT_CONFIDENCE_DEFAULTS.items():
        k.setdefault(key, val)
    cfg["contact_confidence"] = confidence_config(k)
    cfg.setdefault("runtime", {}).setdefault("device", "cuda")
    return cfg


def _sha256(path: Path) -> str:
    with path.open("rb") as source_file:
        return hashlib.file_digest(source_file, "sha256").hexdigest()


@torch.no_grad()
def _pose(mid, betas: torch.Tensor, q: torch.Tensor, scale: float, *, chunk=64):
    """Pose all frames → ``(joints (N,J,3), vertices (N,V,3))`` in the world frame.

    Re-applies the per-person global ``scale`` about the pelvis (joint 0), the
    same operation the kinematic optimizer and viewers use, so the contact SDF
    sees the body at its stored metric size.
    """
    N = q.shape[0]
    jws, vws = [], []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        out = mid.forward(betas[None].expand(e - s, -1), normalize_q(q[s:e], mid.num_joints))
        jw = out.joints_world.tensor()[..., :3]
        vw = out.vertices
        root = jw[:, :1, :]
        jws.append(root + scale * (jw - root))
        vws.append(root + scale * (vw - root))
    return torch.cat(jws, 0), torch.cat(vws, 0)


def run(source: str | Path, predictions_dir: str | Path, config_path: str | Path, *,
        overlay: bool = True, bodies: str = "kinematic_1") -> Path:
    cfg = _load_config(Path(config_path))
    device = cfg["runtime"]["device"]
    cc = cfg["contacts"]
    cg = cfg["contact_gate"]
    pred = Path(predictions_dir)

    # Which optimized body to read contact from: kinematics_1 (default) or the
    # annotation-guided kinematics_2. Output tag mirrors it: contacts_1 / contacts_2.
    kin_path = paths.body_path(pred, bodies)
    tag = bodies.removeprefix("kinematic_")
    scene_path = pred / "geometry" / "scene.npz"
    if not kin_path.exists():
        raise SystemExit(f"missing {kin_path} — run human_optimization {bodies} first")
    if not scene_path.exists():
        raise SystemExit(f"missing {scene_path} — run fuse_geometry.py first (needs vggt --save-depth)")

    transform_path = pred / "geometry" / "transform.npz"
    kin = dict(np.load(kin_path, allow_pickle=True))
    source_body_sha256 = _sha256(kin_path)
    source_scene_sha256 = _sha256(scene_path)
    source_transform_sha256 = _sha256(transform_path)
    tr = dict(np.load(transform_path, allow_pickle=True))
    if not bool(tr.get("metric", True)):  # absent flag = old already-metric artifact
        raise SystemExit(f"geometry under {pred} is up-to-scale — run estimate_scale after fuse")
    sc = dict(np.load(scene_path, allow_pickle=True))
    N, P = int(kin["num_frames"]), int(kin["num_people"])
    H, W = int(kin["video_height"]), int(kin["video_width"])
    object_ids = np.asarray(kin["object_ids"], dtype=np.int32)
    parents = np.asarray(kin["parents"], dtype=np.int32)
    valid_mask = np.asarray(kin["valid_mask"], dtype=bool)
    fps = float(tr.get("fps", kin.get("fps", 30.0)))   # drift window needs the true clip fps

    extr = torch.as_tensor(tr["extrinsics"], dtype=torch.float32, device=device)       # (N,4,4)
    K = torch.as_tensor(tr["intrinsics_px_orig"], dtype=torch.float32, device=device)  # (N,3,3)
    cam_centers = world_from_cam(extr)[:, :3, 3]

    scene_pts, scene_normals, scene_conf = load_scene_cloud(
        sc, cam_centers, conf_drop_q=float(cc.get("conf_drop_q", 0.0)),
        k=int(cc["normal_k"]), device=device)
    print(f"[estimate_contacts] scene cloud {scene_pts.shape[0]} pts, normals: "
          f"{'fuse' if 'normals' in sc else 'PCA fallback'}; "
          f"close<{float(cc['close_thresh_m']) * 100:.0f}cm")

    mid = load_smplx_mid(_DEFAULT_SMPLX_MODEL, device)
    J = mid.num_joints
    mid_names = list(mid.joint_names)
    occ_group_t = torch.as_tensor(occlusion_group_ids(mid_names), device=device)

    joint_contact = np.zeros((P, N, J), dtype=bool)
    joint_sdf_cm = np.full((P, N, J), np.nan, dtype=np.float32)
    joint_distance_cm = np.full((P, N, J), np.nan, dtype=np.float32)
    joints_2d = np.full((P, N, J, 2), np.nan, dtype=np.float32)
    merged_all = np.zeros((P, N, J), dtype=bool)            #frames the span-merge bridged
    confidence_all = {name: np.zeros((P, N, J), np.float32) for name in
                      ("image_visibility", "occlusion_visibility", "occluding_vertices",
                       "visibility", "surface", "motion", "label_confidence")}

    # Pose everyone once. Besides avoiding repeated model inference, retaining the
    # posed meshes lets one tracked person occlude another person's joints.
    posed_people = []
    for p in range(P):
        betas = torch.as_tensor(kin["betas"][p], dtype=torch.float32, device=device)
        q = torch.as_tensor(kin["q"][p], dtype=torch.float32, device=device)
        scale = float(kin["scale"][p])
        # Nearest-joint map from THIS person's shaped rest mesh: body shape moves
        # boundary vertices between joints, so a zero-beta map would mislabel them.
        # Scale is uniform about the pelvis ⇒ nearest-joint is scale-invariant, so the
        # unscaled shaped rest mesh is exact here.
        with torch.no_grad():
            v_rest, j_rest = mid.forward_shape(betas[None])
        v2j = vertex_to_joint(v_rest[0], j_rest[0])                  # (V,) per person
        jw, vw = _pose(mid, betas, q, scale)                         # (N,J,3), (N,V,3) world
        posed_people.append((v2j, jw, vw))

    for p, (v2j, jw, vw) in enumerate(posed_people):
        sdf, dmin, conf = scene_sdf(vw, scene_pts, scene_normals, scene_conf,
                                    prune_margin_m=float(cc["prune_margin_m"]))
        # Raw (unguarded) per-joint min-sdf diagnostic — kept as the continuous channel.
        vcon = contact_from_sdf(sdf, dmin, conf, close_thresh_m=float(cc["close_thresh_m"]),
                                max_dist_m=float(cc["max_dist_m"]), conf_thresh=float(cc["conf_thresh"]))
        _, jsdf = lift_joint_contact(vcon, sdf, v2j, J)
        jdmin = lift_joint_min_distance(dmin, v2j, J)
        # Final contact: a joint-frame must be BOTH near the wall and still. Drift-gate
        # the guarded per-joint min-sdf (frames whose net window drift ≥ 2·still_m are a
        # pass, not a grip → forced non-contact), then temporal hysteresis (enter<close,
        # stay<exit) + min-duration, so labels stop flickering at the boundary.
        gsdf = joint_guarded_min_sdf(sdf, dmin, conf, v2j, J,
                                     max_dist_m=float(cc["max_dist_m"]), conf_thresh=float(cc["conf_thresh"]))
        # The persisted pre-enforcement drift is the honest motion measurement:
        # later attraction/damping must not manufacture apparent stillness. Both
        # the binary gate and no-contact confidence use it, with final-body drift
        # as a per-element fallback for old/missing measurements.
        motion_drift = joint_drift(
            jw, fps=fps, window_s=float(cg["window_s"]),
            valid_mask=torch.as_tensor(valid_mask[p], device=device))
        gate_drift = motion_drift
        # honest gate: when the body carries the refine-start drift_pre (kinematics_1),
        # the stillness gate reads it (the pre-enforcement measurement) instead of re-measuring the
        # final body — the damper can then no longer manufacture false stillness. Fall back per
        # person to the computed final-body drift (kinematics_2 / old npz / NaN).
        if "drift_pre" in kin:
            dp = torch.as_tensor(
                np.asarray(kin["drift_pre"])[p], dtype=jw.dtype, device=jw.device)
            if dp.shape == motion_drift.shape:
                gate_drift = torch.where(
                    torch.isfinite(dp) & torch.isfinite(motion_drift), dp, motion_drift)
        jc_pre = drift_gated_contact(gsdf, gate_drift, close_thresh_m=float(cc["close_thresh_m"]),
                                     exit_thresh_m=float(cc["exit_thresh_m"]), min_frames=int(cc["min_frames"]),
                                     still_m=float(cg["still_m"]))
        # span-merge: bridge short same-joint gaps the joint barely moved across AND stayed near
        # the wall throughout (flicker / brief re-settle); the re-grip guard blocks leave-and-return.
        jc = merge_contact_spans(jc_pre, jw, gsdf, merge_dist_m=float(cc["merge_dist_m"]),
                                 max_gap_s=float(cc["merge_max_gap_s"]), gap_sdf_m=float(cc["merge_sdf_m"]), fps=fps)
        merged_bool = jc & ~jc_pre     #frames the merge bridged — confidence stays honest, the flag marks them
        jw_cam = transform_points(jw, extr)
        px = project(jw_cam, K)                                      # (N,J,2)

        jc_np = jc.cpu().numpy()
        jc_np[~valid_mask[p]] = False
        joint_contact[p] = jc_np
        joint_sdf_cm[p] = jsdf.cpu().numpy()
        joint_distance_cm[p] = (jdmin * 100.0).cpu().numpy()
        joints_2d[p] = px.cpu().numpy()
        merged_np = merged_bool.cpu().numpy(); merged_np[~valid_mask[p]] = False
        merged_all[p] = merged_np
        # Per-joint confidence in the FINAL label.  The binary label is unchanged;
        # the readout only assigns its continuous training weight.
        jc_final = torch.as_tensor(jc_np, dtype=torch.bool, device=device)
        cf = cfg["contact_confidence"]
        other_vw = None
        if P > 1:
            others = []
            for i, person in enumerate(posed_people):
                if i == p:
                    continue
                other_valid = torch.as_tensor(
                    valid_mask[i], dtype=torch.bool, device=device)[:, None, None]
                others.append(torch.where(
                    other_valid, person[2], torch.full_like(person[2], float("nan"))))
            other_vw = torch.stack(others, dim=1)
        n_occ = joint_occlusion_counts(
            vw, jw, cam_centers, v2j, occ_group_t,
            other_vertices=other_vw,
            perp_m=float(cf["occ_perp_m"]), front_m=float(cf["occ_front_m"]),
            subsample=int(cf["occ_subsample"]))
        terms = joint_label_confidence(
            jc_final, gsdf, gate_drift, px, jw_cam[..., 2], n_occ,
            height=H, width=W, cfg=cf,
            window_s=float(cg["window_s"]))
        for name, val in terms.items():
            val[~valid_mask[p]] = 0.0
            confidence_all[name][p] = val
        n_cf = int(jc_np.any(1).sum())
        print(f"[estimate_contacts] obj_id={int(object_ids[p])}: {n_cf}/{N} frames with "
              f"≥1 joint in contact ({int(jc_np.sum())} joint-frame contact instances)")

    bounded_terms = (
        "image_visibility", "occlusion_visibility", "visibility", "surface",
        "motion", "label_confidence")
    for name in bounded_terms:
        value = confidence_all[name]
        if not np.isfinite(value).all() or bool(((value < 0.0) | (value > 1.0)).any()):
            raise RuntimeError(f"invalid confidence term {name}: expected finite values in [0,1]")
    if (not np.isfinite(confidence_all["occluding_vertices"]).all()
            or bool((confidence_all["occluding_vertices"] < 0.0).any())):
        raise RuntimeError("invalid occluding-vertex count")
    if np.isinf(joint_sdf_cm).any() or np.isinf(joint_distance_cm).any():
        raise RuntimeError("joint scene diagnostics may be NaN when missing, but never infinite")

    save = dict(
        joint_contact=joint_contact, joint_sdf_cm=joint_sdf_cm,
        joint_distance_cm=joint_distance_cm, merged=merged_all,
        joint_image_visibility=confidence_all["image_visibility"],
        joint_occlusion_visibility=confidence_all["occlusion_visibility"],
        joint_occluding_vertices=confidence_all["occluding_vertices"],
        joint_visibility=confidence_all["visibility"],
        surface_evidence=confidence_all["surface"],
        motion_evidence=confidence_all["motion"],
        label_confidence=confidence_all["label_confidence"],
        confidence_schema=np.int32(4),
        confidence_method=np.str_("motion-only-joint-label-v1"),
        confidence_motion_source=np.str_("drift_pre_with_final_fallback"),
        source_body=np.str_(kin_path.name), source_body_sha256=np.str_(source_body_sha256),
        source_scene=np.str_(scene_path.name), source_scene_sha256=np.str_(source_scene_sha256),
        source_transform=np.str_(transform_path.name),
        source_transform_sha256=np.str_(source_transform_sha256),
        object_ids=object_ids, joint_names=kin["joint_names"], parents=parents,
        valid_mask=kin["valid_mask"],
        num_people=np.int32(P), num_frames=np.int32(N),
        video_height=np.int32(H), video_width=np.int32(W), fps=np.float32(fps),
        close_thresh_m=np.float32(cc["close_thresh_m"]),
        exit_thresh_m=np.float32(cc["exit_thresh_m"]),
        min_frames=np.int32(cc["min_frames"]),
        max_dist_m=np.float32(cc["max_dist_m"]),
        drift_window_s=np.float32(cg["window_s"]),
        drift_still_m=np.float32(cg["still_m"]),
    )
    save.update({f"confidence_{key}": np.float32(value)
                 for key, value in cfg["contact_confidence"].items()})
    out_path = paths.contacts_path(pred, tag)
    np.savez(out_path, **save)
    print(f"[estimate_contacts] wrote {out_path}")

    if overlay:
        from tools.human_optim.overlay import render_contact_overlay
        ov = render_contact_overlay(source, paths.overlay_path(pred, f"contacts_{tag}"),
                                    joints_2d, joint_contact, parents, valid_mask, H, W, fps,
                                    label_confidence=confidence_all["label_confidence"])
        print(f"[estimate_contacts] wrote {ov}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--bodies", default="kinematic_1", choices=["kinematic_1", "kinematic_2"],
                    help="which refined body under human_optim/ to read contact from "
                         "(default kinematic_1; kinematic_2 for the annotation stage)")
    ap.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    args = ap.parse_args(argv)
    cfg = _load_config(args.config)
    run(args.source, args.predictions, args.config,
        overlay=bool(cfg["contacts"]["overlay"]), bodies=args.bodies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
