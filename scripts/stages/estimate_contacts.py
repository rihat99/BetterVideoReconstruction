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

Finally, when the clip carries an external image model's per-limb contact
probabilities (``predictions/contacts.npz``), the two witnesses are fused by adding
their log-odds (:mod:`tools.human_optim.fusion`): the model can turn a limb on where
the geometry knows nothing, or off where the geometry is barely sure, and clipping its
probability bounds how far it may ever move a label. Without that file the stage is
unchanged (bit-for-bit).

Renders ``contacts_1_overlay.mp4``: the projected SMPL-X skeleton, joints **red**
in contact (dot size ∝ confidence) / **green** otherwise.

Run from CLI::

    .venv/bin/python scripts/stages/estimate_contacts.py \\
        --source video.mp4 --predictions out/video/ --config configs/human_optimization.toml

Reads:  human_optim/kinematic_1.npz, geometry/scene.npz, geometry/transform.npz,
        geometry/human.npz (optional for old artifacts; supplies depth-fit diagnostics)
        predictions/contacts.npz (optional; the image model's per-limb probabilities)
Writes: human_optim/contacts_1.npz          (joint_contact (P,N,J) bool + signed/Euclidean
                                             scene-distance diagnostics + merged + auditable
                                             visibility/evidence terms + label_confidence +
                                             per-joint depth gap/count/trust + the fused
                                             prediction probabilities and what they changed)
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
    COMPLETION_DEFAULTS, DEPTH_FIT_DEFAULTS, NO_DATA_TRUST,
    complete_person, contact_from_sdf, drift_gated_contact, joint_depth_fit, joint_drift,
    joint_guarded_min_sdf,
    joint_occlusion_gap, lift_joint_contact, lift_joint_min_distance, load_dynamics_context,
    merge_contact_spans, project, scene_sdf,
    transform_points, vertex_to_joint, world_from_cam,
)
from tools.human_optim import paths  # noqa: E402
from tools.human_optim.confidence import (  # noqa: E402
    CONTACT_CONFIDENCE_DEFAULTS, CONTACT_CONFIDENCE_METHOD,
    CONTACT_CONFIDENCE_SCHEMA, confidence_config, joint_label_confidence,
    occlusion_group_ids)
from tools.human_optim.fusion import (  # noqa: E402
    FUSION_METHOD, PRED_LIMBS, fuse_person, load_pred_prob_limb)
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
    comp = c.setdefault("completion", {})   # detector-completion knobs (defaults in completion.py)
    for key, val in COMPLETION_DEFAULTS.items():
        comp.setdefault(key, val)
    # Predicted-contact fusion: how far the image model may move a label (see fusion.py).
    c.setdefault("fusion", {}).setdefault("pred_clamp", 0.10)
    d = cfg.setdefault("depth_fit", {})
    for key, val in DEPTH_FIT_DEFAULTS.items():
        d.setdefault(key, val)
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
    df = cfg["depth_fit"]
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
    human_path = pred / "geometry" / "human.npz"
    human_cloud = None
    source_human_sha256 = ""
    if human_path.exists():
        source_human_sha256 = _sha256(human_path)
        with np.load(human_path) as human:
            human_cloud = tuple(torch.as_tensor(
                np.asarray(human[name]), device=device)
                for name in ("points", "frame_id", "obj_id"))
    else:
        print(f"[estimate_contacts] warning: missing {human_path}; depth-fit fields have no data")
    N, P = int(kin["num_frames"]), int(kin["num_people"])
    H, W = int(kin["video_height"]), int(kin["video_width"])
    object_ids = np.asarray(kin["object_ids"], dtype=np.int32)
    parents = np.asarray(kin["parents"], dtype=np.int32)
    valid_mask = np.asarray(kin["valid_mask"], dtype=bool)
    fps = float(tr.get("fps", kin.get("fps", 30.0)))   # drift window needs the true clip fps

    # The image model's per-limb contact probability, aligned to this body's people and
    # frames. Absent file ⇒ fusion is a no-op and every value below is unchanged. A
    # prediction on an invalid body frame is unusable (there is no label to fuse it with).
    pred_clamp = float(cc["fusion"]["pred_clamp"])
    pred_prob_limb = load_pred_prob_limb(pred, object_ids, np.asarray(tr["frame_indices"]))
    do_fusion = pred_prob_limb is not None
    if do_fusion:
        pred_prob_limb[~valid_mask] = np.nan
        n_pred = int(np.isfinite(pred_prob_limb).any(-1).sum())
        print(f"[estimate_contacts] predicted contacts: {n_pred}/{P * N} person-frames, "
              f"clamp {pred_clamp:.2f}")
    else:
        pred_prob_limb = np.full((P, N, len(PRED_LIMBS)), np.nan, np.float32)
        print("[estimate_contacts] fusion: OFF (no predictions/contacts.npz)")
    fusion_changed = np.zeros((P, N, len(PRED_LIMBS)), dtype=np.int8)

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
    faces_t = mid.faces.to(device).long()

    # Contact completion setup (static support solve + union readout). It borrows the
    # dynamics-stage knobs; its friction cones read the ungated scene cloud (conf_drop_q=0,
    # matching the dynamics stage). Gravity follows the dynamics convention: geocalib
    # world-down when present, else world-down [0, 1, 0].
    # Completion runs only when the body carries the persisted pre-enforcement drift
    # (``drift_pre``): its stillness gates need that honest witness. kinematics_1 saves it
    # at refine start, and kinematics_2 forwards it unchanged — attraction and damping have
    # acted on the kin2 POSE, but the stillness evidence still predates any enforcement.
    # An old body npz without drift_pre stays detector-only (re-measured stillness there
    # would be manufactured evidence).
    do_completion = "drift_pre" in kin
    comp_ctx = load_dynamics_context(config_path)
    comp_knobs = cc["completion"]
    grav_path = pred / "geocalib" / "gravity.npz"
    source_gravity = ""
    source_gravity_sha256 = ""
    if grav_path.exists():
        gd = np.load(grav_path)
        gravity_world = np.asarray(gd["gravity_world"], np.float32)
        gsrc = str(gd["source"]) if "source" in gd.files else "geocalib"
        source_gravity = grav_path.name
        source_gravity_sha256 = _sha256(grav_path)
        gravity_source = "geocalib"
    else:
        gravity_world = np.array([0.0, 1.0, 0.0], np.float32)
        gsrc = "fallback_down(no geocalib)"
        gravity_source = "fallback_world_down"
    cone_pts = cone_normals = None
    if do_completion and comp_ctx.friction is not None:
        cone_pts, cone_normals, _ = load_scene_cloud(
            sc, cam_centers, conf_drop_q=0.0,
            k=int(comp_ctx.friction["normal_k"]), device=device)
    if do_completion:
        print(f"[estimate_contacts] completion: {len(comp_ctx.groups)} groups, "
              f"gravity={gravity_world.round(2)} ({gsrc}), "
              f"friction cones {'ON' if cone_pts is not None else 'OFF'}")
    else:
        print(f"[estimate_contacts] completion: OFF for {bodies} (post-enforcement body)")

    joint_contact = np.zeros((P, N, J), dtype=bool)
    contact_source = np.zeros((P, N, J), dtype=np.int8)   # per joint-frame: 1=detector, 2=gate, 4=force
    limb_force_n = np.zeros((P, N, len(comp_ctx.groups)), dtype=np.float32)
    force_share = np.full((P, N, J), np.nan, dtype=np.float32)  # |f|/body-weight at the six limb joints; NaN = no force witness
    body_weight_n = np.full(P, np.nan, dtype=np.float32)   # total baked mass × g per person
    joint_sdf_cm = np.full((P, N, J), np.nan, dtype=np.float32)
    joint_distance_cm = np.full((P, N, J), np.nan, dtype=np.float32)
    joint_depth_gap_cm = np.full((P, N, J), np.nan, dtype=np.float32)
    joint_occlusion_gap_cm = np.full((P, N, J), np.nan, dtype=np.float32)
    joint_depth_points = np.zeros((P, N, J), dtype=np.float32)
    joint_depth_trust = np.full((P, N, J), NO_DATA_TRUST, dtype=np.float32)
    joints_2d = np.full((P, N, J, 2), np.nan, dtype=np.float32)
    merged_all = np.zeros((P, N, J), dtype=bool)            #frames the span-merge bridged
    confidence_all = {name: np.zeros((P, N, J), np.float32) for name in
                      ("image_visibility", "occlusion_visibility",
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
        if human_cloud is not None:
            depth_gap, depth_points, depth_trust = joint_depth_fit(
                vw, *human_cloud, int(object_ids[p]), extr, K, v2j, J,
                cell_px=int(df["cell_px"]), min_points=int(df["min_points"]),
                trust_scale_log=float(df["trust_scale_log"]))
            gap_np = depth_gap.cpu().numpy()
            points_np = depth_points.cpu().numpy()
            trust_np = depth_trust.cpu().numpy()
            gap_np[~valid_mask[p]] = np.nan
            points_np[~valid_mask[p]] = 0.0
            trust_np[~valid_mask[p]] = NO_DATA_TRUST
            joint_depth_gap_cm[p] = gap_np
            joint_depth_points[p] = points_np
            joint_depth_trust[p] = trust_np
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
        joint_sdf_cm[p] = jsdf.cpu().numpy()
        joint_distance_cm[p] = (jdmin * 100.0).cpu().numpy()
        joints_2d[p] = px.cpu().numpy()
        merged_np = merged_bool.cpu().numpy(); merged_np[~valid_mask[p]] = False
        merged_all[p] = merged_np

        # Contact completion: the static support solve + loosened gates, unioned onto the
        # detector labels (add-only). Additions land on each group's own joint; the source
        # bitmask records where each label came from. Confidence below then runs on the
        # COMPLETED labels.
        detector_np = jc_np.copy()
        src = np.where(detector_np, np.int8(1), np.int8(0))
        if do_completion:
            comp = complete_person(
                mid, torch.as_tensor(kin["betas"][p], dtype=torch.float32, device=device),
                torch.as_tensor(kin["q"][p], dtype=torch.float32, device=device),
                valid_mask[p], mid_names,
                detector_contact=detector_np, joint_distance_cm=joint_distance_cm[p],
                gate_drift=gate_drift, gravity_world=gravity_world, fps=fps,
                cone_pts=cone_pts, cone_normals_t=cone_normals, ctx=comp_ctx,
                knobs=comp_knobs, device=device)
            for c, j in enumerate(comp["rep_idx"]):
                jc_np[:, j] |= comp["gate_span"][:, c] | comp["force_span"][:, c]
                src[comp["gate_span"][:, c], j] |= 2
                src[comp["force_span"][:, c], j] |= 4
            limb_force_n[p] = comp["limb_force_n"]
            bw = comp["body_weight_n"]
            body_weight_n[p] = bw
            # Force share |f|/body-weight at each group's own joint feeds the
            # source-conditional confidence. A failed solve (bw = inf) leaves the
            # share NaN, so those frames carry no force witness.
            if np.isfinite(bw) and bw > 0.0:
                for c, j in enumerate(comp["rep_idx"]):
                    force_share[p, :, j] = comp["limb_force_n"][:, c] / bw
        joint_contact[p] = jc_np
        contact_source[p] = src
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
        occ_gap = joint_occlusion_gap(
            vw, jw, cam_centers, faces_t, v2j, occ_group_t, other_vertices=other_vw)
        depth_trust_t = torch.as_tensor(
            joint_depth_trust[p], dtype=jw.dtype, device=device)
        depth_points_t = torch.as_tensor(
            joint_depth_points[p], dtype=jw.dtype, device=device)
        source_t = torch.as_tensor(contact_source[p], dtype=torch.int64, device=device)
        share_t = torch.as_tensor(force_share[p], dtype=jw.dtype, device=device)
        terms = joint_label_confidence(
            jc_final, gsdf, gate_drift, depth_trust_t, depth_points_t, source_t, share_t,
            px, jw_cam[..., 2], occ_gap, height=H, width=W, cfg=cf,
            window_s=float(cg["window_s"]), min_depth_points=int(df["min_points"]))
        # Occlusion gap (cm) is a NaN-or-finite diagnostic, not a bounded [0,1] term:
        # keep its NaN (no surface in front) and NaN the invalid frames, exactly like
        # joint_depth_gap_cm — never route it through the [0,1] zeroing below.
        gap_cm = terms.pop("occlusion_gap") * 100.0
        gap_cm[~valid_mask[p]] = np.nan
        joint_occlusion_gap_cm[p] = gap_cm
        for name, val in terms.items():
            val[~valid_mask[p]] = 0.0
            confidence_all[name][p] = val
        # Predicted-contact fusion runs LAST, on the finished labels and confidence: the
        # geometric witnesses stay exactly as they were, and only the limb-frames where
        # the model disagrees strongly enough move (with their own fused confidence).
        if do_fusion:
            jc_np, fused_conf, fused_src, changed = fuse_person(
                jc_np, confidence_all["label_confidence"][p], contact_source[p],
                pred_prob_limb[p], min_frames=int(cc["min_frames"]), pred_clamp=pred_clamp)
            joint_contact[p] = jc_np
            contact_source[p] = fused_src
            confidence_all["label_confidence"][p] = fused_conf
            fusion_changed[p] = changed
            flips = ", ".join(f"{name} +{int((changed[:, c] > 0).sum())}/"
                              f"-{int((changed[:, c] < 0).sum())}"
                              for c, name in enumerate(PRED_LIMBS))
            print(f"[estimate_contacts] obj_id={int(object_ids[p])}: fusion rescued/vetoed "
                  f"limb-frames — {flips}")
        n_cf = int(jc_np.any(1).sum())
        print(f"[estimate_contacts] obj_id={int(object_ids[p])}: {n_cf}/{N} frames with "
              f"≥1 joint in contact ({int(jc_np.sum())} joint-frame contact instances)")
        finite_joint = np.isfinite(joint_depth_gap_cm[p]).any(0)
        if finite_joint.any():
            med = np.full(J, np.nan, dtype=np.float32)
            for j in np.flatnonzero(finite_joint):
                med[j] = np.nanmedian(joint_depth_gap_cm[p, :, j])
            worst = int(np.nanargmax(np.abs(med)))
            print(f"[estimate_contacts] obj_id={int(object_ids[p])}: depth fit worst "
                  f"{mid_names[worst]} median={float(med[worst]):+.1f}cm")
        else:
            print(f"[estimate_contacts] obj_id={int(object_ids[p])}: depth fit has no matched joints")

    bounded_terms = (
        "image_visibility", "occlusion_visibility", "visibility", "surface",
        "motion", "label_confidence")
    for name in bounded_terms:
        value = confidence_all[name]
        if not np.isfinite(value).all() or bool(((value < 0.0) | (value > 1.0)).any()):
            raise RuntimeError(f"invalid confidence term {name}: expected finite values in [0,1]")
    if np.isinf(joint_occlusion_gap_cm).any():
        raise RuntimeError("joint occlusion gap may be NaN when missing, but never infinite")
    if np.isinf(joint_sdf_cm).any() or np.isinf(joint_distance_cm).any():
        raise RuntimeError("joint scene diagnostics may be NaN when missing, but never infinite")
    if (not np.isfinite(joint_depth_trust).all()
            or bool(((joint_depth_trust < 0.0) | (joint_depth_trust > 1.0)).any())):
        raise RuntimeError("invalid joint_depth_trust: expected finite values in [0,1]")
    if np.isinf(joint_depth_gap_cm).any() or not np.isfinite(joint_depth_points).all():
        raise RuntimeError("joint depth gap may be NaN when missing, but never infinite")
    # Completion outputs: solved forces must be finite (a non-finite solve is caught upstream
    # and falls back to zeros), and any body weight the solve produced must be positive. The
    # NaN (completion skipped / no witness) and +inf (solve failed) sentinels are left as-is.
    if not np.isfinite(limb_force_n).all():
        raise RuntimeError("limb_force_n must be finite (a non-finite solve force reached save)")
    computed_weight = np.isfinite(body_weight_n)
    if bool((body_weight_n[computed_weight] <= 0.0).any()):
        raise RuntimeError("body_weight_n must be positive where the support solve produced a mass")

    save = dict(
        joint_contact=joint_contact, joint_sdf_cm=joint_sdf_cm,
        joint_distance_cm=joint_distance_cm, merged=merged_all,
        contact_source=contact_source, limb_force_n=limb_force_n,
        force_share=force_share, body_weight_n=body_weight_n,
        completion_groups=np.array(list(comp_ctx.groups)),
        completion_method=np.str_(
            "static-support-v1" if do_completion else "skipped(post-enforcement body)"),
        contact_label_schema=np.int32(1),
        contact_label_method=np.str_(
            ("completed-static-support-v1" if do_completion else "detector-v1")
            + (f"+{FUSION_METHOD}" if do_fusion else "")),
        pred_prob_limb=pred_prob_limb.astype(np.float32),
        pred_limbs=np.array(PRED_LIMBS),
        fusion_method=np.str_(FUSION_METHOD if do_fusion else "none"),
        fusion_pred_clamp=np.float32(pred_clamp),
        fusion_changed=fusion_changed,
        joint_depth_gap_cm=joint_depth_gap_cm,
        joint_depth_points=joint_depth_points,
        joint_depth_trust=joint_depth_trust,
        joint_image_visibility=confidence_all["image_visibility"],
        joint_occlusion_visibility=confidence_all["occlusion_visibility"],
        joint_occlusion_gap_cm=joint_occlusion_gap_cm,
        joint_visibility=confidence_all["visibility"],
        surface_evidence=confidence_all["surface"],
        motion_evidence=confidence_all["motion"],
        label_confidence=confidence_all["label_confidence"],
        confidence_schema=np.int32(CONTACT_CONFIDENCE_SCHEMA),
        confidence_method=np.str_(CONTACT_CONFIDENCE_METHOD),
        confidence_motion_source=np.str_("drift_pre_with_final_fallback"),
        source_body=np.str_(kin_path.name), source_body_sha256=np.str_(source_body_sha256),
        source_scene=np.str_(scene_path.name), source_scene_sha256=np.str_(source_scene_sha256),
        source_transform=np.str_(transform_path.name),
        source_transform_sha256=np.str_(source_transform_sha256),
        source_human=np.str_(human_path.name if human_cloud is not None else ""),
        source_human_sha256=np.str_(source_human_sha256),
        source_gravity=np.str_(source_gravity),
        source_gravity_sha256=np.str_(source_gravity_sha256),
        gravity_source=np.str_(gravity_source),
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
    save.update({f"depth_fit_{key}": np.float32(value)
                 for key, value in cfg["depth_fit"].items()})
    save.update({f"completion_{key}": (np.int32(value) if key == "min_span" else np.float32(value))
                 for key, value in comp_knobs.items()})
    if not np.array_equal(joint_contact, contact_source != 0):
        raise RuntimeError("label provenance mismatch: joint_contact != (contact_source != 0)")
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
