"""Visually-guided SMPL-X kinematic optimization (replaces the SMPL kinematic stage).

Refines the per-person SMPL-X trajectory from ``sam3d/smplx_params.npz`` using
the image and the metric geometry the pipeline already computed:

  1. **reprojection** — SMPL-X joints/landmarks → image vs Sapiens keypoints,
  2. **chamfer** — visible posed vertices → metric VGGT human depth points,
  3. **pose prior** — joints (incl. hands) stay near the converter's per-frame fit,
  4. **smoothness** — acceleration + jerk of the world trajectory (Lie-exact).

Each tracked person is optimized independently in the **world** frame; the leaf
variable is ``q (T, 211)`` for ``better_human``'s 52-joint :class:`SMPLXMid`
(articulated fingers, no face), optimized with Adam in coarse-to-fine stages
(root SE3 placed first, then orientation + joints, then the scene terms). The
body is used at its native (converter) size — the metric geometry is already
sized to it by the upstream human-anchored ``estimate_scale`` stage.
Posed-mesh inference lives in ``better_human``; the BetterRobot robot model
(``tools.smplx_robot``) supplies only the Lie smoothness term.

Run from CLI::

    .venv/bin/python scripts/stages/optimize_smplx.py \\
        --source video.mp4 --predictions out/video/ [--config configs/smplx_kinematic.toml]

Reads:  sam3d/smplx_params.npz, sapiens/pose.npz, geometry/human.npz, geometry/transform.npz
Writes: sam3d/smplx_kinematic.npz  (per-person world-frame params, stacked (P, N, ...))
        sam3d/smplx_kinematic_overlay.mp4  (--overlay; reprojected keypoints)
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BETTER_HUMAN_SRC = _REPO_ROOT.parent / "better_human" / "src"
for _p in (str(_BETTER_HUMAN_SRC), str(_REPO_ROOT)):
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402
import tomllib  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.smplx_optim import (  # noqa: E402
    build_correspondence, fold_root_to_world, world_from_cam, subsample_indices,
)
from tools.smplx_optim.optimizer import optimize_person  # noqa: E402
from tools.smplx_robot import load_smplx_mid  # noqa: E402
from tools.geometry.normals import estimate_point_normals  # noqa: E402

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "smplx_kinematic.toml"
_DEFAULT_SMPLX_MODEL = (
    _REPO_ROOT.parent / "better_human" / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
)


# ─────────────────────────────── config ──────────────────────────────────

def _load_config(path: Path) -> dict:
    cfg = tomllib.load(path.open("rb")) if path.exists() else {}
    # Loss weights ([weights] + [smoothness]) come ONLY from the TOML — no Python
    # defaults, so each weight has exactly one home to tune. A config that omits
    # one raises KeyError naming it (see configs/smplx_kinematic.toml).
    cfg.setdefault("weights", {})
    cfg.setdefault("smoothness", {})
    oc = cfg.setdefault("optimizer", {})
    oc.setdefault("lr", 0.01)
    oc.setdefault("iters_stage1", 150)   # Adam stage "root": root SE3 (transl+orient) only
    oc.setdefault("iters_stage2", 300)   # Adam stage "full": all DOFs, no scene terms
    oc.setdefault("iters_stage3", 200)   # Adam stage "refine": + penetration + contact
    oc.setdefault("lr_decay", 0.1)
    oc.setdefault("iters_lbfgs", 30)      # L-BFGS polish iterations after the Adam warm-up (0 = off)
    oc.setdefault("lr_lbfgs", 1.0)        # L-BFGS step scale (strong-Wolfe line search refines it)
    oc.setdefault("lbfgs_history", 10)    # L-BFGS history size
    ch = cfg.setdefault("chamfer", {})
    ch.setdefault("conf_thresh", 0.1)
    ch.setdefault("max_points", 1500)
    ch.setdefault("vertex_subsample", 1000)
    ch.setdefault("chunk", 16)            # frame-chunk for the batched chamfer/penetration cdist
    kp = cfg.setdefault("keypoints", {})
    kp.setdefault("score_thresh", 0.3)
    kp.setdefault("include_hands", False)
    kp.setdefault("gm_scale_px", 40.0)    # Sapiens GM robust scale (uniform; no per-point σ)
    ri = cfg.setdefault("reproj_init", {})  # init-projection 2D anchor (weight [weights].reproj_init)
    ri.setdefault("gm_scale_px", 30.0)    # GM robust scale (px) for the body-joint init anchor
    pn = cfg.setdefault("penetration", {})
    pn.setdefault("conf_thresh", 0.1)     # trust scene points above this VGGT confidence
    pn.setdefault("margin_m", 0.02)       # allowed penetration depth (dead-zone), metres
    pn.setdefault("max_dist_m", 0.15)     # ignore verts farther than this from any scene point
    pn.setdefault("max_pen_m", 0.10)      # occlusion gate: ignore "penetration" deeper than this
    pn.setdefault("huber_m", 0.02)        # Huber quad→linear transition on penetration excess
    pn.setdefault("prune_margin_m", 0.30) # per-frame scene-candidate AABB margin (covers refine motion)
    pn.setdefault("max_candidates", 1500) # cap per-frame scene candidates (nearest-to-box; bounds cdist width)
    pn.setdefault("normal_k", 16)         # kNN for PCA scene-normal estimation
    ct = cfg.setdefault("contact_velocity", {})  # velocity-gated contact repulsion (refine stage)
    ct.setdefault("conf_thresh", 0.1)      # trust scene points above this VGGT confidence
    ct.setdefault("max_dist_m", 0.15)      # ignore verts farther than this from any scene point
    ct.setdefault("max_pen_m", 0.10)       # occlusion gate: ignore "penetration" deeper than this
    ct.setdefault("contact_band_m", 0.03)  # within this signed dist counts as contact (repulsion band)
    ct.setdefault("vel_ref", 1.0)          # velocity scale (m/s): gate g = (vel/vel_ref)^vel_exp
    ct.setdefault("vel_exp", 2.0)          # gate exponent (smooth, threshold-free; >1 ⇒ super-linear)
    at = cfg.setdefault("contact_attraction", {})  # scene-contact attraction (refine stage)
    at.setdefault("conf_thresh", 0.1)      # trust scene points above this VGGT confidence
    at.setdefault("band_m", 0.10)          # pull verts hovering within this of the wall (0<s<band) onto it
    at.setdefault("max_dist_m", 0.15)      # ignore verts farther than this from any scene point
    rt = cfg.setdefault("runtime", {})
    rt.setdefault("device", "cuda")
    return cfg


# ─────────────────────────────── io helpers ──────────────────────────────

def _person_depth_points(human: dict, oid: int, N: int, conf_thresh: float,
                         max_points: int, device) -> list[torch.Tensor]:
    """Per-frame world-frame depth points for one person (conf-gated, capped)."""
    mask = (human["obj_id"] == oid) & (human["confidence"] > conf_thresh)
    pts = human["points"][mask]
    fr = human["frame_id"][mask]
    order = np.argsort(fr, kind="stable")
    pts, fr = pts[order], fr[order]
    bnd = np.searchsorted(fr, np.arange(N + 1))
    out = []
    for t in range(N):
        p = pts[bnd[t]:bnd[t + 1]]
        if len(p) > max_points:
            p = p[np.linspace(0, len(p) - 1, max_points).astype(int)]
        out.append(torch.as_tensor(p, dtype=torch.float32, device=device))
    return out


def _sapiens_for(pose: dict, oid: int, N: int):
    """Return (keypoints (N,308,2), scores (N,308)) for the person with id ``oid``."""
    ids = list(int(x) for x in pose["object_ids"])
    p = ids.index(oid)
    return pose["keypoints"][p], pose["keypoint_scores"][p]


# ─────────────────────────────── main ────────────────────────────────────

def run(source, predictions_dir, config_path, *, device_override=None, overlay=False) -> Path:
    cfg = _load_config(Path(config_path))
    if device_override:
        cfg["runtime"]["device"] = device_override
    device = cfg["runtime"]["device"]
    pred = Path(predictions_dir)
    sam3d = pred / "sam3d"

    sp = dict(np.load(sam3d / "smplx_params.npz", allow_pickle=True))
    pose = dict(np.load(pred / "sapiens" / "pose.npz", allow_pickle=True))
    human = dict(np.load(pred / "geometry" / "human.npz", allow_pickle=True))
    tr = dict(np.load(pred / "geometry" / "transform.npz", allow_pickle=True))

    N = int(sp["num_frames"])
    P = int(sp["num_people"])
    H, W = int(sp["video_height"]), int(sp["video_width"])
    object_ids = np.asarray(sp["object_ids"], dtype=np.int32)
    fps = float(tr.get("fps", 30.0))

    extr = torch.as_tensor(tr["extrinsics"], dtype=torch.float32, device=device)      # (N,4,4)
    K = torch.as_tensor(tr["intrinsics_px_orig"], dtype=torch.float32, device=device)  # (N,3,3)
    wfc = world_from_cam(extr)

    # Scene cloud → penetration loss (optional; the stage runs without it). PCA
    # normals oriented toward the cameras give the SDF its sign; computed once
    # here and shared across people.
    scene_pts = scene_normals = scene_conf = None
    scene_path = pred / "geometry" / "scene.npz"
    if scene_path.exists():
        sc = dict(np.load(scene_path, allow_pickle=True))
        scene_pts = torch.as_tensor(sc["points"], dtype=torch.float32, device=device)
        scene_conf = torch.as_tensor(sc["confidence"], dtype=torch.float32, device=device)
        nk = int(cfg["penetration"]["normal_k"])
        scene_normals = estimate_point_normals(scene_pts, wfc[:, :3, 3], k=nk)
        print(f"[optimize_smplx] scene penetration ON ({scene_pts.shape[0]} pts, PCA normals k={nk})")
    else:
        print("[optimize_smplx] no geometry/scene.npz; penetration loss OFF")

    mid = load_smplx_mid(_DEFAULT_SMPLX_MODEL, device)
    fvids = torch.as_tensor([int(x) for x in mid.frame_vertex_ids], dtype=torch.long, device=device)
    corr = build_correspondence(list(mid.frame_names), include_hands=bool(cfg["keypoints"]["include_hands"]), device=device)
    sub = subsample_indices(mid.num_vertices, int(cfg["chamfer"]["vertex_subsample"]), device=device)
    torch.manual_seed(0)

    nb = mid.num_joints
    out: dict[str, list] = {k: [] for k in (
        "global_orient", "transl", "body_pose", "left_hand_pose", "right_hand_pose", "q")}
    betas_out = []

    for p in range(P):
        oid = int(object_ids[p])
        vmask = sp["valid_mask"][p]
        betas_mean = torch.as_tensor(
            sp["betas"][p][vmask].mean(0), dtype=torch.float32, device=device)

        def t(k):  # per-frame (N, ...) param tensor
            return torch.as_tensor(sp[k][p], dtype=torch.float32, device=device)
        betas_T = betas_mean[None].expand(N, -1)
        q_cam = mid.from_classic(betas_T, t("body_pose"), t("transl"), t("global_orient"),
                                 t("left_hand_pose"), t("right_hand_pose"))
        q_world = fold_root_to_world(q_cam, wfc)
        # Micro-dither joints to break any bit-exact static-frame ties (model.difference).
        q_world[:, 3:] += 1e-6 * torch.randn_like(q_world[:, 3:])

        kp_np, sc_np = _sapiens_for(pose, oid, N)
        target_kp = torch.as_tensor(kp_np, dtype=torch.float32, device=device)
        scores = torch.as_tensor(sc_np, dtype=torch.float32, device=device)
        # Drop evidence on invalid frames (pose prior + smoothness still apply).
        depth = _person_depth_points(human, oid, N, float(cfg["chamfer"]["conf_thresh"]),
                                     int(cfg["chamfer"]["max_points"]), device)
        for i in range(N):
            if not vmask[i]:
                scores[i] = 0.0
                depth[i] = depth[i][:0]

        print(f"[optimize_smplx] person obj_id={oid} ({int(vmask.sum())}/{N} valid) …")
        q_opt, before, after = optimize_person(
            mid, betas_mean, q_world, extrinsics=extr, K=K, depth_points=depth,
            target_kp=target_kp, scores=scores, corr=corr, frame_vertex_ids=fvids,
            sub=sub, cfg=cfg,
            scene_pts=scene_pts, scene_normals=scene_normals, scene_conf=scene_conf, fps=fps)
        print(f"    before: {before}\n    after:  {after}")

        cls = mid.to_classic(betas_T, q_opt)
        # to_classic uses key "translation"; we store it as "transl" (npz schema).
        key_map = {"global_orient": "global_orient", "transl": "translation",
                   "body_pose": "body_pose", "left_hand_pose": "left_hand_pose",
                   "right_hand_pose": "right_hand_pose"}
        for out_k, cls_k in key_map.items():
            out[out_k].append(cls[cls_k].cpu().numpy().astype(np.float32))
        out["q"].append(q_opt.cpu().numpy().astype(np.float32))
        betas_out.append(betas_mean.cpu().numpy().astype(np.float32))

    parents = [-1] + [int(mid.parent_tree[0, i]) for i in range(1, nb)]
    save = {k: np.stack(v, 0) for k, v in out.items()}
    save.update(
        betas=np.stack(betas_out, 0),
        # Body optimized at native (converter) size — no per-body scale now that
        # the metric geometry is human-anchored upstream. Kept as ones so the
        # many consumers that re-apply `scale` about the pelvis stay a no-op.
        scale=np.ones(P, np.float32),
        jaw_pose=np.zeros((P, N, 3), np.float32),
        leye_pose=np.zeros((P, N, 3), np.float32),
        reye_pose=np.zeros((P, N, 3), np.float32),
        expression=np.zeros((P, N, 10), np.float32),
        joint_names=np.array(list(mid.joint_names), dtype="U16"),
        parents=np.array(parents, dtype=np.int32),
        object_ids=object_ids,
        valid_mask=sp["valid_mask"],
        num_people=np.int32(P), num_frames=np.int32(N),
        video_height=np.int32(H), video_width=np.int32(W),
        fps=np.float32(fps), frame=np.array("world", dtype="U8"),
        model_type=np.array("smplx_mid", dtype="U16"), num_betas=np.int32(10),
    )
    out_path = sam3d / "smplx_kinematic.npz"
    np.savez(out_path, **save)
    print(f"[optimize_smplx] wrote {out_path}")

    if overlay:
        from tools.smplx_optim.overlay import render_keypoint_overlay
        ov = render_keypoint_overlay(
            source, sam3d / "smplx_kinematic_overlay.mp4", mid, save, corr, fvids,
            extr, K, target_scheme=pose, fps=fps, device=device)
        print(f"[optimize_smplx] wrote {ov}")
    return out_path


# ─────────────────────────────── cli ─────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    ap.add_argument("--device", default=None)
    ap.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args(argv)
    run(args.source, args.predictions, args.config,
        device_override=args.device, overlay=args.overlay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
