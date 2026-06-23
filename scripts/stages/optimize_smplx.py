"""Visually-guided SMPL-X kinematic optimization (replaces the SMPL kinematic stage).

Refines the per-person SMPL-X trajectory from ``sam3d/smplx_params.npz`` using
the image and the metric geometry the pipeline already computed:

  1. **reprojection** — SMPL-X joints/landmarks → image vs Sapiens keypoints,
  2. **chamfer** — visible posed vertices → metric VGGT human depth points,
  3. **pose prior** — joints (incl. hands) stay near the converter's per-frame fit,
  4. **smoothness** — acceleration + jerk of the world trajectory (Lie-exact).

Each tracked person is optimized independently in the **world** frame; the leaf
variable is ``q (T, 211)`` for ``better_human``'s 52-joint :class:`SMPLXMid`
(articulated fingers, no face), with autograd L-BFGS (no hand-coded Jacobians).
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
from tools.smplx_robot import (  # noqa: E402
    load_smplx_mid, make_smplx_robot_model, remap_q_bh_to_model,
)

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "smplx_kinematic.toml"
_DEFAULT_SMPLX_MODEL = (
    _REPO_ROOT.parent / "better_human" / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
)


# ─────────────────────────────── config ──────────────────────────────────

def _load_config(path: Path) -> dict:
    cfg = tomllib.load(path.open("rb")) if path.exists() else {}
    w = cfg.setdefault("weights", {})
    w.setdefault("reprojection", 0.2)
    w.setdefault("chamfer", 1.0)
    w.setdefault("pose_prior", 400.0)
    w.setdefault("orient_prior", 50.0)
    w.setdefault("scale_prior", 5.0)
    sm = cfg.setdefault("smoothness", {})
    sm.setdefault("accel_root", 3.0)
    sm.setdefault("accel_joint", 6.0)
    sm.setdefault("jerk_root", 15.0)
    sm.setdefault("jerk_joint", 30.0)
    oc = cfg.setdefault("optimizer", {})
    oc.setdefault("lr", 0.01)
    oc.setdefault("iters_stage1", 150)
    oc.setdefault("iters", 300)
    oc.setdefault("lr_decay", 0.1)
    oc.setdefault("scale_init", 1.0)
    oc.setdefault("lr_scale", 0.01)
    ch = cfg.setdefault("chamfer", {})
    ch.setdefault("conf_thresh", 0.1)
    ch.setdefault("max_points", 1500)
    ch.setdefault("vertex_subsample", 1000)
    kp = cfg.setdefault("keypoints", {})
    kp.setdefault("score_thresh", 0.3)
    kp.setdefault("include_hands", False)
    kp.setdefault("robust_px", 40.0)
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

    mid = load_smplx_mid(_DEFAULT_SMPLX_MODEL, device)
    fvids = torch.as_tensor([int(x) for x in mid.frame_vertex_ids], dtype=torch.long, device=device)
    corr = build_correspondence(list(mid.frame_names), include_hands=bool(cfg["keypoints"]["include_hands"]), device=device)
    sub = subsample_indices(mid.num_vertices, int(cfg["chamfer"]["vertex_subsample"]), device=device)
    torch.manual_seed(0)

    nb = mid.num_joints
    out: dict[str, list] = {k: [] for k in (
        "global_orient", "transl", "body_pose", "left_hand_pose", "right_hand_pose", "q")}
    betas_out = []
    scales_out = []

    for p in range(P):
        oid = int(object_ids[p])
        vmask = sp["valid_mask"][p]
        betas_mean = torch.as_tensor(
            sp["betas"][p][vmask].mean(0), dtype=torch.float32, device=device)
        model = make_smplx_robot_model(mid, betas_mean, device=device)
        remap = lambda q: remap_q_bh_to_model(q, mid, model)

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
        q_opt, scale_opt, before, after = optimize_person(
            mid, model, betas_mean, q_world, extrinsics=extr, K=K, depth_points=depth,
            target_kp=target_kp, scores=scores, corr=corr, frame_vertex_ids=fvids,
            sub=sub, cfg=cfg, remap=remap)
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
        scales_out.append(np.float32(scale_opt))

    parents = [-1] + [int(mid.parent_tree[0, i]) for i in range(1, nb)]
    save = {k: np.stack(v, 0) for k, v in out.items()}
    save.update(
        betas=np.stack(betas_out, 0),
        scale=np.asarray(scales_out, np.float32),
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
