"""Annotation-guided SMPL-X kinematic refinement — STAGE 2 (``kinematics_2``).

Continues refining the visually-guided kinematic result
(``sam3d/smplx_kinematic.npz``) using the **manual per-joint contact annotation**
a human made in ``../BetterContactAnnotator`` (saved as ``data/<stem>.npz``). The
annotation is coarse — one joint per hand / foot / knee / hip / elbow, tri-state
``{-1 unlabeled, 0 no-contact, 1 contact}`` — so each annotation joint is expanded
onto the SMPLXMid joints it covers (``left_hand`` → wrist + 15 finger joints,
``left_foot`` → ankle + foot, …) to build a per-frame ``(N, 52)`` label.

Same visual losses as the kinematic stage, plus three annotation-driven terms in
the refine stage (see ``configs/smplx_kinematic_2.toml`` + ``optimize_person``):
a **real attraction** pulling in-contact joints' vertices onto the scene, a
**strong repulsion** pushing no-contact joints' vertices off it, and a **velocity
damper** on in-contact joints. The init q + betas come straight from the
kinematic npz (already world-frame on the 52-joint :class:`SMPLXMid`), so the body
just continues refining from the converged kinematic fit at its native size.

Run from CLI::

    .venv/bin/python scripts/stages/optimize_smplx_2.py \\
        --source data/boulder_1.mp4 --predictions out/boulder_1/ \\
        [--annotations data/boulder_1.npz] [--config configs/smplx_kinematic_2.toml] [--overlay]

Reads:  sam3d/smplx_kinematic.npz (init), data/<stem>.npz (annotation),
        sapiens/pose.npz, geometry/{human,scene,transform}.npz
Writes: sam3d/smplx_kinematic_2.npz, sam3d/smplx_kinematic_2_overlay.mp4 (--overlay)
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.smplx_optim import (  # noqa: E402
    build_correspondence, subsample_indices, world_from_cam,
)
from tools.smplx_optim.optimizer import optimize_person  # noqa: E402
from tools.smplx_robot import load_smplx_mid  # noqa: E402
from tools.geometry.normals import estimate_point_normals  # noqa: E402

# Reuse the kinematic stage's config loader, io helpers + model default.
from optimize_smplx import (  # noqa: E402
    _DEFAULT_SMPLX_MODEL, _load_config, _person_depth_points, _sapiens_for,
)

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "smplx_kinematic_2.toml"


# ─────────────────────── annotation → SMPLXMid mapping ────────────────────

def _hand_smplx_joints(side: str) -> list[str]:
    """The SMPLXMid joints one ``<side>_hand`` annotation covers: wrist + 15 fingers."""
    fingers = ("index", "middle", "pinky", "ring", "thumb")
    return [f"{side}_wrist"] + [f"{side}_{f}{k}" for f in fingers for k in (1, 2, 3)]


# Each coarse annotation joint → the SMPLXMid joints it expands onto. Hands cover
# the wrist + all finger joints (the annotation is one label for the whole hand);
# feet the ankle + foot; the single-joint limbs map 1:1.
_ANN_TO_SMPLX: dict[str, list[str]] = {
    "left_hand": _hand_smplx_joints("left"),
    "right_hand": _hand_smplx_joints("right"),
    "left_foot": ["left_ankle", "left_foot"],
    "right_foot": ["right_ankle", "right_foot"],
    "left_knee": ["left_knee"],
    "right_knee": ["right_knee"],
    "left_hip": ["left_hip"],
    "right_hip": ["right_hip"],
    "left_elbow": ["left_elbow"],
    "right_elbow": ["right_elbow"],
}


def _contact_labels(contacts_p: np.ndarray, ann_names: list[str],
                    mid_names: list[str], N: int) -> np.ndarray:
    """Expand one person's ``(J_ann, N_ann)`` annotation into ``(N, 52)`` SMPLXMid labels.

    Every SMPLXMid joint an annotation joint covers inherits that joint's per-frame
    tri-state series; SMPLXMid joints with no annotation (spine / head / collars /
    shoulders, and any annotation joint left fully unlabeled) stay ``-1``.
    """
    Na = contacts_p.shape[1]
    if Na != N:
        raise ValueError(f"annotation has {Na} frames but the body trajectory has {N}")
    mid_idx = {n: i for i, n in enumerate(mid_names)}
    ann_idx = {n: i for i, n in enumerate(ann_names)}
    lab = np.full((N, len(mid_names)), -1, dtype=np.int8)
    for ann_name, smplx_joints in _ANN_TO_SMPLX.items():
        if ann_name not in ann_idx:
            continue
        series = contacts_p[ann_idx[ann_name]]            # (N,) tri-state
        for jn in smplx_joints:
            if jn in mid_idx:
                lab[:, mid_idx[jn]] = series
    return lab


def _resolve_annotation_path(source, annotations) -> Path:
    """The annotation npz: explicit ``--annotations``, else ``<source>.npz`` /
    ``data/<stem>.npz``."""
    if annotations is not None:
        return Path(annotations)
    src = Path(source)
    cand = src.with_suffix(".npz")
    if cand.exists():
        return cand
    return _REPO_ROOT / "data" / f"{src.stem}.npz"


# ─────────────────────────────── main ────────────────────────────────────

def run(source, predictions_dir, config_path, *, annotations=None,
        device_override=None, overlay=False) -> Path:
    cfg = _load_config(Path(config_path))
    # Operational defaults for the two annotation-loss sections (the weights live
    # only in the TOML, like every other weight — a config omitting them errors).
    aa = cfg.setdefault("ann_attraction", {})
    aa.setdefault("conf_thresh", 0.1)
    aa.setdefault("band_m", 0.30)
    aa.setdefault("max_dist_m", 0.30)
    ar = cfg.setdefault("ann_repulsion", {})
    ar.setdefault("conf_thresh", 0.1)
    ar.setdefault("clearance_m", 0.03)
    ar.setdefault("max_dist_m", 0.15)
    ar.setdefault("max_pen_m", 0.10)
    ar.setdefault("huber_m", 0.02)
    if device_override:
        cfg["runtime"]["device"] = device_override
    device = cfg["runtime"]["device"]
    pred = Path(predictions_dir)
    sam3d = pred / "sam3d"

    # Init from the converged kinematic fit (already world-frame q).
    kin = dict(np.load(sam3d / "smplx_kinematic.npz", allow_pickle=True))
    pose = dict(np.load(pred / "sapiens" / "pose.npz", allow_pickle=True))
    human = dict(np.load(pred / "geometry" / "human.npz", allow_pickle=True))
    tr = dict(np.load(pred / "geometry" / "transform.npz", allow_pickle=True))

    N = int(kin["num_frames"])
    P = int(kin["num_people"])
    H, W = int(kin["video_height"]), int(kin["video_width"])
    object_ids = np.asarray(kin["object_ids"], dtype=np.int32)
    fps = float(tr.get("fps", kin.get("fps", 30.0)))

    extr = torch.as_tensor(tr["extrinsics"], dtype=torch.float32, device=device)        # (N,4,4)
    K = torch.as_tensor(tr["intrinsics_px_orig"], dtype=torch.float32, device=device)   # (N,3,3)
    wfc = world_from_cam(extr)

    # Manual annotation (BetterContactAnnotator sidecar): per-person/joint/frame
    # tri-state contact, aligned to the predictions by SAM3 object id.
    ann_path = _resolve_annotation_path(source, annotations)
    ann = dict(np.load(ann_path, allow_pickle=True))
    ann_contacts = ann["contacts"]                                   # (Pa, J_ann, N_ann)
    ann_names = [str(x) for x in ann["joint_names"]]
    ann_oids = [int(x) for x in ann["object_ids"]]
    print(f"[optimize_smplx_2] annotation {ann_path} — {ann_contacts.shape[0]} people, "
          f"joints {ann_names}, object_ids {ann_oids}")

    # Scene cloud → penetration + annotation attraction/repulsion. Same camera-
    # oriented PCA normals as the kinematic stage (shared across people).
    scene_pts = scene_normals = scene_conf = None
    scene_path = pred / "geometry" / "scene.npz"
    if scene_path.exists():
        sc = dict(np.load(scene_path, allow_pickle=True))
        scene_pts = torch.as_tensor(sc["points"], dtype=torch.float32, device=device)
        scene_conf = torch.as_tensor(sc["confidence"], dtype=torch.float32, device=device)
        nk = int(cfg["penetration"]["normal_k"])
        scene_normals = estimate_point_normals(scene_pts, wfc[:, :3, 3], k=nk)
        print(f"[optimize_smplx_2] scene terms ON ({scene_pts.shape[0]} pts, PCA normals k={nk})")
    else:
        print("[optimize_smplx_2] no geometry/scene.npz; scene + annotation contact terms OFF")

    mid = load_smplx_mid(_DEFAULT_SMPLX_MODEL, device)
    mid_names = list(mid.joint_names)
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
        vmask = np.asarray(kin["valid_mask"][p], dtype=bool)
        betas_mean = torch.as_tensor(kin["betas"][p], dtype=torch.float32, device=device)
        # Init q is already world-frame (kinematic output); micro-dither to break
        # bit-exact static-frame ties in the Lie smoothness, as the kinematic stage does.
        q_world = torch.as_tensor(kin["q"][p], dtype=torch.float32, device=device).clone()
        q_world[:, 3:] += 1e-6 * torch.randn_like(q_world[:, 3:])

        kp_np, sc_np = _sapiens_for(pose, oid, N)
        target_kp = torch.as_tensor(kp_np, dtype=torch.float32, device=device)
        scores = torch.as_tensor(sc_np, dtype=torch.float32, device=device)
        depth = _person_depth_points(human, oid, N, float(cfg["chamfer"]["conf_thresh"]),
                                     int(cfg["chamfer"]["max_points"]), device)
        for i in range(N):
            if not vmask[i]:
                scores[i] = 0.0
                depth[i] = depth[i][:0]

        # Manual contact labels for this person (None ⇒ no annotation for this id ⇒
        # plain re-refine). Unlabeled on invalid frames so no annotation term acts there.
        contact_labels = None
        if oid in ann_oids:
            lab = _contact_labels(ann_contacts[ann_oids.index(oid)], ann_names, mid_names, N)
            lab[~vmask] = -1
            contact_labels = torch.as_tensor(lab, dtype=torch.int8, device=device)
            nc = int((lab == 1).any(1).sum()); nn = int((lab == 0).any(1).sum())
            print(f"[optimize_smplx_2] person obj_id={oid}: contacts on {nc} frames, "
                  f"explicit no-contact on {nn} frames")
        else:
            print(f"[optimize_smplx_2] person obj_id={oid}: NO annotation (re-refine only)")

        print(f"[optimize_smplx_2] person obj_id={oid} ({int(vmask.sum())}/{N} valid) …")
        q_opt, before, after = optimize_person(
            mid, betas_mean, q_world, extrinsics=extr, K=K, depth_points=depth,
            target_kp=target_kp, scores=scores, corr=corr, frame_vertex_ids=fvids,
            sub=sub, cfg=cfg,
            scene_pts=scene_pts, scene_normals=scene_normals, scene_conf=scene_conf, fps=fps,
            contact_labels=contact_labels)
        print(f"    before: {before}\n    after:  {after}")

        betas_T = betas_mean[None].expand(N, -1)
        cls = mid.to_classic(betas_T, q_opt)
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
        # Body optimized at native size (no per-body scale); ones for consumer compat.
        scale=np.ones(P, np.float32),
        jaw_pose=np.zeros((P, N, 3), np.float32),
        leye_pose=np.zeros((P, N, 3), np.float32),
        reye_pose=np.zeros((P, N, 3), np.float32),
        expression=np.zeros((P, N, 10), np.float32),
        joint_names=np.array(mid_names, dtype="U16"),
        parents=np.array(parents, dtype=np.int32),
        object_ids=object_ids,
        valid_mask=kin["valid_mask"],
        num_people=np.int32(P), num_frames=np.int32(N),
        video_height=np.int32(H), video_width=np.int32(W),
        fps=np.float32(fps), frame=np.array("world", dtype="U8"),
        model_type=np.array("smplx_mid", dtype="U16"), num_betas=np.int32(10),
    )
    out_path = sam3d / "smplx_kinematic_2.npz"
    np.savez(out_path, **save)
    print(f"[optimize_smplx_2] wrote {out_path}")

    if overlay:
        from tools.smplx_optim.overlay import render_keypoint_overlay
        ov = render_keypoint_overlay(
            source, sam3d / "smplx_kinematic_2_overlay.mp4", mid, save, corr, fvids,
            extr, K, target_scheme=pose, fps=fps, device=device)
        print(f"[optimize_smplx_2] wrote {ov}")
    return out_path


# ─────────────────────────────── cli ─────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--annotations", default=None,
                    help="manual contact npz (default: <source>.npz or data/<stem>.npz)")
    ap.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    ap.add_argument("--device", default=None)
    ap.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args(argv)
    run(args.source, args.predictions, args.config, annotations=args.annotations,
        device_override=args.device, overlay=args.overlay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
