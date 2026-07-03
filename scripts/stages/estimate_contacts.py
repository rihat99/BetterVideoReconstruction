"""Scene-contact estimation for the optimized SMPL-X bodies.

Reads the refined world-frame SMPL-X trajectory (``sam3d/smplx_kinematic.npz``)
and the static scene cloud (``geometry/scene.npz``) and marks, for every tracked
person, which **joints** are in contact with the scene:

  1. pose each person's :class:`SMPLXMid` mesh per frame (re-applying the stored
     global scale about the pelvis, exactly as the kinematic stage / viewers do),
  2. signed-distance each body vertex to the scene surface (the same point-cloud
     SDF the kinematic stage's penetration loss uses) → per-vertex contact
     (inside the scene, or just outside it within ``close_thresh_m``),
  3. lift to **per-joint** contact by nearest-joint assignment — a joint is in
     contact if any vertex closest to it is (vertex→joint by rest-pose 3D distance).

then renders ``cnt_kin_overlay.mp4``: the projected SMPL-X skeleton on each frame,
joints coloured **red** in contact / **green** otherwise.

Run from CLI::

    .venv/bin/python scripts/stages/estimate_contacts.py \\
        --source video.mp4 --predictions out/video/ [--no-overlay]

Reads:  sam3d/smplx_kinematic.npz, geometry/scene.npz, geometry/transform.npz
Writes: sam3d/cnt_kin.npz          (joint_contact (P,N,J) bool + joint_sdf_cm)
        sam3d/cnt_kin_overlay.mp4  (--overlay; default on)
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

from tools.geometry.normals import estimate_point_normals  # noqa: E402
from tools.smplx_optim import (  # noqa: E402
    contact_from_sdf, lift_joint_contact, project, scene_sdf, transform_points,
    vertex_to_joint, world_from_cam,
)
from tools.smplx_optim.qmath import normalize_q  # noqa: E402
from tools.smplx_robot import load_smplx_mid  # noqa: E402

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "smplx_kinematic.toml"
_DEFAULT_SMPLX_MODEL = (
    _REPO_ROOT.parent / "better_human" / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
)


def _load_config(path: Path) -> dict:
    """Operational knobs for the contact stage (no loss weights here)."""
    cfg = tomllib.load(path.open("rb")) if path.exists() else {}
    c = cfg.setdefault("contacts", {})
    c.setdefault("close_thresh_m", 0.02)   # "just outside" distance counted as contact
    c.setdefault("max_dist_m", 0.10)       # deep-occluder guard (reject spurious deep penetration)
    c.setdefault("conf_thresh", 0.1)       # trust scene points above this VGGT confidence
    c.setdefault("normal_k", 16)           # kNN for PCA scene-normal estimation
    c.setdefault("prune_margin_m", 0.10)   # per-frame scene-AABB margin for the SDF nearest-neighbour
    cfg.setdefault("runtime", {}).setdefault("device", "cuda")
    return cfg


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


def run(source, predictions_dir, config_path, *, device_override=None, overlay=True,
        bodies="smplx_kinematic") -> Path:
    cfg = _load_config(Path(config_path))
    if device_override:
        cfg["runtime"]["device"] = device_override
    device = cfg["runtime"]["device"]
    cc = cfg["contacts"]
    pred = Path(predictions_dir)
    sam3d = pred / "sam3d"

    # Which optimized body to read contact from: the kinematic stage's
    # (smplx_kinematic, default) or the annotation-guided stage 2's
    # (smplx_kinematic_2). Output tag mirrors it: cnt_kin / cnt_kin_2.
    kin_path = sam3d / f"{bodies}.npz"
    tag = "kin" + bodies.removeprefix("smplx_kinematic")
    scene_path = pred / "geometry" / "scene.npz"
    if not kin_path.exists():
        raise SystemExit(f"missing {kin_path} — run the optimize_smplx stage first")
    if not scene_path.exists():
        raise SystemExit(f"missing {scene_path} — run fuse_geometry.py first (needs vggt --save-depth)")

    kin = dict(np.load(kin_path, allow_pickle=True))
    tr = dict(np.load(pred / "geometry" / "transform.npz", allow_pickle=True))
    sc = dict(np.load(scene_path, allow_pickle=True))

    N, P = int(kin["num_frames"]), int(kin["num_people"])
    H, W = int(kin["video_height"]), int(kin["video_width"])
    object_ids = np.asarray(kin["object_ids"], dtype=np.int32)
    parents = np.asarray(kin["parents"], dtype=np.int32)
    valid_mask = np.asarray(kin["valid_mask"], dtype=bool)
    fps = float(kin["fps"])

    extr = torch.as_tensor(tr["extrinsics"], dtype=torch.float32, device=device)       # (N,4,4)
    K = torch.as_tensor(tr["intrinsics_px_orig"], dtype=torch.float32, device=device)  # (N,3,3)
    cam_centers = world_from_cam(extr)[:, :3, 3]

    scene_pts = torch.as_tensor(sc["points"], dtype=torch.float32, device=device)
    scene_conf = torch.as_tensor(sc["confidence"], dtype=torch.float32, device=device)
    scene_normals = estimate_point_normals(scene_pts, cam_centers, k=int(cc["normal_k"]))
    print(f"[estimate_contacts] scene cloud {scene_pts.shape[0]} pts, PCA normals "
          f"k={int(cc['normal_k'])}; close<{float(cc['close_thresh_m']) * 100:.0f}cm")

    mid = load_smplx_mid(_DEFAULT_SMPLX_MODEL, device)
    J = mid.num_joints
    with torch.no_grad():                    # rest-pose geometry → nearest-joint map (once)
        v_rest, j_rest = mid.forward_shape(torch.zeros(1, mid.num_betas, device=device))
    v2j = vertex_to_joint(v_rest[0], j_rest[0])   # (V,) nearest joint center per vertex

    joint_contact = np.zeros((P, N, J), dtype=bool)
    joint_sdf_cm = np.full((P, N, J), np.nan, dtype=np.float32)
    joints_2d = np.full((P, N, J, 2), np.nan, dtype=np.float32)

    for p in range(P):
        betas = torch.as_tensor(kin["betas"][p], dtype=torch.float32, device=device)
        q = torch.as_tensor(kin["q"][p], dtype=torch.float32, device=device)
        scale = float(kin["scale"][p])
        jw, vw = _pose(mid, betas, q, scale)                         # (N,J,3), (N,V,3) world

        sdf, dmin, conf = scene_sdf(vw, scene_pts, scene_normals, scene_conf,
                                    prune_margin_m=float(cc["prune_margin_m"]))
        vcon = contact_from_sdf(sdf, dmin, conf, close_thresh_m=float(cc["close_thresh_m"]),
                                max_dist_m=float(cc["max_dist_m"]), conf_thresh=float(cc["conf_thresh"]))
        jc, jsdf = lift_joint_contact(vcon, sdf, v2j, J)
        px = project(transform_points(jw, extr), K)                  # (N,J,2)

        jc_np = jc.cpu().numpy()
        jc_np[~valid_mask[p]] = False
        joint_contact[p] = jc_np
        joint_sdf_cm[p] = jsdf.cpu().numpy()
        joints_2d[p] = px.cpu().numpy()
        n_cf = int(jc_np.any(1).sum())
        print(f"[estimate_contacts] obj_id={int(object_ids[p])}: {n_cf}/{N} frames with "
              f"≥1 joint in contact ({int(jc_np.sum())} joint-frame contact instances)")

    save = dict(
        joint_contact=joint_contact, joint_sdf_cm=joint_sdf_cm,
        object_ids=object_ids, joint_names=kin["joint_names"], parents=parents,
        valid_mask=kin["valid_mask"],
        num_people=np.int32(P), num_frames=np.int32(N),
        video_height=np.int32(H), video_width=np.int32(W), fps=np.float32(fps),
        close_thresh_m=np.float32(cc["close_thresh_m"]),
        max_dist_m=np.float32(cc["max_dist_m"]),
    )
    out_path = sam3d / f"cnt_{tag}.npz"
    np.savez(out_path, **save)
    print(f"[estimate_contacts] wrote {out_path}")

    if overlay:
        from tools.smplx_optim.overlay import render_contact_overlay
        ov = render_contact_overlay(source, sam3d / f"cnt_{tag}_overlay.mp4",
                                    joints_2d, joint_contact, parents, valid_mask, H, W, fps)
        print(f"[estimate_contacts] wrote {ov}")
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--bodies", default="smplx_kinematic",
                    help="which optimized body npz under sam3d/ to read "
                         "(default smplx_kinematic; smplx_kinematic_2 for the annotation stage)")
    ap.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    ap.add_argument("--device", default=None)
    ap.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args(argv)
    run(args.source, args.predictions, args.config,
        device_override=args.device, overlay=args.overlay, bodies=args.bodies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
