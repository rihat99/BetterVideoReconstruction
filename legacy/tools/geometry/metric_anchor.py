"""Body-anchored metric-scale correction for fused geometry.

``fuse_geometry`` anchors VGGT's up-to-scale geometry to **MoGe2's** monocular
metric depth. On distant / low-texture climbing clips MoGe2 *underestimates*
metric depth (2-5x), so the fused human + scene clouds land too close — e.g. a
19 cm "climber" — and the visually-guided ``optimize_smplx`` then depth-wrenches
(chamfer drags the body off-image, scale runs away). The fitted SMPL-X body is a
**reliable metric anchor**: it carries real human size and reprojects at ~7 px,
and its depth agrees with the FOV+stature physical estimate. So we re-anchor:

    c = median_t( body_front_surface_depth_t / human_cloud_depth_t )

over valid frames, then rescale the fused geometry by ``c`` — scene points, human
points, camera translations and the stored ``scale`` all multiply by ``c``. This
is a pure global similarity (a uniform world scaling matched by the camera
translations), so **every projection is preserved**; only the absolute metric
scale changes. ``c≈1`` when MoGe2 was already right (the body just confirms it);
``c≫1`` rescues the clips MoGe2 got wrong.

The correction is **idempotent** — it stamps ``body_anchor_c`` into
``transform.npz`` and refuses to double-apply — and **reversible** (the original
``geometry/`` is copied to ``geometry.bak`` on first apply).

This is a standalone helper, not (yet) a pipeline stage: run it after
``convert_mhr_to_smplx`` and before ``optimize_smplx`` on clips whose metric
scale is off::

    python tools/geometry/metric_anchor.py --predictions out/<clip>/        # auto obj-id
    python tools/geometry/metric_anchor.py --predictions out/<clip>/ --obj-id 1
    python tools/geometry/metric_anchor.py --predictions out/<clip>/ --revert
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch


def body_anchor_scale(pred: Path, oid: int, mid, device: str = "cpu") -> tuple[float, float, int]:
    """``c = median(body front-surface depth / human-cloud depth)`` over valid frames.

    Both depths are the median camera-z of the camera-FACING surface: the cloud is
    the back-projected metric depth map (a front-only point set), so the body must
    match it with its **front-facing** vertices (back-face cull) rather than the
    full mesh (which samples the back surface too). NB on spread climbing poses the
    two body medians nearly coincide (~1 mm — the depth distribution is near-symmetric
    about its median), so this is a correctness/consistency fix that only bites for
    frontal / close-up poses where the body's depth extent is a large fraction of
    its distance. ``mid`` is a loaded ``SMPLXMid``
    (``tools.smplx_robot.load_smplx_mid``); the body comes from
    ``sam3d/smplx_params.npz`` (camera frame) and the cloud from
    ``geometry/human.npz`` (world) projected back through the extrinsics. Returns
    ``(c, std, n_frames)``.
    """
    from tools.smplx_optim import normalize_q
    from tools.smplx_optim.visibility import front_facing_mask, vertex_normals

    sp = dict(np.load(pred / "sam3d" / "smplx_params.npz", allow_pickle=True))
    tr = dict(np.load(pred / "geometry" / "transform.npz", allow_pickle=True))
    hu = dict(np.load(pred / "geometry" / "human.npz", allow_pickle=True))
    p = [int(x) for x in sp["object_ids"]].index(oid)
    N = int(sp["num_frames"])
    extr = torch.as_tensor(tr["extrinsics"], dtype=torch.float32, device=device)
    vmask = sp["valid_mask"][p]
    betas = torch.as_tensor(sp["betas"][p][vmask].mean(0), dtype=torch.float32, device=device)
    bt = betas[None].expand(N, -1)

    def t(k):
        return torch.as_tensor(sp[k][p], dtype=torch.float32, device=device)

    q_cam = mid.from_classic(bt, t("body_pose"), t("transl"), t("global_orient"),
                             t("left_hand_pose"), t("right_hand_pose"))
    vcam = mid.forward(bt, normalize_q(q_cam, mid.num_joints)).vertices  # (N, V, 3) camera frame
    # The depth cloud only captures the camera-FACING front surface, so the body
    # depth must too: per frame, the median camera-z over front-facing verts (the
    # back-face cull chamfer uses), not over all verts (which also sample the back
    # surface). vcam is camera-frame already, so its normals are too and the cull
    # is a direct dot(normal, position) < 0.
    front = front_facing_mask(vcam, vertex_normals(vcam, mid.faces))  # (N, V) bool
    zc = vcam[..., 2]                                                 # (N, V) camera depth
    H, F, M = hu["points"], hu["frame_id"], hu["obj_id"] == oid
    H, F = H[M], F[M]
    ratios = []
    for k in range(N):
        if not vmask[k]:
            continue
        q = H[F == k]
        if len(q) < 50:
            continue
        Pw = torch.as_tensor(q, dtype=torch.float32, device=device)
        E = extr[k]
        Pc = (E[:3, :3] @ Pw.T + E[:3, 3:4]).T          # world -> camera
        ratios.append((zc[k][front[k]].median() / Pc[:, 2].median()).item())
    if not ratios:
        raise RuntimeError(f"no valid frames to anchor scale for {pred} (obj {oid})")
    r = np.asarray(ratios)
    return float(np.median(r)), float(np.std(r)), len(r)


def _default_obj(pred: Path) -> int:
    """The fitted person's obj-id (first row of the single-person smplx stack)."""
    sp = np.load(pred / "sam3d" / "smplx_params.npz", allow_pickle=True)
    return int(np.asarray(sp["object_ids"])[0])


def apply_correction(pred: Path, oid: int | None = None, *, device: str = "cpu") -> float:
    """Rescale ``geometry/`` by the body-anchored factor ``c`` (idempotent, reversible)."""
    import sys
    bh = Path(__file__).resolve().parents[2].parent / "better_human" / "src"
    if bh.exists() and str(bh) not in sys.path:
        sys.path.insert(0, str(bh))
    from tools.smplx_robot import load_smplx_mid

    pred = Path(pred)
    geo = pred / "geometry"
    tr = dict(np.load(geo / "transform.npz", allow_pickle=True))
    if "body_anchor_c" in tr:
        print(f"  {pred.name}: already corrected (c={float(tr['body_anchor_c']):.3f}) — skipping")
        return float(tr["body_anchor_c"])
    if oid is None:
        oid = _default_obj(pred)
    smplx_model = (Path(__file__).resolve().parents[2].parent / "better_human"
                   / "models" / "smplx" / "SMPLX_NEUTRAL.npz")
    mid = load_smplx_mid(str(smplx_model), device)
    c, sd, n = body_anchor_scale(pred, oid, mid, device=device)

    bak = pred / "geometry.bak"
    if not bak.exists():
        shutil.copytree(geo, bak)
    tr["extrinsics"] = tr["extrinsics"].copy()
    tr["extrinsics"][:, :3, 3] *= c
    old_scale = float(tr["scale"])
    tr["scale"] = np.float32(old_scale * c)
    tr["body_anchor_c"] = np.float32(c)
    tr["scale_premetric"] = np.float32(old_scale)
    np.savez(geo / "transform.npz", **tr)
    for fn in ("scene.npz", "human.npz"):
        d = dict(np.load(geo / fn, allow_pickle=True))
        if d["points"].shape[0]:
            d["points"] = d["points"].astype(np.float32) * c
        np.savez(geo / fn, **d)
    print(f"  {pred.name}(obj{oid}): c={c:.3f} (±{sd:.3f}, n={n})  "
          f"scale {old_scale:.4f} -> {old_scale * c:.4f}  [backup: geometry.bak]")
    return c


def revert(pred: Path) -> None:
    """Restore ``geometry/`` from ``geometry.bak`` (undo a correction)."""
    pred = Path(pred)
    bak = pred / "geometry.bak"
    if not bak.exists():
        print(f"  {pred.name}: no geometry.bak — nothing to revert")
        return
    shutil.rmtree(pred / "geometry")
    shutil.copytree(bak, pred / "geometry")
    print(f"  {pred.name}: reverted geometry/ from geometry.bak")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions", required=True, type=Path, help="clip dir, e.g. out/<stem>/")
    ap.add_argument("--obj-id", type=int, default=None,
                    help="person to anchor to (default: the fitted person)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--revert", action="store_true", help="restore geometry/ from geometry.bak")
    args = ap.parse_args(argv)
    if args.revert:
        revert(args.predictions)
    else:
        apply_correction(args.predictions, args.obj_id, device=args.device)
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    raise SystemExit(main())
