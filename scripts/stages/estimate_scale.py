"""Metric scale for the fused VGGT world, anchored to the SMPL-X human body.

Runs **after** ``fuse_geometry``. VGGT-Omega geometry is up to one arbitrary
per-clip factor (depth and camera translations share it), so fuse writes an
**up-to-scale** world. The SAM-3D-Body → SMPL-X body carries real metric size
and was fit in the *same* VGGT camera, so it is a ruler. Three things happen
here, in order:

1. **Measure.** Per body-valid frame we pose the body with that frame's own
   betas at the converter's translation, keep the camera-facing vertices, and
   look up the fused human cloud (``geometry/human.npz`` — already edge-filtered
   and mask-restricted by fuse) at the pixels those vertices project to. The
   body's camera depth and the cloud's differ by exactly the unknown factor
   ``s``; we take the robust median of ``log(d_body) − log(d_cloud)`` per frame
   and pool the frames (:func:`tools.geometry.human_scale.combine_log_scales`).
   No body-part weighting — every visible vertex counts the same.

2. **Pre-fit** (the dual refinement, on by default). The ruler trusts the
   converter's body placement, but that body can sit a little off its depth
   points — and the offset leaks into ``s``. So the stage alternates: a **body
   round** (:mod:`tools.human_optim.prefit`) re-seats every person on their
   fused depth points with ``log s`` as a joint variable (root gently anchored,
   joints gently held, fingers frozen, shape frozen to one canonical betas per
   person), then the **scale move** re-reads ``s`` from the fit's own
   nearest-vertex correspondences and reconciles the people. A few rounds
   converge to a scale and bodies that agree; the refined bodies land in
   ``geometry/prefit.npz`` (viewer source ``prefit``; a future starting point
   for the kinematic stage). See docs/scale.md for the full reasoning.

3. **Metricize.** The final ``s`` is applied **in place** to the saved fused
   world: scene + human points, camera translations and the voxel size all
   multiply by ``s`` (pixel intrinsics, rotations and normals are scale-free),
   and ``transform.npz`` gets ``metric=True``. Re-running is harmless: on an
   already-metric world the measured ratio is ~1.

Per-clip I/O (``--predictions out/<stem>/``)::

    reads:  sam3d/smplx_params.npz      (per-frame betas + pose, per person)
            vggt/camera.npz             (frame_indices + VGGT-resolution intrinsics)
            geometry/transform.npz      (extrinsics in the cloud's current units)
            geometry/human.npz          (fused per-frame human points, from fuse)
            sapiens/pose.npz            (keypoints anchoring the pre-fit)
    writes: geometry/scale.npz          (scale + per-frame diagnostics)
            geometry/{transform,scene,human}.npz   (rescaled in place, metric=True)
            geometry/prefit.npz         (the aligned bodies, unless prefit=false)
            geometry/diagnostics/scale_over_time.png

Standalone CLI::

    python scripts/stages/estimate_scale.py --predictions out/sport_climb/
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
import os  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.config import load_toml, section  # noqa: E402
from tools.geometry.human_scale import combine_log_scales, weighted_median  # noqa: E402
from tools.geometry.intrinsics import denormalize_K  # noqa: E402
from tools.human_optim import normalize_q  # noqa: E402
from tools.human_optim.keypoints import build_correspondence  # noqa: E402
from tools.human_optim.losses import pad_frames  # noqa: E402
from tools.human_optim.prefit import PrefitWeights, measure_person_scale, prefit_person_round  # noqa: E402
from tools.human_optim.visibility import front_facing_mask, subsample_indices, vertex_normals  # noqa: E402
from tools.smplx_robot import load_smplx_mid  # noqa: E402

_DEFAULT_SMPLX_MODEL = (
    _REPO_ROOT.parent / "better_human" / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
)
_CLASSIC_KEYS = ("betas", "body_pose", "global_orient", "transl",
                 "left_hand_pose", "right_hand_pose")


def _frame_log_scale(
    verts_cam: np.ndarray,
    K: np.ndarray,
    depth_rel: np.ndarray,
    conf: np.ndarray,
    mask: np.ndarray,
    conf_drop_q: float,
    min_inliers: int,
) -> tuple[float, float]:
    """One frame's robust log-scale from front-facing body vertices.

    :param verts_cam: ``(Nf, 3)`` camera-frame front-facing vertices (metric m); their
        ``z`` is the body depth ``d_body``.
    :param K: ``(3, 3)`` VGGT-resolution pixel intrinsics.
    :param depth_rel: ``(Hv, Wv)`` cloud depth in the world's current units (``nan``
        where no cloud point); ``conf`` its confidence; ``mask`` which pixels hold a
        cloud point.
    :param conf_drop_q: drop the lowest ``conf_drop_q`` fraction of hit vertices by
        confidence before reducing.
    :param min_inliers: skip the frame (return ``nan``) below this many inliers.
    :returns: ``(log_scale, weight)`` — ``(nan, 0.0)`` if too few inliers; the weight is
        the summed confidence of the surviving vertices.
    """
    height, width = depth_rel.shape
    depth_body = verts_cam[:, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    safe_z = np.where(depth_body > 0.0, depth_body, 1.0)  # avoid div-by-zero on rare z<=0 verts
    px = np.rint(fx * verts_cam[:, 0] / safe_z + cx).astype(np.int64)
    py = np.rint(fy * verts_cam[:, 1] / safe_z + cy).astype(np.int64)
    inb = (depth_body > 0) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
    if not inb.any():
        return float("nan"), 0.0
    px, py, depth_body = px[inb], py[inb], depth_body[inb]
    drel = depth_rel[py, px]
    cvals = conf[py, px]
    valid = mask[py, px] & np.isfinite(drel) & (drel > 0)
    if int(valid.sum()) < min_inliers:
        return float("nan"), 0.0
    thresh = np.quantile(cvals[valid], conf_drop_q) if conf_drop_q > 0.0 else -np.inf
    keep = valid & (cvals >= thresh)
    if int(keep.sum()) < min_inliers:
        return float("nan"), 0.0
    log_ratio = np.log(depth_body[keep]) - np.log(drel[keep])
    w = cvals[keep]
    return float(weighted_median(log_ratio, w)), float(w.sum())


def _plot(
    path: Path,
    per_frame_scale: np.ndarray,
    per_frame_weight: np.ndarray,
    frame_indices: np.ndarray,
    object_ids: np.ndarray,
    scale_global: float,
) -> None:
    """Scatter per-frame scale (per person) + weight over time, with the global s."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_people = per_frame_scale.shape[0]
    fig, (ax_scale, ax_weight) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for p in range(num_people):
        ok = np.isfinite(per_frame_scale[p])
        if not ok.any():
            continue
        ax_scale.scatter(frame_indices[ok], per_frame_scale[p][ok], s=12, alpha=0.6,
                         label=f"person {int(object_ids[p])}")
        ax_weight.plot(frame_indices[ok], per_frame_weight[p][ok], lw=1.0,
                       label=f"person {int(object_ids[p])}")
    ax_scale.axhline(scale_global, color="k", ls="--", lw=1.2, label=f"global s={scale_global:.3f}")
    ax_scale.set_ylabel("per-frame scale")
    ax_scale.set_title("Human-anchored metric scale over time")
    ax_scale.legend(fontsize=8, ncol=2)
    ax_scale.grid(alpha=0.3)
    ax_weight.set_ylabel("frame weight (Σ conf)")
    ax_weight.set_xlabel("frame index")
    ax_weight.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _group_cloud(human: dict, n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort the fused human cloud by ``(person, frame)`` for contiguous-slice lookup.

    Returns ``(points, confidence, key_sorted, order)``; the slice for
    ``(oid, col)`` is ``order[searchsorted(key_sorted, oid * n_frames + col) :
    searchsorted(key_sorted, oid * n_frames + col + 1)]``.
    """
    pts = np.asarray(human["points"], dtype=np.float32)
    conf = np.asarray(human["confidence"], dtype=np.float32)
    key = np.asarray(human["obj_id"], np.int64) * n_frames + np.asarray(human["frame_id"], np.int64)
    order = np.argsort(key)
    return pts, conf, key[order], order


def _valid_cols(sp: dict, p: int, frame_indices: np.ndarray) -> np.ndarray:
    """VGGT-frame columns where person ``p``'s converter body is valid."""
    vmask = np.asarray(sp["valid_mask"][p], dtype=bool)
    return np.asarray([j for j, idx in enumerate(frame_indices) if vmask[int(idx)]], dtype=int)


def _estimate_metric_scale(
    sp: dict, frame_indices: np.ndarray, K_vggt: np.ndarray, hv: int, wv: int,
    object_ids: np.ndarray, extr: np.ndarray, human: dict, mid, device: str,
    args: argparse.Namespace,
) -> tuple[float, dict]:
    """Robust human-anchored scale from the fused human cloud + per-frame diagnostics.

    ``extr`` must be in the same units as ``human['points']`` (both come from
    ``geometry/`` as fuse — or a previous metricization — left them).
    """
    num_people = len(object_ids)
    n_frames = len(frame_indices)
    per_frame_scale = np.full((num_people, n_frames), np.nan, dtype=np.float32)
    per_frame_weight = np.zeros((num_people, n_frames), dtype=np.float32)
    pooled_log: list[float] = []
    pooled_weight: list[float] = []

    pts, conf, key_sorted, order = _group_cloud(human, n_frames)

    for p in range(num_people):
        oid = int(object_ids[p])
        cols = _valid_cols(sp, p, frame_indices)
        if cols.size == 0:
            print(f"  person obj_id={oid}: no body-valid VGGT frames — skipping")
            continue
        idxs = frame_indices[cols]

        def gather(k: str) -> torch.Tensor:
            return torch.as_tensor(sp[k][p][idxs], dtype=torch.float32, device=device)

        with torch.no_grad():
            betas = gather("betas")
            q_cam = mid.from_classic(betas, gather("body_pose"), gather("transl"),
                                     gather("global_orient"), gather("left_hand_pose"),
                                     gather("right_hand_pose"))
            verts = mid.forward(betas, normalize_q(q_cam, mid.num_joints)).vertices
            front = front_facing_mask(verts, vertex_normals(verts, mid.faces))
        verts_np = verts.cpu().numpy()
        front_np = front.cpu().numpy()

        for fi, col in enumerate(cols):
            col = int(col)
            lo = np.searchsorted(key_sorted, oid * n_frames + col)
            hi = np.searchsorted(key_sorted, oid * n_frames + col + 1)
            if hi - lo < args.min_inlier_verts:
                continue
            sel = order[lo:hi]
            # Cloud points back to this frame's camera; projecting with the same K
            # recovers exactly the pixels fuse backprojected them from.
            E = extr[col]
            Xc = pts[sel] @ E[:3, :3].T + E[:3, 3]
            z = Xc[:, 2]
            K = K_vggt[col]
            u = np.rint(K[0, 0] * Xc[:, 0] / z + K[0, 2]).astype(np.int64)
            v = np.rint(K[1, 1] * Xc[:, 1] / z + K[1, 2]).astype(np.int64)
            ok = (z > 0) & (u >= 0) & (u < wv) & (v >= 0) & (v < hv)
            dmap = np.full((hv, wv), np.nan, np.float32)
            cmap = np.zeros((hv, wv), np.float32)
            dmap[v[ok], u[ok]] = z[ok]
            cmap[v[ok], u[ok]] = conf[sel][ok]
            log_s, weight = _frame_log_scale(
                verts_np[fi][front_np[fi]], K, dmap, cmap, np.isfinite(dmap),
                args.conf_drop_q, args.min_inlier_verts)
            if not np.isfinite(log_s):
                continue
            per_frame_scale[p, col] = float(np.exp(log_s))
            per_frame_weight[p, col] = weight
            pooled_log.append(log_s)
            pooled_weight.append(weight)

    if not pooled_log:
        raise SystemExit("[scale] no usable body frames to anchor scale — is geometry/human.npz empty?")

    res = combine_log_scales(np.asarray(pooled_log), np.asarray(pooled_weight), mad_k=args.mad_k)
    diagnostics = dict(res, per_frame_scale=per_frame_scale, per_frame_weight=per_frame_weight,
                       n_pooled=len(pooled_log))
    return res["scale"], diagnostics


def _replace_npz(path: Path, arrays: dict) -> None:
    """Write ``arrays`` to a temp file and atomically rename it over ``path``."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _metricize_geometry(pred: Path, s: float, method: str = "human-log-ratio") -> float:
    """Bring the saved fused world to metric in place; returns the cumulative scale.

    Everything with length units scales: camera translations, scene + human
    points, the voxel size. Rotations and pixel intrinsics are scale-free.
    ``metric=True`` marks the world as done for the downstream stages.

    Each file carries its own ``world_scale``, and its multiplier is computed
    from that — so if a previous run died mid-rescale, the lagging file is
    simply brought up to the target instead of everything double-scaling.
    The clouds are written first and ``transform.npz`` (which holds the
    ``metric`` flag) last, each via an atomic temp-file rename.
    """
    geo = pred / "geometry"
    tr = dict(np.load(geo / "transform.npz", allow_pickle=True))
    tr_ws = float(tr["scale"])
    total = tr_ws * s
    for name, keys in (("scene.npz", ("points", "voxel_size")), ("human.npz", ("points",))):
        d = dict(np.load(geo / name, allow_pickle=True))
        m = total / float(d.get("world_scale", tr_ws))
        for k in keys:
            d[k] = (np.asarray(d[k], np.float32) * m).astype(np.float32)
        d["world_scale"] = np.float32(total)
        _replace_npz(geo / name, d)
    extr = np.asarray(tr["extrinsics"], np.float32).copy()
    extr[:, :3, 3] *= s
    tr.update(extrinsics=extr, scale=np.float32(total), metric=np.bool_(True),
              scale_method=np.str_(f"{method} (estimate_scale)"))
    _replace_npz(geo / "transform.npz", tr)
    return total


def _prefit_people_data(
    sp: dict, pose: dict, frame_indices: np.ndarray, K_orig: np.ndarray,
    object_ids: np.ndarray, extr: np.ndarray, human: dict, mid, device: str,
    args: argparse.Namespace,
) -> list:
    """Per-person tensors for the pre-fit rounds (``None`` for bodiless people).

    The per-frame depth cloud comes from the fused human points, moved into each
    frame's camera (units follow ``transform.npz``, so the body round multiplies
    by the current ``s``), gated by confidence quantile and capped at
    ``max_points`` by uniform stride. Shape is frozen to one canonical betas per
    person — the per-coefficient median over the valid frames (converter betas
    jitter frame to frame, and a scale problem must not pick its own body size).
    """
    n_frames = len(frame_indices)
    pts, conf, key_sorted, order = _group_cloud(human, n_frames)
    # Sapiens tracks its own person set — rows are aligned by object_id, NOT by
    # the SMPL-X row index (e.g. smplx may hold person [1] of sapiens' [0,1,2]).
    sap_rows = {int(x): i for i, x in enumerate(np.asarray(pose["object_ids"]).ravel())}
    people: list = []
    for p in range(len(object_ids)):
        oid = int(object_ids[p])
        cols = _valid_cols(sp, p, frame_indices)
        if cols.size == 0:
            people.append(None)
            continue
        if oid not in sap_rows:
            print(f"  [prefit] person obj_id={oid}: no Sapiens keypoints — left at the converter pose")
            people.append(None)
            continue
        sap = sap_rows[oid]
        idxs = frame_indices[cols]

        def gather(k: str) -> torch.Tensor:
            return torch.as_tensor(sp[k][p][idxs], dtype=torch.float32, device=device)

        betas_c = np.median(np.asarray(sp["betas"][p][idxs], np.float32), axis=0)
        betas = torch.as_tensor(np.tile(betas_c, (cols.size, 1)), device=device)
        with torch.no_grad():
            q0 = mid.from_classic(betas, gather("body_pose"), gather("transl"),
                                  gather("global_orient"), gather("left_hand_pose"),
                                  gather("right_hand_pose"))
            q0n = normalize_q(q0, mid.num_joints)

        clouds = []
        for col in cols:
            col = int(col)
            lo = np.searchsorted(key_sorted, oid * n_frames + col)
            hi = np.searchsorted(key_sorted, oid * n_frames + col + 1)
            sel = order[lo:hi]
            E = extr[col]
            Xc = pts[sel] @ E[:3, :3].T + E[:3, 3]
            c = conf[sel]
            if args.conf_drop_q > 0.0 and c.size:
                Xc = Xc[c >= np.quantile(c, args.conf_drop_q)]
            if Xc.shape[0] > args.max_points:
                stride = Xc.shape[0] / float(args.max_points)
                Xc = Xc[(np.arange(args.max_points) * stride).astype(np.int64)]
            clouds.append(torch.as_tensor(Xc, dtype=torch.float32, device=device))
        cloud, cloud_valid = pad_frames(clouds, channels=3, device=device, dtype=torch.float32)

        pairs = torch.as_tensor(np.flatnonzero(np.diff(cols) == 1), dtype=torch.long, device=device)
        people.append(dict(
            oid=oid, cols=cols, idxs=idxs, betas=betas, q0n=q0n, q=q0n.clone(),
            cloud=cloud, cloud_valid=cloud_valid,
            K_orig=torch.as_tensor(K_orig[cols], dtype=torch.float32, device=device),
            kp=torch.as_tensor(pose["keypoints"][sap][idxs], dtype=torch.float32, device=device),
            scores=torch.as_tensor(pose["keypoint_scores"][sap][idxs], dtype=torch.float32,
                                   device=device),
            pairs=pairs))
    return people


def _prefit_bodies(
    sp: dict, pose: dict, s0: float, frame_indices: np.ndarray,
    K_orig: np.ndarray, object_ids: np.ndarray, extr: np.ndarray,
    human: dict, mid, device: str, args: argparse.Namespace,
) -> tuple[float, dict, list]:
    """The dual refinement: body rounds at fixed ``s`` ↔ scale re-measurements.

    Starting from the converter pose and the ruler's ``s0``, each round aligns
    every person to their depth cloud scaled by the current ``s`` (the body
    move), then re-measures ``s`` from the fit's own nearest-vertex
    correspondences (the scale move, :func:`measure_person_scale`). Stops early
    once ``s`` settles within 0.1%. Returns
    ``(s_final, diagnostics of the last measurement, people)``.
    """
    corr = build_correspondence(list(mid.frame_names), device=device)
    fvids = torch.as_tensor([int(x) for x in mid.frame_vertex_ids], dtype=torch.long, device=device)
    sub = subsample_indices(mid.num_vertices, args.vertex_subsample, device=device)
    weights = PrefitWeights(
        chamfer=args.w_chamfer, reproj=args.w_reproj, pose=args.w_pose,
        orient=args.w_orient, transl=args.w_transl, smooth=args.w_smooth,
        gm_chamfer_m=args.gm_chamfer_m, gm_reproj_px=args.gm_reproj_px)
    people = _prefit_people_data(sp, pose, frame_indices, K_orig, object_ids, extr,
                                 human, mid, device, args)

    n_frames = len(frame_indices)
    per_frame_scale = np.full((len(object_ids), n_frames), np.nan, dtype=np.float32)
    per_frame_weight = np.zeros((len(object_ids), n_frames), dtype=np.float32)
    s, diag = s0, None
    for r in range(args.rounds):
        for person in people:
            if person is None:
                continue
            print(f"  [prefit] round {r + 1}/{args.rounds} person obj_id={person['oid']} "
                  f"({person['cols'].size} frames, s={s:.4f})")
            # Round 1 fits body + scale jointly (finds each person's depth); later
            # rounds pin the scale to the pooled consensus and only align bodies.
            person["q"], person["s_fit"] = prefit_person_round(
                mid, person["betas"], person["q"], person["q0n"], s0=s,
                cloud=person["cloud"], cloud_valid=person["cloud_valid"],
                K_orig=person["K_orig"], target_kp=person["kp"], scores=person["scores"],
                corr=corr, frame_vertex_ids=fvids, sub=sub, pairs=person["pairs"],
                weights=weights, iters=args.iters, lr=args.lr, freeze_scale=r > 0)
        pooled_log, pooled_w = [], []
        for p, person in enumerate(people):
            if person is None:
                continue
            logs, ws = measure_person_scale(
                mid, person["betas"], person["q"], cloud=person["cloud"],
                cloud_valid=person["cloud_valid"], sub=sub, s=person["s_fit"],
                gm_scale_m=args.gm_chamfer_m)
            per_frame_scale[p, person["cols"]] = np.exp(logs).astype(np.float32)
            per_frame_weight[p, person["cols"]] = ws.astype(np.float32)
            pooled_log.append(logs)
            pooled_w.append(ws)
        res = combine_log_scales(np.concatenate(pooled_log), np.concatenate(pooled_w),
                                 mad_k=args.mad_k)
        s_new = res["scale"]
        diag = dict(res, per_frame_scale=per_frame_scale, per_frame_weight=per_frame_weight,
                    n_pooled=int(np.isfinite(np.concatenate(pooled_log)).sum()))
        print(f"  [prefit] round {r + 1}: s {s:.4f} -> {s_new:.4f}")
        settled = abs(float(np.log(s_new / s))) < 1e-3
        s = s_new
        if settled:
            break
    return s, diag, people


def _save_prefit(pred: Path, sp: dict, object_ids: np.ndarray, people: list,
                 mid, scale_total: float) -> None:
    """Write the aligned bodies to ``geometry/prefit.npz`` (classic params, camera frame)."""
    out = {key: np.asarray(sp[key], dtype=np.float32).copy() for key in _CLASSIC_KEYS}
    for p, person in enumerate(people):
        if person is None:
            continue
        classic = mid.to_classic(person["betas"], person["q"])
        for key in _CLASSIC_KEYS:
            src = classic["translation"] if key == "transl" else classic[key]
            out[key][p, person["idxs"]] = src.cpu().numpy().astype(np.float32)
    out_path = pred / "geometry" / "prefit.npz"
    np.savez(
        out_path,
        valid_mask=np.asarray(sp["valid_mask"], dtype=bool),
        object_ids=np.asarray(object_ids, dtype=np.int32),
        num_people=np.int32(len(object_ids)),
        num_frames=np.int32(sp["num_frames"]),
        video_height=np.int32(sp["video_height"]),
        video_width=np.int32(sp["video_width"]),
        scale=np.float32(scale_total),
        **out,
    )
    print(f"  prefit bodies -> {out_path}")


def estimate_scale(args: argparse.Namespace) -> None:
    pred = Path(args.predictions)
    out_dir = pred / "geometry"
    scale_path = out_dir / "scale.npz"
    tr_path = out_dir / "transform.npz"
    if not tr_path.exists():
        raise SystemExit(f"[scale] missing {tr_path} — run fuse_geometry first")
    # The pre-fit needs Sapiens keypoints as its image anchor; without them the
    # depth direction is unconstrained, so fall back to the ruler-only scale —
    # loudly, and recorded in the method fields.
    prefit = bool(args.prefit)
    if prefit and not (pred / "sapiens" / "pose.npz").exists():
        print(f"  [scale] WARNING: no sapiens/pose.npz under {pred} — "
              "pre-fit disabled, ruler-only scale")
        prefit = False
    method = "prefit-dual" if prefit else "human-log-ratio"

    tr = dict(np.load(tr_path, allow_pickle=True))
    already_metric = bool(tr.get("metric", False))
    done = scale_path.exists() and already_metric
    if done and prefit:  # the requested pre-fit must actually have run
        with np.load(scale_path) as sc:
            done = ("method" in sc.files and str(sc["method"]) == "prefit-dual"
                    and (out_dir / "prefit.npz").exists())
    if args.skip_existing and done:
        print(f"  skip (exists): {scale_path}")
        return

    sp = dict(np.load(pred / "sam3d" / "smplx_params.npz", allow_pickle=True))
    cam = np.load(pred / "vggt" / "camera.npz")
    frame_indices = cam["frame_indices"].astype(int)
    K_vggt = np.asarray(cam["intrinsics_px"], dtype=np.float32)          # (F, 3, 3) @ vggt res
    hv, wv = int(cam["vggt_height"]), int(cam["vggt_width"])
    H, W = int(sp["video_height"]), int(sp["video_width"])
    K_orig = denormalize_K(np.asarray(cam["intrinsics"], dtype=np.float32), H, W)  # (F, 3, 3) @ orig res
    object_ids = np.asarray(sp["object_ids"], dtype=np.int32)
    device = args.device
    # Extrinsics from transform.npz, NOT vggt/camera.npz: they are in the same
    # units as the human cloud, so a re-run on a metric world measures s ≈ 1.
    extr = np.asarray(tr["extrinsics"], dtype=np.float32)
    human = dict(np.load(out_dir / "human.npz", allow_pickle=True))
    if np.asarray(human["points"]).size == 0:
        raise SystemExit(f"[scale] geometry/human.npz has no points under {pred}")
    # If a previous run died mid-rescale, the cloud may lag the extrinsics'
    # units; bring it in line before measuring, or the ratio would be biased.
    hu_ws = float(human.get("world_scale", tr["scale"]))
    if not np.isclose(hu_ws, float(tr["scale"])):
        print(f"  [scale] note: repairing partial rescale (human world_scale {hu_ws:.4f} "
              f"vs transform {float(tr['scale']):.4f})")
        human["points"] = np.asarray(human["points"], np.float32) * (float(tr["scale"]) / hu_ws)

    mid = load_smplx_mid(str(_DEFAULT_SMPLX_MODEL), device)

    # The ruler on the converter bodies gives the starting scale …
    scale_ruler, diag = _estimate_metric_scale(sp, frame_indices, K_vggt, hv, wv,
                                               object_ids, extr, human, mid, device, args)
    scale, people = scale_ruler, None
    if prefit:
        # … and the dual refinement takes it from there (docs/scale.md).
        pose = dict(np.load(pred / "sapiens" / "pose.npz", allow_pickle=True))
        scale, diag, people = _prefit_bodies(sp, pose, scale_ruler, frame_indices,
                                             K_orig, object_ids, extr, human,
                                             mid, device, args)
    s_total = _metricize_geometry(pred, scale, method)
    np.savez(
        scale_path,
        scale=np.float32(scale),
        scale_ruler=np.float32(scale_ruler),
        scale_total=np.float32(s_total),
        log_scale=np.float32(diag["log_scale"]),
        per_frame_scale=diag["per_frame_scale"],
        per_frame_weight=diag["per_frame_weight"],
        frame_indices=frame_indices.astype(np.int32),
        object_ids=object_ids,
        scale_mad=np.float32(diag["mad"]),
        n_frames_used=np.int32(diag["n_used"]),
        n_people=np.int32(len(object_ids)),
        method=np.str_(method),
    )
    _plot(out_dir / "diagnostics" / "scale_over_time.png",
          diag["per_frame_scale"], diag["per_frame_weight"], frame_indices, object_ids, scale)
    print(f"  scale={scale:.4f} (ruler {scale_ruler:.4f}, mad {diag['mad']:.4f}, "
          f"{diag['n_used']}/{diag['n_pooled']} frames) — world now metric "
          f"(total {s_total:.4f}) -> {scale_path}")
    if people is not None:
        _save_prefit(pred, sp, object_ids, people, mid, s_total)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--predictions", required=True, type=Path, help="clip dir, e.g. out/<stem>/")
    p.add_argument("--config", type=Path, default=_REPO_ROOT / "configs" / "pipeline.toml",
                   help="pipeline TOML (default: configs/pipeline.toml)")
    cli = p.parse_args(argv)
    cfg = load_toml(cli.config)
    args = section(cfg, "scale", {
        "conf_drop_q": 0.3,
        "min_inlier_verts": 50,
        "mad_k": 3.0,
        "skip_existing": False,
        # pre-fit: the dual scale + body refinement (prefit=false → ruler-only scale)
        "prefit": True,
        "rounds": 3,
        "iters": 100,
        "lr": 0.02,
        "w_chamfer": 1.0,
        "w_reproj": 0.5,
        "w_pose": 2.0,
        "w_orient": 1.0,
        "w_transl": 0.5,
        "w_smooth": 0.1,
        "gm_chamfer_m": 0.3,
        "gm_reproj_px": 10.0,
        "max_points": 2000,
        "vertex_subsample": 3000,
    })
    args.device = section(cfg, "runtime", {"device": "cuda"}).device
    args.predictions, args.config = cli.predictions, cli.config
    return args


def main(argv: list[str] | None = None) -> int:
    estimate_scale(_parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
