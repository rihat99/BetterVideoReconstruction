"""Per-frame diagnostics for the SMPL-X kinematic stage (``human_optimization`` kinematics_1).

Plots every loss term **as a function of frame index**, for the converter init
vs the refined trajectory, alongside the *evidence* each term depends on
(matched-keypoint availability + score, depth-point count). Use it to spot:

  * sudden keypoint dropouts (panel 2) that unconstrain reprojection → pose pops,
  * bad / sparse depth frames (panels 3–4) that let chamfer wrench the pose,
  * where the optimizer actually moved the pose vs init (panel 5),
  * single-frame jumps in the motion (panels 6–8: root / orient / joint step),
  * which frames the velocity-gated contact loss repels (panel 9: # contacting
    verts + their mean joint speed — fast contacts are what it pushes off).

It reuses the exact correspondence + projection + visibility code the optimizer
uses, so the numbers match the stage (no re-derivation, no fresh mismatch).

    .venv/bin/python scripts/diagnose_human_optim.py --predictions out/sport_climb/

Writes ``out/<stem>/human_optim/diagnostics/loss_person<oid>.png`` (one per person) and
prints a short table of the worst frames per term.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BETTER_HUMAN_SRC = _REPO_ROOT.parent / "better_human" / "src"
for _p in (str(_BETTER_HUMAN_SRC), str(_REPO_ROOT)):
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402
import tomllib  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.human_optim import (  # noqa: E402
    build_correspondence, fold_root_to_world, joint_local_linear_speed, project,
    subsample_indices, transform_points, vertex_to_joint, world_from_cam,
)
from tools.human_optim import paths  # noqa: E402
from tools.human_optim.qmath import normalize_q  # noqa: E402
from tools.human_optim.visibility import front_facing_mask, vertex_normals  # noqa: E402
from tools.smplx_robot import load_smplx_mid, make_smplx_robot_model  # noqa: E402
from tools.geometry.normals import estimate_point_normals  # noqa: E402

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "human_optimization.toml"
_DEFAULT_SMPLX_MODEL = (
    _REPO_ROOT.parent / "better_human" / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
)


# ─────────────────────────────── io helpers ──────────────────────────────

def _person_depth_points(human, oid, N, conf_thresh, max_points, device):
    """Per-frame world depth points for one person (conf-gated, capped) — as the stage does."""
    mask = (human["obj_id"] == oid) & (human["confidence"] > conf_thresh)
    pts, fr = human["points"][mask], human["frame_id"][mask]
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


def _sapiens_for(pose, oid):
    ids = [int(x) for x in pose["object_ids"]]
    p = ids.index(oid)
    return pose["keypoints"][p], pose["keypoint_scores"][p]


# ─────────────────────────── per-frame metrics ───────────────────────────

def _quat_geodesic_deg(qa, qb):
    """Per-element geodesic angle (deg) between unit quats ``(...,4)``."""
    cos = (qa * qb).sum(-1).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.acos(cos))


@torch.no_grad()
def per_frame_metrics(mid, betas, q_world, scale, *, extr, K, depth, target_kp,
                      scores, corr, fvids, sub, score_thresh,
                      scene_pts=None, scene_normals=None, scene_conf=None, pen=None,
                      model=None, fps=30.0, v2j_sub=None, contact=None):
    """Per-frame reprojection (px), chamfer (cm), and joint-step (deg) for one trajectory."""
    T = q_world.shape[0]
    qn = normalize_q(q_world, mid.num_joints)
    out = mid.forward(betas, qn)
    jw = out.joints_world.tensor()[..., :3]
    vw = out.vertices
    root = jw[:, :1, :]
    jw = root + scale * (jw - root)
    vw = root + scale * (vw - root)

    # reprojection: per-frame score-weighted px error + per-keypoint err
    body = jw[:, corr.joint_model, :]
    lm = vw[:, fvids[corr.lm_frame], :]
    pred_px = project(transform_points(torch.cat([body, lm], 1), extr), K)   # (T,K,2)
    tgt = target_kp[:, corr.sap, :]
    err = (pred_px - tgt).norm(dim=-1)                                       # (T,K)
    w = (scores[:, corr.sap] >= score_thresh).float()                       # (T,K)
    reproj = ((w * err).sum(1) / w.sum(1).clamp_min(1.0)).cpu().numpy()      # (T,)
    n_valid = w.sum(1).cpu().numpy()                                         # (T,)
    matched_score = scores[:, corr.sap].mean(1).cpu().numpy()               # (T,)

    # chamfer: per frame, on visible subsampled verts (same culling as the loss)
    n = vertex_normals(vw, mid.faces)
    vcam = transform_points(vw[:, sub], extr)
    ncam = torch.einsum("tij,tvj->tvi", extr[:, :3, :3], n[:, sub])
    vismask = front_facing_mask(vcam, ncam)
    cham = np.full(T, np.nan, np.float32)
    for t, D in enumerate(depth):
        if D.shape[0] == 0:
            continue
        Vv = vw[t, sub][vismask[t]]
        if Vv.shape[0] == 0:
            continue
        d = torch.cdist(Vv, D)
        cham[t] = float(d.min(1).values.mean() + d.min(0).values.mean()) * 100.0  # cm

    # penetration into the scene cloud, per frame (cm) — mirrors scene_penetration —
    # plus the velocity-gated contact diagnostic (# gated contacting verts + their
    # mean joint speed), mirroring contact_velocity_loss on the same per-frame SDF.
    pen_cm = np.full(T, np.nan, np.float32)
    pen_cnt = np.zeros(T, np.float32)
    ct_cnt = np.zeros(T, np.float32)
    ct_speed = np.full(T, np.nan, np.float32)
    have_ct = scene_pts is not None and model is not None and contact is not None
    vel = (joint_local_linear_speed(model, mid, q_world, fps)[:, v2j_sub]
           if have_ct else None)                      # (T, Ns) per-subsampled-vertex joint speed
    if scene_pts is not None:
        band = float(contact["contact_band_m"]) if have_ct else 0.0
        mg = float(pen["prune_margin_m"])
        for tt in range(T):
            V = vw[tt, sub]
            lo, hi = V.amin(0) - mg, V.amax(0) + mg
            m = ((scene_pts >= lo) & (scene_pts <= hi)).all(-1)
            P = scene_pts[m]
            if P.shape[0] == 0:
                pen_cm[tt] = 0.0
                continue
            dmin, idx = torch.cdist(V, P).min(1)
            s = ((V - P[idx]) * scene_normals[m][idx]).sum(-1)
            c = scene_conf[m][idx]
            active = ((c > float(pen["conf_thresh"])) & (dmin < float(pen["max_dist_m"]))
                      & (s > -float(pen["max_pen_m"])))
            pm = active & (s < 0)
            pen_cnt[tt] = float(pm.sum())
            pen_cm[tt] = float((-s[pm]).mean() * 100.0) if bool(pm.any()) else 0.0
            if have_ct:
                cm = active & (s < band)               # gated contacting verts (inside or within band)
                ct_cnt[tt] = float(cm.sum())
                if bool(cm.any()):
                    ct_speed[tt] = float(vel[tt][cm].mean())   # mean joint speed at the contacts

    return dict(reproj=reproj, n_valid=n_valid,
                matched_score=matched_score, cham=cham, pen_cm=pen_cm, pen_cnt=pen_cnt,
                ct_cnt=ct_cnt, ct_speed=ct_speed,
                err=err.cpu().numpy(), w=w.cpu().numpy())


def per_frame_motion(q_world):
    """Per-frame consecutive-frame steps: root transl (m), root orient (deg), joints (deg)."""
    qn = q_world  # already normalized when passed in
    transl = qn[..., :3]
    root_q = qn[..., 3:7]
    nj = (qn.shape[-1] - 7) // 4
    joint_q = qn[..., 7:].reshape(qn.shape[0], nj, 4)

    transl_step = (transl[1:] - transl[:-1]).norm(dim=-1).cpu().numpy()            # (T-1,) m
    orient_step = _quat_geodesic_deg(root_q[1:], root_q[:-1]).cpu().numpy()        # (T-1,) deg
    joint_step = _quat_geodesic_deg(joint_q[1:], joint_q[:-1]).mean(-1).cpu().numpy()  # (T-1,) deg
    return transl_step, orient_step, joint_step


def pose_dev_deg(q_final, q_init, num_joints):
    """Per-frame joint-angle deviation final-vs-init (deg): mean & max over joints."""
    qf = q_final[..., 7:].reshape(q_final.shape[0], num_joints - 1, 4)
    qi = q_init[..., 7:].reshape(q_init.shape[0], num_joints - 1, 4)
    ang = _quat_geodesic_deg(qf, qi)        # (T, J-1)
    return ang.mean(-1).cpu().numpy(), ang.max(-1).values.cpu().numpy()


# ─────────────────────────────── plotting ────────────────────────────────

def _shade_invalid(ax, valid):
    """Grey-shade frames where the converter had no valid fit."""
    bad = ~valid
    if bad.any():
        ax.fill_between(np.arange(len(valid)), 0, 1, where=bad, transform=ax.get_xaxis_transform(),
                        color="0.85", zorder=0, step="mid")


def plot_person(oid, valid, init, final, motion_i, motion_f, posedev, depth_cnt,
                max_points, sap_gm_px, out_png):
    frames = np.arange(len(valid))
    fi = "#bbbbbb"   # init colour (light)
    ff = "#1f77b4"   # final colour
    fig, axes = plt.subplots(9, 1, figsize=(13, 20), sharex=True)

    ax = axes[0]
    ax.plot(frames, init["reproj"], color=fi, lw=1, label="sap init")
    ax.plot(frames, final["reproj"], color=ff, lw=1.2, label="sap final")
    if sap_gm_px > 0:
        ax.axhline(sap_gm_px, color="r", ls=":", lw=0.8, label=f"sap gm_scale={sap_gm_px:g}")
    ax.set_ylabel("reproj\n(px)")
    ax.set_ylim(0, max(40.0, sap_gm_px * 1.2))
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.set_title(f"person obj_id={oid} — per-frame loss diagnostics (grey = invalid frame)")

    ax = axes[1]
    ax.plot(frames, final["n_valid"], color="#2ca02c", lw=1.2, label="# matched kp (score≥thr)")
    ax.set_ylabel("# kp")
    ax2 = ax.twinx()
    ax2.plot(frames, final["matched_score"], color="#d62728", lw=0.9, alpha=0.7, label="mean matched score")
    ax2.set_ylabel("score", color="#d62728")
    ax2.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    ax.plot(frames, init["cham"], color=fi, lw=1, label="init")
    ax.plot(frames, final["cham"], color=ff, lw=1.2, label="final")
    if np.isfinite(final.get("pen_cm", np.array([np.nan]))).any():
        ax.plot(frames, final["pen_cm"], color="#d62728", lw=1.0, label="penetration")
    ax.set_ylabel("chamfer /\npen (cm)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[3]
    ax.plot(frames, depth_cnt, color="#9467bd", lw=1.0)
    ax.axhline(max_points, color="k", ls=":", lw=0.8, label=f"cap={max_points}")
    ax.set_ylabel("depth pts\n/ frame")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[4]
    ax.plot(frames, posedev[0], color="#ff7f0e", lw=1.2, label="mean joint Δ")
    ax.plot(frames, posedev[1], color="#8c564b", lw=0.9, alpha=0.8, label="max joint Δ")
    ax.set_ylabel("pose dev\nvs init (°)")
    ax.legend(loc="upper right", fontsize=8)

    fr2 = np.arange(1, len(valid))
    for ax, key, lbl, unit in (
        (axes[5], 0, "root transl step", "m"),
        (axes[6], 1, "root orient step", "°"),
        (axes[7], 2, "joint step (mean)", "°"),
    ):
        ax.plot(fr2, motion_i[key], color=fi, lw=1, label="init")
        ax.plot(fr2, motion_f[key], color=ff, lw=1.2, label="final")
        ax.set_ylabel(f"{lbl}\n({unit})")
        ax.legend(loc="upper right", fontsize=8)

    # velocity-gated contact panel: # gated contacting verts + their mean joint
    # speed (the contact loss repels the fast ones). NaN speed = no contact frame.
    ax = axes[8]
    has_ct = np.any(np.asarray(final.get("ct_cnt", np.zeros(1))) > 0)
    if has_ct:
        ax.plot(frames, init["ct_cnt"], color=fi, lw=1, label="# contact init")
        ax.plot(frames, final["ct_cnt"], color="#d62728", lw=1.2, label="# contact final")
        ax.set_ylabel("# contact\nverts")
        ax.legend(loc="upper left", fontsize=8)
        axc = ax.twinx()
        axc.plot(frames, final["ct_speed"], color="#17becf", lw=1.0, alpha=0.85)
        axc.set_ylabel("contact speed\n(m/s)", color="#17becf")
    else:
        ax.set_ylabel("# contact\nverts")
        ax.text(0.5, 0.5, "no contact diagnostic", transform=ax.transAxes,
                ha="center", va="center", color="0.5")

    for ax in axes:
        _shade_invalid(ax, valid)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def _worst(name, arr, frames_valid, k=6, unit=""):
    a = np.where(frames_valid, arr, np.nan)
    idx = np.argsort(np.nan_to_num(a, nan=-np.inf))[::-1][:k]
    pairs = ", ".join(f"f{int(i)}={a[i]:.1f}{unit}" for i in idx)
    return f"  {name:22s} {pairs}"


# ─────────────────────────────── main ────────────────────────────────────

def run(predictions_dir, config_path, device):
    pred = Path(predictions_dir)
    sam3d = pred / "sam3d"
    cfg = tomllib.load(Path(config_path).open("rb")) if Path(config_path).exists() else {}
    chcfg = cfg.get("chamfer", {})
    kpcfg = cfg.get("keypoints", {})
    conf_thresh = float(chcfg.get("conf_thresh", 0.1))
    max_points = int(chcfg.get("max_points", 1500))
    vsub = int(chcfg.get("vertex_subsample", 1000))
    score_thresh = float(kpcfg.get("score_thresh", 0.3))
    include_hands = bool(kpcfg.get("include_hands", False))
    sap_gm_px = float(kpcfg.get("gm_scale_px", kpcfg.get("robust_px", 40.0)))
    pncfg = cfg.get("penetration", {})
    pen = dict(conf_thresh=float(pncfg.get("conf_thresh", 0.1)),
               margin_m=float(pncfg.get("margin_m", 0.02)),
               max_dist_m=float(pncfg.get("max_dist_m", 0.15)),
               max_pen_m=float(pncfg.get("max_pen_m", 0.10)),
               prune_margin_m=float(pncfg.get("prune_margin_m", 0.30)),
               normal_k=int(pncfg.get("normal_k", 16)))
    contact = dict(contact_band_m=float(cfg.get("contact_velocity", {}).get("contact_band_m", 0.03)))

    sp = dict(np.load(sam3d / "smplx_params.npz", allow_pickle=True))
    kin = dict(np.load(paths.body_path(pred, "kinematic_1"), allow_pickle=True))
    pose = dict(np.load(pred / "sapiens" / "pose.npz", allow_pickle=True))
    human = dict(np.load(pred / "geometry" / "human.npz", allow_pickle=True))
    tr = dict(np.load(pred / "geometry" / "transform.npz", allow_pickle=True))

    N = int(sp["num_frames"])
    object_ids = [int(x) for x in kin["object_ids"]]
    scales = kin["scale"] if "scale" in kin else np.ones(len(object_ids), np.float32)
    extr = torch.as_tensor(tr["extrinsics"], dtype=torch.float32, device=device)
    K = torch.as_tensor(tr["intrinsics_px_orig"], dtype=torch.float32, device=device)
    wfc = world_from_cam(extr)
    fps = float(tr.get("fps", 30.0))

    scene_pts = scene_normals = scene_conf = None
    scene_path = pred / "geometry" / "scene.npz"
    if scene_path.exists():
        scn = dict(np.load(scene_path, allow_pickle=True))
        scene_pts = torch.as_tensor(scn["points"], dtype=torch.float32, device=device)
        scene_conf = torch.as_tensor(scn["confidence"], dtype=torch.float32, device=device)
        scene_normals = estimate_point_normals(scene_pts, wfc[:, :3, 3], k=pen["normal_k"])

    mid = load_smplx_mid(_DEFAULT_SMPLX_MODEL, device)
    fvids = torch.as_tensor([int(x) for x in mid.frame_vertex_ids], dtype=torch.long, device=device)
    corr = build_correspondence(list(mid.frame_names), include_hands=include_hands, device=device)
    sub = subsample_indices(mid.num_vertices, vsub, device=device)

    diag_dir = paths.diagnostics_dir(pred)

    for p, oid in enumerate(object_ids):
        pk = object_ids.index(oid)  # index into kin arrays (same order)
        valid = np.asarray(sp["valid_mask"][_sp_idx(sp, oid)], bool)
        betas_mean = torch.as_tensor(kin["betas"][pk], dtype=torch.float32, device=device)
        betas_T = betas_mean[None].expand(N, -1)
        scale = float(scales[pk])

        # BetterRobot model + vertex→nearest-joint map for the contact-velocity panel
        # (only when the scene cloud is present), mirroring the optimizer.
        model = v2j_sub = None
        if scene_pts is not None:
            model = make_smplx_robot_model(mid, betas_mean, device=device, dtype=torch.float32)
            v_rest, j_rest = mid.forward_shape(betas_mean[None])
            v2j_sub = vertex_to_joint(v_rest[0], j_rest[0])[sub]

        # init: from converter params, folded to world (no scale)
        spi = _sp_idx(sp, oid)
        def t(k):
            return torch.as_tensor(sp[k][spi], dtype=torch.float32, device=device)
        q_cam = mid.from_classic(betas_T, t("body_pose"), t("transl"), t("global_orient"),
                                 t("left_hand_pose"), t("right_hand_pose"))
        q_init = normalize_q(fold_root_to_world(q_cam, wfc), mid.num_joints)
        # final: stored q is already world-frame, normalized
        q_final = normalize_q(torch.as_tensor(kin["q"][pk], dtype=torch.float32, device=device),
                              mid.num_joints)

        kp_np, sc_np = _sapiens_for(pose, oid)
        target_kp = torch.as_tensor(kp_np, dtype=torch.float32, device=device)
        scores = torch.as_tensor(sc_np, dtype=torch.float32, device=device)
        depth = _person_depth_points(human, oid, N, conf_thresh, max_points, device)
        depth_cnt = np.array([int(d.shape[0]) for d in depth])
        # mirror the stage: invalid frames have no evidence
        for i in range(N):
            if not valid[i]:
                scores[i] = 0.0

        mkw = dict(extr=extr, K=K, depth=depth, target_kp=target_kp, scores=scores, corr=corr,
                   fvids=fvids, sub=sub, score_thresh=score_thresh,
                   scene_pts=scene_pts, scene_normals=scene_normals, scene_conf=scene_conf, pen=pen,
                   model=model, fps=fps, v2j_sub=v2j_sub, contact=contact)
        init = per_frame_metrics(mid, betas_T, q_init, 1.0, **mkw)
        final = per_frame_metrics(mid, betas_T, q_final, scale, **mkw)
        motion_i = per_frame_motion(q_init)
        motion_f = per_frame_motion(q_final)
        posedev = pose_dev_deg(q_final, q_init, mid.num_joints)

        out_png = diag_dir / f"loss_person{oid:02d}.png"
        plot_person(oid, valid, init, final, motion_i, motion_f, posedev, depth_cnt,
                    max_points, sap_gm_px, out_png)
        print(f"[diagnose] wrote {out_png}")

        vv = valid.copy()
        vv2 = valid[1:].copy()
        print(f"  person obj_id={oid}  scale={scale:.3f}  ({int(valid.sum())}/{N} valid)")
        print(_worst("reproj_sap final (px)", final["reproj"], vv, unit="px"))
        print(_worst("chamfer final (cm)", final["cham"], vv, unit="cm"))
        if np.isfinite(final["pen_cm"]).any():
            print(_worst("penetration final (cm)", final["pen_cm"], vv, unit="cm"))
        if np.any(final.get("ct_cnt", np.zeros(1)) > 0):
            print(_worst("contact speed (m/s)", final["ct_speed"], vv, unit="m/s"))
        print(_worst("pose dev max (deg)", posedev[1], vv, unit="°"))
        print(_worst("joint step final (deg)", motion_f[2], vv2, unit="°"))
        print(_worst("root transl step (m)", motion_f[0], vv2, unit="m"))
        lo = np.where(vv, final["n_valid"], np.inf)
        di = np.argsort(lo)[:6]
        print("  min matched-kp frames:   "
              + ", ".join(f"f{int(i)}={int(final['n_valid'][i])}" for i in di))


def _sp_idx(sp, oid):
    return [int(x) for x in sp["object_ids"]].index(oid)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    run(args.predictions, args.config, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
