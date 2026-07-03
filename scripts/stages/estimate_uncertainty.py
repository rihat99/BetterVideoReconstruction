"""Per-frame + per-contact reconstruction uncertainty for the SMPL-X bodies.

A reconstruction is only as trustworthy as the evidence behind it, and that
evidence varies frame to frame (a keypoint dropout, a bad depth patch, a fast
occluded limb). This standalone stage scores that trust. For each tracked person
it reads an optimized body (``kinematic_2`` if present, else
``kinematic_1``) and computes, **per frame**, four complementary axes:

1. **Data-fit loss** ``L(t)`` — the kinematic stage's *data* terms (Sapiens +
   init-anchor reprojection, robustified with the same Geman-McClure kernel, plus
   chamfer to the metric depth cloud), weighted by the same ``[weights]`` the
   optimizer used. Priors / smoothness are deliberately excluded: they measure how
   *regularised* the motion is, not how well it matches the observations. From
   ``L(t)`` we take the clip's best-fit frame ``L* = min_t L(t)`` and a **relative
   score** ``s(t) = L*/L(t) ∈ (0,1]`` (1 at the best frame), plus a robust variant
   referenced to the 5th-percentile loss (a single freak-good frame shouldn't drag
   everyone down).
2. **Observability** — how much evidence constrained the frame: # matched Sapiens
   keypoints, # confident depth points, and the mean VGGT depth confidence. A frame
   can fit well simply because little data could disagree; this axis catches that
   (the relative loss alone cannot).
3. **Absolute residuals** — reprojection in px and chamfer in cm, with fixed
   thresholds. Interpretable on their own and comparable *across* clips (the
   relative score is only within-clip).
4. **Cue disagreement** — the spread between the independent 2D cues (Sapiens vs
   the init-anchor reprojection). When independent observations conflict,
   uncertainty is high — no ground truth needed.

And, **per contact** (every ``contacts_*`` joint-contact), a confidence in [0, 1]
from the SDF **margin** past the contact threshold (``joint_sdf_cm`` — a contact
right at the boundary is uncertain, a deeply-inside one is not) × the contacting
joint's **speed** (a planted grip is slow; a fast "contact" is a limb clipping the
wall) × its **visibility** (front-facing fraction of its vertices) × temporal
**persistence** (a 1-frame flicker is less certain than a sustained grip).

The reprojection / chamfer are recomputed with the optimizer's exact correspondence
+ projection + GM code, so the numbers match the kinematic stage; the per-contact
SDF margin is read straight from ``estimate_contacts`` (no recompute).

Reads (under ``out/<stem>/``)::

    human_optim/<bodies>.npz  optimized body (q, betas, scale, valid_mask)
    sam3d/smplx_params.npz    converter init pose (the init-anchor 2D target)
    human_optim/<cnt>.npz     per-joint contact + joint_sdf_cm (for per-contact)
    sapiens/pose.npz          Goliath-308 keypoints + scores
    geometry/{transform,human}.npz  cameras + per-frame metric human depth cloud

Writes::

    human_optim/uncertainty_<tag>.npz                 per-frame + per-contact arrays
    human_optim/diagnostics/uncertainty_person<oid>.png   the four-axis plot per person

Standalone CLI::

    uv run python scripts/stages/estimate_uncertainty.py --predictions out/video/
    uv run python scripts/stages/estimate_uncertainty.py --predictions out/video/ --bodies kinematic_1
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
_BETTER_HUMAN_SRC = _REPO_ROOT.parent / "better_human" / "src"
for _p in (str(_BETTER_HUMAN_SRC), str(_REPO_ROOT)):
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.human_optim import (  # noqa: E402
    build_correspondence, fold_root_to_world, geman_mcclure, project,
    subsample_indices, transform_points, vertex_to_joint, world_from_cam,
)
from tools.human_optim.qmath import normalize_q  # noqa: E402
from tools.human_optim import paths  # noqa: E402
from tools.human_optim.config import load_config  # noqa: E402
from tools.human_optim.visibility import front_facing_mask, vertex_normals  # noqa: E402
from tools.smplx_robot import load_smplx_mid  # noqa: E402

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "human_optimization.toml"
_DEFAULT_SMPLX_MODEL = (
    _REPO_ROOT.parent / "better_human" / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
)
# Which contacts pair with which body, and the output tag.
_BODY_TO_CONTACT = {
    "kinematic_2": ("contacts_2", "2"),
    "kinematic_1": ("contacts_1", "1"),
}

# ── scoring references (tunable; one home, here) ──────────────────────────────
# Observability is computed PER CLIP: each evidence channel (# keypoints, # depth
# points, mean VGGT depth confidence) is divided by its own 90th
# percentile over the clip's valid frames, so a frame with the clip's usual evidence
# sits at 1 and only a genuine dropout (occlusion, detector miss) drops below. This
# is self-calibrating across clips and pairs with the within-clip relative loss score.
OBS_REF_PCT = 90.0
# A frame is flagged "reliable" when reproj + chamfer are tight and it is observed.
RELIABLE_REPROJ_PX = 15.0
RELIABLE_CHAMFER_CM = 40.0
RELIABLE_OBS = 0.3
# Per-contact confidence references.
MARGIN_REF_CM = 2.0   # sdf depth past the contact threshold for full margin-confidence
SPEED_REF_MS = 0.5    # joint speed at which speed-confidence falls to 1/e
PERSIST_WIN = 2       # ± frames for the persistence window (a 1-frame flicker scores 1/(2W+1))
VIS_FLOOR = 0.5       # an occluded contact keeps this much confidence (a grip presses
                      # into the wall, so it is *expected* to face away — visibility only
                      # gently modulates; the SDF margin carries the "is it touching")


# ─────────────────────────────── io helpers ──────────────────────────────────

def _idx(arr, oid):
    return [int(x) for x in arr].index(oid)


def _person_depth_and_conf(human, oid, N, conf_thresh, max_points, device):
    """Per-frame world depth points (conf-gated, capped) + per-frame depth count
    and mean VGGT confidence for one person — matching the kinematic stage's gate."""
    obj = human["obj_id"]
    fr_all = human["frame_id"]
    conf_all = human["confidence"]
    mine = obj == oid
    # Per-frame mean confidence over ALL of this person's points (raw geometry quality).
    mean_conf = np.zeros(N, np.float32)
    if mine.any():
        fm, cm = fr_all[mine], conf_all[mine]
        sums = np.bincount(fm, weights=cm, minlength=N)
        cnts = np.bincount(fm, minlength=N).astype(np.float32)
        mean_conf = (sums / np.maximum(cnts, 1.0)).astype(np.float32)[:N]
    # Confident points (the chamfer target).
    keep = mine & (conf_all > conf_thresh)
    pts, fr = human["points"][keep], fr_all[keep]
    order = np.argsort(fr, kind="stable")
    pts, fr = pts[order], fr[order]
    bnd = np.searchsorted(fr, np.arange(N + 1))
    out, cnt = [], np.zeros(N, np.int64)
    for t in range(N):
        p = pts[bnd[t]:bnd[t + 1]]
        cnt[t] = len(p)
        if len(p) > max_points:
            p = p[np.linspace(0, len(p) - 1, max_points).astype(int)]
        out.append(torch.as_tensor(p, dtype=torch.float32, device=device))
    return out, cnt, mean_conf


def _sapiens_for(pose, oid):
    p = _idx(pose["object_ids"], oid)
    return pose["keypoints"][p], pose["keypoint_scores"][p]


# ─────────────────────────── per-frame computations ──────────────────────────

@torch.no_grad()
def _reproj_axes(jw, vw, *, extr, K, corr, fvids, body_idx, init_target,
                 target_kp, scores, score_thresh, sap_gm, init_gm):
    """Per-frame reprojection losses (GM, loss-native) + raw px + counts.

    Returns ``L_sap, L_init`` (GM means ∈[0,1)), the raw px errors
    ``reproj_sap_px / reproj_init_px`` (the absolute axis), and ``n_kp``
    (observability). Mirrors ``losses.sapiens_reprojection_gm`` etc. but keeps the
    per-frame axis (reduces over keypoints only)."""
    # Sapiens Goliath-308: score-weighted GM over body joints + landmark verts.
    body = jw[:, corr.joint_model, :]
    lm = vw[:, fvids[corr.lm_frame], :]
    pred_px = project(transform_points(torch.cat([body, lm], 1), extr), K)   # (T,K,2)
    tgt = target_kp[:, corr.sap, :]
    w = (scores[:, corr.sap] >= score_thresh).float()                        # (T,K)
    err = (pred_px - tgt).norm(dim=-1)                                       # (T,K) px
    gm = geman_mcclure(err.square(), pred_px.new_tensor(sap_gm) ** 2)
    denom = w.sum(1).clamp_min(1.0)
    L_sap = (w * gm).sum(1) / denom
    reproj_sap_px = (w * err).sum(1) / denom
    n_kp = w.sum(1)

    # Init-anchor: drift of the final body joints from the converter-pose projection.
    ip = project(transform_points(jw[:, body_idx, :], extr), K)             # (T,Kb,2)
    err_init = (ip - init_target).norm(dim=-1)                              # (T,Kb)
    L_init = geman_mcclure(err_init.square(), ip.new_tensor(init_gm) ** 2).mean(1)
    reproj_init_px = err_init.mean(1)

    return dict(L_sap=L_sap, L_init=L_init, reproj_sap_px=reproj_sap_px,
                reproj_init_px=reproj_init_px, n_kp=n_kp)


@torch.no_grad()
def _chamfer_per_frame(vw, sub, faces, extr, depth_list):
    """Per-frame bidirectional chamfer (cm) on front-facing subsampled verts.

    Same culling the chamfer loss uses; NaN where the frame has no depth points."""
    T = vw.shape[0]
    n = vertex_normals(vw, faces)
    vcam = transform_points(vw[:, sub], extr)
    ncam = torch.einsum("tij,tvj->tvi", extr[:, :3, :3], n[:, sub])
    vis = front_facing_mask(vcam, ncam)
    cham = np.full(T, np.nan, np.float32)
    for t, D in enumerate(depth_list):
        if D.shape[0] == 0:
            continue
        V = vw[t, sub][vis[t]]
        if V.shape[0] == 0:
            continue
        d = torch.cdist(V, D)
        cham[t] = float(d.min(1).values.mean() + d.min(0).values.mean()) * 100.0
    return cham


@torch.no_grad()
def _joint_visibility(vw, faces, extr, v2j_full, J):
    """Per-frame, per-joint front-facing vertex fraction ``(T, J)`` ∈ [0,1]."""
    n = vertex_normals(vw, faces)
    vcam = transform_points(vw, extr)
    ncam = torch.einsum("tij,tvj->tvi", extr[:, :3, :3], n)
    ff = front_facing_mask(vcam, ncam).float()                              # (T, V)
    vis = vw.new_zeros(vw.shape[0], J)
    for j in range(J):
        m = v2j_full == j
        if bool(m.any()):
            vis[:, j] = ff[:, m].mean(1)
    return vis


@torch.no_grad()
def _joint_speed(jw, fps):
    """Per-frame, per-joint world speed ``(T, J)`` (m/s), central finite-difference.

    Its magnitude equals the BetterRobot-Jacobian LOCAL linear speed (the rotation
    preserves norm — see ``tools/human_optim/velocity.py``), so no model is needed."""
    T = jw.shape[0]
    speed = jw.new_zeros(T, jw.shape[1])
    if T >= 3:
        speed[1:-1] = (jw[2:] - jw[:-2]).norm(dim=-1) * (fps / 2.0)
    if T >= 2:
        speed[0] = (jw[1] - jw[0]).norm(dim=-1) * fps
        speed[-1] = (jw[-1] - jw[-2]).norm(dim=-1) * fps
    return speed


def _persistence(jc, win):
    """Per-(frame,joint) fraction of a ±win window also in contact ``(N, J)`` ∈ [0,1]."""
    N = jc.shape[0]
    acc = np.zeros_like(jc, dtype=np.float32)
    for d in range(-win, win + 1):
        lo, hi = max(0, d), min(N, N + d)
        acc[max(0, -d):min(N, N - d)] += jc[lo:hi].astype(np.float32)
    return acc / (2 * win + 1)


# ─────────────────────────────── scoring ─────────────────────────────────────

def _score_person(rep, cham_cm, depth_cnt, mean_conf, valid, weights):
    """Combine the per-frame axes into the data-fit loss + scores. All arrays (N,)."""
    w_sap, w_init, w_ch = weights
    cham_m = cham_cm / 100.0
    L_ch = np.where(np.isnan(cham_m), 0.0, cham_m)   # no-depth frames: drop chamfer (flagged by obs)
    fit_loss = (w_sap * rep["L_sap"] + w_init * rep["L_init"] + w_ch * L_ch).astype(np.float32)
    fit_loss = np.where(valid, fit_loss, np.nan)

    v = valid & np.isfinite(fit_loss)
    if v.any():
        Lmin = float(np.nanmin(fit_loss[v]))
        Lp5 = float(np.percentile(fit_loss[v], 5))
    else:
        Lmin = Lp5 = float("nan")
    fit_score = np.clip(Lmin / np.where(fit_loss > 0, fit_loss, np.nan), 0, 1).astype(np.float32)
    fit_score_robust = np.clip(Lp5 / np.where(fit_loss > 0, fit_loss, np.nan), 0, 1).astype(np.float32)

    # Observability: geometric mean of the per-clip-normalized evidence channels.
    def norm(x):
        vv = valid & np.isfinite(x)
        ref = float(np.percentile(x[vv], OBS_REF_PCT)) if vv.any() else 0.0
        return np.ones_like(x, np.float32) if ref <= 0 else np.clip(x / ref, 0, 1).astype(np.float32)

    comps = [norm(rep["n_kp"]), norm(depth_cnt.astype(np.float32)), norm(mean_conf)]
    obs = np.ones_like(fit_loss)
    for c in comps:
        obs = obs * c
    obs_score = (obs ** (1.0 / len(comps))).astype(np.float32)

    # Cue disagreement: spread of the available reprojection px cues per frame.
    cues = [rep["reproj_sap_px"], rep["reproj_init_px"]]
    stack = np.stack(cues, 0)
    disagreement_px = (np.nanmax(stack, 0) - np.nanmin(stack, 0)).astype(np.float32)

    reliable = (valid & (rep["reproj_sap_px"] < RELIABLE_REPROJ_PX)
                & ((cham_cm < RELIABLE_CHAMFER_CM) | np.isnan(cham_cm))
                & (obs_score > RELIABLE_OBS))
    combined = (fit_score_robust * obs_score).astype(np.float32)
    for a in (fit_score, fit_score_robust, obs_score, disagreement_px, combined):
        a[~valid] = np.nan
    return dict(fit_loss=fit_loss, fit_score=fit_score, fit_score_robust=fit_score_robust,
                obs_score=obs_score, disagreement_px=disagreement_px, combined=combined,
                reliable=reliable, fit_loss_min=np.float32(Lmin), fit_loss_p5=np.float32(Lp5))


def _score_contacts(jc, jsdf_cm, close_thresh_cm, speed, vis, valid):
    """Per-(frame,joint) contact confidence in [0,1], NaN where not in contact."""
    N, J = jc.shape
    conf_margin = np.clip((close_thresh_cm - jsdf_cm) / MARGIN_REF_CM, 0, 1).astype(np.float32)
    conf_speed = np.exp(-np.clip(speed, 0, None) / SPEED_REF_MS).astype(np.float32)
    conf_vis = (VIS_FLOOR + (1.0 - VIS_FLOOR) * np.clip(vis, 0, 1)).astype(np.float32)
    conf_persist = _persistence(jc, PERSIST_WIN).astype(np.float32)
    conf = (conf_margin * conf_speed * conf_vis * conf_persist).astype(np.float32)
    on = jc & valid[:, None]
    nan = np.full((N, J), np.nan, np.float32)
    pack = lambda a: np.where(on, a, nan)
    return dict(contact_conf=pack(conf), contact_conf_margin=pack(conf_margin),
                contact_conf_speed=pack(conf_speed), contact_conf_vis=pack(conf_vis),
                contact_conf_persist=pack(conf_persist), joint_speed_ms=speed.astype(np.float32))


# ─────────────────────────────── plotting ────────────────────────────────────

def _shade_invalid(ax, valid):
    bad = ~valid
    if bad.any():
        ax.fill_between(np.arange(len(valid)), 0, 1, where=bad, transform=ax.get_xaxis_transform(),
                        color="0.85", zorder=0, step="mid")


def _mean_contact_conf(jc, contact_conf):
    """Per-frame mean confidence over the joints in contact (NaN where none)."""
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.where(jc, contact_conf, np.nan), axis=1)


def _make_fig(oid, valid, combined, reproj_px, cham_cm, jc, contact_conf, figsize=(12, 9)):
    """Build the simplified 3-panel uncertainty figure. Returns ``(fig, axes)``.

    Three plain-language panels — confidence, how-far-off, contacts — shared by the
    standalone PNG and the overlay's side strip (so they never drift)."""
    frames = np.arange(len(valid))
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # 1) the single confidence number (1 = trust this frame, red band = low).
    ax = axes[0]
    ax.plot(frames, combined, color="#1f77b4", lw=1.8)
    ax.axhline(0.5, color="0.6", ls="--", lw=0.8)
    ax.fill_between(frames, 0, 1, where=np.nan_to_num(combined) < 0.4,
                    transform=ax.get_xaxis_transform(), color="#ffd6d6", zorder=0, step="mid")
    ax.set_ylabel("confidence")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"person {oid} — confidence  (1 = trust this frame, pink = low)")

    # 2) how far the body is off: image error (px) + depth error (cm).
    ax = axes[1]
    l1, = ax.plot(frames, reproj_px, color="#1f77b4", lw=1.4, label="image error (px)")
    ax.axhline(RELIABLE_REPROJ_PX, color="#1f77b4", ls=":", lw=0.8)
    ax.set_ylabel("image err (px)", color="#1f77b4")
    ax.set_ylim(0, max(40.0, RELIABLE_REPROJ_PX * 1.5))
    axr = ax.twinx()
    l2, = axr.plot(frames, cham_cm, color="#d62728", lw=1.4, label="depth error (cm)")
    axr.axhline(RELIABLE_CHAMFER_CM, color="#d62728", ls=":", lw=0.8)
    axr.set_ylabel("depth err (cm)", color="#d62728")
    ax.set_title("how far off the body is — in the image (blue) and in depth (red)")
    ax.legend(handles=[l1, l2], loc="upper right", fontsize=8)

    # 3) contacts: how sure each touch is (green) + how many (grey).
    ax = axes[2]
    ax.plot(frames, _mean_contact_conf(jc, contact_conf), color="#2ca02c", lw=1.6)
    ax.set_ylabel("contact conf")
    ax.set_ylim(-0.02, 1.05)
    axr = ax.twinx()
    axr.plot(frames, jc.sum(1), color="0.6", lw=1.0)
    axr.set_ylabel("# contacts", color="0.5")
    ax.set_title("contacts — how sure we are each touch is real (green), and how many (grey)")

    for ax in axes:
        _shade_invalid(ax, valid)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    return fig, axes


def _plot_person(oid, valid, combined, reproj_px, cham_cm, jc, contact_conf, out_png):
    fig, _ = _make_fig(oid, valid, combined, reproj_px, cham_cm, jc, contact_conf)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def _conf_color(c):
    """BGR colour for a confidence in [0,1]: red (0, unsure) → green (1, sure)."""
    c = 0.0 if not np.isfinite(c) else float(np.clip(c, 0.0, 1.0))
    return (0, int(255 * c), int(255 * (1.0 - c)))


def _render_overlay(source, out_path, *, joints_2d, joint_contact, contact_conf,
                    valid_mask, parents, panel, H, W, fps):
    """Composite mp4: each frame (with its contact skeleton) beside the uncertainty
    panels, a red playhead sweeping them at the current frame.

    On the frame: bones grey; a joint **in contact** is a fat dot coloured by its
    contact confidence (red = unsure → green = sure), a free joint a small grey dot;
    a top banner shows the frame's overall confidence. The side panels are
    :func:`_make_fig` for the **primary** person (``panel`` dict) with a moving line.
    All people's skeletons are drawn; ``joints_2d (P,N,J,2)``, ``joint_contact`` /
    ``contact_conf (P,N,J)``, ``valid_mask (P,N)`` cover everyone."""
    import tempfile

    import cv2
    from tqdm import tqdm

    from tools.source import resolve_source

    P, N, J = joint_contact.shape

    # Pre-render the side panels ONCE; the playhead is drawn per frame (cheap).
    fig, axes = _make_fig(panel["oid"], panel["valid"], panel["combined"], panel["reproj"],
                          panel["cham"], panel["jc"], panel["cc"], figsize=(8, 9))
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    panel_bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR).copy()
    x0 = float(axes[0].transData.transform((0, 0))[0])          # frame-0 x in canvas px
    x1 = float(axes[0].transData.transform((max(1, N - 1), 0))[0])
    plt.close(fig)
    Hp, Wp = panel_bgr.shape[:2]
    panel_w = max(1, int(round(Wp * (H / Hp))))

    def ok(uv):
        return np.isfinite(uv).all() and abs(uv[0]) < 1e4 and abs(uv[1]) < 1e4

    with tempfile.TemporaryDirectory(prefix="uncertainty_overlay_") as tmp:
        frame_paths, src_fps = resolve_source(Path(source), Path(tmp))
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 float(fps or src_fps or 30.0), (W + panel_w, H))
        try:
            for i in tqdm(range(min(N, len(frame_paths))), desc="uncertainty overlay"):
                bgr = cv2.imread(str(frame_paths[i]))
                if bgr is None:
                    continue
                if bgr.shape[:2] != (H, W):
                    bgr = cv2.resize(bgr, (W, H))
                for p in range(P):
                    if not valid_mask[p, i]:
                        continue
                    pts = joints_2d[p, i]
                    for j in range(1, J):                              # bones (grey)
                        a, b = pts[int(parents[j])], pts[j]
                        if ok(a) and ok(b):
                            cv2.line(bgr, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                                     (170, 170, 170), 1, cv2.LINE_AA)
                    for j in range(J):                                 # joints
                        uv = pts[j]
                        if not ok(uv):
                            continue
                        xy = (int(uv[0]), int(uv[1]))
                        if joint_contact[p, i, j]:                     # contact: colour by confidence
                            cv2.circle(bgr, xy, 5, _conf_color(contact_conf[p, i, j]), -1, cv2.LINE_AA)
                            cv2.circle(bgr, xy, 5, (30, 30, 30), 1, cv2.LINE_AA)
                        else:                                          # free joint: dim grey
                            cv2.circle(bgr, xy, 2, (120, 120, 120), -1, cv2.LINE_AA)
                # top banner: this frame's overall confidence (primary person)
                c0 = panel["combined"][i]
                cv2.rectangle(bgr, (0, 0), (W, 34), (0, 0, 0), -1)
                if np.isfinite(c0):
                    cv2.putText(bgr, f"frame {i}   confidence {c0:.2f}", (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _conf_color(c0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(bgr, f"frame {i}   (not tracked)", (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
                cv2.putText(bgr, "contacts: red = unsure   green = sure", (10, H - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                # side panels + playhead
                ph = panel_bgr.copy()
                xp = int(round(x0 + (x1 - x0) * (i / max(1, N - 1))))
                cv2.line(ph, (xp, 0), (xp, Hp), (0, 0, 255), 2)
                ph = cv2.resize(ph, (panel_w, H))
                writer.write(np.hstack([bgr, ph]))
        finally:
            writer.release()
    return out_path


def _worst(name, arr, valid, k=6, unit="", low=False):
    a = np.where(valid, arr, np.nan)
    order = np.argsort(np.nan_to_num(a, nan=(np.inf if low else -np.inf)))
    idx = order[:k] if low else order[::-1][:k]
    pairs = ", ".join(f"f{int(i)}={a[i]:.1f}{unit}" for i in idx if np.isfinite(a[i]))
    return f"  {name:24s} {pairs}"


# ─────────────────────────────── main ────────────────────────────────────────

def run(predictions_dir, *, source=None, overlay=False, bodies="kinematic_2",
        config_path=_DEFAULT_CONFIG, device="cuda"):
    pred = Path(predictions_dir)
    sam3d = pred / "sam3d"
    if bodies not in _BODY_TO_CONTACT:
        raise SystemExit(f"--bodies must be one of {sorted(_BODY_TO_CONTACT)}")
    body_path = paths.body_path(pred, bodies)
    if not body_path.exists():
        # default fall-back: prefer kinematics_2, else kinematics_1
        alt = "kinematic_1" if bodies == "kinematic_2" else None
        if alt and paths.body_path(pred, alt).exists():
            print(f"[uncertainty] {bodies}.npz absent → using {alt}.npz")
            bodies = alt
            body_path = paths.body_path(pred, alt)
        else:
            raise SystemExit(f"no body npz at {body_path}")
    cnt_name, tag = _BODY_TO_CONTACT[bodies]

    # Data-fit loss uses the SAME weights + kernels as human_optimization's
    # kinematics_1 stage (its data terms only — reproj + chamfer), read from the
    # kinematics_1 column + shared knobs of the single config; else defaults.
    if Path(config_path).exists():
        hc = load_config(config_path)
        w1, knb = hc.weights_for("kinematics_1"), hc.knobs
        weights = (float(w1["reproj_sapiens"]), float(w1["reproj_init"]), float(w1["chamfer"]))
        sap_gm = float(knb["keypoints"]["gm_scale_px"])
        init_gm = float(knb["reproj_init"]["gm_scale_px"])
        score_thresh = float(knb["keypoints"]["score_thresh"])
        conf_thresh = float(knb["chamfer"]["conf_thresh"])
        max_points = int(knb["chamfer"]["max_points"])
        vsub = int(knb["chamfer"]["vertex_subsample"])
        include_hands = bool(knb["keypoints"]["include_hands"])
    else:
        weights = (8.0, 30.0, 10.0)
        sap_gm, init_gm, score_thresh = 40.0, 30.0, 0.3
        conf_thresh, max_points, vsub = 0.1, 1500, 1000
        include_hands = False

    body = dict(np.load(body_path, allow_pickle=True))
    params = dict(np.load(sam3d / "smplx_params.npz", allow_pickle=True))
    pose = dict(np.load(pred / "sapiens" / "pose.npz", allow_pickle=True))
    human = dict(np.load(pred / "geometry" / "human.npz", allow_pickle=True))
    tr = dict(np.load(pred / "geometry" / "transform.npz", allow_pickle=True))
    cnt_path = paths.contacts_path(pred, tag)
    cnt = dict(np.load(cnt_path, allow_pickle=True)) if cnt_path.exists() else None
    if cnt is None:
        print(f"[uncertainty] {cnt_path} absent → per-contact axis skipped")

    N = int(body["num_frames"])
    object_ids = [int(x) for x in body["object_ids"]]
    scales = body["scale"] if "scale" in body else np.ones(len(object_ids), np.float32)
    extr = torch.as_tensor(tr["extrinsics"], dtype=torch.float32, device=device)
    K = torch.as_tensor(tr["intrinsics_px_orig"], dtype=torch.float32, device=device)
    wfc = world_from_cam(extr)
    fps = float(tr.get("fps", body.get("fps", 30.0)))

    mid = load_smplx_mid(_DEFAULT_SMPLX_MODEL, device)
    J = mid.num_joints
    fvids = torch.as_tensor([int(x) for x in mid.frame_vertex_ids], dtype=torch.long, device=device)
    corr = build_correspondence(list(mid.frame_names), include_hands=include_hands, device=device)
    sub = subsample_indices(mid.num_vertices, vsub, device=device)
    jnames = list(mid.joint_names)
    _finger = ("index", "middle", "pinky", "ring", "thumb")
    body_idx = torch.tensor([i for i in range(J) if not any(f in jnames[i].lower() for f in _finger)],
                            dtype=torch.long, device=device)

    diag_dir = paths.diagnostics_dir(pred)

    P = len(object_ids)
    store = {k: np.full((P, N), np.nan, np.float32) for k in
             ("fit_loss", "fit_score", "fit_score_robust", "obs_score", "disagreement_px",
              "combined", "reproj_sap_px", "reproj_init_px", "chamfer_cm",
              "n_kp", "n_depth", "mean_conf")}
    store["reliable"] = np.zeros((P, N), bool)
    store["valid"] = np.zeros((P, N), bool)
    cstore = {k: np.full((P, N, J), np.nan, np.float32) for k in
              ("contact_conf", "contact_conf_margin", "contact_conf_speed",
               "contact_conf_vis", "contact_conf_persist", "joint_speed_ms")}
    cstore["joint_contact"] = np.zeros((P, N, J), bool)
    cstore["joint_sdf_cm"] = np.full((P, N, J), np.nan, np.float32)
    loss_min = np.full(P, np.nan, np.float32)
    loss_p5 = np.full(P, np.nan, np.float32)
    joints_2d = np.full((P, N, J, 2), np.nan, np.float32)   # projected skeleton (for the overlay)

    for pk, oid in enumerate(object_ids):
        valid = np.asarray(body["valid_mask"][pk], bool)
        betas_mean = torch.as_tensor(body["betas"][pk], dtype=torch.float32, device=device)
        betas_T = betas_mean[None].expand(N, -1)
        scale = float(scales[pk])

        # Optimized body (world-frame q) → posed joints/verts, scaled about pelvis.
        q_final = normalize_q(torch.as_tensor(body["q"][pk], dtype=torch.float32, device=device), J)
        out = mid.forward(betas_T, q_final)
        jw = out.joints_world.tensor()[..., :3]
        vw = out.vertices
        root = jw[:, :1, :]
        jw = root + scale * (jw - root)
        vw = root + scale * (vw - root)
        with torch.no_grad():
            joints_2d[pk] = project(transform_points(jw, extr), K).cpu().numpy()

        # Init-anchor target: the converter-pose body projection (the trusted 2D).
        spi = _idx(params["object_ids"], oid)
        tt = lambda k: torch.as_tensor(params[k][spi], dtype=torch.float32, device=device)
        q_cam = mid.from_classic(betas_T, tt("body_pose"), tt("transl"), tt("global_orient"),
                                 tt("left_hand_pose"), tt("right_hand_pose"))
        q_init = normalize_q(fold_root_to_world(q_cam, wfc), J)
        oi = mid.forward(betas_T, q_init)
        jw_init = oi.joints_world.tensor()[..., :3]
        with torch.no_grad():
            init_target = project(transform_points(jw_init[:, body_idx, :], extr), K)

        kp_np, sc_np = _sapiens_for(pose, oid)
        target_kp = torch.as_tensor(kp_np, dtype=torch.float32, device=device)
        scores = torch.as_tensor(sc_np, dtype=torch.float32, device=device)
        scores[torch.as_tensor(~valid, device=device)] = 0.0   # no evidence on invalid frames

        rep = _reproj_axes(jw, vw, extr=extr, K=K, corr=corr, fvids=fvids, body_idx=body_idx,
                           init_target=init_target, target_kp=target_kp, scores=scores,
                           score_thresh=score_thresh, sap_gm=sap_gm, init_gm=init_gm)
        rep = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in rep.items()}

        depth, depth_cnt, mean_conf = _person_depth_and_conf(human, oid, N, conf_thresh, max_points, device)
        cham_cm = _chamfer_per_frame(vw, sub, mid.faces, extr, depth)

        pf = _score_person(rep, cham_cm, depth_cnt, mean_conf, valid, weights)

        for k in ("fit_loss", "fit_score", "fit_score_robust", "obs_score",
                  "disagreement_px", "combined"):
            store[k][pk] = pf[k]
        store["reliable"][pk] = pf["reliable"]
        store["valid"][pk] = valid
        store["reproj_sap_px"][pk] = rep["reproj_sap_px"]
        store["reproj_init_px"][pk] = rep["reproj_init_px"]
        store["chamfer_cm"][pk] = cham_cm
        store["n_kp"][pk] = rep["n_kp"]
        store["n_depth"][pk] = depth_cnt
        store["mean_conf"][pk] = mean_conf
        loss_min[pk] = pf["fit_loss_min"]
        loss_p5[pk] = pf["fit_loss_p5"]

        # ── per-contact ──
        pc = None
        jc = np.zeros((N, J), bool)
        if cnt is not None:
            cpi = _idx(cnt["object_ids"], oid)
            jc = np.asarray(cnt["joint_contact"][cpi], bool)
            jsdf_cm = np.asarray(cnt["joint_sdf_cm"][cpi], np.float32)
            close_cm = float(cnt["close_thresh_m"]) * 100.0
            speed = _joint_speed(jw, fps).detach().cpu().numpy()
            v_rest, j_rest = mid.forward_shape(betas_mean[None])
            v2j_full = vertex_to_joint(v_rest[0], j_rest[0])
            vis = _joint_visibility(vw, mid.faces, extr, v2j_full, J).detach().cpu().numpy()
            pc = _score_contacts(jc, jsdf_cm, close_cm, speed, vis, valid)
            for k, v in pc.items():
                cstore[k][pk] = v
            cstore["joint_contact"][pk] = jc
            cstore["joint_sdf_cm"][pk] = jsdf_cm

        cc_arr = pc["contact_conf"] if pc is not None else np.full((N, J), np.nan, np.float32)
        out_png = diag_dir / f"uncertainty_person{oid:02d}.png"
        _plot_person(oid, valid, pf["combined"], rep["reproj_sap_px"], cham_cm, jc, cc_arr, out_png)
        print(f"[uncertainty] wrote {out_png}")

        nv = int(valid.sum())
        rel = int(pf["reliable"].sum())
        with np.errstate(invalid="ignore"):
            mfs = float(np.nanmean(pf["fit_score_robust"]))
            mcomb = float(np.nanmean(pf["combined"]))
        print(f"  person obj_id={oid}  scale={scale:.3f}  ({nv}/{N} valid, {rel} reliable)")
        print(f"    L*={loss_min[pk]:.3f} (p5={loss_p5[pk]:.3f})  "
              f"mean robust-score={mfs:.3f}  mean combined={mcomb:.3f}")
        print(_worst("worst fit_loss", pf["fit_loss"], valid))
        print(_worst("worst reproj_sap (px)", rep["reproj_sap_px"], valid, unit="px"))
        print(_worst("worst chamfer (cm)", cham_cm, valid, unit="cm"))
        print(_worst("worst disagreement (px)", pf["disagreement_px"], valid, unit="px"))
        print(_worst("lowest obs_score", pf["obs_score"], valid, low=True))
        if pc is not None and jc.any():
            cc = pc["contact_conf"]
            with np.errstate(invalid="ignore"):
                m = float(np.nanmean(cc))
            n_low = int(np.nansum(cc < 0.3))
            print(f"    contacts: {int(jc.sum())} total, mean conf={m:.2f}, {n_low} low (<0.3)")

    out_npz = paths.uncertainty_path(pred, tag)
    np.savez(
        out_npz,
        object_ids=np.asarray(object_ids, np.int32),
        joint_names=body["joint_names"], parents=body["parents"],
        num_people=np.int32(P), num_frames=np.int32(N), fps=np.float32(fps),
        video_height=body.get("video_height", np.int32(0)),
        video_width=body.get("video_width", np.int32(0)),
        bodies=np.str_(bodies), contacts=np.str_(cnt_name if cnt is not None else "none"),
        fit_loss_min=loss_min, fit_loss_p5=loss_p5,
        weights=np.asarray(weights, np.float32),
        **store, **cstore,
    )
    print(f"[uncertainty] wrote {out_npz}")

    if overlay:
        if source is None:
            raise SystemExit("--overlay needs --source (the original video / frames)")
        H, W = int(body["video_height"]), int(body["video_width"])
        panel = dict(oid=object_ids[0], valid=store["valid"][0], combined=store["combined"][0],
                     reproj=store["reproj_sap_px"][0], cham=store["chamfer_cm"][0],
                     jc=cstore["joint_contact"][0], cc=cstore["contact_conf"][0])
        ov = paths.overlay_path(pred, f"uncertainty_{tag}")
        _render_overlay(source, ov, joints_2d=joints_2d, joint_contact=cstore["joint_contact"],
                        contact_conf=cstore["contact_conf"], valid_mask=store["valid"],
                        parents=np.asarray(body["parents"]), panel=panel, H=H, W=W, fps=fps)
        print(f"[uncertainty] wrote {ov}" + ("" if P == 1 else f"  (side panels = person {object_ids[0]})"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=None, help="original video / frames (needed for --overlay)")
    ap.add_argument("--predictions", required=True, help="out/<stem>/ clip directory")
    ap.add_argument("--bodies", default="kinematic_2",
                    choices=sorted(_BODY_TO_CONTACT),
                    help="which refined body to score (default: kinematic_2, "
                         "falls back to kinematic_1)")
    ap.add_argument("--overlay", action="store_true",
                    help="also write uncertainty_<tag>_overlay.mp4 (frame + contact skeleton "
                         "beside the panels with a sweeping playhead); requires --source")
    ap.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    return run(args.predictions, source=args.source, overlay=args.overlay,
               bodies=args.bodies, config_path=args.config, device=args.device)


if __name__ == "__main__":
    raise SystemExit(main())
