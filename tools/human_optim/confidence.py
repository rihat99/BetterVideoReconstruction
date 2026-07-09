"""Per-joint contact-label confidence for the scene-contact readout (round-5 T3; v3 round-11).

Every joint in every frame gets a confidence in its FINAL binary label — whether that
label is *contact* or *no-contact* — from named, monotone signals the readout already
computes. Each signal carries a one-line physical assumption (the governing design rule):

- **p_contact** — how contact-like the geometry + motion look: near the wall (``gsdf``) AND
  either still (pre-enforcement ``drift``) or firmly penetrating. *Assumption: a joint that
  sits on the surface and does not slide is holding it.*
- **vis** (round-11) — did the camera actually see this joint: not hidden behind other body
  parts (geometric self-occlusion), the Sapiens detector confident about it, and in frame.
  *Assumption: an unseen or unpinned joint has an unverified position, and an unverified
  position gives an unverified contact.* Multiplies the SEEN-dependent evidence on both
  classes (the contact side and the distance free-witness), never the motion witness — you
  cannot grip while flying whether or not we saw the joint.
- **the free distance witness** — how far the joint is from everything. Relative first
  (round-11): clearance ABOVE the scene sheet the body rests on (the touching joints if it
  touches, else its own closest joint), so a whole-body depth error cancels; the absolute
  distance + a nothing-within-reach flag are the health-gated fallback for a body that
  touches nothing. *Assumption: clearance above where the body meets the scene is trustworthy;
  clearance without such an anchor inherits the body's unknown offset.*
- **p_fast** — motion alone: a fast joint is a confident no-contact even if unseen or on a
  broken reconstruction. *Assumption: you cannot grip while flying.*
- **persistence** — neighbouring frames agree with this label.
- **body_health** — a floating body (never touches / hovers everywhere) has meaningless
  absolute distances, so its absolute + nothing-within-reach witnesses are muted; the
  relative witness and the motion witness are not.

One no-grad pass, consumed by :mod:`scripts.stages.estimate_contacts` after the labels are
final (post span-merge). Every factor stays in ``[0, 1]`` and monotone.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch

#: Readout-only knobs, shared by ``estimate_contacts`` (writes the fields) and the scorer's
#: ``--reliability`` (recomputes them). NOT registered in ``config.py`` — the optimizer never
#: sees these. Every factor stays monotone; see the per-signal docstring above.
CONTACT_CONFIDENCE_DEFAULTS = dict(
    close_ref_m=0.03,    # p_contact = 0.5 at this signed distance to the wall
    ramp_m=0.03,         # close/far sat ramp half-width (m)
    far_ref_m=0.25,      # p_far = 0.5 at this distance; beyond any hold protrusion ⇒ carve-safe
    still_ref_m=0.03,    # p_still = 0 at drift ≥ 2× this
    pen_ref_m=-0.01,     # p_pen = 1 by this signed penetration depth (m): firm penetration overrides motion doubt
    v_lo=0.15,           # joint speed (m/s) below which motion says nothing
    v_hi=0.5,            # speed at which no-contact is motion-certain
    img_margin_px=10,    # in-frame margin (px) for the visibility gate
    occ_ref=8.0,         # vis: n_occ (occluding body verts) at which the occlusion factor hits 0
    det_lo=0.4,          # vis: Sapiens score at/below which the detector factor is 0
    det_hi=0.75,         # vis: Sapiens score at/above which the detector factor is 1
    vis_floor=0.1,       # vis floor — never hard-zero a class on visibility alone
    health_cr=0.1,       # contact_rate below which a body reads as never-touching (floated) ⇒ mute far branch
    health_hover_cm=15.0,  # median closest-joint dist (cm) at which hover-health → 0
    nan_scene_min_pts=1000,  # a NaN distance counts as "far" only if the scene cloud has ≥ this many points
)

# Coarse annotation joints → SMPLXMid joint names (mirrors stages._ANN_TO_SMPLX; kept local so
# confidence.py need not import the optimizer stack). Order matches the scorer's _COARSE.
_FINGER = ("index", "middle", "pinky", "ring", "thumb")
_COARSE_GROUPS: dict[str, list[str]] = {
    "left_hand": ["left_wrist"] + [f"left_{f}{k}" for f in _FINGER for k in (1, 2, 3)],
    "right_hand": ["right_wrist"] + [f"right_{f}{k}" for f in _FINGER for k in (1, 2, 3)],
    "left_foot": ["left_ankle", "left_foot"], "right_foot": ["right_ankle", "right_foot"],
    "left_knee": ["left_knee"], "right_knee": ["right_knee"],
    "left_hip": ["left_hip"], "right_hip": ["right_hip"],
    "left_elbow": ["left_elbow"], "right_elbow": ["right_elbow"],
}
# Coarse joint → Goliath-308 keypoint indices (keypoints.py convention: body block has no
# wrists; wrists are in the hand blocks, left_wrist=62 / right_wrist=41).
COARSE_TO_GOLIATH: dict[str, list[int]] = {
    "left_hand": [62, 42, 46, 50, 54, 58], "right_hand": [41, 21, 25, 29, 33, 37],
    "left_foot": [13, 15, 16, 17], "right_foot": [14, 18, 19, 20],
    "left_knee": [11], "right_knee": [12], "left_hip": [9], "right_hip": [10],
    "left_elbow": [7], "right_elbow": [8],
}


def coarse_of_joint_ids(mid_names: list[str]) -> np.ndarray:
    """(J,) coarse-group id per SMPLXMid joint (-1 = in no annotated coarse group)."""
    idx = {n: i for i, n in enumerate(mid_names)}
    out = np.full(len(mid_names), -1, np.int32)
    for cid, (_, joints) in enumerate(_COARSE_GROUPS.items()):
        for jn in joints:
            if jn in idx:
                out[idx[jn]] = cid
    return out


def joint_sapiens_score(scores: np.ndarray, mid_names: list[str]) -> np.ndarray:
    """(N,J) per-joint Sapiens detector score — max over the joint's mapped Goliath-308 keypoints.

    ``scores`` is one person's ``(N, 308)`` keypoint scores. Joints in no coarse group (torso,
    spine, head) carry no keypoint → ``nan`` (the detector factor stays neutral for them).
    """
    N = scores.shape[0]
    idx = {n: i for i, n in enumerate(mid_names)}
    out = np.full((N, len(mid_names)), np.nan, np.float32)
    for coarse, joints in _COARSE_GROUPS.items():
        gk = COARSE_TO_GOLIATH[coarse]
        s = scores[:, gk].max(axis=1)                                   # (N,) best mapped keypoint
        for jn in joints:
            if jn in idx:
                out[:, idx[jn]] = s
    return out


def _sat(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def _persistence(labels: np.ndarray, window: int) -> np.ndarray:
    """Fraction of frames within ±``window`` sharing each ``(frame,joint)``'s label. ``(N,J)``."""
    l = labels.astype(np.float64)
    N, J = l.shape
    csum = np.concatenate([np.zeros((1, J)), np.cumsum(l, axis=0)], 0)   # (N+1, J)
    n = np.arange(N)
    lo = np.clip(n - window, 0, N)
    hi = np.clip(n + window + 1, 0, N)
    frac_ones = (csum[hi] - csum[lo]) / np.maximum((hi - lo)[:, None], 1)
    return np.where(l > 0.5, frac_ones, 1.0 - frac_ones).astype(np.float32)


def compute_body_health(jc: np.ndarray, jsdf_cm: np.ndarray, valid: np.ndarray, *, cfg: dict) -> float:
    """Per-person body-health scalar in ``[0, 1]`` from the final labels + closest-joint distance.

    ``1`` for a body that touches the wall in a healthy share of frames and sits on it;
    ``→0`` for a never-touch / always-hovering (floating) body, whose distances to the
    wall are meaningless. It gates ONLY the absolute + nothing-within-reach far witnesses in
    :func:`contact_confidence` (never the motion branch, never the relative-clearance witness).
    Product of two monotone factors: contact rate (share of valid frames with any joint in
    contact) and hover (median over frames of the closest joint's signed distance).
    """
    if valid.sum() < 1:
        return 1.0
    contact_rate = float(jc[valid].any(axis=1).mean())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                # all-NaN frame ⇒ NaN, handled below
        fmin = np.nanmin(jsdf_cm[valid], axis=1)                       # closest joint per frame (cm)
    med = float(np.nanmedian(fmin)) if not np.isnan(fmin).all() else 0.0
    h_contact = min(max(contact_rate / max(float(cfg["health_cr"]), 1e-6), 0.0), 1.0)
    h_hover = min(max(1.0 - med / max(float(cfg["health_hover_cm"]), 1e-6), 0.0), 1.0)
    return h_contact * h_hover


def contact_confidence(jc: torch.Tensor, gsdf: torch.Tensor, jsdf: torch.Tensor, drift: torch.Tensor,
                       jw: torch.Tensor, px: torch.Tensor, scene_pts: torch.Tensor,
                       height: int, width: int, *, n_occ: torch.Tensor, sap_score: torch.Tensor,
                       cfg: dict, window_s: float, fps: float, body_health: float = 1.0):
    """Per-(frame,joint) confidence in the final contact label. All inputs are one person's.

    :param jc: ``(N,J)`` bool final contact label (post-merge).
    :param gsdf: ``(N,J)`` GUARDED per-joint min signed distance to the wall (m), saturated at
        ``max_dist_m`` — the contact-likeness ``p_close``.
    :param jsdf: ``(N,J)`` UNGUARDED per-joint min signed distance (m), finite out to ~1.6 m
        (``NaN`` = no near-surface vertex) — the far-from-wall witnesses.
    :param drift: ``(N,J)`` net window drift (m) — the SAME (pre-enforcement) source as the gate.
    :param jw: ``(N,J,3)`` world joint positions; ``px`` ``(N,J,2)`` their image projection.
    :param scene_pts: ``(M,3)`` the conf-gated scene cloud (degeneracy gate for the NaN witness).
    :param n_occ: ``(N,J)`` count of body vertices occluding each joint (visibility witness).
    :param sap_score: ``(N,J)`` per-joint Sapiens detector score (``nan`` where no keypoint maps).
    :returns: ``(p_contact, vis, label_confidence)`` each ``(N,J)`` float32 numpy —
        ``label_confidence`` is conf(L=1) where the label is contact, conf(L=0) where it is not.
    """
    close_ref, ramp = float(cfg["close_ref_m"]), float(cfg["ramp_m"])
    far_ref, still_ref, pen_ref = float(cfg["far_ref_m"]), float(cfg["still_ref_m"]), float(cfg["pen_ref_m"])
    v_lo, v_hi = float(cfg["v_lo"]), float(cfg["v_hi"])
    margin = float(cfg["img_margin_px"])
    occ_ref, det_lo, det_hi = float(cfg["occ_ref"]), float(cfg["det_lo"]), float(cfg["det_hi"])
    vis_floor = float(cfg["vis_floor"])
    dev = jw.device

    # ── contact-likeness (geometry + motion) ──
    gsdf_f = torch.nan_to_num(gsdf, nan=float("inf"), posinf=float("inf"))
    p_close = _sat((close_ref + ramp - gsdf_f) / (2.0 * ramp))      # 0.5 at close_ref; +inf ⇒ 0
    p_still = _sat(1.0 - drift / (2.0 * still_ref))                 # 1 at drift 0
    p_pen = _sat((close_ref - gsdf_f) / max(close_ref - pen_ref, 1e-6))  # firm penetration overrides motion doubt
    p_contact = p_close * torch.maximum(p_still, p_pen)
    speed = drift / max(window_s, 1e-6)
    p_fast = _sat((speed - v_lo) / max(v_hi - v_lo, 1e-6))          # confident no-contact from motion

    # ── free distance witness: RELATIVE FIRST (round-11 T3b) ──
    jsdf_f = torch.nan_to_num(jsdf, nan=0.0, posinf=0.0, neginf=0.0)  # non-finite / penetrating ⇒ not far
    p_far = _sat((jsdf_f - far_ref + ramp) / (2.0 * ramp))          # absolute distance (health-gated below)
    finite = torch.isfinite(jsdf)
    # Anchor = the scene sheet the body rests on: its own closest joint, surface-clamped to 0.
    # The clamp is load-bearing — a penetrating (negative) closest joint would otherwise INFLATE
    # every other joint's clearance and manufacture confidently-free labels (measured: the
    # unclamped / touching-joint anchor raised the confidently-wrong miss rate). Clamped-min is
    # the round-8 quantity and the only anchor that also exists on a floating body (nothing
    # touching), which is what recovers a floater's genuinely-free joints.
    jbig = torch.where(finite, jsdf, torch.full_like(jsdf, float("inf")))
    closest = torch.clamp(jbig.min(dim=1, keepdim=True).values, min=0.0)   # (N,1) surface-clamped
    rel = torch.nan_to_num(jsdf - closest, nan=0.0, posinf=0.0, neginf=0.0)
    p_far_rel = _sat((rel - far_ref + ramp) / (2.0 * ramp))        # same carve threshold as p_far
    # "Nothing within reach": a NaN distance (no scene vertex within ~1.6 m) is the strongest far
    # evidence — but counts only on a non-degenerate scene, and (like p_far) health-gated.
    scene_ok = 1.0 if int(scene_pts.shape[0]) >= int(cfg["nan_scene_min_pts"]) else 0.0
    p_far_nan = (~finite).float() * scene_ok
    # relative-first: p_far_rel un-gated primary; absolute + nan the health-gated fallback.
    far = torch.maximum(p_far_rel, float(body_health) * torch.maximum(p_far, p_far_nan))

    # ── visibility witness (round-11 T3a): did the camera see + the detector pin this joint? ──
    # Two factors: geometric self-occlusion (n_occ body verts hiding the joint from the camera)
    # and Sapiens detector belief (keypoint score). They enter the two classes ASYMMETRICALLY,
    # by which failure each predicts (measured — full derivation in docs/contact_confidence.md):
    #   • CONTACT side: a low-score joint we call contact is less precise (FP 0.18 vs 0.10) ⇒
    #     detector penalty f_det. Occlusion is NOT penalised here — a deeply-occluded predicted
    #     contact is usually a firm grip pressed behind the body, not a false one.
    #   • FREE side: an occluded / low-score joint we call FREE may be a MISSED hidden contact
    #     (occluded annotated contacts miss at 66% vs 25%) ⇒ its no-contact label is less certain
    #     ⇒ occlusion AND detector penalty f_occ·f_det on the distance witness. This is where the
    #     confidently-wrong misses live, so this is where visibility must reach. It never touches
    #     the motion witness — a fast joint is a confident no-contact whether or not we saw it.
    f_occ = _sat(1.0 - n_occ / max(occ_ref, 1e-6))                # 1 fully visible, 0 by occ_ref occluders
    sd = torch.nan_to_num(sap_score, nan=1.0)                     # unmapped joint ⇒ neutral (1)
    f_det = torch.where(torch.isnan(sap_score), torch.ones_like(sd), _sat((sd - det_lo) / max(det_hi - det_lo, 1e-6)))
    in_img = ((px[..., 0] >= margin) & (px[..., 0] < width - margin) &
              (px[..., 1] >= margin) & (px[..., 1] < height - margin)).float()
    vis_contact = torch.clamp(in_img * f_det, min=vis_floor)                # detector only
    vis_free = torch.clamp(in_img * f_occ * f_det, min=vis_floor)           # occlusion + detector

    persist = torch.as_tensor(_persistence(jc.detach().cpu().numpy().astype(bool),
                                           int(round(window_s * fps))), device=dev)
    conf_pos = persist * vis_contact * p_contact                   # sure contact: trusted keypoint, close, still/penetrating
    conf_neg = persist * torch.maximum(p_fast, vis_free * far)     # motion OR (seen-and-far); occluded free ⇒ less sure
    vis = torch.where(jc.bool(), vis_contact, vis_free)            # the visibility applied to this joint's label
    label_conf = torch.where(jc.bool(), conf_pos, conf_neg)
    to_np = lambda t: t.detach().cpu().numpy().astype(np.float32)
    return to_np(p_contact), to_np(vis), to_np(label_conf)
