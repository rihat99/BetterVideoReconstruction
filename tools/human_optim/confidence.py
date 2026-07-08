"""Per-joint contact-label confidence for the scene-contact readout (round-5 T3).

Every joint in every frame gets a confidence in its FINAL binary label — whether
that label is *contact* or *no-contact* — from named, monotone signals the readout
already computes:

- **p_contact** — how contact-like the geometry + motion look: rises as the joint
  gets closer to the wall (``gsdf``) and stiller (pre-enforcement ``drift``). Firm
  *penetration* (``gsdf`` below the surface) overrides the stillness factor —
  penetration is the strongest possible contact evidence, so a hard grip that
  micro-drifts a few cm is no longer throttled down to ~0.5 (round-7 T4).
- **observability** — how much we actually saw the joint: scene points nearby
  (occlusion check), VGGT depth confidence there, joint in frame.
- **label_confidence** — confidence in the binary label. For a *contact*:
  ``persist · observability · p_contact``. For a *no-contact*: either the joint is
  moving fast (confident from motion alone, even occluded — you cannot grip while
  flying) or it is **clearly far** from the wall. "Far" has three witnesses, combined by
  ``max``: the **absolute** distance ``p_far`` (from the *unguarded* ``jsdf``, past the
  hover band), a **relative** clearance ``p_far_rel`` (round-8 T2a: how far the joint sits
  ABOVE the body's own closest joint — a whole-body depth error cancels, so it certifies
  free joints on a *floating* body where the absolute distance is corrupt), and a
  **nothing-within-reach** flag ``p_far_nan`` (round-8 T2c: a NaN distance on a
  non-degenerate scene). A joint just outside a visible wall (the hover-miss zone, ~3–25 cm)
  or slow + occluded ⇒ low confidence: the honest unknown. (``p_far`` reads the unguarded
  ``jsdf`` — finite out to ~1.6 m — because the guarded ``gsdf`` saturates at ``max_dist_m``
  and cannot tell 15 cm from 40 cm; and its threshold ``far_ref_m`` sits beyond any plausible
  hold protrusion, so a hand on a carved-out hold stays *uncertain* while a hand far in open
  space reads confidently free. This replaces round-5's ``obs · (1 − p_contact)``, which
  scored visible near-wall hovers as *confident* no-contact — the reliability curve's headline
  failure.) The **absolute** and **NaN** witnesses are scaled by a per-person **body_health**
  scalar (:func:`compute_body_health`): a floating body — one that almost never touches or
  hovers everywhere — has meaningless distances, so its "clearly far ⇒ confident no-contact"
  is muted toward 0 (round-7 T4); the **relative** clearance and the *motion* branch are left
  unmuted (relative clearance is immune to the whole-body error by construction).

One no-grad pass, consumed by :mod:`scripts.stages.estimate_contacts` after the
labels are final (post span-merge). Every factor stays in ``[0, 1]`` and monotone.
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
    far_ref_m=0.25,      # p_far = 0.5 at this UNGUARDED distance (jsdf); beyond any hold protrusion ⇒ carve-safe
    still_ref_m=0.03,    # p_still = 0 at drift ≥ 2× this
    pen_ref_m=-0.01,     # p_pen = 1 by this signed penetration depth (m): firm penetration overrides motion doubt
    obs_radius_m=0.15,   # scene points within this of a joint count as "seen"
    obs_min_pts=20,      # n_near for full observability
    obs_k=8,             # k nearest scene points averaged for conf_near
    conf_ref=0.3,        # VGGT conf for full observability
    v_lo=0.15,           # joint speed (m/s) below which motion says nothing
    v_hi=0.5,            # speed at which no-contact is motion-certain
    img_margin_px=10,    # in-frame margin (px) for the visibility gate
    health_cr=0.1,       # contact_rate below which a body reads as never-touching (floated) ⇒ mute far branch
    health_hover_cm=15.0,  # median closest-joint dist (cm) at which hover-health → 0
    nan_scene_min_pts=1000,  # a NaN distance counts as "far" only if the scene cloud has ≥ this many points
)


def _sat(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def _scene_neighbourhood(jw: torch.Tensor, scene_pts: torch.Tensor, scene_conf: torch.Tensor,
                         *, radius_m: float, k: int, chunk: int = 32):
    """Per-(frame,joint) ``(n_near, conf_near)``: scene points within ``radius_m`` of the joint,
    and the mean VGGT confidence of its ``k`` nearest scene points. ``jw (N,J,3)`` world."""
    N, J, _ = jw.shape
    M = int(scene_pts.shape[0])
    dev = jw.device
    n_near = torch.zeros((N, J), device=dev)
    conf_near = torch.zeros((N, J), device=dev)
    if M == 0:
        return n_near, conf_near
    # Bound the (chunk*J, M) cdist to ~1 GB: DB scene clouds reach ~4e5 points (~2.5 GB at chunk 32).
    # A smaller stride gives an identical result (n_near / conf_near are per-(frame,joint)) — lossless.
    chunk = max(1, min(chunk, int(1e9 // (J * M * 4))))
    kk = min(k, M)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        d = torch.cdist(jw[s:e].reshape(-1, 3), scene_pts)          # (f·J, M)
        n_near[s:e] = (d < radius_m).sum(-1).reshape(e - s, J).float()
        idx = d.topk(kk, dim=-1, largest=False).indices             # (f·J, kk) nearest
        conf_near[s:e] = scene_conf[idx].mean(-1).reshape(e - s, J)
    return n_near, conf_near


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
    wall are meaningless. It gates ONLY the far-from-wall no-contact confidence in
    :func:`contact_confidence` (never the motion branch — a fast joint is a confident
    no-contact even on a broken reconstruction). Product of two monotone factors:
    contact rate (share of valid frames with any joint in contact) and hover (median
    over frames of the closest joint's signed distance). ``jc (N,J)`` bool,
    ``jsdf_cm (N,J)`` signed cm, ``valid (N,)`` bool.
    """
    if valid.sum() < 1:
        return 1.0
    contact_rate = float(jc[valid].any(axis=1).mean())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                # all-NaN frame ⇒ NaN, handled below
        fmin = np.nanmin(jsdf_cm[valid], axis=1)                       # closest joint per frame (cm)
    med = float(np.nanmedian(fmin)) if not np.isnan(fmin).all() else 0.0
    # h_contact only penalises a NEAR-ZERO contact rate: in a climbing video a body that
    # essentially never touches is floated off the wall (the climber really is on it), so its
    # distances are meaningless. ``health_cr`` is low so a legitimate dynamic climb with a
    # moderate contact rate reads h_contact = 1 and is judged on hover alone.
    h_contact = min(max(contact_rate / max(float(cfg["health_cr"]), 1e-6), 0.0), 1.0)
    # h_hover falls as the closest joint hovers off the wall; a seated body (med ≤ 0) reads 1.
    h_hover = min(max(1.0 - med / max(float(cfg["health_hover_cm"]), 1e-6), 0.0), 1.0)
    return h_contact * h_hover


def contact_confidence(jc: torch.Tensor, gsdf: torch.Tensor, jsdf: torch.Tensor, drift: torch.Tensor,
                       jw: torch.Tensor, px: torch.Tensor,
                       scene_pts: torch.Tensor, scene_conf: torch.Tensor,
                       height: int, width: int, *, cfg: dict, window_s: float, fps: float,
                       body_health: float = 1.0):
    """Per-(frame,joint) confidence in the final contact label. All inputs are one person's.

    :param jc: ``(N,J)`` bool final contact label (post-merge).
    :param gsdf: ``(N,J)`` GUARDED per-joint min signed distance to the wall (m), saturated at
        ``max_dist_m`` — used for the contact-likeness ``p_close``.
    :param jsdf: ``(N,J)`` UNGUARDED per-joint min signed distance (m), finite out to ~1.6 m
        (``NaN`` = no near-surface vertex) — used for the far-from-wall ``p_far``.
    :param drift: ``(N,J)`` net window drift (m) — the SAME (pre-enforcement) source as the gate.
    :param jw: ``(N,J,3)`` world joint positions; ``px`` ``(N,J,2)`` their image projection.
    :param scene_pts: ``(M,3)`` / ``scene_conf`` ``(M,)`` the conf-gated scene cloud.
    :returns: ``(p_contact, observability, label_confidence)`` each ``(N,J)`` float32 numpy —
        ``label_confidence`` is conf(L=1) where the label is contact, conf(L=0) where it is not.
    """
    close_ref, ramp = float(cfg["close_ref_m"]), float(cfg["ramp_m"])
    far_ref, still_ref, pen_ref = float(cfg["far_ref_m"]), float(cfg["still_ref_m"]), float(cfg["pen_ref_m"])
    obs_min, conf_ref = float(cfg["obs_min_pts"]), float(cfg["conf_ref"])
    v_lo, v_hi = float(cfg["v_lo"]), float(cfg["v_hi"])
    radius, k, margin = float(cfg["obs_radius_m"]), int(cfg["obs_k"]), float(cfg["img_margin_px"])

    gsdf_f = torch.nan_to_num(gsdf, nan=float("inf"), posinf=float("inf"))
    p_close = _sat((close_ref + ramp - gsdf_f) / (2.0 * ramp))      # 0.5 at close_ref; +inf ⇒ 0
    jsdf_f = torch.nan_to_num(jsdf, nan=0.0, posinf=0.0, neginf=0.0)  # non-finite / penetrating ⇒ not far
    p_far = _sat((jsdf_f - far_ref + ramp) / (2.0 * ramp))          # mirror of p_close: 0.5 at far_ref, 1 by far_ref+ramp
    # Relative clearance (round-8 T2a): a joint's distance ABOVE the body's OWN closest joint.
    # A whole-body depth-placement error (a floating body) cancels in the difference, so this
    # witness needs NO health gate — a joint dangling far above a floater's own closest point
    # is provably free even though its absolute distance is corrupt. The closest level is
    # clamped at the surface (0) so a penetrating contact can't inflate the clearance (measured:
    # the signed min leaks +0.17 onto healthy scenes; the surface-clamped min leaks +0.00).
    finite = torch.isfinite(jsdf)
    jbig = torch.where(finite, jsdf, torch.full_like(jsdf, float("inf")))
    closest = torch.clamp(jbig.min(dim=1, keepdim=True).values, min=0.0)   # (N,1) surface-clamped
    rel = torch.nan_to_num(jsdf - closest, nan=0.0, posinf=0.0, neginf=0.0)  # NaN joint / all-NaN frame ⇒ 0
    p_far_rel = _sat((rel - far_ref + ramp) / (2.0 * ramp))        # same carve threshold as p_far
    # "Nothing within reach" (round-8 T2c): a NaN distance (no scene vertex within ~1.6 m) is the
    # strongest far evidence there is — but counts only when the scene is non-degenerate (else NaN
    # just means an empty cloud) and, like p_far, the body is healthy (a floater's distances lie).
    scene_ok = 1.0 if int(scene_pts.shape[0]) >= int(cfg["nan_scene_min_pts"]) else 0.0
    p_far_nan = (~finite).float() * scene_ok
    p_still = _sat(1.0 - drift / (2.0 * still_ref))                 # 1 at drift 0
    # p_pen rises with penetration (gsdf below the surface): firm penetration is the strongest
    # contact evidence, so motion doubt (p_still) must not throttle it. 0 at close_ref (the touch
    # band still needs stillness), 1 by pen_ref. max() ⇒ a still OR firmly-penetrating joint scores high.
    p_pen = _sat((close_ref - gsdf_f) / max(close_ref - pen_ref, 1e-6))
    p_contact = p_close * torch.maximum(p_still, p_pen)
    speed = drift / max(window_s, 1e-6)
    p_fast = _sat((speed - v_lo) / max(v_hi - v_lo, 1e-6))         # confident no-contact from motion

    n_near, conf_near = _scene_neighbourhood(jw, scene_pts, scene_conf, radius_m=radius, k=k)
    in_img = ((px[..., 0] >= margin) & (px[..., 0] < width - margin) &
              (px[..., 1] >= margin) & (px[..., 1] < height - margin)).float()
    obs = _sat(n_near / max(obs_min, 1e-6)) * _sat(conf_near / max(conf_ref, 1e-6)) * in_img

    persist = torch.as_tensor(_persistence(jc.detach().cpu().numpy().astype(bool),
                                           int(round(window_s * fps))), device=jw.device)
    conf_pos = persist * obs * p_contact                           # sure contact: seen, close, still/penetrating, persistent
    # No-contact = motion OR distance witness. body_health mutes the ABSOLUTE + NaN distance
    # witnesses on floating bodies (their far distances are meaningless); the RELATIVE-clearance
    # witness is health-free (it already cancels the floater's whole-body error); the motion
    # branch survives regardless — a fast joint is a confident no-contact even on a broken scene.
    far = torch.maximum(float(body_health) * torch.maximum(p_far, p_far_nan), p_far_rel)
    conf_neg = persist * torch.maximum(p_fast, in_img * far)
    label_conf = torch.where(jc.bool(), conf_pos, conf_neg)
    to_np = lambda t: t.detach().cpu().numpy().astype(np.float32)
    return to_np(p_contact), to_np(obs), to_np(label_conf)
