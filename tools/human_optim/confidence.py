"""Per-joint contact-label confidence for the scene-contact readout (round-5 T3).

Every joint in every frame gets a confidence in its FINAL binary label — whether
that label is *contact* or *no-contact* — from named, monotone signals the readout
already computes:

- **p_contact** — how contact-like the geometry + motion look: rises as the joint
  gets closer to the wall (``gsdf``) and stiller (pre-enforcement ``drift``).
- **observability** — how much we actually saw the joint: scene points nearby
  (occlusion check), VGGT depth confidence there, joint in frame.
- **label_confidence** — confidence in the binary label. For a *contact*:
  ``persist · observability · p_contact``. For a *no-contact*: either the joint is
  moving fast (confident from motion alone, even occluded — you cannot grip while
  flying) or it is **clearly far** from the wall past the ambiguous hover band
  (``p_far``, from the *unguarded* distance ``jsdf``). A joint just outside a visible
  wall (the hover-miss zone, ~3–8 cm) or slow + occluded ⇒ low confidence: the honest
  unknown. (``p_far`` reads the unguarded ``jsdf`` — finite out to ~1.6 m — because the
  guarded ``gsdf`` saturates at ``max_dist_m`` and cannot tell 15 cm from 40 cm; and
  its threshold ``far_ref_m`` sits beyond any plausible hold protrusion, so a hand on a
  carved-out hold stays *uncertain* while a hand far in open space reads confidently free.
  This replaces round-5's ``obs · (1 − p_contact)``, which scored visible near-wall
  hovers as *confident* no-contact — the reliability curve's headline failure.)

One no-grad pass, consumed by :mod:`scripts.stages.estimate_contacts` after the
labels are final (post span-merge). Every factor stays in ``[0, 1]`` and monotone.
"""
from __future__ import annotations

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
    obs_radius_m=0.15,   # scene points within this of a joint count as "seen"
    obs_min_pts=20,      # n_near for full observability
    obs_k=8,             # k nearest scene points averaged for conf_near
    conf_ref=0.3,        # VGGT conf for full observability
    v_lo=0.15,           # joint speed (m/s) below which motion says nothing
    v_hi=0.5,            # speed at which no-contact is motion-certain
    img_margin_px=10,    # in-frame margin (px) for the visibility gate
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


def contact_confidence(jc: torch.Tensor, gsdf: torch.Tensor, jsdf: torch.Tensor, drift: torch.Tensor,
                       jw: torch.Tensor, px: torch.Tensor,
                       scene_pts: torch.Tensor, scene_conf: torch.Tensor,
                       height: int, width: int, *, cfg: dict, window_s: float, fps: float):
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
    far_ref, still_ref = float(cfg["far_ref_m"]), float(cfg["still_ref_m"])
    obs_min, conf_ref = float(cfg["obs_min_pts"]), float(cfg["conf_ref"])
    v_lo, v_hi = float(cfg["v_lo"]), float(cfg["v_hi"])
    radius, k, margin = float(cfg["obs_radius_m"]), int(cfg["obs_k"]), float(cfg["img_margin_px"])

    gsdf_f = torch.nan_to_num(gsdf, nan=float("inf"), posinf=float("inf"))
    p_close = _sat((close_ref + ramp - gsdf_f) / (2.0 * ramp))      # 0.5 at close_ref; +inf ⇒ 0
    jsdf_f = torch.nan_to_num(jsdf, nan=0.0, posinf=0.0, neginf=0.0)  # non-finite / penetrating ⇒ not far
    p_far = _sat((jsdf_f - far_ref + ramp) / (2.0 * ramp))          # mirror of p_close: 0.5 at far_ref, 1 by far_ref+ramp
    p_still = _sat(1.0 - drift / (2.0 * still_ref))                 # 1 at drift 0
    p_contact = p_close * p_still
    speed = drift / max(window_s, 1e-6)
    p_fast = _sat((speed - v_lo) / max(v_hi - v_lo, 1e-6))         # confident no-contact from motion

    n_near, conf_near = _scene_neighbourhood(jw, scene_pts, scene_conf, radius_m=radius, k=k)
    in_img = ((px[..., 0] >= margin) & (px[..., 0] < width - margin) &
              (px[..., 1] >= margin) & (px[..., 1] < height - margin)).float()
    obs = _sat(n_near / max(obs_min, 1e-6)) * _sat(conf_near / max(conf_ref, 1e-6)) * in_img

    persist = torch.as_tensor(_persistence(jc.detach().cpu().numpy().astype(bool),
                                           int(round(window_s * fps))), device=jw.device)
    conf_pos = persist * obs * p_contact                           # sure contact: seen, close, still, persistent
    conf_neg = persist * torch.maximum(p_fast, in_img * p_far)     # sure no-contact: moving, OR in-frame + clearly far
    label_conf = torch.where(jc.bool(), conf_pos, conf_neg)
    to_np = lambda t: t.detach().cpu().numpy().astype(np.float32)
    return to_np(p_contact), to_np(obs), to_np(label_conf)
