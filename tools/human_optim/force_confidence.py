"""Whole-body force confidence — how much to trust a clip's solved contact forces.

Contact forces are solved for the whole body at once (every limb's force balances
the same root residual), so their trustworthiness is a property of the body per
frame — never of one joint. This module scores it from the physics alone:

* **linear residual** — the unexplained root FORCE after the solve, in body
  weights. The dominant signal: if the solve could not balance the linear
  motion, the forces it did assign are allocations of garbage.
* **angular residual** — the unexplained root TORQUE, divided by body weight ×
  a characteristic lever (same convention as the solves). Weighted much more
  gently: the contact cones have little moment authority, so a nonzero angular
  residual is normal even on sequences whose forces are good.
* **overload** — total assigned force far beyond what a human supports.
  Sustained support runs ~1 BW (short landing peaks a few BW); a solve pushing
  several body weights frame after frame is explaining reconstruction error,
  not motion.

Each ingredient maps through the same saturating curve ``1 / (1 + (x / x*)²)``
and the per-frame confidence is their product — in ``(0, 1]``, 1 = fully
trustworthy. The clip score is the mean over valid frames. The constants below
are calibrated on the processed clips (2026-08-24): healthy climbing sits at
linear residual ~0.1–0.5 BW after the solve, the broken data_2 clips at 2–8 BW.

Scene-side evidence (how grounded the contact labels are, scene quality) is
deliberately NOT folded in here — join it per clip from ``contacts_1.npz`` /
``scene_quality`` diagnostics instead, so each score keeps one meaning.
"""

from __future__ import annotations

import numpy as np

#: Characteristic lever (m) turning root torque into force units — matches
#: :data:`tools.human_optim.forces.TORQUE_LENGTH_M`.
LEVER_M = 0.2
#: Linear root residual (body weights) at which confidence halves.
LIN_STAR_BW = 0.5
#: Angular root residual (body weights × lever) at which confidence halves —
#: deliberately looser than the linear star (cones have little moment authority).
ANG_STAR_BW = 1.5
#: Total assigned force (body weights) beyond which the overload term engages.
LOAD_CAP_BW = 3.0
#: Overload excess (body weights above the cap) at which confidence halves.
LOAD_STAR_BW = 1.0


def _saturate(x: np.ndarray, star: float) -> np.ndarray:
    return 1.0 / (1.0 + np.square(x / star))


def force_confidence(
    base_wrench: np.ndarray,    # (N, 6) root residual after the solve (N / N·m)
    frame_forces: np.ndarray,   # (N, F, 3) solved world contact forces (N)
    valid: np.ndarray,          # (N,) bool — tracked frames
    body_weight_n: float,
) -> dict[str, np.ndarray | float]:
    """Per-frame whole-body force confidence + its ingredients.

    Returns ``confidence (N,)`` in ``[0, 1]`` (0 on invalid frames), the raw
    ingredient series ``lin_residual_bw`` / ``ang_residual_bw`` / ``load_bw``
    (each ``(N,)``), and the scalar ``clip`` — the mean confidence over valid
    frames (NaN if none).
    """
    valid = np.asarray(valid, bool)
    bw = max(float(body_weight_n), 1e-6)
    lin = np.linalg.norm(base_wrench[:, :3], axis=-1) / bw
    ang = np.linalg.norm(base_wrench[:, 3:6], axis=-1) / (bw * LEVER_M)
    load = np.linalg.norm(frame_forces, axis=-1).sum(-1) / bw
    overload = np.maximum(0.0, load - LOAD_CAP_BW)
    conf = (_saturate(lin, LIN_STAR_BW) * _saturate(ang, ANG_STAR_BW)
            * _saturate(overload, LOAD_STAR_BW))
    conf = np.where(valid, conf, 0.0).astype(np.float32)
    clip = float(conf[valid].mean()) if valid.any() else float("nan")
    return dict(confidence=conf, lin_residual_bw=lin.astype(np.float32),
                ang_residual_bw=ang.astype(np.float32), load_bw=load.astype(np.float32),
                clip=clip)
