"""Per-clip reconstruction-quality signals and the calibrated failure flags.

Two failure modes, two flags (thresholds in ``[quality]`` of the human-optimization
TOML; calibrated 2026-08-26 against the user's verdict on 23 labelled clips —
high_jump / -bUyE / 1FGkg / pommel_horse bad, everything else including olympics_2
and balance_beam passes):

* ``camera`` — VGGT hallucinated camera travel (a tripod pan read as a dolly).
  Needs BOTH witnesses: a large camera-translation-to-scene-depth ratio AND a
  body carrying acceleration no contact force can explain. The ratio alone
  cannot tell a real handheld move (boulder_1: 0.71, physics coherent at 1 BW)
  from a fake dolly (1FGkg: 0.68 with 2.8 BW inexplicable) — a hallucinated
  dolly corrupts the world *and* the trajectory, a real one corrupts neither.
* ``depth`` — the human depth cloud is smeared along the view ray
  (:mod:`tools.geometry.depth_smear`; the threshold is the smear weight's own
  dead-zone edge, one number with one meaning).

Computed and saved by ``estimate_contacts`` (the earliest stage where all three
signals exist — the residual comes from its force witness); the
``scripts/diagnostics/scene_quality.py`` table imports the same functions.
"""

from __future__ import annotations

import numpy as np


def cam_motion_ratio(extrinsics: np.ndarray, scene_points: np.ndarray) -> float:
    """Camera translation spread / median scene-to-camera distance.

    A hallucinated dolly reads ≫ 0 (high_jump: 1.3); a static camera ~0. Scale
    invariant — both lengths live in the same (possibly up-to-scale) world.
    """
    ext = np.asarray(extrinsics, np.float64)
    R, t = ext[:, :3, :3], ext[:, :3, 3]
    cam = -np.einsum("nij,ni->nj", R, t)
    spread = float(np.linalg.norm(cam - cam.mean(0), axis=1).max()) * 2.0
    depth = float(np.median(np.linalg.norm(
        np.asarray(scene_points, np.float64) - cam.mean(0), axis=1)))
    return spread / max(depth, 1e-9)


def smear_stats(human_points: np.ndarray, frame_id: np.ndarray,
                extrinsics: np.ndarray) -> tuple[float, float]:
    """(median smear, fraction of measurable frames above 2.5) over the clip."""
    from .depth_smear import alongray_smear

    ext = np.asarray(extrinsics, np.float64)
    R, t = ext[:, :3, :3], ext[:, :3, 3]
    cam = -np.einsum("nij,ni->nj", R, t)
    pts, fr = np.asarray(human_points), np.asarray(frame_id)
    order = np.argsort(fr, kind="stable")
    pts, fr = pts[order], fr[order]
    bounds = np.searchsorted(fr, np.arange(ext.shape[0] + 1))
    smears = [alongray_smear(pts[bounds[i]:bounds[i + 1]], cam[i])
              for i in range(ext.shape[0])]
    smears = np.asarray([s for s in smears if np.isfinite(s)])
    if not smears.size:
        return float("nan"), float("nan")
    return float(np.median(smears)), float((smears > 2.5).mean())


def quality_flag(cam_ratio: float, res_before_bw: float, smear_med: float, *,
                 cam_depth_max: float, res_before_bw_max: float,
                 smear_med_max: float) -> str:
    """``"camera"`` / ``"depth"`` / ``"camera+depth"`` / ``""`` (pass).

    NaN in any signal never raises a flag — a missing witness is not evidence
    of failure.
    """
    flags = []
    if cam_ratio > cam_depth_max and res_before_bw > res_before_bw_max:
        flags.append("camera")
    if smear_med > smear_med_max:
        flags.append("depth")
    return "+".join(flags)
