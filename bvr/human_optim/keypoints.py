"""SMPL-X-mid ↔ Sapiens **Goliath-308** keypoint correspondence.

Sapiens here is the ``keypoints308`` (``goliath_308``) model (``pose.npz``'s
``keypoint_scheme == 'goliath_308'``), **not** COCO-wholebody — the two layouts
diverge at index 9. Goliath-308 has **no wrists in the body block**; its body /
foot block is::

    0 nose  1 left_eye  2 right_eye  3 left_ear  4 right_ear
    5 left_shoulder  6 right_shoulder  7 left_elbow  8 right_elbow
    9 left_hip  10 right_hip  11 left_knee  12 right_knee
    13 left_ankle  14 right_ankle
    15 left_big_toe 16 left_small_toe 17 left_heel
    18 right_big_toe 19 right_small_toe 20 right_heel

then per-hand blocks of 21 (fingertip→…→wrist): right hand 21..41
(``right_wrist == 41``), left hand 42..62 (``left_wrist == 62``); fingertips are
``*_thumb4 / *_forefinger4 / *_middle_finger4 / *_ring_finger4 / *_pinky_finger4``.
(COCO-wholebody, which an earlier version wrongly assumed, instead puts wrists at
9/10 and shifts hips→13/14, ankles→15/16, feet→17..22 — mapping SMPL-X wrists to
hips, hips to knees, etc.) Indices verified against
``third_party/sapiens2/.../configs/_base_/keypoints308.py``.

We match those against SMPL-X-mid **joints** (shoulders/elbows/wrists/hips/
knees/ankles — joint centres) and **landmark frames** (nose/eyes/ears, toes/
heels, optional fingertips — surface vertices the model already exposes via
``frame_vertex_ids``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.signal import savgol_filter

# (SMPL-X-mid joint index, Goliath-308 index) — joint centres.
_BODY_PAIRS: tuple[tuple[int, int], ...] = (
    (16, 5), (17, 6),    # shoulders
    (18, 7), (19, 8),    # elbows
    (20, 62), (21, 41),  # wrists (Goliath left_wrist=62, right_wrist=41)
    (1, 9), (2, 10),     # hips
    (4, 11), (5, 12),    # knees
    (7, 13), (8, 14),    # ankles
)

# (SMPL-X-mid frame name, Goliath-308 index) — head + feet surface landmarks.
_LANDMARK_PAIRS: tuple[tuple[str, int], ...] = (
    ("nose", 0), ("leye", 1), ("reye", 2), ("lear", 3), ("rear", 4),
    ("LBigToe", 15), ("LSmallToe", 16), ("LHeel", 17),
    ("RBigToe", 18), ("RSmallToe", 19), ("RHeel", 20),
)

# (SMPL-X-mid frame name, Goliath-308 index) — fingertips (opt-in via include_hands).
_HAND_PAIRS: tuple[tuple[str, int], ...] = (
    ("lthumb", 42), ("lindex", 46), ("lmiddle", 50), ("lring", 54), ("lpinky", 58),
    ("rthumb", 21), ("rindex", 25), ("rmiddle", 29), ("rring", 33), ("rpinky", 37),
)


@dataclass
class Correspondence:
    """Index tensors selecting matched SMPL-X-mid / Sapiens keypoints.

    ``joint_model`` indexes ``joints_world`` (joint centres); ``lm_frame``
    indexes the model's ``frame_names`` list (→ ``frame_vertex_ids`` → posed
    vertices). ``sap`` is the concatenated Sapiens index for [joints, landmarks]
    in that order.
    """

    joint_model: torch.Tensor   # (Kb,) long
    lm_frame: torch.Tensor      # (Kl,) long
    sap: torch.Tensor           # (Kb + Kl,) long


def build_correspondence(
    frame_names: list[str],
    *,
    include_hands: bool = False,
    device: str | torch.device = "cpu",
) -> Correspondence:
    """Resolve the static pairs against this model's ``frame_names``."""
    name_to_frame = {n: i for i, n in enumerate(frame_names)}

    lm_pairs = list(_LANDMARK_PAIRS) + (list(_HAND_PAIRS) if include_hands else [])

    joint_model = [j for j, _ in _BODY_PAIRS]
    lm_frame = [name_to_frame[n] for n, _ in lm_pairs]
    sap = [s for _, s in _BODY_PAIRS] + [s for _, s in lm_pairs]

    long = lambda x: torch.tensor(x, dtype=torch.long, device=device)
    return Correspondence(joint_model=long(joint_model), lm_frame=long(lm_frame), sap=long(sap))


def visibility_weights(
    scores: torch.Tensor,        # (T, 308) Sapiens per-keypoint detection scores
    corr: Correspondence,
    parents: list[int],          # SMPL-X-mid parent joint index per joint (-1 at the root)
    *,
    strength: float,
    window_s: float,
    fps: float,
    frame_valid: torch.Tensor | None = None,   # (T,) bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-keypoint and per-joint **visibility** weights in ``[0, 1]``.

    Sapiens reports a detection *score* per keypoint, not a visibility flag — but a
    joint the camera cannot see is hard to detect, so its score drops. Measured on four
    scenes against an exact camera-ray body-occlusion cast: mean score **0.905 when the
    joint is clear vs 0.614 when a body surface stands in front of it**.

    The score is smoothed over time and divided by the keypoint's own 90th percentile
    over the clip, capped at 1. That percentile is what makes the number mean *visibility*
    rather than *detector calibration*: Sapiens scores hips at ~0.65 and wrists at ~0.95
    even when both are in plain view, so a raw score would permanently distrust the hips.
    Dividing by the keypoint's own clear-view level leaves only its **relative** drop,
    which is the occlusion.

    ``strength`` then scales that drop: ``weight = 1 − strength · (1 − ratio)``, clamped
    to ``[0, 1]``. It is the one knob. ``0`` is off (every weight 1). ``1`` charges the
    drop as measured. Above 1 the *shortfall* is amplified while a clearly-seen keypoint —
    whose ratio is already near 1 — barely moves, so raising it separates the occluded
    joints from the visible ones instead of quietly weakening every term at once.

    A model joint takes the weight of its matched keypoint. The joints with no keypoint
    (spine, neck, collars, head, feet, the 30 fingers) **inherit their parent's** weight;
    SMPL-X-mid lists parents before children, so one forward pass resolves the chain and
    the root starts at 1. So a foot follows its ankle and the fingers follow their wrist,
    while the torso — which no keypoint watches directly — stays at 1.

    Smoothing runs within runs of consecutive valid frames (:func:`smooth_tracks`). An
    untracked frame carries no observation at all, so its weight stays 1: the priors are
    the only thing holding those frames and must not be loosened there.

    Returns ``(keypoint_weights (T, 308), joint_weights (T, J))``.
    """
    num_frames, num_keypoints = int(scores.shape[0]), int(scores.shape[1])
    valid = (frame_valid if frame_valid is not None
             else torch.ones(num_frames, dtype=torch.bool, device=scores.device))
    valid_np = valid.detach().cpu().numpy()
    smoothed = scores.detach().cpu().numpy()
    if window_s > 0.0:
        mask = np.broadcast_to(valid_np[:, None], smoothed.shape)
        smoothed = smooth_tracks(smoothed[..., None], mask, window_s=window_s, fps=fps)[..., 0]
    clear_level = (np.quantile(smoothed[valid_np], 0.9, axis=0) if valid_np.any()
                   else np.ones(num_keypoints, smoothed.dtype))
    ratio = torch.as_tensor(smoothed / np.maximum(clear_level, 1e-3)[None, :],
                            dtype=scores.dtype, device=scores.device).clamp(max=1.0)
    keypoint = (1.0 - strength * (1.0 - ratio)).clamp(0.0, 1.0)
    keypoint = torch.where(valid[:, None], keypoint, torch.ones_like(keypoint))

    joint = torch.ones((num_frames, len(parents)), dtype=scores.dtype, device=scores.device)
    joint[:, corr.joint_model] = keypoint[:, corr.sap[:corr.joint_model.numel()]]
    covered = set(int(j) for j in corr.joint_model.tolist())
    for j, parent in enumerate(parents):
        if j not in covered and parent >= 0:
            joint[:, j] = joint[:, parent]
    return keypoint, joint


def smooth_tracks(x: np.ndarray, valid: np.ndarray, *, window_s: float, fps: float,
                  order: int = 2) -> np.ndarray:
    """Savitzky–Golay-smooth 2D tracks within runs of consecutive valid frames (round-4 §7).

    ``x`` ``(T, K, C)`` and ``valid`` ``(T, K)`` bool → a smoothed **copy** of ``x``. A local
    degree-``order`` polynomial fit over ~``window_s`` seconds (an odd frame window) is applied
    **independently within each maximal run of consecutive valid frames** of each track ``k`` —
    never across a tracking gap, so a keypoint dropping below score (or an untracked frame) does
    not bleed a held value into its neighbours. Runs shorter than ``order + 1`` frames are left
    unchanged; a run shorter than the full window uses a shrunk odd window; ends of a run use SG's
    polynomial edge interpolation. ``order = 2`` preserves quadratic (accelerating / dyno) motion,
    so real fast moves survive while per-frame jitter is removed. ``window_s <= 0`` is a no-op.

    These are *measurements* (fixed optimizer inputs); smoothing one is not the round-1 circularity
    (nothing optimized feeds back into ``x``). The caller keeps the raw tensor for the gate metric.
    """
    if window_s <= 0.0:
        return x
    T = x.shape[0]
    w_full = int(round(float(window_s) * float(fps)))
    if w_full % 2 == 0:
        w_full += 1
    if w_full <= order:
        return x
    out = x.copy()
    for k in range(x.shape[1]):
        vk = valid[:, k]
        t = 0
        while t < T:
            if not vk[t]:
                t += 1
                continue
            s = t
            while t < T and vk[t]:
                t += 1
            L = t - s
            if L <= order:
                continue                                  # too short to fit the polynomial
            w = min(w_full, L if L % 2 == 1 else L - 1)
            if w <= order:
                continue
            out[s:t, k] = savgol_filter(x[s:t, k], window_length=w, polyorder=order,
                                        axis=0, mode="interp")
    return out
