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
