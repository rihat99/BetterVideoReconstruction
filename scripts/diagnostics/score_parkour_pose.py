"""Procrustes MPJPE against the LAAS Parkour ground-truth joint positions.

The dataset ships Vicon joint positions for 16 limb joints — hips, knees, ankles,
toes, shoulders, elbows, wrists, fingers — in the MoCap world frame, one row per
video frame. Its own evaluator (``lib/parkour_evaluator.py``, Python 2) scores a
prediction by solving an orthogonal Procrustes problem **per frame** — rotation,
translation *and* scale — and averaging the residual joint distances in
millimetres. This reproduces that protocol exactly, so the numbers are comparable
to the ones published on this data.

Because the alignment absorbs a rigid transform and a uniform scale, the frame a
prediction lives in does not matter: the SAM-3D-Body initialisation is scored in
its camera frame and the refined bodies in the world frame, and both are fair.

Every body is re-posed with the same pip ``smplx`` neutral model so the rows
compare like with like, and the 16 joints are resolved by NAME on both sides —
our saved bodies drop the jaw and eyes, so the two layouts do not share indices.

One caveat the per-joint table makes visible: MoCap hip and shoulder centres do
not sit where SMPL-X puts those joints (our thigh and upper arm measure ~20 %
shorter than the Vicon skeleton's, while the distal segments agree to a few per
cent). That gap is a floor under every SMPL-family method scored on this data.

Usage:
    uv run --no-sync python scripts/diagnostics/score_parkour_pose.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import smplx
import torch
from smplx.joint_names import JOINT_NAMES

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools import paths  # noqa: E402

PARKOUR = (paths.PARKOUR / "processed").resolve()
MODELS = _REPO.parent / "BetterHuman" / "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#: Each Parkour joint and the SMPL-X joint standing in for it. ``toes`` is the
#: ball of the foot (SMPL-X calls it ``*_foot``); ``fingers`` is the middle
#: knuckle, the central one of the four metacarpal joints.
CORRESPONDENCE = {
    "l_hip": "left_hip", "l_knee": "left_knee", "l_ankle": "left_ankle", "l_toes": "left_foot",
    "r_hip": "right_hip", "r_knee": "right_knee", "r_ankle": "right_ankle", "r_toes": "right_foot",
    "l_shoulder": "left_shoulder", "l_elbow": "left_elbow", "l_wrist": "left_wrist",
    "l_fingers": "left_middle1",
    "r_shoulder": "right_shoulder", "r_elbow": "right_elbow", "r_wrist": "right_wrist",
    "r_fingers": "right_middle1",
}

#: Clip-name prefix -> the parkour technique, in the order the paper reports them.
ACTIONS = {"kv": "Kong-vault", "mu": "Muscle-up", "pu": "Pull-up", "sv": "Safety-vault"}

#: Table row -> the body file under ``out/<clip>/``, in the order the chain runs.
ROWS = {
    "SAM-3D-Body (per-frame init)": "sam3d/smplx_params.npz",
    "+ visual refinement": "human_optim/kinematic_1.npz",
    "+ contact repair": "human_optim/kinematic_2.npz",
    "+ forces": "human_optim/kindyn_1.npz",
    "+ forces, measured contacts": "human_optim_gt_contact/kindyn_1.npz",
}

BODY_KEYS = ("global_orient", "body_pose", "transl", "left_hand_pose", "right_hand_pose",
             "jaw_pose", "leye_pose", "reye_pose", "expression", "betas")
_MODELS: dict[int, torch.nn.Module] = {}


def _model(num_frames: int) -> torch.nn.Module:
    if num_frames not in _MODELS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _MODELS[num_frames] = smplx.create(
                str(MODELS), model_type="smplx", gender="neutral", use_pca=False,
                flat_hand_mean=False, num_betas=10, num_expression_coeffs=10,
                batch_size=num_frames).to(DEVICE)
    return _MODELS[num_frames]


def predicted_joints(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Re-pose one body file: the 16 Parkour joints ``(T, 16, 3)`` and its valid mask."""
    body = np.load(path, allow_pickle=True)
    num_frames = int(body["num_frames"])
    tensors = {}
    for key in BODY_KEYS:
        value = np.asarray(body[key][0], np.float32)
        if value.ndim == 1:                       # betas are per clip, not per frame
            value = np.broadcast_to(value[None], (num_frames, value.shape[0]))
        tensors[key] = torch.as_tensor(np.ascontiguousarray(value), device=DEVICE)
    with torch.no_grad():
        joints = _model(num_frames)(**tensors).joints.cpu().numpy().astype(np.float64)
    columns = [JOINT_NAMES.index(CORRESPONDENCE[name]) for name in CORRESPONDENCE]
    return joints[:, columns], np.asarray(body["valid_mask"][0], bool)


def procrustes_errors(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-frame, per-joint residual in mm after the dataset's own similarity fit."""
    errors = np.empty(predicted.shape[:2])
    for i, (pred, gt) in enumerate(zip(predicted * 1e3, truth * 1e3)):
        pred_centred, gt_centred = pred - pred.mean(0), gt - gt.mean(0)
        cross = pred_centred.T @ gt_centred
        u, _, vt = np.linalg.svd(cross)
        v = vt.T
        rotation = v @ np.diag([1.0, 1.0, np.linalg.det(v) * np.linalg.det(u)]) @ u.T
        scale = np.trace(rotation @ cross) / (pred_centred ** 2).sum()
        errors[i] = np.linalg.norm(pred_centred @ (scale * rotation).T - gt_centred, axis=1)
    return errors


def main() -> int:
    clips = sorted(d for d in (PARKOUR / "out").iterdir()
                   if (d / "human_optim" / "kinematic_1.npz").exists())
    truth = {c.name: np.load(PARKOUR / c.name / "joints_3d.npz") for c in clips}
    gt_columns = {}
    for clip, gt in truth.items():
        names = [str(x) for x in gt["joint_names"]]
        if sorted(names) != sorted(CORRESPONDENCE):
            raise SystemExit(f"[parkour_pose] {clip} has unexpected joint names {names}")
        gt_columns[clip] = [names.index(name) for name in CORRESPONDENCE]

    per_clip: dict[str, dict[str, float]] = {}
    per_joint: dict[str, np.ndarray] = {}
    for label, relative in ROWS.items():
        clip_scores, joint_totals = {}, []
        for clip_dir in clips:
            path = clip_dir / relative
            if not path.exists():
                continue
            gt = truth[clip_dir.name]["joints"][:, gt_columns[clip_dir.name]].astype(np.float64)
            predicted, valid = predicted_joints(path)
            keep = valid[:len(gt)] if len(valid) >= len(gt) else np.ones(len(gt), bool)
            errors = procrustes_errors(predicted[:len(gt)][keep], gt[keep])
            clip_scores[clip_dir.name] = float(errors.mean())
            joint_totals.append(errors.mean(0))
        per_clip[label] = clip_scores
        per_joint[label] = np.mean(joint_totals, axis=0)

    print(f"\nProcrustes MPJPE [mm] — {len(clips)} clips, "
          f"{sum(len(t['joints']) for t in truth.values())} frames\n")
    header = "".join(f"{a:>13s}" for a in ACTIONS.values())
    print(f"{'':32s}{header}{'Avg':>13s}")
    for label, scores in per_clip.items():
        cells = ""
        for prefix in ACTIONS:
            values = [v for c, v in scores.items() if c.startswith(prefix)]
            cells += f"{np.mean(values):13.1f}" if values else f"{'—':>13s}"
        print(f"{label:32s}{cells}{np.mean(list(scores.values())):13.1f}")

    print("\nper joint [mm]")
    names = list(CORRESPONDENCE)
    print(f"{'':32s}" + "".join(f"{n.replace('_', ' '):>12s}" for n in names[:8]))
    for label, errors in per_joint.items():
        print(f"{label:32s}" + "".join(f"{e:12.1f}" for e in errors[:8]))
    print(f"{'':32s}" + "".join(f"{n.replace('_', ' '):>12s}" for n in names[8:]))
    for label, errors in per_joint.items():
        print(f"{label:32s}" + "".join(f"{e:12.1f}" for e in errors[8:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
