"""Export our SAM-3D-Body initialisation as PhysPT's per-frame input.

PhysPT (``../other/PhysPT``) refines a per-frame SMPL motion and infers the forces
behind it. Its demo gets that motion from CLIFF; here it gets ours instead, so every
method in the Parkour comparison starts from the same body. The file written is the
one ``video_inference.py`` reads: per frame the 24 SMPL joint rotations as 6D (the
first two columns of each rotation matrix, row-major) in the camera frame, and 10
SMPL betas.

Two conversions. SMPL-X's 21 body joints ARE SMPL's joints 1..21, so the rotations
copy over; SMPL's two extra hand joints (22, 23) get the identity. The betas do not
copy — the two shape spaces differ — so SMPL betas are fitted by least squares to
reproduce OUR body's rest-pose skeleton over the 22 shared joints, the same recipe as
the Li et al. inputs (``build_our_inputs.py`` there).

Writes ``../other/PhysPT/<dataset>/<clip>_sam3d.json``. Run with the BVR environment::

    uv run --no-sync python scripts/parkour_physpt_inputs.py
    uv run --no-sync python scripts/parkour_physpt_inputs.py --dataset climb_wall_3

``climb_wall_3`` is the instrumented climbing wall: the 68 single-camera trees under
``peter/out_climb_wall_3_single`` whose three-camera reference exists. They are 30 fps and
every frame is written, so ``video_inference.py`` needs ``--dt 0.0333333``.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import smplx
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools import paths  # noqa: E402

PARKOUR = (paths.PARKOUR / "processed").resolve()
PHYSPT = (_REPO.parent / "other" / "PhysPT").resolve()
MODELS = _REPO.parent / "BetterHuman" / "models"
NUM_SHARED_JOINTS = 22


def clips_parkour() -> list[Path]:
    """The Parkour trees carrying a SAM-3D fit."""
    return sorted(d for d in (PARKOUR / "out").iterdir()
                  if (d / "sam3d" / "smplx_params.npz").exists())


def clips_climb_wall_3() -> list[Path]:
    """The climb_wall_3 single-camera trees that have a three-camera reference."""
    gt = paths.PETER / "climb_wall_3_gt" / "human"
    return [paths.PETER / "out_climb_wall_3_single" / d.name
            for d in sorted(gt.iterdir()) if d.is_dir() and not d.name.startswith("_")]


DATASETS = {"parkour": clips_parkour, "climb_wall_3": clips_climb_wall_3}


def rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    """``(..., 3)`` axis-angle -> ``(..., 3, 3)`` rotation matrices."""
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    axis = axis_angle / np.maximum(angle, 1e-12)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    cross = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(*x.shape, 3, 3)
    cos, sin = np.cos(angle)[..., None], np.sin(angle)[..., None]
    return cos * np.eye(3) + sin * cross + (1 - cos) * axis[..., :, None] * axis[..., None, :]


def smpl_rest_joints() -> tuple[np.ndarray, np.ndarray]:
    """``(base, dirs)`` with SMPL rest joints ``= base + dirs @ betas``; from PhysPT's own model."""
    with open(PHYSPT / "assets" / "data" / "smpl" / "neutral.pkl", "rb") as handle:
        params = pickle.load(handle)
    regressor = params["V_regressor"]                              # (24, 6890)
    base = regressor @ params["v_template"]                        # (24, 3)
    dirs = np.einsum("jv,vck->jck", regressor, params["shapedirs"])  # (24, 3, 10)
    return base, dirs


def smplx_rest_joints(betas: np.ndarray) -> np.ndarray:
    """Our body's rest-pose joints ``(22, 3)`` for these SMPL-X betas."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smplx.create(str(MODELS), model_type="smplx", gender="neutral", use_pca=False,
                             num_betas=10, batch_size=1)
    with torch.no_grad():
        joints = model(betas=torch.as_tensor(betas[None], dtype=torch.float32)).joints[0]
    return joints[:NUM_SHARED_JOINTS].numpy().astype(np.float64)


def fit_betas(base: np.ndarray, dirs: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """SMPL betas whose pelvis-relative rest skeleton best matches ``target`` (22, 3)."""
    shared = slice(0, NUM_SHARED_JOINTS)
    design = (dirs[shared] - dirs[0:1]).reshape(-1, dirs.shape[2])
    offset = (base[shared] - base[0:1]).reshape(-1)
    goal = (target - target[0:1]).reshape(-1)
    betas, *_ = np.linalg.lstsq(design, goal - offset, rcond=None)
    residual = np.linalg.norm((design @ betas + offset - goal).reshape(-1, 3), axis=1)
    return betas, float(residual.mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="parkour", choices=sorted(DATASETS))
    parser.add_argument("--clips", default=None, help="comma-separated clip names (default: all)")
    args = parser.parse_args()
    out_dir = PHYSPT / args.dataset
    out_dir.mkdir(exist_ok=True)
    base, dirs = smpl_rest_joints()
    clips = DATASETS[args.dataset]()
    if args.clips:
        wanted = set(args.clips.split(","))
        clips = [c for c in clips if c.name in wanted]
    for clip in clips:
        body = np.load(clip / "sam3d" / "smplx_params.npz")
        orient = np.asarray(body["global_orient"][0], np.float64)               # (N, 3)
        pose = np.asarray(body["body_pose"][0], np.float64).reshape(-1, 21, 3)  # (N, 21, 3)
        n = len(orient)
        rotations = np.concatenate([rodrigues(orient)[:, None], rodrigues(pose),
                                    np.broadcast_to(np.eye(3), (n, 2, 3, 3))], 1)  # (N, 24, 3, 3)
        pose6d = rotations[..., :2].reshape(n, 24 * 6)                            # row-major (3, 2)
        betas, residual = fit_betas(base, dirs, smplx_rest_joints(body["betas"][0].mean(0)))
        payload = {"impath": [], "bboxs_det": [], "datapath": "", "vid_start": [0, n],
                   "pred_pose": pose6d.tolist(),
                   "pred_beta": np.broadcast_to(betas, (n, 10)).tolist()}
        (out_dir / f"{clip.name}_sam3d.json").write_text(json.dumps(payload))
        print(f"{clip.name}: {n} frames, SMPL betas fit residual {residual * 1e3:.1f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
