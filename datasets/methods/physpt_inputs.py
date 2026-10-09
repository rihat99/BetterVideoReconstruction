"""Export our SAM-3D-Body initialisation as PhysPT's per-frame input.

PhysPT (``../other/PhysPT``) refines a per-frame SMPL motion and infers the forces
behind it. Its demo gets that motion from CLIFF; here it gets ours instead, so every
method in a comparison starts from the same body. The file written is the one
``video_inference.py`` reads: per frame the 24 SMPL joint rotations as 6D (the first two
columns of each rotation matrix, row-major) in the camera frame, and 10 SMPL betas. It
never touches ``impath``/``bboxs_det``/``datapath``: no intrinsics, no bounding box, no
translation is needed, because ``GlobalTrajPredictor`` re-predicts the global trajectory
from the pelvis-relative pose alone.

Two conversions. SMPL-X's 21 body joints ARE SMPL's joints 1..21, so the rotations
copy over; SMPL's two extra hand joints (22, 23) get the identity. The betas do not
copy — the two shape spaces differ — so SMPL betas are fitted by least squares to
reproduce OUR body's rest-pose skeleton over the 22 shared joints, the same recipe as
the Li et al. inputs (``build_our_inputs.py`` there).

The clips are ``estmf_inputs.py``'s, per dataset:

* ``parkour`` — every Parkour clip with a SAM-3D fit; every frame written.
* ``climb_wall_3`` — the single-camera trees under ``peter/out_climb_wall_3_single``
  whose three-camera reference exists. They are 30 fps and every frame is written, so
  ``video_inference.py`` needs ``--dt 0.0333333``.
* ``opencap`` — LabValidation's ``clips_natural.txt``. PhysPT ships ``constants.dt =
  1/20`` and its released weights were trained at that rate: ``dt`` sets the
  contact-vertex velocities that gate and damp the spring-mass ground reaction force,
  and the 16-frame window the transformer (and the trajectory predictor) sees is 0.8 s
  long at 20 Hz. These videos are 60 fps, so by default every third frame is taken
  (``--step 3``): exactly 20 Hz, no resampling, the rate the weights assume. ``--step 1``
  feeds all 60 fps frames instead and then ``constants.dt`` must be set to 1/60 by hand.
  The video frame each row came from is written into the json as ``video_frames`` so
  ``opencap_physpt_export.py`` can put the forces back on the video's timeline.
  ``--step`` / ``--offset`` apply to OpenCap only.

Writes ``../other/PhysPT/<dataset>/<clip>_sam3d.json``. Run with the BVR environment::

    uv run --no-sync python datasets/methods/physpt_inputs.py --dataset parkour
    uv run --no-sync python datasets/methods/physpt_inputs.py --dataset climb_wall_3
    uv run --no-sync python datasets/methods/physpt_inputs.py --dataset opencap
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

from bvr.paths import REPO_ROOT

from datasets.methods.estmf_inputs import MODELS, RUNS, select_clips

PHYSPT = (REPO_ROOT.parent / "other" / "PhysPT").resolve()
NUM_SHARED_JOINTS = 22
#: OpenCap's video rate, the one dataset subsampled to PhysPT's 20 Hz (``--step``).
OPENCAP_FPS = 60.0


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


def write_clip(dataset: str, clip: str, base: np.ndarray, dirs: np.ndarray, step: int,
               offset: int, out_dir: Path) -> int:
    """Write one clip's ``<clip>_sam3d.json``; returns the number of rows written."""
    body = np.load(RUNS[dataset] / clip / "sam3d" / "smplx_params.npz", allow_pickle=True)
    frames = np.arange(offset, len(body["global_orient"][0]), step)
    orient = np.asarray(body["global_orient"][0], np.float64)[frames]               # (n, 3)
    pose = np.asarray(body["body_pose"][0], np.float64).reshape(-1, 21, 3)[frames]  # (n, 21, 3)
    n = len(frames)
    rotations = np.concatenate([rodrigues(orient)[:, None], rodrigues(pose),
                                np.broadcast_to(np.eye(3), (n, 2, 3, 3))], 1)       # (n, 24, 3, 3)
    pose6d = rotations[..., :2].reshape(n, 24 * 6)                            # row-major (3, 2)
    betas, residual = fit_betas(base, dirs, smplx_rest_joints(body["betas"][0].mean(0)))
    payload = {"impath": [], "bboxs_det": [], "datapath": "", "vid_start": [0, n],
               "pred_pose": pose6d.tolist(),
               "pred_beta": np.broadcast_to(betas, (n, 10)).tolist()}
    if dataset == "opencap":
        payload.update(video_frames=frames.tolist(), video_fps=OPENCAP_FPS, step=step,
                       physpt_fps=OPENCAP_FPS / step)
    (out_dir / f"{clip}_sam3d.json").write_text(json.dumps(payload))
    print(f"{clip}: {len(body['global_orient'][0])} video frames -> {n} rows, "
          f"SMPL betas fit residual {residual * 1e3:.1f} mm")
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=sorted(RUNS))
    ap.add_argument("--clips", default=None,
                    help="comma-separated clip names, or a file listing them (default: the "
                         "dataset's clip set)")
    ap.add_argument("--step", type=int, default=None,
                    help="opencap only: take every STEPth video frame (default 3 = 60 fps -> "
                         "PhysPT's 20 Hz)")
    ap.add_argument("--offset", type=int, default=0, help="opencap only: first video frame taken")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: ../other/PhysPT/<dataset>)")
    args = ap.parse_args(argv)
    if args.dataset != "opencap" and (args.step not in (None, 1) or args.offset):
        ap.error("--step / --offset subsample OpenCap only; the json of the other datasets "
                 "does not record which video frames it holds")
    step = args.step if args.step is not None else (3 if args.dataset == "opencap" else 1)

    out_dir = args.out or PHYSPT / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    base, dirs = smpl_rest_joints()
    for clip in select_clips(args.dataset, args.clips):
        write_clip(args.dataset, clip, base, dirs, step, args.offset, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
