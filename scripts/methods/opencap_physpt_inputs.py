"""Export our SAM-3D-Body initialisation as PhysPT's per-frame input, OpenCap edition.

The same conversion as ``scripts/parkour_physpt_inputs.py`` — PhysPT's
``video_inference.py`` reads a json holding, per frame, the 24 SMPL joint rotations as 6D
(the first two columns of each rotation matrix, row-major) in the camera frame plus 10
SMPL betas, and nothing else. It never touches ``impath``/``bboxs_det``/``datapath``: no
intrinsics, no bounding box, no translation is needed, because the global trajectory is
re-predicted by ``GlobalTrajPredictor`` from the pelvis-relative pose alone. The
``meta.json`` ``K`` of these clips is therefore unused here.

SMPL-X's 21 body joints ARE SMPL's joints 1..21, so the rotations copy over; SMPL's two
extra hand joints (22, 23) get the identity. The betas do not copy — the two shape spaces
differ — so SMPL betas are fitted by least squares to reproduce OUR body's rest-pose
skeleton over the 22 shared joints.

**Frame rate.** PhysPT ships ``constants.dt = 1/20`` and its released weights were trained
at that rate: ``dt`` sets the contact-vertex velocities that gate and damp the
spring-mass ground reaction force, and the 16-frame window the transformer (and the
trajectory predictor) sees is 0.8 s long at 20 Hz. The LabValidation videos are 60 fps, so
by default every third frame is taken (``--step 3``): exactly 20 Hz, no resampling, the
rate the weights assume. ``--step 1`` feeds all 60 fps frames instead and then
``constants.dt`` must be set to 1/60 by hand. The video frame each row came from is written
into the json as ``video_frames`` so the exporter can put the forces back on the video's
timeline.

Writes ``../other/PhysPT/opencap/<clip>_sam3d.json``. Run with the BVR environment::

    uv run --no-sync python scripts/methods/opencap_physpt_inputs.py
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

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.paths import LABVALIDATION  # noqa: E402

PROCESSED = (LABVALIDATION / "processed").resolve()
PHYSPT = (_REPO.parent / "other" / "PhysPT").resolve()
MODELS = _REPO.parent / "BetterHuman" / "models"
CLIP_LIST = PROCESSED / "out" / "_methods" / "clips_natural.txt"
NUM_SHARED_JOINTS = 22

#: PhysPT's own sampling rate (``constants.dt``); the video rate divided by ``--step``.
PHYSPT_FPS = 20.0


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


def write_clip(clip: str, base: np.ndarray, dirs: np.ndarray, step: int, offset: int,
               out_dir: Path) -> int:
    """Write one clip's ``<clip>_sam3d.json``; returns the number of rows written."""
    body = np.load(PROCESSED / "out" / clip / "sam3d" / "smplx_params.npz", allow_pickle=True)
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
               "pred_beta": np.broadcast_to(betas, (n, 10)).tolist(),
               "video_frames": frames.tolist(), "video_fps": 60.0,
               "step": step, "physpt_fps": 60.0 / step}
    (out_dir / f"{clip}_sam3d.json").write_text(json.dumps(payload))
    print(f"{clip}: {len(body['global_orient'][0])} video frames -> {n} rows at "
          f"{60.0 / step:.1f} fps, SMPL betas fit residual {residual * 1e3:.1f} mm")
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", type=Path, default=CLIP_LIST, help="one clip name per line")
    ap.add_argument("--step", type=int, default=3,
                    help="take every STEPth video frame (3 = 60 fps -> PhysPT's 20 Hz)")
    ap.add_argument("--offset", type=int, default=0, help="first video frame taken")
    ap.add_argument("--out", type=Path, default=PHYSPT / "opencap")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    base, dirs = smpl_rest_joints()
    for clip in args.clips.read_text().split():
        write_clip(clip, base, dirs, args.step, args.offset, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
