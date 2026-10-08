"""Export our pipeline's inputs for the Li et al. CVPR'19 motion-force estimator.

That estimator (``data/Estimating-3D-Motion-Forces``) was built to run on HMR for
the 3D initialisation, OpenPose for 2D joints and a learned contact recogniser.
This writes the same three things from OUR pipeline instead, so the estimator can
be compared on equal footing:

* **3D initialisation** — SAM-3D-Body's per-frame SMPL-X fit, in place of HMR.
* **2D joints** — Sapiens Goliath-308 folded to the 18 COCO joints OpenPose emits.
* **Contacts** — the dataset's own force plates and instrumented bar, in place of
  the learned recogniser: ground truth rather than a guess.

One ``.npz`` per clip, because the two projects cannot share pickles: this repo
runs numpy 2 and the estimator's conda environment runs numpy 1, and a numpy-2
pickle will not load there. ``scripts/build_our_inputs.py`` on the estimator side
turns these into the pickles its loaders expect.

Usage:
    uv run --no-sync python scripts/parkour_estmf_inputs.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import smplx
import torch
from smplx.joint_names import JOINT_NAMES

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools import paths  # noqa: E402

PARKOUR = (paths.PARKOUR / "processed").resolve()
OUT_DIR = (paths.ESTMF / "data"
           / "Parkour-dataset" / "ours").resolve()
MODELS = _REPO.parent / "BetterHuman" / "models"

#: Goliath-308 index for each of OpenPose's 18 COCO joints. ``None`` is the neck,
#: which Goliath does not label: OpenPose puts it midway between the shoulders.
COCO18_FROM_GOLIATH = [0, None, 6, 8, 41, 5, 7, 62, 10, 12, 14, 9, 11, 13, 2, 1, 4, 3]
GOLIATH_SHOULDERS = (5, 6)

#: The contact recogniser's 9 joints, and which ground-truth limb measures each.
#: The four it cannot see are reported as "not in contact": the estimator ignores
#: the neck off the barbell action and the knees on parkour, and a loaded sole
#: already suppresses its own toes inside the estimator's post-processing.
CONTACT_REC_JOINTS = ["neck", "l_fingers", "r_fingers", "l_knee", "r_knee",
                      "l_ankle", "r_ankle", "l_toes", "r_toes"]
LIMB_OF_CONTACT_JOINT = {"l_fingers": "left_hand", "r_fingers": "right_hand",
                         "l_ankle": "left_foot", "r_ankle": "right_foot"}
#: The recogniser's label set, as the estimator's loader decodes it.
IN_CONTACT, NO_CONTACT, OCCLUDED = 0, 1, 2

#: SMPL-X ankles, and where they sit in the 19 Cocoplus joints HMR reports.
ANKLES = {"right_ankle": 0, "left_ankle": 5}
JOINT_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}


def smplx_model(num_frames: int) -> torch.nn.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smplx.create(str(MODELS), model_type="smplx", gender="neutral", use_pca=False,
                            flat_hand_mean=False, num_betas=10, num_expression_coeffs=10,
                            batch_size=num_frames)


def coco18_from_sapiens(pose_file: Path) -> np.ndarray:
    """Sapiens Goliath-308 -> the 18 COCO joints, as ``(nf, 18, 3)`` x/y/confidence."""
    sapiens = np.load(pose_file)
    points = np.asarray(sapiens["keypoints"][0], np.float64)
    scores = np.asarray(sapiens["keypoint_scores"][0], np.float64)
    out = np.zeros((len(points), 18, 3))
    for slot, source in enumerate(COCO18_FROM_GOLIATH):
        if source is None:                               # neck: midway between shoulders
            out[:, slot, :2] = points[:, GOLIATH_SHOULDERS, :].mean(1)
            out[:, slot, 2] = scores[:, GOLIATH_SHOULDERS].min(1)
        else:
            out[:, slot, :2] = points[:, source]
            out[:, slot, 2] = scores[:, source]
    return out


def contact_states(clip: Path) -> np.ndarray:
    """The plates and bar as the recogniser's ``(nf, 9, 4)`` one-hot labels."""
    truth = np.load(clip / "contacts.npz")
    limbs = [str(x) for x in truth["limbs"]]
    contacts, captured = np.asarray(truth["contacts"], bool), np.asarray(truth["captured"], bool)
    labels = np.full((len(contacts), len(CONTACT_REC_JOINTS)), NO_CONTACT)
    for slot, name in enumerate(CONTACT_REC_JOINTS):
        if name not in LIMB_OF_CONTACT_JOINT:
            continue
        column = limbs.index(LIMB_OF_CONTACT_JOINT[name])
        labels[:, slot] = np.where(contacts[:, column], IN_CONTACT, NO_CONTACT)
        # A limb the rig never measured is unknown, not released: report it the
        # way an occlusion is reported, which the estimator carries forward.
        labels[~captured[:, column], slot] = OCCLUDED
    one_hot = np.zeros(labels.shape + (4,), np.uint8)
    np.put_along_axis(one_hot, labels[..., None], 1, axis=-1)
    return one_hot


def export(clip_name: str) -> None:
    clip, run = PARKOUR / clip_name, PARKOUR / "out" / clip_name
    init = np.load(run / "sam3d" / "smplx_params.npz")
    num_frames = int(init["num_frames"])

    # SMPL's 72 axis angles are SMPL-X's global orientation and 21 body joints,
    # then its two hand joints, which SMPL-X replaces with articulated fingers.
    # Those two are leaves in the estimator's skeleton, so zero moves nothing.
    poses = np.zeros((num_frames, 72))
    poses[:, :3] = init["global_orient"][0]
    poses[:, 3:66] = init["body_pose"][0]
    betas = np.asarray(init["betas"][0], np.float64).mean(0)

    model = smplx_model(num_frames)
    with torch.no_grad():
        rest = model(betas=torch.as_tensor(betas[None], dtype=torch.float32))
        posed = model(betas=torch.as_tensor(np.broadcast_to(betas[None], (num_frames, 10)).copy(),
                                            dtype=torch.float32),
                      global_orient=torch.as_tensor(poses[:, :3], dtype=torch.float32),
                      body_pose=torch.as_tensor(poses[:, 3:66], dtype=torch.float32),
                      transl=torch.as_tensor(np.asarray(init["transl"][0]), dtype=torch.float32))
    joints_rest = rest.joints[0, :22].numpy().astype(np.float64)
    joints_cam = posed.joints[0 if num_frames == 1 else slice(None)].numpy().astype(np.float64)

    intrinsics = np.asarray(init["cam_intrinsics"], np.float64)
    focal = float((intrinsics[0, 0] + intrinsics[1, 1]) / 2.0)
    principal = intrinsics[:2, 2]

    # Only the two ankles are ever read back, to stand in where Sapiens is unsure.
    ankles_2d = np.zeros((num_frames, 19, 2))
    for joint, slot in ANKLES.items():
        xyz = joints_cam[:, JOINT_INDEX[joint]]
        ankles_2d[:, slot] = principal + focal * xyz[:, :2] / xyz[:, 2:3]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / f"{clip_name}.npz",
             poses=poses, transl=np.asarray(init["transl"][0], np.float64), betas_smplx=betas,
             smplx_joints_rest=joints_rest, focal=np.float64(focal), principal=principal,
             coco18=coco18_from_sapiens(run / "sapiens" / "pose.npz"),
             hmr_joints_2d=ankles_2d, contact_states=contact_states(clip),
             num_frames=np.int64(num_frames))
    print(f"  {clip_name}: {num_frames} frames, focal {focal:.1f} px")



def main() -> int:
    clips = sorted(d.name for d in (PARKOUR / "out").iterdir()
                   if (d / "sam3d" / "smplx_params.npz").exists())
    print(f"(parkour_estmf_inputs) {len(clips)} clips -> {OUT_DIR}")
    for name in clips:
        export(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
