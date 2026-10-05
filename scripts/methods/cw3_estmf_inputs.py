"""Export our pipeline's inputs for the Li et al. CVPR'19 estimator, climb_wall_3 edition.

The parkour twin of this script is ``scripts/parkour_estmf_inputs.py``. The estimator
wants the same three things; only the sources differ, and a climbing wall needs two
extras that a flat floor did not.

* **3D initialisation** — SAM-3D-Body's per-frame SMPL-X fit, in place of HMR. SAM-3D
  solves in a camera of its own invention, so the body is put back on the rig's
  CALIBRATED camera before it is written out: the pelvis keeps its pixel and the depth
  is scaled by ``f_true / f_sam3d``. Everything else here — the board forces, the wall,
  gravity — is stated in that calibrated camera's frame, so the body has to be too.
* **2D joints** — Sapiens Goliath-308 folded to the 18 COCO joints OpenPose emits.
* **Contacts** — the rig's force boards, in place of the learned recogniser: which hand
  and which foot is loaded, measured, not guessed. Feet become the estimator's SOLE
  channels and hands its FINGERS channels; toes, knees and neck are never in contact.
* **Gravity** — the rig's own down direction, in the camera frame. The estimator
  hard-wires gravity to the camera's +y, which is down only for a level camera; these
  cameras look up at a climbing wall.
* **The principal point** — the calibrated one, not the centre of the frame.

One ``.npz`` per clip, because the two projects cannot share pickles: this repo runs
numpy 2 and the estimator's conda environment runs numpy 1. ``scripts/build_our_inputs.py
--dataset climb_wall_3`` on the estimator side turns these into the pickles its loaders
expect.

Usage:
    uv run --no-sync python scripts/methods/cw3_estmf_inputs.py
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import smplx
import torch
from smplx.joint_names import JOINT_NAMES

_REPO = Path(__file__).resolve().parents[2]
SINGLE = _REPO / "peter" / "out_climb_wall_3_single"
GT = _REPO / "peter" / "climb_wall_3_gt" / "human"
OUT_DIR = (_REPO.parent / "data" / "Estimating-3D-Motion-Forces" / "data"
           / "climb_wall_3" / "ours").resolve()
MODELS = _REPO.parent / "BetterHuman" / "models"

#: Goliath-308 index for each of OpenPose's 18 COCO joints. ``None`` is the neck,
#: which Goliath does not label: OpenPose puts it midway between the shoulders.
COCO18_FROM_GOLIATH = [0, None, 6, 8, 41, 5, 7, 62, 10, 12, 14, 9, 11, 13, 2, 1, 4, 3]
GOLIATH_SHOULDERS = (5, 6)

#: The contact recogniser's 9 joints, in its own order (the estimator's
#: ``lib/contact_states.py``), and which measured limb fills each. The five it does not
#: name are reported free: the estimator drops the neck off the barbell action and the
#: knees on the climbing action, and the toes never carry a board.
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


def contact_states(clip_name: str, num_frames: int) -> tuple[np.ndarray, dict[str, int]]:
    """The boards as the recogniser's ``(nf, 9, 4)`` one-hot labels, plus loaded counts.

    The measured labels come from the multi-view fit, which drops the two frames it has
    no lag for, so they are two rows short of the video; the last row is repeated.
    """
    saved = np.load(GT / clip_name / "human_optim" / "contacts_1.npz", allow_pickle=True)
    limbs = [str(x) for x in saved["limb_names"]]
    measured = np.asarray(saved["limb_contact"][0], bool)
    if len(measured) > num_frames:
        raise SystemExit(f"{clip_name}: {len(measured)} labelled frames, {num_frames} video frames")
    padding = np.repeat(measured[-1:], num_frames - len(measured), axis=0)
    measured = np.concatenate([measured, padding], axis=0)

    labels = np.full((num_frames, len(CONTACT_REC_JOINTS)), NO_CONTACT)
    counts = {}
    for slot, name in enumerate(CONTACT_REC_JOINTS):
        if name not in LIMB_OF_CONTACT_JOINT:
            continue
        loaded = measured[:, limbs.index(LIMB_OF_CONTACT_JOINT[name])]
        labels[:, slot] = np.where(loaded, IN_CONTACT, NO_CONTACT)
        counts[name] = int(loaded.sum())
    one_hot = np.zeros(labels.shape + (4,), np.uint8)
    np.put_along_axis(one_hot, labels[..., None], 1, axis=-1)
    return one_hot, counts


def on_true_camera(joints_cam: np.ndarray, transl: np.ndarray, sam3d_k: np.ndarray,
                   true_k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Move a SAM-3D body from its own camera onto the calibrated one, keeping the picture.

    SAM-3D guesses a focal length; the rig measured one. Re-solving the whole fit under
    the true camera is not on the table here, so the body is translated instead: the
    pelvis keeps its pixel and the depth is scaled by ``f_true / f_sam3d``, so the
    apparent size is preserved. Both are exact for the pelvis and first-order for the
    rest of the body.
    """
    pelvis = joints_cam[:, 0]
    pixel = np.stack([sam3d_k[0, 0] * pelvis[:, 0] / pelvis[:, 2] + sam3d_k[0, 2],
                      sam3d_k[1, 1] * pelvis[:, 1] / pelvis[:, 2] + sam3d_k[1, 2]], 1)
    focal_ratio = (true_k[0, 0] + true_k[1, 1]) / (sam3d_k[0, 0] + sam3d_k[1, 1])
    depth = pelvis[:, 2] * focal_ratio
    moved = np.stack([(pixel[:, 0] - true_k[0, 2]) * depth / true_k[0, 0],
                      (pixel[:, 1] - true_k[1, 2]) * depth / true_k[1, 1], depth], 1)
    shift = moved - pelvis
    return joints_cam + shift[:, None, :], transl + shift


def project(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """``(..., 3)`` camera points -> ``(..., 2)`` pixels."""
    return np.stack([intrinsics[0, 0] * points[..., 0] / points[..., 2] + intrinsics[0, 2],
                     intrinsics[1, 1] * points[..., 1] / points[..., 2] + intrinsics[1, 2]], -1)


def export(clip_name: str) -> dict[str, int]:
    run = SINGLE / clip_name
    init = np.load(run / "sam3d" / "smplx_params.npz")
    num_frames = int(init["num_frames"])

    # SMPL's 72 axis angles are SMPL-X's global orientation and 21 body joints, then its
    # two hand joints, which SMPL-X replaces with articulated fingers. Those two are
    # leaves in the estimator's skeleton, so zero moves nothing.
    poses = np.zeros((num_frames, 72))
    poses[:, :3] = init["global_orient"][0]
    poses[:, 3:66] = init["body_pose"][0]
    betas = np.asarray(init["betas"][0], np.float64).mean(0)
    transl = np.asarray(init["transl"][0], np.float64)

    model = smplx_model(num_frames)
    with torch.no_grad():
        rest = model(betas=torch.as_tensor(betas[None], dtype=torch.float32))
        posed = model(betas=torch.as_tensor(np.broadcast_to(betas[None], (num_frames, 10)).copy(),
                                            dtype=torch.float32),
                      global_orient=torch.as_tensor(poses[:, :3], dtype=torch.float32),
                      body_pose=torch.as_tensor(poses[:, 3:66], dtype=torch.float32),
                      transl=torch.as_tensor(transl, dtype=torch.float32))
    joints_rest = rest.joints[0, :22].numpy().astype(np.float64)
    joints_cam = posed.joints[0 if num_frames == 1 else slice(None)].numpy().astype(np.float64)

    # cam_left is the rig's world frame, so its calibrated intrinsics are the camera the
    # estimator should solve in.
    cameras = np.load(GT / clip_name / "cameras.npz", allow_pickle=True)
    intrinsics = np.asarray(cameras["K"][0], np.float64)
    sam3d_k = np.asarray(init["cam_intrinsics"], np.float64)
    before = project(joints_cam[:, [JOINT_INDEX[j] for j in ANKLES]], sam3d_k)
    joints_cam, transl = on_true_camera(joints_cam, transl, sam3d_k, intrinsics)
    after = project(joints_cam[:, [JOINT_INDEX[j] for j in ANKLES]], intrinsics)
    ankle_shift = float(np.median(np.linalg.norm(after - before, axis=-1)))

    focal = float((intrinsics[0, 0] + intrinsics[1, 1]) / 2.0)
    principal = intrinsics[:2, 2]

    # Only the two ankles are ever read back, to stand in where Sapiens is unsure.
    ankles_2d = np.zeros((num_frames, 19, 2))
    for joint, slot in ANKLES.items():
        ankles_2d[:, slot] = project(joints_cam[:, JOINT_INDEX[joint]], intrinsics)

    # The rig's gravity is stated in its world frame, which is cam_left's; the geometry
    # stage's extrinsics turn any world vector into the camera's.
    transform = np.load(run / "geometry" / "transform.npz")
    gravity = np.asarray(transform["extrinsics"][0][:3, :3] @ np.load(
        run / "geocalib" / "gravity.npz")["gravity_world"], np.float64)
    gravity /= np.linalg.norm(gravity)

    states, counts = contact_states(clip_name, num_frames)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / f"{clip_name}.npz",
             poses=poses, transl=transl, betas_smplx=betas,
             smplx_joints_rest=joints_rest, focal=np.float64(focal), principal=principal,
             coco18=coco18_from_sapiens(run / "sapiens" / "pose.npz"),
             hmr_joints_2d=ankles_2d, contact_states=states,
             num_frames=np.int64(num_frames), fps=np.float64(transform["fps"]),
             gravity=gravity,
             sam3d_focal=np.float64((sam3d_k[0, 0] + sam3d_k[1, 1]) / 2.0))
    print(f"  {clip_name}: {num_frames} frames, focal {focal:.1f} px (SAM-3D had "
          f"{(sam3d_k[0, 0] + sam3d_k[1, 1]) / 2.0:.1f}, ankles moved {ankle_shift:.1f} px), "
          f"gravity [{gravity[0]:+.3f} {gravity[1]:+.3f} {gravity[2]:+.3f}], loaded "
          + " ".join(f"{name} {counts[name]}" for name in sorted(counts)))
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", default=None,
                    help="comma-separated clip names (default: every clip with measured "
                         "contacts, i.e. every directory of the ground-truth tree)")
    args = ap.parse_args(argv)

    clips = sorted(d.name for d in GT.iterdir()
                   if d.is_dir() and not d.name.startswith("_"))
    if args.clips:
        wanted = set(args.clips.split(","))
        clips = [c for c in clips if c in wanted]
    print(f"(cw3_estmf_inputs) {len(clips)} clips -> {OUT_DIR}")
    total = {name: 0 for name in LIMB_OF_CONTACT_JOINT}
    for name in clips:
        for key, value in export(name).items():
            total[key] += value
    print("\nloaded frames over " + str(len(clips)) + " clips: "
          + "  ".join(f"{name} {total[name]}" for name in sorted(total)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
