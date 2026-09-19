"""Export our pipeline's inputs for the Li et al. CVPR'19 estimator, OpenCap edition.

The parkour twin of this script is ``scripts/parkour_estmf_inputs.py``; the three things
the estimator wants are the same, only the sources differ:

* **3D initialisation** — SAM-3D-Body's per-frame SMPL-X fit, in place of HMR. SAM-3D
  solves in a camera of its own invention (its focal is a guess, ~940-1000 px against the
  lab's measured 913), so the body is put back on the TRUE camera's rays before it is
  written out: the pelvis pixel is preserved and the depth is scaled by ``f_true/f_sam3d``.
  Everything downstream — the mocap-frame mapping of the forces above all — assumes the
  estimator's camera is the lab's calibrated one, which is only true if the initialisation
  is expressed there too.
* **2D joints** — Sapiens Goliath-308 folded to the 18 COCO joints OpenPose emits.
* **Contacts** — OUR optimisation's contact labels (``human_optim_floorplane/contacts_2.npz``,
  the ones ``kindyn_1`` itself solved against), in place of the learned recogniser. Unlike
  the parkour export these are *estimates*, not plate measurements: this is our contact
  front end handed to somebody else's dynamics.

**How the 12 contact frames become the recogniser's 9 joints.** Our labels are per contact
frame (heel / ball_inner / ball_outer / big_toe / toe_above per foot, plus the two sit
frames); the recogniser speaks of ``l_ankle`` — the *sole*, four contact points the
estimator pins flat on its ground plane — and ``l_toes``, a single point that can carry a
force while the rest of the foot is in the air. ``--contact-mapping heel``, the default,
uses both the way the estimator means them:

* a HEEL frame loaded (with or without the forefoot) is a loaded **sole**;
* the forefoot loaded with no heel frame is loaded **toes**;
* nothing loaded is a free foot.

The alternative, ``--contact-mapping any``, calls any loaded foot frame a sole and any
loaded forefoot toes. It reads more directly off our labels but is self-defeating: the
estimator's own post-processing (``lib/contact_states.py::adjust_toe_contacts``) drops the
toes wherever the sole is in contact, and under that rule the toes are a subset of the
sole, so the toes channel never fires and every forefoot-only frame — toe-off, a drop-jump
landing — is flattened into a four-point sole. It is kept only so the two can be compared.

Either way the cost is on the other side: our heel labels are the unreliable half (one
walking clip has a 66-frame right stance with no heel frame ever labelled), and under
``heel`` such a stance is carried by the toe point alone.

Hands, neck and knees are reported as never in contact. Neither the chair of the STS
trials (our ``sit_*`` frames, 3193 of them) nor any other non-foot support exists in the
recogniser's vocabulary, so the estimator solves those frames as if the person were
free-standing.

One ``.npz`` per clip, because the two projects cannot share pickles: this repo runs
numpy 2 and the estimator's conda environment runs numpy 1. ``scripts/build_our_inputs.py
--dataset OpenCap`` on the estimator side turns these into the pickles its loaders expect.

Usage:
    uv run --no-sync python scripts/methods/opencap_estmf_inputs.py
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import smplx
import torch
from smplx.joint_names import JOINT_NAMES

_REPO = Path(__file__).resolve().parents[2]
PROCESSED = (_REPO.parent / "data" / "LabValidation_withVideos" / "processed").resolve()
OUT_DIR = (_REPO.parent / "data" / "Estimating-3D-Motion-Forces" / "data"
           / "OpenCap" / "ours").resolve()
MODELS = _REPO.parent / "BetterHuman" / "models"

#: Goliath-308 index for each of OpenPose's 18 COCO joints. ``None`` is the neck,
#: which Goliath does not label: OpenPose puts it midway between the shoulders.
COCO18_FROM_GOLIATH = [0, None, 6, 8, 41, 5, 7, 62, 10, 12, 14, 9, 11, 13, 2, 1, 4, 3]
GOLIATH_SHOULDERS = (5, 6)

#: The contact recogniser's 9 joints, in its own order (see the estimator's
#: ``lib/contact_states.py``).
CONTACT_REC_JOINTS = ["neck", "l_fingers", "r_fingers", "l_knee", "r_knee",
                      "l_ankle", "r_ankle", "l_toes", "r_toes"]
#: Our contact frames, by side and by end of the foot. ``toe_above`` is the dorsum of the
#: toe — it never fires on these trials (6 frames in 37388) but it is a forefoot contact
#: if it does.
HEEL_FRAMES = ["heel_{side}"]
FORE_FRAMES = ["ball_inner_{side}", "ball_outer_{side}", "big_toe_{side}", "toe_above_{side}"]
SIDES = ("left", "right")
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


def contact_states(run: Path, num_frames: int,
                   mapping: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Our contact labels as the recogniser's ``(nf, 9, 4)`` one-hot, plus the counts.

    :param mapping: ``"heel"`` (a sole needs a heel frame, the forefoot alone is toes) or
        ``"any"`` (any loaded frame is a sole); see the module docstring.
    :returns: the one-hot labels and ``{side: [sole, toes, free] frame counts}``.
    """
    path = run / "human_optim_floorplane" / "contacts_2.npz"
    if not path.exists():
        path = run / "human_optim_floorplane" / "contacts_1.npz"
    saved = np.load(path, allow_pickle=True)
    names = [str(x) for x in saved["contact_frame_names"]]
    frame_contact = np.asarray(saved["frame_contact"][0], bool)
    valid = np.asarray(saved["valid_mask"][0], bool)
    if len(frame_contact) != num_frames:
        raise SystemExit(f"{path}: {len(frame_contact)} contact frames, {num_frames} video frames")

    def loaded(templates: list[str], side: str) -> np.ndarray:
        columns = [names.index(t.format(side=side)) for t in templates if t.format(side=side) in names]
        return frame_contact[:, columns].any(1)

    state, counts = {}, {}
    for side in SIDES:
        heel, fore = loaded(HEEL_FRAMES, side), loaded(FORE_FRAMES, side)
        sole = heel if mapping == "heel" else (heel | fore)
        toes = (fore & ~heel) if mapping == "heel" else fore
        state[f"{side[0]}_ankle"], state[f"{side[0]}_toes"] = sole, toes
        counts[side] = np.array([sole.sum(), toes.sum(), (~(sole | toes)).sum()])
    labels = np.full((num_frames, len(CONTACT_REC_JOINTS)), NO_CONTACT)
    for slot, name in enumerate(CONTACT_REC_JOINTS):
        if name in state:
            labels[:, slot] = np.where(state[name], IN_CONTACT, NO_CONTACT)
    # A frame our solve never produced is unknown, not released: report it the way an
    # occlusion is reported, which the estimator carries forward from the frame before.
    labels[~valid] = OCCLUDED
    one_hot = np.zeros(labels.shape + (4,), np.uint8)
    np.put_along_axis(one_hot, labels[..., None], 1, axis=-1)
    return one_hot, counts


def on_true_camera(joints_cam: np.ndarray, transl: np.ndarray, sam3d_k: np.ndarray,
                   true_k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Move a SAM-3D body from its own camera onto the lab's, keeping the picture.

    SAM-3D guesses a focal length; the lab measured one. Re-solving the whole fit under the
    true camera is not on the table here, so the body is translated instead: the pelvis
    keeps its pixel, and the depth is scaled by ``f_true / f_sam3d`` so the apparent size
    is preserved. Both are exact for the pelvis and first-order for the rest of the body.

    :param joints_cam: ``(nf, J, 3)`` joints in SAM-3D's camera, metres.
    :param transl: ``(nf, 3)`` the SMPL-X translation that produced them.
    :returns: the joints and the translation on the true camera's rays.
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


def export(clip_name: str, mapping: str) -> dict[str, np.ndarray]:
    clip, run = PROCESSED / clip_name, PROCESSED / "out" / clip_name
    meta = json.loads((clip / "meta.json").read_text())
    intrinsics = np.asarray(meta["K"], np.float64)
    init = np.load(run / "sam3d" / "smplx_params.npz")
    num_frames = int(init["num_frames"])
    if num_frames != int(meta["n_frames"]):
        raise SystemExit(f"{clip_name}: sam3d has {num_frames} frames, meta.json {meta['n_frames']}")

    # SMPL's 72 axis angles are SMPL-X's global orientation and 21 body joints,
    # then its two hand joints, which SMPL-X replaces with articulated fingers.
    # Those two are leaves in the estimator's skeleton, so zero moves nothing.
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

    states, counts = contact_states(run, num_frames, mapping)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / f"{clip_name}.npz",
             poses=poses, transl=transl, betas_smplx=betas,
             smplx_joints_rest=joints_rest, focal=np.float64(focal), principal=principal,
             coco18=coco18_from_sapiens(run / "sapiens" / "pose.npz"),
             hmr_joints_2d=ankles_2d, contact_states=states,
             num_frames=np.int64(num_frames), fps=np.float64(meta["fps"]),
             sam3d_focal=np.float64((sam3d_k[0, 0] + sam3d_k[1, 1]) / 2.0),
             contact_mapping=np.str_(mapping))
    print(f"  {clip_name}: {num_frames} frames, focal {focal:.1f} px (SAM-3D had "
          f"{(sam3d_k[0, 0] + sam3d_k[1, 1]) / 2.0:.1f}, ankles moved {ankle_shift:.1f} px), "
          + "  ".join(f"{side[0].upper()} sole/toe/free "
                      f"{counts[side][0]}/{counts[side][1]}/{counts[side][2]}"
                      for side in SIDES))
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", type=Path,
                    default=PROCESSED / "out" / "_methods" / "clips_natural.txt",
                    help="file listing the clip names to export, one per line")
    ap.add_argument("--contact-mapping", choices=("heel", "any"), default="heel",
                    help="how our contact frames become the sole and toes channels")
    args = ap.parse_args(argv)

    clips = [line.strip() for line in args.clips.read_text().split() if line.strip()]
    print(f"(opencap_estmf_inputs) {len(clips)} clips, contact mapping "
          f"{args.contact_mapping!r} -> {OUT_DIR}")
    total = {side: np.zeros(3, np.int64) for side in SIDES}
    for name in clips:
        counts = export(name, args.contact_mapping)
        for side in SIDES:
            total[side] += counts[side]
    every = sum(total.values())
    print(f"\ncontact frames over {len(clips)} clips, both feet: sole {every[0]}, "
          f"toes {every[1]}, free {every[2]} of {every.sum()}")
    for side in SIDES:
        print(f"  {side:5s} sole {total[side][0]:6d}  toes {total[side][1]:6d}  "
              f"free {total[side][2]:6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
