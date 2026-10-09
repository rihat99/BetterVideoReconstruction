"""Export our pipeline's inputs for the Li et al. CVPR'19 motion-force estimator.

That estimator (``Estimating-3D-Motion-Forces`` under the data root) was built to run on
HMR for the 3D initialisation, OpenPose for 2D joints and a learned contact recogniser.
This writes the same three things from OUR pipeline instead, so the estimator can be
compared on equal footing:

* **3D initialisation** — SAM-3D-Body's per-frame SMPL-X fit, in place of HMR. SAM-3D
  solves in a camera of its own invention (its focal is a guess, ~940-1000 px against
  OpenCap's measured 913). Where the dataset has a calibrated camera (``opencap``: the
  lab's ``meta.json`` ``K``; ``climb_wall_3``: the rig's ``cam_left``) the body is put
  back on it before it is written out — the pelvis keeps its pixel and the depth is
  scaled by ``f_true / f_sam3d`` (:func:`on_true_camera`) — because everything else the
  estimator is given (forces, wall, gravity) is stated in that camera. ``parkour`` has
  no calibration and keeps SAM-3D's camera, its two focals averaged into the single
  focal the estimator solves with.
* **2D joints** — Sapiens Goliath-308 folded to the 18 COCO joints OpenPose emits.
* **Contacts**, in place of the learned recogniser:

  - ``parkour``: the dataset's own force plates and instrumented bar
    (``<clip>/contacts.npz``), ground truth rather than a guess. A limb the rig never
    measured is reported the way an occlusion is.
  - ``climb_wall_3``: the rig's force boards, through the multi-view fit's labels
    (``climb_wall_3_gt/human/<clip>/human_optim/contacts_1.npz``). Feet become the
    estimator's SOLE channels and hands its FINGERS channels.
  - ``opencap``: OUR optimisation's contact labels (``human_optim_floorplane/
    contacts_2.npz``, the ones ``kindyn_1`` itself solved against). Unlike the other two
    these are *estimates*, not measurements: our contact front end handed to somebody
    else's dynamics.

  Channels a dataset does not fill are reported as never in contact: the estimator
  ignores the neck off the barbell action and the knees on parkour and climbing, and a
  loaded sole already suppresses its own toes inside its post-processing.

``climb_wall_3`` writes two extras a flat floor did not need: **gravity**, the rig's own
down direction in the camera frame (the estimator hard-wires gravity to the camera's +y,
which is down only for a level camera, and these cameras look up at a wall), and the
calibrated **principal point** rather than the centre of the frame.

**OpenCap: how the contact frames become the recogniser's 9 joints.** Our labels are
per contact frame (heel / ball_inner / ball_outer / big_toe / toe_above per foot, plus the
two sit frames); the recogniser speaks of ``l_ankle`` — the *sole*, four contact points
the estimator pins flat on its ground plane — and ``l_toes``, a single point that can
carry a force while the rest of the foot is in the air. ``--contact-mapping heel``, the
default, uses both the way the estimator means them:

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
``heel`` such a stance is carried by the toe point alone. Neither the chair of the STS
trials (our ``sit_*`` frames) nor any other non-foot support exists in the recogniser's
vocabulary, so the estimator solves those frames as if the person were free-standing.

One ``.npz`` per clip in the estimator's ``data/<Parkour-dataset|OpenCap|climb_wall_3>/
ours/``, because the two projects cannot share pickles: this repo runs numpy 2 and the
estimator's conda environment runs numpy 1. ``scripts/build_our_inputs.py [--dataset
OpenCap|climb_wall_3]`` on the estimator side turns these into the pickles its loaders
expect.

Usage:
    uv run --no-sync python scripts/methods/estmf_inputs.py --dataset parkour
    uv run --no-sync python scripts/methods/estmf_inputs.py --dataset opencap [--contact-mapping any]
    uv run --no-sync python scripts/methods/estmf_inputs.py --dataset climb_wall_3 --clips a_00_01_part1
"""
from __future__ import annotations

import argparse
import json
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
OPENCAP = (paths.LABVALIDATION / "processed").resolve()
CW3_GT = paths.PETER / "climb_wall_3_gt" / "human"
#: Per dataset, the root of our pipeline's per-clip trees (``<root>/<clip>/sam3d/...``).
RUNS = {"parkour": PARKOUR / "out", "opencap": OPENCAP / "out",
        "climb_wall_3": paths.PETER / "out_climb_wall_3_single"}
#: Per dataset, where the estimator looks for our inputs.
OUT_DIRS = {"parkour": (paths.ESTMF / "data" / "Parkour-dataset" / "ours").resolve(),
            "opencap": (paths.ESTMF / "data" / "OpenCap" / "ours").resolve(),
            "climb_wall_3": (paths.ESTMF / "data" / "climb_wall_3" / "ours").resolve()}
OPENCAP_CLIPS = OPENCAP / "out" / "_methods" / "clips_natural.txt"
MODELS = _REPO.parent / "BetterHuman" / "models"

#: Goliath-308 index for each of OpenPose's 18 COCO joints. ``None`` is the neck,
#: which Goliath does not label: OpenPose puts it midway between the shoulders.
COCO18_FROM_GOLIATH = [0, None, 6, 8, 41, 5, 7, 62, 10, 12, 14, 9, 11, 13, 2, 1, 4, 3]
GOLIATH_SHOULDERS = (5, 6)

#: The contact recogniser's 9 joints, in its own order (the estimator's
#: ``lib/contact_states.py``), and which measured limb fills each on parkour / climb_wall_3.
CONTACT_REC_JOINTS = ["neck", "l_fingers", "r_fingers", "l_knee", "r_knee",
                      "l_ankle", "r_ankle", "l_toes", "r_toes"]
LIMB_OF_CONTACT_JOINT = {"l_fingers": "left_hand", "r_fingers": "right_hand",
                         "l_ankle": "left_foot", "r_ankle": "right_foot"}
#: OpenCap: our contact frames, by side and by end of the foot. ``toe_above`` is the dorsum
#: of the toe — it never fires on these trials (6 frames in 37388) but it is a forefoot
#: contact if it does.
HEEL_FRAMES = ["heel_{side}"]
FORE_FRAMES = ["ball_inner_{side}", "ball_outer_{side}", "big_toe_{side}", "toe_above_{side}"]
SIDES = ("left", "right")
#: The recogniser's label set, as the estimator's loader decodes it.
IN_CONTACT, NO_CONTACT, OCCLUDED = 0, 1, 2

#: SMPL-X ankles, and where they sit in the 19 Cocoplus joints HMR reports.
ANKLES = {"right_ankle": 0, "left_ankle": 5}
JOINT_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}


def select_clips(dataset: str, value: str | None) -> list[str]:
    """``--clips`` (comma-separated names, or a file of names) or the dataset's clip set.

    The sets: every Parkour clip with a SAM-3D fit, OpenCap's ``clips_natural.txt``, every
    climb_wall_3 clip with a three-camera reference.
    """
    if value is not None:
        return Path(value).read_text().split() if Path(value).is_file() else value.split(",")
    if dataset == "opencap":
        return OPENCAP_CLIPS.read_text().split()
    if dataset == "climb_wall_3":
        return sorted(d.name for d in CW3_GT.iterdir() if d.is_dir() and not d.name.startswith("_"))
    return sorted(d.name for d in RUNS["parkour"].iterdir()
                  if (d / "sam3d" / "smplx_params.npz").exists())


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


def one_hot(labels: np.ndarray) -> np.ndarray:
    """``(nf, 9)`` recogniser labels -> the ``(nf, 9, 4)`` one-hot the estimator loads."""
    out = np.zeros(labels.shape + (4,), np.uint8)
    np.put_along_axis(out, labels[..., None], 1, axis=-1)
    return out


def measured_contacts(limbs: list[str], loaded: np.ndarray,
                      captured: np.ndarray | None) -> tuple[np.ndarray, dict[str, int]]:
    """Measured ``(nf, L)`` limb contacts as the recogniser's one-hot, plus loaded counts.

    :param captured: ``(nf, L)``, False where the rig did not measure the limb: such a
        frame is unknown, not released, so it is reported as an occlusion, which the
        estimator carries forward. ``None``: every frame was measured.
    """
    labels = np.full((len(loaded), len(CONTACT_REC_JOINTS)), NO_CONTACT)
    counts = {}
    for slot, name in enumerate(CONTACT_REC_JOINTS):
        if name not in LIMB_OF_CONTACT_JOINT:
            continue
        column = limbs.index(LIMB_OF_CONTACT_JOINT[name])
        labels[:, slot] = np.where(loaded[:, column], IN_CONTACT, NO_CONTACT)
        counts[name] = int(loaded[:, column].sum())
        if captured is not None:
            labels[~captured[:, column], slot] = OCCLUDED
    return one_hot(labels), counts


def parkour_contacts(clip: str) -> tuple[np.ndarray, dict[str, int]]:
    """The plates and bar of ``<clip>/contacts.npz``."""
    truth = np.load(PARKOUR / clip / "contacts.npz")
    return measured_contacts([str(x) for x in truth["limbs"]],
                             np.asarray(truth["contacts"], bool),
                             np.asarray(truth["captured"], bool))


def climb_wall_3_contacts(clip: str, num_frames: int) -> tuple[np.ndarray, dict[str, int]]:
    """The boards, as labelled by the multi-view fit.

    That fit drops the two frames it has no lag for, so its labels are two rows short of
    the video; the last row is repeated.
    """
    saved = np.load(CW3_GT / clip / "human_optim" / "contacts_1.npz", allow_pickle=True)
    measured = np.asarray(saved["limb_contact"][0], bool)
    if len(measured) > num_frames:
        raise SystemExit(f"{clip}: {len(measured)} labelled frames, {num_frames} video frames")
    padding = np.repeat(measured[-1:], num_frames - len(measured), axis=0)
    return measured_contacts([str(x) for x in saved["limb_names"]],
                             np.concatenate([measured, padding], axis=0), None)


def opencap_contacts(run: Path, num_frames: int,
                     mapping: str) -> tuple[np.ndarray, dict[str, int]]:
    """Our contact labels as the recogniser's one-hot, plus per-side sole/toes/free counts.

    :param mapping: ``"heel"`` (a sole needs a heel frame, the forefoot alone is toes) or
        ``"any"`` (any loaded frame is a sole); see the module docstring.
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
        counts.update({f"{side} sole": int(sole.sum()), f"{side} toes": int(toes.sum()),
                       f"{side} free": int((~(sole | toes)).sum())})
    labels = np.full((num_frames, len(CONTACT_REC_JOINTS)), NO_CONTACT)
    for slot, name in enumerate(CONTACT_REC_JOINTS):
        if name in state:
            labels[:, slot] = np.where(state[name], IN_CONTACT, NO_CONTACT)
    # A frame our solve never produced is unknown, not released: report it the way an
    # occlusion is reported, which the estimator carries forward from the frame before.
    labels[~valid] = OCCLUDED
    return one_hot(labels), counts


def on_true_camera(joints_cam: np.ndarray, transl: np.ndarray, sam3d_k: np.ndarray,
                   true_k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Move a SAM-3D body from its own camera onto the calibrated one, keeping the picture.

    Re-solving the whole fit under the true camera is not on the table here, so the body is
    translated instead: the pelvis keeps its pixel, and the depth is scaled by
    ``f_true / f_sam3d`` so the apparent size is preserved. Both are exact for the pelvis
    and first-order for the rest of the body.

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


def export(dataset: str, clip: str, mapping: str, out_dir: Path) -> dict[str, int]:
    """Write ``<out_dir>/<clip>.npz``; returns the clip's contact counts."""
    run = RUNS[dataset] / clip
    init = np.load(run / "sam3d" / "smplx_params.npz")
    num_frames = int(init["num_frames"])

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
    sam3d_focal = (sam3d_k[0, 0] + sam3d_k[1, 1]) / 2.0
    extra, note = {}, ""
    if dataset == "parkour":
        intrinsics = sam3d_k.copy()
        intrinsics[0, 0] = intrinsics[1, 1] = sam3d_focal
        states, counts = parkour_contacts(clip)
    else:
        if dataset == "opencap":
            meta = json.loads((OPENCAP / clip / "meta.json").read_text())
            if num_frames != int(meta["n_frames"]):
                raise SystemExit(f"{clip}: sam3d has {num_frames} frames, meta.json {meta['n_frames']}")
            intrinsics = np.asarray(meta["K"], np.float64)
            states, counts = opencap_contacts(run, num_frames, mapping)
            extra = {"fps": np.float64(meta["fps"]), "contact_mapping": np.str_(mapping)}
        else:
            # cam_left is the rig's world frame, so its calibrated intrinsics are the camera
            # the estimator should solve in, and the geometry stage's extrinsics turn the
            # rig's world gravity into that camera's.
            cameras = np.load(CW3_GT / clip / "cameras.npz", allow_pickle=True)
            intrinsics = np.asarray(cameras["K"][0], np.float64)
            transform = np.load(run / "geometry" / "transform.npz")
            gravity = np.asarray(transform["extrinsics"][0][:3, :3] @ np.load(
                run / "geocalib" / "gravity.npz")["gravity_world"], np.float64)
            gravity /= np.linalg.norm(gravity)
            states, counts = climb_wall_3_contacts(clip, num_frames)
            extra = {"fps": np.float64(transform["fps"]), "gravity": gravity}
            note = f", gravity [{gravity[0]:+.3f} {gravity[1]:+.3f} {gravity[2]:+.3f}]"
        ankles = [JOINT_INDEX[j] for j in ANKLES]
        before = project(joints_cam[:, ankles], sam3d_k)
        joints_cam, transl = on_true_camera(joints_cam, transl, sam3d_k, intrinsics)
        ankle_shift = float(np.median(np.linalg.norm(
            project(joints_cam[:, ankles], intrinsics) - before, axis=-1)))
        extra["sam3d_focal"] = np.float64(sam3d_focal)
        note = f" (SAM-3D had {sam3d_focal:.1f}, ankles moved {ankle_shift:.1f} px){note}"

    focal = float((intrinsics[0, 0] + intrinsics[1, 1]) / 2.0)
    principal = intrinsics[:2, 2]
    # Only the two ankles are ever read back, to stand in where Sapiens is unsure.
    ankles_2d = np.zeros((num_frames, 19, 2))
    for joint, slot in ANKLES.items():
        ankles_2d[:, slot] = project(joints_cam[:, JOINT_INDEX[joint]], intrinsics)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{clip}.npz",
             poses=poses, transl=transl, betas_smplx=betas,
             smplx_joints_rest=joints_rest, focal=np.float64(focal), principal=principal,
             coco18=coco18_from_sapiens(run / "sapiens" / "pose.npz"),
             hmr_joints_2d=ankles_2d, contact_states=states,
             num_frames=np.int64(num_frames), **extra)
    print(f"  {clip}: {num_frames} frames, focal {focal:.1f} px{note}, "
          + " ".join(f"{name} {count}" for name, count in counts.items()))
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=sorted(RUNS))
    ap.add_argument("--clips", default=None,
                    help="comma-separated clip names, or a file listing them (default: the "
                         "dataset's clip set, see select_clips)")
    ap.add_argument("--contact-mapping", choices=("heel", "any"), default="heel",
                    help="opencap only: how our contact frames become the sole and toes channels")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: the estimator's data/<dataset>/ours)")
    args = ap.parse_args(argv)

    clips = select_clips(args.dataset, args.clips)
    out_dir = args.out or OUT_DIRS[args.dataset]
    print(f"(estmf_inputs {args.dataset}) {len(clips)} clips -> {out_dir}")
    total: dict[str, int] = {}
    for clip in clips:
        for name, count in export(args.dataset, clip, args.contact_mapping, out_dir).items():
            total[name] = total.get(name, 0) + count
    print(f"\ncontact frames over {len(clips)} clips: "
          + "  ".join(f"{name} {count}" for name, count in total.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
