"""Three ways of estimating parkour motion and forces, on the same 28 clips.

The LAAS Parkour dataset measures what a person's hands and feet actually did:
Vicon gives 16 joint positions, floor plates and an instrumented bar give a 3D
force and moment per limb, one row per video frame. Three methods are scored
against it here, all on the same clips, the same frames and the same truth:

* **Li et al., their inputs** — the method as published: HMR for the body
  initialisation, OpenPose for the 2D joints, their learned recogniser for
  contact. Only kv01_PKFC can be run this way, because the repository ships
  those three inputs for that one clip and no other.
* **Li et al., our inputs** — the same solver, given SAM-3D-Body's SMPL-X
  initialisation, Sapiens 2D joints and the rig's measured contact instead.
* **PhysPT** (Zhang et al., CVPR 2024, ``../other/PhysPT``) — a physics-aware
  transformer that refines a per-frame SMPL motion and infers the ground reaction
  forces behind it. Fed our SAM-3D-Body initialisation in place of its CLIFF front
  end (``scripts/methods/physpt_inputs.py --dataset parkour``), so it starts from the same body. It
  models contact with a fixed ground plane only, so a hand on the bar is a force it
  cannot represent.
* **Ours, learned** — our contact-anything model (``../contact_anything``, the
  round-10 ``L_body`` recipe): a temporal transformer behind the frozen per-frame
  body that predicts contact and forces per limb, run on the out-trees with
  ``scripts/predict_reconstruction.py``. Its forces are in body-weight units and
  become newtons through the mass of our optimised body.
* **Ours** — our own kinematics + dynamics stage, on those same inputs.

Two families of metric, because they answer different questions.

The **paper's metrics** are the ones published on this data, so they place our
numbers next to theirs. Pose: per-frame Procrustes alignment (rotation,
translation *and* scale) over the 16 joints, mean residual in millimetres. Force:
the vector difference against the measured wrench, in newtons and newton-metres,
per limb, skipping the frames the rig did not capture.

The **our metrics** rows exist because the paper's two hide things. Its pose
alignment re-fits scale every frame, so a body of the wrong size or a trajectory
that drifts both score as if they were right; ours aligns once per clip and keeps
the true size. Its force metric is a vector difference, which mixes up "how hard"
with "which way" and says nothing about whether the load went to the right limb;
ours separates the size error, how well it tracks over time, and the share of the
total each limb was given.

A **predict nothing** row is included for reference. It is not a method: it is
what a method scores by outputting zero force, and any real number should beat it
comfortably.

Two ground truths ship for the forces, and they are not the same file. The
dataset's own release (``--gt published``, the default) has four channels and
moments about the contact, ~10 cm from the limb. The copy bundled with the
estimator (``--gt bundled``) has five channels, and its hand moments are taken
about some far-away origin — they average 180-220 N.m where the published ones
average 11-15. The linear forces agree between the two to a few newtons, so only
the torque columns really depend on this choice; the paper's published torque
numbers are against the bundled copy. The joint positions are byte-identical in
both, so pose never depends on it.

Usage::

    uv run --no-sync python scripts/diagnostics/compare_parkour_methods.py
    uv run --no-sync python scripts/diagnostics/compare_parkour_methods.py --gt bundled
    uv run --no-sync python scripts/diagnostics/compare_parkour_methods.py --clips kv01_PKFC
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.body import load_body  # noqa: E402

_DIAGNOSTICS = Path(__file__).resolve().parent
if str(_DIAGNOSTICS) not in sys.path:
    sys.path.insert(0, str(_DIAGNOSTICS))
from score_parkour_pose import predicted_joints as _init_joints  # noqa: E402
from tools.human_optim.boards import LIMBS, limb_of_frame  # noqa: E402
from tools.human_optim.qmath import normalize_q  # noqa: E402
from tools.paths import ESTMF, PARKOUR  # noqa: E402

DATASET = PARKOUR.resolve()
TREE = DATASET / "processed"
ESTIMATOR = ESTMF.resolve()

#: Where each force ground truth lives. The joints come from the dataset's copy either
#: way -- the two files are byte-identical.
FORCE_TRUTH = {"published": DATASET / "gt_motion_forces" / "contact_forces_local.pkl",
               "bundled": ESTIMATOR / "data" / "Parkour-dataset" / "gt_motion_forces"
                                    / "contact_forces_local.pkl"}

#: Clip-name prefix -> the parkour technique, in the order the paper reports them.
ACTIONS = {"kv": "Kong-vault", "mu": "Muscle-up", "pu": "Pull-up", "sv": "Safety-vault"}

#: The dataset's four force channels, and our own limb order mapped onto them.
CHANNELS = ("l foot", "r foot", "l hand", "r hand")
PARKOUR_LIMBS = ("left_foot", "right_foot", "left_hand", "right_hand")

#: Each Parkour joint and the joint of our 52-joint body standing in for it.
#: ``toes`` is the ball of the foot; ``fingers`` the middle knuckle.
CORRESPONDENCE = {
    "l_hip": "left_hip", "l_knee": "left_knee", "l_ankle": "left_ankle", "l_toes": "left_foot",
    "r_hip": "right_hip", "r_knee": "right_knee", "r_ankle": "right_ankle", "r_toes": "right_foot",
    "l_shoulder": "left_shoulder", "l_elbow": "left_elbow", "l_wrist": "left_wrist",
    "l_fingers": "left_middle1",
    "r_shoulder": "right_shoulder", "r_elbow": "right_elbow", "r_wrist": "right_wrist",
    "r_fingers": "right_middle1",
}

#: The point each limb's moment is taken about, named in our body.
MOMENT_ABOUT = {"left_foot": "left_ankle", "right_foot": "right_ankle",
                "left_hand": "left_wrist", "right_hand": "right_wrist"}

#: The learned model's six force groups, in its ``forces.npz`` order; ``left_foot`` is
#: the toe and ``left_ankle`` the heel, each applied at the joint of that name.
LEARNED_GROUPS = ("left_hand", "right_hand", "left_foot", "right_foot", "left_ankle", "right_ankle")
GRAVITY = 9.81

PHYSPT = (_REPO.parent / "other" / "PhysPT").resolve()
#: SMPL joint standing in for each Parkour joint (``CORRESPONDENCE`` order).
PHYSPT_JOINTS = [1, 4, 7, 10, 2, 5, 8, 11, 16, 18, 20, 22, 17, 19, 21, 23]
#: PhysPT's 29 contact vertices belong to SMPL body parts; the parts of each Parkour
#: channel, and the SMPL joint its moment is taken about.
PHYSPT_CHANNELS = {"l foot": ({7, 10}, 7), "r foot": ({8, 11}, 8),
                   "l hand": ({20, 22}, 20), "r hand": ({21, 23}, 21)}

#: A limb pushing harder than this counts as loaded, for the frame check.
LOADED_N = 200.0

LABEL = 38
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- truth


def ground_truth(which: str) -> dict[str, dict[str, np.ndarray]]:
    """Every clip's measured joints and wrenches, trimmed to the video's frames."""
    with open(DATASET / "gt_motion_forces" / "joint_3d_positions.pkl", "rb") as handle:
        joints = pickle.load(handle, encoding="latin1")
    with open(FORCE_TRUTH[which], "rb") as handle:
        forces = pickle.load(handle, encoding="latin1")

    truth = {}
    for clip, entry in joints.items():
        names = [str(x) for x in entry["joint_names"]]
        frames = entry["joint_3d_positions"].shape[0] - 3   # the evaluator's own rule
        wrench = forces[clip]
        truth[clip] = dict(
            joints=entry["joint_3d_positions"][:frames][
                :, [names.index(name) for name in CORRESPONDENCE]].astype(np.float64),
            wrench=np.asarray(wrench["contact_forces_local"], np.float64)[:frames, :4],
            captured=np.asarray(wrench["mask_uncaptured_forces"])[:frames, :4] == 0)
    return truth


# ------------------------------------------------------------------------ our body


def init_prediction(clip: str) -> tuple[np.ndarray, None]:
    """The SAM-3D-Body per-frame init, converted to SMPL-X: 16 joints, no forces.

    Posed in the camera frame; the camera is static here, so that is one frame for
    the whole clip and the world metrics are fair.
    """
    joints, _ = _init_joints(TREE / "out" / clip / "sam3d" / "smplx_params.npz")
    return joints, None


def our_prediction(clip: str, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """Our solved body: the 16 joints and the 4 limb wrenches, both in our world."""
    body_file = np.load(TREE / "out" / clip / variant / "kindyn_1.npz", allow_pickle=True)
    joint_names = [str(x) for x in body_file["joint_names"]]
    frame_names = [str(x) for x in body_file["contact_frame_names"]]
    joints_world = np.asarray(body_file["joints_world"][0], np.float64)
    frame_forces = np.asarray(body_file["frame_forces"][0], np.float64)
    points = contact_frame_points(body_file, frame_names)

    joints = joints_world[:, [joint_names.index(CORRESPONDENCE[name])
                              for name in CORRESPONDENCE]]
    wrench = np.zeros((len(joints_world), 4, 6))
    for channel, limb in enumerate(PARKOUR_LIMBS):
        columns = [c for c, name in enumerate(frame_names) if limb_of_frame(name) == limb]
        about = joints_world[:, joint_names.index(MOMENT_ABOUT[limb])]
        force = frame_forces[:, columns]
        wrench[:, channel, :3] = force.sum(1)
        wrench[:, channel, 3:] = np.cross(points[:, columns] - about[:, None], force).sum(1)
    return joints, wrench


@torch.no_grad()
def contact_frame_points(body_file, frame_names: list[str], chunk: int = 64) -> np.ndarray:
    """World positions of the contact frames the forces act at, ``(N, F, 3)``."""
    body = load_body(device=DEVICE, frames_allow=frame_names)
    if list(body.contact_frame_names) != frame_names:
        raise SystemExit("[compare_parkour] the body's contact frames are not the saved ones")
    betas = torch.as_tensor(body_file["betas"][0], dtype=torch.float32, device=DEVICE)
    q = torch.as_tensor(body_file["q"][0], dtype=torch.float32, device=DEVICE)
    scale = float(body_file["scale"][0])
    posed = []
    for start in range(0, q.shape[0], chunk):
        end = min(start + chunk, q.shape[0])
        out = body.forward(betas[None].expand(end - start, -1),
                           normalize_q(q[start:end], body.num_joints))
        root = out.joints[:, :1, :]
        posed.append(root + scale * (out.frames_world[..., :3] - root))
    return torch.cat(posed, 0).cpu().numpy().astype(np.float64)


# ------------------------------------------------------------------ learned model


def learned_prediction(clip: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Our learned model's body and forces: the 16 joints and 4 limb wrenches, our world."""
    tree = TREE / "out" / clip
    if not (tree / "predictions" / "forces.npz").exists():
        return None
    body = np.load(tree / "predictions" / "smplx.npz", allow_pickle=True)
    forces = np.load(tree / "predictions" / "forces.npz", allow_pickle=True)
    reference = np.load(tree / "human_optim_gt_contact" / "kindyn_1.npz", allow_pickle=True)
    joint_names = [str(x) for x in reference["joint_names"]]      # the same 52-joint body
    mass = float(np.asarray(reference["total_mass"], float)[0])
    if [str(x) for x in forces["limbs"]] != list(LEARNED_GROUPS):
        raise SystemExit(f"[compare_parkour] unexpected force groups in {clip}")

    joints_world = np.asarray(body["joints_world"][0], np.float64)
    if not np.isfinite(joints_world).all():
        raise SystemExit(f"[compare_parkour] {clip}: the learned model left frames uncovered")
    newton = np.nan_to_num(np.asarray(forces["forces_world"][0], np.float64)) * mass * GRAVITY
    joint = lambda name: joints_world[:, joint_names.index(name)]  # noqa: E731

    joints = joints_world[:, [joint_names.index(CORRESPONDENCE[name]) for name in CORRESPONDENCE]]
    wrench = np.zeros((len(joints_world), 4, 6))
    for channel, limb in enumerate(PARKOUR_LIMBS):
        side = limb.split("_")[0]
        if limb.endswith("hand"):                       # one force, at the wrist itself
            wrench[:, channel, :3] = newton[:, LEARNED_GROUPS.index(limb)]
        else:                                           # toe + heel; the toe has a lever arm
            toe = newton[:, LEARNED_GROUPS.index(f"{side}_foot")]
            heel = newton[:, LEARNED_GROUPS.index(f"{side}_ankle")]
            wrench[:, channel, :3] = toe + heel
            wrench[:, channel, 3:] = np.cross(joint(f"{side}_foot") - joint(f"{side}_ankle"), toe)
    return joints, wrench


# ----------------------------------------------------------------------- PhysPT


def physpt_model() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PhysPT's SMPL joint regressor, its 29 contact vertices and the body part of each."""
    with open(PHYSPT / "assets" / "data" / "smpl" / "neutral.pkl", "rb") as handle:
        regressor = np.asarray(pickle.load(handle)["V_regressor"], np.float64)   # (24, 6890)
    with open(PHYSPT / "assets" / "data" / "physics.pkl", "rb") as handle:
        physics = pickle.load(handle)
    vertices = np.asarray(physics["contact_vertexidxall"], np.int64)
    parts = np.concatenate([[part] * len(v) for part, v in enumerate(physics["contact_vertexidx"])])
    return regressor, vertices, parts.astype(np.int64)


def physpt_prediction(clip: str) -> tuple[np.ndarray, np.ndarray] | None:
    """PhysPT's refined body and forces: the 16 joints and 4 wrenches, in its own world."""
    path = PHYSPT / "parkour" / f"{clip}_output.npz"
    if not path.exists():
        return None
    saved = np.load(path)
    regressor, vertices, parts = physpt_model()
    frames = np.asarray(saved["frame_idx"])
    if not np.array_equal(frames, np.arange(len(frames))):
        raise SystemExit(f"[compare_parkour] PhysPT frames of {clip} are not the video's")
    mesh = np.asarray(saved["physics_vertices_world"], np.float64)         # (T, 6890, 3)
    force = np.asarray(saved["pred_grf"][0], np.float64)                    # (T-2, 29, 3) N
    joints24 = np.einsum("jv,tvc->tjc", regressor, mesh)
    points = mesh[:len(force), vertices]

    wrench = np.zeros((len(mesh), 4, 6))                                    # zero past T-2
    for channel, name in enumerate(CHANNELS):
        part_set, about = PHYSPT_CHANNELS[name]
        columns = [c for c, part in enumerate(parts) if part in part_set]
        lever = points[:, columns] - joints24[:len(force), about][:, None]
        wrench[:len(force), channel, :3] = force[:, columns].sum(1)
        wrench[:len(force), channel, 3:] = np.cross(lever, force[:, columns]).sum(1)
    return joints24[:, PHYSPT_JOINTS], wrench


# ---------------------------------------------------------------------- estimator


def estimator_prediction(results: Path, save_name: str,
                         clip: str) -> tuple[np.ndarray, np.ndarray] | None:
    """One clip's Li-et-al. output: the 16 joints and the 4 wrenches it predicted."""
    path = results / "run" / "evaluation" / f"{save_name}_s7" / f"{clip}_pred.npz"
    if not path.exists():
        return None
    saved = np.load(path)
    return (np.asarray(saved["joints_16"], np.float64),
            np.asarray(saved["forces_parkour"], np.float64)[:, :4])


# ------------------------------------------------------------------------ metrics


def procrustes_mpjpe(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """The paper's pose metric: per-frame similarity fit, mean residual mm, and its scale."""
    residuals, scales = [], []
    for pred, gt in zip(predicted * 1e3, truth * 1e3):
        pred_centred, gt_centred = pred - pred.mean(0), gt - gt.mean(0)
        cross = pred_centred.T @ gt_centred
        u, _, vt = np.linalg.svd(cross)
        v = vt.T
        rotation = v @ np.diag([1.0, 1.0, np.linalg.det(v) * np.linalg.det(u)]) @ u.T
        scale = np.trace(rotation @ cross) / (pred_centred ** 2).sum()
        residuals.append(np.linalg.norm(pred_centred @ (scale * rotation).T - gt_centred, axis=1))
        scales.append(scale)
    return float(np.mean(residuals)), float(np.mean(scales))


def clip_rotation(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """One rotation for the whole clip, taking a prediction's axes onto the MoCap world's.

    Fitted on the joints with each frame's centroid removed, so only orientation is
    being solved for. This is what lets a force solved in our own world be compared
    with a truth measured in the laboratory's.
    """
    pred = (predicted - predicted.mean(1, keepdims=True)).reshape(-1, 3)
    gt = (truth - truth.mean(1, keepdims=True)).reshape(-1, 3)
    u, _, vt = np.linalg.svd(pred.T @ gt)
    v = vt.T
    return v @ np.diag([1.0, 1.0, np.linalg.det(v) * np.linalg.det(u)]) @ u.T


def rigid_mpjpe(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Our pose metric: one alignment for the clip, true size kept, mean residual mm."""
    turned = predicted @ clip_rotation(predicted, truth).T
    shift = (truth - turned).reshape(-1, 3).mean(0)
    return float(np.linalg.norm((turned + shift - truth) * 1e3, axis=-1).mean())


HIPS = [list(CORRESPONDENCE).index(j) for j in ("l_hip", "r_hip")]


def world_mpjpes(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """The world-grounded pose metrics of WHAM / GVHMR, true size kept, in mm.

    No method shares a frame with the MoCap world, so a clip rotation is fitted first
    (on all frames, as in :func:`rigid_mpjpe`). Then ``MPJPE`` re-centres every frame
    on the mid-hip, ``W-MPJPE`` fits the shift on the first two frames only (drift
    counts), ``WA-MPJPE`` fits it on the whole clip.
    """
    turned = predicted @ clip_rotation(predicted, truth).T
    error = lambda pred: float(np.linalg.norm((pred - truth) * 1e3, axis=-1).mean())  # noqa: E731
    root = lambda joints: joints[:, HIPS].mean(1, keepdims=True)  # noqa: E731
    return {
        "MPJPE": error(turned - root(turned) + root(truth)),
        "W-MPJPE": error(turned + (truth[:2] - turned[:2]).reshape(-1, 3).mean(0)),
        "WA-MPJPE": error(turned + (truth - turned).reshape(-1, 3).mean(0)),
    }


def paper_force_errors(predicted: np.ndarray, truth: np.ndarray,
                       captured: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The paper's force metric: mean vector error per channel, over captured frames only."""
    mask = captured[..., None]
    linear = np.linalg.norm(np.where(mask, predicted[..., :3] - truth[..., :3], 0.0), axis=-1)
    torque = np.linalg.norm(np.where(mask, predicted[..., 3:] - truth[..., 3:], 0.0), axis=-1)
    counts = np.maximum(captured.sum(0), 1)
    return linear.sum(0) / counts, torque.sum(0) / counts


def our_force_metrics(predicted: np.ndarray, truth: np.ndarray,
                      captured: np.ndarray) -> dict[str, float]:
    """Size, timing and distribution of the load, without reference to direction."""
    solved = np.where(captured, np.linalg.norm(predicted[..., :3], axis=-1), np.nan)
    measured = np.where(captured, np.linalg.norm(truth[..., :3], axis=-1), np.nan)
    full = captured.all(1)
    share = lambda size: size / np.maximum(np.nansum(size, 1, keepdims=True), 1e-6)  # noqa: E731
    pair = np.isfinite(solved) & np.isfinite(measured)
    correlation = (np.corrcoef(solved[pair], measured[pair])[0, 1]
                   if solved[pair].std() > 0 else float("nan"))
    return {
        "limb MAE [N]": float(np.nanmean(np.abs(solved - measured))),
        "limb RMSE [N]": float(np.sqrt(np.nanmean((solved - measured) ** 2))),
        "corr": float(correlation),
        "share err [pp]": float(100 * np.nanmean(np.abs(share(solved) - share(measured)))),
        "total MAE [N]": float(np.mean(np.abs(
            np.linalg.norm(predicted[full][..., :3].sum(1), axis=-1)
            - np.linalg.norm(truth[full][..., :3].sum(1), axis=-1)))),
    }


def up_angle(wrench: np.ndarray, captured: np.ndarray) -> float:
    """Degrees between a method's mean loaded-foot force and the MoCap world's up.

    A frame check, not a score: the feet push the person up, so a prediction living
    in the same axes as the truth comes out near zero here.
    """
    force = wrench[:, :2, :3][captured[:, :2]]
    force = force[np.linalg.norm(force, axis=-1) > LOADED_N]
    if not len(force):
        return float("nan")
    mean = force.mean(0) / np.linalg.norm(force.mean(0))
    return float(np.degrees(np.arccos(np.clip(mean[2], -1, 1))))


# ------------------------------------------------------------------------- tables


def collect(clips: list[str], truth: dict, methods: dict) -> dict[str, dict]:
    """Score every method on every clip it produced, pooling the per-frame arrays."""
    scored: dict[str, dict] = {}
    for label, loader in methods.items():
        per_clip, predictions, measurements, masks = {}, [], [], []
        for clip in clips:
            result = loader(clip)
            if result is None:
                continue
            joints, wrench = result
            gt = truth[clip]
            frames = min(len(wrench) if wrench is not None else len(joints), len(gt["joints"]))
            if joints is not None:
                mpjpe, scale = procrustes_mpjpe(joints[:frames], gt["joints"][:frames])
                per_clip[clip] = dict(mpjpe=mpjpe, scale=scale,
                                      rigid=rigid_mpjpe(joints[:frames], gt["joints"][:frames]),
                                      **world_mpjpes(joints[:frames], gt["joints"][:frames]))
            else:
                per_clip[clip] = {}
            if wrench is None:
                continue
            predictions.append(wrench[:frames])
            measurements.append(gt["wrench"][:frames])
            masks.append(gt["captured"][:frames])
        if not per_clip:
            continue
        if not predictions:                                  # pose only, no forces
            scored[label] = dict(clips=per_clip)
            continue
        predicted = np.concatenate(predictions)
        measured = np.concatenate(measurements)
        captured = np.concatenate(masks)
        linear, torque = paper_force_errors(predicted, measured, captured)
        scored[label] = dict(clips=per_clip, linear=linear, torque=torque,
                             up=up_angle(predicted, captured),
                             **our_force_metrics(predicted, measured, captured))
    return scored


def _cell(value: float, width: int = 12, decimals: int = 1) -> str:
    return f"{'—':>{width}s}" if not np.isfinite(value) else f"{value:{width}.{decimals}f}"


def pose_tables(scored: dict) -> None:
    print(f"\n{'=' * 96}\nPOSE\n{'=' * 96}")
    print("\nThe paper's metric — Procrustes MPJPE [mm], per technique (lower is better)\n")
    print(f"{'':{LABEL}s}" + "".join(f"{a:>13s}" for a in ACTIONS.values()) + f"{'Avg':>13s}")
    for label, result in scored.items():
        scores = {c: v["mpjpe"] for c, v in result["clips"].items() if v}
        if not scores:
            continue
        cells = ""
        for prefix in ACTIONS:
            values = [v for c, v in scores.items() if c.startswith(prefix)]
            cells += f"{np.mean(values):13.1f}" if values else f"{'—':>13s}"
        print(f"{label:{LABEL}s}{cells}{np.mean(list(scores.values())):13.1f}")

    print("\nOur metrics — the same poses, without the per-frame re-fit\n")
    print(f"{'':{LABEL}s}{'clips':>7s}{'PA-MPJPE':>11s}{'rigid MPJPE':>13s}{'size':>8s}")
    for label, result in scored.items():
        values = [v for v in result["clips"].values() if v]
        if not values:
            continue
        print(f"{label:{LABEL}s}{len(values):7d}"
              f"{np.mean([v['mpjpe'] for v in values]):11.1f}"
              f"{np.mean([v['rigid'] for v in values]):13.1f}"
              f"{np.mean([v['scale'] for v in values]):8.2f}")
    print("\n  PA-MPJPE      the column above: rotation, shift AND scale re-fitted every frame")
    print("  rigid MPJPE   one rotation and shift for the whole clip, true size kept")
    print("  size          the scale that per-frame fit chose — 1.00 means the body was")
    print("                already the right size, below 1 that the prediction is too big")


def force_tables(scored: dict, which: str) -> None:
    print(f"\n{'=' * 96}\nFORCES — scored against the {which} ground truth\n{'=' * 96}")
    print("\nThe paper's metric — mean vector error against the measured wrench\n")
    print(f"{'':{LABEL}s}{'linear force error [N]':^48s}  {'torque error [N.m]':^48s}")
    print(f"{'':{LABEL}s}" + "".join(f"{c:>12s}" for c in CHANNELS)
          + "  " + "".join(f"{c:>12s}" for c in CHANNELS))
    for label, result in scored.items():
        if "linear" not in result:
            continue
        print(f"{label:{LABEL}s}" + "".join(_cell(v) for v in result["linear"])
              + "  " + "".join(_cell(v) for v in result["torque"]))

    print("\nOur metrics — how big, how well timed, and to which limb\n")
    keys = ("limb MAE [N]", "corr", "share err [pp]", "total MAE [N]")
    print(f"{'':{LABEL}s}" + "".join(f"{k:>16s}" for k in keys))
    for label, result in scored.items():
        if "linear" not in result:
            continue
        print(f"{label:{LABEL}s}" + "".join(
            _cell(result[k], 16, 3 if k == "corr" else 1) for k in keys))
    print("\n  limb MAE      error in how hard each limb pushed, ignoring direction")
    print("  corr          how well that size tracks the measured one over time")
    print("  share err     each limb's share of the total load against the measured share —")
    print("                this is where 'right total, wrong limb' shows up")
    print("  total MAE     error in the total force on the body, all four limbs summed")

    print("\nFrame check — angle between each method's mean loaded-foot force and "
          "the MoCap world's up\n")
    for label, result in scored.items():
        if "up" in result:
            print(f"{label:{LABEL}s}{_cell(result['up'], 8)} deg")


FINAL_ROWS = ("SAM-3D-Body init (converted to SMPL-X)", "Li et al., our inputs (SAM-3D+Sapiens)",
              "PhysPT, our init (SAM-3D)", "Ours (kinematics + dynamics)",
              "Ours, learned model (L_body)")
FINAL_POSE = ("MPJPE", "mpjpe", "WA-MPJPE", "W-MPJPE")     # mpjpe = the PA one, above
FINAL_FORCE = ("limb MAE [N]", "corr", "share err [pp]", "limb RMSE [N]")


def final_table(scored: dict) -> None:
    """One markdown row per method: the four pose metrics, then the four force ones."""
    heads = ["MPJPE", "PA-MPJPE", "WA-MPJPE", "W-MPJPE", "limb MAE [N]", "corr",
             "share err [pp]", "RMSE [N]"]
    print("\n| | " + " | ".join(heads) + " |\n|---|" + "---|" * len(heads))
    for label in FINAL_ROWS:
        if label not in scored:
            continue
        result = scored[label]
        pose = [np.mean([v[k] for v in result["clips"].values()]) for k in FINAL_POSE]
        force = [result.get(k, float("nan")) for k in FINAL_FORCE]
        cells = [f"{x:.1f}" for x in pose] + [
            "—" if not np.isfinite(x) else f"{x:.3f}" if k == "corr" else f"{x:.1f}"
            for k, x in zip(FINAL_FORCE, force)]
        print(f"| {label} | " + " | ".join(cells) + " |")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", default=None, help="comma-separated names (default: all)")
    ap.add_argument("--gt", choices=sorted(FORCE_TRUTH), default="published",
                    help="which force ground truth to score against")
    ap.add_argument("--variant", default="human_optim_gt_contact",
                    help="which of our runs to score (measured contacts by default)")
    args = ap.parse_args(argv)

    truth = ground_truth(args.gt)
    clips = sorted(truth)
    if args.clips:
        clips = [c for c in clips if c in set(args.clips.split(","))]

    def aligned(loader):
        """A method whose world is its own: turn its wrenches into the MoCap axes."""
        def load(clip: str):
            result = loader(clip)
            if result is None:
                return None
            joints, wrench = result
            frames = min(len(joints), len(truth[clip]["joints"]))
            rotation = clip_rotation(joints[:frames], truth[clip]["joints"][:frames])
            turned = wrench.reshape(len(wrench), 4, 2, 3) @ rotation.T
            return joints, turned.reshape(len(wrench), 4, 6)
        return load

    methods = {
        "SAM-3D-Body init (converted to SMPL-X)":
            init_prediction,
        "Li et al., their inputs (HMR+OpenPose)":
            lambda c: estimator_prediction(
                ESTIMATOR / "validation" / "results" / "v25_original", "orig_v25", c),
        "Li et al., our inputs (SAM-3D+Sapiens)":
            lambda c: estimator_prediction(
                ESTIMATOR / "validation" / "results" / "v25_ours", "ours_v25", c),
        "PhysPT, our init (SAM-3D)":
            aligned(physpt_prediction),
        "Ours, learned model (L_body)":
            aligned(learned_prediction),
        "Ours (kinematics + dynamics)":
            aligned(lambda c: our_prediction(c, args.variant)),
        "Predict nothing (zero force)":
            lambda c: (None, np.zeros((len(truth[c]["wrench"]), 4, 6))),
    }

    scored = collect(clips, truth, methods)
    frames = sum(len(truth[c]["joints"]) for c in clips)
    print(f"\nLAAS Parkour — {len(clips)} clips, {frames} frames, 25 fps. "
          "Contacts are MEASURED for every row but the first.")
    pose_tables(scored)
    force_tables(scored, args.gt)
    final_table(scored)
    print("\nPhysPT's forces and both of ours are solved in a world of their own; the rotation")
    print("that best aligns each method's skeleton to the MoCap one turns them into the frame")
    print("the truth lives in. The frame check above is what says that worked.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
