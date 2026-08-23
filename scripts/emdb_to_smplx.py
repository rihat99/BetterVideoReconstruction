"""Convert an EMDB sequence's GT SMPL trajectory to SMPL-X.

EMDB ships gendered SMPL parameters in the world frame. We convert them to the
neutral SMPL-X model our pipeline uses:

- body pose: SMPL body joints 1..21 are the same joints in the same order as
  SMPL-X, so ``body_pose[:, :63]`` copies over directly (SMPL's two extra hand
  joints are dropped; SMPL-X hands are left flat).
- betas: rest-pose joints are linear in betas, so the neutral SMPL-X betas that
  best reproduce the gendered SMPL rest skeleton come from one least-squares
  solve over the 22 shared joints (pelvis-relative).
- transl: both models place the pelvis at ``rest_pelvis + transl``, so a
  constant offset (SMPL rest pelvis minus SMPL-X rest pelvis) aligns the bodies
  in world.

Writes ``gt_smplx.npz`` next to the pkl and reports the FK joint error of the
conversion over the whole sequence.

Usage:
    uv run python scripts/emdb_to_smplx.py --sequence /path/to/P0/10_outdoor_rom
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

MODELS = Path("/data3/rikhat.akizhanov/better/BetterHuman/models")
NUM_BETAS = 10
NUM_SHARED_JOINTS = 22  # pelvis..wrists: identical order in SMPL and SMPL-X


def load_model(path: Path) -> dict:
    """Load the pieces of a SMPL-family npz needed for joint math."""
    d = np.load(path, allow_pickle=True)
    return {
        "v_template": d["v_template"].astype(np.float64),
        "shapedirs": d["shapedirs"][:, :, :NUM_BETAS].astype(np.float64),
        "J_regressor": d["J_regressor"].astype(np.float64),
        "parents": d["kintree_table"][0].astype(np.int64),
    }


def rest_joints(model: dict, betas: np.ndarray) -> np.ndarray:
    """Rest-pose joint locations (J, 3) for the given betas."""
    v = model["v_template"] + model["shapedirs"] @ betas
    return model["J_regressor"] @ v


def rodrigues(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (..., 3) to rotation matrices (..., 3, 3)."""
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = aa / np.maximum(angle, 1e-12)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    K = np.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], axis=-1
    ).reshape(*aa.shape[:-1], 3, 3)
    a = angle[..., None]
    return np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * (K @ K)


def fk_joints(model: dict, betas: np.ndarray, poses: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """World joint positions (N, J, 3) from per-frame axis-angle poses (N, J*3)."""
    J = rest_joints(model, betas)
    parents = model["parents"]
    n_joints = len(parents)
    rots = rodrigues(poses.reshape(len(poses), -1, 3))
    if rots.shape[1] < n_joints:  # absent joints (e.g. flat hands) stay identity
        pad = np.broadcast_to(np.eye(3), (len(poses), n_joints - rots.shape[1], 3, 3))
        rots = np.concatenate([rots, pad], axis=1)
    world_rot = np.zeros((len(poses), n_joints, 3, 3))
    world_pos = np.zeros((len(poses), n_joints, 3))
    world_rot[:, 0] = rots[:, 0]
    world_pos[:, 0] = J[0]
    for j in range(1, n_joints):
        p = parents[j]
        world_rot[:, j] = world_rot[:, p] @ rots[:, j]
        world_pos[:, j] = world_pos[:, p] + (world_rot[:, p] @ (J[j] - J[p])[:, None])[..., 0]
    return world_pos + trans[:, None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True, help="EMDB sequence dir")
    args = parser.parse_args()

    pkl = next(args.sequence.glob("*_data.pkl"))
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    smpl_params = data["smpl"]
    gender = data["gender"]

    smpl = load_model(MODELS / "smpl" / f"SMPL_{gender.upper()}.npz")
    smplx = load_model(MODELS / "smplx" / "SMPLX_NEUTRAL.npz")

    # Fit neutral SMPL-X betas to the gendered SMPL rest skeleton (both
    # pelvis-relative; rest joints are linear in betas -> least squares).
    betas_smpl = smpl_params["betas"].astype(np.float64)
    target = rest_joints(smpl, betas_smpl)[:NUM_SHARED_JOINTS]
    target_rel = target - target[0]
    base = smplx["J_regressor"] @ smplx["v_template"]
    basis = np.einsum("jv,vck->jck", smplx["J_regressor"], smplx["shapedirs"])
    a_mat = (basis[:NUM_SHARED_JOINTS] - basis[0]).reshape(-1, NUM_BETAS)
    b_vec = (target_rel - (base[:NUM_SHARED_JOINTS] - base[0])).reshape(-1)
    betas_x, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    rest_err = np.linalg.norm((a_mat @ betas_x - b_vec).reshape(-1, 3), axis=-1)
    print(f"rest-skeleton fit ({gender} SMPL -> neutral SMPL-X, {NUM_BETAS} betas): "
          f"mean {rest_err.mean() * 1000:.1f} mm, max {rest_err.max() * 1000:.1f} mm")

    # Pose copy + pelvis-offset trans compensation.
    n = data["n_frames"]
    global_orient = smpl_params["poses_root"].astype(np.float64)
    body_pose = smpl_params["poses_body"][:, :63].astype(np.float64)
    offset = target[0] - rest_joints(smplx, betas_x)[0]
    transl = smpl_params["trans"].astype(np.float64) + offset

    # Verify: FK both models over the whole sequence, compare shared joints.
    poses_smpl = np.concatenate([global_orient, smpl_params["poses_body"]], axis=1)
    joints_smpl = fk_joints(smpl, betas_smpl, poses_smpl, smpl_params["trans"].astype(np.float64))
    poses_x = np.concatenate([global_orient, body_pose], axis=1)
    joints_x = fk_joints(smplx, betas_x, poses_x, transl)
    err = np.linalg.norm(
        joints_smpl[:, :NUM_SHARED_JOINTS] - joints_x[:, :NUM_SHARED_JOINTS], axis=-1
    )
    print(f"FK joint error over {n} frames, {NUM_SHARED_JOINTS} joints: "
          f"mean {err.mean() * 1000:.1f} mm, max {err.max() * 1000:.1f} mm")

    out = args.sequence / "gt_smplx.npz"
    np.savez_compressed(
        out,
        betas=betas_x.astype(np.float32),
        global_orient=global_orient.astype(np.float32),
        body_pose=body_pose.astype(np.float32),
        transl=transl.astype(np.float32),
        left_hand_pose=np.zeros((n, 45), np.float32),
        right_hand_pose=np.zeros((n, 45), np.float32),
        jaw_pose=np.zeros((n, 3), np.float32),
        leye_pose=np.zeros((n, 3), np.float32),
        reye_pose=np.zeros((n, 3), np.float32),
        expression=np.zeros((n, 10), np.float32),
        good_frames_mask=data["good_frames_mask"],
        cam_intrinsics=data["camera"]["intrinsics"].astype(np.float32),
        cam_extrinsics=data["camera"]["extrinsics"].astype(np.float32),
        video_width=data["camera"]["width"],
        video_height=data["camera"]["height"],
        fps=30.0,
        gender_source=gender,
        model_type="smplx",
        num_betas=NUM_BETAS,
        frame="world",
        flat_hand_mean=True,
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
