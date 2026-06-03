"""Convert per-frame SMPL params into a BetterRobot motion file.

Run from CLI:

    uv run python scripts/stages/build_robot_motion.py \\
        --predictions out/video/ \\
        [--contacts out/video/interactvlm_hcontact_vertices.npz] \\
        [--manual-contacts out/video/manual_contacts.npz] \\
        [--contact-key pred_contact_3d_smplh] [--contact-threshold 0.5] \\
        [--fps 30] [--visualize] [--port 8081] \\
        [--smooth gaussian|average|none] [--kernel-size 5] [--sigma 0.8] \\
        [--smpl-model /home/rikhat/better/better_human/models/smpl/SMPL_NEUTRAL.npz] \\
        [--device cpu]

Or import:

    from scripts.stages.build_robot_motion import RobotMotionScript
    script = RobotMotionScript()
    script.run("out/video/", visualize=True)

Inputs:
    predictions_dir    folder containing smpl_params.npz
    --contacts         optional InteractVLM-style .npz with per-vertex contact
                       probabilities; aggregated to per-joint for the output.
    --manual-contacts  optional hand-labeled .npz with per-frame 0/1 contact
                       flags for left_hand / right_hand / left_foot /
                       right_foot; expanded to the per-joint matrix. Mutually
                       exclusive with --contacts.

Outputs (into <predictions_dir>/):
    results/0.npz      SMPL-like robot + q(t) trajectory plus a
                       joint_contact (N, 24) per-frame/per-joint contact
                       probability matrix (zeros when no contacts were given).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SMPL_MODEL = (
    _REPO_ROOT.parent / "better_human" / "models" / "smpl" / "SMPL_NEUTRAL.npz"
)


def _add_sys_paths() -> None:
    """Ensure better_human (editable src layout) and tools/ are importable."""
    better_human_src = _REPO_ROOT.parent / "better_human" / "src"
    if better_human_src.exists():
        sys.path.insert(0, str(better_human_src))
    sys.path.insert(0, str(_REPO_ROOT))
    # viewer lives at scripts/view_motion.py (top level), needed by --visualize
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))


_add_sys_paths()


@dataclass
class MotionResult:
    """Return value of ``RobotMotionScript.run``."""

    npz_path: Path
    num_frames: int
    betas_mean: np.ndarray
    fps: float


class RobotMotionScript:
    """Load `smpl_params.npz`, build a shape-aware SMPL-like BetterRobot,
    assemble a Trajectory, and persist a minimal motion file.

    Optionally opens a viser-backed viewer for playback.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        fps: float = 30.0,
        smpl_model_path: str | Path | None = None,
        smooth_method: str = "gaussian",
        smooth_kernel_size: int = 5,
        smooth_sigma: float | None = None,
        contacts_npz: str | Path | None = None,
        contact_key: str = "pred_contact_3d_smplh",
        contact_threshold: float = 0.5,
        manual_contacts_npz: str | Path | None = None,
    ) -> None:
        self.device = device
        self.fps = float(fps)
        self.smpl_model_path = Path(smpl_model_path) if smpl_model_path else _DEFAULT_SMPL_MODEL
        self.smooth_method = smooth_method
        self.smooth_kernel_size = int(smooth_kernel_size)
        self.smooth_sigma = smooth_sigma
        self.contacts_npz = Path(contacts_npz) if contacts_npz else None
        self.contact_key = contact_key
        self.contact_threshold = float(contact_threshold)
        self.manual_contacts_npz = Path(manual_contacts_npz) if manual_contacts_npz else None
        self._smpl = None

    def _load_smpl(self):
        if self._smpl is None:
            from better_human.smpl.model.smpl import SMPL

            self._smpl = SMPL(
                str(self.smpl_model_path),
                gender="neutral",
                num_betas=10,
            ).to(self.device)
            self._smpl.eval()
        return self._smpl

    def run(
        self,
        predictions_dir: str | Path,
        *,
        visualize: bool = False,
        port: int = 8081,
    ) -> MotionResult:
        from better_robot.io.builders import make_smpl_like_model
        from tools.robot_motion import (
            average_betas,
            build_trajectory,
            compute_smpl_body_inertia,
            compute_smpl_joint_offsets,
            load_joint_contacts,
            load_manual_joint_contacts,
            remap_smpl_contacts_to_model,
            remap_smpl_q_to_model,
            smooth_trajectory,
            y_up_to_z_up,
        )

        predictions_dir = Path(predictions_dir)
        src_npz = predictions_dir / "smpl_params.npz"
        if not src_npz.exists():
            raise FileNotFoundError(f"smpl_params.npz not found in {predictions_dir}")

        data = np.load(src_npz)
        q_cam = np.asarray(data["q"], dtype=np.float32)          # (N, 99), camera Y-up
        betas = np.asarray(data["betas"], dtype=np.float32)       # (N, 10)
        valid_mask = np.asarray(data["valid_mask"], dtype=bool)   # (N,)
        num_frames = int(data["num_frames"])
        if q_cam.shape != (num_frames, 99):
            raise ValueError(f"q has unexpected shape {q_cam.shape}, expected ({num_frames}, 99)")

        print(
            f"[robot_motion] loaded {src_npz} — {num_frames} frames, "
            f"{valid_mask.sum()} valid"
        )

        # Camera-estimated motion is Y-up; rotate the whole sequence to Z-up
        # (robotics convention). Only affects the SE3 root; joint quats are
        # parent-local and unchanged.
        q = y_up_to_z_up(q_cam)

        betas_mean = average_betas(betas, valid_mask)            # (10,)

        smpl = self._load_smpl()
        betas_t = torch.from_numpy(betas_mean).to(self.device)
        offsets_t = compute_smpl_joint_offsets(smpl, betas_t)
        offsets = offsets_t.cpu().numpy().astype(np.float32)

        # Neutral-pose SMPL geometry for per-body mass / inertia estimation.
        # ``forward_shape`` expects ``(B, 10)`` betas and returns
        # ``(vertices (B, 6890, 3), joints (B, 24, 3))``.
        with torch.no_grad():
            neutral_vertices, neutral_joints = smpl.forward_shape(betas_t.unsqueeze(0))
        masses, coms, inertias = compute_smpl_body_inertia(
            neutral_vertices[0].cpu(), neutral_joints[0].cpu(), density_system="Custom"
        )

        joint_contact: np.ndarray | None = None
        if self.contacts_npz is not None and self.manual_contacts_npz is not None:
            raise ValueError(
                "pass either --contacts or --manual-contacts, not both"
            )
        if self.manual_contacts_npz is not None:
            joint_contact = load_manual_joint_contacts(
                self.manual_contacts_npz,
                num_frames=num_frames,
                num_joints=24,
            )
            n_contact_frames = int((joint_contact > self.contact_threshold).any(axis=1).sum())
            print(
                f"[robot_motion] manual joint contacts from "
                f"{self.manual_contacts_npz.name} "
                f"(threshold {self.contact_threshold:.2f}) "
                f"— {n_contact_frames}/{num_frames} frames with ≥1 joint in contact"
            )
        elif self.contacts_npz is not None:
            joint_contact = load_joint_contacts(
                self.contacts_npz,
                smpl,
                num_frames=num_frames,
                contact_key=self.contact_key,
                reduce="max",
                num_joints=24,
            )
            n_contact_frames = int((joint_contact > self.contact_threshold).any(axis=1).sum())
            print(
                f"[robot_motion] joint contacts from {self.contacts_npz.name} "
                f"(key {self.contact_key!r}, threshold {self.contact_threshold:.2f}) "
                f"— {n_contact_frames}/{num_frames} frames with ≥1 joint in contact"
            )

        model = make_smpl_like_model(
            name="smpl_motion_body",
            joint_offsets=offsets_t,
            mass_per_body=masses.tolist(),
            com_per_body=coms,
            inertia_per_body=inertias,
            device=self.device,
        )
        total_mass = float(model.body_inertias[1:, 0].sum())
        print(
            f"[robot_motion] built robot: njoints={model.njoints}, "
            f"nq={model.nq}, nv={model.nv}, total_mass={total_mass:.2f} kg"
        )

        # SMPL → BetterRobot (DFS) conversion: happens exactly once here.
        # All downstream consumers (view_motion, optimize_motion) read the
        # saved .npz as BR DFS directly. See docs/motion_npz_schema.md.
        q = remap_smpl_q_to_model(q, model)
        if joint_contact is not None:
            joint_contact = remap_smpl_contacts_to_model(joint_contact, model)

        traj = build_trajectory(q, valid_mask, self.fps)
        print(
            f"[robot_motion] trajectory: B={traj.batch_size}, T={traj.horizon}, "
            f"nq={traj.q.shape[-1]} @ {self.fps} fps"
        )

        if self.smooth_method != "none":
            traj = smooth_trajectory(
                traj,
                method=self.smooth_method,
                kernel_size=self.smooth_kernel_size,
                sigma=self.smooth_sigma,
            )
            print(
                f"[robot_motion] smoothed with {self.smooth_method}"
                f"(kernel_size={self.smooth_kernel_size}"
                + (f", sigma={self.smooth_sigma}" if self.smooth_sigma is not None else "")
                + ")"
            )
        q_out = traj.q[0].cpu().numpy().astype(np.float32)

        # Extract DFS-ordered metadata from the frozen model so the .npz is
        # self-describing. Indices in ``parents_dfs`` are 0-based into the
        # 24-entry ``joint_names_dfs`` array (-1 for the pelvis root).
        joint_names_dfs = tuple(model.body_names[1:])
        parents_dfs = tuple(
            int(model.parents[i]) - 1 for i in range(1, model.njoints)
        )
        offsets_dfs = model.joint_placements[1:, :3].cpu().numpy().astype(np.float32)

        results_dir = predictions_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        out_npz = results_dir / "0.npz"
        # Remap SMPL-canonical COM/inertia arrays into BR DFS so they line
        # up with ``joint_names`` / ``parents`` / ``joint_offsets``.
        from better_robot.io.builders import JOINT_NAMES
        smpl_idx_of = {name: k for k, name in enumerate(JOINT_NAMES)}
        dfs_from_smpl = [smpl_idx_of[n] for n in joint_names_dfs]
        coms_dfs = coms.cpu().numpy().astype(np.float32)[dfs_from_smpl]
        inertias_dfs = inertias.cpu().numpy().astype(np.float32)[dfs_from_smpl]

        np.savez(
            out_npz,
            q=q_out,
            t=traj.t[0].cpu().numpy().astype(np.float32),
            fps=np.float32(self.fps),
            betas_mean=betas_mean,
            joint_names=np.array(joint_names_dfs, dtype="<U20"),
            parents=np.asarray(parents_dfs, dtype=np.int32),
            joint_offsets=offsets_dfs,
            mass_per_body=model.body_inertias[1:, 0].cpu().numpy().astype(np.float32),
            com_per_body=coms_dfs,
            inertia_per_body=inertias_dfs,
            valid_mask=valid_mask,
            num_frames=np.int32(num_frames),
            joint_contact=(
                joint_contact
                if joint_contact is not None
                else np.zeros((num_frames, 24), dtype=np.float32)
            ),
            joint_contact_threshold=np.float32(self.contact_threshold),
        )
        print(f"[robot_motion] wrote {out_npz}")

        if visualize:
            # Delegate to the shared viewer so there is a single canonical
            # entry point used by both build and post-optim visualisation.
            from view_motion import view_motion

            view_motion(
                out_npz, port=port, device=self.device,
                fps=self.fps, contact_threshold=self.contact_threshold,
            )

        return MotionResult(
            npz_path=out_npz,
            num_frames=num_frames,
            betas_mean=betas_mean,
            fps=self.fps,
        )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", required=True,
                   help="folder containing smpl_params.npz")
    p.add_argument("--contacts", default=None,
                   help="optional .npz with per-vertex contact probabilities "
                        "(e.g. interactvlm_hcontact_vertices.npz); aggregated to "
                        "per-joint and saved into results/0.npz")
    p.add_argument("--manual-contacts", default=None,
                   help="optional .npz with per-frame manual labels "
                        "(keys left_hand, right_hand, left_foot, right_foot); "
                        "mutually exclusive with --contacts")
    p.add_argument("--contact-key", default="pred_contact_3d_smplh",
                   help="key in --contacts to use; must have shape (M, 6890) "
                        "(default pred_contact_3d_smplh)")
    p.add_argument("--contact-threshold", type=float, default=0.5,
                   help="joint is 'in contact' when aggregated prob > threshold "
                        "(default 0.5); used by the visualizer and saved alongside "
                        "joint_contact")
    p.add_argument("--fps", type=float, default=30.0,
                   help="frame rate for the trajectory (default 30)")
    p.add_argument("--device", default="cpu",
                   help="torch device for SMPL forward (default cpu)")
    p.add_argument("--smpl-model", default=None,
                   help=f"SMPL neutral .npz (default: {_DEFAULT_SMPL_MODEL})")
    p.add_argument("--visualize", action="store_true",
                   help="launch viser playback after saving the motion")
    p.add_argument("--port", type=int, default=8081,
                   help="viser port when --visualize (default 8081)")
    p.add_argument("--smooth", dest="smooth_method",
                   choices=["gaussian", "average", "none"], default="gaussian",
                   help="temporal smoother for q(t) (default gaussian)")
    p.add_argument("--kernel-size", type=int, default=5,
                   help="smoothing kernel length, odd (default 5)")
    p.add_argument("--sigma", type=float, default=None,
                   help="gaussian sigma (default kernel_size / 6)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    script = RobotMotionScript(
        device=args.device,
        fps=args.fps,
        smpl_model_path=args.smpl_model,
        smooth_method=args.smooth_method,
        smooth_kernel_size=args.kernel_size,
        smooth_sigma=args.sigma,
        contacts_npz=args.contacts,
        contact_key=args.contact_key,
        contact_threshold=args.contact_threshold,
        manual_contacts_npz=args.manual_contacts,
    )
    script.run(
        args.predictions,
        visualize=args.visualize,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
