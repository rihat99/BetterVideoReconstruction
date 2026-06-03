"""View any motion npz (``results/*.npz``) in a viser web viewer.

Extracted from ``build_robot_motion.py``'s ``--visualize`` branch so any
downstream tool (kinematic optimisation, future dynamics passes, manual
motion edits) can share a single playback entry point.

Usage::

    uv run python scripts/view_motion.py <path-to-npz> [--port 8081]

The ``.npz`` is expected to hold motion data in BetterRobot's DFS indexing
— produced by ``build_robot_motion.py`` or any tool that writes through its
schema. Required keys: ``q (T, nq)``,
``t (T,)``, ``fps``, ``joint_offsets (24, 3)``, ``joint_names (24,)``,
``parents (24,)``. Optional: ``joint_contact (T, 24)``,
``joint_contact_threshold``.

Set ``--q-key q_optimized`` (for example) to play a later-stage solution
when both the raw and optimised clips live in the same file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


_REPO_ROOT = Path(__file__).resolve().parents[1]
_BETTER_HUMAN_SRC = _REPO_ROOT.parent / "better_human" / "src"
if _BETTER_HUMAN_SRC.exists() and str(_BETTER_HUMAN_SRC) not in sys.path:
    sys.path.insert(0, str(_BETTER_HUMAN_SRC))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def view_motion(
    npz_path: Path,
    *,
    q_key: str = "q",
    port: int = 8081,
    recenter: bool = True,
    device: str = "cpu",
    fps: float | None = None,
    contact_threshold: float | None = None,
    max_sphere_scale: float = 1.5,
) -> None:
    """Rebuild the model from ``npz_path`` and launch viser playback."""
    from better_robot.io.builders import JOINT_NAMES, build_kinematic_tree_model
    from better_robot.tasks.trajectory import Trajectory
    from tools.robot_motion import play_trajectory

    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    with np.load(npz_path) as data:
        if q_key not in data.files:
            raise KeyError(
                f"key {q_key!r} not found in {npz_path}; available: {tuple(data.files)}"
            )
        q_np = np.asarray(data[q_key], dtype=np.float32)
        t_np = np.asarray(data["t"], dtype=np.float32)
        fps_saved = float(data["fps"])
        offsets_np = np.asarray(data["joint_offsets"], dtype=np.float32)
        saved_names = tuple(str(x) for x in data["joint_names"])
        saved_parents = tuple(int(p) for p in np.asarray(data["parents"]))
        joint_contact = (
            np.asarray(data["joint_contact"])
            if "joint_contact" in data.files
            else None
        )
        saved_threshold = (
            float(data["joint_contact_threshold"])
            if "joint_contact_threshold" in data.files
            else 0.5
        )
        contact_forces = (
            np.asarray(data["contact_forces"], dtype=np.float32)
            if "contact_forces" in data.files
            else None
        )
        contact_force_joints = (
            tuple(str(x) for x in data["contact_force_joints"])
            if "contact_force_joints" in data.files
            else None
        )
        joint_torques = (
            np.asarray(data["joint_torques"], dtype=np.float32)
            if "joint_torques" in data.files
            else None
        )

    if set(saved_names) != set(JOINT_NAMES):
        raise ValueError(
            f"{npz_path.name}: saved joint_names do not form the SMPL-24 body set "
            f"(expected {sorted(set(JOINT_NAMES))}, got {sorted(set(saved_names))})"
        )

    fps_used = float(fps) if fps is not None else fps_saved
    threshold_used = (
        float(contact_threshold) if contact_threshold is not None else saved_threshold
    )

    # The .npz stores joint_names / parents / offsets in BR DFS order (see
    # docs/motion_npz_schema.md). Hand them straight to the builder — no
    # permutation, no use of the SMPL-canonical JOINT_NAMES / PARENTS tuples.
    offsets_t = torch.from_numpy(offsets_np)
    model = build_kinematic_tree_model(
        name=npz_path.parent.name or "motion_body",
        joint_names=saved_names,
        parents=saved_parents,
        translations=offsets_t,
        root_kind="free_flyer",
        child_kind="spherical",
        mass_per_body=1.0,
        device=device,
    )
    print(
        f"[view_motion] loaded {npz_path}  (q='{q_key}', T={q_np.shape[0]}, "
        f"nq={q_np.shape[1]}, fps={fps_used:.2f})"
    )

    # Single-clip Trajectory with leading batch = 1, matching build_robot_motion output.
    q_tensor = torch.from_numpy(q_np).unsqueeze(0)            # (1, T, nq)
    t_tensor = torch.from_numpy(t_np).unsqueeze(0)            # (1, T)
    trajectory = Trajectory(
        t=t_tensor, q=q_tensor,
        model_id=getattr(model, "id", -1),
    )

    # joint_contact is (T, 24) indexed as BR body 1..24 in DFS order. The
    # viewer wants (T, njoints=25) indexed as BR joint 0..24, so prepend a
    # zero column for the universe.
    viz_contacts = None
    if joint_contact is not None:
        viz_contacts = np.concatenate(
            [np.zeros((joint_contact.shape[0], 1), dtype=joint_contact.dtype),
             joint_contact],
            axis=1,
        )

    if contact_forces is not None:
        print(
            f"[view_motion] contact forces: {contact_forces.shape[1]} joints "
            f"({', '.join(contact_force_joints or ())})"
        )

    play_trajectory(
        model,
        trajectory,
        fps=fps_used,
        port=port,
        recenter=recenter,
        joint_contacts=viz_contacts,
        contact_threshold=threshold_used,
        contact_forces=contact_forces,
        contact_force_joints=contact_force_joints,
        joint_torques=joint_torques,
        max_sphere_scale=max_sphere_scale,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("npz_path", type=Path,
                   help="path to a motion npz (e.g. <clip>/results/0.npz)")
    p.add_argument("--q-key", default="q",
                   help="key to read as the trajectory (default 'q')")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--fps", type=float, default=None,
                   help="override playback fps (default: saved value)")
    p.add_argument("--contact-threshold", type=float, default=None,
                   help="override contact threshold for sphere colouring")
    p.add_argument("--no-recenter", action="store_true",
                   help="skip ground-centering of the trajectory before display")
    p.add_argument("--max-sphere-scale", type=float, default=1.5,
                   help="upper bound on torque-driven joint-sphere scale "
                        "(1.0 disables the effect; default 1.5)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    view_motion(
        args.npz_path,
        q_key=args.q_key,
        port=args.port,
        recenter=not args.no_recenter,
        device=args.device,
        fps=args.fps,
        contact_threshold=args.contact_threshold,
        max_sphere_scale=args.max_sphere_scale,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
