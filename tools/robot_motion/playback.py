"""Generic viser playback for a BetterRobot ``Trajectory``.

Extracted from ``scripts/build_robot_motion.py``'s old ``_play`` method so
any script can open a web viewer for a robot + trajectory pair. Keeps the
two latency tricks from the original:

* one batched forward-kinematics pass pre-loads all joint transforms,
* each per-frame scene mutation is batched into a single websocket flush
  via ``server.atomic()``.

Optional per-frame joint-contact coloring: pass ``joint_contacts (T, J)``
(float probabilities or bool mask) and the helper locates the visualizer's
joint-sphere handles and repaints them on frames where a joint is in contact.
"""
from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from better_robot.data_model.model import Model
    from better_robot.tasks.trajectory import Trajectory


def play_trajectory(
    model: "Model",
    trajectory: "Trajectory",
    *,
    fps: float = 30.0,
    port: int = 8081,
    recenter: bool = True,
    joint_contacts: np.ndarray | torch.Tensor | None = None,
    contact_threshold: float = 0.5,
    contact_color: tuple[float, float, float] = (0.90, 0.18, 0.18),
    contact_forces: np.ndarray | torch.Tensor | None = None,
    contact_force_joints: "list[str] | tuple[str, ...] | None" = None,
    force_scale: float = 5e-4,
    joint_torques: np.ndarray | torch.Tensor | None = None,
    max_sphere_scale: float = 1.5,
    button_label: str = "Play trajectory",
) -> None:
    """Open a viser viewer with preloaded-FK playback. Blocks on ``viewer.show``.

    Parameters
    ----------
    model, trajectory
        The robot ``Model`` and ``Trajectory(B=1, T, nq)`` to animate.
    fps
        Target playback frame rate.
    port
        Viser HTTP port.
    recenter
        If True, re-anchors the trajectory via ``ground_centered_trajectory``
        so frame-0 pelvis sits at XY=0 and the lowest joint at Z=0.
    joint_contacts
        Optional ``(T, J)`` array (float or bool). When provided, each joint's
        sphere turns ``contact_color`` on frames where
        ``joint_contacts[t, j] > contact_threshold``, and reverts to the
        viewer's theme default when the joint leaves contact.
    contact_threshold
        Threshold used when ``joint_contacts`` is float-valued.
    contact_color
        RGB triple in ``[0, 1]`` for the "in contact" sphere color.
    contact_forces
        Optional ``(T, N, 3)`` world-frame linear forces at contact joints
        (Stage 2 output). Rendered as arrows at each joint origin, length
        scaled by ``force_scale`` metres per newton.
    contact_force_joints
        Names of the ``N`` joints to which ``contact_forces[:, n, :]``
        corresponds. Required iff ``contact_forces`` is provided.
    force_scale
        Metres-per-newton scale for the force arrows. Default ``5e-4``
        puts a 500 N force at ~0.25 m, which reads well on a human body.
    joint_torques
        Optional ``(T, nv)`` RNEA torques (Stage 2 output). When provided,
        each joint sphere is scaled per frame: the sphere for joint ``j``
        grows with the Euclidean norm of ``tau`` components owned by
        joint ``j`` (i.e. ``tau[:, idx_vs[j] : idx_vs[j]+nvs[j]]``).
    max_sphere_scale
        Upper bound for the torque-driven scale. The mapping is
        ``1 + clamp(|τ_j| / ref, 0, 1) · (max_sphere_scale − 1)`` where
        ``ref`` is the 95th percentile of all per-body magnitudes — so
        the default (1.0) sphere sits at the resting size and the largest
        spheres are ``max_sphere_scale`` times as big.
    button_label
        Label of the GUI button that starts playback.
    """
    from better_robot import forward_kinematics
    from better_robot.data_model.data import Data
    from better_robot.viewer import Visualizer
    from better_robot.viewer.overlays import ForceVectorsOverlay

    from .motion import ground_centered_trajectory

    viz_traj = ground_centered_trajectory(trajectory, model) if recenter else trajectory

    viewer = Visualizer(model, port=port)
    viewer.add_trajectory(viz_traj)
    scene = viewer._scene
    backend = viewer._backend
    if scene is None or backend is None:
        raise RuntimeError("Visualizer failed to initialise (missing scene/backend)")

    # --- Preload: one batched FK call over the whole trajectory ---------
    T = int(viz_traj.horizon)
    q_all = viz_traj.q[0]                                   # (T, nq)
    print(f"[playback] preloading FK for {T} frames...", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        full = forward_kinematics(model, q_all)
    oMi_all = full.oMi.detach().contiguous()                # (T, njoints, 7)
    model_id = id(model)
    print(
        f"[playback] preload done in {time.perf_counter() - t0:.2f}s "
        f"({T} frames, oMi {tuple(oMi_all.shape)})"
    )

    handle_by_j, default_rgb_255, contact_rgb_255, mask = _prepare_contact_coloring(
        scene=scene,
        backend=backend,
        joint_contacts=joint_contacts,
        contact_threshold=contact_threshold,
        contact_color=contact_color,
        expected_T=T,
    )

    # Torque-driven sphere scaling: precompute per-body magnitudes and
    # the (handle, base_radius) pairs we need to push scale into each frame.
    sphere_scale_by_j, per_body_scale = _prepare_torque_scaling(
        scene=scene,
        backend=backend,
        joint_torques=joint_torques,
        model=model,
        expected_T=T,
        max_scale=float(max_sphere_scale),
    )

    # Force-vector overlay: installed only when the caller has contact
    # forces to render (Stage 2 output).
    force_overlay: ForceVectorsOverlay | None = None
    force_joint_ids: torch.Tensor | None = None
    forces_t: torch.Tensor | None = None
    if contact_forces is not None:
        if contact_force_joints is None:
            raise ValueError(
                "contact_forces provided but contact_force_joints is None — "
                "pass the list of joint names matching the second axis of contact_forces"
            )
        forces_np = (
            contact_forces.detach().cpu().numpy()
            if isinstance(contact_forces, torch.Tensor)
            else np.asarray(contact_forces)
        )
        N = len(contact_force_joints)
        if forces_np.ndim != 3 or forces_np.shape != (T, N, 3):
            raise ValueError(
                f"contact_forces has shape {forces_np.shape}; expected ({T}, {N}, 3)"
            )
        force_joint_ids = torch.tensor(
            [model.body_name_to_id[n] for n in contact_force_joints],
            dtype=torch.long,
        )
        forces_t = torch.from_numpy(forces_np).to(dtype=torch.float32)
        force_overlay = ForceVectorsOverlay(
            contact_joint_names=contact_force_joints,
            scale=float(force_scale),
        )
        scene.add_mode(force_overlay)

    playing = threading.Event()

    def _run_preloaded() -> None:
        dt = 1.0 / float(fps) if fps > 0 else 0.0
        next_t = time.perf_counter()
        data = Data(_model_id=model_id, q=q_all[0])
        data.oMi = oMi_all[0]
        data._kinematics_level = 1
        # Start from "no joint in contact" so every True bit on frame 0 triggers
        # a paint; on exit we reset the same way via the finally branch below.
        prev_row = (
            np.zeros(mask.shape[1], dtype=bool) if mask is not None else None
        )
        try:
            for k in range(T):
                data.q = q_all[k]
                data.oMi = oMi_all[k]
                with backend._server.atomic():
                    scene.update(data)
                    if handle_by_j is not None and mask is not None and prev_row is not None:
                        row = mask[k]
                        if not np.array_equal(row, prev_row):
                            for j, handle in handle_by_j.items():
                                if j >= row.shape[0]:
                                    continue
                                if row[j] != prev_row[j]:
                                    handle.color = (
                                        contact_rgb_255 if row[j] else default_rgb_255
                                    )
                            prev_row = row.copy()
                    if force_overlay is not None and forces_t is not None and force_joint_ids is not None:
                        anchors_k = oMi_all[k, force_joint_ids, :3]
                        force_overlay.update_frame(anchors_k, forces_t[k])
                    if sphere_scale_by_j is not None and per_body_scale is not None:
                        row = per_body_scale[k]          # (njoints,) float32
                        for j, handle in sphere_scale_by_j.items():
                            if j >= row.shape[0]:
                                continue
                            s = float(row[j])
                            handle.scale = (s, s, s)
                if dt > 0:
                    next_t += dt
                    slack = next_t - time.perf_counter()
                    if slack > 0:
                        time.sleep(slack)
                    else:
                        # Fell behind: rebase pacing to avoid drift.
                        next_t = time.perf_counter()
        except Exception as e:  # pragma: no cover - playback errors
            print(f"[playback] playback error: {e}")
        finally:
            # Restore default sphere colors so the next playback starts clean.
            if handle_by_j is not None and default_rgb_255 is not None:
                for handle in handle_by_j.values():
                    try:
                        handle.color = default_rgb_255
                    except Exception:
                        pass
            # Restore default sphere scales.
            if sphere_scale_by_j is not None:
                for handle in sphere_scale_by_j.values():
                    try:
                        handle.scale = (1.0, 1.0, 1.0)
                    except Exception:
                        pass
            playing.clear()

    def _on_play(_evt: Any) -> None:
        if playing.is_set():
            return
        playing.set()
        threading.Thread(target=_run_preloaded, daemon=True).start()

    backend.add_gui_button(button_label, _on_play)

    torque_msg = ""
    if per_body_scale is not None:
        torque_msg = f" — torque-driven sphere scaling enabled (max×{max_sphere_scale:.1f})"
    contact_msg = (
        f" — contacts enabled on {int(mask.any(axis=1).sum())}/{T} frames"
        if mask is not None
        else ""
    )
    contact_msg = contact_msg + torque_msg
    print(
        f"[playback] viewer on http://localhost:{port} — "
        f"{T} frames @ {fps} fps preloaded{contact_msg}. "
        f"Click '{button_label}' to start. Ctrl-C to exit."
    )
    viewer.show(block=True)


def _prepare_torque_scaling(
    *,
    scene: Any,
    backend: Any,
    joint_torques: np.ndarray | torch.Tensor | None,
    model: "Model",
    expected_T: int,
    max_scale: float,
) -> tuple[dict[int, Any] | None, np.ndarray | None]:
    """Resolve joint-sphere handles and precompute per-body scale factors.

    Returns ``(handle_by_j, per_body_scale)`` where ``per_body_scale`` is
    a ``(T, njoints)`` array of scale multipliers in ``[1, max_scale]``.
    Returns ``(None, None)`` if ``joint_torques`` is None.
    """
    if joint_torques is None:
        return None, None

    skeleton = scene._modes.get("Skeleton")
    if skeleton is None or not getattr(skeleton, "_sphere_joints", None):
        raise RuntimeError(
            "torque scaling requested but the scene has no SkeletonMode "
            "with joint spheres — cannot locate sphere handles."
        )
    ns = skeleton._ctx.namespace

    handle_by_j: dict[int, Any] = {}
    for j in skeleton._sphere_joints:
        name = f"{ns}/sphere_{j}"
        handle = backend._nodes.get(name)
        if handle is None:
            raise RuntimeError(
                f"torque scaling: expected viser sphere handle at {name!r}, "
                f"not found in backend._nodes"
            )
        handle_by_j[j] = handle

    tau = (
        joint_torques.detach().cpu().numpy()
        if isinstance(joint_torques, torch.Tensor)
        else np.asarray(joint_torques)
    )
    if tau.ndim != 2 or tau.shape[0] != expected_T or tau.shape[1] != model.nv:
        raise ValueError(
            f"joint_torques has shape {tau.shape}; expected ({expected_T}, {model.nv})"
        )

    # Per-body magnitude: ‖τ[:, idx_vs[j] : idx_vs[j]+nvs[j]]‖₂ for each joint.
    njoints = model.njoints
    per_body_mag = np.zeros((expected_T, njoints), dtype=np.float32)
    idx_vs = [int(x) for x in model.idx_vs]
    nvs = [int(x) for x in model.nvs]
    for j in range(njoints):
        n = nvs[j]
        if n <= 0:
            continue
        iv = idx_vs[j]
        block = tau[:, iv : iv + n]                           # (T, n)
        per_body_mag[:, j] = np.linalg.norm(block, axis=-1)

    # Normalise by the 95th percentile so a couple of outlier frames don't
    # compress the dynamic range of the visualisation. Fall back to the
    # max when everything is zero.
    if per_body_mag.max() <= 0:
        per_body_scale = np.ones_like(per_body_mag)
    else:
        ref = float(np.percentile(per_body_mag, 95))
        ref = ref if ref > 1e-9 else float(per_body_mag.max())
        unit = np.clip(per_body_mag / ref, 0.0, 1.0)
        per_body_scale = 1.0 + unit * (max_scale - 1.0)
    return handle_by_j, per_body_scale


def _prepare_contact_coloring(
    *,
    scene: Any,
    backend: Any,
    joint_contacts: np.ndarray | torch.Tensor | None,
    contact_threshold: float,
    contact_color: tuple[float, float, float],
    expected_T: int,
) -> tuple[
    dict[int, Any] | None,
    tuple[int, int, int] | None,
    tuple[int, int, int] | None,
    np.ndarray | None,
]:
    """Resolve joint-sphere handles from the Visualizer and build a boolean
    ``(T, J)`` contact mask. Returns all ``None`` if ``joint_contacts`` is None.

    This touches ``scene._modes`` and ``backend._nodes`` — they're the same
    internals the preload-FK block already reads (``scene.update``,
    ``backend._server.atomic``). If ``better_robot`` refactors either away,
    we fail fast with a clear error rather than silently skipping contacts.
    """
    if joint_contacts is None:
        return None, None, None, None

    skeleton = scene._modes.get("Skeleton")
    if skeleton is None or not getattr(skeleton, "_sphere_joints", None):
        raise RuntimeError(
            "contact coloring requested but the scene has no SkeletonMode "
            "with joint spheres — cannot locate sphere handles."
        )
    ns = skeleton._ctx.namespace  # e.g. "/robot/Skeleton"

    handle_by_j: dict[int, Any] = {}
    for j in skeleton._sphere_joints:
        name = f"{ns}/sphere_{j}"
        handle = backend._nodes.get(name)
        if handle is None:
            raise RuntimeError(
                f"contact coloring: expected viser sphere handle at {name!r}, "
                f"not found in backend._nodes"
            )
        handle_by_j[j] = handle

    theme_rgba = scene._theme.joint_colour
    default_rgb_255 = tuple(int(c * 255) for c in theme_rgba[:3])
    contact_rgb_255 = tuple(int(c * 255) for c in contact_color[:3])

    jc = (
        joint_contacts.detach().cpu().numpy()
        if isinstance(joint_contacts, torch.Tensor)
        else np.asarray(joint_contacts)
    )
    if jc.ndim != 2 or jc.shape[0] != expected_T:
        raise ValueError(
            f"joint_contacts has shape {jc.shape}; expected (T, J) with T={expected_T}"
        )
    if jc.dtype == np.bool_:
        mask = jc.astype(bool)
    else:
        mask = jc.astype(np.float32) > float(contact_threshold)

    return handle_by_j, default_rgb_255, contact_rgb_255, mask
