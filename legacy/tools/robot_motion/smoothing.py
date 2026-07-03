"""Temporal smoothing of motion trajectories.

q layout is ``(B, T, 7 + 4*n_joints)``:

- ``q[..., 0:7]`` — SE(3) floating-base root ``[tx, ty, tz, qx, qy, qz, qw]``.
- ``q[..., 7:]``  — ``n_joints`` parent-local SO(3) quaternions, scalar-last.

Smoothing a kernel over a sequence of rotations/transforms is a weighted
geodesic mean on the manifold, not a component-wise average. This module
implements it via **iterative SLERP / ScLERP**: walking pairwise through the
window and reusing BetterRobot's ``so3.slerp`` / ``se3.sclerp`` primitives so
we never leave the manifold.

Public API:

- :func:`make_average_kernel`, :func:`make_gaussian_kernel` — build weights.
- :func:`kernel_smooth` — apply an arbitrary 1-D kernel to a ``Trajectory``.
- :func:`smooth_trajectory` — high-level dispatch: ``method="gaussian" |
  "average" | "none"``, or pass a custom ``kernel`` / ``smoother`` (protocol
  :class:`TrajectorySmoother`) to plug in e.g. a Kalman smoother.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import torch
import torch.nn.functional as F

from better_robot.lie import se3, so3

if TYPE_CHECKING:
    from better_robot.tasks.trajectory import Trajectory


PadMode = Literal["replicate", "reflect"]


@runtime_checkable
class TrajectorySmoother(Protocol):
    """Strategy interface for pluggable smoothers (e.g. Kalman, Savitzky-Golay)."""

    def __call__(self, traj: "Trajectory") -> "Trajectory": ...


# --------------------------------------------------------------------------
# Kernel constructors
# --------------------------------------------------------------------------

def make_average_kernel(
    size: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Uniform box kernel of length ``size``. Entries sum to 1."""
    _check_size(size)
    return torch.full((size,), 1.0 / size, device=device, dtype=dtype)


def make_gaussian_kernel(
    size: int,
    sigma: float | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """1-D Gaussian kernel of length ``size``, centered, normalized to sum 1.

    ``sigma`` defaults to ``size / 6`` so the kernel spans ±3σ.
    """
    _check_size(size)
    if sigma is None:
        sigma = size / 6.0
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    center = (size - 1) / 2.0
    i = torch.arange(size, device=device, dtype=dtype)
    w = torch.exp(-0.5 * ((i - center) / sigma) ** 2)
    return w / w.sum()


def _check_size(size: int) -> None:
    if size < 1 or size % 2 == 0:
        raise ValueError(f"kernel size must be a positive odd integer, got {size}")


# --------------------------------------------------------------------------
# Window gather with edge padding
# --------------------------------------------------------------------------

def _gather_window(x: torch.Tensor, size: int, mode: PadMode) -> torch.Tensor:
    """Return shape ``(K, T, C)`` window view of ``x (T, C)`` centered at each t.

    Row ``k`` contains ``x`` at time offset ``k - K//2`` from each center,
    with out-of-range indices filled via edge ``mode``.
    """
    T, C = x.shape
    half = size // 2
    x_pad = F.pad(x.transpose(0, 1).unsqueeze(0), (half, half), mode=mode)
    x_pad = x_pad.squeeze(0).transpose(0, 1)  # (T + 2*half, C)
    return torch.stack([x_pad[k : k + T] for k in range(size)], dim=0)


# --------------------------------------------------------------------------
# Weighted geodesic means via iterative (S)LERP
# --------------------------------------------------------------------------

def _iterative_mean(
    samples: torch.Tensor,
    weights: torch.Tensor,
    pairwise_interp,
) -> torch.Tensor:
    """Iterative weighted geodesic mean using a pairwise interpolator.

    ``samples``: ``(K, ...)``. ``weights``: ``(K,)`` non-negative.
    ``pairwise_interp(a, b, alpha) -> c`` computes the α-geodesic from
    ``a`` toward ``b`` (e.g. ``so3.slerp`` or ``se3.sclerp``).

    Walks the window left-to-right, maintaining a running accumulator with
    mass ``w_acc``. At step ``k``, blends the new sample with α = w_k /
    (w_acc + w_k), so the final weight on sample ``k`` is exactly ``w_k``
    on a flat manifold and a close approximation on SO(3)/SE(3) for the
    smooth-motion regime where kernel smoothing is meaningful.
    """
    K = int(samples.shape[0])
    acc = samples[0]
    w_acc = weights[0].clone()
    for k in range(1, K):
        w_k = weights[k]
        denom = w_acc + w_k
        # Skip zero-weight steps to avoid 0/0; `acc` is unchanged.
        if bool((denom <= 0).item()):
            continue
        alpha = w_k / denom
        acc = pairwise_interp(acc, samples[k], alpha)
        w_acc = denom
    return acc


def _weighted_so3_mean(quats: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted mean of ``(K, ..., 4)`` quaternions via iterative SLERP."""
    return _iterative_mean(quats, weights, so3.slerp)


def _weighted_se3_mean(poses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted mean of ``(K, ..., 7)`` SE(3) poses via iterative ScLERP."""
    return _iterative_mean(poses, weights, se3.sclerp)


# --------------------------------------------------------------------------
# Kernel smoother
# --------------------------------------------------------------------------

def kernel_smooth(
    traj: "Trajectory",
    kernel: torch.Tensor,
    *,
    mode: PadMode = "replicate",
) -> "Trajectory":
    """Smooth a ``Trajectory`` with an arbitrary 1-D kernel.

    - SE(3) root (``q[..., 0:7]``) is smoothed with iterative ScLERP.
    - Each SO(3) joint (``q[..., 7+4j : 7+4(j+1)]``) is smoothed with
      iterative SLERP, all joints vectorized.

    The kernel is re-normalized to sum 1. Edge frames use ``mode``
    (``"replicate"`` repeats the boundary value; ``"reflect"`` mirrors).
    """
    from better_robot.tasks.trajectory import Trajectory

    q = traj.q
    if q.ndim != 3:
        raise ValueError(f"expected q of shape (B, T, nq), got {tuple(q.shape)}")
    B, T, nq = q.shape
    if nq < 7 or (nq - 7) % 4 != 0:
        raise ValueError(
            f"expected nq = 7 + 4*n_joints (SE(3) root + SO(3) joints), got nq={nq}"
        )

    size = int(kernel.shape[0])
    _check_size(size)
    if size == 1:
        return Trajectory(t=traj.t, q=q.clone(), v=traj.v, a=traj.a, tau=traj.tau,
                          model_id=traj.model_id)
    if T < size:
        raise ValueError(f"trajectory too short: T={T} < kernel size {size}")

    w = kernel.to(dtype=q.dtype, device=q.device)
    w = w / w.sum()

    n_joints = (nq - 7) // 4
    out = q.clone()
    for b in range(B):
        qb = q[b]  # (T, nq)

        # SE(3) root: (T, 7) → (K, T, 7) → (T, 7) via ScLERP.
        root_win = _gather_window(qb[:, :7], size, mode=mode)
        out[b, :, :7] = _weighted_se3_mean(root_win, w)

        # SO(3) joints: vectorize across the joint dimension.
        # (T, 4*n_joints) → gather → (K, T, 4*n_joints) → reshape to
        # (K, T, n_joints, 4), run iterative SLERP, reshape back.
        joint_flat = qb[:, 7:]
        joint_win = _gather_window(joint_flat, size, mode=mode)
        joint_win = joint_win.reshape(size, T, n_joints, 4)
        joint_smooth = _weighted_so3_mean(joint_win, w)  # (T, n_joints, 4)
        out[b, :, 7:] = joint_smooth.reshape(T, n_joints * 4)

    return Trajectory(
        t=traj.t, q=out, v=traj.v, a=traj.a, tau=traj.tau, model_id=traj.model_id,
    )


# --------------------------------------------------------------------------
# High-level dispatch
# --------------------------------------------------------------------------

def smooth_trajectory(
    traj: "Trajectory",
    *,
    method: Literal["gaussian", "average", "none"] = "gaussian",
    kernel_size: int = 5,
    sigma: float | None = None,
    kernel: torch.Tensor | None = None,
    smoother: TrajectorySmoother | None = None,
    mode: PadMode = "replicate",
) -> "Trajectory":
    """Smooth ``traj`` and return a new ``Trajectory``.

    Selection order (first match wins):

    1. ``smoother`` given — delegate to the custom callable. Use this hook to
       plug in Kalman smoothers, Savitzky-Golay, learned models, etc.
    2. ``kernel`` given — kernel-based smoothing with that 1-D weight tensor.
    3. ``method == "none"`` — identity (returns a clone).
    4. ``method == "average"`` — box kernel of ``kernel_size``.
    5. ``method == "gaussian"`` — Gaussian kernel (``sigma`` defaults to
       ``kernel_size / 6``).
    """
    if smoother is not None:
        return smoother(traj)
    if kernel is not None:
        return kernel_smooth(traj, kernel, mode=mode)
    if method == "none":
        from better_robot.tasks.trajectory import Trajectory
        return Trajectory(
            t=traj.t, q=traj.q.clone(), v=traj.v, a=traj.a, tau=traj.tau,
            model_id=traj.model_id,
        )
    device = traj.q.device
    dtype = traj.q.dtype
    if method == "average":
        k = make_average_kernel(kernel_size, device=device, dtype=dtype)
    elif method == "gaussian":
        k = make_gaussian_kernel(kernel_size, sigma, device=device, dtype=dtype)
    else:
        raise ValueError(
            f"unknown method {method!r}; expected 'gaussian'|'average'|'none', "
            f"or pass `kernel=` / `smoother=`."
        )
    return kernel_smooth(traj, k, mode=mode)


__all__ = [
    "TrajectorySmoother",
    "make_average_kernel",
    "make_gaussian_kernel",
    "kernel_smooth",
    "smooth_trajectory",
]
