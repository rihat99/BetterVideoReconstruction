"""Friction-cone geometry for contact forces.

Two pieces, both precomputed off a frozen trajectory: :func:`cone_normals` reads a
surface normal per (frame, contact joint) from the fused scene cloud, and
:func:`cone_generators` turns each normal into the four unit edge directions of the
friction pyramid at half-angle ``atan mu``. A force built as a nonnegative
combination of the four edges cannot leave the pyramid inscribed in the true cone
— the dynamics solve uses exactly that to make foot/ankle forces physical by
construction (see :mod:`tools.human_optim.dynamics`).
"""

from __future__ import annotations

import numpy as np
import torch


def cone_normals(joint_xyz: torch.Tensor, scene_pts: torch.Tensor,
                  scene_normals: torch.Tensor, k: int, max_dist: float
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Surface normal per ``(frame, cone joint)`` from the scene cloud, detached.

    ``joint_xyz`` is ``(T, Cc, 3)`` world joint positions. For each one, average the
    stored normals of its ``k`` nearest scene points and normalize — the fuse normals
    point out of the wall toward the climber, i.e. the direction a pushing contact
    force on the body should have. Returns ``(normals (T, Cc, 3), valid (T, Cc))``;
    ``valid`` is ``False`` where the nearest scene point is beyond ``max_dist`` (joint
    too far from any surface for a meaningful cone) or where the neighbours' normals
    disagree (mean magnitude < 0.5 — e.g. a thin hold or concave corner averages
    opposing normals to ~zero, and normalizing that would aim the cone arbitrarily).
    ``q`` is frozen, so this is a one-time precompute done on CPU with a KDTree.
    """
    from scipy.spatial import cKDTree

    T, Cc, _ = joint_xyz.shape
    dev = joint_xyz.device
    pts = scene_pts.detach().cpu().numpy()
    nrm = scene_normals.detach().cpu().numpy()
    query = joint_xyz.detach().cpu().numpy().reshape(-1, 3)          # (T*Cc, 3)
    kk = min(int(k), pts.shape[0])
    dist, idx = cKDTree(pts).query(query, k=kk)
    idx = idx.reshape(query.shape[0], kk)
    dist = dist.reshape(query.shape[0], kk)
    n = nrm[idx].mean(axis=1)                                        # (T*Cc, 3)
    coherence = np.linalg.norm(n, axis=-1)                           # 1 = neighbours agree, 0 = cancel
    n /= np.maximum(coherence[:, None], 1e-9)
    valid = (dist[:, 0] <= float(max_dist)) & (coherence >= 0.5)
    normals = torch.as_tensor(n.reshape(T, Cc, 3), dtype=torch.float32, device=dev)
    valid_t = torch.as_tensor(valid.reshape(T, Cc), dtype=torch.bool, device=dev)
    return normals, valid_t


def cone_generators(normals: torch.Tensor, mu: float) -> torch.Tensor:
    """Unit edge directions ``(..., 4, 3)`` of the friction pyramid around ``normals``.

    Per normal ``n̂`` a tangent basis ``(t1, t2)`` is built and each edge is
    ``normalize(n̂ ± μ·t)`` — a direction on the cone boundary at half-angle
    ``atan μ``. Any nonnegative combination of the four edges lies in the pyramid
    they span, which is inscribed in the true cone (exact along the edge planes,
    slightly conservative between them).
    """
    n = normals
    seed = torch.zeros_like(n)
    seed[..., 0] = 1.0
    seed[n[..., 0].abs() > 0.9] = torch.tensor([0.0, 1.0, 0.0], device=n.device)
    t1 = torch.cross(n, seed, dim=-1)
    t1 = t1 / t1.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    t2 = torch.cross(n, t1, dim=-1)
    edges = torch.stack([n + mu * t1, n - mu * t1, n + mu * t2, n - mu * t2], dim=-2)
    return edges / edges.norm(dim=-1, keepdim=True).clamp_min(1e-9)
