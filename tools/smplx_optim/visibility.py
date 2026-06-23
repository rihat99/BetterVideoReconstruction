"""Mesh visibility for the chamfer term.

The depth points only see the front of the body, so chamfer must compare them
against **front-facing** mesh vertices. We approximate visibility with a
back-face cull in the camera frame (a vertex is visible when its outward normal
points back toward the camera). This is a cheap proxy for true z-buffer
occlusion — fine for the largely-convex human surface. The cull mask is used
detached (it gates which vertices participate; gradients still flow through the
vertex positions that survive).
"""

from __future__ import annotations

import torch


def vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Per-vertex normals ``(B, V, 3)`` (area-weighted face normals, normalized).

    ``vertices`` ``(B, V, 3)``; ``faces`` ``(F, 3)`` long.
    """
    B, V, _ = vertices.shape
    v0 = vertices[:, faces[:, 0]]
    v1 = vertices[:, faces[:, 1]]
    v2 = vertices[:, faces[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, F, 3), magnitude ∝ area

    normals = vertices.new_zeros((B, V, 3))
    idx = faces.reshape(-1)  # (3F,)
    contrib = torch.cat([fn, fn, fn], dim=1)  # (B, 3F, 3)
    normals.index_add_(1, idx, contrib)
    return normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def front_facing_mask(
    vertices_cam: torch.Tensor,
    normals_cam: torch.Tensor,
) -> torch.Tensor:
    """Boolean ``(B, V)`` mask: vertices whose normal faces the camera.

    Both inputs are in the camera frame (origin at the camera). A vertex is
    front-facing when the view ray (camera→vertex) opposes its normal,
    i.e. ``dot(normal, position) < 0``.
    """
    return (normals_cam * vertices_cam).sum(dim=-1) < 0.0


def subsample_indices(num_vertices: int, n_target: int, device: str | torch.device = "cpu") -> torch.Tensor:
    """Deterministic uniform-stride vertex subsample of size ``≤ n_target``."""
    if n_target >= num_vertices:
        return torch.arange(num_vertices, device=device)
    stride = num_vertices / float(n_target)
    idx = (torch.arange(n_target, device=device) * stride).long().clamp_max(num_vertices - 1)
    return torch.unique(idx)
