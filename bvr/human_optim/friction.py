"""Friction-cone geometry for contact forces.

:func:`cone_generators` turns a contact normal into the four unit edge directions
of the friction pyramid at half-angle ``atan mu``. A force built as a nonnegative
combination of the four edges cannot leave the pyramid inscribed in the true cone
— the force witness and kindyn use exactly that to make contact forces physical
by construction.
"""

from __future__ import annotations

import torch


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
