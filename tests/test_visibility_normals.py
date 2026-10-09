"""Vertex-normal accumulation must land each face normal on that face's vertices.

A regression guard for the indexing bug where the per-face contributions were
stacked in block order (`cat([fn, fn, fn])`) while the scatter indices were in
per-face-vertex order (`faces.reshape(-1)`), scrambling normals by ~88°.
"""
from __future__ import annotations

import torch

from bvr.human_optim.visibility import front_facing_mask, vertex_normals


def _reference_vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Obvious per-face accumulation loop — independent of the vectorized code."""
    normals = torch.zeros_like(vertices)
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        fn = torch.cross(v1 - v0, v2 - v0, dim=-1)  # magnitude ∝ area
        for vi in face:
            normals[vi] += fn
    return normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def test_vertex_normals_match_reference_on_tetrahedron():
    # Every vertex touches 3 faces with distinct normals, so a scrambled
    # accumulation cannot accidentally agree with the correct one.
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=torch.long)

    got = vertex_normals(vertices[None], faces)[0]
    expected = _reference_vertex_normals(vertices, faces)

    assert torch.allclose(got, expected, atol=1e-5)


def test_vertex_normals_of_a_flat_ccw_quad_point_up():
    # Two CCW triangles in the z=0 plane → every vertex normal is +z.
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)

    normals = vertex_normals(vertices[None], faces)[0]

    assert torch.allclose(normals, torch.tensor([0.0, 0.0, 1.0]).expand(4, 3), atol=1e-5)


def test_front_facing_mask_sees_a_plane_facing_the_camera():
    # A vertex at +z with a normal pointing back toward the camera (−z) is visible.
    vertices_cam = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
    normals_cam = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])

    mask = front_facing_mask(vertices_cam[None], normals_cam[None])[0]

    assert mask.tolist() == [True, False]
