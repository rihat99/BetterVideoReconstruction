"""``sole_offsets``: a contact frame is pushed out along its outward normal (``tools.body``)."""

from __future__ import annotations

import pytest
import torch

from tools.body import frame_normals, load_body

_FEET = ["heel_left", "heel_right", "big_toe_left", "big_toe_right"]


@pytest.fixture(scope="module")
def bodies():
    bare = load_body(device="cpu", frames_allow=_FEET)
    shod = load_body(device="cpu", frames_allow=_FEET, sole_offsets={"heel": 0.03})
    return bare, shod


def _frames(body):
    betas = torch.zeros(1, 10)
    q = torch.zeros(1, body.nq)
    q[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])                        # identity root (xyzw)
    q[:, 7:] = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(body.num_joints - 1)  # identity joints
    return body.forward(betas, q).frames_world


def test_heel_moves_along_its_normal_and_toes_stay(bodies):
    bare, shod = bodies
    f_bare, f_shod = _frames(bare), _frames(shod)
    normals = frame_normals(f_bare)
    for i, name in enumerate(bare.contact_frame_names):
        shift = f_shod[0, i, :3] - f_bare[0, i, :3]
        expected = 0.03 * normals[0, i] if name.startswith("heel") else torch.zeros(3)
        torch.testing.assert_close(shift, expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(f_shod[..., 3:], f_bare[..., 3:])      # rotations untouched


def test_offsets_are_exposed_per_frame(bodies):
    _, shod = bodies
    torch.testing.assert_close(shod.contact_frame_offsets, torch.tensor([0.03, 0.03, 0.0, 0.0]))


def test_unknown_stem_is_refused():
    with pytest.raises(ValueError, match="palm"):
        load_body(device="cpu", frames_allow=_FEET, sole_offsets={"palm": 0.01})
