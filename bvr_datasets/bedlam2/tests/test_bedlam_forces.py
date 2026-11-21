"""The frozen-pose force solve behind ``bvr_datasets/bedlam2/label_bedlam2.py --step forces``.

A body standing still on its labelled feet must come out carrying its own weight there and nowhere
else, and a body without a label must carry nothing, its weight left unexplained at the root.
"""

from __future__ import annotations

import numpy as np

from bvr import body, config
from bvr_datasets.bedlam2.forces import scene_forces

_SOLES = ("heel_left", "heel_right", "ball_inner_left", "ball_outer_left",
          "ball_inner_right", "ball_outer_right", "big_toe_left", "big_toe_right")


def _solve(directory, human, soles: bool, frames: int = 8) -> dict:
    """Write one standing person's scene files to ``directory`` and solve its forces."""
    q = np.zeros((1, frames, human.model.nq), np.float32)
    q[..., 6::4] = 1.0                                             # unit quaternions: the rest pose, y up
    names = list(human.contact_frames.names)
    contact = np.zeros((1, frames, len(names)), bool)
    if soles:
        contact[..., [names.index(name) for name in _SOLES]] = True
    np.savez(directory / "smplx.npz", q=q, betas=np.zeros((1, 10), np.float32), object_ids=np.zeros(1, np.int32),
             fps=np.float32(30.0), joint_names=np.array(human.spec.joint_names), parents=np.array(human.spec.parents))
    np.savez(directory / "gravity.npz", gravity_world=np.array([0.0, -1.0, 0.0], np.float32))
    np.savez(directory / "contacts.npz", frame_contact=contact, contact_frame_names=np.array(names),
             object_ids=np.zeros(1, np.int32))
    return scene_forces(directory, human=human, config=config.load(), edge_weight=3000.0, edge_speed_m_s=0.8)


def test_a_standing_body_carries_its_weight_on_its_feet(tmp_path):
    human = body.load("cpu")
    out = _solve(tmp_path, human, soles=True)

    weight = float(out["total_mass"][0]) * 9.81
    total = out["frame_forces"][0].sum(axis=1)                     # (N, 3) world newtons
    assert np.allclose(total[:, 1], weight, rtol=0.05)             # pushes up by the body weight
    assert np.abs(total[:, [0, 2]]).max() < 0.05 * weight          # nothing sideways
    assert np.linalg.norm(out["base_wrench"][0, :, :3], axis=-1).max() < 0.05 * weight  # the residual is explained
    assert not out["frame_forces"][~out["frame_contact"]].any()    # only labelled frames push


def test_without_contact_there_is_no_force_and_the_body_is_unsupported(tmp_path):
    human = body.load("cpu")
    out = _solve(tmp_path, human, soles=False)

    weight = float(out["total_mass"][0]) * 9.81
    assert not out["frame_forces"].any()
    assert np.allclose(np.linalg.norm(out["joint_torques"][0, :, :3], axis=-1), weight, rtol=0.05)
