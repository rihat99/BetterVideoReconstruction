"""``bvr.stages.solve_dynamics``: what the ``kindyn_1`` file says beside a tracking gap and where a track ends."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from bvr import body, files, paths
from bvr.stages.solve_dynamics import save
from bvr.terms import dynamics, objective
from bvr.terms.motion import Time


def test_no_inverse_dynamics_is_written_where_its_stencil_reads_an_untracked_frame(tmp_path: Path) -> None:
    human, frames, time = body.load("cpu"), 12, Time(30.0)
    tracked = np.ones((2, frames), bool)
    tracked[0, 8:] = False                   # the first person's track ends mid-clip
    tracked[1, 5:7] = False                  # the second's has a gap
    q = torch.zeros(2, frames, human.model.nq)
    q[..., 6::4] = 1.0                       # unit quaternions: both stand still at rest
    q[torch.as_tensor(~tracked), :3] = 3.0   # an untracked frame holds no pose of the person
    names = list(human.contact_frames.names)
    held = torch.zeros(2, frames, len(names), dtype=torch.bool)
    held[..., names.index("heel_left")] = torch.as_tensor(tracked)
    down = np.array([0.0, -1.0, 0.0], np.float32)
    identity = dynamics.under_gravity(body.bake(human, torch.zeros(2, 10)), torch.as_tensor(down))
    skin = body.frame_normals(body.contact_frames(objective.posed(identity, q)))
    edges = dynamics.cone_edges(skin, 30.0) * held[..., None, None]
    start = {"object_ids": np.arange(2, dtype=np.int32), "valid_mask": tracked, "fps": np.float32(30.0),
             "video_width": np.int32(64), "video_height": np.int32(64),
             "drift_pre": np.zeros((2, frames, len(names)), np.float32),
             "auto_labels": np.zeros((2, frames, len(names)), np.int8)}
    clip = paths.Clip("clip", tmp_path / "clip.mp4", tmp_path)

    # Activations of minus one body weight: no force, so the root carries the whole weight.
    save(clip, start, identity, q, torch.full((*held.shape, 4), -1.0), edges, held, down, time)

    written = files.load_npz(clip.path(paths.KINDYN_1), "bvr-kindyn")
    counted = dynamics.stencil_rows(time, torch.as_tensor(tracked)).numpy()
    assert counted.sum(axis=1).tolist() == [7, 8]
    root = np.linalg.norm(written["joint_torques"][..., :3], axis=-1)
    weight = np.broadcast_to(written["total_mass"][:, None] * dynamics.GRAVITY, root.shape)
    assert np.allclose(root[counted], weight[counted], rtol=1e-3)
    for key in ("joint_torques", "base_wrench", "torque_mag"):
        assert not written[key][~counted].any(), key
