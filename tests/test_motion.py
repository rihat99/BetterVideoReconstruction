"""The frame-rate rule: every derivative steps over about 1/30 s and is priced per 1/30 s."""
from __future__ import annotations

import pytest
import torch

from bvr.terms.motion import Time


@pytest.mark.parametrize(("fps", "stride"), [(24, 1), (25, 1), (29.97, 1), (30, 1), (60, 2), (100, 3), (120, 4)])
def test_the_stride_is_the_whole_number_of_frames_nearest_to_a_thirtieth_of_a_second(fps: float, stride: int) -> None:
    assert Time(fps).stride == stride


def test_at_30_fps_a_difference_is_the_plain_difference_of_neighbouring_frames() -> None:
    values = torch.randn(2, 9, 3, generator=torch.Generator().manual_seed(0))
    assert torch.equal(Time(30.0).difference(values), values[:, 1:] - values[:, :-1])


def test_a_motion_reads_the_same_at_every_frame_rate() -> None:
    # A point moving at 0.6 m/s moves 0.02 m per thirtieth of a second, however often it is filmed.
    for fps in (24.0, 25.0, 30.0, 60.0, 120.0):
        position = 0.6 * torch.arange(40)[None, :, None] / fps
        assert torch.allclose(Time(fps).difference(position), torch.tensor(0.02), atol=1e-6)


def test_a_row_counts_only_if_every_frame_from_its_first_to_its_last_is_tracked() -> None:
    tracked = torch.ones(1, 12, dtype=torch.bool)
    tracked[0, 5] = False
    time = Time(60.0)  # stride 2: a second derivative reads t, t + 2, t + 4 and spans the frames between
    rows = time.rows(tracked, 2)
    assert rows.shape == (1, 8)
    assert rows[0].tolist() == [True, False, False, False, False, False, True, True]
