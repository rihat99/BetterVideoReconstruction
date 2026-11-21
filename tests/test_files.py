"""``bvr.files``: the helpers the stages read and write through."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from bvr import files


def test_a_video_is_written_at_a_rate_mpeg4_cannot_hold_exactly(tmp_path: Path) -> None:
    # 119.683 fps is the rate of a 4x frame-interpolated 29.92 fps clip; its exact time base overflows MPEG-4's.
    writer = files.video_writer(tmp_path / "overlay.mp4", 119.68310301507537, (64, 48))
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    assert sum(1 for _ in files.frames(tmp_path / "overlay.mp4")) == 3
    assert abs(files.frame_rate(tmp_path / "overlay.mp4") - 119.68) < 0.01
