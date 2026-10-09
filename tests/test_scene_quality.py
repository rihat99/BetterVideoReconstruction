"""The calibrated per-clip quality flags (bvr/geometry/quality.py)."""

from __future__ import annotations

from bvr.geometry.quality import quality_flag

_KNOBS = dict(cam_depth_max=0.4, res_before_bw_max=2.5, smear_med_max=2.0)


def test_camera_flag_needs_both_witnesses() -> None:
    """A real handheld move (big cam ratio, coherent physics — boulder_1) passes;
    a hallucinated dolly (big ratio AND inexplicable residual — 1FGkg) is flagged."""
    assert quality_flag(0.71, 1.0, 0.7, **_KNOBS) == ""
    assert quality_flag(0.68, 2.8, 0.7, **_KNOBS) == "camera"
    assert quality_flag(0.1, 7.7, 0.7, **_KNOBS) == "", "residual alone is not a camera flag"


def test_depth_flag_and_combination() -> None:
    assert quality_flag(0.0, 1.0, 7.9, **_KNOBS) == "depth"
    assert quality_flag(1.3, 7.7, 7.9, **_KNOBS) == "camera+depth"


def test_missing_witnesses_never_flag() -> None:
    nan = float("nan")
    assert quality_flag(nan, nan, nan, **_KNOBS) == ""
    assert quality_flag(0.9, nan, nan, **_KNOBS) == ""
