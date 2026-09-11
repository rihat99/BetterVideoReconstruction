"""The viewer's group (sport) filter over the clip list (``clips_in_group``)."""

from __future__ import annotations

from pathlib import Path

from tools.viewer.app import clips_in_group

_CLIPS = [Path("a_0001"), Path("b_0002"), Path("c_0003"), Path("d_0004")]
_GROUPS = {"a_0001": "pommel", "b_0002": "beam", "c_0003": "pommel"}  # d_0004 unlabeled


def test_all_keeps_every_clip_in_order():
    assert clips_in_group(_CLIPS, _GROUPS, "all") == _CLIPS


def test_label_keeps_only_that_group():
    assert clips_in_group(_CLIPS, _GROUPS, "pommel") == [Path("a_0001"), Path("c_0003")]
    assert clips_in_group(_CLIPS, _GROUPS, "beam") == [Path("b_0002")]


def test_unlabeled_clips_match_no_label():
    assert Path("d_0004") not in clips_in_group(_CLIPS, _GROUPS, "pommel")
    assert clips_in_group(_CLIPS, {}, "pommel") == []
