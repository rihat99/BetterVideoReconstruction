"""Where the data lives: one place for every root outside the repo.

The data root defaults to ``<repo>/../data``; set ``BVR_DATA_ROOT`` to move it.
Example clips and their outputs sit under ``EXAMPLES`` (``data``, ``data_2``,
``out``, ``out_2``); the climbing-rig recordings under ``PETER``.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("BVR_DATA_ROOT", REPO_ROOT.parent / "data"))
EXAMPLES = DATA_ROOT / "examples"
PETER = DATA_ROOT / "peter"
LABVALIDATION = DATA_ROOT / "LabValidation_withVideos"
PARKOUR = DATA_ROOT / "Parkour-dataset"
CLIMBING = DATA_ROOT / "ClimbingVideos"
ESTMF = DATA_ROOT / "Estimating-3D-Motion-Forces"
