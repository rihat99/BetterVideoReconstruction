"""Where the data lives: the one root outside the repo.

The data root defaults to ``<repo>/../data``; set ``BVR_DATA_ROOT`` to move it.
The example clips and their outputs sit under ``EXAMPLES`` (``data``, ``data_2``,
``out``, ``out_2``); every dataset's own root lives in its ``datasets/<name>/paths.py``.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("BVR_DATA_ROOT", REPO_ROOT.parent / "data"))
EXAMPLES = DATA_ROOT / "examples"
