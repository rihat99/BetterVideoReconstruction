"""Make repository namespace modules win over similarly named site packages."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT in sys.path:
    sys.path.remove(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)
