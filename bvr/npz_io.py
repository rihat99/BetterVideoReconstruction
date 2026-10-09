"""Atomic ``.npz`` writing shared by every stage that persists an artifact.

A sweep killed mid-write must never leave a truncated npz behind: resume checks
are existence-based, so a torn file would read as *done* and poison every later
stage. Writing to a sibling temp file and renaming over the target makes the
artifact either the complete old version or the complete new one.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def atomic_savez(path: str | Path, **arrays) -> None:
    """``np.savez(path, **arrays)`` via a temp file + atomic rename."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)
