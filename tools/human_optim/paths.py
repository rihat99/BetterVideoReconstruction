"""Output-path resolver for the human-optimization artifacts under ``out/<stem>/``.

Everything about an *optimized* human lives together in ``human_optim/`` —
``kinematic_1/2.npz``, ``dynamics.npz``, per-body ``contacts_*``,
overlays, and ``diagnostics/`` — kept apart from the raw body *estimates* in
``sam3d/`` (``smplx_params.npz``, the converter output, is the optimization INPUT
and stays there).
"""

from __future__ import annotations

from pathlib import Path

def human_optim_dir(pred: str | Path) -> Path:
    """The ``human_optim/`` directory for a clip (created on demand)."""
    d = Path(pred) / "human_optim"
    d.mkdir(parents=True, exist_ok=True)
    return d


def body_path(pred: str | Path, name: str) -> Path:
    """Refined-body npz by basename (``root`` / ``kinematic_1`` / ``kinematic_2`` /
    ``kindyn_1``)."""
    return human_optim_dir(pred) / f"{name}.npz"



def contacts_path(pred: str | Path, tag: str) -> Path:
    """Per-joint scene-contact npz for body ``tag`` (``1`` / ``2``)."""
    return human_optim_dir(pred) / f"contacts_{tag}.npz"


def overlay_path(pred: str | Path, name: str) -> Path:
    """Debug overlay mp4 (``name`` e.g. ``kinematic_1``, ``contacts_2``)."""
    return human_optim_dir(pred) / f"{name}_overlay.mp4"

