"""The BVR 3D viewer: ``out/<stem>/`` clips drawn with BetterRobot / BetterHuman viewer layers.

* :mod:`tools.viewer.loading` — one clip's numpy (scene, cameras, gravity, body sources, video),
* :mod:`tools.viewer.layers` — the adapter that turns it into BetterRobot layers,
* :mod:`tools.viewer.datasets_layers` — readers for one dataset (BEDLAM2, force plates, wall
  sensors, the scene DB); the core never imports it, ``scripts/view_3d.py`` plugs it in,
* :mod:`tools.viewer.app` — the sidebar and clip switching, and :func:`view_3d`.

``scripts/view_3d.py`` is the CLI.
"""
from __future__ import annotations

from .app import view_3d

__all__ = ["view_3d"]
