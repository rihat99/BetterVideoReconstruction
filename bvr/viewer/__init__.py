"""The BVR 3D viewer: ``out/<stem>/`` clips drawn with BetterRobot / BetterHuman viewer layers.

* :mod:`bvr.viewer.loading` — one clip's numpy (scene, cameras, gravity, body sources, video),
* :mod:`bvr.viewer.layers` — the adapter that turns it into BetterRobot layers,
* the datasets' readers (``datasets/<name>/viewer.py``, ``datasets/_rigs.py``): the core never
  imports them, ``scripts/view_3d.py`` composes them into the ``read`` it hands the app,
* :mod:`bvr.viewer.app` — the sidebar and clip switching, and :func:`view_3d`.

``scripts/view_3d.py`` is the CLI.
"""
from __future__ import annotations

from .app import view_3d

__all__ = ["view_3d"]
