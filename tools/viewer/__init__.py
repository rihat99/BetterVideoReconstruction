"""viser 3D viewer for a batch of ``out/<stem>/`` fusion-geometry reconstructions.

A mechanical split of the former single-file ``scripts/view_3d.py`` into cohesive
concern modules:

* :mod:`tools.viewer.helpers` — small pure geometry / colour helpers,
* :mod:`tools.viewer.loading` — the ``out/<stem>/`` numpy loaders (no viser),
* :mod:`tools.viewer.scene` — :class:`~tools.viewer.scene.ClipScene`, every viser
  node for one clip,
* :mod:`tools.viewer.app` — the persistent-GUI :class:`~tools.viewer.app.Viewer`
  and the :func:`~tools.viewer.app.view_3d` entry point.

``scripts/view_3d.py`` stays the thin CLI wrapper around :func:`view_3d`.
"""
from __future__ import annotations

from .app import view_3d

__all__ = ["view_3d"]
