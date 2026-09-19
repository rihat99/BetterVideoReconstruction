"""The force-plate wrench algebra the OpenCap scripts share.

A force plate states its measurement as a *triple* — a force ``F``, a centre of pressure
``p`` on the floor, and a free moment ``T`` — while physics states it as a wrench. The two
are the same thing written two ways, and that conversion is used in three places: the
importer resamples 2000 Hz plate rows onto video frames (which must be done on the
WRENCH, since ``mean(p × F) ≠ mean(p) × mean(F)``), the exporter turns our solved point
forces back into a triple, and the tests round-trip the dataset's own files. One
implementation, so the three cannot drift.

**The identity, derived sign by sign.** About the ground origin a plate's triple is the
wrench

    F_O = F,    M_O = p × F + T,    T = T_y ŷ

— the free moment of a flat plate is vertical only, because the COP is *defined* as the
point where the horizontal moment components vanish. Inverting for the triple, with
``p_y`` pinned to the floor plane ``y = y0``, uses the three components of ``p × F``
(``x: p_y F_z − p_z F_y``, ``y: p_z F_x − p_x F_z``, ``z: p_x F_y − p_y F_x``):

    p_z = (y0 · F_z − M_x) / F_y
    p_x = (M_z + y0 · F_x) / F_y
    T_y = M_y − (p_z F_x − p_x F_z)

Note which component carries ``y0``: the ``x`` moment row pairs it with ``F_z`` and the
``z`` row pairs it with ``F_x``. ``tests/test_opencap_forces.py`` round-trips every loaded
sample of the dataset's own force files through this to ~1e-16 m and ~1e-13 N·m.

The inversion divides by ``F_y``, so it says nothing where the plate carries no weight.
:func:`cop_free_torque` returns that as a mask rather than a policy — the importer writes
NaN there, the exporter zeroes an unloaded row and places a fallback point on the rare
frame that carries a large horizontal force with no vertical one.
"""

from __future__ import annotations

import numpy as np

#: Vertical force above which a plate's COP is a defined, meaningful point.
LOADED_N = 20.0


def origin_moment(cop: np.ndarray, force: np.ndarray, free_torque: np.ndarray) -> np.ndarray:
    """``(N, 3)`` moment of a plate triple about the ground ORIGIN — the averageable form."""
    return np.cross(cop, force) + free_torque


def cop_free_torque(force: np.ndarray, moment: np.ndarray,
                    y0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wrench about the ground origin → ``(cop (N, 3), free_torque (N, 3), defined (N,))``.

    :param force: ``(N, 3)`` force in the OpenSim ground frame, newtons.
    :param moment: ``(N, 3)`` moment of that force about the ground ORIGIN, N·m.
    :param y0: the floor plane the COP is placed on, metres.
    :returns: the COP, the free moment (vertical only), and the mask of rows where the
        vertical force clears :data:`LOADED_N` — the rows where a COP exists at all. The
        two vectors are set to zero where it does not; the caller decides what belongs
        there instead.
    """
    fx, fy, fz = force.T
    mx, my, mz = moment.T
    defined = fy > LOADED_N
    denom = np.where(defined, fy, 1.0)
    pz = (y0 * fz - mx) / denom
    px = (mz + y0 * fx) / denom
    cop = np.stack([px, np.full_like(px, y0), pz], 1)
    torque = np.zeros_like(cop)
    torque[:, 1] = my - (pz * fx - px * fz)
    gate = defined[:, None]
    return cop * gate, torque * gate, defined


def vertical_free_torque(point: np.ndarray, force: np.ndarray, moment: np.ndarray) -> np.ndarray:
    """``(N, 3)`` vertical moment left over about ``point`` — the free torque of a given COP.

    Used where :func:`cop_free_torque` cannot place the COP itself (no vertical load) and
    the caller supplies a point of its own. Only the vertical residual is kept; the
    horizontal one is dropped, exactly as a plate's triple drops it.
    """
    residual = moment - np.cross(point, force)
    out = np.zeros_like(force)
    out[:, 1] = residual[:, 1]
    return out
