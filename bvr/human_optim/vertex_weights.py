"""Per-vertex extremity emphasis for the depth terms (geodesic distance to hands / feet).

A pose-invariant ``(V,)`` weight field over the SMPL-X template: geodesic surface
distance to the nearest hand / foot vertex, a falloff of scale ``radius`` metres,
log-mapped onto ``[lo, hi]`` so the core is divided and the tips multiplied by the
same factor. :func:`geodesic_endeffector_weights` is the one entry point; the scale
stage's pre-fit and the chamfer term build it once and index it per frame.
"""

from __future__ import annotations

import numpy as np

#: Anatomical groups, extremity → core. :func:`classify_joints` maps every
#: SMPL-X-mid joint name to exactly one of these.
GROUPS = ("hand", "foot", "forearm", "shin", "upper_arm", "thigh", "chest", "torso", "head")

_FINGERS = ("index", "middle", "pinky", "ring", "thumb")


def _np(x) -> np.ndarray:
    """Detach + move to host as a numpy array (accepts torch tensors or arrays)."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def classify_joints(joint_names) -> list[str]:
    """Map each SMPL-X-mid joint name to one anatomical :data:`GROUPS` label.

    Substring rules over the standard SMPL-X names (robust to ``left_`` / ``right_``
    prefixes): both wrists and all finger joints fold into ``hand``; ankles and toes
    into ``foot``. Raises if any joint is unclassified (guards against a renamed
    skeleton silently mis-weighting).
    """
    out: list[str] = []
    for name in joint_names:
        n = str(name).lower()
        if "wrist" in n or any(f in n for f in _FINGERS):
            group = "hand"
        elif "ankle" in n or "foot" in n or "toe" in n:
            group = "foot"
        elif "elbow" in n:
            group = "forearm"
        elif "knee" in n:
            group = "shin"
        elif "shoulder" in n:
            group = "upper_arm"
        elif "hip" in n:
            group = "thigh"
        elif "collar" in n:
            group = "chest"
        elif "spine" in n or n == "pelvis":
            group = "torso"
        elif "neck" in n or "head" in n:
            group = "head"
        else:
            raise ValueError(f"unclassified SMPL-X joint {name!r}")
        out.append(group)
    return out


def _edge_graph(verts: np.ndarray, faces: np.ndarray):
    """Sparse ``(V, V)`` CSR of undirected mesh-edge euclidean lengths."""
    from scipy.sparse import csr_matrix

    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    i, j = e[:, 0], e[:, 1]
    length = np.linalg.norm(verts[i] - verts[j], axis=1)
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    data = np.concatenate([length, length])
    return csr_matrix((data, (rows, cols)), shape=(len(verts), len(verts)))


def endeffector_seed_mask(lbs_weights, joint_names, *,
                          seed_groups=("hand", "foot")) -> np.ndarray:
    """Boolean ``(V,)`` — vertices whose ``argmax`` LBS joint is in ``seed_groups``."""
    groups = np.array(classify_joints(joint_names))
    v2j = _np(lbs_weights).argmax(axis=1)
    return np.isin(groups[v2j], list(seed_groups))


def geodesic_endeffector_field(verts, faces, lbs_weights, joint_names, *,
                               seed_groups=("hand", "foot")) -> np.ndarray:
    """Geodesic surface distance ``(V,)`` in metres to the nearest seed vertex.

    Seeds are the hand / foot vertices (:func:`endeffector_seed_mask`); multi-source
    Dijkstra over the mesh-edge graph gives every vertex its distance to the closest
    one. The head is naturally far from all seeds. Feed the result to :func:`falloff`.
    """
    from scipy.sparse.csgraph import dijkstra

    verts = _np(verts).astype(np.float64)
    faces = _np(faces).astype(np.int64)
    seeds = np.flatnonzero(endeffector_seed_mask(lbs_weights, joint_names,
                                                 seed_groups=seed_groups))
    dist = dijkstra(_edge_graph(verts, faces), directed=False, indices=seeds,
                    min_only=True)
    return dist                                            # (V,), 0 at seeds


def falloff(dist, *, kind: str = "gaussian", radius: float = 0.7,
            gamma: float = 2.0) -> np.ndarray:
    """Map distance ``(V,)`` → ``x ∈ [0, 1]`` (``1`` at a seed, decaying outward).

    ``radius`` (m) is the decay scale; ``gamma`` sharpens ``power``. ``kind`` is the
    "decreasing rate":

    * ``linear``     — ``max(0, 1 − d/radius)``
    * ``power``      — the linear ramp to the ``gamma`` (``2`` = quadratic)
    * ``smoothstep`` — ``3x²−2x³`` of the ramp (flat at both ends)
    * ``gaussian``   — ``exp(−(d/radius)²)`` (never hard-zero)
    * ``exp``        — ``exp(−d/radius)``
    """
    d = np.asarray(dist, dtype=np.float64)
    ramp = np.clip(1.0 - d / radius, 0.0, 1.0)
    if kind == "linear":
        return ramp
    if kind == "power":
        return ramp ** gamma
    if kind == "smoothstep":
        return ramp * ramp * (3.0 - 2.0 * ramp)
    if kind == "gaussian":
        return np.exp(-((d / radius) ** 2))
    if kind == "exp":
        return np.exp(-d / radius)
    raise ValueError(f"unknown falloff {kind!r}")


def scale_to_range(x, *, lo: float = 0.4, hi: float = 2.0, mode: str = "linear") -> np.ndarray:
    """Map ``x ∈ [0, 1]`` to weight units ``[lo, hi]``.

    ``mode="linear"`` maps affinely (``lo + (hi−lo)·x``). ``mode="log"`` maps
    geometrically (``lo·(hi/lo)**x``), so ``x=0.5`` lands on the geometric mean
    ``√(lo·hi)``; for a wide range like ``[0.1, 10]`` that puts the neutral weight
    ``1.0`` at the midpoint and makes the field a symmetric multiplicative emphasis
    (a tip is ``×10``, the core ``÷10``).
    """
    x = np.asarray(x, dtype=np.float64)
    if mode == "linear":
        return lo + (hi - lo) * x
    if mode == "log":
        return lo * (hi / lo) ** x
    raise ValueError(f"unknown scale mode {mode!r}")


def geodesic_endeffector_weights(vertices_template, faces, lbs_weights, joint_names, *,
                                 kind: str = "linear", radius: float = 1.0,
                                 lo: float = 0.1, hi: float = 10.0,
                                 seed_groups=("hand", "foot")) -> np.ndarray:
    """The ``(V,)`` geodesic-to-endeffector weight field the depth terms consume.

    Convenience for the chosen field: geodesic distance to the nearest hand / foot
    seed, a ``kind`` falloff of scale ``radius`` (m), log-mapped to ``[lo, hi]``
    (default **linear · r1.0**, core ÷10 … tip ×10). Pose-invariant, so both
    :mod:`bvr.stages.estimate_scale` and the chamfer term build it once from the
    SMPL-X-mid template and index it per frame.
    """
    geo = geodesic_endeffector_field(vertices_template, faces, lbs_weights, joint_names,
                                     seed_groups=seed_groups)
    return scale_to_range(falloff(geo, kind=kind, radius=radius), lo=lo, hi=hi, mode="log")
