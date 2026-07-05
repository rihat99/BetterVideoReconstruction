"""Per-SMPL-X-vertex weight fields to emphasize/deemphasize regions in depth terms.

VGGT-Omega depth does not preserve human proportions — it stretches the extremities
(hands/feet/legs) in camera-z. The scale ruler (:mod:`scripts.stages.estimate_scale`)
and the chamfer term (:mod:`tools.human_optim.losses`) both consume this human depth
*per vertex* with a uniform weight. This module builds a per-vertex weight ``w (V,)``
over the SMPL-X-mid template so those terms can later weight regions unequally.

Every generator is driven by one **profile** — a base weight per anatomical group
(hand / foot / … / torso / head) — so the field is direction-agnostic: the
extremity-up profile proposed for this pipeline and its extremity-down inverse are
just two profiles. Weights are a property of vertex identity (canonical,
pose-invariant), so they are built once from the template.

Two generator families:

* :func:`group_weights` — semantic: each vertex takes its anatomical group's base
  weight, either as a smooth LBS-weighted blend (``soft=True``) or a hard
  ``lbs_weights.argmax`` assignment (``soft=False``). Head-low is trivial (low
  head / neck base).
* :func:`geodesic_endeffector_field` + :func:`falloff` — geometric: geodesic surface
  distance from each vertex to the nearest hand / foot seed, mapped through a tunable
  falloff (linear / power / smoothstep / gaussian / exp — the "decreasing rate"). The
  head is naturally low (far from every hand / foot seed).

Nothing here is wired into optimization. Visualize candidates with
``scripts/render_vertex_weights.py`` and pick one.

.. note::
   SMPL-X vertex density is not uniform (hands and head are far more finely
   tessellated). A raw per-vertex weight therefore acts partly as a per-mesh-density
   weight; :func:`vertex_areas` returns the barycentric per-vertex area so a consumer
   can area-normalize if it wants a true per-surface weighting. Off by default here —
   the visualization shows the raw field.
"""
from __future__ import annotations

import numpy as np

#: Anatomical groups, extremity → core. :func:`classify_joints` maps every
#: SMPL-X-mid joint name to exactly one of these.
GROUPS = ("hand", "foot", "forearm", "shin", "upper_arm", "thigh", "chest", "torso", "head")

#: Named base-weight profiles (group → weight). A profile is the single knob that sets
#: the field's *direction*; a generator only translates a profile into a per-vertex
#: field. Edit / add freely — these are the candidates to compare.
PROFILES: dict[str, dict[str, float]] = {
    # Emphasize the contacting extremities, trust the torso least (the proposed idea).
    "extremity_up": {
        "hand": 2.0, "foot": 2.0, "forearm": 1.4, "shin": 1.4,
        "upper_arm": 1.0, "thigh": 1.0, "chest": 0.6, "torso": 0.3, "head": 0.4,
    },
    # The inverse (trust the torso, distrust the stretched extremities) — the more
    # defensible direction for the single global scale factor.
    "extremity_down": {
        "hand": 0.4, "foot": 0.4, "forearm": 0.7, "shin": 0.7,
        "upper_arm": 1.0, "thigh": 1.0, "chest": 1.3, "torso": 1.6, "head": 0.5,
    },
    "uniform": {g: 1.0 for g in GROUPS},
}

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


def joint_base_weights(joint_names, profile: dict[str, float]) -> np.ndarray:
    """Per-joint base weight ``(J,)`` = each joint's group weight from ``profile``."""
    groups = classify_joints(joint_names)
    return np.array([float(profile[g]) for g in groups], dtype=np.float64)


def lbs_group_labels(lbs_weights, joint_names) -> np.ndarray:
    """Per-vertex group label ``(V,)`` by ``argmax`` LBS joint (object dtype)."""
    groups = np.array(classify_joints(joint_names), dtype=object)
    return groups[_np(lbs_weights).argmax(axis=1)]


def group_weights(lbs_weights, joint_names, profile: dict[str, float], *,
                  soft: bool = True) -> np.ndarray:
    """Per-vertex weight ``(V,)`` from a group ``profile``.

    :param lbs_weights: ``(V, J)`` SMPL-X-mid skinning matrix (``mid.lbs_weights``).
    :param joint_names: length-``J`` joint names (``mid.joint_names``).
    :param profile: group → base weight (see :data:`PROFILES`).
    :param soft: ``True`` blends the group bases by the LBS weights
        (``w_v = Σ_j lbs[v,j]·base[j]``, smooth); ``False`` gives each vertex its
        ``argmax`` joint's group weight (hard, blocky).
    """
    weights = _np(lbs_weights).astype(np.float64)          # (V, J)
    base_j = joint_base_weights(joint_names, profile)      # (J,)
    if soft:
        return weights @ base_j                            # (V,)
    return base_j[weights.argmax(axis=1)]                  # (V,)


#: SMPL-X 27-part vertex segmentation (Meshcapade ``smplx_vert_segmentation.json``,
#: exposed as ``mid.vertices_segmentation``) → the 9 anatomical :data:`GROUPS`. Note
#: the SMPL naming: ``Arm`` is the upper arm, ``ForeArm`` the forearm; ``UpLeg`` the
#: thigh, ``Leg`` the calf/shin; ``Shoulder`` is the shoulder-girdle patch.
_SEG_PART_TO_GROUP = {
    "leftHand": "hand", "rightHand": "hand",
    "leftHandIndex1": "hand", "rightHandIndex1": "hand",
    "leftFoot": "foot", "rightFoot": "foot",
    "leftToeBase": "foot", "rightToeBase": "foot",
    "leftForeArm": "forearm", "rightForeArm": "forearm",
    "leftLeg": "shin", "rightLeg": "shin",
    "leftArm": "upper_arm", "rightArm": "upper_arm",
    "leftUpLeg": "thigh", "rightUpLeg": "thigh",
    "leftShoulder": "chest", "rightShoulder": "chest",
    "hips": "torso", "spine": "torso", "spine1": "torso", "spine2": "torso",
    "neck": "head", "head": "head",
    "leftEye": "head", "rightEye": "head", "eyeballs": "head",
}


def segmentation_group_labels(vertices_segmentation, num_vertices: int, *,
                              fallback: np.ndarray | None = None) -> np.ndarray:
    """Per-vertex group label ``(V,)`` from the SMPL-X part segmentation.

    :param vertices_segmentation: part name → vertex-index list
        (``mid.vertices_segmentation``, 27 parts, with overlaps + a few gaps).
    :param num_vertices: ``V`` (10475).
    :param fallback: optional ``(V,)`` labels for the unlabeled vertices (e.g.
        :func:`lbs_group_labels`); unlabeled vertices default to ``torso`` otherwise.

    Overlapping parts are resolved by extremity-first priority (:data:`GROUPS`
    order — ``hand`` beats ``torso`` on a shared vertex).
    """
    order = {g: i for i, g in enumerate(GROUPS)}
    best = np.full(num_vertices, len(GROUPS), dtype=np.int64)   # current priority (lower wins)
    label_idx = np.full(num_vertices, -1, dtype=np.int64)
    for part, idx in vertices_segmentation.items():
        group = _SEG_PART_TO_GROUP.get(part)
        if group is None:
            continue
        prio = order[group]
        idx = np.asarray(idx, dtype=np.int64)
        win = idx[prio < best[idx]]
        label_idx[win] = prio
        best[win] = prio
    groups_arr = np.array(GROUPS, dtype=object)
    labels = np.where(label_idx >= 0, groups_arr[np.clip(label_idx, 0, None)], "torso")
    labels = labels.astype(object)
    if fallback is not None:
        empty = label_idx < 0
        labels[empty] = np.asarray(fallback, dtype=object)[empty]
    return labels


def segmentation_weights(vertices_segmentation, joint_names, lbs_weights,
                         profile: dict[str, float]) -> np.ndarray:
    """Per-vertex weight ``(V,)`` from the SMPL-X part segmentation + a ``profile``.

    A hard (blocky) field like :func:`group_weights` with ``soft=False``, but from
    the artist-labeled mesh parts rather than the LBS argmax. The LBS argmax fills
    the segmentation's few unlabeled vertices.
    """
    num_vertices = _np(lbs_weights).shape[0]
    fallback = lbs_group_labels(lbs_weights, joint_names)
    labels = segmentation_group_labels(vertices_segmentation, num_vertices, fallback=fallback)
    return np.array([float(profile[g]) for g in labels], dtype=np.float64)


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
    :mod:`scripts.stages.estimate_scale` and the chamfer term build it once from the
    SMPL-X-mid template and index it per frame.
    """
    geo = geodesic_endeffector_field(vertices_template, faces, lbs_weights, joint_names,
                                     seed_groups=seed_groups)
    return scale_to_range(falloff(geo, kind=kind, radius=radius), lo=lo, hi=hi, mode="log")


def normalize(weights, *, mean_one: bool = True, clamp=(0.25, 3.0)) -> np.ndarray:
    """Optionally rescale ``weights`` to unit mean, then clamp.

    Unit mean keeps a weighted loss comparable in magnitude to the uniform one so a
    downstream loss weight need not be re-tuned per profile.
    """
    weights = np.asarray(weights, dtype=np.float64).copy()
    if mean_one:
        mean = float(weights.mean())
        if mean > 1e-9:
            weights /= mean
    if clamp is not None:
        weights = np.clip(weights, clamp[0], clamp[1])
    return weights


def vertex_areas(verts, faces) -> np.ndarray:
    """Per-vertex area ``(V,)`` — a third of each incident triangle's area.

    Sums to the mesh area. Use to area-normalize a weight field
    (``w *= areas / areas.mean()``) so densely-tessellated regions (hands, head) are
    not over-counted purely for having more vertices.
    """
    verts = _np(verts).astype(np.float64)
    faces = _np(faces).astype(np.int64)
    tri = verts[faces]                                     # (F, 3, 3)
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)  # (F,)
    out = np.zeros(len(verts), dtype=np.float64)
    np.add.at(out, faces.ravel(), np.repeat(area / 3.0, 3))
    return out
