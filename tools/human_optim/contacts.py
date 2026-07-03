"""Scene-contact estimation for the optimized SMPL-X bodies.

Given a posed (world-frame, scaled) SMPL-X mesh and the static scene point cloud,
mark which body vertices touch the scene with a point-cloud SDF, then lift the
per-vertex contacts to **per-joint** contacts by nearest-joint assignment (each
vertex belongs to the joint center it is spatially closest to in the rest pose).

The SDF is the same signed distance the kinematic stage's scene-penetration loss
uses (:func:`tools.human_optim.losses.scene_penetration`): for each body vertex,
the nearest scene point ``p`` with its outward PCA normal ``n`` gives a signed
distance ``(v − p)·n`` — negative behind the surface (penetrating), positive in
front — using the camera-oriented normals from
:func:`tools.geometry.normals.estimate_point_normals`. A vertex is *in contact*
when its signed distance falls below a small threshold (inside the scene, or just
outside it).

This is a one-shot analysis (no gradients, run once per clip), so the scene
nearest-neighbour is a plain per-frame pruned ``cdist`` rather than the
chunked-batched machinery the per-iteration losses need.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def scene_sdf(
    vertices: torch.Tensor,        # (T, V, 3) posed world-frame (scaled) vertices
    scene_pts: torch.Tensor,       # (M, 3) world scene points
    scene_normals: torch.Tensor,   # (M, 3) outward unit normals
    scene_conf: torch.Tensor,      # (M,) per-point confidence
    *,
    prune_margin_m: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-vertex signed distance to the scene surface, frame by frame.

    Returns ``(sdf, dmin, conf)``, each ``(T, V)``:

      * ``sdf``  — signed Euclidean distance ``sign((v−p)·n)·‖v−p‖`` (metres);
        negative = behind the surface (penetrating). The magnitude is the true
        Euclidean distance to the nearest scene point; only the **sign** comes
        from the normal, so a vertex far to the *side* of a surface reads as far
        (no lateral false "closeness").
      * ``dmin`` — Euclidean distance to that nearest scene point (metres).
      * ``conf`` — confidence of that nearest scene point.

    The scene cloud is large (~10⁵ pts) but only points near the body matter, so
    each frame prunes to the posed-vertex AABB + ``prune_margin_m`` before the
    ``cdist`` (the body is ~2 m; the pruned set is a few thousand points). Frames
    whose body box contains no scene point return ``+inf`` distance / zero
    confidence (⇒ no contact).
    """
    T, V = vertices.shape[0], vertices.shape[1]
    dev, dt = vertices.device, vertices.dtype
    INF = torch.finfo(dt).max
    sdf = torch.full((T, V), INF, device=dev, dtype=dt)
    dmin = torch.full((T, V), INF, device=dev, dtype=dt)
    conf = torch.zeros((T, V), device=dev, dtype=scene_conf.dtype)
    for t in range(T):
        v = vertices[t]                                          # (V, 3)
        lo = v.amin(0) - prune_margin_m
        hi = v.amax(0) + prune_margin_m
        m = ((scene_pts >= lo) & (scene_pts <= hi)).all(dim=-1)  # (M,) bool
        if not bool(m.any()):
            continue
        cp, cn, cc = scene_pts[m], scene_normals[m], scene_conf[m]
        d = torch.cdist(v[None], cp[None])[0]                    # (V, Mc)
        dm, idx = d.min(dim=1)                                   # (V,)
        p, n = cp[idx], cn[idx]
        signed = ((v - p) * n).sum(dim=-1)                       # (V,) perpendicular sign
        sdf[t] = torch.where(signed < 0, -dm, dm)               # signed Euclidean SDF
        dmin[t] = dm
        conf[t] = cc[idx]
    return sdf, dmin, conf


def contact_from_sdf(
    sdf: torch.Tensor,      # (T, V) signed Euclidean SDF (m)
    dmin: torch.Tensor,     # (T, V) Euclidean NN distance (m)
    conf: torch.Tensor,     # (T, V) NN-point confidence
    *,
    close_thresh_m: float,
    max_dist_m: float,
    conf_thresh: float,
) -> torch.Tensor:
    """Boolean per-vertex contact ``(T, V)``.

    A vertex is in contact when it is **inside** the scene (``sdf < 0``) or
    **just outside** it (``0 ≤ sdf < close_thresh_m``) — i.e. ``sdf <
    close_thresh_m`` — provided it is genuinely near a confident surface. Two
    guards (mirroring the penetration loss) keep strays from false-flagging:
    ``dmin < max_dist_m`` rejects a vertex whose only "penetration" is a deep,
    almost-always-spurious correspondence behind an occluder, and ``conf >
    conf_thresh`` trusts only well-observed scene points.
    """
    return (sdf < close_thresh_m) & (dmin < max_dist_m) & (conf > conf_thresh)


def vertex_to_joint(vertices: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
    """Map each vertex to its **nearest joint center** ``(V,)`` (rest-pose 3D).

    ``vertices (V, 3)`` and ``joints (J, 3)`` are the shaped rest-pose mesh and
    joint locations (``SMPLBase.forward_shape``); each vertex is assigned to the
    spatially closest joint. This is the "vertex belongs to joint" relation used
    to lift per-vertex contacts to per-joint contacts.

    Nearest-joint, rather than LBS-``argmax`` (the joint that *deforms* the
    vertex), keeps the contact label where the surface physically is. Skinning
    dominance pulls a whole limb segment up to its *proximal* joint — the thigh,
    including flesh sitting right at the knee, all reads "hip" — displacing the
    contact 10–30 cm from the touch. Nearest-joint roughly halves that
    displacement and orphans nothing (every joint still owns ≥25 verts, fingers
    included).
    """
    return torch.cdist(vertices[None], joints[None])[0].argmin(dim=1)


def lift_joint_contact(
    vcontact: torch.Tensor,         # (T, V) bool per-vertex contact
    sdf: torch.Tensor,              # (T, V) signed SDF (m) — for the diagnostic
    v2j: torch.Tensor,              # (V,) long vertex→joint
    num_joints: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lift per-vertex contacts to per-joint ``(T, J)``.

    A joint is in contact when **any** vertex assigned to it is in contact (a
    ``scatter`` max over the joint's vertices). Also returns ``joint_sdf_cm
    (T, J)`` — the *minimum* signed distance (cm) over each joint's vertices, the
    continuous quantity behind the boolean (handy for thresholding / diagnosis).
    A joint with no near-surface vertex this frame gets ``nan``.
    """
    T = vcontact.shape[0]
    dev = vcontact.device
    INF = torch.finfo(sdf.dtype).max
    idx = v2j[None].expand(T, -1)                               # (T, V)
    jc = torch.zeros((T, num_joints), device=dev)
    jc.scatter_reduce_(1, idx, vcontact.to(jc.dtype), reduce="amax", include_self=True)
    jsdf = torch.full((T, num_joints), INF, device=dev, dtype=sdf.dtype)
    jsdf.scatter_reduce_(1, idx, sdf, reduce="amin", include_self=True)
    jsdf_cm = torch.where(jsdf < INF * 0.5, jsdf * 100.0, torch.full_like(jsdf, float("nan")))
    return jc > 0.5, jsdf_cm
