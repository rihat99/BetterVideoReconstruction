"""Per-person world-frame kinematic optimization (Adam warm-up → L-BFGS polish).

The leaf variables are ``q (T, 211)`` in ``better_human`` order; quaternions are
normalized inside every forward (so there is no exp-retraction singularity). Each
iteration: ``better_human`` forward → world joints / vertices → the weighted losses
(two robust reprojection terms, chamfer, pose / orientation priors, smoothness,
scene penetration, scene-contact attraction). Visibility and the chamfer /
penetration nearest-neighbours are recomputed detached each call.

The schedule is a list of :class:`PhaseSpec` — each phase names its trainable DOF
groups, iteration count, whether the scene terms are active, its optimizer (Adam or
L-BFGS), and **its own weight column**. The caller (the ``human_optimization``
dispatcher) maps the user-facing stages onto phases: ``root`` (rigid placement,
joints frozen, no scene) then ``kinematics_1`` (``full`` → ``refine`` → ``lbfgs``);
``kinematics_2`` reuses the same phase shape with its own column + the annotation
terms. Priors and the init-projection anchor are pinned to the *call's input* pose
and threaded across all its phases — so root and kinematics_1 share one call
(anchored to the caller's starting pose: the scale stage's pre-fit when present,
else the converter) rather than running as separate calls.

All posed-mesh inference stays in ``better_human``. This module *defines the
problem* — the forward, the named loss terms, the phase plan — and hands the
loop to the shared engine (:mod:`tools.optim`), which owns the optimizers,
gradient masking and per-term logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .kindyn import DynamicsSpec

from tools.body import frame_normals

from .camera import project, transform_points
from .contacts import detect_frame_contacts, joint_drift, vertex_to_joint
from .keypoints import smooth_tracks
from tools.optim import Phase, Problem, State, run_phases

from .losses import (
    acquisition_penalty, attraction_penalty, chamfer_loss, clearance_penalty,
    init_reprojection_gm, joint_acceleration_loss, joint_velocity_loss, pad_frames,
    penetration_penalty, pose_prior_loss, radial_velocity_loss, root_orient_prior_loss,
    sapiens_reprojection_gm, scene_penetration, scene_signed_distance, seating_penalty,
)
from .motion import tangent_difference
from .prefit import finger_freeze_mask
from .qmath import normalize_q
from .visibility import front_facing_mask, vertex_normals


@dataclass
class Metrics:
    reproj_sap_px: float  # Sapiens Goliath-308 keypoint reprojection (score-weighted)
    reproj_init_px: float  # body-joint drift from the starting pose's projection (init anchor)
    chamfer_m: float
    accel: float
    jerk: float
    snap: float           # ‖snap‖ (4th tangent difference) — the highest-frequency jitter
    pen_cm: float = float("nan")  # mean contact-frame penetration depth into the scene (nan = no scene)
    pen_n: int = 0                # number of penetrating contact-frame instances (summed over frames)
    depth_m: float = float("nan")  # mean pelvis camera-frame depth (diagnostic)
    radial_cms: float = float("nan")  # mean per-joint radial (viewing-ray) speed, cm/s

    def __str__(self) -> str:
        ini = "n/a" if self.reproj_init_px != self.reproj_init_px else f"{self.reproj_init_px:.2f}px"
        dep = "" if self.depth_m != self.depth_m else f"  depth={self.depth_m:.2f}m"
        pen = "" if self.pen_cm != self.pen_cm else f"  pen={self.pen_cm:.2f}cm/{self.pen_n}f"
        rad = "" if self.radial_cms != self.radial_cms else f"  radial={self.radial_cms:.1f}cm/s"
        return (f"reproj_sap={self.reproj_sap_px:.2f}px  reproj_init={ini}  "
                f"chamfer={self.chamfer_m * 100:.2f}cm  "
                f"‖accel‖={self.accel:.4f}  ‖jerk‖={self.jerk:.4f}  ‖snap‖={self.snap:.4f}{dep}{pen}{rad}")


@dataclass
class PhaseSpec:
    """One optimization phase: a DOF group + iteration budget + its own weight column.

    ``weights`` is the flat ``{loss term: weight}`` dict for this phase (a stage's
    column from the 4-stage config). ``dofs`` names the trainable ``q`` groups
    (``transl`` / ``orient`` / ``joints``); the rest are frozen by zeroing their
    gradient. ``scene`` builds the scene candidates and enables the scene terms.
    ``optimizer`` is ``"adam"`` (masked, scheduled warm-up) or ``"lbfgs"`` (all DOFs
    free, strong-Wolfe polish).
    """
    name: str
    weights: dict[str, float]
    iters: int
    dofs: tuple[str, ...]
    scene: bool
    optimizer: str = "adam"


def _dof_mask(active: tuple[str, ...], nq: int, device, *,
              freeze_fingers: bool = False) -> torch.Tensor:
    """``(nq,)`` 0/1 mask selecting the ``q`` DOFs trainable this phase.

    ``q`` packs the root translation (``[:3]``), the root quaternion (``[3:7]``),
    then the parent-local joint quaternions (``[7:]``). Freezing a DOF group is
    done by zeroing its gradient after backward, so only the named groups move.
    ``freeze_fingers`` additionally pins the 30 finger quaternions (the ``q``
    tail, :func:`~tools.human_optim.prefit.finger_freeze_mask`).
    """
    m = torch.zeros(nq, device=device)
    if "transl" in active:
        m[0:3] = 1.0
    if "orient" in active:
        m[3:7] = 1.0
    if "joints" in active:
        m[7:] = 1.0
    if freeze_fingers:
        m = m * finger_freeze_mask(nq, device)
    return m


def _visibility(vertices_sub_det: torch.Tensor, normals_sub: torch.Tensor,
                extrinsics: torch.Tensor) -> torch.Tensor:
    """Front-facing mask ``(T, Ns)`` for already-subsampled vertices."""
    vcam = transform_points(vertices_sub_det, extrinsics)
    ncam = torch.einsum("tij,tvj->tvi", extrinsics[:, :3, :3], normals_sub)
    return front_facing_mask(vcam, ncam)


def _padded_scene_candidates(box_points_det: torch.Tensor, scene_pts: torch.Tensor,
                             scene_normals: torch.Tensor, scene_conf: torch.Tensor,
                             margin: float, max_candidates: int):
    """Per-frame scene candidates near the body, padded to ``(T, Mc, …)`` for the
    chunked-batched :func:`~tools.human_optim.losses.scene_penetration`.

    The scene cloud is static but large (~10⁵ points over the whole route); only
    points near the body can ever be penetrated. ``box_points_det`` ``(T, K, 3)`` is
    the detached body point set the box is drawn around — the 52 joints plus the 12
    contact frames, which needs no mesh. Each frame keeps the scene points inside
    that AABB + ``margin`` (a margin covering both the joints-to-skin gap and the
    body's motion over the stage), then — if that still exceeds ``max_candidates`` —
    the ``max_candidates`` nearest to the box centre (the farther ones can't be any
    body point's nearest neighbour, so this is near-lossless but bounds the padded
    width, hence the per-iter ``cdist`` cost). Built once from the post-placement
    body and reused across a scene phase's iterations. Returns the four padded
    tensors ``(points, normals, confidence, valid)``."""
    pts, nrm, cnf = [], [], []
    for t in range(box_points_det.shape[0]):
        lo = box_points_det[t].amin(0) - margin
        hi = box_points_det[t].amax(0) + margin
        m = ((scene_pts >= lo) & (scene_pts <= hi)).all(dim=-1)
        cp, cn, cc = scene_pts[m], scene_normals[m], scene_conf[m]
        if cp.shape[0] > max_candidates:
            center = (0.5 * (lo + hi))[None]
            keep = torch.cdist(center, cp)[0].topk(max_candidates, largest=False).indices
            cp, cn, cc = cp[keep], cn[keep], cc[keep]
        pts.append(cp)
        nrm.append(cn)
        cnf.append(cc)
    dev, dt = box_points_det.device, box_points_det.dtype
    pad_p, valid = pad_frames(pts, channels=3, device=dev, dtype=dt)
    pad_n, _ = pad_frames(nrm, channels=3, device=dev, dtype=dt)
    pad_c, _ = pad_frames(cnf, channels=None, device=dev, dtype=scene_conf.dtype)
    return pad_p, pad_n, pad_c, valid


@torch.no_grad()
def _build_contact_patches(
    vw_sub: torch.Tensor,        # (T, Ns, 3) posed subsampled verts (detached), current pose
    group_vert_ids: list,        # per group: LongTensor of subsampled-vertex ids owned by the group
    group_active: torch.Tensor,  # (T, G) bool — group labeled in-contact at this valid frame
    scene_pts: torch.Tensor, scene_normals: torch.Tensor, scene_conf: torch.Tensor,
    extrinsics: torch.Tensor, K: torch.Tensor, image_h: int, image_w: int,
    *, radius_px: float, patch_k: int, conf_thresh: float,
) -> dict:
    """Per positive (frame, annotation-group) instance: a small TARGET PATCH of wall points.

    A contact label means SOME surface point of the group touches the wall, so the target
    is the wall visible where the limb appears — NOT the whole Voronoi patch. For each frame
    with an in-contact group, the scene is projected once; the patch is the scene points whose
    projection lands within ``radius_px`` of the group's member-vertex pixel centroid (confident
    + in front of the camera), capped to the ``patch_k`` nearest in 3D to the member centroid,
    with their stored normals. An instance is UNRESOLVED — no pull, counted — when the group's
    member centroid is behind the camera or its pixel centroid falls outside the ``image_h`` /
    ``image_w`` frame (plus a ``radius_px`` margin), or its footprint sees no qualifying wall
    point. The camera / bounds guard matters because :func:`~tools.human_optim.camera.project`
    clamps negative depth, so a behind-camera limb would otherwise resolve a bogus patch.
    Returns ``{"bundles": [(member_ids, frames, pts, nrm, valid)], "resolved": int,
    "unresolved": int}``, one bundle per group (``pts/nrm`` padded to ``patch_k``; ``valid``
    masks the pad)."""
    T, G = vw_sub.shape[0], len(group_vert_ids)
    dev, dt = vw_sub.device, vw_sub.dtype
    conf_ok = scene_conf > conf_thresh
    acc = [([], [], [], []) for _ in range(G)]          # per group: frames, pts, nrm, valid
    resolved = unresolved = 0
    for t in range(T):
        if not bool(group_active[t].any()):
            continue
        cam = transform_points(scene_pts[None], extrinsics[t:t + 1])[0]   # (S, 3) camera frame
        in_front = cam[:, 2] > 1e-6
        uv = project(cam[None], K[t:t + 1])[0]                            # (S, 2) pixels
        for g in range(G):
            if not bool(group_active[t, g]):
                continue
            vmem = vw_sub[t][group_vert_ids[g]]                           # (M, 3) member verts, world
            vmem_cam = transform_points(vmem[None], extrinsics[t:t + 1])[0]   # (M, 3) camera frame
            if float(vmem_cam[:, 2].mean()) <= 0.0:                       # limb centroid behind the camera
                unresolved += 1
                continue
            px_c = project(vmem_cam[None], K[t:t + 1])[0].mean(0)         # (2,) member pixel centroid
            if not (-radius_px <= float(px_c[0]) <= image_w + radius_px
                    and -radius_px <= float(px_c[1]) <= image_h + radius_px):   # off-image limb
                unresolved += 1
                continue
            near = ((uv - px_c).norm(dim=-1) < radius_px) & in_front & conf_ok
            if not bool(near.any()):
                unresolved += 1
                continue
            cp, cn = scene_pts[near], scene_normals[near]
            k = min(int(patch_k), cp.shape[0])
            idx = (cp - vmem.mean(0)).norm(dim=-1).topk(k, largest=False).indices
            pts, nrm = cp[idx], cn[idx]                                   # (k, 3)
            valid = torch.zeros(patch_k, dtype=torch.bool, device=dev)
            valid[:k] = True
            if k < patch_k:                                              # pad to a common width
                pts = torch.cat([pts, pts.new_zeros(patch_k - k, 3)], 0)
                nrm = torch.cat([nrm, nrm.new_zeros(patch_k - k, 3)], 0)
            acc[g][0].append(t); acc[g][1].append(pts); acc[g][2].append(nrm); acc[g][3].append(valid)
            resolved += 1
    bundles = []
    for g in range(G):
        fr, ptl, nrl, val = acc[g]
        if fr:
            bundles.append((group_vert_ids[g], torch.tensor(fr, dtype=torch.long, device=dev),
                            torch.stack(ptl, 0), torch.stack(nrl, 0), torch.stack(val, 0)))
    return {"bundles": bundles, "resolved": resolved, "unresolved": unresolved}


def _contact_instance_nn(vw_sub: torch.Tensor, patches: dict):
    """Per resolved instance: the closest ``(member vertex, patch point)`` pair.

    Returns ``(v_star, p_star, n_star, count)`` — ``v_star (I, 3)`` the posed member vertex
    (gradient flows through it), ``p_star`` / ``n_star`` its nearest patch point + that point's
    normal (both detached constants), ``count`` = resolved instances (the fixed loss
    denominator). The ``argmin`` over member-verts × patch-points is detached. ``None`` when
    there are no resolved instances."""
    bundles = patches["bundles"]
    if not bundles:
        return None
    INF = torch.finfo(vw_sub.dtype).max
    vs, ps, ns = [], [], []
    for member_ids, frames, pts, nrm, valid in bundles:
        posed = vw_sub[frames][:, member_ids]                            # (n, M, 3) grad through vw_sub
        n, M, k = posed.shape[0], posed.shape[1], pts.shape[1]
        with torch.no_grad():
            d = torch.cdist(posed.detach(), pts).masked_fill(~valid[:, None, :], INF)  # (n, M, k)
            flat = d.reshape(n, M * k).argmin(dim=1)                     # (n,)
            vi, pi = flat // k, flat % k
        ar = torch.arange(n, device=vw_sub.device)
        vs.append(posed[ar, vi])                                         # (n, 3) grad
        ps.append(pts[ar, pi]); ns.append(nrm[ar, pi])
    return torch.cat(vs, 0), torch.cat(ps, 0), torch.cat(ns, 0), int(patches["resolved"])


@torch.no_grad()
def _metrics(mid, betas, q_world, *, extrinsics, K, depth_pad, depth_valid, target_kp,
            scores, corr, frame_vertex_ids, sub, score_thresh, chunk=16,
            body_idx=None, init_target=None,
            scene_pts=None, scene_normals=None, scene_conf=None, pen=None,
            cam_centers=None, fps=30.0, frame_valid=None) -> Metrics:
    qn = normalize_q(q_world, mid.num_joints)
    out = mid.forward(betas, qn)
    jw = out.joints_world.tensor()[..., :3]
    fw = out.frames_world[..., :3]                    # (T, F, 3) contact-frame points
    vw = out.vertices
    # Sapiens reprojection in raw pixels (un-normalized) over scored keypoints
    body = jw[:, corr.joint_model, :]
    lm = vw[:, frame_vertex_ids[corr.lm_frame], :]
    pred_px = project(transform_points(torch.cat([body, lm], 1), extrinsics), K)
    tgt = target_kp[:, corr.sap, :]
    w = (scores[:, corr.sap] >= score_thresh).float()
    err = (pred_px - tgt).norm(dim=-1)
    reproj_sap = float((w * err).sum() / w.sum().clamp_min(1.0))
    # Init-anchor drift: mean body-joint pixel distance from the starting pose's
    # projection (honest px, not GM). ~0 at init; grows as the 3D terms pull the
    # body off its trusted starting 2D.
    if init_target is not None:
        ip = project(transform_points(jw[:, body_idx, :], extrinsics), K)
        reproj_init = float((ip - init_target).norm(dim=-1).mean())
    else:
        reproj_init = float("nan")
    # chamfer
    n = vertex_normals(vw, mid.faces)
    vismask = _visibility(vw[:, sub], n[:, sub], extrinsics)
    cham = float(chamfer_loss(vw[:, sub], vismask, depth_pad, depth_valid, chunk=chunk))
    # penetration into the scene cloud (diagnostic — the same SDF and the same 12
    # contact-frame points the loss uses; candidates pruned to each frame's body box)
    pen_cm, pen_n = float("nan"), 0
    if scene_pts is not None:
        cand = _padded_scene_candidates(torch.cat([jw, fw], 1), scene_pts, scene_normals, scene_conf,
                                        float(pen["prune_margin_m"]), int(pen["max_candidates"]))
        _, dcm, npen = scene_penetration(
            fw, *cand, conf_thresh=float(pen["conf_thresh"]),
            margin_m=float(pen["margin_m"]), max_dist_m=float(pen["max_dist_m"]),
            max_pen_m=float(pen["max_pen_m"]), huber_m=float(pen["huber_m"]), chunk=chunk)
        pen_cm, pen_n = float(dcm), int(npen)
    # mean pelvis camera-frame depth over frames with depth (a placement diagnostic)
    pelvis_z = transform_points(jw[:, :1, :], extrinsics)[:, 0, 2]   # (T,)
    fhd = depth_valid.any(dim=1)
    depth_m = (float((pelvis_z * fhd).sum() / fhd.sum().clamp_min(1))
               if bool(fhd.any()) else float(pelvis_z.mean()))
    # accel / jerk / snap norms (raw tangent diffs)
    d1 = tangent_difference(qn, mid.num_joints)
    accel = d1[1:] - d1[:-1]
    jerk = accel[1:] - accel[:-1]
    snap = jerk[1:] - jerk[:-1]
    # mean per-joint radial (viewing-ray) speed cm/s — the depth-jitter diagnostic
    radial_cms = float("nan")
    if cam_centers is not None and jw.shape[0] > 1:
        vstep = jw[1:] - jw[:-1]                                # (T-1, J, 3)
        ray = jw[:-1] - cam_centers[:-1, None, :]
        ray = ray / ray.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        radial = (vstep * ray).sum(dim=-1).abs()               # (T-1, J) per-step radial speed
        if frame_valid is not None:                            # match the loss: both-ends-valid steps only
            g = frame_valid[1:] & frame_valid[:-1]
            radial_cms = (float(radial[g].mean()) * float(fps) * 100.0) if bool(g.any()) else float("nan")
        else:
            radial_cms = float(radial.mean()) * float(fps) * 100.0
    return Metrics(reproj_sap, reproj_init, cham, float(accel.norm()), float(jerk.norm()),
                   float(snap.norm()), pen_cm, pen_n, depth_m, radial_cms)


def optimize_person(
    mid, betas_mean: torch.Tensor, q_world_init: torch.Tensor, *,
    extrinsics: torch.Tensor, K: torch.Tensor, depth_points: list[torch.Tensor],
    target_kp: torch.Tensor, scores: torch.Tensor, corr, frame_vertex_ids: torch.Tensor,
    sub: torch.Tensor, knobs: dict, phases: list[PhaseSpec],
    scene_pts: torch.Tensor | None = None, scene_normals: torch.Tensor | None = None,
    scene_conf: torch.Tensor | None = None, fps: float = 30.0,
    contact_labels: torch.Tensor | None = None,
    contact_conf: torch.Tensor | None = None,
    contact_groups: list | None = None,
    image_hw: tuple[int, int] | None = None,
    chamfer_weights: torch.Tensor | None = None,
    frame_valid: torch.Tensor | None = None,
    cam_centers: torch.Tensor | None = None,
    auto_contact: dict | None = None,
    freeze_fingers: bool = False,
    dynamics: DynamicsSpec | None = None,
) -> tuple[torch.Tensor, Metrics, Metrics, dict]:
    """Optimize one person over ``phases``. Returns ``(q_world_opt, before, after, aux)``.

    ``aux`` carries the detection-time signals the caller persists into the body
    npz: ``drift_pre (T, 12)`` — the net window drift of the 12 **contact frames** on the
    refine-start (pre-attraction) body, the honest pre-enforcement stillness measurement —
    and ``auto_labels (T, 12)`` int8 — the frozen labels the attraction used. Both
    are ``None`` when auto detection did not run (manual ``kinematics_2``, no scene, or a
    zero ``ann_attraction`` weight); the caller then writes NaN / ``-1`` for that person.

    Priors (pose / root-orientation) and the init-projection anchor are pinned to
    ``q_world_init`` — the *call's input* pose — and held across every phase, so
    ``root`` and ``kinematics_1`` belong to one call (anchored to the caller's
    starting pose: the scale pre-fit when present, else the converter). Each phase
    supplies its own weight column via :class:`PhaseSpec`; a term whose weight is 0
    in a phase contributes nothing there.

    ``contact_labels`` (optional ``(T, 52)`` int8 tri-state per SMPLXMid joint:
    ``1`` in-contact, ``0`` no-contact, ``-1`` unlabeled) turns on the
    **kinematics_2** annotation-driven scene terms in the scene phases — a *real*
    attraction pulling the vertices of in-contact joints onto the wall, a *strong*
    repulsion pushing the vertices of no-contact joints off it, and a velocity
    damper on in-contact joints (weights ``ann_*``). It defaults to the plain
    kinematics behaviour (no annotation terms). ``contact_conf`` (optional
    ``(T, 52)`` float in [0,1]) scales the attraction's vertex weights
    and the velocity damper's per-joint contribution by each joint's label
    confidence, so an uncertain auto-label pulls proportionally less; ``None`` keeps
    every labelled joint at weight 1.

    ``contact_groups`` (optional ``list[list[int]]`` — each annotation group's member
    SMPLXMid joint indices) turns on the **kinematics_2 contact-patch repair**: per
    positive ``(frame, group)`` instance a target patch of wall points is built once per
    scene phase (:func:`_build_contact_patches`) and the ``ann_acquisition`` /
    ``ann_seating`` terms pull the group's single closest vertex onto it and seat it at
    slight penetration. Off (``None``) or at zero acquisition/seating weight it is inert.
    ``image_hw`` (``(H, W)``) is required for it — the patch builder uses it to reject a
    group whose pixel footprint falls off-image (or behind the camera).

    ``auto_contact`` (optional dict) enables **kinematics_1** auto-contact mode: when
    ``contact_labels`` is not supplied and the ``ann_attraction`` weight is > 0, labels
    are detected **once** from the detached pose at the first scene phase
    (:func:`~tools.human_optim.contacts.detect_frame_contacts` at the ``[auto_contact]``
    operating point) and **frozen** — on the 12 **contact frames**, positive-only
    (``1`` on a detected frame, ``-1`` elsewhere, never ``0``), so only the
    attraction + velocity terms fire, never the clearance repulsion. In this mode those
    terms act on the contact-frame POINTS, not on vertices or joints. ``cam_centers``
    (``(T, 3)`` world camera centers) drives the radial-velocity damper + its
    diagnostic; both default off.

    ``freeze_fingers`` pins the 30 finger quaternions in every phase (Adam mask and the
    L-BFGS polish alike). The finger joints still exist and still carry the two fingertip
    contact frames — they simply follow the wrist instead of being fitted, because at
    cloud/keypoint resolution there is no evidence to fit them with.

    ``frame_valid`` (optional ``(T,)`` bool) excludes untracked frames — whose
    converter pose was held from the nearest valid frame and so is glued to the
    moving camera — from the init-projection anchor and the scene terms (just as
    the zeroed Sapiens scores / emptied depth cloud already exclude them from the
    reprojection / chamfer terms). Defaults to all-valid (a no-op on gap-free clips).

    The body is used at its native (converter) size — the metric geometry is
    already sized to it by the human-anchored ``estimate_scale`` stage upstream.

    ``dynamics`` (optional :class:`~tools.human_optim.kindyn.DynamicsSpec`) turns
    on the **kindyn_1** joint kinematics + dynamics mode: a world-force leaf per
    contact group per frame is added (plus a friction-pyramid activation leaf where
    the spec froze a cone), the RNEA base wrench / torques are threaded
    through the shared forward (recomputed from the *current* ``q`` so the physics
    residual pulls on the pose), the five dynamics terms are registered, and the
    forces are warm-started at the initial pose before the phases run. The solved
    forces + diagnostics are returned in ``aux['dynamics']``. ``None`` keeps the
    pure kinematic behaviour.
    """
    T = q_world_init.shape[0]
    betas = betas_mean[None].expand(T, -1).contiguous()

    # Shared (per-clip) knob sections — operational constants, not per-stage weights.
    oc = knobs["optimizer"]
    kp = knobs["keypoints"]
    score_thresh = float(kp["score_thresh"])
    sap_gm_px = float(kp["gm_scale_px"])
    pen_cfg = knobs["penetration"]
    chunk = int(knobs["chamfer"]["chunk"])
    max_cand = int(pen_cfg["max_candidates"])

    # Union over the phases about to run — decides what to SET UP once (the per-iter
    # term inclusion is gated by each phase's own weight below).
    def _wmax(term: str) -> float:
        return max((float(ph.weights.get(term, 0.0)) for ph in phases), default=0.0)

    use_init = _wmax("reproj_init") > 0.0
    init_gm_px = float(knobs["reproj_init"]["gm_scale_px"]) if use_init else 0.0
    use_pen = scene_pts is not None
    use_ann = contact_labels is not None          # manual labels present from the start (kinematics_2)
    # kinematics_1 auto-contact mode: bind labels once at the first scene phase (below) then
    # freeze. DETECTION is guarded on the ann_attraction weight-max, so a 0-weight column never
    # runs the detector. Auto attraction reads its band from [auto_contact]; the manual
    # (kinematics_2) path keeps [ann_attraction].
    use_auto = auto_contact is not None and not use_ann and _wmax("ann_attraction") > 0.0
    ann_at_cfg = auto_contact if use_auto else (knobs["ann_attraction"] if use_ann else None)
    ann_rep_cfg = knobs["ann_repulsion"] if use_ann else None   # clearance repulsion is manual-only
    ann_vel_ang_w = float(knobs.get("ann_velocity", {}).get("angular_weight", 0.0))  # angular/linear ratio
    # kinematics_2 contact-patch repair (ann_acquisition + ann_seating): per positive
    # (frame, annotation-group) instance, pull ONE closest vertex toward a per-instance
    # target patch of wall points and seat it at slight penetration. Needs the annotation
    # group→member-joint mapping (contact_groups) and non-zero acquisition/seating weight.
    use_contact_patch = (use_ann and contact_groups is not None and image_hw is not None
                         and (_wmax("ann_acquisition") > 0.0 or _wmax("ann_seating") > 0.0))

    depth_pad, depth_valid = pad_frames(depth_points, channels=3,
                                        device=q_world_init.device, dtype=q_world_init.dtype)

    # `v2j_sub` maps each subsampled vertex to its nearest rest-pose joint, so the
    # kinematics_2 manual joint labels can reach vertices (attraction / repulsion masks).
    # Auto (kinematics_1) mode needs no such map: its labels already live on the 12 contact
    # frames, which ARE points.
    v2j_sub = None
    if use_ann:
        with torch.no_grad():
            v_rest, j_rest = mid.forward_shape(betas_mean[None])
        v2j_sub = vertex_to_joint(v_rest[0], j_rest[0])[sub]   # (Ns,) nearest joint per vertex

    # kinematics_2 contact-patch groups: per annotation group, the subsampled-vertex ids it owns
    # (via v2j_sub) + a per-frame in-contact flag (any member joint labeled 1). A group with no
    # owned subsampled vertex is dropped — there is no vertex to seat.
    group_vert_ids: list = []
    group_active = None
    if use_contact_patch:
        cols = []
        for members in contact_groups:
            mj = torch.as_tensor(members, dtype=torch.long, device=q_world_init.device)
            vids = torch.nonzero((v2j_sub[:, None] == mj[None, :]).any(1), as_tuple=False).squeeze(1)
            if vids.numel() == 0:
                continue
            group_vert_ids.append(vids)
            cols.append((contact_labels[:, mj] == 1).any(dim=1))   # (T,) group in-contact this frame
        group_active = (torch.stack(cols, 1) if cols
                        else torch.zeros((T, 0), dtype=torch.bool, device=q_world_init.device))

    # per-vertex label-confidence weight (kinematics_2-on-auto). Scales the attraction; the
    # velocity damper takes the per-joint conf directly. None ⇒ every active vertex weighted 1.
    conf_v = None
    if contact_conf is not None and v2j_sub is not None:
        conf_v = contact_conf[:, v2j_sub].clamp(0.0, 1.0)      # (T, Ns)

    incontact_v = nocontact_v = incontact_f = None
    def _bind_labels(labels: torch.Tensor) -> None:
        """Set ``contact_labels`` + the mask of the points the annotation terms act on.

        Auto (kinematics_1) labels live on the contact frames, so the mask is the labels
        themselves; manual (kinematics_2) labels live on joints and are spread onto the
        subsampled vertices through ``v2j_sub``.
        """
        nonlocal contact_labels, incontact_v, nocontact_v, incontact_f
        contact_labels = labels
        if use_auto:
            incontact_f = labels == 1                         # (T, F) per contact frame
            return
        vlabel = labels[:, v2j_sub]                           # (T, Ns) tri-state per vertex
        incontact_v, nocontact_v = (vlabel == 1), (vlabel == 0)
    if use_ann:
        _bind_labels(contact_labels)                          # manual labels bind now (kinematics_2)

    # Init-projection 2D anchor target: each BODY joint's projection at the call's
    # input pose. body_idx excludes the 30 finger joints (tiny in image space, they
    # would swamp the uniform mean); the anchor holds those 22 major joints
    # image-locked. Built once (frozen). See losses.init_reprojection_gm.
    jnames = list(mid.joint_names)
    _finger = ("index", "middle", "pinky", "ring", "thumb")
    body_idx = torch.tensor(
        [i for i in range(mid.num_joints) if not any(f in jnames[i].lower() for f in _finger)],
        dtype=torch.long, device=q_world_init.device)
    init_target = None
    if use_init:
        with torch.no_grad():
            o0i = mid.forward(betas, normalize_q(q_world_init, mid.num_joints))
            jw0i = o0i.joints_world.tensor()[..., :3]
            init_target = project(transform_points(jw0i[:, body_idx, :], extrinsics), K)

    # Smooth the 2D TARGET tracks (not the trajectory) so converter / Sapiens per-frame
    # jitter is not imprinted into the objective and the smoothness terms stop fighting the targets.
    # Only the LOSS sees the smoothed targets; the gate metric (reproj_init) is reported vs the RAW
    # init target and reproj_sap vs the raw Sapiens keypoints, so smoothing cannot flatter the numbers.
    # Validity gates the SG filter within runs (frame_valid for init; score >= thresh for Sapiens);
    # the scores themselves are untouched. window_s = 0 ⇒ no-op.
    kp_smooth_s = float(kp.get("smooth_window_s", 0.0))
    init_smooth_s = float(knobs["reproj_init"].get("smooth_window_s", 0.0)) if use_init else 0.0
    target_kp_loss = target_kp
    if kp_smooth_s > 0.0:
        vkp = (scores >= score_thresh).detach().cpu().numpy()
        sm = smooth_tracks(target_kp.detach().cpu().numpy(), vkp, window_s=kp_smooth_s, fps=fps)
        target_kp_loss = torch.as_tensor(sm, dtype=target_kp.dtype, device=target_kp.device)
    init_target_loss = init_target
    if init_smooth_s > 0.0 and init_target is not None:
        vin = (frame_valid.detach().cpu().numpy() if frame_valid is not None else np.ones(T, dtype=bool))
        vin = np.broadcast_to(vin[:, None], init_target.shape[:2])
        sm = smooth_tracks(init_target.detach().cpu().numpy(), vin, window_s=init_smooth_s, fps=fps)
        init_target_loss = torch.as_tensor(sm, dtype=init_target.dtype, device=init_target.device)

    mk = lambda q: _metrics(
        mid, betas, q, extrinsics=extrinsics, K=K, depth_pad=depth_pad, depth_valid=depth_valid,
        target_kp=target_kp, scores=scores, corr=corr, frame_vertex_ids=frame_vertex_ids,
        sub=sub, score_thresh=score_thresh, chunk=chunk,
        body_idx=body_idx, init_target=init_target,
        scene_pts=scene_pts, scene_normals=scene_normals, scene_conf=scene_conf, pen=pen_cfg,
        cam_centers=cam_centers, fps=fps, frame_valid=frame_valid)
    before = mk(q_world_init)

    q_init = q_world_init.detach().clone()   # the anchor (priors + init reprojection)

    # ── the problem: one forward, named terms, phase-driven weights ──────────
    # The engine (tools.optim) owns the loop; this builds the kinematic Problem.
    # ``scene_cell`` holds the current phase's padded scene candidates (set by each
    # phase's on_start — None outside scene phases), so the lazy "sig" state entry
    # and the scene terms always read the running phase's candidates.
    scene_cell: dict = {"cand": None}
    # kinematics_2 contact patches (per scene phase's on_start; None outside scene phases).
    patch_cell: dict = {"patches": None}

    def _forward() -> State:
        st = State()
        qn = normalize_q(q, mid.num_joints)
        out = mid.forward(betas, qn)
        jw7 = out.joints_world.tensor()          # (T, J, 7) world SE3: position + scalar-last quat
        st.put("qn", qn)
        st.put("jw", jw7[..., :3])
        st.put("jwq", jw7[..., 3:7])
        st.put("vw", out.vertices)
        st.put("fw", out.frames_world[..., :3])       # (T, F, 3) contact-frame points
        st.put("fwq", out.frames_world[..., 3:7])     # (T, F, 4) contact-frame orientations

        def _vis():
            with torch.no_grad():
                vw = st["vw"]
                normals = vertex_normals(vw.detach(), mid.faces)
                return _visibility(vw.detach()[:, sub], normals[:, sub], extrinsics)

        def _sig():
            # One detached nearest-neighbour pass over the subsampled vertices, shared
            # by the kinematics_2 annotation terms (repulsion / manual attraction).
            cand = scene_cell["cand"]
            if cand is None:
                return None
            return scene_signed_distance(st["vw"][:, sub], *cand, chunk=chunk)

        def _sig_frames():
            # The same signed-distance pass over the 12 contact-frame points — shared by
            # penetration and the kinematics_1 contact attraction. 12 points instead of
            # 1000 verts, so the nearest-neighbour pass is ~83× cheaper.
            cand = scene_cell["cand"]
            if cand is None:
                return None
            return scene_signed_distance(st["fw"], *cand, chunk=chunk)

        def _inst_nn():
            # Per-instance closest (member vertex, patch point), shared by acquisition + seating.
            patches = patch_cell["patches"]
            if patches is None:
                return None
            return _contact_instance_nn(st["vw"][:, sub], patches)

        st.put_lazy("vismask", _vis)
        st.put_lazy("sig", _sig)
        st.put_lazy("sig_f", _sig_frames)
        st.put_lazy("inst_nn", _inst_nn)
        st.put_lazy("d1", lambda: tangent_difference(st["qn"], mid.num_joints))
        st.put_lazy("accel", lambda: st["d1"][1:] - st["d1"][:-1])
        st.put_lazy("jerk", lambda: st["accel"][1:] - st["accel"][:-1])
        st.put_lazy("snap", lambda: st["jerk"][1:] - st["jerk"][:-1])
        if dynamics is not None:
            dynamics.populate(st, f_free, gen_u)
        return st

    problem = Problem(_forward)
    q = problem.add_leaf("q", q_world_init)
    nq = q.shape[1]
    # kindyn_1: a free world-force leaf per contact group per frame — plus, where a
    # friction cone is frozen (feet/ankles), a second leaf of four pyramid-generator
    # activations that replaces it — and the five dynamics terms (registered here,
    # active only where their kindyn column weight is > 0).
    f_free = gen_u = None
    if dynamics is not None:
        f_free = problem.add_leaf("f_free", dynamics.zero_forces())
        gen_init = dynamics.zero_generators()
        if gen_init is not None:
            gen_u = problem.add_leaf("gen_u", gen_init)
        dynamics.register(problem)

    def _build_scene_cand():
        """Per-frame padded scene candidates from the *current* pose, or None.

        Built once at the start of a scene phase: by then the body is seated (the
        root / full phases placed it), so the candidates pruned to each frame's
        body box stay valid for the whole phase — reused every iteration. The box is
        drawn around the joints and the contact frames (no mesh); its 0.30 m margin
        covers the joints-to-skin gap, so the kinematics_2 vertex terms still see
        every scene point their vertices could reach."""
        if not use_pen:
            return None
        with torch.no_grad():
            o0 = mid.forward(betas, normalize_q(q, mid.num_joints))
            box = torch.cat([o0.joints_world.tensor()[..., :3], o0.frames_world[..., :3]], 1)
            cand = _padded_scene_candidates(box, scene_pts, scene_normals,
                                            scene_conf, float(pen_cfg["prune_margin_m"]), max_cand)
        if frame_valid is not None:      # untracked frames: drop their candidates → all scene terms skip them
            pad_p, pad_n, pad_c, valid = cand
            cand = (pad_p, pad_n, pad_c, valid & frame_valid[:, None])
        return cand

    def _detect_auto_labels() -> tuple[torch.Tensor, torch.Tensor]:
        """Detect frozen kinematics_1 contact labels once from the current detached pose.

        Runs :func:`~tools.human_optim.contacts.detect_frame_contacts` on the 12 contact
        frames at the ``[auto_contact]`` operating point. Returns ``(labels, drift)``:
        ``labels (T, F)`` int8 with ``1`` on a detected contact frame / ``-1`` everywhere
        else — never ``0``, so the clearance repulsion (which needs a ``0``) can never fire
        in auto mode; and ``drift (T, F)`` — the refine-start net window drift the detection
        gated on, kept so the caller can persist it as the honest pre-enforcement stillness
        measurement. No gradient path (``no_grad`` forward from the detached leaf), run once
        per stage — so contact detection cannot feed back into the fit."""
        ac = auto_contact
        with torch.no_grad():
            frames7 = mid.forward(betas, normalize_q(q, mid.num_joints)).frames_world
        points = frames7[..., :3]
        drift = joint_drift(points, fps=fps, window_s=float(ac["window_s"]),
                            valid_mask=frame_valid)
        detected = detect_frame_contacts(
            points, frame_normals(frames7), drift, scene_pts, scene_normals, scene_conf,
            frame_names=mid.contact_frame_names,
            close_thresh_m=float(ac["detect_thresh_m"]), exit_thresh_m=float(ac["detect_exit_m"]),
            min_frames=int(ac["min_frames"]), still_m=float(ac["still_m"]),
            max_dist_m=float(ac["max_dist_m"]), conf_thresh=float(ac["conf_thresh"]),
            prune_margin_m=float(ac["prune_margin_m"]))
        keep = detected.contact
        if frame_valid is not None:
            keep = keep & frame_valid[:, None]
        lab = torch.full_like(keep, -1, dtype=torch.int8)
        lab[keep] = 1
        return lab, drift

    # ── term registration (order fixes the summation order) ─────────────────
    # Each term reads the shared per-iteration State; a term is active in a phase
    # iff its weight there is non-zero (the engine skips the rest). The scene
    # penalty heads share a lazy nearest-neighbour pass — "sig_f" over the 12 contact
    # frames (penetration + auto attraction), "sig" over the subsampled vertices
    # (the kinematics_2 terms) — and return zeros outside scene phases (both are
    # None there).
    problem.add_term("reproj_sapiens", lambda st: sapiens_reprojection_gm(
        st["jw"], st["vw"], extrinsics, K, target_kp_loss, scores, corr, frame_vertex_ids,
        score_thresh=score_thresh, gm_scale_px=sap_gm_px))
    if use_init:
        problem.add_term("reproj_init", lambda st: init_reprojection_gm(
            st["jw"], body_idx, extrinsics, K, init_target_loss,
            gm_scale_px=init_gm_px, frame_mask=frame_valid))
    problem.add_term("chamfer", lambda st: chamfer_loss(
        st["vw"][:, sub], st["vismask"], depth_pad, depth_valid, chunk=chunk,
        vertex_weights=chamfer_weights))
    # Fitted fingers drop out of the pose prior: their starting values are the converter's
    # hand pose, which is exactly what unfreezing them says we no longer trust.
    prior_joints = (None if freeze_fingers
                    else int(finger_freeze_mask(nq, q.device)[7:].sum()) // 4)
    problem.add_term("pose_prior", lambda st: pose_prior_loss(
        st["qn"], q_init, mid.num_joints, num_prior_joints=prior_joints))
    problem.add_term("orient_prior", lambda st: root_orient_prior_loss(st["qn"], q_init))

    def _msq(x: torch.Tensor) -> torch.Tensor:
        return x.square().mean() if x.numel() > 0 else q.new_zeros(())   # short clip → 0, not NaN

    problem.add_term("accel_root", lambda st: _msq(st["accel"][:, :6]))
    problem.add_term("accel_joint", lambda st: _msq(st["accel"][:, 6:]))
    problem.add_term("jerk_root", lambda st: _msq(st["jerk"][:, :6]))
    problem.add_term("jerk_joint", lambda st: _msq(st["jerk"][:, 6:]))
    problem.add_term("snap_root", lambda st: _msq(st["snap"][:, :6]))
    problem.add_term("snap_joint", lambda st: _msq(st["snap"][:, 6:]))

    if use_pen:
        def _penetration(st):
            # On the 12 contact-frame points, not on mesh vertices: those points ARE
            # where contact is modelled, and keeping them out of the wall is what stops
            # a wrong-facing frame (a toe_above buried in a hold) from reading as touching.
            sig = st["sig_f"]
            if sig is None:
                return q.new_zeros(())
            loss, depth_cm, n_pen = penetration_penalty(
                *sig, conf_thresh=float(pen_cfg["conf_thresh"]), margin_m=float(pen_cfg["margin_m"]),
                max_dist_m=float(pen_cfg["max_dist_m"]), max_pen_m=float(pen_cfg["max_pen_m"]),
                huber_m=float(pen_cfg["huber_m"]))
            return loss, {"cm": depth_cm, "n": n_pen}
        problem.add_term("penetration", _penetration)

    if cam_centers is not None:
        problem.add_term("radial_velocity", lambda st: radial_velocity_loss(
            st["jw"], cam_centers, frame_valid=frame_valid))

    # Contact attraction pulls labeled points ONTO the wall: the 12 contact-frame POINTS
    # for kinematics_1's auto labels (band from [auto_contact]), the in-contact joints'
    # vertices for kinematics_2's manual labels (band from [ann_attraction]). Clearance
    # repulsion pushes no-contact-joint verts off — kinematics_2 only, since auto labels
    # carry no 0s (nocontact_v is empty).
    if (use_ann or use_auto) and use_pen:
        def _attraction(st):
            sig = st["sig_f" if use_auto else "sig"]
            active = incontact_f if use_auto else incontact_v
            if sig is None or active is None:
                return q.new_zeros(())
            loss, gap_cm, n_attr = attraction_penalty(
                *sig, conf_thresh=float(ann_at_cfg["conf_thresh"]),
                band_m=float(ann_at_cfg["band_m"]), max_dist_m=float(ann_at_cfg["max_dist_m"]),
                active_mask=active, active_weight=None if use_auto else conf_v)
            return loss, {"cm": gap_cm, "n": n_attr}
        problem.add_term("ann_attraction", _attraction)
    if use_ann and use_pen:
        def _repulsion(st):
            sig = st["sig"]
            if sig is None or nocontact_v is None:
                return q.new_zeros(())
            loss, gap_cm, n_push = clearance_penalty(
                *sig, conf_thresh=float(ann_rep_cfg["conf_thresh"]),
                clearance_m=float(ann_rep_cfg["clearance_m"]), max_dist_m=float(ann_rep_cfg["max_dist_m"]),
                max_pen_m=float(ann_rep_cfg["max_pen_m"]), huber_m=float(ann_rep_cfg["huber_m"]),
                active_mask=nocontact_v)
            return loss, {"cm": gap_cm, "n": n_push}
        problem.add_term("ann_repulsion", _repulsion)

    # kinematics_2 contact-patch repair: acquisition pulls each positive instance's single
    # closest vertex toward its target patch (long-range, bounded pseudo-Huber); seating then
    # seats that vertex at slight penetration and — being symmetric — pushes it back OUT after
    # overshoot (the anti-burial term). Both read the one shared per-instance NN "inst_nn".
    if use_contact_patch:
        aq_cfg = knobs["ann_acquisition"]
        se_cfg = knobs["ann_seating"]

        def _acquisition(st):
            nn = st["inst_nn"]
            if nn is None:
                return q.new_zeros(())
            v_star, p_star, _, cnt = nn
            loss, d_cm, n = acquisition_penalty(v_star, p_star, delta_m=float(aq_cfg["delta_m"]),
                                                resolved_count=cnt)
            return loss, {"cm": d_cm, "n": n}
        problem.add_term("ann_acquisition", _acquisition)

        def _seating(st):
            nn = st["inst_nn"]
            if nn is None:
                return q.new_zeros(())
            v_star, p_star, n_star, cnt = nn
            loss, seat_cm, n = seating_penalty(
                v_star, p_star, n_star, target_m=float(se_cfg["target_m"]),
                band_m=float(se_cfg["band_m"]), huber_m=float(se_cfg["huber_m"]), resolved_count=cnt)
            return loss, {"cm": seat_cm, "n": n}
        problem.add_term("ann_seating", _seating)
    if use_ann or use_auto:
        # Velocity damper (hold the labeled points still) needs only world placements, so it
        # runs in every phase the labels are present (refine + lbfgs for auto, all
        # phases for manual). Auto labels damp the contact FRAMES, manual ones the joints —
        # the loss is the same, only the (T, K) point set differs.
        pos_key, quat_key = ("fw", "fwq") if use_auto else ("jw", "jwq")

        def _velocity(st):
            if contact_labels is None:
                return q.new_zeros(())
            loss, speed_cms, ang_deg_s, n_steps = joint_velocity_loss(
                st[pos_key], contact_labels, fps=fps, joint_weight=contact_conf,
                world_quats=st[quat_key], angular_weight=ann_vel_ang_w)
            return loss, {"cm_s": speed_cms, "deg_s": ang_deg_s, "n": n_steps}
        problem.add_term("ann_velocity", _velocity)

        # Contact acceleration damper (steady, jerk-free holds) — the Cartesian 2nd
        # difference of the same labeled points.
        def _acceleration(st):
            if contact_labels is None:
                return q.new_zeros(())
            loss, accel_cms2, n_stencils = joint_acceleration_loss(
                st[pos_key], contact_labels, fps=fps, joint_weight=contact_conf)
            return loss, {"cm_s2": accel_cms2, "n": n_stencils}
        problem.add_term("ann_acceleration", _acceleration)

    # ── run the phases in order ──────────────────────────────────────────────
    # Adam (per-coordinate steps) tolerates the very different gradient scales here
    # (metre root translation vs joint quaternions, pixel reproj vs metre chamfer)
    # that would stall L-BFGS cold; the final L-BFGS phase (all DOFs free, strong-
    # Wolfe) then sharpens the already-close fit. The engine builds a fresh
    # optimizer per phase (no stale momentum leaks across an unfreeze) and freezes
    # inactive DOF groups by masking the gradient.
    auto_drift_pre: list = [None]   # (T, num_joints) refine-start drift, persisted by the caller

    def _phase_start(ph: PhaseSpec):
        def on_start() -> None:
            scene_cell["cand"] = _build_scene_cand() if ph.scene else None
            # Bind the frozen kinematics_1 labels once, at the first scene phase (the body has
            # seated by then).
            if use_auto and contact_labels is None and ph.scene and scene_cell["cand"] is not None:
                lab, auto_drift_pre[0] = _detect_auto_labels()
                _bind_labels(lab)
                per_frame = " ".join(f"{name}={int((lab[:, i] == 1).sum())}"
                                     for i, name in enumerate(mid.contact_frame_names))
                print(f"    [auto-contact] frozen detected labels @ {ph.name}: "
                      f"{int((lab == 1).any(1).sum())}/{lab.shape[0]} frames w/ contact  ({per_frame})")
            if use_contact_patch and use_pen and ph.scene:
                with torch.no_grad():
                    vw_sub0 = mid.forward(betas, normalize_q(q, mid.num_joints)).vertices[:, sub]
                aq_cfg = knobs["ann_acquisition"]
                patch_cell["patches"] = _build_contact_patches(
                    vw_sub0, group_vert_ids, group_active, scene_pts, scene_normals, scene_conf,
                    extrinsics, K, image_hw[0], image_hw[1], radius_px=float(aq_cfg["radius_px"]),
                    patch_k=int(aq_cfg["patch_k"]), conf_thresh=float(aq_cfg["conf_thresh"]))
                pk = patch_cell["patches"]
                print(f"    [contact-patch] {pk['resolved']} resolved / {pk['unresolved']} "
                      f"unresolved instances @ {ph.name}")
            else:
                patch_cell["patches"] = None
        return on_start

    # L-BFGS frees every DOF group (the polish), but a frozen group must STAY frozen there
    # too — so the finger freeze is the one mask that survives into the L-BFGS phase.
    lbfgs_mask = {"q": finger_freeze_mask(nq, q.device)} if freeze_fingers else None
    engine_phases = [
        Phase(ph.name, int(ph.iters), ph.weights, optimizer=ph.optimizer,
              grad_mask=(lbfgs_mask if ph.optimizer == "lbfgs" else
                         {"q": _dof_mask(ph.dofs, nq, q.device, freeze_fingers=freeze_fingers)}),
              on_start=_phase_start(ph))
        for ph in phases
    ]
    # kindyn_1: seed the forces at the (frozen) initial pose so the base-wrench
    # gradient on q is meaningful from the first phase (zero forces ⇒ freefall).
    if dynamics is not None:
        warm_f, warm_gen = dynamics.warmstart(normalize_q(q_world_init, mid.num_joints))
        with torch.no_grad():
            f_free.copy_(warm_f)
            if gen_u is not None:
                gen_u.copy_(warm_gen)
    run_phases(problem, engine_phases, lr=float(oc["lr"]), lr_decay=float(oc["lr_decay"]),
               lr_lbfgs=float(oc["lr_lbfgs"]), lbfgs_history=int(oc["lbfgs_history"]))

    with torch.no_grad():
        q_opt = normalize_q(q.detach(), mid.num_joints)
    aux = {"drift_pre": auto_drift_pre[0],                         # (T, J) refine-start drift, or None
           "auto_labels": contact_labels if use_auto else None}   # (T, J) int8 frozen labels, or None
    if dynamics is not None:
        aux["dynamics"] = dynamics.extract(q_opt, f_free.detach(),
                                           None if gen_u is None else gen_u.detach(), mid)
    return q_opt, before, mk(q_opt), aux
