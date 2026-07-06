"""Per-person world-frame kinematic optimization (Adam warm-up → L-BFGS polish).

The leaf variables are ``q (T, 211)`` in ``better_human`` order; quaternions are
normalized inside every forward (so there is no exp-retraction singularity). Each
iteration: ``better_human`` forward → world joints / vertices → the weighted losses
(two robust reprojection terms, chamfer, pose / orientation priors, smoothness,
scene penetration, scene-contact attraction, velocity-gated contact repulsion).
Visibility and the chamfer / penetration nearest-neighbours are recomputed detached
each call.

The schedule is a list of :class:`PhaseSpec` — each phase names its trainable DOF
groups, iteration count, whether the scene terms are active, its optimizer (Adam or
L-BFGS), and **its own weight column**. The caller (the ``human_optimization``
dispatcher) maps the user-facing stages onto phases: ``root`` (rigid placement,
joints frozen, no scene) then ``kinematics_1`` (``full`` → ``refine`` → ``lbfgs``);
``kinematics_2`` reuses the same phase shape with its own column + the annotation
terms. Priors and the init-projection anchor are pinned to the *call's input* pose
and threaded across all its phases — so root and kinematics_1 share one call
(anchored to the converter pose) rather than running as separate calls.

The contact loss's per-joint speed comes from a BetterRobot kinematic model built
once per person (:mod:`tools.human_optim.velocity`); all posed-mesh inference stays
in ``better_human``. This module assembles the objective and drives ``torch.optim``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tools.smplx_robot import make_smplx_robot_model

from .camera import project, transform_points
from .contacts import vertex_to_joint
from .losses import (
    chamfer_loss, clearance_repulsion, contact_velocity_loss, depth_push_loss,
    init_reprojection_gm, joint_velocity_loss, pad_frames,
    pose_prior_loss, root_orient_prior_loss, sapiens_reprojection_gm,
    scene_attraction, scene_penetration, smoothness_loss,
)
from .motion import tangent_difference
from .qmath import normalize_q
from .velocity import joint_local_linear_speed
from .visibility import front_facing_mask, vertex_normals


@dataclass
class Metrics:
    reproj_sap_px: float  # Sapiens Goliath-308 keypoint reprojection (score-weighted)
    reproj_init_px: float  # body-joint drift from the converter-pose projection (init anchor)
    chamfer_m: float
    accel: float
    jerk: float
    pen_cm: float = float("nan")  # mean penetration depth into the scene cloud (nan = no scene)
    pen_n: int = 0                # number of penetrating vertex-instances (summed over frames)
    contact_n: int = 0            # gated contacting vertex-instances (velocity-gated contact loss)
    contact_cm: float = float("nan")  # mean inside-depth of those contacting verts (cm)
    depth_m: float = float("nan")  # mean pelvis camera-frame depth (the z-push target / diagnostic)
    attract_n: int = 0            # near-wall vertex-instances in the attraction band (summed over frames)
    attract_cm: float = float("nan")  # mean outside-gap of those near-wall verts (cm)

    def __str__(self) -> str:
        ini = "n/a" if self.reproj_init_px != self.reproj_init_px else f"{self.reproj_init_px:.2f}px"
        dep = "" if self.depth_m != self.depth_m else f"  depth={self.depth_m:.2f}m"
        pen = "" if self.pen_cm != self.pen_cm else f"  pen={self.pen_cm:.2f}cm/{self.pen_n}v"
        ct = "" if self.contact_cm != self.contact_cm else f"  contact={self.contact_cm:.2f}cm/{self.contact_n}v"
        at = "" if self.attract_cm != self.attract_cm else f"  attract={self.attract_cm:.2f}cm/{self.attract_n}v"
        return (f"reproj_sap={self.reproj_sap_px:.2f}px  reproj_init={ini}  "
                f"chamfer={self.chamfer_m * 100:.2f}cm  "
                f"‖accel‖={self.accel:.4f}  ‖jerk‖={self.jerk:.4f}{dep}{pen}{ct}{at}")


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


def _dof_mask(active: tuple[str, ...], nq: int, device) -> torch.Tensor:
    """``(nq,)`` 0/1 mask selecting the ``q`` DOFs trainable this phase.

    ``q`` packs the root translation (``[:3]``), the root quaternion (``[3:7]``),
    then the parent-local joint quaternions (``[7:]``). Freezing a DOF group is
    done by zeroing its gradient after backward, so only the named groups move.
    """
    m = torch.zeros(nq, device=device)
    if "transl" in active:
        m[0:3] = 1.0
    if "orient" in active:
        m[3:7] = 1.0
    if "joints" in active:
        m[7:] = 1.0
    return m


def _visibility(vertices_sub_det: torch.Tensor, normals_sub: torch.Tensor,
                extrinsics: torch.Tensor) -> torch.Tensor:
    """Front-facing mask ``(T, Ns)`` for already-subsampled vertices."""
    vcam = transform_points(vertices_sub_det, extrinsics)
    ncam = torch.einsum("tij,tvj->tvi", extrinsics[:, :3, :3], normals_sub)
    return front_facing_mask(vcam, ncam)


def _padded_scene_candidates(vertices_sub_det: torch.Tensor, scene_pts: torch.Tensor,
                             scene_normals: torch.Tensor, scene_conf: torch.Tensor,
                             margin: float, max_candidates: int):
    """Per-frame scene candidates near the body, padded to ``(T, Mc, …)`` for the
    chunked-batched :func:`~tools.human_optim.losses.scene_penetration`.

    The scene cloud is static but large (~10⁵ points over the whole route); only
    points near the body can ever be penetrated. Each frame keeps the scene
    points inside its posed-vertex AABB + ``margin`` (a margin covering the body's
    motion over the stage), then — if that still exceeds ``max_candidates`` — the
    ``max_candidates`` nearest to the box centre (the farther ones can't be any
    vertex's nearest neighbour, so this is near-lossless but bounds the padded
    width, hence the per-iter ``cdist`` cost). Built once from the post-placement
    body and reused across a scene phase's iterations. Returns the four padded
    tensors ``(points, normals, confidence, valid)``."""
    pts, nrm, cnf = [], [], []
    for t in range(vertices_sub_det.shape[0]):
        lo = vertices_sub_det[t].amin(0) - margin
        hi = vertices_sub_det[t].amax(0) + margin
        m = ((scene_pts >= lo) & (scene_pts <= hi)).all(dim=-1)
        cp, cn, cc = scene_pts[m], scene_normals[m], scene_conf[m]
        if cp.shape[0] > max_candidates:
            center = (0.5 * (lo + hi))[None]
            keep = torch.cdist(center, cp)[0].topk(max_candidates, largest=False).indices
            cp, cn, cc = cp[keep], cn[keep], cc[keep]
        pts.append(cp)
        nrm.append(cn)
        cnf.append(cc)
    dev, dt = vertices_sub_det.device, vertices_sub_det.dtype
    pad_p, valid = pad_frames(pts, channels=3, device=dev, dtype=dt)
    pad_n, _ = pad_frames(nrm, channels=3, device=dev, dtype=dt)
    pad_c, _ = pad_frames(cnf, channels=None, device=dev, dtype=scene_conf.dtype)
    return pad_p, pad_n, pad_c, valid


@torch.no_grad()
def _metrics(mid, betas, q_world, *, extrinsics, K, depth_pad, depth_valid, target_kp,
            scores, corr, frame_vertex_ids, sub, score_thresh, chunk=16,
            body_idx=None, init_target=None,
            scene_pts=None, scene_normals=None, scene_conf=None, pen=None,
            model=None, v2j_sub=None, fps=30.0, contact=None, attract=None) -> Metrics:
    qn = normalize_q(q_world, mid.num_joints)
    out = mid.forward(betas, qn)
    jw = out.joints_world.tensor()[..., :3]
    vw = out.vertices
    # Sapiens reprojection in raw pixels (un-normalized) over scored keypoints
    body = jw[:, corr.joint_model, :]
    lm = vw[:, frame_vertex_ids[corr.lm_frame], :]
    pred_px = project(transform_points(torch.cat([body, lm], 1), extrinsics), K)
    tgt = target_kp[:, corr.sap, :]
    w = (scores[:, corr.sap] >= score_thresh).float()
    err = (pred_px - tgt).norm(dim=-1)
    reproj_sap = float((w * err).sum() / w.sum().clamp_min(1.0))
    # Init-anchor drift: mean body-joint pixel distance from the converter-pose
    # projection (honest px, not GM). ~0 at init; grows as the 3D terms pull the
    # body off its trusted converter 2D.
    if init_target is not None:
        ip = project(transform_points(jw[:, body_idx, :], extrinsics), K)
        reproj_init = float((ip - init_target).norm(dim=-1).mean())
    else:
        reproj_init = float("nan")
    # chamfer
    n = vertex_normals(vw, mid.faces)
    vismask = _visibility(vw[:, sub], n[:, sub], extrinsics)
    cham = float(chamfer_loss(vw[:, sub], vismask, depth_pad, depth_valid, chunk=chunk))
    # penetration into the scene cloud (diagnostic — same SDF the loss uses, on
    # all subsampled verts; candidates pruned to each frame's body box) + the
    # velocity-gated contact diagnostic (reuses the same candidates)
    pen_cm, pen_n = float("nan"), 0
    contact_n, contact_cm = 0, float("nan")
    attract_n, attract_cm = 0, float("nan")
    if scene_pts is not None:
        cand = _padded_scene_candidates(vw[:, sub], scene_pts, scene_normals, scene_conf,
                                        float(pen["prune_margin_m"]), int(pen["max_candidates"]))
        _, dcm, npen = scene_penetration(
            vw[:, sub], *cand, conf_thresh=float(pen["conf_thresh"]),
            margin_m=float(pen["margin_m"]), max_dist_m=float(pen["max_dist_m"]),
            max_pen_m=float(pen["max_pen_m"]), huber_m=float(pen["huber_m"]), chunk=chunk)
        pen_cm, pen_n = float(dcm), int(npen)
        if attract is not None:
            _, agm, na = scene_attraction(
                vw[:, sub], *cand, conf_thresh=float(attract["conf_thresh"]),
                band_m=float(attract["band_m"]), max_dist_m=float(attract["max_dist_m"]), chunk=chunk)
            attract_n, attract_cm = int(na), float(agm)
        if model is not None and contact is not None:
            speed = joint_local_linear_speed(model, mid, q_world, fps)  # (T, J)
            _, cdm, cn = contact_velocity_loss(
                vw[:, sub], speed[:, v2j_sub], *cand,
                conf_thresh=float(contact["conf_thresh"]), max_dist_m=float(contact["max_dist_m"]),
                max_pen_m=float(contact["max_pen_m"]), contact_band_m=float(contact["contact_band_m"]),
                vel_ref=float(contact["vel_ref"]), vel_exp=float(contact["vel_exp"]), chunk=chunk)
            contact_n, contact_cm = int(cn), float(cdm)
    # mean pelvis camera-frame depth over frames with depth (the z-push diagnostic)
    pelvis_z = transform_points(jw[:, :1, :], extrinsics)[:, 0, 2]   # (T,)
    fhd = depth_valid.any(dim=1)
    depth_m = (float((pelvis_z * fhd).sum() / fhd.sum().clamp_min(1))
               if bool(fhd.any()) else float(pelvis_z.mean()))
    # accel / jerk norms (raw tangent diffs)
    d1 = tangent_difference(qn, mid.num_joints)
    accel = d1[1:] - d1[:-1]
    jerk = accel[1:] - accel[:-1]
    return Metrics(reproj_sap, reproj_init, cham, float(accel.norm()), float(jerk.norm()),
                   pen_cm, pen_n, contact_n, contact_cm, depth_m, attract_n, attract_cm)


def optimize_person(
    mid, betas_mean: torch.Tensor, q_world_init: torch.Tensor, *,
    extrinsics: torch.Tensor, K: torch.Tensor, depth_points: list[torch.Tensor],
    target_kp: torch.Tensor, scores: torch.Tensor, corr, frame_vertex_ids: torch.Tensor,
    sub: torch.Tensor, knobs: dict, phases: list[PhaseSpec],
    scene_pts: torch.Tensor | None = None, scene_normals: torch.Tensor | None = None,
    scene_conf: torch.Tensor | None = None, fps: float = 30.0,
    contact_labels: torch.Tensor | None = None,
    chamfer_weights: torch.Tensor | None = None,
    frame_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Metrics, Metrics]:
    """Optimize one person over ``phases``. Returns ``(q_world_opt, before, after)``.

    Priors (pose / root-orientation) and the init-projection anchor are pinned to
    ``q_world_init`` — the *call's input* pose — and held across every phase, so
    ``root`` and ``kinematics_1`` belong to one call (anchored to the converter
    pose). Each phase supplies its own weight column via :class:`PhaseSpec`; a term
    whose weight is 0 in a phase contributes nothing there.

    ``contact_labels`` (optional ``(T, 52)`` int8 tri-state per SMPLXMid joint:
    ``1`` in-contact, ``0`` no-contact, ``-1`` unlabeled) turns on the
    **kinematics_2** annotation-driven scene terms in the scene phases — a *real*
    attraction pulling the vertices of in-contact joints onto the wall, a *strong*
    repulsion pushing the vertices of no-contact joints off it, and a velocity
    damper on in-contact joints (weights ``ann_*``). It defaults to the plain
    kinematics behaviour (no annotation terms).

    ``frame_valid`` (optional ``(T,)`` bool) excludes untracked frames — whose
    converter pose was held from the nearest valid frame and so is glued to the
    moving camera — from the init-projection anchor and the scene terms (just as
    the zeroed Sapiens scores / emptied depth cloud already exclude them from the
    reprojection / chamfer terms). Defaults to all-valid (a no-op on gap-free clips).

    The body is used at its native (converter) size — the metric geometry is
    already sized to it by the human-anchored ``estimate_scale`` stage upstream.
    """
    T = q_world_init.shape[0]
    betas = betas_mean[None].expand(T, -1).contiguous()

    # Shared (per-clip) knob sections — operational constants, not per-stage weights.
    oc = knobs["optimizer"]
    kp = knobs["keypoints"]
    score_thresh = float(kp["score_thresh"])
    sap_gm_px = float(kp["gm_scale_px"])
    pen_cfg = knobs["penetration"]
    contact_cfg = knobs["contact_velocity"]
    attract_cfg = knobs["contact_attraction"]
    chunk = int(knobs["chamfer"]["chunk"])
    max_cand = int(pen_cfg["max_candidates"])

    # Union over the phases about to run — decides what to SET UP once (the per-iter
    # term inclusion is gated by each phase's own weight below).
    def _wmax(term: str) -> float:
        return max((float(ph.weights.get(term, 0.0)) for ph in phases), default=0.0)

    use_init = _wmax("reproj_init") > 0.0
    init_gm_px = float(knobs["reproj_init"]["gm_scale_px"]) if use_init else 0.0
    use_pen = scene_pts is not None
    use_contact = use_pen and _wmax("contact_velocity") > 0.0
    use_attract = use_pen and _wmax("contact_attraction") > 0.0
    use_ann = contact_labels is not None
    ann_at_cfg = knobs["ann_attraction"] if use_ann else None
    ann_rep_cfg = knobs["ann_repulsion"] if use_ann else None

    depth_pad, depth_valid = pad_frames(depth_points, channels=3,
                                        device=q_world_init.device, dtype=q_world_init.dtype)
    frame_has_depth = depth_valid.any(dim=1)   # (T,) — frames the depth-push acts on

    # `v2j_sub` maps each subsampled vertex to its nearest rest-pose joint — needed
    # by the velocity-gated contact loss AND by the kinematics_2 annotation terms
    # (to inherit each vertex's joint contact label), so build it for either. The
    # BetterRobot kinematic model is built once per person (betas fixed) only for
    # the velocity-gated contact loss's Jacobian speed.
    model = v2j_sub = None
    if use_contact or use_ann:
        with torch.no_grad():
            v_rest, j_rest = mid.forward_shape(betas_mean[None])
        v2j_sub = vertex_to_joint(v_rest[0], j_rest[0])[sub]   # (Ns,) nearest joint per subsampled vert
    if use_contact:
        model = make_smplx_robot_model(mid, betas_mean, device=q_world_init.device,
                                       dtype=q_world_init.dtype)
    incontact_v = nocontact_v = None
    if use_ann:
        vlabel = contact_labels[:, v2j_sub]                   # (T, Ns) tri-state
        incontact_v, nocontact_v = (vlabel == 1), (vlabel == 0)

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

    mk = lambda q: _metrics(
        mid, betas, q, extrinsics=extrinsics, K=K, depth_pad=depth_pad, depth_valid=depth_valid,
        target_kp=target_kp, scores=scores, corr=corr, frame_vertex_ids=frame_vertex_ids,
        sub=sub, score_thresh=score_thresh, chunk=chunk,
        body_idx=body_idx, init_target=init_target,
        scene_pts=scene_pts, scene_normals=scene_normals, scene_conf=scene_conf, pen=pen_cfg,
        model=model, v2j_sub=v2j_sub, fps=fps, contact=contact_cfg,
        attract=attract_cfg if use_attract else None)
    before = mk(q_world_init)

    q_init = q_world_init.detach().clone()   # the anchor (priors + init reprojection)
    q = q_init.clone().requires_grad_(True)
    nq = q.shape[1]

    # ── shared forward + objective ───────────────────────────────────────────
    # One definition of the loss, reused by every phase (Adam and L-BFGS descend
    # exactly the same objective); each call is given the running phase's weights.
    def _build_scene_cand():
        """Per-frame padded scene candidates from the *current* pose, or None.

        Built once at the start of a scene phase: by then the body is seated (the
        root / full phases placed it), so the candidates pruned to each frame's
        body box stay valid for the whole phase — reused every iteration."""
        if not use_pen:
            return None
        with torch.no_grad():
            v0 = mid.forward(betas, normalize_q(q, mid.num_joints)).vertices
            cand = _padded_scene_candidates(v0[:, sub], scene_pts, scene_normals,
                                            scene_conf, float(pen_cfg["prune_margin_m"]), max_cand)
        if frame_valid is not None:      # untracked frames: drop their candidates → all scene terms skip them
            pad_p, pad_n, pad_c, valid = cand
            cand = (pad_p, pad_n, pad_c, valid & frame_valid[:, None])
        return cand

    def _build_vel_gate():
        """Static per-subsampled-vertex joint speed ``(T, Ns)`` from the current
        pose, or None — the velocity input to the contact loss's gate (detached,
        computed once per scene phase; refinement barely changes joint speeds)."""
        if not use_contact:
            return None
        with torch.no_grad():
            speed = joint_local_linear_speed(model, mid, q, fps)   # (T, J)
        return speed[:, v2j_sub]                                   # (T, Ns)

    def _forward_losses(w: dict, scene_cand, vel_gate):
        """Forward the posed mesh and assemble the phase-weighted loss. Returns
        ``(loss, components)`` — ``components`` carries the per-term tensors +
        penetration / contact diagnostics for logging + diagnostics."""
        qn = normalize_q(q, mid.num_joints)
        out = mid.forward(betas, qn)
        jw, vw = out.joints_world.tensor()[..., :3], out.vertices
        with torch.no_grad():
            normals = vertex_normals(vw.detach(), mid.faces)
            vismask = _visibility(vw.detach()[:, sub], normals[:, sub], extrinsics)
        w_init = float(w["reproj_init"])
        w_push = float(w["depth_push"])
        w_attract = float(w["contact_attraction"])
        w_ann_attract = float(w.get("ann_attraction", 0.0))
        w_ann_repel = float(w.get("ann_repulsion", 0.0))
        w_ann_vel = float(w.get("ann_velocity", 0.0))
        L_sap = sapiens_reprojection_gm(jw, vw, extrinsics, K, target_kp, scores, corr,
                                        frame_vertex_ids, score_thresh=score_thresh, gm_scale_px=sap_gm_px)
        L_init = (init_reprojection_gm(jw, body_idx, extrinsics, K, init_target,
                                       gm_scale_px=init_gm_px, frame_mask=frame_valid)
                  if (use_init and w_init > 0.0) else jw.new_zeros(()))
        L_ch = chamfer_loss(vw[:, sub], vismask, depth_pad, depth_valid, chunk=chunk,
                            vertex_weights=chamfer_weights)
        L_push = (depth_push_loss(jw, extrinsics, frame_has_depth)
                  if w_push != 0.0 else jw.new_zeros(()))
        L_pp = pose_prior_loss(qn, q_init, mid.num_joints)
        L_ro = root_orient_prior_loss(qn, q_init)
        L_sm = smoothness_loss(qn, mid.num_joints, accel_root=float(w["accel_root"]),
                               accel_joint=float(w["accel_joint"]), jerk_root=float(w["jerk_root"]),
                               jerk_joint=float(w["jerk_joint"]))
        if scene_cand is not None:
            L_pen, pen_cm, pen_n = scene_penetration(
                vw[:, sub], *scene_cand, conf_thresh=float(pen_cfg["conf_thresh"]),
                margin_m=float(pen_cfg["margin_m"]), max_dist_m=float(pen_cfg["max_dist_m"]),
                max_pen_m=float(pen_cfg["max_pen_m"]), huber_m=float(pen_cfg["huber_m"]), chunk=chunk)
        else:
            L_pen, pen_cm, pen_n = vw.new_zeros(()), float("nan"), 0
        if vel_gate is not None:                       # velocity-gated contact (scene phase, weight>0)
            L_ct, ct_cm, ct_n = contact_velocity_loss(
                vw[:, sub], vel_gate, *scene_cand, conf_thresh=float(contact_cfg["conf_thresh"]),
                max_dist_m=float(contact_cfg["max_dist_m"]), max_pen_m=float(contact_cfg["max_pen_m"]),
                contact_band_m=float(contact_cfg["contact_band_m"]), vel_ref=float(contact_cfg["vel_ref"]),
                vel_exp=float(contact_cfg["vel_exp"]), chunk=chunk)
        else:
            L_ct, ct_cm, ct_n = vw.new_zeros(()), float("nan"), 0
        if scene_cand is not None and use_attract and w_attract > 0.0:  # contact attraction (scene phase)
            L_at, at_cm, at_n = scene_attraction(
                vw[:, sub], *scene_cand, conf_thresh=float(attract_cfg["conf_thresh"]),
                band_m=float(attract_cfg["band_m"]), max_dist_m=float(attract_cfg["max_dist_m"]), chunk=chunk)
        else:
            L_at, at_cm, at_n = vw.new_zeros(()), float("nan"), 0
        # kinematics_2 annotation-driven terms. The attraction (in-contact verts →
        # ONTO the wall) and the strong repulsion (no-contact verts → OFF the wall)
        # both need the scene candidates, so run in the scene phases; the velocity
        # damper (slow in-contact joints) needs only the world joints, so it runs
        # in every phase the labels are supplied.
        L_ann_at = L_ann_rep = vw.new_zeros(())
        ann_at_cm = ann_rep_cm = float("nan"); ann_at_n = ann_rep_n = 0
        if scene_cand is not None and use_ann:
            if w_ann_attract > 0.0:
                L_ann_at, ann_at_cm, ann_at_n = scene_attraction(
                    vw[:, sub], *scene_cand, conf_thresh=float(ann_at_cfg["conf_thresh"]),
                    band_m=float(ann_at_cfg["band_m"]), max_dist_m=float(ann_at_cfg["max_dist_m"]),
                    active_mask=incontact_v, chunk=chunk)
            if w_ann_repel > 0.0:
                L_ann_rep, ann_rep_cm, ann_rep_n = clearance_repulsion(
                    vw[:, sub], *scene_cand, conf_thresh=float(ann_rep_cfg["conf_thresh"]),
                    clearance_m=float(ann_rep_cfg["clearance_m"]), max_dist_m=float(ann_rep_cfg["max_dist_m"]),
                    max_pen_m=float(ann_rep_cfg["max_pen_m"]), huber_m=float(ann_rep_cfg["huber_m"]),
                    active_mask=nocontact_v, chunk=chunk)
        L_ann_vel = vw.new_zeros(()); ann_vel_cms = float("nan"); ann_vel_n = 0
        if use_ann and w_ann_vel > 0.0:
            L_ann_vel, ann_vel_cms, ann_vel_n = joint_velocity_loss(jw, contact_labels, fps=fps)
        loss = (float(w["reproj_sapiens"]) * L_sap + w_init * L_init + float(w["chamfer"]) * L_ch
                + w_push * L_push + float(w["pose_prior"]) * L_pp + float(w["orient_prior"]) * L_ro + L_sm
                + float(w["penetration"]) * L_pen + float(w["contact_velocity"]) * L_ct + w_attract * L_at
                + w_ann_attract * L_ann_at + w_ann_repel * L_ann_rep + w_ann_vel * L_ann_vel)
        return loss, dict(sap=L_sap, init=L_init, ch=L_ch, push=L_push, pp=L_pp, sm=L_sm,
                          pen=L_pen, pen_cm=pen_cm, pen_n=pen_n, has_pen=scene_cand is not None,
                          ct=L_ct, ct_cm=ct_cm, ct_n=ct_n, has_ct=vel_gate is not None,
                          at=L_at, at_cm=at_cm, at_n=at_n, has_at=scene_cand is not None and use_attract,
                          ann_at=L_ann_at, ann_at_cm=ann_at_cm, ann_at_n=ann_at_n,
                          ann_rep=L_ann_rep, ann_rep_cm=ann_rep_cm, ann_rep_n=ann_rep_n,
                          ann_vel=L_ann_vel, ann_vel_cms=ann_vel_cms, ann_vel_n=ann_vel_n,
                          has_ann=use_ann and scene_cand is not None)

    def _log(tag, it, loss, c, w):
        w_pen, w_contact, w_attract = float(w["penetration"]), float(w["contact_velocity"]), float(w["contact_attraction"])
        w_rep_sap, w_rep_init, w_push = float(w["reproj_sapiens"]), float(w["reproj_init"]), float(w["depth_push"])
        w_ann_attract, w_ann_repel, w_ann_vel = float(w.get("ann_attraction", 0.0)), float(w.get("ann_repulsion", 0.0)), float(w.get("ann_velocity", 0.0))
        pen_str = (f" pen={float(c['pen_cm']):.1f}cm/{int(c['pen_n'])}v·{w_pen:g}={w_pen * c['pen'].item():.3f}"
                   if c["has_pen"] else "")
        ct_str = (f" contact={float(c['ct_cm']):.1f}cm/{int(c['ct_n'])}v·{w_contact:g}={w_contact * c['ct'].item():.3f}"
                  if c["has_ct"] else "")
        at_str = (f" attract={float(c['at_cm']):.1f}cm/{int(c['at_n'])}v·{w_attract:g}={w_attract * c['at'].item():.3f}"
                  if c["has_at"] else "")
        ann_str = (f" ann_at={float(c['ann_at_cm']):.1f}cm/{int(c['ann_at_n'])}v·{w_ann_attract:g}={w_ann_attract * c['ann_at'].item():.3f}"
                   f" ann_rep={float(c['ann_rep_cm']):.1f}cm/{int(c['ann_rep_n'])}v·{w_ann_repel:g}={w_ann_repel * c['ann_rep'].item():.3f}"
                   f" ann_vel={float(c['ann_vel_cms']):.1f}cm/s·{w_ann_vel:g}={w_ann_vel * c['ann_vel'].item():.4f}"
                   if c["has_ann"] else "")
        push_str = (f" push·{w_push:g}={w_push * c['push'].item():.3f}" if w_push != 0.0 else "")
        init_str = (f"rep_init={c['init'].item():.4f}·{w_rep_init:g}={w_rep_init * c['init'].item():.2f} "
                    if w_rep_init > 0.0 else "")
        print(f"    [{tag:9s}] it {it:4d}  loss={loss.item():.4f}  "
              f"(rep_sap={c['sap'].item():.4f}·{w_rep_sap:g}={w_rep_sap * c['sap'].item():.2f} "
              f"{init_str}ch={c['ch'].item() * 100:.1f}cm pose={c['pp'].item():.4f} "
              f"smooth={c['sm'].item():.3f}{push_str}{pen_str}{ct_str}{at_str}{ann_str})")

    # ── run the phases in order ──────────────────────────────────────────────
    # Adam (per-coordinate steps) tolerates the very different gradient scales here
    # (metre root translation vs joint quaternions, pixel reproj vs metre chamfer)
    # that would stall L-BFGS cold; the final L-BFGS phase (all DOFs free, strong-
    # Wolfe) then sharpens the already-close fit. A fresh optimizer per phase, so
    # no stale momentum leaks across an unfreeze; DOFs frozen by zeroing the grad.
    for ph in phases:
        scene_cand = _build_scene_cand() if ph.scene else None
        vel_gate = _build_vel_gate() if ph.scene else None
        if ph.optimizer == "lbfgs":
            opt = torch.optim.LBFGS([q], lr=float(oc["lr_lbfgs"]), max_iter=int(ph.iters),
                                    history_size=int(oc["lbfgs_history"]), line_search_fn="strong_wolfe")
            step = [0]

            def closure():
                opt.zero_grad()
                loss, c = _forward_losses(ph.weights, scene_cand, vel_gate)
                loss.backward()
                step[0] += 1
                if step[0] == 1 or step[0] % 5 == 0:
                    _log(ph.name, step[0], loss, c, ph.weights)
                return loss
            if int(ph.iters) > 0:
                opt.step(closure)
            continue
        mask = _dof_mask(ph.dofs, nq, q.device)
        opt = torch.optim.Adam([{"params": [q], "lr": float(oc["lr"])}])
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[ph.iters // 2, 3 * ph.iters // 4], gamma=float(oc["lr_decay"]))
        for it in range(int(ph.iters)):
            opt.zero_grad()
            loss, c = _forward_losses(ph.weights, scene_cand, vel_gate)
            loss.backward()
            q.grad.mul_(mask)  # freeze the DOF groups not active this phase
            opt.step()
            sched.step()
            if it % max(1, ph.iters // 4) == 0 or it == ph.iters - 1:
                _log(ph.name, it, loss, c, ph.weights)

    with torch.no_grad():
        q_opt = normalize_q(q.detach(), mid.num_joints)
    return q_opt, before, mk(q_opt)
