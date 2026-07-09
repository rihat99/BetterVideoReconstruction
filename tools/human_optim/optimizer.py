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
(anchored to the converter pose) rather than running as separate calls.

All posed-mesh inference stays in ``better_human``; this module assembles the
objective and drives ``torch.optim``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .camera import project, transform_points
from .contacts import (
    drift_gated_contact, joint_drift, joint_guarded_min_sdf, scene_sdf, vertex_to_joint,
)
from .keypoints import smooth_tracks
from .losses import (
    attraction_penalty, chamfer_loss, clearance_penalty, init_reprojection_gm,
    joint_velocity_loss, pad_frames, penetration_penalty, pose_prior_loss,
    radial_velocity_loss, root_orient_prior_loss, sapiens_reprojection_gm,
    scene_penetration, scene_signed_distance, smoothness_loss, wall_contact_loss,
)
from .motion import tangent_difference
from .qmath import normalize_q
from .visibility import front_facing_mask, vertex_normals


@dataclass
class Metrics:
    reproj_sap_px: float  # Sapiens Goliath-308 keypoint reprojection (score-weighted)
    reproj_init_px: float  # body-joint drift from the converter-pose projection (init anchor)
    chamfer_m: float
    accel: float
    jerk: float
    snap: float           # ‖snap‖ (4th tangent difference) — the highest-frequency jitter
    pen_cm: float = float("nan")  # mean penetration depth into the scene cloud (nan = no scene)
    pen_n: int = 0                # number of penetrating vertex-instances (summed over frames)
    depth_m: float = float("nan")  # mean pelvis camera-frame depth (diagnostic)
    radial_cms: float = float("nan")  # mean per-joint radial (viewing-ray) speed, cm/s

    def __str__(self) -> str:
        ini = "n/a" if self.reproj_init_px != self.reproj_init_px else f"{self.reproj_init_px:.2f}px"
        dep = "" if self.depth_m != self.depth_m else f"  depth={self.depth_m:.2f}m"
        pen = "" if self.pen_cm != self.pen_cm else f"  pen={self.pen_cm:.2f}cm/{self.pen_n}v"
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
def _median_closest_ee(ee_world: torch.Tensor, scene_pts: torch.Tensor,
                       frame_valid: torch.Tensor | None, chunk: int) -> float:
    """Median over valid frames of the closest EE joint's distance to the scene (m).

    The float-gate signal: a grossly floating body has all its hands/feet tens of cm from
    the wall every frame, a seated / dynamic body has at least one near it most frames.
    """
    T = ee_world.shape[0]
    mins = []
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        d = torch.cdist(ee_world[s:e].reshape(-1, 3), scene_pts)     # ((e-s)·Nee, Ms)
        mins.append(d.min(1).values.reshape(e - s, -1).min(1).values)  # (e-s,) closest EE / frame
    m = torch.cat(mins)
    if frame_valid is not None:
        m = m[frame_valid]
    return float(m.median()) if m.numel() else float("inf")


@torch.no_grad()
def _metrics(mid, betas, q_world, *, extrinsics, K, depth_pad, depth_valid, target_kp,
            scores, corr, frame_vertex_ids, sub, score_thresh, chunk=16,
            body_idx=None, init_target=None,
            scene_pts=None, scene_normals=None, scene_conf=None, pen=None,
            cam_centers=None, fps=30.0, frame_valid=None) -> Metrics:
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
    # all subsampled verts; candidates pruned to each frame's body box)
    pen_cm, pen_n = float("nan"), 0
    if scene_pts is not None:
        cand = _padded_scene_candidates(vw[:, sub], scene_pts, scene_normals, scene_conf,
                                        float(pen["prune_margin_m"]), int(pen["max_candidates"]))
        _, dcm, npen = scene_penetration(
            vw[:, sub], *cand, conf_thresh=float(pen["conf_thresh"]),
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
    chamfer_weights: torch.Tensor | None = None,
    frame_valid: torch.Tensor | None = None,
    cam_centers: torch.Tensor | None = None,
    auto_contact: dict | None = None,
) -> tuple[torch.Tensor, Metrics, Metrics, dict]:
    """Optimize one person over ``phases``. Returns ``(q_world_opt, before, after, aux)``.

    ``aux`` carries the detection-time signals the caller persists into the body
    npz: ``drift_pre (T, num_joints)`` — ``joint_drift`` of the
    refine-start (pre-attraction) body, the honest pre-enforcement stillness measurement —
    and ``auto_labels (T, num_joints)`` int8 — the frozen labels the attraction used. Both
    are ``None`` when auto detection did not run (manual ``kinematics_2``, no scene, or a
    zero ``ann_attraction`` weight); the caller then writes NaN / ``-1`` for that person.

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
    kinematics behaviour (no annotation terms). ``contact_conf`` (optional
    ``(T, 52)`` float in [0,1]) scales the attraction's vertex weights
    and the velocity damper's per-joint contribution by each joint's label
    confidence, so an uncertain auto-label pulls proportionally less; ``None`` keeps
    every labelled joint at weight 1.

    ``auto_contact`` (optional dict) enables **kinematics_1** auto-contact mode: when
    ``contact_labels`` is not supplied and the ``ann_attraction`` weight is > 0, labels
    are detected **once** from the detached pose at the first scene phase (the contact
    readout at the ``[auto_contact]`` operating point) and **frozen** — positive-only
    (``1`` on detected group joints, ``-1`` elsewhere, never ``0``), so only the
    attraction + velocity terms fire, never the clearance repulsion. ``cam_centers``
    (``(T, 3)`` world camera centers) drives the radial-velocity damper + its
    diagnostic; both default off.

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
    # kinematics_1 auto-contact mode: detect labels once at the first scene phase (below) then
    # freeze. Guarded on the ann_attraction weight-max, so a 0-weight column never detects
    # (keeps parity with HEAD). Auto attraction reads its band from [auto_contact]; the manual
    # (kinematics_2) path keeps [ann_attraction].
    use_auto = auto_contact is not None and not use_ann and _wmax("ann_attraction") > 0.0
    ann_at_cfg = auto_contact if use_auto else (knobs["ann_attraction"] if use_ann else None)
    ann_rep_cfg = knobs["ann_repulsion"] if use_ann else None   # clearance repulsion is manual-only

    depth_pad, depth_valid = pad_frames(depth_points, channels=3,
                                        device=q_world_init.device, dtype=q_world_init.dtype)

    # `v2j_full` / `v2j_sub` map each (subsampled) vertex to its nearest rest-pose joint. The
    # subsampled map inherits joint contact labels onto vertices (attraction / repulsion masks);
    # the full map lifts the auto-detection per-vertex SDF to per-joint. Built once from the
    # shaped rest mesh (label-independent).
    v2j_full = v2j_sub = None
    if use_ann or use_auto:
        with torch.no_grad():
            v_rest, j_rest = mid.forward_shape(betas_mean[None])
        v2j_full = vertex_to_joint(v_rest[0], j_rest[0])      # (V,) nearest joint per vertex
        v2j_sub = v2j_full[sub]                               # (Ns,) subsampled
    auto_group_mask = None
    if use_auto:
        auto_group_mask = torch.zeros(mid.num_joints, dtype=torch.bool, device=q_world_init.device)
        auto_group_mask[auto_contact["group_joints"]] = True   # only these joints may be labeled 1

    # per-vertex label-confidence weight (kinematics_2-on-auto). Scales the attraction; the
    # velocity damper takes the per-joint conf directly. None ⇒ every active vertex weighted 1.
    conf_v = None
    if contact_conf is not None and v2j_sub is not None:
        conf_v = contact_conf[:, v2j_sub].clamp(0.0, 1.0)      # (T, Ns)

    incontact_v = nocontact_v = None
    def _bind_labels(labels: torch.Tensor) -> None:
        """Set ``contact_labels`` + the per-subsampled-vertex in/no-contact masks (from v2j_sub)."""
        nonlocal contact_labels, incontact_v, nocontact_v
        contact_labels = labels
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

    # Round-10 T3-B wall-contact prior + a rigid seat phase, gated to floating bodies. Off (parity)
    # unless [loss_weights.scene].wall_contact > 0.
    use_wall = _wmax("wall_contact") > 0.0 and use_pen
    wall_cfg = knobs.get("wall_contact", {}) if use_wall else None
    ee_idx = None
    if use_wall:
        _ee_names = ("left_wrist", "right_wrist", "left_foot", "right_foot")
        ee_idx = torch.tensor([jnames.index(n) for n in _ee_names],
                              dtype=torch.long, device=q_world_init.device)
        # Per-body float gate: seat/wall the body ONLY if it is grossly floating (closest EE
        # median off the wall > gate). A seated / dynamic body (labeled clips) is left exactly
        # as baseline — so the wall pull cannot manufacture false contacts on it (round-10).
        gate_m = float(wall_cfg.get("float_gate_m", 0.0))
        if gate_m > 0.0:
            with torch.no_grad():
                ee0 = mid.forward(betas, normalize_q(q_world_init, mid.num_joints)
                                  ).joints_world.tensor()[..., :3][:, ee_idx]
            closest = _median_closest_ee(ee0, scene_pts, frame_valid, chunk)
            if closest < gate_m:
                print(f"    [wall] body seated (closest-EE median {closest * 100:.1f}cm < gate "
                      f"{gate_m * 100:.0f}cm) → seat+wall OFF (baseline fit)")
                use_wall = False
                phases = [ph for ph in phases if ph.name != "seat"]
            else:
                print(f"    [wall] body floating (closest-EE median {closest * 100:.1f}cm ≥ gate "
                      f"{gate_m * 100:.0f}cm) → seat+wall ON")

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

    def _build_wall_cand():
        """Per-frame padded scene points near the body, with a WIDE margin — the wall a
        floating body must reach sits tens of cm away, beyond the penetration prune box.
        Returns ``(points, valid)`` or None. Built once at the scene phase like the
        penetration candidates."""
        if not use_wall:
            return None
        with torch.no_grad():
            v0 = mid.forward(betas, normalize_q(q, mid.num_joints)).vertices
            pad_p, _pad_n, _pad_c, valid = _padded_scene_candidates(
                v0[:, sub], scene_pts, scene_normals, scene_conf,
                float(wall_cfg["prune_margin_m"]), int(wall_cfg["max_candidates"]))
        if frame_valid is not None:
            valid = valid & frame_valid[:, None]
        return pad_p, valid

    def _detect_auto_labels() -> tuple[torch.Tensor, torch.Tensor]:
        """Detect frozen kinematics_1 contact labels once from the current detached pose.

        Runs the contact readout at the ``[auto_contact]`` operating point (scene SDF →
        guarded per-joint min → drift/stillness gate → hysteresis), keeps only joints in the
        configured groups on valid frames. Returns ``(labels, drift)``: ``labels (T, num_joints)``
        int8 with ``1`` on detected group joints / ``-1`` everywhere else — never ``0``, so
        ``nocontact_v`` stays empty and the clearance repulsion can never fire in auto mode; and
        ``drift (T, num_joints)`` — the refine-start net window drift the detection gated on, kept
        so the caller can persist it as the honest pre-enforcement stillness measurement. No
        gradient path (``no_grad`` forward from the detached leaf), run once per stage — so
        contact detection cannot feed back into the fit."""
        ac = auto_contact
        with torch.no_grad():
            o = mid.forward(betas, normalize_q(q, mid.num_joints))
            jw_full = o.joints_world.tensor()[..., :3]
            vw_full = o.vertices
        sdf, dmin, conf = scene_sdf(vw_full, scene_pts, scene_normals, scene_conf,
                                    prune_margin_m=float(ac["prune_margin_m"]))
        gsdf = joint_guarded_min_sdf(sdf, dmin, conf, v2j_full, mid.num_joints,
                                     max_dist_m=float(ac["max_dist_m"]), conf_thresh=float(ac["conf_thresh"]))
        drift = joint_drift(jw_full, fps=fps, window_s=float(ac["window_s"]))
        jc = drift_gated_contact(gsdf, drift, close_thresh_m=float(ac["detect_thresh_m"]),
                                 exit_thresh_m=float(ac["detect_exit_m"]), min_frames=int(ac["min_frames"]),
                                 still_m=float(ac["still_m"]))          # (T, J) bool
        keep = jc & auto_group_mask[None, :]                           # only configured group joints
        if frame_valid is not None:
            keep = keep & frame_valid[:, None]
        lab = torch.full((jc.shape[0], mid.num_joints), -1, dtype=torch.int8, device=q.device)
        lab[keep] = 1
        return lab, drift

    def _forward_losses(w: dict, scene_cand, wall_cand=None):
        """Forward the posed mesh and assemble the phase-weighted loss. Returns
        ``(loss, components)`` — ``components`` carries the per-term tensors +
        penetration diagnostics for logging + diagnostics."""
        qn = normalize_q(q, mid.num_joints)
        out = mid.forward(betas, qn)
        jw, vw = out.joints_world.tensor()[..., :3], out.vertices
        with torch.no_grad():
            normals = vertex_normals(vw.detach(), mid.faces)
            vismask = _visibility(vw.detach()[:, sub], normals[:, sub], extrinsics)
        w_init = float(w["reproj_init"])
        w_ann_attract = float(w.get("ann_attraction", 0.0))
        w_ann_repel = float(w.get("ann_repulsion", 0.0))
        w_ann_vel = float(w.get("ann_velocity", 0.0))
        w_radial = float(w.get("radial_velocity", 0.0))
        have_labels = contact_labels is not None      # manual (kinematics_2) or auto-detected (kinematics_1)
        L_sap = sapiens_reprojection_gm(jw, vw, extrinsics, K, target_kp_loss, scores, corr,
                                        frame_vertex_ids, score_thresh=score_thresh, gm_scale_px=sap_gm_px)
        L_init = (init_reprojection_gm(jw, body_idx, extrinsics, K, init_target_loss,
                                       gm_scale_px=init_gm_px, frame_mask=frame_valid)
                  if (use_init and w_init > 0.0) else jw.new_zeros(()))
        L_ch = chamfer_loss(vw[:, sub], vismask, depth_pad, depth_valid, chunk=chunk,
                            vertex_weights=chamfer_weights)
        L_pp = pose_prior_loss(qn, q_init, mid.num_joints)
        L_ro = root_orient_prior_loss(qn, q_init)
        L_sm = smoothness_loss(qn, mid.num_joints, accel_root=float(w["accel_root"]),
                               accel_joint=float(w["accel_joint"]), jerk_root=float(w["jerk_root"]),
                               jerk_joint=float(w["jerk_joint"]), snap_root=float(w.get("snap_root", 0.0)),
                               snap_joint=float(w.get("snap_joint", 0.0)))
        L_radial = (radial_velocity_loss(jw, cam_centers, frame_valid=frame_valid)
                    if (w_radial > 0.0 and cam_centers is not None) else jw.new_zeros(()))
        # Scene terms share ONE detached nearest-neighbour pass (scene_signed_distance):
        # penetration + contact attraction + clearance repulsion are penalty heads on it, so
        # the default (penetration + attraction) costs one cdist, not two.
        L_pen, pen_cm, pen_n = vw.new_zeros(()), float("nan"), 0
        L_ann_at = L_ann_rep = vw.new_zeros(())
        ann_at_cm = ann_rep_cm = float("nan"); ann_at_n = ann_rep_n = 0
        if scene_cand is not None:
            sig = scene_signed_distance(vw[:, sub], *scene_cand, chunk=chunk)
            L_pen, pen_cm, pen_n = penetration_penalty(
                *sig, conf_thresh=float(pen_cfg["conf_thresh"]), margin_m=float(pen_cfg["margin_m"]),
                max_dist_m=float(pen_cfg["max_dist_m"]), max_pen_m=float(pen_cfg["max_pen_m"]),
                huber_m=float(pen_cfg["huber_m"]))
            # Contact attraction pulls in-contact-joint verts ONTO the wall (kinematics_1 auto
            # labels read their band from [auto_contact]; kinematics_2 from [ann_attraction]).
            # Clearance repulsion pushes no-contact-joint verts off — kinematics_2 only, since
            # auto labels carry no 0s (nocontact_v is empty).
            if have_labels:
                if w_ann_attract > 0.0:
                    L_ann_at, ann_at_cm, ann_at_n = attraction_penalty(
                        *sig, conf_thresh=float(ann_at_cfg["conf_thresh"]),
                        band_m=float(ann_at_cfg["band_m"]), max_dist_m=float(ann_at_cfg["max_dist_m"]),
                        active_mask=incontact_v, active_weight=conf_v)
                if w_ann_repel > 0.0 and ann_rep_cfg is not None:   # manual-only; auto has no ann_rep_cfg
                    L_ann_rep, ann_rep_cm, ann_rep_n = clearance_penalty(
                        *sig, conf_thresh=float(ann_rep_cfg["conf_thresh"]),
                        clearance_m=float(ann_rep_cfg["clearance_m"]), max_dist_m=float(ann_rep_cfg["max_dist_m"]),
                        max_pen_m=float(ann_rep_cfg["max_pen_m"]), huber_m=float(ann_rep_cfg["huber_m"]),
                        active_mask=nocontact_v)
        # velocity damper (slow in-contact joints) needs only the world joints, so it runs in
        # every phase the labels are present (refine + lbfgs for auto, all phases for manual).
        L_ann_vel = vw.new_zeros(()); ann_vel_cms = float("nan"); ann_vel_n = 0
        if have_labels and w_ann_vel > 0.0:
            L_ann_vel, ann_vel_cms, ann_vel_n = joint_velocity_loss(jw, contact_labels, fps=fps,
                                                                    joint_weight=contact_conf)
        # T3-B wall-contact prior: pull the closest EE joint onto the wall (refine phase).
        w_wall = float(w.get("wall_contact", 0.0))
        L_wall = vw.new_zeros(()); wall_cm = float("nan"); wall_n = 0
        if wall_cand is not None and w_wall > 0.0:
            L_wall, wall_cm, wall_n = wall_contact_loss(
                jw[:, ee_idx], wall_cand[0], wall_cand[1], margin_m=float(wall_cfg["margin_m"]),
                temp_m=float(wall_cfg["temp_m"]), gm_scale_m=float(wall_cfg["gm_scale_m"]),
                chunk=chunk, frame_mask=frame_valid)
        loss = (float(w["reproj_sapiens"]) * L_sap + w_init * L_init + float(w["chamfer"]) * L_ch
                + float(w["pose_prior"]) * L_pp + float(w["orient_prior"]) * L_ro + L_sm
                + float(w["penetration"]) * L_pen + w_radial * L_radial
                + w_ann_attract * L_ann_at + w_ann_repel * L_ann_rep + w_ann_vel * L_ann_vel
                + w_wall * L_wall)
        return loss, dict(sap=L_sap, init=L_init, ch=L_ch, pp=L_pp, sm=L_sm,
                          pen=L_pen, pen_cm=pen_cm, pen_n=pen_n, has_pen=scene_cand is not None,
                          radial=L_radial, radial_w=w_radial,
                          ann_at=L_ann_at, ann_at_cm=ann_at_cm, ann_at_n=ann_at_n,
                          ann_rep=L_ann_rep, ann_rep_cm=ann_rep_cm, ann_rep_n=ann_rep_n,
                          ann_vel=L_ann_vel, ann_vel_cms=ann_vel_cms, ann_vel_n=ann_vel_n,
                          has_ann=have_labels and scene_cand is not None,
                          wall=L_wall, wall_cm=wall_cm, wall_n=wall_n, wall_w=w_wall)

    def _log(tag, it, loss, c, w):
        w_pen = float(w["penetration"])
        w_rep_sap, w_rep_init = float(w["reproj_sapiens"]), float(w["reproj_init"])
        w_ann_attract, w_ann_repel, w_ann_vel = float(w.get("ann_attraction", 0.0)), float(w.get("ann_repulsion", 0.0)), float(w.get("ann_velocity", 0.0))
        pen_str = (f" pen={float(c['pen_cm']):.1f}cm/{int(c['pen_n'])}v·{w_pen:g}={w_pen * c['pen'].item():.3f}"
                   if c["has_pen"] else "")
        ann_str = (f" ann_at={float(c['ann_at_cm']):.1f}cm/{int(c['ann_at_n'])}v·{w_ann_attract:g}={w_ann_attract * c['ann_at'].item():.3f}"
                   f" ann_rep={float(c['ann_rep_cm']):.1f}cm/{int(c['ann_rep_n'])}v·{w_ann_repel:g}={w_ann_repel * c['ann_rep'].item():.3f}"
                   f" ann_vel={float(c['ann_vel_cms']):.1f}cm/s·{w_ann_vel:g}={w_ann_vel * c['ann_vel'].item():.4f}"
                   if c["has_ann"] else "")
        init_str = (f"rep_init={c['init'].item():.4f}·{w_rep_init:g}={w_rep_init * c['init'].item():.2f} "
                    if w_rep_init > 0.0 else "")
        w_rad = float(c.get("radial_w", 0.0))
        rad_str = (f" rad·{w_rad:g}={w_rad * c['radial'].item():.3f}" if w_rad > 0.0 else "")
        w_wall = float(c.get("wall_w", 0.0))
        wall_str = (f" wall={float(c['wall_cm']):.1f}cm/{int(c['wall_n'])}f·{w_wall:g}={w_wall * c['wall'].item():.3f}"
                    if w_wall > 0.0 else "")
        print(f"    [{tag:9s}] it {it:4d}  loss={loss.item():.4f}  "
              f"(rep_sap={c['sap'].item():.4f}·{w_rep_sap:g}={w_rep_sap * c['sap'].item():.2f} "
              f"{init_str}ch={c['ch'].item() * 100:.1f}cm pose={c['pp'].item():.4f} "
              f"smooth={c['sm'].item():.3f}{pen_str}{rad_str}{ann_str}{wall_str})")

    # ── run the phases in order ──────────────────────────────────────────────
    # Adam (per-coordinate steps) tolerates the very different gradient scales here
    # (metre root translation vs joint quaternions, pixel reproj vs metre chamfer)
    # that would stall L-BFGS cold; the final L-BFGS phase (all DOFs free, strong-
    # Wolfe) then sharpens the already-close fit. A fresh optimizer per phase, so
    # no stale momentum leaks across an unfreeze; DOFs frozen by zeroing the grad.
    auto_drift_pre = None    # (T, num_joints) refine-start drift, persisted by the caller
    for ph in phases:
        scene_cand = _build_scene_cand() if ph.scene else None
        wall_cand = _build_wall_cand() if ph.scene else None
        if use_auto and contact_labels is None and scene_cand is not None:
            lab, auto_drift_pre = _detect_auto_labels()   # detect once at the first scene phase, then freeze
            _bind_labels(lab)
            per_g = " ".join(f"{g}={int((lab[:, m] == 1).any(1).sum())}"
                             for g, m in auto_contact["group_members"].items() if m)
            print(f"    [auto-contact] frozen labels @ {ph.name}: "
                  f"{int((lab == 1).any(1).sum())}/{lab.shape[0]} frames w/ contact  ({per_g})")
        if ph.optimizer == "lbfgs":
            opt = torch.optim.LBFGS([q], lr=float(oc["lr_lbfgs"]), max_iter=int(ph.iters),
                                    history_size=int(oc["lbfgs_history"]), line_search_fn="strong_wolfe")
            step = [0]

            def closure():
                opt.zero_grad()
                loss, c = _forward_losses(ph.weights, scene_cand, wall_cand)
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
            loss, c = _forward_losses(ph.weights, scene_cand, wall_cand)
            loss.backward()
            q.grad.mul_(mask)  # freeze the DOF groups not active this phase
            opt.step()
            sched.step()
            if it % max(1, ph.iters // 4) == 0 or it == ph.iters - 1:
                _log(ph.name, it, loss, c, ph.weights)

    with torch.no_grad():
        q_opt = normalize_q(q.detach(), mid.num_joints)
    aux = {"drift_pre": auto_drift_pre,                            # (T, J) refine-start drift, or None
           "auto_labels": contact_labels if use_auto else None}   # (T, J) int8 frozen labels, or None
    return q_opt, before, mk(q_opt), aux
