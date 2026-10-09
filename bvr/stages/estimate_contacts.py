"""Scene-contact estimation on the contact frames of the optimized SMPL-X bodies.

Reads a refined world-frame SMPL-X trajectory (``human_optim/kinematic_1.npz``) and
the static scene cloud (``geometry/scene.npz``) and answers one question for every
tracked person, every video frame and each of the **contact frames** — the named
mesh points (toes, heels, knees, palms, fingertips) baked onto the body, each
carrying its own outward surface normal: *is this point touching the scene?*

Two readouts answer it, and the stage unions them.

**Geometry** — pose each person per frame (re-applying the stored global scale about
the pelvis) and read the contact-frame world poses straight out of forward kinematics;
for each one find the nearest scene point; then run the shared detector
(:func:`~bvr.human_optim.contacts.detect_frame_contacts`): near the wall AND still,
with one exception — a step pressing into the surface counts however fast it moves,
*if* the force solve confirms it carries load (``[contact_force].confirm_bw``) or was
muted there (``mute_res_bw``) — with hysteresis, a minimum duration and a span merge
behind a re-grip guard (bridged frames flagged in ``merged``). Knee frames are dropped
from this arm: a knee rides near the wall for most of a climb, and its geometric
labels measured ~15% precise on the annotated corpus.

**Force** — freeze the pose and ask what force is unexplained, then solve which of the
contact frames could have supplied it (:mod:`bvr.human_optim.forces`). A frame that has
to carry real load is in contact whatever its distance says, which is how a contact on a
badly reconstructed body is recovered at all. See ``docs/force_contacts.md``.

**The floor gate** (``[contacts].floor_gate_m``, 0 = off) crosses both arms: where the
gravity stage accepted a ground plane, a foot frame higher than that above it cannot carry
a label, however near a scene point it is and however much load the solve routes through
it. Only the foot frames are gated — everything else touches things that are not the floor.

The union is then debounced once (a contact lasts at least ``min_frames``, and so does a
release) and each foot is given a single toe patch per contact — whichever side of the
toe the force solve loaded (the sole-side frame standing on the hold, ``toe_above``
pressed against one), once for the whole contact rather than per step.

Stillness reads the body's persisted pre-enforcement drift (``drift_pre``) — motion
measured before any contact attraction acted. The force solve requires it and will not
re-measure drift on an enforced body.

Renders ``contacts_<tag>_overlay.mp4``: the projected contact points, **red** in
contact / **green** otherwise.

Run from CLI::

    .venv/bin/python bvr/stages/estimate_contacts.py \\
        --source video.mp4 --predictions out/video/ --config configs/human_optimization.toml

The frame labels are folded onto **report joints** (each frame's parent joint; every
hand frame reports at its wrist) for downstream per-joint consumers, and every folded
label gets a **confidence**: the fitted probability that it agrees with a human
annotator, from the joint, its drift and its distance to the next label change (see
``docs/contact_confidence.md``). The per-frame confidence is its report joint's value.

Reads:  human_optim/kinematic_1.npz, geometry/scene.npz, geometry/transform.npz,
        geometry/human.npz (the depth-smear quality metric; optional)
Writes: human_optim/contacts_1.npz          (frame_contact (P,N,F) bool + the signed /
                                             Euclidean / normal / drift signals behind it
                                             + the joint_contact fold and its
                                             joint_label_confidence / frame_label_confidence)
        human_optim/contacts_1_overlay.mp4  ([contacts].overlay; default on)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import argparse
import json

import numpy as np
import torch

from bvr import determinism
from bvr.paths import REPO_ROOT
from bvr.body import frame_normals, load_body
from bvr.config import load_config
from bvr.geometry.normals import load_scene_cloud
from bvr.geometry.quality import cam_motion_ratio, quality_flag, smear_stats
from bvr.npz_io import atomic_savez
from bvr.human_optim import (
    DETECT_MIN_FRAMES, detect_frame_contacts, merge_contact_spans,
    min_span_frames,
    project, transform_points, world_from_cam,
)
from bvr.human_optim import paths
from bvr.human_optim.contacts import FOOT_FRAMES
from bvr.human_optim.confidence import (
    CONTACT_CONFIDENCE_METHOD, confidence_coefficients, fold_contacts, fold_drift,
    frame_report_joints, label_confidence,
)
from bvr.human_optim.forces import (
    NO_SCENE_M,
    ForceKnobs, close_gaps, far_still_veto, force_labels, min_span, resolve_toe_conflicts,
    solve_mute, solve_person_forces,
)
from bvr.human_optim.qmath import normalize_q

_DEFAULT_CONFIG = REPO_ROOT / "configs" / "human_optimization.toml"

#: Bumped from 1 (the per-joint era) — the label axis is now the contact frames. The
#: force witness (method "frames-v2") only ADDS fields and changes no shape, so readers of
#: schema 2 keep working; ``contact_label_method`` is what distinguishes the two.
CONTACT_LABEL_SCHEMA = 2

CONTACT_LABEL_METHOD = "frames-v4"   # v1 = geometry only; v2 adds the force witness;
                                     # v3 debounce + force hysteresis; v4 the toe rule reads
                                     # the solved load allocation (v3 read its vertical sign,
                                     # which renamed every bat hang to a stance)


def _load_config(path: Path) -> dict:
    """Operational knobs for the contact stage (no loss weights here)."""
    cfg = load_config(path) if path.exists() else {}
    c = cfg.setdefault("contacts", {})
    c.setdefault("close_thresh_m", 0.03)   # enter: "just outside" distance counted as contact
    c.setdefault("exit_thresh_m", 0.035)   # stay: looser hysteresis exit (kills boundary flicker)
    c.setdefault("min_span_s", 0.1)        # drop contact spans shorter than this (seconds; frames = ceil(s*fps))
    c.setdefault("max_dist_m", 0.10)       # far guard (a match behind an occluder is not a surface)
    c.setdefault("conf_thresh", 0.1)       # residual absolute floor (VGGT conf ≥~1 ⇒ non-binding)
    c.setdefault("conf_drop_q", 0.1)       # quantile confidence gate: drop the lowest scene voxels
    c.setdefault("normal_k", 16)           # kNN for PCA scene-normal estimation
    c.setdefault("prune_margin_m", 0.10)   # per-frame scene-AABB margin for the nearest-neighbour
    c.setdefault("merge_dist_m", 0.03)     # span-merge: bridge a gap the point barely moved across
    c.setdefault("merge_max_gap_s", 2.0)   # …only gaps shorter than this
    c.setdefault("merge_sdf_m", 0.08)      # …re-grip guard: it stayed this near the wall throughout
    c.setdefault("floor_gate_m", 0.0)      # a FOOT frame may only be labelled this high above the
                                           # fitted ground plane; 0 = off (every other config)
    c.setdefault("overlay", True)          # render the contact overlay
    if not bool(c.get("use_scene", True)):
        raise SystemExit("[estimate_contacts] use_scene=false is not supported by the contact-frame "
                         "detector — contact is measured against the scene cloud only")
    f = cfg.setdefault("contact_force", {})
    f.setdefault("theta_bw", 0.02)          # a frame carrying this fraction of body weight is loaded
    f.setdefault("theta_hold_bw", 0.01)     # …and stays loaded until it drops below this (hysteresis)
    f.setdefault("gap_dstar_m", 0.05)       # gap price: force this far from the scene is expensive
    f.setdefault("stillness_vstar_mps", 0.2)  # stillness price: force on a limb this fast is expensive
    f.setdefault("far_still_x", 4.0)        # far force labels need stillness at this multiple of still_m; 0 = off
    f.setdefault("confirm_bw", 0.02)        # load the PENETRATION EXCEPTION must carry where the solve spoke; 0 = off
    f.setdefault("mute_res_bw", 0.5)        # …unless the solve left this much residual unexplained
    g = cfg.setdefault("contact_gate", {})
    g.setdefault("window_s", 0.35)         # drift window (s) for the stillness gate
    g.setdefault("still_m", 0.03)          # net drift (m) at which stillness = 0.5
    g.setdefault("still_override_m", -1.0)  # closeness that outranks stillness; -1 = off
    q = cfg.setdefault("quality", {})
    q.setdefault("cam_depth_max", 0.4)     # camera flag: cam-travel/scene-depth above this ...
    q.setdefault("res_before_bw_max", 2.5)  # ... AND pre-force residual above this (both witnesses)
    q.setdefault("smear_med_max", 2.0)     # depth flag: median along-ray smear above this
    # Label confidence: the fitted logistic coefficients. There are no defaults — a
    # config without them has never been calibrated, and guessing numbers would be worse
    # than saying so.
    if "contact_confidence" not in cfg:
        raise SystemExit(f"[estimate_contacts] {path} has no [contact_confidence] section — "
                         "run datasets/climbingvideos/calibrate_contact_confidence.py --write on it")
    cfg["contact_confidence"] = confidence_coefficients(cfg["contact_confidence"])
    cfg.setdefault("runtime", {}).setdefault("device", "cuda")
    return cfg


def _sha256(path: Path) -> str:
    with path.open("rb") as source_file:
        return hashlib.file_digest(source_file, "sha256").hexdigest()


@torch.no_grad()
def _pose(body, betas: torch.Tensor, q: torch.Tensor, scale: float,
          *, chunk: int = 64) -> torch.Tensor:
    """Pose all frames → contact-frame world poses ``(N, F, 7)``.

    Re-applies the per-person global ``scale`` about the pelvis (joint 0), the same
    operation the kinematic optimizer and viewers use, so the contact detector sees
    the body at its stored metric size. A uniform scale about a point moves
    positions only — the frame rotations, and therefore the outward normals, are
    untouched.
    """
    num_frames = q.shape[0]
    posed = []
    for start in range(0, num_frames, chunk):
        end = min(start + chunk, num_frames)
        out = body.forward(betas[None].expand(end - start, -1),
                           normalize_q(q[start:end], body.num_joints))
        frames = out.frames_world.clone()
        root = out.joints[:, :1, :]
        frames[..., :3] = root + scale * (frames[..., :3] - root)
        posed.append(frames)
    return torch.cat(posed, 0)


def _drift_pre(kin: dict, person: int, shape, device: str) -> torch.Tensor:
    """The persisted pre-enforcement drift ``(N, F)`` for one person, or a hard error.

    The stillness witnesses price force / gate far labels by how still a frame is. If
    that stillness were measured on the *enforced* body, the optimizer's own contact
    damping would be voting for its own labels, so there is deliberately no fallback
    here: a body without ``drift_pre`` cannot have a contact readout. NaN entries are
    legitimate — an untracked window (or a person detection never ran on) carries no
    stillness evidence, and every consumer treats NaN drift as "not still".
    """
    if "drift_pre" not in kin:
        raise SystemExit("[estimate_contacts] kinematic_1.npz has no drift_pre — the force "
                         "witness needs pre-enforcement stillness and will not re-measure "
                         "it on the enforced body; re-run human_optimization")
    drift = torch.as_tensor(np.asarray(kin["drift_pre"])[person],
                            dtype=torch.float32, device=device)
    if tuple(drift.shape) != tuple(shape):
        raise SystemExit("[estimate_contacts] drift_pre has the wrong shape; "
                         "refusing to guess")
    return drift


def _finite_cm(values: torch.Tensor) -> np.ndarray:
    """Metres → centimetres as float32, with the ``±inf`` sentinels turned into ``NaN``."""
    out = (values * 100.0).cpu().numpy().astype(np.float32)
    return np.where(np.isfinite(out), out, np.nan).astype(np.float32)


def floor_gate(points: torch.Tensor, centroid: torch.Tensor, up: torch.Tensor,
               foot_columns: list[int], gate_m: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``(height (N, F), allowed (N, F))`` for the floor gate.

    ``height`` is each contact frame's signed distance above the fitted ground plane,
    measured along ``up`` from ``centroid``. ``allowed`` says which frames may carry a
    label: a FOOT frame only within ``gate_m`` of that plane, everything else always —
    a hand or a sit bone touches things that are not the floor.
    """
    height = ((points - centroid) * up).sum(-1)
    allowed = torch.ones_like(height, dtype=torch.bool)
    allowed[:, foot_columns] = height[:, foot_columns] <= gate_m
    return height, allowed


def _count_spans(mask: np.ndarray) -> int:
    """Number of contiguous ``True`` runs in a 1-D bool array."""
    return int(np.count_nonzero(mask[1:] & ~mask[:-1]) + (1 if mask.size and mask[0] else 0))


def run(source: str | Path, predictions_dir: str | Path, config_path: str | Path, *,
        overlay: bool = True, bodies: str = "kinematic_1") -> Path:
    cfg = _load_config(Path(config_path))
    device = cfg["runtime"]["device"]
    cc = cfg["contacts"]
    cg = cfg["contact_gate"]
    pred = Path(predictions_dir)

    # Which optimized body to read contact from: kinematics_1 (default) or the
    # annotation-guided kinematics_2. Output tag mirrors it: contacts_1 / contacts_2.
    kin_path = paths.body_path(pred, bodies)
    tag = bodies.removeprefix("kinematic_")
    scene_path = pred / "geometry" / "scene.npz"
    transform_path = pred / "geometry" / "transform.npz"
    if not kin_path.exists():
        raise SystemExit(f"missing {kin_path} — run human_optimization {bodies} first")
    if not scene_path.exists():
        raise SystemExit(f"missing {scene_path} — run fuse_geometry.py first (needs vggt --save-depth)")

    kin = dict(np.load(kin_path, allow_pickle=True))
    tr = dict(np.load(transform_path, allow_pickle=True))
    if not bool(tr.get("metric", True)):  # absent flag = old already-metric artifact
        raise SystemExit(f"geometry under {pred} is up-to-scale — run estimate_scale after fuse")
    scene_npz = dict(np.load(scene_path, allow_pickle=True))

    num_frames, num_people = int(kin["num_frames"]), int(kin["num_people"])
    height, width = int(kin["video_height"]), int(kin["video_width"])
    object_ids = np.asarray(kin["object_ids"], dtype=np.int32)
    parents = np.asarray(kin["parents"], dtype=np.int32)
    valid_mask = np.asarray(kin["valid_mask"], dtype=bool)
    fps = float(tr.get("fps", kin.get("fps", 30.0)))   # the drift window needs the true clip fps
    min_frames = min_span_frames(cc["min_span_s"], fps)   # the span rules count frames

    extrinsics = torch.as_tensor(tr["extrinsics"], dtype=torch.float32, device=device)       # (N,4,4)
    intrinsics = torch.as_tensor(tr["intrinsics_px_orig"], dtype=torch.float32, device=device)
    cam_centers = world_from_cam(extrinsics)[:, :3, 3]

    scene_pts, scene_normals, scene_conf = load_scene_cloud(
        scene_npz, cam_centers, conf_drop_q=float(cc["conf_drop_q"]),
        k=int(cc["normal_k"]), device=device)
    print(f"[estimate_contacts] scene cloud {scene_pts.shape[0]} pts, normals: "
          f"{'fuse' if 'normals' in scene_npz else 'PCA fallback'}; "
          f"close<{float(cc['close_thresh_m']) * 100:.0f}cm, "
          f"still<{float(cg['still_m']) * 100:.1f}cm")

    body = load_body(device=device, frames_allow=cc.get("frames") or None,
                     sole_offsets=cc.get("sole_offset_m") or None)
    frame_names = list(body.contact_frame_names)
    num_contact_frames = len(frame_names)
    knee_columns = [i for i, n in enumerate(frame_names) if n.startswith("knee")]
    heel_columns = [i for i, n in enumerate(frame_names) if n.startswith("heel")]
    cf = cfg["contact_force"]
    force_knobs = ForceKnobs(gap_dstar_m=float(cf["gap_dstar_m"]),
                             stillness_vstar_mps=float(cf["stillness_vstar_mps"]))

    cconf = cfg["contact_confidence"]

    # Gravity for the force solve: GeoCalib's world-down when the stage ran, else VGGT's
    # world-down convention — the same fallback kindyn uses.
    gravity_path = pred / "geocalib" / "gravity.npz"
    if gravity_path.exists():
        gravity_world = np.asarray(np.load(gravity_path)["gravity_world"], np.float32)
    else:
        gravity_world = np.array([0.0, 1.0, 0.0], np.float32)

    # ── the floor gate ([contacts].floor_gate_m) ─────────────────────────────────────
    # A foot only touches the floor when it is ON the floor. Where the gravity stage
    # accepted a ground plane, this stage measures every FOOT frame's height above it
    # (signed distance along up_world from ground_centroid) and refuses a contact label
    # above floor_gate_m, on both readouts. Nothing else is gated: a hand, a knee or a
    # sit bone touches things that are not the floor. 0 = off.
    floor_gate_m = float(cc["floor_gate_m"])
    foot_columns = [i for i, n in enumerate(frame_names) if n.rsplit("_", 1)[0] in FOOT_FRAMES]
    ground = None
    if floor_gate_m > 0.0 and foot_columns and gravity_path.exists():
        geo = np.load(gravity_path)
        if "ground_accepted" in geo.files and bool(geo["ground_accepted"]):
            ground = tuple(torch.as_tensor(np.asarray(geo[k], np.float32), device=device)
                           for k in ("ground_centroid", "up_world"))
            print(f"[estimate_contacts] floor gate {floor_gate_m * 100:.0f} cm on "
                  f"{len(foot_columns)} foot frame(s)")
        else:
            print(f"[estimate_contacts] floor gate {floor_gate_m * 100:.0f} cm asked for but "
                  "the gravity stage accepted no ground plane — not gating")

    shape = (num_people, num_frames, num_contact_frames)
    frame_floor_height_m = np.full(shape, np.nan, dtype=np.float32)
    frame_contact = np.zeros(shape, dtype=bool)
    merged_all = np.zeros(shape, dtype=bool)
    frame_sdf_cm = np.full(shape, np.nan, dtype=np.float32)
    frame_distance_cm = np.full(shape, np.nan, dtype=np.float32)
    frame_drift_m = np.full(shape, np.nan, dtype=np.float32)
    frame_normal_dot = np.full(shape, np.nan, dtype=np.float32)
    frames_2d = np.full((*shape, 2), np.nan, dtype=np.float32)
    frame_force = np.zeros((*shape, 3), dtype=np.float32)
    frame_force_bw = np.zeros(shape, dtype=np.float32)
    frame_force_label = np.zeros(shape, dtype=bool)
    frame_contact_geom = np.zeros(shape, dtype=bool)
    frame_pressed = np.zeros(shape, dtype=bool)
    force_dist_m = np.full(shape, np.nan, dtype=np.float32)
    frame_wall_dot = np.full(shape, -1.0, dtype=np.float32)
    body_weight_n = np.zeros(num_people, dtype=np.float32)
    residual_before_n = np.zeros((num_people, num_frames), dtype=np.float32)
    residual_after_n = np.zeros((num_people, num_frames), dtype=np.float32)
    contact_source = np.zeros(shape, dtype=np.int8)

    for p in range(num_people):
        object_id = int(object_ids[p])
        betas = torch.as_tensor(kin["betas"][p], dtype=torch.float32, device=device)
        q = torch.as_tensor(kin["q"][p], dtype=torch.float32, device=device)
        frames7 = _pose(body, betas, q, float(kin["scale"][p]))
        points, normals = frames7[..., :3], frame_normals(frames7)
        person_valid = torch.as_tensor(valid_mask[p], device=device)

        # The floor gate's verdict, one bool per (video frame, contact frame): False only
        # where a FOOT frame sits higher than floor_gate_m above the fitted ground plane.
        floor_ok = torch.ones((num_frames, num_contact_frames), dtype=torch.bool, device=device)
        if ground is not None:
            floor_height, floor_ok = floor_gate(points, *ground, foot_columns, floor_gate_m)
            frame_floor_height_m[p] = floor_height.cpu().numpy().astype(np.float32)

        # The persisted pre-enforcement drift is the honest motion measurement: the
        # optimizer's own damping must not be able to manufacture apparent stillness,
        # so there is no re-measure on this body. NaN entries carry no stillness
        # evidence and every consumer treats them as "not still".
        drift = _drift_pre(kin, p, (num_frames, num_contact_frames), device)

        # ── the force witness ────────────────────────────────────────────────────
        # Solved first, because the detector below reads it too. Physics answers two
        # different questions here: what the body's motion needs carried (the labels
        # further down, which ADD contacts geometry could not see) and whether a
        # pressing-but-moving step is a plant or a sweep (the load gate, which holds
        # the penetration exception to a load check).
        force_solution = solve_person_forces(
            body, betas, q, valid_mask[p], drift_pre=drift,
            drift_window_s=float(cg["window_s"]), gravity_world=gravity_world, fps=fps,
            scene_pts=scene_pts, scene_normals=scene_normals, scene_conf=scene_conf,
            knobs=force_knobs)
        load_bw = torch.as_tensor(
            force_solution.magnitude / max(force_solution.body_weight_n, 1e-9),
            device=device)
        # …except where the solve balanced nothing at all (a rope carrying the climber,
        # a trajectory too broken to explain). There its silence is not evidence.
        load_mute = torch.as_tensor(
            solve_mute(force_solution.residual_force_after_n, force_solution.body_weight_n,
                       mute_res_bw=float(cf["mute_res_bw"])), device=device)

        detected = detect_frame_contacts(
            points, normals, drift, scene_pts, scene_normals, scene_conf,
            frame_names=frame_names,
            close_thresh_m=float(cc["close_thresh_m"]), exit_thresh_m=float(cc["exit_thresh_m"]),
            min_frames=DETECT_MIN_FRAMES, still_m=float(cg["still_m"]),
            still_override_m=float(cg["still_override_m"]),
            max_dist_m=float(cc["max_dist_m"]), conf_thresh=float(cc["conf_thresh"]),
            prune_margin_m=float(cc["prune_margin_m"]),
            load_bw=load_bw, min_load_bw=float(cf["confirm_bw"]), load_mute=load_mute)
        # The gate acts on the geometric arm here, before the span rules, so a swing foot
        # never enters them; the force arm is gated the same way further down.
        gated = int((detected.contact & ~floor_ok).sum())
        geometric = detected.contact & floor_ok

        # Bridge short same-frame gaps the point barely moved across AND stayed near the
        # wall throughout (flicker / brief re-settle); the re-grip guard blocks a genuine
        # leave-and-return. The merge works one frame at a time, so on its own it can
        # bridge a toe over a step where the pair tiebreak had already ruled that toe out —
        # putting both toes of a foot back in contact at once. The tiebreak's verdict is
        # final, so bridged steps never land on a suppressed one.
        contact = merge_contact_spans(
            geometric, points, detected.guarded_sdf,
            merge_dist_m=float(cc["merge_dist_m"]), max_gap_s=float(cc["merge_max_gap_s"]),
            gap_sdf_m=float(cc["merge_sdf_m"]), fps=fps) & ~detected.toe_suppressed
        # A knee rides near the wall (and still) for most of a climb without touching:
        # measured on the annotated corpus, geometric knee labels are ~15% precise —
        # actively harmful. The force arm already refuses knees (below), so the stage
        # simply never labels a knee. A real kneebar is the known cost.
        contact[:, knee_columns] = False
        merged = contact & ~geometric

        points_cam = transform_points(points, extrinsics)                      # (N, F, 3)
        pixels = project(points_cam, intrinsics)                               # (N, F, 2)

        # ── the force LABELS ─────────────────────────────────────────────────────
        # The geometric detector above needs the frame to be NEAR the scene. When the
        # reconstruction puts the body in the wrong place that never happens and a real
        # contact is simply lost. So ask the same solve for the other direction: call a
        # frame in contact when it has to carry real load, wherever it sits. Add-only,
        # with exactly one exception: the toe pair (see below), where "at most one toe
        # per foot" outranks add-only.
        # A FAR force label needs stillness ([contact_force].far_still_x). Within the
        # detector's own trust radius ([contacts].max_dist_m) load alone suffices — a fast
        # foot plant that carries weight is a contact. Beyond it geometry is blind, and the
        # load claim needs the stillness witness: a misregistered body that is really
        # supported holds still (the gtIGgs floater, a beam routine whose beam cloud sits
        # half a metre off), while the mid-air scraps a bad-depth flight parks on flying
        # limbs move fast and are cut. The bar is a multiple of still_m looser than the
        # detector's, because a badly reconstructed body jitters — measured 2026-08-25:
        # at 4x, climbing F1 pays .007 while the high_jump/1FGkg far floods drop to zero.
        # The solve itself is untouched — residuals, load allocation and the toe rule see
        # every candidate. NaN distance = no scene evidence = far.
        # Heels never take a force label: within one foot the solve's load split is
        # indeterminate (a flat foot can carry its weight entirely through the ball),
        # so small solved heel loads say nothing — measured ~25% precise at the shipped
        # threshold. The heel's geometric arm is untouched; the raw magnitudes still
        # feed the residuals and the toe rule below.
        label_magnitude = force_solution.magnitude.copy()
        label_magnitude[:, heel_columns] = 0.0
        if float(cf["far_still_x"]) > 0.0:
            veto = far_still_veto(
                force_solution.dist_m, drift.cpu().numpy(),
                near_m=float(cc["max_dist_m"]), still_m=float(cg["still_m"]),
                far_still_x=float(cf["far_still_x"]))
            label_magnitude = np.where(veto, 0.0, label_magnitude)
        force_label = force_labels(
            label_magnitude, frame_names=frame_names, valid=valid_mask[p],
            threshold_n=float(cf["theta_bw"]) * force_solution.body_weight_n,
            hold_n=float(cf["theta_hold_bw"]) * force_solution.body_weight_n,
            min_frames=DETECT_MIN_FRAMES)
        force_label[:, knee_columns] = False        # force on a knee is ~all false positives
        floor_ok_np = floor_ok.cpu().numpy()
        gated += int((force_label & ~floor_ok_np).sum())
        force_label &= floor_ok_np
        if ground is not None:
            print(f"[estimate_contacts] floor gate removed {gated} foot frame-labels "
                  f"(both readouts, before the span rules)")
        contact_np = contact.cpu().numpy() | force_label
        contact_np[~valid_mask[p]] = False

        # The evidence exception, carried through to the union rules: a surviving press
        # (pressing into the surface, and load-confirmed where the solve was heard)
        # outranks the duration rules the same way it outranks stillness. A one-frame
        # bounce that presses in and carries weight is a contact, not flicker, and the
        # post-weld duration rule must not take it back.
        pressed = (detected.pressed & ~detected.toe_suppressed & floor_ok).cpu().numpy()

        # One debounce over the union. Each readout already drops its own too-short spans,
        # but the union puts two of them end to end and can leave a one- or two-frame hole
        # between them that neither arm ever saw. The same number governs both directions:
        # a contact lasts at least min_frames, and so does a release.
        contact_np = close_gaps(contact_np, min_frames)
        contact_np[~valid_mask[p]] = False          # never bridge across untracked steps
        # The fps-aware duration rule, applied only now that the union is welded. At
        # <=30 fps this is a no-op (every arm above already enforces 3 frames); at high
        # fps it drops the still-isolated micro-spans the welding could not fuse.
        contact_np = min_span(contact_np, min_frames, pressed)
        contact_np[~valid_mask[p]] = False

        # Both readouts keep one toe per foot, but they can keep DIFFERENT ones, and the
        # union would then hold both at once. The toe rule settles it for the whole contact
        # at once — see resolve_toe_conflicts. It only moves a label between the two frames
        # of one foot, so "this foot is in contact" is untouched.
        contact_np = resolve_toe_conflicts(contact_np, force_solution.magnitude,
                                           frame_names=frame_names,
                                           min_frames=DETECT_MIN_FRAMES, keep=pressed)
        merged_np = merged.cpu().numpy() & contact_np      # a bridged step that lost its
        merged_np[~valid_mask[p]] = False                  # toe is no longer a contact
        frame_contact_geom[p] = contact.cpu().numpy() & valid_mask[p][:, None]
        frame_floor_height_m[p][~valid_mask[p]] = np.nan
        frame_pressed[p] = pressed
        force_dist_m[p] = np.where(force_solution.dist_m < NO_SCENE_M,
                                   force_solution.dist_m, np.nan)
        frame_force[p] = force_solution.forces
        frame_force_bw[p] = force_solution.magnitude / force_solution.body_weight_n
        frame_force_label[p] = force_label
        frame_wall_dot[p] = force_solution.normal_dot
        body_weight_n[p] = force_solution.body_weight_n
        residual_before_n[p] = force_solution.residual_force_before_n
        residual_after_n[p] = force_solution.residual_force_after_n
        frame_contact[p] = contact_np
        merged_all[p] = merged_np
        frame_sdf_cm[p] = _finite_cm(detected.sdf)
        frame_distance_cm[p] = _finite_cm(detected.dmin)
        frame_normal_dot[p] = detected.normal_dot.cpu().numpy().astype(np.float32)
        frame_drift_m[p] = drift.cpu().numpy().astype(np.float32)
        frames_2d[p] = pixels.cpu().numpy()
        for array in (frame_sdf_cm[p], frame_distance_cm[p],
                      frame_normal_dot[p], frame_drift_m[p]):
            array[~valid_mask[p]] = np.nan

        # Where the label came from, saved for auditing: 2 = the geometric arm (near
        # AND still — or pressing with the solve's load confirmation; span-merge and
        # union-weld fills included); 4 = the force witness. The frames era has no
        # external detector, so bit 1 never appears.
        src = ((contact_np & (frame_contact_geom[p] | merged_np)) * 2
               + (contact_np & force_label) * 4).astype(np.int8)
        src[contact_np & (src == 0)] = 2
        contact_source[p] = src

        # How often did the toe tiebreak have to choose? Both toe frames of a foot were
        # candidates at that step and the normal picked one; the count is the size of the
        # both-fire problem the rule removes.
        toe_tiebreak = (detected.toe_suppressed & person_valid[:, None]).sum(dim=0).cpu().numpy()

        n_valid = int(valid_mask[p].sum())
        print(f"[estimate_contacts] obj_id={object_id}: "
              f"{int(contact_np.any(1).sum())}/{num_frames} frames with ≥1 contact "
              f"({int(contact_np.sum())} frame-point contact instances, "
              f"{int(merged_np.sum())} bridged)")
        for i, name in enumerate(frame_names):
            rate = float(contact_np[valid_mask[p], i].mean()) if n_valid else float("nan")
            print(f"    {name:<17s} rate={rate:5.1%}  spans={_count_spans(contact_np[:, i]):3d}  "
                  f"toe-tiebreak suppressed={int(toe_tiebreak[i]):4d}")

    # ── the per-clip quality flag ([quality]) — can this reconstruction be trusted? ──
    # Computed here because this is the earliest stage where all three signals exist:
    # the camera ratio and the smear come from geometry/, the residual from the force
    # witness above. Saved with its raw metrics so the verdict is re-derivable.
    qc = cfg["quality"]
    cam_ratio = cam_motion_ratio(tr["extrinsics"], scene_npz["points"])
    human_path = pred / "geometry" / "human.npz"
    smear_med = float("nan")
    if human_path.exists():
        with np.load(human_path) as human:
            smear_med, _ = smear_stats(human["points"], human["frame_id"],
                                       human["obj_id"], tr["extrinsics"])
    res_bw = residual_before_n / np.maximum(body_weight_n[:, None], 1e-9)
    res_before_bw = float(np.median(res_bw[valid_mask])) if valid_mask.any() else float("nan")
    flag = quality_flag(cam_ratio, res_before_bw, smear_med,
                        cam_depth_max=float(qc["cam_depth_max"]),
                        res_before_bw_max=float(qc["res_before_bw_max"]),
                        smear_med_max=float(qc["smear_med_max"]))
    print(f"[estimate_contacts] quality: cam/depth {cam_ratio:.2f}, smear med {smear_med:.2f}, "
          f"res_before {res_before_bw:.2f} BW → flag '{flag or 'pass'}'")

    # ── frame → joint, then the label confidence ─────────────────────────────────
    # Each frame reports at its parent joint, except hand frames, which all report at
    # their wrist (the joint we usually call "the hand"). A joint is in contact when any
    # of its frames is, and is as still as its stillest frame. The confidence — how often
    # a label like this one agrees with a human annotator — is fitted per joint, so it is
    # computed on the folded joints and broadcast back to their frames.
    joint_names = [str(n) for n in kin["joint_names"]]
    report_joint = frame_report_joints(body.contact_frame_parents, joint_names, parents)
    joint_contact = fold_contacts(frame_contact, report_joint, len(parents))
    joint_drift = fold_drift(frame_drift_m, report_joint, len(parents))
    joint_confidence = np.stack([
        label_confidence(joint_contact[p], joint_drift[p], valid_mask[p], fps,
                         joint_names, cconf)
        for p in range(num_people)])
    frame_label_confidence = joint_confidence[..., report_joint]

    save = dict(
        contact_label_schema=np.int32(CONTACT_LABEL_SCHEMA),
        contact_label_method=np.str_(CONTACT_LABEL_METHOD),
        quality_flag=np.str_(flag),                       # "" = pass; "camera" / "depth" / "camera+depth"
        quality_cam_ratio=np.float32(cam_ratio),          # camera-travel spread / median scene depth
        quality_smear_med=np.float32(smear_med),          # median along-ray human-cloud smear
        quality_res_before_bw=np.float32(res_before_bw),  # median pre-force root residual, body weights
        frame_contact=frame_contact,
        frame_force=frame_force,                 # (P,N,12,3) world contact force, newtons
        frame_force_bw=frame_force_bw,           # (P,N,F) the same as a fraction of body weight
        frame_force_label=frame_force_label,     # (P,N,F) the force decision BEFORE the union
        frame_contact_geom=frame_contact_geom,   # (P,N,F) the geometric arm BEFORE the union
        frame_pressed=frame_pressed,             # (P,N,F) penetration-exception keep mask
        force_dist_m=force_dist_m,               # (P,N,F) scene distance as the force solve saw it (1 m search; NaN = no scene evidence)
        frame_wall_dot=frame_wall_dot,           # (P,N,F) dot(cone axis, wall normal), force search radius
        body_weight_n=body_weight_n,             # (P,) baked mass x g
        residual_force_before_n=residual_before_n,   # (P,N) root residual with no contact force
        residual_force_after_n=residual_after_n,     # (P,N) and after the solve
        force_theta_bw=np.float32(cf["theta_bw"]),
        force_theta_hold_bw=np.float32(cf["theta_hold_bw"]),
        force_gap_dstar_m=np.float32(cf["gap_dstar_m"]),
        force_stillness_vstar_mps=np.float32(cf["stillness_vstar_mps"]),
        force_far_still_x=np.float32(cf["far_still_x"]),
        force_confirm_bw=np.float32(cf["confirm_bw"]),
        force_mute_res_bw=np.float32(cf["mute_res_bw"]),
        still_override_m=np.float32(cg["still_override_m"]),
        frame_sdf_cm=frame_sdf_cm,
        frame_distance_cm=frame_distance_cm,
        frame_drift_m=frame_drift_m,
        frame_normal_dot=frame_normal_dot,
        merged=merged_all,
        contact_source=contact_source,                  # (P,N,F) int8: 2=geometric arm, 4=force witness
        frame_label_confidence=frame_label_confidence,  # (P,N,F) its report joint's confidence
        frame_report_joint=report_joint,                # (F,) frame → report joint (hands → wrist)
        joint_contact=joint_contact,                    # (P,N,J) OR of the joint's frames
        joint_label_confidence=joint_confidence,        # (P,N,J) P(label agrees with an annotator);
                                                        #         NaN where the stage makes no statement
        confidence_method=np.str_(CONTACT_CONFIDENCE_METHOD),
        confidence_motion_source=np.str_("drift_pre"),
        confidence_coefficients=np.str_(json.dumps(cconf)),
        contact_frame_names=np.array(frame_names),
        frame_parent_joint=np.array(body.contact_frame_parents, dtype=np.int64),
        frame_vertex=np.array(body.contact_frame_vertices, dtype=np.int64),
        joint_names=kin["joint_names"], parents=parents,
        object_ids=object_ids, valid_mask=valid_mask,
        num_people=np.int32(num_people), num_frames=np.int32(num_frames),
        video_height=np.int32(height), video_width=np.int32(width), fps=np.float32(fps),
        source_body=np.str_(kin_path.name), source_body_sha256=np.str_(_sha256(kin_path)),
        source_scene=np.str_(scene_path.name), source_scene_sha256=np.str_(_sha256(scene_path)),
        source_transform=np.str_(transform_path.name),
        source_transform_sha256=np.str_(_sha256(transform_path)),
        close_thresh_m=np.float32(cc["close_thresh_m"]),
        exit_thresh_m=np.float32(cc["exit_thresh_m"]),
        min_span_s=np.float32(cc["min_span_s"]),
        min_frames=np.int32(min_frames),
        max_dist_m=np.float32(cc["max_dist_m"]),
        conf_thresh=np.float32(cc["conf_thresh"]),
        prune_margin_m=np.float32(cc["prune_margin_m"]),
        merge_dist_m=np.float32(cc["merge_dist_m"]),
        merge_max_gap_s=np.float32(cc["merge_max_gap_s"]),
        merge_sdf_m=np.float32(cc["merge_sdf_m"]),
        floor_gate_m=np.float32(floor_gate_m if ground is not None else 0.0),
        frame_floor_height_m=frame_floor_height_m,   # (P,N,F) height above the fitted ground
                                                     # plane, m; NaN where the gate is off
        drift_window_s=np.float32(cg["window_s"]),
        drift_still_m=np.float32(cg["still_m"]),
    )
    if np.isinf(frame_sdf_cm).any() or np.isinf(frame_distance_cm).any():
        raise RuntimeError("frame scene diagnostics may be NaN when missing, but never infinite")
    out_path = paths.contacts_path(pred, tag)
    atomic_savez(out_path, **save)
    print(f"[estimate_contacts] wrote {out_path}")

    if overlay:
        from bvr.human_optim.overlay import render_contact_overlay
        rendered = render_contact_overlay(
            source, paths.overlay_path(pred, f"contacts_{tag}"), frames_2d, frame_contact,
            valid_mask, height, width, fps)
        print(f"[estimate_contacts] wrote {rendered}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    determinism.apply()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--bodies", default="kinematic_1", choices=["kinematic_1", "kinematic_2"],
                    help="which refined body under human_optim/ to read contact from "
                         "(default kinematic_1; kinematic_2 for the annotation stage)")
    ap.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    args = ap.parse_args(argv)
    cfg = _load_config(args.config)
    run(args.source, args.predictions, args.config,
        overlay=bool(cfg["contacts"]["overlay"]), bodies=args.bodies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
