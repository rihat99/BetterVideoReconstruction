"""Scene-contact estimation on the 12 contact frames of the optimized SMPL-X bodies.

Reads a refined world-frame SMPL-X trajectory (``human_optim/kinematic_1.npz``) and
the static scene cloud (``geometry/scene.npz``) and answers one question for every
tracked person, every video frame and each of the 12 **contact frames** — the named
mesh points (toes, heels, knees, palms, fingertips) baked onto the body, each
carrying its own outward surface normal: *is this point touching the scene?*

Two readouts answer it, and the stage unions them.

**Geometry** — pose each person per frame (re-applying the stored global scale about
the pelvis) and read the contact-frame world poses straight out of forward kinematics;
for each one find the nearest scene point; then run the shared detector
(:func:`~tools.human_optim.contacts.detect_frame_contacts`): near the wall AND still,
with hysteresis, a minimum duration and a span merge behind a re-grip guard (bridged
frames flagged in ``merged``).

**Force** — freeze the pose and ask what force is unexplained, then solve which of the
12 frames could have supplied it (:mod:`tools.human_optim.forces`). A frame that has to
carry real load is in contact whatever its distance says, which is how a contact on a
badly reconstructed body is recovered at all. See ``docs/force_contacts.md``.

The union is then debounced once (a contact lasts at least ``min_frames``, and so does a
release) and each foot is given a single toe patch per contact — the force decides
whether it was standing on the hold (``toe_under``) or hooked under it (``toe_above``),
once for the whole contact rather than per step.

Stillness reads the body's persisted pre-enforcement drift (``drift_pre``) — motion
measured before any contact attraction acted. The force solve requires it and will not
re-measure drift on an enforced body.

Renders ``contacts_<tag>_overlay.mp4``: the 12 projected contact points, **red** in
contact / **green** otherwise.

Run from CLI::

    .venv/bin/python scripts/stages/estimate_contacts.py \\
        --source video.mp4 --predictions out/video/ --config configs/human_optimization.toml

Reads:  human_optim/kinematic_1.npz, geometry/scene.npz, geometry/transform.npz
Writes: human_optim/contacts_1.npz          (frame_contact (P,N,12) bool + the signed /
                                             Euclidean / normal / drift signals behind it)
        human_optim/contacts_1_overlay.mp4  ([contacts].overlay; default on)
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.body import frame_normals, load_body  # noqa: E402
from tools.config import load_toml  # noqa: E402
from tools.geometry.normals import load_scene_cloud  # noqa: E402
from tools.human_optim import (  # noqa: E402
    detect_frame_contacts, joint_drift, merge_contact_spans,
    project, transform_points, world_from_cam,
)
from tools.human_optim import paths  # noqa: E402
from tools.human_optim.forces import (  # noqa: E402
    ForceKnobs, close_gaps, force_labels, resolve_toe_conflicts, solve_person_forces,
)
from tools.human_optim.qmath import normalize_q  # noqa: E402

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "human_optimization.toml"

#: Bumped from 1 (the per-joint era) — the label axis is now the 12 contact frames. The
#: force witness (method "frames-v2") only ADDS fields and changes no shape, so readers of
#: schema 2 keep working; ``contact_label_method`` is what distinguishes the two.
CONTACT_LABEL_SCHEMA = 2

CONTACT_LABEL_METHOD = "frames-v3"   # v1 = geometry only; v2 adds the force witness;
                                     # v3 the toe support rule, debounce + force hysteresis


def _load_config(path: Path) -> dict:
    """Operational knobs for the contact stage (no loss weights here)."""
    cfg = load_toml(path) if path.exists() else {}
    c = cfg.setdefault("contacts", {})
    c.setdefault("close_thresh_m", 0.03)   # enter: "just outside" distance counted as contact
    c.setdefault("exit_thresh_m", 0.035)   # stay: looser hysteresis exit (kills boundary flicker)
    c.setdefault("min_frames", 3)          # drop contact spans shorter than this many frames
    c.setdefault("max_dist_m", 0.10)       # far guard (a match behind an occluder is not a surface)
    c.setdefault("conf_thresh", 0.1)       # residual absolute floor (VGGT conf ≥~1 ⇒ non-binding)
    c.setdefault("conf_drop_q", 0.1)       # quantile confidence gate: drop the lowest scene voxels
    c.setdefault("normal_k", 16)           # kNN for PCA scene-normal estimation
    c.setdefault("prune_margin_m", 0.10)   # per-frame scene-AABB margin for the nearest-neighbour
    c.setdefault("merge_dist_m", 0.03)     # span-merge: bridge a gap the point barely moved across
    c.setdefault("merge_max_gap_s", 2.0)   # …only gaps shorter than this
    c.setdefault("merge_sdf_m", 0.08)      # …re-grip guard: it stayed this near the wall throughout
    c.setdefault("overlay", True)          # render the contact overlay
    if not bool(c.get("use_scene", True)):
        raise SystemExit("[estimate_contacts] use_scene=false is not supported by the contact-frame "
                         "detector — contact is measured against the scene cloud only")
    f = cfg.setdefault("contact_force", {})
    f.setdefault("theta_bw", 0.05)          # a frame carrying this fraction of body weight is loaded
    f.setdefault("theta_hold_bw", 0.03)     # …and stays loaded until it drops below this (hysteresis)
    f.setdefault("gap_dstar_m", 0.05)       # gap price: force this far from the scene is expensive
    f.setdefault("stillness_vstar_mps", 0.2)  # stillness price: force on a limb this fast is expensive
    g = cfg.setdefault("contact_gate", {})
    g.setdefault("window_s", 0.35)         # drift window (s) for the stillness gate
    g.setdefault("still_m", 0.03)          # net drift (m) at which stillness = 0.5
    cfg.setdefault("runtime", {}).setdefault("device", "cuda")
    return cfg


def _sha256(path: Path) -> str:
    with path.open("rb") as source_file:
        return hashlib.file_digest(source_file, "sha256").hexdigest()


@torch.no_grad()
def _pose(body, betas: torch.Tensor, q: torch.Tensor, scale: float, *, chunk: int = 64):
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
    """The persisted pre-enforcement drift ``(N, 12)`` for one person, or a hard error.

    The force solve prices force by how still a frame is. If that stillness were measured
    on the *enforced* body, the optimizer's own contact damping would be voting for its own
    labels, so there is deliberately no fallback here: a body without ``drift_pre`` cannot
    have a force witness computed from it.
    """
    if "drift_pre" not in kin:
        raise SystemExit("[estimate_contacts] kinematic_1.npz has no drift_pre — the force "
                         "witness needs pre-enforcement stillness and will not re-measure "
                         "it on the enforced body; re-run human_optimization")
    drift = torch.as_tensor(np.asarray(kin["drift_pre"])[person],
                            dtype=torch.float32, device=device)
    if tuple(drift.shape) != tuple(shape) or not bool(torch.isfinite(drift).all()):
        raise SystemExit("[estimate_contacts] drift_pre is the wrong shape or has "
                         "non-finite entries; refusing to guess")
    return drift


def _finite_cm(values: torch.Tensor) -> np.ndarray:
    """Metres → centimetres as float32, with the ``±inf`` sentinels turned into ``NaN``."""
    out = (values * 100.0).cpu().numpy().astype(np.float32)
    return np.where(np.isfinite(out), out, np.nan).astype(np.float32)


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

    body = load_body(device=device)
    frame_names = list(body.contact_frame_names)
    num_contact_frames = len(frame_names)
    knee_columns = [i for i, n in enumerate(frame_names) if n.startswith("knee")]
    cf = cfg["contact_force"]
    force_knobs = ForceKnobs(gap_dstar_m=float(cf["gap_dstar_m"]),
                             stillness_vstar_mps=float(cf["stillness_vstar_mps"]))

    # Gravity for the force solve: GeoCalib's world-down when the stage ran, else VGGT's
    # world-down convention — the same fallback the dynamics stage uses.
    gravity_path = pred / "geocalib" / "gravity.npz"
    if gravity_path.exists():
        gravity_world = np.asarray(np.load(gravity_path)["gravity_world"], np.float32)
    else:
        gravity_world = np.array([0.0, 1.0, 0.0], np.float32)

    shape = (num_people, num_frames, num_contact_frames)
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
    frame_wall_dot = np.full(shape, -1.0, dtype=np.float32)
    body_weight_n = np.zeros(num_people, dtype=np.float32)
    residual_before_n = np.zeros((num_people, num_frames), dtype=np.float32)
    residual_after_n = np.zeros((num_people, num_frames), dtype=np.float32)

    for p in range(num_people):
        object_id = int(object_ids[p])
        betas = torch.as_tensor(kin["betas"][p], dtype=torch.float32, device=device)
        q = torch.as_tensor(kin["q"][p], dtype=torch.float32, device=device)
        posed = _pose(body, betas, q, float(kin["scale"][p]))                 # (N, F, 7)
        points, normals = posed[..., :3], frame_normals(posed)
        person_valid = torch.as_tensor(valid_mask[p], device=device)

        # The persisted pre-enforcement drift is the honest motion measurement: the
        # optimizer's own damping must not be able to manufacture apparent stillness.
        # Fall back per element to drift re-measured on this body (old npz / NaN).
        drift = joint_drift(points, fps=fps, window_s=float(cg["window_s"]),
                            valid_mask=person_valid)
        if "drift_pre" in kin:
            persisted = torch.as_tensor(
                np.asarray(kin["drift_pre"])[p], dtype=drift.dtype, device=drift.device)
            if persisted.shape == drift.shape:
                drift = torch.where(
                    torch.isfinite(persisted) & torch.isfinite(drift), persisted, drift)

        detected = detect_frame_contacts(
            points, normals, drift, scene_pts, scene_normals, scene_conf,
            frame_names=frame_names,
            close_thresh_m=float(cc["close_thresh_m"]), exit_thresh_m=float(cc["exit_thresh_m"]),
            min_frames=int(cc["min_frames"]), still_m=float(cg["still_m"]),
            max_dist_m=float(cc["max_dist_m"]), conf_thresh=float(cc["conf_thresh"]),
            prune_margin_m=float(cc["prune_margin_m"]))

        # Bridge short same-frame gaps the point barely moved across AND stayed near the
        # wall throughout (flicker / brief re-settle); the re-grip guard blocks a genuine
        # leave-and-return. The merge works one frame at a time, so on its own it can
        # bridge a toe over a step where the pair tiebreak had already ruled that toe out —
        # putting both toes of a foot back in contact at once. The tiebreak's verdict is
        # final, so bridged steps never land on a suppressed one.
        contact = merge_contact_spans(
            detected.contact, points, detected.guarded_sdf,
            merge_dist_m=float(cc["merge_dist_m"]), max_gap_s=float(cc["merge_max_gap_s"]),
            gap_sdf_m=float(cc["merge_sdf_m"]), fps=fps) & ~detected.toe_suppressed
        merged = contact & ~detected.contact

        pixels = project(transform_points(points, extrinsics), intrinsics)     # (N, F, 2)

        # ── the force witness ────────────────────────────────────────────────────
        # The geometric detector above needs the frame to be NEAR the scene. When the
        # reconstruction puts the body in the wrong place that never happens and a real
        # contact is simply lost. So ask physics as well: solve the world forces at the 12
        # frames that explain the body's leftover root wrench, and call a frame in contact
        # when it has to carry real load. Add-only, with exactly one exception: the toe
        # pair (see below), where "at most one toe per foot" outranks add-only.
        force_solution = solve_person_forces(
            body, betas, q, valid_mask[p], drift_pre=_drift_pre(kin, p, drift.shape, device),
            drift_window_s=float(cg["window_s"]), gravity_world=gravity_world, fps=fps,
            scene_pts=scene_pts, scene_normals=scene_normals, scene_conf=scene_conf,
            knobs=force_knobs)
        force_label, _ = force_labels(
            force_solution.magnitude, frame_names=frame_names, valid=valid_mask[p],
            threshold_n=float(cf["theta_bw"]) * force_solution.body_weight_n,
            hold_n=float(cf["theta_hold_bw"]) * force_solution.body_weight_n,
            min_frames=int(cc["min_frames"]), support=force_solution.support)
        force_label[:, knee_columns] = False        # force on a knee is ~all false positives
        contact_np = contact.cpu().numpy() | force_label
        contact_np[~valid_mask[p]] = False

        # One debounce over the union. Each readout already drops its own too-short spans,
        # but the union puts two of them end to end and can leave a one- or two-frame hole
        # between them that neither arm ever saw. The same number governs both directions:
        # a contact lasts at least min_frames, and so does a release.
        contact_np = close_gaps(contact_np, int(cc["min_frames"]))
        contact_np[~valid_mask[p]] = False          # never bridge across untracked steps

        # Both readouts keep one toe per foot, but they can keep DIFFERENT ones, and the
        # union would then hold both at once. The toe rule settles it for the whole contact
        # at once — see resolve_toe_conflicts. It only moves a label between the two frames
        # of one foot, so "this foot is in contact" is untouched.
        contact_np = resolve_toe_conflicts(contact_np, force_solution.support,
                                           frame_names=frame_names,
                                           min_frames=int(cc["min_frames"]))
        merged_np = merged.cpu().numpy() & contact_np      # a bridged step that lost its
        merged_np[~valid_mask[p]] = False                  # toe is no longer a contact
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

    save = dict(
        contact_label_schema=np.int32(CONTACT_LABEL_SCHEMA),
        contact_label_method=np.str_(CONTACT_LABEL_METHOD),
        frame_contact=frame_contact,
        frame_force=frame_force,                 # (P,N,12,3) world contact force, newtons
        frame_force_bw=frame_force_bw,           # (P,N,12) the same as a fraction of body weight
        frame_force_label=frame_force_label,     # (P,N,12) the force decision BEFORE the union
        frame_wall_dot=frame_wall_dot,           # (P,N,12) dot(cone axis, wall normal), force search radius
        body_weight_n=body_weight_n,             # (P,) baked mass x g
        residual_force_before_n=residual_before_n,   # (P,N) root residual with no contact force
        residual_force_after_n=residual_after_n,     # (P,N) and after the solve
        force_theta_bw=np.float32(cf["theta_bw"]),
        force_theta_hold_bw=np.float32(cf["theta_hold_bw"]),
        force_gap_dstar_m=np.float32(cf["gap_dstar_m"]),
        force_stillness_vstar_mps=np.float32(cf["stillness_vstar_mps"]),
        frame_sdf_cm=frame_sdf_cm,
        frame_distance_cm=frame_distance_cm,
        frame_drift_m=frame_drift_m,
        frame_normal_dot=frame_normal_dot,
        merged=merged_all,
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
        min_frames=np.int32(cc["min_frames"]),
        max_dist_m=np.float32(cc["max_dist_m"]),
        conf_thresh=np.float32(cc["conf_thresh"]),
        prune_margin_m=np.float32(cc["prune_margin_m"]),
        merge_dist_m=np.float32(cc["merge_dist_m"]),
        merge_max_gap_s=np.float32(cc["merge_max_gap_s"]),
        merge_sdf_m=np.float32(cc["merge_sdf_m"]),
        drift_window_s=np.float32(cg["window_s"]),
        drift_still_m=np.float32(cg["still_m"]),
    )
    if np.isinf(frame_sdf_cm).any() or np.isinf(frame_distance_cm).any():
        raise RuntimeError("frame scene diagnostics may be NaN when missing, but never infinite")
    out_path = paths.contacts_path(pred, tag)
    np.savez(out_path, **save)
    print(f"[estimate_contacts] wrote {out_path}")

    if overlay:
        from tools.human_optim.overlay import render_contact_overlay
        rendered = render_contact_overlay(
            source, paths.overlay_path(pred, f"contacts_{tag}"), frames_2d, frame_contact,
            None, valid_mask, height, width, fps)
        print(f"[estimate_contacts] wrote {rendered}")
    return out_path


def main(argv: list[str] | None = None) -> int:
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
