"""Scene-contact estimation on the 12 contact frames of the optimized SMPL-X bodies.

Reads a refined world-frame SMPL-X trajectory (``human_optim/kinematic_1.npz``) and
the static scene cloud (``geometry/scene.npz``) and answers one question for every
tracked person, every video frame and each of the 12 **contact frames** — the named
mesh points (toes, heels, knees, palms, fingertips) baked onto the body, each
carrying its own outward surface normal: *is this point touching the scene?*

Four steps, no more:

  1. pose each person per frame (re-applying the stored global scale about the
     pelvis) and read the contact-frame world poses straight out of forward
     kinematics — position and outward normal, no mesh work;
  2. for every contact frame find the nearest scene point: signed perpendicular
     distance, Euclidean distance, that point's confidence, and how well the two
     normals face each other;
  3. run the shared detector (:func:`~tools.human_optim.contacts.detect_frame_contacts`)
     — surface + **normal window** + stillness, then hysteresis and a minimum
     duration;
  4. bridge short flicker gaps behind a re-grip guard (bridged frames flagged in
     ``merged``).

The normal window is what makes toe hooks work: ``toe_under`` and ``toe_above`` sit
millimetres apart on the same toe, but their skin faces opposite ways, so at most
one of them can be touching a given hold.

Stillness reads the body's persisted pre-enforcement drift (``drift_pre``) when
present — motion measured before any contact attraction acted — and falls back to
drift re-measured on the final body otherwise.

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
    detect_frame_contacts, frame_guarded_sdf, joint_drift, merge_contact_spans,
    project, transform_points, world_from_cam,
)
from tools.human_optim import paths  # noqa: E402
from tools.human_optim.qmath import normalize_q  # noqa: E402

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "human_optimization.toml"

#: Bumped from 1 (the per-joint era) — the label axis is now the 12 contact frames.
CONTACT_LABEL_SCHEMA = 2
CONTACT_LABEL_METHOD = "frames-v1"


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
    c.setdefault("normal_max_angle_deg", 90.0)  # normal window: how far the skin may look away
    c.setdefault("normal_k", 16)           # kNN for PCA scene-normal estimation
    c.setdefault("prune_margin_m", 0.10)   # per-frame scene-AABB margin for the nearest-neighbour
    c.setdefault("merge_dist_m", 0.03)     # span-merge: bridge a gap the point barely moved across
    c.setdefault("merge_max_gap_s", 2.0)   # …only gaps shorter than this
    c.setdefault("merge_sdf_m", 0.08)      # …re-grip guard: it stayed this near the wall throughout
    c.setdefault("overlay", True)          # render the contact overlay
    if not bool(c.get("use_scene", True)):
        raise SystemExit("[estimate_contacts] use_scene=false is not supported by the contact-frame "
                         "detector — contact is measured against the scene cloud only")
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
          f"normal window ±{float(cc['normal_max_angle_deg']):.0f}°")

    body = load_body(device=device)
    frame_names = list(body.contact_frame_names)
    num_contact_frames = len(frame_names)

    shape = (num_people, num_frames, num_contact_frames)
    frame_contact = np.zeros(shape, dtype=bool)
    merged_all = np.zeros(shape, dtype=bool)
    frame_sdf_cm = np.full(shape, np.nan, dtype=np.float32)
    frame_distance_cm = np.full(shape, np.nan, dtype=np.float32)
    frame_drift_m = np.full(shape, np.nan, dtype=np.float32)
    frame_normal_dot = np.full(shape, np.nan, dtype=np.float32)
    frames_2d = np.full((*shape, 2), np.nan, dtype=np.float32)

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
            close_thresh_m=float(cc["close_thresh_m"]), exit_thresh_m=float(cc["exit_thresh_m"]),
            min_frames=int(cc["min_frames"]), still_m=float(cg["still_m"]),
            max_dist_m=float(cc["max_dist_m"]), conf_thresh=float(cc["conf_thresh"]),
            normal_max_angle_deg=float(cc["normal_max_angle_deg"]),
            prune_margin_m=float(cc["prune_margin_m"]))

        # Bridge short same-frame gaps the point barely moved across AND stayed near the
        # wall throughout (flicker / brief re-settle); the re-grip guard blocks a genuine
        # leave-and-return.
        contact = merge_contact_spans(
            detected.contact, points, detected.guarded_sdf,
            merge_dist_m=float(cc["merge_dist_m"]), max_gap_s=float(cc["merge_max_gap_s"]),
            gap_sdf_m=float(cc["merge_sdf_m"]), fps=fps)
        merged = contact & ~detected.contact

        pixels = project(transform_points(points, extrinsics), intrinsics)     # (N, F, 2)

        contact_np = contact.cpu().numpy()
        contact_np[~valid_mask[p]] = False
        merged_np = merged.cpu().numpy()
        merged_np[~valid_mask[p]] = False
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

        # How often did the surface guards accept a point that the normal window then
        # rejected? That difference IS the toe-hook discrimination, so it is worth
        # reporting: the same guarded signal with the window opened to 180°.
        open_window = frame_guarded_sdf(
            detected.sdf, detected.dmin, detected.conf, detected.normal_dot,
            max_dist_m=float(cc["max_dist_m"]), conf_thresh=float(cc["conf_thresh"]),
            normal_max_angle_deg=180.0)
        close_thresh = float(cc["close_thresh_m"])
        window_rejected = ((open_window < close_thresh) & (detected.guarded_sdf >= close_thresh)
                           & person_valid[:, None]).sum(dim=0).cpu().numpy()

        n_valid = int(valid_mask[p].sum())
        print(f"[estimate_contacts] obj_id={object_id}: "
              f"{int(contact_np.any(1).sum())}/{num_frames} frames with ≥1 contact "
              f"({int(contact_np.sum())} frame-point contact instances, "
              f"{int(merged_np.sum())} bridged)")
        for i, name in enumerate(frame_names):
            rate = float(contact_np[valid_mask[p], i].mean()) if n_valid else float("nan")
            print(f"    {name:<17s} rate={rate:5.1%}  spans={_count_spans(contact_np[:, i]):3d}  "
                  f"normal-gate rejected={int(window_rejected[i]):4d}")

    save = dict(
        contact_label_schema=np.int32(CONTACT_LABEL_SCHEMA),
        contact_label_method=np.str_(CONTACT_LABEL_METHOD),
        frame_contact=frame_contact,
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
        normal_max_angle_deg=np.float32(cc["normal_max_angle_deg"]),
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
