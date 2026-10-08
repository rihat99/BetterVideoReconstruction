"""Contacts and forces for raw AMASS SMPL-X motions, by the BEDLAM2 readout.

BEDLAM2's bodies are AMASS motions, so the two measurements ``scripts/label_bedlam2.py``
makes apply to an AMASS ``*_stageii.npz`` as they are: :mod:`tools.bedlam_contacts`
labels the 35 contact frames against a floor plane (height, stillness, which way the
skin faces) and :mod:`tools.bedlam_forces` solves the force every labelled frame
carries at the frozen pose with ``kindyn_1``'s force objective. The knobs are the same
too — ``[bedlam]`` and ``[bedlam_forces]`` in ``configs/pipeline.toml``. The pose goes
in the way ``scripts/import_bedlam2.py`` brings it in: 16 locked-head betas truncated
to the pipeline's 10, and the model's hand mean subtracted (AMASS hands are absolute).
The mocap is subsampled to the 30 fps the force solve is tuned at (every 4th frame of a
120 fps capture); ``source_frame`` maps each output frame back to the original.

What AMASS does not give is the floor. Its world is z-up with the floor nominally at
z = 0, but the fitted mesh does not stand there: on CMU 49_06 the still feet sit 4 cm
below it. So the floor is measured: a horizontal plane at the median height of each
frame's lowest contact frame, over the frames where that point is still (calm speed
below ``still_mps``). Where the body rests, it rests on the floor.

``--step label`` writes, per motion, ``<stem>_contacts.npz`` in the AMASS world frame:

* ``frame_contact (N, 35)`` bool and ``frame_force (N, 35, 3)`` newtons — zero wherever
  the frame is not in contact — with ``frame_position (N, 35, 3)``, where each acts;
* the same per SMPL-X joint: ``joint_contact (N, 52)`` (any of the joint's contact frames
  touches) and ``joint_force (N, 52, 3)`` (their forces summed), ``joint_position``;
* ``frame_names``, ``frame_vertex`` (SMPL-X vertex of each frame), ``frame_parent_joint``,
  ``joint_names``, the floor (``floor_height_m``, ``up``), ``total_mass_kg``, ``q`` and
  the three contact measurements behind every label;

and once ``contact_frame_vertices.npz``: contact frame id → SMPL-X vertex id table.

``--step render`` draws ``<stem>_contacts.mp4`` from that file at half speed: two
views of the mesh with contact frames (green on the floor, red off it) and force arrows,
over a strip comparing the vertical and horizontal contact force with what the centre
of mass needs, ``m (a_com - g)``, and a timeline of the frames in contact.

Usage::

    CUDA_VISIBLE_DEVICES=2 EGL_DEVICE_ID=2 uv run python scripts/label_amass.py \\
        --motions ../data/retarget_test/motions --save-dir ../data/retarget_test/results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.bedlam_contacts import (  # noqa: E402
    CONTACT_LABEL_METHOD, ENTER_M, EXIT_M, MAX_TILT_DEG, MIN_SPAN_S, STILL_MPS, WINDOW_S,
    calm_speed, contact_from_floor, frame_positions)
from tools.bedlam_forces import FORCE_METHOD, person_forces  # noqa: E402
from tools.body import load_body  # noqa: E402
from tools.config import load_toml, section  # noqa: E402
from tools.human_optim.config import FPS_REF, load_config  # noqa: E402

LOCKED_HEAD = Path("/home/rikhat.akizhanov/better/BetterHuman/models/smplx_lockedhead/SMPLX_NEUTRAL.npz")
NUM_BETAS = 10
GRAVITY = 9.81
UP = np.array([0.0, 0.0, 1.0], np.float32)      # AMASS is z-up
ARROW_M_PER_BW = 0.8                             # arrow length of one body weight


def motion_to_q(body, motion: Path, device
                ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """AMASS SMPL-X stage-II → ``(q (N, 211), betas (10,), fps, source_frame (N,))``.

    Posed as import_bedlam2 does it, at every ``round(fps / 30)``-th mocap frame.
    """
    data = np.load(motion)
    model, gender = str(data["surface_model_type"]), str(data["gender"])
    if model != "smplx" or gender != "neutral":
        raise ValueError(f"{motion}: {gender} {model}; the pipeline body is neutral SMPL-X")
    hand = np.load(LOCKED_HEAD, allow_pickle=True)
    step = max(1, round(float(data["mocap_frame_rate"]) / FPS_REF))
    source_frame = np.arange(0, len(data["poses"]), step)
    poses, trans = data["poses"][source_frame], data["trans"][source_frame]
    betas = np.asarray(data["betas"][:NUM_BETAS], np.float32)
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
    with torch.no_grad():
        q = body.from_classic(t(betas)[None], t(poses[:, 3:66]), t(trans),
                              t(poses[:, :3]), t(poses[:, 75:120] - hand["hands_meanl"]),
                              t(poses[:, 120:165] - hand["hands_meanr"]))
    return q.cpu().numpy(), betas, float(data["mocap_frame_rate"]) / step, source_frame


def floor_height(positions: torch.Tensor, velocities: torch.Tensor, *, up: torch.Tensor,
                 fps: float, window_s: float, still_mps: float) -> float:
    """Median height of each frame's lowest contact frame, over the frames it is still."""
    height = positions @ up                                                  # (N, F)
    speed = calm_speed(velocities.norm(dim=-1), fps=fps, window_s=window_s)
    rows = torch.arange(len(height), device=height.device)
    lowest = height.argmin(dim=1)
    still = speed[rows, lowest] < still_mps
    if not bool(still.any()):
        raise ValueError("no frame where the lowest point is still: the floor is unknown")
    return float(height[rows, lowest][still].median())


def label_motion(motion: Path, *, body, force_body, device, contact_knobs: dict,
                 force_knobs: dict) -> dict:
    """Contact labels and forces for one motion, keyed by npz field name."""
    q, betas, fps, source_frame = motion_to_q(body, motion, device)
    up = torch.as_tensor(UP, device=device)
    positions, velocities, normals, _ = frame_positions(
        body, betas, q, device, up=up, centroid=torch.zeros(3, device=device), fps=fps)
    floor = floor_height(positions, velocities, up=up, fps=fps,
                         window_s=contact_knobs["window_s"], still_mps=contact_knobs["still_mps"])
    out = contact_from_floor(positions, velocities, normals, up=up, centroid=up * floor,
                             fps=fps, **contact_knobs)
    contact = out["contact"].cpu().numpy()

    solved = person_forces(force_body, betas, q, contact, gravity_world=-UP, fps=fps,
                           **force_knobs)
    parents = np.array(body.contact_frame_parents, dtype=np.int64)
    joint_contact = np.zeros((len(q), body.num_joints), bool)
    for frame, joint in enumerate(parents):
        joint_contact[:, joint] |= contact[:, frame]

    return {
        "motion": np.str_(motion.name), "fps": np.float32(fps),
        "num_frames": np.int32(len(q)), "source_frame": source_frame,
        "floor_height_m": np.float32(floor), "up": UP,
        "betas": betas, "q": q.astype(np.float32),
        "total_mass_kg": np.float32(solved["total_mass"]),
        "frame_names": np.array(body.contact_frame_names),
        "frame_vertex": np.array(body.contact_frame_vertices, dtype=np.int64),
        "frame_parent_joint": parents,
        "frame_contact": contact,
        "frame_force": solved["frame_forces"],                     # (N, F, 3) world N
        "frame_position": positions.cpu().numpy(),                 # (N, F, 3)
        "joint_names": np.array(body.joint_names),
        "joint_contact": joint_contact,
        "joint_force": solved["force_world"],                      # (N, J, 3) world N
        "joint_position": solved["joints_world"],                  # (N, J, 3)
        "frame_height_m": out["height"].cpu().numpy(),
        "frame_speed_mps": out["speed"].cpu().numpy(),
        "frame_tilt_deg": out["tilt"].cpu().numpy(),
        "root_residual": solved["base_wrench"],                    # (N, 6) root-local, unexplained
        "contact_method": np.str_(CONTACT_LABEL_METHOD + "+amass-floor"),
        "force_method": np.str_(FORCE_METHOD),
        "contact_knobs": np.str_(json.dumps(contact_knobs, sort_keys=True)),
    }


def com_force_needed(body, betas: np.ndarray, q: np.ndarray, fps: float, device) -> np.ndarray:
    """External force ``m (a_com - g)`` in body weights ``(N, 3)``, NaN at the clip edges.

    The centre of mass from BetterRobot's CCRBA, differenced over the same ±``fps/30``
    frame stencil the force solve uses, so the two see the same acceleration noise.
    """
    from better_robot.dynamics import center_of_mass

    from tools.human_optim.qmath import normalize_q

    model = body.shaped(torch.as_tensor(betas, device=device)).model
    with torch.no_grad():
        qn = normalize_q(torch.as_tensor(q, device=device), body.num_joints)
        com = center_of_mass(model, model.fk(qn)).cpu().numpy().astype(np.float64)
    step = max(1, round(fps / FPS_REF))
    accel = np.full_like(com, np.nan)
    accel[step:-step] = (com[2 * step:] - 2 * com[step:-step] + com[:-2 * step]) * (fps / step) ** 2
    return accel / GRAVITY + UP


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """OpenGL camera pose (``-z`` forward, ``+y`` up) at ``eye`` looking at ``target``."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, UP)
    right /= np.linalg.norm(right)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, np.cross(right, forward), -forward, eye
    return pose


def arrows(starts: np.ndarray, vectors: np.ndarray):
    """One trimesh of arrows from ``starts (K, 3)`` along ``vectors (K, 3)`` metres."""
    import trimesh

    parts = []
    for start, vector in zip(starts, vectors):
        length = float(np.linalg.norm(vector))
        if length < 0.01:
            continue
        head = min(0.06, 0.4 * length)
        align = trimesh.geometry.align_vectors([0, 0, 1], vector / length)
        shaft = trimesh.creation.cylinder(radius=0.008, height=length - head, sections=12)
        shaft.apply_translation([0, 0, (length - head) / 2])
        cone = trimesh.creation.cone(radius=0.022, height=head, sections=16)
        cone.apply_translation([0, 0, length - head])
        arrow = trimesh.util.concatenate([shaft, cone])
        arrow.apply_transform(align)
        arrow.apply_translation(start)
        parts.append(arrow)
    return trimesh.util.concatenate(parts) if parts else None


def checkerboard(center: np.ndarray, half: float, height: float, tile: float = 0.5):
    """A grey checkerboard floor at ``height`` around ``center`` (xy), tiles of ``tile`` m."""
    import trimesh

    tiles, count = [], int(np.ceil(half / tile))
    for i in range(-count, count):
        for j in range(-count, count):
            box = trimesh.creation.box(extents=[tile, tile, 0.002])
            box.apply_translation([center[0] + (i + 0.5) * tile, center[1] + (j + 0.5) * tile,
                                   height - 0.001])
            box.visual.face_colors = [205, 205, 205, 255] if (i + j) % 2 else [170, 170, 170, 255]
            tiles.append(box)
    return trimesh.util.concatenate(tiles)


def plot_strip(d: dict, needed: np.ndarray, width: int, height: int):
    """The force and contact-timeline strip ``(H, W, 3)`` plus a time → pixel-column map."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fps, contact = float(d["fps"]), d["frame_contact"]
    time = np.arange(len(contact)) / fps
    weight = float(d["total_mass_kg"]) * GRAVITY
    total = d["frame_force"].sum(axis=1) / weight                                # (N, 3) BW
    used = np.flatnonzero(contact.any(axis=0))
    fig, (ax_f, ax_c) = plt.subplots(2, 1, figsize=(width / 100, height / 100), dpi=100,
                                     sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    ax_f.plot(time, needed[:, 2], color="black", lw=1.2, ls="--", label="vertical, needed m(a-g)")
    ax_f.plot(time, total[:, 2], color="tab:orange", lw=1.8, label="vertical, solved contacts")
    ax_f.plot(time, np.linalg.norm(needed[:, :2], axis=1), color="grey", lw=0.9, ls="--",
              label="horizontal, needed")
    ax_f.plot(time, np.linalg.norm(total[:, :2], axis=1), color="tab:blue", lw=1.2,
              label="horizontal, solved")
    ax_f.set_ylabel("force / body weight")
    ax_f.legend(loc="upper left", fontsize=7, ncol=4, frameon=False)
    ax_f.grid(alpha=0.3)
    ax_c.imshow(contact[:, used].T, aspect="auto", interpolation="nearest", cmap="Greens",
                extent=(0, time[-1], len(used) - 0.5, -0.5), vmin=0, vmax=1.3)
    ax_c.set_yticks(range(len(used)), [str(d["frame_names"][i]) for i in used], fontsize=6)
    ax_c.set_xlabel("time (s)")
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    x0 = ax_f.transData.transform((0.0, 0.0))[0]
    x1 = ax_f.transData.transform((1.0, 0.0))[0]
    plt.close(fig)
    return image, lambda seconds: int(round(x0 + (x1 - x0) * seconds))


def render_motion(npz: Path, out: Path, *, body, device, size=(720, 540)) -> None:
    """Draw ``<stem>_contacts.mp4``: two 3D views over the force/contact strip."""
    import os

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import cv2
    import imageio.v2 as imageio
    import pyrender
    import trimesh

    d = dict(np.load(npz))
    fps, q, betas = float(d["fps"]), d["q"], d["betas"]
    contact, force, where = d["frame_contact"], d["frame_force"], d["frame_position"]
    weight = float(d["total_mass_kg"]) * GRAVITY
    floor = float(d["floor_height_m"])
    with torch.no_grad():
        vertices = torch.cat([
            body.forward(torch.as_tensor(betas, device=device)[None],
                         torch.as_tensor(q[s:s + 128], device=device)).vertices.cpu()
            for s in range(0, len(q), 128)]).numpy()
    faces = body.faces.cpu().numpy()
    needed = com_force_needed(body, betas, q, fps, device)

    # Cameras follow the pelvis (smoothed over a second), looking across the direction
    # the body travels and from a raised three-quarter angle.
    pelvis = d["joint_position"][:, 0]
    kernel = np.ones(int(fps)) / int(fps)
    track = np.stack([np.convolve(np.pad(pelvis[:, k], len(kernel) // 2, mode="edge"),
                                  kernel, mode="valid")[:len(q)] for k in range(3)], axis=1)
    travel = pelvis[-1, :2] - pelvis[0, :2]
    distance = float(np.linalg.norm(travel))
    travel = travel / distance if distance > 0.3 else np.array([1.0, 0.0])
    side = np.array([-travel[1], travel[0], 0.0])
    oblique = 0.6 * side + 0.8 * np.array([travel[0], travel[1], 0.0])
    views = [side * 3.4 + UP * 0.35, oblique * 3.0 + UP * 1.6]           # eye minus target

    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    skin = pyrender.MetallicRoughnessMaterial(baseColorFactor=(0.72, 0.74, 0.8, 0.55),
                                              alphaMode="BLEND", metallicFactor=0.0)
    material = lambda rgb: pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(*rgb, 1.0), metallicFactor=0.0, roughnessFactor=0.6)
    on, off, arrow_mat = material((0.1, 0.75, 0.2)), material((0.85, 0.15, 0.1)), material((1.0, 0.55, 0.0))

    scene = pyrender.Scene(ambient_light=(0.35, 0.35, 0.35), bg_color=(1.0, 1.0, 1.0))
    extent = np.ptp(pelvis[:, :2], axis=0).max() / 2 + 3.0
    scene.add(pyrender.Mesh.from_trimesh(checkerboard(pelvis[:, :2].mean(0), extent, floor),
                                         smooth=False))
    camera = scene.add(pyrender.PerspectiveCamera(yfov=np.radians(40), aspectRatio=size[0] / size[1]))
    light = scene.add(pyrender.DirectionalLight(intensity=3.0))
    scene.add(pyrender.DirectionalLight(intensity=1.5), pose=look_at(np.array([1.0, 2.0, 5.0]), np.zeros(3)))
    renderer = pyrender.OffscreenRenderer(*size)
    strip, column = plot_strip(d, needed, 2 * size[0], 360)

    writer = imageio.get_writer(out, fps=fps / 2, codec="libx264", quality=8, macro_block_size=2)
    for t in range(len(q)):
        nodes = [scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices[t], faces, process=False),
                                                      material=skin))]
        for mask, radius, mat in ((contact[t], 0.018, on), (~contact[t], 0.009, off)):
            if mask.any():
                poses = np.tile(np.eye(4), (int(mask.sum()), 1, 1))
                poses[:, :3, :3] *= radius
                poses[:, :3, 3] = where[t, mask]
                nodes.append(scene.add(pyrender.Mesh.from_trimesh(sphere, material=mat, poses=poses)))
        mesh = arrows(where[t, contact[t]], force[t, contact[t]] / weight * ARROW_M_PER_BW)
        if mesh is not None:
            nodes.append(scene.add(pyrender.Mesh.from_trimesh(mesh, material=arrow_mat)))
        panels = []
        for offset in views:
            target = track[t] * np.array([1, 1, 0]) + UP * (floor + 0.8)
            pose = look_at(target + offset, target)
            scene.set_pose(camera, pose)
            scene.set_pose(light, pose)
            panels.append(renderer.render(scene)[0])
        for node in nodes:
            scene.remove_node(node)

        frame = np.concatenate([np.concatenate(panels, axis=1), strip], axis=0).copy()
        x = column(t / fps)
        cv2.line(frame, (x, size[1]), (x, frame.shape[0] - 1), (220, 0, 0), 1)
        total = force[t].sum(0) / weight
        text = (f"{d['motion']}  frame {t}/{len(q) - 1}  t={t / fps:.2f}s  (1/2 speed)  "
                f"frames in contact {int(contact[t].sum())}  force {total[2]:+.2f} BW up, "
                f"{np.linalg.norm(total[:2]):.2f} BW sideways")
        cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        writer.append_data(frame)
    writer.close()
    renderer.delete()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--step", default="all", choices=("label", "render", "all"))
    p.add_argument("--motions", type=Path, required=True,
                   help="an AMASS *_stageii.npz, or a directory of them")
    p.add_argument("--save-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=_REPO_ROOT / "configs" / "pipeline.toml")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    motions = sorted(args.motions.glob("*.npz")) if args.motions.is_dir() else [args.motions]
    if not motions:
        print(f"[label_amass] no motions under {args.motions}", flush=True)
        return 1
    args.save_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_toml(args.config)
    bedlam = vars(section(cfg, "bedlam", {
        "window_s": WINDOW_S, "still_mps": STILL_MPS, "enter_m": ENTER_M, "exit_m": EXIT_M,
        "min_span_s": MIN_SPAN_S, "max_tilt_deg": MAX_TILT_DEG}))
    contact_knobs = {k: float(bedlam[k]) for k in (
        "window_s", "still_mps", "enter_m", "exit_m", "min_span_s", "max_tilt_deg")}
    forces = section(cfg, "bedlam_forces", {
        "config": "configs/human_optimization.toml", "edge_speed_m_s": 0.0, "weights": {},
        "device": "cpu"})
    hoc = load_config(_REPO_ROOT / forces.config)
    force_knobs = {
        "device": torch.device(forces.device),
        "weights": {**hoc.weights_for("kindyn_1"),
                    **{k: float(v) for k, v in forces.weights.items()}},
        "dyn_opt": hoc.knobs["dynamics_optimizer"],
        "torque_table": hoc.knobs["torque"]["multipliers"],
        "half_angle_deg": float(hoc.knobs.get("friction", {}).get("half_angle_deg", 45.0)),
        "edge_speed_m_s": float(forces.edge_speed_m_s)}
    device = torch.device(section(cfg, "runtime", {"device": "cuda"}).device
                          if torch.cuda.is_available() else "cpu")
    body = load_body(device=device)

    stems = [m.name.removesuffix(".npz").removesuffix("_stageii") for m in motions]
    if args.step in ("label", "all"):
        force_body = load_body(device=force_knobs["device"])
        for motion, stem in zip(motions, stems):
            fields = label_motion(motion, body=body, force_body=force_body, device=device,
                                  contact_knobs=contact_knobs, force_knobs=force_knobs)
            np.savez_compressed(args.save_dir / f"{stem}_contacts.npz", **fields)
            weight = float(fields["total_mass_kg"]) * GRAVITY
            print(f"[label] {stem}: {fields['num_frames']} frames @ {fields['fps']:.0f} fps, "
                  f"floor z={float(fields['floor_height_m']):+.3f} m, mass "
                  f"{float(fields['total_mass_kg']):.1f} kg, frames in contact "
                  f"{fields['frame_contact'].any(1).mean():.1%}, root residual "
                  f"{np.linalg.norm(fields['root_residual'][:, :3], axis=1).mean() / weight:.3f} BW "
                  f"mean", flush=True)
        np.savez_compressed(
            args.save_dir / "contact_frame_vertices.npz",
            frame_ids=np.arange(len(body.contact_frame_names), dtype=np.int64),
            frame_names=np.array(body.contact_frame_names),
            vertex_ids=np.array(body.contact_frame_vertices, dtype=np.int64),
            parent_joint=np.array(body.contact_frame_parents, dtype=np.int64),
            parent_joint_name=np.array([body.joint_names[j] for j in body.contact_frame_parents]))
    if args.step in ("render", "all"):
        for stem in stems:
            out = args.save_dir / f"{stem}_contacts.mp4"
            render_motion(args.save_dir / f"{stem}_contacts.npz", out, body=body, device=device)
            print(f"[render] {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
