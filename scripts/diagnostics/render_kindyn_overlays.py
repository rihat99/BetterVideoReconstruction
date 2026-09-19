"""Draw the pipeline's own contacts and forces (``kindyn_1``) onto a measured clip.

``render_prediction_overlays.py`` does this for the two external models. This is the
same picture for **our** reconstruction, so the two read side by side: same discs,
same arrows, same five stacked plots, same moving cursor.

Four rigs are covered, and they measure different things:

``peter/out_campus_board_single/`` — the campus board, where each limb pulls on its
own single-axis force plate. So every limb can be compared on its own, and the
comparable number is the **downward load** on the hold.

``peter/out_climb_wall_single/`` — the climbing wall, where four 3-axis cells hold up
**one shared structure**. A cell reading is that corner's reaction, not the limb it is
named after: a left-hand-only touch shows up on all four cells. Only the **sum** of
the cells is a clean force — the total the climber puts into the wall. So the wall
gets one comparison, on the total, and because both sides are 3D vectors whose
sideways direction is not well pinned down, they are compared by **size** alone.

``peter/out_climb_wall_2_single/`` — the demonstrator wall, where each limb has its
**own instrumented board**, so a channel IS that limb's contact force (a released
limb reads ~0; verified at import). Every limb panel is a real comparison here:
black is the limb's own board, nothing is shaded, and the sizes compare in world
axes because the board frame is pinned by the datasheet rotation. Each limb panel
also carries the **direction error** — the angle between the solve's and the
board's force vectors — as a dashed line in the limb's own colour on its own
0–180° right-hand axis, drawn only where both sides read above 50 N (a
near-zero vector has no direction). Per-limb contact
truth comes from the measured force the same way the campus plates' labels did —
bigger than 50 N. The climb_wall_2 recipe restricts contact to hand and foot
frames (``configs/human_optimization_peter.toml``), so every newton the solve
applies is inside one of the four limb panels — yet the top panel is still not the
four below it added up, because both totals are the length of a *vector sum* and
limb forces pointing against each other partially cancel in it.

``peter/out_climb_wall_3_single/`` — the same demonstrator wall, 2026-09 session
(two climbers, wall at 0 / 16 / 32 degrees, the camera moved between the three).
Read exactly like climb_wall_2: ``sensor_forces.npz`` written by
``scripts/peter_sensor_forces.py`` with that inclination's own datasheet rotation.

``peter/climb_wall_3_gt/human/`` — the climb_wall_3 clips reconstructed from all THREE
calibrated cameras (the ground-truth chain, ``scripts/peter_gt/``). Drawn on the
undistorted ``cam_left`` view, read exactly like climb_wall_3; ``sensor_forces.npz``
sits at the clip root there.

``parkour/out/`` — the LAAS Parkour dataset, where the feet land on floor force
plates and the hands grab an instrumented bar, so a channel IS that limb's 3D
contact force: the limb panels compare directly, like the demonstrator wall.
One difference: the measured vectors live in the MoCap world, whose yaw is not
our world's yaw, so only their **sizes** are compared — no measured arrows
are drawn on the video and no angle statistic is printed. Frames where a foot
landed off the plates carry no measurement: the black trace gaps there and that
limb's inner ring goes grey.

What is on screen, per limb:

*A disc* at the limb's joint — the wrist for a hand, the foot for a foot. On the
campus board it is a two-ring disc: **inner** is the force plate, **outer** is what
``kindyn_1`` used, so one colour means the two agree and a two-tone disc is a
disagreement. Red is contact, green is free, grey means that limb had no plate. The
wall has no per-limb truth to show, so there it is a single white-rimmed disc holding
``kindyn_1``'s label alone — deliberately a different shape, so a reader never
mistakes a model-only overlay for perfect agreement.

*An arrow* along the contact force the solve put on the limb — everything it pushes
through the hand's points summed into one vector, likewise for each foot — drawn in
3D: the joint and the joint-plus-force are both projected through the clip's camera,
so an arrow shortens with distance the way a real vector would. Force the solve puts
elsewhere on the body (knees, torso, …) is deliberately not drawn. Every solve
arrow is **green** and every sensor arrow **red** — which limb an arrow belongs
to, its position already says. On the demonstrator wall the sensor's own limb
force draws from the same joint, so size and direction compare on the video
itself. A tiny dot-and-word legend in the corner is the pane's only text.

*Five plots* beside the video — the total first, then one per limb, with the
``kindyn_1`` trace in the limb's colour, one point per video frame.

On the campus board every panel is a **downward load** in newtons, the part of the
contact force pointing along gravity, and the plate's own trace is drawn in black
behind it on its 1000 Hz clock. That is the only quantity a single-axis plate and a
3D solve can share. Session 1 instrumented the hands only (see ``peter/README.md``),
so there a foot's inner ring is grey, its panel is shaded, and the total adds up the
hands alone.

On the wall every panel is instead the plain **size** of a force. Only the top panel
can be compared, and it holds two sizes: the wall's measured total in black against
the size of the solve's total. Both are the length of a vector sum, so the top panel
is deliberately *not* the four panels below it added up. The four limb panels are
shaded, because the wall never measured a single limb. Each clip's ratio and median
angle between the two totals is printed to the console, so a trial whose cells were
never zeroed gives itself away instead of looking like a bad reconstruction.

Writes ``<clip>/human_optim/kindyn_forces_overlay.mp4``.

Naming an optimization directory instead of a clip draws that run: pass
``<clip>/human_optim_gt_contact`` for the board-labelled variant chain
(``docs/board_contacts.md``), and the overlay lands beside its own ``kindyn_1.npz``.

Usage:
    uv run --no-sync python scripts/diagnostics/render_kindyn_overlays.py
    uv run --no-sync python scripts/diagnostics/render_kindyn_overlays.py 07_peter_climbing trial_10
    uv run --no-sync python scripts/diagnostics/render_kindyn_overlays.py \
        peter/out_climb_wall_3_single/a_00_01_part1/human_optim_gt_contact
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

_DIAGNOSTICS = Path(__file__).resolve().parent
if str(_DIAGNOSTICS) not in sys.path:
    sys.path.insert(0, str(_DIAGNOSTICS))

from render_prediction_overlays import (  # noqa: E402
    CAMPUS_ROOT, LIMBS, PANEL_SIZE, PETER, PLATE_ROOT, WALL_ROOT, WALL_SOURCE,
    _draw_contact_discs, _fit, _force_panel, _wall_panel, _wired,
)
from score_pipeline_contacts import _limb_load, _limb_members  # noqa: E402

WALL2_SOURCE = PETER / "climb_wall_2"
WALL2_ROOT = PETER / "out_climb_wall_2_single"
WALL3_SOURCE = PETER / "climb_wall_3"
WALL3_ROOT = PETER / "out_climb_wall_3_single"
GT_ROOT = PETER / "climb_wall_3_gt" / "human"
PARKOUR_SOURCE = Path("/home/rikhat.akizhanov/better/data/Parkour-dataset/processed")
PARKOUR_ROOT = PARKOUR_SOURCE / "out"

METRES_PER_NEWTON = 1.2e-3  # the arrow length the viewer uses for a dynamics force
MIN_ARROW_N = 1.0  # below this the arrow is under a pixel; the disc already says "in contact"
BOARD_CONTACT_N = 50.0  # measured force above this is "in contact" — the rule the campus GT used
LIMB_REP_JOINT = ["left_wrist", "right_wrist", "left_foot", "right_foot"]  # where a disc sits
SOLVE_RGB = (45, 200, 70)  # every kindyn arrow — green; the disc already says which limb
MEASURED_RGB = (230, 40, 40)  # every sensor arrow — red


@dataclass(frozen=True)
class Clip:
    """One clip to render, and whatever its rig measured to compare against."""

    name: str
    video: Path  #: the source video
    tree: Path  #: the out tree holding geometry/ and human_optim/
    #: ``(N, 4)`` int8 per-limb contact truth: ``1`` contact, ``0`` free, ``-1`` no plate
    plate_contact: np.ndarray | None = None
    plate_newton: np.ndarray | None = None  #: ``(T, 4)`` plate log, downward load in newtons
    plate_fps: float = 0.0
    wired: np.ndarray | None = None  #: ``(4,)`` which limbs had a plate
    #: ``(N, 3)`` world-frame total the wall measured, in the reaction sense; wall only
    measured_total: np.ndarray | None = None
    #: ``(N, 4, 3)`` the same for each corner cell, in ``LIMBS`` order; wall only
    measured_cell: np.ndarray | None = None
    #: True when each channel is that limb's own board (``climb_wall_2``) rather than
    #: a corner reaction of one shared structure — limb panels then compare directly
    limbs_measured: bool = False
    #: False when the measured vectors are in some other frame (parkour's MoCap
    #: world) — sizes still compare, but arrows and angles would be meaningless
    measured_in_world: bool = True
    #: Which optimization directory under ``tree`` to draw — ``human_optim`` normally,
    #: ``human_optim_gt_contact`` for the board-labelled variant (docs/board_contacts.md)
    optim_dir: str = "human_optim"


def _project(points: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray,
             scale: tuple[float, float]) -> np.ndarray:
    """Project world points ``(..., 3)`` to pane pixels ``(..., 2)``, NaN behind the camera.

    ``extrinsic`` is OpenCV cam-from-world and ``intrinsic`` the pixel matrix at the
    original video size, so ``scale`` rescales to the drawing pane. The pane's width
    is rounded to a multiple of 16, which is why x and y carry separate scales.
    """
    camera = points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    depth = np.where(camera[..., 2] > 1e-3, camera[..., 2], np.nan)
    return np.stack([(intrinsic[0, 0] * camera[..., 0] / depth + intrinsic[0, 2]) * scale[0],
                     (intrinsic[1, 1] * camera[..., 1] / depth + intrinsic[1, 2]) * scale[1]], -1)


def _draw_force_arrows(frame: np.ndarray, anchors: np.ndarray, forces: np.ndarray,
                       colours: list[tuple[int, int, int]], extrinsic: np.ndarray,
                       intrinsic: np.ndarray, scale: tuple[float, float],
                       thickness: int = 3) -> None:
    """Draw one arrow per limb from its anchor joint, in place.

    ``anchors (4, 3)`` are this frame's world limb-joint positions and ``forces
    (4, 3)`` one world force vector per limb, coloured by ``colours``.
    """
    for anchor, force, colour in zip(anchors, forces, colours):
        if np.linalg.norm(force) < MIN_ARROW_N:
            continue
        ends = _project(np.stack([anchor, anchor + force * METRES_PER_NEWTON]),
                        extrinsic, intrinsic, scale)
        if not np.isfinite(ends).all():
            continue
        tail, head = (int(round(ends[0][0])), int(round(ends[0][1]))), \
                     (int(round(ends[1][0])), int(round(ends[1][1])))
        cv2.arrowedLine(frame, tail, head, colour, thickness, cv2.LINE_AA, tipLength=0.22)


def _draw_legend(frame: np.ndarray, entries: list[tuple[tuple[int, int, int], str]]) -> None:
    """A tiny corner legend — one dot + word per entry, nothing else."""
    for row, (colour, text) in enumerate(entries):
        y = 16 + 17 * row
        cv2.circle(frame, (12, y), 4, colour, -1, cv2.LINE_AA)
        cv2.putText(frame, text, (22, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (22, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)


def campus_clip(name: str) -> Clip:
    """Describe a campus_board clip — one single-axis force plate per limb."""
    plates = PLATE_ROOT / name
    wired = _wired(name)
    truth = np.asarray(np.load(plates / "contacts.npz")["contacts"], np.int8)
    truth[:, ~wired] = -1  # a session-1 foot column is zero because there was no sensor
    forces = np.load(plates / "forces.npz")
    return Clip(name=name, video=plates / "cam_left.mp4", tree=CAMPUS_ROOT / name,
                plate_contact=truth, plate_newton=forces["forces"],
                plate_fps=float(forces["fps"]), wired=wired)


def wall_clip(name: str) -> Clip:
    """Describe a climb_wall trial — four cells under one structure, so only a total.

    ``sensor_forces.npz`` is what ``scripts/peter_sensor_forces.py`` wrote: the cell
    readings rotated into the reconstruction's world frame and flipped to the reaction
    sense, one row per video frame, so its total is directly comparable to the solve's.
    Its channels come in ``LIMBS`` order, which is checked rather than assumed — drawing
    a cell under the wrong limb would be invisible and wrong.
    """
    sensor = np.load(WALL_ROOT / name / "human_optim" / "sensor_forces.npz")
    channels = [str(limb) for limb in sensor["limbs"]]
    if channels != LIMBS:
        raise ValueError(f"sensor channels are {channels}, expected {LIMBS}")
    return Clip(name=name, video=WALL_SOURCE / name / "cam_left.mp4", tree=WALL_ROOT / name,
                measured_total=np.asarray(sensor["total_world"], float),
                measured_cell=np.asarray(sensor["reaction_world"], float))


def wall2_clip(name: str) -> Clip:
    """Describe a climb_wall_2 trial — the demonstrator rig, one board per limb.

    Unlike the first wall, a channel here IS that limb's contact force, so the limb
    panels compare directly, and per-limb contact truth comes from the measured force
    the same way the campus plates' labels did: bigger than ``BOARD_CONTACT_N``.
    """
    sensor = np.load(WALL2_ROOT / name / "human_optim" / "sensor_forces.npz")
    channels = [str(limb) for limb in sensor["limbs"]]
    if channels != LIMBS:
        raise ValueError(f"sensor channels are {channels}, expected {LIMBS}")
    measured = np.asarray(sensor["reaction_world"], float)
    truth = (np.linalg.norm(measured, axis=-1) > BOARD_CONTACT_N).astype(np.int8)
    return Clip(name=name, video=WALL2_SOURCE / name / "cam_left.mp4", tree=WALL2_ROOT / name,
                plate_contact=truth,
                measured_total=np.asarray(sensor["total_world"], float),
                measured_cell=measured, limbs_measured=True)


def wall3_clip(name: str) -> Clip:
    """Describe a climb_wall_3 clip — the demonstrator rig again, one board per limb.

    Same reading as :func:`wall2_clip`; the rotation inside ``sensor_forces.npz`` is
    the one for this clip's wall inclination.
    """
    sensor = np.load(WALL3_ROOT / name / "human_optim" / "sensor_forces.npz")
    channels = [str(limb) for limb in sensor["limbs"]]
    if channels != LIMBS:
        raise ValueError(f"sensor channels are {channels}, expected {LIMBS}")
    measured = np.asarray(sensor["reaction_world"], float)
    truth = (np.linalg.norm(measured, axis=-1) > BOARD_CONTACT_N).astype(np.int8)
    return Clip(name=name, video=WALL3_SOURCE / name / "cam_left.mp4", tree=WALL3_ROOT / name,
                plate_contact=truth,
                measured_total=np.asarray(sensor["total_world"], float),
                measured_cell=measured, limbs_measured=True)


def gt_clip(name: str) -> Clip:
    """Describe a climb_wall_3 multi-view ground-truth clip — the same boards as
    :func:`wall3_clip`, drawn on its undistorted ``cam_left`` video."""
    sensor = np.load(GT_ROOT / name / "sensor_forces.npz", allow_pickle=True)
    channels = [str(limb) for limb in sensor["limbs"]]
    if channels != LIMBS:
        raise ValueError(f"sensor channels are {channels}, expected {LIMBS}")
    measured = np.asarray(sensor["reaction_world"], float)
    truth = (np.linalg.norm(measured, axis=-1) > BOARD_CONTACT_N).astype(np.int8)
    return Clip(name=name, video=GT_ROOT / name / "views" / "cam_left" / "video.mp4",
                tree=GT_ROOT / name, plate_contact=truth,
                measured_total=np.asarray(sensor["total_world"], float),
                measured_cell=measured, limbs_measured=True)


def parkour_clip(name: str) -> Clip:
    """Describe a LAAS Parkour clip — floor plates under the feet, a sensed bar in the hands.

    A channel IS that limb's 3D contact force (see ``parkour/README.md``), already one
    row per video frame, so the limb panels compare directly like the demonstrator
    wall's. But the vectors live in the MoCap world, not ours, so only sizes compare
    (``measured_in_world=False``). Frames the rig could not measure — a foot landing
    off the plates — are NaN, so the black trace gaps instead of drawing a bad zero;
    the total is NaN whenever any limb is unmeasured, because a sum with a missing
    limb is not the total. Contact truth is the import's own 50 N labels, with the
    unmeasured frames marked "no data" rather than "free".
    """
    gt = np.load(PARKOUR_SOURCE / name / "forces.npz")
    channels = [str(limb) for limb in gt["limbs"]]
    if channels != LIMBS:
        raise ValueError(f"force channels are {channels}, expected {LIMBS}")
    measured = np.asarray(gt["forces"][:, :, :3], float)
    captured = np.asarray(gt["captured"], bool)
    truth = np.asarray(np.load(PARKOUR_SOURCE / name / "contacts.npz")["contacts"], np.int8)
    truth[~captured] = -1
    total = measured.sum(1)
    total[~captured.all(1)] = np.nan
    measured[~captured] = np.nan
    return Clip(name=name, video=PARKOUR_SOURCE / name / "video.mp4", tree=PARKOUR_ROOT / name,
                plate_contact=truth, measured_total=total, measured_cell=measured,
                limbs_measured=True, measured_in_world=False)


def render(clip: Clip) -> Path:
    """Write the ``kindyn_1`` overlay for one clip and return its path."""
    tree = clip.tree
    optim = tree / clip.optim_dir
    kindyn = np.load(optim / "kindyn_1.npz", allow_pickle=True)
    transform = np.load(tree / "geometry" / "transform.npz")

    frame_indices = transform["frame_indices"]
    n_frames = len(frame_indices)
    if not np.array_equal(frame_indices, np.arange(n_frames)):
        raise ValueError("pipeline frames are not the video frames — the rig's clock cannot be aligned")
    # The multi-view GT body is a few frames shorter than the video (cut to the slowest camera's lag).
    n_frames = min(n_frames, int(kindyn["num_frames"]))
    people = [int(x) for x in kindyn["object_ids"]]
    if len(people) != 1:
        raise ValueError(f"the rig measures one climber but kindyn_1 holds {people}")
    truth = None if clip.plate_contact is None else clip.plate_contact[:n_frames]
    if truth is not None and len(truth) < n_frames:
        raise ValueError(f"the plate log covers {len(truth)} frames but the clip has {n_frames}")

    joint_names = [str(name) for name in kindyn["joint_names"]]
    members = _limb_members(joint_names)                                  # (4,) joint columns per limb
    # Where the solve actually pushes, as columns of the 52 body joints, folded to limbs.
    if "contact_force_joints" in kindyn.files:
        # The older solve pushed through a few named joints; read which ones.
        force_groups = [str(name) for name in np.asarray(kindyn["contact_force_joints"]).reshape(-1)]
        group_members = _limb_members(force_groups)
        group_joints = [[joint_names.index(force_groups[g]) for g in limb] for limb in group_members]
    else:
        # The frame-level solve saves per-joint forces with the contact-frame forces
        # already folded onto their parent joints, so the limbs fold off the 52 joints.
        group_members, group_joints = members, members
    rep_joints = [joint_names.index(name) for name in LIMB_REP_JOINT]
    joints_world = np.asarray(kindyn["joints_world"][0], float)           # (N, 52, 3)
    forces_world = np.asarray(kindyn["contact_forces_world"][0], float)   # (N, 52, 3)
    # One summed force per limb — what the arrows draw and the wall panels size up.
    # Points no limb owns (knees, torso, …) are left out; the total still counts them.
    limb_force = np.stack([forces_world[:, limb].sum(1) for limb in group_joints], 1)  # (N, 4, 3)
    valid = np.asarray(kindyn["valid_mask"][0], bool)
    fps = float(kindyn["fps"])

    # The label the solve used: its own if the stage saved it, else "a force was solved here".
    if "joint_contact" in kindyn.files:
        joint_contact = np.asarray(kindyn["joint_contact"][0], bool)
        contact = np.stack([joint_contact[:, limb].any(1) for limb in members], 1)
    else:
        active = np.linalg.norm(np.asarray(kindyn["contact_forces"][0], float), axis=-1) > 1e-3
        contact = np.stack([active[:, limb].any(1) for limb in group_members], 1)

    if clip.measured_total is None:
        panel, column_of, cursor_top, cursor_bottom = _force_panel(
            _limb_load(kindyn, 0)[None], clip.wired, fps,
            clip.plate_newton, clip.plate_fps, 50.0)
    else:
        # Either wall rig. The solve's total takes every joint it pushed through — on
        # the frame-level solve that includes points no limb owns (knees, torso, …),
        # which sit over no board. The clip's own agreement is printed to the console,
        # not gated, so an unzeroed trial shows itself instead of blaming the solve.
        limb_size = np.linalg.norm(limb_force, axis=-1)
        total_vec = forces_world.sum(1)
        model_total = np.linalg.norm(total_vec, axis=-1)
        measured = clip.measured_total[:n_frames]
        measured_size = np.linalg.norm(measured, axis=-1)
        cell = clip.measured_cell[:n_frames]
        cell_size = np.linalg.norm(cell, axis=-1)
        # Per-limb direction error — only where a channel IS that limb's force in our
        # world frame, and only where both sides carry a real force (a near-zero
        # vector has no direction, so its angle would be noise).
        limb_angle = None
        if clip.limbs_measured and clip.measured_in_world:
            both = (limb_size > BOARD_CONTACT_N) & (cell_size > BOARD_CONTACT_N)
            limb_angle = np.full_like(limb_size, np.nan)
            limb_angle[both] = np.degrees(np.arccos(np.clip(
                (limb_force[both] * cell[both]).sum(-1)
                / (limb_size[both] * cell_size[both]), -1.0, 1.0)))
        panel, column_of, cursor_top, cursor_bottom = _wall_panel(
            limb_size, cell_size, model_total, measured_size, fps, model_label="solve",
            limbs_measured=clip.limbs_measured,
            measured_label=None if clip.measured_in_world else "plate",
            total_label="wall total" if clip.measured_in_world else "plate total",
            limb_angle=limb_angle)
        if clip.measured_in_world:
            both = (model_total > 1.0) & (measured_size > 1.0)
            angle = np.degrees(np.arccos(np.clip(
                (total_vec[both] * measured[both]).sum(1)
                / (model_total[both] * measured_size[both]), -1.0, 1.0)))
            print(f"[overlays] {clip.name}: total size ratio "
                  f"{model_total.mean() / measured_size.mean():.2f}, "
                  f"median angle {np.median(angle):.0f} deg")
        else:
            # The rig's frame is not ours (MoCap yaw unknown): sizes compare, angles do not.
            finite = np.isfinite(measured_size)
            print(f"[overlays] {clip.name}: total size ratio "
                  f"{model_total[finite].mean() / measured_size[finite].mean():.2f} "
                  f"(sizes only - the measured vectors are not in our world frame)")

    # The sensor's own per-limb force, drawn beside the solve's — only where
    # a channel IS a limb force (the demonstrator wall), not where it is a corner cell,
    # and only when its vectors share our world frame (parkour's do not).
    measured_limb = (clip.measured_cell[:n_frames]
                     if clip.limbs_measured and clip.measured_in_world else None)

    legend = [(SOLVE_RGB, "kindyn")]
    if measured_limb is not None:
        legend.append((MEASURED_RGB, "sensor"))

    capture = cv2.VideoCapture(str(clip.video))
    source = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    pane_size = _fit(source, PANEL_SIZE[1])
    pane_scale = (pane_size[0] / source[0], pane_size[1] / source[1])

    path = optim / "kindyn_forces_overlay.mp4"
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7)
    for frame_index in tqdm(range(n_frames), desc=f"[overlays] {clip.name}", leave=False):
        ok, bgr = capture.read()
        if not ok:
            break
        pane = cv2.resize(bgr, pane_size)[..., ::-1].copy()
        if valid[frame_index]:
            extrinsic = transform["extrinsics"][frame_index]
            intrinsic = transform["intrinsics_px_orig"][frame_index]
            anchors = joints_world[frame_index][rep_joints]
            pixels = _project(anchors, extrinsic, intrinsic, pane_scale)
            _draw_contact_discs(pane, pixels, contact[frame_index],
                                None if truth is None else truth[frame_index])
            if measured_limb is not None:  # the sensor first, so the solve draws on top
                _draw_force_arrows(pane, anchors, measured_limb[frame_index],
                                   [MEASURED_RGB] * 4, extrinsic, intrinsic, pane_scale,
                                   thickness=2)
            _draw_force_arrows(pane, anchors, limb_force[frame_index], [SOLVE_RGB] * 4,
                               extrinsic, intrinsic, pane_scale)
        _draw_legend(pane, legend)

        cursor = panel.copy()
        column = np.clip(column_of(frame_index / fps), 1, panel.shape[1] - 2)
        cursor[cursor_top:cursor_bottom, column - 1:column + 2] = (205, 30, 30)
        writer.append_data(np.hstack([pane, cursor]))

    capture.release()
    writer.close()
    return path


TREES = [(CAMPUS_ROOT, campus_clip), (WALL_ROOT, wall_clip), (WALL2_ROOT, wall2_clip),
         (WALL3_ROOT, wall3_clip), (GT_ROOT, gt_clip), (PARKOUR_ROOT, parkour_clip)]


def _targets(names: list[str]) -> list[Clip]:
    """Resolve clip names (or clip directories) against the trees, or list every clip whose dynamics stage ran."""
    if not names:
        return [build(clip.name) for root, build in TREES for clip in sorted(root.iterdir())
                if (clip / "human_optim" / "kindyn_1.npz").exists()]
    clips = []
    for name in names:
        # A clip directory picks its own tree (a name alone takes the first tree holding it —
        # climb_wall_3 clips exist both single-view and in the multi-view GT tree). An
        # optimization directory holding a kindyn_1 picks BOTH its clip and which run to
        # draw, which is how the board-labelled variant is asked for.
        path, optim_dir = Path(name), "human_optim"
        if path.is_dir() and (path / "kindyn_1.npz").exists():
            path, optim_dir = path.parent, path.name
        if path.is_dir():
            clip_dir = path.resolve()
            match = next((build for root, build in TREES if clip_dir.parent == root.resolve()), None)
            name = clip_dir.name
        else:
            match = next((build for root, build in TREES
                          if (root / name / optim_dir / "kindyn_1.npz").exists()), None)
        if match is None:
            roots = ", ".join(str(root) for root, _ in TREES)
            raise SystemExit(f"[overlays] no kindyn_1 for {name!r} under any of: {roots}")
        clips.append(replace(match(name), optim_dir=optim_dir))
    return clips


if __name__ == "__main__":
    for target in _targets(sys.argv[1:]):
        try:
            print(f"[overlays] {target.name} -> {render(target)}")
        except Exception as error:  # noqa: BLE001 — one bad clip must not kill the batch
            print(f"[overlays] {target.name}: FAILED — {type(error).__name__}: {error}")
