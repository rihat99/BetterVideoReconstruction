"""Draw the pipeline's own contacts and forces (``kindyn_1``) onto a measured clip.

``render_prediction_overlays.py`` does this for the two external models. This is the
same picture for **our** reconstruction, so the two read side by side: same discs,
same arrows, same five stacked plots, same moving cursor.

Two rigs are covered, and they measure very different things:

``peter/out_campus_board_single/`` — the campus board, where each limb pulls on its
own single-axis force plate. So every limb can be compared on its own, and the
comparable number is the **downward load** on the hold.

``peter/out_climb_wall_single/`` — the climbing wall, where four 3-axis cells hold up
**one shared structure**. A cell reading is that corner's reaction, not the limb it is
named after: a left-hand-only touch shows up on all four cells. Only the **sum** of
the cells is a clean force — the total the climber puts into the wall. So the wall
gets one comparison, on the total, and because both sides are 3D vectors whose
sideways direction is not well pinned down, they are compared by **size** alone.

What is on screen, per limb:

*A disc* at the limb's joint — the wrist for a hand, the foot for a foot. On the
campus board it is a two-ring disc: **inner** is the force plate, **outer** is what
``kindyn_1`` used, so one colour means the two agree and a two-tone disc is a
disagreement. Red is contact, green is free, grey means that limb had no plate. The
wall has no per-limb truth to show, so there it is a single white-rimmed disc holding
``kindyn_1``'s label alone — deliberately a different shape, so a reader never
mistakes a model-only overlay for perfect agreement.

*An arrow* along the contact force the solve put on the body, drawn in 3D: the joint
and the joint-plus-force are both projected through the clip's camera, so an arrow
shortens with distance the way a real vector would. One arrow is drawn per point the
solve pushes through, and which points those are is read from the file — today four,
one per limb, but it used to also give the ankles their own point.

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
shaded, because the wall never measured a single limb. The caption prints that clip's
own ratio and median angle between the two totals, so a trial whose cells were never
zeroed gives itself away instead of looking like a bad reconstruction.

The mass is not an assumption here, unlike the prediction overlay: ``kindyn_1`` solved
its forces in newtons using the SMPL-X body's own mass, which is printed on the pane
because it is known to run a few kilos high.

Writes ``<clip>/human_optim/kindyn_forces_overlay.mp4``.

Usage:
    uv run --no-sync python scripts/diagnostics/render_kindyn_overlays.py
    uv run --no-sync python scripts/diagnostics/render_kindyn_overlays.py 07_peter_climbing trial_10
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

_DIAGNOSTICS = Path(__file__).resolve().parent
if str(_DIAGNOSTICS) not in sys.path:
    sys.path.insert(0, str(_DIAGNOSTICS))

from render_prediction_overlays import (  # noqa: E402
    CAMPUS_ROOT, CONTACT_RGB, FREE_RGB, LIMB_RGB, LIMBS, NODATA_RGB, PANEL_SIZE, PLATE_ROOT,
    WALL_ROOT, WALL_SOURCE, _draw_caption, _draw_contact_discs, _fit, _force_panel,
    _wall_panel, _wired,
)
from score_pipeline_contacts import _limb_load, _limb_members  # noqa: E402

METRES_PER_NEWTON = 1.2e-3  # the arrow length the viewer uses for a dynamics force
MIN_ARROW_N = 1.0  # below this the arrow is under a pixel; the disc already says "in contact"
LIMB_REP_JOINT = ["left_wrist", "right_wrist", "left_foot", "right_foot"]  # where a disc sits


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


def _draw_force_arrows(frame: np.ndarray, joints: np.ndarray, forces: np.ndarray,
                       group_joints: list[list[int]], extrinsic: np.ndarray,
                       intrinsic: np.ndarray, scale: tuple[float, float]) -> None:
    """Draw one arrow per active contact group, in place, coloured by its limb.

    ``joints (52, 3)`` and ``forces (52, 3)`` are this frame's world joint positions
    and world contact forces; ``group_joints`` lists the joint columns of each limb.
    """
    for limb, columns in enumerate(group_joints):
        for column in columns:
            force = forces[column]
            if np.linalg.norm(force) < MIN_ARROW_N:
                continue
            ends = _project(np.stack([joints[column], joints[column] + force * METRES_PER_NEWTON]),
                            extrinsic, intrinsic, scale)
            if not np.isfinite(ends).all():
                continue
            tail, head = (int(round(ends[0][0])), int(round(ends[0][1]))), \
                         (int(round(ends[1][0])), int(round(ends[1][1])))
            cv2.arrowedLine(frame, tail, head, LIMB_RGB[limb], 3, cv2.LINE_AA, tipLength=0.22)


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


def render(clip: Clip) -> Path:
    """Write the ``kindyn_1`` overlay for one clip and return its path."""
    tree = clip.tree
    kindyn = np.load(tree / "human_optim" / "kindyn_1.npz", allow_pickle=True)
    transform = np.load(tree / "geometry" / "transform.npz")

    frame_indices = transform["frame_indices"]
    n_frames = len(frame_indices)
    if not np.array_equal(frame_indices, np.arange(n_frames)):
        raise ValueError("pipeline frames are not the video frames — the rig's clock cannot be aligned")
    people = [int(x) for x in kindyn["object_ids"]]
    if len(people) != 1:
        raise ValueError(f"the rig measures one climber but kindyn_1 holds {people}")
    truth = None if clip.plate_contact is None else clip.plate_contact[:n_frames]
    if truth is not None and len(truth) < n_frames:
        raise ValueError(f"the plate log covers {len(truth)} frames but the clip has {n_frames}")

    joint_names = [str(name) for name in kindyn["joint_names"]]
    members = _limb_members(joint_names)                                  # (4,) joint columns per limb
    # Where the solve actually pushes, as columns of the 52 body joints, folded to limbs.
    force_groups = [str(name) for name in np.asarray(kindyn["contact_force_joints"]).reshape(-1)]
    group_members = _limb_members(force_groups)
    group_joints = [[joint_names.index(force_groups[g]) for g in limb] for limb in group_members]
    rep_joints = [joint_names.index(name) for name in LIMB_REP_JOINT]
    joints_world = np.asarray(kindyn["joints_world"][0], float)           # (N, 52, 3)
    forces_world = np.asarray(kindyn["contact_forces_world"][0], float)   # (N, 52, 3)
    valid = np.asarray(kindyn["valid_mask"][0], bool)
    mass = float(kindyn["total_mass"][0])
    fps = float(kindyn["fps"])

    # The label the solve used: its own if the stage saved it, else "a force was solved here".
    if "joint_contact" in kindyn.files:
        joint_contact = np.asarray(kindyn["joint_contact"][0], bool)
        contact = np.stack([joint_contact[:, limb].any(1) for limb in members], 1)
        label_note = "kindyn_1 contact"
    else:
        active = np.linalg.norm(np.asarray(kindyn["contact_forces"][0], float), axis=-1) > 1e-3
        contact = np.stack([active[:, limb].any(1) for limb in group_members], 1)
        label_note = "kindyn_1 force > 0"

    caption: list[tuple[tuple[int, int, int] | None, str]] = []
    if clip.measured_total is None:
        panel, column_of, cursor_top, cursor_bottom = _force_panel(
            _limb_load(kindyn, 0)[None], clip.wired, fps,
            clip.plate_newton, clip.plate_fps, 50.0)
        caption += [
            (None, f"inner disc = force plate,  outer ring = {label_note}"),
            (None, f"arrows = kindyn_1 contact force,  {1 / METRES_PER_NEWTON:.0f} N per metre"),
            (None, f"graphs: downward load [N],  black = plate,  body mass {mass:.1f} kg"),
        ]
    else:
        # The wall's four cells hold up one structure, so the only comparable number is
        # the total, and only its size: the fit that put the cells into world axes barely
        # pins down the sideways direction. The clip's own agreement is printed rather
        # than gated, so an unzeroed trial shows itself instead of blaming the solve.
        solved = np.stack([forces_world[:, limb].sum(1) for limb in group_joints], 1)  # (N, 4, 3)
        limb_size = np.linalg.norm(solved, axis=-1)
        model_total = np.linalg.norm(solved.sum(1), axis=-1)
        measured = clip.measured_total[:n_frames]
        measured_size = np.linalg.norm(measured, axis=-1)
        panel, column_of, cursor_top, cursor_bottom = _wall_panel(
            limb_size, np.linalg.norm(clip.measured_cell[:n_frames], axis=-1),
            model_total, measured_size, fps, model_label="solve")
        both = (model_total > 1.0) & (measured_size > 1.0)
        angle = np.degrees(np.arccos(np.clip(
            (solved.sum(1)[both] * measured[both]).sum(1)
            / (model_total[both] * measured_size[both]), -1.0, 1.0)))
        caption += [
            (None, f"disc = {label_note}  (the wall measures no single limb)"),
            (None, f"arrows = kindyn_1 contact force,  {1 / METRES_PER_NEWTON:.0f} N per metre"),
            (None, f"graphs: force size [N],  black = measured,  body mass {mass:.1f} kg"),
            (None, "limb graphs: black is that CORNER's cell, not that limb's force"),
            (None, f"wall total vs solve:  size ratio "
                   f"{model_total.mean() / measured_size.mean():.2f},"
                   f"  median angle {np.median(angle):.0f} deg"),
        ]
    caption += [(CONTACT_RGB, "contact"), (FREE_RGB, "no contact")]
    if truth is not None and bool((truth < 0).any()):
        caption.append((NODATA_RGB, "no plate on this limb"))
    caption += [(LIMB_RGB[limb], LIMBS[limb].replace("_", " ")) for limb in range(4)]

    capture = cv2.VideoCapture(str(clip.video))
    source = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    pane_size = _fit(source, PANEL_SIZE[1])
    pane_scale = (pane_size[0] / source[0], pane_size[1] / source[1])

    path = tree / "human_optim" / "kindyn_forces_overlay.mp4"
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7)
    for frame_index in tqdm(range(n_frames), desc=f"[overlays] {clip.name}", leave=False):
        ok, bgr = capture.read()
        if not ok:
            break
        pane = cv2.resize(bgr, pane_size)[..., ::-1].copy()
        if valid[frame_index]:
            extrinsic = transform["extrinsics"][frame_index]
            intrinsic = transform["intrinsics_px_orig"][frame_index]
            pixels = _project(joints_world[frame_index][rep_joints], extrinsic, intrinsic, pane_scale)
            _draw_contact_discs(pane, pixels, contact[frame_index],
                                None if truth is None else truth[frame_index])
            _draw_force_arrows(pane, joints_world[frame_index], forces_world[frame_index],
                               group_joints, extrinsic, intrinsic, pane_scale)
        _draw_caption(pane, caption)

        cursor = panel.copy()
        column = np.clip(column_of(frame_index / fps), 1, panel.shape[1] - 2)
        cursor[cursor_top:cursor_bottom, column - 1:column + 2] = (205, 30, 30)
        writer.append_data(np.hstack([pane, cursor]))

    capture.release()
    writer.close()
    return path


TREES = [(CAMPUS_ROOT, campus_clip), (WALL_ROOT, wall_clip)]


def _targets(names: list[str]) -> list[Clip]:
    """Resolve clip names against both trees, or list every clip whose dynamics stage ran."""
    if not names:
        return [build(clip.name) for root, build in TREES for clip in sorted(root.iterdir())
                if (clip / "human_optim" / "kindyn_1.npz").exists()]
    clips = []
    for name in names:
        match = next((build for root, build in TREES
                      if (root / name / "human_optim" / "kindyn_1.npz").exists()), None)
        if match is None:
            roots = ", ".join(str(root) for root, _ in TREES)
            raise SystemExit(f"[overlays] no kindyn_1 for {name!r} under any of: {roots}")
        clips.append(match(name))
    return clips


if __name__ == "__main__":
    for target in _targets(sys.argv[1:]):
        try:
            print(f"[overlays] {target.name} -> {render(target)}")
        except Exception as error:  # noqa: BLE001 — one bad clip must not kill the batch
            print(f"[overlays] {target.name}: FAILED — {type(error).__name__}: {error}")
