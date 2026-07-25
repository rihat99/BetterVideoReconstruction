"""Draw the pipeline's own contacts and forces (``kindyn_1``) onto a campus_board clip.

``render_prediction_overlays.py`` does this for the two external models. This is the
same picture for **our** reconstruction, so the two read side by side: same discs,
same arrows, same five stacked plots, same moving cursor.

What is on screen, per limb:

*A disc* at the limb's joint — the wrist for a hand, the foot for a foot. **Inner**
ring is the force plate, **outer** ring is what ``kindyn_1`` used: red for contact,
green for free, grey when that limb had no plate. So one colour means the two agree
and a two-tone disc is a disagreement.

*An arrow* along the contact force the solve put on the body, drawn in 3D: the joint
and the joint-plus-force are both projected through the clip's camera, so an arrow
shortens with distance the way a real vector would. A foot can show two arrows,
because the solve gives the foot and the ankle their own contact point.

*Five plots* beside the video — the sum over limbs, then one per limb. The plate
trace is black on its own 1000 Hz clock; the ``kindyn_1`` trace is in the limb's
colour, one point per video frame. Both are the same number: the **downward load** on
the hold, the part of the contact force pointing along gravity. That is the only
quantity a single-axis plate and a 3D solve can share.

Session 1 of campus_board instrumented the hands only (see ``peter/README.md``), so on
those clips a foot's inner ring is grey, its panel is shaded, and the sum panel adds
up the hands alone.

The mass is not an assumption here, unlike the prediction overlay: ``kindyn_1`` solved
its forces in newtons using the SMPL-X body's own mass, which is printed on the pane
because it is known to run a few kilos high.

Writes ``<clip>/human_optim/kindyn_forces_overlay.mp4``.

Usage:
    uv run --no-sync python scripts/diagnostics/render_kindyn_overlays.py
    uv run --no-sync python scripts/diagnostics/render_kindyn_overlays.py 07_peter_climbing
"""

from __future__ import annotations

import sys
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
    _draw_caption, _draw_contact_discs, _fit, _force_panel, _wired,
)
from score_pipeline_contacts import _limb_load, _limb_members  # noqa: E402

METRES_PER_NEWTON = 1.2e-3  # the arrow length the viewer uses for a dynamics force
MIN_ARROW_N = 1.0  # below this the arrow is under a pixel; the disc already says "in contact"
#: the force groups that make up each limb, in ``LIMBS`` order — a foot pushes through two points
LIMB_GROUPS = [["left_wrist"], ["right_wrist"], ["left_foot", "left_ankle"],
               ["right_foot", "right_ankle"]]
LIMB_REP_JOINT = ["left_wrist", "right_wrist", "left_foot", "right_foot"]


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


def render(clip: str) -> Path:
    """Write the ``kindyn_1`` overlay for one campus_board clip and return its path."""
    tree = CAMPUS_ROOT / clip
    kindyn = np.load(tree / "human_optim" / "kindyn_1.npz", allow_pickle=True)
    transform = np.load(tree / "geometry" / "transform.npz")
    plates = np.load(PLATE_ROOT / clip / "forces.npz")
    video = PLATE_ROOT / clip / "cam_left.mp4"

    frame_indices = transform["frame_indices"]
    n_frames = len(frame_indices)
    if not np.array_equal(frame_indices, np.arange(n_frames)):
        raise ValueError("pipeline frames are not the video frames — the plate clock cannot be aligned")
    people = [int(x) for x in kindyn["object_ids"]]
    if len(people) != 1:
        raise ValueError(f"the plates measure one climber but kindyn_1 holds {people}")
    truth = np.asarray(np.load(PLATE_ROOT / clip / "contacts.npz")["contacts"], np.int8)[:n_frames]
    if len(truth) < n_frames:
        raise ValueError(f"the plate log covers {len(truth)} frames but the clip has {n_frames}")

    joint_names = [str(name) for name in kindyn["joint_names"]]
    members = _limb_members(joint_names)                                  # (4,) joint columns per limb
    group_joints = [[joint_names.index(name) for name in limb] for limb in LIMB_GROUPS]
    rep_joints = [joint_names.index(name) for name in LIMB_REP_JOINT]
    joints_world = np.asarray(kindyn["joints_world"][0], float)           # (N, 52, 3)
    forces_world = np.asarray(kindyn["contact_forces_world"][0], float)   # (N, 52, 3)
    valid = np.asarray(kindyn["valid_mask"][0], bool)
    mass = float(kindyn["total_mass"][0])
    fps = float(kindyn["fps"])
    load = _limb_load(kindyn, 0)                                          # (N, 4) newtons

    # The label the solve used: its own if the stage saved it, else "a force was solved here".
    if "joint_contact" in kindyn.files:
        joint_contact = np.asarray(kindyn["joint_contact"][0], bool)
        contact = np.stack([joint_contact[:, limb].any(1) for limb in members], 1)
        label_note = "kindyn_1 contact"
    else:
        active = np.linalg.norm(np.asarray(kindyn["contact_forces"][0], float), axis=-1) > 1e-3
        groups = [str(name) for name in np.asarray(kindyn["contact_force_joints"]).reshape(-1)]
        contact = np.stack([active[:, [groups.index(n) for n in limb]].any(1)
                            for limb in LIMB_GROUPS], 1)
        label_note = "kindyn_1 force > 0"

    wired = _wired(clip)
    truth[:, ~wired] = -1  # a session-1 foot column is zero because there was no sensor
    panel, column_of, cursor_top, cursor_bottom = _force_panel(
        load[None], wired, fps, plates["forces"], float(plates["fps"]), 50.0)

    caption: list[tuple[tuple[int, int, int] | None, str]] = [
        (None, f"inner disc = force plate,  outer ring = {label_note}"),
        (None, f"arrows = kindyn_1 contact force,  {1 / METRES_PER_NEWTON:.0f} N per metre"),
        (None, f"graphs: downward load [N],  black = plate,  body mass {mass:.1f} kg"),
        (CONTACT_RGB, "contact"), (FREE_RGB, "no contact"),
    ]
    if bool((truth < 0).any()):
        caption.append((NODATA_RGB, "no plate on this limb"))
    caption += [(LIMB_RGB[limb], LIMBS[limb].replace("_", " ")) for limb in range(4)]

    capture = cv2.VideoCapture(str(video))
    source = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    pane_size = _fit(source, PANEL_SIZE[1])
    pane_scale = (pane_size[0] / source[0], pane_size[1] / source[1])

    path = tree / "human_optim" / "kindyn_forces_overlay.mp4"
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7)
    for frame_index in tqdm(range(n_frames), desc=f"[overlays] {clip}", leave=False):
        ok, bgr = capture.read()
        if not ok:
            break
        pane = cv2.resize(bgr, pane_size)[..., ::-1].copy()
        if valid[frame_index]:
            extrinsic = transform["extrinsics"][frame_index]
            intrinsic = transform["intrinsics_px_orig"][frame_index]
            pixels = _project(joints_world[frame_index][rep_joints], extrinsic, intrinsic, pane_scale)
            _draw_contact_discs(pane, pixels, contact[frame_index], truth[frame_index])
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


def _targets(names: list[str]) -> list[str]:
    """Resolve clip names, or list every campus_board clip whose dynamics stage ran."""
    if names:
        return names
    return sorted(clip.name for clip in CAMPUS_ROOT.iterdir()
                  if (clip / "human_optim" / "kindyn_1.npz").exists())


if __name__ == "__main__":
    for target in _targets(sys.argv[1:]):
        try:
            print(f"[overlays] {target} -> {render(target)}")
        except Exception as error:  # noqa: BLE001 — one bad clip must not kill the batch
            print(f"[overlays] {target}: FAILED — {type(error).__name__}: {error}")
