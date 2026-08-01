"""Draw the supervised six-group force model (``forces_sup.npz``) onto a measured clip.

``render_prediction_overlays.py`` covers the earlier four-limb force model and
``render_kindyn_overlays.py`` the pipeline's own solve. This is the same picture for the
**supervised** model — the one trained directly against the corpus kindyn forces — so all
three overlays read side by side: same arrows, same five stacked plots, same moving cursor.

The prediction file differs from ``forces.npz`` in two ways this script leans on:

*Six groups, not four.* The model predicts one force per kindyn contact group —
``left_hand, right_hand, left_foot (big-toe), right_foot, left_ankle (heel),
right_ankle`` — so a foot's load is split between its toe and its heel. Every arrow is
drawn on its own (the heels get their own colours), and for the plots the six groups are
folded back to the four plate limbs: hand = hand, foot = toe + heel.

*World-frame forces via the body root.* The model reports in the kindyn body-root frame;
``predict_reconstruction.py`` already rotated a copy into the reconstruction's world axes
with the kindyn root quaternion (``forces_world``, body-weight units, NaN where the
dynamics stage had no valid body). Because both the model's forces and the rigs' measured
forces then live in world axes, the wall comparison can print a median angle the way the
kindyn overlay does — something the local-world-aligned prediction overlay could not.

The two rigs are compared exactly as in the other overlays: on the campus board every
panel is a **downward load** in newtons against that limb's plate (session 1 wired the
hands only — those foot panels are shaded); on the wall only the **total** is a clean
quantity, compared by size, with each corner cell's trace shaded inside its limb panel.
Body-weight units become newtons through the SMPL-X mass in ``human_optim/kindyn_1.npz``,
printed on the pane because it is known to run a few kilos high.

Writes ``<clip>/predictions/forces_sup_overlay.mp4``.

Usage:
    uv run --no-sync python scripts/diagnostics/render_sup_force_overlays.py
    uv run --no-sync python scripts/diagnostics/render_sup_force_overlays.py 01_hanging trial_10
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

from render_kindyn_overlays import Clip, campus_clip, wall_clip  # noqa: E402
from render_prediction_overlays import (  # noqa: E402
    CAMPUS_ROOT, LIMB_RGB, PANEL_SIZE, WALL_ROOT, _draw_caption, _fit,
    _force_panel, _wall_panel,
)

GRAVITY = 9.81
ARROW_METRES_PER_BW = 0.35  # same drawing scale as the prediction overlay
MIN_ARROW_BW = 0.02  # below this an arrow is under a pixel of ink
SUP_GROUPS = ["left_hand", "right_hand", "left_foot", "right_foot",
              "left_ankle", "right_ankle"]  # kindyn's contact_force_joints column order
GROUP_FOLD = [[0], [1], [2, 4], [3, 5]]  # six groups -> the four plate limbs
GROUP_RGB = LIMB_RGB + [(148, 103, 189), (227, 119, 194)]  # heels get their own colours
GROUP_LABEL = ["left hand", "right hand", "left foot (toe)", "right foot (toe)",
               "left heel", "right heel"]


def _draw_force_arrows(pane: np.ndarray, anchor_cam: np.ndarray, force_world: np.ndarray,
                       extrinsic: np.ndarray, intrinsic: np.ndarray,
                       scale: tuple[float, float]) -> None:
    """Draw one arrow per active group along the model's force on the body, in place.

    ``anchor_cam (6, 3)`` is the model's own camera-frame anchor point per group and
    ``force_world (6, 3)`` its world-frame force in body weights; the force is rotated
    into the camera with the frame's cam-from-world rotation, scaled to metres, and both
    endpoints are projected — so an arrow shortens with distance like a real vector.
    """
    rot = extrinsic[:3, :3]
    for group in range(len(SUP_GROUPS)):
        start, force = anchor_cam[group], force_world[group]
        if not (np.isfinite(start).all() and np.isfinite(force).all()):
            continue
        if np.linalg.norm(force) < MIN_ARROW_BW or start[2] <= 1e-3:
            continue
        end = start + (rot @ force) * ARROW_METRES_PER_BW
        if end[2] <= 1e-3:
            continue
        tail, head = (
            (int(round((intrinsic[0, 0] * p[0] / p[2] + intrinsic[0, 2]) * scale[0])),
             int(round((intrinsic[1, 1] * p[1] / p[2] + intrinsic[1, 2]) * scale[1])))
            for p in (start, end))
        cv2.arrowedLine(pane, tail, head, GROUP_RGB[group], 3, cv2.LINE_AA, tipLength=0.22)


def render(clip: Clip) -> Path:
    """Write the supervised-model overlay for one clip and return its path."""
    tree = clip.tree
    sup = np.load(tree / "predictions" / "forces_sup.npz", allow_pickle=True)
    kindyn = np.load(tree / "human_optim" / "kindyn_1.npz", allow_pickle=True)
    transform = np.load(tree / "geometry" / "transform.npz")

    groups = [str(x) for x in sup["limbs"]]
    if groups != SUP_GROUPS:
        raise ValueError(f"forces_sup limbs are {groups}, expected {SUP_GROUPS}")
    frame_indices = transform["frame_indices"]
    n_frames = len(frame_indices)
    if not np.array_equal(frame_indices, np.arange(n_frames)):
        raise ValueError("pipeline frames are not the video frames — the rig's clock cannot be aligned")
    sup_ids = [int(x) for x in sup["object_ids"]]
    if len(sup_ids) != 1:
        raise ValueError(f"the rig measures one climber but the predictions hold {sup_ids}")
    kindyn_ids = [int(x) for x in kindyn["object_ids"]]
    mass = float(np.asarray(kindyn["total_mass"], float)[kindyn_ids.index(sup_ids[0])])

    fps = float(sup["fps"])
    forces_world = np.asarray(sup["forces_world"], np.float32)[0, :n_frames]  # (N, 6, 3) bw
    anchor_cam = np.asarray(sup["anchor_cam"], np.float32)[0, :n_frames]      # (N, 6, 3)
    down = np.asarray(kindyn["gravity_world"], float)
    down = down / np.linalg.norm(down)

    # NaN rows (untracked frames, kindyn-invalid frames) plot as zero load; the arrows
    # skip them per-group instead, so the video pane never shows an invented force.
    filled = np.nan_to_num(forces_world) * mass * GRAVITY                     # (N, 6, 3) newtons
    folded = np.stack([filled[:, g].sum(1) for g in GROUP_FOLD], 1)           # (N, 4, 3)
    caption: list[tuple[tuple[int, int, int] | None, str]] = []
    if clip.measured_total is None:
        load = -(folded @ down)                                               # (N, 4) downward N
        panel, column_of, cursor_top, cursor_bottom = _force_panel(
            load[None], clip.wired, fps, clip.plate_newton, clip.plate_fps, 50.0)
        caption += [
            (None, "arrows = supervised force model (6 kindyn groups)"),
            (None, f"graphs: downward load [N],  black = plate,  foot = toe + heel,"
                   f"  model x m.g,  m = {mass:.1f} kg"),
        ]
    else:
        # The wall's cells hold up one structure, so only the total is a clean force.
        # Both sides are world-frame vectors here (the kindyn root quaternion put the
        # model there), so the angle between the totals is meaningful — print it.
        limb_size = np.linalg.norm(folded, axis=-1)
        total = folded.sum(1)                                                 # (N, 3)
        model_total = np.linalg.norm(total, axis=-1)
        measured = clip.measured_total[:n_frames]
        measured_size = np.linalg.norm(measured, axis=-1)
        panel, column_of, cursor_top, cursor_bottom = _wall_panel(
            limb_size, np.linalg.norm(clip.measured_cell[:n_frames], axis=-1),
            model_total, measured_size, fps, model_label="model")
        both = (model_total > 1.0) & (measured_size > 1.0)
        angle = np.degrees(np.arccos(np.clip(
            (total[both] * measured[both]).sum(1)
            / (model_total[both] * measured_size[both]), -1.0, 1.0)))
        caption += [
            (None, "arrows = supervised force model (6 kindyn groups)"),
            (None, f"graphs: force size [N],  black = measured,  foot = toe + heel,"
                   f"  m = {mass:.1f} kg"),
            (None, "limb graphs: black is that CORNER's cell, not that limb's force"),
            (None, f"wall total vs model:  size ratio "
                   f"{model_total.mean() / measured_size.mean():.2f},"
                   f"  median angle {np.median(angle):.0f} deg"),
        ]
    caption += [(GROUP_RGB[g], GROUP_LABEL[g]) for g in range(len(SUP_GROUPS))]

    capture = cv2.VideoCapture(str(clip.video))
    source = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    pane_size = _fit(source, PANEL_SIZE[1])
    pane_scale = (pane_size[0] / source[0], pane_size[1] / source[1])

    path = tree / "predictions" / "forces_sup_overlay.mp4"
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7)
    for frame_index in tqdm(range(n_frames), desc=f"[overlays] {clip.name}", leave=False):
        ok, bgr = capture.read()
        if not ok:
            break
        pane = cv2.resize(bgr, pane_size)[..., ::-1].copy()
        _draw_force_arrows(pane, anchor_cam[frame_index], forces_world[frame_index],
                           transform["extrinsics"][frame_index],
                           transform["intrinsics_px_orig"][frame_index], pane_scale)
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
    """Resolve clip names against both trees, or list every clip the model predicted."""
    if not names:
        return [build(clip.name) for root, build in TREES for clip in sorted(root.iterdir())
                if (clip / "predictions" / "forces_sup.npz").exists()]
    clips = []
    for name in names:
        match = next((build for root, build in TREES
                      if (root / name / "predictions" / "forces_sup.npz").exists()), None)
        if match is None:
            roots = ", ".join(str(root) for root, _ in TREES)
            raise SystemExit(f"[overlays] no forces_sup for {name!r} under any of: {roots}")
        clips.append(match(name))
    return clips


if __name__ == "__main__":
    for target in _targets(sys.argv[1:]):
        try:
            print(f"[overlays] {target.name} -> {render(target)}")
        except Exception as error:  # noqa: BLE001 — one bad clip must not kill the batch
            print(f"[overlays] {target.name}: FAILED — {type(error).__name__}: {error}")
