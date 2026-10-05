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

The rigs are compared exactly as in the other overlays: on the campus board every
panel is a **downward load** in newtons against that limb's plate (session 1 wired the
hands only — those foot panels are shaded); on the wall only the **total** is a clean
quantity, compared by size, with each corner cell's trace shaded inside its limb panel.
Body-weight units become newtons through the SMPL-X mass in ``human_optim/kindyn_1.npz``,
printed on the pane because it is known to run a few kilos high.

The demonstrator wall (``peter/out_climb_wall_2_single/``) instead gets the
**kindyn-style picture**, so the model's overlay and the solve's read side by side:
every limb panel is a real comparison against that limb's own board (nothing shaded),
with the model-vs-board direction error as a dashed line in the limb's colour on its
own 0–180° right axis, drawn where both sides read above 50 N. On the video, one
summed **green** arrow per limb (foot = toe + heel) beside the board's **red** arrow,
two-ring contact discs (outer ring = the model's contact head above 0.5, inner = the
board above 50 N), and a tiny dot-and-word legend instead of the caption; the clip's
total ratio, median angle and the mass print to the console.

Writes ``<clip>/predictions/forces_sup_overlay.mp4``, or — with ``--run <name>`` — reads
``<clip>/predictions/<name>/forces_sup.npz`` and writes the overlay beside it (the layout
contact_anything's ``predict_reconstruction.py`` uses since round 11). A dump written on
the 35 contact frames (``contact_set`` ``frames35``) is folded onto the six kindyn groups
first: probabilities by max, forces by vector sum, each group's arrow anchored at the
force-weighted mean of its member frames' anchors.

Usage:
    uv run --no-sync python scripts/diagnostics/render_sup_force_overlays.py
    uv run --no-sync python scripts/diagnostics/render_sup_force_overlays.py 01_hanging trial_10
    uv run --no-sync python scripts/diagnostics/render_sup_force_overlays.py --run climbing_frames35 a_00_01_part1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

_DIAGNOSTICS = Path(__file__).resolve().parent
if str(_DIAGNOSTICS) not in sys.path:
    sys.path.insert(0, str(_DIAGNOSTICS))

from render_kindyn_overlays import (  # noqa: E402
    BOARD_CONTACT_N, MEASURED_RGB, SOLVE_RGB, WALL2_ROOT, WALL3_ROOT, Clip, campus_clip,
    wall2_clip, wall3_clip, wall_clip, _draw_force_arrows as _draw_limb_arrows, _draw_legend,
    _project,
)
from render_prediction_overlays import (  # noqa: E402
    CAMPUS_ROOT, LIMB_RGB, PANEL_SIZE, WALL_ROOT, _draw_caption, _draw_contact_discs,
    _fit, _force_panel, _wall_panel,
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
SUP_CONTACT_PROB = 0.5  # the model's contact head is a probability; 0.5 = its own claim
CONTACT_ANYTHING = Path(__file__).resolve().parents[2].parent / "contact_anything"


def _six_groups(sup) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(forces_world, anchor_cam, contact_probs)`` of a dump on the six kindyn groups.

    A ``kindyn6`` dump (or an older one without ``contact_set``) is returned as is; a
    ``frames35`` dump is folded with contact_anything's own contract: probabilities by
    max, forces by sum, anchors by the force-magnitude-weighted mean of the group's member
    frames (plain mean where no member carries force).
    """
    forces_world = np.asarray(sup["forces_world"], np.float32)[0]   # (N, K, 3) bw
    anchor_cam = np.asarray(sup["anchor_cam"], np.float32)[0]       # (N, K, 3)
    probs = np.asarray(sup["contact_probs"], np.float32)[0]         # (N, K)
    name = str(sup["contact_set"]) if "contact_set" in sup.files else "kindyn6"
    if name == "kindyn6":
        return forces_world, anchor_cam, probs
    if str(CONTACT_ANYTHING) not in sys.path:
        sys.path.insert(0, str(CONTACT_ANYTHING))
    from model.contact_frames import contact_set  # noqa: E402
    slots = contact_set(name)
    if [str(x) for x in sup["limbs"]] != list(slots.slot_names):
        raise ValueError(f"forces_sup limbs do not match contact set {name!r}")
    group_of = slots.group_of
    n_frames = forces_world.shape[0]
    folded_anchor = np.full((n_frames, len(SUP_GROUPS), 3), np.nan, np.float32)
    for group in range(len(SUP_GROUPS)):
        members = np.flatnonzero(group_of == group)
        if not members.size:
            continue
        points = anchor_cam[:, members]                              # (N, m, 3)
        weight = np.linalg.norm(np.nan_to_num(forces_world[:, members]), axis=-1)  # (N, m)
        weight = np.where(np.isfinite(points).all(-1), weight, 0.0)
        flat = np.where(weight.sum(1, keepdims=True) > 0, weight,
                        np.isfinite(points).all(-1).astype(np.float32))
        total = flat.sum(1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            folded_anchor[:, group] = (np.nan_to_num(points) * flat[..., None]).sum(1) / total
    return slots.fold_sum(np.nan_to_num(forces_world)), folded_anchor, slots.fold_max(np.nan_to_num(probs))


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


def render(clip: Clip, run: str | None = None) -> Path:
    """Write the supervised-model overlay for one clip and return its path."""
    tree = clip.tree
    pred_dir = tree / "predictions" if run is None else tree / "predictions" / run
    sup = np.load(pred_dir / "forces_sup.npz", allow_pickle=True)
    kindyn = np.load(tree / "human_optim" / "kindyn_1.npz", allow_pickle=True)
    transform = np.load(tree / "geometry" / "transform.npz")

    groups = [str(x) for x in sup["limbs"]]
    if groups != SUP_GROUPS and "contact_set" not in sup.files:
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
    forces_world, anchor_cam, contact_probs = _six_groups(sup)
    forces_world = forces_world[:n_frames]                                    # (N, 6, 3) bw
    anchor_cam = anchor_cam[:n_frames]                                        # (N, 6, 3)
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
        # A wall rig. On the first wall the cells hold up one structure, so only the
        # total is a clean force; on the demonstrator wall a channel IS that limb's
        # board, so every panel compares — with the model-vs-board direction error
        # as a dashed line, exactly like the kindyn overlay. Both sides are
        # world-frame vectors here (the kindyn root quaternion put the model there),
        # so the angles are meaningful.
        limb_size = np.linalg.norm(folded, axis=-1)
        total = folded.sum(1)                                                 # (N, 3)
        model_total = np.linalg.norm(total, axis=-1)
        measured = clip.measured_total[:n_frames]
        measured_size = np.linalg.norm(measured, axis=-1)
        cell = clip.measured_cell[:n_frames]
        cell_size = np.linalg.norm(cell, axis=-1)
        limb_angle = None
        if clip.limbs_measured:
            both = (limb_size > BOARD_CONTACT_N) & (cell_size > BOARD_CONTACT_N)
            limb_angle = np.full_like(limb_size, np.nan)
            limb_angle[both] = np.degrees(np.arccos(np.clip(
                (folded[both] * cell[both]).sum(-1)
                / (limb_size[both] * cell_size[both]), -1.0, 1.0)))
        panel, column_of, cursor_top, cursor_bottom = _wall_panel(
            limb_size, cell_size, model_total, measured_size, fps, model_label="model",
            limbs_measured=clip.limbs_measured, limb_angle=limb_angle)
        both = (model_total > 1.0) & (measured_size > 1.0)
        angle = np.degrees(np.arccos(np.clip(
            (total[both] * measured[both]).sum(1)
            / (model_total[both] * measured_size[both]), -1.0, 1.0)))
        if clip.limbs_measured:
            print(f"[overlays] {clip.name}: total size ratio "
                  f"{model_total.mean() / measured_size.mean():.2f}, "
                  f"median angle {np.median(angle):.0f} deg, m = {mass:.1f} kg")
        else:
            caption += [
                (None, "arrows = supervised force model (6 kindyn groups)"),
                (None, f"graphs: force size [N],  black = measured,  foot = toe + heel,"
                       f"  m = {mass:.1f} kg"),
                (None, "limb graphs: black is that CORNER's cell, not that limb's force"),
                (None, f"wall total vs model:  size ratio "
                       f"{model_total.mean() / measured_size.mean():.2f},"
                       f"  median angle {np.median(angle):.0f} deg"),
            ]
    if clip.limbs_measured:
        # The kindyn-style picture: two-ring discs, one summed green arrow per limb,
        # the board's own force in red, and a tiny legend instead of a caption.
        valid = np.asarray(sup["valid_mask"], bool)[0, :n_frames]
        probs = contact_probs[:n_frames]                                      # (N, 6)
        limb_contact = np.stack([probs[:, g].max(1) for g in GROUP_FOLD], 1) > SUP_CONTACT_PROB
        truth = clip.plate_contact[:n_frames]
        measured_limb = clip.measured_cell[:n_frames]
        legend = [(SOLVE_RGB, "model"), (MEASURED_RGB, "sensor")]
    else:
        caption += [(GROUP_RGB[g], GROUP_LABEL[g]) for g in range(len(SUP_GROUPS))]

    capture = cv2.VideoCapture(str(clip.video))
    source = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    pane_size = _fit(source, PANEL_SIZE[1])
    pane_scale = (pane_size[0] / source[0], pane_size[1] / source[1])

    path = pred_dir / "forces_sup_overlay.mp4"
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7)
    for frame_index in tqdm(range(n_frames), desc=f"[overlays] {clip.name}", leave=False):
        ok, bgr = capture.read()
        if not ok:
            break
        pane = cv2.resize(bgr, pane_size)[..., ::-1].copy()
        extrinsic = transform["extrinsics"][frame_index]
        intrinsic = transform["intrinsics_px_orig"][frame_index]
        if clip.limbs_measured:
            if valid[frame_index]:
                # The model anchors live in the camera; the shared helpers want world.
                rot, t = extrinsic[:3, :3], extrinsic[:3, 3]
                anchors = (anchor_cam[frame_index, :4] - t) @ rot              # (4, 3) world
                pixels = _project(anchors, extrinsic, intrinsic, pane_scale)
                _draw_contact_discs(pane, pixels, limb_contact[frame_index], truth[frame_index])
                _draw_limb_arrows(pane, anchors, measured_limb[frame_index],
                                  [MEASURED_RGB] * 4, extrinsic, intrinsic, pane_scale,
                                  thickness=2)
                _draw_limb_arrows(pane, anchors, folded[frame_index], [SOLVE_RGB] * 4,
                                  extrinsic, intrinsic, pane_scale)
            _draw_legend(pane, legend)
        else:
            _draw_force_arrows(pane, anchor_cam[frame_index], forces_world[frame_index],
                               extrinsic, intrinsic, pane_scale)
            _draw_caption(pane, caption)

        cursor = panel.copy()
        column = np.clip(column_of(frame_index / fps), 1, panel.shape[1] - 2)
        cursor[cursor_top:cursor_bottom, column - 1:column + 2] = (205, 30, 30)
        writer.append_data(np.hstack([pane, cursor]))

    capture.release()
    writer.close()
    return path


TREES = [(CAMPUS_ROOT, campus_clip), (WALL_ROOT, wall_clip), (WALL2_ROOT, wall2_clip),
         (WALL3_ROOT, wall3_clip)]


def _targets(names: list[str], run: str | None) -> list[Clip]:
    """Resolve clip names against both trees, or list every clip the model predicted."""
    sub = Path("predictions") if run is None else Path("predictions") / run
    if not names:
        return [build(clip.name) for root, build in TREES for clip in sorted(root.iterdir())
                if (clip / sub / "forces_sup.npz").exists()]
    clips = []
    for name in names:
        match = next((build for root, build in TREES
                      if (root / name / sub / "forces_sup.npz").exists()), None)
        if match is None:
            roots = ", ".join(str(root) for root, _ in TREES)
            raise SystemExit(f"[overlays] no forces_sup for {name!r} under any of: {roots}")
        clips.append(match(name))
    return clips


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None,
                        help="read predictions/<run>/forces_sup.npz and write the overlay beside it")
    parser.add_argument("clips", nargs="*")
    args = parser.parse_args()
    for target in _targets(args.clips, args.run):
        try:
            print(f"[overlays] {target.name} -> {render(target, args.run)}")
        except Exception as error:  # noqa: BLE001 — one bad clip must not kill the batch
            print(f"[overlays] {target.name}: FAILED — {type(error).__name__}: {error}")
