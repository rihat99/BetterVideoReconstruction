"""Draw one method's limb forces against the measured ones on a Parkour clip.

One video per (clip, method): the clip on the left with, at each limb, a two-ring
contact disc (inner = the plate or bar reads above 50 N, outer = the method's force
does; grey inner = that limb was not captured), the measured force as a **red** arrow
and the method's as a **green** arrow, both drawn in 3D through the clip's camera; and
on the right the five stacked plots of ``render_kindyn_overlays.py`` (total, then one
per limb: black = measured, colour = the method, dashed = the angle between them).

Everything is placed in OUR reconstruction's world, because that is the only world
the video's camera is known in. The MoCap skeleton is registered to our optimised
body once per clip (a rotation and a shift, no scale — both are metric), which puts
the measured joints and forces in our world; each method's forces reach the MoCap
axes first through the same skeleton alignment ``compare_parkour_methods.py`` scores
them with, then follow. So every method is drawn at the SAME anchor pixels — the
measured wrists and ankles — and the arrows compare like for like.

Writes ``../data/Parkour-dataset/comparison/videos/<clip>_<method>.mp4``.

Usage:
    uv run --no-sync python scripts/diagnostics/render_parkour_method_overlays.py
    uv run --no-sync python scripts/diagnostics/render_parkour_method_overlays.py --clips kv01_PKFC
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

import compare_parkour_methods as cmp  # noqa: E402
from render_kindyn_overlays import (  # noqa: E402
    BOARD_CONTACT_N, MEASURED_RGB, SOLVE_RGB, _draw_force_arrows, _draw_legend, _project)
from render_prediction_overlays import PANEL_SIZE, _draw_contact_discs, _fit, _wall_panel  # noqa: E402

OUT_DIR = cmp.DATASET / "comparison" / "videos"
#: One clip per technique, in the paper's order.
DEFAULT_CLIPS = ["kv01_PKFC", "mu02_PKLP", "pu02_PKLP", "sv01_PKFC"]
#: The harness's channel order (l foot, r foot, l hand, r hand) -> the plot order
#: of the shared panel (left hand, right hand, left foot, right foot).
PANEL_ORDER = [2, 3, 0, 1]
#: The MoCap joint each limb's arrow and disc sit on, in panel order.
ANCHOR_JOINTS = ["l_wrist", "r_wrist", "l_ankle", "r_ankle"]

METHODS = {
    "li_their_inputs": ("Li et al., their inputs", lambda c: cmp.estimator_prediction(
        cmp.ESTIMATOR / "validation" / "results" / "v25_original", "orig_v25", c)),
    "li_our_inputs": ("Li et al., our inputs", lambda c: cmp.estimator_prediction(
        cmp.ESTIMATOR / "validation" / "results" / "v25_ours", "ours_v25", c)),
    "physpt": ("PhysPT, our init", cmp.physpt_prediction),
    "ours_learned": ("Ours, learned (L_body)", cmp.learned_prediction),
    "ours_optim": ("Ours, optimisation", lambda c: cmp.our_prediction(c, "human_optim_gt_contact")),
}


def render(clip: str, key: str, truth: dict) -> Path | None:
    label, loader = METHODS[key]
    result = loader(clip)
    if result is None:
        return None
    joints, wrench = result
    tree = cmp.TREE / "out" / clip
    transform = np.load(tree / "geometry" / "transform.npz")
    gt_joints, gt_wrench, captured = truth["joints"], truth["wrench"], truth["captured"]
    n = min(len(joints), len(wrench), len(gt_joints), len(transform["frame_indices"]))

    # MoCap -> our world: registered on our optimised body (a rotation and a shift).
    reference, _ = cmp.our_prediction(clip, "human_optim_gt_contact")
    world_from_mocap = cmp.clip_rotation(gt_joints[:n], reference[:n])
    shift = (reference[:n] - gt_joints[:n] @ world_from_mocap.T).reshape(-1, 3).mean(0)
    anchors = gt_joints[:n][:, [list(cmp.CORRESPONDENCE).index(j) for j in ANCHOR_JOINTS]]
    anchors = anchors @ world_from_mocap.T + shift                          # (n, 4, 3)
    # The method's forces: its own world -> MoCap axes (as scored) -> our world.
    mocap_from_method = cmp.clip_rotation(joints[:n], gt_joints[:n])
    force = wrench[:n, :, :3] @ mocap_from_method.T @ world_from_mocap.T
    measured = gt_wrench[:n, :, :3] @ world_from_mocap.T
    force, measured, captured = force[:, PANEL_ORDER], measured[:, PANEL_ORDER], captured[:n][:, PANEL_ORDER]

    size, gt_size = np.linalg.norm(force, axis=-1), np.linalg.norm(measured, axis=-1)
    gt_size = np.where(captured, gt_size, np.nan)
    total = np.linalg.norm(force.sum(1), axis=-1)
    gt_total = np.where(captured.all(1), np.linalg.norm(measured.sum(1), axis=-1), np.nan)
    both = (size > BOARD_CONTACT_N) & (np.nan_to_num(gt_size) > BOARD_CONTACT_N)
    angle = np.full_like(size, np.nan)
    angle[both] = np.degrees(np.arccos(np.clip(
        (force[both] * measured[both]).sum(-1) / (size[both] * gt_size[both]), -1.0, 1.0)))
    truth_contact = np.where(captured, (np.nan_to_num(gt_size) > BOARD_CONTACT_N).astype(np.int8), -1)
    fps = float(transform["fps"])
    panel, column_of, cursor_top, cursor_bottom = _wall_panel(
        size, gt_size, total, gt_total, fps, model_label=label, limbs_measured=True,
        measured_label="plate", total_label="plate total", limb_angle=angle)

    capture = cv2.VideoCapture(str(cmp.TREE / clip / "video.mp4"))
    source = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    pane_size = _fit(source, PANEL_SIZE[1])
    pane_scale = (pane_size[0] / source[0], pane_size[1] / source[1])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{clip}_{key}.mp4"
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7)
    for frame_index in tqdm(range(n), desc=f"[overlays] {clip} {key}", leave=False):
        ok, bgr = capture.read()
        if not ok:
            break
        pane = cv2.resize(bgr, pane_size)[..., ::-1].copy()
        extrinsic = transform["extrinsics"][frame_index]
        intrinsic = transform["intrinsics_px_orig"][frame_index]
        pixels = _project(anchors[frame_index], extrinsic, intrinsic, pane_scale)
        _draw_contact_discs(pane, pixels, size[frame_index] > BOARD_CONTACT_N, truth_contact[frame_index])
        _draw_force_arrows(pane, anchors[frame_index], np.nan_to_num(measured[frame_index]),
                           [MEASURED_RGB] * 4, extrinsic, intrinsic, pane_scale, thickness=2)
        _draw_force_arrows(pane, anchors[frame_index], force[frame_index], [SOLVE_RGB] * 4,
                           extrinsic, intrinsic, pane_scale)
        _draw_legend(pane, [(SOLVE_RGB, label), (MEASURED_RGB, "measured")])
        cursor = panel.copy()
        column = np.clip(column_of(frame_index / fps), 1, panel.shape[1] - 2)
        cursor[cursor_top:cursor_bottom, column - 1:column + 2] = (205, 30, 30)
        writer.append_data(np.hstack([pane, cursor]))
    capture.release()
    writer.close()
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", default=",".join(DEFAULT_CLIPS))
    ap.add_argument("--methods", default=",".join(METHODS))
    args = ap.parse_args()
    truth = cmp.ground_truth("published")
    for clip in args.clips.split(","):
        for key in args.methods.split(","):
            path = render(clip, key, truth[clip])
            print(f"{clip} {key}: {'no prediction' if path is None else path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
