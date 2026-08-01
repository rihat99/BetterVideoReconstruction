"""Render overlays for the two external models' predictions.

Two models write into a clip's ``predictions/`` folder: a contact model
(``contacts.npz``) and a force model (``forces.npz``). This script draws both
back onto the video, and puts the force plates beside them where plates exist.

Three trees are covered, with less to compare against in each:

``peter/out_campus_board_single/`` — clips filmed on the instrumented campus
board, so every overlay is a **comparison** against the measured force plates.

``out/`` — ordinary climbing clips. No force plates ever filmed these, but each
one carries a hand-made contact annotation in ``data/<stem>.npz``, so contacts
are still compared against a truth; only the forces have nothing to check
against. These clips can be portrait or landscape and can hold more than one
climber; both are handled.

``peter/out_climb_wall_single/`` — the climb_wall trials. Nobody annotated the
contacts here, so that overlay is still the model alone, but the wall's own load
cells are now mapped into world axes and the force overlay compares against them.
The four cells are the supports of **one structure**, so a cell is that corner's
reaction rather than the limb it is named after, and only their sum is a clean
force. The comparison is therefore on the **total**, and by **size** only.

``contacts_overlay.mp4``
    The video with a marker on each of the four contact keypoints — the wrists
    and ankles, taken from the Sapiens pose. Where a truth exists (the plates on
    campus_board, the hand annotation under ``out/``) the marker is a two-ring
    disc: **inner** is the truth, **outer** is the model, separated by a white
    strip, so one colour is agreement and a two-tone disc is a disagreement. With
    no truth at all it is a single disc showing the model alone. Red means
    contact, green means no contact, grey means the truth does not cover that
    limb.

``forces_overlay.mp4``
    The video with the model's force arrows drawn on the body, beside five
    stacked plots — the total, then one per limb — with a cursor that follows the
    video. Whatever the rig measured is drawn in black behind the model's trace.
    On the campus board that is each limb's plate, and every panel is a downward
    load. On the wall it is the load cells, every panel is a plain force size, and
    only the top panel is a like-for-like comparison — a limb panel's black line
    is that corner's cell, which tracks *when* a limb loads up without being that
    limb's force, so those panels stay shaded.

Two things the plots depend on, both checked against the data:

*The sign.* A plate measures one axis, so the comparison is one number per limb:
the downward load on the hold. For the model that is ``+F_y`` in its
local-world-aligned frame, whose ``+y`` points up — the file's own
``lwa_to_opencv_cam_flip`` says so, and a hanging clip confirms it, because the
limb forces there sum to +1.06 body weights, which is what holds the climber up.
Measured over the 24 campus_board clips, the plate reading and ``+F_y`` agree in
sign on 22; the two exceptions are hanging clips whose plate reading is constant
to within its own noise, so their correlation carries no information.

*The mass.* The model reports body weights, so its curves are multiplied by
``m·g`` using the SMPL-X body mass from ``human_optim/kindyn_1.npz``. That is the
one assumption in the plot, and it is printed on the video pane because it is
known to run high — around 61 kg where the measured totals imply about 55. A
tree whose dynamics stage never ran has no such file (climb_wall today), and
those clips are plotted in **body weights** rather than handed an invented mass.

Session 1 of campus_board instrumented the hands only (see ``peter/README.md``),
so on those clips a zero foot column means there was no sensor, not that the foot
was free. Those feet are drawn grey, their panels are shaded, and the sum panel
adds up the hands alone.

Usage:
    uv run python scripts/diagnostics/render_prediction_overlays.py
    uv run python scripts/diagnostics/render_prediction_overlays.py boulder_1 02_gianna_hanging
    uv run python scripts/diagnostics/render_prediction_overlays.py --forces
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PETER = ROOT / "peter"
PLATE_ROOT = PETER / "campus_board"
CAMPUS_ROOT = PETER / "out_campus_board_single"
WALL_SOURCE = PETER / "climb_wall"
WALL_ROOT = PETER / "out_climb_wall_single"
OUT_ROOT = ROOT / "out"
DATA_ROOT = ROOT / "data"

LIMBS = ["left_hand", "right_hand", "left_foot", "right_foot"]
LIMB_KEYPOINT = [62, 41, 13, 14]  # Goliath-308: left/right wrist, left/right ankle
LIMB_RGB = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40)]
SUM_RGB = (110, 70, 175)
PERSON_STYLE = ["-", "--", ":", "-."]
CONTACT_THRESHOLD = 0.3  # on the contact model's probability
GRAVITY = 9.81

CONTACT_RGB = (225, 55, 45)
FREE_RGB = (45, 185, 95)
NODATA_RGB = (135, 135, 135)
WHITE = (255, 255, 255)

# Both overlays keep every side a multiple of 16, or ffmpeg silently rescales them.
CONTACT_HEIGHT = 720  # the contacts overlay is the video alone; width follows the aspect
PANEL_SIZE = (720, 896)  # the plot column; its height also sets the video pane's height
RADIUS_INNER, RADIUS_STRIP, RADIUS_OUTER = 6, 9, 13  # the plate-vs-model disc
RADIUS_SOLO, RADIUS_SOLO_RIM = 8, 11  # the model-only disc, drawn with a white rim
ARROW_METRES_PER_BW = 0.35  # how long one body weight of force is drawn


@dataclass(frozen=True)
class Clip:
    """One clip to render, and the plate data to compare it against if any."""

    name: str
    video: Path  #: the source video
    tree: Path  #: the prediction tree holding sapiens/, human_optim/ and predictions/
    #: ``(P, N, 4)`` int8 contact truth: ``1`` contact, ``0`` free, ``-1`` not known
    truth_contact: np.ndarray | None = None
    truth_label: str = ""  #: what the truth is, for the caption
    plate_newton: np.ndarray | None = None  #: ``(T, 4)`` plate log in newtons
    plate_fps: float = 0.0
    wired: np.ndarray | None = None  #: ``(4,)`` which limbs had a plate; None = no plates at all
    #: ``(N, 3)`` world-frame total the climb_wall measured, in the reaction sense
    measured_total: np.ndarray | None = None
    #: ``(N, 4, 3)`` the same for each corner cell, in ``LIMBS`` order
    measured_cell: np.ndarray | None = None


def _wired(clip: str) -> np.ndarray:
    """Return ``(4,)`` bool — which limbs had a plate. Session 1 wired the hands only."""
    plates = np.ones(4, bool)
    if "peter" not in clip and "gianna" not in clip:
        plates[2:] = False
    return plates


def campus_board_clip(name: str) -> Clip:
    """Describe an instrumented campus_board clip, plates included."""
    plates = PLATE_ROOT / name
    forces = np.load(plates / "forces.npz")
    wired = _wired(name)
    truth = np.asarray(np.load(plates / "contacts.npz")["contacts"], np.int8)[None]  # (1, N, 4)
    truth[:, :, ~wired] = -1  # a session-1 foot column is zero because there is no sensor
    return Clip(name=name, video=plates / "cam_left.mp4", tree=CAMPUS_ROOT / name,
                truth_contact=truth, truth_label="force plate",
                plate_newton=forces["forces"], plate_fps=float(forces["fps"]), wired=wired)


def _annotation_contacts(name: str, predicted_ids: list[int]) -> np.ndarray | None:
    """The four limb columns of ``data/<stem>.npz``, ordered to match the predicted people.

    The hand-made annotation stores ``(P, J, N)`` over a 10- or 14-joint schema, so
    the limbs are looked up **by name**, and its people are matched **by object id**:
    a clip can carry annotation tracks the annotator marked ``ignored``, and on
    ``competition_2`` the predicted climber is id 1 while track 0 is ignored, so
    matching by position would take the wrong person. Values pass through as
    stored — ``1`` contact, ``0`` free, ``-1`` never annotated.
    """
    path = DATA_ROOT / f"{name}.npz"
    if not path.exists():
        return None
    ann = np.load(path, allow_pickle=True)
    joint_names = [str(x) for x in ann["joint_names"]]
    if not all(limb in joint_names for limb in LIMBS):
        return None
    columns = [joint_names.index(limb) for limb in LIMBS]
    contacts = np.asarray(ann["contacts"], np.int8)
    annotated_ids = [int(x) for x in ann["object_ids"]]
    ignored = (np.asarray(ann["ignored"], bool) if "ignored" in ann.files
               else np.zeros(len(annotated_ids), bool))
    truth = np.full((len(predicted_ids), contacts.shape[2], 4), -1, np.int8)
    for row, object_id in enumerate(predicted_ids):
        if object_id in annotated_ids and not ignored[annotated_ids.index(object_id)]:
            truth[row] = contacts[annotated_ids.index(object_id)][columns].T
    return truth


def out_clip(name: str) -> Clip:
    """Describe an ordinary clip from ``out/``, with the hand annotation if one exists."""
    tree = OUT_ROOT / name
    predicted_ids = [int(x) for x in np.load(tree / "predictions" / "contacts.npz")["object_ids"]]
    return Clip(name=name, video=DATA_ROOT / f"{name}.mp4", tree=tree,
                truth_contact=_annotation_contacts(name, predicted_ids),
                truth_label="hand annotation")


def climb_wall_clip(name: str) -> Clip:
    """Describe a climb_wall trial, with the wall's own load cells if they were mapped.

    No clip here has a hand contact annotation, so the contacts overlay still shows the
    model alone. The forces overlay now has something to check against:
    ``human_optim/sensor_forces.npz`` (written by ``scripts/peter_sensor_forces.py``)
    holds the four load cells rotated into world axes and flipped to the reaction sense,
    one row per video frame. The four cells are the supports of ONE structure, so a cell
    is a corner reaction rather than that limb's force and only their sum is a clean
    quantity — which is why the wall is compared by size, on the total. Channel order is
    checked, not assumed: drawing a cell under the wrong limb would be invisible.
    """
    sensor_path = WALL_ROOT / name / "human_optim" / "sensor_forces.npz"
    if not sensor_path.exists():
        return Clip(name=name, video=WALL_SOURCE / name / "cam_left.mp4", tree=WALL_ROOT / name)
    sensor = np.load(sensor_path)
    channels = [str(limb) for limb in sensor["limbs"]]
    if channels != LIMBS:
        raise SystemExit(f"[overlays] {name}: sensor channels are {channels}, expected {LIMBS}")
    return Clip(name=name, video=WALL_SOURCE / name / "cam_left.mp4", tree=WALL_ROOT / name,
                measured_total=np.asarray(sensor["total_world"], float),
                measured_cell=np.asarray(sensor["reaction_world"], float))


def _rows_for(available_ids: np.ndarray, wanted_ids: list[int], clip: str, what: str) -> list[int]:
    """Map object ids to rows of a per-person array, failing loudly if one is missing."""
    order = [int(x) for x in available_ids]
    missing = [object_id for object_id in wanted_ids if object_id not in order]
    if missing:
        raise SystemExit(f"[overlays] {clip}: {what} has no object id {missing} (it holds {order})")
    return [order.index(object_id) for object_id in wanted_ids]


def _fit(source: tuple[int, int], height: int) -> tuple[int, int]:
    """Size a pane to ``height``, keeping the source aspect, width a multiple of 16.

    Rounding the width moves the aspect by well under a percent, which is why the
    drawing code carries separate x and y scales rather than one number.
    """
    width = max(16, int(round(height * source[0] / source[1] / 16)) * 16)
    return width, height


def _intrinsics(anchor_cam: np.ndarray, anchor_2d: np.ndarray) -> tuple[float, float, float, float]:
    """Recover ``(fx, fy, cx, cy)`` from the anchors the force model stored.

    The prediction file carries every contact anchor twice, once in camera
    coordinates and once in pixels, which pins the pinhole camera down exactly —
    so the force arrows need no external calibration and cannot disagree with the
    model's own geometry. The fit reproduces the stored pixels to under a pixel.
    """
    points = anchor_cam.reshape(-1, 3)
    pixels = anchor_2d.reshape(-1, 2)
    keep = (points[:, 2] > 1e-3) & np.isfinite(pixels).all(1)
    points, pixels = points[keep], pixels[keep]
    design = np.zeros((2 * len(points), 4))
    design[0::2, 0], design[0::2, 2] = points[:, 0] / points[:, 2], 1.0
    design[1::2, 1], design[1::2, 3] = points[:, 1] / points[:, 2], 1.0
    fx, fy, cx, cy = np.linalg.lstsq(design, pixels.reshape(-1), rcond=None)[0]
    return float(fx), float(fy), float(cx), float(cy)


def _draw_caption(frame: np.ndarray, entries: list[tuple[tuple[int, int, int] | None, str]]) -> None:
    """Draw a translucent caption box in the top-left corner.

    Each entry is a colour swatch plus its text; a ``None`` colour writes the
    text alone, for the heading lines.
    """
    font, scale, weight, row_h, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, 24, 10
    measure = lambda size: max(cv2.getTextSize(text, font, size, weight)[0][0] for _, text in entries)  # noqa: E731
    room = frame.shape[1] - 4 * pad - 12
    if measure(scale) > room:  # a portrait pane is too narrow for the caption at full size
        scale = max(0.26, scale * room / measure(scale))
    widths = [cv2.getTextSize(text, font, scale, weight)[0][0] for _, text in entries]
    box = frame[pad:pad + 2 * pad + row_h * len(entries), pad:pad + 3 * pad + max(widths) + 12]
    box[:] = (0.35 * box + 0.65 * np.array([25, 25, 25])).astype(np.uint8)
    for row, (colour, text) in enumerate(entries):
        baseline = 2 * pad + row_h * row + 6
        if colour is not None:
            cv2.circle(frame, (2 * pad + 4, baseline - 5), 7, colour, -1, cv2.LINE_AA)
        cv2.putText(frame, text, (3 * pad + 6, baseline), font, scale, (245, 245, 245), weight, cv2.LINE_AA)


def _draw_contact_discs(frame: np.ndarray, pixels: np.ndarray, model: np.ndarray,
                        truth: np.ndarray | None) -> None:
    """Stamp one marker per limb onto an RGB frame, in place.

    With ``truth`` given the marker is a two-ring disc, truth inside and model
    outside. Without it there is nothing to compare, so a single disc shows the
    model alone — deliberately a different shape, so a reader never mistakes a
    model-only overlay for a perfect agreement.
    """
    for limb in range(4):
        if not np.isfinite(pixels[limb]).all():
            continue
        centre = (int(round(pixels[limb][0])), int(round(pixels[limb][1])))
        outer = CONTACT_RGB if model[limb] else FREE_RGB
        if truth is None:
            cv2.circle(frame, centre, RADIUS_SOLO_RIM, WHITE, -1, cv2.LINE_AA)
            cv2.circle(frame, centre, RADIUS_SOLO, outer, -1, cv2.LINE_AA)
            continue
        inner = {1: CONTACT_RGB, 0: FREE_RGB}.get(int(truth[limb]), NODATA_RGB)
        cv2.circle(frame, centre, RADIUS_OUTER, outer, -1, cv2.LINE_AA)
        cv2.circle(frame, centre, RADIUS_STRIP, WHITE, -1, cv2.LINE_AA)
        cv2.circle(frame, centre, RADIUS_INNER, inner, -1, cv2.LINE_AA)


def _draw_force_arrows(frame: np.ndarray, anchor_cam: np.ndarray, force_lwa: np.ndarray,
                       flip: np.ndarray, camera: tuple[float, float, float, float],
                       scale: tuple[float, float]) -> None:
    """Draw one arrow per limb along the model's force on the body, in place.

    The force is rotated into the camera frame with the file's own axis flip,
    scaled to metres, added to the 3D anchor, and both ends are projected — so an
    arrow shortens with distance the way a real vector would.
    """
    fx, fy, cx, cy = camera
    scale_x, scale_y = scale
    force_cam = force_lwa * flip
    for limb in range(4):
        start = anchor_cam[limb]
        if start[2] <= 1e-3 or np.linalg.norm(force_lwa[limb]) < 0.02:
            continue
        end = start + force_cam[limb] * ARROW_METRES_PER_BW
        if end[2] <= 1e-3:
            continue
        tail = (int(round((fx * start[0] / start[2] + cx) * scale_x)),
                int(round((fy * start[1] / start[2] + cy) * scale_y)))
        head = (int(round((fx * end[0] / end[2] + cx) * scale_x)),
                int(round((fy * end[1] / end[2] + cy) * scale_y)))
        cv2.arrowedLine(frame, tail, head, LIMB_RGB[limb], 3, cv2.LINE_AA, tipLength=0.22)


def _force_panel(model_load: np.ndarray, wired: np.ndarray, video_fps: float,
                 plate_newton: np.ndarray | None = None, plate_fps: float = 0.0,
                 floor: float = 50.0):
    """Render the five stacked force plots once.

    Returns the panel image, a function from clip time to its pixel column, and
    the rows the cursor should span. The plots are drawn once per clip and only
    the cursor moves, which is what keeps the render cheap.

    ``model_load (P, N, 4)`` is each climber's downward load, one value per video
    frame; ``plate_newton (T, 4)`` the plate log at its own rate, or None.
    ``floor`` is the smallest y-axis top, in whatever unit the load carries.
    """
    people = model_load.shape[0]
    model_time = np.arange(model_load.shape[1]) / video_fps
    summed = np.flatnonzero(wired)  # a sum over limbs the plates could not see is not a sum

    fig, axes = plt.subplots(5, 1, figsize=(PANEL_SIZE[0] / 100, PANEL_SIZE[1] / 100), dpi=100, sharex=True)
    rows = [(-1, "sum", SUM_RGB)] + [(limb, LIMBS[limb].replace("_", " "), LIMB_RGB[limb]) for limb in range(4)]

    for axis, (limb, name, colour) in zip(axes, rows):
        pick = (lambda a: a[:, summed].sum(1)) if limb < 0 else (lambda a: a[:, limb])
        has_plate = plate_newton is not None and (limb < 0 or wired[limb])
        drawn = []
        if has_plate:
            plate_curve = pick(plate_newton)
            axis.plot(np.arange(len(plate_newton)) / plate_fps, plate_curve,
                      color="black", lw=1.3, alpha=0.8, label="plate")
            drawn.append(plate_curve)
        elif plate_newton is not None:
            axis.set_facecolor("0.94")  # a limb this clip could not measure
        for person in range(people):
            curve = pick(model_load[person])
            axis.plot(model_time, curve, color=np.array(colour) / 255.0, lw=2.1,
                      ls=PERSON_STYLE[person % len(PERSON_STYLE)],
                      label="model" if people == 1 else f"person {person}")
            drawn.append(curve)

        span = np.concatenate(drawn)
        low, high = min(0.0, float(span.min())), max(floor, float(span.max()))
        headroom = 0.26 if limb < 0 else 0.16  # room for the in-panel name, and the legend up top
        axis.set_ylim(low - 0.06 * (high - low), high + headroom * (high - low))
        # The name rides inside the panel so the left margin only has to fit tick numbers.
        shown = name if has_plate or plate_newton is None else f"{name}  (no plate)"
        axis.text(0.008, 0.96, shown, transform=axis.transAxes, fontsize=13, va="top", ha="left",
                  color="0.15" if has_plate or plate_newton is None else "0.45")
        axis.tick_params(labelsize=11)
        axis.grid(alpha=0.25, lw=0.6)

    if plate_newton is not None or people > 1:
        axes[0].legend(loc="upper right", fontsize=11, ncol=2, framealpha=0.9,
                       borderpad=0.3, handlelength=1.4)
    axes[-1].set_xlim(0.0, float(model_time[-1]))
    fig.subplots_adjust(left=0.108, right=0.996, top=0.996, bottom=0.038, hspace=0.07)
    fig.canvas.draw()
    panel = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

    height = panel.shape[0]
    column_of = lambda t: int(round(axes[0].transData.transform((t, 0.0))[0]))  # noqa: E731
    top = height - int(round(axes[0].transAxes.transform((0.0, 1.0))[1]))
    bottom = height - int(round(axes[-1].transAxes.transform((0.0, 0.0))[1]))
    plt.close(fig)
    return panel, column_of, top, bottom


def _wall_panel(limb_size: np.ndarray, cell_size: np.ndarray, model_total: np.ndarray,
                measured_total: np.ndarray, fps: float, model_label: str = "model",
                floor: float = 50.0):
    """Render the climb_wall's five stacked size plots once, mirroring :func:`_force_panel`.

    Returns the panel image, a function from clip time to its pixel column, and the
    rows the cursor should span — the same four things, so a frame loop does not care
    which rig it is drawing. Shared by this overlay and the ``kindyn_1`` one, so the
    model's and the reconstruction's forces are read against the wall the same way.

    ``limb_size (N, 4)`` is how big the predicted force at each limb is, ``model_total
    (N,)`` how big their vector sum is, ``measured_total (N,)`` how big the wall's own
    total is, and ``cell_size (N, 4)`` how big each corner cell's reading is, in the
    same limb order. Everything is a plain size in newtons, because the wall's cells
    and a 3D prediction share no axis convention worth trusting sideways.

    Only the top panel is a like-for-like comparison. The black line in a limb panel is
    that corner's cell, which is dominated by the nearest limb but mixed across all
    four — so it tracks *when* a limb loads up without being that limb's force. The
    limb panels stay shaded to keep saying so. Because both totals are the length of a
    vector sum, the top panel is not the four below it added up.
    """
    time = np.arange(len(model_total)) / fps
    fig, axes = plt.subplots(5, 1, figsize=(PANEL_SIZE[0] / 100, PANEL_SIZE[1] / 100),
                             dpi=100, sharex=True)
    rows = [(None, "total force size", SUM_RGB)] + \
           [(limb, LIMBS[limb].replace("_", " "), LIMB_RGB[limb]) for limb in range(4)]

    for axis, (limb, name, colour) in zip(axes, rows):
        measured = measured_total if limb is None else cell_size[:, limb]
        axis.plot(time, measured, color="black", lw=1.3, alpha=0.8,
                  label="wall total" if limb is None else "corner cell")
        if limb is not None:
            axis.set_facecolor("0.94")  # the wall cannot measure one limb on its own
        curve = model_total if limb is None else limb_size[:, limb]
        axis.plot(time, curve, color=np.array(colour) / 255.0, lw=2.1, label=model_label)

        span = np.concatenate([measured, curve])
        low, high = min(0.0, float(span.min())), max(floor, float(span.max()))
        headroom = 0.26 if limb is None else 0.22  # room for the in-panel name and the legend
        axis.set_ylim(low - 0.06 * (high - low), high + headroom * (high - low))
        shown = name if limb is None else f"{name}  (cell, not this limb)"
        axis.text(0.008, 0.96, shown, transform=axis.transAxes, fontsize=13, va="top", ha="left",
                  color="0.15" if limb is None else "0.45")
        axis.tick_params(labelsize=11)
        axis.grid(alpha=0.25, lw=0.6)

    # The top panel's black line and a limb panel's black line are different quantities,
    # so the first of each gets its own legend rather than one shared key up top.
    for axis in (axes[0], axes[1]):
        axis.legend(loc="upper right", fontsize=11, ncol=2, framealpha=0.9,
                    borderpad=0.3, handlelength=1.4)
    axes[-1].set_xlim(0.0, float(time[-1]))
    fig.subplots_adjust(left=0.108, right=0.996, top=0.996, bottom=0.038, hspace=0.07)
    fig.canvas.draw()
    panel = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

    height = panel.shape[0]
    column_of = lambda t: int(round(axes[0].transData.transform((t, 0.0))[0]))  # noqa: E731
    top = height - int(round(axes[0].transAxes.transform((0.0, 1.0))[1]))
    bottom = height - int(round(axes[-1].transAxes.transform((0.0, 0.0))[1]))
    plt.close(fig)
    return panel, column_of, top, bottom


def render(clip: Clip, forces_only: bool = False) -> list[Path]:
    """Write the overlays for one clip and return the paths written.

    ``forces_only`` skips the contacts overlay. Only the force plots depend on the
    body mass from ``kindyn_1``, so when the dynamics stage is re-run it is the force
    overlay alone that goes stale.
    """
    pred_dir = clip.tree / "predictions"
    contacts = np.load(pred_dir / "contacts.npz")
    forces = np.load(pred_dir / "forces.npz")
    pose = np.load(clip.tree / "sapiens" / "pose.npz")
    kindyn_path = clip.tree / "human_optim" / "kindyn_1.npz"

    people, n_frames = contacts["probs"].shape[:2]
    # Every per-person array is looked up by object id, never by position: the models may
    # cover a subset of the tracked people. On competition_2 the prediction holds object 1
    # alone while Sapiens tracked 0, 1 and 2, so row 0 there is a spectator, not the climber.
    predicted_ids = [int(x) for x in contacts["object_ids"]]
    pose_rows = _rows_for(pose["object_ids"], predicted_ids, clip.name, "sapiens/pose.npz")
    # The body mass turns the model's body weights into newtons. A tree that never ran the
    # dynamics stage has none, so those clips are plotted in body weights rather than being
    # given an invented mass.
    mass = None
    if kindyn_path.exists():
        kindyn = np.load(kindyn_path)
        rows = _rows_for(kindyn["object_ids"], predicted_ids, clip.name, "human_optim/kindyn_1.npz")
        mass = np.asarray(kindyn["total_mass"], float)[rows]
    if clip.plate_newton is not None and people != 1:
        raise SystemExit(f"[overlays] {clip.name}: the plates measure one climber but the predictions hold {people}")

    fps = float(contacts["fps"])
    wired = clip.wired if clip.wired is not None else np.ones(4, bool)
    model_contact = contacts["probs"] > CONTACT_THRESHOLD  # (P, N, 4)
    tracked = contacts["valid_mask"]  # (P, N)
    truth_contact = clip.truth_contact[:, :n_frames] if clip.truth_contact is not None else None
    if truth_contact is not None and truth_contact.shape[0] != people:
        raise SystemExit(f"[overlays] {clip.name}: {truth_contact.shape[0]} annotated people "
                         f"but {people} predicted")
    keypoints = pose["keypoints"][pose_rows][:, :n_frames][:, :, LIMB_KEYPOINT]  # (P, N, 4, 2)

    force_lwa = forces["forces"]  # (P, N, 4, 3) body weights, +y up
    anchor_cam = forces["anchor_cam"]  # the contact points the force model used
    axis_flip = forces["lwa_to_opencv_cam_flip"]
    # + F_y in the local-world-aligned frame is the downward load on the hold.
    if mass is None:
        model_load, unit, floor = force_lwa[..., 1], "body weights", 1.0
        unit_note = "no body mass here, so the model's own unit"
    else:
        model_load, unit, floor = force_lwa[..., 1] * mass[:, None, None] * GRAVITY, "N", 50.0
        unit_note = f"model x m.g,  m = {', '.join(f'{value:.1f}' for value in mass)} kg"
    camera = _intrinsics(anchor_cam, forces["anchor_points_2d"])
    if clip.measured_total is None:
        panel, column_of, cursor_top, cursor_bottom = _force_panel(
            model_load, wired, fps, clip.plate_newton, clip.plate_fps, floor)
    else:
        # The wall measures a corner reaction per cell and only their sum is a clean
        # force, so the comparison is on the total. It is a comparison of SIZE only: the
        # model reports in its own local-world-aligned axes and the cells in the
        # reconstruction's world axes, and nothing ties those two frames together here,
        # so an angle between them would be meaningless (the kindyn_1 overlay can show
        # one because both of its sides are already in world axes).
        if mass is None:
            raise SystemExit(f"[overlays] {clip.name}: the wall reads newtons but this tree has "
                             f"no body mass, so the model cannot be put in the same unit")
        force_n = force_lwa[0] * mass[0] * GRAVITY  # (N, 4, 3) newtons
        limb_size = np.linalg.norm(force_n, axis=-1)
        model_total = np.linalg.norm(force_n.sum(1), axis=-1)
        measured_size = np.linalg.norm(clip.measured_total[:n_frames], axis=-1)
        panel, column_of, cursor_top, cursor_bottom = _wall_panel(
            limb_size, np.linalg.norm(clip.measured_cell[:n_frames], axis=-1),
            model_total, measured_size, fps, model_label="model")

    if truth_contact is None:
        contact_caption: list[tuple[tuple[int, int, int] | None, str]] = [
            (None, f"disc = contact model (p > {CONTACT_THRESHOLD})")]
    else:
        contact_caption = [(None, f"inner disc = {clip.truth_label},"
                                  f"  outer ring = model (p > {CONTACT_THRESHOLD})")]
    contact_caption += [(CONTACT_RGB, "contact"), (FREE_RGB, "no contact")]
    if truth_contact is not None and bool((truth_contact < 0).any()):
        contact_caption.append((NODATA_RGB, f"no {clip.truth_label} on this limb"))

    force_caption: list[tuple[tuple[int, int, int] | None, str]] = [
        (None, "arrows = model force on the body")]
    if clip.measured_total is None:
        force_caption.append((None, f"graphs: downward load [{unit}],  {unit_note}"))
    else:
        force_caption += [
            (None, f"graphs: force size [N],  black = measured,  {unit_note}"),
            (None, "limb graphs: black is that CORNER's cell, not that limb's force"),
            (None, f"wall total vs model:  size ratio "
                   f"{model_total.mean() / measured_size.mean():.2f}"),
        ]
    force_caption += [(LIMB_RGB[limb], LIMBS[limb].replace("_", " ")) for limb in range(4)]

    capture = cv2.VideoCapture(str(clip.video))
    source = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    disc_size = _fit(source, CONTACT_HEIGHT)
    video_size = _fit(source, PANEL_SIZE[1])
    disc_scale = (disc_size[0] / source[0], disc_size[1] / source[1])
    video_scale = (video_size[0] / source[0], video_size[1] / source[1])

    contact_path = pred_dir / "contacts_overlay.mp4"
    force_path = pred_dir / "forces_overlay.mp4"
    contact_writer = (None if forces_only else
                      imageio.get_writer(contact_path, fps=fps, codec="libx264", quality=7))
    force_writer = imageio.get_writer(force_path, fps=fps, codec="libx264", quality=7)
    for frame_index in tqdm(range(n_frames), desc=f"[overlays] {clip.name}", leave=False):
        ok, bgr = capture.read()
        if not ok:
            break

        video_pane = cv2.resize(bgr, video_size)[..., ::-1].copy()
        disc_frame = None if forces_only else cv2.resize(bgr, disc_size)[..., ::-1].copy()
        for person in range(people):
            if not tracked[person, frame_index]:
                continue
            if disc_frame is not None:
                _draw_contact_discs(disc_frame, keypoints[person, frame_index] * disc_scale,
                                    model_contact[person, frame_index],
                                    None if truth_contact is None else truth_contact[person, frame_index])
            _draw_force_arrows(video_pane, anchor_cam[person, frame_index], force_lwa[person, frame_index],
                               axis_flip, camera, video_scale)
        _draw_caption(video_pane, force_caption)
        if contact_writer is not None:
            _draw_caption(disc_frame, contact_caption)
            contact_writer.append_data(disc_frame)

        cursor = panel.copy()
        column = np.clip(column_of(frame_index / fps), 1, panel.shape[1] - 2)
        cursor[cursor_top:cursor_bottom, column - 1:column + 2] = (205, 30, 30)
        force_writer.append_data(np.hstack([video_pane, cursor]))

    capture.release()
    force_writer.close()
    if contact_writer is None:
        return [force_path]
    contact_writer.close()
    return [contact_path, force_path]


TREES = [(CAMPUS_ROOT, campus_board_clip), (OUT_ROOT, out_clip), (WALL_ROOT, climb_wall_clip)]


def _targets(names: list[str]) -> list[Clip]:
    """Resolve clip names to clips, or list every tree when given none."""
    if not names:
        return [build(clip.name) for root, build in TREES
                for clip in sorted(root.iterdir())
                if (clip / "predictions" / "contacts.npz").exists()]
    clips = []
    for name in names:
        match = next((build for root, build in TREES
                      if (root / name / "predictions" / "contacts.npz").exists()), None)
        if match is None:
            roots = ", ".join(str(root) for root, _ in TREES)
            raise SystemExit(f"[overlays] no predictions for {name!r} under any of: {roots}")
        clips.append(match(name))
    return clips


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("clips", nargs="*", help="clip names; with none, every tree is rendered")
    parser.add_argument("--forces", action="store_true",
                        help="write forces_overlay.mp4 only — the contacts overlay does not "
                             "use the body mass, so re-running the dynamics stage cannot stale it")
    args = parser.parse_args()
    for target in _targets(args.clips):
        paths = render(target, forces_only=args.forces)
        print(f"[overlays] {target.name} -> {paths[0].parent}/"
              f"{{{', '.join(path.name for path in paths)}}}")
