"""Helpers the diagnostics share: tables, contact rates, the limb fold, overlay drawing.

Paper-style PNG tables (``_render_table``), confusion counts to rates (``_rates``),
the joint-to-limb fold and the downward load per limb (``_limb_members``,
``_limb_load``), and the overlay pieces the force renderers draw with — contact
discs, captions and the five stacked force plots (``_force_panel``, ``_wall_panel``).
They came out of the campus_board scorers and renderer, retired 2026-10-08 (see git
history for ``score_prediction_contacts.py``, ``score_pipeline_contacts.py`` and
``render_prediction_overlays.py``).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LIMBS = ["left_hand", "right_hand", "left_foot", "right_foot"]
LIMB_RGB = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40)]
SUM_RGB = (110, 70, 175)
PERSON_STYLE = ["-", "--", ":", "-."]

CONTACT_RGB = (225, 55, 45)
FREE_RGB = (45, 185, 95)
NODATA_RGB = (135, 135, 135)
WHITE = (255, 255, 255)

# The overlays keep every side a multiple of 16, or ffmpeg silently rescales them.
PANEL_SIZE = (720, 896)  # the plot column; its height also sets the video pane's height
RADIUS_INNER, RADIUS_STRIP, RADIUS_OUTER = 6, 9, 13  # the plate-vs-model disc
RADIUS_SOLO, RADIUS_SOLO_RIM = 8, 11  # the model-only disc, drawn with a white rim


def _wired(clip: str) -> np.ndarray:
    """Return ``(4,)`` bool — which limbs had a plate. Session 1 wired the hands only."""
    plates = np.ones(4, bool)
    if "peter" not in clip and "gianna" not in clip:
        plates[2:] = False
    return plates


def _rates(counts: dict[str, int]) -> dict[str, float | int | None]:
    """Turn a confusion count into precision, recall, F1 and accuracy.

    A rate whose denominator is zero is reported as ``None`` rather than 0.0 —
    "never predicted a contact" and "every contact was wrong" are not the same
    statement, and a JSON reader should be able to tell them apart.
    """
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {**counts,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
            "frames": tp + fp + fn + tn,
            "positives": tp + fn}


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


_ROW_H, _MARGIN = 0.34, 0.16  # inches: one table row, and the blank border around a table


def _table_size(columns: list[tuple[str, float, str]], rows: list[list[str]],
                groups: tuple[tuple[str, int, int], ...] = ()) -> tuple[float, float]:
    """The ``(width, height)`` in inches a table occupies, border included."""
    widths = np.array([width for _, width, _ in columns], float)
    return (0.92 * widths.sum(),
            2 * _MARGIN + _ROW_H * (1 + bool(groups) + len(rows)) + 0.02)


def _draw_table(fig, block_left: float, block_top: float,
                columns: list[tuple[str, float, str]], rows: list[list[str]],
                groups: tuple[tuple[str, int, int], ...] = (), rule_before: int | None = None,
                bold: frozenset[tuple[int, int]] = frozenset()) -> None:
    """Draw one minimal paper-style table into ``fig``: horizontal rules only, no colour.

    ``block_left`` / ``block_top`` place the table's top-left corner, in inches from
    the figure's top-left, so several tables can share one figure. ``columns`` is
    ``(label, width, align)`` per column with widths in arbitrary units, ``groups``
    optional ``(label, first, last)`` spans drawn above the header, ``rule_before`` a
    row index to separate with a thin rule, and ``bold`` a set of ``(row, column)``
    cells to emphasise.
    """
    widths = np.array([width for _, width, _ in columns], float)
    edges = np.concatenate([[0.0], np.cumsum(widths / widths.sum())])
    row_h, margin = _ROW_H, _MARGIN
    fig_width, fig_height = fig.get_size_inches()
    block_width, _ = _table_size(columns, rows, groups)

    left = (block_left + margin) / fig_width
    span = (block_width - 2 * margin) / fig_width

    def place(text: str, column: int, y: float, size: float, weight: str = "normal", colour: str = "black") -> None:
        align = columns[column][2]
        anchor = {"l": edges[column], "r": edges[column + 1], "c": 0.5 * (edges[column] + edges[column + 1])}[align]
        pad = {"l": 0.004, "r": -0.004, "c": 0.0}[align]
        fig.text(left + span * (anchor + pad), y, text, fontsize=size, weight=weight, color=colour,
                 ha={"l": "left", "r": "right", "c": "center"}[align], va="center")

    def rule(y: float, thickness: float, first: int = 0, last: int | None = None) -> None:
        last = len(columns) - 1 if last is None else last
        fig.add_artist(plt.Line2D([left + span * edges[first], left + span * edges[last + 1]], [y, y],
                                  color="black", lw=thickness, transform=fig.transFigure))

    top = 1 - (block_top + margin) / fig_height
    step = row_h / fig_height
    rule(top, 1.5)
    cursor = top - 0.55 * step
    if groups:
        for label, first, last in groups:
            centre = 0.5 * (edges[first] + edges[last + 1])
            fig.text(left + span * centre, cursor, label, fontsize=10, ha="center", va="center")
            rule(cursor - 0.42 * step, 0.7, first, last)
        cursor -= step
    for column, (label, _, _) in enumerate(columns):
        place(label, column, cursor, 10)
    rule(cursor - 0.45 * step, 0.9)  # the rule sits between the rows; it does not take a row of its own

    for index, row in enumerate(rows):
        cursor -= step
        if index == rule_before:
            rule(cursor + 0.5 * step, 0.5)
        for column, cell in enumerate(row):
            emphasis = "bold" if index == rule_before or (index, column) in bold else "normal"
            place(cell, column, cursor, 10, emphasis)
    rule(cursor - 0.5 * step, 1.5)


def _render_table(path: Path, columns: list[tuple[str, float, str]], rows: list[list[str]],
                  groups: tuple[tuple[str, int, int], ...] = (), rule_before: int | None = None,
                  bold: frozenset[tuple[int, int]] = frozenset()) -> None:
    """Write one table to its own PNG, sized to fit it exactly."""
    fig = plt.figure(figsize=_table_size(columns, rows, groups), dpi=200)
    fig.patch.set_facecolor("white")
    _draw_table(fig, 0.0, 0.0, columns, rows, groups, rule_before, bold)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


_FINGER = ("index", "middle", "pinky", "ring", "thumb")


def _limb_members(joint_names: list[str]) -> list[list[int]]:
    """Joint columns per limb, in ``LIMBS`` order — hand = wrist + fingers, foot = foot + ankle.

    Mirrors the retired dynamics stage's hand fold (wrist + fingers -> wrist)
    by name rather than by index, and without pulling torch into a CPU script. A name the
    list does not hold is simply skipped, so the same rule folds the body's 52 joints and
    the much shorter list of joints the force solve puts a contact on.
    """
    index = {name: column for column, name in enumerate(joint_names)}
    present = lambda *names: [index[name] for name in names if name in index]  # noqa: E731
    hands = [present(f"{side}_wrist")
             + [column for name, column in index.items()
                if name.startswith(f"{side}_") and any(finger in name for finger in _FINGER)]
             for side in ("left", "right")]
    feet = [present(f"{side}_foot", f"{side}_ankle") for side in ("left", "right")]
    return hands + feet


def _limb_load(kindyn: np.lib.npyio.NpzFile, person: int) -> np.ndarray:
    """``(N, 4)`` downward load in newtons per limb — the plate-comparable scalar.

    A limb's load is the sum over whichever contact points the solve gave it. Which
    points those are is read from the file rather than assumed, because it has changed:
    the solve used to push through six joints (the wrists, the feet and the ankles),
    then four (the wrists and the feet), and the frame-level solve pushes through
    contact frames whose forces reach the file already folded onto their parent
    joints — so there the groups are the 52 body joints.
    """
    gravity = np.asarray(kindyn["gravity_world"], float)
    down = gravity / np.linalg.norm(gravity)
    if "contact_force_joints" in kindyn.files:
        groups = [str(name) for name in np.asarray(kindyn["contact_force_joints"]).reshape(-1)]
        forces = np.asarray(kindyn["contact_forces"][person], float)
    else:
        groups = [str(name) for name in kindyn["joint_names"]]
        forces = np.asarray(kindyn["contact_forces_world"][person], float)
    load = -(forces @ down)  # (N, groups)
    members = _limb_members(groups)
    if not all(members):  # a limb with no contact point would silently read as zero load
        raise ValueError(f"no solved contact point for {LIMBS[members.index([])]} among {groups}")
    return np.stack([load[:, limb].sum(1) for limb in members], 1)


def _fit(source: tuple[int, int], height: int) -> tuple[int, int]:
    """Size a pane to ``height``, keeping the source aspect, width a multiple of 16.

    Rounding the width moves the aspect by well under a percent, which is why the
    drawing code carries separate x and y scales rather than one number.
    """
    width = max(16, int(round(height * source[0] / source[1] / 16)) * 16)
    return width, height


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
                floor: float = 50.0, limbs_measured: bool = False,
                measured_label: str | None = None, total_label: str = "wall total",
                limb_angle: np.ndarray | None = None):
    """Render the climb_wall's five stacked size plots once, mirroring :func:`_force_panel`.

    Returns the panel image, a function from clip time to its pixel column, and the
    rows the cursor should span — the same four things, so a frame loop does not care
    which rig it is drawing. Shared by every force overlay, so a model's and the
    reconstruction's forces are read against the wall the same way.

    ``limb_size (N, 4)`` is how big the predicted force at each limb is, ``model_total
    (N,)`` how big their vector sum is, ``measured_total (N,)`` how big the wall's own
    total is, and ``cell_size (N, 4)`` how big each corner cell's reading is, in the
    same limb order. Everything is a plain size in newtons, because the wall's cells
    and a 3D prediction share no axis convention worth trusting sideways.

    On the first wall only the top panel is a like-for-like comparison. The black line
    in a limb panel is that corner's cell, which is dominated by the nearest limb but
    mixed across all four — so it tracks *when* a limb loads up without being that
    limb's force. The limb panels stay shaded to keep saying so. Because both totals
    are the length of a vector sum, the top panel is not the four below it added up.

    ``limbs_measured`` is for rigs where each limb has its own sensor and a channel
    IS that limb's force (the ``climb_wall_2`` demonstrator, parkour): the limb
    panels then compare directly, so nothing is shaded and the black line is
    labelled as the limb's own sensor (``measured_label`` overrides the name;
    ``total_label`` names the top panel's black line). Measured rows may hold NaN
    where the rig recorded nothing — the black line simply gaps there.

    ``limb_angle (N, 4)`` — when given, each limb panel also carries the angle between
    the predicted and measured limb vectors as a dashed line in the limb's own colour,
    on its own right-hand axis (fixed 0–180°, so the force curves keep their scale),
    NaN where either side is too small to have a direction.
    """
    time = np.arange(len(model_total)) / fps
    fig, axes = plt.subplots(5, 1, figsize=(PANEL_SIZE[0] / 100, PANEL_SIZE[1] / 100),
                             dpi=100, sharex=True)
    rows = [(None, "total force size", SUM_RGB)] + \
           [(limb, LIMBS[limb].replace("_", " "), LIMB_RGB[limb]) for limb in range(4)]

    angle_twin = None
    for axis, (limb, name, colour) in zip(axes, rows):
        measured = measured_total if limb is None else cell_size[:, limb]
        axis.plot(time, measured, color="black", lw=1.3, alpha=0.8,
                  label=total_label if limb is None else
                  (measured_label or ("board" if limbs_measured else "corner cell")))
        if limb is not None and not limbs_measured:
            axis.set_facecolor("0.94")  # the first wall cannot measure one limb on its own
        curve = model_total if limb is None else limb_size[:, limb]
        axis.plot(time, curve, color=np.array(colour) / 255.0, lw=2.1, label=model_label)
        if limb is not None and limb_angle is not None:
            twin = axis.twinx()
            # Same colour as the limb's force line; long dashes are what say "angle".
            twin.plot(time, limb_angle[:, limb], color=np.array(colour) / 255.0,
                      lw=1.8, ls=(0, (5, 3)), label="angle")
            twin.set_ylim(0.0, 190.0)
            twin.set_yticks([0, 90, 180])
            twin.tick_params(labelsize=9, labelcolor="0.4")
            if angle_twin is None:
                angle_twin = twin

        span = np.concatenate([measured, curve])
        low, high = min(0.0, float(np.nanmin(span))), max(floor, float(np.nanmax(span)))
        headroom = 0.26 if limb is None else 0.22  # room for the in-panel name and the legend
        axis.set_ylim(low - 0.06 * (high - low), high + headroom * (high - low))
        shown = name if limb is None or limbs_measured else f"{name}  (cell, not this limb)"
        axis.text(0.008, 0.96, shown, transform=axis.transAxes, fontsize=13, va="top", ha="left",
                  color="0.15" if limb is None or limbs_measured else "0.45")
        axis.tick_params(labelsize=11)
        axis.grid(alpha=0.25, lw=0.6)

    # The top panel's black line and a limb panel's black line are different quantities,
    # so the first of each gets its own legend rather than one shared key up top.
    axes[0].legend(loc="upper right", fontsize=11, ncol=2, framealpha=0.9,
                   borderpad=0.3, handlelength=1.4)
    handles, labels = axes[1].get_legend_handles_labels()
    if angle_twin is not None:  # fold the twin's line in; draw on the twin so it sits on top
        twin_handles, twin_labels = angle_twin.get_legend_handles_labels()
        handles, labels = handles + twin_handles, labels + twin_labels
    (axes[1] if angle_twin is None else angle_twin).legend(
        handles, labels, loc="upper right", fontsize=11, ncol=3 if angle_twin is not None else 2,
        framealpha=0.9, borderpad=0.3, handlelength=1.4)
    axes[-1].set_xlim(0.0, float(time[-1]))
    fig.subplots_adjust(left=0.108, right=0.996 if limb_angle is None else 0.955,
                        top=0.996, bottom=0.038, hspace=0.07)
    fig.canvas.draw()
    panel = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

    height = panel.shape[0]
    column_of = lambda t: int(round(axes[0].transData.transform((t, 0.0))[0]))  # noqa: E731
    top = height - int(round(axes[0].transAxes.transform((0.0, 1.0))[1]))
    bottom = height - int(round(axes[-1].transAxes.transform((0.0, 0.0))[1]))
    plt.close(fig)
    return panel, column_of, top, bottom
