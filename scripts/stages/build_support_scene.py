"""Synthetic support scene: the fitted floor plane, plus the surfaces the body rests on.

The fused VGGT cloud is unreliable on the OpenCap LabValidation footage — good enough to
fit the flat floor (``geocalib/gravity.npz`` accepts a plane within 2 deg of the plates)
and not good enough for anything else. Every consumer of scene geometry reads one file,
``geometry/scene.npz``: the kinematics stages (penetration + contact attraction through
``tools.human_optim.optimizer``) and the contact readout (``estimate_contacts.py`` /
``tools.human_optim.contacts.detect_frame_contacts``). So this stage REPLACES that input
rather than touching any detector, and writes it with the fuse schema
(``points``/``colors``/``confidence``/``normals``/``count``/``voxel_size``/``world_scale``)
into the clip directory it is pointed at — a variant farm whose ``geometry/`` is a real
directory of symlinks to the real ``transform.npz`` / ``scale.npz`` / ``human.npz``.

Two things end up in the cloud.

**The floor** — a lattice on the accepted ground plane, normals ``up_world``, confidence 1,
covering ``ground_extent`` plus a margin (and the body's own footprint when a body is
given). Spacing is about half the fused voxel: the plane's *signed* distance
``(v - p)·n`` is exact at any spacing because every normal is ``up_world``, but the
Euclidean nearest-neighbour distance the ``max_dist_m`` guards use is inflated by up to
``spacing/sqrt(2)`` in the plane.

**Plateaus** — the two surfaces the trials rest on that neither the plates nor the cloud
report: the drop-jump box and the sit-to-stand chair. A contact frame that holds still at a
constant height above the floor is standing (or sitting) on something horizontal at that
height, so per person and frame group (left foot, right foot, sit bones) the stage clusters
the heights of the still frames and draws one horizontal disc per surviving mode. A surface
has to earn that, and :func:`detect_plateaus` documents each test: the mode must be
separated from the floor's own distribution, flat, held without a break, and — for the sit
bones — actually below the knees. The heights are measured, never assumed.

Runnable standalone and idempotent — it reads only ``geocalib/gravity.npz`` and the body,
so re-running rewrites the same cloud over the one it wrote before::

    uv run --no-sync python scripts/stages/build_support_scene.py \\
        --predictions out/<clip>/_floorplane --config configs/human_optimization_floorplane.toml
    uv run --no-sync python scripts/stages/build_support_scene.py \\
        --predictions out/<clip>/_floorplane --bodies kinematic_1 \\
        --config configs/human_optimization_floorplane.toml

Writes::

    geometry/scene.npz    the fuse schema + the plateau diagnostics (``plateau_*``,
                          one row per cluster considered, ``plateau_accepted`` marking
                          the ones that became patches) and the floor parameters
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.body import load_body  # noqa: E402
from tools.config import load_toml  # noqa: E402
from tools.npz_io import atomic_savez  # noqa: E402
from tools.human_optim import paths  # noqa: E402
from tools.human_optim.qmath import normalize_q  # noqa: E402

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "human_optimization.toml"

#: Lattice pitch of every synthetic surface, m — about half the fused voxel (0.048-0.054 m
#: on these clips), which bounds the in-plane nearest-neighbour error at spacing/sqrt(2).
SPACING_M = 0.025
#: Extra floor radius beyond ``ground_extent / 2`` (and beyond the body's footprint), m.
FLOOR_MARGIN_M = 0.5
#: Radius of one plateau patch, m — a foot or a seat plus the in-plane spread of the
#: still frames that found it.
PLATEAU_RADIUS_M = 0.35
#: Contact-frame stems that can carry a body's weight on a horizontal surface. ``toe_above``
#: is deliberately absent: it is the TOP of the toes, and a foot does not rest on it.
SUPPORT_FOOT_FRAMES = ("heel", "ball_inner", "ball_outer", "big_toe")
#: A cluster reaching this low is the floor (or its upper tail), not a surface above it.
#: The lattice already models the floor, and a tail that is merely truncated at this height
#: must never become a shelf — which is why clustering runs over every height from 0 up.
PLATEAU_MIN_HEIGHT_M = 0.10
#: Nothing a foot or a sit bone can rest on sits higher than this above the floor, m.
PLATEAU_MAX_HEIGHT_M = 0.80
#: Single-linkage gap that splits the still heights of a group into clusters, m.
PLATEAU_BIN_M = 0.02
#: Height range one surface captures around its own mode, m. Without it, single-linkage
#: chains a 9 cm ramp of still heights into one "surface" whose patch misses most of the
#: frames that found it, and a single sample bridging two real shelves welds them together.
PLATEAU_MAX_SPAN_M = 0.05
#: Height quantile of a cluster's members the patch is placed at. A plateau's normal points
#: one way (up), so anything under the patch is pushed up: the surface belongs under the
#: frames that found it, not through the middle of them.
PLATEAU_HEIGHT_Q = 0.10
#: Members further than this in-plane from the cluster's median centre are not on the same
#: surface and are dropped before the centre and the durations are measured, m.
PLATEAU_INPLANE_M = 0.5
#: A cluster becomes a patch once its still frames run this long WITHOUT A BREAK.
PLATEAU_MIN_STILL_S = 0.3
#: A sit cluster is a seat only if the sit bones are this far BELOW the knees (median over
#: the cluster's samples, m, negative = below). Standing sit bones ride ~0.29 m above the
#: knees and the bottom of a squat sits level with them; a real chair puts them below.
PLATEAU_SIT_BELOW_KNEE_M = -0.03

#: Point colours, so the viewer separates the two surfaces at a glance (RGB uint8).
FLOOR_COLOR = (128, 128, 128)
PLATEAU_COLOR = (214, 124, 48)


@dataclass(frozen=True)
class Plateau:
    """One horizontal surface a frame group was found resting on."""

    #: Frame group that found it: ``foot_left`` / ``foot_right`` / ``sit``.
    group: str
    #: Person row in the body npz.
    person: int
    #: Height above the fitted ground plane the patch is drawn at, m
    #: (:data:`PLATEAU_HEIGHT_Q` of the members' heights).
    height_m: float
    #: ``(3,)`` world centre of the patch, on the plateau's own height.
    center: np.ndarray
    #: Height of the lowest / highest member of the cluster, m.
    height_lo_m: float
    height_hi_m: float
    #: First and last video frame contributing to the cluster.
    frame_start: int
    frame_end: int
    #: Distinct video frames in the cluster, and the longest UNBROKEN run of them — the
    #: run is what acceptance is decided on.
    steps: int
    longest_run: int
    #: ``(frame, contact-frame)`` samples in the cluster.
    samples: int
    #: Median of (sit height − mean knee height) over the members, m; NaN off the sit group.
    sit_below_knee_m: float
    #: Whether the cluster held up as a surface and was drawn.
    accepted: bool
    #: Why it was not drawn (``""`` when it was): ``floor`` / ``short`` / ``high`` /
    #: ``above_knees``.
    reject: str


def plane_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal in-plane axes ``(3,), (3,)`` for a unit plane normal ``up (3,)``."""
    seed = np.eye(3, dtype=np.float32)[int(np.argmin(np.abs(up)))]
    first = np.cross(up, seed)
    first = first / np.linalg.norm(first)
    return first.astype(np.float32), np.cross(up, first).astype(np.float32)


def disc_points(center: np.ndarray, up: np.ndarray, radius_m: float,
                spacing_m: float) -> np.ndarray:
    """``(M, 3)`` lattice points on the horizontal disc of ``radius_m`` about ``center``.

    :param center: ``(3,)`` world centre, which the lattice is built symmetrically about.
    :param up: ``(3,)`` unit plane normal; every returned point satisfies
        ``(p - center)·up == 0`` to float32.
    """
    axis_u, axis_v = plane_basis(up)
    steps = np.arange(-int(np.ceil(radius_m / spacing_m)),
                      int(np.ceil(radius_m / spacing_m)) + 1, dtype=np.float32) * spacing_m
    grid_u, grid_v = np.meshgrid(steps, steps, indexing="ij")
    inside = grid_u**2 + grid_v**2 <= radius_m**2
    return (center[None] + grid_u[inside, None] * axis_u[None]
            + grid_v[inside, None] * axis_v[None]).astype(np.float32)


def frame_groups(frame_names: list[str]) -> dict[str, list[int]]:
    """Contact-frame columns per support group: ``foot_left`` / ``foot_right`` / ``sit``.

    Only the frames that can carry weight on a horizontal surface join a group
    (:data:`SUPPORT_FOOT_FRAMES`). Groups that the config's contact-frame allowlist left
    empty are dropped, so a body without sit bones simply never grows a seat.
    """
    groups: dict[str, list[int]] = {"foot_left": [], "foot_right": [], "sit": []}
    for column, name in enumerate(frame_names):
        base, _, side = name.rpartition("_")
        if base in SUPPORT_FOOT_FRAMES:
            groups[f"foot_{side}"].append(column)
        elif base == "sit":
            groups["sit"].append(column)
    return {name: columns for name, columns in groups.items() if columns}


def planar_offset(points: np.ndarray, centroid: np.ndarray, up: np.ndarray) -> np.ndarray:
    """In-plane part ``(..., 3)`` of ``points (..., 3) − centroid``, i.e. the height removed."""
    offset = points - centroid
    return offset - (offset @ up)[..., None] * up


def planar_radius(points: np.ndarray, centroid: np.ndarray, up: np.ndarray) -> float:
    """Largest finite in-plane distance of ``points (..., 3)`` from ``centroid``, or 0.

    Untracked frames carry NaN positions, which must not poison the floor's extent, so the
    reduction ignores them and an all-NaN input measures nothing rather than returning NaN.
    """
    radius = np.linalg.norm(planar_offset(points, centroid, up), axis=-1)
    return float(radius[np.isfinite(radius)].max(initial=0.0))


def _longest_run(steps: np.ndarray) -> int:
    """Longest run of consecutive integers in ``steps``, which need not be sorted."""
    unique = np.unique(steps)
    if unique.size == 0:
        return 0
    breaks = np.flatnonzero(np.diff(unique) > 1)
    starts, ends = np.r_[0, breaks + 1], np.r_[breaks, unique.size - 1]
    return int((ends - starts + 1).max())


def _height_modes(heights: np.ndarray) -> list[np.ndarray]:
    """Split sorted ``heights (M,)`` into flat modes; returns index arrays, lowest first.

    Densest :data:`PLATEAU_BIN_M` bin first: its median is a mode, and every height within
    half a :data:`PLATEAU_MAX_SPAN_M` of it belongs to that mode. Repeat on what is left.
    Seeking the mode rather than cutting at the widest gap is what keeps one sample bridging
    two shelves from welding them: the bridge joins whichever mode it is near, and the far
    shelf keeps its own.
    """
    remaining, modes = np.arange(heights.size), []
    while remaining.size:
        values = heights[remaining]
        bins = np.floor(values / PLATEAU_BIN_M).astype(np.int64)
        edges, counts = np.unique(bins, return_counts=True)
        centre = float(np.median(values[bins == edges[int(np.argmax(counts))]]))
        take = np.abs(values - centre) <= 0.5 * PLATEAU_MAX_SPAN_M
        modes.append(remaining[take])
        remaining = remaining[~take]
    return sorted(modes, key=lambda index: float(heights[index[0]]))


def detect_plateaus(points_world: np.ndarray, knee_height: np.ndarray, drift: np.ndarray,
                    valid: np.ndarray, groups: dict[str, list[int]], centroid: np.ndarray,
                    up: np.ndarray, *, still_m: float, fps: float,
                    person: int = 0) -> list[Plateau]:
    """Horizontal surfaces a person's frame groups rest on, one per height mode.

    A sample is ``(frame, contact frame)`` where the body is tracked, the contact frame sits
    at or above the floor plane, and its pre-enforcement drift is under ``still_m``. That
    stillness bar is deliberately stricter than the pipeline's — ``stillness_weight`` only
    reaches zero at ``2 · still_m`` — because a surface has to hold a frame still, not merely
    slow it down.

    Clustering runs over EVERY height from the plane up, so the upper tail of the floor
    distribution stays attached to the floor instead of being truncated into a shelf: heights
    are split wherever consecutive sorted values differ by more than :data:`PLATEAU_BIN_M`,
    any cluster reaching below :data:`PLATEAU_MIN_HEIGHT_M` is the floor, and what is left is
    split into flat modes at most :data:`PLATEAU_MAX_SPAN_M` wide (:func:`_height_modes`).
    Members further
    than :data:`PLATEAU_INPLANE_M` from the piece's median in-plane centre are on some other
    surface and are dropped. A piece becomes a patch when its still frames run
    :data:`PLATEAU_MIN_STILL_S` seconds WITHOUT A BREAK — scattered frames at one height are
    a coincidence, not a rest — and, for the sit group, only when the sit bones sit below the
    knees (:data:`PLATEAU_SIT_BELOW_KNEE_M`), which standing and the bottom of a squat do not.
    Every cluster is returned, accepted or not, with the reason it was dropped.

    :param points_world: ``(T, F, 3)`` posed contact-frame world positions.
    :param knee_height: ``(T,)`` mean knee-joint height above the plane, m (the sit test).
    :param drift: ``(T, F)`` persisted pre-enforcement net window drift, m (NaN = unknown).
    :param valid: ``(T,)`` bool — frames where this person is tracked.
    :param groups: support group name → contact-frame columns (:func:`frame_groups`).
    :param centroid: ``(3,)`` a point on the fitted ground plane.
    :param up: ``(3,)`` unit plane normal pointing away from the floor.
    """
    height = (points_world - centroid) @ up                       # (T, F)
    eligible = (np.isfinite(drift) & (drift < still_m) & valid[:, None]
                & np.isfinite(height) & (height >= 0.0))
    min_run = max(1, int(round(PLATEAU_MIN_STILL_S * fps)))
    found: list[Plateau] = []
    for group, columns in groups.items():
        rows, cols = np.nonzero(eligible[:, columns])
        if rows.size == 0:
            continue
        cols = np.asarray(columns)[cols]
        order = np.argsort(height[rows, cols], kind="stable")
        rows, cols = rows[order], cols[order]
        sorted_height = height[rows, cols]
        breaks = np.flatnonzero(np.diff(sorted_height) > PLATEAU_BIN_M)
        for lo, hi in zip(np.r_[0, breaks + 1], np.r_[breaks, sorted_height.size - 1]):
            if sorted_height[lo] <= PLATEAU_MIN_HEIGHT_M:
                floor_offset = np.median(planar_offset(
                    points_world[rows[lo:hi + 1], cols[lo:hi + 1]], centroid, up), axis=0)
                found.append(_plateau(group, person, rows[lo:hi + 1], sorted_height[lo:hi + 1],
                                      centroid, up, floor_offset, float("nan"), reject="floor"))
                continue
            for mode in _height_modes(sorted_height[lo:hi + 1]):
                mode_rows, mode_cols = rows[lo:hi + 1][mode], cols[lo:hi + 1][mode]
                mode_points = points_world[mode_rows, mode_cols]
                keep = _in_plane_members(mode_points, centroid, up)
                member_rows = mode_rows[keep]
                member_height = sorted_height[lo:hi + 1][mode][keep]
                offset = np.median(planar_offset(mode_points[keep], centroid, up), axis=0)
                below_knee = (float(np.median(member_height - knee_height[member_rows]))
                              if group == "sit" else float("nan"))
                run = _longest_run(member_rows)
                reject = ("short" if run < min_run else
                          "high" if member_height[0] > PLATEAU_MAX_HEIGHT_M else
                          "above_knees" if group == "sit"
                          and not below_knee < PLATEAU_SIT_BELOW_KNEE_M else "")
                found.append(_plateau(group, person, member_rows, member_height,
                                      centroid, up, offset, below_knee, reject=reject))
    return found


def _in_plane_members(points: np.ndarray, centroid: np.ndarray, up: np.ndarray) -> np.ndarray:
    """``(M,)`` bool: members within :data:`PLATEAU_INPLANE_M` of their own median centre."""
    offset = planar_offset(points, centroid, up)
    return np.linalg.norm(offset - np.median(offset, axis=0), axis=-1) <= PLATEAU_INPLANE_M


def _plateau(group: str, person: int, rows: np.ndarray, heights: np.ndarray,
             centroid: np.ndarray, up: np.ndarray, offset: np.ndarray,
             below_knee: float, *, reject: str) -> Plateau:
    """One :class:`Plateau` from a cluster's member frames, heights and in-plane centre."""
    patch_height = float(np.quantile(heights, PLATEAU_HEIGHT_Q))
    # A coordinate-wise median of in-plane vectors need not be in the plane; re-project it,
    # or the patch would sit millimetres off the height it reports.
    in_plane = offset - (offset @ up) * up
    return Plateau(
        group=group, person=person, height_m=patch_height,
        center=(centroid + in_plane + patch_height * up).astype(np.float32),
        height_lo_m=float(heights[0]), height_hi_m=float(heights[-1]),
        frame_start=int(rows.min()), frame_end=int(rows.max()),
        steps=int(np.unique(rows).size), longest_run=_longest_run(rows),
        samples=int(heights.size), sit_below_knee_m=below_knee,
        accepted=not reject, reject=reject)


def _posed_frames(body, betas: torch.Tensor, q: torch.Tensor, scale: float,
                  joint_columns: list[int], *, chunk: int = 64
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Pose every frame, as the stages do: ``(frames (T, F, 3), joints (T, len(cols), 3))``.

    Both are world positions with the stored per-person scale re-applied about the pelvis.
    """
    frames, joints = [], []
    for start in range(0, q.shape[0], chunk):
        end = min(start + chunk, q.shape[0])
        out = body.forward(betas[None].expand(end - start, -1),
                           normalize_q(q[start:end], body.num_joints))
        root = out.joints[:, :1, :]
        frames.append(root + scale * (out.frames_world[..., :3] - root))
        joints.append(root + scale * (out.joints[:, joint_columns, :] - root))
    return torch.cat(frames, 0), torch.cat(joints, 0)


def _plateau_arrays(plateaus: list[Plateau]) -> dict:
    """The ``plateau_*`` diagnostic arrays, one row per cluster considered."""
    return dict(
        plateau_group=np.array([p.group for p in plateaus], dtype=np.str_),
        plateau_person=np.array([p.person for p in plateaus], dtype=np.int32),
        plateau_height_m=np.array([p.height_m for p in plateaus], dtype=np.float32),
        plateau_center=(np.stack([p.center for p in plateaus]) if plateaus
                        else np.zeros((0, 3), np.float32)).astype(np.float32),
        plateau_height_lo_m=np.array([p.height_lo_m for p in plateaus], dtype=np.float32),
        plateau_height_hi_m=np.array([p.height_hi_m for p in plateaus], dtype=np.float32),
        plateau_frame_start=np.array([p.frame_start for p in plateaus], dtype=np.int32),
        plateau_frame_end=np.array([p.frame_end for p in plateaus], dtype=np.int32),
        plateau_steps=np.array([p.steps for p in plateaus], dtype=np.int32),
        plateau_longest_run=np.array([p.longest_run for p in plateaus], dtype=np.int32),
        plateau_samples=np.array([p.samples for p in plateaus], dtype=np.int32),
        plateau_sit_below_knee_m=np.array([p.sit_below_knee_m for p in plateaus],
                                          dtype=np.float32),
        plateau_accepted=np.array([p.accepted for p in plateaus], dtype=bool),
        plateau_reject=np.array([p.reject for p in plateaus], dtype=np.str_),
        plateau_radius_m=np.float32(PLATEAU_RADIUS_M),
        plateau_min_height_m=np.float32(PLATEAU_MIN_HEIGHT_M),
        plateau_max_height_m=np.float32(PLATEAU_MAX_HEIGHT_M),
        plateau_bin_m=np.float32(PLATEAU_BIN_M),
        plateau_max_span_m=np.float32(PLATEAU_MAX_SPAN_M),
        plateau_height_q=np.float32(PLATEAU_HEIGHT_Q),
        plateau_inplane_m=np.float32(PLATEAU_INPLANE_M),
        plateau_min_still_s=np.float32(PLATEAU_MIN_STILL_S),
        plateau_sit_below_knee_max_m=np.float32(PLATEAU_SIT_BELOW_KNEE_M),
    )


def run(predictions_dir: str | Path, config_path: str | Path, *,
        bodies: str | None = None) -> Path:
    """Write the clip's synthetic ``geometry/scene.npz``; return its path."""
    cfg = load_toml(Path(config_path))
    pred = Path(predictions_dir)
    gravity_path = pred / "geocalib" / "gravity.npz"
    if not gravity_path.exists():
        raise SystemExit(f"missing {gravity_path} — run estimate_gravity_geocalib.py first")
    geo = dict(np.load(gravity_path, allow_pickle=True))
    if not bool(geo.get("ground_accepted", False)):
        raise SystemExit(f"[build_support_scene] {gravity_path} accepted no ground plane "
                         f"({geo.get('ground_reject', '')!r}) — there is no floor to build on")
    centroid = np.asarray(geo["ground_centroid"], np.float32)
    # The lattice IS the fitted plane, so its normal is that plane's own ``ground_normal``,
    # not ``up_world`` — those two differ whenever gravity came from GeoCalib instead of the
    # floor fit (the fit tolerates up to ground_max_tilt_deg between them). ``up_world`` only
    # settles which way is up.
    up = np.asarray(geo["ground_normal"], np.float32)
    up = up / np.linalg.norm(up)
    if float(up @ np.asarray(geo["up_world"], np.float32)) < 0.0:
        up = -up
    floor_radius = 0.5 * float(geo["ground_extent"]) + FLOOR_MARGIN_M

    plateaus: list[Plateau] = []
    patches: list[np.ndarray] = []
    if bodies is not None:
        kin_path = paths.body_path(pred, bodies)
        if not kin_path.exists():
            raise SystemExit(f"missing {kin_path} — run human_optimization {bodies} first")
        kin = dict(np.load(kin_path, allow_pickle=True))
        device = cfg["runtime"]["device"]
        body = load_body(device=device, frames_allow=cfg["contacts"].get("frames") or None,
                         sole_offsets=cfg["contacts"].get("sole_offset_m") or None)
        frame_names = list(body.contact_frame_names)
        if [str(n) for n in kin["contact_frame_names"]] != frame_names:
            raise SystemExit(f"[build_support_scene] {kin_path} was solved on a different "
                             "contact-frame set than this config builds — refusing to guess")
        groups = frame_groups(frame_names)
        joint_names = [str(n) for n in kin["joint_names"]]
        knee_columns = [joint_names.index(n) for n in ("left_knee", "right_knee")]
        fps = float(np.load(pred / "geometry" / "transform.npz")["fps"])
        still_m = float(cfg["contact_gate"]["still_m"])
        valid_mask = np.asarray(kin["valid_mask"], bool)
        drift_pre = np.asarray(kin["drift_pre"], np.float32)
        for person in range(int(kin["num_people"])):
            frames, knees = _posed_frames(
                body, torch.as_tensor(kin["betas"][person], dtype=torch.float32, device=device),
                torch.as_tensor(kin["q"][person], dtype=torch.float32, device=device),
                float(kin["scale"][person]), knee_columns)
            points = frames.cpu().numpy()
            knee_height = ((knees.cpu().numpy() - centroid) @ up).mean(1)
            plateaus += detect_plateaus(points, knee_height, drift_pre[person],
                                        valid_mask[person], groups, centroid, up,
                                        still_m=still_m, fps=fps, person=person)
            floor_radius = max(floor_radius,
                               planar_radius(points[valid_mask[person]], centroid, up)
                               + FLOOR_MARGIN_M)
        for plateau in plateaus:
            print(f"[build_support_scene] plateau {plateau.group} p{plateau.person}: "
                  f"h={plateau.height_m:.3f} m span=[{plateau.height_lo_m:.3f}, "
                  f"{plateau.height_hi_m:.3f}] frames {plateau.frame_start}-{plateau.frame_end} "
                  f"({plateau.steps} still, longest run {plateau.longest_run}, "
                  f"{plateau.samples} samples, sit-knee {plateau.sit_below_knee_m:+.3f} m) "
                  f"{'ACCEPTED' if plateau.accepted else 'rejected: ' + plateau.reject}")
            if plateau.accepted:
                patches.append(disc_points(plateau.center, up, PLATEAU_RADIUS_M, SPACING_M))

    floor = disc_points(centroid, up, floor_radius, SPACING_M)
    points = np.concatenate([floor, *patches], 0).astype(np.float32)
    colors = np.concatenate(
        [np.tile(np.array(FLOOR_COLOR, np.uint8), (len(floor), 1)),
         *(np.tile(np.array(PLATEAU_COLOR, np.uint8), (len(p), 1)) for p in patches)], 0)
    transform = np.load(pred / "geometry" / "transform.npz")
    out_path = pred / "geometry" / "scene.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_savez(
        out_path,
        points=points, colors=colors,
        confidence=np.ones(len(points), np.float32),
        normals=np.tile(up, (len(points), 1)).astype(np.float32),
        count=np.ones(len(points), np.int32),
        voxel_size=np.float32(SPACING_M),
        world_scale=np.float32(transform["scale"]),
        synthetic=np.bool_(True),
        floor_center=centroid, floor_normal=up,
        floor_radius_m=np.float32(floor_radius),
        floor_points=np.int32(len(floor)),
        floor_margin_m=np.float32(FLOOR_MARGIN_M),
        spacing_m=np.float32(SPACING_M),
        bodies=np.str_(bodies or ""),
        **_plateau_arrays(plateaus))
    print(f"[build_support_scene] wrote {out_path}: {len(floor)} floor pts "
          f"(r={floor_radius:.2f} m, spacing {SPACING_M * 100:.1f} cm) + "
          f"{len(points) - len(floor)} pts on {len(patches)} plateau(s)")
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions", required=True, type=Path,
                    help="clip dir whose geometry/ the scene is written into (a variant farm)")
    ap.add_argument("--bodies", default=None, choices=["kinematic_1", "kinematic_2"],
                    help="body under human_optim/ to read the plateaus off; omit for a "
                         "floor-only scene (the pass that runs before any body exists)")
    ap.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    args = ap.parse_args(argv)
    run(args.predictions, args.config, bodies=args.bodies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
