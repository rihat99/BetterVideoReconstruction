"""The physics-free rival of the force witness: relative gap + stillness, nothing else.

The force witness claims that solving contact forces finds contacts geometry cannot
see. That claim is only worth anything if a detector built from the *same two
quantities the force prices use* — how far a contact frame is from the scene relative
to the other frames of that step, and how still it is — cannot do the same job. This
script builds that rival, gives it the same tuning privilege the force recipe had, and
scores both through the judge's own fold.

Everything structural is shared with the shipped stage, so the only difference is the
signal:

  * the same 12 contact frames on the same ``kinematic_1`` bodies;
  * the same candidate policy — a frame-step with no scene point in the 1 m search box
    is missing evidence, not "far", and is not a candidate
    (:data:`tools.human_optim.forces.NO_SCENE_M`);
  * the same relative gap, computed exactly as
    :func:`tools.human_optim.forces.solve_frame_forces` computes it (distance minus the
    step's nearest *candidate* frame, clamped to ``[0, 1]``);
  * the same stillness — the persisted pre-enforcement ``drift_pre``, never re-measured;
  * the same readout policy — minimum span, knee columns dropped, union with the
    geometric gates, at most one toe per foot.

The label rule is one family with three knobs: a frame is in contact when

    ‖( relgap / d★ , speed / v★ )‖_p  <  1

so ``p = ∞`` is the plain AND of two thresholds, ``p = 2`` an ellipse, ``p = 1`` a
straight trade. The left-hand side doubles as the ranking score (smaller = more in
contact), which is what breaks toe conflicts and what the threshold-free AP measures.
Raising the radius above 1 sweeps the whole precision-recall frontier without
re-gridding, because it just scales ``d★`` and ``v★`` together.

Nothing is written into ``out/<scene>/``. Usage::

    # grid the control on the 13 in-sample scenes, and the force arm beside it
    uv run --no-sync python scripts/diagnostics/contacts_control.py --grid --force
    # a frozen recipe on a held-out farm
    uv run --no-sync python scripts/diagnostics/contacts_control.py \\
        --root out/_expK_heldout --annotations scratch/force_recovery/heldout_data \\
        --recipe 0.05 0.20 inf
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Three agents share this checkout, and a mid-run edit to tools/human_optim would silently
# change what this control measures. ``BVR_PIN=<dir>`` (a ``git archive`` of tools/ + the
# judge at a chosen commit) takes precedence over the live tree, so a readout stays frozen
# while the library moves underneath it.
_PIN = os.environ.get("BVR_PIN")
if _PIN:
    for _path in (str(Path(_PIN) / "scripts"), str(_PIN)):
        sys.path.insert(0, _path)

from score_frame_contacts import _GROUPS, _add_counts, _empty_counts, _pool, fold_for_schema, metrics  # noqa: E402

from tools.body import frame_normals, load_body  # noqa: E402
from tools.config import load_toml  # noqa: E402
from tools.geometry.normals import load_scene_cloud  # noqa: E402
from tools.human_optim.camera import world_from_cam  # noqa: E402
from tools.human_optim.contacts import (  # noqa: E402
    DETECT_MIN_FRAMES, detect_frame_contacts, frame_scene_signals, joint_drift,
    merge_contact_spans,
)
from tools.human_optim.forces import (  # noqa: E402
    NO_SCENE_M, SEARCH_MARGIN_M, ForceKnobs, force_labels, resolve_toe_conflicts,
    solve_person_forces,
)
from tools.human_optim.qmath import normalize_q  # noqa: E402

_CONFIG = _REPO_ROOT / "configs" / "human_optimization.toml"
_OUT_DIR = _REPO_ROOT / "scratch" / "round2_control"

#: The frozen shipped force recipe (the round-2 simplification baked the rest in).
_FORCE_KNOBS = ForceKnobs(gap_dstar_m=0.05, stillness_vstar_mps=0.2)

#: The three scenes whose bodies float off the wall — the force witness's headline claim.
_FLOATERS = ("gtIGgs_y2nY_0024", "vcRBT9Q9jXM_0029", "9RvrpYoWDoo_0001")

#: Score below which a non-candidate frame can never be labelled or win a toe tiebreak.
_UNRANKED = -1e9


class PersonSignals:
    """One person's physics-free inputs, all ``(T, 12)`` numpy unless noted."""

    def __init__(self, gates: np.ndarray, relgap: np.ndarray, speed: np.ndarray,
                 candidate: np.ndarray, valid: np.ndarray, object_id: int) -> None:
        #: Geometric detector labels — the shipped gates, recomputed from this body.
        self.gates = gates
        #: Distance to the scene minus the step's nearest candidate frame, clamped [0, 1] m.
        self.relgap = relgap
        #: Pre-enforcement drift / window, m/s.
        self.speed = speed
        #: Frames the control is allowed to label (scene evidence exists and the step is tracked).
        self.candidate = candidate
        #: ``(T,)`` bool — steps where this person is tracked.
        self.valid = valid
        self.object_id = object_id


def prepare_scene(stem: str, root: Path, body, config: dict, device: str, *,
                  drift_from_body: bool) -> tuple[list[PersonSignals], dict]:
    """Load one scene tree and compute the gates plus the two control signals."""
    pred = root / stem
    kin = dict(np.load(pred / "human_optim" / "kinematic_1.npz", allow_pickle=True))
    transform = dict(np.load(pred / "geometry" / "transform.npz", allow_pickle=True))
    scene_npz = dict(np.load(pred / "geometry" / "scene.npz", allow_pickle=True))
    contacts = config["contacts"]
    gate = config["contact_gate"]

    extrinsics = torch.as_tensor(transform["extrinsics"], dtype=torch.float32, device=device)
    scene_pts, scene_normals, scene_conf = load_scene_cloud(
        scene_npz, world_from_cam(extrinsics)[:, :3, 3],
        conf_drop_q=float(contacts["conf_drop_q"]), k=int(contacts["normal_k"]), device=device)

    fps = float(transform.get("fps", kin.get("fps", 30.0)))
    window_s = float(gate["window_s"])
    valid_mask = np.asarray(kin["valid_mask"], bool)
    frame_names = list(body.contact_frame_names)

    people: list[PersonSignals] = []
    for person, object_id in enumerate([int(o) for o in kin["object_ids"]]):
        betas = torch.as_tensor(kin["betas"][person], dtype=torch.float32, device=device)
        q = torch.as_tensor(kin["q"][person], dtype=torch.float32, device=device)
        posed = _pose(body, betas, q, float(kin["scale"][person]))
        points, normals = posed[..., :3], frame_normals(posed)
        valid = valid_mask[person]
        valid_t = torch.as_tensor(valid, device=device)

        # The gate path's drift, exactly as estimate_contacts builds it: re-measured on
        # this body, then overwritten element-wise by the persisted pre-enforcement value.
        drift = joint_drift(points, fps=fps, window_s=window_s, valid_mask=valid_t)
        persisted = torch.as_tensor(np.asarray(kin["drift_pre"])[person],
                                    dtype=drift.dtype, device=drift.device)
        if persisted.shape == drift.shape:
            drift = torch.where(torch.isfinite(persisted) & torch.isfinite(drift),
                                persisted, drift)

        detected = detect_frame_contacts(
            points, normals, drift, scene_pts, scene_normals, scene_conf,
            frame_names=frame_names,
            close_thresh_m=float(contacts["close_thresh_m"]),
            exit_thresh_m=float(contacts["exit_thresh_m"]),
            min_frames=DETECT_MIN_FRAMES, still_m=float(gate["still_m"]),
            max_dist_m=float(contacts["max_dist_m"]),
            conf_thresh=float(contacts["conf_thresh"]),
            prune_margin_m=float(contacts["prune_margin_m"]))
        gates = (merge_contact_spans(
            detected.contact, points, detected.guarded_sdf,
            merge_dist_m=float(contacts["merge_dist_m"]),
            max_gap_s=float(contacts["merge_max_gap_s"]),
            gap_sdf_m=float(contacts["merge_sdf_m"]), fps=fps) & ~detected.toe_suppressed
        ).cpu().numpy()
        gates[~valid] = False

        # The force path's own nearest-neighbour: a metre-wide box, so an empty one really
        # means no reconstructed geometry anywhere near, not that the box was small.
        _, dist_m, _, _ = frame_scene_signals(
            points, normals, scene_pts, scene_normals, scene_conf,
            prune_margin_m=SEARCH_MARGIN_M)

        # Stillness must be the PERSISTED pre-enforcement drift. The one exception is a
        # body no contact term ever acted on, where the freshly measured drift already is
        # pre-enforcement (the pure-visual control).
        if drift_from_body:
            drift_pre = joint_drift(points, fps=fps, window_s=window_s, valid_mask=valid_t)
        else:
            drift_pre = torch.as_tensor(np.asarray(kin["drift_pre"])[person],
                                        dtype=torch.float32, device=device)
            if not bool(torch.isfinite(drift_pre).all()):
                raise SystemExit(f"{stem}: drift_pre has non-finite entries — refusing to "
                                 "re-measure stillness on a possibly enforced body")

        candidate, relgap = _relative_gap(dist_m, valid_t)
        speed = drift_pre.cpu().numpy() / window_s
        speed = np.where(np.isfinite(speed), speed, np.inf)
        people.append(PersonSignals(gates, relgap.cpu().numpy(), speed,
                                    candidate.cpu().numpy(), valid, object_id))

    meta = dict(fps=fps, window_s=window_s, frame_names=frame_names,
                object_ids=[p.object_id for p in people], valid_mask=valid_mask,
                num_frames=int(kin["num_frames"]),
                knee_columns=[i for i, n in enumerate(frame_names) if n.startswith("knee")],
                min_frames=DETECT_MIN_FRAMES)
    return people, meta


@torch.no_grad()
def _pose(body, betas: torch.Tensor, q: torch.Tensor, scale: float, *, chunk: int = 64
          ) -> torch.Tensor:
    """Pose every frame → contact-frame world poses ``(T, 12, 7)``, as the stage does."""
    posed = []
    for start in range(0, q.shape[0], chunk):
        end = min(start + chunk, q.shape[0])
        out = body.forward(betas[None].expand(end - start, -1),
                           normalize_q(q[start:end], body.num_joints))
        frames = out.frames_world.clone()
        root = out.joints[:, :1, :]
        frames[..., :3] = root + scale * (frames[..., :3] - root)
        posed.append(frames)
    return torch.cat(posed, 0)


def _relative_gap(dist_m: torch.Tensor, valid: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """The candidate mask and relative gap, byte-for-byte the force solve's own recipe.

    Reproduces :func:`tools.human_optim.forces.solve_frame_forces`: a frame with no scene
    point in the search box is *not* a candidate, and the datum the gap is measured
    against is the nearest **candidate** frame of that step.
    """
    known = torch.isfinite(dist_m) & (dist_m < NO_SCENE_M)
    candidate = known & valid[:, None]
    gap = torch.where(known, dist_m, torch.zeros_like(dist_m))
    near = torch.where(candidate, gap, torch.full_like(gap, float("inf"))).amin(dim=1)
    gap = gap - torch.where(torch.isfinite(near), near, torch.zeros_like(near))[:, None]
    return candidate, gap.clamp(0.0, 1.0)


def control_score(person: PersonSignals, gap_m: float, speed_mps: float, power: float
                  ) -> np.ndarray:
    """``(T, 12)`` distance-to-contact ``‖(relgap/d★, speed/v★)‖_p`` — smaller is more contact.

    Non-candidate frames get ``+inf``, so no threshold and no toe tiebreak can ever pick one.
    """
    scaled_gap = person.relgap / gap_m
    scaled_speed = person.speed / speed_mps
    if math.isinf(power):
        norm = np.maximum(scaled_gap, scaled_speed)
    else:
        norm = (scaled_gap ** power + scaled_speed ** power) ** (1.0 / power)
    return np.where(person.candidate, norm, np.inf)


def control_labels(people: list[PersonSignals], meta: dict, *, gap_m: float,
                   speed_mps: float, power: float, radius: float = 1.0
                   ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The control's final labels per person, under the shipped stage's own policy.

    The score is turned into a magnitude-like quantity (``radius − norm``, positive exactly
    where the rule fires) so the *identical* policy functions the force readout uses apply
    unchanged: threshold, one toe per foot decided by the score, minimum span, knee columns
    dropped, union with the gates, and a final toe resolution on the union.
    """
    labels, scores = [], []
    for person in people:
        norm = control_score(person, gap_m, speed_mps, power)
        magnitude = np.where(np.isfinite(norm), radius - norm, _UNRANKED)
        fired, _ = force_labels(magnitude, frame_names=meta["frame_names"],
                                valid=person.valid, threshold_n=0.0,
                                min_frames=meta["min_frames"])
        fired[:, meta["knee_columns"]] = False
        union = person.gates | fired
        union[~person.valid] = False
        labels.append(resolve_toe_conflicts(union, magnitude, frame_names=meta["frame_names"],
                                            min_frames=meta["min_frames"]))
        scores.append(magnitude)
    return labels, scores


def force_arm(stem: str, root: Path, body, config: dict, device: str, people: list[PersonSignals],
              meta: dict, thetas: tuple[float, ...], *, drift_from_body: bool
              ) -> tuple[dict[float, list[np.ndarray]], list[np.ndarray]]:
    """Re-solve the shipped force witness on the same bodies → labels per θ, plus ``|f|/BW``."""
    pred = root / stem
    kin = dict(np.load(pred / "human_optim" / "kinematic_1.npz", allow_pickle=True))
    transform = dict(np.load(pred / "geometry" / "transform.npz", allow_pickle=True))
    scene_npz = dict(np.load(pred / "geometry" / "scene.npz", allow_pickle=True))
    contacts = config["contacts"]

    extrinsics = torch.as_tensor(transform["extrinsics"], dtype=torch.float32, device=device)
    scene_pts, scene_normals, scene_conf = load_scene_cloud(
        scene_npz, world_from_cam(extrinsics)[:, :3, 3],
        conf_drop_q=float(contacts["conf_drop_q"]), k=int(contacts["normal_k"]), device=device)

    gravity_path = pred / "geocalib" / "gravity.npz"
    gravity = (np.asarray(np.load(gravity_path)["gravity_world"], np.float32)
               if gravity_path.exists() else np.array([0.0, 1.0, 0.0], np.float32))

    per_theta: dict[float, list[np.ndarray]] = {theta: [] for theta in thetas}
    relative_magnitude: list[np.ndarray] = []
    for person, signals in enumerate(people):
        betas = torch.as_tensor(kin["betas"][person], dtype=torch.float32, device=device)
        q = torch.as_tensor(kin["q"][person], dtype=torch.float32, device=device)
        drift_pre = torch.as_tensor(signals.speed * meta["window_s"],
                                    dtype=torch.float32, device=device)
        if drift_from_body:
            drift_pre = torch.nan_to_num(drift_pre, posinf=0.0)
        solution = solve_person_forces(
            body, betas, q, signals.valid, drift_pre=drift_pre,
            drift_window_s=meta["window_s"], gravity_world=gravity, fps=meta["fps"],
            scene_pts=scene_pts, scene_normals=scene_normals, scene_conf=scene_conf,
            knobs=_FORCE_KNOBS)
        relative_magnitude.append(solution.magnitude / solution.body_weight_n)
        for theta in thetas:
            label, _ = force_labels(solution.magnitude, frame_names=meta["frame_names"],
                                    valid=signals.valid,
                                    threshold_n=theta * solution.body_weight_n,
                                    min_frames=meta["min_frames"])
            label[:, meta["knee_columns"]] = False
            union = signals.gates | label
            union[~signals.valid] = False
            per_theta[theta].append(resolve_toe_conflicts(
                union, solution.magnitude, frame_names=meta["frame_names"],
                min_frames=meta["min_frames"]))
    return per_theta, relative_magnitude


def score_labels(stem: str, labels: list[np.ndarray], meta: dict, annotations: Path
                 ) -> dict[str, dict[str, int]]:
    """Per-coarse-joint confusion counts of ``(T, 12)`` labels — the judge's own fold."""
    ann = np.load(annotations / f"{stem}.npz", allow_pickle=True)
    ann_names = [str(n) for n in ann["joint_names"]]
    ann_contacts = np.transpose(ann["contacts"].astype(np.int8), (0, 2, 1))
    ann_ids = [int(o) for o in ann["object_ids"]]
    ignored = ann["ignored"].astype(bool)
    fold, _ = fold_for_schema(ann_names)
    ann_column = {name: i for i, name in enumerate(ann_names)}
    frame_index = {name: i for i, name in enumerate(meta["frame_names"])}

    joints = {name: _empty_counts() for name in fold}
    for person, object_id in enumerate(meta["object_ids"]):
        if object_id not in ann_ids or bool(ignored[ann_ids.index(object_id)]):
            continue
        ann_person = ann_ids.index(object_id)
        for name, frames in fold.items():
            predicted = np.any(labels[person][:, [frame_index[f] for f in frames]], axis=1)
            truth = ann_contacts[ann_person][:, ann_column[name]]
            keep = (truth >= 0) & meta["valid_mask"][person]
            _add_counts(joints[name], predicted[keep], truth[keep] == 1)
    return joints


def ranking_cells(stem: str, labels_scores: list[np.ndarray], gates: list[np.ndarray],
                  meta: dict, annotations: Path) -> dict[str, np.ndarray]:
    """Flatten every scorable ``(person, step, coarse joint)`` cell for threshold-free ranking."""
    ann = np.load(annotations / f"{stem}.npz", allow_pickle=True)
    ann_names = [str(n) for n in ann["joint_names"]]
    ann_contacts = np.transpose(ann["contacts"].astype(np.int8), (0, 2, 1))
    ann_ids = [int(o) for o in ann["object_ids"]]
    ignored = ann["ignored"].astype(bool)
    fold, _ = fold_for_schema(ann_names)
    ann_column = {name: i for i, name in enumerate(ann_names)}
    frame_index = {name: i for i, name in enumerate(meta["frame_names"])}

    out: dict[str, list[np.ndarray]] = {"score": [], "truth": [], "gated": []}
    for person, object_id in enumerate(meta["object_ids"]):
        if object_id not in ann_ids or bool(ignored[ann_ids.index(object_id)]):
            continue
        for name, frames in fold.items():
            columns = [frame_index[f] for f in frames]
            truth = ann_contacts[ann_ids.index(object_id)][:, ann_column[name]]
            keep = (truth >= 0) & meta["valid_mask"][person]
            out["score"].append(labels_scores[person][:, columns].max(axis=1)[keep])
            out["truth"].append(truth[keep] == 1)
            out["gated"].append(gates[person][:, columns].any(axis=1)[keep])
    return {key: np.concatenate(value) if value else np.zeros(0)
            for key, value in out.items()}


def average_precision(scores: np.ndarray, truth: np.ndarray) -> dict[str, float | int | None]:
    """AUROC and average precision of one score against a boolean truth."""
    from scipy.stats import rankdata

    positives, negatives = int(truth.sum()), int((~truth).sum())
    if not positives or not negatives:
        return {"n": int(truth.size), "pos": positives, "auroc": None, "ap": None}
    ranks = rankdata(scores)
    auroc = (ranks[truth].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    order = np.argsort(-scores, kind="stable")
    hits = truth[order].cumsum()
    precision = hits / np.arange(1, truth.size + 1)
    return {"n": int(truth.size), "pos": positives, "auroc": float(auroc),
            "ap": float(precision[truth[order]].sum() / positives)}


def write_fake_tree(destination: Path, stem: str, labels: list[np.ndarray], source: Path,
                    tag: str = "1") -> Path:
    """Copy a real ``contacts_<tag>.npz`` with only ``frame_contact`` replaced, for the judge."""
    out_dir = destination / stem / "human_optim"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(source, allow_pickle=True))
    data["frame_contact"] = np.stack(labels)
    path = out_dir / f"contacts_{tag}.npz"
    np.savez(path, **data)
    return path


def _pool_scene_rows(rows: list[dict[str, dict[str, int]]], *,
                     drop: set[str] = frozenset()) -> dict[str, int]:
    return _pool([counts for joints in rows for name, counts in joints.items()
                  if name not in drop])


def _fmt(counts: dict[str, int]) -> str:
    row = metrics(counts)
    show = lambda value: "      -" if value is None else f"{value:7.3f}"  # noqa: E731
    return (f"{row['tp']:6d} {row['fp']:6d} {row['fn']:6d} "
            f"{show(row['precision'])}{show(row['recall'])}{show(row['f1'])}")


_HEAD = f"{'arm':38s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'prec':>7s} {'rec':>7s} {'F1':>7s}"

#: The grid the control is tuned on — the same privilege the force θ had.
_GAP_GRID = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.60, 1.00)
_SPEED_GRID = (0.03, 0.05, 0.08, 0.12, 0.20, 0.30, 0.50, 0.80, 1.50, 5.00)
_POWER_GRID = (1.0, 2.0, float("inf"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=_REPO_ROOT / "out",
                        help="scene-tree root (default out/)")
    parser.add_argument("--scenes", nargs="*", default=None, help="scene stems (default: all)")
    parser.add_argument("--annotations", type=Path, default=_REPO_ROOT / "data",
                        help="annotation root holding <stem>.npz")
    parser.add_argument("--grid", action="store_true", help="tune the control's thresholds")
    parser.add_argument("--recipe", nargs=3, metavar=("GAP_M", "SPEED_MPS", "POWER"),
                        default=None, help="evaluate one frozen recipe")
    parser.add_argument("--radius", nargs="*", type=float, default=[1.0],
                        help="rule radii to sweep for the precision-recall frontier")
    parser.add_argument("--force", action="store_true", help="also re-solve the force witness")
    parser.add_argument("--force-thetas", nargs="*", type=float, default=[0.05],
                        help="force thresholds as a fraction of body weight")
    parser.add_argument("--drift-from-body", action="store_true",
                        help="measure stillness on the supplied body; ONLY valid when no "
                             "contact term ever acted on it (pure-visual control)")
    parser.add_argument("--write-trees", type=Path, default=None,
                        help="write contacts_1.npz-shaped labels here for the real judge")
    parser.add_argument("--json", type=Path, default=None, help="write the full numbers here")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    scenes = args.scenes or sorted(
        path.stem for path in args.annotations.glob("*.npz")
        if (args.root / path.stem / "human_optim" / "kinematic_1.npz").exists())
    config = load_toml(_CONFIG)
    body = load_body(device=args.device)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    prepared: dict[str, tuple[list[PersonSignals], dict]] = {}
    gate_counts: dict[str, dict] = {}
    force_counts: dict[float, dict[str, dict]] = {theta: {} for theta in args.force_thetas}
    force_rank: list[dict[str, np.ndarray]] = []
    for stem in scenes:
        started = time.time()
        people, meta = prepare_scene(stem, args.root, body, config, args.device,
                                     drift_from_body=args.drift_from_body)
        prepared[stem] = (people, meta)
        gate_counts[stem] = score_labels(stem, [p.gates for p in people], meta, args.annotations)
        candidate_rate = float(np.mean([p.candidate[p.valid].mean() for p in people]))
        print(f"[{stem}] P={len(people)} N={meta['num_frames']} "
              f"cand={candidate_rate:.3f} prep={time.time() - started:.1f}s", flush=True)
        if args.force:
            per_theta, magnitudes = force_arm(
                stem, args.root, body, config, args.device, people, meta,
                tuple(args.force_thetas), drift_from_body=args.drift_from_body)
            for theta, labels in per_theta.items():
                force_counts[theta][stem] = score_labels(stem, labels, meta, args.annotations)
                if args.write_trees is not None and theta == args.force_thetas[0]:
                    write_fake_tree(args.write_trees / "force", stem, labels,
                                    args.root / stem / "human_optim" / "contacts_1.npz")
            force_rank.append(ranking_cells(stem, magnitudes, [p.gates for p in people],
                                            meta, args.annotations))
        torch.cuda.empty_cache()

    report: dict = {"scenes": scenes, "root": str(args.root)}
    print("\n" + "=" * len(_HEAD))
    print(_HEAD)
    pooled_gates = _pool_scene_rows(list(gate_counts.values()))
    # The floor any detector must clear: label every scorable cell in contact. Worth
    # printing beside every arm, because on this label distribution (~46% positive) a very
    # loose rule already scores well and F1 alone would flatter it.
    positives = pooled_gates["tp"] + pooled_gates["fn"]
    total = positives + pooled_gates["fp"] + pooled_gates["tn"]
    rate = positives / total
    print(f"{'ALWAYS-CONTACT floor':38s} {positives:6d} {total - positives:6d} {0:6d} "
          f"{rate:7.3f}{1.0:7.3f}{2 * rate / (1 + rate):7.3f}")
    print(f"{'GATES only':38s} {_fmt(pooled_gates)}")
    report["gates"] = {"pooled": metrics(_pool_scene_rows(list(gate_counts.values()))),
                       "per_scene": {s: metrics(_pool_scene_rows([c]))
                                     for s, c in gate_counts.items()}}
    for theta in args.force_thetas:
        if not force_counts[theta]:
            continue
        rows = list(force_counts[theta].values())
        print(f"{f'FORCE θ={theta} ∪ gates':38s} {_fmt(_pool_scene_rows(rows))}")
        report.setdefault("force", {})[str(theta)] = {
            "pooled": metrics(_pool_scene_rows(rows)),
            "per_scene": {s: metrics(_pool_scene_rows([c]))
                          for s, c in force_counts[theta].items()}}

    if force_rank:
        pooled_rank = {key: np.concatenate([row[key] for row in force_rank])
                       for key in force_rank[0]}
        missed = ~pooled_rank["gated"]
        report.setdefault("force", {})["ranking"] = {
            "all": average_precision(pooled_rank["score"], pooled_rank["truth"]),
            "gate_missed": average_precision(pooled_rank["score"][missed],
                                             pooled_rank["truth"][missed])}
    if args.grid:
        report["grid"] = _run_grid(prepared, args.annotations)
    if args.recipe is not None:
        gap_m, speed_mps = float(args.recipe[0]), float(args.recipe[1])
        power = float(args.recipe[2])
        report["control"] = _run_recipe(prepared, args.annotations, gap_m, speed_mps, power,
                                        args.radius, gate_counts, force_counts,
                                        args.write_trees, args.root)
    _print_ranking(report)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1, default=float))
        print(f"\nwrote {args.json}")
    return 0


def _print_ranking(report: dict) -> None:
    """Threshold-free ranking of both signals, on every cell and on the gate-missed ones."""
    rows = {name: report.get(name, {}).get("ranking")
            for name in ("force", "control")}
    if not any(rows.values()):
        return
    print("\n" + "=" * 78)
    print("SIGNAL QUALITY (threshold-free, coarse-joint cells pooled)")
    print("  ALL    = every scorable cell    MISSED = only cells the geometric gates missed")
    print("=" * 78)
    print(f"{'signal':10s} {'subset':8s} {'cells':>8s} {'pos':>7s} {'AUROC':>7s} {'AP':>7s}")
    show = lambda value: "      -" if value is None else f"{value:7.3f}"  # noqa: E731
    for name, ranking in rows.items():
        if not ranking:
            continue
        for subset, key in (("ALL", "all"), ("MISSED", "gate_missed")):
            stats = ranking[key]
            print(f"{name:10s} {subset:8s} {stats['n']:8d} {stats['pos']:7d} "
                  f"{show(stats['auroc'])}{show(stats['ap'])}")


def _run_grid(prepared: dict, annotations: Path) -> list[dict]:
    """Score every ``(d★, v★, p)`` on the prepared scenes and print the leaderboard."""
    rows = []
    for power in _POWER_GRID:
        for gap_m in _GAP_GRID:
            for speed_mps in _SPEED_GRID:
                counts = {}
                for stem, (people, meta) in prepared.items():
                    labels, _ = control_labels(people, meta, gap_m=gap_m,
                                               speed_mps=speed_mps, power=power)
                    counts[stem] = score_labels(stem, labels, meta, annotations)
                pooled = metrics(_pool_scene_rows(list(counts.values())))
                rows.append({"gap_m": gap_m, "speed_mps": speed_mps, "power": power, **pooled,
                             "per_scene": {s: _pool_scene_rows([c])
                                           for s, c in counts.items()}})
    rows.sort(key=lambda row: (row["f1"] or 0.0), reverse=True)
    print("\n" + "=" * 78)
    print(f"CONTROL GRID — {len(rows)} recipes, pooled over {len(prepared)} scenes")
    print("=" * 78)
    print(f"{'d*(m)':>7s} {'v*(m/s)':>8s} {'p':>5s} {'TP':>6s} {'FP':>6s} {'FN':>6s} "
          f"{'prec':>7s} {'rec':>7s} {'F1':>7s}")
    for row in rows[:20]:
        print(f"{row['gap_m']:7.3f} {row['speed_mps']:8.3f} {row['power']:5.1f} "
              f"{row['tp']:6d} {row['fp']:6d} {row['fn']:6d} "
              f"{row['precision']:7.3f}{row['recall']:7.3f}{row['f1']:7.3f}")
    return rows


def _run_recipe(prepared: dict, annotations: Path, gap_m: float, speed_mps: float,
                power: float, radii: list[float], gate_counts: dict, force_counts: dict,
                write_trees: Path | None, root: Path) -> dict:
    """Evaluate one frozen control recipe at every radius: tables + the ranking readout."""
    out: dict = {"recipe": {"gap_m": gap_m, "speed_mps": speed_mps, "power": power},
                 "radii": {}}
    knees = set(_GROUPS["knees"])
    for radius in radii:
        counts, rank_rows = {}, []
        for stem, (people, meta) in prepared.items():
            labels, scores = control_labels(people, meta, gap_m=gap_m, speed_mps=speed_mps,
                                            power=power, radius=radius)
            counts[stem] = score_labels(stem, labels, meta, annotations)
            rank_rows.append(ranking_cells(stem, scores, [p.gates for p in people],
                                           meta, annotations))
            if write_trees is not None and radius == radii[0]:
                write_fake_tree(write_trees / "control", stem, labels,
                                root / stem / "human_optim" / "contacts_1.npz")
        rows = list(counts.values())
        pooled = metrics(_pool_scene_rows(rows))
        print(f"{f'CONTROL r={radius} ∪ gates':38s} {_fmt(_pool_scene_rows(rows))}")
        out["radii"][str(radius)] = {
            "pooled": pooled,
            "no_knee": metrics(_pool_scene_rows(rows, drop=knees)),
            "no_sport_climb": metrics(_pool_scene_rows(
                [c for s, c in counts.items() if s != "sport_climb"])),
            "per_scene": {s: metrics(_pool_scene_rows([c])) for s, c in counts.items()},
            "per_group": {group: metrics(_pool([c[name] for c in rows for name in names
                                                if name in c]))
                          for group, names in _GROUPS.items()},
        }
        if radius == radii[0]:
            pooled_rank = {key: np.concatenate([row[key] for row in rank_rows])
                           for key in rank_rows[0]}
            missed = ~pooled_rank["gated"]
            out["ranking"] = {
                "all": average_precision(pooled_rank["score"], pooled_rank["truth"]),
                "gate_missed": average_precision(pooled_rank["score"][missed],
                                                 pooled_rank["truth"][missed])}

    print("\n" + "=" * 78)
    print(f"PER-SCENE — control d★={gap_m} v★={speed_mps} p={power} r={radii[0]}")
    print("=" * 78)
    print(f"{'scene':22s} {'gates F1':>9s} {'control F1':>11s} " +
          " ".join(f"{'force θ' + str(t):>12s}" for t in force_counts if force_counts[t]))
    first = out["radii"][str(radii[0])]["per_scene"]
    for stem in prepared:
        gate_f1 = metrics(_pool_scene_rows([gate_counts[stem]]))["f1"]
        show = lambda value: "        -" if value is None else f"{value:9.3f}"  # noqa: E731
        cells = " ".join(
            f"{(metrics(_pool_scene_rows([force_counts[t][stem]]))['f1'] or 0.0):12.3f}"
            for t in force_counts if force_counts[t])
        mark = " *" if stem in _FLOATERS else "  "
        print(f"{stem:20s}{mark} {show(gate_f1)} {first[stem]['f1'] or 0.0:11.3f} {cells}")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
