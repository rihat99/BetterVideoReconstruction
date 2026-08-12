"""Measure the force witness for contact on the 12 contact frames.

The geometric detector labels a contact frame when it is *near* the scene and
*still*. That works when the reconstructed depth is good. When it is not — a body
that floats off the wall, a limb that came out too short — the distance gate never
fires and the contact is simply lost. This diagnostic asks whether **force** sees
those contacts anyway.

For every scene it runs :mod:`tools.human_optim.forces` on the finished
``kinematic_1`` body: RNEA once with nothing touching, then a friction-coned solve
for the world forces at the 12 contact frames that explain the leftover root
wrench. A frame carrying more than a threshold fraction of body weight is called
in contact, and the labels are scored against the manual annotations with the same
fold the judge (``scripts/score_frame_contacts.py``) uses.

Nothing is written into ``out/<scene>/``; results go to ``scratch/force_recovery/``.

Usage::

    uv run --no-sync python scripts/diagnostics/force_recovery.py
    uv run --no-sync python scripts/diagnostics/force_recovery.py --scenes boulder_1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.body import frame_normals, load_body  # noqa: E402
from tools.config import load_toml  # noqa: E402
from tools.geometry.normals import load_scene_cloud  # noqa: E402
from tools.human_optim.camera import world_from_cam  # noqa: E402
from tools.human_optim.contacts import frame_scene_signals, joint_drift, toe_pair_slots  # noqa: E402
from tools.human_optim.forces import (  # noqa: E402
    ForceKnobs, FrameDynamics, contact_axes, force_labels, min_span, solve_frame_forces,
)
from tools.human_optim.qmath import normalize_q  # noqa: E402

sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from score_frame_contacts import (  # noqa: E402
    _GROUPS, _add_counts, _empty_counts, _pool, fold_for_schema, metrics,
)

_CONFIG = _REPO_ROOT / "configs" / "human_optimization.toml"
_OUT_DIR = _REPO_ROOT / "scratch" / "force_recovery"

#: Where the manual annotations live (a one-slot box so --annotations can retarget it).
_ANNOTATIONS = [_REPO_ROOT / "data"]

#: The three scenes the user calls out as floaters — bad depth, so the distance gate
#: never fires. Whether force recovers their contacts is the headline question.
_FLOATERS = ("gtIGgs_y2nY_0024", "vcRBT9Q9jXM_0029", "9RvrpYoWDoo_0001")

_HANDS = ("palm_left", "palm_right", "finger_left", "finger_right")

#: Nearest-neighbour search-box margin for the force path — see :func:`prepare_scene`.
_FORCE_PRUNE_MARGIN_M = 1.0

#: The recipe every ablation varies one knob away from.
_BASE = ForceKnobs()

#: name -> the one knob changed. Solved per scene; the θ sweep is a readout on top.
_VARIANTS: dict[str, ForceKnobs] = {
    "base(axis=frame)": _BASE,
    "axis=scene": replace(_BASE, cone_axis="scene"),
    "axis=scene_signed": replace(_BASE, cone_axis="scene_signed"),
    "no-motion(v=a=0)": replace(_BASE, use_motion=False),
    "effort λτ=0.1": replace(_BASE, lambda_tau=0.1),
    "effort λτ=1": replace(_BASE, lambda_tau=1.0),
    "effort λτ=10": replace(_BASE, lambda_tau=10.0),
    "hands-free": replace(_BASE, free_frames=_HANDS),
    "absolute-gap": replace(_BASE, relative_gap=False),
    "cand≤0.30m": replace(_BASE, candidate_max_dist_m=0.30),
    "smooth=0.1": replace(_BASE, smooth_weight=0.1),
    "torque-L=0.2m": replace(_BASE, torque_length_m=0.2),
    # Second round: the two knobs that each helped on their own, together.
    "λτ=1+L=0.2m": replace(_BASE, lambda_tau=1.0, torque_length_m=0.2),
    "λτ=1+signed": replace(_BASE, lambda_tau=1.0, cone_axis="scene_signed"),
    "λτ=1+L=0.2m+signed": replace(_BASE, lambda_tau=1.0, torque_length_m=0.2,
                                  cone_axis="scene_signed"),
    "torque-L=0.5m": replace(_BASE, torque_length_m=0.5),
    # The Phase-A recommendation: static gravity, frame axes, relative gap, no distance
    # cap, effort rows on, root torque weighted up.
    # THE PRE-REGISTERED RECIPE (frozen before any held-out scoring; mirrors
    # scripts/stages/estimate_contacts.py::_FORCE_INTERNALS).
    "recommended": replace(_BASE, torque_length_m=0.2),
}

#: Force thresholds, as a fraction of body weight.
_THETAS = (0.01, 0.02, 0.03, 0.05, 0.10)

#: Physics-free label generators, scored exactly like the force labels. They ask how much
#: of the lift is force and how much is the geometry the gap price already knows:
#: ``loose-gate`` is the old completion-era operating point (absolute distance + drift),
#: ``rel-gap`` mimics the relative gap price with no dynamics anywhere.
_CONTROLS: dict[str, tuple[str, float, float]] = {
    f"{kind} d<{dist:.2f} s<{drift:.2f}": (kind, dist, drift)
    for kind, dists in (("loose-gate", (0.15, 0.30)), ("rel-gap", (0.05, 0.10)))
    for dist in dists for drift in (0.08, 0.12)
}


def _all_scenes(root: Path | None = None) -> list[str]:
    """Every scene with both a manual annotation and a finished contacts_1 readout."""
    root = root or _REPO_ROOT / "out"
    return sorted(p.stem for p in _ANNOTATIONS[0].glob("*.npz")
                  if (root / p.stem / "human_optim" / "contacts_1.npz").exists())


class ScenePerson:
    """One person's prepared inputs — everything the solve needs, computed once."""

    def __init__(self, dynamics: FrameDynamics, valid: np.ndarray, drift: torch.Tensor,
                 dist_m: torch.Tensor, sdf_m: torch.Tensor, normal_dot: torch.Tensor,
                 axes: dict[str, tuple[torch.Tensor, torch.Tensor]], baseline: np.ndarray,
                 object_id: int) -> None:
        self.dynamics = dynamics
        self.valid = valid
        self.drift = drift
        self.dist_m = dist_m
        self.sdf_m = sdf_m
        self.normal_dot = normal_dot
        self.axes = axes
        self.baseline = baseline
        self.object_id = object_id


def prepare_scene(stem: str, body, config: dict, device: str, *,
                  root: Path | None = None, drift_from_body: bool = False
                  ) -> tuple[list[ScenePerson], dict]:
    """Load one ``out/<stem>/`` tree and precompute the pose-dependent inputs."""
    pred = (root or _REPO_ROOT / "out") / stem
    kin = dict(np.load(pred / "human_optim" / "kinematic_1.npz", allow_pickle=True))
    contacts = dict(np.load(pred / "human_optim" / "contacts_1.npz", allow_pickle=True))
    transform = dict(np.load(pred / "geometry" / "transform.npz", allow_pickle=True))
    scene_npz = dict(np.load(pred / "geometry" / "scene.npz", allow_pickle=True))

    gravity_path = pred / "geocalib" / "gravity.npz"
    if gravity_path.exists():
        gravity_npz = np.load(gravity_path)
        gravity = np.asarray(gravity_npz["gravity_world"], np.float32)
        reliable = bool(gravity_npz["reliable"]) if "reliable" in gravity_npz.files else False
        gravity_source = f"geocalib(reliable={reliable})"
    else:
        gravity = np.array([0.0, 1.0, 0.0], np.float32)
        gravity_source = "fallback_down"

    extrinsics = torch.as_tensor(transform["extrinsics"], dtype=torch.float32, device=device)
    scene_pts, scene_normals, scene_conf = load_scene_cloud(
        scene_npz, world_from_cam(extrinsics)[:, :3, 3],
        conf_drop_q=float(config["contacts"]["conf_drop_q"]),
        k=int(config["contacts"]["normal_k"]), device=device)

    fps = float(transform.get("fps", kin.get("fps", 30.0)))
    window_s = float(config["contact_gate"]["window_s"])
    valid_mask = np.asarray(kin["valid_mask"], bool)
    frame_contact = np.asarray(contacts["frame_contact"], bool)
    contact_ids = [int(o) for o in contacts["object_ids"]]

    people: list[ScenePerson] = []
    for person, object_id in enumerate([int(o) for o in kin["object_ids"]]):
        betas = torch.as_tensor(kin["betas"][person], dtype=torch.float32, device=device)
        q = torch.as_tensor(kin["q"][person], dtype=torch.float32, device=device)
        dynamics = FrameDynamics(body, betas, q, gravity_world=gravity, fps=fps, device=device)
        valid = valid_mask[person]

        # The detector's 10 cm prune margin is right for "is it touching?"; for the force
        # solve a step whose search box catches nothing must mean the body really is far
        # from all reconstructed geometry, not that the box was small — so the box is
        # opened to a metre and anything still empty is honestly "no scene evidence".
        _, dist_m, _, _ = frame_scene_signals(
            dynamics.points, dynamics.normals, scene_pts, scene_normals, scene_conf,
            prune_margin_m=_FORCE_PRUNE_MARGIN_M)

        # Stillness is the PERSISTED pre-enforcement drift: measured before any contact
        # term acted on this body. Re-measuring on the enforced body would be evidence the
        # optimizer manufactured, so a missing measurement is a hard error, never a silent
        # substitution. (The pure-visual control passes a body where nothing was ever
        # enforced, so its freshly measured drift IS pre-enforcement — see --drift-from-body.)
        if "drift_pre" in kin:
            drift = torch.as_tensor(np.asarray(kin["drift_pre"])[person],
                                    dtype=torch.float32, device=device)
            if not bool(torch.isfinite(drift).all()):
                raise SystemExit(f"{stem}: drift_pre has non-finite entries")
        elif drift_from_body:
            drift = joint_drift(dynamics.points, fps=fps, window_s=window_s,
                                valid_mask=torch.as_tensor(valid, device=device))
        else:
            raise SystemExit(f"{stem}: kinematic_1.npz has no drift_pre — refusing to "
                             "re-measure stillness on a body that may have been enforced")

        axes = {mode: contact_axes(
            dynamics.points, dynamics.normals, scene_pts, scene_normals, mode=mode,
            k=_BASE.normal_k, max_dist_m=_BASE.axis_max_dist_m)
            for mode in ("frame", "scene", "scene_signed")}

        row = contact_ids.index(object_id) if object_id in contact_ids else person
        people.append(ScenePerson(
            dynamics, valid, drift, dist_m,
            torch.as_tensor(np.asarray(contacts["frame_sdf_cm"])[row] / 100.0, device=device),
            torch.as_tensor(np.asarray(contacts["frame_normal_dot"])[row], device=device),
            axes, frame_contact[row], object_id))

    meta = dict(fps=fps, window_s=window_s, gravity_source=gravity_source,
                frame_names=[str(n) for n in contacts["contact_frame_names"]],
                object_ids=[p.object_id for p in people],
                valid_mask=valid_mask, num_frames=int(kin["num_frames"]))
    return people, meta


def score_labels(stem: str, labels: list[np.ndarray], meta: dict) -> dict[str, dict[str, int]]:
    """Score ``(T, 12)`` labels per person against ``data/<stem>.npz`` — the judge's fold."""
    ann = np.load(_ANNOTATIONS[0] / f"{stem}.npz", allow_pickle=True)
    ann_names = [str(n) for n in ann["joint_names"]]
    ann_contacts = np.transpose(ann["contacts"].astype(np.int8), (0, 2, 1))   # (P, N, J)
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


def _combinations(labels: np.ndarray, baseline: np.ndarray, dist_m: np.ndarray,
                  drift_m: np.ndarray, veto_m: float, veto_drift_m: float,
                  knee_columns: list[int]) -> dict[str, np.ndarray]:
    """The decision rules compared at every operating point.

    Every union is taken on the 12 contact frames, before any fold onto the annotated
    coarse joints. ``gates`` is the shipped geometric readout (``contacts_1``). The
    **veto** is the two gate signals used only to *reject*: a frame that force says is
    loaded but that is far from any surface or visibly moving is thrown out. The
    ``−knees`` rule drops force labels on the two knee frames only at readout — the knees
    stay candidates in the solve, so removing them cannot push their share of the
    residual onto another limb.
    """
    vetoed = labels & (dist_m < veto_m) & (drift_m < veto_drift_m)
    no_knee = labels.copy()
    no_knee[..., knee_columns] = False
    return {
        "force": labels,
        "force|gates": labels | baseline,
        "force&veto": vetoed,
        "force&veto|gates": vetoed | baseline,
        "force−knees|gates": no_knee | baseline,
    }


def control_labels(kind: str, dist_m: np.ndarray, drift_m: np.ndarray, valid: np.ndarray,
                   *, dist_thresh: float, drift_thresh: float, min_frames: int
                   ) -> np.ndarray:
    """Physics-free labels ``(T, 12)`` from geometry alone — the attribution control.

    ``"loose-gate"`` thresholds the raw distance to the scene; ``"rel-gap"`` thresholds
    distance minus the step's nearest contact frame, the same relative trick the force
    solve's gap price uses. Both then apply the drift gate and the same minimum span as
    the force readout. No toe tiebreak is applied (there is no force to break ties with),
    which makes the control slightly *more* generous than the force labels.
    """
    gap = np.nan_to_num(dist_m, nan=10.0, posinf=10.0)
    gap = np.minimum(gap, 10.0)
    if kind == "rel-gap":
        gap = gap - gap.min(axis=1, keepdims=True)
    fires = (gap < dist_thresh) & (drift_m < drift_thresh) & np.asarray(valid, bool)[:, None]
    return min_span(fires, min_frames)


def write_fake_tree(root: Path, tag: str, stem: str, labels: list[np.ndarray],
                    source: Path) -> Path:
    """Write real ``contacts_1.npz``-shaped labels so the actual judge can score them.

    The scene's own readout is copied verbatim and only ``frame_contact`` is replaced, so
    every field the judge reads (``valid_mask``, ``object_ids``, ``contact_frame_names``)
    is byte-identical to production.
    """
    destination = root / tag / stem / "human_optim"
    destination.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(source / "human_optim" / "contacts_1.npz", allow_pickle=True))
    data["frame_contact"] = np.stack(labels)
    np.savez(destination / "contacts_1.npz", **data)
    return destination / "contacts_1.npz"


def joint_candidates(stem: str, meta: dict, per_person: list[dict[str, np.ndarray]]
                     ) -> dict[str, np.ndarray]:
    """Flatten every scorable ``(person, step, coarse joint)`` cell into ranking arrays.

    A threshold hides how good a signal is. This asks the threshold-free question:
    ordered by each signal, how well do the truly-touching cells come first? Every
    signal is folded onto the annotated coarse joint the same way the judge folds the
    labels — the best of the joint's contact frames.
    """
    ann = np.load(_ANNOTATIONS[0] / f"{stem}.npz", allow_pickle=True)
    ann_names = [str(n) for n in ann["joint_names"]]
    ann_contacts = np.transpose(ann["contacts"].astype(np.int8), (0, 2, 1))
    ann_ids = [int(o) for o in ann["object_ids"]]
    ignored = ann["ignored"].astype(bool)
    fold, _ = fold_for_schema(ann_names)
    ann_column = {name: i for i, name in enumerate(ann_names)}
    frame_index = {name: i for i, name in enumerate(meta["frame_names"])}

    out: dict[str, list[np.ndarray]] = {k: [] for k in
                                        ("force", "near", "relnear", "still", "truth", "baseline")}
    for person, object_id in enumerate(meta["object_ids"]):
        if object_id not in ann_ids or bool(ignored[ann_ids.index(object_id)]):
            continue
        row = per_person[person]
        # The relative gap is exactly what the solve's price uses, with no dynamics — the
        # threshold-free control for "is the lift physics, or the geometry already in the price?"
        distance = np.minimum(np.nan_to_num(row["dist_m"], nan=10.0, posinf=10.0), 10.0)
        relative = distance - distance.min(axis=1, keepdims=True)
        for name, frames in fold.items():
            columns = [frame_index[f] for f in frames]
            truth = ann_contacts[ann_ids.index(object_id)][:, ann_column[name]]
            keep = (truth >= 0) & meta["valid_mask"][person]
            out["force"].append(row["magnitude"][:, columns].max(axis=1)[keep]
                                / row["body_weight_n"])
            out["near"].append(-distance[:, columns].min(axis=1)[keep])
            out["relnear"].append(-relative[:, columns].min(axis=1)[keep])
            out["still"].append(-np.nan_to_num(row["drift_m"][:, columns], nan=10.0
                                               ).min(axis=1)[keep])
            out["truth"].append((truth[keep] == 1))
            out["baseline"].append(row["baseline"][:, columns].any(axis=1)[keep])
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}


def _ranking(scores: np.ndarray, truth: np.ndarray) -> dict[str, float | int | None]:
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
    return {"n": int(truth.size), "pos": positives,
            "auroc": float(auroc), "ap": float(precision[truth[order]].sum() / positives)}


def _fmt(counts: dict[str, int]) -> str:
    m = metrics(counts)
    show = lambda v: "    -  " if v is None else f"{v:7.3f}"  # noqa: E731
    return (f"{m['tp']:6d} {m['fp']:6d} {m['fn']:6d} "
            f"{show(m['precision'])}{show(m['recall'])}{show(m['f1'])}")


_HEAD = f"{'variant':34s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'prec':>7s} {'rec':>7s} {'F1':>7s}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes", nargs="*", default=None, help="scene stems (default: all)")
    parser.add_argument("--variants", nargs="*", default=None, help="variant names (default: all)")
    parser.add_argument("--veto-m", type=float, default=0.30, help="gate-veto distance (m)")
    parser.add_argument("--veto-drift-m", type=float, default=0.15, help="gate-veto drift (m)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--annotations", type=Path, default=None,
                        help="annotation root (default data/) — for held-out farms")
    parser.add_argument("--root", type=Path, default=None,
                        help="scene-tree root (default out/) — for control farms")
    parser.add_argument("--drift-from-body", action="store_true",
                        help="allow measuring stillness on the supplied body; ONLY valid "
                             "when no contact term ever acted on it (pure-visual control)")
    parser.add_argument("--write-trees", type=Path, default=None,
                        help="write contacts_1.npz-shaped labels here so the real judge "
                             "(scripts/score_frame_contacts.py) can score them")
    parser.add_argument("--controls", action="store_true",
                        help="also score the physics-free geometry-only label generators")
    args = parser.parse_args()

    if args.annotations is not None:
        _ANNOTATIONS[0] = args.annotations
    scenes = args.scenes or _all_scenes(args.root)
    variants = {k: v for k, v in _VARIANTS.items()
                if args.variants is None or k in args.variants}
    config = load_toml(_CONFIG)
    body = load_body(device=args.device)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # counts[variant][theta][rule][joint] and the per-scene breakdown
    pooled: dict = {}
    per_scene: dict = {}
    residuals: dict = {}
    toe_story: dict = {}
    ranking: dict = {}
    controls: dict = {}

    for stem in scenes:
        started = time.time()
        people, meta = prepare_scene(stem, body, config, args.device,
                                     root=args.root, drift_from_body=args.drift_from_body)
        print(f"[{stem}] P={len(people)} N={meta['num_frames']} fps={meta['fps']:.1f} "
              f"gravity={meta['gravity_source']} prep={time.time() - started:.1f}s", flush=True)

        baseline_labels = [p.baseline for p in people]
        baseline_counts = score_labels(stem, baseline_labels, meta)
        per_scene.setdefault(stem, {})["baseline"] = baseline_counts
        pooled.setdefault("baseline", {}).setdefault("-", {}).setdefault("gates", []).append(
            baseline_counts)

        knee_columns = [i for i, n in enumerate(meta["frame_names"]) if n.startswith("knee")]
        if args.controls:
            for control, (kind, dist_thresh, drift_thresh) in _CONTROLS.items():
                labels = [control_labels(
                    kind, person.dist_m.cpu().numpy(), person.drift.cpu().numpy(), person.valid,
                    dist_thresh=dist_thresh, drift_thresh=drift_thresh,
                    min_frames=int(config["contacts"]["min_frames"])) for person in people]
                union = [a | b for a, b in zip(labels, baseline_labels)]
                counts = score_labels(stem, union, meta)
                controls.setdefault(control, []).append(counts)
                per_scene[stem][f"CONTROL {control}|gates"] = counts

        for name, knobs in variants.items():
            solve_started = time.time()
            solutions = []
            for person in people:
                axes, axes_valid = person.axes[knobs.cone_axis]
                solutions.append(solve_frame_forces(
                    person.dynamics, valid=person.valid, axes=axes, axes_valid=axes_valid,
                    drift_m=person.drift, drift_window_s=meta["window_s"],
                    dist_m=person.dist_m, knobs=knobs))
            elapsed = time.time() - solve_started

            before = np.concatenate([s.residual_force_before_n[p.valid]
                                     for s, p in zip(solutions, people)])
            after = np.concatenate([s.residual_force_after_n[p.valid]
                                    for s, p in zip(solutions, people)])
            torque_before = np.concatenate([s.residual_torque_before_nm[p.valid]
                                            for s, p in zip(solutions, people)])
            torque_after = np.concatenate([s.residual_torque_after_nm[p.valid]
                                           for s, p in zip(solutions, people)])
            residuals.setdefault(name, {})[stem] = dict(
                force_before=float(before.mean()), force_after=float(after.mean()),
                torque_before=float(torque_before.mean()), torque_after=float(torque_after.mean()),
                body_weight_n=float(solutions[0].body_weight_n), seconds=elapsed,
                kkt=float(max(s.kkt_violation for s in solutions)),
                candidate_rate=float(np.mean([s.candidate[p.valid].mean()
                                              for s, p in zip(solutions, people)])))

            ranking.setdefault(name, []).append(joint_candidates(stem, meta, [
                dict(magnitude=solution.magnitude, dist_m=person.dist_m.cpu().numpy(),
                     drift_m=person.drift.cpu().numpy(), baseline=person.baseline,
                     body_weight_n=solution.body_weight_n)
                for solution, person in zip(solutions, people)]))

            for theta in _THETAS:
                labels_per_person, splits = [], []
                for solution, person in zip(solutions, people):
                    labels, split = force_labels(
                        solution.magnitude, frame_names=meta["frame_names"], valid=person.valid,
                        threshold_n=theta * solution.body_weight_n,
                        min_frames=int(config["contacts"]["min_frames"]))
                    labels_per_person.append(labels)
                    splits.append(split)
                for rule, rule_labels in _combinations(
                        np.stack(labels_per_person), np.stack(baseline_labels),
                        np.stack([p.dist_m.cpu().numpy() for p in people]),
                        np.stack([p.drift.cpu().numpy() for p in people]),
                        args.veto_m, args.veto_drift_m, knee_columns).items():
                    counts = score_labels(stem, list(rule_labels), meta)
                    pooled.setdefault(name, {}).setdefault(theta, {}).setdefault(
                        rule, []).append(counts)
                    per_scene[stem][f"{name}|θ{theta}|{rule}"] = counts
                    if args.write_trees is not None and rule == "force|gates":
                        write_fake_tree(args.write_trees,
                                        f"{_slug(name)}_t{round(theta * 1000):03d}",
                                        stem, list(rule_labels),
                                        (args.root or _REPO_ROOT / 'out') / stem)

                if theta == 0.05:
                    toe_story.setdefault(name, {})[stem] = _toe_report(
                        solutions, people, meta, theta)
            print(f"    {name:24s} solve={elapsed:5.1f}s  |F| {before.mean():6.1f} → "
                  f"{after.mean():6.1f} N   |τ| {torque_before.mean():5.1f} → "
                  f"{torque_after.mean():5.1f} N·m  kkt={max(s.kkt_violation for s in solutions):.2e}",
                  flush=True)
        del people
        torch.cuda.empty_cache()

    _print_tables(pooled, per_scene, residuals, toe_story, scenes)
    if "recommended" in pooled:
        _print_recommended(per_scene, scenes)
    if controls:
        _print_controls(controls, per_scene)
    rank_report = _print_ranking(ranking)
    if args.write_trees is not None:
        print(f"\nfake trees under {args.write_trees} — score with:\n"
              f"  uv run --no-sync python scripts/score_frame_contacts.py "
              f"--predictions {args.write_trees}/<tag>/")
    payload = dict(
        pooled={v: {str(t): {r: metrics(_pool([c for j in rows for c in j.values()]))
                             for r, rows in rules.items()}
                    for t, rules in thetas.items()} for v, thetas in pooled.items()},
        per_scene={s: {k: metrics(c) for k, c in _flatten(rows).items()}
                   for s, rows in per_scene.items()},
        residuals=residuals, toe_story=toe_story, ranking=rank_report)
    out_path = _OUT_DIR / "force_recovery.json"
    out_path.write_text(json.dumps(payload, indent=1, default=float))
    print(f"\nwrote {out_path}")
    return 0


def _print_recommended(per_scene: dict, scenes: list[str], theta: float = 0.05) -> None:
    """The one table that goes to the user: every scene, baseline vs the recommendation."""
    print("\n" + "=" * 78)
    print(f"RECOMMENDED RECIPE, θ={theta}, rule = force ∪ gates — all scenes")
    print("=" * 78)
    print(f"{'scene':22s} {'base F1':>8s} {'new F1':>8s} {'Δ':>7s} {'new P':>7s} {'new R':>7s} "
          f"{'force-only P':>13s}")
    for stem in scenes:
        rows = _flatten(per_scene[stem])
        base = metrics(rows["baseline"])
        union = metrics(rows[f"recommended|θ{theta}|force|gates"])
        only = metrics(rows[f"recommended|θ{theta}|force"])
        show = lambda v: "      -" if v is None else f"{v:7.3f}"  # noqa: E731
        delta = None if base["f1"] is None or union["f1"] is None else union["f1"] - base["f1"]
        mark = " *" if stem in _FLOATERS else "  "
        print(f"{stem:20s}{mark} {show(base['f1'])[1:]} {show(union['f1'])[1:]} {show(delta)} "
              f"{show(union['precision'])}{show(union['recall'])}  {show(only['precision'])}")


def _slug(name: str) -> str:
    """Variant name → a filesystem-safe tag."""
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def _print_controls(controls: dict, per_scene: dict) -> None:
    """The physics-free attribution grid: pooled, then the three floaters."""
    print("\n" + "=" * 78)
    print("ATTRIBUTION CONTROLS — geometry-only labels ∪ gates (no dynamics at all)")
    print("=" * 78)
    print(f"{'control':34s} {_HEAD[35:]}")
    for control, rows in controls.items():
        print(f"{control:34s} {_fmt(_pool([c for j in rows for c in j.values()]))}")
    for stem in _FLOATERS:
        if stem not in per_scene:
            continue
        rows = _flatten(per_scene[stem])
        print(f"\n  {stem}")
        for key in ("baseline", *(k for k in rows if k.startswith("CONTROL"))):
            if key in rows:
                print(f"  {key:44s} {_fmt(rows[key])}")


def _flatten(rows: dict) -> dict[str, dict[str, int]]:
    """Per-scene rows are ``{key: joint_counts}``; pool each row's joints for the JSON."""
    return {key: _pool(list(joints.values())) for key, joints in rows.items()}


def _toe_report(solutions, people, meta, theta: float) -> dict:
    """How the solve splits each foot's two toe frames, and whether normal_dot agrees."""
    names = meta["frame_names"]
    report = dict(one=0, both=0, neither=0, agree=0, disagree=0,
                  agree_penetrating=0, agree_outside=0,
                  disagree_penetrating=0, disagree_outside=0)
    for solution, person in zip(solutions, people):
        magnitude = solution.magnitude
        threshold = theta * solution.body_weight_n
        normal_dot = person.normal_dot.cpu().numpy()
        sdf = person.sdf_m.cpu().numpy()
        for under, above in toe_pair_slots(names):
            fires_under = (magnitude[:, under] > threshold) & person.valid
            fires_above = (magnitude[:, above] > threshold) & person.valid
            report["both"] += int((fires_under & fires_above).sum())
            report["one"] += int((fires_under ^ fires_above).sum())
            report["neither"] += int((~fires_under & ~fires_above & person.valid).sum())
            # Where exactly one toe carries force, does the normal tiebreak pick the same one?
            single = fires_under ^ fires_above
            force_picks_under = fires_under & single
            normal_picks_under = normal_dot[:, under] >= normal_dot[:, above]
            agree = single & (force_picks_under == normal_picks_under)
            report["agree"] += int(agree.sum())
            disagree = single & ~agree
            report["disagree"] += int(disagree.sum())
            winner_sdf = np.where(force_picks_under, sdf[:, under], sdf[:, above])
            for tag, mask in (("agree", agree), ("disagree", disagree)):
                report[f"{tag}_penetrating"] += int((mask & (winner_sdf < 0)).sum())
                report[f"{tag}_outside"] += int((mask & (winner_sdf >= 0)).sum())
    return report


def _print_tables(pooled: dict, per_scene: dict, residuals: dict, toe_story: dict,
                  scenes: list[str]) -> None:
    baseline_rows = pooled["baseline"]["-"]["gates"]
    baseline_pooled = _pool([counts for joints in baseline_rows for counts in joints.values()])

    print("\n" + "=" * 78)
    print("RESIDUAL: mean root |F| (N) before → after the solve, per scene")
    print("=" * 78)
    header = f"{'scene':22s} {'BW(N)':>7s} " + " ".join(f"{n[:14]:>15s}" for n in residuals)
    print(header)
    for stem in scenes:
        weight = next((r[stem]["body_weight_n"] for r in residuals.values() if stem in r), np.nan)
        cells = " ".join(
            f"{residuals[n][stem]['force_before']:6.0f}→{residuals[n][stem]['force_after']:<8.0f}"
            if stem in residuals[n] else " " * 15 for n in residuals)
        print(f"{stem:22s} {weight:7.0f} {cells}")

    print("\n" + "=" * 78)
    print("POOLED SCORES (all scenes, the judge's frame→joint fold)")
    print("=" * 78)
    knees = set(_GROUPS["knees"])
    no_knee = lambda rows: _pool([c for j in rows for n, c in j.items() if n not in knees])  # noqa: E731

    def without_rope(key: str) -> dict[str, int]:
        """Pooled counts with sport_climb dropped — its climber hangs on an unmodelled rope."""
        return _pool([c for stem, rows in per_scene.items() if stem != "sport_climb"
                      for c in rows[key].values() if key in rows])

    print(_HEAD + "  | no-knee P/R/F1        | no-sport_climb P/R/F1")
    print(f"{'BASELINE contacts_1':34s} {_fmt(baseline_pooled)}  | "
          f"{_fmt(no_knee(baseline_rows))[21:]} | {_fmt(without_rope('baseline'))[21:]}")
    for variant, thetas in pooled.items():
        if variant == "baseline":
            continue
        for theta, rules in thetas.items():
            for rule, rows in rules.items():
                counts = _pool([row for scene_rows in rows for row in scene_rows.values()])
                print(f"{variant + f' θ{theta} ' + rule:34s} {_fmt(counts)}  | "
                      f"{_fmt(no_knee(rows))[21:]} | "
                      f"{_fmt(without_rope(f'{variant}|θ{theta}|{rule}'))[21:]}")
        print("-" * len(_HEAD))

    print("\n" + "=" * 78)
    print("PER-GROUP at θ=0.05 (hands / feet+ankles / knees)")
    print("=" * 78)
    print(f"{'variant / rule':34s} " + " ".join(f"{g:>24s}" for g in _GROUPS))

    def group_row(label: str, rows: list) -> None:
        cells = []
        for members in _GROUPS.values():
            counts = _pool([row[j] for row in rows for j in members if j in row])
            m = metrics(counts)
            cells.append("           -            " if m["f1"] is None else
                         f"P{m['precision'] or 0:.3f} R{m['recall'] or 0:.3f} F{m['f1']:.3f}")
        print(f"{label:34s} " + " ".join(f"{c:>24s}" for c in cells))

    group_row("BASELINE contacts_1", baseline_rows)
    for variant, thetas in pooled.items():
        if variant == "baseline":
            continue
        for rule in ("force", "force|gates"):
            group_row(f"{variant} {rule}", thetas.get(0.05, {}).get(rule, []))

    print("\n" + "=" * 78)
    print("PER-SCENE F1 at θ=0.05, rule=force|gates (baseline in the first column)")
    print("=" * 78)
    variant_names = [v for v in pooled if v != "baseline"]
    print(f"{'scene':22s} {'base':>6s} " + " ".join(f"{v[:12]:>12s}" for v in variant_names))
    for stem in scenes:
        rows = _flatten(per_scene[stem])
        cells = []
        for variant in variant_names:
            key = f"{variant}|θ0.05|force|gates"
            f1 = metrics(rows[key])["f1"] if key in rows else None
            cells.append("     -" if f1 is None else f"{f1:12.3f}")
        base_f1 = metrics(rows["baseline"])["f1"]
        mark = " *" if stem in _FLOATERS else "  "
        print(f"{stem:20s}{mark} {base_f1 if base_f1 is None else f'{base_f1:6.3f}'} "
              + " ".join(cells))

    print("\n" + "=" * 78)
    print(f"FLOATERS {_FLOATERS} — every rule at θ=0.05")
    print("=" * 78)
    for stem in _FLOATERS:
        if stem not in per_scene:
            continue
        rows = _flatten(per_scene[stem])
        print(f"\n{stem}")
        print(f"  {'rule':44s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'prec':>7s} {'rec':>7s} {'F1':>7s}")
        for key in ("baseline", *(k for k in rows if "θ0.05" in k)):
            if key in rows:
                print(f"  {key:44s} {_fmt(rows[key])}")

    print("\n" + "=" * 78)
    print("TOE PAIR (θ=0.05): steps where the solve loads one toe / both / neither,")
    print("and whether the force winner matches today's normal_dot tiebreak")
    print("=" * 78)
    print(f"{'variant':24s} {'one':>7s} {'both':>7s} {'agree':>7s} {'disagr':>7s} "
          f"{'agr|pen':>8s} {'agr|out':>8s} {'dis|pen':>8s} {'dis|out':>8s} {'dis%|pen':>9s} "
          f"{'dis%|out':>9s}")
    for variant, per in toe_story.items():
        total = {k: sum(row[k] for row in per.values()) for k in next(iter(per.values()))}
        rate = lambda d, a: f"{d / (d + a):9.3f}" if d + a else "        -"  # noqa: E731
        print(f"{variant:24s} {total['one']:7d} {total['both']:7d} {total['agree']:7d} "
              f"{total['disagree']:7d} {total['agree_penetrating']:8d} {total['agree_outside']:8d} "
              f"{total['disagree_penetrating']:8d} {total['disagree_outside']:8d} "
              f"{rate(total['disagree_penetrating'], total['agree_penetrating'])} "
              f"{rate(total['disagree_outside'], total['agree_outside'])}")


def _print_ranking(ranking: dict) -> dict:
    """Threshold-free signal quality: how well each signal ranks true contacts first."""
    print("\n" + "=" * 78)
    print("SIGNAL QUALITY (threshold-free, coarse-joint cells pooled over all scenes)")
    print("  ALL      = every scorable cell")
    print("  MISSED   = only the cells the geometric detector labelled 'no contact'")
    print("=" * 78)
    print(f"{'variant':22s} {'signal':8s} {'subset':8s} {'cells':>8s} {'pos':>7s} "
          f"{'AUROC':>7s} {'AP':>7s}")
    report: dict = {}
    show = lambda v: "      -" if v is None else f"{v:7.3f}"  # noqa: E731
    for variant, rows in ranking.items():
        pooled = {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}
        missed = ~pooled["baseline"]
        signals = (["force", "near", "relnear", "still"] if variant == next(iter(ranking))
                   else ["force"])
        for signal in signals:
            for subset, mask in (("ALL", np.ones_like(missed)), ("MISSED", missed)):
                stats = _ranking(pooled[signal][mask], pooled["truth"][mask])
                report.setdefault(variant, {})[f"{signal}|{subset}"] = stats
                print(f"{variant:22s} {signal:8s} {subset:8s} {stats['n']:8d} "
                      f"{stats['pos']:7d} {show(stats['auroc'])} {show(stats['ap'])}")
    return report


if __name__ == "__main__":
    raise SystemExit(main())
