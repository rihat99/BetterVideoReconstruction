"""Label BEDLAM2 ground truth: person masks, boxes, per-frame validity, ground contact.

``datasets/bedlam2/import_bedlam2.py`` composes BEDLAM2's pose ground truth into
``features/gt/``. This script measures three things that ground truth implies but
does not state, writing them next to it in the same directory:

``gt/masks/NNNNNN.png``
    One 8-bit instance map per frame — ``0`` background, ``p + 1`` for person ``p``
    — from a z-buffered rasterisation of the posed SMPL-X meshes. One file per
    frame however many people are in it, and occlusion between them is exact.

``gt/bboxes.npz``
    Every person's box and per-frame validity for the whole sequence:
    ``bbox_full`` (the projected surface's box, *unclipped*, so how far the person
    reaches past the edge stays readable), ``bbox_visible``, ``mask_area``,
    ``surface_area`` (before anyone occludes them), ``out_of_frame_ratio``,
    ``valid (P, N)``, ``person_valid (P,)`` and ``valid_frac (P,)``.

``gt/contacts.npz``
    Which of the 35 body contact frames are on the floor, per person per frame, in
    the label schema the viewer already draws, plus the three measurements behind every
    decision (``frame_height_m``, ``frame_speed_mps``, ``frame_tilt_deg``) so a label can
    be audited without posing anything again. See :mod:`datasets.bedlam2.contacts` for why
    height, stillness and facing are the three measurements, and for the cases where
    they are not enough.

``gt/forces.npz``
    The force every labelled contact frame carries, solved at the frozen ground-truth
    pose by the ``kindyn_1`` force objective (:mod:`datasets.bedlam2.forces`): world
    newtons per contact frame, their sums per joint, the joint torques and the root
    wrench that stays unexplained. The same fields as ``kindyn_1.npz``, so the viewer
    draws them the same way. Needs ``contacts.npz``.

``--step contacts`` prints two numbers per run. The first is the support check: of the
frames where the *mesh* comes within a centimetre of the floor, how many carry no label
at all — it should be a fraction of a percent. The second is **plane suspect**: how
many frames have the whole body too high to label and too low to be airborne. That one
exists because the support check shares the labels' floor plane, so it cannot see a
person standing on something the plane does not know about; a scene with a large
suspect share has a floor that is wrong, not a detector that is.

A person-frame is **valid** when at most a third of ``bbox_full`` lies outside the
image, at least one pixel of the person survives the other bodies, and no vertex is
closer to the lens than the near plane. A person is valid when any of their frames
is; a scene is valid when any of its people is. That last rule is deliberately literal
and very permissive — one good frame in five hundred keeps a person — so
``valid_frac`` is stored beside ``person_valid`` for anyone who wants a stricter cut.

``--step select`` acts on that: it reports the scenes with no valid person and, with
``--apply``, clears their ``human_selected`` flag in ``scenes/scenes.db`` and writes
a rollback JSON beside it. Nothing is ever deleted, and invalid *people* are marked
in ``bboxes.npz`` (``person_valid``) rather than cut out of ``smplx.npz`` — the pose
ground truth stays exactly as imported.

Usage::

    # the 200 scenes `view_3d.py --bedlam` browses
    uv run python datasets/bedlam2/label_bedlam2.py --step masks --sample 200
    uv run python datasets/bedlam2/label_bedlam2.py --step contacts --sample 200
    uv run python datasets/bedlam2/label_bedlam2.py --step forces --sample 200 --workers 32
    uv run python datasets/bedlam2/label_bedlam2.py --step select --sample 200

    # all 18,076
    uv run python datasets/bedlam2/label_bedlam2.py --step all --workers 8
    uv run python datasets/bedlam2/label_bedlam2.py --step select --apply

Contact knobs live in ``datasets/bedlam2/configs/bedlam.toml`` under ``[bedlam]`` and the force
solve's under ``[bedlam_forces]``; this CLI takes only targets.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bvr.paths import REPO_ROOT

from datasets.climbingvideos import scene_db
from datasets.bedlam2.paths import BEDLAM2_OUR
from datasets.bedlam2.contacts import (
    ENTER_M,
    EXIT_M,
    MAX_TILT_DEG,
    MIN_SPAN_S,
    STILL_MPS,
    SUSPECT_M,
    TOUCH_M,
    WINDOW_S,
    save_contacts,
    scene_contacts,
)
from datasets.bedlam2.forces import save_forces, scene_forces
from datasets.bedlam2.masks import save_boxes, scene_masks
from bvr.config import load_config, section
from bvr.config import load_human_config

_STATE: dict = {}


def _init_worker(data_root: str, device: str, knobs: dict, workers: int) -> None:
    """One body per worker process (and one CUDA context, or a fair share of the cores)."""
    import os

    import torch

    from bvr.body import load_body

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu":   # a fair share of the cores, capped: 192 threads a worker only thrash
        torch.set_num_threads(min(16, max(1, (os.cpu_count() or 1) // workers)))
    _STATE.update(paths=scene_db.ScenePaths(Path(data_root)), device=dev,
                  body=load_body(device=dev), knobs=knobs)


def _masks_worker(scene_id: str) -> tuple[str, str, int, int]:
    gt = _STATE["paths"].feature_dir("gt", scene_id)
    try:
        fields = scene_masks(gt, body=_STATE["body"], device=_STATE["device"],
                             masks_dir=gt / "masks")
        save_boxes(gt / "bboxes.npz", fields)
        return scene_id, "", int(fields["valid"].sum()), int(fields["person_valid"].sum())
    except Exception as exc:                       # a corrupt scene must not kill the batch
        return scene_id, f"{type(exc).__name__}: {exc}", 0, 0


def _forces_worker(scene_id: str) -> tuple[str, str, float, float]:
    gt = _STATE["paths"].feature_dir("gt", scene_id)
    try:
        fields = scene_forces(gt, body=_STATE["body"], device=_STATE["device"],
                              **_STATE["knobs"]["forces"])
        save_forces(gt / "forces.npz", fields)
        weight = fields["total_mass"] * 9.81                                  # (P,)
        residual = fields["base_wrench_diagnostics"][:, 2] / weight           # after, in BW
        return scene_id, "", float(residual.mean()), float(fields["total_mass"].mean())
    except Exception as exc:
        return scene_id, f"{type(exc).__name__}: {exc}", 0.0, 0.0


def _contacts_worker(scene_id: str) -> tuple[str, str, float, float, float]:
    gt = _STATE["paths"].feature_dir("gt", scene_id)
    try:
        fields = scene_contacts(gt, body=_STATE["body"], device=_STATE["device"],
                                knobs=_STATE["knobs"]["contacts"])
        save_contacts(gt / "contacts.npz", fields)
        touching = fields["touching"]
        held = fields["frame_contact"].any(-1)
        unsupported = float((touching & ~held).sum()) / max(1, int(touching.sum()))
        return (scene_id, "", float(fields["frame_contact"].mean()), unsupported,
                float(fields["plane_suspect"]))
    except Exception as exc:
        return scene_id, f"{type(exc).__name__}: {exc}", 0.0, 0.0, 0.0


def _run_pool(name: str, worker, scene_ids: list[str], *, workers: int,
              data_root: Path, device: str, knobs: dict) -> list[tuple]:
    """Fan ``worker`` over scenes, reporting progress and tolerating per-scene failure."""
    print(f"[{name}] {len(scene_ids):,} scene(s), {workers} worker(s)", flush=True)
    ctx = mp.get_context("spawn")                  # CUDA cannot survive a fork
    results, failures = [], []
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(str(data_root), device, knobs, workers)) as pool:
        for done, row in enumerate(pool.imap_unordered(worker, scene_ids, chunksize=1), 1):
            results.append(row)
            if row[1]:
                failures.append(f"{row[0]}: {row[1]}")
            if done % 25 == 0 or done == len(scene_ids):
                print(f"  {done}/{len(scene_ids)}  ({len(failures)} failed)", flush=True)
    for line in failures[:20]:
        print(f"  FAIL {line}", flush=True)
    print(f"[{name}] done; {len(failures)} failure(s)", flush=True)
    return results


def run_masks(paths, rows, *, workers: int, device: str, skip_existing: bool) -> None:
    todo = [r["scene_id"] for r in rows
            if not skip_existing
            or not (paths.feature_dir("gt", r["scene_id"]) / "bboxes.npz").is_file()]
    results = _run_pool("masks", _masks_worker, todo, workers=workers,
                        data_root=paths.data_root, device=device, knobs={})
    frames = sum(r[2] for r in results)
    people = sum(r[3] for r in results)
    print(f"[masks] {frames:,} valid person-frames, {people:,} valid people", flush=True)


def run_contacts(paths, rows, *, workers: int, device: str, skip_existing: bool,
                 knobs: dict) -> None:
    todo = [r["scene_id"] for r in rows
            if not skip_existing
            or not (paths.feature_dir("gt", r["scene_id"]) / "contacts.npz").is_file()]
    results = _run_pool("contacts", _contacts_worker, todo, workers=workers,
                        data_root=paths.data_root, device=device, knobs=knobs)
    ok = [r for r in results if not r[1]]
    if ok:
        rate = float(np.mean([r[2] for r in ok]))
        unsupported = float(np.mean([r[3] for r in ok]))
        suspect = float(np.mean([r[4] for r in ok]))
        print(f"[contacts] mean label rate {rate:.3%} of person-frame-contact-frames; "
              f"frames where the mesh touches the floor but nothing is labelled: "
              f"{unsupported:.3%}", flush=True)
        # The support check shares the labels' floor plane, so it is blind to a body
        # standing on something the plane does not know about. This is that blind spot.
        print(f"[contacts] plane suspect (body too high to label, too low to be "
              f"airborne): {suspect:.3%} of person-frames", flush=True)


def run_forces(paths, rows, *, workers: int, device: str, skip_existing: bool,
               knobs: dict) -> None:
    todo = [r["scene_id"] for r in rows
            if not skip_existing
            or not (paths.feature_dir("gt", r["scene_id"]) / "forces.npz").is_file()]
    # The solve is a few hundred tiny L-BFGS steps: launch-bound on a GPU, where
    # workers only queue behind each other, so it runs on the cores and scales with them.
    results = _run_pool("forces", _forces_worker, todo, workers=workers,
                        data_root=paths.data_root, device=device, knobs=knobs)
    ok = [r for r in results if not r[1]]
    if ok:
        residual = np.array([r[2] for r in ok])
        print(f"[forces] root residual left after the solve, in body weights: mean "
              f"{residual.mean():.3f}, worst scene {residual.max():.3f}; body mass "
              f"{np.mean([r[3] for r in ok]):.1f} kg mean", flush=True)


def run_select(paths, rows, *, apply: bool) -> None:
    """Report — and optionally clear — the scenes whose every person is unusable."""
    missing, dead, people_dropped, frames = [], [], 0, [0, 0]
    for row in rows:
        box = paths.feature_dir("gt", row["scene_id"]) / "bboxes.npz"
        if not box.is_file():
            missing.append(row["scene_id"])
            continue
        d = np.load(box)
        person_valid = d["person_valid"]
        people_dropped += int((~person_valid).sum())
        frames[0] += int(d["valid"][person_valid].sum())
        frames[1] += int(d["valid"][person_valid].size)
        if not person_valid.any():
            dead.append(row["scene_id"])
    kept = len(rows) - len(missing) - len(dead)
    print(f"[select] {len(rows):,} scene(s): {kept:,} keep, {len(dead):,} have no valid "
          f"person, {len(missing):,} not measured yet", flush=True)
    print(f"[select] {people_dropped:,} person(s) with no valid frame; of the people that "
          f"survive, {frames[0]:,}/{frames[1]:,} frames are valid "
          f"({frames[0] / max(1, frames[1]):.2%})", flush=True)
    if not apply:
        print("[select] dry run — pass --apply to clear human_selected on the dead scenes",
              flush=True)
        return
    if not dead:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rollback = paths.scenes_root / f"_unselect_{stamp}.json"
    rollback.write_text(json.dumps(
        {"reason": "label_bedlam2: no person has a valid frame", "scene_ids": dead}, indent=1))
    conn = paths.connect_rw()
    with conn:
        conn.executemany("UPDATE scenes SET human_selected = 0 WHERE scene_id = ?",
                         [(s,) for s in dead])
    conn.close()
    print(f"[select] cleared human_selected on {len(dead):,} scene(s); "
          f"rollback list at {rollback}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--step", default="all",
                   choices=("masks", "contacts", "forces", "select", "all"))
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs" / "bedlam.toml")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--scenes", default=None, help="comma-separated scene_ids")
    p.add_argument("--sample", type=int, default=None,
                   help="a deterministic random subset (--sample 200 --seed 0 is the "
                        "set `view_3d.py --bedlam` browses)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--skip-existing", action="store_true",
                   help="leave scenes that already have the step's artifact alone")
    p.add_argument("--apply", action="store_true",
                   help="select: actually clear human_selected (default is a dry run)")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    cfg = load_config(args.config)
    # Defaults come from the module, never a second copy: a --config without a
    # [bedlam] section must behave exactly like the shipped readout.
    contact_knobs = vars(section(cfg, "bedlam", {
        "window_s": WINDOW_S, "still_mps": STILL_MPS, "enter_m": ENTER_M, "exit_m": EXIT_M,
        "min_span_s": MIN_SPAN_S, "max_tilt_deg": MAX_TILT_DEG, "touch_m": TOUCH_M,
        "suspect_m": SUSPECT_M,
    }))
    forces = section(cfg, "bedlam_forces", {
        "config": "configs/human_optimization.toml", "edge_speed_m_s": 0.0, "weights": {},
        "device": "cpu"})
    # The force solve's objective is kindyn_1's; the worker pool gets the parsed config.
    knobs = {"contacts": contact_knobs,
             "forces": {"config": load_human_config(REPO_ROOT / forces.config),
                        "weight_overrides": {k: float(v) for k, v in forces.weights.items()},
                        "edge_speed_m_s": float(forces.edge_speed_m_s)}}
    device = section(cfg, "runtime", {"device": "cuda"}).device

    paths = scene_db.ScenePaths(args.data_root or BEDLAM2_OUR)
    with paths.connect() as conn:
        rows = scene_db.select_targets(conn, categories=(), rope_supported=None)
    rows = scene_db.resolve_rows(rows, scenes=args.scenes, sample=args.sample,
                                 limit=args.limit, seed=args.seed)
    if not rows:
        print("[label_bedlam2] no scenes match the selection", flush=True)
        return 1

    steps = ("masks", "contacts", "forces", "select") if args.step == "all" else (args.step,)
    for step in steps:
        if step == "masks":
            run_masks(paths, rows, workers=args.workers, device=device,
                      skip_existing=args.skip_existing)
        elif step == "contacts":
            run_contacts(paths, rows, workers=args.workers, device=device,
                         skip_existing=args.skip_existing, knobs=knobs)
        elif step == "forces":
            run_forces(paths, rows, workers=args.workers, device=str(forces.device),
                       skip_existing=args.skip_existing, knobs=knobs)
        elif step == "select":
            run_select(paths, rows, apply=args.apply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
