#!/usr/bin/env python3
"""DB-driven SAM 3 human tracking over the ClimbingVideos scene database.

This is the first stage adapted from the single-clip pipeline to the
``../data/ClimbingVideos`` scene DB (mirrors how ``../BetterImageReconstruction``
migrated its stages). It runs on the curated **364-scene** target set
(``human_selected = 1 AND vlm_category IN (1,2) AND vlm_rope_supported = 0`` —
indoor bouldering + indoor rope, no rope support; the same target
``BetterParsing``'s tracking stage uses).

Seeding strategy: YOLO already tracked the VLM-selected climber across each clip
(``features/tracking/<aa>/<bb>/<scene>.npz``), but that trajectory is coarse and
drops the subject. So on the **first frame the track appears** we run SAM 3
**text "person" grounding** (as in the standalone pipeline), keep the detection
whose mask best matches the YOLO human-of-interest box (highest bbox IoU ≥
``--iou-thresh``), and then track **that clean person mask** across the rest of
the clip. The YOLO box only *selects* which person; SAM 3 defines the silhouette
and does the tracking. If the target isn't grounded on the seed frame (e.g. a
fade-in / black frame), it falls back to a plain box seed.

Output mirrors the DB feature layout — the standalone ``track_sam3`` outputs
(per-object per-frame masks, ``mask_overlay.mp4``, ``bboxes.npz``) land directly
in::

    ../data/ClimbingVideos/features/sam3/<aa>/<bb>/<scene_id>/

(``<aa>/<bb>`` = ``scene_id[:2]/scene_id[2:4]``, same sharding as the yolo/
tracking sidecars). Resumable: a scene with a ``bboxes.npz`` is skipped under
``--skip-existing``.

Orchestration: the scene list is split across the configured GPUs (default
1,2,3); one ``track_sam3.py`` worker subprocess runs per GPU over a disjoint
stride shard, each seeded from a per-shard JSON manifest. The tested SAM 3
tracker is reused unchanged via that manifest — this script only does scene
selection, first-frame-bbox extraction, and fan-out.

    # 20 random scenes on GPUs 1,2,3, overlays gathered for inspection:
    uv run python scripts/track_sam3_scenes.py --sample 20

    # dry-run: print the selected scenes + seed bboxes, no GPU work:
    uv run python scripts/track_sam3_scenes.py --sample 20 --dry-run

    # full target set, resumable:
    uv run python scripts/track_sam3_scenes.py --all --skip-existing
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import scene_db  # noqa: E402  (scripts/ on path)
from scene_db import ScenePaths, select_targets  # noqa: E402

_TRACK_SAM3 = _REPO_ROOT / "scripts" / "stages" / "track_sam3.py"

# Shared DB / layout defaults (see scene_db).
_DEFAULT_DATA_ROOT = scene_db.DEFAULT_DATA_ROOT
_DEFAULT_CATEGORIES = scene_db.DEFAULT_CATEGORIES
_DEFAULT_ROPE_SUPPORTED = scene_db.DEFAULT_ROPE_SUPPORTED


# ``ScenePaths`` + ``select_targets`` are imported from ``scene_db`` (shared with
# the vggt/sapiens driver, so the DB layout has a single source of truth).


# ------------------------- first-frame bbox prompt -------------------------


def first_frame_prompts(
    paths: ScenePaths, scene_id: str, *,
    text: str = "person", iou_thresh: float = 0.2, min_area_frac: float = 0.002,
) -> list[dict]:
    """SAM 3 ``select_bbox`` seed prompts from the YOLO human-of-interest track.

    Reads ``features/tracking/<scene>.npz`` (CSR-packed per-human boxes
    ``[frame, x1, y1, x2, y2, conf]``) and returns one prompt per *matched*
    human, taking that human's earliest tracked frame (98% start at frame 0; the
    rest use their first appearance). Each prompt is a ``select_bbox`` entry:
    SAM 3 text-grounds ``text`` on the seed frame and keeps the detection whose
    mask best matches this box (IoU ≥ ``iou_thresh``), then tracks that clean
    person mask — the box only *selects* the person. Returns ``[]`` when there
    is no usable track.
    """
    npz = paths.tracking_npz(scene_id)
    if not npz.is_file():
        return []
    d = np.load(npz, allow_pickle=True)
    matched = d["human_matched"]
    offsets = d["human_box_offsets"]
    boxes = d["boxes"]

    prompts: list[dict] = []
    for i in range(len(matched)):
        if not bool(matched[i]):
            continue
        sl = boxes[offsets[i]:offsets[i + 1]]
        if sl.shape[0] == 0:
            continue
        row = sl[int(sl[:, 0].argmin())]          # earliest frame for this human
        x1, y1, x2, y2 = (float(v) for v in row[1:5])
        if x2 - x1 < 2.0 or y2 - y1 < 2.0:        # skip degenerate boxes
            continue
        prompts.append({
            "select_bbox": [x1, y1, x2, y2],
            "text": text,
            "label": "person",
            "iou_thresh": iou_thresh,
            "min_area_frac": min_area_frac,
        })
    return prompts


def build_manifest(
    paths: ScenePaths, rows: list[sqlite3.Row], *,
    text: str, iou_thresh: float, min_area_frac: float,
) -> list[dict]:
    """One ``track_sam3`` manifest entry per scene with a usable seed prompt."""
    entries: list[dict] = []
    for row in rows:
        sid = row["scene_id"]
        prompts = first_frame_prompts(
            paths, sid, text=text, iou_thresh=iou_thresh, min_area_frac=min_area_frac,
        )
        if not prompts:
            print(f"  [skip] {sid}: no matched YOLO track / usable first-frame bbox")
            continue
        entries.append({
            "source": str(paths.clip_path(row)),
            "out_dir": str(paths.sam3_dir(sid)),
            "prompts": prompts,
        })
    return entries


# ------------------------------ orchestration ------------------------------


def gather_overlays(paths: ScenePaths, rows: list[sqlite3.Row], dest: Path) -> int:
    """Symlink each scene's ``mask_overlay.mp4`` into ``dest`` as ``<scene>.mp4``."""
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for row in rows:
        sid = row["scene_id"]
        src = paths.sam3_dir(sid) / "mask_overlay.mp4"
        if not src.is_file():
            continue
        link = dest / f"{sid}.mp4"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(src.resolve())
        n += 1
    return n


def scene_presence(paths: ScenePaths, scene_id: str) -> tuple[float, int, int] | None:
    """``(primary_present_frac, n_obj, n_frames)`` from a scene's ``bboxes.npz``.

    ``primary_present_frac`` is the fraction of frames where object 0 (the primary
    tracked subject) has a non-empty mask — the tracking-quality signal. Returns
    ``None`` when the scene produced no output (a hard failure).
    """
    npz = paths.sam3_dir(scene_id) / "bboxes.npz"
    if not npz.is_file():
        return None
    d = np.load(npz, allow_pickle=True)
    bpo = d["bboxes_per_obj"].astype(np.float32)          # (n_obj, N, 4) xyxy
    b = bpo[0]
    area = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return float((area > 0).mean()), int(bpo.shape[0]), int(bpo.shape[1])


def discard_failures(paths: ScenePaths, rows: list[sqlite3.Row],
                     min_presence: float) -> list[dict]:
    """Unmark ``human_selected = 0`` for scenes whose tracking failed.

    A scene fails when it produced no output, or its primary-subject mask is
    present in fewer than ``min_presence`` of frames (SAM 3 lost the climber —
    e.g. the fade-in hard case). Records ``filter_reason`` for auditability /
    reversal and returns the discard records.
    """
    discards: list[dict] = []
    for row in rows:
        sid = row["scene_id"]
        res = scene_presence(paths, sid)
        if res is None:
            discards.append({"scene_id": sid, "present": 0.0,
                             "n_frames": int(row["n_frames"]), "reason": "no_output"})
        elif res[0] < min_presence:
            discards.append({"scene_id": sid, "present": round(res[0], 4),
                             "n_frames": res[2], "reason": "low_presence"})

    if discards:
        conn = paths.connect_rw()
        try:
            for d in discards:
                conn.execute(
                    "UPDATE scenes SET human_selected = 0, filter_reason = ? WHERE scene_id = ?",
                    (f"sam3_track_failed(present={d['present']:.2f})", d["scene_id"]),
                )
            conn.commit()
        finally:
            conn.close()
    return discards


def run_worker(paths: ScenePaths, rows: list[sqlite3.Row], gpu: int, i: int, n: int,
               work_dir: Path, args: argparse.Namespace) -> subprocess.Popen | None:
    """Spawn one ``track_sam3.py`` on ``gpu`` over shard ``i/n`` of ``rows``."""
    shard_rows = scene_db.stride_shard(rows, i, n)
    entries = build_manifest(
        paths, shard_rows,
        text=args.text, iou_thresh=args.iou_thresh, min_area_frac=args.ground_min_area_frac,
    )
    if not entries:
        print(f"[shard {i}/{n} gpu={gpu}] no scenes — skipping")
        return None

    # Per-shard save-dir so concurrent workers don't clobber each other's
    # throwaway _batch_manifest.jsonl (per-scene outputs go to their own out_dir).
    shard_dir = work_dir / f"shard_{i}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest = shard_dir / "manifest.json"
    manifest.write_text(json.dumps(entries, indent=2))

    cmd = [
        sys.executable, str(_TRACK_SAM3),
        "--source", str(manifest),
        "--save-dir", str(shard_dir),       # only used for the batch manifest log
        "--masks-subdir", ".",              # data straight into features/sam3/<...>/<scene>/
        "--device", "cuda",                 # GPU pinned via CUDA_VISIBLE_DEVICES below
        "--confidence", str(args.confidence),
        "--part-length", str(args.part_length),
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    if not args.overlay:
        cmd.append("--no-overlay")

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    print(f"[shard {i}/{n} gpu={gpu}] {len(entries)} scene(s) → {manifest.name}")
    return subprocess.Popen(cmd, env=env)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT,
                   help=f"ClimbingVideos data root (default: {_DEFAULT_DATA_ROOT}).")
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--sample", type=int, default=None,
                     help="process a random subset of N target scenes (deterministic per --seed).")
    sel.add_argument("--limit", type=int, default=None,
                     help="process the first N target scenes (by scene_id).")
    sel.add_argument("--all", action="store_true",
                     help="process every target scene (default when none of --sample/--limit given).")
    sel.add_argument("--scenes", default=None,
                     help="comma-separated scene_ids to process (overrides --sample/--limit; "
                          "handy for re-running specific scenes).")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for --sample (default: 0).")
    p.add_argument("--devices", default="1,2,3",
                   help="comma-separated GPU ids, one worker each (default: 1,2,3).")
    p.add_argument("--categories", default=",".join(map(str, _DEFAULT_CATEGORIES)),
                   help="comma-separated vlm_category filter (default: 1,2 = indoor boulder+rope).")
    p.add_argument("--rope-supported", type=int, default=_DEFAULT_ROPE_SUPPORTED,
                   help="vlm_rope_supported filter (default: 0; pass -1 to not constrain).")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip scenes that already have features/sam3/<...>/bboxes.npz.")
    p.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=True,
                   help="write per-scene mask_overlay.mp4 (default: ON).")
    p.add_argument("--debug-overlay-dir", type=Path, default=None,
                   help="gather the selected scenes' overlays here as <scene>.mp4 "
                        "(default: <features>/sam3/_debug_overlays).")
    p.add_argument("--discard-failed", action="store_true",
                   help="after tracking, set human_selected=0 in scenes.db for scenes whose "
                        "primary-subject mask presence < --min-presence (hard cases like a "
                        "fade-in seed frame); records filter_reason + a _discarded.json sidecar.")
    p.add_argument("--min-presence", type=float, default=0.9,
                   help="min fraction of frames with a non-empty primary mask to KEEP a scene "
                        "(default: 0.9; below → discarded when --discard-failed).")
    p.add_argument("--text", default="person",
                   help="SAM 3 text-grounding phrase for the seed frame (default: person).")
    p.add_argument("--iou-thresh", type=float, default=0.2,
                   help="min IoU between a grounded person mask and the YOLO box to "
                        "select it as the seed (default: 0.2; below → fall back to box seed).")
    p.add_argument("--ground-min-area-frac", type=float, default=0.002,
                   help="drop grounded masks below this area frac before IoU-matching "
                        "(default: 0.002; low so small/distant climbers survive).")
    p.add_argument("--confidence", type=float, default=0.5,
                   help="SAM 3 text-grounding confidence on the seed frame (default: 0.5).")
    p.add_argument("--part-length", type=int, default=200,
                   help="frames per SAM 3 propagation part (default: 200).")
    p.add_argument("--dry-run", action="store_true",
                   help="print the selected scenes + seed bboxes and exit (no GPU).")
    args = p.parse_args(argv)

    paths = ScenePaths(args.data_root)
    if not paths.scenes_db.is_file():
        print(f"ERROR: scenes.db not found at {paths.scenes_db}", file=sys.stderr)
        return 1

    categories = tuple(int(x) for x in args.categories.split(",") if x.strip() != "")
    rope = None if args.rope_supported < 0 else args.rope_supported

    with paths.connect() as conn:
        rows = select_targets(conn, categories=categories, rope_supported=rope)

    # Selection: explicit scene_ids / random sample / prefix / all.
    rows = scene_db.resolve_rows(
        rows, scenes=args.scenes, sample=args.sample, limit=args.limit, seed=args.seed,
    )

    print(f"[track_sam3_scenes] target scenes: {len(rows)} "
          f"(categories={categories}, rope_supported={rope})")
    if not rows:
        return 0

    if args.dry_run:
        for row in rows:
            sid = row["scene_id"]
            prompts = first_frame_prompts(
                paths, sid, text=args.text, iou_thresh=args.iou_thresh,
                min_area_frac=args.ground_min_area_frac,
            )
            boxes = ", ".join(
                f"[{','.join(f'{v:.0f}' for v in pr['select_bbox'])}]" for pr in prompts
            )
            print(f"  {sid:24s} {row['width']}x{row['height']} {row['n_frames']}f  "
                  f"seed={len(prompts)} select_bbox(es): {boxes or '(none)'}")
        return 0

    gpus = [int(x) for x in args.devices.split(",") if x.strip() != ""]
    n = len(gpus)
    work_dir = paths.features_root / "sam3" / "_runs"

    procs = []
    for i, gpu in enumerate(gpus):
        proc = run_worker(paths, rows, gpu, i, n, work_dir, args)
        if proc is not None:
            procs.append((i, gpu, proc))

    codes = [(i, gpu, proc.wait()) for (i, gpu, proc) in procs]
    for i, gpu, code in codes:
        print(f"[shard {i}/{n} gpu={gpu}] exit code {code}")

    if args.overlay:
        dest = args.debug_overlay_dir or (paths.features_root / "sam3" / "_debug_overlays")
        got = gather_overlays(paths, rows, dest)
        print(f"[overlays] gathered {got}/{len(rows)} overlay(s) → {dest}")

    if args.discard_failed and not all(code == 0 for _, _, code in codes):
        # A worker crashed → its unprocessed scenes would look like "no_output"
        # and be wrongly discarded. Skip the auto-discard; re-run (skip-existing
        # resumes) once workers are clean, then it will discard only real failures.
        print("\n[discard] SKIPPED — a worker exited non-zero. Re-run with "
              "--skip-existing to finish, then the discard pass will run.")
    elif args.discard_failed:
        discards = discard_failures(paths, rows, args.min_presence)
        sam3_root = paths.features_root / "sam3"
        sam3_root.mkdir(parents=True, exist_ok=True)
        (sam3_root / "_discarded.json").write_text(json.dumps(discards, indent=2))
        if discards:
            drows = [r for r in rows if r["scene_id"] in {d["scene_id"] for d in discards}]
            gather_overlays(paths, drows, sam3_root / "_discarded_overlays")
        print(f"\n[discard] {len(discards)}/{len(rows)} scene(s) discarded "
              f"(presence < {args.min_presence}); human_selected → 0, filter_reason recorded")
        for d in sorted(discards, key=lambda x: x["present"]):
            print(f"   {d['scene_id']:24s} present={d['present']:.2f}  ({d['reason']})")

    return 0 if all(code == 0 for _, _, code in codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
