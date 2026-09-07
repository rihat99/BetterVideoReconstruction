"""Assign train/test ``dataset_split`` labels to new curated scenes.

The split is decided per SOURCE VIDEO — every scene of a video lands on one
side, never both. Videos that already carry a label keep it (and pass it on to
any of their scenes that are still unlabelled). Among the fully-new videos,
test videos are drawn at random from the SMALL ones only (few scenes — big
videos are too self-similar to make good test material) until test holds about
``test_frames_frac`` of the corpus frames. Everything else becomes train.

Knobs live in ``configs/corpus.toml`` ``[split]``; the CLI takes only
``--config`` and ``--dry-run``. Applying writes the labels into ``scenes.db``
(a deliberate, user-approved write to the upstream DB, alongside the QA
discard) and saves the applied assignment as JSON next to the run logs.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import tomllib
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from scene_db import ScenePaths, select_targets  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_REPO_ROOT / "configs" / "corpus.toml")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    args = parser.parse_args(argv)

    with args.config.open("rb") as f:
        cfg = tomllib.load(f)
    corpus = cfg.get("corpus", {})
    split = cfg.get("split", {})
    test_frac = float(split.get("test_frames_frac", 0.10))
    max_scenes = int(split.get("max_test_video_scenes", 4))
    seed = int(split.get("seed", 0))

    data_root = (_REPO_ROOT / corpus.get("data_root", "../data/ClimbingVideos")).resolve()
    paths = ScenePaths(data_root)
    conn = paths.connect()
    rope_value = int(corpus.get("rope_supported", 0))
    rows = select_targets(
        conn,
        categories=tuple(int(v) for v in corpus.get("categories", [1, 2])),
        rope_supported=None if rope_value < 0 else rope_value,
    )
    labels = {
        r["scene_id"]: r["dataset_split"]
        for r in conn.execute("SELECT scene_id, dataset_split FROM scenes")
    }

    by_video: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_video.setdefault(row["video_id"], []).append(row)

    total_frames = sum(r["n_frames"] for r in rows)
    target = round(test_frac * total_frames)
    assign: dict[str, str] = {}  # scene_id -> side, previously-unlabelled scenes only

    # 1) Existing labels win; unlabelled scenes of a labelled video inherit its side.
    test_frames = 0
    candidates: list[str] = []
    for video, scenes in by_video.items():
        sides = {labels[r["scene_id"]] for r in scenes} - {None}
        if len(sides) > 1:
            raise SystemExit(f"video {video} already has scenes on BOTH sides — fix manually")
        if sides:
            side = sides.pop()
            for r in scenes:
                if labels[r["scene_id"]] is None:
                    assign[r["scene_id"]] = side
            if side == "test":
                test_frames += sum(r["n_frames"] for r in scenes)
        else:
            candidates.append(video)

    # 2) Draw small new videos into test until the frame target is reached.
    small = [v for v in candidates if len(by_video[v]) <= max_scenes]
    random.Random(seed).shuffle(small)
    chosen: list[str] = []
    for video in small:
        if test_frames >= target:
            break
        chosen.append(video)
        test_frames += sum(r["n_frames"] for r in by_video[video])
        for r in by_video[video]:
            assign[r["scene_id"]] = "test"

    # 3) Every remaining unlabelled scene is train.
    for row in rows:
        if labels[row["scene_id"]] is None and row["scene_id"] not in assign:
            assign[row["scene_id"]] = "train"

    n_test = sum(1 for side in assign.values() if side == "test")
    print(f"corpus: {len(rows)} scenes / {len(by_video)} videos / {total_frames} frames")
    print(f"target test frames: {target} ({test_frac:.0%}); reached: {test_frames} "
          f"({test_frames / total_frames:.1%})")
    print(f"new test videos ({len(chosen)}, ≤{max_scenes} scenes each): {sorted(chosen)}")
    print(f"assigning: {n_test} test + {len(assign) - n_test} train "
          f"(previously unlabelled scenes)")
    if args.dry_run:
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = _REPO_ROOT / f"dataset_split_assign_{stamp}.json"
    out.write_text(json.dumps(assign, indent=1, sort_keys=True))
    print(f"assignment saved: {out} (rollback = set these scene_ids back to NULL)")

    rw = paths.connect_rw()
    with rw:
        rw.executemany(
            "UPDATE scenes SET dataset_split = ? WHERE scene_id = ?",
            [(side, sid) for sid, side in assign.items()],
        )
    rw.close()
    print("scenes.db updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
