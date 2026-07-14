"""Shared ClimbingVideos scene-DB access, feature layout, and row selection.

The corpus driver resolves every stage's scenes, on-disk paths, and target set
the same way, so that logic lives here.

Feature layout is **stage-first**, matching the existing DB dirs
(``features/{yolo,tracking,vlm,sam3}/<aa>/<bb>/<scene_id>``): each stage writes
under ``features/<stage>/<aa>/<bb>/<scene_id>/`` where ``<aa>/<bb> =
scene_id[:2]/scene_id[2:4]``.

The scenes DB (``scenes/scenes.db``) is owned by the upstream ``../BetterParsing``
pipeline and is opened **read-only** here; :meth:`ScenePaths.connect_rw` exists
only for the explicit, auditable discard of scenes whose tracking failed.
"""
from __future__ import annotations

import random
import sqlite3
from pathlib import Path

# ``../data/ClimbingVideos`` relative to the repo root (this file is scripts/).
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = _REPO_ROOT.parent / "data" / "ClimbingVideos"

# The curated target set: human-selected indoor bouldering + indoor rope, no
# rope support (== BetterParsing's SceneTrackConfig default). Overridable on CLI.
DEFAULT_CATEGORIES = (1, 2)
DEFAULT_ROPE_SUPPORTED = 0


def shard(scene_id: str) -> tuple[str, str]:
    """``scene_id`` → its two-level shard (``aa``, ``bb``)."""
    return scene_id[:2], scene_id[2:4]


class ScenePaths:
    """Resolve DB rows to on-disk clip / feature paths for one data root."""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.scenes_root = self.data_root / "scenes"
        self.scenes_db = self.scenes_root / "scenes.db"
        self.features_root = self.data_root / "features"

    # ----- DB connections --------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        # Read-only: this DB is owned by the upstream BetterParsing pipeline.
        conn = sqlite3.connect(f"file:{self.scenes_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def connect_rw(self) -> sqlite3.Connection:
        # Read-write — only for the explicit discard of failed scenes.
        conn = sqlite3.connect(self.scenes_db)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    # ----- path resolution -------------------------------------------------

    def clip_path(self, row: sqlite3.Row) -> Path:
        return self.scenes_root / row["rel_path"]

    def feature_dir(self, stage: str, scene_id: str) -> Path:
        """``features/<stage>/<aa>/<bb>/<scene_id>`` — the per-scene stage dir."""
        aa, bb = shard(scene_id)
        return self.features_root / stage / aa / bb / scene_id

    def tracking_npz(self, scene_id: str) -> Path:
        aa, bb = shard(scene_id)
        return self.features_root / "tracking" / aa / bb / f"{scene_id}.npz"

    def sam3_dir(self, scene_id: str) -> Path:
        return self.feature_dir("sam3", scene_id)

    def sam3_bboxes(self, scene_id: str) -> Path:
        """The SAM 3 ``bboxes.npz`` the pose / body stages consume."""
        return self.sam3_dir(scene_id) / "bboxes.npz"

    def vggt_camera(self, scene_id: str) -> Path:
        """The VGGT-Omega ``camera.npz`` the body stage reads intrinsics from."""
        return self.feature_dir("vggt", scene_id) / "camera.npz"


def select_targets(
    conn: sqlite3.Connection,
    *,
    categories: tuple[int, ...],
    rope_supported: int | None,
) -> list[sqlite3.Row]:
    """The curated target scenes, ordered by ``scene_id`` (stable)."""
    q = ("SELECT scene_id, video_id, rel_path, width, height, fps, n_frames "
         "FROM scenes WHERE human_selected = 1")
    params: list = []
    if categories:
        q += f" AND vlm_category IN ({','.join('?' * len(categories))})"
        params += list(categories)
    if rope_supported is not None:
        q += " AND vlm_rope_supported = ?"
        params.append(rope_supported)
    q += " ORDER BY scene_id"
    return conn.execute(q, params).fetchall()


def resolve_rows(
    rows: list[sqlite3.Row],
    *,
    scenes: str | list[str] | None = None,
    sample: int | None = None,
    limit: int | None = None,
    seed: int = 0,
) -> list[sqlite3.Row]:
    """Narrow ``rows`` by the mutually-exclusive selection flags.

    ``scenes`` (an id list or comma-separated ids) wins, then a deterministic random
    ``sample`` (by ``seed``), then a ``limit`` prefix; otherwise all rows.
    Unknown ids in ``scenes`` are reported and skipped.
    """
    if scenes is not None:
        values = scenes if isinstance(scenes, list) else scenes.split(",")
        want = [s.strip() for s in values if s.strip()]
        by_id = {r["scene_id"]: r for r in rows}
        missing = [s for s in want if s not in by_id]
        if missing:
            print(f"WARNING: {len(missing)} scene(s) not in the target set: {missing}")
        return [by_id[s] for s in want if s in by_id]
    if sample is not None:
        k = min(sample, len(rows))
        return sorted(random.Random(seed).sample(rows, k), key=lambda r: r["scene_id"])
    if limit is not None:
        return rows[:limit]
    return rows


def stride_shard(rows: list, i: int, n: int) -> list:
    """Rows ``i, i+n, i+2n, …`` — a stable disjoint partition of ``rows``."""
    return rows[i::n]


def parse_int_csv(s: str) -> tuple[int, ...]:
    """``"1,2,3"`` → ``(1, 2, 3)`` (empty fields dropped)."""
    return tuple(int(x) for x in s.split(",") if x.strip() != "")
