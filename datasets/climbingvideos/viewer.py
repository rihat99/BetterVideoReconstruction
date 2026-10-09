"""Browse the ClimbingVideos scene DB in the viewer (``scripts/view_3d.py --scenes``).

The DB keeps its features stage-first (``features/<stage>/<aa>/<bb>/<scene>/``), so the selected
scenes are aliased into a throwaway ``out/<stem>/``-shaped symlink farm the generic viewer
serves. BEDLAM2 shares the DB layout and reuses :func:`select_scenes` / :func:`serve_scene_farm`
with its one feature stage.
"""
from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path

from bvr.config import load_config
from bvr.paths import REPO_ROOT
from datasets.climbingvideos import scene_db

#: Which scenes to browse: ``[corpus]`` (data_root, categories, rope_supported, seed) + ``[viewer]``.
CORPUS_CONFIG = Path(__file__).resolve().parent / "configs" / "corpus.toml"

#: The stage feature dirs the loaders read, each aliased as ``<farm>/<stage>``.
FARM_STAGES = ("geometry", "sam3d", "vggt", "human_optim", "geocalib")


def select_scenes(data_root: Path, *, categories: tuple[int, ...], rope_supported: int | None,
                  scene_ids: str | None, sample: int | None, limit: int | None, seed: int) -> list:
    """The DB's target rows under ``data_root``, narrowed by ids / a random sample / a limit."""
    paths = scene_db.ScenePaths(data_root)
    with paths.connect() as conn:
        rows = scene_db.select_targets(conn, categories=categories, rope_supported=rope_supported)
    return scene_db.resolve_rows(rows, scenes=scene_ids, sample=sample, limit=limit, seed=seed)


def _link_scene_farm(paths, row, farm_root: Path, *, stages: tuple[str, ...],
                     needs: tuple[str, str]) -> bool:
    """Alias one DB scene's stage-first feature dirs into an ``out/<stem>/``-shaped symlink farm.

    Builds ``<farm_root>/<scene_id>/<stage>`` links (skipping stages never produced) plus the source
    clip as ``<farm_root>/<scene_id><ext>`` for the video pane. ``needs`` is the ``(stage, file)``
    the viewer cannot draw without; a scene missing it is skipped.
    """
    sid = row["scene_id"]
    if not (paths.feature_dir(needs[0], sid) / needs[1]).exists():
        return False
    farm = farm_root / sid
    farm.mkdir(parents=True, exist_ok=True)
    for stage in stages:
        src, link = paths.feature_dir(stage, sid), farm / stage
        if src.exists() and not link.exists():
            link.symlink_to(src)
    clip = paths.clip_path(row)
    vlink = farm_root / f"{sid}{clip.suffix}"
    if clip.exists() and not vlink.exists():
        vlink.symlink_to(clip)
    return True


def serve_scene_farm(view, data_root: Path, rows: list, *, stages: tuple[str, ...],
                     needs: tuple[str, str]) -> None:
    """Alias ``rows`` into a throwaway symlink farm and serve it through ``view(root, video_dir=...)``,
    the same entry point a local ``out/`` tree uses."""
    if not rows:
        print("[view_3d] no target scenes match the selection", flush=True)
        return
    paths = scene_db.ScenePaths(data_root)
    farm_root = Path(tempfile.mkdtemp(prefix="bvr_view_"))
    atexit.register(shutil.rmtree, farm_root, ignore_errors=True)   # symlinks only; targets untouched
    n = sum(_link_scene_farm(paths, row, farm_root, stages=stages, needs=needs) for row in rows)
    if not n:
        print(f"[view_3d] none of {len(rows)} target scene(s) have {needs[0]}/{needs[1]} under {data_root}",
              flush=True)
        return
    print(f"[view_3d] prepared {n}/{len(rows)} scene(s) from {data_root}", flush=True)
    view(farm_root, video_dir=farm_root)


def run_scene_db(view, config_path: Path = CORPUS_CONFIG) -> None:
    """Browse the corpus's target scenes (``[corpus]`` of ``corpus.toml``, narrowed by its
    ``[viewer]`` scene_ids / sample / limit knobs)."""
    config = load_config(config_path)
    corpus, knobs = config.get("corpus", {}), config.get("viewer", {})
    data_root = Path(corpus.get("data_root", scene_db.DEFAULT_DATA_ROOT))
    if not data_root.is_absolute():
        data_root = (REPO_ROOT / data_root).resolve()
    rope = int(corpus.get("rope_supported", scene_db.DEFAULT_ROPE_SUPPORTED))
    rows = select_scenes(data_root, categories=tuple(int(c) for c in corpus.get("categories", [1, 2])),
                         rope_supported=None if rope < 0 else rope,
                         scene_ids=knobs.get("scene_ids") or None,
                         sample=int(knobs["sample"]) if knobs.get("sample", "") != "" else None,
                         limit=int(knobs["limit"]) if knobs.get("limit", "") != "" else None,
                         seed=int(corpus.get("seed", 0)))
    serve_scene_farm(view, data_root, rows, stages=FARM_STAGES, needs=("geometry", "scene.npz"))
