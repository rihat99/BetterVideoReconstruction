"""Standard precompute-script runner: load model once, iterate over inputs.

Each precompute step subclasses :class:`Step` (``load`` / ``process`` /
``release``) and calls :func:`run_batch` from its ``main()``. The runner
parses :func:`add_standard_args` (``--source``, ``--save-dir``, ...) and
dispatches inputs through :func:`iter_inputs`, which accepts a single image
or video, a directory of either, a glob, or a ``.txt`` / ``.json`` manifest.

Contract:

  1. ``Step.load()`` is called once at process start (after at least one
     unskipped input is found).
  2. ``Step.process(item)`` is called per :class:`InputItem`. The step is
     responsible for honoring ``args.skip_existing`` — the runner doesn't
     know each step's output paths.
  3. ``Step.release()`` is called once before exit.

Outputs convention: each item lands at ``<save_dir>/<source.stem>/...`` so a
folder-batch run materializes N parallel ``<run_dir>`` trees compatible with
the orchestrator's per-source layout. Single-input invocations from
``run_phy_sic.py`` hit the same code path with one item.

Source forms accepted by :func:`iter_inputs`:

  * single image (``.jpg/.jpeg/.png/.bmp/.webp``) or video file
    (``.mp4/.mov/.avi/.mkv/.webm``), case-insensitive
  * directory — globbed for image+video extensions, sorted by stem (numeric
    if possible, lexicographic otherwise)
  * glob pattern containing ``* ? [`` — expanded with :mod:`glob`
  * ``.txt`` manifest — one source path per line, ``#`` comments allowed
  * ``.json`` manifest — list of dicts with keys ``source`` (or legacy
    ``image``, required), ``mask`` (optional), ``out_dir`` (optional,
    overrides the default ``<save_dir>/<source.stem>/``), ``prompts``
    (optional list passed through as :attr:`InputItem.prompts` — used by
    steps like ``track_objects_sam3.py`` for per-item prompt overrides;
    ignored by steps that don't read it)
"""
from __future__ import annotations

import argparse
import glob
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

from tools.source import IMAGE_EXTS, VIDEO_EXTS

_MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS
_MANIFEST_EXTS = {".txt", ".json"}
_GLOB_CHARS = set("*?[")


@dataclass
class InputItem:
    """One unit of work for a precompute Step."""

    source_path: Path
    out_dir: Path                      # <save_dir>/<source.stem>/ unless overridden by manifest
    mask_path: Path | None = None
    prompts: list[dict] | None = None  # opaque per-item prompt list — only some steps read this

    @property
    def stem(self) -> str:
        return self.source_path.stem


# -----------------------------------------------------------------------------
# Source resolution
# -----------------------------------------------------------------------------

def _looks_like_glob(s: str) -> bool:
    return any(c in s for c in _GLOB_CHARS)


def _list_media_dir(folder: Path) -> list[Path]:
    paths = [p for p in folder.iterdir() if p.suffix.lower() in _MEDIA_EXTS]
    try:
        paths.sort(key=lambda p: int(p.stem))
    except ValueError:
        paths.sort()
    return paths


def _resolve_mask(image: Path, mask: Path | None, mask_dir: Path | None) -> Path | None:
    """Per-image mask resolution. ``--mask`` wins over ``--mask-dir``.

    For ``--mask-dir``, probes ``<mask_dir>/<stem>.{png,jpg,jpeg,webp,bmp}``
    in that order and returns the first hit. Returns ``None`` if neither
    flag is set or no probe matches (the caller decides if that's an error).
    """
    if mask is not None:
        return mask
    if mask_dir is None:
        return None
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        candidate = mask_dir / f"{image.stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _read_txt_manifest(path: Path) -> list[Path]:
    items: list[Path] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(Path(line))
    return items


def _read_json_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"manifest {path} must be a JSON list of objects")
    return data


def iter_inputs(
    source: str | Path,
    save_dir: Path,
    *,
    mask: Path | None = None,
    mask_dir: Path | None = None,
) -> Iterator[InputItem]:
    """Yield :class:`InputItem` for every image implied by ``source``."""
    save_dir = Path(save_dir)
    src_str = str(source)
    src_path = Path(source)

    # Glob pattern (literal *, ?, [)
    if _looks_like_glob(src_str):
        paths = sorted(Path(p) for p in glob.glob(src_str))
        if not paths:
            raise FileNotFoundError(f"glob matched nothing: {src_str}")
        for p in paths:
            yield InputItem(p, save_dir / p.stem, _resolve_mask(p, mask, mask_dir))
        return

    # JSON / TXT manifest — extension-driven
    suffix = src_path.suffix.lower()
    if suffix == ".json":
        for entry in _read_json_manifest(src_path):
            src_key = "source" if "source" in entry else ("image" if "image" in entry else None)
            if src_key is None:
                raise ValueError(f"manifest entry missing 'source' (or 'image'): {entry}")
            src = Path(entry[src_key])
            out_dir = Path(entry["out_dir"]) if "out_dir" in entry else save_dir / src.stem
            entry_mask = Path(entry["mask"]) if "mask" in entry else _resolve_mask(src, mask, mask_dir)
            yield InputItem(src, out_dir, entry_mask, prompts=entry.get("prompts"))
        return
    if suffix == ".txt":
        for src in _read_txt_manifest(src_path):
            yield InputItem(src, save_dir / src.stem, _resolve_mask(src, mask, mask_dir))
        return

    # Directory of images / videos
    if src_path.is_dir():
        paths = _list_media_dir(src_path)
        if not paths:
            raise FileNotFoundError(f"no image or video files in directory {src_path}")
        for p in paths:
            yield InputItem(p, save_dir / p.stem, _resolve_mask(p, mask, mask_dir))
        return

    # Single image or video
    if src_path.is_file() and suffix in _MEDIA_EXTS:
        yield InputItem(src_path, save_dir / src_path.stem,
                        _resolve_mask(src_path, mask, mask_dir))
        return

    raise FileNotFoundError(
        f"--source must be an image/video file, directory, glob, or .txt/.json manifest: {source}"
    )


# -----------------------------------------------------------------------------
# Step protocol + batch runner
# -----------------------------------------------------------------------------

class Step(Protocol):
    """Per-step contract. ``load``/``release`` run once; ``process`` runs per item."""

    def load(self) -> None: ...
    def process(self, item: InputItem) -> None: ...
    def release(self) -> None: ...


def add_standard_args(parser: argparse.ArgumentParser) -> None:
    """Attach the shared precompute-script flags. Step-specific flags go on top."""
    parser.add_argument(
        "--source", required=True, type=str,
        help="single image/video | directory | glob pattern | .txt or .json manifest",
    )
    parser.add_argument(
        "--save-dir", required=True, type=Path,
        help="parent dir; per-item outputs land in <save-dir>/<stem>/...",
    )
    parser.add_argument(
        "--config", default=None, type=Path,
        help="optional TOML/YAML config (interpretation is step-specific)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="step may skip items whose outputs are already on disk",
    )
    parser.add_argument("--device", default="cuda", help="cuda | cpu (default: cuda)")
    parser.add_argument(
        "--dtype", default="fp16", choices=("fp16", "bf16", "fp32"),
        help="model dtype hint passed to the step (default: fp16)",
    )


def run_batch(step: Step, args: argparse.Namespace) -> int:
    """Resolve inputs, load the step once, iterate, write a JSONL manifest.

    Returns 0 on full success, 1 if any item raised. Errors don't abort the
    batch — a benchmark run over 50 frames shouldn't die because frame 7's
    mask is corrupt.
    """
    items = list(iter_inputs(
        args.source, args.save_dir,
        mask=getattr(args, "mask", None),
        mask_dir=getattr(args, "mask_dir", None),
    ))
    print(f"[precompute] {len(items)} item(s) from {args.source}")
    if not items:
        return 0

    args.save_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.save_dir / "_batch_manifest.jsonl"
    n_ok = n_skip = n_err = 0

    step.load()
    try:
        with manifest_path.open("w") as mf:
            for i, item in enumerate(items, 1):
                t0 = time.monotonic()
                status = "ok"
                err: str | None = None
                try:
                    step.process(item)
                except SkipItem as e:
                    status = "skipped"
                    n_skip += 1
                    print(f"[{i}/{len(items)}] {item.stem} skipped ({e})")
                except Exception as e:
                    status = "err"
                    err = f"{type(e).__name__}: {e}"
                    n_err += 1
                    print(f"[{i}/{len(items)}] {item.stem} ERR {err}")
                    traceback.print_exc()
                else:
                    n_ok += 1
                    dt = time.monotonic() - t0
                    print(f"[{i}/{len(items)}] {item.stem} ok dt={dt:.2f}s")
                mf.write(json.dumps({
                    "stem": item.stem,
                    "source": str(item.source_path),
                    "out_dir": str(item.out_dir),
                    "status": status,
                    "duration_s": round(time.monotonic() - t0, 3),
                    "error": err,
                }) + "\n")
                mf.flush()
    finally:
        step.release()

    print(f"[precompute] done: {n_ok} ok, {n_skip} skipped, {n_err} err -> {manifest_path}")
    return 0 if n_err == 0 else 1


class SkipItem(Exception):
    """Raise from ``Step.process`` to mark an item as skipped (vs errored)."""


__all__ = [
    "InputItem",
    "Step",
    "SkipItem",
    "add_standard_args",
    "iter_inputs",
    "run_batch",
]
