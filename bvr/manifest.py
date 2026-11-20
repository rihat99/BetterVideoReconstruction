"""Stage manifests: what ran, on what, with which code and config.

Every step of every clip ends by writing ``<clip>/manifests/<step>.json``. A step is done only if
its manifest says so, and a resume skips it only while the config it read, the code and the
checksums of its inputs are what the manifest recorded (:func:`stale`). A file that merely exists
proves nothing: it may be half a run's, or an older config's.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

from bvr.paths import REPO_ROOT

FORMAT = "bvr-stage"
FORMAT_VERSION = 1


def record(*, step: str, status: str, error: str | None, started: str, finished: str, seconds: float,
           command: list[str], config: dict, code: dict, environment: dict, device: str, seed: int,
           peak_gpu_bytes: int, inputs: dict, outputs: dict) -> dict:
    """Return a manifest: every field, in the order it is written.

    :param status: ``"complete"`` or ``"failed"``; ``error`` says why it failed.
    :param config: the config tables the step read, resolved.
    :param inputs: per input, named relative to the clip, where it was read and its checksum.
    :param outputs: per output, named relative to the clip, its checksum.
    """
    return {
        "format": FORMAT, "format_version": FORMAT_VERSION,
        "step": step, "status": status, "error": error,
        "started": started, "finished": finished, "seconds": seconds,
        "command": command,
        "config": config, "config_sha256": digest(config),
        "code": code, "environment": environment,
        "device": device, "seed": seed, "peak_gpu_bytes": peak_gpu_bytes,
        "inputs": inputs, "outputs": outputs,
    }


def write(path: Path, manifest: dict) -> None:
    """Write a manifest; the file is whole or absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, path)


def read(path: Path) -> dict | None:
    """Return the manifest at ``path``, or ``None`` when the step never finished there."""
    return json.loads(path.read_text()) if path.exists() else None


def stale(manifest: dict | None, *, config: dict, code: dict, inputs: dict) -> str | None:
    """Say why ``manifest`` does not vouch for a step with this config, code and inputs.

    :return: ``None`` when the step is complete and nothing it depends on has changed.
    """
    if manifest is None:
        return "no manifest"
    if manifest["status"] != "complete":
        return f"status is {manifest['status']}"
    if manifest["config_sha256"] != digest(config):
        return "the config changed"
    if (manifest["code"]["commit"], manifest["code"]["diff_sha256"]) != (code["commit"], code["diff_sha256"]):
        return "the code changed"
    recorded = {name: entry["checksum"] for name, entry in manifest["inputs"].items()}
    current = {name: entry["checksum"] for name, entry in inputs.items()}
    changed = sorted(name for name in recorded.keys() | current.keys() if recorded.get(name) != current.get(name))
    return f"inputs changed: {', '.join(changed)}" if changed else None


def digest(value: object) -> str:
    """The sha256 of a JSON value, independent of key order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def checksum(path: Path) -> str | None:
    """A checksum of a file's contents, or of a folder's listing; ``None`` when nothing is there.

    A folder of frames is too large to hash on every resume, so its checksum covers the sorted
    names and sizes of its files.
    """
    if path.is_file():
        contents = hashlib.sha256()
        with path.open("rb") as file:
            while block := file.read(1 << 20):
                contents.update(block)
        return "sha256:" + contents.hexdigest()
    if path.is_dir():
        listing = sorted((str(file.relative_to(path)), file.stat().st_size)
                         for file in path.rglob("*") if file.is_file())
        return "listing:" + digest(listing)
    return None


def code(save_dir: Path | None) -> dict:
    """The state of this repository's code: commit, dirty flag and the checksum of its diff.

    A dirty tree's diff against the commit is saved once under ``<save_dir>/_runs/``, named by
    the commit and the diff's checksum; ``None`` saves nothing (a dry run). Untracked files set
    the dirty flag and are not in the diff.
    """
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    diff = _git("diff", "HEAD") if dirty else ""
    state = {"commit": commit, "dirty": dirty, "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(), "diff": None}
    if diff and save_dir is not None:
        saved = Path("_runs") / f"{commit[:12]}-{state['diff_sha256'][:12]}" / "code.diff"
        if not (save_dir / saved).exists():
            (save_dir / saved).parent.mkdir(parents=True, exist_ok=True)
            (save_dir / saved).write_text(diff)
        state["diff"] = str(saved)
    return state


def environment() -> dict:
    """The interpreter, torch, CUDA, the lock file and the commit of every library and submodule."""
    import torch  # here, so that reading a manifest or printing a dry run does not load torch

    submodules = {line.split()[1]: line.split()[0].lstrip("+-U")
                  for line in _git("submodule", "status").splitlines()}
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "uv_lock_sha256": hashlib.sha256((REPO_ROOT / "uv.lock").read_bytes()).hexdigest(),
        "libraries": {name: _library(name) for name in ("better-robot", "better-human")},
        "submodules": submodules,
    }


def _library(name: str) -> str:
    """``<version>@<commit>`` of an installed library; ``+dirty`` marks a modified checkout."""
    distribution = importlib.metadata.distribution(name)
    origin = json.loads(distribution.read_text("direct_url.json") or "{}")
    if "vcs_info" in origin:
        return f"{distribution.version}@{origin['vcs_info']['commit_id']}"
    if "dir_info" not in origin:
        return distribution.version
    checkout = origin["url"].removeprefix("file://")
    dirty = "+dirty" if _git("status", "--porcelain", cwd=checkout) else ""
    return f"{distribution.version}@{_git('rev-parse', 'HEAD', cwd=checkout)}{dirty}"


def _git(*arguments: str, cwd: str | Path = REPO_ROOT) -> str:
    return subprocess.run(["git", *arguments], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
