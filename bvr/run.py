"""The stage table and the code that runs it.

The chain, in order::

    sam3 -> vggt -> sapiens -> sam3d -> smplx -> fuse -> scale -> gravity
         -> kinematics_1 -> contacts_1 -> kinematics_2 -> contacts_2 -> kindyn_1

:data:`STEPS` has one row per step: its stage module, the config tables it reads, and the files it
reads and writes under a clip's directory. Stages meet only through those files.

:func:`chain` runs steps over the clips of a source, one process per step over all its clips, so
a model loads once and its memory is gone before the next step. :func:`main` is the entry point
of every stage module, which makes ``python -m bvr.stages.<module>`` the same command the chain
starts. A clip that fails is data: its manifest says ``failed``, the other clips go on, and the
exit code is 1 at the end.
"""
from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from bvr import config as configuration
from bvr import manifest, paths
from bvr.paths import Clip

if TYPE_CHECKING:
    import torch

log = logging.getLogger("bvr.run")

#: The clip's video, as an input of a step.
VIDEO = "video"


@dataclass(frozen=True)
class Step:
    """One row of the stage table."""

    name: str                   # what --steps, the manifests and the logs call it
    module: str                 # python -m bvr.stages.<module>
    sections: tuple[str, ...]   # the config tables it reads, as dotted paths
    inputs: tuple[str, ...]     # VIDEO, and files and folders relative to the clip
    outputs: tuple[str, ...]    # files and folders relative to the clip


_GEOMETRY = (paths.TRANSFORM, paths.SCENE, paths.HUMAN)
_KINEMATIC = ("optimizer", "chamfer", "keypoints", "reprojection_start", "penetration", "contact_attraction",
              "contact_acquisition", "contact_velocity", "auto_contact", "contact_gate")
_CONTACTS = ("contacts", "contact_force", "contact_gate", "contact_confidence", "quality", "friction")

STEPS = (
    Step("sam3", "track_sam3", ("sam3",), (VIDEO,), (paths.SAM3,)),
    Step("vggt", "estimate_camera_vggt", ("vggt",), (VIDEO,), (paths.VGGT,)),
    Step("sapiens", "estimate_sapiens", ("sapiens",), (VIDEO, paths.SAM3_BBOXES), (paths.SAPIENS,)),
    Step("sam3d", "estimate_body_sam3d", ("sam3d",), (VIDEO, paths.SAM3, paths.SAM3_BBOXES, paths.VGGT_CAMERA),
         (paths.SAM3D_PARAMS, paths.SAM3D_OVERLAY)),
    Step("smplx", "convert_mhr_to_smplx", ("smplx",), (paths.SAM3D_PARAMS,), (paths.SMPLX_PARAMS,)),
    Step("fuse", "fuse_geometry", ("fuse",),
         (VIDEO, paths.VGGT_CAMERA, paths.VGGT_DEPTH, paths.SAM3, paths.SAM3_BBOXES, paths.SMPLX_PARAMS),
         (paths.FUSED,)),
    Step("scale", "estimate_scale", ("scale",), (paths.FUSED, paths.SAPIENS_POSE, paths.SMPLX_PARAMS),
         (*_GEOMETRY, paths.SCALE, paths.PREFIT)),
    Step("gravity", "estimate_gravity", ("gravity",), (VIDEO, paths.TRANSFORM, paths.SCENE), (paths.GEOCALIB,)),
    Step("kinematics_1", "refine_kinematics", (*_KINEMATIC, "stages.root", "stages.kinematics_1"),
         (paths.SMPLX_PARAMS, paths.PREFIT, paths.SAPIENS_POSE, *_GEOMETRY), (paths.KINEMATIC_1,)),
    Step("contacts_1", "estimate_contacts", _CONTACTS, (paths.KINEMATIC_1, *_GEOMETRY, paths.GRAVITY),
         (paths.CONTACTS_1,)),
    Step("kinematics_2", "refine_kinematics", (*_KINEMATIC, "stages.kinematics_2"),
         (paths.KINEMATIC_1, paths.CONTACTS_1, paths.SAPIENS_POSE, *_GEOMETRY), (paths.KINEMATIC_2,)),
    Step("contacts_2", "estimate_contacts", _CONTACTS, (paths.KINEMATIC_2, *_GEOMETRY, paths.GRAVITY),
         (paths.CONTACTS_2,)),
    Step("kindyn_1", "solve_dynamics", (*_KINEMATIC, "stages.kindyn_1", "torque_multipliers", "friction"),
         (paths.KINEMATIC_2, paths.CONTACTS_2, paths.GRAVITY, paths.SAPIENS_POSE, *_GEOMETRY), (paths.KINDYN_1,)),
)


def chain(source: Path, save_dir: Path, *, config_path: Path | None = None, steps: Sequence[str] | None = None,
          upstream: Path | None = None, skip_existing: bool = False, shard: tuple[int, int] = (1, 1),
          dry_run: bool = False) -> int:
    """Run steps of the chain over the clips of ``source``; return the process exit code.

    :param source: a video, or a folder of videos (each a clip).
    :param save_dir: the run root; a clip's tree is ``<save_dir>/<stem>/``.
    :param config_path: a delta over ``configs/bvr.toml``.
    :param steps: steps to run, in chain order; default ``[run].steps`` of the config.
    :param upstream: another run root; a stage reads from it what this run does not have.
    :param skip_existing: skip a step for a clip whose manifest still vouches for it.
    :param shard: ``(i, n)`` takes every n-th clip starting at the i-th, so n runs fill one root.
    :param dry_run: log the commands and touch nothing.
    """
    _log_to_console(logging.getLogger("bvr"))
    config = configuration.load(config_path)
    selected = _steps(config["run"]["steps"] if steps is None else steps)
    clips = paths.clips(source, save_dir, upstream)[shard[0] - 1::shard[1]]
    code = manifest.code(None if dry_run else save_dir)
    log.info("%d clip(s) -> %s; steps: %s", len(clips), save_dir, ", ".join(step.name for step in selected))
    failed: set[str] = set()
    rewritten: set[str] = set()
    for step in selected:
        todo = [clip for clip in clips if clip.name not in failed]
        if skip_existing:
            todo = [clip for clip in todo if clip.name in rewritten or _stale(step, clip, config, code)]
        if not todo:
            log.info("%s: nothing to do", step.name)
            continue
        command = [sys.executable, "-m", f"bvr.stages.{step.module}", "--step", step.name,
                   "--source", str(source), "--save-dir", str(save_dir),
                   *([] if config_path is None else ["--config", str(config_path)]),
                   *([] if upstream is None else ["--upstream", str(upstream)]),
                   "--clips", ",".join(clip.name for clip in todo)]
        log.info("%s", shlex.join(command))
        if dry_run:
            # A step that would run rewrites the inputs of the steps after it.
            rewritten.update(clip.name for clip in todo)
            continue
        subprocess.run(command, check=False)
        failed.update(clip.name for clip in todo if not _complete(step, clip))
    for name in sorted(failed):
        log.error("%s failed; see %s", name, save_dir / name / "manifests")
    return 1 if failed else 0


@dataclass(frozen=True)
class Stage:
    """What a stage module's ``load`` receives: one step of one run."""

    step: str                   # the step's name; a module that serves two steps reads it
    config: dict                # the config tables of the step's row, and [run]
    device: torch.device
    generator: torch.Generator  # seeded from [run].seed, on ``device``


def main(load: Callable[[Stage], Callable[[Clip], None]]) -> None:
    """Entry point of a stage module: run its step over the clips named on the command line.

    ``load`` is called once, after the settings of ``[run]`` are applied and only if a clip has
    work to do. It loads what the clips share (a model, the body) and returns the function that
    processes one clip: it reads through ``clip.find`` and writes through ``clip.path``.
    """
    module = sys.modules[load.__module__].__spec__.name.rpartition(".")[2]
    sys.exit(stage(module, load, sys.argv[1:]))


def stage(module: str, load: Callable[[Stage], Callable[[Clip], None]], arguments: Sequence[str]) -> int:
    """Run one step of ``bvr.stages.<module>`` over its clips; return the process exit code."""
    import cv2  # here, so that the chain's own process, which starts the stages, does not load cv2 or torch
    import torch

    steps = {step.name: step for step in STEPS if step.module == module}
    if not steps:
        raise ValueError(f"bvr.stages.{module} has no row in bvr.run.STEPS")
    parser = argparse.ArgumentParser(prog=f"python -m bvr.stages.{module}")
    parser.add_argument("--source", type=Path, required=True, help="a video, or a folder of videos")
    parser.add_argument("--save-dir", type=Path, required=True, help="the run root: <save-dir>/<stem>/")
    parser.add_argument("--config", type=Path, help="a delta over configs/bvr.toml")
    parser.add_argument("--upstream", type=Path, help="a run root to read missing inputs from")
    parser.add_argument("--step", choices=sorted(steps), required=len(steps) > 1, default=next(iter(steps)))
    parser.add_argument("--clips", help="comma-separated clip names (default: every clip of the source)")
    parser.add_argument("--skip-existing", action="store_true", help="skip clips whose manifest still holds")
    options = parser.parse_args(arguments)
    step = steps[options.step]
    config = configuration.load(options.config)
    clips = paths.clips(options.source, options.save_dir, options.upstream)
    if options.clips is not None:
        names = options.clips.split(",")
        if unknown := sorted(set(names) - {clip.name for clip in clips}):
            parser.error(f"no clip named {', '.join(unknown)} at {options.source}")
        clips = [clip for clip in clips if clip.name in names]
    # A module started with ``python -m`` is ``__main__``: its own logger is outside ``bvr``.
    loggers = [logging.getLogger(name) for name in {"bvr", load.__module__} if name == "bvr" or name[:4] != "bvr."]
    for logger in loggers:
        _log_to_console(logger)
    code = manifest.code(options.save_dir)
    if options.skip_existing:
        clips = [clip for clip in clips if _stale(step, clip, config, code)]
    if not clips:
        log.info("%s: nothing to do", step.name)
        return 0

    stage_config = _config(step, config)
    settings = stage_config["run"]
    torch.set_num_threads(settings["threads"])
    # OpenCV keeps a pool of its own, one thread per core of the host unless capped.
    cv2.setNumThreads(settings["threads"])
    torch.manual_seed(settings["seed"])
    if settings["deterministic"]:
        # cuBLAS reads this when it creates its handle, so it is set before the first CUDA call.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    device = torch.device(settings["device"])
    generator = torch.Generator(device).manual_seed(settings["seed"])
    on_gpu = device.type == "cuda"
    environment = manifest.environment()
    command = ["python", "-m", f"bvr.stages.{module}", *arguments]
    process = load(Stage(step.name, stage_config, device, generator))
    failed = 0
    for clip in clips:
        inputs = _inputs(step, clip)
        _clear(step, clip)
        clip.log(step.name).parent.mkdir(parents=True, exist_ok=True)
        file_log = logging.FileHandler(clip.log(step.name), mode="w")
        for logger in loggers:
            logger.addHandler(file_log)
        log.info("%s: %s", step.name, clip.name)
        if on_gpu:
            torch.cuda.reset_peak_memory_stats(device)
        started, clock = datetime.now(UTC), time.monotonic()
        try:
            process(clip)
            outputs = {name: manifest.checksum(clip.path(name)) for name in step.outputs}
            if not any(outputs.values()):
                raise FileNotFoundError(f"{step.name} wrote none of {', '.join(step.outputs)}")
            error = None
        except Exception as raised:  # a clip that fails is data: it is recorded and the next one runs
            log.exception("%s: %s failed", step.name, clip.name)
            outputs, error, failed = {}, f"{type(raised).__name__}: {raised}", failed + 1
        manifest.write(clip.manifest(step.name), manifest.record(
            step=step.name, status="complete" if error is None else "failed", error=error,
            started=started.isoformat(timespec="seconds"), finished=datetime.now(UTC).isoformat(timespec="seconds"),
            seconds=round(time.monotonic() - clock, 3), command=command, config=stage_config, code=code,
            environment=environment, seed=settings["seed"],
            device=f"{device} ({torch.cuda.get_device_name(device)})" if on_gpu else str(device),
            peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if on_gpu else 0,
            inputs=inputs, outputs=outputs))
        for logger in loggers:
            logger.removeHandler(file_log)
        file_log.close()
    return 1 if failed else 0


def _steps(names: Sequence[str]) -> list[Step]:
    """The rows of ``names``, which must be steps of the chain in its order."""
    order = [step.name for step in STEPS]
    if unknown := [name for name in names if name not in order]:
        raise ValueError(f"unknown step(s) {', '.join(unknown)}; the chain is {', '.join(order)}")
    places = [order.index(name) for name in names]
    if any(first >= second for first, second in zip(places, places[1:])):
        raise ValueError(f"steps must follow the chain's order {', '.join(order)}; got {', '.join(names)}")
    return [STEPS[place] for place in places]


def _config(step: Step, config: dict) -> dict:
    """What ``step`` reads: the tables of its row, and ``[run]`` without the chain selection."""
    selected = configuration.select(config, step.sections)
    selected["run"] = {key: value for key, value in config["run"].items() if key != "steps"}
    return selected


def _inputs(step: Step, clip: Clip) -> dict:
    """Where each input of ``step`` is read for ``clip``, with its checksum (``None`` if absent)."""
    found = {name: clip.video if name == VIDEO else clip.find(name) for name in step.inputs}
    return {name: {"path": str(path), "checksum": manifest.checksum(path)} for name, path in found.items()}


def _stale(step: Step, clip: Clip, config: dict, code: dict) -> bool:
    reason = manifest.stale(manifest.read(clip.manifest(step.name)), config=_config(step, config), code=code,
                            inputs=_inputs(step, clip))
    log.info("%s: %s: %s", step.name, clip.name, f"runs ({reason})" if reason else "complete, skipped")
    return reason is not None


def _complete(step: Step, clip: Clip) -> bool:
    written = manifest.read(clip.manifest(step.name))
    return written is not None and written["status"] == "complete"


def _clear(step: Step, clip: Clip) -> None:
    """Move what a rerun of ``step`` replaces to the trash: outputs, stagings and the manifest."""
    label = f"{step.name}_{time.strftime('%Y%m%d-%H%M%S')}"
    trash, attempt = clip.trash(label), 1
    while trash.exists():  # a second rerun within the same second: nothing in the trash is overwritten
        attempt += 1
        trash = clip.trash(f"{label}_{attempt}")
    replaced = [clip.manifest(step.name)]
    for name in step.outputs:
        replaced += [clip.path(name), clip.path(name + ".staging")]
    for path in replaced:
        if path.exists():
            target = trash / path.relative_to(clip.directory)
            target.parent.mkdir(parents=True, exist_ok=True)
            path.rename(target)


def _log_to_console(logger: logging.Logger) -> None:
    """Send a logger to the console at INFO; a stage adds a DEBUG file per clip."""
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(console)
        logger.setLevel(logging.DEBUG)
