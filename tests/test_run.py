"""The glue of a run: the stage table, the loop over clips, manifests, resume, the trash, upstream."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bvr import config, files, manifest, paths, run
from bvr.paths import Clip

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Rows of the stage table whose module the rebuild has not written yet; a row leaves with its module.
PENDING: set[str] = set()


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """A folder of two clips. No stage here decodes them, so a clip is a few bytes."""
    folder = tmp_path / "videos"
    folder.mkdir()
    for name in ("one", "two"):
        (folder / f"{name}.mp4").write_bytes(name.encode())
    return folder


@pytest.fixture
def arguments(tmp_path: Path, source: Path) -> list[str]:
    delta = tmp_path / "cpu.toml"
    delta.write_text('base = "configs/bvr.toml"\n[run]\ndevice = "cpu"\nthreads = 1\n')
    return ["--source", str(source), "--save-dir", str(tmp_path / "out"), "--config", str(delta)]


def gravity(calls: list[str], fail: str | None = None):
    """A stand-in for the gravity stage: it writes ``geocalib/gravity.npz`` and counts its calls."""
    def load(stage: run.Stage):
        assert stage.step == "gravity" and set(stage.config) == {"gravity", "run"}

        def process(clip: Clip) -> None:
            calls.append(clip.name)
            if clip.name == fail:
                raise RuntimeError("no floor")
            files.save_npz(clip.path(paths.GRAVITY), "bvr-gravity", gravity_world=np.array([0.0, 1.0, 0.0]))
        return process
    return load


def test_every_step_has_its_module() -> None:
    assert [step.name for step in run.STEPS][:8] == [
        "sam3", "vggt", "sapiens", "sam3d", "smplx", "fuse", "scale", "gravity"]
    for step in run.STEPS:
        written = (REPO_ROOT / "bvr" / "stages" / f"{step.module}.py").exists()
        assert written != (step.module in PENDING), step.module


def test_a_stage_writes_a_manifest_with_every_field(tmp_path: Path, arguments: list[str]) -> None:
    calls: list[str] = []
    assert run.stage("estimate_gravity", gravity(calls), arguments) == 0
    assert calls == ["one", "two"]
    written = json.loads((tmp_path / "out" / "one" / "manifests" / "gravity.json").read_text())
    assert list(written) == [
        "format", "format_version", "step", "status", "error", "started", "finished", "seconds", "command",
        "config", "config_sha256", "code", "environment", "device", "seed", "peak_gpu_bytes", "inputs", "outputs"]
    assert (written["format"], written["step"], written["status"]) == ("bvr-stage", "gravity", "complete")
    assert written["config"]["gravity"]["num_frames"] == 48 and written["config"]["run"]["device"] == "cpu"
    assert set(written["code"]) == {"commit", "dirty", "diff_sha256", "diff"}
    assert set(written["environment"]) == {"python", "torch", "cuda", "uv_lock_sha256", "libraries", "submodules"}
    assert set(written["environment"]["libraries"]) == {"better-robot", "better-human"}
    assert written["inputs"]["video"]["checksum"].startswith("sha256:")
    assert written["inputs"]["geometry/transform.npz"]["checksum"] is None
    assert written["outputs"]["geocalib"].startswith("listing:")
    assert (tmp_path / "out" / "one" / "logs" / "gravity.log").exists()
    assert files.load_npz(tmp_path / "out" / "one" / paths.GRAVITY, "bvr-gravity")["gravity_world"].shape == (3,)


def test_resume_trusts_the_manifest_and_a_rerun_moves_the_old_output_to_the_trash(
        tmp_path: Path, arguments: list[str], source: Path) -> None:
    calls: list[str] = []
    run.stage("estimate_gravity", gravity(calls), arguments)
    assert run.stage("estimate_gravity", gravity(calls), [*arguments, "--skip-existing"]) == 0
    assert calls == ["one", "two"]                       # both manifests still hold: nothing ran

    (source / "two.mp4").write_bytes(b"another video")   # an input changed: that clip alone reruns
    run.stage("estimate_gravity", gravity(calls), [*arguments, "--skip-existing"])
    assert calls == ["one", "two", "two"]
    trashed = list((tmp_path / "out" / "_trash" / "two").glob("gravity_*"))
    assert len(trashed) == 1
    assert (trashed[0] / paths.GRAVITY).exists() and (trashed[0] / "manifests" / "gravity.json").exists()
    assert not (tmp_path / "out" / "_trash" / "one").exists()

    changed = tmp_path / "changed.toml"                  # a key the step reads changed: both rerun
    changed.write_text(f'base = "{tmp_path / "cpu.toml"}"\n[gravity]\nnum_frames = 12\n')
    run.stage("estimate_gravity", gravity(calls), [*arguments[:4], "--config", str(changed), "--skip-existing"])
    assert calls[3:] == ["one", "two"]
    assert len(list((tmp_path / "out" / "_trash" / "two").glob("gravity_*"))) == 2   # the first copy is kept


def test_a_clip_that_fails_is_recorded_and_the_others_run(tmp_path: Path, arguments: list[str]) -> None:
    calls: list[str] = []
    assert run.stage("estimate_gravity", gravity(calls, fail="one"), arguments) == 1
    assert calls == ["one", "two"]
    failed = manifest.read(tmp_path / "out" / "one" / "manifests" / "gravity.json")
    assert (failed["status"], failed["error"]) == ("failed", "RuntimeError: no floor")
    assert manifest.read(tmp_path / "out" / "two" / "manifests" / "gravity.json")["status"] == "complete"
    assert manifest.stale(failed, config=failed["config"], code=failed["code"], inputs=failed["inputs"]) == "status is failed"


def test_a_clip_reads_from_upstream_what_its_run_does_not_have(tmp_path: Path) -> None:
    clip = Clip("one", tmp_path / "one.mp4", tmp_path / "out" / "one", tmp_path / "up" / "one")
    assert clip.find(paths.SCENE) == tmp_path / "up" / "one" / paths.SCENE
    files.save_npz(clip.path(paths.SCENE), "bvr-scene", points=np.zeros((1, 3)))
    assert clip.find(paths.SCENE) == clip.path(paths.SCENE) == tmp_path / "out" / "one" / paths.SCENE


def test_a_dry_run_names_every_step_and_touches_nothing(tmp_path: Path, source: Path, caplog) -> None:
    caplog.set_level("INFO", logger="bvr")
    assert run.chain(source, tmp_path / "out", dry_run=True) == 0
    commands = [record.getMessage() for record in caplog.records if " -m bvr.stages." in record.getMessage()]
    assert [command.split(" --step ")[1].split()[0] for command in commands] == config.load()["run"]["steps"]
    assert all(command.endswith("--clips=one,two") for command in commands)
    assert not (tmp_path / "out").exists()


def test_a_clip_whose_name_starts_with_a_dash_reaches_its_stage(tmp_path: Path, source: Path, arguments: list[str],
                                                                caplog) -> None:
    (source / "-zq.mp4").write_bytes(b"dash")
    caplog.set_level("INFO", logger="bvr")
    run.chain(source, tmp_path / "out", steps=["gravity"], dry_run=True)
    command = next(record.getMessage() for record in caplog.records if " -m bvr.stages." in record.getMessage())
    calls: list[str] = []
    # The stage reads the chain's own token for the clips, after the fixture's source, root and config.
    assert run.stage("estimate_gravity", gravity(calls), [*arguments, command.split()[-1]]) == 0
    assert calls == ["-zq", "one", "two"]


def test_steps_follow_the_chain(tmp_path: Path, source: Path) -> None:
    with pytest.raises(ValueError, match="unknown step"):
        run.chain(source, tmp_path / "out", steps=["sam4"], dry_run=True)
    with pytest.raises(ValueError, match="chain's order"):
        run.chain(source, tmp_path / "out", steps=["vggt", "sam3"], dry_run=True)


def test_an_old_file_reads_with_the_new_frame_name(tmp_path: Path) -> None:
    np.savez(tmp_path / "old.npz", contact_frame_names=np.array(["head", "left_palm"]), joint_names=np.array(["head"]))
    old = files.load_npz(tmp_path / "old.npz", "bvr-contacts")
    assert list(old["contact_frame_names"]) == ["head_top", "left_palm"] and list(old["joint_names"]) == ["head"]
    files.save_npz(tmp_path / "new.npz", "bvr-contacts", contact_frame_names=np.array(["head_top"]))
    with pytest.raises(ValueError, match="expected bvr-kindyn"):
        files.load_npz(tmp_path / "new.npz", "bvr-kindyn")
