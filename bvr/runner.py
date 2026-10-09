"""Stage commands over one clip root, and the variant farms the chain runs through.

A clip root is ``out/<stem>/`` — the per-clip output tree every stage reads and writes
(the table in CLAUDE.md). This module is the ONE place a stage's command line is
assembled: :func:`stage_command` turns a chain step into the argv of its
``python -m bvr.stages.<name>`` call over a clip root, and :func:`sweep_command` does the
same for the five model stages, which run once over a JSON manifest of clips (the model
loads once and sweeps them all). ``scripts/pipeline.py`` and
``datasets/climbingvideos/corpus.py`` both call these; nothing else builds a stage argv.

The chain, in order::

    sam3 → vggt → sapiens → sam3d → smplx → fuse → scale [→ gravity]
         → kinematics_1 → contacts_1 → kinematics_2 → contacts_2 → kindyn_1

``--variant <name>`` re-runs the human chain into ``<clip>/human_optim_<name>/`` through a
symlink farm ``<clip>/_<name>/`` whose ``human_optim`` IS the variant directory and whose
inputs link back to the real tree, so the stock stages run unchanged on it
(:func:`build_farm`); :func:`build_view_farm` gives the farms clip-named aliases for the
viewer.
"""
from __future__ import annotations

import sys
from pathlib import Path

from bvr.config import load_config

#: manifest step → (stage module under bvr.stages, the output that means a clip is done).
#: These five sweep every clip in one process; the rest run per clip.
MANIFEST_STEPS: dict[str, tuple[str, str]] = {
    "sam3":    ("track_sam3",           "sam3/bboxes.npz"),
    "vggt":    ("estimate_camera_vggt", "vggt/camera.npz"),
    "sapiens": ("estimate_sapiens",     "sapiens/pose.npz"),
    "sam3d":   ("estimate_body_sam3d",  "sam3d/params.npz"),
    "smplx":   ("convert_mhr_to_smplx", "sam3d/smplx_params.npz"),
}
#: per-clip step → the file that means it is done.
CLIP_OUTPUT: dict[str, str] = {
    "fuse":         "geometry/transform.npz",
    "scale":        "geometry/scale.npz",
    "gravity":      "geocalib/gravity.npz",
    "kinematics_1": "human_optim/kinematic_1.npz",
    "contacts_1":   "human_optim/contacts_1.npz",
    "kinematics_2": "human_optim/kinematic_2.npz",
    "contacts_2":   "human_optim/contacts_2.npz",
    "kindyn_1":     "human_optim/kindyn_1.npz",
    "overlay":      "human_optim/kindyn_forces_overlay.mp4",
}
#: Synthetic-support-scene steps → the body their plateaus are read off (None = floor only).
#: Both write the same ``geometry/scene.npz``, so an existence check cannot tell them apart:
#: they are never resume-skipped, which costs seconds and is safe because they are idempotent.
SUPPORT_STEPS: dict[str, str | None] = {
    "support_scene_floor":    None,
    "support_scene_plateaus": "kinematic_1",
}
ALL_STEPS = [*MANIFEST_STEPS, *CLIP_OUTPUT, *SUPPORT_STEPS]
#: The plain variant chain. kinematics_1 is recomputed too — a variant that changes the
#: contact-frame allowlist changes its auto-contacts.
VARIANT_STEPS = ["kinematics_1", "contacts_1", "kinematics_2", "contacts_2", "kindyn_1"]
#: Steps that write into ``geometry/`` (or ``geocalib/``) of whatever tree they are pointed
#: at. A variant farm links both back to the real clip, so under ``--variant`` these would
#: rewrite the shared inputs every run depends on.
GEOMETRY_STEPS = ("fuse", "scale", "gravity")
FARM_LINKS = ("sam3", "vggt", "sapiens", "sam3d", "geometry", "geocalib")
#: Contents of ``geometry/`` a synthetic-scene farm links back to the real tree; its
#: ``scene.npz`` is the one file the variant owns. ``prefit.npz`` is optional (only clips
#: whose scale stage pre-fit the body have one).
GEOMETRY_LINKS = ("transform.npz", "scale.npz", "human.npz", "prefit.npz")

#: Human-optimization steps → the ``--stages`` they dispatch; ``kinematics_1`` dispatches the
#: human config's ``[stages].run`` (``root,kinematics_1`` by default: the two share one call).
_HUMAN_STAGES = {"kinematics_1": None, "kinematics_2": "kinematics_2", "kindyn_1": "kindyn_1"}
_CONTACT_BODIES = {"contacts_1": "kinematic_1", "contacts_2": "kinematic_2"}


def sweep_command(step: str, manifest: Path, out_root: Path, pipeline_config: Path, *,
                  skip_existing: bool = False) -> list[str]:
    """Argv of one model stage over a JSON manifest of clips (the model loads once).

    :param skip_existing: let the stage skip the clips it has finished (its own knob, forced on).
    """
    module, _ = MANIFEST_STEPS[step]
    return [sys.executable, "-m", f"bvr.stages.{module}", "--source", str(manifest),
            "--save-dir", str(out_root), "--config", str(pipeline_config),
            *(["--skip-existing"] if skip_existing else [])]


def step_done(step: str, clip_root: Path) -> bool:
    """Is this per-clip step's output complete under ``clip_root``?

    Existence of :data:`CLIP_OUTPUT`, except for ``scale``, which the stage itself judges
    more strictly: the world must be metric (a re-fused ``transform.npz`` resets the flag, and
    a stale ``scale.npz`` must not hide that) and a pre-fit scale needs its ``prefit.npz``.
    """
    output = CLIP_OUTPUT.get(step)
    if output is None or not (clip_root / output).exists():
        return False
    if step != "scale":
        return True
    import numpy as np

    transform_path = clip_root / "geometry" / "transform.npz"
    if not transform_path.exists():
        return False
    with np.load(transform_path) as transform:
        if not ("metric" in transform.files and bool(transform["metric"])):
            return False
    with np.load(clip_root / output) as scale:
        if "method" in scale.files and str(scale["method"]) == "prefit-dual":
            return (clip_root / "geometry" / "prefit.npz").exists()
    return True


def stage_command(step: str, clip_root: Path, *, videos: list[Path], pipeline_config: Path,
                  human_config: Path, variant: str | None = None, measured: bool = False,
                  overlay_module: str | None = None) -> list[str]:
    """Argv of one per-clip step over ``clip_root`` (``out/<stem>/`` or a variant farm).

    :param videos: one video per camera view, in the ``camera.npz`` view order; only fuse
        takes more than the first.
    :param measured: measure the contacts on the rig's force sensors instead of estimating them.
    :param overlay_module: the ``overlay`` step's renderer (a ``python -m`` module name, from
        the variant's recipe), run on the real run directory, not the farm.
    """
    common = ["--source", str(videos[0]), "--predictions", str(clip_root)]
    if step in SUPPORT_STEPS:
        bodies = SUPPORT_STEPS[step]
        return [sys.executable, "-m", "bvr.stages.build_support_scene",
                "--predictions", str(clip_root), "--config", str(human_config),
                *(["--bodies", bodies] if bodies else [])]
    if step == "overlay":
        if overlay_module is None:
            raise ValueError("the overlay step needs the variant's `overlay` module")
        run_dir = clip_root.parent / f"human_optim_{variant}" if variant else clip_root / "human_optim"
        return [sys.executable, "-m", overlay_module, str(run_dir)]
    if step == "fuse":
        return [sys.executable, "-m", "bvr.stages.fuse_geometry",
                "--source", ",".join(str(v) for v in videos),
                "--predictions", str(clip_root), "--config", str(pipeline_config)]
    if step == "scale":
        return [sys.executable, "-m", "bvr.stages.estimate_scale",
                "--predictions", str(clip_root), "--config", str(pipeline_config)]
    if step == "gravity":
        return [sys.executable, "-m", "bvr.stages.estimate_gravity_geocalib", *common,
                "--config", str(pipeline_config)]
    if step in _CONTACT_BODIES:
        bodies = _CONTACT_BODIES[step]
        if measured:
            return [sys.executable, "-m", "bvr.stages.measure_contacts",
                    "--predictions", str(clip_root), "--config", str(human_config),
                    "--bodies", bodies]
        return [sys.executable, "-m", "bvr.stages.estimate_contacts", *common,
                "--bodies", bodies, "--config", str(human_config)]
    if step in _HUMAN_STAGES:
        stages = _HUMAN_STAGES[step] or ",".join(
            load_config(human_config).get("stages", {}).get("run", ("root", "kinematics_1")))
        return [sys.executable, "-m", "bvr.stages.human_optimization", *common,
                "--stages", stages, "--config", str(human_config)]
    raise ValueError(f"unknown per-clip step {step!r}; known: {ALL_STEPS}")


def _farm_link(link: Path, target: Path, *, dry_run: bool = False) -> None:
    """Point ``link`` at the relative ``target``; a link already pointing there is kept.

    Anything else in the way is refused, never re-pointed: a farm laid out by other code
    (e.g. a variant linking another variant's geometry) would otherwise be rewired under
    the results already computed through it.
    """
    if link.is_symlink() and link.resolve() == (link.parent / target).resolve():
        return
    if link.is_symlink() or link.exists():
        current = link.readlink() if link.is_symlink() else "a real file or directory"
        raise SystemExit(f"[pipeline] {link} already points to {current}, not {target}; "
                         "move it aside to rebuild this farm")
    if not dry_run:
        link.symlink_to(target)


def build_farm(clip_dir: Path, variant: str, *, synthetic_scene: bool = False,
               inherit: tuple[str, ...] = (), dry_run: bool = False) -> Path:
    """Create (or refresh) the clip's symlink farm; return the path to point the stages at.

    :param synthetic_scene: give the farm a real ``geometry/`` directory — the variant's
        own ``scene.npz`` beside symlinks to the real tree's other geometry artifacts —
        instead of linking the whole directory.
    :param inherit: files of the normal run's ``human_optim/`` linked into the variant
        directory instead of being recomputed.
    :param dry_run: only check that the existing links agree; write nothing.
    """
    out_dir, farm = clip_dir / f"human_optim_{variant}", clip_dir / f"_{variant}"
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        farm.mkdir(exist_ok=True)
    for name in inherit:
        _farm_link(out_dir / name, Path("..") / "human_optim" / name, dry_run=dry_run)
    for name in (*FARM_LINKS, "human_optim"):
        if synthetic_scene and name == "geometry":
            continue
        _farm_link(farm / name, Path("..") / (out_dir.name if name == "human_optim" else name),
                   dry_run=dry_run)
    if synthetic_scene:
        geometry = farm / "geometry"
        if geometry.is_symlink():
            raise SystemExit(f"[pipeline] {geometry} is a link to {geometry.readlink()}, but "
                             f"--variant {variant} owns its geometry/; move it aside to rebuild")
        if not dry_run:
            geometry.mkdir(exist_ok=True)
        for name in GEOMETRY_LINKS:
            if not (clip_dir / "geometry" / name).exists():
                continue
            _farm_link(geometry / name, Path("..") / ".." / "geometry" / name, dry_run=dry_run)
    return farm


def build_view_farm(out_root: Path, variant: str, *, dry_run: bool = False) -> Path:
    """``out/_view_<variant>/<clip>`` → each clip's ``_<variant>`` farm, for the viewer.

    ``scripts/view_3d.py`` wants a root of clip-named directories and finds each clip's
    footage through ``_batch_manifest.jsonl`` beside them, so the farms (all named
    ``_<variant>``) get a clip-named alias here and the manifest is linked in.
    ``dry_run`` prints each link instead of writing it.
    """
    view = out_root / f"_view_{variant}"
    if not dry_run:
        view.mkdir(exist_ok=True)
    links = [(view / farm.parent.name, Path("..") / farm.parent.name / farm.name)
             for farm in sorted(out_root.glob(f"*/_{variant}"))]
    for link, target in [*links, (view / "_batch_manifest.jsonl", Path("..") / "_batch_manifest.jsonl")]:
        if dry_run:
            print(f"[pipeline] would link {link} -> {target}")
        _farm_link(link, target, dry_run=dry_run)
    return view
