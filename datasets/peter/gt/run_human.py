"""Run the climb_wall_3 human ground-truth chain over a set of clips.

Steps, in order (any subset via ``--steps``; every step resumes):

    prepare  → per-clip working tree (undistorted upright videos, calibrated cameras,
               scene, gravity, sensor forces)          datasets/peter/gt/prepare_human.py
    sam3     → person masks, one manifest sweep per view (the tracker loads once)
    sapiens  → Sapiens2 Goliath-308 keypoints per view (the fit's 2D witness)
    sam3d    → SAM-3D-Body per view, with the CALIBRATED intrinsics
    smplx    → MHR -> SMPL-X per view
    sync     → each view's frame lag behind cam_left (epipolar test on the 2D joints) → cameras.npz
    cam3     → cam_3's pose re-solved from the climber (PnP on keypoints triangulated by the front pair)
    fit      → kinematic_1: the multi-view kinematic fit (pose, + shape while no frozen shape exists)
    shape    → one frozen shape per participant from the fitted clips, then the fit again with it
    contacts → contacts_1: board force labels on the body's contact frames
    kin2     → kinematic_2: the labelled frames seated on the wall (contact-patch repair)
    kindyn   → kindyn_1: pose + contact forces against the labels, compared with the boards
    overlay  → three-view overlay videos (kinematic_1 / kinematic_2 / kindyn_1)   overlay_human.py
               + cam_left force overlay with the force plots   datasets/peter/render_kindyn_overlays.py
    (all of these: datasets/peter/gt/fit_multiview.py)

The model stages are the standard stage CLIs run on each view's tree
(``views/<view>/``), so nothing about them is special. The GPU is chosen at start:
the first one with (almost) nothing running on it; if every GPU is busy the run waits.

    uv run --no-sync python datasets/peter/gt/run_human.py
    uv run --no-sync python datasets/peter/gt/run_human.py --clips a_00_01_part3 --steps fit,contacts,kindyn
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


from bvr.paths import REPO_ROOT
from datasets.peter.paths import PETER

HUMAN_ROOT = PETER / "climb_wall_3_gt/human"
VIEW_NAMES = ["cam_left", "cam_right", "cam_3"]
STEPS = ["prepare", "sam3", "sapiens", "sam3d", "smplx", "sync", "cam3", "fit", "shape", "contacts", "kin2",
         "kindyn", "overlay"]
DEFAULT_CLIPS = ["a_00_01_part3", "b_00_02_part2", "a_16_01_part2",
                 "b_16_01_part3", "a_32_01_part3", "b_32_02_part2"]
PIPELINE_CONFIG = REPO_ROOT / "pipeline_gt.toml"


def free_gpu(max_used_mib: int = 8000, max_util: int = 15, poll_s: int = 60) -> str:
    """Index of the first GPU with little memory and load in use; waits until one frees up."""
    while True:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
        for line in out.strip().splitlines():
            idx, used, util = (int(x) for x in line.split(","))
            if used <= max_used_mib and util <= max_util:
                return str(idx)
        print(f"[run_human] every GPU busy — waiting {poll_s}s", flush=True)
        time.sleep(poll_s)


def run(cmd: list[str], env: dict) -> int:
    print(f"\n=== {' '.join(str(c) for c in cmd[1:])} ===", flush=True)
    return subprocess.run([str(c) for c in cmd], env=env, cwd=REPO_ROOT).returncode


def view_manifest(clips: list[str], view: str, step: str) -> Path:
    """One manifest entry per clip for ``view``; the stage writes into ``views/<view>/``."""
    entries = []
    for clip in clips:
        view_dir = HUMAN_ROOT / clip / "views" / view
        entries.append({"source": str(view_dir / "video.mp4"), "out_dir": str(view_dir),
                        "bboxes": str(view_dir / "sam3" / "bboxes.npz"),
                        "camera": str(view_dir / "vggt" / "camera.npz")})
    tag = hashlib.sha1(",".join(clips).encode()).hexdigest()[:8]   # per clip selection: two runs never share one
    path = HUMAN_ROOT / "_manifests" / f"{step}_{view}_{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", default=",".join(DEFAULT_CLIPS))
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--gpu", default=None, help="GPU index (default: the first free one)")
    ap.add_argument("--views", default=",".join(VIEW_NAMES),
                    help="model stages only run on these views — split the views over free GPUs")
    args = ap.parse_args()
    clips = [c.strip() for c in args.clips.split(",") if c.strip()]
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    if any(v not in VIEW_NAMES for v in views):
        raise SystemExit(f"unknown view in {views}; known {VIEW_NAMES}")
    bad = [s for s in steps if s not in STEPS]
    if bad:
        raise SystemExit(f"unknown step(s) {bad}; known {STEPS}")

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu or free_gpu())
    print(f"[run_human] {len(clips)} clip(s), steps {steps}, GPU {env['CUDA_VISIBLE_DEVICES']}")
    py = sys.executable
    failed: list[str] = []
    if "prepare" in steps:
        rc = run([py, REPO_ROOT / "prepare_human.py", "--clips", ",".join(clips), "--skip-existing"], env)
        if rc != 0:
            raise SystemExit(f"[run_human] prepare failed (rc={rc}); nothing else runs")
    for step, module in (("sam3", "track_sam3"), ("sapiens", "estimate_sapiens"),
                         ("sam3d", "estimate_body_sam3d"),
                         ("smplx", "convert_mhr_to_smplx")):
        if step not in steps:
            continue
        for view in views:
            manifest = view_manifest(clips, view, step)
            started = time.monotonic()
            rc = run([py, "-m", f"bvr.stages.{module}", "--source", manifest, "--save-dir", manifest.parent,
                      "--config", PIPELINE_CONFIG], env)
            print(f"[run_human] {step} {view}: rc={rc} in {(time.monotonic() - started) / 60:.1f} min",
                  flush=True)
            if rc != 0:
                failed.append(f"{step} {view}")
    # The per-clip stages: a clip whose stage fails is dropped from the later stages, so a
    # stale body can never feed the next stage as if it were fresh.
    live = list(clips)

    def per_clip(stage: str, targets: list[str], drop: list[str] | None = None) -> None:
        """Run ``stage`` per clip; a failure drops the clip (or ``drop``) from the live list."""
        for clip in targets:
            started = time.monotonic()
            rc = run([py, REPO_ROOT / "fit_multiview.py", "--clip", clip, "--stage", stage], env)
            print(f"[run_human] {stage} {clip}: rc={rc} in {(time.monotonic() - started) / 60:.1f} min", flush=True)
            if rc != 0:
                failed.append(f"{stage} {clip}")
                for c in (drop if drop is not None else [clip]):
                    if c in live:
                        live.remove(c)

    for step in ("sync", "cam3", "fit"):
        if step in steps:
            per_clip(step, list(live))
    if "shape" in steps:
        # One shape per participant (the first letter of the clip name), then every live clip
        # of that participant is fitted again with the frozen shape.
        for who in sorted({c[0] for c in live}):
            mine = [c for c in live if c[0] == who]
            had_shape = (HUMAN_ROOT / "_shapes" / f"{who.upper()}.npz").exists()
            per_clip("shape", mine[:1], drop=mine)      # no shape → none of that participant's clips go on
            if mine[0] in live and not had_shape:       # a shape that already existed was used by the fit above
                per_clip("fit", mine)
    for step in ("contacts", "kin2", "kindyn"):
        if step in steps:
            per_clip(step, list(live))
    if "overlay" in steps:
        for clip in live:
            for body in ("kinematic_1", "kinematic_2", "kindyn_1"):
                if (HUMAN_ROOT / clip / "human_optim" / f"{body}.npz").exists():
                    rc = run([py, REPO_ROOT / "overlay_human.py", "--clip", clip, "--body", body], env)
                    if rc != 0:
                        failed.append(f"overlay {clip} {body}")
            if (HUMAN_ROOT / clip / "human_optim" / "kindyn_1.npz").exists():
                # the monocular chain's picture: cam_left with the solved + measured force arrows
                # and the force plots → human_optim/kindyn_forces_overlay.mp4
                rc = run([py, "-m", "datasets.peter.render_kindyn_overlays", HUMAN_ROOT / clip], env)
                if rc != 0:
                    failed.append(f"forces overlay {clip}")
    if failed:
        raise SystemExit(f"[run_human] FAILED: {failed}")
    print("[run_human] all steps finished")


if __name__ == "__main__":
    main()
