# BetterVideoReconstruction — Claude Code Guide

Monocular-video human reconstruction: a video of one person → their **motion
trajectory + per-contact-joint forces**, expressed on a shape-aware BetterRobot
SMPL body. Track the person → estimate per-frame camera intrinsics → fit a
per-frame MHR body → convert to SMPL → pack into a BetterRobot motion
trajectory → kinematic optimization → dynamics (contact-force) optimization.

This repo was split off from `../BetterHumanForce` the same way
`../BetterImageReconstruction` was: keep the video + motion/dynamics parts,
drop the image-only machinery (FLUX eraser, static scene/object alignment,
single-image refinement), and rewrite for cleanness on the way out.

## Status: small-batch, file-based (DB migration is later)

Results live on disk under `out/<stem>/` — one tree per input clip. There is
**no database yet**; the layout deliberately mirrors
`../BetterImageReconstruction`'s `scripts/stages/` so the eventual switch to a
SQLite/DB dataset is a small diff. Re-processing = re-running a stage (use
`--skip-existing` to skip clips whose outputs exist).

## The pipeline & `out/<stem>/` contract

Run in this order (the orchestrator does it for you; each stage is also a
standalone CLI). "ACTIVE" stages are implemented; "PLACEHOLDER" stages parse
args and raise `NotImplementedError`.

| # | Stage (`scripts/stages/…`) | Status | Reads | Writes (under `out/<stem>/`) |
|---|---|---|---|---|
| 1 | `track_sam3.py` | ACTIVE | video/frames | `sam3/<obj:02d>/frame_NNNNNN.png`, `sam3/mask_overlay.mp4`, `sam3/bboxes.npz` |
| 2 | `estimate_camera_vggt.py` | ACTIVE | frames | `vggt/camera.npz` (multi-view: `extrinsics (T,4,4)`, normalized + pixel `intrinsics`, records the VGGT inference resolution; ≤`max_frames`, default 500); `vggt/depth/…` with `--save-depth` |
| 3 | `estimate_camera_moge2.py` | ACTIVE | frames | `moge2/camera.npz` (`intrinsics (T,3,3)` normalized, `fov_x_deg`, …); `moge2/depth/…` with `--save-depth` |
| 4 | `estimate_sapiens.py` | ACTIVE | frames, `sam3/bboxes.npz` | `sapiens/pose.npz`, `sapiens/pointmap.npz`, `sapiens/overlay/…` |
| 5 | `fuse_geometry.py` | ACTIVE | `vggt/`, `moge2/depth`, `sam3/` | `geometry/transform.npz` (metric scale + original-res cameras), `geometry/scene.npz`, `geometry/human.npz` — see `docs/geometry_fusion_plan.md` |
| 6 | `estimate_body_sam3d.py` | ACTIVE | `sam3/bboxes.npz`, `sam3/`, `moge2/camera.npz`, frames | `body_params.npz` (single person, MHR params) |
| 7 | `convert_mhr_to_smpl.py` | ACTIVE | `body_params.npz` | `smpl_params.npz` (betas, body_pose, global_orient, transl, `q (N,99)`) |
| 8 | `build_robot_motion.py` | ACTIVE | `smpl_params.npz`, optional contacts | `results/0.npz` (Stage 0 motion — see `docs/motion_npz_schema.md`) |
| 9 | `optimize_motion.py` | ACTIVE | `results/0.npz`, `configs/kinematic.toml` | `results/1.npz` (kinematic) |
| 10 | `optimize_dynamics.py` | ACTIVE | `results/1.npz`, `configs/dynamics.toml` | `results/2.npz` (+ `contact_forces`, `joint_torques`) |
| – | `classify_gemma.py` | PLACEHOLDER | frames, `sam3/bboxes.npz`, `sam3/` | `gemma/result.json` (triage / contact graph) |

Viewers: `scripts/view_motion.py out/<stem>/results/2.npz` (viser playback with
contact-force arrows); `scripts/view_3d.py out/<stem>/` (viser: fused `geometry/`
scene cloud + camera frustums + per-climber human points);
`scripts/render_smpl_contact.py --source v.mp4 --predictions out/<stem>/`.

`vggt`, `moge2`, `sapiens`, and the `gemma` placeholder are **precompute_runner
Steps** (operate on `--source` / `--save-dir`); a *folder* of videos loads each
model once and sweeps the batch. The post-track stages take `--predictions
<clip-dir>`.

## Setup

See [`INSTALL.md`](INSTALL.md) for env / submodule / checkpoint setup.

## Run

```bash
# Whole chain on one clip (each model loads once for that clip).
uv run python scripts/pipeline.py --source video.mp4 --save-dir out/

# Batch of clips; skip clips already done.
uv run python scripts/pipeline.py --source clips/ --save-dir out/ --skip-existing

# A subset of stages.
uv run python scripts/pipeline.py --source video.mp4 --save-dir out/ --steps sam3,moge2

# Or run a stage standalone (full flag set lives on each stage CLI):
uv run python scripts/stages/track_sam3.py --source video.mp4 --save-dir out/ --text person
uv run python scripts/stages/estimate_camera_moge2.py --source video.mp4 --save-dir out/
uv run python scripts/stages/fuse_geometry.py --predictions out/video/
uv run python scripts/stages/estimate_body_sam3d.py --source video.mp4 --predictions out/video/
uv run python scripts/stages/convert_mhr_to_smpl.py --source video.mp4 --predictions out/video/
uv run python scripts/stages/build_robot_motion.py --predictions out/video/ --fps 30
uv run python scripts/stages/optimize_motion.py   out/video/ --config configs/kinematic.toml
uv run python scripts/stages/optimize_dynamics.py out/video/ --config configs/dynamics.toml
```

Stage knobs + step selection for the orchestrator live in
[`configs/pipeline.toml`](configs/pipeline.toml).

## Project layout

```
BetterVideoReconstruction/
├── CLAUDE.md / README.md / INSTALL.md
├── pyproject.toml / .gitmodules
├── docs/motion_npz_schema.md      # results/*.npz DFS-ordered schema (source of truth)
├── configs/
│   ├── pipeline.toml              # orchestrator: per-stage knobs + [steps]
│   ├── kinematic.toml             # optimize_motion (Stage 1)
│   └── dynamics.toml              # optimize_dynamics (Stage 2)
├── scripts/
│   ├── pipeline.py                # load-once subprocess orchestrator
│   ├── view_motion.py             # viser playback of results/*.npz
│   ├── view_3d.py                 # viser: geometry/ scene cloud + cameras + human points
│   ├── render_smpl_contact.py     # SMPL mesh/skeleton + contact overlay
│   ├── _pym_preload.py            # libstdc++ LD_PRELOAD shim for pymomentum
│   └── stages/                    # the per-stage CLIs in the table above
├── tools/                         # importable library (repo root added to sys.path by scripts)
│   ├── source.py                  # image/folder/video → frame paths
│   ├── precompute_runner.py       # Step protocol + run_batch + out/<stem>/ layout
│   ├── mask_io.py
│   ├── adapters/                  # stateless model wrappers
│   │   ├── moge2.py               # MoGe2 load + infer + intrinsics helpers
│   │   ├── sapiens2.py            # Sapiens2 pose/pointmap/normal
│   │   ├── gemma_vlm.py           # Gemma 4 VLM (backs the placeholder)
│   │   └── vggt_omega.py          # VGGT-Omega multi-view camera + depth
│   ├── mhr_to_smpl/               # MHR → better_human SMPL fitter
│   ├── geometry/                  # depth clean/align + backproject + voxel merge (fuse stage)
│   └── robot_motion/              # motion/dynamics core (+ data/ mass & inertia tables)
└── third_party/                   # submodules: sam3, sam-3d-body, sapiens2, MHR, moge, vggt-omega
```

## Conventions & deliberate choices

- **Single person.** SAM 3 can track several, but the body→motion chain fits
  one (`--obj-id`, default 0 = the primary tracked person). Multi-person is a
  future extension.
- **SMPL, not SMPL-X**, on the motion path. The kinematics/dynamics core
  (`tools/robot_motion`, `smpl_inertia`, the 24-joint tree) is built on SMPL;
  the MHR→SMPL-X converter is intentionally not ported.
- **Camera is its own stage.** `estimate_camera_moge2.py` writes per-frame
  intrinsics to `moge2/camera.npz`; `estimate_body_sam3d.py` reads `K` from
  there (falling back to a one-off MoGe2 frame-0 estimate if absent).
- **VGGT-Omega is additive, runs right after `sam3`.** It is a *multi-view*
  model (one forward over the whole clip), so it caps at `max_frames` (default
  500, uniformly subsampled) and runs at a reduced internal resolution. It writes
  its own `vggt/camera.npz` — cross-frame `extrinsics` + normalized `intrinsics`
  (same convention as `moge2/camera.npz`) **plus the VGGT inference resolution**
  so depth/point maps can be mapped back to full-res frames. It does *not* yet
  feed the body stage (that still reads MoGe2's `moge2/camera.npz`); the two
  camera stages coexist. **VGGT geometry is up-to-scale, not metric** (one
  arbitrary per-clip scale shared by depth + camera translations; the npz records
  `scale='up_to_scale'`) — metricized downstream by `fuse_geometry.py`.
- **`fuse_geometry.py` turns VGGT metric.** After `moge2`, it estimates one
  global scale (per-frame median depth ratio → inlier-weighted median across
  frames — robust to MoGe2's per-frame noise), denormalizes VGGT intrinsics to
  the original resolution (FOV preserved), and writes a metric **scene** cloud
  (globally voxel-merged) + per-frame **human** points (all tracked people,
  unmerged) under `geometry/`. Needs `vggt` + `moge2` run with `save_depth`.
- **The motion `.npz` is DFS-ordered and self-describing** — every consumer
  rebuilds the BetterRobot model straight from the arrays. See
  [`docs/motion_npz_schema.md`](docs/motion_npz_schema.md). (`build_robot_motion`
  writes `results/0.npz`, not the old `robot_motion.npz`.)
- **The SAM 3 tracker lives in its stage**, not in `tools/adapters/`. It is a
  *stateful video-tracking Step* (tracking state, video chunking, overlay/IO),
  not a stateless inference wrapper like `moge2`/`sapiens2`/`gemma_vlm`.
- **MoGe and VGGT-Omega are editable submodules** (`third_party/`), not git-pip
  deps — folds them into `uv sync` and pins them.
- **Stages tolerate per-item failure**: precompute Steps print the error and
  keep going (a corrupt frame shouldn't kill a batch).

## What this guide intentionally does NOT cover

- Per-stage algorithmic detail — see each stage's module docstring.
- `BetterRobot` internals (kinematics, RNEA, residuals, viewer) — see
  `../BetterRobot/CLAUDE.md`. The optimizers and `tools/robot_motion` depend on
  it heavily.

## See also

- [`docs/motion_npz_schema.md`](docs/motion_npz_schema.md) — `results/*.npz` schema.
- [`docs/geometry_fusion_plan.md`](docs/geometry_fusion_plan.md) — `fuse_geometry` design + `geometry/*.npz` contract.
- [`README.md`](README.md) — user-facing overview.
- [`INSTALL.md`](INSTALL.md) — env, submodules, model assets.
