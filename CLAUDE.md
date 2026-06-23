# BetterVideoReconstruction — Claude Code Guide

Monocular-video human reconstruction: a video → a per-person **SMPL-X body
trajectory** for everyone tracked, plus — for one selected person — their
**motion trajectory + per-contact-joint forces** on a shape-aware BetterRobot
SMPL body. Track the people → VGGT + MoGe2 cameras → fuse to metric geometry →
Sapiens keypoints → fit a per-frame MHR body for **every tracked person**
(SAM-3D-Body, using the VGGT camera) → convert each to **SMPL-X** →
**visually-guided kinematic refinement** (2D reprojection + depth chamfer + pose
prior + acceleration/jerk smoothness). The default chain ends at the refined
SMPL-X bodies; the SMPL → BetterRobot motion trajectory → kinematic → dynamics
(contact-force) path is **SMPL-only, single-person** and run standalone.

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
standalone CLI). The **default chain ends at `optimize_smplx`** (rows 1–8); the
`smpl → motion → optimize_*` rows are the SMPL-only path, off the default chain
(run via `--steps` or standalone). "ACTIVE" = implemented; "PLACEHOLDER" parses
args and raises `NotImplementedError`.

| # | Stage (`scripts/stages/…`) | Status | Reads | Writes (under `out/<stem>/`) |
|---|---|---|---|---|
| 1 | `track_sam3.py` | ACTIVE | video/frames | `sam3/<obj:02d>/frame_NNNNNN.png`, `sam3/mask_overlay.mp4`, `sam3/bboxes.npz` |
| 2 | `estimate_camera_vggt.py` | ACTIVE | frames | `vggt/camera.npz` (multi-view: `extrinsics (T,4,4)`, normalized + pixel `intrinsics`, records the VGGT inference resolution; ≤`max_frames`, default 500); `vggt/depth/…` with `--save-depth` |
| 3 | `estimate_camera_moge2.py` | ACTIVE | frames | `moge2/camera.npz` (`intrinsics (T,3,3)` normalized, `fov_x_deg`, …); `moge2/depth/…` with `--save-depth` |
| 4 | `fuse_geometry.py` | ACTIVE | `vggt/`, `moge2/depth`, `sam3/` | `geometry/transform.npz` (metric scale + original-res VGGT cameras), `geometry/scene.npz`, `geometry/human.npz` — see `docs/geometry_fusion_plan.md` |
| 5 | `estimate_sapiens.py` | ACTIVE | frames, `sam3/bboxes.npz` | `sapiens/pose.npz`, `sapiens/overlay/…` (keypoints only by default; `pointmap`/`normal` opt-in) |
| 6 | `estimate_body_sam3d.py` | ACTIVE | `sam3/bboxes.npz`, `sam3/`, `geometry/transform.npz` (VGGT camera @ orig res), frames | `sam3d/params.npz` (**all tracked people**, MHR params stacked `(P,N,…)` + `object_ids (P,)`), `sam3d/overlay.mp4` (combined — every person, one colour each) |
| 7 | `convert_mhr_to_smplx.py` | ACTIVE | `sam3d/params.npz` | `sam3d/smplx_params.npz` (per-person stacked `(P,N,…)`: betas, body/hand pose, global_orient, transl, jaw/eye/expr), `sam3d/smplx_overlay.mp4` (combined) |
| 8 | `optimize_smplx.py` | ACTIVE | `sam3d/smplx_params.npz`, `sapiens/pose.npz`, `geometry/human.npz`, `geometry/transform.npz` | `sam3d/smplx_kinematic.npz` (**all tracked people**, refined **world-frame** params stacked `(P,N,…)` + `q (P,N,211)` + `scale (P,)`, 52-joint `SMPLXMid`, `frame='world'`), `sam3d/smplx_kinematic_overlay.mp4` (`--overlay`) |
| – | `convert_mhr_to_smpl.py` | ACTIVE (standalone) | `sam3d/params.npz` (person 0, via `--person`) | `smpl_params.npz` (betas, body_pose, global_orient, transl, `q (N,99)`) |
| – | `build_robot_motion.py` | ACTIVE (standalone) | `smpl_params.npz`, optional contacts | `results/0.npz` (Stage 0 motion — see `docs/motion_npz_schema.md`) |
| – | `optimize_motion.py` | ACTIVE (standalone) | `results/0.npz`, `configs/kinematic.toml` | `results/1.npz` (kinematic) |
| – | `optimize_dynamics.py` | ACTIVE (standalone) | `results/1.npz`, `configs/dynamics.toml` | `results/2.npz` (+ `contact_forces`, `joint_torques`) |
| – | `classify_gemma.py` | PLACEHOLDER | frames, `sam3/bboxes.npz`, `sam3/` | `gemma/result.json` (triage / contact graph) |

Viewers: `scripts/view_motion.py out/<stem>/results/2.npz` (viser playback with
contact-force arrows); `scripts/view_3d.py out/<stem>/` (viser: fused `geometry/`
scene cloud + camera frustums + per-climber human points + the refined SMPL-X
bodies — prefers `sam3d/smplx_kinematic.npz`, falls back to `smplx_params.npz`);
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
uv run python scripts/stages/estimate_camera_vggt.py --source video.mp4 --save-dir out/ --save-depth
uv run python scripts/stages/estimate_camera_moge2.py --source video.mp4 --save-dir out/ --save-depth
uv run python scripts/stages/fuse_geometry.py --source video.mp4 --predictions out/video/
uv run python scripts/stages/estimate_sapiens.py --source video.mp4 --save-dir out/ --tasks pose
uv run python scripts/stages/estimate_body_sam3d.py --source video.mp4 --predictions out/video/
uv run python scripts/stages/convert_mhr_to_smplx.py --source video.mp4 --predictions out/video/
uv run python scripts/stages/optimize_smplx.py --source video.mp4 --predictions out/video/ --overlay

# SMPL → motion/dynamics path (SMPL-only; off the default chain):
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
│   ├── kinematic.toml             # optimize_motion (SMPL Stage 1)
│   ├── dynamics.toml              # optimize_dynamics (SMPL Stage 2)
│   └── smplx_kinematic.toml       # optimize_smplx (visually-guided SMPL-X kinematics)
├── scripts/
│   ├── pipeline.py                # load-once subprocess orchestrator
│   ├── view_motion.py             # viser playback of results/*.npz
│   ├── view_3d.py                 # viser: geometry/ scene cloud + cameras + human points
│   ├── render_smpl_contact.py     # SMPL mesh/skeleton + contact overlay
│   ├── diagnose_smplx.py          # per-frame plots of every optimize_smplx loss term
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
│   ├── mhr_to_smpl/               # MHR → better_human SMPL fitter (motion path)
│   ├── mhr_to_smplx/              # MHR → SMPL-X fitter (pip smplx; default body output)
│   ├── smplx_robot/               # SMPLXMid → BetterRobot 52-joint model + q remap (kinematic)
│   ├── smplx_optim/               # visually-guided SMPL-X losses (reproject/chamfer/pose/smoothness)
│   ├── geometry/                  # depth clean/align + backproject + voxel merge (fuse stage)
│   └── robot_motion/              # motion/dynamics core (+ data/ mass & inertia tables)
└── third_party/                   # submodules: sam3, sam-3d-body, sapiens2, MHR, moge, vggt-omega
```

## Conventions & deliberate choices

- **Body + SMPL-X are multi-person; the motion path is single-person.**
  `estimate_body_sam3d.py` and `convert_mhr_to_smplx.py` fit **every** person
  SAM 3 tracked (`--obj-ids`, default = all) and stack the results on a leading
  `(P, N, …)` axis (with `object_ids (P,)`), rendering one combined overlay with
  a colour per person. The SMPL → motion/dynamics chain is still single-person
  (the kinematics/dynamics core assumes one body): `convert_mhr_to_smpl.py`
  selects one person from the stack (`--person`, default 0 = the primary).
- **Default body is SMPL-X; the motion path is SMPL.** The default chain ends at
  `convert_mhr_to_smplx.py` (pip `smplx`, loading `better_human/models/smplx/`) —
  SMPL-X captures the hands MHR predicts (jaw/eye/face left at zero). Its outputs
  live **inside `sam3d/`** next to the body params (`sam3d/smplx_params.npz` +
  `sam3d/smplx_overlay.mp4`), not at the clip root. The kinematics/dynamics core
  (`tools/robot_motion`, `smpl_inertia`, the 24-joint tree) is built on **SMPL**,
  so the motion path keeps `convert_mhr_to_smpl.py` (with the BetterRobot
  `q (N,99)` packing) and runs standalone. SMPL-X now has a BetterRobot
  **kinematic** model too — `tools/smplx_robot` builds a 52-joint `SMPLXMid`
  model used by `optimize_smplx` for Lie-exact smoothness, and the stage writes
  `q (P,N,211)` — but no dynamics chain yet (dynamics stays SMPL-only).
- **The SMPL-X kinematic stage is visually-guided, multi-person, world-frame.**
  `optimize_smplx.py` refines each tracked person's SMPL-X trajectory against the
  image + geometry: **2D reprojection** to Sapiens keypoints, **chamfer** of
  visible posed vertices to the metric VGGT human depth (confidence-gated,
  back-face-culled), a **joints-only pose prior**, and **acceleration + jerk**
  smoothness (exact Lie `model.difference`, separate root/joint weights). The
  variable is `q (T,211)` for `better_human`'s `SMPLXMid` (pelvis + 21 body + 30
  hands; jaw/eyes dropped) **plus one per-person global scale** that scales the
  body uniformly about the pelvis (skeleton + mesh), optimised directly as the
  autograd leaf (quaternions renormalised in the forward — no exp-retraction
  singularity) with **Adam** in **two coarse-to-fine stages** (stage 1 places the
  body rigidly — root translation + scale, pose frozen; stage 2 unfreezes
  orientation + joints), robust to the heterogeneous gradient scales that stall
  L-BFGS here. All posed-mesh inference lives in `better_human`
  (`SMPLXMid.forward`); BVR only adds the BetterRobot model (`tools/smplx_robot`)
  + the visual losses (`tools/smplx_optim`). Output is **world-frame**
  (`frame='world'`) in `sam3d/smplx_kinematic.npz` (with a `scale (P,)` field that
  consumers — overlay, `view_3d` — re-apply about the pelvis); weights in
  `configs/smplx_kinematic.toml`.
  - **Keypoint correspondence is Goliath-308, not COCO-wholebody.** Sapiens here
    emits the `goliath_308` scheme (`pose.npz` `keypoint_scheme`), whose body block
    has **no wrists** and so diverges from COCO-wholebody at index 9 (hips 9/10,
    knees 11/12, ankles 13/14, feet 15–20; wrists live in the hand blocks at
    41/62). `tools/smplx_optim/keypoints.py` maps SMPL-X-mid joints/landmarks to
    these indices (verified against `third_party/sapiens2/.../keypoints308.py`).
    Getting this right is what makes reprojection a reliable signal — init
    reprojection is ~5px, i.e. the converter's per-frame pose is already good.
  - **Keeping the motion natural — multi-stage + trust the image.** Single-stage
    Adam fell into a local minimum where the joints *deform* to chase the depth
    cloud (off-image **and** contorted), so the optimiser is **coarse-to-fine**:
    stage 1 places the body **rigidly** (translation + scale, pose frozen at the
    converter's good per-frame pose — frozen by zeroing the inactive `q.grad`
    groups, fresh Adam per stage); stage 2 unfreezes orientation + joints and
    refines locally. The metric human cloud is **wholesale-wrong on some frames**
    (per-frame depth scale swings ±30% / mask leakage → the image-consistent depth
    is 1–3 m off the cloud), and good-vs-bad-frame depth error differs ~10×, so no
    single chamfer weight wins both: **chamfer is deliberately modest** and the
    image (reprojecting at ~5px) is the trusted signal — the body sits on the cloud
    where depth is good and follows the image where it isn't (absolute depth there
    then follows smoothness, the honest monocular answer). Other knobs: a **strong
    joints-only pose prior**; a **gentle root-orientation prior** (`orient_prior`)
    that damps the orientation runaway the finite-difference smoothness leaves
    under-constrained at the clip boundary — root **translation** stays free for
    depth; a **truncated reprojection** (`robust_px`) outlier guard; and the
    **global scale**. `scripts/diagnose_smplx.py` plots every loss per frame
    (+ keypoint/depth availability) to catch these.
- **Camera is its own stage; the body uses the VGGT camera.**
  `estimate_camera_vggt.py` + `estimate_camera_moge2.py` write per-frame
  intrinsics; `fuse_geometry.py` denormalizes VGGT's to the original resolution
  (`geometry/transform.npz` `intrinsics_px_orig`). `estimate_body_sam3d.py` reads
  `K` from there (fallbacks: `vggt/camera.npz` denormalized → `moge2/camera.npz`
  → one-off MoGe2 frame-0). This is why `fuse` runs before the body stage.
- **VGGT-Omega is additive, runs right after `sam3`.** It is a *multi-view*
  model (one forward over the whole clip), so it caps at `max_frames` (default
  500, uniformly subsampled) and runs at a reduced internal resolution. It writes
  its own `vggt/camera.npz` — cross-frame `extrinsics` + normalized `intrinsics`
  (same convention as `moge2/camera.npz`) **plus the VGGT inference resolution**
  so depth/point maps can be mapped back to full-res frames. It **feeds the body
  stage** via `fuse_geometry.py` (which denormalizes its intrinsics to the
  original resolution); the two camera stages coexist — MoGe2 supplies the metric
  depth that scales VGGT. **VGGT geometry is up-to-scale, not metric** (one
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
