# BetterVideoReconstruction — Claude Code Guide

Monocular video → a per-person **SMPL-X body trajectory** for everyone tracked,
plus **per-contact-joint forces** on the refined bodies. Track the people (SAM 3)
→ VGGT camera → Sapiens keypoints → per-frame MHR body for every person
(SAM-3D-Body) → convert to **SMPL-X** → fuse the up-to-scale point clouds →
**anchor VGGT's metric scale to the human body** (rescales the fused geometry in
place) → **visually-guided kinematic refinement**. The
default chain ends at the refined SMPL-X bodies; scene contacts, gravity,
annotation-guided refinement, and force dynamics are standalone SMPL-X stages.

The algorithm docs live in `docs/` and are the reference for auditing the
optimization — keep them current when the code changes:

- [`docs/scale.md`](docs/scale.md) — human-ruler metric scale + in-place metricization
- [`docs/fusion.md`](docs/fusion.md) — up-to-scale clouds, relative two-pass voxel merge
- [`docs/human_optimization.md`](docs/human_optimization.md) — the 4 stages, every loss term
- [`docs/contacts.md`](docs/contacts.md) + [`docs/contact_confidence.md`](docs/contact_confidence.md) — contact labels + confidence

## Experimental branch — working principles

This is the `experimental` branch: try ideas, simplify the code. Commits land
here (not `main`). Four rules guide every change:

1. **Keep new features simple and clean.** Add the least code that does the job.
   If it starts sprawling, step back and find the smaller version.
2. **Write docs a person can actually read.** Short, plain sentences. Explain a
   term the first time it shows up. No walls of jargon.
3. **Never over-engineer.** No configurability, abstraction, or "future-proofing"
   the task did not ask for.
4. **Prefer methods that are elegant and clever over methods held together by
   rules.** Avoid piling on gates, thresholds, and hand-tuned special cases; reach
   for the idea that makes them unnecessary.

## Git workflow

Commit directly to **`experimental`** — do **not** create branches (this
overrides the default "branch before committing" behavior for this repo).

## Configs over CLIs

**Every behavior knob lives in a TOML; CLIs take only targets** (what to
process: `--source`, `--predictions`, `--save-dir`, stage/scene selection,
`--config`). Three config files:

- [`configs/pipeline.toml`](configs/pipeline.toml) — one section per stage
  (`[sam3] [vggt] [sapiens] [scale] [fuse] [sam3d] [smplx] [gravity] [viewer]`),
  read by the stage CLIs themselves + `[steps]`/`[runtime]` for the orchestrator.
- [`configs/human_optimization.toml`](configs/human_optimization.toml) — the four
  human-optimization stages: loss weights as length-4 columns
  `[root, kinematics_1, kinematics_2, dynamics]` (a term is off where its column
  is 0), per-stage `[stages.*]` options, shared knob sections, `[contacts]` +
  `[contact_confidence]`.
- [`configs/corpus.toml`](configs/corpus.toml) — ClimbingVideos DB fan-out (data
  root, devices, QA/discard) + the corpus tools (`[export]`, `[score]`).

## Two pipelines, shared stages

- **Local, file-based** — `scripts/pipeline.py` runs the chain on a video/folder
  → `out/<stem>/` (one tree per clip). Resume with `--skip-existing`.
- **ClimbingVideos DB** — `scripts/corpus.py --stage <s>` fans ANY stage over the
  scene DB (`features/<stage>/<aa>/<bb>/<scene>/`), stride-sharded across the
  configured GPUs. Manifest stages (sam3/vggt/sapiens/sam3d/smplx) sweep with one
  model load per GPU; farm stages (geometry/human_optim/contacts) fake a local
  `out/<stem>/` tree with symlinks and call the same stage entry points. Stage
  knobs come from the same pipeline/human-optimization TOMLs (written per shard
  with the corpus overrides), so the two pipelines cannot drift. Selection:
  `--scenes/--sample/--limit`, plus `--dry-run`. DB access in `scripts/scene_db.py`.
- `scripts/export_contact_dataset.py` packages a standalone 22-joint train/test
  contact corpus; `scripts/score_contacts.py` scores contacts against manual
  annotations.

## The pipeline & `out/<stem>/` contract

Default chain in order (rows 1–8; `human_optimization` runs its `root` +
`kinematics_1` sub-stages). The rest are standalone SMPL-X stages.

| # | Stage (`scripts/stages/…`) | Reads | Writes (under `out/<stem>/`) |
|---|---|---|---|
| 1 | `track_sam3.py` | video/frames | `sam3/<obj:02d>/frame_*.png`, `sam3/mask_overlay.mp4`, `sam3/bboxes.npz` |
| 2 | `estimate_camera_vggt.py` | frames | `vggt/camera.npz` (multi-view extrinsics + intrinsics, ≤`max_frames`), `vggt/depth/…` (`save_depth` — the chain needs it) |
| 3 | `estimate_sapiens.py` | frames, `sam3/bboxes.npz` | `sapiens/pose.npz` (Goliath-308), overlay |
| 4 | `estimate_body_sam3d.py` | `sam3/`, `vggt/camera.npz`, frames | `sam3d/params.npz` (all people, `(P,N,…)` + `object_ids`), overlay |
| 5 | `convert_mhr_to_smplx.py` | `sam3d/params.npz` | `sam3d/smplx_params.npz` (`(P,N,…)` classic SMPL-X), overlay |
| 6 | `fuse_geometry.py` | `vggt/`, `sam3/`, `sam3d/smplx_params.npz`, frames | `geometry/transform.npz` (cameras, `metric=False`), `geometry/scene.npz` (two-pass voxel merge + hidden-point prune + per-point normals), `geometry/human.npz` (per-frame human points) |
| 7 | `estimate_scale.py` | `sam3d/smplx_params.npz`, `vggt/camera.npz`, `sapiens/pose.npz`, `geometry/` | `geometry/scale.npz` + rescales `transform/scene/human.npz` in place (`metric=True`); `geometry/prefit.npz` (dual scale+body pre-fit bodies, unless `prefit=false`) |
| 8 | `human_optimization.py` (`root`+`kinematics_1`) | `sam3d/smplx_params.npz`, `sapiens/pose.npz`, `geometry/` (init pose = `prefit.npz` when present) | `human_optim/kinematic_1.npz` (world-frame `q (P,N,211)` + `drift_pre` + `auto_labels`), optional overlay |
| – | `human_optimization.py --stages kinematics_2` | `kinematic_1.npz`, labels (`data/<stem>.npz` manual, or `contacts_1.npz` auto — `[stages.kinematics_2].labels`) | `human_optim/kinematic_2.npz` |
| – | `estimate_contacts.py` | a refined body (`--bodies`), `geometry/` | `human_optim/contacts_<tag>.npz` (per-joint labels + confidence) |
| – | `estimate_gravity_geocalib.py` | frames, `geometry/transform.npz` | `geocalib/gravity.npz` (world-frame down + `reliable` flag) |
| – | `human_optimization.py --stages dynamics` | `kinematic_2.npz`, `contacts_2.npz`, gravity | `human_optim/dynamics.npz` (world contact forces + joint torques) |

Viewer: `scripts/view_3d.py out/<stem>/` (viser: fused scene + cameras + human
points + bodies; the **`body source`** dropdown switches `original` / `prefit` /
`kinematics` / `kinematics_2` / `dynamics`; a **`scene normals`** debug layer
draws the per-point normal arrows). `--scenes` browses the DB.

## Run

```bash
# Whole chain on one clip (knobs in configs/pipeline.toml).
uv run python scripts/pipeline.py --source video.mp4 --save-dir out/
# Batch of clips; resume.
uv run python scripts/pipeline.py --source clips/ --save-dir out/ --skip-existing
# One stage standalone (all stages follow this shape).
uv run python scripts/stages/estimate_scale.py --predictions out/video/
uv run python scripts/stages/human_optimization.py --source video.mp4 --predictions out/video/
# Corpus: any stage over the scene DB.
uv run python scripts/corpus.py --stage geometry --limit 10 --dry-run
```

## Project layout

```
BetterVideoReconstruction/
├── CLAUDE.md / README.md / INSTALL.md
├── docs/                          # the algorithm docs (see top of this file)
├── configs/                       # pipeline.toml, human_optimization.toml, corpus.toml
├── scripts/
│   ├── pipeline.py                # LOCAL pipeline (subprocess orchestrator)
│   ├── corpus.py                  # DATASET pipeline: any stage over the scene DB
│   ├── scene_db.py                # scene-DB access + target selection
│   ├── export_contact_dataset.py  # dataset packaging (frames/masks/labels)
│   ├── score_contacts.py          # contact scoring vs manual annotations
│   ├── view_3d.py                 # viser viewer
│   ├── _pym_preload.py            # libstdc++ LD_PRELOAD shim for pymomentum
│   └── stages/                    # the per-stage CLIs in the table above
├── tools/                         # importable library
│   ├── optim.py                   # the shared optimization engine (Problem/Phase/run_phases)
│   ├── config.py                  # tiny TOML section helper
│   ├── source.py / mask_io.py / precompute_runner.py
│   ├── adapters/                  # stateless model wrappers (sapiens2, vggt_omega, geocalib)
│   ├── mhr_to_smplx/              # MHR → SMPL-X fitter (pip smplx)
│   ├── smplx_robot/               # SMPLXMid ↔ BetterRobot 52-joint model; inertia + dynamics
│   ├── human_optim/               # the stage machinery: losses/optimizer/stages/prefit/contacts/confidence/…
│   ├── geometry/                  # depth clean + backproject + voxel + normals + human scale
│   └── viewer/                    # view_3d internals
└── third_party/                   # submodules: sam3, sam-3d-body, sapiens2, MHR, vggt-omega, geocalib
```

## Conventions & deliberate choices

- **One optimization engine.** `tools/optim.py`: a stage describes its fit as a
  `Problem` (leaf tensors + one forward + named loss terms over a lazy `State`)
  and a list of `Phase` rows (weight column, iters, Adam/L-BFGS, DOF masks). A
  term is active iff its phase weight is non-zero, so the TOML weight columns ARE
  the objective. Kinematics (`tools/human_optim/optimizer.py`) and the scale
  stage's pre-fit run on it; `dynamics` keeps its own small L-BFGS solve (its
  weights multiply inside squared norms).
- **Multi-person throughout.** Every stage from `sam3d` on carries a leading
  `(P, N, …)` axis with `object_ids (P,)`.
- **Body is SMPL-X.** `convert_mhr_to_smplx` uses pip `smplx` (the only place);
  the optimization path poses via `better_human` (`SMPLXMid`, 52 joints,
  `q (N,211)`). No per-body scale — `estimate_scale` sizes the world to the body.
- **Keypoints are Goliath-308**, not COCO-wholebody (`tools/human_optim/keypoints.py`).
- **Camera is VGGT's** (scale-invariant intrinsics); **gravity is GeoCalib's**
  (world-frame, `reliable` flag, world-down fallback); **VGGT geometry is
  up-to-scale**: `fuse` builds the clouds unscaled (`metric=False`), then
  `estimate_scale` measures `s` against the fused human points and rescales the
  saved geometry in place (`metric=True`; downstream stages check the flag).
- **`root`+`kinematics_1` share ONE optimization call** — priors and the
  init-projection anchor stay pinned to the starting pose across both (the
  scale stage's pre-fit when `geometry/prefit.npz` exists, else the converter).
- **Contact honesty:** the readout gates stillness on `drift_pre` (measured
  before any contact enforcement acted); kinematics_1 auto-labels are detected
  once, frozen, positive-only. See `docs/contacts.md`.
- **Stages tolerate per-item failure** (a corrupt scene must not kill a batch),
  and the corpus contacts stage keeps sha256+schema provenance freshness.
- **GPU runs use `CUDA_VISIBLE_DEVICES=3`** (workspace convention for experiments).

## See also

- [`README.md`](README.md) — user-facing overview. [`INSTALL.md`](INSTALL.md) — setup.
- `../BetterRobot/CLAUDE.md` — the robotics library the optimizers build on.
