# BetterVideoReconstruction — Claude Code Guide

Monocular video → a per-person **SMPL-X body trajectory** for everyone tracked,
plus **per-contact-joint forces** on the refined bodies. Track the people (SAM 3)
→ VGGT camera → Sapiens keypoints → per-frame MHR body for every person
(SAM-3D-Body) → convert to **SMPL-X** → fuse the up-to-scale point clouds →
**anchor VGGT's metric scale to the human body** (rescales the fused geometry in
place) → **visually-guided kinematic refinement**. The
default chain ends at the refined SMPL-X bodies; scene contacts, gravity,
annotation-guided refinement, and the joint kinematics+dynamics stage are
standalone SMPL-X stages.

The algorithm docs live in `docs/` and are the reference for auditing the
optimization — keep them current when the code changes:

- [`docs/scale.md`](docs/scale.md) — human-ruler metric scale + in-place metricization
- [`docs/fusion.md`](docs/fusion.md) — up-to-scale clouds, relative two-pass voxel merge
- [`docs/human_optimization.md`](docs/human_optimization.md) — the 4 stages, every loss term
- [`docs/contacts.md`](docs/contacts.md) + [`docs/contact_confidence.md`](docs/contact_confidence.md) — contact labels + confidence
- [`docs/force_contacts.md`](docs/force_contacts.md) — the force witness: contact detection by inverse dynamics, end to end
- [`datasets/peter/board_contacts.md`](datasets/peter/board_contacts.md) — contacts MEASURED by the peter rigs'
  force boards instead of estimated, and the `human_optim_gt_contact/` variant chain
- [`docs/quality.md`](docs/quality.md) — depth-smear downweighting, the scene-quality table, whole-body force confidence

## Working principles

Try ideas, simplify the code. Four rules guide every change:

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

Work happens on the **`dev`** branch (this worktree); `main` stays as it is until
dev is merged. Commit directly to `dev` — do not create further branches. The sibling
libraries are the `dev` worktrees next to this repo (`../BetterRobot`, `../BetterHuman`,
see `INSTALL.md`).

## Data layout

Data lives outside the repo, in `../data` (set `BVR_DATA_ROOT` to move it; the
resolver is `bvr/paths.py`): `examples/{data,data_2,out,out_2}` (example clips +
their results), `peter/` (the force-rig recordings and their out trees), and the
datasets (`ClimbingVideos`, `LabValidation_withVideos`, `Parkour-dataset`,
`BEDLAM2_our`, ...). Nothing data-like lives in the repo. `datasets/climbingvideos/configs/corpus.toml`
`data_root` is resolved against the repo root by `corpus.py`,
`assign_dataset_split.py` and `calibrate_contact_confidence.py`, not via
`BVR_DATA_ROOT`.

## Configs over CLIs

**Every behavior knob lives in a TOML; CLIs take only targets** (what to
process: `--source`, `--predictions`, `--save-dir`, stage/scene selection,
`--config`). The config files:

One loader, `bvr.config.load_config`: a TOML may name a `base` (a path relative to the repo
root) and is merged over it table by table, so a dataset config holds only the knobs it
changes (a loss-weight column is one value, replaced whole).

- [`configs/pipeline.toml`](configs/pipeline.toml) — one section per stage
  (`[sam3] [vggt] [sapiens] [scale] [fuse] [sam3d] [smplx] [gravity] [viewer]`),
  read by the stage CLIs themselves + `[steps]`/`[human_optimization]` for a
  `scripts/pipeline.py --source` run.
- [`datasets/peter/configs/pipeline_peter.toml`](datasets/peter/configs/pipeline_peter.toml) — the static-camera
  peter rigs and Parkour, over the base: one tracked person, overlays off, no pre-fit, chunked VGGT.
- [`datasets/opencap/configs/pipeline_opencap.toml`](datasets/opencap/configs/pipeline_opencap.toml) — OpenCap
  LabValidation, over the peter config: no frame subsampling and leaner VGGT chunks.
- [`configs/human_optimization.toml`](configs/human_optimization.toml) — the four
  human-optimization stages: loss weights as length-4 columns
  `[root, kinematics_1, kinematics_2, kindyn_1]` (a term is off where its
  column is 0), per-stage `[stages.*]` options, shared knob sections, `[contacts]` +
  `[contact_force]`.
- [`datasets/peter/configs/human_optimization_peter.toml`](datasets/peter/configs/human_optimization_peter.toml) —
  the rigs, Parkour and OpenCap, over the general config: contact frames on hands and
  feet only, kinematic chamfer off, the `[measured_contacts]` thresholds.
- [`datasets/opencap/configs/human_optimization_floorplane.toml`](datasets/opencap/configs/human_optimization_floorplane.toml) —
  the OpenCap `floorplane` variant, over the general config: feet + sit-bone frames, no
  hands, contact against a synthetic support scene (`floor_gate_m = 0`), shod heels.
- [`datasets/opencap/configs/human_optimization_floorplane_edgev.toml`](datasets/opencap/configs/human_optimization_floorplane_edgev.toml) —
  floorplane with the `kindyn_1` edge knobs on (`force_edge`, `edge_speed_m_s`).
- [`datasets/climbingvideos/configs/corpus.toml`](datasets/climbingvideos/configs/corpus.toml) — ClimbingVideos DB fan-out (data
  root, devices, QA/discard), the corpus tools (`[split]`) and which scenes the viewer
  browses (`[viewer]`).
- `datasets/<name>/configs/dataset.toml` (`parkour`, `opencap`, `peter`) — the measured
  datasets `scripts/pipeline.py --dataset` runs (roots, clips or rig views, out trees,
  their configs) and the `[variants.*]` chains.

## Two runners, shared stages

- **Clips** — `scripts/pipeline.py` runs the chain over a target: `--source` (a
  video/folder → `<save-dir>/<stem>/`, one tree per clip) or `--dataset <name>` (a
  `datasets/<name>/configs/dataset.toml` table: `parkour`, `opencap`, and the peter
  rigs `campus_board`, `climb_wall_2`, `climb_wall_3` → `<out>/<clip>/`, the same
  tree). Every stage command comes from `bvr/runner.py` (`stage_command`,
  `sweep_command` → `python -m bvr.stages.<name>`). The five model stages sweep every
  clip with one model load; the rest run clip by clip. `--skip-existing` resumes (complete
  per-clip steps and finished sweep items are skipped; a `--source` run recomputes
  everything without it, a dataset run always resumes its per-clip steps), `--shard i/n` across
  GPUs, `--dry-run` to print the commands. `--variant <name>` re-runs the human chain
  into `<clip>/human_optim_<name>/` (`floorplane`: a synthetic support scene;
  `gt_contact`: the boards' measured contacts); `--views a,b` reconstructs a rig clip
  from several cameras at once.
- **ClimbingVideos DB** — `datasets/climbingvideos/corpus.py --stage <s>` fans ANY stage over the
  scene DB (`features/<stage>/<aa>/<bb>/<scene>/`), stride-sharded across the
  configured GPUs. Manifest stages (sam3/vggt/sapiens/sam3d/smplx) sweep with one
  model load per GPU; farm stages (geometry/human_optim/contacts) fake a local
  `out/<stem>/` tree with symlinks and call the same stage entry points. Stage
  knobs come from the same pipeline/human-optimization TOMLs (written per shard
  with the corpus overrides) and the stage commands from the same `bvr/runner.py`,
  so the two runners cannot drift. Selection:
  `--scenes/--sample/--limit`, plus `--dry-run`. DB access in `datasets/climbingvideos/scene_db.py`.
- `datasets/examples/score_frame_contacts.py` scores the stage's contact-frame labels
  against manual annotations.

## The pipeline & `out/<stem>/` contract

Default chain in order (rows 1–8; `human_optimization` runs its `root` +
`kinematics_1` sub-stages). The rest are standalone SMPL-X stages.

| # | Stage (`bvr/stages/…`) | Reads | Writes (under `out/<stem>/`) |
|---|---|---|---|
| 1 | `track_sam3.py` | video/frames | `sam3/<obj:02d>/frame_*.png`, `sam3/mask_overlay.mp4`, `sam3/bboxes.npz` |
| 2 | `estimate_camera_vggt.py` | frames | `vggt/camera.npz` (multi-view extrinsics + intrinsics, ≤`max_frames`), `vggt/depth/…` (`save_depth` — the chain needs it) |
| 3 | `estimate_sapiens.py` | frames, `sam3/bboxes.npz` | `sapiens/pose.npz` (Goliath-308), overlay |
| 4 | `estimate_body_sam3d.py` | `sam3/`, `vggt/camera.npz`, frames | `sam3d/params.npz` (all people, `(P,N,…)` + `object_ids`), overlay |
| 5 | `convert_mhr_to_smplx.py` | `sam3d/params.npz` | `sam3d/smplx_params.npz` (`(P,N,…)` classic SMPL-X), overlay |
| 6 | `fuse_geometry.py` | `vggt/`, `sam3/`, `sam3d/smplx_params.npz`, frames | `geometry/transform.npz` (cameras, `metric=False`), `geometry/scene.npz` (two-pass voxel merge + hidden-point prune + per-point normals), `geometry/human.npz` (per-frame human points) |
| 7 | `estimate_scale.py` | `sam3d/smplx_params.npz`, `vggt/camera.npz`, `sapiens/pose.npz`, `geometry/` | `geometry/scale.npz` + rescales `transform/scene/human.npz` in place (`metric=True`); `geometry/prefit.npz` (dual scale+body pre-fit bodies, unless `prefit=false`) |
| 8 | `human_optimization.py` (`root`+`kinematics_1`) | `sam3d/smplx_params.npz`, `sapiens/pose.npz`, `geometry/` (init pose = `prefit.npz` when present) | `human_optim/kinematic_1.npz` (world-frame `q (P,N,211)` + per-contact-frame `drift_pre`/`auto_labels (P,N,F)`), optional overlay |
| – | `human_optimization.py --stages kinematics_2` | `kinematic_1.npz` + `contacts_1.npz` frame labels | `human_optim/kinematic_2.npz` (contact-patch repair: each labelled contact frame pulled onto the wall seen at its own pixel; `drift_pre` forwarded) |
| – | `estimate_contacts.py` | a refined body (`--bodies`), `geometry/`, its `drift_pre` | `human_optim/contacts_<tag>.npz` — **per-contact-frame** labels (schema 2 `frames-v4`: one column per contact frame, geometry-gated-by-solved-load ∪ the force witness (stillness decides only where the solve explained nothing), toe rule = whichever side the solve loaded, debounce + hysteresis; see `docs/force_contacts.md`) |
| – | `measure_contacts.py` | a refined body (`--bodies`), `human_optim/sensor_forces.npz`, `geometry/` | `human_optim/contacts_<tag>.npz` — the same schema, labels **measured** by a rig's force boards instead of estimated (`datasets/peter/board_contacts.md`) |
| – | `estimate_gravity_geocalib.py` | frames, `geometry/transform.npz`, `geometry/scene.npz` | `geocalib/gravity.npz` (world-frame down + `source`: reliable GeoCalib > gated ground-plane fit > camera-down fallback; the accepted floor plane is saved for the viewer) |
| – | `build_support_scene.py` | `geocalib/gravity.npz` (+ a body for the plateaus) | `geometry/scene.npz` REPLACED by a synthetic support scene: the floor plane as a lattice + one horizontal patch per surface the body rests on (the `floorplane` variant, written into its farm) |
| – | `human_optimization.py --stages kindyn_1` | a refined body + its contacts (default `kinematic_2.npz` + `contacts_2.npz`; `[stages.kindyn_1].bodies`), gravity | `human_optim/kindyn_1.npz` (pose AND per-contact-frame forces solved together; each force in a 45° cone about that frame's contact normal) |

Viewer: `scripts/view_3d.py out/<stem>/` (or a root of clips: a pager in the sidebar
switches between them). It is a thin adapter: `bvr/viewer/loading.py` reads a clip into
plain numpy (`ClipData`), `bvr/viewer/layers.py` turns it into BetterRobot / BetterHuman
viewer layers (points, arrows, cameras, ground, skinned body, skeleton, spheres), and
`bvr/viewer/app.py` is the sidebar: the **`body source`** dropdown switches `original`
(converter) / `prefit` / `kinematics` / `kinematics_2` / `kindyn_1`, each when its file
exists; the **layer** checkboxes toggle one *kind* each (`layers.KINDS`: SMPL-X body,
skeleton, contact frames red/green with their normal arrows, contact forces per joint or
per frame, root force/torque, human points, scene points and normals, cameras, gravity,
ground plane); the **video pane** plays the source footage in sync (`[viewer].video`).
Bodies are posed once per clip on the CPU. Dataset readers are composed in
`scripts/view_3d.py`, not in `bvr/viewer`: measured rig forces (`datasets/_rigs.py`),
OpenCap plate estimates (`datasets/opencap/viewer.py`), and the two DB browsers —
`--scenes` (ClimbingVideos, `datasets/climbingvideos/viewer.py`, scene choice in
`[viewer]` of `datasets/climbingvideos/configs/corpus.toml`) and `--bedlam` (BEDLAM2
ground truth, `datasets/bedlam2/viewer.py`, `[viewer]` of
`datasets/bedlam2/configs/bedlam.toml`; one **`ground truth`** source, cameras, ground
plane and gravity, no scene points; where `datasets/bedlam2/label_bedlam2.py` has run
also its contact frames and solved forces).

**BEDLAM2 labels** — `datasets/bedlam2/label_bedlam2.py` measures what that pose ground truth
implies but does not state, writing it beside it under `features/gt/`:
`--step masks` rasterises the posed meshes into one 8-bit instance map per frame
(`masks/NNNNNN.png`, `0` background and `p+1` per person, so occlusion between people
is exact) plus `bboxes.npz` — unclipped boxes, mask areas and the per-frame `valid`
flag (at most a third of the box outside the image, at least one visible pixel, nothing
inside the near plane); `--step contacts` labels the 35 body contact frames against the
floor from three measurements — height, stillness, and which way the skin faces
(`contacts.npz`, the schema the viewer draws); `--step forces` solves the force on
every labelled frame at the frozen ground-truth pose with `kindyn_1`'s force objective
(`forces.npz`, the fields of `kindyn_1.npz`, so the viewer draws them; the RNEA is
affine in the forces at a fixed pose, so a scene takes a second or two); `--step select`
reports the scenes with no valid person and, with `--apply`, clears their
`human_selected` in the DB with a rollback JSON. Nothing is deleted and `smplx.npz` is
never rewritten — invalid people are flagged in `bboxes.npz` and the loaders skip them.
Knobs in `[bedlam]` and `[bedlam_forces]` of `datasets/bedlam2/configs/bedlam.toml`.

## Run

```bash
# Whole chain on one clip (knobs in configs/pipeline.toml, chain = its [steps].run).
uv run python scripts/pipeline.py --source ../data/examples/data/boulder_1.mp4 --save-dir ../data/examples/out/
# Batch of clips, resumed: complete per-clip steps and finished sweep items are skipped (without the flag everything reruns).
uv run python scripts/pipeline.py --source ../data/examples/data/ --save-dir ../data/examples/out/ --skip-existing
# The standalone human steps after the default chain, in order.
uv run python scripts/pipeline.py --source ../data/examples/data/ --save-dir ../data/examples/out/ --steps contacts_1,kinematics_2,contacts_2,kindyn_1
# One stage standalone (all stages follow this shape).
uv run python -m bvr.stages.estimate_scale --predictions ../data/examples/out/boulder_1/
uv run python -m bvr.stages.human_optimization --source ../data/examples/data/boulder_1.mp4 --predictions ../data/examples/out/boulder_1/
# Corpus: any stage over the scene DB.
uv run python datasets/climbingvideos/corpus.py --stage geometry --limit 10 --dry-run
# A measured dataset (datasets/<name>/configs/dataset.toml); --dry-run prints the commands, runs nothing.
uv run python scripts/pipeline.py --dataset climb_wall_3 --variant gt_contact --dry-run
# View a clip (or a root of clips) in the browser.
uv run python scripts/view_3d.py ../data/examples/out/boulder_1/
# Tests (CPU).
CUDA_VISIBLE_DEVICES="" uv run pytest -q
```

`../data` is `$BVR_DATA_ROOT`. `BVR_DETERMINISTIC=1` makes a GPU run bit-reproducible
(`bvr/determinism.py`; off by default).

## Project layout

```
BetterVideoReconstruction/
├── CLAUDE.md / README.md / INSTALL.md
├── docs/                          # the algorithm docs (see top of this file)
├── configs/                       # the general TOMLs: pipeline.toml, human_optimization.toml
├── scripts/
│   ├── pipeline.py                # the chain over clips: --source video/folder or --dataset <name> (+ variants)
│   ├── view_3d.py                 # the viewer CLI (composes the dataset readers into bvr/viewer)
│   └── scene_quality.py           # per-clip reconstruction-quality table
├── bvr/                           # the generic core, an installable package (never imports datasets/)
│   ├── stages/                    # the per-stage CLIs in the table above (python -m bvr.stages.<name>)
│   │   └── _pym_preload.py        # libstdc++ LD_PRELOAD shim for pymomentum
│   ├── runner.py                  # the ONE place a stage argv is assembled (+ the variant farms)
│   ├── config.py                  # the TOML loader (base + override merge) and the human-optimization config
│   ├── paths.py                   # where the data lives (DATA_ROOT, EXAMPLES)
│   ├── body.py                    # the one place a human body is built (BetterHuman SMPL-X + contact frames)
│   ├── determinism.py             # BVR_DETERMINISTIC=1
│   ├── npz_io.py / source.py / mask_io.py / precompute_runner.py
│   ├── adapters/                  # stateless model wrappers (sapiens2, vggt_omega, geocalib)
│   ├── mhr_to_smplx/              # MHR → SMPL-X fitter (pip smplx)
│   ├── human_optim/               # the human stages: solve.py (BetterRobot optim glue), optimizer / prefit /
│   │                              #   stages / kindyn / forces / contacts / measured_contacts / confidence / losses / …
│   ├── geometry/                  # depth clean + backproject + voxel + normals + human scale + quality
│   └── viewer/                    # loading.py (ClipData) + layers.py (BR/BH layers) + app.py (sidebar)
├── datasets/                      # one package per dataset / benchmark; each has a README.md
│   ├── _common.py / _rigs.py      # shared diagnostics helpers; measured rig forces in the viewer
│   ├── examples/                  # the annotated example clips: annotations.py, score_frame_contacts.py
│   ├── climbingvideos/            # the scene DB: corpus.py, scene_db.py, viewer.py, assign_dataset_split.py,
│   │                              #   calibrate_contact_confidence.py, configs/corpus.toml, tests/
│   ├── peter/                     # the force rigs: sensor_forces.py, compare_*, render_*, board_contacts.md,
│   │                              #   configs/{dataset,pipeline_peter,human_optimization_peter}.toml, gt/ (3-camera GT chain)
│   ├── opencap/                   # OpenCap LabValidation: import, sync, plates, sensor_forces, exports, scorers,
│   │                              #   viewer.py, configs/{dataset,pipeline_opencap,human_optimization_floorplane*}.toml, tests/
│   ├── parkour/                   # LAAS Parkour: import, sensor_forces, scorers, configs/dataset.toml
│   ├── bedlam2/                   # BEDLAM2 ground truth: import, label (masks/contacts/forces), viewer.py, configs/bedlam.toml, tests/
│   ├── amass/                     # label_amass.py (AMASS motions through the BEDLAM2 readout)
│   └── methods/                   # inputs for and scores of the published baselines (Li et al., PhysPT, …)
├── tests/                         # the core's tests (dataset tests live next to their dataset)
└── third_party/                   # submodules sam3, sam-3d-body, sapiens2, MHR, geocalib; vendored vggt-omega
```

## Conventions & deliberate choices

- **One optimizer, BetterRobot's.** Every fit is a `better_robot.optim.Problem`
  (named variables, one cached forward, named scalar terms) run by
  `better_robot.optim.run` over `Phase` rows; `bvr/human_optim/solve.py` is the thin
  glue (the lazy `State` the terms share, term registration, the Adam / L-BFGS
  factories, the per-term logger). A term is active iff its phase weight is non-zero,
  so the TOML weight columns ARE the objective. Kinematics, the kindyn warm start, the
  force witness, the scale pre-fit and the MHR→SMPL-X converter all run on it.
- **Multi-person throughout.** Every stage from `sam3d` on carries a leading
  `(P, N, …)` axis with `object_ids (P,)`.
- **Body is SMPL-X.** `convert_mhr_to_smplx` uses pip `smplx` (the only place);
  the optimization path poses via `better_human` (`SMPLXMid`, 52 joints,
  `q (N,211)`). No per-body scale — `estimate_scale` sizes the world to the body.
- **Keypoints are Goliath-308**, not COCO-wholebody (`bvr/human_optim/keypoints.py`).
- **Camera is VGGT's** (scale-invariant intrinsics); **gravity is the gravity stage's**
  (world-frame; reliable GeoCalib, else a gated floor-plane fit to the fused
  scene, else world-down); **VGGT geometry is
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
