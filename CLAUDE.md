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
- [`docs/board_contacts.md`](docs/board_contacts.md) — contacts MEASURED by the peter rigs'
  force boards instead of estimated, and the `human_optim_gt_contact/` variant chain
- [`docs/contact_fusion.md`](docs/contact_fusion.md) — retired per-joint-era mechanism (kept as a design record)
- [`docs/quality.md`](docs/quality.md) — depth-smear downweighting, the scene-quality table, whole-body force confidence
- [`docs/contact_points.md`](docs/contact_points.md) — investigation (not implemented): contact on mesh points instead of joints

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
  `[root, kinematics_1, kinematics_2, kindyn_1]` (a term is off where its
  column is 0), per-stage `[stages.*]` options, shared knob sections, `[contacts]` +
  `[contact_force]`.
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
- `scripts/score_frame_contacts.py` scores the stage's contact-frame labels
  against manual annotations.

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
| 8 | `human_optimization.py` (`root`+`kinematics_1`) | `sam3d/smplx_params.npz`, `sapiens/pose.npz`, `geometry/` (init pose = `prefit.npz` when present) | `human_optim/kinematic_1.npz` (world-frame `q (P,N,211)` + per-contact-frame `drift_pre`/`auto_labels (P,N,12)`), optional overlay |
| – | `human_optimization.py --stages kinematics_2` | `kinematic_1.npz` + `contacts_1.npz` frame labels (per-joint `data/<stem>.npz` via `[stages.kinematics_2].labels = "manual"`) | `human_optim/kinematic_2.npz` (contact-patch repair: each labelled contact frame pulled onto the wall seen at its own pixel; `drift_pre` forwarded) |
| – | `estimate_contacts.py` | a refined body (`--bodies`), `geometry/`, its `drift_pre` | `human_optim/contacts_<tag>.npz` — **per-contact-frame** labels (schema 2 `frames-v4`: 12 frames, geometry-gated-by-solved-load ∪ the force witness (stillness decides only where the solve explained nothing), toe rule = whichever side the solve loaded, debounce + hysteresis; see `docs/force_contacts.md`) |
| – | `estimate_gravity_geocalib.py` | frames, `geometry/transform.npz`, `geometry/scene.npz` | `geocalib/gravity.npz` (world-frame down + `source`: reliable GeoCalib > gated ground-plane fit > camera-down fallback; the accepted floor plane is saved for the viewer) |
| – | `human_optimization.py --stages kindyn_1` | a refined body + its contacts (default `kinematic_2.npz` + `contacts_2.npz`; `[stages.kindyn_1].bodies`), gravity | `human_optim/kindyn_1.npz` (pose AND per-contact-frame forces solved together; each force in a 45° cone about that frame's contact normal) |

Viewer: `scripts/view_3d.py out/<stem>/` (viser: fused scene + cameras + human
points + bodies; the **`body source`** dropdown switches `original` / `prefit` /
`kinematics` / `mhr_1_sup` — the kindyn body on the MHR rig
(`human_optim/mhr_1.npz`), mesh only, no contacts or forces — plus the
frozen-era sources when their files exist; the skeleton
draws all 52 joints — fingers as small 0.30× spheres; the **`contact frames`**
layer draws the 12 contact frames red/green from `contacts_1` with a thin
connector to each parent joint and a 6 cm outward-normal arrow; a
**`scene normals`** debug layer draws the per-point normal arrows; the **ground
plane** draws as a translucent grey fill under grey half-metre cells). `--scenes`
browses the ClimbingVideos DB; `--scenes athletics` the AthleticsVideos DB, with a
**`sport`** dropdown in the sidebar that narrows the scene list to one sport.
`--bedlam` browses the BEDLAM2 ground truth (`../data/BEDLAM2_our`, one feature
stage `gt/`): a single **`ground truth`** body source, the cameras, the ground
plane and gravity — the synthetic environments ship no geometry, so there are no
scene points. Where `scripts/label_bedlam2.py` has run it also draws the **contact
frames** red/green from `gt/contacts.npz` and drops the people `gt/bboxes.npz` found
no usable frame for, and draws the solved forces from `gt/forces.npz` on that body
the way it draws `kindyn_1`'s. Video pane on; all 18 076 scenes in the pager unless
`[viewer].scene_ids`/`sample`/`limit` narrows them (a scene loads only when selected).

**BEDLAM2 labels** — `scripts/label_bedlam2.py` measures what that pose ground truth
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
Knobs in `[bedlam]` and `[bedlam_forces]`.

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
│   ├── import_bedlam2.py          # BEDLAM2 → the ClimbingVideos layout (pose ground truth)
│   ├── label_bedlam2.py           # BEDLAM2 masks / boxes / validity / ground contact
│   ├── score_frame_contacts.py    # contact-frame scoring vs manual annotations
│   ├── view_3d.py                 # viser viewer
│   ├── _pym_preload.py            # libstdc++ LD_PRELOAD shim for pymomentum
│   └── stages/                    # the per-stage CLIs in the table above
├── tools/                         # importable library
│   ├── optim.py                   # the shared optimization engine (Problem/Phase/run_phases)
│   ├── config.py                  # tiny TOML section helper
│   ├── source.py / mask_io.py / precompute_runner.py
│   ├── bedlam_masks.py / bedlam_contacts.py   # what label_bedlam2.py measures
│   ├── adapters/                  # stateless model wrappers (sapiens2, vggt_omega, geocalib)
│   ├── mhr_to_smplx/              # MHR → SMPL-X fitter (pip smplx)
│   ├── smplx_robot/               # SMPL-X body loader + frozen-q trajectory finite-diff (model + inertia now come baked off the BetterHuman body)
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
  stage's pre-fit run on it; the force witness keeps its own small projected-Adam
  solve (its weights multiply inside squared norms).
- **Multi-person throughout.** Every stage from `sam3d` on carries a leading
  `(P, N, …)` axis with `object_ids (P,)`.
- **Body is SMPL-X.** `convert_mhr_to_smplx` uses pip `smplx` (the only place);
  the optimization path poses via `better_human` (`SMPLXMid`, 52 joints,
  `q (N,211)`). No per-body scale — `estimate_scale` sizes the world to the body.
- **Keypoints are Goliath-308**, not COCO-wholebody (`tools/human_optim/keypoints.py`).
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
