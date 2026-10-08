# Report I — method-input exporters (mars_top_opus, 2026-10-08, base HEAD 0cf1281)

Result: seven exporters became three scripts in `scripts/methods/`. All outputs are byte- or bit-identical to the HEAD scripts on every clip of every dataset. Full suite: 250 passed. No real external dir was written.

## 1. What replaced what
| old (trashed) | new |
|---|---|
| scripts/parkour_estmf_inputs.py | `scripts/methods/estmf_inputs.py --dataset parkour` |
| scripts/methods/opencap_estmf_inputs.py | `estmf_inputs.py --dataset opencap [--contact-mapping heel\|any]` |
| scripts/methods/cw3_estmf_inputs.py | `estmf_inputs.py --dataset climb_wall_3 [--clips a,b]` |
| scripts/parkour_physpt_inputs.py | `scripts/methods/physpt_inputs.py --dataset parkour\|climb_wall_3` |
| scripts/methods/opencap_physpt_inputs.py | `physpt_inputs.py --dataset opencap [--step 3] [--offset 0]` |
| scripts/parkour_prepare_interactvlm.py | `scripts/methods/baseline_farm.py --dataset parkour` |
| scripts/peter_gt/prepare_baselines.py | `baseline_farm.py --dataset climb_wall_3 [--skip-existing] [--jobs 8]` |

Common flags: `--clips` (comma list, or a file of names for estmf/physpt) and `--out` (target dir; default = old location). Default outputs unchanged: `Estimating-3D-Motion-Forces/data/{Parkour-dataset,OpenCap,climb_wall_3}/ours`, `../other/PhysPT/<dataset>`, `Parkour-dataset/processed/others/_farm`, `peter/climb_wall_3_gt/others/_farm`. Lines: estmf 687 → 397, physpt 281 → 161, farm 186 → 163 (1154 → 721).

Design: estmf_inputs has one copy of `smplx_model`, `coco18_from_sapiens`, `on_true_camera`, `project`, constants, `one_hot`, `measured_contacts`; per-dataset code is the contact source and one `if dataset` block (camera + extra npz keys); `select_clips` holds each dataset's default clip set. physpt_inputs has one copy of `rodrigues`, `smpl_rest_joints`, `smplx_rest_joints`, `fit_betas` and imports `MODELS, RUNS, select_clips` from estmf_inputs; frame bookkeeping keys written for opencap only; `--step/--offset` other than 1/0 on parkour/cw3 exit 2. baseline_farm uses `scene_db.shard` for the fan-out, one `extract` in a process pool, per-dataset clip lister + features step.

## 2. Files
Created: scripts/methods/estmf_inputs.py, physpt_inputs.py, baseline_farm.py. Edited: methods/opencap_physpt_export.py:23 (docstring), peter_gt/run_baselines.sh:3 (comment; it never called prepare_baselines). Trashed: the 7 old scripts. Leftover reference fixed by the main session: scripts/diagnostics/compare_parkour_methods.py:17.

## 3. Equivalence (CUDA_VISIBLE_DEVICES=2; all CPU)
Harness in scratchpad `I_old/` (HEAD copies with `_REPO` pinned and output dir / clip list overridden) and `I_old/cmp_dirs.py` (file sets, npz keys/dtypes/shapes/array_equal, json/jpg/txt bytes, symlink targets).

| run | result |
|---|---|
| estmf parkour (all 28) | 28/28 npz exact |
| estmf opencap heel / any (all 64 each) | 64/64 + 64/64 exact |
| estmf climb_wall_3 (all 68) | 68/68 exact |
| physpt parkour (28) / climb_wall_3 (68) / opencap step 3 (64) / opencap step 1 offset 1 (2) | all json bytes equal |
| farm parkour (all 28) | 2353/2353 files equal |
| farm climb_wall_3 (1 clip decoded) | 484/484 equal (475 jpg, 8 symlink targets, test_scenes.txt) |
Extra: the new listers reproduce the real farms' `results/test_scenes.txt` byte for byte (28 and 68 rows) and match all 544 symlinks of the real cw3 farm. Stdout totals agree. `find -newer` shows no file written under the real external dirs. `--help` exits 0; all 8 methods modules import; ruff F,E9 clean.

## 4. Tests
250 passed in 38.5 s (R and S editing in parallel).

## 5. Behaviour differences (outputs unchanged)
1. baseline_farm decodes exactly the clip's frame count for both datasets and fails if the video is short (all 68 + 28 videos match).
2. parkour farm gains `--skip-existing`, `--jobs`, `--clips`, `--out`; no per-clip fps print.
3. `--clips` naming an unknown clip fails at load (old cw3 silently dropped it); opencap also accepts a comma list.
4. estmf stdout: per-clip contact counts + one summary line.
5. A `--clips` subset in baseline_farm writes test_scenes.txt with only those clips (documented in --help).

## 6. Not done / noted
- `physpt_model` still duplicated at diagnostics/compare_parkour_methods.py:255 and methods/opencap_physpt_export.py:83.
- opencap_frames35_export.py not folded into opencap_learned_export.py (differs in load_prediction, foot points, foot_triples, saved keys; a fold saves almost nothing).
- External docstrings still name the old scripts (outside this repo): data/Estimating-3D-Motion-Forces/scripts/build_our_inputs.py:5, run_opencap.py:8, run_climb_wall_3.py:18, NOTES.md:81,157,162; other/PhysPT/run_parkour.py:6, run_opencap.py:5; contact_anything/scripts/physpt_dumps.py:4.
- Not verified: cw3 farm with all 68 clips decoded (metadata and links checked for all 68); downstream consumers not run (outputs identical).

## Open questions
1. Repoint the external docstrings listed above?
2. Pre-existing quirk kept: parkour projects ankle stand-ins with the averaged SAM-3D focal, opencap/cw3 with fx/fy separately.
