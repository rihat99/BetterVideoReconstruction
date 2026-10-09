# Report 7 — review fixes (mars_top, 2026-10-09; saved by the main session; committed as dev ebf3c99)

Proofs in artifacts/dev7/ (help_7.log, pytest_7.log, dryrun_diff_7_datasets.txt, local_actions_7.txt, config_merge_equivalence_7.txt, regression_7_kin1.log, compare_7.{log,json}, the 5 `--source` dry-run captures, scripts). Removed function saved in artifacts/dev7/removed_draw_caption.py.

## 1. Runner resume semantics (review finding 1) — one flag, two defaults
- `scripts/pipeline.py:408` `resume = args.skip_existing or dataset is not None`; per-clip skip at :431 `if resume and step_done(step, clip_dir)`. A `--source` run recomputes every selected step unless `--skip-existing` (main's local runner); a `--dataset` run keeps run_dataset's resume (its pipeline configs set the stages' `skip_existing = true`). Said in `--help` (:277-281), the docstring (:19-27), CLAUDE.md (:103-105, :185).
- `bvr/runner.py:93-118` `step_done(step, clip_root)`: `CLIP_OUTPUT` existence; for `scale` main's complete check — transform.npz present with `metric=True`, and when scale.npz says `method == "prefit-dual"` also prefit.npz (estimate_scale.py:505-510); missing transform.npz → not done. Used for both targets.
- `bvr/runner.py:78-90` `sweep_command(..., skip_existing=False)` appends `--skip-existing`; pipeline.py forwards it to the five sweeps (:400-401) and the extra-view sam3 sweeps (:199-200). sam3d and smplx CLIs gained the flag (`estimate_body_sam3d.py:507-508,515`, `convert_mhr_to_smplx.py:488-489,497`).
- kinematics_1: main's local runner passed no `--stages` (the HUMAN config's `[stages].run` ran; main pipeline.py:146-150); run_dataset and corpus forced `root,kinematics_1`. Now `stage_command` dispatches `",".join([stages].run)` of the human config on every runner (`bvr/runner.py:70-72`, :138-141), default `root,kinematics_1`; every shipped config says exactly that → every argv unchanged.
- Dry-run proof: 21 dataset/variant + 4 corpus cases vs 4c's captures: 0 differing lines in all 25 (normalised for step 6's `measure_contacts` rename). `--source` cases = main's semantics (local_actions_7.txt): no flag → everything runs; `--skip-existing` → finished sweeps not launched, finished per-clip steps skipped; `local_steps` identical. `step_done` edge cases checked by hand.

## 2. `load_config` (finding 2)
- `bvr/config.py:29-67`: base-cycle guard (`ValueError("config base cycle: a -> b -> a")`, self-base too) and one stderr line listing delta keys the base has no value for (dotted paths); whole tables the base lacks are exempt.
- Legit delta-only keys now have base defaults: `configs/human_optimization.toml` `[stages.kindyn_1].mass_kg = 0`, `.edge_speed_m_s = 0.0`, `[loss_weights.dynamics].force_edge = [0,0,0,0]`, `[contacts].floor_gate_m = 0.0`; `sole_offset_m` (inline table, exempt) stays delta-only. Every shipped config loads with no warning.
- `config_merge_equivalence.py` re-run: no value differs beyond the known 4c/6 ones; the 4 new base keys appear as "only in MERGED" with the defaults the stages read anyway.
- `load_human_config` (:121-129) warns once on weight rows that name no term (vocabulary = the general config's `[loss_weights.*]` rows; a group the general config lacks is a chain's own terms). The peter GT chain's five rows moved into `[loss_weights.multiview]` in `datasets/peter/gt/human_gt.toml:191-205` (groups are flattened; 33 rows equal).

## 3. Stale text (finding 7)
converter.py:7 → `bvr.body`; bedlam.toml:13 → `datasets/bedlam2/contacts.py ENTER_M`; compare_opencap_plates.py:66 → `datasets/opencap/*_export.py`; climbingvideos/README.md:15 → `configs/corpus.toml`; three `:mod:scripts.stages.*` docstrings (vertex_weights.py:155, human_scale.py:3, mask_io.py:45) → `bvr.stages.*`. Left as historical: docs/force_contacts.md:382, converter.py:169 (Meta's MHR path).

## 4. Orphans and runaway scripts (finding 8)
- `_draw_caption` removed (its only caller, render_sup_force_overlays.py, was retired). Pre-existing dead on main, left: `bvr/precompute_runner.py:59 _MANIFEST_EXTS`, `bvr/stages/track_sam3.py:240 TrackResult`, `datasets/climbingvideos/scene_db.py:28 DEFAULT_CATEGORIES`, `datasets/peter/gt/view_gt.py:31 camera_pose`.
- score_init_world.py:101, calibrate_cameras.py:391, score_parkour_pose.py:117: zero-argument argparse at the entry, so `--help` exits before any work.
- label_bedlam2.py:114-115: `torch.set_num_threads(min(16, max(1, cpu_count // workers)))`.

## 5. Verification
`--help` 60/60 rc=0 (help_7.log); pytest 229 passed, 4 skipped (193 s), no `[config]` warning; dry-run diffs 25/25 unchanged; regression (GPU 2, BVR_DETERMINISTIC=1) `scripts/pipeline.py --source 9RvrpYoWDoo_0001 --save-dir <scratch> --steps kinematics_1` on a linked scratch tree: kinematic_1 27/27 keys identical to human_optim_dev3.

## Open questions
1. kinematics_1 follows the human config's `[stages].run` on all runners (main's local-runner behaviour); the fixed `root,kinematics_1` is one line (`bvr/runner.py:72`). Main session: kept as is (parity with main's local runner; all shipped configs identical).
2. GT chain's rows in `[loss_weights.multiview]` vs explicit extra terms in `load_human_config` — kept.
3. `sole_offset_m = {}` in the base for visibility? — not added.
