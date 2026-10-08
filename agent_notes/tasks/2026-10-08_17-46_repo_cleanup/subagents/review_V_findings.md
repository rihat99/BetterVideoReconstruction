# Adversarial review of 5a1fc9a..b0b3dce (mars_aux, read-only, 2026-10-08)

Verdict: no code path exercised breaks. 250 tests pass, every changed script imports cleanly, core edits are path defaults only, runner and exporters match old behaviour. Two medium hazards, six low, two nits. Triage (main session) in the last column.

| # | sev | finding | triage |
|---|---|---|---|
| 1 | MED | `_batch_manifest.jsonl` under data/peter/out_climb_wall_3_single (17 rows), out_campus_board_single/multi still hold `.../BetterVideoReconstruction/peter/...` in `source`/`out_dir`; viewer `_manifest_video` (tools/viewer/loading.py:366-389) finds the video for 0/17 (before the move it resolved). cw2 and examples/out_2 manifests pointed at `BetterVideoReconstruction-dev` (broken before). | fixed in step F: rewrite prefixes in the manifests (originals copied to trash/manifests) |
| 2 | MED | run_dataset.py:394-395 never checks the human config exists; build_farm (:146-167) re-points every farm link silently. On disk 64 `_floorplane_edgev` + 161 `_floorplane_massedgev` farms link `geometry → ../_floorplane/geometry`; a real `--variant floorplane_edgev` run would re-point them to the real cloud and report "exists, skipping". Same code as old run_opencap, but edgev is now advertised in datasets.toml. | fixed in step F: SystemExit on missing config; SystemExit on link conflict |
| 3 | LOW | baseline_farm.py:143-147 `--clips` rewrites the shared farm's results/test_scenes.txt with only the selected clips. | fixed in step F: only written without --clips |
| 4 | LOW | .gitignore no longer guards data/ out/ peter/ scratch/ etc.; `pipeline.py --save-dir out/` would show videos as untracked. | fixed in step F: anchored entries re-added |
| 5 | LOW | `BVR_DATA_ROOT` half-honoured: corpus.toml `data_root` is resolved against the repo root by corpus.py:213-216,242, assign_dataset_split.py:47, calibrate_contact_confidence.py:408-410, diagnostics/score_test_contacts.py:66-67; scene_db and score_published_* follow the env var. Unset (default) → same tree. | documented in CLAUDE.md (step F) |
| 6 | LOW | peter_sensor_forces.py:133 default --tree now exists, so a bare run rewrites 68 live sensor_forces.npz (recomputed: values identical, only mtimes change). | informational, left |
| 7 | LOW | score_published_contacts.py:167,223-229 truncates predictions to label length on ClimbingVideos where the old path raised. | disclosed in report S, left |
| 8 | LOW | data/Estimating-3D-Motion-Forces/scripts/run_climb_wall_3.py:40 default SOURCE still BVR/peter/climb_wall_3 (links exist; only new clips would dangle). `_talk_figs/scripts/rig_plots.py:7,10` (+ result_plots.py:81) import compare_climb_wall_2 and score_pipeline_contacts (trashed) → broken. | run_climb_wall_3.py fixed in step F; _talk_figs left (outside repo, noted in REPORT) |
| 9 | NIT | `--dry-run` still builds `_view_<v>` farms (run_dataset.py:388-392). | fixed in step F |
| 10 | NIT | docs/board_contacts.md:109,112,154 cite run_gt_contact at b0b3dce (fixed in working tree by step D); tools/human_optim/contacts.py:12 and stages.py:247 (frozen) mention score_contacts. | docs committed in 5/5; core comments left |

## Verified OK
- Core: stages.py one import + fallback lines 780/783; viewer/loading.py import, docstring, default at 420; ground.py docstring only; scripts/stages/* and tools/optim.py untouched.
- Resolver: no code/TOML/shell value resolves data/, out/, peter/ against repo or cwd at b0b3dce.
- Import hygiene: all 45 changed scripts import from a foreign cwd; `tools` resolves to the repo package, `_common` to diagnostics; no cycles; no basename collisions. contact_anything scripts import with their venv.
- Runner: datasets.toml matches old constants value by value; argv per stage identical incl. order; no env passing in either; manifest hash/name, shard slicing, mask-reuse and extra-view link layout identical; clip discovery identical.
- Exporters: old vs new re-run on one clip per case — identical keys/dtypes/shapes/values.
- Trashed files: only retired-marked references, external docstrings, and _talk_figs.
- Configs: load_config OK on every remaining human_optimization*.toml and peter_gt/human_gt.toml; no reference to a trashed config name.
- .gitignore: zero tracked binaries; trash/ ignored (954 MB).
- Runs: pytest 250 passed; dry-runs for climb_wall_3, opencap edgev, gt_contact, campus_board 2-view; --help of the six new CLIs; nothing written.

## Not verified
No real GPU stage through run_dataset; exporters one clip per case (reports I/S did the full sets); symlink scan time-bounded.
