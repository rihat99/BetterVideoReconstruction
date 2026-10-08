# Report P — purge (2026-10-08)

Repo: /home/rikhat.akizhanov/better/BetterVideoReconstruction (main). Nothing committed, nothing `rm`-ed.
Result: all 8 steps done, nothing stopped. Full test suite **250 passed in 38.10s** (`CUDA_VISIBLE_DEVICES=2 uv run --no-sync pytest tests/ -q -p no:cacheprovider`, exit 0).

## Deviation to note
- `trash/` was **not** gitignored (`git check-ignore trash` exit 1; no `trash/` dir existed). I used the user's global `trash` command (`~/.local/bin/trash`, same destination `<repo>/trash/` as trash.py), which appended `trash/` to `.gitignore` (now line 258). `git check-ignore -v trash` now matches `.gitignore:258:trash/`.

## Moved (plain `mv`) — scratch/ -> peter/_provenance/
- 13 scripts: import_climb_wall_3, trim_climb_wall_2, trim_climb_wall_3, cw2_wall_datasheet, cw3_wall_datasheet, cw2_gravity_fit, cw3_gravity_fit, cw2_wall_fit, cw2_wall_geometric, wall_check_climb_wall_2, climb_wall_2_counter, climb_wall_3_counter, cw2_boards_before_after (.py)
- 6 arrays: R_cw2_wall_to_world{,_datasheet,_geometric}.npy, R_cw3_wall_to_world_{00,16,32}.npy
- 2 dirs: cw2_gravity/ (23M), cw3_gravity/ (26M) — both under 50 MB
- New: peter/_provenance/README.md (5 lines; notes the scripts resolve `parents[1]` as repo root, so they do not run as-is from here — verified by grep)

## Trashed -> /home/rikhat.akizhanov/better/BetterVideoReconstruction/trash/
- Dirs: scratch/ (remaining 32 entries, untracked), benchmark_tables/ (8 tracked), experiments/ (7 tracked)
- Root: dataset_split_assign_20260913_105412.json (tracked)
- Scripts (11, all tracked): run_flat_torque.py, diagnostics/motion_smoothness.py, diagnostics/compare_climb_wall_2.py, emdb_split_clips.py, emdb_to_smplx.py, import_peter.py, backfill_static_camera.py, score_contacts.py, diagnostics/contacts_control.py, run_data2.py, run_athletics_chain.sh
- Configs (11, all tracked): human_optimization_floorplane_{mass,edge,edge3k,massedge,massedge3k,massedgev}.toml, human_optimization_labvalidation.toml, pipeline_data2.toml, corpus_athletics.toml, corpus_athletics_pommel.toml, pipeline_parkour.toml
- Backup copy (cp, not a removal): trash/peter_README.before_purge_20261008.md (peter/ is gitignored, so no git history for its README)
- Remaining configs/: corpus, human_optimization, human_optimization_peter, human_optimization_floorplane, human_optimization_floorplane_edgev, pipeline, pipeline_peter, pipeline_opencap

## pipeline_parkour vs pipeline_peter check (step 5)
`diff` shows: header comments (lines 1-5), the inline comment on `top_k = 1` (value identical), and `[vggt] frame_chunk/token_chunk` (0/0 vs 128/65536). Parsed-TOML comparison (105 keys each): only `vggt.frame_chunk` and `vggt.token_chunk` differ. Proceeded.

## Files edited (line refs are post-edit)
| file | lines | change |
|---|---|---|
| .gitignore | 258 | `trash/` appended by the trash command |
| scripts/peter_sensor_forces.py | 66, 67, 89, 91 | comment `scratch/` -> `<data root>/peter/_provenance/` |
| tools/geometry/ground.py | 13 | docstring `scratch/cw2_gravity_fit.py` -> `<data root>/peter/_provenance/cw2_gravity_fit.py` |
| scripts/assign_dataset_split.py | 114, 13 | rollback JSON now `paths.scenes_db.parent / f"dataset_split_assign_{stamp}.json"`; docstring "next to the run logs" -> "next to ``scenes.db``" |
| scripts/run_parkour.py | 112, 22 | `--config` default and docstring -> `configs/pipeline_peter.toml` |
| scripts/opencap_sensor_forces.py | 41, 46 | example run `human_optim_labvalidation` -> `human_optim_floorplane` |
| scripts/export_opencap_mot.py | 48, 53 | same |
| scripts/diagnostics/compare_opencap_forces.py | 71 | same |
| docs/contacts.md | 148 | "It is on in ...labvalidation.toml" -> "It was on in ... (retired 2026-10-08, see git history)"; evidence paragraph kept |
| docs/fusion.md | 249-254 | backfill paragraph to past tense + "(retired 2026-10-08 once the backfill was done, see git history)" |
| CLAUDE.md | 76-77, old 166 | dropped the `score_contacts.py` parenthetical and its layout-tree line |
| docs/force_contacts.md | 382 | "reproduced all of this; script retired 2026-10-08, see git history." |
| scripts/diagnostics/compare_climb_wall_3_runs.py | 32, 121 | `compare_climb_wall_2.py` citations marked retired |
| configs/pipeline_opencap.toml | 2 | header: pipeline_parkour.toml marked retired |
| configs/human_optimization_floorplane.toml | 1 | header: labvalidation.toml marked retired |
| configs/human_optimization_floorplane_edgev.toml | 3, 5 | massedgev twin and labvalidation.toml marked retired |
| peter/README.md (gitignored) | 5, 237, 267-268, 285, 309-310, 331, 339-340, 346 | `scratch/...` -> `_provenance/...` (all 11 cited paths verified to exist); import_peter.py marked retired |
| git index | — | `git add scripts/label_amass.py` (status `A`) |

## Verification
- `git status`: 16 M, 38 D (8 benchmark_tables + 7 experiments + 11 configs + 11 scripts + 1 json), 1 A (label_amass.py), `?? agent_notes/`.
- pytest: 250 passed, 0 failed, 0 errors, 38.10s.
- `uv run --no-sync python scripts/assign_dataset_split.py --help` exit 0; rollback dir resolves to /home/rikhat.akizhanov/better/data/ClimbingVideos/scenes (exists, scenes.db present). Not run for real (it writes the DB).
- `run_parkour.py --help` exit 0; opencap_sensor_forces, export_opencap_mot, peter_sensor_forces, tools.geometry.ground import OK; all 8 edited .py files `ast.parse` OK; the 3 edited TOMLs + pipeline_peter.toml parse with tomllib.

## Leftover references, deliberately left
Core (do not edit):
- tools/human_optim/contacts.py:12 — "(``scripts/score_contacts.py``) still imports" (now stale)
- tools/human_optim/stages.py:246 — comment "shared with estimate_contacts / score_contacts"
run_opencap (rewritten later):
- scripts/run_opencap.py:44 — usage example `--variant labvalidation`
- scripts/run_opencap.py:254 — `f"human_optimization_{args.variant}.toml"`: `--variant labvalidation` and the 6 trashed floorplane variants now have no config file (would fail at load time)
Tests (pass today; they use "labvalidation" only as a plain variant name and exit before the config is read):
- tests/test_support_scene.py:369, 391, 402, 407, 408, 418, 426
Code output paths (a behavior change, not a text fix; these scripts recreate `benchmark_tables/` when run):
- scripts/diagnostics/compare_climb_wall_3_runs.py:41 (docstring), :67 `OUT = _REPO / "benchmark_tables"`
- scripts/diagnostics/score_combined_table.py:24-25 (docstring), :52 `OUT = _REPO / "benchmark_tables"`
Kept on purpose (not references to removed files):
- configs/human_optimization_floorplane.toml:396 and _edgev.toml:403 "(labvalidation run)" — the on-disk run, not the config
- scripts/diagnostics/score_gt_pose.py:59, compare_climb_wall_3_runs.py:72 — `human_optim_flat_torque` is the on-disk run dir (68/68), not the trashed script
- scripts/peter_sensor_forces.py:18 "scratchpad ``wall_total_fit.py``" — a session scratchpad, never in scratch/
- .gitignore:238 `scratch/` — still useful to keep a future scratch/ out of git

## Stopped on
Nothing.

## Open questions
1. benchmark_tables/ writers (above): redirect their `OUT` to a data/results dir, or leave for the later step?
2. run_opencap.py:254 variant -> config lookup now breaks for `labvalidation` and the 6 trashed variants; the planned runner rewrite should drop them from any variant list/help.
3. Core docstring tools/human_optim/contacts.py:8-12 is now stale: it keeps the per-joint pieces because score_contacts.py imports them. The trashed script also imported `_ANN_TO_SMPLX`, `_hand_smplx_joints`, `resolve_annotation_path` from tools/human_optim/stages.py (trash/score_contacts.py:49). Whether those now have no user was NOT checked; core is frozen, so I only flag it.
4. Should `trash/` live in .gitignore (done now by the user's trash command) or should removals go to the workspace trash /home/rikhat.akizhanov/better/trash (the user's memory note says "../trash")?
