# Report M — move data out of the repo, one path resolver (mars_top_opus, 2026-10-08)

Result: all 5 dirs moved with plain `mv` on the same filesystem. 625 symlinks re-pointed, 0 broken links. One resolver, `tools/paths.py`, used by ~50 files. Sibling repos fixed. Full suite 250 passed.

## 1. Moves
Pre-checks: no process had cwd/open file under the dirs; all GPU processes belong to other users; same device (37); DATA/peter and DATA/examples did not exist.

| from (BVR/) | to | size |
|---|---|---|
| data | /home/rikhat.akizhanov/better/data/examples/data | 32M |
| data_2 | .../data/examples/data_2 | 37M |
| out | .../data/examples/out | 11G |
| out_2 | .../data/examples/out_2 | 2.3G |
| peter | /home/rikhat.akizhanov/better/data/peter | 69G |

Repo root now: agent_notes CLAUDE.md configs docs INSTALL.md pyproject.toml README.md scripts tests third_party tools trash uv.lock.

## 2. Resolver — tools/paths.py
Constants: `REPO_ROOT`, `DATA_ROOT = $BVR_DATA_ROOT or REPO_ROOT.parent/"data"`, `EXAMPLES`, `PETER`, `LABVALIDATION`, `PARKOUR`, `CLIMBING`, `ESTMF`. All exist; env override verified.
Import pattern: `sys.path.insert(0, repo_root)` then `from tools.paths import X` — required because the venv's site-packages holds detectron2's own top-level `tools` package. Files with a local `PARKOUR`/`ESTMF` name use `from tools import paths`.

## 3. Edited files (60 + tools/paths.py)
Core (path defaults only): tools/human_optim/stages.py:28,780,783 (fallbacks → EXAMPLES/"data", EXAMPLES/"data_2"; docstrings :772,:797 left); tools/viewer/loading.py (default video dir → EXAMPLES/"data"; boulder_1 video pane loads).
§2a sites: score_frame_contacts.py; diagnostics/score_prediction_contacts.py (PETER from resolver); diagnostics/render_prediction_overlays.py (PETER, OUT_ROOT, DATA_ROOT; dead WALL_SOURCE/WALL_ROOT → climb_wall_3 trees); render_kindyn_overlays.py + render_sup_force_overlays.py (removed the `(WALL_ROOT, wall_clip)` TREES row, see deviations; PARKOUR_SOURCE); run_peter.py, run_gt_contact.py (root constants); peter_sensor_forces.py (PETER; default --tree out_climb_wall_3_single; dropped RIGS["climb_wall"]); parkour_physpt_inputs.py, methods/cw3_estmf_inputs.py; compare_climb_wall_3_runs.py + score_combined_table.py (OUT = PETER/"benchmark_tables"); compare_gt_contact.py, render_cw3_estmf_overlays.py, score_gt_contacts.py, score_gt_pose.py, score_init_world.py; all peter_gt/*.py (PETER); peter_gt/prepare_human.py:203,206,282 `relative_to(_REPO)` → `relative_to(PETER.parent)` (else ValueError); peter_gt/run_all_clips.sh + run_baselines.sh (`${BVR_DATA_ROOT:-../data}/peter`; bash -n ok; 68 clips listed).
§2c sites → resolver: run_opencap, run_parkour, import_opencap, import_parkour, import_bedlam2, opencap_sensor_forces, parkour_sensor_forces, scene_db (inserts repo root itself), parkour_prepare_interactvlm, parkour_estmf_inputs, methods/{contact_recognizer, opencap_estmf_export, opencap_estmf_inputs, opencap_physpt_export, opencap_physpt_inputs}, diagnostics/{compare_opencap_forces, compare_opencap_plates, compare_parkour_methods, score_baseline_contacts, score_baseline_pose, score_parkour_pose}, tests/test_opencap_forces.py.
Docs: README.md:75,78 (../data/examples/...); CLAUDE.md:94, docs/human_optimization.md:281; view_3d.py:64,130, diagnostics/scene_quality.py:28, tests/test_body_density.py:14.
.gitignore: removed data/ data_2/ out/ out_2/ peter/ peter scratch/, the 4 configs/corpus_* lines, the 2 !tools/robot_motion lines; added agent_notes/**/vis/ and agent_notes/**/artifacts/.

## 4. Symlinks
| set | before | after |
|---|---|---|
| DATA/peter absolute links into BVR/peter | 544 | 0 |
| DATA/Estimating-3D-Motion-Forces/data/climb_wall_3/videos | 68 | 0 |
| DATA/willd_videos links into BVR data/out/out_2 (not in brief) | 13 | 0 |
`find DATA/peter DATA/examples DATA/Estimating-3D-Motion-Forces DATA/willd_videos -xtype l` = 0. body_models links resolve. No other link under better/ (depth 7) pointed into the moved dirs.

## 5. Sibling repos (edited)
contact_anything (HEAD c6a5b45): score_rig.py:463-464, fig_climb_wall_3.py:57-58, trivial_baselines.py:46,49,75-76,82, prepare_wild.py:115 → `_ROOT.parent/"data"/...`. score_parkour_rig.py:47 unchanged (import path only). All --help exit 0.
BetterContactAnnotator (HEAD a528ad1): backend/app/config.py:69 RECON_OUT default → data/examples/out.
Not changed (outside line list): contact_anything/CLAUDE.md:258-259 and BetterContactAnnotator/README.md:36-37 still mention the old paths.

## 6. Verification (repo root, CUDA_VISIBLE_DEVICES=2)
- pytest: 250 passed in 37.29s, 0 skipped.
- Brief grep: empty. No `parent / "data"` outside tools/paths.py.
- --help: all 41 edited argparse scripts exit 0; 15 non-argparse scripts import cleanly; every path constant exists except peter/benchmark_tables (created on write) and scene_db.ATHLETICS_DATA_ROOT (missing before the move too).
- Unused names (AST, HEAD vs now): only R_SENSOR_TO_WORLD, CALIB_CLIPS, NOTE in peter_sensor_forces.py (kept).
- pipeline.py --skip-existing on boulder_1: run 1 — sam3, vggt skipped; sapiens CUDA OOM while loading its model (another user's PID holds 86 GB of GPU 2; precompute_runner loads before checking skips — predates this change). Run 2 without sapiens: rc=0 in 17 s, every stage "skip (output exists)".
- Side effect: the runs created DATA/examples/out/_batch_manifest.jsonl; trashed to ~/.trash.

## Deviations
1. Removed `(WALL_ROOT, wall_clip)` TREES rows in render_kindyn_overlays and render_sup_force_overlays (WALL_ROOT now equals WALL3_ROOT; the old row pointed at a missing tree and already crashed). `wall_clip()` now unused, kept.
2. prepare_human.py relative_to fix.
3. 13 willd_videos symlinks fixed beyond the brief.
4. Rewrite-list scripts got constant swaps only; docstrings still say `peter/` (run_peter:1,3,247; run_gt_contact:126; render_prediction_overlays:9,18).

## Open questions
1. peter_sensor_forces.py: R_SENSOR_TO_WORLD / CALIB_CLIPS / NOTE describe the vanished climb_wall rig — delete or keep?
2. wall_clip() in render_kindyn_overlays.py unused — drop in step S?
3. precompute_runner loads models before checking skips (sapiens/sam3d need GPU memory even for --skip-existing). Not touched (next to frozen code).
4. scene_db.ATHLETICS_DATA_ROOT points at missing data/AthleticsVideos (pre-existing).
