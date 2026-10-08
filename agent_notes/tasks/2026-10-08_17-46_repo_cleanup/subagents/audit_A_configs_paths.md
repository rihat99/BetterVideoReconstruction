# Audit A — configs, hardcoded paths, root files, tests (read-only, mars_aux, 2026-10-08)

Repo: `/home/rikhat.akizhanov/better/BetterVideoReconstruction`, main @ 5a1fc9a. Untracked: `agent_notes/`, `scripts/label_amass.py`.

## Key findings
1. The floorplane configs were run on OpenCap (`/home/rikhat.akizhanov/better/data/LabValidation_withVideos/processed/out/`), not on climb_wall_3. The commit message of ce0498f is misleading. peter/ holds no floorplane runs.
2. Three floorplane configs are live: `floorplane` (161 clips, the reference that later method scripts read), `floorplane_edgev` (64 clips, the last variant run, Sep 17 19:58) and `floorplane_massedgev` (161 clips). All three are rows in `_methods/methods_table.md`.
3. edgev's knobs (force_edge 3000, edge_speed 0.8) were carried into `configs/pipeline.toml` [bedlam_forces] at lines 146-152 (commit 5a1fc9a). The other five variants (mass, edge, edge3k, massedge, massedge3k) were 5-clip ablations with no references anywhere.
4. No committed code reproduces the variant runs on disk. They are kindyn-only farms (geometry linked to `../_floorplane/geometry`, kinematics and contacts linked from `human_optim_floorplane`). `run_opencap.py --variant floorplane_edgev` would run a different chain.
5. `human_optimization_floorplane.toml` must stay. tests/test_support_scene.py:41 loads it, and run_opencap.py:104-107 has the `floorplane` chain built in.
6. `labvalidation` was superseded on the same day. floorplane = labvalidation + `floor_gate_m=0` + heel sole offset 0.03. Dropping it means editing docs/contacts.md:148 and three docstrings.
7. `pipeline_data2.toml` is stale: last used Aug 25 and read only by run_data2.py. The parkour, peter and opencap pipeline configs are live and differ only in VGGT chunking.
8. Code that breaks after the move: about 45 hardcoded sites (see 2a). Plus tools/viewer/loading.py:420 (video pane goes blank) and two CLI defaults.
9. tools/human_optim/stages.py:779-782 is frozen core code and falls back to repo `data/` and `data_2/`. Harmless while each `.npz` sits next to its video. score_contacts.py:265/528 will silently score nothing without `--source`.
10. Symlinks that will break: 544 absolute links in `peter/climb_wall_3_gt/others/_farm` and 68 in `../data/Estimating-3D-Motion-Forces/data/climb_wall_3/videos`. All 1448 relative links inside peter/ stay inside peter/.
11. Other repos also read these folders: contact_anything (score_rig, fig_climb_wall_3, trivial_baselines, prepare_wild) and BetterContactAnnotator (`RECON_OUT` default is BVR/out).
12. Tests: 250 collected, 0 errors. No test touches data/, data_2/, out/, out_2/ or peter/, so the suite cannot catch the move. test_bedlam_forces needs the repo root as cwd.
13. Run tests with `uv run --no-sync`. INSTALL.md:46-48 says a plain `uv run` re-syncs and removes the hand-installed better_robot and better_human.
14. Root file `dataset_split_assign_20260913_105412.json` is tracked and read by nothing. It is a scenes.db rollback record, the same kind of file commit 3043dcd already removed.

## 1. configs/

### 1a. Who references each config
Every config is tracked. All 11 human_optimization*.toml and `scripts/peter_gt/human_gt.toml` load via `tools.human_optim.config.load_config` on CPU.

| config | commits (last) | code / test refs | doc refs |
|---|---|---|---|
| pipeline.toml | 18 (5a1fc9a, Oct 5) | pipeline.py:213, view_3d.py:239, backfill_static_camera.py:130, label_amass.py:352, stages/{estimate_scale:593, fuse_geometry:469, estimate_gravity_geocalib:360} defaults, corpus*.toml:13-14 `pipeline_config` | CLAUDE.md, README |
| pipeline_peter.toml | 6 (ca7f72b, Sep 11) | run_peter.py:260; copied into peter_gt/vggt_gt.toml:2 and pipeline_gt.toml:2 | — |
| pipeline_parkour.toml | 1 (c5f9a26, Aug 24) | run_parkour.py:112 | header of pipeline_opencap.toml:2 |
| pipeline_opencap.toml | 1 (ce0498f) | run_opencap.py:235 | — |
| pipeline_data2.toml | 3 (860cbd3, Aug 25) | run_data2.py:111 only | — |
| corpus.toml | 7 (867de9a, Sep 7) | corpus.py:36, score_contacts.py:53, assign_dataset_split.py:35, calibrate_contact_confidence.py:45, diagnostics/score_test_contacts.py:67 | CLAUDE.md |
| corpus_athletics.toml, corpus_athletics_pommel.toml | 1 each (eac87b4, Sep 11) | used only via `corpus.py --config`; usage line in run_athletics_chain.sh:4; derived copies in scratch/ dated Sep 9 | — |
| human_optimization.toml | 52 (a3bb5fd) | stages/human_optimization.py:47, pipeline.py:67, corpus.toml:14, score_contacts:52, recompute/calibrate_contact_confidence, contacts_control:87, label_bedlam2:297, label_amass:368, pipeline.toml:144, tests test_bedlam_forces:43,89 and test_contact_confidence:145 | CLAUDE.md |
| human_optimization_peter.toml | 13 (ce0498f) | run_peter:263, run_parkour:115, run_opencap:254 (default), run_gt_contact:132, run_flat_torque:95, stages/measure_contacts_boards:54, pipeline_opencap.toml:146; base of peter_gt/human_gt.toml | docs/board_contacts.md:119, contacts.md:53, contact_fusion.md:91 |
| human_optimization_labvalidation.toml | 1 (ce0498f) | only through `run_opencap --variant labvalidation` (run_opencap.py:254). Tests pass that argv but exit before reading the file | docs/contacts.md:148; docstrings opencap_sensor_forces:41,46, export_opencap_mot:48,53, compare_opencap_forces:71 |
| human_optimization_floorplane.toml | 1 (ce0498f) | tests/test_support_scene.py:41 (5 tests use it); run_opencap `VARIANT_CHAINS["floorplane"]` (:104-107) | build_support_scene.py:36,39 |
| floorplane_{mass,edge,edge3k,edgev,massedge,massedge3k,massedgev} | 1 each (ce0498f) | none, apart from the `human_optimization_{variant}.toml` naming pattern in run_opencap.py:254 | none |

Lineage (config headers, confirmed by TOML diff):
- human_optimization → peter: chamfer [10,10,0,0] becomes [0,0,0,0]; `contacts.frames` limited to hands + feet; adds `[boards]`.
- peter → labvalidation: frames become feet + `sit_left/right` with no hands; chamfer back to [10,10,0,0]; adds `contacts.floor_gate_m=0.17`.
- labvalidation → floorplane: `floor_gate_m` 0.17 → 0; adds `contacts.sole_offset_m.heel=0.03`.

### 1b. Floorplane variants vs `floorplane` (key-level)

| variant | delta vs floorplane |
|---|---|
| mass | `stages.kindyn_1.mass_from_meta=true` |
| edge | `loss_weights.dynamics.force_edge=[0,0,0,120]` |
| edge3k | force_edge 3000 |
| edgev | force_edge 3000, `edge_speed_m_s=0.8`, `mass_from_meta=false` |
| massedge | mass_from_meta + force_edge 120 |
| massedge3k | mass_from_meta + force_edge 3000 |
| massedgev | mass_from_meta + force_edge 3000 + edge_speed 0.8 |

### 1c. What was actually used (on disk, LabValidation `processed/out/`)

| run dir | clips with kindyn_1 | newest kindyn_1 | downstream use |
|---|---|---|---|
| human_optim (peter cfg) | 12 | Sep 16 03:07 | — |
| human_optim_labvalidation | 12 | Sep 16 14:56 | `_compare_labvalidation` |
| human_optim_floorplane | 161 | Sep 17 12:31 | default `--kindyn-run` of methods/opencap_frames35_export:225 and opencap_learned_export:327; contacts in opencap_estmf_inputs:126; compare_opencap_plates:394; `_methods/learned_root/*` farms link `_floorplane` (used Sep 21); methods_table row "floorplane baseline" |
| human_optim_floorplane_massedgev | 161 | Sep 17 16:25 | `_plates_..._massedgev_clips_natural`; methods_table row |
| human_optim_floorplane_edgev | 64 | Sep 17 19:58 (last variant) | `_plates_floorplane_edgev_clips_natural` (20:17); methods_table row; knobs adopted in pipeline.toml:146-152 |
| mass / edge / massedge | 5 | Sep 17 15:21 | `_plates_*` only |
| edge3k / massedge3k | 5 | Sep 17 15:29-15:30 | `_plates_*` only |

Caveat: the variant farms link `geometry -> ../_floorplane/geometry` and kinematic/contact files into `human_optim_floorplane` — a kindyn_1-only rerun. No committed code builds this layout (`run_opencap.build_farm` :129-154 links the clip's real geometry for any variant not in `VARIANT_CHAINS`; `VARIANT_STEPS` reruns kinematics_1 through kindyn_1). So `--variant floorplane_edgev` would NOT reproduce the edgev numbers.

### 1d. Recommendation

| config | rec | evidence |
|---|---|---|
| pipeline.toml, human_optimization.toml, human_optimization_peter.toml, corpus.toml | keep | above |
| human_optimization_floorplane.toml | keep | test + VARIANT_CHAINS + the reference run for methods |
| floorplane_edgev | keep | last-run variant, row in the final table, knobs adopted into pipeline.toml |
| floorplane_massedgev | keep, or merge-into edgev | differs only by `mass_from_meta=true`; merge needs derive-at-launch like run_flat_torque.py:18-21, because the loader has no defaults |
| floorplane_{mass,edge,edge3k,massedge,massedge3k} | drop | 5-clip ablations, no references |
| human_optimization_labvalidation.toml | drop, or keep as a record | superseded by floorplane; then fix docs/contacts.md:148 and the docstrings listed above |
| pipeline_opencap.toml, pipeline_peter.toml | keep | live runners |
| pipeline_parkour.toml | keep, or merge-into pipeline_peter | identical except VGGT `frame_chunk/token_chunk` 0 vs 128/65536 (numerics differ ~3e-4 at bf16). Parkour last ran Sep 21 |
| pipeline_data2.toml | drop with run_data2.py, or keep and fix paths | stale since Aug 25; runner hardcodes repo `data_2/` and `out_2/` |
| corpus_athletics(.toml, _pommel) | keep while the athletics corpus matters | only record of the athletics and pommel recipes; ran Sep 9 |

Note: `[human_optimization].config` in pipeline_opencap.toml:146 is read only by pipeline.py. No run_*.py calls load_toml.

## 2. Hardcoded paths

Grep: 426 lines in 118 files. Python lines: 6 CLI defaults, 152 in code, 147 docstrings, 26 help text, 7 comments. Most in-code lines are absolute `/home/...` or `../data/...`, unaffected by the move.

Path resolution: no shared root resolver. ~60 files each define `_REPO_ROOT`/`_REPO`/`REPO`/`ROOT = Path(__file__).resolve().parents[N]`. `tools/human_optim/paths.py` only resolves paths inside one clip. `tools/config.py` has no path logic. `corpus.py:213 _repo_path` resolves relative to the repo root.

cwd-relative (need cwd = repo root): pipeline.py:104 and per-stage `--config`; view_3d.py:268 `[viewer].video_dir`; calibrate_contact_confidence.py:408 (`../data/ClimbingVideos`); tests/test_bedlam_forces.py:43,89; stages/estimate_contacts.py:118 silently uses `{}` when the config path is missing.

### 2a. Sites that break after the move

| file:line | snippet | class | after the move |
|---|---|---|---|
| tools/viewer/loading.py:420 | `video_dir ... else _REPO_ROOT / "data"` | b (overridable via `[viewer].video_dir`) | no video pane for out/ clips; no crash |
| tools/human_optim/stages.py:779,782 (FROZEN) | `_REPO_ROOT/"data"/<stem>.npz`, else `"data_2"` | b | fine if video and npz move together (:776 checks `<source>.npz` first) |
| scripts/score_contacts.py:265,528 | `resolve_annotation_path(source or pred, None)` | b (indirect) | without `--source`: skips scoring silently |
| scripts/score_frame_contacts.py:138 | `_REPO_ROOT / "data" / f"{scene}.npz"` | b | FileNotFoundError |
| scripts/diagnostics/contacts_control.py:429,432 | `--root` default `_REPO_ROOT/"out"`, `--annotations` default `_REPO_ROOT/"data"` | a | defaults point at missing dirs |
| scripts/diagnostics/render_prediction_overlays.py:90,95,96 (+168,189,191) | `PETER=ROOT/"peter"`, `OUT_ROOT`, `DATA_ROOT` | b | breaks; constants imported by render_kindyn_overlays:121-131, render_sup_force_overlays:68-74, compare_climb_wall_2:39, render_parkour_method_overlays:41-43 |
| scripts/diagnostics/score_prediction_contacts.py:42-44 | `PETER = parents[2]/"peter"` | b | breaks; imported by score_pipeline_contacts:47 |
| scripts/run_data2.py:43-44 | `_REPO_ROOT/"data_2"`, `"out_2"` | b | SystemExit "no clips" |
| scripts/run_peter.py:68 (267,276-277) | `_PETER = _REPO_ROOT/"peter"` | b | breaks |
| scripts/run_flat_torque.py:98-99 | `_REPO_ROOT/"peter"/...` | b | breaks |
| scripts/run_gt_contact.py:87-88 | `_REPO_ROOT/"peter"/f"out_{dataset}_single"` | b | peter datasets break |
| scripts/peter_sensor_forces.py:134,136 | `REPO/"peter"/args.tree` | b | breaks |
| scripts/import_peter.py:301 | `--dest` default `_REPO_ROOT/"peter"` | a | writes a new repo peter/ |
| scripts/parkour_physpt_inputs.py:53-54 | climb_wall_3 GT and single tree | b | breaks |
| scripts/methods/cw3_estmf_inputs.py:41-42 | SINGLE, GT | b | breaks |
| scripts/diagnostics/compare_climb_wall_3_runs.py:66 | POSE_JSON | b | breaks |
| scripts/diagnostics/compare_gt_contact.py:191 | `--tree` default | a | default breaks |
| scripts/diagnostics/render_cw3_estmf_overlays.py:33-36 | GT, VIDEOS, SINGLE, OUT_DIR | b | breaks |
| scripts/diagnostics/score_combined_table.py:51 | CW | b | breaks |
| scripts/diagnostics/score_gt_contacts.py:69-71 | GT_ROOT, OURS_ROOT, OTHERS | b | breaks |
| scripts/diagnostics/score_gt_pose.py:49-51 | same | b | breaks |
| scripts/diagnostics/score_init_world.py:40 | CW_RESULTS | b | breaks |
| scripts/peter_gt/calibrate_cameras.py:64-65 | ORIG, GT | b | breaks |
| scripts/peter_gt/extract_frames.py:24-25 | ORIG, OUT | b | breaks |
| scripts/peter_gt/fit_multiview.py:82 | HUMAN_ROOT | b | breaks |
| scripts/peter_gt/overlay_human.py:33 | HUMAN_ROOT | b | breaks |
| scripts/peter_gt/prepare_baselines.py:26-29 | CLIPS, GT, SINGLE, FARM | b | breaks |
| scripts/peter_gt/prepare_human.py:54-57 | 4 roots | b | breaks |
| scripts/peter_gt/rebuild_scene_calibrated.py:76 | HUMAN_ROOT | b | breaks |
| scripts/peter_gt/reconstruct_vggt.py:46 | GT_ROOT | b | breaks |
| scripts/peter_gt/run_human.py:43 | HUMAN_ROOT | b | breaks |
| scripts/peter_gt/view_gt.py:22 | GT_ROOT | b | breaks |
| scripts/peter_gt/run_all_clips.sh:13,21 | `peter/climb_wall_3/*/` after `cd` | b | breaks |
| scripts/peter_gt/run_baselines.sh:12-15 | `$REPO/peter/...` (:17 sources `/data3/...` conda, already stale) | b | breaks |

Already stale before the move: `peter/climb_wall` and `peter/out_climb_wall_single` (render_prediction_overlays.py:93-94, peter_sensor_forces default `--tree`) do not exist.

### 2b. Documentation only (class c)
README.md:75,78 is a runnable example that will stop working (`--source data/boulder_1.mp4 --save-dir out/`, `view_3d.py out/boulder_1/`).
Other doc-only mentions: view_3d.py:65,130; backfill_static_camera.py:20,128; diagnostics/scene_quality.py:28; run_peter.py:1,3,247; run_data2.py:1,3; run_flat_torque.py:91; run_gt_contact.py:126; peter_sensor_forces.py:4,131; import_peter.py:1,10; score_frame_contacts.py:5,131; score_contacts.py:6,643; stages/human_optimization.py:97 (frozen); tools/human_optim/{stages.py:772,797; kindyn.py:712} (frozen); tools/viewer/loading.py:398,411; tests/test_body_density.py:14; diagnostics and peter_gt docstrings; docs/quality.md:3,109; docs/fusion.md:177; docs/human_optimization.md:281; CLAUDE.md:95.
Generic `out/<stem>/` contract mentions (~150) describe a user-chosen `--save-dir`; they do not break.

### 2c. Not affected by the move
- Absolute `/home/rikhat.akizhanov/better/...`: run_opencap:60, run_parkour:45, run_gt_contact:78, import_opencap:107, import_parkour:57, opencap_sensor_forces:66, parkour_sensor_forces:41, compare_opencap_{forces,plates}:96, render_kindyn_overlays:132, emdb_to_smplx:31, import_bedlam2:119, label_amass:64, tools/body.py:38,51, tools/viewer/loading.py:43, tests/test_opencap_forces.py:42.
- `_REPO.parent/"data"/...`: scene_db.py:23,32,44, import_bedlam2:113-114, methods/* and parkour_* scripts, diagnostics/{score_baseline_*, score_combined_table:50, score_init_world:39, score_parkour_pose:39, compare_parkour_methods:87-89, render_cw3_estmf:31}, corpus*.toml `data_root`.
- Dead but harmless: `_REPO_ROOT.parent/"better_human"/...` in tools/human_optim/stages.py:41, stages/estimate_scale.py:81, score_contacts.py:54 (folder does not exist; real one is `BetterHuman`; path ignored per tools/smplx_robot/model.py:22).

### 2d. Symlinks and outside consumers (these break)
- `peter/climb_wall_3_gt/others/_farm/**`: 544 absolute links into `.../BetterVideoReconstruction/peter/climb_wall_3/...` (built by peter_gt/prepare_baselines.py).
- `/home/rikhat.akizhanov/better/data/Estimating-3D-Motion-Forces/data/climb_wall_3/videos/*.mp4`: 68 absolute links into repo peter/.
- The other 1448 relative links in peter/ resolve inside peter/ and are move-safe. `data/body_models/*` (2 links) point outside the repo; nothing references them.
- contact_anything/scripts: score_rig.py:463-464 (default), fig_climb_wall_3.py:57-58 (hardcoded), trivial_baselines.py:49,75-82 (hardcoded), prepare_wild.py:115 (`--moving-root` default `BVR/out_2`), score_parkour_rig.py:47 `_BVR`.
- BetterContactAnnotator/backend/app/config.py:69: `RECON_OUT` default `BVR/out` (env overridable).
- Same filesystem, so `mv` is a rename. Sizes: peter 69G, out 11G, out_2 2.3G, data 32M, data_2 37M.

### 2e. Fix options
1. One resolver module (e.g. `tools/paths.py`: `REPO_ROOT`, `DATA_ROOT = env BVR_DATA_ROOT or REPO_ROOT.parent/"data"`, `EXAMPLES`, `PETER`), replace ~45 sites in ~30 files.
2. Fix only each family's root constant (render_prediction_overlays, score_prediction_contacts, score_gt_contacts + runners + peter_gt): fewer lines, root still duplicated ~25 times.
3. Symlinks: rejected by the user.
Either way the 612 absolute symlinks must be re-pointed or rebuilt, and the two outside repos' defaults updated.

## 3. Root loose files
- `dataset_split_assign_20260913_105412.json`: tracked (c20756a, 97f424f), read by nothing; written by assign_dataset_split.py:114. Rollback record for `../data/ClimbingVideos/scenes/scenes.db`.
- README.md: broken example at :75,78; uses plain `uv run`.
- INSTALL.md: :23-25 "six submodules" incl. vggt-omega; :52-53 "VGGT-Omega is an editable submodule". Reality: 5 submodules; vggt-omega is vendored.
- CLAUDE.md: says "experimental branch", only `main` exists. Lists 3 configs.
- pyproject.toml: no `[project.scripts]`; transformers/accelerate/bitsandbytes kept "for the legacy VLM stage" with no direct import anywhere.
- Submodules all used: sam3 (track_sam3.py), sam-3d-body (estimate_body_sam3d.py, convert_mhr_to_smplx.py; `.venv/.../sam-3d-body.pth` holds an absolute repo path), sapiens2, MHR, geocalib; vggt-omega vendored (tools/adapters/vggt_omega.py, peter_gt/reconstruct_vggt.py).
- .claude/: worktrees/ empty; gitignored.
- .gitignore stale entries: configs/corpus_onepergpu|contacts2|contacts2_onepergpu|retry.toml, `!tools/robot_motion/...`, bare `peter` line in the Django block.
- Dirs to remove: experiments/ (7 tracked), benchmark_tables/ (8 tracked; recreated by compare_climb_wall_3_runs:181 and score_combined_table:151). scratch/ untracked; peter_sensor_forces.py:66-91 and tools/geometry/ground.py:13 cite scratch/cw2_*.py and cw3_*.py as provenance of hardcoded calibration constants. run_athletics_chain.sh:14,23-25 recreates scratch/.

## 4. tests/
- `uv run --no-sync pytest tests/ -q --co -p no:cacheprovider`: 250 tests, 22 files, 0 collection errors.
- No test reads data/, data_2/, out/, out_2/ or peter/. The suite cannot detect move breakage.
- Outside deps (not moving): BetterHuman `SMPLX_NEUTRAL.npz` and `better_contacts.json` (tools/body.py:38,51); test_opencap_forces:139 reads LabValidation, skips if missing.
- Config deps: test_support_scene.py:41 (floorplane); test_bedlam_forces.py:43,89 (cwd-relative); test_contact_confidence.py:145.
- Scripts imported by tests (do not rename/trash without updating): run_opencap, build_support_scene, export_opencap_mot, opencap_plates, opencap_sensor_forces, diagnostics/compare_opencap_forces, import_opencap, sync_opencap, scripts.corpus, scripts.recompute_contact_confidence, scripts.stages.{estimate_contacts, estimate_scale, fuse_geometry, human_optimization}.
- "Nothing broke" check: full suite with `--no-sync`; grep for `"peter"`, `"data_2"`, `"out_2"`, `_REPO_ROOT / "data"`, `_REPO_ROOT / "out"`; `--help` of each changed script; `find -xtype l` on moved trees; open one viewer clip.

## Open questions
1. Keep edgev and massedgev as two files, or derive massedgev at launch? Commit the kindyn-only variant farm to run_opencap?
2. Drop labvalidation, or keep as the documented step between peter and floorplane?
3. Drop pipeline_data2 and run_data2.py, or fix their paths to point at examples?
4. Fix the outside repos (contact_anything, BetterContactAnnotator) and the 612 absolute symlinks as part of this cleanup?
5. Keep the scratch/ calibration-provenance scripts somewhere before scratch/ is trashed?

Not verified: no tests run (collection only), no GPU job, no script run after a simulated move.
