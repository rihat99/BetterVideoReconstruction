# Audit B — scripts and tools inventory, duplicates (read-only, mars_aux, 2026-10-08)

Repo /home/rikhat.akizhanov/better/BetterVideoReconstruction, branch `main`, HEAD 5a1fc9a.

**How it was checked.** Text-reference scanner over scripts/, tools/, tests/, configs/, docs/, CLAUDE.md, README.md, INSTALL.md, pyproject.toml, .gitignore (Python imports separated from text mentions; scratch/ and experiments/ counted separately). AST comparison of every top-level function in scripts/ and tools/. Every non-stage script imported on CPU with PYTHONDONTWRITEBYTECODE=1: all 76 OK. Commit dates via `git log -1 --format=%cs`; `2026-09-19 ce0498f` is one bulk "path move" commit.

**Abbreviations:** P/ = repo-local peter/, LV = ../data/LabValidation_withVideos/processed, PK = ../data/Parkour-dataset/processed, CV = ../data/ClimbingVideos, EST = ../data/Estimating-3D-Motion-Forces.

## Summary
1. All 76 non-stage scripts import cleanly; dead code is stale data paths, not broken imports.
2. Inventory: 35 top-level scripts, 24 diagnostics, 8 methods, 16 peter_gt, 12 core stages, 33 tools modules. Only untracked file: scripts/label_amass.py.
3. Biggest duplication: the runner family. run_data2.py and run_parkour.py are the same file with different paths; run_opencap/run_peter repeat the step tables and stage-command builder; run_gt_contact and run_flat_torque share an identical build_farm. ~1.4k lines could become one run_dataset.py plus configs/datasets.toml.
4. pipeline.py cannot replace the runners today: 8 steps only (pipeline.py:57), stops on first failure (:170), runs sam3d/smplx once per clip (:58), no sharding.
5. Method-input exporters repeat code: three estmf exporters share 3 identical functions, two PhysPT exporters share 3, two scripts build the same baseline farm layout.
6. Safe to trash now: run_flat_torque.py (ablation done, 68/68 trees), diagnostics/motion_smoothness.py, diagnostics/compare_climb_wall_2.py, emdb_split_clips.py + emdb_to_smplx.py (EMDB absent), import_peter.py (source dir gone), backfill_static_camera.py (migration done 13/13 out, 10/10 out_2).
7. Needs a decision: score_contacts.py (CLAUDE.md calls it frozen), diagnostics/contacts_control.py (cited in docs, writes into scratch/), run_athletics_chain.sh (AthleticsVideos absent, writes into scratch/), label_amass.py (untracked).
8. diagnostics/score_prediction_contacts.py has 8 importers and score_pipeline_contacts.py 5, almost all for small helpers; campus_board-era scripts survive only as libraries.
9. Risk: scratch/ holds the only copies of the climb_wall_2/3 import, trim and calibration-fit scripts, cited as the source of constants in peter_sensor_forces.py:66-91 and tools/geometry/ground.py:13.
10. Risk: core reads repo-local data/ and data_2/ (tools/human_optim/stages.py:779-782, tools/viewer/loading.py:420).
11. Four scripts write into dirs being removed and recreate them: compare_climb_wall_3_runs.py and score_combined_table.py (benchmark_tables/), contacts_control.py and run_athletics_chain.sh (scratch/).
12. Stale defaults: peter_sensor_forces.py --tree out_climb_wall_single and climb_wall roots in render_prediction_overlays.py:93-94 point at dirs that no longer exist.
13. Docs drift: CLAUDE.md omits diagnostics/, methods/, peter_gt/, every run_*.py, two stage CLIs and tools/body.py; says branch is "experimental". README documents retired export_contact_dataset.py and retired --stages dynamics.
14. peter_gt/ is a self-contained GT pipeline (entry run_human.py): nothing imports it, 10 scripts read its output tree, fit_multiview.py imports 9 private names from core.

## 1. Inventory

### scripts/ top level (35)
| path | purpose | inputs → outputs | imported by (I) / mentioned in (M) | last | verdict |
|---|---|---|---|---|---|
| _pym_preload.py | LD_PRELOAD shim for pymomentum | – | I: stages/estimate_body_sam3d, convert_mhr_to_smplx; M: CLAUDE, INSTALL, pipeline.py | 06-03 | core |
| assign_dataset_split.py | train/test split for curated scenes; writes scenes.db | CV scenes.db, corpus.toml [split] → DB + dataset_split_assign_<ts>.json | none | 09-07 | keep (one-off DB tool, ask) |
| backfill_static_camera.py | stamps static-camera flag on old results | out/, out_2/, --corpus → camera npz in place | M: docs/fusion.md:250 | 09-06 | dead? (done) |
| calibrate_contact_confidence.py | fits contact-label confidence; --write rewrites TOML | annotated CV scenes → [contact_confidence] in human_optimization*.toml | M: README:69, docs/contact_confidence.md, confidence.py:27,200, estimate_contacts.py:158, 11 TOMLs | 09-13 | keep |
| corpus.py | any stage over the scene DB | CV/AthleticsVideos → features/<stage>/… | I: tests/test_independent_script_fixes.py:7; M: CLAUDE, README, docs, configs | 09-11 | core |
| emdb_split_clips.py | splits EMDB sequences into ≤500-frame clips | --root EMDB → clips/ | none | 09-19 bulk | dead? (EMDB absent) |
| emdb_to_smplx.py | EMDB GT SMPL → SMPL-X | --sequence → npz | only scratch/emdb_metrics.py | 09-19 bulk (real 08-23) | dead? |
| export_opencap_mot.py | our foot forces → OpenSim .mot + estimated_forces.npz | LV/out/<clip>/<run> | I: tests/test_opencap_forces.py:27, compare_opencap_plates:92, 4 methods/*_export | 09-19 | keep |
| import_bedlam2.py | BEDLAM2 → ClimbingVideos layout | ../data/BEDLAM2 → ../data/BEDLAM2_our | M: CLAUDE, label_bedlam2, scene_db | 10-05 | keep |
| import_opencap.py | imports OpenCap LabValidation | ../data/LabValidation_withVideos → LV/<clip>/ | I: sync_opencap, tests; M: run_opencap, pyproject | 09-19 | keep |
| import_parkour.py | imports LAAS Parkour | ../data/Parkour-dataset → PK/<clip>/ | M: run_parkour, parkour_sensor_forces | 09-19 | keep |
| import_peter.py | imports ETH "peter" recordings | /data3/.../climb_forces/data/eth (MISSING) → P/campus_board, P/climb_wall (MISSING) | none (P/README.md) | 07-24 | dead? |
| label_amass.py (UNTRACKED) | AMASS contacts + forces via BEDLAM2 readout | --motions (../data/retarget_test) → --save-dir | none | untracked | ask |
| label_bedlam2.py | BEDLAM2 masks/boxes/contacts/forces | BEDLAM2_our gt/ → gt/ | M: CLAUDE, pipeline.toml, tests, viewer | 10-05 | keep |
| opencap_plates.py | plate wrench algebra (library) | – | I: 9 files | 09-19 | keep |
| opencap_sensor_forces.py | OpenCap plates → world sensor_forces.npz | LV/<clip>/forces.npz + meta.json → LV/out/<clip>/<run>/ | I: sync_opencap, export_mot, tests, 3 methods | 09-19 | keep |
| parkour_estmf_inputs.py | Li et al. inputs, Parkour | PK/out → EST/data/parkour/ours | M: methods/* docstrings | 09-16 | consolidate |
| parkour_physpt_inputs.py | PhysPT inputs, --dataset {parkour, climb_wall_3} | PK/out, P/out_climb_wall_3_single → ../other/PhysPT/<ds>/ | M: compare_parkour_methods, methods/opencap_physpt_inputs | 10-05 | consolidate |
| parkour_prepare_interactvlm.py | corpus-shaped farm over Parkour for InteractVLM | PK/out → PK/others/_farm | none | 10-05 | consolidate |
| parkour_sensor_forces.py | Parkour force GT → sensor_forces.npz | PK/<clip>/forces.npz → PK/out/<clip>/human_optim/ | M: docs/board_contacts.md | 09-19 | keep |
| peter_sensor_forces.py | peter rig forces → world sensor_forces.npz; holds R_CW* constants | P/<rig>/<clip>/forces.npz → P/out_<rig>_single/<clip>/human_optim/ | I: peter_gt/prepare_human.py:48; M: viewer, renderers | 09-11 | keep; default --tree (line 130) MISSING |
| pipeline.py | local orchestrator, 8 steps | video/folder → out/<stem>/ | M: CLAUDE, README, configs, docs | 08-25 | core |
| recompute_contact_confidence.py | rewrites confidence inside contacts npz | in place | I: tests/test_contact_confidence.py:155,193; M: docs | 09-13 | keep |
| run_athletics_chain.sh | post-smplx corpus chain, AthleticsVideos | ../data/AthleticsVideos (MISSING); derived TOMLs into scratch/ (23-25) | only scratch/resume_waiter.sh | 09-11 | dead? / ask |
| run_data2.py | full chain over data_2 | data_2/*.mp4 → out_2/ (43-44) | none | 08-25 | consolidate |
| run_flat_torque.py | kindyn_1 flat-torque ablation | P/out_climb_wall_3_single → human_optim_flat_torque (68/68 exist) | none | 09-19 bulk | trash (done) |
| run_gt_contact.py | contact chain on measured board contacts | P/out_<ds>_single or PK/out → human_optim_gt_contact | M: docs/board_contacts.md, run_opencap, run_flat_torque | 09-19 | consolidate |
| run_opencap.py | full chain + --variant farms over OpenCap | LV → LV/out | I: tests/test_support_scene.py:31,37, tests/test_viewer_plate_forces.py:10 | 09-19 | consolidate (tests pin it) |
| run_parkour.py | full chain over Parkour | PK → PK/out | M: configs/pipeline_parkour.toml | 09-19 | consolidate |
| run_peter.py | full chain over peter, single/multi view | P/<ds>/<clip>/cam_*.mp4 → P/out_<ds>_{single,multi} | M: configs/pipeline_peter.toml | 08-25 | consolidate |
| scene_db.py | scene-DB access (library) | – | I: corpus, view_3d, label_bedlam2, assign_dataset_split | 10-05 | core |
| score_contacts.py | frozen per-joint-era scorer | out/<s>/kinematic_1 + data/<s>.npz | M: CLAUDE:77,166, tools/human_optim/contacts.py:12 | 09-13 | dead? / ask |
| score_frame_contacts.py | frames-era contact scorer vs manual labels | out/<s>/contacts_<tag> + data/<s>.npz (138) | I: calibrate_contact_confidence, contacts_control, score_baseline_contacts, score_gt_contacts; M: CLAUDE | 08-23 | keep |
| sync_opencap.py | OpenCap video↔mocap offset | LV/<clip> + out/<clip>/sapiens → sync.json | I: tests; M: import_opencap | 09-19 | keep |
| view_3d.py | viser viewer | out/<stem>/, CV/Athletics DB, BEDLAM2_our | M: CLAUDE, README | 10-05 | core |

### scripts/diagnostics/ (24)
| path | purpose | inputs → outputs | I / M | last | verdict |
|---|---|---|---|---|---|
| compare_climb_wall_2 | kindyn_1 vs supervised forces, cw2 | P/out_climb_wall_2_single → png | none | 08-24 | dead? |
| compare_climb_wall_3_runs | 3 runs on cw3 vs rig | P/climb_wall_3_gt + out_cw3 → benchmark_tables/ (67) | none | 09-19 | keep; redirect output |
| compare_gt_contact | board-labelled vs estimated chain | P/out_climb_wall_3_single | I: compare_climb_wall_3_runs; M: docs/board_contacts.md | 09-19 | keep |
| compare_opencap_forces | our-world force diagnostics vs plates | LV/out → out/_compare/*.png | I: tests (best_lag) | 09-19 | keep |
| compare_opencap_plates | published-protocol GRF metrics (%BW) | LV/out/<clip>/<run>/estimated_forces.npz | M: methods exporters | 10-05 | keep |
| compare_parkour_methods | 3 methods on 28 Parkour clips | PK/out + method outputs | I: render_parkour_method_overlays | 09-19 | keep |
| contacts_control | physics-free control experiment | out/ + data/ → scratch/round2_control (88) | M: docs/force_contacts.md | 08-25 | dead? / ask |
| motion_smoothness | kinematics_1 jitter metric | --predictions → json | none | 08-12 | dead? |
| render_cw3_estmf_overlays | Li et al. on cw3 video | P/… → P/climb_wall_3_gt/others/estmf/videos | none | 10-05 | keep |
| render_kindyn_overlays | kindyn force overlay on measured clip | campus/cw2/cw3/gt/parkour → mp4 | I: 3 renderers; subprocess from run_gt_contact, peter_gt/run_human | 09-19 | keep |
| render_parkour_method_overlays | one method vs measured, Parkour | → ../data/Parkour-dataset/comparison/videos | none | 09-19 | keep |
| render_prediction_overlays | campus_board external-model overlays + shared draw helpers | P/campus_board, out/, data/, P/climb_wall (MISSING, 93-96) | I: 3 renderers | 08-24 | keep as library |
| render_sup_force_overlays | supervised model overlay | <clip>/predictions/forces_sup.npz + ../contact_anything | I: compare_climb_wall_2 | 10-05 | keep |
| scene_quality | reconstruction-quality table | --predictions | M: docs/quality.md, tools/geometry/quality.py | 08-25 | keep |
| score_baseline_contacts | published contact models, CV test split | CV results | none | 09-11 | keep (merge candidate) |
| score_baseline_pose | published pose models, CV test split | CV results | I: score_gt_pose, score_init_world | 09-11 | keep |
| score_combined_table | both benchmarks side by side | → benchmark_tables/ (52) | none | 09-15 | keep; redirect |
| score_gt_contacts | published contact models vs cw3 boards | → P/climb_wall_3_gt/others | I: compare_climb_wall_3_runs | 09-19 | keep (merge candidate) |
| score_gt_pose | published pose vs cw3 multi-view GT | → others/gt_pose_* | I: score_init_world | 09-19 | keep |
| score_init_world | SAM-3D init world metrics; patches both jsons | – | none | 09-15 | keep |
| score_parkour_pose | PA-MPJPE vs Parkour GT | – | I: compare_parkour_methods | 09-19 | keep |
| score_pipeline_contacts | pipeline vs campus_board plates | → out_campus_board_single/results | I: 5 files (_limb_members/_limb_load) | 09-06 | keep as library |
| score_prediction_contacts | external models vs campus_board plates | same | I: 8 files (mostly _render_table) | 08-01 | keep as library |
| score_test_contacts | contact readings, CV test split | CV → <data_root>/results | M: experiments/ only | 08-25 | keep? ask |

### scripts/methods/ (8)
All write into external repos (EST, ../other/PhysPT) or LV/out/<clip>/method_*. None imported except opencap_learned_export (by opencap_frames35_export).
- Keep: contact_recognizer (10-05), opencap_estmf_export, opencap_learned_export, opencap_frames35_export, opencap_physpt_export.
- Consolidate: cw3_estmf_inputs, opencap_estmf_inputs, opencap_physpt_inputs.

### scripts/peter_gt/ (16)
Climb_wall_3 multi-view ground-truth chain, last commits 09-11 to 09-19. Entry: run_all_clips.sh → run_human.py. Scene side: calibrate_cameras, extract_frames, reconstruct_vggt, clean_scene, rebuild_scene_calibrated, view_gt (+ vggt_gt.toml). Human side: prepare_human, fit_multiview, overlay_human (+ human_gt.toml, a 493-line copy of the peter config, and pipeline_gt.toml). Baselines: prepare_baselines + run_baselines.sh (models in ../other/* conda envs). Imports scripts/peter_sensor_forces.py (prepare_human.py:48), core tools.human_optim.* incl. 9 private names (fit_multiview.py:76-79), tools.geometry. Calls stage CLIs and render_kindyn_overlays.py (run_human.py:166). Nothing imports it; output P/climb_wall_3_gt/{human,others} is read by 10 scripts. Verdict: keep as is; every path is `_REPO / "peter/..."` (11 files).

### scripts/stages/ (12, core)
track_sam3, estimate_camera_vggt, estimate_sapiens, estimate_body_sam3d, convert_mhr_to_smplx, fuse_geometry, estimate_scale, human_optimization, estimate_contacts, estimate_gravity_geocalib, build_support_scene (I: tests/test_support_scene.py), measure_contacts_boards.

### tools/ (non-core)
All keep; every module has a live importer. adapters/*: their stages. bedlam_*: label_bedlam2, label_amass, tests. body.py: 22 importers (not in CLAUDE layout). config.py: 21 importers. geometry/*: each imported (surface.py only by viewer/loading.py, rasterize.py only by bedlam_masks.py). mask_io, npz_io, precompute_runner, source: stage infra. mhr_to_smplx/converter: via convert_mhr_to_smplx.py:185. smplx_robot/dynamics: kindyn, forces, bedlam_contacts, tests. smplx_robot/model.py: shim, model_path ignored (19-30), used by core. smplx_robot/vertex_weights: stages.py. viewer/*: view_3d, tests. dynamics.py vs human_optim/motion.py: deliberately separate (motion.py:26-28). tools/optim.py and tools/human_optim/*: core.

**Untracked:** only scripts/label_amass.py (plus agent_notes/).

## 2. Duplicate families

**A. Runners (run_data2, run_parkour, run_opencap, run_peter, run_gt_contact, run_flat_torque; 1,388 lines).**
- Shared: identical MANIFEST_STEPS/CLIP_OUTPUT tables (data2:48, parkour:49, opencap:64, peter:74); `_clip_command` (data2:68 == parkour:69; opencap:175 0.90; peter:94 0.88); `_manifest_name` (data2:98, parkour:99, opencap:210, peter:123); `_parse_shard` (opencap:216, peter:217, 0.98); same skip-if-exists / break-on-failure loop.
- run_data2 vs run_parkour: 124 diff lines, nearly all docstring; code differs only in data root, clip glob (`*.mp4` vs `*/video.mp4`) and default configs.
- Real differences: run_peter --views multi-view (mask reuse, extra-view tracking, symlinks); run_opencap --variant farms, synthetic support-scene steps, --view-farm; run_gt_contact board contacts + overlay step; run_flat_torque derived TOML. build_farm flat_torque:73 == gt_contact:91.
- pipeline.py cannot absorb them: 8 steps only (:57), sam3d/smplx once per clip (:58), aborts on first failure (:170), no manifest or --shard.
- Option (a), full merge: `scripts/run_dataset.py --dataset {data2,parkour,opencap,campus_board,climb_wall_2,climb_wall_3} [--clips a,b] [--steps ...] [--shard i/n] [--variant gt_contact|floorplane|labvalidation|...] [--views cam_left,cam_right]`. Dataset roots, clip glob, out root and the two configs in configs/datasets.toml; variant recipes under [variants.*]. Est. ~400 lines. Fits "Configs over CLIs" and puts every data path the move touches in one TOML.
- Option (b), lighter: shared scripts/_chain.py (step tables, clip_command, manifest, shard, farm, loop) with thin per-dataset wrappers.
- Either way, tests import run_opencap (VARIANT_CHAINS, VARIANT_STEPS, build_farm, owns_scene, build_view_farm) and must move too. run_athletics_chain.sh / corpus.py are the DB path; keep separate.

**B. Importers/labellers.** import_{bedlam2,opencap,parkour,peter} parse four unrelated formats; almost nothing shared. No consolidation. label_amass/label_bedlam2 already share tools/bedlam_*; keep.

**C. Sensor forces + OpenCap family.** The three *_sensor_forces.py write one schema but each transform differs (peter: datasheet rotation + resampling; parkour: linear part only; opencap: meta.json calibration). Keep separate. opencap_plates / export_opencap_mot / sync_opencap / import_opencap already factored. Optional later: package scripts/opencap/.

**D. Method inputs.**
- estmf trio (parkour_estmf_inputs, methods/opencap_estmf_inputs, methods/cw3_estmf_inputs; 669 lines): identical coco18_from_sapiens (parkour:66, opencap:102, cw3:76), smplx_model (58/94/68), on_true_camera (cw3:119 == opencap:159); repeated constants. Differs: contact_states source, cw3 gravity + principal point. Sketch: `scripts/methods/estmf_inputs.py --dataset {parkour,opencap,climb_wall_3} [--clips] [--contact-mapping heel|any]`.
- PhysPT pair (parkour_physpt_inputs, methods/opencap_physpt_inputs; 271 lines): identical fit_betas, smpl_rest_joints, smplx_rest_joints; rodrigues 0.75 (third copy emdb_to_smplx.py:53). Sketch: `scripts/methods/physpt_inputs.py --dataset {parkour,climb_wall_3,opencap} [--step N] [--offset K]`.
- Baseline farms (parkour_prepare_interactvlm 79 lines, peter_gt/prepare_baselines 95 lines): same layout; subdir()/sub() duplicate scene_db.shard. Sketch: `scripts/methods/baseline_farm.py --dataset {parkour,climb_wall_3} [--skip-existing] [--jobs N]`.
- Exporters already share export_opencap_mot.plate_triple/floor_y. frames35 could become `opencap_learned_export.py --contact-set {six,frames35}` (low priority).
- Move the three parkour_* method scripts into scripts/methods/.

**E. Contact scoring.** score_frame_contacts (live hub), score_contacts (frozen per-joint era; imports private names from core stages.py:49), calibrate_contact_confidence, recompute_contact_confidence. diagnostics/score_* are separate benchmarks. Real siblings: score_baseline_contacts vs score_gt_contacts (vertex_rows 0.85, ours 0.75, _count 0.71) → `score_published_contacts.py --benchmark {climbingvideos,climb_wall_3}`; score_baseline_pose vs score_gt_pose (gt already imports baseline) → `score_published_pose.py --benchmark ...`.

**F. Overlays / compare_*.** Five render() bodies 0.09 similar: different renderers. Share helpers by importing from render_prediction_overlays / render_kindyn_overlays; _draw_force_arrows exists 3× (0.23-0.29). No single-script merge; move shared helpers into a small module so the campus_board renderer can retire. compare_opencap_forces and compare_opencap_plates answer different questions: keep both.

**G. One-offs.** backfill_static_camera, assign_dataset_split: DB migration tools. emdb_*: data absent. scene_db, view_3d, _pym_preload: core.

## 3. Shared helpers copied across scripts
| helper | copies | similarity |
|---|---|---|
| runner step tables + _clip_command + _manifest_name + loop | run_data2:48/68/98, run_parkour:49/69/99, run_opencap:64/175/210, run_peter:74/94/123 | 0.88-1.0 |
| build_farm | run_flat_torque:73, run_gt_contact:91 (identical); run_opencap:129 (variant) | 1.0 / 0.18 |
| _parse_shard | run_opencap:216, run_peter:217 | 0.98 |
| coco18_from_sapiens, smplx_model | parkour_estmf_inputs:66/58, methods/opencap_estmf_inputs:102/94, methods/cw3_estmf_inputs:76/68 | identical ×3 |
| on_true_camera | methods/cw3_estmf_inputs:119, methods/opencap_estmf_inputs:159 | identical |
| fit_betas, smpl_rest_joints, smplx_rest_joints | parkour_physpt_inputs:93/72/82, methods/opencap_physpt_inputs:86/65/75 | identical |
| rodrigues | emdb_to_smplx:53, opencap_physpt_inputs:54, parkour_physpt_inputs:61 | 0.75 |
| physpt_model | diagnostics/compare_parkour_methods:254, methods/opencap_physpt_export:82 | identical |
| _wired | diagnostics/render_prediction_overlays:138, diagnostics/score_prediction_contacts:56 | identical |
| farm sub/subdir + JPEG extract | parkour_prepare_interactvlm:31-48, peter_gt/prepare_baselines:34-56 (= scene_db.shard) | same logic |
| vertex_rows / ours / _count | score_baseline_contacts:143/103/160, score_gt_contacts:142/105/164 | 0.71-0.85 |
| load_prediction | methods/opencap_frames35_export:73, methods/opencap_learned_export:102 | 0.78 |
| count_frames | import_peter:110, peter_gt/prepare_human:93 | 0.88 |

## 4. CLAUDE.md entry points vs inventory
- Documented and present: pipeline.py, corpus.py, scene_db.py, import_bedlam2.py, label_bedlam2.py, score_frame_contacts.py, score_contacts.py, view_3d.py, _pym_preload.py, the 10 stage CLIs, tools/{optim,config,source,mask_io,precompute_runner,bedlam_masks,bedlam_contacts} + subpackages.
- Present but undocumented: all of diagnostics/, methods/, peter_gt/; every run_*.py and run_athletics_chain.sh; import_{opencap,parkour,peter}, label_amass, the three *_sensor_forces, opencap_plates, sync_opencap, export_opencap_mot; calibrate/recompute_contact_confidence (in README/docs), assign_dataset_split, backfill_static_camera, emdb_*, parkour_* method scripts; stage CLIs build_support_scene.py and measure_contacts_boards.py; tools/body.py, npz_io.py, bedlam_forces.py.
- Documented but missing/stale: CLAUDE.md:27-46 "experimental" branch (repo on main, no such branch); `view_3d --scenes athletics` (../data/AthleticsVideos absent); README.md:64 scripts/export_contact_dataset.py (retired a82388a, 2026-08-25); README.md:57 `--stages dynamics` (retired; core docstring stages/human_optimization.py:3-11 still says "dynamics"); docs/contact_confidence.md cites tools/human_optim/depth_fit.py (missing); peter/README.md describes climb_wall/ and climb_wall_orig/ (missing).

## Ranked proposal

**(i) Safe to trash now:**
1. scripts/run_flat_torque.py (ablation done, 68/68; no references).
2. scripts/diagnostics/motion_smoothness.py (no references; 08-12).
3. scripts/diagnostics/compare_climb_wall_2.py (one-off cw2 figure).
4. scripts/emdb_split_clips.py, scripts/emdb_to_smplx.py (EMDB absent).
5. scripts/import_peter.py (source missing; output layout gone).
6. scripts/backfill_static_camera.py (migration done; also drop docs/fusion.md:250).

**(ii) Consolidation candidates, by payoff:**
1. Runners → run_dataset.py + configs/datasets.toml (a) or _chain.py (b). Tests pinning run_opencap must follow.
2. Method inputs → methods/estmf_inputs.py --dataset, methods/physpt_inputs.py --dataset --step, methods/baseline_farm.py --dataset; move parkour_* into methods/.
3. Diagnostics helpers (_render_table, _limb_members, _limb_load, _wired, draw helpers) → one shared module; then campus_board-only entry points (score_prediction_contacts, score_pipeline_contacts, render_prediction_overlays) become trash candidates.
4. score_{baseline,gt}_contacts → score_published_contacts.py --benchmark; same for pose (optional).
5. Fix stale defaults while touching files: peter_sensor_forces.py:130; render_prediction_overlays.py:93-94; compare_climb_wall_3_runs.py:67 and score_combined_table.py:52 (benchmark_tables/); contacts_control.py:88 and run_athletics_chain.sh:23-25 (scratch/).

**(iii) Keep as is:** stages/, pipeline.py, corpus.py, scene_db.py, view_3d.py, _pym_preload.py, import_{bedlam2,opencap,parkour}.py, label_bedlam2.py, OpenCap family, parkour/peter_sensor_forces.py, score_frame_contacts.py, calibrate/recompute_contact_confidence.py, all of peter_gt/, methods/*_export.py, contact_recognizer.py, the cw3/OpenCap/Parkour scorers and renderers, scene_quality.py, all of tools/.

## Open questions
1. score_contacts.py (frozen per-joint era; CLAUDE.md:77,166): trash and edit CLAUDE.md?
2. diagnostics/contacts_control.py: keep as reproducible record or trash?
3. AthleticsVideos and EMDB absent. Gone for good? Decides run_athletics_chain.sh, corpus_athletics*.toml, `--scenes athletics`, emdb_*.
4. scripts/label_amass.py untracked: commit or trash?
5. Before scratch/ goes, save import_climb_wall_3.py, trim_climb_wall_{2,3}.py, cw{2,3}_wall_datasheet.py, cw{2,3}_gravity_fit.py and R_cw*_wall_to_world*.npy somewhere?
6. data/ move breaks core's manual-label lookup (stages.py:779-782), viewer video lookup (loading.py:420), score_frame_contacts.py:138. Core cannot be edited.
7. Runner consolidation: option (a) or (b)?
8. PK/out holds 9 variant dirs (human_optim_abl_*, _chamfer, _scenefree, _gt_scenefree, _gt_contact_chamfer) that no tracked or scratch script produces.
