# Investigation C — BVR map: generic core vs dataset code, loss inventory, baselines (mars_aux, read-only, 2026-10-08)

Side effect: an empty gitignored `.venv` (108 KB) was created in the dev worktree by a `uv run` call; no packages.

## Verdict
- Split is mostly clean at file level: ~60% of scripts/ is dataset-specific; the core (tools/, scripts/stages/) has ~10 dataset hooks woven in (paths, viewer, kindyn meta.json mass, manual-annotation defaults, boards.py).
- The rewrite must cover five optimizer sites: tools/optim.py (kinematic stages + scale prefit); kindyn force warm-start L-BFGS (kindyn.py:409; BEDLAM2/AMASS labels reuse it); force witness projected Adam (forces.py:267-276); MHR→SMPL-X converter Adam (converter.py:211-250); fit_multiview via `optimize_person(extra_terms=, fit_betas=)` (optimizer.py:390-391).
- Most terms are not squared residuals (§3): GM on r², tanh-saturated distances, linear attraction, Huber/pseudo-Huber hinges, 1−(q·q0)². Many average over an active set that changes per iteration. q is a raw 211-vector normalised in the forward (not a manifold variable). GN/LM would change results; keep Adam/L-BFGS via BR TorchOptim.
- No on-disk output is a valid main baseline: examples kindyn_1.npz (2026-09-11) lacks `mass_source`/`edge_speed_scale` that kindyn.py:894-895 now writes; kindyn/body changed in ce0498f and 5a1fc9a. Regenerate on main HEAD with main BR/BH, twice (noise floor), before comparing dev.

## 1. Inventory
GENERIC: scripts pipeline.py, _pym_preload.py, run_dataset.py (recipe runner), view_3d.py (mixed), recompute_contact_confidence.py, diagnostics/scene_quality.py; all scripts/stages (build_support_scene only used by OpenCap floorplane); tools: optim.py (→BR), body.py, config.py, npz_io.py, source.py, mask_io.py, precompute_runner.py, adapters/*, mhr_to_smplx/*, smplx_robot/*, geometry/* except rasterize.py, human_optim/* except boards.py, viewer/helpers.py; configs pipeline.toml (mixed), human_optimization.toml; tests conftest, test_body_density, test_body_sole_offset, test_contact_confidence, test_contact_frames, test_forces, test_fps_scaling, test_fuse_scale, test_occlusion_weighting, test_optim_engine (→BR), test_scene_quality, test_visibility_normals, test_viewer_camera_pose, test_viewer_multiview, test_kin2_kindyn_frames (mixed), test_support_scene (mixed); docs scale, fusion, human_optimization, contacts, contact_confidence, force_contacts, quality.

Mixed files:
| file | dataset part → where |
|---|---|
| tools/paths.py:15-19 | LABVALIDATION, PARKOUR, CLIMBING, ESTMF, PETER → each dataset; core keeps DATA_ROOT, EXAMPLES |
| tools/human_optim/stages.py:595-783 | `_ANN_TO_SMPLX*`, `_ann_mapping`, `annotation_frame_labels`, `resolve_annotation_path` (examples' manual annotations; score_frame_contacts imports the maps) |
| tools/human_optim/kindyn.py:724-727 | `mass_from_meta` reads `<source>/../meta.json` (OpenCap) → explicit mass input |
| tools/human_optim/boards.py | LIMBS + sensor→frame logic (peter/parkour/OpenCap); FOOT_FRAMES also used by estimate_contacts.py:313 → core contacts |
| tools/viewer/loading.py | load_bedlam_data/_bedlam_gravity/_bedlam_contacts (838-957) → bedlam2; _load_sensor_forces/_load_plate_estimate (134-186) → rigs/OpenCap; _mhr_skinned (678-770), kindyn_manual legacy |
| tools/viewer/app.py, scene.py | groups/sport (app.py:65-125,498) athletics; _is_clip bedlam gt/ (505-514); plate/sensor layers scene.py:386-389,917-950 |
| scripts/view_3d.py:93-127,187-201 | --scenes (DB), athletics, --bedlam |
| scripts/stages/measure_contacts_boards.py | core contact-source hook from sensor_forces.npz (peter/parkour/OpenCap gt_contact) |
| configs/pipeline.toml:123-154 | [bedlam], [bedlam_forces] → bedlam2; [viewer] scene selection → ClimbingVideos |
| tests test_support_scene, test_viewer_plate_forces, test_kin2_kindyn_frames | farm tests with runner; plate tests → opencap; annotation-schema tests with the annotation path |

DATASET files:
| tag | files |
|---|---|
| climbingvideos | corpus.py, scene_db.py, assign_dataset_split.py, calibrate_contact_confidence.py, diagnostics/score_test_contacts.py (broken), configs/corpus.toml, tests/test_independent_script_fixes.py |
| examples (manual annotations) | score_frame_contacts.py + stages.py parts |
| peter rigs | peter_sensor_forces.py; diagnostics/compare_gt_contact.py, compare_climb_wall_3_runs.py, render_kindyn_overlays.py (also parkour/OpenCap), render_cw3_estmf_overlays.py; peter_gt/* (15 files incl. human_gt.toml 493-line copy); configs/pipeline_peter.toml (parkour too), human_optimization_peter.toml (parkour, OpenCap too); docs/board_contacts.md |
| opencap | import_opencap, sync_opencap, opencap_sensor_forces, opencap_plates, export_opencap_mot; diagnostics/compare_opencap_{forces,plates}; methods/opencap_{estmf,frames35,learned,physpt}_export; configs/pipeline_opencap.toml, human_optimization_floorplane{,_edgev}.toml; tests/test_opencap_forces.py |
| parkour | import_parkour, parkour_sensor_forces; diagnostics/compare_parkour_methods, render_parkour_method_overlays, score_parkour_pose |
| bedlam2 | import_bedlam2, label_bedlam2; tools/bedlam_{masks,contacts,forces}.py; tools/geometry/rasterize.py; tests/test_bedlam_{forces,labels}.py, test_viewer_bedlam.py |
| amass | label_amass.py (depends on bedlam modules + kindyn) |
| methods | methods/{baseline_farm,contact_recognizer,estmf_inputs,physpt_inputs}; diagnostics/{score_published_contacts,score_published_pose,score_combined_table,score_init_world,render_sup_force_overlays}; diagnostics/_common.py |
| athletics (absent) | scene_db.py:37-38, view_3d/viewer group code, tests/test_viewer_groups.py |
| EMDB | none left (metric naming in score_published_pose.py:29,40) |
| retired docs | docs/contact_fusion.md, docs/contact_points.md |

## 2. Target layout (proposal)
```
bvr/ (or keep tools/)   problems/ (kinematics.py, kindyn.py, prefit.py, witness.py: variables + terms → BR optim)
                        residuals.py (pure term fns: losses.py + kindyn/prefit/forces term bodies)
                        contacts/ geometry/ body.py viewer/ io (npz_io, source, mask_io, paths: DATA_ROOT only)
                        stages/ (today scripts/stages)
scripts/  pipeline.py run_dataset.py view_3d.py
datasets/<name>/  import_*.py, *_sensor_forces.py, scorers, exporters, configs/*.toml, README.md, tests/
datasets/methods/ published-baseline adapters (estmf, physpt, farms, score_published_*)
```
Hooks datasets need from core: (1) contact-source plug writing contacts_<tag>.npz schema 2 from measured limb on/off; sensor_forces.npz `(N,4,3)` LIMBS order + optional cop_world; (2) variant chains + symlink farms (run_dataset.build_farm, [variants.*]) incl. replace scene.npz (build_support_scene); (3) body knobs [contacts].frames allowlist, sole_offset_m, floor_gate_m; (4) explicit subject mass input; (5) problem-extension API (extra terms + variables like betas, as optimize_person(extra_terms=, fit_betas=)); (6) viewer extension points (extra body sources, sensor/plate layers, scene-list providers); (7) manual-annotation reader + joint maps.
Configs: only pipeline.toml (minus [bedlam*]) and human_optimization.toml generic. Dataset configs are full copies differing in 3-10 lines (peter: chamfer 0, 18-frame allowlist, [boards]; floorplane: feet+sit allowlist, floor_gate_m 0, sole_offset_m, [boards]; edgev: edge_speed_m_s 0.8, mass_from_meta, force_edge 3000; pipeline_peter/opencap: overlay/skip/chunk/prefit=false). Proposal: loader supports `base = "…"` + overrides; needs a decision (today "the loader has no defaults"). Two config paths exist: HumanOptimConfig.knobs filters sections (config.py:89-91) vs raw load_toml for [contacts], [contact_force], [quality], [boards], [contact_confidence].

## 3. Loss terms that must survive (weights [root, kin1, kin2, kindyn]; sites optimizer.py:800-943, kindyn.py:289-349)
| term | inputs | form | weights | win |
|---|---|---|---|---|
| reproj_sapiens | projected joints+landmarks vs Sapiens 308 kp (SG-smoothed 0.35 s); w = score≥0.3 × visibility | GM r²/(s²+r²) s=40 px; Σw·ρ/Σscore | 8×4 | 1 |
| reproj_init | 22 joints vs own initial projection; frame_valid | GM s=30; masked mean | 30×4 | 1 |
| chamfer | 1000 subsampled front-facing verts ↔ per-frame human cloud ≤1500; detached NN both ways | unsquared sat·tanh(d/sat) sat=0.25; geodesic emphasis; smear weights | 10,10,0,0 | 1 |
| pose_prior | joint quats vs init; visibility; fingers dropped when fitted | 1−(q·q0)² mean | 300×4 | 1 |
| orient_prior | root quat vs init | 1−(q·q0)² | 50×4 | 1 |
| velocity/accel/jerk/snap _root/_joint | Lie tangent diffs order 1-4 (motion.tangent_difference); valid-row gates | squared mean | vel 0; accel root 24,400,24,24 joint 48,800,48,48; jerk 300/500; snap 300/500 | 2-5 |
| radial_velocity | joint step on detached camera ray | squared | 0,400,100,100 | 2 |
| penetration | contact-frame signed distance to nearest scene candidate (rebuilt per phase) | Huber(relu(−s−0.01),0.02); gated conf, dist≤0.15, pen≤0.10; mean active | 0,12000,12000,12000 | 1 |
| ann_attraction | labelled frames 0<s<band; kin1 labels frozen at first scene phase | linear mean s | 0,10,0,20 | 1 |
| ann_acquisition | closest member point → patch point (patch per phase) | pseudo-Huber δ=0.10; ÷ fixed count | 0,0,1000,0 | 1 |
| ann_seating | s vs −5 mm in ±0.10 band | Huber | 0×4 | 1 |
| ann_repulsion | no-contact verts (needs manual 0 labels) | Huber(relu(0.03−s)) | 0,0,4000,4000 (inert with auto labels) | 1 |
| ann_velocity | labelled points in contact both ends + 0.1·(1−(q_t·q_{t+1})²) | squared | 0,500,4000,4000 | 2 |
| ann_acceleration | second diff | squared | 0×4 | 3 |
| base_wrench / _ang | RNEA root rows; v,a central diff spacing=round(fps/30); forces masked | 0.5·res_scale·Σ sq; res_scale=1/(frames·BW²) | 0,0,0,50 / 300 | 2·spacing+1 |
| force_magnitude / force_edge | masked frame forces; edge onset/release speed-scaled | squared | 1 / 0 (edgev 3000) | 1 |
| force_{smooth,accel,jerk}_{mag,dir} | k-th force diff ∥/⊥ window mean dir; persisting masks | squared | 120/60, 640/320, 400/200 | 2/3/4 |
| torque_magnitude/smooth/accel | strength-weighted τ[6:] and diffs | squared | 30, 8, 30 | RNEA win +0/1/2 |
Variables: q raw (T,211) normalised in forward (optimizer.py:668); DOF masks transl/orient/joints in Adam, finger mask into L-BFGS (990); betas (10) only in multi-view GT fit; kindyn f_free (T,C,3) + gen_u (T,C,4) (force = softplus(gen_u)×pyramid; with cones=true f_free is dead, kindyn.py:197-200); C = 35 (general) / 18 (peter) / 12 (floorplane).
Phases: root (Adam, root DOFs, no scene); kin1 full→refine→lbfgs(30); kin2 root/full/refine with scene, L-BFGS 0; kindyn one L-BFGS 150 after force-only warm-start L-BFGS (kindyn.py:382-459); Adam lr 0.01 MultiStep 0.1 at 1/2, 3/4; fps_scaled_weights kindyn only (optimizer.py:997); on_start rebuilds scene candidates, auto labels, patches.
Other solves: prefit (prefit.py:265-301; weights pipeline.toml [scale] w_* 1/.5/2/1/.5/.1; one-sided GM chamfer, Sapiens GM 10 px, pose, orient, transl sq, smooth; vars q + log_s frozen from round 2; 4 rounds × 150 Adam; off in peter/OpenCap); force witness (forces.py:251-276; convex quadratic in λ≥0, per-frame, 2000 projected Adam); converter (edge L1 + vertex L2, pip smplx, 4 Adam stages).

## 4. "Unnecessary" candidates
| item | evidence | proposal |
|---|---|---|
| kindyn_manual source | nothing writes it (kindyn.py:874); loading.py:1013,1031,1047; view_3d.py:37; paths.py:23; scene.py:7,308 | delete |
| mhr_1_sup + MHR loader | loading.py:678-770,1016-1020; no producer; files only in ClimbingVideos features | ask |
| per-joint manual-label path | contact_axis="joints", v2j_sub, contact_groups (optimizer.py:419-431,562-606), ann_repulsion (536,879-890), labels="manual" in k2/kindyn, stages.py:595-783 | ask (drops repulsion + joint-axis branches) |
| all-zero terms | velocity_root/joint, ann_seating, ann_acceleration (toml:79-80,118,121); force_edge 0 in base | ask |
| [stages.kinematics_1].labels must be "auto" (stages.py:493) | | delete key |
| dead config plumbing | config.py:91 section "contact" (no TOML); corpus.toml [score] (42-49) no reader | delete |
| stale text | human_optimization.py:3-11, help :89 list retired `dynamics`; `_BETTER_HUMAN_SRC=../better_human/src` (absent) in human_optimization.py, convert_mhr_to_smplx.py, estimate_scale.py, peter_gt/overlay_human.py; "12 contact frames" in docs/human_optimization.md:202,206,409, contacts.py:3, forces.py:1, CLAUDE.md:142 | fix |
| BH paths hard-wired to main | body.py:37-52 absolute `/home/.../BetterHuman/models/...`; BR/BH hand-installed --no-deps (pyproject.toml:7-8,95-96) | must change |
| Athletics | scene_db.py:37-38; view_3d; viewer groups; test_viewer_groups.py | delete |
| broken/learned-model scripts | score_test_contacts.py (:169 KeyError), render_sup_force_overlays.py | move to contact_anything or delete |
| retired docs | contact_fusion.md, contact_points.md | → agent_notes |
| vanished rig constants | peter_sensor_forces.py:55-61 | delete |
| recompute_contact_confidence.py (+tests) | migration tool | ask |
| legacy shims | loading.py:618 single-person promote; toe_under names in score_frame_contacts; paths.py docstring dynamics.npz | delete with care |
| three orchestrators | pipeline.py, run_dataset.py, corpus.py | option: one clip-root runner + DB fan-out |
| data leftovers | examples/out/*/human_optim/{kindyn_1_prev,uncertainty_1,uncertainty_2}.npz, moge2/, predictions/ | user's call |

## 5. Baselines
| output | command | main result | status | cost |
|---|---|---|---|---|
| examples human chain (13 clips) | human_optimization root,kinematics_1 → estimate_contacts → kinematics_2 → contacts 2 → kindyn_1 | `<clip>/human_optim/{kinematic_1,contacts_1,kinematic_2,contacts_2,kindyn_1}.npz` | k1 08-25, k2 08-30, contacts 09-06, kindyn 09-11: STALE | kindyn ~50 s/clip; k1/k2 ~1-3 min/clip (guess) |
| examples contact F1 | score_frame_contacts.py --predictions ../data/examples/out/ --contacts 1|2 --json | none stored | — | CPU seconds |
| examples scale/prefit | fuse_geometry → estimate_scale (rescales in place: fuse must rerun first) | geometry/{scale,prefit}.npz | — | needs vggt/depth (present) |
| OpenCap plates (64 clips) | run_dataset opencap --variant floorplane → export_opencap_mot → compare_opencap_plates | `out/_plates_floorplane_clips_natural/{summary.md,metrics.npz}` 09-21: MAE V 19.0 %BW, r V 0.62, CoP 55.3 mm, stance F1 0.97 | | 64 clips ≤900 fr @60 fps |
| OpenCap methods table | `_methods/methods_table.py` (in the data tree) | methods_table.md 09-17 | not in repo | |
| climb_wall_3 (68 clips) | run_dataset climb_wall_3 [--variant gt_contact] → score_published_pose → compare_climb_wall_3_runs | last copy in main repo trash/benchmark_tables/climb_wall_3_runs_metrics.json: F1 .916, MAE 80.1 N, angle 11.6°, corr .654, MPJPE 57.5, PA 32.8 | stale 09-08/14 | 68 clips |
| ClimbingVideos contacts (30 test) | score_test_contacts (broken) | results/contact_metrics.json 08-03 F1 .932 | not reproducible | 5 GPUs |
| ClimbingVideos published baselines | score_published_* --benchmark climbingvideos | results/baseline_* 09-10/15 | pose ref = our kindyn_1 | CPU |
| BEDLAM2 labels | label_bedlam2 --step contacts|forces | features/gt/.../{contacts,forces}.npz 09-19 | exercises warm-start L-BFGS | CPU 1-2 s/scene |
| unit tests | pytest | 250 pass | | |
GPU memory not recorded anywhere; measure with torch.cuda.max_memory_allocated during the main rerun.

Recommended regression set: boulder_1 (445 fr, 30 fps, annotated, prefit); gtIGgs_y2nY_0024 (471 fr, 60 fps, spacing=2); 3DAgLpPXVCs_0132 (363 fr, 60 fps, two people); 9RvrpYoWDoo_0001 (104 fr, smoke); one measured clip (climb_wall_3 a_00_01_part1 --variant gt_contact, or OpenCap subject2_DJ1_cam1 floorplane). Tables: (A) OpenCap plates 12-clip import set, all 64 at the end; (B) score_frame_contacts over the 13 examples. Plus BEDLAM2 forces spot check (20 scenes) and pytest. Run in a scratch data root (BVR_DATA_ROOT) so the reference trees are not overwritten.
Tolerances (set against two main-vs-main runs first): per clip joint pos mean ≤5 mm, max ≤3 cm; reproj ±0.5 px; chamfer ±0.2 cm; contact-label agreement ≥0.97; frame forces median ≤2 %BW; base_wrench after-RMS ±10%. OpenCap: MAE V ±1 %BW, r V ±0.03, CoP ±5 mm, stance F1 ±0.01. cw3: F1 ±0.01, MAE ±5 N, angle ±1°, MPJPE ±2 mm. Examples pooled contact F1 ±0.01.

## Open questions
1. Solver changes allowed (LM/GN, manifold q)? 2. Converter, force witness, kindyn warm-start onto BR optim? 3. Retire per-joint manual-label path, ann_repulsion, all-zero terms? 4. mhr_1_sup needed? 5. Base+override config overlays? 6. measure_contacts_boards + sensor_forces schema in core? 7. tools/ → bvr/? 8. Merge pipeline.py and run_dataset.py?
