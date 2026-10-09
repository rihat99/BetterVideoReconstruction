# Report 4 — structure (mars_top, 2026-10-09)

Sub-step 4a finished by a second worker (the first hit its context limit mid-4a). Nothing committed.
Scratch scripts (untrusted automation, kept for the record): /tmp/claude-1003/-home-rikhat-akizhanov-better/cb45a0c1-6981-4fe2-a421-62ef38728597/scratchpad/{retire_a.py,retire_b.py,retire_b2.py,retire_b3.py,retire_docs.py,retire_c.py}
(retire_b3 = retire_b2 minus the already-applied kindyn section, with a fixed section regex — retire_b2's `(?:[^\n\[]*\n)*` would also have deleted the comment block above `[ann_velocity]`.)

## Baseline (addendum) — DONE
`data_dev/examples/out/<clip>/human_optim/` copied to `human_optim_dev3/` for all 4 clips (08:55) before any run.

## SUBSTEP 4a READY

### What moved (all `git mv`, history follows)
| From | To |
|---|---|
| `tools/` (whole package) | `bvr/` |
| `scripts/stages/*.py` (12 stage CLIs) | `bvr/stages/<name>.py` + `bvr/stages/__init__.py`; runnable as `python -m bvr.stages.<name>` |
| `scripts/_pym_preload.py` | `bvr/stages/_pym_preload.py` (re-execs as `python -m <module>` when run as a module) |
| `tools/config.py` + `tools/human_optim/config.py` | one `bvr/config.py` (`load_toml`, `section`, `STAGE_ORDER`, `FPS_REF`, `fps_scaled_weights`, `HumanOptimConfig`, `load_human_config`) |
| `docs/contact_fusion.md`, `docs/contact_points.md` | `agent_notes/docs/retired/` |

- pyproject.toml: `[build-system]` setuptools + `[tool.setuptools.packages.find] include = ["bvr*"]`; `uv pip install -e . --no-deps --no-build-isolation` done. Verified from `/`: `bvr.__file__` resolves to the dev worktree. 4b must add `datasets*`.
- sys.path repo-root inserts + `_HERE`/`_REPO_ROOT` removed from every `bvr/stages/*.py` and from scripts/pipeline.py, run_dataset.py, corpus.py (`REPO_ROOT` from `bvr.paths`). The other scripts/*.py keep theirs until 4b moves them. tests/conftest.py keeps its insert (needed for `from scripts…`).
- Runners call `python -m bvr.stages.<name>`: pipeline.py (`_MODULES`), run_dataset.py (`MANIFEST_STEPS`, `_clip_command` with `-m`), corpus.py (`StageSpec.cli` = module name; farm worker `from bvr.stages import …`), peter_gt/run_human.py, reconstruct_vggt.py.
- corpus.py `_IMPLEMENTATION_INPUTS` had 8 paths still built from the literal `"tools"` (the rename sed only rewrote `tools.`/`tools/`) → `_is_fresh` was always False; fixed to `"bvr"` (this was the one pytest failure of the first full run).
- `bvr/paths.py` still carries the dataset roots (CLIMBING, PETER, …): they move with their datasets in 4b (scene_db.py etc. import them today).

### What was retired
| Item | Where it went / what was cut |
|---|---|
| `kindyn_manual`, `mhr_1_sup`, Athletics (`ATHLETICS_DATA_ROOT`/`ATHLETICS_SPORTS` in scene_db.py, `--scenes athletics` text) | code already gone; last mentions removed from scene_db.py and CLAUDE.md |
| per-joint manual-label path | optimizer.py: `contact_axis`/`contact_groups` params, `v2j_sub`, `on_frames`, `incontact_v`/`nocontact_v`, vertex `_sig`; terms `ann_repulsion`, `ann_seating`, `ann_acceleration`, `velocity_root/joint`; losses.py `clearance_penalty`, `seating_penalty`, `joint_acceleration_loss`; contacts.py `vertex_to_joint`; stages.py `_contact_labels`, `labels_mode`; kindyn.py manual branch (`run_kindyn(source, pred, cfg, device, *, overlay, bodies, cones, mass_from_meta, edge_speed_scale)`); human_optimization.py `--annotations`, `labels=`/`annotations=` kwargs |
| `[stages.*].labels = "auto"` (3 rows), weight rows `velocity_root/joint`, `ann_seating`, `ann_repulsion`, `ann_acceleration`, sections `[ann_seating]`, `[ann_repulsion]` | cut from all 5 TOMLs (configs/human_optimization{,_peter,_floorplane,_floorplane_edgev}.toml, scripts/peter_gt/human_gt.toml) |
| `bvr/config.py` knob section `"contact"`, `_DIFF_ORDER` entries for the cut terms | gone |
| corpus.toml `[score]` | gone |
| `bvr/human_optim/config.py`, `scripts/diagnostics/score_test_contacts.py`, `scripts/diagnostics/render_sup_force_overlays.py`, `scripts/recompute_contact_confidence.py` (+ its 2 tests in test_contact_confidence.py) | trashed → `/home/rikhat.akizhanov/better/BetterVideoReconstruction/trash/` (the worktree shares the main checkout's trash dir) |
| peter_sensor_forces.py vanished-rig docstring, `R_SENSOR_TO_WORLD`, `CALIB_CLIPS`, `NOTE` | gone |
| "12 contact frames" text (the set is F = 35) | `(T, F)`/`(P,N,F)`/"contact frames" in forces.py, contacts.py, kindyn.py, estimate_contacts.py, score_frame_contacts.py, estmf_inputs.py, 2 tests, CLAUDE.md, docs/human_optimization.md, contacts.md, force_contacts.md. Kept on purpose: "legacy 12-frame climbing set" (bvr/body.py, contacts.py, test fixtures), OpenCap's "12 default clips", "12 comparison clips", "Twelve of thirteen scenes". |
| doc mentions | CLAUDE.md (two doc bullets, score_test_contacts, kinematics_2 `labels = "manual"` cell, `mhr_1_sup`, athletics, recompute), README.md (contact_points bullet), docs/contacts.md, contact_confidence.md, human_optimization.md (manual-label switches, ann_acceleration/ann_repulsion bullets, `labels = "predicted"` paragraph, ann_seating bullet → a 6-line "why it is gone" note without the name) |
| viewer loading.py legacy shims | none found |

`_ANN_TO_SMPLX*`, `_ann_mapping`, `_ANN_TO_FRAMES`, `annotation_frame_labels`, `resolve_annotation_path` stay in stages.py until 4b moves them to `datasets/examples/annotations.py` (score_frame_contacts needs them).

### Verification
- `python -m compileall -q bvr scripts tests` OK. Retired-name grep (`kindyn_manual|mhr_1_sup|contact_axis|contact_groups|v2j_sub|ann_repulsion|ann_seating|ann_acceleration|velocity_root|velocity_joint|labels_mode|recompute_contact_confidence|score_test_contacts|render_sup_force|ATHLETICS|human_optim\.config|contact_fusion\.md|contact_points\.md|clearance_penalty|seating_penalty|joint_acceleration_loss|vertex_to_joint`) over bvr scripts tests configs docs CLAUDE.md README.md INSTALL.md pyproject.toml: empty. `"tools"` literal: only the MHR submodule path in convert_mhr_to_smplx.py (correct).
- pytest (`CUDA_VISIBLE_DEVICES="" BVR_DATA_ROOT=data_dev`): **234 passed, 4 skipped** (baseline 236 + 4; the two recompute tests are gone). Log: artifacts/dev4/pytest_4a.log.
- `--help` rc=0 for all 12 `python -m bvr.stages.<name>` (track_sam3, estimate_camera_vggt, estimate_sapiens, estimate_body_sam3d, convert_mhr_to_smplx, fuse_geometry, estimate_scale, human_optimization, estimate_contacts, measure_contacts_boards, estimate_gravity_geocalib, build_support_scene) and scripts/pipeline.py, run_dataset.py, corpus.py, view_3d.py.
- Regression (GPU 2, `BVR_DETERMINISTIC=1`, `--stages root,kinematics_1` on 9RvrpYoWDoo_0001 into `human_optim/`; log artifacts/dev4/regression_4a_kin1.log), compare_runs.py A=human_optim_dev3 B=human_optim (artifacts/dev4/compare_4a_kin1.log):
  `kinematic_1  pelvis 0.00/0.0 mm  root 0.000/0.00 deg  joints 0.000/0.00 deg  auto_labels agree 1.0000  [27/27 keys identical]`
  (the kinematic_2/kindyn_1/contacts rows of that output are the untouched step-3 files, not re-run — not evidence; the full 4-clip chain is the 4c check.)
- Bit-identity argument for the cut terms: BR `Problem._active()` skips zero-weight terms, so the all-zero rows never entered the sum; `ann_repulsion` (4000 in the kin2/kindyn columns) was guarded by `use_rep = use_ann and not on_frames` in HEAD's optimizer and never added on the frames axis the whole chain uses.

## 4b / 4c — not started; notes for the successor
- 4b per brief; `datasets` namespace → pyproject `include = ["bvr*", "datasets*"]` + `[tool.pytest.ini_options] testpaths = ["tests", "datasets"]`; `datasets/examples/annotations.py` takes `_ANN_TO_SMPLX`, `_ANN_TO_SMPLX_14`, `_ann_mapping`, `_ANN_TO_FRAMES`, `annotation_frame_labels`, `resolve_annotation_path` (+`EXAMPLES`) from bvr/human_optim/stages.py and test_kin2_kindyn_frames' annotation tests; scripts/score_frame_contacts.py + calibrate_contact_confidence import from there.
- Dataset roots leave `bvr/paths.py` in 4b (scene_db.py imports CLIMBING, DATA_ROOT from it today).
- `kindyn.mass_from_meta` → explicit `mass_kg` input in 4b; `mass_source` must stay "shape" for the examples.
- Viewer split per the addendum (bvr/viewer/datasets_layers.py → datasets/<name>/viewer.py; sensor/plate kinds via `ForceArrows.kind`; "bedlam2_gt" tag out of core).
- CLAUDE.md layout block still lists `scripts/_pym_preload.py` and `scripts/stages/` (step 6 rewrites the layout; only retired mentions were touched here).
- 4c: `bvr/config.py` gets `load_config(path) -> dict` with `base = "…"` deep merge (weight arrays replaced whole); `load_human_config` built on it; one runner `scripts/run.py` with `stage_command(name, clip_root, config)` shared by pipeline/run_dataset/corpus (today's per-step argv tables: scripts/pipeline.py `_MODULES`/`_build_perclip_args`, scripts/run_dataset.py `_clip_command`).

## Open questions
1. kindyn_1.npz `source_labels` is now a constant "auto" (kept only so compare_runs reports every key identical) — drop in step 6?
2. The brief's "delete the sys.path repo-root inserts": done for bvr/stages and the three runners; the other scripts/*.py keep theirs until 4b moves them (their inserts also serve sibling `import scene_db`-style imports).
3. The trashed files landed in the MAIN checkout's `trash/` (`/home/rikhat.akizhanov/better/BetterVideoReconstruction/trash/`), not under the worktree — the `trash` helper resolves the git common dir. Fine as a copy; say if you want them under dev/.
4. docs/contacts.md:174 and docs/quality.md:3 still say "athletics" as the name of a historical data batch (data_2) — history, not an Athletics-DB remnant; left as is.

## SUBSTEP 4b READY — datasets/<name>/ split (mars_top ×2, 2026-10-09)
Nothing committed. All moves are `git mv` (73 renames, 30 new files, 33 edits in `git status`). Scratch (untrusted automation): scratchpad `b4/{rewrite_imports.py, sweep_paths.py, unused_imports.py, corpus_abs.toml}`.

### What moved where (all git mv)
| From | To |
|---|---|
| scripts/diagnostics/_common.py | datasets/_common.py (shared by peter/opencap/parkour/methods) |
| scripts/score_frame_contacts.py; stages.py annotation block (`_ANN_TO_SMPLX*`, `_ann_mapping`, `_ANN_TO_FRAMES`, `annotation_frame_labels`, `resolve_annotation_path`) | datasets/examples/score_frame_contacts.py; datasets/examples/annotations.py (public names `ANN_TO_SMPLX`, `ANN_TO_SMPLX_14`, `ann_mapping`, `ANN_TO_FRAMES`); its 5 tests → datasets/examples/tests/test_annotations.py (cut from tests/test_kin2_kindyn_frames.py) |
| scripts/{corpus,scene_db,assign_dataset_split,calibrate_contact_confidence}.py, configs/corpus.toml, tests/test_independent_script_fixes.py | datasets/climbingvideos/{…}.py, configs/corpus.toml, tests/test_corpus.py; + new viewer.py (scene-DB farm: `select_scenes`, `serve_scene_farm`, `run_scene_db`) + paths.py (CLIMBING) |
| scripts/peter_sensor_forces.py, scripts/peter_gt/* (15 files), diagnostics/{compare_gt_contact, compare_climb_wall_3_runs, render_kindyn_overlays, render_cw3_estmf_overlays}.py, docs/board_contacts.md, configs/{human_optimization_peter, pipeline_peter}.toml | datasets/peter/{…}.py, datasets/peter/gt/*, datasets/peter/board_contacts.md, datasets/peter/configs/*; + paths.py (PETER) |
| scripts/{import_opencap, sync_opencap, opencap_sensor_forces, opencap_plates, export_opencap_mot}.py, diagnostics/compare_opencap_{forces,plates}.py, methods/opencap_*_export.py (4), configs/{human_optimization_floorplane, _edgev, pipeline_opencap}.toml, tests/{test_opencap_forces, test_support_scene, test_viewer_plate_forces}.py | datasets/opencap/{…}, configs/, tests/; + viewer.py (`load_plate_estimate`, `plate_arrows`, `plate_forces`) + paths.py (LABVALIDATION) |
| scripts/{import_parkour, parkour_sensor_forces}.py, diagnostics/{compare_parkour_methods, render_parkour_method_overlays, score_parkour_pose}.py | datasets/parkour/ + paths.py (PARKOUR) |
| scripts/{import_bedlam2, label_bedlam2}.py, bvr/bedlam_{masks,contacts,forces}.py, bvr/geometry/rasterize.py, tests/{test_bedlam_forces, test_bedlam_labels, test_viewer_bedlam}.py, pipeline.toml `[bedlam]`+`[bedlam_forces]` | datasets/bedlam2/{import_bedlam2, label_bedlam2, masks, contacts, forces, rasterize}.py, tests/, configs/bedlam.toml (label_bedlam2/label_amass `--config` default); bvr/viewer/datasets_layers.py → datasets/bedlam2/viewer.py (BEDLAM part) + paths.py (BEDLAM2, BEDLAM2_OUR) |
| scripts/label_amass.py | datasets/amass/label_amass.py |
| scripts/methods/{baseline_farm, contact_recognizer, estmf_inputs, physpt_inputs}.py, diagnostics/{score_published_contacts, score_published_pose, score_combined_table, score_init_world}.py | datasets/methods/ + paths.py (ESTMF) |
| scripts/diagnostics/scene_quality.py | scripts/scene_quality.py (generic) |
| datasets_layers sensor part | datasets/_rigs.py (`load_sensor_forces`, `sensor_arrows`, `forces_source`, `measured_forces`, `SENSOR_FLOOR_N`) |

Final layout: `datasets/{examples,climbingvideos,peter,opencap,parkour,bedlam2,amass,methods}/` each with `README.md` (new, 15–20 lines: what the data is, where it lives, how to import / run / score / view, which configs), `paths.py` where the dataset has a root, `configs/` where it has one (climbingvideos, peter, opencap, bedlam2), `tests/` where it has tests (examples, climbingvideos, opencap, bedlam2 — the four empty `tests/` dirs the first worker had made were trashed). `scripts/` is down to `pipeline.py`, `run_dataset.py`, `scene_quality.py`, `view_3d.py`; `configs/` to `datasets.toml`, `human_optimization.toml`, `pipeline.toml`.

### Hooks / core changes
- `datasets` is a package (`__init__.py` per dataset; pyproject `include = ["bvr*", "datasets*"]`, `[tool.pytest.ini_options] testpaths = ["tests","datasets"] pythonpath = ["."]`; `uv pip install -e . --no-deps --no-build-isolation` re-run → `import datasets` works from `/`). No sys.path inserts remain in datasets/ or scripts/; sibling imports are absolute `datasets.<pkg>.<module>`. Nothing in `bvr/` imports `datasets`.
- `bvr/paths.py` keeps only REPO_ROOT, DATA_ROOT, EXAMPLES; roots live in `datasets/<name>/paths.py`, resolved from `bvr.paths.DATA_ROOT`.
- `bvr/human_optim/boards.py` STAYS in core as the `sensor_forces.npz` schema (LIMBS, limb_of_frame, limb_contact, frame_contact) — deviation from the brief's "datasets/peter/boards.py or datasets/_rigs.py": the core stage `measure_contacts_boards` reads that schema and the core never imports datasets. `HAND_FRAMES`/`FOOT_FRAMES` → `bvr/human_optim/contacts.py`. `measure_contacts_boards --config` is now REQUIRED (its default was the peter config).
- `kindyn.mass_from_meta` → `run_kindyn(mass_kg: float | None)`; `[stages.kindyn_1].mass_kg` (0/absent = shape-baked; `mass_kg = 0` in the edgev toml); kindyn_1.npz `mass_source` = "config"/"shape" (was "meta"/"shape").
- Viewer: `layers.KINDS` lost "sensor"/"plate"; `App` adds a checkbox (off) for any `item.kind` it does not know (`ForceArrows.kind` is the label); the info panel prints an unknown gravity `source` verbatim; no BEDLAM names in bvr/viewer. `scripts/view_3d.py` composes `read_clip` from `datasets/bedlam2/viewer`, `datasets/_rigs`, `datasets/opencap/viewer` and dispatches `--scenes`/`--bedlam` to `climbingvideos.viewer.run_scene_db` / `bedlam2.viewer.run_scene_db`.
- `scripts/run_dataset.py`: configs in `configs/datasets.toml` are paths relative to the repo root (`datasets/peter/configs/pipeline_peter.toml`, …); overlay step runs `python -m datasets.peter.render_kindyn_overlays`; variant default config `configs/human_optimization_<variant>.toml`.
- `datasets/climbingvideos/corpus.py` `--config` default = its own `configs/corpus.toml` (data_root still resolved against REPO_ROOT; `pipeline_config = "configs/pipeline.toml"` unchanged).
- Absolute main-BH paths (label_amass, import_bedlam2) → `bvr.body.models_dir() / "smplx_lockedhead" / "SMPLX_NEUTRAL.npz"`; `_REPO.parent / "BetterHuman" / "models"` in methods/parkour scorers → `models_dir()`. Broken `from tools import paths` leftovers in estmf_inputs.py and parkour_sensor_forces.py fixed.
- Fixed a 4a regression: `datasets/peter/gt/rebuild_scene_calibrated.py` imported `STAGE` from `reconstruct_vggt`, which 4a's commit e2e7bb7 had replaced by a `-m bvr.stages.estimate_camera_vggt` call (verified with `git log -S STAGE`); the import is dropped and its one `subprocess.run` now uses `[sys.executable, "-m", "bvr.stages.estimate_camera_vggt", …]` too.
- Text sweep: every old path/module name of a moved file rewritten in docstrings, docs, configs, CLAUDE.md, README.md (80 files; CLAUDE.md layout tree block left for step 6). board_contacts.md relative links fixed. INSTALL.md preload sentence → `bvr/stages/`.

### Verification (artifacts/dev4/)
- `python -m compileall bvr datasets scripts tests` OK; unused-import check: only pre-existing ones.
- pytest (`CUDA_VISIBLE_DEVICES="" BVR_DATA_ROOT=data_dev`, final run after the READMEs + fix): **234 passed, 4 skipped** in 126 s (pytest_4b_final.log) — identical to 4a (236 + 4 at step start minus the 2 retired recompute tests; 5 tests moved to datasets/examples/tests, 7 to opencap/bedlam2/climbingvideos).
- `--help`: **48/48 rc=0** (help_4b_final.log: 36 `python -m datasets.…` CLIs, scripts.{pipeline,run_dataset,scene_quality,view_3d}, bvr.stages.{measure_contacts_boards,estimate_contacts,human_optimization}, 4 file-form spot checks). Before the rebuild_scene_calibrated fix it was 47/48 (help_4b.log). The 8 non-argparse CLIs import with rc=0.
- Dry-run diffs (BVR_DATA_ROOT=/home/rikhat.akizhanov/better/data — data_dev has no peter/OpenCap/ClimbingVideos; a dry run writes nothing): dryrun_{before,after}_{cw3_gt_contact,opencap_floorplane,corpus_geometry}.txt — cw3 gt_contact identical (0 diff lines); opencap floorplane identical modulo the config path (`configs/human_optimization_floorplane.toml` → `datasets/opencap/configs/…`, 0 lines after masking); corpus geometry `--limit 2` identical (0 diff lines; run from datasets/climbingvideos/corpus.py with scratchpad b4/corpus_abs.toml = corpus.toml with an absolute data_root).
- Regression (GPU 2, `BVR_DETERMINISTIC=1`, `python -m bvr.stages.human_optimization --stages root,kinematics_1` on 9RvrpYoWDoo_0001 into `human_optim/`, rc=0; regression_4b_kin1.log), compare_runs.py A=human_optim_dev3 B=human_optim (compare_4b_kin1.log):
  `kinematic_1  pelvis 0.00/0.0 mm  root 0.000/0.00 deg  joints 0.000/0.00 deg  auto_labels agree 1.0000  [27/27 keys identical]`
  (kinematic_2/kindyn_1/contacts rows are the untouched step-3 files; the full 4-clip chain is the 4c check.)

### Open questions (4b) — resolved here
- Q6 (viewer info panel prints "bedlam2_gt" instead of "BEDLAM2 ground truth"): KEPT as is. The core deliberately prints a dataset's own gravity tag verbatim (app.py `# a dataset's own tag … prints as it is`) and `datasets/bedlam2/tests/test_viewer_bedlam.py:104` pins `source == "bedlam2_gt"`; changing the wording would mean changing a test for a cosmetic string. The dataset side can map it later if wanted.
- Q7 (rebuild_scene_calibrated `--help` rc=1): FIXED (see Hooks) — it was a 4a regression, not pre-existing.
- Empty `datasets/{amass,methods,parkour,peter}/tests/` dirs: trashed (git never saw them).

### Open questions (4b) — left for the main session
1. boards.py kept in core (reason above) — rename `boards.py`/`measure_contacts_boards`/`contacts = "boards"` to something rig-neutral in step 6? Touches a stage name, datasets.toml and the docs, so not done here.
2. `annotation_frame_labels`, `ANN_TO_FRAMES`, `resolve_annotation_path` (datasets/examples/annotations.py) have NO caller: kinematics_2's manual branch died in 4a; only `ANN_TO_SMPLX*`/`ann_mapping` serve score_frame_contacts (verified by grep over the repo). The 5 moved tests test only `annotation_frame_labels`. Recommendation: delete the three + the 5 tests in step 6 (−5 tests); not done because the 4a brief explicitly scheduled them to be kept and the test count would move.
3. `[viewer]` DB-selection keys (data_root, categories, rope_supported, scene_ids, sample, limit, seed) still sit in configs/pipeline.toml (ClimbingVideos-specific) — 4c config split?
4. `mass_kg` is per config, not per subject (OpenCap subjects differ; the importer still writes `mass_kg` into meta.json). The feature was off everywhere (`mass_from_meta = false`), so nothing changes today; a per-clip `mass.json` would be the alternative the brief named.
5. Dataset script basenames keep their redundant prefixes (`datasets/opencap/opencap_plates.py`, `datasets/peter/peter_sensor_forces.py`) — rename in step 6 or 4c?
6. READMEs are 15–20 physical lines (brief: 5–15) because command lines are not wrapped; peter (20) and opencap (20) have the most moving parts.
