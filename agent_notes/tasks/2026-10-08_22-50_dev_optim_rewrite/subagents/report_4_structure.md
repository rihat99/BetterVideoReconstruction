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
