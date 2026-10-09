# Report 6 — env Option C, cleanups, dev-layout docs (mars_top, 2026-10-09)

Nothing committed. Worktree HEAD 42ff251 (code 28f023e). `git status`: 10 renames (`git mv`), 1 deletion (trashed), ~41 edits, uv.lock. Artifacts: `agent_notes/tasks/2026-10-08_22-50_dev_optim_rewrite/artifacts/dev6/`. Scratch (untrusted automation): scratchpad `s6/{rename_refs.py, hand_edits.py, claude_md.py, help_sweep_6.sh, venv_c/ (a 5 GB scratch venv), uv.lock.bak, pyproject.toml.bak}`.

## 1. Env Option C — DONE, verified
- The old blocker is gone: dev/BetterRobot and dev/BetterHuman now pin the same `pytorch-cu129` index as BVR (their pyproject `[[tool.uv.index]]`), so the two path deps resolve. pyproject.toml now:
  ```toml
  dependencies = [
      # The sibling libraries, editable from the dev worktrees next to this repo (uv.sources).
      "better-robot",
      "better-human",
      # the viewer's UI library (bvr/viewer runs on better-robot's viser-based viewer)
      "viser>=1.0",
      ...                      # the rest unchanged; `yourdfpy` dropped (better-robot's own runtime dep, comes transitively)
  ]
  [tool.uv.sources]
  better-robot = { path = "../BetterRobot", editable = true }
  better-human = { path = "../BetterHuman", editable = true }
  ```
- `uv lock`: resolved 216 packages in 2 s; lock diff = the two editable packages added, `yourdfpy` now reached through better-robot, the project `virtual` → `editable`.
- Fresh env from the lock: `UV_PROJECT_ENVIRONMENT=<scratch>/venv_c uv sync --frozen` (29 s from cache) → `better_robot`/`better_human` import from `/home/rikhat.akizhanov/better/dev/{BetterRobot,BetterHuman}/src/…`, `bvr`/`datasets` from the worktree, torch 2.8.0+cu129, viser 1.0.30.
- Real `.venv`: `uv sync --dry-run` would uninstall ONLY detectron2 (exact sync) + reinstall the project and decord — so I did NOT run `uv sync` there. A plain `uv run` is an *inexact* sync (`uv run --help`: `--exact` is opt-in) and is now safe: run on the real venv it reinstalled the project (editable) + decord and kept detectron2 and `sam-3d-body.pth`; `uv run python -c "import better_robot, better_human, bvr, datasets"` → all four resolve to dev/. Same sequence proven in the scratch venv (detectron2 hand-installed there, then `uv run` kept it, `uv sync --inexact` kept it, `uv sync` removed it). INSTALL.md says so ("`uv run` vs `uv sync` afterwards").
- NOT declared (manual steps in INSTALL.md §3): **detectron2** — a source build whose outcome depends on the build machine (nvcc 13.3 is on this box while torch is cu129; with a GPU visible its setup.py compiles CUDA kernels), its setup.py imports torch at metadata time, and it only needs to be importable; **sam-3d-body** — no pyproject.toml, stays a `.pth`; **MHR assets** — a download (symlink to main's in the dev layout).

## 2. Cleanups (one line each)
- `bvr/human_optim/boards.py` → `measured_contacts.py`; `bvr/stages/measure_contacts_boards.py` → `measure_contacts.py` (log prefix `[measure_contacts]`); `[boards]` TOML section → `[measured_contacts]` (human_optimization_peter.toml, peter/gt/human_gt.toml, both readers); variant key `contacts = "boards"` → `"measured"` (peter dataset.toml, opencap comment, `scripts/pipeline.py`, `bvr/runner.py` param `measured=`). 35 files re-pointed. Kept: the file name `datasets/peter/board_contacts.md` (a peter doc about peter's boards) and the stored method stamp `boards-v3` (provenance string, also stamped by the GT chain).
- `annotation_frame_labels`, `ANN_TO_FRAMES`, `resolve_annotation_path` deleted from `datasets/examples/annotations.py` (grep: no caller; the `EXAMPLES`/`Path` imports went with them); `datasets/examples/tests/test_annotations.py` (all 5 tests were of `annotation_frame_labels`) trashed → `/home/rikhat.akizhanov/better/BetterVideoReconstruction/trash/tests/`; examples README updated.
- Prefixes: `datasets/opencap/opencap_{plates,sensor_forces,estmf_export,frames35_export,learned_export,physpt_export}.py` → `datasets/opencap/{…}.py`; `datasets/peter/peter_sensor_forces.py` → `sensor_forces.py`; `datasets/parkour/parkour_sensor_forces.py` → `sensor_forces.py` (word-boundary rewrite; `compare_opencap_plates.py` etc. are not prefixes and stay).
- `kindyn_1.npz` constant `source_labels` dropped (kindyn.py; no reader anywhere).
- `[contact_confidence].fit` note: the calibrator path dropped from the TOML value and from `FIT_NOTE` in `calibrate_contact_confidence.py` (`--write` writes the same string) → the stored provenance string no longer changes on renames.
- Docs naming the old engine: CLAUDE.md layout line + "One optimization engine" paragraph (→ "One optimizer, BetterRobot's": Problem / run / Phase + solve.py in 5 lines), docs/scale.md:131, docs/human_optimization.md "The engine" paragraph. README's stale "manual `kinematics_2` mode" sentence gone with the rewrite. `CUDA_VISIBLE_DEVICES=3` convention line kept.

## 3. Docs for the dev layout
- CLAUDE.md: "Working principles" no longer names a branch; Git workflow → dev branch, main stays as is; viewer paragraph → the BR/BH-layer viewer (loading/layers/app, body-source dropdown, layer kinds, video pane, dataset readers composed in view_3d.py, `--scenes`/`--bedlam` with their own `[viewer]` configs); BEDLAM knobs → their file; Run block (plain `uv run`, tests line, data-root + determinism note); Project layout tree rewritten to the real tree; `[measured_contacts]` + `measure_contacts.py` in the config list / stage table.
- README.md rewritten short: what it does + stage table, install pointer, run one clip, view it, run a dataset, doc links.
- INSTALL.md rewritten for Option C: prerequisites (sibling dev checkouts), `uv sync --managed-python`, the three manual steps, `uv run` vs `uv sync`, env-var table (`BVR_DATA_ROOT`, `BETTERHUMAN_MODELS_DIR`, `BVR_DETERMINISTIC`, `CUDA_VISIBLE_DEVICES`), verify, checkpoints, notes.
- Every repo path in the three files exists (scripted check, 84 paths; the remaining hits were bare stage names in the table and non-repo paths); every command checked: pipeline `--source … --dry-run` (boulder_1; folder + `--steps`), `--dataset climb_wall_3 --variant gt_contact --dry-run`, corpus `--dry-run`, view_3d / scene_quality / stage `--help`.

## 4. Verification
- `compileall bvr datasets scripts tests` OK. Leftover grep for the old names: only local variable names in peter scripts and the GT chain's own `source_labels="boards"` (fit_multiview.py:840, left).
- `--help` sweep (`help_6.log`, the 4c list with the 9 renamed modules): **57/57 rc=0**.
- Dry-runs vs 4c's captures (BVR_DATA_ROOT=…/data, read-only): real-root cw3 gt_contact / parkour / opencap floorplane / corpus geometry `--limit 2`: **0 differing lines** each; scratch-root cw3 gt_contact (empty out trees): 0 differing lines after normalising `measure_contacts_boards → measure_contacts`; the two contacts steps print `-m bvr.stages.measure_contacts --predictions …/_gt_contact --config …/human_optimization_peter.toml --bodies …`.
- pytest (CPU, `BVR_DATA_ROOT=data_dev`, after the real-venv `uv run`): **229 passed, 4 skipped** in 150 s (`pytest_6.log`) = 4c's 234 − the 5 trashed annotation tests.
- Regression (GPU 2, `BVR_DETERMINISTIC=1`, `scripts/pipeline.py --source …/9RvrpYoWDoo_0001.mp4 --steps kinematics_1,contacts_1,kinematics_2,contacts_2,kindyn_1`, `human_optim/` trashed first; 5/5 steps rc=0; `regression_6_chain.log`) vs `human_optim_dev3` (`compare_6.log/json`): `kinematic_1` **27/27**, `kinematic_2` **27/27** keys identical (pelvis 0.00 mm, joints 0.000°, labels agree 1.0000); `kindyn_1` **46/47** — the one key is `source_labels`, `only_in_a` (removed on purpose), every number identical (jpos 0.00 mm, forces 0.00 %BW, bw_after 0.0 %); `contacts_1`/`contacts_2` **69/70** — the one key is the 0-d provenance string `confidence_coefficients`, whose JSON differs in the `fit` field only (A: `… (scripts/calibrate_contact_confidence.py --write)`, B: the note without the path; checked field by field), every numeric key identical (frame_force 0.00 %BW, frame_contact agree 1.0000).

## Open questions
1. `fit_multiview.py` (peter GT chain) still writes its own `source_labels="boards"` into its kindyn_1.npz — drop too, or keep as the GT chain's stamp?
2. `datasets/peter/board_contacts.md` keeps its name and the `boards-v3` method stamp (peter-specific wording); rename to `measured_contacts.md` if the doc is meant to be rig-neutral.
3. `uv run --no-sync` still appears in ~60 docstrings / dataset READMEs; harmless (it only skips the sync), left as is — a one-line sed if you want them plain `uv run`.
4. Non-prefix dataset basenames (`compare_opencap_plates.py`, `import_opencap.py`, `score_parkour_pose.py`, …) untouched — the brief named prefixes only.
5. The scratch venv `scratchpad/s6/venv_c` (5 GB) can be deleted (your rm; the scratchpad is session-local).
