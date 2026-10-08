# PLAN — repo cleanup (2026-10-08)

Repo: /home/rikhat.akizhanov/better/BetterVideoReconstruction, branch main, commits directly to main.
Scope: scripts, configs, tools helpers, data layout, root files. NOT touched: tools/human_optim, tools/optim.py, scripts/stages/* (core optimization).

## Decisions from intake (user)
- Data root: /home/rikhat.akizhanov/better/data. data, data_2, out, out_2 -> data/examples/<name>; peter -> data/peter.
- Paths: update code/configs (single resolver), no symlinks.
- Removals: everything via trash (repo trash/), nothing rm'd. benchmark_tables, experiments, scratch go.
- GPU: 2 only (CUDA_VISIBLE_DEVICES=2) for any smoke run.

## Steps
1. Audit (read-only, 2 agents): configs usage + hardcoded paths + root files + tests (A); scripts/tools inventory + duplicate families (B).
2. Move data dirs; fix every path hit from audit A → verify: `uv run pytest tests/` collection + non-GPU tests pass; grep shows no stale relative data/out/peter refs; one smoke run of the pipeline on an example video (GPU 2) if feasible.
3. Trash benchmark_tables, experiments, scratch, __pycache__, root loose files agreed with user → verify: git status clean of them, tests still collect.
4. Configs: keep only the floorplane variant actually used (per audit evidence, confirmed by user); drop stale pipeline_*/corpus* variants; → verify: every remaining config loads via the project's config loader.
5. Scripts/tools consolidation per audit B, each merge confirmed with the user → verify: `--help` of each new CLI, dry-run / small run where data allows, tests.
6. Update CLAUDE.md / README / INSTALL to the new layout; REPORT.md; adversarial pass (mars_aux) on the final diff.

Each step = one commit on main.

## Decisions round 2 (user, 2026-10-08 ~18:30)
- Floorplane configs: keep `human_optimization_floorplane.toml` + `_floorplane_edgev.toml`; drop mass, edge, edge3k, massedge, massedge3k, massedgev.
- Drop: human_optimization_labvalidation.toml; pipeline_data2.toml + run_data2.py; corpus_athletics*.toml + run_athletics_chain.sh; pipeline_parkour.toml merged into pipeline_peter.toml (parkour uses peter's VGGT chunking from now on; numerics differ ~3e-4).
- Core: one-line default-path edits allowed in tools/human_optim/stages.py:779-782 and tools/viewer/loading.py:420.
- Outside refs: re-point the 612 absolute symlinks; update path defaults in contact_anything and BetterContactAnnotator.
- Runners → one `scripts/run_dataset.py` + `configs/datasets.toml`.
- Trash: run_flat_torque, diagnostics/motion_smoothness, diagnostics/compare_climb_wall_2, emdb_split_clips, emdb_to_smplx, import_peter, backfill_static_camera, score_contacts, diagnostics/contacts_control, root dataset_split_assign_*.json (future records next to the DB).
- Commit scripts/label_amass.py.
- scratch provenance (cw2/cw3 import/trim/calibration scripts + R_*.npy) → data/peter/_provenance; rest of scratch trashed.
- Also merge: method inputs (estmf trio, PhysPT pair, baseline farms → scripts/methods/*_inputs.py --dataset; parkour_* into methods/); published-model scorers (--benchmark); diagnostics shared-helper module then trash campus_board-era entry points; benchmark table writers → data/peter/benchmark_tables.
- Verification: full pytest (--no-sync, GPU 2), --help of every changed CLI, broken-symlink scan, pipeline.py --skip-existing on one moved example clip (GPU 2), run_dataset dry-run per dataset.

## Execution order (one commit each, agents do not commit)
P. purge (trash lists, config drops, provenance, label_amass, doc refs)
M. move data + tools/paths.py resolver + all path sites + core 2 spots + symlinks + sibling repos + README example
R/I/S in parallel: runners; method inputs; scorers + diagnostics helpers
D. docs (CLAUDE.md layout, README, INSTALL), smoke checks, adversarial review, REPORT.md
