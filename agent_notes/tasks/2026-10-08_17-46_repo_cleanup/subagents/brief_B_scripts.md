Read brief_common.md in this folder first.

# Audit B — scripts and tools inventory, duplicates

Report path: /home/rikhat.akizhanov/better/BetterVideoReconstruction/agent_notes/tasks/2026-10-08_17-46_repo_cleanup/subagents/audit_B_scripts.md

1. Inventory every .py/.sh/.toml under scripts/ (top level, scripts/diagnostics, scripts/methods, scripts/peter_gt, scripts/stages — stages are the pipeline core, list them but they are not cleanup candidates) and the top-level modules in tools/ (tools/*.py, tools/adapters, tools/geometry, tools/mhr_to_smplx, tools/smplx_robot, tools/viewer; tools/human_optim is the core, list only). For each: one row — path | purpose (1 line, from docstring/argparse) | inputs → outputs (which dirs/datasets) | imported-by / referenced-by (grep the repo incl. CLAUDE.md, README.md, docs/, .sh files, pyproject.toml) | last commit date (`git log -1 --format=%cs -- path`; "untracked" if not tracked) | verdict candidate: core / keep / consolidate / likely-dead. Note untracked files (git status) explicitly.
2. Duplicate families. Inspect and diff these groups and any others you find; for each group say what is shared, what differs, and whether one parametrised script (argument/flag) could replace them, with a concrete sketch of the CLI (e.g. `scripts/run_dataset.py --dataset {data2,opencap,parkour,peter}`): 
   - scripts/run_data2.py, run_opencap.py, run_parkour.py, run_peter.py, run_flat_torque.py, run_gt_contact.py, run_athletics_chain.sh, corpus.py, pipeline.py
   - scripts/import_*.py (bedlam2, opencap, parkour, peter) and label_*.py
   - scripts/*_sensor_forces.py (opencap, parkour, peter), opencap_plates.py, sync_opencap.py, export_opencap_mot.py
   - scripts/parkour_estmf_inputs.py, parkour_physpt_inputs.py, parkour_prepare_interactvlm.py vs scripts/methods/*_inputs.py, *_export.py, cw3_estmf_inputs.py
   - scripts/score_contacts.py, score_frame_contacts.py, calibrate_contact_confidence.py, recompute_contact_confidence.py vs scripts/diagnostics/score_*.py
   - scripts/diagnostics/render_*_overlays.py, compare_*.py
   - scripts/backfill_static_camera.py, assign_dataset_split.py, emdb_*.py, scene_db.py, view_3d.py, _pym_preload.py
   - scripts/peter_gt/* (is this a self-contained GT pipeline? which parts are reused elsewhere?)
3. Shared helpers: are there functions copy-pasted across scripts (same name / near-identical body)? List the top offenders with file:line.
4. What does CLAUDE.md say the canonical entry points are? Compare against the inventory: anything documented but missing, or present but undocumented.
Deliver a ranked proposal: (i) safe to trash now (nothing references it, superseded), (ii) consolidation candidates with sketched CLI, (iii) keep as is. Evidence for each.
