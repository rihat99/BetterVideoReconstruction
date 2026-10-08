Read brief_common.md, brief_consolidation_common.md, then audit_B_scripts.md §1 (diagnostics table), §2E, §2F, §3.

# Step S — published-model scorers and the diagnostics helper module

Your file set: everything under scripts/diagnostics/ (24 files minus the two already trashed) and the new files below. Nothing else.

Do:
1. `scripts/diagnostics/score_published_contacts.py --benchmark {climbingvideos,climb_wall_3}` replaces score_baseline_contacts.py + score_gt_contacts.py (shared vertex_rows / ours / _count / six_joint; benchmark-specific GT loading selected by --benchmark). Outputs go where each old script wrote them (climb_wall_3: PETER/climb_wall_3_gt/others; climbingvideos: the CV results dir).
2. `scripts/diagnostics/score_published_pose.py --benchmark {climbingvideos,climb_wall_3}` replaces score_baseline_pose.py + score_gt_pose.py the same way. score_init_world.py imports both old ones: repoint it.
3. `scripts/diagnostics/_common.py` (one module, plain functions): the helpers that other diagnostics import from the campus_board-era scripts — `_render_table` (from score_prediction_contacts), `_limb_members` / `_limb_load` (score_pipeline_contacts), `_wired` (duplicated in render_prediction_overlays + score_prediction_contacts), the draw helpers / `_draw_force_arrows` family and the root constants that render_kindyn_overlays, render_sup_force_overlays, render_parkour_method_overlays, compare_climb_wall_3_runs and compare_gt_contact import from render_prediction_overlays / score_prediction_contacts / score_pipeline_contacts. Find the exact import lists with grep before you start. Then trash score_prediction_contacts.py, score_pipeline_contacts.py and render_prediction_overlays.py (campus_board-only entry points, last real use Aug–Sep). If any of the three still has an entry-point behaviour that another live script runs as a subprocess, keep that one and say why.
4. Repoint every importer; verify with `uv run --no-sync python -c "import ..."` for each diagnostics module (run from repo root, same import style the scripts already use).

Equivalence: run HEAD copies and new scorers on the real benchmarks (climb_wall_3 and the ClimbingVideos test split, `--limit`/subset if a flag exists; else full — these are CPU/light GPU scorers, should be minutes) writing JSON/tables to scratchpad dirs, and diff the numbers (exact or allclose). For the renderers, run one overlay clip old vs new to a scratchpad dir and compare frame count + a checksum or np.allclose on a sampled frame. Report commands and results.
Report: /home/rikhat.akizhanov/better/BetterVideoReconstruction/agent_notes/tasks/2026-10-08_17-46_repo_cleanup/subagents/report_S_scorers_diagnostics.md
