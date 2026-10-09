# STATE — BetterVideoReconstruction (dev worktree) / 2026-10-08_22-50_dev_optim_rewrite (updated 2026-10-09 10:10 Dubai)

## Pointers
- Task status: PLAN.md final (11 decisions + off-hours assumptions); step 2 DONE + committed (dev 46ebb69 code, a7027b9 notes; report_2 §4 gate PASS); step 3 committed (fffc58a); step 5 committed (9796bba); 4a/4b committed (e2e7bb7, d282814), 4c running; REPORT.md not started.
- Worktree: /home/rikhat.akizhanov/better/dev/BetterVideoReconstruction (branch dev: d282814 4b ← e2e7bb7 4a ← 9796bba step 5 ← fffc58a step 3 ← 46ebb69 step 2 ← main 4e2c152; 4c edits uncommitted). Own .venv (torch 2.8.0+cu129, editable dev/BetterRobot + dev/BetterHuman); env doc agent_notes/docs/dev_env.md. Use `uv run --no-sync`, CUDA_VISIBLE_DEVICES=2, BVR_DATA_ROOT=/home/rikhat.akizhanov/better/data_dev.
- Key docs (all under the task folder /home/rikhat.akizhanov/better/dev/BetterVideoReconstruction/agent_notes/tasks/2026-10-08_22-50_dev_optim_rewrite/):
  PLAN.md; subagents/inv_A_br_optim.md (BR optim API, bit-exact parity probe via TorchOptim+scalar terms); inv_B_bh_env.md (BH API delta table, env); inv_C_bvr_map.md (generic/dataset map, loss inventory, retire list); inv_D_viewer.md (BR viewer vs BVR viewer); report_0_reference.md (main refs A/B in data_dev, noise floor, GATE + tolerances); report_1_env.md; report_G_br_gaps.md (BR dev commits 72a7d02 0c3d167 b6602aa 64d3fb5 2dd0a9d 1f3fbe3 4b7b1da: per-term diagnostics, valid masks, timeline shrink, ImagePanel, per-step colours, tilted ground, set_fps); brief_2_api_migration.md; brief_3_optim_port.md + brief_5_viewer.md written (ready to launch after step 2 commit).
  compare_runs.py (task folder) compares two out roots; artifacts/ref/ has run scripts + det_wrapper.py (deterministic mode).
- Previous task (main repo cleanup) is done: main 4e2c152; REPORT at /home/rikhat.akizhanov/better/BetterVideoReconstruction/agent_notes/tasks/2026-10-08_17-46_repo_cleanup/REPORT.md.

## Running jobs
- Step 3 DONE + committed (dev fffc58a; report_3_optim_port.md final, gate PASS, pytest 233/4 skipped). tests/test_contact_confidence.py fix (uses the new viewer loader) is uncommitted and goes with the viewer commit.
- Step 5 DONE + committed (dev 9796bba: viewer on BR/BH layers incl. G2 solid arrows + 5b fixes; pytest 236/4 skipped; reports 5, 5r, G2, 5b).
- Step G2 DONE: BR dev commit 8828842 (ArrowsLayer shaft_radius=/head_radius= solid style, 1674 tests); BVR layers.py uses old radii; screenshots vis/viewer_boulder_1{,_forces,_forces_vs_old,_forces_vs_old_close}.png; report_G2_solid_arrows.md.


## Next steps
1. 4a DONE (e2e7bb7), 4b DONE (d282814: datasets/<name>/ split, 8 READMEs, pytest 234/4, 48/48 --help). 4c (mars_top: base+override configs, one runner, [viewer] DB keys → climbingvideos, full 4-clip regression bit-identical to human_optim_dev3, F1 0.912/0.931) RUNNING since 10:10 → 'SUBSTEP 4c READY' → commit 'Dev 6/N'. One agent per sub-step; successors on context limit (handoff in report_4).
2. Then step 4 (bvr/ package, datasets/<name>/ split, retire list, base+override configs, single runner, corpus → datasets/climbingvideos), step 6 (Option C env, docs, adversarial review, REPORT.md), TG ping (TG not configured on this machine: token missing in ~/.config/autoresearch/telegram.env).

## Assumptions pending user confirmation
- Regression gate = report_0 recommendation (×3 same-input noise per stage, pooled F1 ±0.01); dev gets an env-var deterministic switch.
- Contact-frame JSON stays in BetterHuman (resolved from the installed package).

## Notes for REPORT.md
- Trashed files from the dev worktree land in the MAIN checkout's trash/ (the helper routes a worktree's trash there): /home/rikhat.akizhanov/better/BetterVideoReconstruction/trash/.
- kindyn_1.npz `source_labels` kept as constant "auto" for compare key identity; drop in step 6 (open question).
- Scratch /tmp farms left by viewer checks: /tmp/bvr_view_i7lz5o_l, /tmp/bvr_view_upj2jo06, /tmp/bvr_view_5_90h8hs (need user rm).
- Step 6 to-do list from 4b open questions: rename boards.py/measure_contacts_boards/`contacts = "boards"` rig-neutral; delete uncalled annotation_frame_labels/ANN_TO_FRAMES/resolve_annotation_path (+5 tests); drop redundant basename prefixes (opencap/opencap_plates.py, peter/peter_sensor_forces.py); drop kindyn_1.npz constant `source_labels`; CLAUDE.md layout/README/INSTALL rewrite.
