# STATE — BetterVideoReconstruction (dev worktree) / 2026-10-08_22-50_dev_optim_rewrite (updated 2026-10-09 08:00 Dubai)

## Pointers
- Task status: PLAN.md final (11 decisions + off-hours assumptions); step 2 DONE + committed (dev 46ebb69 code, a7027b9 notes; report_2 §4 gate PASS); step 3 committed (fffc58a); step 5 awaiting review + G2; REPORT.md not started.
- Worktree: /home/rikhat.akizhanov/better/dev/BetterVideoReconstruction (branch dev: fffc58a = step 3 on 46ebb69 = step 2 on main 4e2c152; step 5 viewer edits + test_contact_confidence.py uncommitted). Own .venv (torch 2.8.0+cu129, editable dev/BetterRobot + dev/BetterHuman); env doc agent_notes/docs/dev_env.md. Use `uv run --no-sync`, CUDA_VISIBLE_DEVICES=2, BVR_DATA_ROOT=/home/rikhat.akizhanov/better/data_dev.
- Key docs (all under the task folder /home/rikhat.akizhanov/better/dev/BetterVideoReconstruction/agent_notes/tasks/2026-10-08_22-50_dev_optim_rewrite/):
  PLAN.md; subagents/inv_A_br_optim.md (BR optim API, bit-exact parity probe via TorchOptim+scalar terms); inv_B_bh_env.md (BH API delta table, env); inv_C_bvr_map.md (generic/dataset map, loss inventory, retire list); inv_D_viewer.md (BR viewer vs BVR viewer); report_0_reference.md (main refs A/B in data_dev, noise floor, GATE + tolerances); report_1_env.md; report_G_br_gaps.md (BR dev commits 72a7d02 0c3d167 b6602aa 64d3fb5 2dd0a9d 1f3fbe3 4b7b1da: per-term diagnostics, valid masks, timeline shrink, ImagePanel, per-step colours, tilted ground, set_fps); brief_2_api_migration.md; brief_3_optim_port.md + brief_5_viewer.md written (ready to launch after step 2 commit).
  compare_runs.py (task folder) compares two out roots; artifacts/ref/ has run scripts + det_wrapper.py (deterministic mode).
- Previous task (main repo cleanup) is done: main 4e2c152; REPORT at /home/rikhat.akizhanov/better/BetterVideoReconstruction/agent_notes/tasks/2026-10-08_17-46_repo_cleanup/REPORT.md.

## Running jobs
- Step 3 DONE + committed (dev fffc58a; report_3_optim_port.md final, gate PASS, pytest 233/4 skipped). tests/test_contact_confidence.py fix (uses the new viewer loader) is uncommitted and goes with the viewer commit.
- Step 5 viewer rebuild DONE (uncommitted; report_5_viewer.md saved; 19 viewer tests pass; screenshots vis/viewer_boulder_1{,_forces}.png). Review agent (mars_aux, brief_5r_viewer_review.md) running: live view_3d process + --bedlam check + adversarial pass → report_5r_viewer_review.md. Launched 01:30 Dubai.
- Step G2 (mars_top_opus, brief_G2_solid_arrows.md): solid arrow style in BR dev ArrowsLayer (commits on dev/BetterRobot) + BVR tools/viewer/layers.py force/wrench/sensor/gravity/normal arrows switched to it + re-recorded screenshots; report → report_G2_solid_arrows.md. Launched 02:00 Dubai (user asked for clear arrows like before; PLAN decision 12).
- Steps 3b, 5-review and G2 share the dev working tree (disjoint files) and GPU 2.

## Next steps
2. Viewer commit waits for BOTH the step-5 review and G2. When they report: fix real bugs (small → myself or a mars_top_opus), then commit ONLY the viewer paths (tools/viewer/*, scripts/view_3d.py, tests/test_viewer_*, tools/geometry/surface.py deletion) as "Dev 3/N: viewer on BetterRobot/BetterHuman layers" (step 3 files may still be dirty — add explicitly).
3. Then step 4 (bvr/ package, datasets/<name>/ split, retire list, base+override configs, single runner, corpus → datasets/climbingvideos), step 6 (Option C env, docs, adversarial review, REPORT.md), TG ping (TG not configured on this machine: token missing in ~/.config/autoresearch/telegram.env).

## Assumptions pending user confirmation
- Regression gate = report_0 recommendation (×3 same-input noise per stage, pooled F1 ±0.01); dev gets an env-var deterministic switch.
- Contact-frame JSON stays in BetterHuman (resolved from the installed package).
