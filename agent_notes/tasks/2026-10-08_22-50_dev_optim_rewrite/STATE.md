# STATE — BetterVideoReconstruction (dev worktree) / 2026-10-08_22-50_dev_optim_rewrite (updated 2026-10-09 00:35 Dubai)

## Pointers
- Task status: PLAN.md final (11 decisions + off-hours assumptions); REPORT.md not started.
- Worktree: /home/rikhat.akizhanov/better/dev/BetterVideoReconstruction (branch dev = main 4e2c152, NO commits yet; step 2 edits are uncommitted in the working tree). Own .venv (torch 2.8.0+cu129, editable dev/BetterRobot + dev/BetterHuman); env doc agent_notes/docs/dev_env.md. Use `uv run --no-sync`, CUDA_VISIBLE_DEVICES=2, BVR_DATA_ROOT=/home/rikhat.akizhanov/better/data_dev.
- Key docs (all under the task folder /home/rikhat.akizhanov/better/dev/BetterVideoReconstruction/agent_notes/tasks/2026-10-08_22-50_dev_optim_rewrite/):
  PLAN.md; subagents/inv_A_br_optim.md (BR optim API, bit-exact parity probe via TorchOptim+scalar terms); inv_B_bh_env.md (BH API delta table, env); inv_C_bvr_map.md (generic/dataset map, loss inventory, retire list); inv_D_viewer.md (BR viewer vs BVR viewer); report_0_reference.md (main refs A/B in data_dev, noise floor, GATE + tolerances); report_1_env.md; report_G_br_gaps.md (BR dev commits 72a7d02 0c3d167 b6602aa 64d3fb5 2dd0a9d 1f3fbe3 4b7b1da: per-term diagnostics, valid masks, timeline shrink, ImagePanel, per-step colours, tilted ground, set_fps); brief_2_api_migration.md; brief_3_optim_port.md + brief_5_viewer.md written (ready to launch after step 2 commit).
  compare_runs.py (task folder) compares two out roots; artifacts/ref/ has run scripts + det_wrapper.py (deterministic mode).
- Previous task (main repo cleanup) is done: main 4e2c152; REPORT at /home/rikhat.akizhanov/better/BetterVideoReconstruction/agent_notes/tasks/2026-10-08_17-46_repo_cleanup/REPORT.md.

## Running jobs
- Subagent step 2 (mars_top, "BR/BH API migration on dev"): edits dev worktree (tools/body.py → BH dev API; kindyn/forces/bedlam_contacts/label_amass → BR dev FK/RNEA/Jacobians; dissolve tools/smplx_robot; remove mhr_1_sup + kindyn_manual + dead sys.path hacks), then pytest, then regression on GPU 2 (4 clips, dev vs data_dev/examples/out/<clip>/human_optim_A via compare_runs.py). Its report → subagents/report_2_api_migration.md (or returned as text; save it). Expected finish within ~1–2 h of 01:00.

## Next steps
1. When step 2 reports: check pytest green + regression within report_0 gate; `git add -A . ':!agent_notes'` + commit "Dev 1/N: BetterHuman + BetterRobot dev API migration" on dev; save report; commit agent_notes.
2. Launch in parallel: step 3 optim port (mars_top; brief to write: remove tools/optim.py, bvr problems with cached state + scalar terms on BR TorchOptim factories, per-term logging via State.extra, kindyn warm start as second Problem, witness via bounds, converter + multi-view fit; deterministic env switch; gate per report_0) and step 5 viewer rebuild on BR/BH layers (mars_top_opus; inv_D mapping; dataset layers → datasets/<name>/ later).
3. Then step 4 (bvr/ package, datasets/<name>/ split, retire list, base+override configs, single runner, corpus → datasets/climbingvideos), step 6 (Option C env, docs, adversarial review, REPORT.md), TG ping (TG not configured on this machine: token missing in ~/.config/autoresearch/telegram.env).

## Assumptions pending user confirmation
- Regression gate = report_0 recommendation (×3 same-input noise per stage, pooled F1 ±0.01); dev gets an env-var deterministic switch.
- Contact-frame JSON stays in BetterHuman (resolved from the installed package).
