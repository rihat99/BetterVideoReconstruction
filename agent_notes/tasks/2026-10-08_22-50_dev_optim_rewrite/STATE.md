# STATE — BetterVideoReconstruction (dev worktree) / 2026-10-08_22-50_dev_optim_rewrite (updated 2026-10-09 13:55 Dubai)

## Pointers
- Task status: DONE. PLAN.md final (14 decisions); all code committed on dev (last 96feaa5 = Dev 9/N viewer look); REPORT.md final; index line dropped.
- Worktree: /home/rikhat.akizhanov/better/dev/BetterVideoReconstruction (branch dev: 69d338b ← ebf3c99 step 7 ← ba64306 step 6 ← 28f023e 4c ← d282814 4b ← e2e7bb7 4a ← 9796bba step 5 ← fffc58a step 3 ← 46ebb69 step 2 ← main 4e2c152; working tree clean). Own .venv (torch 2.8.0+cu129, editable dev/BetterRobot + dev/BetterHuman); env doc agent_notes/docs/dev_env.md. Use `uv run --no-sync`, CUDA_VISIBLE_DEVICES=2, BVR_DATA_ROOT=/home/rikhat.akizhanov/better/data_dev.
- Key docs (all under the task folder /home/rikhat.akizhanov/better/dev/BetterVideoReconstruction/agent_notes/tasks/2026-10-08_22-50_dev_optim_rewrite/):
  PLAN.md; subagents/inv_A_br_optim.md (BR optim API, bit-exact parity probe via TorchOptim+scalar terms); inv_B_bh_env.md (BH API delta table, env); inv_C_bvr_map.md (generic/dataset map, loss inventory, retire list); inv_D_viewer.md (BR viewer vs BVR viewer); report_0_reference.md (main refs A/B in data_dev, noise floor, GATE + tolerances); report_1_env.md; report_G_br_gaps.md (BR dev commits 72a7d02 0c3d167 b6602aa 64d3fb5 2dd0a9d 1f3fbe3 4b7b1da: per-term diagnostics, valid masks, timeline shrink, ImagePanel, per-step colours, tilted ground, set_fps); brief_2_api_migration.md; brief_3_optim_port.md + brief_5_viewer.md written (ready to launch after step 2 commit).
  compare_runs.py (task folder) compares two out roots; artifacts/ref/ has run scripts + det_wrapper.py (deterministic mode).
- Previous task (main repo cleanup) is done: main 4e2c152; REPORT at /home/rikhat.akizhanov/better/BetterVideoReconstruction/agent_notes/tasks/2026-10-08_17-46_repo_cleanup/REPORT.md.

## Running jobs
- Step 3 DONE + committed (dev fffc58a; report_3_optim_port.md final, gate PASS, pytest 233/4 skipped). tests/test_contact_confidence.py fix (uses the new viewer loader) is uncommitted and goes with the viewer commit.
- Step 5 DONE + committed (dev 9796bba: viewer on BR/BH layers incl. G2 solid arrows + 5b fixes; pytest 236/4 skipped; reports 5, 5r, G2, 5b).
- Step G2 DONE: BR dev commit 8828842 (ArrowsLayer shaft_radius=/head_radius= solid style, 1674 tests); BVR layers.py uses old radii; screenshots vis/viewer_boulder_1{,_forces,_forces_vs_old,_forces_vs_old_close}.png; report_G2_solid_arrows.md.


## Next steps
1. DONE 13:50: step 5c handed back (report_5c_viewer_look.md); committed 96feaa5 (+ BR dev 26dfb8d); user's viewer restarted on the new code (pid 1506808, port 8082, HTTP 200). Original instruction kept for reference: When step 5c (viewer look) hands back: read its report (save to subagents/report_5c_viewer_look.md if returned as text), check tests/test_viewer_* + screenshots vis/viewer_boulder_1{,_forces,_torque}.png, commit viewer files as "Dev 9/N: viewer sidebar and skeleton look as before" (+ BR dev commit hash in the message), then RESTART the user's viewer: kill pid 2670485, relaunch `cd dev worktree; nohup env BVR_DATA_ROOT=/home/rikhat.akizhanov/better/data CUDA_VISIBLE_DEVICES=2 uv run --no-sync python scripts/view_3d.py /home/rikhat.akizhanov/better/data/examples/out > <scratchpad>/viewer/view_3d.log 2>&1 &`, curl 200, tell the user (http://10.67.33.41:8082) and ask what else looks off.
2. DONE 13:40: investigation F handed back; artifacts/force_diff/report_F.md saved; REPORT.md §4 paragraph added; user told.
3. DONE 13:55: finalize REPORT.md (commit table incl. Dev 9/N), commit agent_notes, drop the task line from the index (/home/rikhat.akizhanov/better/BetterVideoReconstruction/.git/autoresearch/STATE.md), tell the user. TG not configured.
4. Leave running for the user: the viewer process (pid 1506808, port 8082, launched from the dev worktree with BVR_DATA_ROOT=/home/rikhat.akizhanov/better/data, log in the session scratchpad viewer/view_3d.log). Kill it when they say so. Possible follow-ups the user may ask for: more viewer-look tweaks; bounded-λ force solver (report_F).

## Assumptions pending user confirmation
- Regression gate = report_0 recommendation (×3 same-input noise per stage, pooled F1 ±0.01); dev gets an env-var deterministic switch.
- Contact-frame JSON stays in BetterHuman (resolved from the installed package).

## Notes for REPORT.md
- Trashed files from the dev worktree land in the MAIN checkout's trash/ (the helper routes a worktree's trash there): /home/rikhat.akizhanov/better/BetterVideoReconstruction/trash/.
- kindyn_1.npz `source_labels` kept as constant "auto" for compare key identity; drop in step 6 (open question).
- Scratch /tmp farms left by viewer checks: /tmp/bvr_view_i7lz5o_l, /tmp/bvr_view_upj2jo06, /tmp/bvr_view_5_90h8hs (need user rm).
- Step 6 to-do list from 4b open questions: rename boards.py/measure_contacts_boards/`contacts = "boards"` rig-neutral; delete uncalled annotation_frame_labels/ANN_TO_FRAMES/resolve_annotation_path (+5 tests); drop redundant basename prefixes (opencap/opencap_plates.py, peter/peter_sensor_forces.py); drop kindyn_1.npz constant `source_labels`; CLAUDE.md layout/README/INSTALL rewrite.
