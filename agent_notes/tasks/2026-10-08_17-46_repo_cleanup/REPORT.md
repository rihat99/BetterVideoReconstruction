# REPORT — BetterVideoReconstruction repo cleanup (2026-10-08)

Repo: /home/rikhat.akizhanov/better/BetterVideoReconstruction, branch main. Five commits, 5a1fc9a → e06721f, plus two small commits in each of contact_anything (916250c, d71d064) and BetterContactAnnotator (91c37af, 4c4e8c3). Net: 125 files, +3364 / −13634 lines. The core optimization code (tools/human_optim, tools/optim.py, scripts/stages) is untouched except two one-line path defaults.

## What changed

| commit | step | content |
|---|---|---|
| 8e38aec | 1/5 purge | trashed benchmark_tables/, experiments/, scratch/ (calibration provenance kept in peter/_provenance), root dataset_split JSON, 11 one-off scripts, 11 configs; run_parkour → pipeline_peter.toml; label_amass.py added |
| 0cf1281 | 2/5 move | data, data_2, out, out_2 → ../data/examples/<name>; peter → ../data/peter; new tools/paths.py resolver (BVR_DATA_ROOT override) replaces every per-script root constant (60 files); 625 symlinks re-pointed; sibling repos fixed |
| 8bda531 | 3/5 runner | scripts/run_dataset.py + configs/datasets.toml replace run_parkour / run_opencap / run_peter / run_gt_contact |
| b0b3dce | 4/5 merges | scripts/methods/{estmf_inputs,physpt_inputs,baseline_farm}.py --dataset replace 7 exporters; diagnostics/score_published_{contacts,pose}.py --benchmark replace 4 scorers; diagnostics/_common.py absorbs the helpers of 3 campus_board-era scripts, which retire |
| e06721f | 5/5 docs + guards | CLAUDE.md / README / INSTALL for main + new layout; run_dataset refuses a missing config and a farm link that already points elsewhere; --dry-run touches nothing; anchored data ignores |

Data layout now: `/home/rikhat.akizhanov/better/data/examples/{data,data_2,out,out_2}` (32M, 37M, 11G, 2.3G), `/home/rikhat.akizhanov/better/data/peter` (69G, with `_provenance/` holding the cw2/cw3 calibration scripts and rotation .npy files from the old scratch/).

Configs kept (9): pipeline, pipeline_peter, pipeline_opencap, human_optimization, human_optimization_peter, human_optimization_floorplane, human_optimization_floorplane_edgev, corpus, datasets. Floorplane evidence: all variants ran on OpenCap LabValidation; `floorplane` is the reference run that tests and method scripts read; `edgev` is the last variant run (Sep 17) whose knobs were adopted into pipeline.toml. Note the on-disk edgev/massedgev farms were kindyn-only reruns no committed code reproduces (documented in datasets.toml; a real `--variant floorplane_edgev` now stops with a link-conflict message instead of silently re-pointing those farms).

Scripts: 95 → 77 files under scripts/. Trashed scripts (git keeps them): run_flat_torque, run_data2, run_athletics_chain.sh, score_contacts, diagnostics/contacts_control, motion_smoothness, compare_climb_wall_2, emdb_split_clips, emdb_to_smplx, import_peter, backfill_static_camera, the 4 runners, 7 exporters, 4 scorers, 3 campus_board-era diagnostics.

## Verification
- Full pytest (250 tests) passes after every step; last run 37.8 s on GPU 2.
- `--help` / import of every changed script OK; no code path resolves data/out/peter against the repo or cwd (reviewer grep).
- Runner: 21/21 synthetic cases produce identical stage commands to the old runners (200 commands); 7 real `--skip-existing` runs on processed clips skip everything in 0.1 s.
- Exporters: bit-identical outputs on every clip of every dataset (parkour 28, opencap 64×2, climb_wall_3 68; PhysPT 28/68/64/2; farms 2353 + 484 files).
- Scorers and renderers: byte-identical JSON/PNG on both full benchmarks; overlays frame-equal (5 clips).
- pipeline.py --skip-existing on boulder_1 through the moved tree: all stages skip, rc=0 (sapiens stage needs GPU memory to load before it checks skips; GPU 2 was 86 GB occupied by another user, so that stage was excluded from the smoke run — pre-existing behaviour).
- Adversarial review (subagents/review_V_findings.md): no broken path; 2 medium + 6 low findings, all fixed or documented (report_F_review_fixes.md).
- Not verified: no real GPU stage executed through run_dataset; baseline_farm on climb_wall_3 only decoded one clip (metadata and all 544 links checked for all 68).

## Known leftovers (your call)
- `/home/rikhat.akizhanov/better/_talk_figs/scripts/rig_plots.py` and result_plots.py import two trashed scripts (compare_climb_wall_2, score_pipeline_contacts); recover from trash/ or git if the talk figures are needed again.
- Pre-existing failures untouched: render_sup_force_overlays on campus_board clips (older forces_sup.npz format), diagnostics/score_test_contacts.py (KeyError pred_limbs), scene_db.ATHLETICS_DATA_ROOT (dataset absent).
- `configs/corpus.toml data_root` is repo-relative, not BVR_DATA_ROOT (documented in CLAUDE.md).
- peter_sensor_forces.py still carries R_SENSOR_TO_WORLD / CALIB_CLIPS / NOTE for the vanished climb_wall rig (record only).
- External docstrings naming old script names: data/Estimating-3D-Motion-Forces (NOTES.md, scripts), other/PhysPT, contact_anything/scripts/physpt_dumps.py.
- `trash/` holds 954 MB (47 entries incl. manifest backups); empty it when satisfied.

## Files
- Audits: subagents/audit_A_configs_paths.md, audit_B_scripts.md
- Step reports: subagents/report_{P_purge,M_move_paths,R_runners,I_method_inputs,S_scorers_diagnostics,D_docs,F_review_fixes}.md
- Review: subagents/review_V_findings.md
