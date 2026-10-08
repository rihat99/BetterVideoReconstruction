Read brief_common.md, brief_consolidation_common.md, then audit_B_scripts.md §2D and §3.

# Step I — method-input exporters

Your file set: scripts/parkour_estmf_inputs.py, scripts/parkour_physpt_inputs.py, scripts/parkour_prepare_interactvlm.py, scripts/methods/* (all 8), scripts/peter_gt/prepare_baselines.py and scripts/peter_gt/run_baselines.sh (only to repoint at the new farm script if it calls prepare_baselines), plus the new files below.

Do:
1. `scripts/methods/estmf_inputs.py --dataset {parkour,opencap,climb_wall_3} [--clips ...] [--contact-mapping heel|any]` replaces parkour_estmf_inputs.py, methods/opencap_estmf_inputs.py, methods/cw3_estmf_inputs.py. One copy of coco18_from_sapiens, smplx_model, on_true_camera and the constants; dataset-specific parts (contact_states source, cw3 gravity + principal point) selected by --dataset. Output locations unchanged (they feed the external Estimating-3D-Motion-Forces repo).
2. `scripts/methods/physpt_inputs.py --dataset {parkour,climb_wall_3,opencap} [--step N] [--offset K]` replaces parkour_physpt_inputs.py and methods/opencap_physpt_inputs.py. One copy of fit_betas, smpl_rest_joints, smplx_rest_joints, rodrigues.
3. `scripts/methods/baseline_farm.py --dataset {parkour,climb_wall_3} [--skip-existing] [--jobs N]` replaces parkour_prepare_interactvlm.py and peter_gt/prepare_baselines.py; use scene_db.shard instead of the private sub/subdir copies if it is the same logic (check).
4. physpt_model is duplicated in diagnostics/compare_parkour_methods.py:254 and methods/opencap_physpt_export.py:82 — diagnostics is another agent's file set, so leave compare_parkour_methods alone and just note it.
5. Low priority, only if trivial: fold methods/opencap_frames35_export.py into opencap_learned_export.py as `--contact-set {six,frames35}`; otherwise leave both and say so.
6. Trash the superseded scripts; fix every reference (grep the repo for each old basename; docstrings in methods/*; peter_gt/run_baselines.sh).

Equivalence: for each new script and each dataset where data exists, run the HEAD copy and the new script on ONE clip with outputs redirected to scratchpad dirs (add nothing to the real external dirs — if a script has no output-dir flag, add `--out` as a target argument; that is allowed) and diff the produced files (np.allclose for npz/npy, byte diff for text). Report the commands and the diff results.
Report: /home/rikhat.akizhanov/better/BetterVideoReconstruction/agent_notes/tasks/2026-10-08_17-46_repo_cleanup/subagents/report_I_method_inputs.md
