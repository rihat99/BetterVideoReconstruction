# Report 1 — dev environment, Option A (mars_top, 2026-10-08)

Verdict: the dev env is built and verified. All third-party imports work and CUDA works on GPU 2. The BVR test suite (main: 250 pass) gives 104 passed, 1 failed, 9 test files not collected. Every failure comes from the BR/BH dev API change. Two BR symbols block all collection today: `update_frame_placements` (tools/body.py:28) and `compute_joint_jacobians` (tools/bedlam_contacts.py:57). Wall time ≈ 5 min; all installs came from the uv cache.

## Steps
1. `git submodule update --init` took 22 s. Five submodules are checked out at the same commits as main: MHR 6d14668, geocalib 747054d, sam-3d-body b5c765a, sam3 44ef224, sapiens2 0e51c12. Size is ≈230 MB on disk. The git object stores went to main's `.git/worktrees/BetterVideoReconstruction/modules/` (240 MB). That is standard git worktree layout, and main's working tree is untouched (clean, HEAD 4e2c152). I symlinked `third_party/MHR/assets` → main's assets (212 MB). Inside the dev MHR submodule the symlink shows as `?? assets`, because the pattern `assets/` matches directories only. Harmless.
2. `uv sync --frozen` took 4.9 s from cache. Versions: CPython 3.12.14, torch 2.8.0+cu129, torchvision 0.23.0+cu129, pymomentum-gpu 0.1.110.post0, numpy 2.4.6, viser 1.0.30. Every editable finder (mhr, sam3, sapiens, vggt-omega, geocalib) points into dev/. I grepped the .pth, finder and direct_url files: 0 refer to main.
   - `uv pip install -e ../BetterRobot -e ../BetterHuman --no-deps`: better-robot 0.2.0 → dev/BetterRobot/src, better-human 0.1.0.dev0 → dev/BetterHuman/src.
   - detectron2 0.6 (a1ce2f9): a cached CPU wheel, installed in 0.9 s. `_C` has no CUDA, and the .so is byte-size-identical to main's (18 656 784 B).
   - `sam-3d-body.pth` → dev/third_party/sam-3d-body.
3. Verify (`CUDA_VISIBLE_DEVICES=2 .venv/bin/python`):
   - better_robot / better_human load from /home/rikhat.akizhanov/better/dev/...; torch 2.8.0+cu129 sees cuda=True, 1 device (RTX PRO 6000 Blackwell), and a test matmul ran.
   - OK: viser, yourdfpy, pypose, mhr, mhr.mhr, pymomentum.geometry, detectron2(+.config), sam3(+.model_builder), sam_3d_body, sapiens(+pose.models, dense.models), vggt_omega(+models), geocalib, smplx, utils3d, open3d, decord, imageio_ffmpeg, trimesh.
   - These cover every module that tools/adapters/*.py and scripts/stages/*.py import.
   - `_pym_preload` re-exec plus `MHR.from_files(lod=1)` loads through the assets symlink. Caveat: `torch` must be imported before `pymomentum`, otherwise `ImportError: initialization failed`. The stages already do this (convert_mhr_to_smplx.py:57,67).
4. Tests, command as briefed. Collection errors stop pytest, so `-x --maxfail=200` alone gives only "9 errors during collection". I re-ran with `--continue-on-collection-errors`: 105 collected, 104 passed, 1 failed, 9 errors, 4.8 s. About 145 tests (134 `def test_` plus parametrize) sit in the 9 uncollected files.
5. Wrote `agent_notes/docs/dev_env.md` (15 lines).

## Failure list, grouped by root cause (the step-2 migration work list)
The runtime failures (pytest):
- A. `ImportError: cannot import name 'update_frame_placements' from 'better_robot.kinematics'` at tools/body.py:28. It hits 8 collection errors and 1 failure:
  - test_bedlam_forces.py:15 (via tools/bedlam_forces.py:33 → human_optim/kindyn.py:43 → body.py)
  - test_body_density.py:8, test_body_sole_offset.py:8, test_contact_frames.py:11, test_kin2_kindyn_frames.py:25, test_occlusion_weighting.py:19
  - test_opencap_forces.py:27 (via scripts/export_opencap_mot.py:78), test_support_scene.py:32 (via scripts/stages/build_support_scene.py:63)
  - FAILED test_fuse_scale.py::test_kinematic_init_prefers_matching_prefit, at test_fuse_scale.py:264 → scripts/stages/human_optimization.py:44 → kindyn.py:43 → body.py:28
- B. `ImportError: cannot import name 'compute_joint_jacobians' from 'better_robot.kinematics.jacobian'` at tools/bedlam_contacts.py:57. It hits 1 collection error: test_bedlam_labels.py:20.

The static probe imports every BR/BH symbol BVR uses into the dev venv (scratch script `probe_symbols.py`). These breaks sit behind A/B and will surface once A/B are fixed:
- C. BR kinematics removed (`better_robot.kinematics` and `.jacobian`): `update_frame_placements`, `forward_kinematics`, `compute_joint_jacobians`, `get_frame_jacobian`.
  - Used at tools/body.py:28; human_optim/kindyn.py:214,369,370; human_optim/forces.py:129-131; bedlam_contacts.py:57.
  - Dev replacements exist: `kinematics.forward_kinematics_state`, `kinematics.frame_jacobian`.
- D. BR dynamics: `better_robot.dynamics.rnea` is missing. Used at human_optim/kindyn.py:250,368 and forces.py:128. Dev has `dynamics.inverse_dynamics(model, state, v=, a=, fext=)`.
- E. BR module `better_robot.dynamics.centroidal` is gone (scripts/label_amass.py:159). Dev has `dynamics.center_of_mass(model, state)`.
- F. BH package `better_human.bodies.*` is gone:
  - tools/body.py:32 (`smpl_family.config_smplx.SMPLX_CONFIG`) → `families.tables.SMPLX`
  - tools/body.py:33 (`smpl_family.smplx.SMPLXClassic`) → `families.classic.SMPLXClassic` / `smplx_from_classic` / `smplx_to_classic`
  - tools/viewer/loading.py:722 (`bodies.mhr.body.MHRClassic`) → `families.mhr.MHRClassic` / `mhr_from_classic`
- G. BH top-level classes are gone: `bh.SMPLX` (tools/body.py:136,344) → `families.smplx` + `Body` + `load_density_table`; `bh.ParametricBody` (body.py:209,219) → `Identity` from `Body.bake`; `bh.MHR` (viewer/loading.py:687) → `bh.load("mhr", ...)`. All dev replacements resolve.
- H. Still fine on dev: `better_robot.lie.se3/so3` (body.py:29, camera.py:18, motion.py:34, viewer/loading.py:720, test_contact_frames.py:9).
- Not covered by the probe (attribute-level, runtime only; see inv_B §1): `.shaped(...)`, `.robot`, `.values.*`, `data.joint_pose_world`, `frame_pose_world` and the `dataclasses.replace(model.values, gravity=)` sites listed in inv_B §1 / inv_A §4.

## Torch note (from inv_B, not re-run)
BR/BH dev venvs run torch 2.13; BVR stays on 2.8 because of pymomentum-gpu. On 2.8 the only known red test is BR `test_matrix_fk_jacrev_equals_jacfwd_on_mixed_kinds` (torch.func plus `.tolist()`, `_fk_lean.py:51`). BVR and BR optim do not use torch.func.

## Not done / open
- I did not fix anything and made no commits. I did not run a pipeline smoke and did not re-run inv_B's parity numbers.
- I did not run main's suite for a baseline; the 250-pass figure comes from the cleanup REPORT.md:22.
- Open: should the MHR `assets` symlink go into the submodule's `.git/info/exclude`? It needs no repo change.
