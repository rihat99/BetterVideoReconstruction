# Investigation B — BetterHuman dev API delta and the dev environment (mars_aux, read-only, 2026-10-08)

Verdict: BH dev rewrote the whole API (main `SMPLX`/`ParametricBody` → dev `FamilySpec` → `Body` → `bake` → `Identity` → `forward` → `HumanState`). The SMPL-X body BVR uses is numerically the same (measured), so the port touches tools/body.py, the MHR viewer path and ~8 `.shaped(...)` call sites. tools/smplx_robot/ can be dissolved. The dev worktree cannot `uv sync` yet (submodules empty). The cu126 reason for hand-installing BR/BH is stale.

## Parity measurements (BVR main venv torch 2.8.0+cu129; dev via PYTHONPATH to dev/BetterRobot/src:dev/BetterHuman/src; random betas + 4 poses, CPU)
| quantity (SMPL-X neutral, use_face=False, 52 joints, Dempster, 35 contact frames) | main vs dev max abs |
|---|---|
| q from classic (4,211); round trip | 0; 0 |
| vertices (4,10475,3) / joint poses (4,52,7) | 6.0e-7 m / 2.4e-7 |
| contact-frame positions (4,35,3) | 2.4e-7 m; 19/140 quaternions sign-flipped (same rotation) → compare rotations, not raw quats |
| template, v_shaped, rest_joints, pelvis, faces, part ids | identical |
| body_inertias (53,10) | 0 (71.114 kg) |
| joint names/parents, 21 keypoints, contact names/parents/frame_ids (53..87), nq 211, nv 159, nframes 88 | identical |
| model gravity | main (0,0,−9.81), dev (0,−9.81,0) (y-up fix); every BVR RNEA path overrides gravity anyway |
| MHR LOD1 (viewer): q (2,132) / joints / vertices | 0 / 3.6e-7 / 1.0e-6 |
Speed (T=300, forward = vertices+joints+frames, main→dev): CPU fwd 91→81 ms, fwd+bwd 223→192 ms; CUDA fwd 4.2→3.7 ms, fwd+bwd 16.6→11.2 ms; bake 26→29 / 36→44 ms. First body load 3.3 s main, 8.1 s dev.

## 1. BH API delta (only tools/body.py and tools/viewer/loading.py import better_human; everything else goes through Body / body.shaped)
| BVR file:line | BH main | BH dev | change |
|---|---|---|---|
| body.py:31-33 | `bh`, `bodies.smpl_family.config_smplx.SMPLX_CONFIG`, `...smplx.SMPLXClassic` | `better_human.bodies` gone → `families.tables.SMPLX`, `families.classic.{SMPLXClassic, smplx_from_classic, smplx_to_classic}` | rewrite imports |
| body.py:344-352 | `bh.SMPLX(path, use_face=False, use_hands=True, num_betas=10, density="Dempster", device, contact_frames=json)` | `spec=bh.families.smplx(path, num_betas=10, face=False, hands=True, device=d)`; `dens=families.assets.load_density_table("body_densities.json","Dempster",[spec.joint_names[j] for j in spec.parts.joints],spec.vertices)`; `bh.Body(spec, density=dens, contact_frames=json)` | dev Body rejects a table name (core/body.py:359-367); keep num_betas=10 (default 16) |
| body.py:141-142,155 | `base.values.v_template` | `body.spec.vertices`, `body.dtype`, `body.device` | rename |
| body.py:145,147,153,163 | `structure.faces/joint_names/num_vertices/parents` | `body.faces/...` | rename |
| body.py:151 | `base.robot.nq` | `body.model.nq` | rename |
| body.py:157-159 | `SMPLX_CONFIG.frame_names/frame_vertex_ids` | `tables.SMPLX.keypoint_names/keypoint_vertex_ids` (`HumanState.landmarks()`) | rename; identical values |
| body.py:161,211-217 | dense LBS from skinning_weights/indices | `body.skin_matrix` (V,52) dense | delete `_dense_lbs_weights` |
| body.py:167-195 | `base.contact_frames` | `body.contact_frames` (same dataclass) | works; cleaner: filtered mapping to `Body(contact_frames=)` |
| body.py:219-242 | `base.with_shape(betas=)` → ParametricBody | `body.bake(betas=)` → Identity | dtype/device/width checked strictly (body.py:346-351) |
| body.py:257-263 | `shaped.fk(q)`, `vertices_from_data`, `data.joint_pose_world[...,1:,:]`, `update_frame_placements` + `frame_pose_world[...,ids]` | `st=identity.forward(q)`; `st.vertices`, `st.joint_poses`, `st.frame_poses[...,ids,:]` | `_Posed.model/.data` → `st.model`, `st.kinematics` (BR KinematicState); bedlam_contacts.py:132-133 Jacobians → BR dev `frame_jacobian(model, state, …)` |
| body.py:277-280 | `values.pelvis_offset/v_shaped/rest_joints` | `identity.pelvis/v_shaped/rest_joints` | same convention |
| body.py:297-307/315-316 | `shaped.from_classic(SMPLXClassic(..., num_pca_comps=None))` / `to_classic` | `q,_=smplx_from_classic(identity, SMPLXClassic(...))` / `smplx_to_classic(identity, q, num_pca_comps=None)` | dev returns (q, stream) |
| kindyn.py:621-633; forces.py:138-148; label_amass.py:163; bedlam_contacts.py:178; test_kin2_kindyn_frames.py:392,411 | `body.shaped(b).robot`, `.values.body_inertias`, `dataclasses.replace(model.values, gravity=g6)` | `identity.model`, `identity.model.values.body_inertias`, `model.with_values(gravity=)` | keep the gravity override |
| peter_gt/fit_multiview.py:306-309 | `.values.v_shaped`, `.values.body_inertias` | `identity.v_shaped`, `identity.model.values.body_inertias` | rename |
| tests/test_body_density.py:26,38-40 | `.shaped().values.density` | `body.density` (52,) | rename |
| viewer/loading.py:687 | `bh.MHR(path, lod=1, use_expression=False, use_correctives=False, compute_mass=False)` | `bh.load("mhr", path, lod=1, expression=False, correctives=False, mass=False)` | rename |
| viewer/loading.py:722,730,740-763 | `bodies.mhr.body.MHRClassic`; `mhr.from_classic`→(shaped,q); `native_joint_poses(fk(q))`; `values.inverse_bind_poses/rest_joints/v_shaped/skinning_weight_matrix`; `structure.faces` | `families.mhr.{MHRClassic, mhr_from_classic}`→(identity,q,stream); `identity.forward(q,stream).joint_poses`; `body.spec.meta["inverse_bind_poses"]`; `identity.rest_joints/v_shaped`; `body.skin_matrix`; `body.faces` | parity verified |
| human_optimization.py:36; estimate_scale.py:58,81; convert_mhr_to_smplx.py:115-118; peter_gt/overlay_human.py:26; human_optim/stages.py:42 | `<parent>/better_human/src` sys.path hack + model paths | dirs exist in neither layout | delete as dead |
| body.py:38,51; viewer/loading.py:43; label_amass.py:64; import_bedlam2.py:124 | absolute paths into main BetterHuman | dev/BetterHuman/models symlinks to main's models; better_contacts.json tracked in BH, identical today | resolve via installed BH or `BETTERHUMAN_MODELS_DIR` |
Stale: pyproject.toml:77-80 and convert_mhr_to_smplx.py:4-6 say BH SMPLX has no face; dev `smplx(face=True)` streams expression at 5.1e-7 parity with pip smplx (pip smplx could be dropped later).

## 2. tools/smplx_robot
- model.py (30 lines): `load_smplx_mid` just redirects to `load_body`, ignores model_path → delete (callers estimate_scale.py:546, stages.py:267; loading.py:478 own copy).
- dynamics.py (50 lines): central-difference v/a on the manifold with `spacing` and zero-pinned endpoints (used by kindyn.py:252,372, forces.py:151, bedlam_contacts.py:120, test_fps_scaling). BR dev has only `terms.robot.finite_difference` (forward difference, no spacing/pin) → not drop-in; keep and move next to users (e.g. human_optim/motion.py) or ask BR to host it.
- vertex_weights.py (335 lines): only `geodesic_endeffector_weights` + helpers (~130 lines) used (stages.py:30,274); PROFILES, joint_base_weights, lbs_group_labels, group_weights, segmentation_*, normalize, vertex_areas dead → move the used part into human_optim.
- Inertia from BH (bit-identical), RNEA from BR dev `dynamics.inverse_dynamics(model, state, v=, a=, fext=)`.
- BH dev ships BR-optim terms (dev terms.py:23-199): shaped, human_state, landmarks, points, reprojection, ground, contact, pose_prior, shape_prior, contact_wrenches, root_wrench.
- Net: delete the package; ~180 lines move.

## 3. Environment
Today (main): uv-managed CPython 3.12, `uv sync` from uv.lock, torch 2.8.0+cu129 (pinned for pymomentum-gpu 0.1.110); third_party editable via uv path sources (pyproject.toml:97-101, absolute finders into main); BR/BH hand-installed `uv pip install -e ../BetterRobot -e ../BetterHuman --no-deps` (INSTALL.md:32; .pth → main src); detectron2 0.6 outside the lock; sam-3d-body.pth hand-written.
Stale reason: INSTALL.md:41-46 says BR/BH pin cu126; all four now pin pytorch-cu129 like BVR.
BR/BH own venvs: torch 2.13.0+cu129 (BR py3.11, BH py3.12); BH dev sources point better-robot at ../BetterRobot = dev/BetterRobot.
Dev worktree: .venv empty; 5 submodules uninitialised (vendored vggt-omega present); `../data` → dev/data, does not exist.
Torch 2.8 check: BH dev tests 265 passed, 1 skipped; BR dev optim/dynamics/kinematics/lie/data_model 321 passed, 1 failed (`test_matrix_fk_jacrev_equals_jacfwd_on_mixed_kinds`, `.tolist()` under jacfwd at `_fk_lean.py:51`; passes on 2.13). Neither BVR nor BR optim uses torch.func → not blocking.

Option A (recommended): own venv in the worktree, same lock, BR/BH from dev:
```bash
cd /home/rikhat.akizhanov/better/dev/BetterVideoReconstruction
git submodule update --init      # ~250 MB
ln -s /home/rikhat.akizhanov/better/BetterVideoReconstruction/third_party/MHR/assets third_party/MHR/assets  # 212 MB gitignored
uv sync --frozen                 # torch 2.8.0+cu129
uv pip install --python .venv/bin/python -e ../BetterRobot -e ../BetterHuman --no-deps   # ../ = better/dev
CUDA_VISIBLE_DEVICES="" uv pip install --python .venv/bin/python 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps  # sam3d stage only
echo "$PWD/third_party/sam-3d-body" > .venv/lib/python3.12/site-packages/sam-3d-body.pth
export BVR_DATA_ROOT=/home/rikhat.akizhanov/better/data
.venv/bin/python -c "import better_robot,better_human,torch;print(better_robot.__file__,better_human.__file__,torch.__version__)"
```
Option B (no install, quick checks): `PYTHONPATH=dev/BetterRobot/src:dev/BetterHuman/src <main .venv>/bin/python ...` (third_party from main; never `uv run`/`uv pip` there).
Option C (clean end state, unverified): uncomment pyproject.toml:7-8 and :95-96, `uv lock && uv sync` on dev; detectron2 would need a git source + no-build-isolation-package.

## 4. third_party and .pth with a second checkout
- Main venv finders are absolute into main (sam3, sapiens2, MHR, geocalib, vggt-omega, sam-3d-body.pth).
- Repo-relative code breaks on dev until submodules init: tools/adapters/sapiens2.py:41-47 (configs), convert_mhr_to_smplx.py:75-82 (MHR mapping npz), `uv sync` (path sources without pyproject).
- MHR assets: `third_party/MHR/mhr/io.py:32` loads `<pkg>/../assets` (212 MB gitignored download) → symlink.
- Checkpoints are in user-level caches (HF cache, torch hub) → shared.
- The optim rewrite needs no third_party (fuse, scale, human_optim, contacts, kindyn, build_support import none) → testable on main's existing out trees.
- `configs/corpus.toml data_root` repo-relative → breaks from dev; a dev/data symlink would send dev writes into main's data.

## Open questions
1. 35-frame contact JSON: keep in BH studio_data (resolved from installed BH) or copy into BVR configs/?
2. Option A now + C as cleanup target, or C directly?
3. Keep torch 2.8 pin (pymomentum-gpu) while BR/BH dev test on 2.13?
4. Dev runs write separate out trees (e.g. `human_optim_dev/`) so main's results are never overwritten?
