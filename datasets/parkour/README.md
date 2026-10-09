# parkour — LAAS Parkour

The LAAS Parkour dataset (Li et al., CVPR 2019): 28 clips of four actions (kong vault `kv`, muscle-up `mu`, pull-up `pu`, safety
vault `sv`), one subject each, one static 720x576 camera. Truth: Vicon (16 joints) plus floor plates and an instrumented bar — one
6D contact wrench per limb per frame.

- **Where:** `<data root>/Parkour-dataset/` (the `zongmianli/Parkour-dataset` repo: `videos/`, `frames/`, `gt_motion_forces/`; `paths.py: PARKOUR`);
  imported `Parkour-dataset/processed/<clip>/{video.mp4, forces.npz, contacts.npz, joints_3d.npz}`; results `processed/out/<clip>/`.
- **Import:** `uv run --no-sync python datasets/parkour/import_parkour.py [--clips …]` (bakes the 3-frame video/MoCap offset);
  `sensor_forces.py [--source …]` → `<clip>/human_optim/sensor_forces.npz` (the linear part per limb, reaction sense;
  `world_aligned = False`: the MoCap yaw is not ours, so readers compare sizes only).
- **Run:** `uv run --no-sync python scripts/pipeline.py --dataset parkour [--variant gt_contact] [--dry-run]` (`configs/dataset.toml`) with the peter configs
  (`datasets/peter/configs/pipeline_peter.toml`, `human_optimization_peter.toml`).
- **Score:** `score_parkour_pose.py` (the dataset's own per-frame Procrustes MPJPE); `compare_parkour_methods.py [--clips …] [--gt …] [--variant …]`
  scores Li et al. (their / our inputs), PhysPT and our chain on the same frames and truth; `render_parkour_method_overlays.py [--clips --methods]`
  draws one video per clip and method. Baseline inputs: `datasets/methods/{estmf_inputs,physpt_inputs,baseline_farm}.py --dataset parkour`.
