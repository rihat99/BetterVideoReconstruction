# parkour — LAAS Parkour

The LAAS Parkour dataset (Li et al., CVPR 2019): 28 clips of four actions (kong vault `kv`, muscle-up `mu`, pull-up `pu`, safety
vault `sv`), one subject each, one static 720x576 camera. Truth: Vicon (16 joints) plus floor plates and an instrumented bar — one
6D contact wrench per limb per frame.

- **Where:** `<data root>/Parkour-dataset/` (the `zongmianli/Parkour-dataset` repo: `videos/`, `frames/`, `gt_motion_forces/`; `paths.py: PARKOUR`);
  imported `Parkour-dataset/processed/<clip>/{video.mp4, forces.npz, contacts.npz, joints_3d.npz}`; the stored runs in `processed/out/<clip>/`.
- **Import:** `uv run python bvr_datasets/parkour/import_parkour.py [--clips …]` (bakes the 3-frame video/MoCap offset).
- **Videos:** a run reads a folder of links `<clip>.mp4`:
  `mkdir -p <videos> && for c in <data root>/Parkour-dataset/processed/*/; do [ -f "${c}video.mp4" ] && ln -s "${c}video.mp4" "<videos>/$(basename $c).mp4"; done`
- **Run:** the force-rig recipe of `bvr_datasets/peter/` (`configs/peter.toml`, then `python -m bvr_datasets.peter.chain`, see its README);
  `sensor_forces.py --save-dir <run>` → `<run>/<clip>/human_optim/sensor_forces.npz` (the linear part per limb, reaction sense;
  `world_aligned = False`: the MoCap yaw is not ours, so readers compare sizes only); `chain --measured` for the boards' labels.
- **Score:** `score_parkour_pose.py --save-dir <run> [--measured <run 2>]` (the dataset's own per-frame Procrustes MPJPE);
  `bvr_datasets/peter/compare_gt_contact.py --estimated <run> --measured <run 2>` (forces against the plates and the bar).
