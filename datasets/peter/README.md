# peter — the climbing force rigs

The "peter" demonstrator rigs, where contact is **measured**, not estimated: `campus_board` (a single-axis plate per limb),
`climb_wall_2` and `climb_wall_3` (an inclined wall with a 3-axis force board under each limb's hold, ~99 Hz; climb_wall_3 is
the 2026-09 session: two climbers, wall at 0 / 16 / 32 deg), two or three synchronized static cameras. See `board_contacts.md`.

- **Where:** `<data root>/peter/<rig>/<clip>/{cam_left,cam_right,cam_3}.mp4` + `forces.npz` (`paths.py: PETER`); results in
  `peter/out_<rig>_single/<clip>/` (cam_left) and `out_<rig>_multi/` (`--views`); the climb_wall_3 ground truth in `peter/climb_wall_3_gt/`.
- **Run:** `uv run --no-sync python scripts/pipeline.py --dataset climb_wall_3|climb_wall_2|campus_board [--views a,b] [--dry-run]`
  (`configs/dataset.toml`; `configs/pipeline_peter.toml` + `configs/human_optimization_peter.toml`, deltas over the general configs: one tracked person, overlays off,
  hand + foot contact frames only, kinematic chamfer off — Parkour and OpenCap use these too).
- **Measured contacts:** `peter_sensor_forces.py --tree out_climb_wall_3_single|out_climb_wall_2_single` rotates the board logs into our
  world (datasheet rotation, no fit) → `<clip>/human_optim/sensor_forces.npz`; `scripts/pipeline.py --dataset climb_wall_3 --variant gt_contact`
  then re-runs contacts_1 → kindyn_1 on the boards' labels (`bvr.stages.measure_contacts_boards`) into `<clip>/human_optim_gt_contact/`.
- **Score:** `compare_gt_contact.py [--tree …]` (estimated vs measured labels and forces); `render_kindyn_overlays.py [<rig> <clip>]`
  (contact discs + force plots on the video); `compare_climb_wall_3_runs.py` → `peter/benchmark_tables/`; `render_cw3_estmf_overlays.py`.
- **Ground truth (`gt/`):** `run_human.py --clips … --steps prepare,…,overlay --gpu N` (or `run_all_clips.sh <gpu …>`) is the three-camera
  chain — chessboard calibration (`calibrate_cameras.py`), the empty-wall cloud (`extract_frames.py`, `reconstruct_vggt.py`, `clean_scene.py`,
  `rebuild_scene_calibrated.py`, `view_gt.py`), the multi-view fit (`prepare_human.py`, `fit_multiview.py`, `overlay_human.py`); configs
  `gt/{pipeline_gt,vggt_gt,human_gt}.toml`. `run_baselines.sh <model> <gpu>` runs a published model on the `datasets/methods/baseline_farm.py` farm.
