# peter — the climbing force rigs

The "peter" demonstrator rigs, where contact is **measured**, not estimated: `campus_board` (a single-axis plate per limb),
`climb_wall_2` and `climb_wall_3` (an inclined wall with a 3-axis force board under each limb's hold, ~99 Hz; climb_wall_3 is
the 2026-09 session: two climbers, wall at 0 / 16 / 32 deg), two or three synchronized static cameras. See `board_contacts.md`.

- **Where:** `<data root>/peter/<rig>/<clip>/{cam_left,cam_right,cam_3}.mp4` + `forces.npz` (`paths.py: PETER`); the stored
  single-view runs in `peter/out_<rig>_single/<clip>/`, the climb_wall_3 ground truth in `peter/climb_wall_3_gt/`.
- **Videos:** the pipeline names a clip by its video, so a run reads a folder of links `<clip>.mp4`:
  `mkdir -p <videos> && for c in <data root>/peter/climb_wall_3/*/; do ln -s "${c}cam_left.mp4" "<videos>/$(basename $c).mp4"; done`
- **Run** (`configs/peter.toml`, also used by Parkour and OpenCap: one tracked person, overlays off, no pre-fit,
  kinematic chamfer off; `chain.py` runs the human steps on a body with contact frames on the hands and feet only):
  ```bash
  uv run python scripts/pipeline.py --source <videos> --save-dir <run> --config bvr_datasets/peter/configs/peter.toml
  uv run python -m bvr_datasets.peter.chain --source <videos> --save-dir <run> --config bvr_datasets/peter/configs/peter.toml
  ```
- **Measured contacts:** `sensor_forces.py --rig climb_wall_3 --save-dir <run>` rotates the board logs into our world
  (datasheet rotation, no fit) → `<run>/<clip>/human_optim/sensor_forces.npz`; then
  `python -m bvr_datasets.peter.chain --measured --source <videos> --save-dir <run 2> --upstream <run> --config …` re-runs
  contacts_1 → kindyn_1 on the boards' labels into a second root.
- **Score:** `compare_gt_contact.py --estimated <run> --measured <run 2>` (estimated vs measured labels and forces).
