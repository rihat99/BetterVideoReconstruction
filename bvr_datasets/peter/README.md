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
  A stored run keeps its measured run inside each clip, `<clip>/_gt_contact/`, so `--measured` takes a folder of
  links, one per clip: `mkdir -p <links> && for c in <data root>/peter/out_climb_wall_3_single/[!_]*/; do ln -s "${c}_gt_contact" "<links>/$(basename $c)"; done`,
  then `--estimated <data root>/peter/out_climb_wall_3_single --measured <links>`.
- **Ground truth (`gt/`, climb_wall_3):** three calibrated cameras place a body that one camera cannot. A trial has a
  prepared tree, `peter/climb_wall_3_gt/human/<trial>/`: `cameras.npz` (the calibration, the cameras' frame offsets and
  how well each one's keypoints agree), each camera's model outputs under `views/<camera>/`, the wall's cloud and plane in
  `geometry/`, gravity and `sensor_forces.npz`. `gt/fit_multiview.py` runs the human steps on it with the image terms of all
  three cameras, the boards' contact labels and the wall's plane, and pulls each limb's solved force to its board; it reads
  the trees as the upstream and writes `<run>/<trial>/human_optim/`. `gt/shape.py` first fits one shape per climber to
  their measured height and mass, from the trials given (three of each, one per wall angle); a run without shapes of its own
  reads the upstream's `_shapes/`.
  ```bash
  T=<data root>/peter/climb_wall_3_gt/human
  mkdir -p <videos> && for c in $T/[ab]_*/; do ln -s "${c}views/cam_left/video.mp4" "<videos>/$(basename $c).mp4"; done
  uv run python -m bvr_datasets.peter.gt.shape --source <videos> --save-dir <run> --upstream $T --config bvr_datasets/peter/gt/gt.toml \
      --clips a_00_01_part3,a_16_01_part2,a_32_01_part3,b_00_02_part2,b_16_01_part3,b_32_02_part2
  uv run python -m bvr_datasets.peter.gt.fit_multiview --source <videos> --save-dir <run> --upstream $T --config bvr_datasets/peter/gt/gt.toml
  ```
  The scene and the trees come from the other scripts of `gt/`, run in this order on one root (`<root>/scene/<set>/`,
  `<root>/human/<trial>/`; the stored one is `<data root>/peter/climb_wall_3_gt`). A set is one placement of the
  cameras: `00`, `16`, `16b`, `16c`, `32` (`cam_3` was bumped twice at 16 degrees).
  ```bash
  G=<root, an absolute path>; P=bvr_datasets/peter/gt
  uv run python $P/extract_frames.py --save-dir $G            # a people-free frame per camera and set (CPU, 7 s)
  uv run python $P/calibrate_cameras.py --save-dir $G         # chessboard intrinsics and poses of sets 00, 16, 32 (CPU)
  uv run python $P/reconstruct_vggt.py --save-dir $G          # VGGT on the three stills: scene.npz in VGGT's cameras
  uv run python $P/clean_scene.py --save-dir $G               # scene_clean.npz (CPU, 35 s)
  uv run python $P/rebuild_scene_calibrated.py --save-dir $G  # scene_calib.npz: VGGT's depth in the calibrated cameras
  uv run python $P/prepare_human.py --save-dir $G --clips <trials>   # the trees; --scene-only refreshes geometry/
  for t in <trials>; do         # the model steps on each camera of a trial (GPU, about 40 min a trial of 475 frames)
    mkdir -p $G/videos/$t && for c in cam_left cam_right cam_3; do ln -s $G/human/$t/views/$c/video.mp4 $G/videos/$t/$c.mp4; done
    uv run python scripts/pipeline.py --source $G/videos/$t --save-dir $G/human/$t/views \
        --config bvr_datasets/peter/configs/peter.toml --steps sam3,sapiens,sam3d,smplx
  done
  uv run python $P/prepare_human.py --save-dir $G --clips <trials> --cam3   # cam_3's pose from the climber (CPU)
  uv run python $P/view_gt.py --save-dir $G                   # the clouds and both cameras of each set, in a browser
  ```
  `scene_calib.npz` is the wall every tree carries (`geometry/scene.npz`, its plane `geometry/wall.npz`);
  `scene.npz` gives the pose of `cam_3` in sets `16b` and `16c`, which have no calibration of their own, and with
  `scene_clean.npz` the checks of `rebuild_scene_calibrated.py`. `cameras.npz` holds each camera's frame lag behind
  `cam_left` (`frame_offset`, the chessboard's, from the set's `calibration.npz`). `--cam3` completes it for the fit:
  it triangulates the climber's Sapiens keypoints with the front pair, re-solves the pose of `cam_3` from them (kept
  when it lowers `cam_3`'s reprojection error; the pose before stays in `extrinsics_before_pnp`) and writes each
  camera's keypoint spread, `view_sigma_px`, which weighs the cameras in the fit.
