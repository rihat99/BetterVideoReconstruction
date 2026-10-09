# opencap — OpenCap LabValidation

OpenCap LabValidation (Uhlrich et al., PLOS Comp Biol 2023): 10 subjects x 17 trials, five static iPhones (720x1280 portrait,
60 fps), Vicon markers (100 Hz) and two floor force plates (2000 Hz). We import 12 clips: subjects 2 / 3 / 4 x walking1, squats1,
STS1, DJ1 from camera `Cam1`.

- **Where:** raw `<data root>/LabValidation_withVideos/subject*/` (`paths.py: LABVALIDATION`); imported
  `LabValidation_withVideos/processed/<subject>_<trial>_<cam>/{video.mp4, forces.npz, markers.npz, meta.json, sync.json}`; results `processed/out/<clip>/`.
- **Import:** `uv run --no-sync python datasets/opencap/import_opencap.py [--subjects 2,3,4 --trials … --camera Cam1 --skip-existing]`;
  `sync_opencap.py [--clips …] [--joints RUN]` measures each clip's video-to-mocap offset from the video → `sync.json`, then
  `import_opencap.py --resample-only` re-reads the plates at the corrected times; `opencap_sensor_forces.py [--run human_optim]`
  rotates the plates into our world → `<clip>/human_optim/sensor_forces.npz`.
- **Run:** `uv run --no-sync python scripts/run_dataset.py --dataset opencap [--variant floorplane|floorplane_edgev] [--dry-run]` with
  `configs/pipeline_opencap.toml` + `datasets/peter/configs/human_optimization_peter.toml`; `floorplane` (`configs/human_optimization_floorplane.toml`)
  runs on a synthetic support scene — feet + sit-bone frames, no hands, `floor_gate_m = 0`; `_edgev` adds the `kindyn_1` edge knobs.
- **Score:** `export_opencap_mot.py --run <run>` folds our foot forces into the plates' triple (force, centre of pressure, free moment;
  `opencap_plates.py`) → `<run>/estimated_forces.npz` + OpenSim `.mot`; `compare_opencap_plates.py --run <run>` scores it with the published
  GRF protocol; `compare_opencap_forces.py --run <run>` compares per-foot newtons. `opencap_{estmf,physpt,learned,frames35}_export.py` put the baselines in the same contract.
- **View / tests:** `scripts/view_3d.py processed/out/<clip>/` draws the plates (orange, `datasets/_rigs.py`) and the exported estimate
  (blue, `viewer.py`); `tests/test_opencap_forces.py`, `test_support_scene.py`, `test_viewer_plate_forces.py`.
