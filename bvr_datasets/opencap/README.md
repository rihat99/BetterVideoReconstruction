# opencap — OpenCap LabValidation

OpenCap LabValidation (Uhlrich et al., PLOS Comp Biol 2023): 10 subjects x 17 trials, five static iPhones (720x1280 portrait,
60 fps), Vicon markers (100 Hz) and two floor force plates (2000 Hz). We import camera `Cam1`.

- **Where:** raw `<data root>/LabValidation_withVideos/subject*/` (`paths.py: LABVALIDATION`); imported
  `LabValidation_withVideos/processed/<clip>/{video.mp4, forces.npz, markers.npz, meta.json, sync.json}`; the stored runs in
  `processed/out/<clip>/` (`human_optim/` on 12 clips; `human_optim_floorplane/` on 161 is of a recipe this code no longer has).
- **Import:** `uv run python bvr_datasets/opencap/import_opencap.py [--subjects 2,3,4 --trials … --camera Cam1 --skip-existing]`;
  `sync_opencap.py --save-dir <run> [--clips …] [--joints]` measures each clip's video-to-mocap offset from the run's keypoints
  (or its kindyn joints) → `sync.json`, then `import_opencap.py --resample-only` re-reads the plates at the corrected times.
- **Videos:** a run reads a folder of links `<clip>.mp4`:
  `mkdir -p <videos> && for c in <data root>/LabValidation_withVideos/processed/*/; do [ -f "${c}video.mp4" ] && ln -s "${c}video.mp4" "<videos>/$(basename $c).mp4"; done`
- **Run:** the force-rig recipe of `bvr_datasets/peter/` with `configs/opencap.toml` (the peter delta, no frame subsampling,
  leaner VGGT chunks): `scripts/pipeline.py`, then `python -m bvr_datasets.peter.chain` (see its README);
  `sensor_forces.py --save-dir <run>` rotates the plates into our world → `<run>/<clip>/human_optim/sensor_forces.npz`.
- **Score:** `export_opencap_mot.py --save-dir <run>` folds our foot forces into the plates' triple (force, centre of pressure,
  free moment; `plates.py`) → `human_optim/estimated_forces.npz` + OpenSim `.mot`; `compare_opencap_plates.py --save-dir <run>
  [--clips <list>]` scores it with the published GRF protocol (→ `<run>/_plates/summary.md`); `compare_opencap_forces.py
  --save-dir <run>` compares per-foot newtons (→ `<run>/_compare/`).
- **Tests:** `tests/test_opencap_forces.py`.
