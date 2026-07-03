# legacy/

Archived code from the BetterVideoReconstruction cleanup. These components were
tested and then dropped from the pipeline; they are kept for reference and are
**not maintained** — their internal paths/imports may no longer resolve as-is.
Nothing under `legacy/` is imported by the active pipeline.

## What's here and why it was dropped

- **MoGe2 metric depth** — `scripts/stages/estimate_camera_moge2.py`,
  `tools/adapters/moge2.py`, `tools/geometry/metric_anchor.py`,
  `third_party/moge/`. Replaced by the human-body-anchored `estimate_scale` stage:
  VGGT's up-to-scale geometry is metricized with the SMPL-X body as a ruler, not
  MoGe2 depth. The camera-intrinsics helpers MoGe2 provided
  (`denormalize_K` / `normalize_K`) now live in `tools/geometry/intrinsics.py`.

- **MAMMA `ma_2d` dense landmarks** — `scripts/stages/estimate_ma_2d.py`,
  `tools/adapters/mamma.py`, `third_party/mamma/`. Was a weight-0 alternative
  reprojection target in `optimize_smplx`; the init-projection anchor + Sapiens
  Goliath-308 keypoints are the reprojection signal now.

- **SMPL (not SMPL-X) motion / dynamics path** —
  `scripts/stages/{convert_mhr_to_smpl,build_robot_motion,optimize_motion,optimize_dynamics}.py`,
  `scripts/{view_motion,render_smpl_contact}.py`, `tools/robot_motion/`,
  `tools/mhr_to_smpl/`, `configs/{kinematic,dynamics}.toml`,
  `docs/motion_npz_schema.md`. The old simplistic-loss motion optimizer + viewer.
  The active dynamics path is the SMPL-X force-only RNEA stage
  (`optimize_dynamics_smplx`); its one reused helper,
  `velocity_acceleration_from_trajectory`, now lives in
  `tools/smplx_robot/dynamics.py`.

- **Gemma VLM triage** — `scripts/stages/classify_gemma.py`,
  `tools/adapters/gemma_vlm.py`. A never-implemented placeholder stage.
