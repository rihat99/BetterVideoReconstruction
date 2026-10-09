# methods — the published baselines

The methods we compare against and the glue that runs them on our data: Li et al. CVPR 2019 "Estimating 3D Motion and Forces"
(`estmf`, with its contact recogniser), PhysPT (CVPR 2024), and the contact / pose models DECO, BSTRO, GVHMR, PromptHMR, InteractVLM.
Their inputs are built from OUR pipeline so every method starts from the same body; their outputs are scored on our two benchmarks.

- **Where:** the estimator at `<data root>/Estimating-3D-Motion-Forces/` (`paths.py: ESTMF`); the other models under
  `/home/rikhat.akizhanov/better/other/<model>/`, each in its own conda env; predictions land in the dataset trees
  (`peter/climb_wall_3_gt/others/<model>/`, `…/out/<clip>/method_*`).
- **Inputs:** `baseline_farm.py --dataset parkour|climb_wall_3 [--clips …]` lays a dataset out like a ClimbingVideos corpus so the runners
  (`datasets/peter/gt/run_baselines.sh`) read it unchanged; `estmf_inputs.py --dataset parkour|opencap|climb_wall_3` (SAM-3D-Body init on the
  true camera, Sapiens 2D joints, measured contact); `physpt_inputs.py --dataset …` (per-frame 6D rotations + fitted SMPL betas);
  `contact_recognizer.py --dataset Parkour-dataset|climb_wall_3|OpenCap` (Li et al.'s recogniser in one pass).
- **Score:** `score_published_contacts.py --benchmark climbingvideos|climb_wall_3` (108 annotated scenes / the four force boards);
  `score_published_pose.py --benchmark climbingvideos|climb_wall_3` (reference: our `kindyn_1` body / the calibrated three-camera fit);
  `score_init_world.py` (world-frame metrics of the SAM-3D-Body init); `score_combined_table.py` → `peter/benchmark_tables/`.
  The baselines' OpenCap exporters live in `datasets/opencap/`, the Parkour comparison in `datasets/parkour/`.
- **Shared:** `datasets/_common.py` (PNG tables, confusion rates, the limb fold, overlay drawing).
