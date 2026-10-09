# examples — the annotated example clips

The short climbing clips the local pipeline is demonstrated and regression-tested on, each with a manual contact
annotation on coarse joints (a legacy 10-joint or the 14-joint schema; per frame 1 contact, 0 free, -1 never annotated).

- **Where:** `<data root>/examples/` (`../data` or `$BVR_DATA_ROOT`; `bvr.paths.EXAMPLES`). `data/`, `data_2/` hold `<clip>.mp4`
  plus the annotation `<clip>.npz` (`joint_names`, `contacts (P, J, N)` — time axis last); results go to `out/<clip>/`, `out_2/<clip>/`.
- **Run:** `uv run --no-sync python scripts/pipeline.py --source ../data/examples/data/ --save-dir ../data/examples/out/`
  (`configs/pipeline.toml` + `configs/human_optimization.toml`), then the standalone SMPL-X stages
  (`python -m bvr.stages.estimate_contacts`, `human_optimization --stages kinematics_2|kindyn_1`) as in the main README.
- **Score:** `uv run --no-sync python datasets/examples/score_frame_contacts.py --predictions ../data/examples/out/ --contacts 1|2 [--json f]`
  folds `out/<clip>/human_optim/contacts_<tag>.npz` onto the annotated joints (hand ← palm | finger, foot ← toe/ball frames,
  ankle ← heel, knee ← knee) and reports per-joint precision / recall / F1 against `examples/data/<clip>.npz`.
- **Files:** `annotations.py` (the joint schemas `ANN_TO_SMPLX` / `ANN_TO_SMPLX_14`, `ann_mapping`),
  `score_frame_contacts.py`. View a result: `scripts/view_3d.py ../data/examples/out/<clip>/`.
