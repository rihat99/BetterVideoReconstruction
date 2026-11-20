# examples — the annotated example clips

The short climbing clips the pipeline is demonstrated and regression-tested on, each with a manual contact
annotation on coarse joints (a 10-joint or a 14-joint schema; per frame 1 contact, 0 free, -1 never annotated).

- **Where:** `<data root>/examples/` (`../data` or `$BVR_DATA_ROOT`; `bvr.paths.EXAMPLES`). `data/`, `data_2/` hold `<clip>.mp4`
  plus the annotation `<clip>.npz` (`joint_names`, `contacts (P, J, N)`, time axis last).
- **Run:** `uv run python scripts/pipeline.py --source <data root>/examples/data --save-dir <run root>
  --steps kinematics_1,contacts_1,kinematics_2,contacts_2,kindyn_1 [--upstream <a run root with the earlier steps>]`.
- **Score:** `uv run python bvr_datasets/examples/score_frame_contacts.py --predictions <run root> --contacts 1|2 [--json <file>]`
  folds `<clip>/human_optim/contacts_<tag>.npz` onto the annotated joints (hand ← palm | finger, foot ← toe/ball frames,
  ankle ← heel, knee ← knee) and reports per-joint precision / recall / F1 against `examples/data/<clip>.npz`.
- **View:** `uv run python scripts/view_3d.py --source <data root>/examples/data --save-dir <run root>`.
