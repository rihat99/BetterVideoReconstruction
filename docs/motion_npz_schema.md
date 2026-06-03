# Motion `.npz` schema (`results/<stage>.npz`)

The motion stages communicate through a single self-describing `.npz`
layout written under `out/<stem>/results/`:

| File | Producer | Adds over previous |
|------|----------|--------------------|
| `results/0.npz` | `build_robot_motion.py` (Stage 0) | the full base schema below |
| `results/1.npz` | `optimize_motion.py` (Stage 1, kinematic) | refined `q`; `stage_index`, `source_index`, `config_path` |
| `results/2.npz` | `optimize_dynamics.py` (Stage 2, RNEA) | `contact_forces`, `contact_force_joints`, `contact_force_threshold`, `joint_torques` |

Every consumer (`optimize_motion`, `optimize_dynamics`, `view_motion`,
`render_smpl_contact`) rebuilds the BetterRobot model **directly from the
arrays in the file** — there is no hidden dependency on a live SMPL model
or on the SMPL-canonical joint order.

## The DFS-order contract (read this first)

`build_robot_motion.py` performs the SMPL → BetterRobot remap **exactly
once**. Everything stored in the `.npz` is already in BetterRobot's
depth-first (DFS) body order — *not* SMPL-canonical order. Downstream code
consumes the arrays without any permutation.

- `joint_names`, `parents`, `joint_offsets`, `mass_per_body`,
  `com_per_body`, `inertia_per_body` are all length-24, DFS-ordered, and
  index-aligned with each other. Index `i` is BetterRobot **body `i`**
  (0 = pelvis root).
- `parents[i]` is the 0-based index into these arrays of body `i`'s parent
  (`-1` for the pelvis root).
- `joint_contact` (and any per-joint force/torque arrays) use the same DFS
  order, with column `i` corresponding to BR **body `i + 1`** (the contact
  array omits the root, so it is 24 columns starting at body 1). This is
  why the optimizers compute contact columns as
  `model.body_name_to_id[name] - 1`.

To rebuild the model, pass `joint_names` / `parents` / `joint_offsets`
straight to `better_robot.io.builders.build_kinematic_tree_model` with
`root_kind="free_flyer"`, `child_kind="spherical"` (see
`scripts/stages/optimize_motion.py:_rebuild_model` and
`scripts/view_motion.py`).

## Base schema (`results/0.npz`)

`T` = number of frames, `nq` = 99, `nv` = 75 for the SMPL-24 body
(free-flyer root = 7 pos-dims / 6 vel-dims, 23 spherical joints = 4 / 3
dims each).

| Key | Shape | dtype | Meaning |
|-----|-------|-------|---------|
| `q` | `(T, 99)` | f32 | per-frame configuration. `q[:, 0:3]` root translation, `q[:, 3:7]` root quaternion (w, x, y, z), `q[:, 7:99]` = 23 joint quaternions (4 each), DFS order. Z-up world frame. |
| `t` | `(T,)` | f32 | per-frame timestamps (seconds) |
| `fps` | scalar | f32 | frame rate |
| `betas_mean` | `(10,)` | f32 | clip-mean SMPL shape (averaged over valid frames) |
| `joint_names` | `(24,)` | `<U20` | DFS-ordered body names (`body_…`) |
| `parents` | `(24,)` | i32 | 0-based DFS parent index (`-1` for root) |
| `joint_offsets` | `(24, 3)` | f32 | parent→child translation in rest pose (DFS) |
| `mass_per_body` | `(24,)` | f32 | per-body mass (kg), DFS |
| `com_per_body` | `(24, 3)` | f32 | per-body center of mass, DFS |
| `inertia_per_body` | `(24, 3, 3)` | f32 | per-body inertia tensor, DFS |
| `valid_mask` | `(T,)` | bool | frames with a successful body fit (invalid frames are forward-filled in `q`) |
| `num_frames` | scalar | i32 | `== T` |
| `joint_contact` | `(T, 24)` | f32 | per-frame per-joint contact probability (0 when no contacts supplied); column `i` ↔ BR body `i + 1` |
| `joint_contact_threshold` | scalar | f32 | threshold a joint must exceed to count as "in contact" |

## Stage 1 additions (`results/1.npz`)

Same schema; `q` is the kinematically-optimized trajectory. Plus:

| Key | dtype | Meaning |
|-----|-------|---------|
| `stage_index` | i32 | output stage index (1) |
| `source_index` | i32 | input stage index it was produced from (0) |
| `config_path` | str | the `configs/kinematic.toml` used |

## Stage 2 additions (`results/2.npz`)

Everything in `results/1.npz` plus the inferred dynamics. `N` = number of
contact joints (4 by default: left/right foot, left/right hand).

| Key | Shape | dtype | Meaning |
|-----|-------|-------|---------|
| `contact_forces` | `(T, N, 3)` | f32 | world-frame linear force at each contact joint (zeroed on non-contact frames) |
| `contact_force_joints` | `(N,)` | `<U20` | contact joint names, aligned with `contact_forces` axis 1 |
| `contact_force_threshold` | scalar | f32 | `joint_contact` threshold used to gate contacts |
| `joint_torques` | `(T, nv)` | f32 | RNEA joint torques at the final forces (`nv = 75`); `[:, :6]` is the residual base wrench (driven toward zero) |

(`stage_index` / `source_index` / `config_path` are updated to 2 / 1 /
`configs/dynamics.toml`.)
