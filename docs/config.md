# Configuration

Every setting of the pipeline is a key of one file, `configs/bvr.toml`, which holds every default once with a comment
on its meaning and unit. Nothing else holds a default: code reads `config["chamfer"]["sat_m"]`, never
`.get(key, default)`. The command line names only what to process.

## Deltas

A run that needs other values names a delta with `--config`. A delta starts with its base, a path from the
repository root (the defaults, or another delta), and lists only what it changes:

```toml
base = "configs/bvr.toml"

[vggt]
max_frames = 2100           # a long clip: a camera for every frame

[stages.kindyn_1.weights]
neighbour_root = 0          # 0 turns a term off in that stage
neighbour_joint = 0
```

`bvr.config.load` lays the delta over its base table by table and raises `ConfigError`, before any compute, for:

- a key or a table the base does not have (the message names the nearest one);
- a value of another type than the base's (an integer may stand for a float; a boolean is not a number);
- a file without `base`, and a cycle of bases.

A table named `weights` is the one place where a delta may add a key. A term that a stage's weight table does not
name is off in that stage; a name that is no term of the stage raises when its problem is built.

## Which step reads which table

Each step reads only the tables of its row in `bvr.run.STEPS`, and its manifest hashes them: a change to
`[contacts]` reruns the contact steps under `--skip-existing`, not SAM 3.

| tables | read by |
|---|---|
| `[run]` (but `steps`) | every step: `device`, `threads` (torch and OpenCV), `seed`, `deterministic` |
| `[run].steps` | `scripts/pipeline.py`: the steps run without `--steps` |
| `[sam3]`, `[vggt]`, `[sapiens]`, `[sam3d]`, `[smplx]`, `[fuse]`, `[gravity]` | the step of that name |
| `[scale]`, `[scale.weights]` | `scale` ([`scale.md`](scale.md)) |
| `[optimizer]`, `[chamfer]`, `[keypoints]`, `[reprojection_start]`, `[penetration]`, `[contact_attraction]`, `[contact_acquisition]`, `[contact_velocity]`, `[auto_contact]` | `kinematics_1`, `kinematics_2`, `kindyn_1` ([`fitting.md`](fitting.md)) |
| `[stages.root]`, `[stages.kinematics_1]` and their `weights` | `kinematics_1` |
| `[stages.kinematics_2]` and its `weights` | `kinematics_2` |
| `[stages.kindyn_1]` and its `weights`, `[torque_multipliers]` | `kindyn_1` ([`forces.md`](forces.md)) |
| `[contact_gate]` | the three fitting steps and the two contact steps |
| `[friction]` | `contacts_1`, `contacts_2`, `kindyn_1` |
| `[contacts]`, `[contact_force]`, `[contact_confidence]`, `[quality]` | `contacts_1`, `contacts_2` ([`contacts.md`](contacts.md)) |
| `[viewer]` | `scripts/view_3d.py` ([`viewer.md`](viewer.md)) |

Weights of difference terms are per change over 1/30 s ([`frame_rate.md`](frame_rate.md)); the force and torque
terms of `kindyn_1` are in body weights squared. `[contact_confidence]` is fitted, not tuned: its tables are written by
`bvr_datasets/climbingvideos/calibrate_contact_confidence.py --write`.

## Deltas in the repository

| file | for |
|---|---|
| `bvr_datasets/peter/configs/peter.toml` | the static-camera force rigs (peter, Parkour, OpenCap): one person, overlays off, no confidence cut in fuse and scale, no pre-fit, kinematic chamfer off |
| `bvr_datasets/opencap/configs/opencap.toml` | OpenCap over `peter.toml`: up to 900 VGGT frames |

The other TOML files under `bvr_datasets/` (`bedlam2/configs/bedlam.toml`, `climbingvideos/configs/corpus.toml`) are
settings of those tools, not deltas.

## Settings worth knowing

- `[run].deterministic = true`: deterministic torch algorithms; two runs from the video are then identical in every
  output, at 15 to 20 % more time. Without it the human steps repeat to a mean body-joint distance of 0.1 to 2.3 mm
  (`kinematic_1` to `kindyn_1`) on a single-person clip, and only to 9 to 16 mm in `kindyn_1` on the three-person
  `competition_2` (p95 near 30 mm).
- `[vggt].max_frames` (500): the longest clip the `vggt` step takes; a longer one stops it, so raise the key or
  trim the clip. VGGT sees every frame in one forward, which holds about 1150 frames on a 96 GB card; the wild
  clips were run with 2100 (`docs/pipeline.md`).
- `[sam3].top_k` and `min_area_frac` decide who is tracked: every person grounded on frame 0 by default. In-the-wild
  clips were run on the largest person only (`top_k = 1`, `min_area_frac = 0.001`, and `[sam3d].overlay = false`).
