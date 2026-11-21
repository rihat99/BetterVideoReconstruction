# The pipeline

A run takes videos and writes one tree per clip, `<save-dir>/<stem>/`. It is a chain of 13 steps that meet only
through the files of that tree. The table `bvr.run.STEPS` is the one place that says, for each step, its module, the
config tables it reads, and the files it reads and writes.

## The steps

| step | module (`bvr/stages/`) | what it does | writes |
|---|---|---|---|
| `sam3` | `track_sam3` | finds the people on frame 0 from a text prompt (or a box, mask or points) and tracks their masks | `sam3/` |
| `vggt` | `estimate_camera_vggt` | one forward of VGGT-Omega over the clip: a camera and a depth map per frame, right up to one unknown scale | `vggt/` |
| `sapiens` | `estimate_sapiens` | Sapiens2 keypoints in every person's box | `sapiens/pose.npz` |
| `sam3d` | `estimate_body_sam3d` | SAM 3D Body: an MHR body per person and frame, with VGGT's intrinsics | `sam3d/params.npz` |
| `smplx` | `convert_mhr_to_smplx` | fits SMPL-X to each MHR body, frame by frame | `sam3d/smplx_params.npz` |
| `fuse` | `fuse_geometry` | the depth maps as one static scene cloud and per-frame human points, still up to scale | `geometry/fused/` |
| `scale` | `estimate_scale` | reads the metric scale off the bodies, seats the bodies on their depth, writes the metric world | `geometry/*.npz` |
| `gravity` | `estimate_gravity` | the direction of gravity, from GeoCalib or the floor | `geocalib/gravity.npz` |
| `kinematics_1` | `refine_kinematics` | fits every body trajectory to the keypoints, the depth points and the scene | `human_optim/kinematic_1.npz` |
| `contacts_1` | `estimate_contacts` | which contact frames touch the scene, and how far to trust each label | `human_optim/contacts_1.npz` |
| `kinematics_2` | `refine_kinematics` | pulls each labelled contact onto the wall the camera sees there | `human_optim/kinematic_2.npz` |
| `contacts_2` | `estimate_contacts` | the contact readout again, on the seated bodies | `human_optim/contacts_2.npz` |
| `kindyn_1` | `solve_dynamics` | pose and contact forces solved together by inverse dynamics | `human_optim/kindyn_1.npz` |

The default chain, `[run].steps`, ends at `kinematics_1`; `--steps` runs any of the 13 in this order. The steps
`sam3`, `vggt`, `sapiens`, `sam3d` and `gravity` each import their own third-party model, and only they do. `fuse` and `scale` are in
[`fusion.md`](fusion.md) and [`scale.md`](scale.md), the two kinematic steps in [`fitting.md`](fitting.md), the
contact steps in [`contacts.md`](contacts.md), `kindyn_1` in [`forces.md`](forces.md). The README lists the files;
the docstring of each stage module lists every array it writes.

Every step is multi-person: from `sapiens` on, arrays are `(P, N, ...)` with `object_ids (P,)`, the SAM 3 objects,
and `valid_mask (P, N)`. The body is BetterHuman's neutral SMPL-X with 10 shape coefficients
and 35 contact frames (`bvr.body`): named points of the skin whose z axis is the outward normal there. A person
without a tracked frame or without keypoints is written to the body files as they came, not fitted.

## How a run works

- One process per step over all its clips (`bvr.run.chain`), so a model or the body loads once per step and its
  memory is gone before the next. The process applies `[run]`: device, torch and OpenCV thread caps, the seed, and
  with `[run].deterministic = true` deterministic torch algorithms (bit-identical reruns, 15 to 20 % slower).
- A step writes its files whole or not at all: an npz goes to a temporary name and is renamed, a folder is written as
  `<name>.staging` and renamed (`bvr.files`).
- The manifest `<clip>/manifests/<step>.json` is written last. It records the status (`complete` or `failed`, with
  the error), the config tables the step read, the commit and a checksum of any uncommitted diff, the environment
  (python, torch, CUDA, the lock, the commits of BetterRobot, BetterHuman and the submodules), the device, the seed,
  every input and output with a checksum, the seconds and the peak GPU memory. The step's debug log is
  `<clip>/logs/<step>.log`.
- A clip that fails is data: its manifest says `failed`, the other clips go on, the later steps skip it, and the
  exit code is 1.
- A rerun first moves what it replaces (the step's outputs and manifest) to `<save-dir>/_trash/<clip>/<step>_<time>/`.
  Nothing is deleted; empty that folder yourself.

## Resume, upstream, shards

- `--skip-existing` skips a step for a clip while its manifest says `complete` and the config tables it read, the
  code (commit and uncommitted diff) and the checksums of its inputs are unchanged; the log says which of them
  changed otherwise. So a new commit reruns every step. A tree written before the rebuild has no manifests: its
  files are valid inputs, never completed steps.
- `--upstream <root>` makes a step read an input it does not find in `<save-dir>/<clip>/` from `<upstream>/<clip>/`
  (`bvr.paths.Clip.find`); outputs always go to `--save-dir`, and the manifest records where each input came from.
  This is how a second run of some steps sits beside the first without copying it:

  ```bash
  uv run python scripts/pipeline.py --source <videos> --save-dir <new root> --upstream <old root> \
      --steps kinematics_1,contacts_1,kinematics_2,contacts_2,kindyn_1 --config <delta.toml>
  ```

- `--shard i/n` takes every n-th clip from the i-th, so n runs on n GPUs fill one root.
- A step alone is `python -m bvr.stages.<module> --source <videos> --save-dir <root>` with the same `--config`,
  `--upstream` and `--skip-existing`, `--clips a,b` for some clips, and `--step` for the two modules that serve two
  steps (`refine_kinematics`, `estimate_contacts`).

## What a step costs

Measured on one clip of 104 frames and one person, the 13 steps from the video, on a GPU another user shared: wall
time inside each step without the model load (from the manifests), and torch's peak allocation.

| step | s | GPU | step | s | GPU |
|---|---|---|---|---|---|
| `sam3` | 9.8 | 8.5 GB | `gravity` | 5.1 | |
| `vggt` | 10.6 | 12.2 GB | `kinematics_1` | 20.1 | 1.0 GB |
| `sapiens` | 49.9 | 10.3 GB | `contacts_1` | 3.2 | |
| `sam3d` | 94.5 | 3.5 GB | `kinematics_2` | 13.2 | |
| `smplx` | 10.6 | | `contacts_2` | 3.6 | |
| `fuse` | 2.8 | | `kindyn_1` | 8.0 | |
| `scale` | 8.9 | | whole job | 387 | |

VGGT's memory grows with the frame count: it runs one forward over the clip, and `[vggt].mode = "balanced"` keeps
about 1024 tokens a frame at any aspect ratio. Torch's peak allocation is 4.3 GB plus 0.076 GB per frame:

| frames | 104 | 250 | 471 | 500 | 750 | 1000 |
|---|---|---|---|---|---|---|
| peak allocation | 12.2 GB | 22.9 GB | 39.3 GB | 42.2 GB | 61.1 GB | 80.1 GB |
| forward | 5 s | 20 s | 62 s | 72 s | 163 s | 280 s |

A 96 GB card holds about 1150 frames, a 48 GB card about 560. `nvidia-smi` shows more while the card has room (62 GB
at 471 frames): blocks the allocator keeps cached and gives back under pressure.
