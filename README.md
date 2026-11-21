# BetterVideoReconstruction

BetterVideoReconstruction (BVR) turns one monocular video into, for every person it tracks:

- an SMPL-X body trajectory in a metric world frame,
- on every frame, which of 35 named points of the body's skin touch the scene,
- the contact forces at those points and the joint torques that make the motion physically consistent.

Five models give the evidence: SAM 3 (person masks), VGGT-Omega (cameras and depth), Sapiens2 (2D keypoints),
SAM 3D Body (a body per person and frame) and GeoCalib (gravity). BVR's own steps convert the body to SMPL-X, fuse
the depth into point clouds, size the world from the body, fit the motion to the images and the depth, read the
contacts, and solve pose and forces together by inverse dynamics. The fits run on
[BetterRobot](https://github.com/rihat99/BetterRobot) (kinematics, inverse dynamics, the optimiser) with
[BetterHuman](https://github.com/rihat99/better_human) bodies.

```text
video -> sam3 -> vggt -> sapiens -> sam3d -> smplx -> fuse -> scale -> gravity
      -> kinematics_1 -> contacts_1 -> kinematics_2 -> contacts_2 -> kindyn_1
```

## Install

[`INSTALL.md`](INSTALL.md) has the details. In short: clone with submodules, put BetterRobot and BetterHuman beside
the repository, `uv sync --managed-python`, add one `.pth` line for SAM 3D Body, point `BETTERHUMAN_MODELS_DIR` at
the SMPL-X and MHR body models, and accept the licences of the gated model repositories on Hugging Face. A CUDA GPU
is needed for the model steps.

## Run a clip

Data lives outside the repository under `$BVR_DATA_ROOT` (default: the folder `data` beside the repository). The
dataset tools read it and stop with an error if neither exists; a run names its own `--source` and `--save-dir`.

```bash
uv run python scripts/pipeline.py --source $BVR_DATA_ROOT/examples/data/boulder_1.mp4 --save-dir ../runs
uv run python scripts/pipeline.py --source $BVR_DATA_ROOT/examples/data/boulder_1.mp4 --save-dir ../runs \
    --steps gravity,contacts_1,kinematics_2,contacts_2,kindyn_1
```

The first command runs the default chain, `sam3` to `kinematics_1` (`[run].steps` of `configs/bvr.toml`); the second
adds gravity, the contacts and the forces. `--source` may be a folder: every video in it is a clip. Each clip gets
its own tree, `<save-dir>/<stem>/`. All 13 steps on a 104-frame clip took 6.5 minutes on one shared GPU, most of it
in SAM 3D Body and Sapiens.

The command line names what to process; every setting is a key of `configs/bvr.toml`, and `--config <delta.toml>`
names a file that changes some of them (`docs/config.md`). Other options: `--steps` picks any steps in chain order,
`--upstream <root>` reads what this run lacks from another run root, `--skip-existing` skips a step whose manifest
still holds, `--shard i/n` splits the clips over several runs, `--dry-run` prints the commands. Each step is also a
module of its own, `python -m bvr.stages.<module>`, with `--source`, `--save-dir`, `--config`, `--upstream` and
`--skip-existing`. `docs/pipeline.md` explains the steps
and the run.

## The output tree

Frame `k` of every file is the `k`-th decoded frame of the video. People are stacked first, frames second,
`(P, N, ...)`, with `object_ids (P,)` and `valid_mask (P, N)` (the frames a person is tracked on). Lengths are in
metres, forces in newtons, rotations as quaternions `[qx, qy, qz, qw]`. Every npz names its `format` and
`format_version`, and the docstring of the stage module that writes it lists every array.

```text
<save-dir>/<clip>/
  sam3/bboxes.npz           bboxes_per_obj (P, N, 4) in pixels, object_ids, labels, the video size
  sam3/<id>/frame_*.png     each person's mask per frame; mask_overlay.mp4 beside them
  vggt/camera.npz           cameras of the sampled frames (frame_indices): extrinsics, intrinsics, up to scale
  vggt/depth/frame_*.npz    depth and its confidence per sampled frame
  sapiens/pose.npz          keypoints (P, N, 308, 2) in pixels (Goliath-308), keypoint_scores; overlay/pose.mp4
  sam3d/params.npz          the MHR body per person and frame: mhr_model_params (204), shape_params (45), pred_cam_t
  sam3d/overlay.mp4         the fitted bodies over the video
  sam3d/smplx_params.npz    SMPL-X fitted to each MHR body: betas (10), poses, transl in the camera frame, fit_error
  geometry/fused/           the point clouds up to scale (transform.npz, scene.npz, human.npz)
  geometry/transform.npz    metric cameras: extrinsics (T, 4, 4) camera from world, intrinsics_px_orig, fps
  geometry/scene.npz        the static scene: points (M, 3), colors, confidence, normals
  geometry/human.npz        each person's depth points per frame: points, frame_id, obj_id
  geometry/scale.npz        the metric factor, its per-frame readings, smear_median
  geometry/prefit.npz       the bodies the scale step seated on their depth (SMPL-X, camera frame)
  geocalib/gravity.npz      gravity_world (3,), source (geocalib, ground or fallback_down), the floor plane
  human_optim/kinematic_1.npz   body file: q (P, N, 211) in the world, the classic SMPL-X fields, drift_pre
  human_optim/contacts_1.npz    frame_contact (P, N, 35), each witness's labels, label confidence, quality_flag
  human_optim/kinematic_2.npz   body file: the bodies seated on the contacts of contacts_1
  human_optim/contacts_2.npz    as contacts_1, read on kinematic_2
  human_optim/kindyn_1.npz      body file with frame_forces (P, N, 35, 3), joint_torques, base_wrench,
                                total_mass, gravity_world, force_confidence (P, N)
  manifests/<step>.json     written last: status, config, code, environment, inputs and outputs with checksums
  logs/<step>.log           the step's debug log
<save-dir>/_trash/          what a rerun replaced, per clip and step; nothing is deleted
<save-dir>/_runs/           the uncommitted diff of the code that ran, once per code state
```

A body file holds `q (P, N, 211)`: per person and frame the root pose `[tx, ty, tz, qx, qy, qz, qw]` in the world
and 51 joint quaternions, which BetterHuman poses; beside it the same bodies as the classic SMPL-X fields (`betas`,
`global_orient`, `body_pose`, `left_hand_pose`, `right_hand_pose`, `transl`), `joint_names`, `parents` and
`contact_frame_names`. Files written before the rebuild have no `format` and name the contact frame `head_top` by its
old name `head`; `bvr.files.load_npz` reads both.

## View

```bash
uv run python scripts/view_3d.py --source $BVR_DATA_ROOT/examples/data --save-dir ../runs
```

A browser viewer on BetterRobot's viewer, at `http://localhost:8082` (`[viewer].port`): the scene and human points,
the camera on its path with the video frame on it, gravity and the floor, and for the chosen body source (converter,
pre-fit, `kinematics_1`, `kinematics_2`, `kindyn_1`) the posed mesh, the skeleton, the contact frames (red in
contact, green free), the contact forces and the root residual; the video plays in step. A pager switches the clips
of a folder. `--record <file.mp4>` writes the first clip to a video instead (`docs/viewer.md`).

## Datasets

`bvr_datasets/` holds the code around the measured and labelled datasets, on top of the same chain: the contact
scorer of the annotated example clips, the BEDLAM2 and ClimbingVideos tools that make PACT's training labels, and
the force-rig recipes and scorers of the peter climbing walls, LAAS Parkour and OpenCap LabValidation.
[`docs/datasets.md`](docs/datasets.md) says what each one does and how to run it; each folder's `README.md` has the
commands.

## Documentation

- [`INSTALL.md`](INSTALL.md): environment, submodules, body models, model downloads.
- [`docs/pipeline.md`](docs/pipeline.md): the steps, what each reads and writes, and how a run resumes.
- [`docs/frame_rate.md`](docs/frame_rate.md): the one frame-rate rule of every time derivative.
- [`docs/fusion.md`](docs/fusion.md) and [`docs/scale.md`](docs/scale.md): the point clouds and the metric scale.
- [`docs/fitting.md`](docs/fitting.md): the kinematic fits, their phases and every term.
- [`docs/contacts.md`](docs/contacts.md): how contact labels are read and how far to trust them.
- [`docs/forces.md`](docs/forces.md): pose and forces solved together, and `force_confidence`.
- [`docs/config.md`](docs/config.md): the config file, deltas, and which step reads which table.
- [`docs/viewer.md`](docs/viewer.md) and [`docs/datasets.md`](docs/datasets.md).
