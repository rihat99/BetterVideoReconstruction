# BetterVideoReconstruction

Reconstruct a person's **3D motion and contact forces** from a monocular video.
Given a clip of one person, the pipeline tracks them, recovers per-frame camera
intrinsics, fits a 3D body each frame, converts it to a shape-aware SMPL
BetterRobot, and runs a two-stage optimization: a kinematic pass that smooths
and anchors the trajectory, then a dynamics pass that infers per-contact-joint
forces via inverse dynamics (RNEA).

It is the video-focused sibling of
[`BetterImageReconstruction`](../BetterImageReconstruction) (image/scene) and
[`BetterHumanForce`](../BetterHumanForce) (the original research repo this was
split from). All three sit next to the [`BetterRobot`](../BetterRobot)
PyTorch-native robotics library, which provides the kinematics, RNEA, and
optimization residuals used here.

## Pipeline

```
video ─▶ SAM 3 track ─▶ MoGe2 camera ─▶ Sapiens2 pose ─┐
              │                                         ├▶ SAM-3D-Body (MHR)
              └▶ VGGT-Ω (multi-view camera + depth)     │      │
                                                        │      ▼
                                                   MHR → SMPL ─▶ build_robot_motion
                                                                      │
                                            kinematic optimize ◀──────┘
                                                   │
                                            dynamics optimize ─▶ motion + contact forces
```

| Stage | Model | Output |
|-------|-------|--------|
| sam3 | SAM 3 (video) | per-person masks + bbox tracks |
| vggt | VGGT-Omega | multi-view cameras + depth (≤500 frames; records the VGGT inference resolution) |
| moge2 | MoGe2 | per-frame camera intrinsics (+ optional depth) |
| sapiens | Sapiens2-5B | 308-keypoint pose + metric pointmap |
| fuse | metric align + fuse | metric scene + per-frame human point clouds (`geometry/`) |
| body | SAM-3D-Body | per-frame MHR body params (single person) |
| smpl | MHR→SMPL fit | per-frame SMPL params (`q(t)`) |
| motion | BetterRobot | shape-aware SMPL robot + smoothed trajectory |
| optimize_motion | L-BFGS / `CostStack` | kinematically refined `q(t)` |
| optimize_dynamics | RNEA | per-contact-joint forces + joint torques |

**VGGT-Omega** runs right after the tracker as a multi-view camera/geometry
stage; it writes its own `vggt/camera.npz` (cross-frame extrinsics + depth, plus
the inference resolution) and is additive — the body stage still reads MoGe2's
`moge2/camera.npz`. Its access-gated checkpoints are not bundled (see
[`INSTALL.md`](INSTALL.md)).

**Future / placeholder stage:** Gemma 4 VLM triage + contact graph
(`classify_gemma`).

## Quick start

```bash
# See INSTALL.md for env + submodule + checkpoint setup first.
uv run python scripts/pipeline.py --source video.mp4 --save-dir out/

# Then view the result (motion + contact-force arrows) in a viser web viewer:
uv run python scripts/view_motion.py out/video/results/2.npz
```

Outputs land under `out/<stem>/` (per clip). The current build is file-based
and small-batch — a DB-backed dataset is planned later. Each stage is also
runnable on its own; see [`CLAUDE.md`](CLAUDE.md) for the full stage table, the
`out/<stem>/` contract, and conventions, and
[`docs/motion_npz_schema.md`](docs/motion_npz_schema.md) for the motion `.npz`
schema.

## Documentation

- [`INSTALL.md`](INSTALL.md) — environment, submodules, model assets.
- [`CLAUDE.md`](CLAUDE.md) — architecture, pipeline contract, conventions.
- [`docs/motion_npz_schema.md`](docs/motion_npz_schema.md) — `results/*.npz` layout.
