# BetterVideoReconstruction — Claude Code Guide

Monocular video → a per-person **SMPL-X body trajectory** for everyone tracked,
plus **per-contact-joint forces** on the refined bodies. Track the people → VGGT
camera → Sapiens keypoints → fit a per-frame MHR body for **every tracked person**
(SAM-3D-Body, using the VGGT camera) → convert each to **SMPL-X** → **anchor
VGGT's metric scale to the human body** → fuse to metric geometry →
**visually-guided kinematic refinement** (2D reprojection + depth chamfer + pose
prior + acceleration/jerk smoothness + scene contact). The default chain ends at
the refined SMPL-X bodies; scene contacts, gravity, annotation-guided refinement,
force dynamics, and trust-scoring are standalone SMPL-X stages.

Code that was tested and then dropped — **MoGe2** depth, **MAMMA `ma_2d`**
landmarks, the **SMPL** (not SMPL-X) motion/dynamics path, and the **Gemma**
placeholder — lives under **`legacy/`** and is imported by nothing here; see
[`legacy/README.md`](legacy/README.md).

## Two pipelines, shared stages

- **Local, file-based** — `scripts/pipeline.py` runs the chain on a video / folder
  of videos → `out/<stem>/` (one tree per clip). No database yet; re-processing =
  re-running a stage (`--skip-existing`).
- **ClimbingVideos dataset** — `scripts/track_sam3_scenes.py` (SAM 3, seeded from
  the YOLO first-frame bbox) + `scripts/precompute_scenes.py`
  (`--stage vggt|sapiens|sam3d|smplx`, GPU-sharded) over the scene DB, writing
  `features/<stage>/<aa>/<bb>/<scene>/`. Shared DB access + row selection live in
  `scripts/scene_db.py`.

Both reuse the same per-stage CLIs in `scripts/stages/` and the `tools/` library.

## The pipeline & `out/<stem>/` contract

Run in this order (the orchestrator does it for you; each stage is also a
standalone CLI). The **default chain ends at `optimize_smplx`** (rows 1–8); the
rest are standalone SMPL-X stages, off the default chain.

| # | Stage (`scripts/stages/…`) | Reads | Writes (under `out/<stem>/`) |
|---|---|---|---|
| 1 | `track_sam3.py` | video/frames | `sam3/<obj:02d>/frame_NNNNNN.png`, `sam3/mask_overlay.mp4`, `sam3/bboxes.npz` |
| 2 | `estimate_camera_vggt.py` | frames | `vggt/camera.npz` (multi-view: `extrinsics (T,4,4)`, normalized + pixel `intrinsics`, VGGT inference resolution; ≤`max_frames`, default 500); `vggt/depth/…` with `--save-depth` |
| 3 | `estimate_sapiens.py` | frames, `sam3/bboxes.npz` | `sapiens/pose.npz`, `sapiens/overlay/…` (keypoints by default; `pointmap`/`normal` opt-in) |
| 4 | `estimate_body_sam3d.py` | `sam3/bboxes.npz`, `sam3/`, `vggt/camera.npz` (denormalized to orig res), frames | `sam3d/params.npz` (**all tracked people**, MHR params stacked `(P,N,…)` + `object_ids (P,)`), `sam3d/overlay.mp4` |
| 5 | `convert_mhr_to_smplx.py` | `sam3d/params.npz` | `sam3d/smplx_params.npz` (per-person `(P,N,…)`: betas, body/hand pose, global_orient, transl, jaw/eye/expr), `sam3d/smplx_overlay.mp4` |
| 6 | `estimate_scale.py` | `sam3d/smplx_params.npz`, `vggt/camera.npz`, `vggt/depth/`, `sam3/` | `geometry/scale.npz` (human-body-anchored metric scale + per-frame diagnostics), `geometry/diagnostics/scale_over_time.png` |
| 7 | `fuse_geometry.py` | `vggt/`, `geometry/scale.npz`, `sam3/` | `geometry/transform.npz` (metric scale + orig-res VGGT cameras), `geometry/scene.npz`, `geometry/human.npz` — see `docs/geometry_fusion_plan.md` |
| 8 | `optimize_smplx.py` | `sam3d/smplx_params.npz`, `sapiens/pose.npz`, `geometry/{human,transform,scene}.npz` | `sam3d/smplx_kinematic.npz` (**all tracked people**, refined **world-frame** params `(P,N,…)` + `q (P,N,211)` + `scale (P,)`≡1 (compat), 52-joint `SMPLXMid`, `frame='world'`), `sam3d/smplx_kinematic_overlay.mp4` (`--overlay`) |
| – | `optimize_smplx_2.py` | `sam3d/smplx_kinematic.npz`, **`data/<stem>.npz`** (manual contact annotation), `sapiens/pose.npz`, `geometry/{human,scene,transform}.npz` | `sam3d/smplx_kinematic_2.npz` (annotation-guided refinement; same schema), `sam3d/smplx_kinematic_2_overlay.mp4` (`--overlay`) |
| – | `estimate_contacts.py` | `sam3d/<bodies>.npz` (`--bodies`, default `smplx_kinematic`), `geometry/{scene,transform}.npz`, frames | `sam3d/cnt_<tag>.npz` (`joint_contact (P,N,J)` bool + `joint_sdf_cm`), `sam3d/cnt_<tag>_overlay.mp4` (skeleton; joints **red**=contact / **green**=free) |
| – | `estimate_gravity_geocalib.py` | frames, `geometry/transform.npz` | `geocalib/gravity.npz` (world-frame unit `gravity_world (3,)` down + per-frame diagnostics + `reliable` flag; magnitude 9.81 applied by dynamics) |
| – | `optimize_dynamics_smplx.py` | `sam3d/smplx_kinematic_2.npz` (frozen `q`), `sam3d/cnt_kin_2.npz`, `geocalib/gravity.npz` | `sam3d/smplx_dynamics.npz` (force-only RNEA: world `contact_forces`, `joint_torques (P,N,159)`, + viewer-ready `joints_world`/`torque_mag`/`contact_forces_world (P,N,52,…)`) |
| – | `estimate_uncertainty.py` | `sam3d/<bodies>.npz` (`--bodies`, default `smplx_kinematic_2`→`smplx_kinematic`), `sam3d/smplx_params.npz`, `sam3d/cnt_<tag>.npz`, `sapiens/pose.npz`, `geometry/{human,transform}.npz`, frames (`--overlay`) | `sam3d/uncertainty_<tag>.npz` (per-frame data-fit loss + relative/observability/residual/cue axes, per-contact confidence), `sam3d/diagnostics/uncertainty_person<oid>.png`, `sam3d/uncertainty_<tag>_overlay.mp4` (`--overlay`) |

Viewer: `scripts/view_3d.py out/<stem>/` (viser: fused `geometry/` scene cloud +
camera frustums + GeoCalib gravity arrow + per-climber human points + the refined
SMPL-X bodies — a **`body source` dropdown** switches `original`
(`smplx_params.npz`) / `kinematics` (`smplx_kinematic.npz`) / `kinematics_2`
(`smplx_kinematic_2.npz`) / **`dynamics`** (`smplx_dynamics.npz` — no mesh: the
52-joint skeleton as red torque-scaled spheres + white cylinder bones + cyan
contact-force arrows), defaulting to the most-refined body present).
`scripts/diagnose_smplx.py` plots every `optimize_smplx` loss term per frame.

`vggt` and `sapiens` are **precompute_runner Steps** (operate on `--source` /
`--save-dir`; a *folder* of videos loads each model once and sweeps the batch).
The post-track stages (`sam3d`, `smplx`, `scale`, `fuse`, `optimize_smplx`) take
`--predictions <clip-dir>`.

## Setup

See [`INSTALL.md`](INSTALL.md) for env / submodule / checkpoint setup.

## Run

```bash
# Whole chain on one clip (each model loads once for that clip).
uv run python scripts/pipeline.py --source video.mp4 --save-dir out/

# Batch of clips; skip clips already done.
uv run python scripts/pipeline.py --source clips/ --save-dir out/ --skip-existing

# A subset of stages.
uv run python scripts/pipeline.py --source video.mp4 --save-dir out/ --steps sam3,vggt

# Or run a stage standalone (full flag set lives on each stage CLI):
uv run python scripts/stages/track_sam3.py --source video.mp4 --save-dir out/ --text person
uv run python scripts/stages/estimate_camera_vggt.py --source video.mp4 --save-dir out/ --save-depth
uv run python scripts/stages/estimate_sapiens.py --source video.mp4 --save-dir out/ --tasks pose
uv run python scripts/stages/estimate_body_sam3d.py --source video.mp4 --predictions out/video/
uv run python scripts/stages/convert_mhr_to_smplx.py --source video.mp4 --predictions out/video/
uv run python scripts/stages/estimate_scale.py --predictions out/video/            # human-anchored metric scale (before fuse)
uv run python scripts/stages/fuse_geometry.py --source video.mp4 --predictions out/video/   # applies the scale
uv run python scripts/stages/optimize_smplx.py --source video.mp4 --predictions out/video/ --overlay
uv run python scripts/stages/optimize_smplx_2.py --source video.mp4 --predictions out/video/ --overlay   # annotation-guided (needs data/<stem>.npz)
uv run python scripts/stages/estimate_contacts.py --source video.mp4 --predictions out/video/            # per-joint scene contact
uv run python scripts/stages/estimate_gravity_geocalib.py --source video.mp4 --predictions out/video/    # world-frame gravity
uv run python scripts/stages/optimize_dynamics_smplx.py --predictions out/video/                         # force-only RNEA
uv run python scripts/stages/estimate_uncertainty.py --source video.mp4 --predictions out/video/ --overlay   # reconstruction-trust scoring
```

Orchestrator knobs + step selection live in
[`configs/pipeline.toml`](configs/pipeline.toml).

## Project layout

```
BetterVideoReconstruction/
├── CLAUDE.md / README.md / INSTALL.md
├── pyproject.toml / .gitmodules
├── docs/geometry_fusion_plan.md   # fuse_geometry design + geometry/*.npz contract
├── configs/
│   ├── pipeline.toml              # orchestrator: per-stage knobs + [steps]
│   ├── smplx_kinematic.toml       # optimize_smplx (visually-guided SMPL-X kinematics)
│   ├── smplx_kinematic_2.toml     # optimize_smplx_2 (annotation-guided refinement)
│   └── smplx_dynamics.toml        # optimize_dynamics_smplx (force-only RNEA)
├── scripts/
│   ├── pipeline.py                # LOCAL pipeline (load-once subprocess orchestrator)
│   ├── precompute_scenes.py       # DATASET pipeline: vggt/sapiens/sam3d/smplx fan-out
│   ├── track_sam3_scenes.py       # DATASET pipeline: SAM 3 seeding + fan-out
│   ├── scene_db.py                # shared scene-DB access + target selection
│   ├── view_3d.py                 # viser: geometry + cameras + bodies + dynamics
│   ├── diagnose_smplx.py          # per-frame plots of every optimize_smplx loss term
│   ├── _pym_preload.py            # libstdc++ LD_PRELOAD shim for pymomentum
│   └── stages/                    # the per-stage CLIs in the table above
├── tools/                         # importable library (repo root on sys.path via scripts)
│   ├── source.py                  # image/folder/video → frame paths
│   ├── precompute_runner.py       # Step protocol + run_batch + out/<stem>/ layout
│   ├── mask_io.py
│   ├── adapters/                  # stateless model wrappers
│   │   ├── sapiens2.py            # Sapiens2 pose/pointmap/normal
│   │   ├── vggt_omega.py          # VGGT-Omega multi-view camera + depth
│   │   └── geocalib.py            # GeoCalib single-image camera + gravity calib
│   ├── mhr_to_smplx/              # MHR → SMPL-X fitter (pip smplx; the body output)
│   ├── smplx_robot/               # SMPLXMid → BetterRobot 52-joint model + q remap; inertia.py + dynamics.py
│   ├── smplx_optim/               # visually-guided SMPL-X losses + optimizer + overlay
│   └── geometry/                  # depth clean + backproject + voxel merge + normals + surface + intrinsics + human-anchored scale
└── third_party/                   # submodules: sam3, sam-3d-body, sapiens2, MHR, vggt-omega, geocalib
```

## Conventions & deliberate choices

- **Multi-person throughout.** `estimate_body_sam3d` and `convert_mhr_to_smplx`
  fit **every** person SAM 3 tracked (`--obj-ids`, default = all), stacking on a
  leading `(P, N, …)` axis (with `object_ids (P,)`) and rendering one combined
  overlay (a colour per person). All downstream SMPL-X stages carry the `(P, …)`
  axis.
- **Body is SMPL-X.** `convert_mhr_to_smplx` (pip `smplx`, loading
  `better_human/models/smplx/`) captures the hands MHR predicts (jaw/eye/face left
  at zero). Its outputs live **inside `sam3d/`** next to the body params. SMPL-X has
  a BetterRobot 52-joint kinematic model (`tools/smplx_robot.make_smplx_robot_model`
  builds a `SMPLXMid`); the kinematic stage writes `q (P,N,211)`.
- **`optimize_smplx` is visually-guided, multi-person, world-frame.** It refines
  each person's trajectory against image + geometry: two **robust (Geman-McClure)
  2D reprojection** terms — the detector-free **init-projection anchor** (each body
  joint pinned to its OWN converter-pose projection, identity correspondence) plus
  a downweighted **Sapiens** cross-check — **chamfer** of visible posed vertices to
  the metric VGGT human depth, a gentle **depth-push**, a **joints-only pose prior**
  + gentle root-orientation prior, **acceleration + jerk** smoothness (exact Lie
  tangent — batched `so3`/`se3`), a **scene-penetration** repulsion (point-cloud
  SDF) and a **scene-contact attraction** (its inverse — the term that *raises* the
  contact rate), plus an off-by-default velocity-gated contact repulsion. The leaf
  is `q (T,211)` for `SMPLXMid` (quaternions renormalised in the forward — no
  exp-retraction singularity). **No per-body scale** — the metric geometry is sized
  to the body upstream by `estimate_scale`, so the body is used at native size
  (`smplx_kinematic.npz` still writes `scale (P,)`≡1 for consumer compat). The fit
  is an **Adam warm-up** in three coarse-to-fine stages (`root`: rigid placement;
  `full`: all DOFs, no scene terms; `refine`: add the scene terms) then a short
  **L-BFGS** polish. Posed-mesh inference is `better_human`; BVR adds the visual
  losses + the batched Lie smoothness (`tools/smplx_optim`). Chamfer + penetration
  are **chunked-batched** cdists (the per-frame loop was launch-bound — the real
  speed lever). **All loss weights live only in `configs/smplx_kinematic.toml`.**
  - **Keypoint correspondence is Goliath-308, not COCO-wholebody.** Sapiens emits
    `goliath_308` (no wrists in the body block; wrists live in the hand blocks).
    `tools/smplx_optim/keypoints.py` maps SMPL-X-mid joints to these indices —
    getting it right is what makes reprojection reliable (init reprojection is ~5px,
    i.e. the converter pose is already good).
  - **Chamfer owns DEPTH.** Reprojection pins the image plane, not the camera-ray
    distance, so absolute depth comes from chamfer to the metric cloud. Too small
    and the body floats off the cloud at its apparent-size depth; too large and it
    wrenches the body off-image to chase a locally-wrong cloud — the init-anchor +
    Sapiens hold the image while chamfer seats depth.
  - **Scene-penetration** (`losses.scene_penetration`, weight `[weights].penetration`)
    pushes posed vertices out of the static scene cloud (`geometry/scene.npz`, which
    excludes the dilated human). Nearest scene point + its camera-oriented PCA normal
    (`tools/geometry/normals.py`) sign the distance; a soft dead-zone (`margin_m`) +
    Huber ramp tolerate small penetration. Refine stage only; candidates AABB-pruned
    per frame and built once from the seated body.
  - **Scene-contact attraction** (`losses.scene_attraction`, weight
    `[weights].contact_attraction`) is the inverse: a vertex just outside the surface
    (`0 < s < band_m`) is pulled onto it, so near-wall parts (a climbing hand a few
    cm off) make contact. The two partition the signed-distance axis and never fight.
    It exists because nothing else pulls the body to the *scene* (chamfer targets the
    *human* depth cloud, penetration only pushes off).
- **Scene contact is a separate one-shot stage** (`estimate_contacts`, off the
  default chain): reads the refined body + `geometry/scene.npz` and marks
  **per-joint** scene contact for every person, reusing the penetration SDF +
  PCA normals (no gradients → a plain pruned `cdist`). A vertex is in contact when
  its signed distance is below `close_thresh_m`; a **joint** is in contact if any of
  its vertices is (vertex→joint by **nearest joint centre**, not LBS argmax — which
  would displace contact to the proximal joint). `--bodies` selects the body
  (`smplx_kinematic` → `cnt_kin`, `smplx_kinematic_2` → `cnt_kin_2`). Core geometry
  in `tools/smplx_optim/contacts.py`; knobs in `smplx_kinematic.toml` `[contacts]`.
- **`kinematics_2` is annotation-guided refinement.** `optimize_smplx_2` continues
  refining `smplx_kinematic.npz` using **manual per-joint contact labels**
  (`data/<stem>.npz` from `../BetterContactAnnotator`: tri-state over 10 coarse
  joints). It runs the kinematic losses plus three annotation-driven terms in the
  refine stage (weights `[weights].ann_*`): a **real attraction** pulling in-contact
  joints ONTO the wall, a **clearance repulsion** pushing no-contact joints past the
  contact threshold, and **velocity damping** on in-contact joints. Unlabeled joints
  feel none of these. `view_3d`'s `body source` compares original / kinematics /
  kinematics_2.
- **SMPL-X dynamics is a force-only RNEA pass** (`optimize_dynamics_smplx`): with
  the kinematic `q` frozen, solve per-contact-joint **world-frame forces** so the
  RNEA **base wrench** `tau[:6]` (the un-actuatable free-flyer root) vanishes; joint
  torques are whatever RNEA returns. Reads `smplx_kinematic_2.npz`, `cnt_kin_2.npz`
  (SDF contacts on that body — NOT the coarse annotation), and `geocalib/gravity.npz`.
  Real inertia for the 52-joint body comes from an LBS-argmax vertex→joint
  segmentation + per-part convex hulls renormalised to `total_mass_kg`
  (`tools/smplx_robot/inertia.py`); the per-person scale is folded in. Force-bearing
  joints default to hands (resultant at the wrist) + feet, active when any sub-joint
  is in contact — **no upward-only prior** (a climber's hand pulls down/in on a
  hold). Temporal `force_smooth`/`torque_smooth` terms cut per-frame flicker. `v`/`a`
  are central finite-differences of `q` (`velocity_acceleration_from_trajectory`,
  now in `tools/smplx_robot/dynamics.py`), **not** denoised. Config
  `configs/smplx_dynamics.toml`; `[runtime].device` defaults to **cpu**.
- **Reconstruction trust is a separate scoring stage** (`estimate_uncertainty`):
  per-frame + per-contact trust of an optimized body. The headline is a per-frame
  **data-fit loss** `L(t)` = the kinematic stage's *data* terms only
  (`reproj_sapiens·GM + reproj_init·GM + chamfer`, same weights + GM kernel as
  `optimize_smplx`; priors/smoothness excluded) → a **relative score**
  `s = L*/L(t) ∈ (0,1]` (robust variant vs the 5th-percentile loss). Three more
  axes: **observability** (per-clip-normalised geomean of # Sapiens kp / # confident
  depth points / mean VGGT depth-conf), **absolute residuals** (px + cm, fixed
  thresholds), **cue disagreement** (px spread between the Sapiens vs init-anchor
  reprojections). `combined = robust_score × obs_score`. **Per contact**, a
  confidence = SDF margin × joint speed × visibility × temporal persistence. The npz
  keeps every axis; the plot is a simplified 3-panel (confidence / fit-error /
  contacts), shared with the `--overlay` video.
- **Camera is its own stage; the body uses the VGGT camera.** VGGT intrinsics are
  scale-invariant, so the scaleless VGGT camera is exact for the body (which
  predicts its own metric translation). `estimate_body_sam3d` reads `K` from
  `vggt/camera.npz` (normalized, denormalized to orig res on the fly via
  `tools/geometry/intrinsics.py`); `fuse_geometry` writes the same `K` into
  `geometry/transform.npz` (`intrinsics_px_orig`) for later stages.
- **Gravity is its own stage (GeoCalib), recovered in the world frame.** The camera
  moves/tilts so it can't supply gravity. `estimate_gravity_geocalib` runs GeoCalib
  on sampled frames — each gives a camera-frame **up** vector (`gravity.vec3d`;
  physical gravity is its negative) + a confidence — maps them into the world frame
  (`g_world = -Rᵀ·up_cam`) and robustly combines. `reliable` needs a confidence
  floor + camera-rotation diversity + low inlier spread. On our clips only the
  outdoor `sport_climb` is reliable; the indoor gym clips fall back to world-down
  (`+y`; VGGT anchors its world to the ~upright camera 0). `view_3d` draws the arrow
  **red** when reliable, **amber** on the fallback.
- **VGGT-Omega geometry is up-to-scale, not metric** (one arbitrary per-clip factor
  on depth + camera translations; `scale='up_to_scale'`). It is a *multi-view* model
  (one forward over the clip), so it caps at `max_frames` (default 500) and runs at a
  reduced internal resolution recorded in the npz for mapping back to full-res.
- **Metric scale is the human body as a ruler** (`estimate_scale`, runs before
  `fuse`). Per person, per body-valid frame, it poses the SMPL-X body with that
  frame's own betas, front-face-culls, projects with VGGT's intrinsics, and over the
  `sam3` mask ∩ confident VGGT depth takes per-vertex `log(d_body) − log(d_vggt)`;
  the robust per-frame centre is pooled (MAD-gated, confidence-weighted, log space,
  `tools/geometry/human_scale.py`) into one clip factor `s` → `geometry/scale.npz`.
  Because VGGT is single-scale, this one `s` metricizes the whole clip; `fuse`
  applies it.
- **`fuse_geometry` turns VGGT metric.** It reads `s` from `geometry/scale.npz`,
  denormalizes VGGT intrinsics to the original resolution, and backprojects VGGT
  depth × `s` into a metric **scene** cloud (voxel-merged) + per-frame **human**
  points (unmerged) under `geometry/`. Needs `vggt` run with `--save-depth` and
  `estimate_scale` first.
- **The SAM 3 tracker lives in its stage**, not in `tools/adapters/` — it is a
  *stateful video-tracking Step* (tracking state, chunking, overlay/IO), not a
  stateless inference wrapper like `sapiens2`/`vggt_omega`/`geocalib`.
- **VGGT-Omega and GeoCalib are editable submodules** (`third_party/`), not
  git-pip deps — folded into `uv sync` and pinned.
- **Stages tolerate per-item failure**: precompute Steps print the error and keep
  going (a corrupt frame shouldn't kill a batch).

## What this guide intentionally does NOT cover

- Per-stage algorithmic detail — see each stage's module docstring.
- `BetterRobot` internals (kinematics, RNEA, residuals, viewer) — see
  `../BetterRobot/CLAUDE.md`. The optimizers depend on it heavily.

## See also

- [`docs/geometry_fusion_plan.md`](docs/geometry_fusion_plan.md) — `fuse_geometry` design + `geometry/*.npz` contract.
- [`README.md`](README.md) — user-facing overview.
- [`INSTALL.md`](INSTALL.md) — env, submodules, model assets.
- [`legacy/README.md`](legacy/README.md) — what was dropped and why.
