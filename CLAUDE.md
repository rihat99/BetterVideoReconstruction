# BetterVideoReconstruction — Claude Code Guide

Monocular video → a per-person **SMPL-X body trajectory** for everyone tracked,
plus **per-contact-joint forces** on the refined bodies. Track the people → VGGT
camera → Sapiens keypoints → fit a per-frame MHR body for **every tracked person**
(SAM-3D-Body, using the VGGT camera) → convert each to **SMPL-X** → **anchor
VGGT's metric scale to the human body** → fuse to metric geometry →
**visually-guided kinematic refinement** (2D reprojection + depth chamfer + pose
prior + acceleration/jerk smoothness + scene contact). The default chain ends at
the refined SMPL-X bodies; scene contacts, gravity, annotation-guided refinement,
and force dynamics are standalone SMPL-X stages.

Code that was tested and then dropped — **MoGe2** depth, **MAMMA `ma_2d`**
landmarks, the **SMPL** (not SMPL-X) motion/dynamics path, and the **Gemma**
placeholder — lives under **`legacy/`** and is imported by nothing here; see
[`legacy/README.md`](legacy/README.md).

## Experimental branch — working principles

This is the `experimental` branch. It exists to try out ideas and to make the
code simpler, without disturbing the working pipeline on `main`. Commits land on
`experimental` here (not `main`). Four rules guide every change on this branch:

1. **Keep new features simple and clean.** Add the least code that does the job.
   If it starts sprawling, step back and find the smaller version.
2. **Write docs a person can actually read.** Short, plain sentences. Explain a
   term the first time it shows up. No walls of jargon.
3. **Never over-engineer.** No configurability, abstraction, or "future-proofing"
   the task did not ask for.
4. **Prefer methods that are elegant and clever over methods held together by
   rules.** Avoid piling on gates, thresholds, and hand-tuned special cases; reach
   for the idea that makes them unnecessary.

## Git workflow

Commit directly to **`main`** — do **not** create branches. Land all work on
`main` (this overrides the default "branch before committing on the default
branch" behavior for this repo).

## Two pipelines, shared stages

- **Local, file-based** — `scripts/pipeline.py` runs the chain on a video / folder
  of videos → `out/<stem>/` (one tree per clip). No database yet; re-processing =
  re-running a stage (`--skip-existing`).
- **ClimbingVideos dataset** — `scripts/track_sam3_scenes.py` (SAM 3, seeded from
  the YOLO first-frame bbox) + `scripts/precompute_scenes.py`
  (`--stage vggt|sapiens|sam3d|smplx`, GPU-sharded) over the scene DB, writing
  `features/<stage>/<aa>/<bb>/<scene>/`. The `--predictions` stages that don't fit
  the manifest fan-out uses `scripts/precompute_contacts_scenes.py`, a symlink-farm
  contact readout written **alongside the body** in `features/human_optim/<scene>/`
  to mirror the local
  `out/<stem>/human_optim/` layout; `--out-root` redirects for smoke tests).
  `scripts/export_contact_dataset.py` packages those features into a standalone
  22-joint train/test contact corpus; manual annotations fill test labels only.
  Shared DB access + row selection live in `scripts/scene_db.py`.

Both reuse the same per-stage CLIs in `scripts/stages/` and the `tools/` library.

## The pipeline & `out/<stem>/` contract

Run in this order (the orchestrator does it for you; each stage is also a
standalone CLI). The **default chain ends at `human_optimization`** (rows 1–8,
which runs its `root` + `kinematics_1` sub-stages); the rest are standalone
SMPL-X stages, off the default chain.

| # | Stage (`scripts/stages/…`) | Reads | Writes (under `out/<stem>/`) |
|---|---|---|---|
| 1 | `track_sam3.py` | video/frames | `sam3/<obj:02d>/frame_NNNNNN.png`, `sam3/mask_overlay.mp4`, `sam3/bboxes.npz` |
| 2 | `estimate_camera_vggt.py` | frames | `vggt/camera.npz` (multi-view: `extrinsics (T,4,4)`, normalized + pixel `intrinsics`, VGGT inference resolution; ≤`max_frames`, default 500); `vggt/depth/…` with `--save-depth` |
| 3 | `estimate_sapiens.py` | frames, `sam3/bboxes.npz` | `sapiens/pose.npz`, `sapiens/overlay/…` (keypoints by default; `pointmap`/`normal` opt-in) |
| 4 | `estimate_body_sam3d.py` | `sam3/bboxes.npz`, `sam3/`, `vggt/camera.npz` (denormalized to orig res), frames | `sam3d/params.npz` (**all tracked people**, MHR params stacked `(P,N,…)` + `object_ids (P,)`), `sam3d/overlay.mp4` |
| 5 | `convert_mhr_to_smplx.py` | `sam3d/params.npz` | `sam3d/smplx_params.npz` (per-person `(P,N,…)`: betas, body/hand pose, global_orient, transl, jaw/eye/expr), `sam3d/smplx_overlay.mp4` |
| 6 | `estimate_scale.py` | `sam3d/smplx_params.npz`, `vggt/camera.npz`, `vggt/depth/`, `sam3/` | `geometry/scale.npz` (human-body-anchored metric scale + per-frame diagnostics), `geometry/diagnostics/scale_over_time.png` |
| 7 | `fuse_geometry.py` | `vggt/`, `geometry/scale.npz`, `sam3/`, `sam3d/smplx_params.npz` | `geometry/transform.npz` (metric scale + orig-res VGGT cameras + **body-anchor** gate diagnostics), `geometry/scene.npz`, `geometry/human.npz` |
| 8 | `human_optimization.py` (`root`+`kinematics_1`) | `sam3d/smplx_params.npz`, `sapiens/pose.npz`, `geometry/{human,transform,scene}.npz` | `human_optim/kinematic_1.npz` (**all tracked people**, refined **world-frame** params `(P,N,…)` + `q (P,N,211)` + `scale (P,)`≡1 (compat), 52-joint `SMPLXMid`, `frame='world'`), `human_optim/kinematic_1_overlay.mp4` (`--overlay`) |
| – | `human_optimization.py --stages kinematics_2` | `human_optim/kinematic_1.npz`, **`data/<stem>.npz`** (manual contact annotation), `sapiens/pose.npz`, `geometry/{human,scene,transform}.npz` | `human_optim/kinematic_2.npz` (annotation-guided refinement; same schema), `human_optim/kinematic_2_overlay.mp4` (`--overlay`) |
| – | `estimate_contacts.py` | `human_optim/<bodies>.npz` (`--bodies`, default `kinematic_1`), `geometry/{scene,transform}.npz`, frames | `human_optim/contacts_<tag>.npz` (`joint_contact (P,N,J)` bool, signed-SDF and Euclidean-distance diagnostics, `merged`, and schema-4 per-joint confidence fields). Confidence is `visibility × label evidence`: contact uses soft surface evidence; no-contact uses motion only, so a stationary free joint is untrusted regardless of reconstructed wall distance. There is no clearance/floating-body, detector, body-health, or persistence gate. Detected full body occlusion (at least five sampled occluders) is exactly zero-confidence. See [`docs/contact_confidence.md`](docs/contact_confidence.md). The optional overlay draws **red**=contact / **green**=free with contact-dot size ∝ confidence. |
| – | `estimate_gravity_geocalib.py` | frames, `geometry/transform.npz` | `geocalib/gravity.npz` (world-frame unit `gravity_world (3,)` down + per-frame diagnostics + `reliable` flag; magnitude 9.81 applied by dynamics) |
| – | `human_optimization.py --stages dynamics` | `human_optim/kinematic_2.npz` (frozen `q`), `human_optim/contacts_2.npz`, `geocalib/gravity.npz` | `human_optim/dynamics.npz` (force-only RNEA: world `contact_forces`, `joint_torques (P,N,159)`, + viewer-ready `joints_world`/`torque_mag`/`contact_forces_world (P,N,52,…)`) |

Viewer: `scripts/view_3d.py out/<stem>/` (viser: fused `geometry/` scene cloud +
camera frustums + GeoCalib gravity arrow + per-climber human points + the refined
SMPL-X bodies — a **`body source` dropdown** switches `original`
(`sam3d/smplx_params.npz`) / `kinematics` (`human_optim/kinematic_1.npz`) /
`kinematics_2` (`human_optim/kinematic_2.npz`) / **`dynamics`**
(`human_optim/dynamics.npz` — no mesh: the 52-joint skeleton as red torque-scaled
spheres + white cylinder bones + cyan contact-force arrows), defaulting to the
most-refined body present). `scripts/diagnose_human_optim.py` plots every
`human_optimization` kinematics_1 loss term per frame, per person.

`vggt` and `sapiens` are **precompute_runner Steps** (operate on `--source` /
`--save-dir`; a *folder* of videos loads each model once and sweeps the batch).
The post-track stages (`sam3d`, `smplx`, `scale`, `fuse`, `human_optimization`) take
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
uv run python scripts/stages/human_optimization.py --source video.mp4 --predictions out/video/ --overlay   # root + kinematics_1 (default chain)
uv run python scripts/stages/human_optimization.py --source video.mp4 --predictions out/video/ --stages kinematics_2   # annotation-guided (needs data/<stem>.npz)
uv run python scripts/stages/estimate_contacts.py --source video.mp4 --predictions out/video/            # per-joint scene contact (--bodies kinematic_1)
uv run python scripts/stages/estimate_gravity_geocalib.py --source video.mp4 --predictions out/video/    # world-frame gravity
uv run python scripts/stages/human_optimization.py --predictions out/video/ --stages dynamics            # force-only RNEA (needs contacts_2 + gravity)
```

Orchestrator knobs + step selection live in
[`configs/pipeline.toml`](configs/pipeline.toml).

## Project layout

```
BetterVideoReconstruction/
├── CLAUDE.md / README.md / INSTALL.md
├── pyproject.toml / .gitmodules
├── docs/contact_confidence.md     # per-joint confidence formula + contacts artifact schema
├── configs/
│   ├── pipeline.toml              # orchestrator: per-stage knobs + [steps]
│   └── human_optimization.toml    # all 4 human-optimization stages: 4-column [loss_weights.*] tables + shared knobs
├── scripts/
│   ├── pipeline.py                # LOCAL pipeline (load-once subprocess orchestrator)
│   ├── precompute_scenes.py       # DATASET pipeline: vggt/sapiens/sam3d/smplx fan-out
│   ├── precompute_contacts_scenes.py  # DATASET pipeline: contact readout (symlink farm)
│   ├── export_contact_dataset.py # DATASET packaging: frames/masks/intrinsics/contact labels
│   ├── track_sam3_scenes.py       # DATASET pipeline: SAM 3 seeding + fan-out
│   ├── scene_db.py                # shared scene-DB access + target selection
│   ├── view_3d.py                 # viser: geometry + cameras + bodies + dynamics
│   ├── audit_scenes.py            # round-7 corpus floater audit → features/audit/bodies.csv
│   ├── autopsy_scenes.py          # round-7: assign each flagged body a cause (a/b/c/d)
│   ├── diagnose_human_optim.py    # per-frame per-person plots of every kinematics_1 loss term
│   ├── diagnose_depth_streaks.py  # round-9 T0: VGGT depth-streak meter + contact go/no-go slice
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
│   ├── human_optim/               # the 4-stage human optimizer: config/paths/body/motion/losses/optimizer/stages + contacts/overlay/diagnostics
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
- **`human_optimization` is one stage, four sub-stages, one config.** A single
  `human_optimization.py` dispatcher runs four stages: **`root`** (rigid placement —
  root SE3 only, joints frozen), **`kinematics_1`** (the visually-guided kinematic
  fit), **`kinematics_2`** (annotation-guided), **`dynamics`** (force-only RNEA). The
  default chain runs `root` + `kinematics_1`; the latter two are opt-in and, for now,
  algorithmically **frozen**. `root` and `kinematics_1` share ONE optimization call —
  `root` is its first phase, and the priors / init-anchor stay pinned to the converter
  input pose across both (so root cannot be a separate call). Machinery lives in
  **`tools/human_optim/`** (config / paths / body / motion / losses / optimizer /
  stages), keeping the dispatcher thin.
  - **The kinematic fit** (`kinematics_1`) refines each person's world-frame `q
    (T,211)` for `SMPLXMid` against image + geometry: two **robust (Geman-McClure) 2D
    reprojection** terms — the detector-free **init-projection anchor** (each body
    joint pinned to its OWN converter-pose projection) + a downweighted **Sapiens**
    cross-check — **chamfer** of visible posed vertices to the metric VGGT human
    depth, a **joints-only pose prior** + gentle
    root-orientation prior, **acceleration + jerk** smoothness (exact Lie tangent,
    batched `so3`/`se3` in `tools/human_optim/motion.py`), a **scene-penetration**
    repulsion and its inverse **auto-contact attraction** (the term that *raises*
    contact — see below).
    Quaternions are renormalised in the forward (no exp-retraction singularity). It runs
    `full` (all DOFs, no scene) → `refine` (add scene terms on the settled body) → a
    short **L-BFGS** polish. Chamfer + penetration are **chunked-batched** cdists (the
    per-frame loop was
    launch-bound — the real speed lever). Round-12: the human cloud is sampled with
    **per-limb quotas** (`[chamfer] sampling="limb_quota"`, default — hands/feet keep up
    to 200 points each instead of being drowned by the torso; robust/anisotropic chamfer
    kernels were tried, measured worse, and removed — the raw along-ray chamfer pull is
    load-bearing). The later physics-gated de-smear second pass was measured and
    removed because its improvement ceiling was negligible.
    Gross whole-clip depth errors are instead handled upstream by the
    **fuse body anchor** (see the `fuse_geometry` bullet).
  - **No per-body scale** — `estimate_scale` sizes the metric geometry to the body
    upstream, so the body is used at native size (`kinematic_1.npz` writes
    `scale (P,)`≡1 for consumer compat). Frozen at 1.0; one `apply_scale` path is
    kept for a future unfreeze. Posed-mesh inference is `better_human`; **no pip
    `smplx` on the optimization path** (only the upstream `convert_mhr_to_smplx`).
  - **One config, table-shaped.** `configs/human_optimization.toml` holds every loss
    weight as a length-4 array **`[root, kinematics_1, kinematics_2, dynamics]`**,
    grouped into `[loss_weights.kinematic|motion|scene|annotation|dynamics]` tables (a
    term is off in a stage whose column is 0); shared operational knobs are separate
    scalar sections. `config.py` flattens each column into a per-stage `{term: weight}`
    dict the optimizer consumes.
  - **Keypoint correspondence is Goliath-308, not COCO-wholebody.** Sapiens emits
    `goliath_308` (no wrists in the body block; wrists live in the hand blocks).
    `tools/human_optim/keypoints.py` maps SMPL-X-mid joints to these indices —
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
  - **Auto-contact attraction** (kinematics_1): contact labels are detected **once**
    at the start of the refine phase (the contact readout at the `[auto_contact]`
    operating point), **frozen**, and the labeled joints' vertices just outside the
    surface (`0 < s < band_m`) are pulled onto it (`losses.attraction_penalty`, weight
    `[loss_weights.annotation].ann_attraction`; `ann_velocity` also damps the labeled
    joints). Attraction and penetration partition the signed-distance axis and never
    fight. It exists because nothing else pulls the body to the *scene* (chamfer
    targets the *human* depth cloud, penetration only pushes off). The readout stays
    honest: `estimate_contacts` gates stillness on `drift_pre` — measured BEFORE the
    attraction/damping acted.
    (Round-10's wall-contact seating pull + rigid `seat` phase were dropped, and
    round-12's physics-gated de-smear followed in round 13 — both superseded by the
    **fuse body anchor**, which corrects the *world* where its gate says the depth
    evidence is refuted instead of pushing the body.)
- **Scene contact is a separate one-shot stage** (`estimate_contacts`, off the
  default chain): reads the refined body + `geometry/scene.npz` and marks
  **per-joint** scene contact for every person, reusing the penetration SDF +
  PCA normals (no gradients → a plain pruned `cdist`). A vertex is in contact when
  its signed distance is below `close_thresh_m`; a **joint** is in contact if any of
  its vertices is (vertex→joint by **nearest joint centre**, not LBS argmax — which
  would displace contact to the proximal joint). `--bodies` selects the body
  (`kinematic_1` → `contacts_1`, `kinematic_2` → `contacts_2`). Core geometry
  in `tools/human_optim/contacts.py`; knobs in `human_optimization.toml` `[contacts]`.
- **`kinematics_2` is annotation-guided refinement.** `human_optimization --stages
  kinematics_2` continues refining `kinematic_1.npz` using **manual per-joint contact
  labels** (`data/<stem>.npz` from `../BetterContactAnnotator`: tri-state over 10
  coarse joints). It runs the kinematic losses plus three annotation-driven terms in
  the refine phase (`[loss_weights.annotation]`): a **real attraction** pulling
  in-contact joints ONTO the wall, a **clearance repulsion** pushing no-contact joints
  past the contact threshold, and **velocity damping** on in-contact joints. Unlabeled
  joints feel none of these. `view_3d`'s `body source` compares original / kinematics /
  kinematics_2.
- **SMPL-X dynamics is a force-only RNEA pass** (`human_optimization --stages
  dynamics`): with the kinematic `q` frozen, solve per-contact-joint **world-frame
  forces** so the RNEA **base wrench** `tau[:6]` (the un-actuatable free-flyer root)
  vanishes; joint torques are whatever RNEA returns. Reads `kinematic_2.npz`,
  `contacts_2.npz` (SDF contacts on that body — NOT the coarse annotation), and
  `geocalib/gravity.npz`. Real inertia for the 52-joint body comes from an LBS-argmax
  vertex→joint segmentation + per-part convex hulls renormalised to
  `[inertia].total_mass_kg` (`tools/smplx_robot/inertia.py`); the per-person scale is
  folded in. Force-bearing joints default to hands (resultant at the wrist) + feet,
  active when any sub-joint is in contact — **no upward-only prior** (a climber's hand
  pulls down/in on a hold). Temporal `force_smooth`/`torque_smooth` terms cut per-frame
  flicker. `v`/`a` are central finite-differences of `q`
  (`velocity_acceleration_from_trajectory` in `tools/smplx_robot/dynamics.py`),
  **not** denoised. `[stages.dynamics].device` defaults to **cpu**.
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
  `estimate_scale` first. Round-13 **body anchor** (`--body-anchor`, default on;
  `tools/human_optim/depth_anchor.py`): the human-ruler scale is blind to
  human-vs-scene bias, so fuse gates every clip on the SAM-3D init body (metric from
  apparent size, camera-frame position scale-invariant) — extremities > 0.28 m off
  the fused scene (or buried > 0.10 m) refute the clip's human depth, and the world
  is rescaled so the wall meets the body (exact correspondences, burial-budgeted,
  ground-truth-verified twice, vetoed on thin evidence / no-op / person
  disagreement); the refuted person's human cloud stays at the body-consistent
  scale. Fired ⇒ downstream artifacts of that clip are stale (fuse warns); not
  fired ⇒ byte-identical build + diagnostics in `transform.npz`.
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

- [`docs/contact_confidence.md`](docs/contact_confidence.md) — per-joint contact-confidence formula and schema.
- [`docs/ISSUE.md`](docs/ISSUE.md) — current scale, fusion, and human-optimization audit.
- [`README.md`](README.md) — user-facing overview.
- [`INSTALL.md`](INSTALL.md) — env, submodules, model assets.
- [`legacy/README.md`](legacy/README.md) — what was dropped and why.
