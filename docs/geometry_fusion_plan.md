# Plan — metric alignment & point-cloud fusion (VGGT × MoGe2 × SAM3)

> **Status: proposal for review.** Nothing here is implemented yet. This
> document is the design the user asked for *before* writing code.

## Goal

Add one new per-clip stage that runs **after the `moge2` camera stage** and turns
VGGT-Omega's up-to-scale, reduced-resolution geometry into a **metric,
original-resolution** scene, split into a static **scene/background** point cloud
and a dynamic **human** point cloud — **all tracked people**, via SAM3 masks.

Two problems to solve, in the user's framing:

1. **Scale.** VGGT depth + camera translations are *up-to-scale* (one arbitrary
   per-clip factor; `vggt/camera.npz` records `scale='up_to_scale'`). MoGe2 gives
   *metric* per-frame depth but is **per-frame and noisy** (no cross-frame
   consistency, outliers on some frames). We estimate **one global scale `s`**
   that metricizes the whole VGGT clip, robustly against MoGe2 noise.
2. **Resolution.** VGGT runs at a reduced internal resolution (here `384×688`)
   while frames are `720×1280`. We map VGGT's camera intrinsics to the original
   resolution. **FOV is held fixed (VGGT's FOV is authoritative)** — so this is a
   pure denormalize, no re-estimation.

### The one identity that makes this simple

VGGT depth is *backprojected through the VGGT cameras*, so depth and camera
translation share the **same** unknown scale. Therefore metricizing is a single
global multiply — **never touch rotation or FOV**:

```
X_world_metric      = s · X_world_uptoscale
camera_center_metric = s · camera_center_uptoscale   (i.e. scale extrinsic t by s)
```

Equivalently: multiply every VGGT depth value by `s` *and* every extrinsic
translation by `s`; rotations and intrinsics (FOV) are unchanged. We exploit this
— we estimate `s` once, then scale the final world points and the camera centers.

---

## Inputs consumed (all already on disk under `out/<stem>/`)

| Path | Keys used | Notes |
|------|-----------|-------|
| `vggt/camera.npz` | `extrinsics (T,4,4)` cam-from-world OpenCV; `intrinsics (T,3,3)` normalized; `intrinsics_px (T,3,3)` @ vggt res; `fov_x_deg/fov_y_deg`; `frame_indices (T,)`; `image_height/width`; `vggt_height/width`; `scale` | `T ≤ num_frames_total`; `frame_indices` maps VGGT frames → original frame numbers |
| `vggt/depth/frame_NNNNNN.npz` | `depth (Hv,Wv) f16` (up-to-scale, z-depth); `depth_conf (Hv,Wv) f16` (higher = better) | `NNNNNN` = **original** frame index. **Requires `[vggt] save_depth = true`** |
| `moge2/depth/frame_NNNNNN.npz` | `depth (Ho,Wo) f16` (metric); `mask (Ho,Wo) bool` (already sky/edge-filtered) | metric anchor. **Requires `[moge2] save_depth = true`** |
| `sam3/<obj:02d>/frame_NNNNNN.png` | per-object binary mask @ original res | human = **union of all tracked objects** (all are people); `--person-obj-ids` to restrict |
| `--source` | RGB frames (video / folder) | per-point colour |

Both `save_depth` flags are already `true` in `configs/pipeline.toml`. If either
`vggt/depth/` or `moge2/depth/` is missing the stage **fails fast** with an
actionable message (don't silently re-run a 5B model).

**Frame alignment.** For VGGT frame `k`, the original index is
`idx = frame_indices[k]`; load `moge2/depth/frame_{idx:06d}.npz` and
`sam3/<obj>/frame_{idx:06d}.png`. (For `sport_climb`, `frame_indices == arange(271)`,
so everything lines up 1:1; the indexing matters only when VGGT subsamples a
>500-frame clip.)

---

## Algorithm

### Step A — clean both depth maps (the user's "remove edges first")

Edges between objects are where every depth estimator flies pixels into the void;
MoGe ships the fix and we reuse it for **both** sources.

- **MoGe2 valid mask** = the saved `mask` (already excludes sky/invalid **and**
  ran `utils3d.np.depth_map_edge` inside `infer_frame`). Use as-is.
- **VGGT valid mask** = `finite(depth) & depth>0` **and**
  `~utils3d.np.depth_map_edge(depth, rtol=…)` (same built-in MoGe uses) **and** a
  **confidence gate**. Conf is unbounded and its scale drifts per frame, so gate
  by a *per-frame quantile* (e.g. drop the bottom `conf_drop_q≈0.3`), not an
  absolute threshold.

### Step B — robust global scale `s` (noise-tolerant by construction)

Per-pixel ratio `r = depth_moge / depth_vggt` is the scale at that pixel. We make
it doubly robust:

1. Resize MoGe2 depth **and** its mask from `(Ho,Wo)` down to VGGT res `(Hv,Wv)`
   (nearest for the mask; nearest/area for depth) so pixels correspond. (Down, not
   up — VGGT is the coarser grid and far cheaper.)
2. Co-visible set = `vggt_valid & moge_valid_resized`. **Per frame**, take the
   **median** of `r` over that set → `s_k` (kills bad pixels within a frame).
3. Global `s = median_k(s_k)`, weighted by each frame's inlier count (kills whole
   bad frames). Report `MAD(s_k)` as a quality signal.
4. *(optional refinement)* drop frames with `|s_k − s| > 3·MAD` and recompute.

Median-of-medians is intentionally chosen over a least-squares / single pooled
mean: the user explicitly called out per-frame MoGe2 outliers, and this is robust
to both pixel- and frame-level outliers while staying one cheap pass. (No RANSAC
needed; documented as a possible upgrade if `MAD` is ever large.)

> Slight aspect-ratio mismatch (`688/384=1.792` vs `1280/720=1.778`, <1%) means a
> VGGT pixel and a MoGe pixel aren't *exactly* the same ray; a global median over
> millions of pixels absorbs this. Scale estimation uses **z-depth ratios at
> matched pixels**, never cross-camera triangulation.

### Step C — resolution / intrinsics transform (FOV fixed)

VGGT intrinsics are stored **normalized** (`fx/W, fy/H`), which is
resolution-independent and FOV-preserving. So original-res pixel `K` is a pure
denormalize against the original frame size (reuse
`tools.adapters.moge2.denormalize_K`):

```
K_orig = denormalize_K(intrinsics_normalized, image_height, image_width)
```

- **Backprojection** uses `intrinsics_px` (already @ vggt res) against the vggt-res
  depth grid — these must match the depth resolution.
- **Reported camera model** for downstream consumers is `K_orig` (original res).
- Both describe the *same pinhole camera* (same FOV) → the 3D points are identical
  regardless of which grid we backproject on. The transform is bookkeeping, not a
  reprojection.

### Step D — backproject to metric world points (GPU)

For each VGGT frame `k` (chunked over frames to bound memory):

```
X_cam   = depth_k · Kpx_k⁻¹ · [u, v, 1]ᵀ            # z-depth backprojection
X_world = R_kᵀ (X_cam − t_k)                          # extrinsics = [R|t] cam-from-world
X_world_metric = s · X_world                          # the one global multiply
```

All in torch on `--device`. Keep `depth_conf` alongside each surviving point (used
as the voxel representative weight in Step F).

### Step E — human / scene split (SAM3 masks)

The SAM3 tracker is prompted with "person", so **every tracked object is a
person**. The human region is the **union of all `sam3/<obj>/` masks** (not just
obj 0): per frame, resize each object's mask from original res down to vggt res
(nearest) and OR them. Each human point is tagged with its `obj_id` (lowest id
wins on the rare overlap) so the people stay distinguishable. `--person-obj-ids
0,1` can restrict the set if ever needed. (This split is multi-person even though
the downstream **motion** path still fits a single person.)

The split is **asymmetric** — masks are imperfect, and the two clouds want
opposite error trade-offs:

- **Scene** = `valid_scene & ~dilate(human, --human-dilate)`. Masks under-cover
  the silhouette, leaking human-edge pixels into the background, so the scene
  subtracts a *dilated* human band (elliptical, default 8 px @ vggt res). The
  dilation band belongs to neither cloud — it is **dropped**, not reassigned —
  which is what keeps the scene clean.
- **Human** = `valid_human & human` (the tight mask). The human keeps **every**
  in-mask point: `valid_human` is the edge filter only (`finite & depth>0 & ~edge`)
  with **no confidence gate**, so `--conf-drop-q` thins only the scene.

### Step F — prune / merge (the "super-fast" core: voxel hashing, not kNN)

Raw count is huge — `271 × 384×688 ≈ 71.6 M` points before filtering — and many
overlap (the user's case: little camera motion ⇒ depth maps stack on the same
surface). **Neighbor search (KD-tree / radius) over tens of millions of points is
the slow trap.** Instead use **voxel-grid hashing**, which is `O(N)`, single-pass,
GPU-native, and *inherently* merges overlapping points:

```
ijk  = floor((X − aabb_min) / voxel_size)             # integer voxel coords
key  = (i·ny + j)·nz + k                               # pack to one int64 (exact)
uniq, inverse, counts = torch.unique(key, return_inverse, return_counts)
centroid = scatter_add(X, inverse) / counts           # one representative / voxel
conf_rep = scatter_max(depth_conf, inverse)            # keep best-conf attrs
```

- **Default: single batch pass** — backproject all frames, concat, one
  `torch.unique` + `scatter`. ~72 M int64 keys + xyz fit comfortably on a ≥24 GB
  GPU (<~2 GB, well under a second).
- **Fallback for very large clips: incremental `VoxelAccumulator`** — this is the
  user's "iteratively adding frames." Keep a running `key → (Σxyz, count, max_conf)`
  on GPU; per frame-chunk, `torch.unique` the *union* of current keys and the new
  chunk's keys and `scatter_add` into it. Peak memory = occupied voxels + one
  chunk, not all raw points. Occupied voxels ≪ raw points (that's the whole point
  of merging), so this stays bounded on arbitrarily long clips.

**Scene vs human are merged differently — important design call:**

- **Scene is static** → merge **globally** across all frames into one fused cloud.
  This is exactly where overlap-merging pays off. `voxel_scene ≈ 3–5 cm`.
- **Human is dynamic** (moves frame-to-frame) → merging across frames would smear
  the person into a blob. So **do not merge the human across frames**: voxel-
  downsample **within each frame only** (`voxel_human ≈ 1–2 cm`, finer) and keep
  per-frame clouds tagged by frame **and `obj_id`**. (A canonical-frame human cloud would need
  motion compensation — out of scope; flagged below.)

---

## Output contract — `out/<stem>/<geometry-folder>/`

| File | Keys |
|------|------|
| `transform.npz` | `scale ()`; `extrinsics (T,4,4)` **metric** (t·s, R unchanged); `intrinsics_px_orig (T,3,3)` @ original res; `intrinsics (T,3,3)` normalized (unchanged); `fov_x_deg/fov_y_deg (T,)` (unchanged); `frame_indices (T,)`; `image_height/width`, `vggt_height/width`; diagnostics: `per_frame_scale (T,)`, `per_frame_inliers (T,)`, `scale_mad ()`, `scale_method` |
| `scene.npz` | `points (M,3) f32` metric; `colors (M,3) u8`; `confidence (M,)`; `count (M,)`; `voxel_size ()` |
| `human.npz` | `points (P,3) f32` metric; `colors (P,3) u8`; `frame_id (P,) i32`; `obj_id (P,) i32`; `confidence (P,)`; `frame_indices (T,)` |

`transform.npz` records `scale`, so any later consumer can metricize raw
`vggt/depth` on the fly without re-deriving it — we keep scaled depth as an
intermediate, not a saved product.

---

## New reusable tools — `tools/geometry/` (new subpackage)

Per the user's "some functions as generic tools / algorithms as separate scripts."
Each is small and importable by the stage.

| Module | Responsibility |
|--------|----------------|
| `clean.py` | `clean_depth(depth, conf=None, conf_drop_q, edge_rtol, normal=None) → mask` — wraps `utils3d.np.depth_map_edge` / `normal_map_edge` (the MoGe built-in) + conf-quantile gate |
| `depth_align.py` | `robust_scale(vggt_depths, moge_depths, masks, conf) → (s, per_frame_scale, inliers, mad)` — Step B |
| `backproject.py` | `backproject(depth, K_px, extrinsic) → world_points` (torch, chunked) — Step D |
| `voxel.py` | `voxel_downsample(points, voxel_size)` + `group_reduce` — Step F |

---

## Pipeline integration

New per-clip stage script `scripts/stages/<stage>.py` (reads `--predictions`,
optional `--source` for color — mirrors `estimate_body_sam3d.py`). It loads no
heavy model, so it stays in the **per-clip** group of `pipeline.py`. Registration
points (all small, mirroring the existing pattern):

- `DEFAULT_RUN`: insert after `sapiens`, before `body` (it depends only on
  `sam3`+`vggt`+`moge2`; running it before `body` keeps cameras/geometry together).
- `_SCRIPTS`, `_perclip_output` (→ `<geometry-folder>/transform.npz`),
  `_build_perclip_args` (pass `--predictions`, `--person-obj-ids`, voxel sizes,
  `conf_drop_q`, `edge_rtol`, `--save-ply`, `--device`; `--source` for color).
- `_default_config()` + a `[<stage>]` block in `configs/pipeline.toml`.
- Docstring stage diagram + `CLAUDE.md` stage table + `README.md` table + a short
  schema note in `docs/`.

It runs standalone too:
`python scripts/stages/<stage>.py --predictions out/sport_climb/ [--source data/sport_climbing.mp4]`.

---

## Decided

- **Name:** step `fuse`, script `scripts/stages/fuse_geometry.py`, output folder
  `geometry/`. (Model stages are model-named — `sam3`, `moge2`, `vggt`, `sapiens`;
  this is a *processing* stage, so a function name fits the convention.)
- **Human cloud = per-frame, never merged across frames** (the human moves;
  merging would smear it into a blob). Scene stays globally merged.
- **All tracked people are split out as human, not just obj 0** — human = union of
  all `sam3/<obj>/` masks, each point tagged with `obj_id`. `--person-obj-ids`
  restricts the set if ever needed. The single-person motion path downstream is
  unaffected (it still fits obj 0).

### Assumptions I'll make unless told otherwise

- Scale estimated over **all** valid co-visible pixels (scene **and** human) — same
  frame, so depth is valid surface depth either way. (Could restrict to scene-only.)
- VGGT/MoGe `depth` are **z-depth** (standard for both); backprojection assumes z.
- World frame stays VGGT's (first cam at origin, OpenCV). We only **scale** it — no
  gravity/floor alignment (not requested).

---

## Verification

- **Scale sanity:** `s ≈ median(moge)/median(vggt) ≈ 6.96/0.71 ≈ 9.8` on
  `sport_climb`; `scale_mad / s` small (e.g. <10%) ⇒ MoGe2 frames agree.
- **Metric check:** scene extent and human height in metres are physically
  plausible (person ≈ 1.5–2 m tall).
- **Reprojection:** project metric scene points back through `K_orig` + metric
  extrinsics into a couple of frames; they should land on the right pixels.
- **Merge check:** fused scene point count ≪ raw `Σ valid pixels` (overlap merged),
  and reducing `voxel_size` monotonically increases the count.
- **Split check:** overlay human points' source pixels on the SAM3 mask — contained.
- **Inspect:** load `geometry/scene.npz` / `human.npz` (points + `colors`) in viser or a notebook.

## Out of scope (this stage)

Feeding metric cameras into the body stage (it still reads `moge2/camera.npz`);
gravity/floor fitting; motion-compensated human reconstruction; **multi-person
motion** (the geometry split *is* multi-person, but the downstream motion path
stays single-person); meshing the point cloud.
