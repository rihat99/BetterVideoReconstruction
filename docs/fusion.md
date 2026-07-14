# The fusion stage: metric point clouds + the body anchor

`scripts/stages/fuse_geometry.py` — runs after `estimate_scale`, needs the VGGT stage
run with depth saving on.

This doc explains how the metric 3D world is built from VGGT's output, and how the
**body anchor** decides whether that world can be trusted.

## What fusion produces

Three files under `geometry/`:

- `transform.npz` — the metric scale, the camera poses with metric translations, and
  the camera intrinsics at the original video resolution. Every later stage reads its
  cameras from here.
- `scene.npz` — one static, merged point cloud of everything that is not a person: the
  wall, the holds, the ground.
- `human.npz` — the person's depth points, kept per frame and unmerged (the person
  moves, so merging frames would smear them).

## How the clouds are built

For every frame:

1. **Clean the depth.** Two filters. An *edge filter* removes "flying pixels" — points
   VGGT smears across depth discontinuities. A *confidence gate* drops the least
   confident 30% of pixels. The scene gets both; the human cloud gets only the edge
   filter, because the person is exactly where we cannot afford to throw points away.
2. **Split person from background.** The SAM3 masks say which pixels belong to which
   tracked person. The scene subtracts a *dilated* (grown by 8 px) union of every
   tracked person, so imperfect mask edges do not leak person pixels into the wall.
   The human cloud keeps the tight mask, and only for people the pipeline actually
   reconstructed — a tracked bystander is carved out of the scene but not added to the
   human cloud.
3. **Backproject** the remaining pixels to 3D using VGGT's depth times the metric scale
   `s` from `geometry/scale.npz`.

Then, once per clip:

4. **Merge the scene** by voxel hashing: points falling in the same 5 cm cell become
   one point. The scene is static, so this is lossless at the resolution we use it.
5. **Drop small clusters.** Connected groups of fewer than 25 voxels are removed. These
   are floating depth fragments and off-surface blobs — for example an untracked person
   VGGT placed in mid-air.

## The body anchor: catching a wrong world

### The problem it solves

The metric scale comes from comparing the body with *the person's own depth pixels*.
That makes it blind to one failure: when VGGT places the person's depth at a different
relative distance than the wall (the "floater" failure — the human cloud hovers half a
metre off the wall). The scale is then internally consistent but the world is wrong:
the body, placed at its own depth, floats in the air. No later stage can fix this,
because the kinematic losses all assume the world is roughly right.

### The gate

The SAM-3D body is an independent witness: its depth comes from the person's apparent
size in the image, not from VGGT. So fuse poses the initial bodies in the fused world
and measures, per valid frame:

- **gap** — the distance from the four extremities (wrists and feet) to the nearest
  scene point. A climber touches the wall constantly; a median gap above **0.28 m**
  (healthy clips top out at 0.225, floaters start at 0.346 on the calibration set) says
  the world and the body disagree grossly.
- **burial** — the signed distance of the deepest extremity *behind* the wall surface.
  A median burial beyond **0.10 m** is the same disagreement in the other direction.

Small disagreements never fire: the init body is only approximate, so only gross
disagreement is treated as evidence.

### The fix: rescale the world, not the body

If the gate fires, the human depth ruler is refuted, and the whole world (scene and
camera translations) is rescaled by a factor `r` so the wall meets the body:

1. For each valid frame, take the body vertex nearest the wall and its matched scene
   point with its surface normal. Because a world rescale moves each scene point along
   a known line, the signed body-to-wall distance after rescaling by `r` can be written
   exactly per correspondence. Near-tangent matches (the normal almost perpendicular to
   the viewing direction) are rejected — their predicted crossing point is unstable.
2. Choose the `r` closest to 1 such that at least a quarter of the frames come within
   2 cm of the wall (the *touch* rule) while the most-buried 5% of frames stay within
   8 cm behind the surface (the *burial budget*). If the two rules conflict, burial
   wins: never buy contact by burying the body.
3. **Verify, don't trust the prediction.** The prediction assumed the nearest-neighbour
   matches stay fixed, but they switch as the world rescales. The stage re-poses the
   body in the candidate world and measures the true gaps; if the burial budget is
   violated, `r` is backed off toward 1 and re-checked (up to 6 times). After the new
   world is actually built (re-voxelization can shift surfaces by up to a voxel), the
   check runs once more on the final cloud; a violation vetoes everything.

The refuted person's own depth points are *not* rescaled — they keep the body-consistent
scale, because they were the wrong ruler in the first place.

### The vetoes

A world change must never ship unverified or half-justified. The rescale is dropped
(and the baseline world rebuilt bit-identically) when:

- a fired person keeps too few usable correspondences — a clip-wide claim needs
  clip-wide evidence;
- two fired people demand scales differing by more than 5% — one clip has one world,
  so contradiction falsifies the single-scale explanation;
- a *healthy* person would be pushed out of a plausible band (median gap between
  −5 cm and +20 cm) by the rescale;
- the chosen `r` is within 2% of 1 — relabeling the clip "refuted" while applying a
  no-op rescues nothing;
- verification loses support or cannot satisfy the burial budget.

### What a fired anchor means downstream

Rescaling changes the world, so every artifact built in the old world — the kinematic
fits, contacts, dynamics — is stale for that clip, and fuse prints a warning saying so.
When the anchor does not fire, the output is byte-identical to a build without the
anchor; the gate only ever adds diagnostics to `transform.npz` (`rescale_r`,
`depth_refuted`, per-person gap/burial/support, and the verified true gaps).
