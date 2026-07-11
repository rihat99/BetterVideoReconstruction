# UPDATE — round 13: rescuing the floaters, and where the ceiling really is

*2026-07-12. You asked for recall around 0.5–0.6 on the new hard test set (the level
the easy scenes reach), precision held at ~0.9, no marginal tweaks, and marginal
mechanisms deleted rather than switched off. This round ships one big mechanism,
kills three tempting ones with measurements taken before building them, and ends
with an honest statement of what the remaining gap needs. Rounds 1–12 live in git
history.*

## First, a correction to every earlier number on this set

The geometry folders copied into `out_2/` were fused before the fps fix existed, so
their `transform.npz` had no fps and every optimizer window treated 24/25/60-fps
clips as 30 fps. All baselines were regenerated at native fps before anything else
was measured. The honest starting point on the 8 scenes is **P 0.889 / R 0.167 /
F1 0.281** (production cell: 3 cm, stillness 0.03, pre-attraction gate) — slightly
worse than the numbers we had been quoting.

## What shipped: the world now gets anchored to the body (commit `7a26984`)

The diagnosis from the test-set forensics held up: on the worst scenes VGGT puts the
whole climber 0.2–1.0 m in front of the wall (or inside it), and our metric scale —
which compares the body to *its own* VGGT depth — cannot see that error at all. The
one independent depth witness in the pipeline is the SAM-3D init body: its distance
from the camera comes from apparent size, so it does not move when the world is
rescaled.

`fuse_geometry` now checks every clip: if the init body's extremities sit further
than 28 cm from the fused scene on the median frame (healthy scenes top out at
22.5 cm; floaters start at 34.6 cm), or are buried deeper than 10 cm, the clip's
scale is corrected so the wall meets the body. The correction is chosen from exact
body-to-scene correspondences, must let a quarter of the frames touch without
burying the worst 5% deeper than 8 cm, is verified against ground-truth distances
(twice — once in the candidate world, once against the scene that is actually
written), and is vetoed whenever the evidence is thin, the effect would be a no-op,
or the people in the clip disagree. The refuted person's own depth cloud is kept at
its body-consistent scale, so body, evidence and wall all agree; healthy scenes are
untouched byte-for-byte.

Results on the 8 scenes, against the regenerated baseline:

| | P | R | F1 |
|---|---|---|---|
| native-fps baseline | 0.889 | 0.167 | 0.281 |
| **round 13** | **0.918** | **0.234** | **0.373** |

The two scenes that scored exactly zero are rescued outright — 9Rvr to F1 0.474 and
gtIGgs to 0.372, both at precision 1.0 — and re-scoring with the threshold shrunk by
the same rescale factor confirms the gains are real geometry, not a threshold
artifact. Two scenes refused the rescale, correctly: vcRB is simultaneously floating
(median +35 cm) and buried (worst 5% at −16 cm) — a dispersion no single scale can
fix — and 33OD's body is *compressed into* the scene (extremities −20 cm, torso
−54 cm), which a scale could only trade one way or the other.

## Three mechanisms measured to death before they were built

You asked for experimentation, and the discipline that paid off was measuring each
candidate's ceiling on the real data *before* writing the mechanism.

1. **Per-limb "still limb far from wall" evidence correction** (the natural widening
   of round-12's de-smear). Replaying its exact license over the current bodies:
   only 38% of the missed contacts happen while the limb is still — the hard ceiling
   is **R ≤ 0.446 even if every licensed miss converted**, and the false-positive
   risk lands precisely on ankle labels (448 free-ankle cells inside the license vs
   an allowed budget of ~405 extra false positives). Rejected.
2. **Casting each hand/foot's pixel ray onto the wall and using the intersection as
   depth evidence** (works on moving frames too). Measured oracle: to reach the
   misses the search window must be 0.6 m, and at that window 11–14% of the
   genuinely-free cells acquire a wall target the stillness gate would not veto —
   precision ceiling 0.825 *at perfect optimizer response*, and at the response rate
   lever 1 actually achieved the expected gain is ≈ +0.05 recall. It would also
   fight the contacts we already get right (97% of current true positives have their
   "target" more than 5 cm from where the joint correctly sits). Rejected.
3. **A global scale correction alone** (round 12's refuted version) — the reason is
   now sharper: rescaling moves the cloud with the wall, so the body keeps following
   its broken evidence. Lever 1 works precisely because it *also* pins the refuted
   person's cloud to the body scale.

## Where the remaining gap actually is

After lever 1, the misses decompose into: 52% of cells hover 10–30 cm off the wall
(per-frame depth noise — the init body is only good to ±20 cm per frame, and VGGT's
per-frame human depth on these scenes is worse), 19% sit 3–10 cm away, 12% are still
beyond 30 cm (mostly the two vetoed scenes), and 14% are geometrically correct but
vetoed as moving. Every geometric witness we have is now in use. What separates a
gripping hand from a hand hovering 15 cm off the wall is *appearance*, not geometry —
which is exactly the learned per-joint contact model the ClimbingVideos_v1 dataset
was exported for. When that model exists, its per-frame contact probability becomes
the license for seating evidence (replacing stillness), and the false-positive
problem that killed mechanisms 1 and 2 above is solved by the classifier instead of
by physics proxies.

Until then, 0.37 pooled F1 (0.23 recall) at 0.92 precision is where honest geometry
stops on this set. The easy-scene level (0.5–0.6 recall) is not reachable from the
current evidence; the numbers above say so more credibly than another round of
knob-turning would.

## De-smear: measured on this set, and the verdict — deleted

Round-12's two-pass evidence trim was re-measured head-to-head on the lever-1 world
across all 8 scenes: with it P .918 / R .234 / F1 .373, without it P .914 / R .230 /
F1 .367 — a pooled contribution of **+0.006 F1**. Per scene it behaves exactly as
designed: +0.018 on gtIGgs and +0.015 on CRqu (its target pathology), bit-identical
wherever its gate stays quiet, and −0.011 on one healthy scene. On the old 30-scene
set its share was ~+0.009.

That sits squarely inside your "marginal → delete, don't just turn off" bar, and two
independent judgments (mine and codex's) agreed on deletion for one decisive reason:
the mechanism can no longer grow. The per-limb widening — the only path to more
coverage — was measured this round to a hard ceiling of R ≤ 0.446 with its
false-positive risk concentrated on ankle labels, and was rejected. A permanently
narrow mechanism worth six pooled F1 thousandths is maintenance surface, not a
capability; its niche (still, provably-unsupported limbs) will be owned outright by
the learned contact license when that model lands. The code, its config section, and
its optimizer/loss plumbing are removed — the two-pass machinery, the warm-start and
weighted-evidence hooks, all of it. `docs/round12_depth_root_cause.md` stays as the
historical record with a removal note. The 8-scene numbers after deletion are the
"without" column above.

## Cleanup in this round

- The fps regeneration (above) — all `out_2` numbers now come from native-fps runs.
- `docs/geometry_fusion_plan.md` restored (it had been deleted while CLAUDE.md and
  README still pointed at it) — it now carries the current `geometry/*.npz` contract
  including the body-anchor fields.
- The corpus `features/` trees are now stale twice over (round-12 defaults AND the
  round-13 fuse gate); a full re-run is ~7 h on one GPU and is your call.
