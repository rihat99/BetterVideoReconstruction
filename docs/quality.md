# Quality: knowing when not to trust the reconstruction

Three related pieces, added 2026-08-24 after the athletics (data_2) batch showed
two failure modes the pipeline could not see: VGGT breaking dramatically on
broadcast sports footage, and contact forces being solved confidently on top of
broken kinematics.

## 1. Depth smear — the per-frame human-depth failure

VGGT's per-frame human depth has one characteristic catastrophic failure: fast
motion or fast zoom smears the person's points **along the viewing ray**, turning
a ~0.3 m-deep body into a streak of metres pointing at the camera (pommel_horse:
4 m+ streaks). Everything downstream that consumes those points then pulls the
body toward garbage — the chamfer data term drags the fit into the streak, and
the metric-scale ruler sizes the whole world to it (the "human huge" scenes).

The detector is one number per frame (`bvr/geometry/depth_smear.py`):

    smear = (depth extent along the view ray) / (largest lateral extent)

Both extents are robust (p90 − p10) about the cloud centroid. A real person has
more lateral extent (their height in view) than depth, so healthy frames sit
below 1; smeared frames read several. The ratio is scale-invariant, so it works
in metric and up-to-scale units alike.

Measured on every processed clip: all 13 annotated climbing scenes have max
smear ≤ 1.6. pommel_horse has 86 % of frames above 2.5 (median 7.9),
floor_exercise 45 %, high_jump 27 %, olympics_2 16 %.

**The weight.** Each frame gets `min(1, (2 / smear)^4)` — a **dead zone**: at or
below smear 2 the weight is exactly 1, so every clean clip behaves bit-identically
and there is no knob. Beyond it the weight falls fast (0.2 at smear 3, ~0.004 at
8). The weight multiplies, always on:

* the kinematic chamfer's per-frame average (`chamfer_loss(frame_weights=…)`) —
  a smeared frame pulls the body far less, and the image terms hold it instead;
* the pre-fit's cloud chamfer, the same way;
* the scale votes — both the ruler's per-frame weight and the pre-fit's
  per-frame scale measurement — so a streaked frame cannot size the world.

A frame with too few points to measure keeps weight 1: absence of a witness is
not evidence of bad depth.

## 2. The scene-quality table — finding the scenes where VGGT failed

The flag is **part of the pipeline**: `estimate_contacts` computes it (the
earliest stage where all three signals exist — the residual comes from its own
force witness) and saves it into `contacts_<tag>.npz` as `quality_flag` plus the
raw metrics behind it (`quality_cam_ratio`, `quality_smear_med`,
`quality_res_before_bw`), so every consumer of the contact labels can see
whether to trust them. Thresholds live in `[quality]` of
`configs/human_optimization.toml`; the shared logic in
`bvr/geometry/quality.py`. `scripts/diagnostics/scene_quality.py` prints one
row of raw metrics per `out/<stem>/` tree plus the same **`flag`** column
(recomputed with the same functions, so table and pipeline cannot drift):

| column | meaning | reads |
|---|---|---|
| `cam/depth` | camera translation spread ÷ median scene-to-camera distance | a tripod pan hallucinated as a dolly reads ≫ 1 (high_jump: 1.3) |
| `smear med`, `>2.5` | the per-frame smear above | pommel_horse: 7.9 / 86 % |
| `scale mad` | per-frame log-scale spread from `scale.npz` | frames never agreed on one world size |
| `far lbl` | fraction of contact labels > 10 cm from the scene | how much of the contact story floats in mid-air |
| `res b/a` | median root residual before/after the force witness, in body weights | ~1 before is normal support; ≫ 1 before *and* after = kinematics no force can explain (high_jump: 8.4/8.2) |
| `f-conf` | whole-body force confidence (below) | |

No single column catches everything: pommel_horse balances its residual fine on
a garbage world (`f-conf` 0.80) and is caught by `smear`; high_jump's clouds are
mostly fine per frame and is caught by `cam/depth` and `res`. Use the table, not
one number.

**The flags** encode the user's verdict on the first labelled batch (2026-08-26:
high_jump / -bUyE / 1FGkg / pommel_horse are bad; olympics_2 / balance_beam pass,
as does every climbing scene). Two failure modes, two flags:

* **`camera`** — `cam/depth` above `cam_depth_max` (0.4) **and** `res b` above
  `res_before_bw_max` (2.5 BW). Both witnesses must agree, because the camera
  ratio alone cannot tell a real handheld move from a hallucinated dolly:
  boulder_1 reads 0.71 with coherent physics (1.0 BW) and must pass, while
  1FGkg reads 0.68 carrying 2.8 BW of acceleration no contact force can
  explain. A fake dolly corrupts the world *and* the trajectory; a real one
  corrupts neither.
* **`depth`** — median smear above `smear_med_max` (2.0) — the same 2.0 as the
  smear weight's dead zone, one number with one meaning.

The thresholds were set on 23 labelled clips; revisit them as the dataset grows.

## 3. Force confidence — how much to trust the solved forces

Contact forces are solved for the whole body at once — every limb balances the
same root residual — so their trustworthiness is a property of the **body per
frame**, never of one joint. `bvr/human_optim/force_confidence.py` scores it
from the physics alone; the kindyn stage saves it as `force_confidence (P, N)`
plus the per-clip mean `force_confidence_clip (P,)`.

Three ingredients, each mapped through the same saturating curve
`1 / (1 + (x/x*)²)` and multiplied:

* **linear residual** (dominant, halves at 0.5 BW) — the unexplained root force
  after the solve. If the solve could not balance the linear motion, the forces
  it did assign are allocations of garbage.
* **angular residual** (gentle, halves at 1.5 BW·lever) — the contact cones have
  little moment authority, so nonzero angular residual is normal even on good
  sequences.
* **overload** (engages above 3 BW total) — a solve pushing several body weights
  frame after frame is explaining reconstruction error, not motion.

Calibration on the processed clips: healthy climbing scores 0.81–0.94
(sport_climb 0.50 — honest: the rope carries force we cannot model), the broken
data_2 clips 0.18–0.36. Scene-side evidence (label grounding, smear) is
deliberately *not* folded in — join it per clip from the table above, so each
score keeps one meaning.
