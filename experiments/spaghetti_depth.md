# Spaghetti depth — investigation record

**Question.** VGGT's per-frame depth sometimes stretches the *person* along the
camera rays — a limb reads longer than a real limb, or the body sits at the
wrong depth. We call this spaghettization. The fitted SMPL-X body has fixed
human proportions, so it cannot follow: it floats off the wall, or a hand never
reaches the hold it visibly grips. The idea under test: since the error lives
along rays, **slide each human cloud point along its own ray** to undo the
stretch — pinning points that sit near the scene (a hand on the wall is
probably right) and squeezing only the unsupported parts. Continuous, no gates.

Work happened on the worktree branch `spaghetti-depth` (cut from `dev`
@ 592943e). Raw artifacts: `…/.claude/worktrees/spaghetti-depth/experiments/spaghetti_depth/`
(`recon/`, `discriminator/`, `aniso/` — scripts, JSON, tables, heatmaps, logs).
Design briefs and an external adversarial review (codex) are in
`~/.claude/handoffs/spaghetti_depth_*` and `spaghetti_adjudicate.codex.out`.

## Verdict in one paragraph

The premise is half right and the fix built on it does not work as intended.
The stretch signal is real, smooth, and measurable — but "pin the cloud at the
wall, squeeze the rest" fails twice over. First, warping the cloud toward the
fitted body is circular (the body was fit to that same cloud), and with only
"don't move near the wall" as an anchor the mathematically correct smooth warp
is *zero everywhere*. Second, the softer version we did build — an anisotropic
chamfer that keeps full along-ray pull only for wall-corroborated points —
is numerically sound and mildly helps contacts (+0.016 mean F1, nothing
regresses), but leaves the actual spaghetti untouched: the worst scenes' bodies
still stop ~20% of depth short of the wall. The reason is not tuning. On
genuinely stretched limbs the cloud demands more reach than the skeleton has,
so *no reweighting of the same evidence* can close the gap; and the "severe"
scenes turn out to split into two different diseases that no single
cloud-domain correction treats.

## What was done, in order

1. **Recon** (13 local scenes, 15 person-tracks). Measured the per-point
   along-ray residual `r = log(z_body / z_cloud)` between the scale-stage
   pre-fit body and the fused human cloud, its spatial structure, its relation
   to distance-from-scene, and its stability over time.
2. **Design + adversarial review.** Three candidate formulations (warp the
   cloud with a scene-anchored smooth field; make the chamfer radially
   compliant; estimate a low-order warp inside the pre-fit) were designed,
   then attacked by codex acting as an external adversary.
3. **Discriminator probes** (4 severe + 2 control scenes). Two diagnostics
   built to be immune to the pipeline's own priors: a *fit-free* limb-span
   test (can any fixed-proportion body span this cloud at all?) and a
   *prior-free* overfit refit (chamfer + reprojection only, priors zeroed).
4. **Implementation + A/B.** The surviving design — the radially compliant
   chamfer — was implemented behind a default-off knob and evaluated on all 13
   annotated scenes at three bandwidths, scored against the manual contact
   labels.

## Findings

### The signal is real and smooth — but not "one scale per limb"

Within a frame, ~70% of the residual field's variance is captured by a coarse
20-pixel piecewise field (range 0.61–0.81 across tracks); it is temporally
smooth (lag-1 autocorrelation ~0.9). But a **single constant per limb — the
simple version of the sliding idea — captures only ~26%**. The stretch varies
*within* a limb, so any correction must be a spatially varying field, not a
per-limb linear squeeze.

The residual is also **blind to whole-body floaters**: the pre-fit body rides
its own cloud along the ray, so a uniform depth offset leaves no residual
(per-frame median r ≈ 0 on every track). Whatever detects floaters, it will
not be this.

### The wall-anchor premise is shakier than it looks

Against the pre-fit body, the residual is *largest* near the wall, not
smallest — the cloud reaches the wall, the body falls short (signed r at the
wall is negative for all 15 tracks; worst −0.25). By itself that does not
refute "the wall end of the cloud is correct" (a compromise fit *must* show
its residual at the extremities), but three things undermine using scene
proximity as ground truth: the scene and the human cloud come from the same
VGGT model, so the scene is not an independent witness; points that VGGT
placed at wall depth are *selected* to be "near scene", baking in the
conclusion; and anchors are scarce — typically only **2–7% of a person's
points lie within 3 voxels of the scene**.

### Two different diseases, one symptom

The discriminator split the severe scenes cleanly:

- **Real distal stretch** — `gtIGgs_y2nY_0024` (right arm) and
  `vcRBT9Q9jXM_0029` (both arms): the cloud's elbow→hand straight-line span is
  **1.33–1.41× the actual forearm+hand bone length** (controls < 0.85), with
  clean hand landmarks and one-arm dominance. No fixed-proportion body at any
  pose or depth can fit that cloud. This is true spaghetti.
- **Whole-limb / placement error** — `9RvrpYoWDoo_0001`,
  `3DAgLpPXVCs_0132` (person 1): the near-wall gap is just as bad (r down to
  −0.25) and a prior-free overfit cannot close it either (84–108% of the gap
  retained, even while sacrificing reprojection) — but the limb spans are
  body-feasible. The cloud is displaced, not stretched. Squeezing limbs here
  would be the wrong medicine.

A fix that operates on the cloud has to know which disease it is treating.
Nothing in the pipeline can tell them apart per-scene ahead of time.

### Why the cloud-warp design was rejected before building

The natural realization of the sliding idea — a smooth per-point log-depth
correction solved as a screened interpolation, pinned to zero near the scene —
dies on a structural point: with only the pin as a boundary condition, the
optimal smooth field is **zero everywhere**. To produce a nonzero correction it
needs a forcing term in the interior, and the only available one is "move the
cloud toward the fitted body" — self-distillation: the prior-anchored body
manufactures the target, the cloud moves to it, and the next fit is praised
for agreeing. On top of that, warping `human.npz` in place corrupts downstream
contracts (kinematics_1's limb quotas would amplify synthetic evidence,
auto-contact labels and `drift_pre` get contaminated, scale re-runs stop being
idempotent, and the depth-fit "spaghetti detector" would be grading our own
homework). The cloud was left untouched.

### What was built instead, and what happened

**Radially compliant chamfer** (worktree commits `cfd0130` + NaN fix; knob
`[chamfer] radial_sigma_vox`, default 0 = exactly the old behavior). Each
human depth point's chamfer residual is split about its own viewing ray; the
along-ray component is weighted by `1/(1+(d/σ)²)` where `d` is the point's
distance to the scene cloud. A hand-on-hold point (corroborated by the wall)
keeps full depth pull; an unsupported interior point fades continuously to
transverse-only, so a stretched cloud cannot drag the body along the ray. One
bandwidth knob, no gates, cloud never modified.

Two lessons from the evaluation:

- **A silent NaN bug first invalidated 10/13 runs.** The hand-written metric
  `√(‖e‖² − (1−τ)(e·r̂)²)` hits exactly zero (fp32 cancellation, or
  vertex-point coincidence), where sqrt's backward is infinite; one NaN
  poisons the whole solve, and the stage tolerates it — writing an all-NaN
  body with exit code 0. (`torch.cdist` guards its internal sqrt; a
  hand-rolled one needs a floor. Check *finiteness*, never return codes.)
- **After the fix (all 39 runs finite), the effect on spaghetti is nil.**
  Near-wall residual across σ = 0 / 2 / 6 voxels: 9Rvr 0.227 → 0.225 → 0.221,
  gtIGgs 0.204 → 0.200 → 0.202. Hand/foot depth gaps flat (gtIGgs stuck at
  31 cm). No dose-response. Side effect: at σ=6 contact F1 vs the manual
  annotations improves +0.016 on average (9 of 15 tracks up, worst regression
  −0.007, reprojection slightly better everywhere) at 1.7× stage runtime — a
  small free win, kept default-off.

### Why it could not have worked

Three numbers close the case. (1) A climber's cloud sits a median **26
voxels** off the wall, so at any usable bandwidth only ~2% of points are
wall-trusted (σ=6 gives mean trust 0.079) — widening σ far enough to reach the
limbs hands the torso its along-ray hold right back. (2) On a genuinely
stretched limb the too-deep forearm cloud pulls the arm **toward** the wall —
the same direction as the wall points — so making it compliant removes friend
and foe together; that is why σ=2 measured slightly *worse* than baseline on
vcRBT. (3) The binding constraint is anatomy versus cloud span: the prior-free
overfit, with every prior removed and reprojection allowed to degrade, still
retained 84–108% of the near-wall gap. **When the evidence itself is
infeasible for the skeleton, reweighting the evidence cannot fix it.**

## Where this leaves us

- The chamfer-shaping lever is exhausted. The implementation is kept (safe,
  small win, default-off) on branch `spaghetti-depth`.
- The evidence points at **contact boundary conditions** as the real lever:
  gtIGgs scores F1 0.008 in *every* configuration because the hand never gets
  near the wall, so contact is never detected, so no attraction ever fires — a
  chicken-and-egg that evidence reweighting cannot break. A limb the image
  shows touching should be pulled to the **wall** (the one trustworthy
  surface), via contact hypotheses that do not require the body to already be
  close — the kinematics_2 attraction machinery is the natural host. This is
  also the only formulation that is indifferent to the two-disease split.
- Whole-body placement errors (the floater family) remain a separate, upstream
  problem: invisible to every body-vs-own-cloud residual, untouched by any of
  this, still wanting the "general mechanism" promised when the fuse body
  anchor was removed.

*Companion record: [`FINDINGS.md`](FINDINGS.md) (the limb-length premise test —
whose verdict this investigation independently corroborates: the depth
residuals are not body-shape errors, and the body must stay frozen).*
