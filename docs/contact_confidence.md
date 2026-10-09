# Contact-label confidence

`estimate_contacts` decides, for every tracked person and every video frame,
which body points touch the scene. Those labels are not always right. The
**confidence** is a number between 0 and 1 saved beside each label that says how
often a label like this one turns out to be what a human annotator would have
written. A learner uses it to weight its training loss. It never changes the
label itself.

The number is fitted, not designed. `scripts/calibrate_contact_confidence.py`
compares the automatic labels with the manual annotations on the corpus scenes
that have them and fits a tiny model; the fitted numbers live in
`[contact_confidence]` in `configs/human_optimization.toml`. The code is
`bvr/human_optim/confidence.py`.

## The unit: a report joint

Contacts are detected on **contact frames** — named points baked onto the skin
(palms, fingertips, heels, toes, knees). People do not annotate 35 skin points;
they annotate joints. So the frame labels are first folded onto **report
joints**: every frame reports at its parent joint, and every hand frame (palm,
fingertip, thumb) reports at its wrist. A joint is in contact when any of its
frames is.

Confidence is computed on those folded joints, because that is the unit the
annotations — and the learner — speak in. Each contact frame then simply carries
its report joint's number.

Eight joints have fitted coefficients: the two wrists ("hand"), the two feet
(the forefoot, i.e. the toes), the two ankles (the heel) and the two knees. Left
and right share coefficients. Everywhere else — elbows, hips, torso, head — the
confidence is `NaN`: the stage makes no statement there, and a consumer reading
a non-finite value should treat the label as unassessed rather than as zero.

## What the old weight was doing

The confidence this replaced was hand-built: visibility times evidence, with the
evidence capped according to which arm of the detector produced the label. Read
as "how often is this label right", it did not work. Measured on 259 114 rows —
one per (scene, person, frame, annotated joint) over the 118 annotated scenes
(after the manual-label check finished on 2026-09-12):

| | old weight | new confidence |
|---|---|---|
| AUROC, contact labels | 0.65 | **0.82** |
| AUROC, free labels | 0.71 | **0.87** |
| calibration error (ECE), contact | 0.40 | **0.008** |
| calibration error (ECE), free | 0.44 | **0.018** |

AUROC is how well the number separates right labels from wrong ones (0.5 = no
better than a coin). ECE is how far the stated probability is from the observed
truth — the old mean weight was 0.49 while 92 % of the labels were in fact
right. The new numbers are out-of-fold: the model is refitted five times, each
time without the videos it is then scored on, so scenes cut from the same video
cannot flatter it.

Where the old weight was most wrong, split by whether the joint was actually
still (net drift under 8 cm over the readout window) and what the annotator
wrote:

| cell | share of rows | labels right | old weight | new confidence |
|---|---|---|---|---|
| still & contact | 33 % | 0.920 | 0.54 | 0.92 |
| still & free | 23 % | 0.938 | 0.40 | 0.93 |
| moving & contact | 8 % | 0.539 | 0.28 | 0.74 |
| moving & free | 37 % | 0.966 | 0.54 | 0.93 |

The still-and-free row is the one that mattered. Those labels are 94 % right and
the old weight muted them to 0.40 — so a learner trained on it was taught "still
means contact". The new confidence gives them 0.93, which is what the data say.
The remaining weak cell is moving-and-contact: only 54 % of those labels are
right and the model still states 0.74, so it is the one place the confidence is
meaningfully over-optimistic.

## The formula

Three witnesses, one logistic model per label class (one for "the label says
contact", one for "the label says free"):

```text
x_drift = log(drift_m + 0.005)                drift over the readout window, metres
x_trans = log1p(30 * min(t_s, 1.0))           t_s = seconds to the nearest label change
logit   = group_term + w_drift * x_drift + w_trans * x_trans
conf    = 1 / (1 + exp(-logit))
```

* **Which joint.** `group_term` is the joint's own intercept: hand, foot, ankle
  or knee. Accuracy differs enormously between them (knee labels are 99 %
  right because a knee is essentially never labelled in contact; forefoot labels
  are 83 % right), so the joint is the single strongest witness. Each group's term
  IS its intercept — the model carries no separate global one, which keeps every
  coefficient pinned by the data and the fit reproducible from run to run.
* **How still it was.** `drift_m` is the joint's net displacement over the 0.35 s
  readout window, taken from `drift_pre` — the motion measured *before* any
  contact enforcement acted on the body, so the optimizer's own damping cannot
  manufacture stillness. A joint's drift is the smallest among its frames: the
  stillest frame speaks for the joint. Where the window is undefined (an
  untracked stretch) the confidence is `NaN`. More drift makes a contact label
  less believable and a free label more believable, which is why the two classes
  fit this witness with opposite signs.
* **How far from a label change.** Labels are least reliable exactly where they
  flip. A flip between two consecutive tracked frames sits halfway between them,
  so both frames are 0 s from it and the distance grows one frame at a time.
  Untracked gaps are not flips, and neither is the start or the end of a track. A
  joint whose label never changes is treated as being at the 1 s cap. The
  distance is measured in seconds and then converted to 30-fps-equivalent frames,
  so a 60-fps clip and a 30-fps clip mean the same thing by it.

## What was dropped

Everything else that used to feed the weight, and several things that never did,
were tested as extra features on the same rows: the signed distance to
the scene cloud, the solved force share, the pre-union arm agreement, the span
length, image-border and body-occlusion visibility, the depth-fit trust, the
Sapiens keypoint score, the limb's reprojection residual, and the per-clip
quality metrics. Each added at most 0.01 AUROC on top of the three witnesses
above.

Two of them were also circular: the scene distance and the stillness *are* how
the geometric arm decides, so letting them certify the label would be the label
endorsing itself. The ray-cast body-occlusion witness and the depth-fit trust
were removed outright (`bvr/human_optim/depth_fit.py` is deleted), which also
removes the most expensive part of the stage.

The honest reading of that result: the annotations say no image or depth
evidence available here predicts label errors beyond what the joint, its
stillness and its distance to a flip already say.

## Refitting

```bash
uv run --no-sync python scripts/calibrate_contact_confidence.py \
    --config configs/human_optimization.toml --write
```

It reads every `features/annotation/<shard>/<scene>/annotation.npz` in the
ClimbingVideos DB together with that scene's `contacts_1.npz`, folds both the
same way, fits, prints the tables above, and replaces the `[contact_confidence]`
section of each config given. Existing artifacts pick the new numbers up when
the contacts stage is re-run.

The coefficients in the configs were fitted on 2026-09-12, after the manual
labels had been checked; the `fit` string in the TOML says so. Refit after any
change to the labeller or the annotations, and refresh the numbers in this
document from the script's printout.

One known gap in the fit: the contact head has no knee rows at all, because the
stage never labels a knee in contact. Its knee coefficient is therefore 0 and
unused. And the fit is done on all annotated scenes at once, which are the
downstream learner's *test* split — good for the learner, but it means the
confidence has no fully held-out set of its own. When the fit was instead done on
the 107 test scenes and scored on the 11 annotated train scenes (measured on the
labels before the final check), the free head held up (AUROC 0.96) but the
contact head dropped to 0.67. Those 11 scenes are
the hardest in the corpus, so treat 0.67 as a floor rather than as the expected
number, and re-check it after a refit.

## What is saved

In `human_optim/contacts_<tag>.npz`, per person `P` and video frame `N`, over
`F` contact frames and `J` model joints:

| Field | Shape | Meaning |
|---|---|---|
| `joint_contact` | `(P, N, J)` bool | The label, folded: any of the joint's frames touching |
| `joint_label_confidence` | `(P, N, J)` | P(this label agrees with an annotator); `NaN` = no statement |
| `frame_label_confidence` | `(P, N, F)` | Its report joint's confidence |
| `frame_report_joint` | `(F,)` | Which joint each frame reports at (hands → wrist) |
| `contact_source` | `(P, N, F)` int8 | 2 = geometric arm, 4 = force witness (0 off-contact) |

Plus `confidence_method` (`calibrated-v2`), `confidence_motion_source`
(`drift_pre`) and `confidence_coefficients`, the exact fitted numbers as a JSON
string, so any artifact says how its confidence was produced.

## What it cannot do

* It contains no image evidence. It cannot tell you that a hand is hidden behind
  a body, only that labels on hands drift-and-flip the way they do.
* It does not repair a wrong label. A 0.2 confidence marks a label as doubtful;
  it does not flip it.
* Net displacement over the window misses out-and-back motion: a hand that leaves
  a hold and returns inside 0.35 s reads as still.
* The moving-and-contact cell stays over-confident (0.74 stated against 0.54
  observed). Those are the reconstruction's genuinely hard frames, and none of
  the witnesses available here separates them.
* It moves little for the easy majority: 77 % of all rows land in the top
  confidence bin, because most labels really are right. Weighting a loss by it
  mostly reweights the ~5 % of rows that come out below 0.7.
