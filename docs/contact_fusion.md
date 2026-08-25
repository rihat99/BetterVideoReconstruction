# Contact fusion: combining detected and predicted contacts

> **Retired.** This describes the per-joint-era mechanism; `fusion.py`, the
> `labels = "predicted"` wiring and the peter configs that used them are deleted
> from the code and live in git history. Kept as a design record.

The pipeline has two independent opinions about whether a limb touches the wall.

- The **geometric labels** from the contact stage ([contacts.md](contacts.md)):
  precise when the fused scene cloud has geometry near the limb, but blind where
  it does not. A hand gripping a rung that was erased with the human mask reads
  "no contact" simply because there is nothing to measure against.
- The **predicted contacts** from an external image model (the climb4 network,
  `predictions/contacts.npz`): a per-frame probability for each of four limbs
  (left/right hand, left/right foot), computed from the video alone. It sees the
  picture, so missing scene geometry does not blind it — but it has false
  positives, and it knows nothing about the 3D body.

Fusion combines the two at the very end of `estimate_contacts.py`, after the
detector, the completion pass, and the confidence computation. It changes labels
only; everything upstream is untouched. It runs whenever the clip has a
`predictions/contacts.npz` — for `contacts_1` and `contacts_2` alike; without
one, the stage prints "fusion: OFF" and the output is the pure geometric
labeling.

## The idea: add the evidence

Each source is turned into a probability of contact, and probabilities are
combined by adding their **log-odds** (the log-odds of `p` is
`log(p / (1 - p))` — 0 means "no idea", positive means "contact", negative
means "free"). Adding log-odds is the textbook way to merge two independent
witnesses: each one pushes the belief by however much it actually knows.

Per limb and frame:

1. **Geometric probability.** The limb's label `L` is the OR of `joint_contact`
   over its member joints (a hand is the wrist plus the fingers; a foot limb is
   the foot plus the ankle). Its confidence `c` is the label confidence of the
   strongest touching member (if in contact) or the weakest representative
   joint (if free). Then `p_geo = 0.5 + 0.5·c` for a contact label and
   `0.5 − 0.5·c` for a free label. A label the confidence machinery does not
   believe in (`c ≈ 0`) contributes nothing — which is exactly the
   "no scene data here" case.
2. **Model probability.** The network's probability, clamped into
   `[0.10, 0.90]` (`[contacts.fusion].pred_clamp`). The clamp bounds how loudly
   the model may argue — its only guard. A frame with no usable prediction
   counts as 0.5 (silence).
3. **Fused label.** `logit(p_geo) + logit(p_pred) > 0`. An exact tie keeps the
   geometric label, so "no usable prediction" is a guaranteed no-op. The fused
   confidence is `|2·sigmoid(sum) − 1|`, back on the same 0–1 scale.

There are no gates. A confident geometric label cannot be flipped by the
clamped model; a low-confidence one can. That single mechanism handles both
directions: the model rescues contacts where geometry is absent, and geometry
overrules the model's false positives wherever it actually measured something.

## Writing the changes back

- **Rescue** (fused says contact, geometry said free): the limb's
  representative joints are set — the wrist for a hand, the foot *and* the
  ankle for a foot limb (the same convention the completion pass uses).
  `contact_source` gains bit 8.
- **Veto** (fused says free, geometry said contact): every member joint of the
  limb is cleared.
- Both kinds of change are dropped if they last less than `min_span_s`
  (0.1 s) — the stage's usual span rule, applied symmetrically.
- Changed joints get the fused confidence; untouched labels keep the witness
  confidence they already had.

The output records what happened: `pred_prob_limb` (the probabilities fusion
actually used, NaN where absent), `fusion_changed` (+1 rescue / −1 veto per
limb-frame), `fusion_method`, and the `+predfuse-logodds-v1` suffix on
`contact_label_method`. A scene with no `predictions/` file is byte-identical
to the unfused output apart from the schema number.

## Measured effect

Against the manual ground-truth labels of the 13 `out/` scenes (4-limb fold,
~20k scored limb-frames), fusion moves pooled precision/recall/F1 from
.908/.672/**.772** to .899/.774/**.832** — ten points of recall for one of
precision, 2000 rescues against 320 vetoes. The result is flat across clamp
values 0.05–0.35: almost every rescue lands on a limb whose geometric
confidence is near zero, so it is the geometric confidence, not the clamp,
that keeps the model's false positives out.

## Where else the predictions enter

On the peter force-plate datasets (static cameras), `kinematics_1` takes its
frozen contact labels **directly from the predictions** instead of the geometric
detector (`[stages.kinematics_1] labels = "predicted"`,
`configs/human_optimization_peter.toml`). The static-camera VGGT confidence is
uniformly low near the limbs and thin structures, so the peter configs also
un-gate the clouds (`conf_drop_q = 0` in `[fuse]`, `[scale]` and `[chamfer]`,
`human_dilate = 2`) — that restores the campus board to the scene and the hands
to the human cloud, letting the prefit and chamfer anchor the arms at the right
depth. With those labels and that geometry, the ordinary kinematics_2
acquisition+seating then sits the hands on the board. See
[human_optimization.md](human_optimization.md).
