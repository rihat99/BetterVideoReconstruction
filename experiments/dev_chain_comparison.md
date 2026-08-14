# Dev chain vs main chain vs the learned model (2026-08-14)

The dev branch's contact chain had only ever been judged on the 13 annotated
`data/` scenes. This runs it on the corpus **test split** — the 30 held-out,
hand-annotated ClimbingVideos scenes — and puts three readings of contact on the
ruler that split has always used.

## What was run

The full dev chain (`kinematics_1 → contacts_1 → kinematics_2 → contacts_2 →
kindyn_1`, HEAD `f466af7`, the default `configs/human_optimization.toml`) over
all 30 test scenes, writing to a new feature tree
**`features/human_optim_dev/`** so the existing `features/human_optim/` (the
main-branch artifacts from 2026-07-26) stays untouched. One GPU (4), two
stride shards, per-scene symlink farms; 30/30 scenes, zero failures, ~57 min.
Runner and scorer live in `scratch/dev_testset/` (`run_shard.sh`,
`score_compare.py`).

One asymmetry to keep in mind: the main chain's `contacts_2` has the learned
image model **fused in** (log-odds fusion, 2026-07-26); the dev chain dropped
that path — its labels are geometry plus the force witness only. So "main vs
dev" is also "with model vs without model".

## The ruler

The same 4-limb fold as `scripts/diagnostics/score_test_contacts.py` (my
scorer reproduces its stored `fused2` and `learned` rows to the digit): a hand
is its wrist plus fingers, a foot is the foot OR the ankle; truth is the manual
tri-state annotation, unlabeled limb-frames dropped. Dev frame labels fold as
palms+fingers → hand, toes+heel → foot. Tables (PNG + `metrics.json`):
`<data_root>/results/dev_comparison/`.

## Test split — 30 scenes, 32,376 limb-frames

| Contact Detection Method | P | R | F1 |
|---|---|---|---|
| Optimization | **0.962** | 0.905 | 0.932 |
| Optimization + model | 0.953 | 0.935 | **0.944** |
| Optimization + forces | 0.949 | 0.883 | 0.915 |
| Learned model (p > 0.5) | 0.927 | **0.939** | 0.933 |

What each method is:

- **Optimization** — the main chain's `contacts_1` with its fusion flipped back
  (`fusion_changed` records every flip): the geometric readout before the model
  enters the chain anywhere. Reproduces the official table's old
  "Optimization" row to the digit. There is no model-free `contacts_2` in the
  main chain — the kinematics_2 body was repaired toward the *fused* contacts_1
  labels, so the model's contacts are baked into that geometry and a
  contacts_2-pre-fusion reading silently re-detects them (the first version of
  this row made that mistake; it read .937).
- **Optimization + model** — the main chain's `contacts_2`: the learned image
  model fused in, and the body repaired with the fused labels.
- **Optimization + forces** — the dev chain's `contacts_2`: the force witness
  instead of the model, no image model anywhere.
- **Learned model** — the image model alone at its stored 0.5 threshold.

Always-contact floor: **0.878** (climbers are usually touching — quote it next
to every F1 here). The split is easy, as expected: everything scores high.
Model fusion buys +1.2 F1 (.932 → .944). The no-model comparison favors
plain optimization here: forces .915 vs .932 (position-matched at contacts_1,
the force chain reads .906) — on easy scenes the completion machinery's
recall beats the force witness.

## Hard examples — the 13 annotated `data/` scenes, 19,802 limb-frames

Main row = the main checkout's `out/<scene>/human_optim/contacts_2.npz`
(2026-07-26), dev row = the dev checkout's regenerated trees, learned row =
`out/<scene>/predictions/contacts.npz`.

| Contact Detection Method | P | R | F1 |
|---|---|---|---|
| Optimization | **0.908** | 0.672 | 0.772 |
| Optimization + model | 0.900 | 0.776 | 0.833 |
| Optimization + forces | 0.889 | 0.763 | 0.821 |
| Learned model (p > 0.5) | 0.874 | **0.827** | **0.850** |

Always-contact floor: 0.819. Here the no-model comparison flips hard: plain
optimization drops to .772 (recall .672 — the floaters' feet go missing),
while optimization + forces holds .821. The force witness is worth +4.9 F1
over plain optimization on hard scenes — what it was built for — and even
position-matched at contacts_1 the force chain is ahead (.795 vs .772). Model
fusion is worth +6.1 F1 (.772 → .833); the force chain gets within 1.2 of
optimization + model with no image model at all.

Two caveats before reading a winner:

- **6 of the 13 scenes are in the learned model's train split** (gtIGgs, vcRBT,
  2nwG, 3DAg, 9Rvr, PfwFWSoPF — it trained on those frames with pipeline
  pseudo-labels). On the 7 scenes it never saw, the readings converge:
  optim+model .786, optimization .766, optim+forces .773, learned .788 — near-ties at or below
  that subset's always-contact floor of .795. On raw limb-level F1, nothing
  currently beats "always touching" on the hardest unseen scenes; the methods
  earn their keep on precision (~.80–.84 vs the floor's .66), not F1.
- **Per limb the chains split**: dev beats main on feet (right foot .824 vs
  .789) and loses on hands (~.85 vs ~.89). The force witness helps feet; hands
  are where the model fusion pays.

Note the limb fold also hides most of what the dev chain changed — toe
side, per-frame placement, force consistency are invisible once everything
collapses to "foot touched"; the joint-level judge on the 13 scenes
(contacts_2 F1 .771) remains the sharper instrument for those.
