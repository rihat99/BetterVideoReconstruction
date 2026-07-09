# UPDATE_AGENT — round 10: seat the flying bodies (clean → diagnose → release + re-anchor)

## Outcome (2026-07-09) — SHIPPED

Wall-contact seating of floating bodies, adopted as default. Raw metrics + renders in
`out/_round10/results.md`.

- **T1** — stripped round-9 null knobs (commit `19954b3`), parity bit-exact on GPU 3
  (canonical `out/`/`features/` reproduce bit-exact on GPU 3, ~1e-3 on GPU 0). The
  0.645-vs-0.678 recall gap was purely the `--gate-source final`-vs-`pre` scorer flag.
- **T2** — the disease is a **rigid along-ray float**: the body sits ON its human cloud
  (\|δ\|=0.6–3 cm, cloud fits its limb shape to ~6 cm) while the whole cloud+body float
  20–60 cm *toward the camera* in front of a **credible wall** (verified). The plan's
  cause letters were **swapped** vs `autopsy.csv` (`b_detached` = the cloud-off-wall
  target; `a_off_cloud` = the guard). competition_2 is **not** a floater.
- **T3/T4** — built two levers + a rigid **seat** phase. The winner is **wall-contact
  pull + seat phase + a per-body float gate** (`[wall_contact].float_gate_m=0.15`):
  5 gross floaters seated 24–51 cm → **1.3–5.8 cm** (98–100 % touching); healthy +
  labeled clips left **bit-exact baseline** (contact guards P/FPR **±0.000**). The
  **depth-release** lever (T3-A) added nothing on these floaters and was **stripped**.
  Defaults set to the winner (`wall_contact=[0,200,0,0]`, `iters.seat=150`). The
  361-scene corpus re-run was deferred (user's call).

Plan that produced this outcome follows.

---


*2026-07-09, rev 10. You are the implementing agent. Read `docs/UPDATE.md`
rev 12 first — it is the contract. Rounds 1–9: git history (round 9 = an
honest null; its outcome table is in this file's git history). Trust
function/key names over line numbers: the planning session's anchors date
from `cc92cdb` and round 9 edited these files.*

## §0 The goal, in the user's words — read twice

> "We are solving the flying body — VGGT-Omega human depth being stretched.
> Borders are less important than overall depth position. We should trust
> depth less, but still get some info from it, like limbs location."

Operationalized: VGGT human depth is demoted to **limb-arrangement
evidence** (relative depth structure); **absolute placement** comes from the
scene cloud + the domain fact that a climber on the wall touches it.
Deliverable = flagged floating bodies visibly seated on the wall, measured
by seating metrics and confirmed by renders, with the labeled-clip contact
metrics holding as **guards**. A +0.01 pooled-F1 result is NOT success this
round. A visibly-seated floater set with guards intact IS — even if pooled
F1 barely moves.

**Frozen, untouchable:** annotations `data/*.npz`; the contact-confidence
formula + `confidence_schema=1`; `estimate_contacts` readout knobs;
`kinematics_2` + `dynamics` (their config columns stay 0 for new terms);
canonical corpus `features/` trees (all experimental corpus runs REDIRECTED
via `--out-root`; round 9 was told to add it to the geometry/human-optim
drivers — verify it exists, add if not).

## §1 T1 — cleanup (first commit; everything recoverable via git history)

Run `git status` first; identify and preserve anything user-edited that is
not round-9 debris (check whether `scripts/render_vertex_weights.py` was
already committed in round 9).

Remove **forward** (one commit, "round 10 T1: strip round-9 null knobs" —
do not `git revert`):

1. Discard the uncommitted T2a corridor diff (working tree).
2. Delete the round-9 opt-in knob code + config keys + CLI flags, restoring
   pre-round-9 code exactly in behavior: `[chamfer].robust_scale_m`,
   `robust_from_phase`, `cloud_edge_ref_px`, `cloud_use_conf` (losses/
   stages/config paths); `estimate_scale --vertex-weighting` / `--edge-rtol`
   (restore the hardcoded extremity-up call); fuse's per-point
   `edge_dist_px` write and any reader.
3. Delete `out/_round9/` and any redirected corpus scratch from rounds 9
   (list contents in the commit message before deleting).
4. KEEP: `scripts/diagnose_depth_streaks.py` (commit `5cdc052`) — the
   instrument survives, its contact-F1 go/no-go yardstick is retired.
   KEEP `score_contacts.py` (guards).
5. Parity proof: after the strip, re-run fuse + human_optim + contacts on
   one local clip and confirm bit-exact (defaults were all off/parity, so
   the strip must be behavior-neutral).
6. Time-boxed (≤45 min): the baseline-recall discrepancy (re-derived R
   0.645 vs the round-4 record 0.678 — likely a scorer cell/flag mismatch,
   e.g. merge/target-smoothing settings). Explain it or record it as
   unresolved in `out/_round10/results.md` and move on. Do not rabbit-hole.
7. If round 9 didn't: purge the stale "depth-push" mention from CLAUDE.md.

## §2 Verified code anchors (re-verify before building; names > line numbers)

- **Chamfer**: `chamfer_loss` in `tools/human_optim/losses.py` —
  bidirectional raw L2 over `torch.cdist`, per-frame chunked;
  `vertex_weights` on the v2d side only. Camera access for rays: the
  optimizer already holds per-frame cameras for reprojection/visibility
  (`optimizer.py`, world→cam transforms; `geometry/transform.npz` has
  orig-res intrinsics + extrinsics).
- **Cloud loading**: `_person_depth_points` in `tools/human_optim/stages.py`
  (per-frame conf quantile 0.3, `max_points` 1500); invalid/untracked frames
  emptied via the `frame_valid` machinery (ISSUES round 2, Issue 5).
- **Scene terms**: `scene_signed_distance` / penetration / attraction in
  `losses.py`; shared per-frame scene neighborhood via `_build_scene_cand`
  (AABB-pruned candidates, built once from the seated body, refine phase).
  Scene cloud = `geometry/scene.npz` (excludes dilated human); normals =
  `tools/geometry/normals.py` (PCA, camera-oriented).
- **Config**: `configs/human_optimization.toml`, 4-column weight arrays
  `[root, kinematics_1, kinematics_2, dynamics]` in `[loss_weights.*]`
  tables; a term with column 0 is not built. New scene-term weights go in
  `[loss_weights.scene]`.
- **Seating numbers already exist**: `contacts_<tag>.npz` carries
  `joint_sdf_cm (P,N,J)` (unguarded per-joint min signed distance to scene)
  + `body_health (P,)`. The coarse-joint map `_ANN_TO_SMPLX` (stages.py)
  names the 10 annotated groups; hands/feet EE subsets derive from it.
- **The target population**: `features/audit/{bodies,autopsy,cloud_gap}.csv`
  (round 7): ~80 flagged bodies / 363; causes: (a) 65% human cloud off-wall,
  body faithfully follows; (b) 22% body off its own cloud; (c) ~10% broken
  upstream; (d) ~2% limbs. `scripts/audit_scenes.py` regenerates flags —
  verify it can point at a redirected tree (add a root flag if needed).
- **Corpus drivers**: `precompute_geometry_scenes.py`,
  `precompute_human_optim_scenes.py`, `precompute_contacts_scenes.py`, all
  `--scenes` comma-list; check `--out-root` coverage (§0).
- **Scorer (guards)**: `scripts/score_contacts.py --predictions <dirs>`;
  pooled excludes sport_climb; re-derive the guard baseline yourself at
  current HEAD and pin the exact cell/flags in `results.md`.

## §3 T2 — diagnose the disease on the bodies that have it (before any loss)

Eval-set selection (commit the list): from `autopsy.csv`, ~15 cause-(a)
scenes + ~5 cause-(b) + 8–10 unflagged healthy controls; plus the 7 showcase
scenes (`PfwFWSoPF_k_0026, 2nwG1Qa5k_c_0115, 9RvrpYoWDoo_0000,
CRquJ8H1QLI_0001, 3DAgLpPXVCs_0132, gtIGgs_y2nY_0024, vcRBT9Q9jXM_0029`);
plus the 5 labeled locals as guards. Baseline for corpus scenes is FREE: the
canonical round-8 `features/human_optim/` outputs (round-9 knobs were
parity) — pull baseline seating metrics straight from existing
`contacts_1.npz` `joint_sdf_cm`.

On ~10 cause-(a) scenes + competition_2 + 2 healthy controls, measure per
person, per frame (extend `diagnose_depth_streaks.py` or a sibling script):

1. **δ(t)** — robust median along-ray offset between the fitted body's
   visible vertices and their matched human-cloud points (the exact
   statistic T3-A will subtract; rays from the frame's camera center).
2. **cloud→scene gap** (the round-7 `cloud_gap` statistic) and **body
   d_seat(t)** = min over EE joints of distance to scene.
3. **Residual limb structure**: chamfer residual before vs after removing
   δ(t). If removing one scalar per frame collapses residuals to a few cm,
   the user's "limbs location survives" hypothesis is confirmed and T3-A is
   well-posed. If not, say so — T3-A weakens to a crude chamfer down-weight.
4. **δ stability**: per-clip constant (⇒ a per-clip bias correction at fuse
   becomes a menu option) vs per-frame wander (⇒ per-frame handling in-loop).
5. **Wall credibility spot-check** (eyes): the scene must actually exist
   where the body should sit — it is about to become the depth owner.

Output: `out/_round10/diagnosis.md` + plots; conclusions set T3 knobs.
Commit the summary. If the diagnosis refutes the premises (e.g. no wall in
the scene cloud on most flagged scenes), STOP and report — do not build
losses on a refuted premise.

## §4 T3 — the two levers (built together, tested factorially)

### T3-A: depth-release — offset-tolerant chamfer

In `chamfer_loss` (or a thin wrapper): per frame, per person, compute
δ = weighted median over visible matched pairs of `(p_cloud − v) · r̂`
(r̂ = unit ray from that frame's camera center; detach δ — recomputed each
iteration like `vismask`). Shift the cloud by `−λ·δ·r̂` before both chamfer
directions. Knob `[chamfer].depth_release = λ ∈ [0,1]`, default 0 = parity
(bit-exact). Guard: <50 matched pairs ⇒ δ=0. Log per-frame δ (free
diagnostics). Note for the report: at λ=1 chamfer is invariant to global
along-ray shifts — depth is then owned by T3-B + smoothness, by design.

### T3-B: re-anchor — wall-contact prior

New term `wall_contact` (weight column in `[loss_weights.scene]`, default
[0,0,0,0]): per valid frame, per person, take the 4 EE joints (L/R hand,
L/R foot; joint centres are already computed in the optimizer), distance to
nearest scene point via the shared scene-neighborhood machinery; combine
with a softmin (logsumexp, temperature ~5–10 cm) → `s_min(t)`; loss =
saturating penalty `gm(max(0, s_min − margin))` with margin ~5 cm (joint
centre sits inside the flesh) and GM scale ~0.3–0.5 m (airborne frames cost
a bounded amount). Phase: refine (where scene terms live) — and run one arm
with it active in `full` too; a badly flying body may need the pull before
refine. Skip untracked frames (`frame_valid`). Log per clip: fraction of
frames with `s_min > 0.5 m` after optimization (unfixable) and frames where
the term moved the root > 20 cm (eyeball those renders).

### The matrix (locals + eval scenes, redirected; one results table)

| arm | depth_release λ | wall_contact | chamfer weight |
|---|---|---|---|
| P (parity) | 0 | off | 10 |
| A | 1 | off | 10 |
| B | 0 | on | 10 |
| A+B | 1 | on | 10 |
| C+B (crude control) | 0 | on | 2.5 |

(Add a λ=0.5 arm only if the diagnosis argues for it.) A alone is expected
to drift (no depth owner) — it is in the matrix to prove that, not to win.
**The fancy winner must beat C+B meaningfully or C+B ships instead** —
simplicity rule.

### Gates (adopt only if all hold)

- **Flagged eval scenes**: median d_seat drops to touching range (calibrate
  "touching" on the healthy controls first — likely ≤ 5–10 cm) on ≥ half of
  the cause-(a) scenes; audit flag count on the eval set drops; cause-(b)
  scenes at minimum do not worsen.
- **Healthy controls**: seating metrics within noise; zero new flags.
- **Labeled guards**: precision ≥ baseline − 0.005, FPR ≤ baseline + 0.005
  (recall/F1 up is welcome, not required); reproj within +0.3 px.
- **Eyes, mandatory**: before/after renders (showcase + ≥5 flagged scenes +
  competition_2) into `out/_round10/results.md`; the user owns the verdict.
- Standing self-confirmation rule: seating improves under B *by
  construction* — the independent legs are the guards, the post-release
  chamfer residuals, and the renders. Any arm that raises FPR alongside
  seating is doing the wrong thing.

## §5 T4 — wrap

Winner arm → confirmation run over the full eval set; final table + renders
in `results.md`; TOML defaults set to the winner ONLY if gates passed
unambiguously (otherwise present and leave off); sync CLAUDE.md + prepend
the outcome to this file; the 361-scene corpus re-run with the winner
(~7 h human_optim + contacts) is the USER'S decision — ask, don't run.

Menu if A+B underdelivers (do not start without evidence): per-clip
constant depth-bias correction at fuse (only if T2 finds δ clip-constant);
root-depth temporal prior anchored on scene-seated frames.

## §6 Discipline

- Per-task commits on `main` (`Round 10 T1: …`), numbers in messages;
  parity proof before every sweep; raw metrics + renders accumulate in
  `out/_round10/results.md` as you go.
- Skepticism: re-verify every §2 anchor you build on; this file's claims
  are leads, not ground truth. Report nulls plainly — this project ships
  them.
- Never leave experimental output in canonical corpus `features/`;
  redirect, or restore before finishing.
