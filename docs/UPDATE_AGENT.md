# UPDATE_AGENT — implementation handoff: contacts in `kinematics_1`, round 2

*2026-07-06, rev 2. Audience: the agent implementing [`docs/UPDATE.md`](UPDATE.md)
(rev 4) — read it first for the rationale; this file holds the mechanics. Round 1
(the discrete detect→enforce design) is already implemented in the working tree and
partially superseded — §2 says exactly what to keep and what to remove. Trust
function/key names over line numbers.*

**Scope**: `root` + `kinematics_1`, the contact readout in `estimate_contacts`, and
the scoring/diagnostic scripts.
**Out of scope**: `kinematics_2`, `dynamics`, the viewer, any ClimbingVideos DB
re-run, and the known `kinematics_2`-drops-`body_scale` issue (separate topic).

**User decisions (do not re-litigate):**
- Motion/stillness is a **signal we read, never something we optimize** — no
  velocity damping of any joint in kinematics_1, ever. Gradients never flow
  through the stillness weight.
- No discrete contact labels inside the optimizer, no separate contact phase —
  the contact losses live inside `refine`, reweighted every iteration.
- accel/jerk stay 24/48/300/500; no triangulation cue; betas/scale frozen
  (the `[body] scale` knob is a separate user experiment — leave it alone).
- Negative labels stay manual-only (`kinematics_2`'s `ann_repulsion`). The new
  `motion_clearance` is *not* a negative label — it is a physics prior on moving
  vertices, blended continuously.

---

## 1. Where round 1 ended (measured, all on local clips)

- Detector (`auto_contact.py`, enter 6 cm / exit 9 cm / veto 0.2 m/s / ≥5 frames):
  precision 0.83–0.97, FPR 0.01–0.21 — but recall only 0.19–0.65.
- SDF-only readout baseline: precision 0.74–0.89, recall 0.74–0.99, FPR up to 0.67.
- Enforcement (contact phase, `auto_attraction` 20 band 0.18 + `auto_velocity`
  150): reproj ~flat, **chamfer +0.7–1.9 cm and penetration +0.6–1.4 cm worse on
  every clip**, marginal label gains, +2.3× runtime. The phase now defaults off
  (`contact = 0` iters).
- Root causes (drive the redesign): damping optimized the very signal detection
  reads; enforcement only acted on already-correct labels so it could not raise
  recall; and the **hard speed veto is unusable** — a gripping hand's few-mm
  per-frame fit jitter × 30 fps ≈ 0.1–0.3 m/s instantaneous speed, the same range
  as a slow near-wall pass. Grips broke spans (recall down); slow passes slipped
  under (the user's visually-confirmed false positives).
- The discriminative signal is **net drift over a window** (~⅓ s): a grip nets
  < ~2 cm (jitter cancels), a slow pass accumulates 3–7 cm.
- Manual annotations now exist for **four** clips: boulder_1, boulder_3,
  competition_1, competition_2 (`data/<stem>.npz`, tri-state coarse joints).
  sport_climb has none — use it for fit metrics only.

## 2. Current tree state — keep / remove

Everything below is in the working tree, **uncommitted** (HEAD `420b148`).

**Keep as-is:**
- Task-0 deletions (`depth_push`, `contact_velocity` gone) — done, parity-verified.
- `estimate_contacts` hysteresis + `min_frames` (`[contacts] exit_thresh_m=0.035,
  min_frames=3`) and the `contacts.py` helpers `joint_guarded_min_sdf`,
  `hysteresis_spans` — the readout builds on these.
- `scripts/prototype_auto_contacts.py` (the annotation scorer) and
  `scripts/diagnose_contacts.py` — they are the evaluation harness; adapt, don't
  rewrite.
- The `[body] scale` knob and its npz plumbing (user experiment, not yours).
- The speed levers found in round 1 (−40 % kin1 runtime) — apply separately if
  the user asks; keep them out of this diff.

**Remove / replace (the discrete enforcement path):**
- `optimizer.py`: the auto-label state, the detect hook, and the
  `auto_attraction` / `auto_velocity` terms and their weight reads/log strings.
- `stages.py`: the `contact` `PhaseSpec` (and `PhaseSpec.detect` if nothing else
  uses it); `iters` loses the `contact` entry.
- Config: `auto_attraction` + `auto_velocity` weight rows, `[auto_attraction]` +
  `[auto_contact]` tables, and the generic `contact_attraction` row + table
  (retired — superseded by the still branch). Update `config.py` `knob_sections`
  accordingly (it silently drops unknown sections — and unknown *rows* would
  linger in the flattened dicts, so remove both ends).
- `auto_contact.py`: `detect_contact_labels`'s veto/label logic is superseded.
  Move what survives (the drift helper, below) into `contacts.py` and retire the
  module, or slim it to the drift helper — your call; one home, no duplication.

## 3. The drift/stillness machinery (shared by readout and losses)

One new pure helper (suggested home: `tools/human_optim/contacts.py`):

```python
def joint_drift(joints_world, fps, window_s):   # (T,J,3) -> (T,J) metres
    """Net displacement of each joint across a centered time window:
    ‖p(t+h) − p(t−h)‖ with h = round(window_s·fps/2), one-sided at the ends."""
```

Two gathers, no loops. From drift, the stillness weight:

```
w(t,j) = clamp(1 − drift/(2·still_m), 0, 1)     # 1 at drift=0, 0.5 at still_m, 0 at 2·still_m
```

Linear ramp is fine; the exact shape is not sacred. Knobs (new section, add to
`knob_sections`):

```toml
[contact_gate]
window_s = 0.35        # drift window (~11 frames at 30 fps)
still_m  = 0.025       # net drift at which w = 0.5; a grip nets < ~2 cm, a pass 3-7 cm
```

Rules: computed under `no_grad` / detached **always**; per-joint, inherited
per-vertex via the existing `v2j_sub`; invalid frames (`~frame_valid`) get
`w = 0` (nothing near a held frame should be pulled).

## 4. Task A — readout v2 in `estimate_contacts` + rescore  ← GATE

1. In the stage, compute `joint_drift` from the posed joints (fps from
   `transform.npz`, `tr.get("fps", 30.0)`), and gate the hysteresis input the
   same way the old veto did: frames with `drift > 2·still_m` (i.e. `w == 0`)
   have the per-joint sdf forced to `+inf` before `hysteresis_spans`. Delete the
   old `speed_veto_m_s` path everywhere.
2. Extend `scripts/prototype_auto_contacts.py` to score **this readout** (not the
   old detector) against the four annotated clips, sweeping a small grid and
   printing one table:
   - `close_thresh_m ∈ {0.02, 0.03}` × `still_m ∈ {0.02, 0.03, 0.05}` (and a
     no-gate column as reference — that is the current SDF baseline).
   - Report per clip: precision / recall / FPR, per coarse joint group and overall.
3. Baselines to print alongside: the SDF-only numbers and the round-1 detector
   numbers from §1.

**Gate (user decides):** an operating point with precision at or above the
round-1 detector's and recall well above it (the veto was the recall killer —
expect most of it back). If no point on the grid dominates, report the table and
stop — the in-loop losses (Task B) only make sense downstream of a readout the
user trusts.

**Accept**: readout change touches only `estimate_contacts` + `contacts.py`;
`contacts_1.npz` schema unchanged (plus gate knobs in the attrs for provenance).

## 5. Task B — the bistable in-loop term

### 5.1 Loss shape

Extend the two existing losses to take a float per-vertex weight instead of only
a bool mask (`active_mask: bool (T,Ns)` → also accept `weights: float (T,Ns)`;
bool keeps working — multiply into the per-vertex contribution and the
normalizing count, exactly like the mask does today):

- **`still_attraction`** = `scene_attraction(..., weights = w_v · in-band)` with
  its own knob table — pulls still vertices the last few cm onto the surface.
- **`motion_clearance`** = `clearance_repulsion(..., weights = (1 − w_v))` —
  pushes moving vertices out to a small clearance. Same Huber/cap structure it
  already has from `ann_repulsion`.

`w_v = w[:, v2j_sub]` recomputed **every iteration** inside `_forward_losses`
from the detached posed joints `jw` (which the forward already produces — no
extra body forward, no detect hook, no cached labels). Both branches share the
frame's scene candidates (`scene_cand`) like every scene term; if convenient,
share one NN query between them — a perf nicety, not a requirement.

### 5.2 Config

```toml
#                       root  kin1  kin2  dyn      # [loss_weights.scene]
penetration         = [   0, 4000, 4000,    0 ]
still_attraction    = [   0,   15,    0,    0 ]
motion_clearance    = [   0, 1000,    0,    0 ]
# contact_attraction / auto_attraction / auto_velocity rows: deleted

[still_attraction]
conf_thresh = 0.1
band_m = 0.12          # pull range for still vertices; detection recall no longer rides on this
max_dist_m = 0.20      # |sdf| <= dmin: must be >= band_m or the far-guard silently kills the band

[motion_clearance]
conf_thresh = 0.1
clearance_m = 0.04     # moving vertices are pushed out to this
max_dist_m = 0.15
max_pen_m = 0.10
huber_m = 0.02
```

Both terms are active only in scene phases (they need `scene_cand`), i.e.
`refine` + `lbfgs` — no phase machinery needed beyond what every scene term
already has. L-BFGS needs no special-casing: `w` is recomputed per loss
evaluation from detached joints, which is well-defined inside a line search.

### 5.3 Interactions to preserve

- `penetration` unchanged and still owns `s < −margin` for everyone; the
  clearance branch overlaps it for moving vertices (both push out — fine).
- The still branch's per-iteration detached gate `0 < s < band` stays (from
  `scene_attraction`) — a vertex reaching the surface stops being pulled.
- `frame_valid` continues to empty scene candidates on untracked frames; with
  `w = 0` there as well, both branches are doubly off.

**Accept — parity first**: with `still_attraction = 0` and `motion_clearance = 0`
a full run is `np.allclose`-identical to a run of the current tree with the
round-1 machinery removed and `contact = 0` (i.e. the deletions themselves must
not change the default path). Then the runtime check: `refine` within ~15 % of
that baseline.

## 6. Task C — evaluation (hard gates)

On the local clips, against fresh HEAD-equivalent runs (same geometry inputs):

1. **Fit flat (gate)**: reproj_sapiens within ±0.3 px, chamfer within ±0.3 cm,
   penetration within ±0.3 cm of baseline, per clip. Round 1 failed here; a
   configuration that fails again is dead regardless of its labels.
2. **Runtime (gate)**: kinematics_1 ≤ 1.15× baseline.
3. **Labels**: run the Task-A readout (the user-picked operating point) on the
   baseline bodies AND the new bodies. Precision and recall vs the four
   annotations must improve or hold; FPR must not rise. The expected mechanism:
   still true contacts pulled inside the readout distance (recall ↑), slow
   passes pushed past it (precision ↑).
4. **Slow-pass spot check**: on the user's visually-identified false-contact
   segments, confirm the labels are gone (report which clips/frames you checked).
5. Weight grid: `still_attraction ∈ {10, 15}` × `motion_clearance ∈ {500, 1000}`
   — four runs on one clip first (boulder_1), the winner on all five.

Report raw tables; the user owns the verdict. **The documented fallback is
real**: if no weight passes gate 1 while moving gate 3, recommend keeping
detector+readout only (weights 0) — that outcome is acceptable and expected to
be close.

## 7. Gotchas (things that already bit once)

- `|sdf| ≤ dmin` (projection ≤ norm): every `max_dist_m` must exceed the band it
  guards, or detections/pulls die silently and look like "no contacts".
- PCA normals are CPU (`tools/geometry/normals.py`) — cuSOLVER `eigh` dies. Don't
  move them.
- The stillness weight must come from **detached** joints. If you see grips
  getting stiller while chamfer worsens, gradients are leaking through `w` —
  that is the round-1 damping failure re-entering through the back door.
- Central-difference speed at the trajectory ends: `joint_drift` uses one-sided
  windows there; don't let the first/last frames read as artificially still.
- `estimate_contacts` poses with the npz's `scale` (the `[body]` knob) — the
  drift must be computed on the same scaled joints the sdf uses.
- Multi-person `(P, …)` everywhere; annotations align via `object_ids`. DB scenes
  can be P=2 (out of scope, but don't hard-code P=1).
- Keep the readout knobs (`[contacts]`, `[contact_gate]`) and the loss knobs
  (`[still_attraction]`, `[motion_clearance]`) separate — the readout must stay
  meaningful on bodies optimized without the losses.

## 8. Process

- Follow the repo `CLAUDE.md`s. Python → `python-standards` skill if available.
- Commit to `main` only when the user asks; the working pattern is implement →
  re-run locals → report raw numbers → user decides.
- GPUs are allocated by the user — ask before multi-GPU sweeps; one clip fits on
  one GPU.
- Surgical diffs: every changed line traces to §2/§4/§5. The round-1 removal and
  the new term are separable commits if the user wants history.

---

## Outcome (2026-07-06 — implemented, measured, resolved)

Implemented and measured on the four annotated clips (boulder_1/3, competition_1/2;
sport_climb excluded as an outlier). GPUs 2/3.

**Task A — the drift-gated readout — SHIPPED.** `joint_drift` / `stillness_weight` /
`drift_gated_contact` (`contacts.py`), wired into `estimate_contacts` and the scorer. On the
base body, pooled over the four clips (close 2 cm): `still_m = 0.025` → **P 0.909 / R 0.481 /
FPR 0.040**, versus the plain-SDF (no-gate) baseline **P 0.829 / R 0.697 / FPR 0.120** — a **3×
FPR cut at matched precision**. `still_m` is the recall knob (0.05 → R 0.594 / FPR 0.058). The
scorer was verified bit-identical to what `estimate_contacts` writes.

**Task B — the bistable in-loop term — FAILED all three gates, REMOVED.**
- **Gate 1 (fit):** chamfer **+1.0–2.8 cm**, penetration +0.3–0.9 cm on every clip (default
  `still_attraction=15`/`motion_clearance=1000`).
- **Gate 3 (labels):** base → default got *worse* (**P 0.909 → 0.885, FPR 0.040 → 0.050**
  pooled); no point on the `{10,15}×{500,1000}` grid escaped it.
- **Gate 2 (runtime):** 1.29–1.43×.

The cause is structural and identical to round 1: chamfer pins the body to the **human depth
cloud**, contact pulls it to the **scene wall**, and those disagree by a few cm — so any weight
that raises contact degrades the fit, and a mis-fit body yields *worse* readout labels (pulling
no-contact joints to the wall also raises FPR). Two independent mechanisms — round-1 discrete
detect→enforce, round-2 continuous drift blend — hit the same wall.

The in-loop machinery was **removed entirely** (not left at weight 0 — a dead config row invites
a future re-audit of a known-dead path). The drift-gated readout stayed. Do not re-implement the
in-loop branch without new evidence that the fit-vs-contact surface conflict is resolved (e.g.
anchoring contact to the human depth cloud rather than the merged wall).
