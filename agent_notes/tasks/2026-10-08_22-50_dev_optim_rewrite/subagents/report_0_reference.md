# Report 0 — reference runs on MAIN for the regression set (mars_top_opus, 2026-10-09)

Main-side references for all 4 regression clips, twice (passes A and B); all 40 stages rc=0. The main chain is NOT reproducible run to run even with its `torch.manual_seed(0)`: kinematic_1 differs <1 mm mean, growing through kinematics_2 and kindyn_1 to joint positions mean 3–7 mm / max 183 mm and per-frame forces median 0.7–1.3 %BW / max 204 %BW. `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` (wrapper, no repo edit, no slowdown) makes the whole chain bit-identical (checked on 9Rvrp).

## Gate decision (main session, off-hours assumption 2026-10-09 ~01:00 Dubai)
Use the agent's recommended gate: (a) iteration-0 loss terms from the logs must be exact; (b) each dev stage fed main A's inputs, within ×3 of the same-input noise (contact labels exact; kin_2 pelvis mean ≤6 mm, joint mean ≤1.05°; kindyn_1 forces median ≤0.4 %BW, base-wrench-after ≤6.3%; kindyn_1 joint positions mean ≤5 mm, boulder_1 ≤19 mm); (c) whole chain: pooled contact F1 ±0.01 plus the ×3-noise column below for kin_2/kindyn_1. Where ×3 noise exceeds inv_C's caps, ×3 noise wins (inv_C's caps sat below the measured noise). Dev runs use the deterministic wrapper so dev-vs-dev is exact. Measure body joints without fingers as a secondary number.

## 1. Setup
Main BVR HEAD 4e2c152 clean; venv imports main BR (7e2008f) + main BH (793d716); torch 2.8.0+cu129. Scratch root `/home/rikhat.akizhanov/better/data_dev/examples` (1.7 GB): `data/<clip>.{mp4,npz}`; `out/<clip>/{sam3,vggt,sapiens,sam3d,geometry,geocalib}` (boulder_1 639 MB, gtIGgs 432, 3DAg 336, 9Rvrp 193); not copied: moge2/, predictions/, human_optim/. All 4 clips exist; `transform.npz metric=True` (prefit-dual) and `geometry/prefit.npz` present. No input file written during runs. Annotations exist for the scorer; config defaults `labels="auto"`.

## 2. Commands
`artifacts/ref/run_ref.sh <A|B> <clips>` from the main checkout, `CUDA_VISIBLE_DEVICES=2 BVR_DATA_ROOT=/home/rikhat.akizhanov/better/data_dev uv run --no-sync python` per stage (S = data_dev/examples/data/<clip>.mp4, P = data_dev/examples/out/<clip>):
```
scripts/stages/human_optimization.py --source S --predictions P --stages root,kinematics_1
scripts/stages/estimate_contacts.py  --source S --predictions P --bodies kinematic_1   # -> contacts_1
scripts/stages/human_optimization.py --source S --predictions P --stages kinematics_2
scripts/stages/estimate_contacts.py  --source S --predictions P --bodies kinematic_2   # -> contacts_2
scripts/stages/human_optimization.py --source S --predictions P --stages kindyn_1
```
Pass A moved to `P/human_optim_A/`; pass B in `P/human_optim/`. Scorer: `score_frame_contacts.py --predictions data_dev/examples/out --contacts {1,2} --json`. GPU wait 0 min.

## 3. Wall time (s, A/B) and peak VRAM
| clip | kin_1 | cont_1 | kin_2 | cont_2 | kindyn_1 | total | peak MiB |
|---|---|---|---|---|---|---|---|
| boulder_1 (445 fr) | 71/69 | 19/18 | 66/66 | 18/17 | 56/42 | 229/212 | 4194 |
| gtIGgs (471) | 57/56 | 13/17 | 49/49 | 13/19 | 42/47 | 175/188 | 4394 |
| 3DAg (363×2) | 77/76 | 14/14 | 66/68 | 15/15 | 61/63 | 234/236 | 4164 |
| 9Rvrp (104) | 41/43 | 16/18 | 40/41 | 16/15 | 35/34 | 148/152 | 1636 |
Per-stage peaks ≈ 4.2 GB for kinematics/kindyn, 1.1 GB for contacts. A sibling process (≤676 MiB) shared GPU 2 during three stages; timings within 3%.

## 4. Noise floor, A vs B (mean / p99 / max; "joint" = 51 parent-local quaternion angles)
| clip | kin_1 pelvis mm · joint ° | kin_2 pelvis mm · joint ° | kindyn_1 joint pos mm · force %BW (med/p95/max) · base-wrench-after Δ | contacts_1 / contacts_2 agreement (F1 A vs B) |
|---|---|---|---|---|
| boulder_1 | 0.19/1.6/2.0 · 0.01/0.20/1.1 | 2.75/11.3/13.3 · 0.67/2.4/13.7 | 7.20/31.8/183 · 1.11/12.7/50 · 0.5% | .9979 (.996) / .9904 (.982) |
| gtIGgs | 0.30/2.2/3.5 · 0.04/0.36/1.6 | 2.03/8.7/13.4 · 0.54/2.3/9.5 | 3.21/16.6/38 · 0.72/5.2/38 · 1.8% | .9999 (1.000) / .9857 (.976) |
| 3DAg | 0.51/4.4/6.8 · 0.04/0.38/2.5 | 3.36/19.8/27.4 · 0.65/3.6/24.4 | 5.56/35.6/63 · 1.26/8.2/204 · 5.5% | .9994 (.998) / .9941 (.987) |
| 9Rvrp | 0.57/4.3/4.3 · 0.04/0.36/1.2 | 2.98/11.4/11.8 · 0.72/3.4/6.9 | 4.74/18.0/34 · 1.20/11.3/31 · 4.1% | .9907 (.978) / .9920 (.982) |
Maxima come from contact labels flipping between runs (force 0 vs hundreds of N; 3DAg sacrum 0 vs 1226 N) and fingers (boulder 183 mm is left_middle3; 22 body joints max 102 mm, p99 20 mm). Total force per frame differs median 2.5–5.5 %BW, p95 12–26. kinematic_1 auto_labels agree 100%. Root orientation ~0.1° mean kin_1, ≤0.7° kin_2/kindyn.

## 5. Where the noise comes from
- Deterministic mode (`artifacts/ref/det_wrapper.py`): 9Rvrp D1/D2 bit-identical, no warnings; D1 inside the A/B spread (vs A: kindyn joints 4.4 mm mean, forces median 1.2 %BW).
- Same inputs, one stage at a time (I1, I2 fed pass A's inputs): contacts deterministic (labels identical, forces ≤3e-4 N); kinematics_2 pelvis 1.93 mm mean (max 25.6), joint 0.35° mean (max 7.8°); kindyn_1 joints 6.4 mm mean / p99 29 / max 43 on boulder_1 only (pelvis drift), 0.15–1.2 mm on the others; forces median ≤0.14 %BW, max 8.7; base-wrench-after ≤2.1%.
- The chain amplifies: ~0.5 mm in kin_1 → 2–3 mm in kin_2, partly via flipped contact_1 labels.
- `[root] it 0` / `[full] it 0` loss lines (every term, 4 decimals) are identical A vs B → exact wiring check for dev's terms. kindyn_1 prints no per-iteration lines.

## 6. Contact F1 vs manual annotations (A / B)
| | 3DAg | 9Rvrp | boulder_1 | gtIGgs | pooled F1 (P / R) |
|---|---|---|---|---|---|
| contacts_1 | .899/.899 | .940/.947 | .882/.883 | .942/.943 | .912/.912 (.930/.930, .894/.895) |
| contacts_2 | .937/.936 | .965/.957 | .887/.889 | .956/.954 | .932/.931 (.927/.925, .936/.936) |

## 7. Tolerances (worst-clip noise ×3 vs inv_C cap)
| metric (dev vs main A) | noise | ×3 | inv_C | use |
|---|---|---|---|---|
| kin_1 pelvis mean / max | 0.57 / 6.8 mm | 1.7 / 20 | 5 / 30 | 1.7 / 20 mm |
| kin_1 joint angle mean / p99 | 0.044 / 0.38° | 0.13 / 1.2 | – | 0.13° / 1.2° |
| kin_2 pelvis mean / p99 | 3.36 / 19.8 mm | 10 / 60 | 5 / 30 | 10 / 60 (×3 noise) |
| kin_2 joint angle mean / p99 | 0.72 / 3.6° | 2.2 / 11 | – | 2.2° / 11° |
| kindyn_1 joint pos mean / p99 | 7.2 / 35.6 mm | 22 / 107 | 5 / 30 | 22 / 107 (×3 noise) |
| kindyn_1 force median | 1.26 %BW | 3.8 | 2 | 3.8 %BW |
| kindyn_1 base-wrench-after Δ | 5.5% | 16% | 10% | 16% |
| contacts_1 / contacts_2 agreement | .9907 / .9857 | .972 / .957 | .97 | .972 / .957 |
| pooled F1 Δ (contacts 1, 2) | .000 / .001 | .003 | .01 | ±0.01 |

## Open questions
1. (answered above) ×3 noise for the whole chain; gate mainly on same-input per-stage comparisons.
2. Opt-in deterministic switch in dev (one line per stage or env wrapper)? → dev gets an env-var switch (decision for step 3).
3. boulder_1 kindyn_1 shows 4 mm pelvis drift even with identical inputs; keep that clip's tolerance looser.
4. Doc drift: files carry 35 contact frames, CLAUDE.md says 12.

## Paths
- `compare_runs.py` in the task folder: `compare_runs.py rootA rootB --clips a,b [--dir-a human_optim_A] [--dir-b human_optim] [--json f] [--verbose]`; identical trees → "all keys identical"; D1 vs D2 exact.
- `artifacts/ref/`: run_ref.sh, run_det.sh, det_wrapper.py, run_iso.sh; timing.tsv; noise_A_vs_B.{txt,json}; score_{A,B}_contacts_{1,2}.{txt,json}; `<clip>/{A,B}/<stage>.{log,vram}`; det/ (D1/D2 + noise files); iso/ (noise_A_vs_I1.json, noise_I1_vs_I2.json).
- References: `/home/rikhat.akizhanov/better/data_dev/examples/out/<clip>/human_optim_A/` (A) and `.../human_optim/` (B); extra runs `data_dev/det/D{1,2}/`, `data_dev/iso/I{1,2}/`.
