# BetterVideoReconstruction

An application on BetterRobot and BetterHuman: the code is the plain package `bvr/` at the repository root, entry
points are scripts in `scripts/`, every setting is a key of `configs/bvr.toml`, dataset code is `bvr_datasets/` and
third-party models are in `third_party/`. `README.md` is the user's guide and `docs/` explains each part; this file
holds what the code cannot tell.

## Commands

From the repository root, with `BETTERHUMAN_MODELS_DIR` and `BVR_DATA_ROOT` set (the first trap below). CPU times
measured on the shared 384-thread host with `OMP_NUM_THREADS=16`; GPU times are from the rebuild's runs on one GPU
that another user shared.

```bash
uv sync --managed-python                                  # the environment; 2 s once cached
uv run pytest tests/contract -q                           # edge table, strict config, no absolute path; 1.5 s
CUDA_VISIBLE_DEVICES= uv run pytest tests -q              # every core test, CPU, about 11 s
CUDA_VISIBLE_DEVICES= uv run pytest bvr_datasets -q       # the dataset tests, CPU, 11 s warm, 70 s cold
uvx ruff check --select F,E9 bvr bvr_datasets scripts tests   # unused names and syntax; no ruff config exists

uv run python scripts/pipeline.py --source <video|folder> --save-dir <root> --dry-run   # the step commands, 0.2 s
uv run python scripts/pipeline.py --source <video|folder> --save-dir <root> --steps <a,b,...>
#   all 13 steps on a 104-frame clip: 6.5 min, peak 12 GB (VGGT); per step in docs/pipeline.md
uv run python scripts/pipeline.py --source $BVR_DATA_ROOT/examples/data --save-dir <root> \
    --upstream $BVR_DATA_ROOT/bvr_baseline_2026-11-20/examples --steps kinematics_1,contacts_1,kinematics_2,contacts_2,kindyn_1
#   the five human steps on the 13 example clips: 21 min
uv run python bvr_datasets/examples/score_frame_contacts.py --predictions <root> --contacts 1   # under a second
uv run python scripts/compare_runs.py <root a> <root b> --annotations $BVR_DATA_ROOT/examples/data   # 8 s a clip
uv run python scripts/view_3d.py --source <video|folder> --save-dir <root>                # serves on [viewer].port
uv run python scripts/view_3d.py --source <video> --save-dir <root> --record <file.mp4>    # 67 s, two people, CPU
```

A step alone is `python -m bvr.stages.<module> --source <video|folder> --save-dir <root>`, with `--config`,
`--upstream`, `--skip-existing`, `--clips a,b` and, for a module that serves two steps, `--step`. A run resumes
with `--skip-existing`: a step is skipped for a clip while its manifest, `<clip>/manifests/<step>.json`, matches the
config tables the step reads, the code (commit and diff) and the checksums of its inputs.

## Gates that must not be weakened

Never skip, xfail, loosen a tolerance or delete an assertion to get green; report the number instead.

- The contract tests of `tests/contract/` (below) and every test of `tests/` and `bvr_datasets/`.
- Results of the human steps stay similar. The rebuilt code's numbers on the 13 example clips, run by the third
  command above: pooled annotation F1 0.842 (`contacts_1`) and 0.856 (`contacts_2`); `kindyn_1` support 0.888 body
  weights, root residual median 0.147 body weights, force confidence 0.841. A change is judged with
  `scripts/compare_runs.py` against a run of the code before it on the same upstream, per frame-rate group (30, 24
  to 25, 60 fps), and in body weights for forces.

## Absences, each with its enforcer

- `bvr` imports neither `bvr_datasets` nor a script; stages never import one another (they meet through files; what
  two share is `bvr.fit`); `bvr.run` starts stages by module name; each package imports only its row of `ALLOWED`
  in `tests/contract/test_layer_dependencies.py`, whose `UNPORTED` lists the modules not yet held to it.
- Each third-party model (`sam3`, `sapiens`, `vggt_omega`, `sam_3d_body`, `geocalib`) is imported by one stage
  module (`MODELS`, same file).
- No step reads a config table the base lacks; a delta cannot add a key outside a `weights` table
  (`tests/contract/test_config.py`).
- No tracked `.py`, `.toml`, `.md` or `.sh` names an absolute path (`tests/contract/test_no_absolute_paths.py`).

## Traps

- The data root. `bvr.paths.DATA_ROOT` is `$BVR_DATA_ROOT`, else the folder `data` beside the checkout. From the
  dev worktree (`dev/BetterVideoReconstruction`) that is `dev/data`, from a lane `dev/.lanes/data`; neither exists.
  Set `BVR_DATA_ROOT` to the workspace's `data/` (`../../data` from the dev worktree, `../../../data` from a lane).
  Without it the dataset paths point nowhere and `pytest bvr_datasets` skips four OpenCap tests instead of failing.
- `BETTERHUMAN_MODELS_DIR` has no default. The steps `smplx`, `scale` and every step after `gravity` load SMPL-X
  through it, and `smplx` also `MHR/converted/mhr_lod1.npz`.
- Model outputs move with the environment. SAM 3D Body on torch 2.13 (its cuDNN) differs from the torch 2.8 runs by
  about 1 mm on joints, up to 87 mm at fingertips; accepted. VGGT-Omega reproduces the stored outputs of 11 of the 13
  example clips and of the two wild clips checked (field of view within 0.04 degrees, median depth within 0.1 %).
  `9RvrpYoWDoo_0001` and `PfwFWSoPF_k_0026` differ (0.3 degrees, 3 to 4 %): their videos were re-cut and re-encoded on
  2026-07-03, two days after VGGT ran on the earlier cut, which is gone. Compare changes to the human steps on a fixed
  upstream, never a run from the video against a stored tree.
- VGGT memory. `third_party/vggt-omega` is the authors' code, unchanged: one forward over the whole clip, whose peak
  allocation is 4.3 GB plus 0.076 GB per frame (42 GB at 500 frames, 80 GB at 1000; `mode = "balanced"` keeps about
  1024 tokens a frame at any aspect ratio). A 96 GB card holds about 1150 frames. `nvidia-smi` shows up to 1.6 times
  the allocation while the card has room; that is the allocator's cache, not a need. Clips longer than
  `[vggt].max_frames` (500) are subsampled; no such clip has been run through the fitting steps.
- The in-the-wild clips need a delta the repository does not hold (the largest person only, a camera per frame):

  ```toml
  base = "configs/bvr.toml"
  [sam3]
  top_k = 1                 # 2 for parkour_duo, which needs its own run: SAM 3 takes no per-clip prompt
  min_area_frac = 0.001
  [vggt]
  max_frames = 2100         # a camera per frame; the two clips over 1150 frames do not fit one forward on 96 GB
  [sam3d]
  overlay = false
  ```

- torch 2.13 aborts the process (inductor, CPU) in a compiled MHR call with gradients at batch 8 or more. Nothing in
  BVR compiles today; do not wrap the `smplx` step's MHR forward in `kinematics.compiled` on the CPU.
- BetterRobot's L-BFGS takes a first step of `-g / max(|g|_1, 1)` and only halves it: a variable far from unit scale
  never moves. That is why kindyn_1's force activations are in body weights. Its joint solve is a fixed budget
  (`iterations.joint = 125`), stopped on purpose; its result depends on the count.
- BetterRobot `optim.run` derives every phase from the original problem: data a phase's `on_start` returns lasts one
  phase. Labels go in with `problem.with_(data=)` between two runs. `TorchOptim` refuses a free mask on a manifold
  variable, so `q` is a plain tensor and the cached value `pose` projects it.
- `[run].threads` caps torch and OpenCV, not numpy's BLAS: keep `OMP_NUM_THREADS` set (uncapped runs were 7 times
  slower here). Sapiens' heatmap decode is also slow on this host from glibc returning memory;
  `MALLOC_MMAP_THRESHOLD_=33554432` in the environment before Python starts cut the step from 97 to 84 s.
- Without `[run].deterministic = true` two runs of the human steps differ by about 0.3, 2 and 2.3 mm
  (`kinematic_1`, `kinematic_2`, `kindyn_1`); with it they are identical and 15 to 20 % slower.
- Trees written before the rebuild have no manifests and no `format` key, say `head` for the contact frame
  `head_top`, and their `scale.npz` lacks `smear_median` (the contact steps then measure the smear themselves).
  Read every npz through `bvr.files.load_npz`.
- A clip is named by its video's stem, so datasets with one folder per clip run from a folder of links. A recipe
  that runs a step with its own `load` (`bvr_datasets/peter/chain.py`) still records `python -m bvr.stages.<module>`
  as the manifest's command.
- The `smplx` step reads the MHR to SMPL-X map from the `third_party/MHR` submodule (`bvr.paths.mhr_to_smplx_mapping`).
- `uv sync` re-locks when BetterRobot's or BetterHuman's `pyproject.toml` changed (they are path sources): that lock
  change is theirs, commit it on its own.

## Conventions the code does not announce

- `q (P, T, 211)`: the root pose `[tx, ty, tz, qx, qy, qz, qw]` in the world, then 51 joint quaternions (body joints
  1 to 21, then 30 finger joints); 52 joints, 159 velocity coordinates. People first, time second; every term
  returns one loss per person, `(P,)`.
- Frame `k` of every file is the `k`-th decoded frame. The model steps read the video as JPEG frames of quality 95
  (`bvr.files.extract_frames`), which is part of their results.
- Every time derivative steps over `h = max(1, round(fps / 30))` frames and every weight is per change over 1/30 s
  (`bvr.terms.motion.Time`, `docs/frame_rate.md`); windows are seconds, except `[contact_gate].detect_min_frames`.
- Body mass comes from BetterHuman's default densities (77.4 kg for the neutral mean body); compare forces in body
  weights.
- A stage module defines `load(stage: run.Stage) -> process(clip: paths.Clip)` and ends with `run.main(load)`. It
  reads through `clip.find(paths.X)`, writes through `clip.path(paths.X)` with `files.save_npz` or
  `files.staged_directory`, and logs through `logging`. A table it reads or a file it writes is added to its row of
  `bvr.run.STEPS`: the manifest hashes the tables, and a rerun moves the outputs to `_trash/`.

## Working rules

From the principles repository (chapters 01 to 12; chapter 10 governs this file, chapter 12 the application shape);
no test checks them, a reviewer does.

- Write the least code that solves the problem; no setting, switch or abstraction the task did not ask for.
- No default in code: a new setting is a key of `configs/bvr.toml` with a comment on its meaning and unit.
- Options name what to process, never how; how is the config.
- Update `docs/` and this file in the change that changes what they say.

## Where things live

- The steps, their tables and files: `bvr.run.STEPS`; every output path: `bvr.paths`; file formats and versions:
  `bvr.files.FORMATS`; the keys of each file: the docstring of the stage module that writes it.
- Data under `$BVR_DATA_ROOT`: `examples/` (the 13 annotated clips in `data/`, more in `data_2/`, stored trees in
  `out/`, `out_2/`), `ClimbingVideos/`, `BEDLAM2/`, `BEDLAM2_our/`, `peter/`, `LabValidation_withVideos/`,
  `Parkour-dataset/`, `willd_videos/` (the wild clips). Model weights in the Hugging Face cache; body models in
  `$BETTERHUMAN_MODELS_DIR`.
- `agent_notes/` (one folder per task) and `trash/` are ignored by git; nothing tracked cites them.
