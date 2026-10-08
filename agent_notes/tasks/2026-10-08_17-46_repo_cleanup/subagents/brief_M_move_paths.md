Read brief_common.md in this folder first, then subagents/audit_A_configs_paths.md (section 2 is your work list) and report_P_purge.md (what was already removed).

# Step M — move data out of the repo and fix every path (you MAY edit/move files; do NOT git commit; never rm)

Repo: /home/rikhat.akizhanov/better/BetterVideoReconstruction (BVR). Data root: /home/rikhat.akizhanov/better/data (DATA).

## 1. Move (same filesystem; plain `mv` is a rename — never cp, never rsync)
Before moving: `ps aux | grep -c BetterVideoReconstruction` and `nvidia-smi --query-compute-apps=pid,process_name --format=csv` — if any process has the repo as cwd or an open file under the dirs below, STOP and report. Check `stat -c %d BVR DATA` are equal.
- `mkdir -p DATA/examples`; `mv BVR/data DATA/examples/data`; `mv BVR/data_2 DATA/examples/data_2`; `mv BVR/out DATA/examples/out`; `mv BVR/out_2 DATA/examples/out_2`; `mv BVR/peter DATA/peter` (DATA/peter must not already exist — check first).
- Afterwards `du -sh` each target and confirm the repo root no longer has these dirs.

## 2. One resolver
Create `tools/paths.py` (small, no deps): `REPO_ROOT = Path(__file__).resolve().parents[1]`; `DATA_ROOT = Path(os.environ.get("BVR_DATA_ROOT", REPO_ROOT.parent / "data"))`; `EXAMPLES = DATA_ROOT / "examples"`; `PETER = DATA_ROOT / "peter"`; plus `LABVALIDATION = DATA_ROOT / "LabValidation_withVideos"`, `PARKOUR = DATA_ROOT / "Parkour-dataset"`, `CLIMBING = DATA_ROOT / "ClimbingVideos"`, `ESTMF = DATA_ROOT / "Estimating-3D-Motion-Forces"`. Short module docstring. Do not make it configurable beyond the one env var.

## 3. Fix every site in audit A §2a (and the stale ones in §2c that hard-code `/home/rikhat.akizhanov/better/data/...` or `_REPO.parent / "data"` — switch those to the resolver too, so there is one place). Rules:
- Replace `_REPO_ROOT / "peter"` → `PETER`, `_REPO_ROOT / "data"` → `EXAMPLES / "data"`, `"data_2"` → `EXAMPLES / "data_2"`, `"out"` → `EXAMPLES / "out"`, `"out_2"` → `EXAMPLES / "out_2"`. Remove each file's private `_REPO_ROOT`-style constant only if your change made it unused.
- Scripts that will be REWRITTEN in the next step still get the minimal fix now so nothing is broken in between: scripts/run_peter.py, run_parkour.py, run_opencap.py, run_gt_contact.py, parkour_estmf_inputs.py, parkour_physpt_inputs.py, parkour_prepare_interactvlm.py, scripts/methods/*, scripts/diagnostics/score_baseline_*, score_gt_*, score_prediction_contacts, score_pipeline_contacts, render_prediction_overlays. Minimal = swap the root constant, nothing else.
- Core, allowed exactly as path-default edits and nothing more: tools/human_optim/stages.py:779-782 (fallback dirs → EXAMPLES/"data", EXAMPLES/"data_2") and tools/viewer/loading.py:420 (default video dir → EXAMPLES/"data"). Do not touch anything else under tools/human_optim or scripts/stages; if a fix seems to need it, leave it and report.
- Scripts that import `scripts.X` or run via `uv run python scripts/...` : check how `tools` is importable from scripts (look at how existing scripts import `tools.config`) and follow the same pattern for `tools.paths`.
- benchmark table writers: scripts/diagnostics/compare_climb_wall_3_runs.py:67 and score_combined_table.py:52 → `PETER / "benchmark_tables"` (mkdir as they already do).
- CLI defaults in audit A §2a class (a) rows → the resolver; help strings updated.
- scripts/peter_gt/run_all_clips.sh and run_baselines.sh: derive the peter root from `${BVR_DATA_ROOT:-$REPO/../data}/peter`.
- Dead references already stale before the move (peter/climb_wall, out_climb_wall_single in render_prediction_overlays.py:93-94 and peter_sensor_forces.py default --tree): change the default to an existing tree (climb_wall_3 / out_climb_wall_3_single) and drop the dead RIGS["climb_wall"] entry if it is only a dict literal; report what you did.
- README.md:75,78 example commands: use `../data/examples/data/boulder_1.mp4` and `../data/examples/out/` (check the clip name exists). CLAUDE.md:95 and the doc-only hits in audit A §2b: fix only the ones that state a fixed location (not the generic `out/<stem>/` contract).
- .gitignore: remove the now-meaningless `data/`, `data_2/`, `out/`, `out_2/`, `peter/`, bare `peter`, `scratch/`, the four `configs/corpus_*.toml` lines and the `!tools/robot_motion/...` lines; keep `temp/`, `*.log`, `trash/`, `.claude/`; add `agent_notes/**/vis/` and `agent_notes/**/artifacts/`.

## 4. Symlinks (audit A §2d)
- `find DATA/peter -type l -lname '*BetterVideoReconstruction/peter*'` and `find DATA/Estimating-3D-Motion-Forces/data/climb_wall_3/videos -type l -lname '*BetterVideoReconstruction*'`: rewrite each target by replacing the prefix `/home/rikhat.akizhanov/better/BetterVideoReconstruction/peter` with `/home/rikhat.akizhanov/better/data/peter` (ln -sfn with the new target; keep a count). Expected ~544 + 68. Then `find DATA/peter DATA/examples DATA/Estimating-3D-Motion-Forces -xtype l | wc -l` must be 0 (or list the leftovers, with what they pointed to).
- Also check `DATA/examples/data/body_models/*` (2 links) resolve.

## 5. Sibling repos (edit, do not commit; list exact diffs in the report)
- /home/rikhat.akizhanov/better/contact_anything/scripts: score_rig.py:463-464, fig_climb_wall_3.py:57-58, trivial_baselines.py:49,75-82, prepare_wild.py:115, score_parkour_rig.py:47 — point at DATA/peter and DATA/examples/out_2 respectively. Verify each line by reading it first.
- /home/rikhat.akizhanov/better/BetterContactAnnotator/backend/app/config.py:69 RECON_OUT default → DATA/examples/out.

## 6. Verify (all from the repo root, CUDA_VISIBLE_DEVICES=2)
- `uv run --no-sync pytest tests/ -q -p no:cacheprovider` full suite; report counts and any failure verbatim.
- `grep -rn --include=*.py --include=*.sh --include=*.toml --include=*.md -E '_REPO(_ROOT)? */ *"(peter|data|data_2|out|out_2)"|parents\[[0-9]\] */ *"(peter|data|out)' scripts tools tests docs configs README.md CLAUDE.md INSTALL.md` → must be empty except what you deliberately left (list).
- `uv run --no-sync python <script> --help` for every script you edited that has argparse (list pass/fail).
- `uv run --no-sync python scripts/pipeline.py --source ../data/examples/data/<one clip>.mp4 --save-dir ../data/examples/out/ --skip-existing` with CUDA_VISIBLE_DEVICES=2 — should skip every stage of an already processed clip and exit quickly; paste the tail of its output. If it starts real GPU work for more than a minute, kill it and report.
- `uv run --no-sync python scripts/view_3d.py --help`.

Report: /home/rikhat.akizhanov/better/BetterVideoReconstruction/agent_notes/tasks/2026-10-08_17-46_repo_cleanup/subagents/report_M_move_paths.md — moves done with sizes, resolver API, every edited file (repo + sibling repos) with one line each, symlink counts before/after, verification outputs, anything left or stopped on. Under 150 lines.
