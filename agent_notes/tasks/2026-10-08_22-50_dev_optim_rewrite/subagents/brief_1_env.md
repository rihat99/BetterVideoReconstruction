Read brief_common.md, then inv_B_bh_env.md §3–4 (Option A commands) and inv_A_br_optim.md §4 (API breaks to expect).

# Step 1 — dev environment, Option A (you MAY install into the dev worktree's .venv only; never touch the main checkout's .venv; no git commits; never rm)

Worktree: /home/rikhat.akizhanov/better/dev/BetterVideoReconstruction. Follow inv_B Option A exactly, in this order, verifying each step:
1. `git submodule update --init` (≈250 MB; report time). Symlink MHR assets from the main checkout as in the recipe. Check `third_party/` matches main's submodule commits (`git submodule status` in both).
2. `uv sync --frozen` into the worktree's .venv (uv-managed 3.12, torch 2.8.0+cu129). Then `uv pip install --python .venv/bin/python -e ../BetterRobot -e ../BetterHuman --no-deps` (../ is /home/rikhat.akizhanov/better/dev). detectron2 + sam-3d-body.pth as in the recipe (CPU build, may take minutes).
3. Verify: `.venv/bin/python -c "import better_robot, better_human, torch; print(...)"` → paths under /home/rikhat.akizhanov/better/dev/ and torch 2.8.0+cu129; `torch.cuda.is_available()` with CUDA_VISIBLE_DEVICES=2; `import viser`; `import mhr`; `import pymomentum`; `import detectron2`; `import sam3` (whatever the adapters import — read tools/adapters/*.py and scripts/stages/*.py imports and try each).
4. Run `CUDA_VISIBLE_DEVICES=2 BVR_DATA_ROOT=/home/rikhat.akizhanov/better/data uv run --no-sync pytest tests/ -q -p no:cacheprovider -x --maxfail=200` (full suite; expect failures where BR/BH APIs changed). Produce the failure list grouped by root cause (ImportError symbol X, AttributeError ..., etc.) with file:line — this is the migration work list for step 2. Do not fix anything.
5. Write a 15-line `agent_notes/docs/dev_env.md` in the worktree: how the dev env was built, the env vars (BVR_DATA_ROOT), the two commands to re-create it, and the known torch 2.8 vs 2.13 note from inv_B.
Report → subagents/report_1_env.md (or text): each step's result, versions, the failure list, time taken. Under 100 lines.
