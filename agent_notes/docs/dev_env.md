# Dev environment (dev worktree, built 2026-10-08)
Own `.venv` in `/home/rikhat.akizhanov/better/dev/BetterVideoReconstruction`: uv-managed CPython 3.12.14, same `uv.lock` as main (torch 2.8.0+cu129, pymomentum-gpu 0.1.110), third_party editable from this worktree, BetterRobot/BetterHuman editable from `dev/` (`--no-deps`), detectron2 0.6 CPU build, `sam-3d-body.pth` → `third_party/sam-3d-body`.
One-time setup: `git submodule update --init` (≈230 MB, 22 s); `ln -s /home/rikhat.akizhanov/better/BetterVideoReconstruction/third_party/MHR/assets third_party/MHR/assets` (212 MB, shared with main);
`CUDA_VISIBLE_DEVICES="" uv pip install --python .venv/bin/python 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps`; `echo "$PWD/third_party/sam-3d-body" > .venv/lib/python3.12/site-packages/sam-3d-body.pth`.
Re-create the env (run from the worktree root; the second command must follow every `uv sync`):
```bash
uv sync --frozen
uv pip install --python .venv/bin/python -e ../BetterRobot -e ../BetterHuman --no-deps   # ../ = /home/rikhat.akizhanov/better/dev
```
Env vars: `BVR_DATA_ROOT=/home/rikhat.akizhanov/better/data` (required: the default `<repo>/../data` = `dev/data` does not exist; tools/paths.py:13). GPU runs: `CUDA_VISIBLE_DEVICES=2`.
Always `uv run --no-sync` (or `.venv/bin/python`): a plain `uv run`/`uv sync` drops the hand-installed BR, BH and detectron2.
Tests: `CUDA_VISIBLE_DEVICES=2 BVR_DATA_ROOT=/home/rikhat.akizhanov/better/data uv run --no-sync pytest tests/ -q -p no:cacheprovider`.
pymomentum: import `scripts/_pym_preload` first and `torch` before `pymomentum`/`mhr` (else `ImportError: initialization failed`).
The MHR assets symlink shows as untracked inside the MHR submodule (its `.gitignore` has `assets/`, which matches directories only); harmless.
Torch note: BR/BH dev venvs use torch 2.13.0+cu129; BVR stays on 2.8 for pymomentum-gpu. On 2.8, BH dev tests pass (265 passed, 1 skipped) and BR dev optim/dynamics/kinematics/lie/data_model pass except `test_matrix_fk_jacrev_equals_jacfwd_on_mixed_kinds` (`.tolist()` under jacfwd, `_fk_lean.py:51`; passes on 2.13). Neither BVR nor BR optim uses torch.func, so this does not block (measured by inv_B, not re-run here).
