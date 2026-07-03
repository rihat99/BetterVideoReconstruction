# Installation

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- CUDA-compatible GPU with CUDA 12.6+
- Sibling `BetterRobot/` and `better_human/` checkouts next to this repo
  (the editable installs in `pyproject.toml` point at `../BetterRobot` and
  `../better_human`).

## 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/rihat99/BetterVideoReconstruction.git
cd BetterVideoReconstruction
```

If already cloned without submodules:

```bash
git submodule update --init --recursive
```

The six submodules under `third_party/` are `sam3`, `sam-3d-body`,
`sapiens2`, `MHR`, `vggt-omega`, and `geocalib` (no `sam-3d-objects` — this
repo is human-only).

## 2. Create venv and install dependencies

```bash
uv venv --python 3.12
uv sync
```

This installs:

- `better-robot` and `better-human` (editable, from sibling directories)
- `sam3`, `mhr`, `sapiens`, and `vggt-omega` (editable, from `third_party/`)
- `pymomentum-gpu` (built against torch cu129 — the uv lock pins the matching torch build)
- All `sam-3d-body` dependencies (listed in `pyproject.toml`)

Unlike the sibling repos, **VGGT-Omega is an editable submodule** here
(not a git-pip dep), so `uv sync` installs it directly — no extra git step.

### MHR model assets

`mhr` loads FBX + model assets from `third_party/MHR/assets/` at runtime.
Download and unpack them once:

```bash
cd third_party/MHR
curl -OL https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip
unzip assets.zip
cd ../..
```

### pymomentum-gpu libstdc++ preload

PyPI `pymomentum-gpu` wheels are built against GCC 14 (`GLIBCXX_3.4.33`+) and
ship their own libstdc++ as `libstdc++-pymomentum.so.6`. On systems with an
older system libstdc++ (e.g. Ubuntu 22.04 GCC 12, which only exports up to
`GLIBCXX_3.4.30`), mixing the two at runtime segfaults inside
`Character.with_blend_shape` (see [momentum#939](https://github.com/facebookresearch/momentum/issues/939)).

Every pymomentum-using script in `scripts/` imports `_pym_preload` first, which
re-execs the interpreter with `LD_PRELOAD` pointed at the bundled libstdc++. If
you invoke pymomentum outside these scripts, set it yourself:

```bash
export LD_PRELOAD=$(python -c "import sysconfig,pathlib; print(pathlib.Path(sysconfig.get_paths()['purelib'])/'pymomentum_gpu.libs'/'libstdc++-pymomentum.so.6')")
```

## 3. Install detectron2 (git-only dep)

`detectron2` lives outside the uv lock (git pin, `--no-deps`) because it has
heavy build requirements and only needs to be importable for `sam-3d-body`:

```bash
uv pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
```

If `VIRTUAL_ENV` points elsewhere, prefix with `VIRTUAL_ENV=$(pwd)/.venv`.

## 4. Add sam-3d-body to the Python path

`sam-3d-body` has no `pyproject.toml`, so we add it via a `.pth` file:

```bash
SITE=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
echo "$(pwd)/third_party/sam-3d-body" > "$SITE/sam-3d-body.pth"
```

## 5. Verify

```bash
.venv/bin/python -c "
import better_robot, better_human, sam3, sam_3d_body, detectron2, vggt_omega
print('All imports OK')
"
```

Note: use `.venv/bin/python` directly (not `uv run`) to avoid re-sync, which
drops `detectron2` (it is not in the lock file).

## Sapiens2 checkpoints

`scripts/stages/estimate_sapiens.py` runs Meta's Sapiens2-5B for pose /
pointmap / normal. The 5B `.safetensors` weights (~10 GB per task) auto-download
from HuggingFace into the standard HF cache on first use:

- `facebook/sapiens2-pose-5b`     → `sapiens2_5b_pose.safetensors`
- `facebook/sapiens2-pointmap-5b` → `sapiens2_5b_pointmap.safetensors`
- `facebook/sapiens2-normal-5b`   → `sapiens2_5b_normal.safetensors`

Pre-download into a fixed location and pass `--checkpoint-root <root>` (or set
`[sapiens] checkpoint_root` in the pipeline config) to avoid the HF cache; the
layout is `<root>/<task>/sapiens2_5b_<task>.safetensors`.

## VGGT-Omega checkpoints

`scripts/stages/estimate_camera_vggt.py` runs VGGT-Omega multi-view camera +
depth (active; in the default pipeline right after `sam3`). Its checkpoints are
HuggingFace **access-gated**: request access at
<https://huggingface.co/facebook/VGGT-Omega>, then either

- download `vggt_omega_1b_512.pt` and point the stage at it —
  `--checkpoint /path/to/vggt_omega_1b_512.pt` (or set `[vggt] checkpoint` in
  `configs/pipeline.toml`); or
- authenticate (`huggingface-cli login`) and let the stage pull the default
  weight from `facebook/VGGT-Omega` on first use.

VGGT-Omega is multi-view — one forward over the whole clip — so peak GPU memory
grows with frame count (~43 GB for 500 frames @ 512 on an A100, per the upstream
benchmark). The stage caps at `--max-frames` (default 500, uniformly
subsampled); lower it or `--image-resolution` if you hit OOM. The
256-resolution text-aligned checkpoint pairs with `--image-resolution 256` (and
`VGGTOmega(enable_alignment=True)`). See `tools/adapters/vggt_omega.py`.

## Notes

- **numpy override:** `sam3` and `vggt-omega` pin `numpy<2` but `better-robot`
  requires `numpy>=2.0`. The override in `pyproject.toml` forces `numpy>=2.0`
  (works fine with all three).
- **xtcocotools build:** Needs numpy/cython at build time; configured via
  `[tool.uv.extra-build-dependencies]` in `pyproject.toml`.
- **detectron2** is installed outside the uv lock (git pin, `--no-deps`)
  because it has heavy build requirements and only needs to be importable for
  `sam-3d-body`.
