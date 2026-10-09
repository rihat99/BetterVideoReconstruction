# Installation

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- A CUDA GPU; the lock pins the PyTorch **cu129** wheels (`pymomentum-gpu`'s ABI).
- The sibling checkouts `../BetterRobot` and `../BetterHuman` next to this repo
  (in the dev layout: `/home/rikhat.akizhanov/better/dev/{BetterRobot,BetterHuman,BetterVideoReconstruction}`).
  `pyproject.toml` declares them as editable path dependencies, so `uv sync`
  installs them from there.

## 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/rihat99/BetterVideoReconstruction.git
cd BetterVideoReconstruction
```

If already cloned without submodules: `git submodule update --init --recursive`.

The five submodules under `third_party/` are `sam3`, `sam-3d-body`, `sapiens2`,
`MHR` and `geocalib`; `vggt-omega` sits beside them as a vendored copy (patched
for chunked low-memory inference).

## 2. Build the environment

```bash
uv sync --managed-python
```

This creates `.venv` with a uv-managed CPython 3.12 (`xtcocotools` builds a C
extension, so a system Python without headers fails) and installs everything in
`uv.lock`: `better-robot` and `better-human` editable from `../`, the
`third_party/` packages editable (`sam3`, `mhr`, `sapiens`, `vggt-omega`,
`geocalib`), `pymomentum-gpu` with the matching torch cu129 build, and every
`sam-3d-body` dependency (listed in `pyproject.toml`, since that submodule has no
`pyproject.toml` of its own).

## 3. The manual steps

Two things cannot be declared in the lock; do them once after `uv sync`.

**detectron2** (only needs to be importable for `sam-3d-body`): it is a source
build whose outcome depends on the build machine — with a GPU visible and a CUDA
toolkit on the path it compiles CUDA kernels against whatever `nvcc` it finds —
so it is installed by hand as a CPU build, pinned, without its own dependencies:

```bash
CUDA_VISIBLE_DEVICES="" uv pip install --python .venv/bin/python 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
```

**sam-3d-body** (no `pyproject.toml`): add it to the Python path with a `.pth` file:

```bash
echo "$PWD/third_party/sam-3d-body" > "$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')/sam-3d-body.pth"
```

**MHR model assets**: `mhr` loads FBX + model assets from `third_party/MHR/assets/`:

```bash
cd third_party/MHR
curl -OL https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip
unzip assets.zip
cd ../..
```

(In the dev layout the folder is a symlink to the main checkout's `assets/`.)

### `uv run` vs `uv sync` afterwards

`uv run python …` is safe: it syncs the lock but never removes a package it does
not know, so detectron2 and the `.pth` file survive. A plain `uv sync` is an
*exact* sync and uninstalls detectron2; use `uv sync --inexact`, or repeat the
detectron2 line after it.

## 4. Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `BVR_DATA_ROOT` | where the data lives (`examples/`, `peter/`, the datasets) | `<repo>/../data` (in the dev layout that path does not exist — set it) |
| `BETTERHUMAN_MODELS_DIR` | BetterHuman's licensed body models (`smplx/SMPLX_NEUTRAL.npz`, `smplx_lockedhead/`, …) | `../BetterHuman/models` |
| `BVR_DETERMINISTIC` | `1` makes a GPU run bit-reproducible (deterministic torch kernels, no cuDNN autotune) | off |
| `CUDA_VISIBLE_DEVICES` | the GPU to use; `""` runs on the CPU (tests) | all |

## 5. Verify

```bash
uv run python -c "import better_robot, better_human, bvr, datasets, sam3, sam_3d_body, detectron2, vggt_omega; print(better_robot.__file__)"
CUDA_VISIBLE_DEVICES="" uv run pytest -q
```

The first line must print a path under `../BetterRobot`.

## Model checkpoints

### Sapiens2

`bvr/stages/estimate_sapiens.py` runs Meta's Sapiens2-5B for pose / pointmap /
normal. The 5B `.safetensors` weights (~10 GB per task) auto-download from
HuggingFace into the standard HF cache on first use:

- `facebook/sapiens2-pose-5b`     → `sapiens2_5b_pose.safetensors`
- `facebook/sapiens2-pointmap-5b` → `sapiens2_5b_pointmap.safetensors`
- `facebook/sapiens2-normal-5b`   → `sapiens2_5b_normal.safetensors`

Pre-download into a fixed location and set `[sapiens] checkpoint_root` in
`configs/pipeline.toml` to avoid the HF cache; the layout is
`<root>/<task>/sapiens2_5b_<task>.safetensors`.

### VGGT-Omega

`bvr/stages/estimate_camera_vggt.py` runs VGGT-Omega (multi-view camera + depth,
right after `sam3`). Its checkpoints are HuggingFace **access-gated**: request
access at <https://huggingface.co/facebook/VGGT-Omega>, then either download
`vggt_omega_1b_512.pt` and set `[vggt] checkpoint` in `configs/pipeline.toml`,
or authenticate (`huggingface-cli login`) and let the stage pull the default
weight on first use.

VGGT-Omega is multi-view — one forward over the whole clip — so peak GPU memory
grows with frame count (~43 GB for 500 frames @ 512 on an A100, per the upstream
benchmark). The stage caps at `[vggt] max_frames` (default 500, uniformly
subsampled); lower it or the image resolution if you hit OOM. See
`bvr/adapters/vggt_omega.py`.

## Notes

- **pymomentum-gpu libstdc++ preload:** the PyPI wheels are built against GCC 14
  and ship their own `libstdc++-pymomentum.so.6`; on an older system libstdc++
  (Ubuntu 22.04) mixing the two segfaults inside `Character.with_blend_shape`
  ([momentum#939](https://github.com/facebookresearch/momentum/issues/939)).
  Every pymomentum-using stage in `bvr/stages/` imports `_pym_preload` first,
  which re-execs the interpreter with `LD_PRELOAD` pointed at the bundled
  libstdc++. Outside these stages set it yourself:

  ```bash
  export LD_PRELOAD=$(python -c "import sysconfig,pathlib; print(pathlib.Path(sysconfig.get_paths()['purelib'])/'pymomentum_gpu.libs'/'libstdc++-pymomentum.so.6')")
  ```

  Import `torch` before `pymomentum` / `mhr` (else `ImportError: initialization failed`).
- **numpy override:** `sam3` and `vggt-omega` pin `numpy<2` but `better-robot`
  requires `numpy>=2.0`; `[tool.uv] override-dependencies` forces `numpy>=2.0`
  (works with all three).
- **xtcocotools build:** needs numpy/cython at build time; configured via
  `[tool.uv.extra-build-dependencies]` in `pyproject.toml`.
- **Torch version:** BetterRobot and BetterHuman run on newer torch in their own
  envs; this repo stays on torch 2.8 (`pymomentum-gpu`), which both support.
