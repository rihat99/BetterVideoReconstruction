#!/bin/bash
# Run one published model over the climb_wall_3_gt clips, on the farm that
# scripts/peter_gt/prepare_baselines.py builds. Each model lives in its own conda
# environment, so the caller picks the model and the GPU and this sorts the rest out.
#
#   run_baselines.sh <deco|bstro|gvhmr|prompthmr|interactvlm> <gpu> [extra args...]
#
# Predictions land in peter/climb_wall_3_gt/others/<model>/<clip>/.
set -euo pipefail

MODEL="$1"; GPU="$2"; shift 2
REPO=/data3/rikhat.akizhanov/better/BetterVideoReconstruction
OTHER=/data3/rikhat.akizhanov/better/other
FARM="$REPO/peter/climb_wall_3_gt/others/_farm"
OUT="$REPO/peter/climb_wall_3_gt/others"

source /data3/rikhat.akizhanov/miniconda3/etc/profile.d/conda.sh
export CUDA_VISIBLE_DEVICES="$GPU"

case "$MODEL" in
  deco|bstro)
    conda activate "$MODEL"; cd "$OTHER/$MODEL"
    exec python run_climbing_testset.py --data-root "$FARM" --out-root "$OUT/$MODEL" \
         --skip-existing --sanity "$@" ;;
  gvhmr)
    conda activate gvhmr; cd "$OTHER/GVHMR"
    exec python tools/demo/run_climbing_testset.py --data-root "$FARM" --out "$OUT/gvhmr" \
         --check --skip-existing "$@" ;;
  prompthmr)
    conda activate prompthmr; cd "$OTHER/PromptHMR"
    exec python scripts/run_climbing_testset.py --data-root "$FARM" --out-root "$OUT/prompthmr" \
         --skip-existing "$@" ;;
  interactvlm)
    conda activate interactvlm; cd "$OTHER/InteractVLM"
    exec python -u run_climbing_testset.py --data_root "$FARM" --results_root "$OUT" \
         --contact_classes wall scene "$@" ;;
  *) echo "unknown model: $MODEL" >&2; exit 2 ;;
esac
