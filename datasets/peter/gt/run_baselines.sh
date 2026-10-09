#!/bin/bash
# Run one published model over the climb_wall_3_gt clips, on the farm that
# datasets/methods/baseline_farm.py --dataset climb_wall_3 builds. Each model lives in its own conda
# environment, so the caller picks the model and the GPU and this sorts the rest out.
#
#   run_baselines.sh <deco|bstro|gvhmr|prompthmr|interactvlm> <gpu> [extra args...]
#
# Predictions land in ${BVR_DATA_ROOT:-../data}/peter/climb_wall_3_gt/others/<model>/<clip>/.
set -euo pipefail

MODEL="$1"; GPU="$2"; shift 2
REPO=/home/rikhat.akizhanov/better/BetterVideoReconstruction
OTHER=/home/rikhat.akizhanov/better/other
PETER="${BVR_DATA_ROOT:-$REPO/../data}/peter"
FARM="$PETER/climb_wall_3_gt/others/_farm"
OUT="$PETER/climb_wall_3_gt/others"

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
