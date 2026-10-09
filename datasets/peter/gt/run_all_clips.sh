#!/bin/bash
# The whole climb_wall_3 human GT chain over every clip, sharded over the GPUs given as arguments
# (e.g. `run_all_clips.sh 2 3 5`, a GPU may be listed twice for two shards on it); each shard runs datasets/peter/gt/run_human.py on its own GPU with
# the steps in $STEPS (default: all, prepare .. overlay); steps that already produced their output are skipped by the stages.
#   nohup datasets/peter/gt/run_all_clips.sh 2 3 5 > ../data/peter/climb_wall_3_gt/human/_logs/all_clips.log 2>&1 &
# CLIP_LIST=a_00_01_part1,b_00_01_part2 restricts the run to those clips.
set -u
cd "$(dirname "$0")/../.."
PETER="${BVR_DATA_ROOT:-$PWD/../data}/peter"
[ $# -ge 1 ] || { echo "usage: [STEPS=prepare,sam3,...] $0 <gpu> [gpu ...]"; exit 2; }
STEPS=${STEPS:-prepare,sam3,sapiens,sam3d,smplx,sync,fit,shape,contacts,kin2,kindyn,overlay}
GPUS=("$@")
if [ -n "${CLIP_LIST:-}" ]; then IFS=, read -ra CLIPS <<< "$CLIP_LIST"      # a comma-separated subset
else mapfile -t CLIPS < <(ls -d "$PETER"/climb_wall_3/*/ | xargs -n1 basename | sort); fi
echo "[all_clips] ${#CLIPS[@]} clips over GPUs ${GPUS[*]}"
for i in "${!GPUS[@]}"; do
  shard=()
  for ((j=i; j<${#CLIPS[@]}; j+=${#GPUS[@]})); do shard+=("${CLIPS[$j]}"); done
  list=$(IFS=,; echo "${shard[*]}")
  echo "[all_clips] shard $i on GPU ${GPUS[$i]}: ${#shard[@]} clips"
  uv run --no-sync python datasets/peter/gt/run_human.py --gpu "${GPUS[$i]}" --clips "$list" --steps "$STEPS" \
    > "$PETER/climb_wall_3_gt/human/_logs/all_clips_shard${i}_gpu${GPUS[$i]}.log" 2>&1 &
done
wait
echo "[all_clips] every shard finished"
