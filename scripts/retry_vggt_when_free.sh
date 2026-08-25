#!/usr/bin/env bash
# Wait until GPU 7 or 4 is nearly empty (<=3000 MiB used => ~44 GiB free), then
# retry the 7 vggt scenes that OOM at ~42 GiB next to neighbour processes, and
# push them through sam3d + smplx (resume mode skips everything already done).
set -u
cd "$(dirname "$0")/.."

SCENES="OXknR84A_l0_0020 OXknR84A_l0_0021 VKYPH7pyEeA_0017"

while :; do
    for gpu in 7 4; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
        if [ "$used" -le 3000 ]; then
            break 2
        fi
    done
    sleep 60
done
echo "=== $(date '+%F %T') GPU $gpu has ${used} MiB used — retrying the 7 vggt scenes ==="
sed "s/^devices = .*/devices = [$gpu]                       # single shard, claimed while empty/" \
    configs/corpus.toml > configs/corpus_retry.toml

for stage in vggt sam3d smplx; do
    echo "=== $(date '+%F %T') corpus.py --stage $stage (retry on GPU $gpu) ==="
    if [ "$stage" = vggt ]; then
        uv run --no-sync python scripts/corpus.py --stage vggt --config configs/corpus_retry.toml --scenes $SCENES
    else
        uv run --no-sync python scripts/corpus.py --stage "$stage" --config configs/corpus_retry.toml
    fi
    rc=$?
    echo "=== $(date '+%F %T') exit ${rc}: --stage $stage ==="
done
echo "### retry done ($(date '+%F %T'))"
