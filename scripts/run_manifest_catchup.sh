#!/usr/bin/env bash
# Manifest catch-up (2026-08-25): finish the corpus up to smplx ONLY — no
# geometry or later stages. vggt runs one shard per GPU: the growth sweep
# showed two co-located VGGT workers OOM a 47 GiB card (each wants 20-30 GiB).
# sam3/sapiens/sam3d/smplx were fine co-located, so they keep the main config's
# [4,4,7,7]. skip_existing=true makes every stage a resume — sam3/sapiens rerun
# cheaply and only pick up scenes the upstream DB added since the last pass.
set -u
cd "$(dirname "$0")/.."

MAIN=configs/corpus.toml
ONE=configs/corpus_onepergpu.toml
# GPU 7 only: the 7 remaining vggt scenes are 448-488-frame forwards (~42 GiB)
# and neighbour processes currently hold ~9 GiB on GPU 4 — only 7 has room.
sed 's/^devices = .*/devices = [7]                       # single shard, emptiest GPU/' "$MAIN" > "$ONE"

run() {
    echo
    echo "=== $(date '+%F %T') corpus.py $* ==="
    uv run --no-sync python scripts/corpus.py "$@"
    local rc=$?
    echo "=== $(date '+%F %T') exit ${rc}: corpus.py $* ==="
}

run --stage sam3 --config "$MAIN"
run --stage vggt --config "$ONE"
run --stage sapiens --config "$MAIN"
run --stage sam3d --config "$MAIN"
run --stage smplx --config "$MAIN"
echo
echo "### done ($(date '+%F %T'))"
