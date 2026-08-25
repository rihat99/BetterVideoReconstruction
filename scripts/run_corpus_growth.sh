#!/usr/bin/env bash
# Corpus growth sweep (2026-08-24): bring every curated ClimbingVideos scene up
# to kindyn_1 on GPUs 4 and 7. skip_existing=true in configs/corpus.toml makes
# every stage a resume — only scenes missing the stage output run, so the 361
# finished scenes and the partially-processed ones are never recomputed.
#
# The whole chain runs twice: pass A with two shards per GPU (fast), then pass B
# with one shard per GPU. Pass B is the straggler pass — co-located shards can
# OOM when two big scenes collide (mostly in geometry), and rerunning with
# skip_existing retries exactly the failures. A clean pass A makes pass B a
# cheap no-op sweep.
#
# The contacts runs are pinned to scenes still MISSING their artifact: the
# contacts freshness check is provenance-based (implementation mtimes), so an
# unrestricted run would regenerate the finished scenes' labels under newer
# code and break their already-built kin2/kindyn chains.
set -u
cd "$(dirname "$0")/.."

MAIN=configs/corpus.toml
ONE=configs/corpus_onepergpu.toml
C2=configs/corpus_contacts2.toml
C2ONE=configs/corpus_contacts2_onepergpu.toml
sed 's/^devices = .*/devices = [4, 7]                    # one shard per GPU (straggler pass)/' "$MAIN" > "$ONE"
sed 's/^bodies = "kinematic_1"/bodies = "kinematic_2"/' "$MAIN" > "$C2"
sed 's/^bodies = "kinematic_1"/bodies = "kinematic_2"/' "$ONE" > "$C2ONE"

run() {
    echo
    echo "=== $(date '+%F %T') corpus.py $* ==="
    uv run --no-sync python scripts/corpus.py "$@"
    echo "=== $(date '+%F %T') exit $?: corpus.py $* ==="
}

missing() {  # scene ids of curated targets still missing human_optim/<artifact>
    uv run --no-sync python - "$1" <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from scene_db import ScenePaths, select_targets
paths = ScenePaths(Path("../data/ClimbingVideos").resolve())
with paths.connect() as conn:
    rows = select_targets(conn, categories=(1, 2), rope_supported=0)
print(" ".join(r["scene_id"] for r in rows
               if not (paths.feature_dir("human_optim", r["scene_id"]) / sys.argv[1]).is_file()))
EOF
}

chain() {  # $1 = corpus config, $2 = its contacts_2 (bodies=kinematic_2) copy
    run --stage sam3 --config "$1"
    run --stage vggt --config "$1"
    run --stage sapiens --config "$1"
    run --stage sam3d --config "$1"
    run --stage smplx --config "$1"
    run --stage geometry --config "$1"
    run --stage human_optim --config "$1"
    local m1 m2
    m1=$(missing contacts_1.npz)
    [ -n "$m1" ] && run --stage contacts --config "$1" --scenes $m1
    run --stage kinematics_2 --config "$1"
    m2=$(missing contacts_2.npz)
    [ -n "$m2" ] && run --stage contacts --config "$2" --scenes $m2
    run --stage kindyn --config "$1"
}

echo "### pass A: two shards per GPU ($(date '+%F %T'))"
chain "$MAIN" "$C2"
echo
echo "### pass B: straggler pass, one shard per GPU ($(date '+%F %T'))"
chain "$ONE" "$C2ONE"
echo
echo "### done ($(date '+%F %T'))"
