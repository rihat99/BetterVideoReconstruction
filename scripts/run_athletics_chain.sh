#!/usr/bin/env bash
# Post-smplx chain over the AthleticsVideos DB, one corpus config at a time:
#   geometry → gravity → human_optim (kin1) → contacts_1 → kinematics_2 → contacts_2 → kindyn_1
# Usage: scripts/run_athletics_chain.sh configs/corpus_athletics.toml [configs/corpus_athletics_pommel.toml ...]
# Geometry runs with the config's device list (2 shards/GPU: the pre-fit holds ~20-24 GiB),
# then a one-shard-per-GPU straggler pass redoes the scenes a big+big collision OOMed
# (a failed scale leaves the world non-metric, which skip_existing treats as not done);
# the lighter stages run 3 shards/GPU (derived configs in scratch/). Every stage resumes
# (skip_existing) and tolerates per-scene failures; grep the log for "FAILED after".
# A stage's rc=1 only means some target scene lacks the artifact (e.g. the 18 no-YOLO
# scenes) — judge failures by the per-shard "[worker] N/M" lines.
set -u
cd "$(dirname "$0")/.."
mkdir -p scratch
stamp=$(date +%Y%m%d_%H%M)
run() {  # run <stage> <config>
    echo "=== $(date '+%F %T') stage=$1 config=$2"
    uv run --no-sync python scripts/corpus.py --stage "$1" --config "$2"
    echo "=== $(date '+%F %T') stage=$1 rc=$?"
}
for cfg in "$@"; do
    name=$(basename "$cfg" .toml)
    solo=scratch/${name}_1pg.toml        # 1 shard per GPU (geometry stragglers)
    dense=scratch/${name}_3pg.toml       # 3 shards per GPU
    c2=scratch/${name}_3pg_c2.toml       # + contacts on kinematic_2
    python3 - "$cfg" "$solo" "$dense" "$c2" <<'PY'
import re, sys
src, solo, dense, c2 = sys.argv[1:]
text = open(src).read()
gpus = sorted({int(g) for g in re.search(r"^devices\s*=\s*\[([^\]]*)\]", text, re.M).group(1).split(",")})
def with_devices(per_gpu):
    return re.sub(r"^devices\s*=\s*\[[^\]]*\]", "devices = " + str([g for g in gpus for _ in range(per_gpu)]),
                  text, count=1, flags=re.M)
open(solo, "w").write(with_devices(1))
dense_text = with_devices(3)
open(dense, "w").write(dense_text)
open(c2, "w").write(dense_text.replace('bodies = "kinematic_1"', 'bodies = "kinematic_2"', 1))
PY
    run geometry     "$cfg"
    run geometry     "$solo"
    run gravity      "$dense"
    run human_optim  "$dense"
    run contacts     "$dense"
    run kinematics_2 "$dense"
    run contacts     "$c2"
    run kindyn       "$dense"
done
echo "=== $(date '+%F %T') all done"
