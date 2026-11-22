#!/usr/bin/env bash
set -euo pipefail

recipe=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

usage() {
  printf '%s\n' 'Usage: scripts/dual-128k-fp4.sh [check|start|restart|stop]' \
    'Runs the opt-in two-slot profile using a prepared image on all three Sparks.' \
    'See docs/dual-128k-fp4.md for preparation and rollback ownership.'
}

case "${1:-start}" in
  help|-h|--help) usage; exit 0 ;;
  check|start|restart|stop) ;;
  *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then usage >&2; exit 2; fi

# This wrapper owns only these containers and this state directory.
export CONTAINER_NAME=glm53-full-tf-dual-128k-fp4
export STATE_DIR="$HOME/.local/state/glm53-full-tensorfold-dual-128k-fp4"
export PREPARE=0 KEEP_OLD_PCACHES=1
export CP=0 PARALLEL=${PARALLEL:-2} CONTEXT=${CONTEXT:-131072} KV=${KV:-fp4} DENSE=q4 DRAFTER=dspark MTP=0
export PREFILL_ROWS=${PREFILL_ROWS:-3072} PREFILL_SPLIT=1 PREFILL_OVERLAP=1
export TF_GLM_MULTI_GRAPHS=top TF_GLM_MULTI_PREFILL=1
export TF_GLM_FILL_ROWS=${TF_GLM_FILL_ROWS:-1024} TF_GLM_FILL_SHARE=${TF_GLM_FILL_SHARE:-0.5}
export TF_GLM_MULTI_KEEP_HOME_CACHE=1 TF_GLM_MULTI_DRAFT_BORROW=0
export TF_GLM_FILL_BUDGET_MS=0 TF_GLM_FILL_DRAFTS=0
export DISK_CACHE=0 FLOOR_GIB=6 OVERHEAD_GIB=10 MEMORY_RESERVE_GIB=16 GUARD=1 GUARD_KILL_GIB=3
# Verify windows of 24 rows, for one stream and for both together (0146), so copy drafts run longer; 24-row all-gathers
# stay on RoCE with TF_ROCE_MAX_KB=512 (16 KiB a row). Replies stay equal to their serial references (tools/exact.py).
export TF_GLM_MAX_ROWS=${TF_GLM_MAX_ROWS:-24} TF_GLM_MULTI_ROWS=${TF_GLM_MULTI_ROWS:-24} TF_ROCE_MAX_KB=${TF_ROCE_MAX_KB:-512}
# 24 kept prompts instead of 8: an agent's tool runs no longer push out a shared system prompt (~6k pool tokens)
export TF_GLM_CACHE_ENTRIES=${TF_GLM_CACHE_ENTRIES:-24}
# Expert usage counters (0144), for the own-quantization work; TF_GLM_EXPERT_COUNTS=0 turns them off
export TF_GLM_EXPERT_COUNTS=${TF_GLM_EXPERT_COUNTS:-1}

check_image() (
  cd "$recipe"
  source ./scripts/config.sh
  source ./scripts/nodes.sh
  check_workers

  probe=$(cat <<'PY'
from hashlib import sha256
from pathlib import Path
from tensorfold.families.glm5_next.cuda import engine, multi, verify

multi_source = Path(multi.__file__).read_bytes()
engine_source = Path(engine.__file__).read_bytes()
required = {
    "verify.py": (Path(verify.__file__).read_bytes(), (b"TF_GLM_MULTI_ROWS",)),
    "multi.py": (multi_source, (
        b"replay_overtakes", b"TF_GLM_MULTI_KEEP_HOME_CACHE", b'"home_cache"',
    )),
    "engine.py": (engine_source, (
        b"TF_GLM_MULTI_KEEP_HOME_CACHE", b"TF_GLM_MULTI_DRAFT_BORROW",
    )),
}
for name, (source, features) in required.items():
    missing = [feature.decode() for feature in features if feature not in source]
    if missing:
        raise SystemExit(f"{name} is missing preset support: {', '.join(missing)}")
print(sha256(multi_source).hexdigest(), sha256(engine_source).hexdigest())
PY
  )
  image_fingerprint() {
    local image=$1 probe=$2 label hashes
    label=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$image") || return 1
    [[ "$label" =~ ^[0-9a-f]{12}$ ]] || return 1
    hashes=$(docker run --pull=never --rm --network none --memory 2g --entrypoint python "$image" -B -c "$probe") || return 1
    [[ "$hashes" =~ ^[0-9a-f]{64}\ [0-9a-f]{64}$ ]] || return 1
    printf '%s %s\n' "$label" "$hashes"
  }
  expected=$(image_fingerprint "$IMAGE" "$probe") || die "rank 0 cannot verify the prepared image $IMAGE"
  printf -v remote_arguments '%q ' "$IMAGE" "$probe"
  remote_probe="$(declare -f image_fingerprint)"$'\n'"image_fingerprint $remote_arguments"
  for rank in $(worker_ids); do
    actual=$(worker "$rank" bash -s <<< "$remote_probe") || die "rank $rank cannot verify the prepared image $IMAGE"
    [[ "$actual" == "$expected" ]] || die "rank $rank uses a different patch label or TensorFold module in $IMAGE"
  done
  read -r patch_hash multi_hash engine_hash <<< "$expected"
  log "Verified replay and cache relocation modules on all three ranks in $IMAGE"
  printf '  tf.patches %s\n  multi.py %s\n  engine.py %s\n' "$patch_hash" "$multi_hash" "$engine_hash"
)

case "${1:-start}" in
  stop) exec "$recipe/stop.sh" ;;
  check) check_image ;;
  start) check_image; exec "$recipe/start.sh" ;;
  restart) check_image; exec "$recipe/start.sh" restart ;;
esac
