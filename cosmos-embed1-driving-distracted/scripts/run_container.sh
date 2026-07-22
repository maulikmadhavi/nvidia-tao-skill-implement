#!/usr/bin/env bash
# docker run wrapper (Linux/macOS, zsh/bash) — driving-distracted clip-classification
# experiment. Uses the OFFLINE cosmos-embed image (deps baked). Reuses the model
# snapshot and hf_cache already living in the driving-violations experiment so we
# don't duplicate 7.6 GB. The raw dataset is mounted read-only at /data.
#
#   ./scripts/run_container.sh "python /exp/scripts/10_embed.py ..."
#
# Paths default to the sibling-experiment layout but are overridable via env vars
# (the Windows D:\ defaults won't exist on a POSIX host — set these to match yours):
#   DATA_ROOT   dataset root (shortclips_pos_neg)
#   SHARED      driving-violations workspace holding model/ and hf_cache/
#   IMAGE       offline image tag
#
# Mounts:
#   /data         dataset root (shortclips_pos_neg)      ro
#   /splits       workspace/splits (metadata + prompts)  ro
#   /results      workspace/results                      rw
#   /model        shared Cosmos-Embed1-224p snapshot     ro
#   /hf_cache     shared HF cache                         rw
#   /exp/scripts  this experiment's scripts              ro

set -euo pipefail

if [[ $# -lt 1 || -z "$1" ]]; then
    echo "usage: $0 \"<command to run inside the container>\"" >&2
    exit 2
fi
CMD="$1"

# script dir → experiment root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
EXP="$(dirname "$SCRIPT_DIR")"

IMAGE="${IMAGE:-cosmos-embed-offline:7.0.1}"
DATA_ROOT="${DATA_ROOT:-/data/research_data/driving_violation/shortclips_pos_neg/shortclips_pos_neg}"
SHARED="${SHARED:-$(dirname "$EXP")/cosmos-embed1-driving-violations/workspace}"

docker_args=(
    run --rm --gpus all --ipc=host --shm-size=16g
    --ulimit memlock=-1 --ulimit stack=67108864
    -v "${DATA_ROOT}:/data:ro"
    -v "${EXP}/workspace/splits:/splits:ro"
    -v "${EXP}/workspace/results:/results"
    -v "${SHARED}/model:/model:ro"
    -v "${SHARED}/hf_cache:/hf_cache"
    -v "${EXP}/scripts:/exp/scripts:ro"
    "$IMAGE" bash -lc "$CMD"
)

echo "docker ${docker_args[*]}"
exec docker "${docker_args[@]}"
