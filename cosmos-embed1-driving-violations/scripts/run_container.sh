#!/usr/bin/env bash
# Canonical docker run wrapper (Linux) — driving-violations experiment.
# Same mounts/flags as run_container.ps1. Offline image, no pip preamble.
#
#   ./run_container.sh "python /exp/scripts/10_infer_chunks.py ..."
#   PORTS=7860:7860 ./run_container.sh "python /exp/demo/app.py"
#   HF_CACHE=/path/to/hf_cache ./run_container.sh "..."     # cache override
#   DETACH=1 ./run_container.sh "..."                       # run detached
set -euo pipefail

CMD="${1:?usage: run_container.sh '<in-container command>'}"
IMAGE="${IMAGE:-cosmos-embed-offline:7.0.1}"
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CACHE="${HF_CACHE:-$EXP/workspace/hf_cache}"

ARGS=(run --rm --gpus all --ipc=host --shm-size=16g
      --ulimit memlock=-1 --ulimit stack=67108864
      -v "$EXP/workspace/data:/data:ro"
      -v "$EXP/workspace/specs:/specs:ro"
      -v "$EXP/workspace/results:/results"
      -v "$EXP/workspace/model:/model"
      -v "$HF_CACHE:/hf_cache"
      -v "$EXP/scripts:/exp/scripts:ro"
      -v "$EXP/deploy:/exp/deploy"
      -v "$EXP/demo:/exp/demo:ro"
      -v "$EXP/selftest:/exp/selftest:ro")
[[ -n "${PORTS:-}" ]] && ARGS+=(-p "$PORTS")
[[ -n "${DETACH:-}" ]] && ARGS+=(-d)
ARGS+=("$IMAGE" bash -lc "$CMD")

echo "docker ${ARGS[*]}"
exec docker "${ARGS[@]}"
