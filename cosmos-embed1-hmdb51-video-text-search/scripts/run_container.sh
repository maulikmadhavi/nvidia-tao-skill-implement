#!/usr/bin/env bash
# Canonical docker run wrapper for the cosmos-embed experiment (Linux/macOS, zsh/bash).
# Single source of truth for image + mounts + flags so they never drift.
#
# Usage:
#   ./run_container.sh "cosmos-embed1 evaluate -e /specs/evaluate_zeroshot.yaml results_dir=/results/evaluate_zeroshot"
#   PORTS=7860:7860 ./run_container.sh "python /exp/demo/app.py"
#   DETACH=1 ./run_container.sh "..."
#
# Deviations from the skill's DOCKER_COMMON (deliberate): --shm-size=16g (not 64g);
# no --network=host (use PORTS for the demo). protobuf<7 preamble is mandatory
# (wandb import pitfall). Pass HF_TOKEN in the environment to authenticate pulls.

set -euo pipefail

CMD="${1:?usage: run_container.sh '<in-container command>'}"
IMAGE="${IMAGE:-nvcr.io/nvidia/tao/tao-toolkit:7.0.1-cosmos-embed}"
EXP="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"   # experiment root (this file lives in scripts/)

ARGS=(run --rm --gpus all --ipc=host --shm-size=16g
      --ulimit memlock=-1 --ulimit stack=67108864
      -e HF_TOKEN
      -e WANDB_DISABLED=true
      -e WANDB_MODE=disabled
      -e HUGGINGFACE_HUB_CACHE=/hf_cache
      -v "$EXP/workspace/data:/data:ro"
      -v "$EXP/workspace/specs:/specs:ro"
      -v "$EXP/workspace/results:/results"
      -v "$EXP/workspace/model:/model"
      -v "$EXP/workspace/hf_cache:/hf_cache"
      -v "$EXP/scripts:/exp/scripts:ro"
      -v "$EXP/deploy:/exp/deploy"
      -v "$EXP/demo:/exp/demo:ro")
[[ -n "${PORTS:-}" ]] && ARGS+=(-p "$PORTS")
[[ -n "${DETACH:-}" ]] && ARGS+=(-d)
ARGS+=("$IMAGE" bash -lc "python -m pip install --quiet 'protobuf<7' && $CMD")

echo "docker ${ARGS[*]}"
exec docker "${ARGS[@]}"
