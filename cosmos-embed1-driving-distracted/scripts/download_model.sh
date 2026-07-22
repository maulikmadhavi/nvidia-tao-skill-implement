#!/usr/bin/env bash
# Pull the Cosmos-Embed1-224p snapshot for the driving-distracted experiment
# into MODEL_DIR — the dir run_container.sh / run_steps.sh mount at
# /model/Cosmos-Embed1-224p. The offline image runs with HF_HUB_OFFLINE=1, so
# the model must be materialized on disk (a real snapshot, not cache symlinks).
#
#   ./download_model.sh
#   MODEL_DIR=/data/you/Cosmos-Embed1-224p ./download_model.sh
#
# Env knobs:
#   MODEL_DIR   where to write the snapshot   (default: ./_model/Cosmos-Embed1-224p)
#   HF_HOME / XDG_CACHE_HOME   point HF's intermediate cache at a roomy disk
#   HF_HUB_ENABLE_HF_TRANSFER  set 1 for faster parallel downloads (needs hf_transfer)

set -euo pipefail

REPO="nvidia/Cosmos-Embed1-224p"   # NOTE: the video embedder — NOT Cosmos3-Nano
MODEL_DIR="${MODEL_DIR:-$(pwd)/_model/Cosmos-Embed1-224p}"

# hf CLI (huggingface_hub >= 0.26); fall back to the legacy name if needed
if command -v hf >/dev/null 2>&1; then HF=hf
elif command -v huggingface-cli >/dev/null 2>&1; then HF=huggingface-cli
else echo "no 'hf' / 'huggingface-cli' on PATH — pip install -U 'huggingface_hub[cli]'" >&2; exit 1; fi

echo "[i] repo:      $REPO"
echo "[i] MODEL_DIR: $MODEL_DIR"
mkdir -p "$MODEL_DIR"
[[ -n "${HF_HOME:-}" ]] && echo "[i] HF_HOME:   $HF_HOME"
df -h "$MODEL_DIR" 2>/dev/null || true

# retry until the pull completes (resumable — safe to re-run)
until "$HF" download "$REPO" --local-dir "$MODEL_DIR"; do
    echo "[!] download interrupted — retrying in 10s..."; sleep 10
done

# completeness check: the loader needs config.json + the trust_remote_code files
if [[ -f "$MODEL_DIR/config.json" ]]; then
    echo "[i] DONE — snapshot at $MODEL_DIR"
    echo "    run with:  MODEL_DIR=$MODEL_DIR ./run_steps.sh"
else
    echo "[x] $MODEL_DIR/config.json missing after download — snapshot incomplete" >&2
    exit 1
fi
