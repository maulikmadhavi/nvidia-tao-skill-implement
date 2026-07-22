#!/usr/bin/env bash
# Stage every weight the offline run needs onto disk:
#   1) the Cosmos-Embed1-224p snapshot -> MODEL_DIR (mounted at /model/Cosmos-Embed1-224p)
#   2) bert-base-uncased -> the HF cache (HF_CACHE, mounted at /hf_cache)
#
# Why bert too: the Cosmos Q-Former calls BertConfig.from_pretrained("bert-base-uncased")
# at load time (modeling_qformer.py). The offline image runs HF_HUB_OFFLINE=1 and can't
# fetch it, so it must already sit in the mounted HF cache or the model never loads.
#
#   ./download_model.sh
#   MODEL_DIR=/data/you/Cosmos-Embed1-224p HF_CACHE=/data/you/hfcache ./download_model.sh
#
# Env knobs:
#   MODEL_DIR   where to write the model snapshot  (default: ./_model/Cosmos-Embed1-224p)
#   HF_CACHE    HF hub cache for bert (and /hf_cache mount)  (default: ./_hf_cache)
#   HF_HOME / XDG_CACHE_HOME    point HF's intermediate cache at a roomy disk
#   HF_HUB_ENABLE_HF_TRANSFER   set 1 for faster parallel downloads (needs hf_transfer)

set -euo pipefail

MODEL_REPO="nvidia/Cosmos-Embed1-224p"   # NOTE: the video embedder — NOT Cosmos3-Nano
BERT_REPO="bert-base-uncased"            # bare alias the Q-Former asks for by default
MODEL_DIR="${MODEL_DIR:-$(pwd)/_model/Cosmos-Embed1-224p}"
HF_CACHE="${HF_CACHE:-$(pwd)/_hf_cache}"

# hf CLI (huggingface_hub >= 0.26); fall back to the legacy name if needed
if command -v hf >/dev/null 2>&1; then HF=hf
elif command -v huggingface-cli >/dev/null 2>&1; then HF=huggingface-cli
else echo "no 'hf' / 'huggingface-cli' on PATH — pip install -U 'huggingface_hub[cli]'" >&2; exit 1; fi

# retry-until wrapper (downloads are resumable — safe to re-run)
pull() {  # pull <description> <hf download args...>
    local desc="$1"; shift
    until "$HF" download "$@"; do
        echo "[!] $desc interrupted — retrying in 10s..."; sleep 10
    done
}

echo "[i] MODEL_DIR: $MODEL_DIR   (<- $MODEL_REPO)"
echo "[i] HF_CACHE:  $HF_CACHE    (<- $BERT_REPO)"
[[ -n "${HF_HOME:-}" ]] && echo "[i] HF_HOME:   $HF_HOME"
mkdir -p "$MODEL_DIR" "$HF_CACHE"
df -h "$MODEL_DIR" 2>/dev/null || true

# 1) model snapshot as real files (not cache symlinks) for the bind mount
pull "model snapshot" "$MODEL_REPO" --local-dir "$MODEL_DIR"

# 2) bert into the HF hub cache -> $HF_CACHE/models--bert-base-uncased/...
#    (container reads it via HUGGINGFACE_HUB_CACHE=/hf_cache)
pull "bert-base-uncased" "$BERT_REPO" --cache-dir "$HF_CACHE"

# completeness checks
ok=1
[[ -f "$MODEL_DIR/config.json" ]] || { echo "[x] $MODEL_DIR/config.json missing — model snapshot incomplete" >&2; ok=0; }
ls "$HF_CACHE"/models--bert-base-uncased/snapshots/*/config.json >/dev/null 2>&1 \
    || { echo "[x] bert config not found under $HF_CACHE/models--bert-base-uncased — bert stage incomplete" >&2; ok=0; }
[[ $ok -eq 1 ]] || exit 1

echo "[i] DONE."
echo "    run with:  MODEL_DIR=$MODEL_DIR HF_CACHE=$HF_CACHE ./run_steps.sh"
