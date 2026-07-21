#!/usr/bin/env bash
# Build the air-gap migration bundle into bundle/ (run on the INTERNET machine).
# Linux/Ubuntu twin of make_bundle.ps1.
#
#   ./scripts/make_bundle.sh                        # manifest dry-run (no 20GB docker save)
#   ./scripts/make_bundle.sh --full                 # also docker save the offline image (~20+GB)
#   ./scripts/make_bundle.sh --model-source /path   # snapshots live somewhere else
#
# Bundle contents:
#   cosmos-embed-offline.tar   docker image with baked deps  (--full only)
#   model/                     FULL local snapshots: Cosmos-Embed1-224p (incl.
#                              model_converted.pth — the CLI needs it and does NOT
#                              read the HF hub cache) + bert-base-uncased
#   kit/                       this experiment folder (scripts/specs/demo/deploy/selftest/docs)
#   MANIFEST.txt               sizes + SHA256 of the tar
set -euo pipefail

FULL=0
MODEL_SOURCE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full) FULL=1; shift ;;
    --model-source) MODEL_SOURCE="${2:?--model-source needs a path}"; shift 2 ;;
    *) echo "usage: make_bundle.sh [--full] [--model-source /path]" >&2; exit 2 ;;
  esac
done

EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE="$EXP/bundle"
IMAGE="cosmos-embed-offline:7.0.1"
mkdir -p "$BUNDLE"

gb() { awk -v b="$1" 'BEGIN { printf "%.2f", b / 1073741824 }'; }

lines=("bundle built: $(date -Iseconds)" "image: $IMAGE" "")

# 1. docker image
if [[ "$FULL" -eq 1 ]]; then
  TAR_PATH="$BUNDLE/cosmos-embed-offline.tar"
  echo "docker save (this is 20+GB, takes a while) ..."
  docker save -o "$TAR_PATH" "$IMAGE"
  SHA="$(sha256sum "$TAR_PATH" | cut -d' ' -f1)"
  lines+=("cosmos-embed-offline.tar  $(gb "$(stat -c%s "$TAR_PATH")") GB  sha256=$SHA")
else
  lines+=("cosmos-embed-offline.tar  SKIPPED (dry-run; rerun with --full)")
fi

# 2. full model snapshots (populate first with scripts/fetch_snapshots.py)
[[ -n "$MODEL_SOURCE" ]] || MODEL_SOURCE="$EXP/workspace/model"
if [[ -d "$MODEL_SOURCE/Cosmos-Embed1-224p" && -d "$MODEL_SOURCE/bert-base-uncased" ]]; then
  echo "copying model snapshots from $MODEL_SOURCE ..."
  mkdir -p "$BUNDLE/model"
  cp -a "$MODEL_SOURCE/." "$BUNDLE/model/"
  lines+=("model/  $(gb "$(du -sb "$BUNDLE/model" | cut -f1)") GB (Cosmos-Embed1-224p full snapshot + bert-base-uncased)")
else
  lines+=("model/  MISSING: run scripts/fetch_snapshots.py first (needs internet once)")
fi

# 3. experiment kit (code + docs, no workspace outputs)
echo "copying kit ..."
KIT="$BUNDLE/kit"
for d in scripts specs demo deploy selftest; do
  mkdir -p "$KIT/$d"
  cp -a "$EXP/$d/." "$KIT/$d/"
done
cp "$EXP"/*.md "$KIT/"
cp "$EXP/Dockerfile.offline" "$KIT/"
lines+=("kit/  (scripts, specs, demo, deploy, selftest, docs, Dockerfile.offline)")

printf '%s\n' "${lines[@]}" > "$BUNDLE/MANIFEST.txt"
echo "----"
cat "$BUNDLE/MANIFEST.txt"
