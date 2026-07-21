#!/usr/bin/env bash
# RUNBOOK Phase 1 — data preparation (host: python3 stdlib + ffmpeg).
#   ./scripts/phase1_prepare.sh /path/to/lesson_videos /path/to/annotations.csv
set -euo pipefail
VIDEOS="${1:?usage: phase1_prepare.sh <videos_dir> <annotations.csv>}"
ANNOT="${2:?usage: phase1_prepare.sh <videos_dir> <annotations.csv>}"
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== chunk + label (00) =="
python3 "$EXP/scripts/00_chunk_videos.py" \
  --videos "$VIDEOS" --annotations "$ANNOT" --out "$EXP/workspace/data" --seed 42
cp "$ANNOT" "$EXP/workspace/data/annotations.csv"

echo "== metadata + specs (01) =="
python3 "$EXP/scripts/01_make_metadata.py" \
  --data "$EXP/workspace/data" --templates "$EXP/specs" --specs "$EXP/workspace/specs"

cat "$EXP/workspace/data/val_videos.txt" "$EXP/workspace/data/test_videos.txt" \
  > "$EXP/workspace/data/valtest_videos.txt"
echo "PHASE 1 DONE — review the gate lines above and chunk_manifest.csv"
