#!/usr/bin/env bash
# Launch the timeline review demo at http://127.0.0.1:7860 (IPv4).
#   ./run_demo.sh [phase] [mode]     # defaults: finetuned glue
set -euo pipefail
PHASE="${1:-finetuned}"
MODE="${2:-glue}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORTS=7860:7860 "$DIR/scripts/run_container.sh" \
  "python /exp/demo/app.py --scores /results/scores/$PHASE --events /results/events/${PHASE}_${MODE}/events.json --videos /data/full"
