#!/usr/bin/env bash
# RUNBOOK Phase 2 — zero-shot baseline: score val+test videos, tune glue on val,
# run both inference modes, evaluate on test (event F1 + chunk PR-AUC).
#   ./scripts/phase2_zeroshot_baseline.sh
set -euo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$EXP/scripts/run_container.sh"
mkdir -p "$EXP/reports"

echo "== 2.1 score matrices (container, GPU) =="
"$RUN" "python /exp/scripts/10_infer_chunks.py --model /model/Cosmos-Embed1-224p \
  --videos /data/full --list /data/valtest_videos.txt \
  --prompts /data/prompts.json --out /results/scores/baseline"

echo "== 2.2 tune glue params on VAL =="
python3 "$EXP/scripts/14_tune_thresholds.py" --scores "$EXP/workspace/results/scores/baseline" \
  --annotations "$EXP/workspace/data/annotations.csv" --videos "$EXP/workspace/data/val_videos.txt" \
  --out "$EXP/workspace/results/thresholds_baseline.json"

echo "== 2.3 post-process: isolated + glue =="
for MODE in isolated glue; do
  python3 "$EXP/scripts/11_glue_postprocess.py" --scores "$EXP/workspace/results/scores/baseline" \
    --prompts "$EXP/workspace/data/prompts.json" \
    --thresholds "$EXP/workspace/results/thresholds_baseline.json" \
    --mode "$MODE" --out "$EXP/workspace/results/events/baseline_$MODE"
done

echo "== 2.4 metrics on TEST =="
for MODE in isolated glue; do
  python3 "$EXP/scripts/13_eval_event_level.py" \
    --events "$EXP/workspace/results/events/baseline_$MODE/events.json" \
    --annotations "$EXP/workspace/data/annotations.csv" --videos "$EXP/workspace/data/test_videos.txt" \
    --label "baseline/$MODE" --out "$EXP/reports/event_eval_baseline_$MODE.json"
done
"$RUN" "python /exp/scripts/12_eval_chunk_level.py --scores /results/scores/baseline \
  --annotations /data/annotations.csv --videos /data/test_videos.txt \
  --thresholds /results/thresholds_baseline.json --phase baseline --out /results/chunk_eval_baseline"

echo "PHASE 2 DONE — baseline reports in $EXP/reports and workspace/results/chunk_eval_baseline"
