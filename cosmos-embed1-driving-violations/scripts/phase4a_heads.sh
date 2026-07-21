#!/usr/bin/env bash
# RUNBOOK Phase 4a — per-violation heads on the frozen encoder (HEADS_DESIGN.md).
# Run AFTER phase 2 (needs val+test scores in /results/scores/$SCORES); adds
# train scores, fits heads (15), rescoring (16), tuning (14, probability grid),
# both inference modes (11), test metrics (13+12), comparison print.
#   ./scripts/phase4a_heads.sh
#   TAG=heads_v2 SCORES=finetuned MODEL=/results/export_hf/cosmos_embed1_driving_hf \
#     ./scripts/phase4a_heads.sh          # Phase 5b: refit heads after LoRA
set -euo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$EXP/scripts/run_container.sh"
TAG="${TAG:-heads_v1}"
SCORES="${SCORES:-baseline}"
MODEL="${MODEL:-/model/Cosmos-Embed1-224p}"
mkdir -p "$EXP/reports"

echo "== 4a.1 score TRAIN videos with the same encoder (completes the embedding cache) =="
"$RUN" "python /exp/scripts/10_infer_chunks.py --model $MODEL \
  --videos /data/full --list /data/train_videos.txt \
  --prompts /data/prompts.json --out /results/scores/$SCORES"

echo "== 4a.2 fit heads (15, container, CPU) =="
"$RUN" "python /exp/scripts/15_train_heads.py --scores /results/scores/$SCORES \
  --annotations /data/annotations.csv --train-videos /data/train_videos.txt \
  --val-videos /data/val_videos.txt --out /results/heads/$TAG.npz \
  --report /results/heads/${TAG}_report.json"

echo "== 4a.3 head probabilities + unknown queue for all scored videos (16) =="
"$RUN" "python /exp/scripts/16_score_heads.py --scores /results/scores/$SCORES \
  --heads /results/heads/$TAG.npz --out /results/scores/$TAG"

echo "== 4a.4 tune glue params on VAL (probability grid) =="
python3 "$EXP/scripts/14_tune_thresholds.py" --scores "$EXP/workspace/results/scores/$TAG" \
  --annotations "$EXP/workspace/data/annotations.csv" --videos "$EXP/workspace/data/val_videos.txt" \
  --thr-grid 0.05:0.95:0.02 \
  --out "$EXP/workspace/results/thresholds_$TAG.json"

echo "== 4a.5 post-process: isolated + glue =="
for MODE in isolated glue; do
  python3 "$EXP/scripts/11_glue_postprocess.py" --scores "$EXP/workspace/results/scores/$TAG" \
    --prompts "$EXP/workspace/data/prompts.json" \
    --thresholds "$EXP/workspace/results/thresholds_$TAG.json" \
    --mode "$MODE" --out "$EXP/workspace/results/events/${TAG}_$MODE"
done

echo "== 4a.6 metrics on TEST =="
for MODE in isolated glue; do
  python3 "$EXP/scripts/13_eval_event_level.py" \
    --events "$EXP/workspace/results/events/${TAG}_$MODE/events.json" \
    --annotations "$EXP/workspace/data/annotations.csv" --videos "$EXP/workspace/data/test_videos.txt" \
    --label "$TAG/$MODE" --out "$EXP/reports/event_eval_${TAG}_$MODE.json"
done
"$RUN" "python /exp/scripts/12_eval_chunk_level.py --scores /results/scores/$TAG \
  --annotations /data/annotations.csv --videos /data/test_videos.txt \
  --thresholds /results/thresholds_$TAG.json --phase $TAG --out /results/chunk_eval_$TAG"

echo "== 4a.7 comparison: $SCORES vs $TAG =="
python3 - "$EXP" "$SCORES" "$TAG" <<'PY'
import json, sys
from pathlib import Path
exp, base, tag = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
print(f"{'run':26} {'event P':>8} {'event R':>8} {'event F1':>8}")
for phase in (base, tag):
    for mode in ("isolated", "glue"):
        p = exp / "reports" / f"event_eval_{phase}_{mode}.json"
        if p.exists():
            m = json.loads(p.read_text())["micro"]
            print(f"{phase + '/' + mode:26} {m['precision']:8.4f} {m['recall']:8.4f} {m['f1']:8.4f}")
    c = exp / "workspace" / "results" / f"chunk_eval_{phase}" / "metrics.json"
    if c.exists():
        print(f"{phase + ' chunk mAP':26} {json.loads(c.read_text())['mAP']}")
PY

echo "PHASE 4a DONE — heads: workspace/results/heads/$TAG.npz, report: ${TAG}_report.json,"
echo "  review queue: workspace/results/scores/$TAG/review_queue.json"
echo "Phase 4b (LoRA) ONLY if a class with >= 30 positive train windows still has low test AP."
