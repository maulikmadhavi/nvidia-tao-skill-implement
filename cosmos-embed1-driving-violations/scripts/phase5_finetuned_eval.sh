#!/usr/bin/env bash
# RUNBOOK Phase 5 — export the fine-tuned model, evaluate the EXPORTED artifact
# in both modes on test, produce all comparison reports.
#   ./scripts/phase5_finetuned_eval.sh
set -euo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$EXP/scripts/run_container.sh"
mkdir -p "$EXP/reports"

CKPT_NAME="$(cat "$EXP/workspace/results/train/train/checkpoints/latest_checkpoint.txt")"
echo "== 5.1 export HF from $CKPT_NAME =="
"$RUN" "cosmos-embed1 export -e /specs/export_hf.yaml results_dir=/results/export_hf \
  export.checkpoint=/results/train/train/checkpoints/$CKPT_NAME"
cp "$EXP/workspace/model/Cosmos-Embed1-224p/processor_config.json" \
   "$EXP/workspace/results/export_hf/cosmos_embed1_driving_hf/"

echo "== 5.2 score val+test with the EXPORTED model =="
"$RUN" "python /exp/scripts/10_infer_chunks.py --model /results/export_hf/cosmos_embed1_driving_hf \
  --videos /data/full --list /data/valtest_videos.txt \
  --prompts /data/prompts.json --out /results/scores/finetuned"

echo "== 5.3 re-tune glue on VAL (fine-tuned score scales differ) =="
python3 "$EXP/scripts/14_tune_thresholds.py" --scores "$EXP/workspace/results/scores/finetuned" \
  --annotations "$EXP/workspace/data/annotations.csv" --videos "$EXP/workspace/data/val_videos.txt" \
  --out "$EXP/workspace/results/thresholds_finetuned.json"

echo "== 5.4 post-process + metrics on TEST =="
for MODE in isolated glue; do
  python3 "$EXP/scripts/11_glue_postprocess.py" --scores "$EXP/workspace/results/scores/finetuned" \
    --prompts "$EXP/workspace/data/prompts.json" \
    --thresholds "$EXP/workspace/results/thresholds_finetuned.json" \
    --mode "$MODE" --out "$EXP/workspace/results/events/finetuned_$MODE"
  python3 "$EXP/scripts/13_eval_event_level.py" \
    --events "$EXP/workspace/results/events/finetuned_$MODE/events.json" \
    --annotations "$EXP/workspace/data/annotations.csv" --videos "$EXP/workspace/data/test_videos.txt" \
    --label "finetuned/$MODE" --out "$EXP/reports/event_eval_finetuned_$MODE.json"
done
"$RUN" "python /exp/scripts/12_eval_chunk_level.py --scores /results/scores/finetuned \
  --annotations /data/annotations.csv --videos /data/test_videos.txt \
  --thresholds /results/thresholds_finetuned.json --phase finetuned --out /results/chunk_eval_finetuned"

echo "== 5.5 the 2x2 summary ({baseline,finetuned} x {isolated,glue}) =="
python3 - "$EXP" <<'PY'
import json, sys
from pathlib import Path
exp = Path(sys.argv[1])
print(f"{'run':26} {'event P':>8} {'event R':>8} {'event F1':>8}")
for phase in ("baseline", "finetuned"):
    for mode in ("isolated", "glue"):
        p = exp / "reports" / f"event_eval_{phase}_{mode}.json"
        if p.exists():
            m = json.loads(p.read_text())["micro"]
            print(f"{phase + '/' + mode:26} {m['precision']:8.4f} {m['recall']:8.4f} {m['f1']:8.4f}")
    c = exp / "workspace" / "results" / f"chunk_eval_{phase}" / "metrics.json"
    if c.exists():
        print(f"{phase + ' chunk mAP':26} {json.loads(c.read_text())['mAP']}")
PY

echo "PHASE 5 DONE — deployable model: workspace/results/export_hf/cosmos_embed1_driving_hf"
echo "Next: ./demo/run_demo.sh finetuned glue  ->  http://127.0.0.1:7860"
