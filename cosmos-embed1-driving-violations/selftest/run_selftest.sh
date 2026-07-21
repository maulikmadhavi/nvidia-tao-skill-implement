#!/usr/bin/env bash
# End-to-end pipeline self-test on synthetic lessons (bash twin of run_selftest.ps1).
# NOTE: occupies workspace/data — run BEFORE loading real data, or clean afterwards.
# Needs a directory of HMDB chunk mp4s for source material (bundled or copied):
#   ./selftest/run_selftest.sh [/path/to/hmdb_clips]
set -euo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$EXP/scripts/run_container.sh"
ST="$EXP/selftest/out"
TAX="$ST/taxonomy_selftest.json"
HMDB_ARGS=()
[[ -n "${1:-}" ]] && HMDB_ARGS=(--hmdb "$1")

echo "== [1/11] synthetic lessons =="
python3 "$EXP/selftest/make_synthetic.py" --out "$ST" "${HMDB_ARGS[@]}"

echo "== [2/11] chunk + label (00) =="
python3 "$EXP/scripts/00_chunk_videos.py" --videos "$ST/videos" --annotations "$ST/annotations.csv" \
  --out "$EXP/workspace/data" --taxonomy "$TAX" --split 34,33,33 --neg-ratio 4 --seed 42
cp "$ST/annotations.csv" "$EXP/workspace/data/annotations.csv"

echo "== [3/11] metadata + specs (01) =="
python3 "$EXP/scripts/01_make_metadata.py" --data "$EXP/workspace/data" --taxonomy "$TAX" \
  --templates "$EXP/specs" --specs "$EXP/workspace/specs"

echo "== [4/11] score all lessons (10, container) =="
"$RUN" "python /exp/scripts/10_infer_chunks.py --model /model/Cosmos-Embed1-224p \
  --videos /data/full --list all --prompts /data/prompts.json --out /results/scores/selftest"

echo "== [5/11] tune thresholds on val lesson (14) =="
python3 "$EXP/scripts/14_tune_thresholds.py" --scores "$EXP/workspace/results/scores/selftest" \
  --annotations "$EXP/workspace/data/annotations.csv" --videos "$EXP/workspace/data/val_videos.txt" \
  --out "$EXP/workspace/results/thresholds_selftest.json"

echo "== [6/11] glue + isolated post-processing (11) =="
for MODE in glue isolated; do
  python3 "$EXP/scripts/11_glue_postprocess.py" --scores "$EXP/workspace/results/scores/selftest" \
    --prompts "$EXP/workspace/data/prompts.json" \
    --thresholds "$EXP/workspace/results/thresholds_selftest.json" \
    --mode "$MODE" --out "$EXP/workspace/results/events/selftest_$MODE"
done

echo "== [7/11] event-level eval on test lesson (13) =="
python3 "$EXP/scripts/13_eval_event_level.py" \
  --events "$EXP/workspace/results/events/selftest_glue/events.json" \
  --annotations "$EXP/workspace/data/annotations.csv" --videos "$EXP/workspace/data/test_videos.txt" \
  --label "selftest/zero-shot/glue" --out "$EXP/workspace/results/event_eval_selftest_glue.json"

echo "== [8/11] chunk-level eval (12, container) =="
"$RUN" "python /exp/scripts/12_eval_chunk_level.py --scores /results/scores/selftest \
  --annotations /data/annotations.csv --videos /data/test_videos.txt \
  --thresholds /results/thresholds_selftest.json --phase selftest-zeroshot --out /results/chunk_eval_selftest"

F1="$(python3 -c "import json; print(json.load(open('$EXP/workspace/results/event_eval_selftest_glue.json'))['micro']['f1'])")"
if python3 -c "exit(0 if $F1 > 0 else 1)"; then
  echo "SELFTEST GATE PASSED: glue recovered planted events (micro F1 = $F1)"
else
  echo "SELFTEST GATE FAILED: micro F1 = 0 (no planted event recovered)"; exit 1
fi

echo "== [9/11] fit heads on train lesson (15, container) =="
"$RUN" "python /exp/scripts/15_train_heads.py --scores /results/scores/selftest \
  --annotations /data/annotations.csv --train-videos /data/train_videos.txt \
  --val-videos /data/val_videos.txt --out /results/heads/selftest_heads.npz \
  --report /results/heads/selftest_heads_report.json"

echo "== [10/11] head probabilities + unknown queue (16, container) =="
"$RUN" "python /exp/scripts/16_score_heads.py --scores /results/scores/selftest \
  --heads /results/heads/selftest_heads.npz --out /results/scores/selftest_heads"

echo "== [11/11] heads: tune (14, probability grid) -> glue (11) -> event eval (13) =="
python3 "$EXP/scripts/14_tune_thresholds.py" --scores "$EXP/workspace/results/scores/selftest_heads" \
  --annotations "$EXP/workspace/data/annotations.csv" --videos "$EXP/workspace/data/val_videos.txt" \
  --thr-grid 0.05:0.95:0.02 \
  --out "$EXP/workspace/results/thresholds_selftest_heads.json"
python3 "$EXP/scripts/11_glue_postprocess.py" --scores "$EXP/workspace/results/scores/selftest_heads" \
  --prompts "$EXP/workspace/data/prompts.json" \
  --thresholds "$EXP/workspace/results/thresholds_selftest_heads.json" \
  --mode glue --out "$EXP/workspace/results/events/selftest_heads_glue"
python3 "$EXP/scripts/13_eval_event_level.py" \
  --events "$EXP/workspace/results/events/selftest_heads_glue/events.json" \
  --annotations "$EXP/workspace/data/annotations.csv" --videos "$EXP/workspace/data/test_videos.txt" \
  --label "selftest/heads/glue" --out "$EXP/workspace/results/event_eval_selftest_heads_glue.json"

F1H="$(python3 -c "import json; print(json.load(open('$EXP/workspace/results/event_eval_selftest_heads_glue.json'))['micro']['f1'])")"
if [ -f "$EXP/workspace/results/heads/selftest_heads_report.json" ] && python3 -c "exit(0 if $F1H > 0 else 1)"; then
  echo "SELFTEST HEADS GATE PASSED: heads glue recovered planted events (micro F1 = $F1H)"
else
  echo "SELFTEST HEADS GATE FAILED: heads report missing or heads glue micro F1 = 0"; exit 1
fi
