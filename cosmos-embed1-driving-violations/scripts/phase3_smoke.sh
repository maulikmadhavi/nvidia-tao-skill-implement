#!/usr/bin/env bash
# RUNBOOK Phase 3 — cheap smoke of the full train→export→load chain (run BEFORE
# real training). Gates: 1-iter train with LoRA (~0.83% trainable), checkpoint
# written, export prints "Inference verification PASSED", HF load gate passes.
#   ./scripts/phase3_smoke.sh
set -euo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$EXP/scripts/run_container.sh"

echo "== 3.1 one-iteration train =="
"$RUN" "cosmos-embed1 train -e /specs/train.yaml results_dir=/results/train_smoke \
  train.max_iter=1 train.validation_iter=2 train.checkpoint_iter=1 \
  train.optim.warmup_steps=0 train.optim.lr_decay_iters=1 \
  dataset.train_dataset.batch_size=2 dataset.val_dataset.batch_size=2 \
  dataset.train_dataset.workers=0 dataset.val_dataset.workers=0"

echo "== 3.2 export HF from the 1-iter checkpoint =="
"$RUN" "cosmos-embed1 export -e /specs/export_hf.yaml results_dir=/results/export_smoke \
  export.checkpoint=/results/train_smoke/train/checkpoints/iter_000000001.pt \
  export.hf_output_dir=/results/export_smoke/hf"
# export omits processor_config.json (processor silently defaults to 448p without it)
cp "$EXP/workspace/model/Cosmos-Embed1-224p/processor_config.json" \
   "$EXP/workspace/results/export_smoke/hf/"

echo "== 3.3 HF load + embed gate =="
FIRST_TEST_VIDEO="$(head -1 "$EXP/workspace/data/test_videos.txt")"
"$RUN" "python /exp/scripts/gate_hf_load.py /results/export_smoke/hf /data/full/${FIRST_TEST_VIDEO}.mp4"

echo "PHASE 3 DONE — chain proven; safe to start real training (phase4_train.sh)"
