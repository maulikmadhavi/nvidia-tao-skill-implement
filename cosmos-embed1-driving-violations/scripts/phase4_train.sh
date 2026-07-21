#!/usr/bin/env bash
# RUNBOOK Phase 4 — LoRA fine-tuning.
#   ./scripts/phase4_train.sh                      # uses max_iter from specs/train.yaml (600)
#   ./scripts/phase4_train.sh 900                  # override max_iter (lr_decay_iters follows)
#   ./scripts/phase4_train.sh 900 "dataset.train_dataset.batch_size=2"   # extra overrides
#
# Size max_iter from a short probe: watch the first "Iteration N ... Time: Xs"
# lines, then max_iter ~= wall_budget_seconds / s_per_iter.
# (reference: RTX 5070 12GB was ~20 s/iter at batch 4 -> 600 iters ~ 3.5 h)
# VRAM ladder if OOM: batch 4 -> model.network.visual_encoder.checkpoint_activations=true -> batch 2
set -euo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$EXP/scripts/run_container.sh"

OVERRIDES=""
if [[ -n "${1:-}" ]]; then
  OVERRIDES="train.max_iter=$1 train.optim.lr_decay_iters=$1"
fi
OVERRIDES="$OVERRIDES ${2:-}"

"$RUN" "cosmos-embed1 train -e /specs/train.yaml results_dir=/results/train $OVERRIDES"

echo "PHASE 4 DONE — checkpoint: workspace/results/train/train/checkpoints/$(cat "$EXP/workspace/results/train/train/checkpoints/latest_checkpoint.txt")"
