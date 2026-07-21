# RUNBOOK — Driving-lesson violation detection (air-gapped)

Everything below runs WITHOUT internet. The server is assumed Ubuntu/Linux.
Every phase ends with a GATE — do not continue past a failed gate.

**Fast path — one bash script per phase** (each is exactly the commands of its
section below; all syntax-gated under the container's Linux bash):

```bash
./scripts/phase1_prepare.sh /path/to/lesson_videos /path/to/annotations.csv
./scripts/phase2_zeroshot_baseline.sh
./scripts/phase3_smoke.sh
./scripts/phase4a_heads.sh           # heads on the frozen encoder (seconds of CPU after scoring)
./scripts/phase4_train.sh            # 4b LoRA — ONLY if heads plateau; or phase4_train.sh <max_iter>
./scripts/phase5_finetuned_eval.sh   # prints the {baseline,finetuned}x{isolated,glue} 2x2
TAG=heads_v2 SCORES=finetuned MODEL=/results/export_hf/cosmos_embed1_driving_hf \
  ./scripts/phase4a_heads.sh         # 5b: refit heads on the fine-tuned encoder (seconds)
./demo/run_demo.sh finetuned glue    # http://127.0.0.1:7860
```

The sections below explain what each phase does, its gates, and the individual
commands (for debugging or partial re-runs). `<EXP>` = this folder.

The whole pipeline was validated end-to-end on the build machine via
`selftest/run_selftest.ps1` (synthetic lessons; glue recovered all planted events,
micro event-F1 0.86 zero-shot), and the offline image passed a `--network none` gate.

---

## Phase 0 — One-time setup

Follow `MIGRATION.md` (docker load, copy kit, copy `bundle/model` → `workspace/model`,
GPU gate, offline gate). GATE: both gate scripts print `GATE PASSED`.

## Phase 1 — Data preparation

Inputs you provide:
- `~100` lesson videos in one directory (any ffmpeg-readable format)
- `annotations.csv` per `annotations_schema.md` (video, class, start_sec, end_sec)

```bash
cd <EXP>
python3 scripts/00_chunk_videos.py \
  --videos /path/to/lesson_videos --annotations /path/to/annotations.csv \
  --out workspace/data --seed 42
cp /path/to/annotations.csv workspace/data/annotations.csv
python3 scripts/01_make_metadata.py --data workspace/data \
  --templates specs --specs workspace/specs
```

What this does: transcodes full videos into `workspace/data/full/` (256px h264);
splits at VIDEO level, stratified by violation content (~72 train / 8 val / 20 test);
labels 2s chunks (overlap >= 0.5s, multi-label); train split keeps all positives +
±0.5s jitter copies and downsamples negatives to 4:1; cuts train/val/test chunk mp4s;
emits `chunk_manifest.csv`, `{train,val,test}.json`, `{split}_videos.txt`, `prompts.json`,
and renders `workspace/specs/`.

GATES (printed): every event covered by >= 1 chunk (watch for the WARN about
too-short events); per-split video/chunk/positive counts look sane; 01 confirms all
chunk mp4s exist. Sanity-check `workspace/data/chunk_manifest.csv` by eye: a few
known events should map to correctly-labeled chunks.

## Phase 2 — Zero-shot baseline (both inference modes)

```bash
# 2.1 score matrices for val+test videos (the expensive step; ~1-2 min per lesson)
cat workspace/data/val_videos.txt workspace/data/test_videos.txt > workspace/data/valtest_videos.txt
./scripts/run_container.sh "python /exp/scripts/10_infer_chunks.py \
  --model /model/Cosmos-Embed1-224p --videos /data/full --list /data/valtest_videos.txt \
  --prompts /data/prompts.json --out /results/scores/baseline"

# 2.2 tune glue params on VAL only
python3 scripts/14_tune_thresholds.py --scores workspace/results/scores/baseline \
  --annotations workspace/data/annotations.csv --videos workspace/data/val_videos.txt \
  --out workspace/results/thresholds_baseline.json

# 2.3 both modes -> events
python3 scripts/11_glue_postprocess.py --scores workspace/results/scores/baseline \
  --prompts workspace/data/prompts.json --thresholds workspace/results/thresholds_baseline.json \
  --mode isolated --out workspace/results/events/baseline_isolated
python3 scripts/11_glue_postprocess.py --scores workspace/results/scores/baseline \
  --prompts workspace/data/prompts.json --thresholds workspace/results/thresholds_baseline.json \
  --mode glue --out workspace/results/events/baseline_glue

# 2.4 metrics on TEST
python3 scripts/13_eval_event_level.py --events workspace/results/events/baseline_glue/events.json \
  --annotations workspace/data/annotations.csv --videos workspace/data/test_videos.txt \
  --label baseline/glue --out reports/event_eval_baseline_glue.json
python3 scripts/13_eval_event_level.py --events workspace/results/events/baseline_isolated/events.json \
  --annotations workspace/data/annotations.csv --videos workspace/data/test_videos.txt \
  --label baseline/isolated --out reports/event_eval_baseline_isolated.json
./scripts/run_container.sh "python /exp/scripts/12_eval_chunk_level.py \
  --scores /results/scores/baseline --annotations /data/annotations.csv \
  --videos /data/test_videos.txt --thresholds /results/thresholds_baseline.json \
  --phase baseline --out /results/chunk_eval_baseline"
```

GATE: chunk-level `mAP` well above the positive-rate floor; per-class AP table shows
which violations zero-shot already sees (expect trainer-vs-driver role confusion —
that is what fine-tuning fixes). Keep these numbers: they are the baseline row.

## Phase 3 — Smoke chain (cheap, before any real training)

```bash
./scripts/run_container.sh "cosmos-embed1 train -e /specs/train.yaml results_dir=/results/train_smoke \
  train.max_iter=1 train.validation_iter=2 train.checkpoint_iter=1 \
  train.optim.warmup_steps=0 train.optim.lr_decay_iters=1 \
  dataset.train_dataset.batch_size=2 dataset.val_dataset.batch_size=2 \
  dataset.train_dataset.workers=0 dataset.val_dataset.workers=0"
./scripts/run_container.sh "cosmos-embed1 export -e /specs/export_hf.yaml results_dir=/results/export_smoke \
  export.checkpoint=/results/train_smoke/train/checkpoints/iter_000000001.pt \
  export.hf_output_dir=/results/export_smoke/hf"
cp workspace/model/Cosmos-Embed1-224p/processor_config.json workspace/results/export_smoke/hf/
./scripts/run_container.sh "python /exp/scripts/gate_hf_load.py /results/export_smoke/hf /data/full/$(head -1 workspace/data/test_videos.txt).mp4"
```

GATE: train completes 1 iter with LoRA (`trainable% ~0.83`), checkpoint written,
export prints `Inference verification PASSED`, gate_hf_load prints `GATE PASSED`.
NOTE (verified on build machine): the multi-label metadata (duplicate mp4, different
captions) must load here — if the loader errors on duplicates, see "Fallbacks" below.

## Phase 4a — Per-violation heads on the frozen encoder

Design + decision log: `HEADS_DESIGN.md`. A head = logistic regression per
class on the window embeddings that `10` caches in every `scores_*.npz`
(V key). Fits in seconds on CPU; the encoder is untouched; all downstream
scripts run unchanged because `16` emits the exact `10` file format
(S = probabilities).

```bash
./scripts/phase4a_heads.sh    # after phase 2; TAG=heads_v1 SCORES=baseline by default
```

Steps inside: `10` on the train videos (completes the embedding cache in
`scores/baseline`) → `15` fits heads + picks C by val AP + computes the
unknown band → `16` writes probability scores + `unknown_*.csv` +
`review_queue.json` → `14` retunes glue on val with `--thr-grid 0.05:0.95:0.02`
(probabilities, not cosines) → `11` both modes → `13` + `12` on test →
comparison print (baseline vs heads_v1).

GATES: `15` prints the head-vs-prompt val AP table (heads should beat the
prompt on classes with enough positives) and warns on any class with < 10
positive train windows (keep the prompt score for those). `16` warns if a
class's unknown-rate exceeds 20% (band too wide to review). Keep
`review_queue.json` — it is the reviewer worklist AND the future feedback
record for refitting heads.

**Phase 4b decision rule:** run LoRA below ONLY if a class with >= 30
positive train windows still has test AP clearly below requirement after
heads. Otherwise heads_v1 is the shipping configuration — skip to Phase 6/7
(demo works on `scores/heads_v1` + `thresholds_heads_v1.json`).

## Phase 4b — LoRA fine-tuning (only if heads plateau)

Size the run first:

```bash
# measure ~20 iters, read the "Iteration N ... Time: Xs" lines, then set max_iter:
#   max_iter = your_wall_budget_seconds / measured_s_per_iter   (keep lr_decay_iters == max_iter)
# reference: RTX 5070 12GB was ~20 s/iter at batch 4 -> 600 iters ~ 3.5 h
./scripts/run_container.sh "cosmos-embed1 train -e /specs/train.yaml results_dir=/results/train"
```

VRAM ladder if OOM: batch 4 → `model.network.visual_encoder.checkpoint_activations=true` → batch 2.
GATE: loss trending down; `validation_eval` top-1 improving across checkpoints;
`workspace/results/train/train/checkpoints/iter_*.pt` + `latest_checkpoint.txt` written.

## Phase 5 — Fine-tuned evaluation (both modes) + comparison

```bash
CKPT=/results/train/train/checkpoints/$(cat workspace/results/train/train/checkpoints/latest_checkpoint.txt)

# export FIRST, then evaluate the exported artifact (what ships is what's measured)
./scripts/run_container.sh "cosmos-embed1 export -e /specs/export_hf.yaml results_dir=/results/export_hf export.checkpoint=$CKPT"
cp workspace/model/Cosmos-Embed1-224p/processor_config.json workspace/results/export_hf/cosmos_embed1_driving_hf/

./scripts/run_container.sh "python /exp/scripts/10_infer_chunks.py \
  --model /results/export_hf/cosmos_embed1_driving_hf --videos /data/full \
  --list /data/valtest_videos.txt --prompts /data/prompts.json --out /results/scores/finetuned"

# re-tune glue on val (fine-tuned scores have different scales), then repeat 2.3/2.4
python3 scripts/14_tune_thresholds.py --scores workspace/results/scores/finetuned \
  --annotations workspace/data/annotations.csv --videos workspace/data/val_videos.txt \
  --out workspace/results/thresholds_finetuned.json
python3 scripts/11_glue_postprocess.py --scores workspace/results/scores/finetuned \
  --prompts workspace/data/prompts.json --thresholds workspace/results/thresholds_finetuned.json \
  --mode glue --out workspace/results/events/finetuned_glue
python3 scripts/11_glue_postprocess.py --scores workspace/results/scores/finetuned \
  --prompts workspace/data/prompts.json --thresholds workspace/results/thresholds_finetuned.json \
  --mode isolated --out workspace/results/events/finetuned_isolated
python3 scripts/13_eval_event_level.py --events workspace/results/events/finetuned_glue/events.json \
  --annotations workspace/data/annotations.csv --videos workspace/data/test_videos.txt \
  --label finetuned/glue --out reports/event_eval_finetuned_glue.json
python3 scripts/13_eval_event_level.py --events workspace/results/events/finetuned_isolated/events.json \
  --annotations workspace/data/annotations.csv --videos workspace/data/test_videos.txt \
  --label finetuned/isolated --out reports/event_eval_finetuned_isolated.json
./scripts/run_container.sh "python /exp/scripts/12_eval_chunk_level.py \
  --scores /results/scores/finetuned --annotations /data/annotations.csv \
  --videos /data/test_videos.txt --thresholds /results/thresholds_finetuned.json \
  --phase finetuned --out /results/chunk_eval_finetuned"
```

GATE: fine-tuned >= baseline on per-class AP and event F1 (glue). If not: lower lr to
5e-5 or increase iters; check per-class — classes with < ~20 positive training chunks
may need more annotation or longer jitter augmentation before they move.

Final comparison = the four `reports/event_eval_*.json` + the two `chunk_eval_*/metrics.json`.
Report the 2x2: {baseline, finetuned} x {isolated, glue} — glue should win on event F1,
isolated is the per-chunk reference.

**Phase 5b — refit heads on the fine-tuned encoder (encoder v2).** LoRA changed
the embedding space, so v1 heads must NEVER be applied to finetuned scores
(`16` refuses — encoder check). Refit is seconds:

```bash
TAG=heads_v2 SCORES=finetuned MODEL=/results/export_hf/cosmos_embed1_driving_hf \
  ./scripts/phase4a_heads.sh
```

Then the full comparison is baseline / heads_v1 / finetuned / heads_v2.

## Phase 6 — Deploy artifact

`workspace/results/export_hf/cosmos_embed1_driving_hf/` (with the copied
`processor_config.json`) is the deployable model — load anywhere with
`AutoModel.from_pretrained(dir, trust_remote_code=True)`. Production inference =
`10_infer_chunks.py` (scores) + `11_glue_postprocess.py` (events) with
`thresholds_finetuned.json` — the same two scripts, unchanged.

## Phase 7 — Review demo (test videos only)

```bash
./demo/run_demo.sh finetuned glue     # then open http://127.0.0.1:7860
```

Per video: score trajectories per class (SVG), detected event bars, click the
timeline to seek the video. GATE: spot-check a few annotated events — the trajectory
should visibly rise inside the annotated window.

---

## Fallbacks / troubleshooting

| Symptom | Action |
|---|---|
| Loader rejects duplicate `video` rows (multi-label) at Phase 3 | Set `random_caption: true` stays; replace duplicate rows by ONE row per chunk with the rarest violation's caption (script change in 01: pick `labels.split(';')[0]` after sorting by class frequency). Expect slightly weaker multi-label training signal. |
| CUDA OOM in training | batch 4 → `checkpoint_activations=true` → batch 2 (see Phase 4b). |
| `0 videos found` from the msrvtt loader | metadata `video` filenames must exactly match files under `/data/video` — rerun 01 and check its gate. |
| Events shorter than 1s never detected | they never overlap a chunk by 0.5s — re-annotate to >= 1s or lower `--min-overlap`. |
| Glue misses brief real events but flags long false ones | per-class `window`/`min_consec` too aggressive — retune 14 with a wider grid, or set the class's `kind` defaults in `taxonomy.py`. |
| Demo unreachable | use `http://127.0.0.1:7860`, never `localhost` (IPv6 proxy issue on Docker Desktop). |
| ffmpeg/ffprobe hang at 0% CPU in background shells | already mitigated (`stdin=DEVNULL` + timeouts); do the same for any new subprocess call. |

## Class imbalance guidance (recap)

- Never rebalance val/test — tune thresholds on val, report on the true test distribution.
- Headline metrics: per-class AP (chunk level) and event F1 @ IoU 0.3 (glue). Accuracy is meaningless here.
- To grow rare positives: annotate more events of that class, or widen jitter (`--jitter 1.0`
  gives ±1s copies). Keep an eye on `support` in `chunk_eval_*/metrics.json`.
