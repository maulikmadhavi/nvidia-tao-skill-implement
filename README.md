# nvidia-tao-skill-implement

Implementation archive of NVIDIA TAO **Cosmos-Embed1** skill work — scripts, specs,
docs, and result summaries only. **No datasets, model weights, or large binaries are
stored here** (see `.gitignore`); every heavy artifact is regenerable from these
scripts plus the model snapshot fetched at build time.

## Experiments

### `cosmos-embed1-driving-distracted/`
Single-label 7-way classification of ~3s dashcam clips (6 distracted-driving
violations + `no_violation`) on the frozen Cosmos-Embed1-224p encoder.

| tier | val accuracy | macro-F1 |
|---|---|---|
| Zero-shot (cosine to text prompts → argmax) | 94.5% | 0.885 |
| Linear probe (LogReg on frozen embeddings) | **98.9%** | **0.977** |

Split: stratified 80/20 → 724 train / 181 val. The linear probe essentially solves
the task; encoder LoRA is unnecessary. Result summaries live in
`workspace/results/baseline/` (metrics JSON, per-class CSV, confusion PNG/CSV).

### `cosmos-embed1-driving-violations/`
Air-gapped kit for multi-label violation detection on full driving-lesson videos:
zero-shot → per-violation heads on the frozen encoder → LoRA (only if heads plateau),
with glue post-processing (median-smoothed trajectories → tuned thresholds →
hysteresis → event segments). Includes a synthetic self-test and offline Docker
build. See its `README.md`, `MIGRATION.md`, `RUNBOOK.md`.

### `cosmos-embed1-hmdb51-video-text-search/`
HMDB51 video-text retrieval / zero-shot action classification with Cosmos-Embed1;
baseline vs. LoRA-finetuned comparison. Reports in `reports/`.

## Running

Each experiment folder has its own README with the exact commands. The pipelines run
Cosmos-Embed1 inside the TAO cosmos-embed Docker image; the model snapshot and raw
data are provided locally (not in this repo).

## What is intentionally excluded
Model weights (`*.safetensors`, `*.pth`, `*.pt`, `*.bin`, `*.onnx`), videos
(`*.mp4`, `*.avi`), embedding caches (`*.npz`), Docker image tars, and the
`workspace/{model,data,hf_cache}` + `bundle/` trees.
