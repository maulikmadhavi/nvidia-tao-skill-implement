# Cosmos-Embed1 video-text search — 7-step reproducible pipeline

Experiment root: `D:\tao-skill-bank\exp\cosmos-embed1-hmdb51-video-text-search\`
Engine: `nvcr.io/nvidia/tao/tao-toolkit:7.0.1-cosmos-embed` — every container step goes through
`scripts\run_container.ps1` (mounts `workspace\{data,specs,results,model,hf_cache}` as `/data /specs /results /model /hf_cache`, plus `scripts deploy demo` under `/exp/`).

## Step 1 — Data collection

| | |
|---|---|
| Run | `python scripts\00_download_hmdb51.py --out workspace\raw` then `python scripts\01_extract_hmdb51.py --raw workspace\raw` |
| Input | Dataset URL (HF mirror `jili5044/hmdb51/resolve/main/hmdb51.zip`; original Serre URLs are dead) |
| Output | `workspace\raw\hmdb51_extracted\<class>\*.avi` (6,766 clips) + `workspace\raw\class_dirs.json` (class → dir map) |
| Gate | 51 class dirs, ≥6,000 clips; downloads <1 GiB rejected (error-page guard) |

## Step 2 — Data preparation

| | |
|---|---|
| Run | `02_sample_and_split.py --raw workspace\raw --out workspace\data --seed 42 --per-class 30` → `03_transcode.py` → `04_make_metadata.py` |
| Input | `class_dirs.json` + raw `.avi` + the 51-entry caption map (inside `02_`, template "a video of a person {gerund}") |
| Output | `workspace\data\split_manifest.csv` (video_id, class, label, split, caption, source path — **single source of truth**, seed 42, 22/2/6 per class = 1,122 train / 102 val / 306 test); `workspace\data\video\*.mp4` (1,530, h264 short-side-256, ffprobe-verified); `workspace\data\{train,val,test}.json` (MSR-VTT rows `{"video_id","video","caption"}`); `caption_to_label.json`, `class_prompts.json`; rendered `workspace\specs\evaluate_{zeroshot,finetuned}.yaml` |
| Gate | exact 1,530 = 1,122+102+306; every metadata `video` exists on disk (loader silently finds "0 videos" on mismatch) |

Invariant: the exact caption strings must be identical in train metadata, `caption_to_label`, `class_prompts.json`, and the demo.

## Step 3 — Zero-shot baseline

| | |
|---|---|
| Run | `run_container.ps1 -Cmd "cosmos-embed1 evaluate -e /specs/evaluate_zeroshot.yaml results_dir=/results/evaluate_zeroshot"` then `10_extract_embeddings.py --model nvidia/Cosmos-Embed1-224p --phase baseline` + `11_compute_metrics.py --phase baseline` |
| Input | `test.json` + pretrained weights (`evaluate.checkpoint: null` = zero-shot; weights ungated, auto-downloaded to `/hf_cache`) |
| Output | `workspace\results\evaluate_zeroshot\evaluate\metrics.json` (container top-k) + `workspace\results\metrics\baseline\` — metrics.json, per_class_metrics.csv, confusion_matrix.{csv,png}, **sim_video_x_class.csv / similarity_scores.npz / retrieval_log.csv** (all matching scores) |
| Result | top-1 65.7%, top-5 89.2%, macro-F1 62.4, t2v mAP 70.1 |

## Step 4 — Training parameter setting

| | |
|---|---|
| Input | Skill template `skills\models\tao-finetune-cosmos-embed\references\spec_template_train.yaml` |
| Output | `workspace\specs\train.yaml` |

Key choices for 12GB VRAM: variant 224p; **LoRA** r=8 α=16 dropout=0.1 (9.8M/1.18B trainable), `freeze_visual_encoder: true`, `transformer_engine: false`, `fsdp_shard_size: 1`; batch 4 (VRAM ceiling — batch 8 OOMs); lr **1e-4** (≈10× the full-FT default, LoRA rule of thumb), cosine, warmup 50; `use_captioning_loss: false` (template captions carry no captioning signal); **max_iter = wall-clock budget ÷ measured s/iter** — measured ~20 s/iter → 600 iters (~2 epochs, 3.5h). Always keep `lr_decay_iters == max_iter`.

## Step 5 — Training

| | |
|---|---|
| Run | First a 1-iter **smoke chain** (train smoke overrides → evaluate 8 samples → export HF → HF load) — this caught the processor-config export bug before the real run. Then `run_container.ps1 -Cmd "cosmos-embed1 train -e /specs/train.yaml results_dir=/results/train"` |
| Input | `train.yaml`, `train.json`/`val.json`, `/data/video/*.mp4` |
| Output | `workspace\results\train\train\checkpoints\iter_000000600.pt` (+ `latest_checkpoint.txt`; the `.pth` symlink is 0 bytes through Windows mounts — always pass the exact iter file); val metrics every 150 iters in console.log |
| Gate | loss trending down; val top-1 66.7% → 78.4% |

## Step 6 — Evaluation + deploy

| | |
|---|---|
| Run | evaluate (ckpt) → export HF → **copy `processor_config.json` from the 224p snapshot into the exported dir** (export omits it; processor otherwise defaults to 448p and crashes) → `10_/11_ --phase finetuned` **on the exported dir** → `12_compare_phases.py` → copy model to `deploy\model\` → `deploy\run_search.ps1 -Index` |
| Input | `iter_000000600.pt`, `test.json` |
| Output | `workspace\results\export_hf\cosmos_embed1_hmdb51_hf\` (LoRA-merged, parity cosine 1.0) → `deploy\model\`; `workspace\results\metrics\finetuned\*`; `reports\comparison.md` + `reports\final_report.md`; `workspace\results\deploy\index.npz` (306 test embeddings) |
| Result | acc 77.5% (+11.8), macro-F1 76.4 (+14.0), t2v R@1 94.1, mAP 82.7; search gate: bike query 5/5, paraphrase 3/3 |

Deployable = `deploy\model\` + `deploy\search_cli.py` (`index`/`search`) + `deploy\run_search.ps1`.

## Step 7 — Demo (test data only)

| | |
|---|---|
| Run | `demo\run_demo.ps1` → open **http://127.0.0.1:7860** (IPv4 only; Docker Desktop drops `::1`) |
| Input | `deploy\model\` + `index.npz` (precomputed; only the query text is embedded live) |
| Output | Web UI: free-text search → top-k playable test clips with cosine scores. Zero-dependency stdlib server (`pip install gradio` is impossible in this container — pinned httpx/httpcore) |

---

# Next experiment: driving-lesson violation detection — design changes

Task shift: 51-way balanced **classification** → **multi-label detection with rare positives** over 10-sec chunks
(labels like `phone_use`, `no_seatbelt`, …, plus the dominant `no_violation`).

## Challenge 1 — very few positives (most chunks are no-violation)

- **Never rebalance the test set.** Keep every positive; split **stratified per violation type at the source-video level** (all chunks of one lesson/video stay on one side — chunks from the same recording are near-duplicates and leak).
- **Training set**: keep all positives, downsample negatives to ~3:1–5:1 neg:pos. **Test set**: true distribution.
- **Multiply positives instead of collecting more**: temporal jitter — cut several overlapping 10s windows around each violation event (e.g., ±2s shifts) → 3–5× positives for free. (Step 2 owns this; the manifest gets an `event_id` column so all windows of one event stay in one split.)
- **Metrics change (Step 3/6)**: accuracy is meaningless at high imbalance. Report **per-class PR-AUC / AP**, precision@fixed-recall (e.g., P@R=0.95 for safety-critical classes), and F1 at tuned thresholds. Keep the score logs — they're what you tune thresholds on.
- **Captions**: one per violation, driver-centric and concrete: "the driver is using a mobile phone", "the driver is not wearing a seatbelt", plus an explicit negative prompt "the driver is driving normally with no violation" (gives the contrastive loss a real negative anchor and the detector a calibration reference).

## Challenge 2 — one 10s chunk can carry multiple violations

- **Prediction rule changes**: argmax over prompts is wrong. Score each class independently — `s_c = cos(video_emb, prompt_c)` — and predict **every class with `s_c ≥ θ_c`**, where each per-class threshold θ_c is tuned on the val split (maximize F1, or fix recall). `no_violation` = no class above threshold (or its own prompt wins).
- **Training metadata**: emit **one MSR-VTT row per (chunk, violation) pair** — same `video`, different `caption`; a chunk with phone+seatbelt appears twice. Smoke-gate this first (1-iter train) to verify the loader accepts duplicate video ids with different captions; if it dedupes, fall back to `random_caption: true` with a per-chunk caption list, or a combined caption ("… using a phone and not wearing a seatbelt") as last resort.
- **In-batch false negatives get worse** (two chunks sharing a violation class in one batch): keep batch small, or sample batches with distinct classes.
- **Evaluation code (Step 6)**: swap `11_compute_metrics.py`'s argmax block for a multi-label variant — labels become a binary matrix [N×C]; outputs: per-class AP + PR curves, mAP, micro/macro F1 at θ_c, exact-match ratio; the confusion matrix becomes a **co-occurrence error matrix** (which violations get confused/missed together).
- **Container `topk_classification` assumes single-label** — its metrics.json is not meaningful here; the custom metrics path (scripts 10/11) is the authoritative one. Everything else (specs, train action, export, deploy CLI, demo) carries over unchanged.
- **8 frames over 10s may miss brief events** (a 1-2s phone glance): embed 2–4 overlapping 8-frame windows per chunk and **max-pool the per-class scores** — a one-line change in `embed_lib.embed_video`, reused by eval, deploy, and demo.

## What carries over unchanged

Steps 1/2 scripts (swap the source + caption map), Step 4 spec (same LoRA/VRAM logic), Step 5 command,
export + processor-config fix, `deploy\search_cli.py`, the stdlib demo, and all the container gotchas in
`reports\final_report.md`.
