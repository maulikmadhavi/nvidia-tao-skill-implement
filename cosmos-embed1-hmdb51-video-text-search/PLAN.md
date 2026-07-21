# Cosmos-Embed1 Video-Text Search on HMDB51 — Experiment Spec

## Context

Fine-tune NVIDIA **Cosmos-Embed1** (joint video-text embedding model) for **video-text search** using
HMDB51 action-recognition data: curate a sampled dataset, measure a **zero-shot baseline**, fine-tune on
**80%** / test on **20%**, produce full metrics (accuracy, precision, recall, F1, confusion matrix,
Recall@K, logged matching scores), save the final model, and deliver a **deployable search script + model**
and a **demo over test data only**. First artifact: `PLAN.md` in the experiment dir.

Experiment root: `D:\tao-skill-bank\exp\cosmos-embed1-hmdb51-video-text-search\`

## Verified environment facts

| Item | Status |
|---|---|
| GPU | RTX 5070, **12GB VRAM** (Blackwell sm_120), driver 591.86 |
| RAM / disk | 32GB RAM, 804GB free on D: |
| Docker | Docker Desktop 29.2.1 + WSL2 — **daemon not running** (user starts it in Phase 0) |
| Host Python | 3.13, **no torch** → host scripts stdlib-only; torch work runs in-container |
| ffmpeg / 7-Zip | ffmpeg 7.1.1 ✓, 7-Zip 24.09 ✓ (.rar extract + .avi→.mp4 transcode) |
| HF token | **missing** — user must supply `HF_TOKEN` + accept `nvidia/Cosmos-Embed1-224p` terms on HF |

## Core approach

Reuse the repo's existing skill **`skills\models\tao-finetune-cosmos-embed`** (verified against its
SKILL.md and `references\spec_template_*.yaml`):

- Container `nvcr.io/nvidia/tao/tao-toolkit:7.0.1-cosmos-embed` (versions.yaml `tao_toolkit.cosmos_embed`);
  CLI `cosmos-embed1 train|evaluate|inference|export`; every action prefixed with
  `python -m pip install "protobuf<7"` (known wandb pitfall).
- Variant **Cosmos-Embed1-224p** (8 frames, 224², embed dim 256); **LoRA** fine-tune
  (`model.lora.enabled=true`, `transformer_engine=false`, `fsdp_shard_size=1`), frozen ViT-g visual encoder.
- Zero-shot baseline = `evaluate` with `evaluate.checkpoint: null` (pretrained weights).
- Data format: MSR-VTT style — `/data/video/*.mp4` glob + JSON rows `{"video_id","video","caption"}`.
- Final model via `export.mode=huggingface` (+ optional ONNX).
- Docker deltas for this host: `--shm-size=16g` (not 64g), **no** `--network=host` (use `-p 7860:7860` for demo).

We write what the repo lacks: HMDB51 download/sample/split/transcode/metadata scripts (host, stdlib),
custom metrics + score logging (in-container), deploy CLI, Gradio demo.

## Directory layout & scripts

```
exp\cosmos-embed1-hmdb51-video-text-search\
├── PLAN.md                          # this spec (Step 0)
├── scripts\
│   ├── 00_download_hmdb51.py        # host: hmdb51_org.rar (~2.1GB) + test_train_splits.rar from Serre Lab; resume + size check
│   ├── 01_extract_hmdb51.py         # host: 7z two-stage (outer rar → 51 class rars → .avi); gate: 51 classes, ~6,766 avi
│   ├── 02_sample_and_split.py       # host: seed 42; 51×30=1,530 clips; per class 22 train/2 val/6 test → 1,122/102/306;
│   │                                #   sanitized ids {class}_{0001..}; 51-entry gerund caption map;
│   │                                #   → split_manifest.csv + class_prompts.json
│   ├── 03_transcode.py              # host: ffmpeg .avi→.mp4 (scale=-2:256, libx264, yuv420p, crf 20, -an); ffprobe-verify all
│   ├── 04_make_metadata.py          # host: train/val/test.json (MSR-VTT rows) + caption_to_label.json;
│   │                                #   hard gate: every metadata `video` exists on disk
│   ├── 10_extract_embeddings.py     # container: HF AutoModel(trust_remote_code) baseline=HF repo / finetuned=exported dir;
│   │                                #   8 uniform frames per test mp4 → L2-normed video embs [306×256] + 51 class-prompt embs → NPZ
│   ├── 11_compute_metrics.py        # container: pip install scikit-learn matplotlib; all metrics + score logs from NPZ
│   ├── 12_compare_phases.py         # host stdlib: baseline vs finetuned table → reports\comparison.md; cross-check vs container topk
│   └── run_container.ps1            # canonical docker run wrapper (single source of flags)
├── workspace\
│   ├── raw\                         # rars + extracted .avi (not mounted)
│   ├── data\                        # → /data:ro   (video\*.mp4, *.json, manifests)
│   ├── specs\                       # → /specs:ro  (6 spec YAMLs)
│   ├── results\                     # → /results   (train/evaluate/export/metrics)
│   ├── model\                       # → /model
│   └── hf_cache\                    # → /hf_cache  (persistent HF downloads)
├── reports\                         # final metrics CSV/PNG/JSON + comparison.md
├── deploy\  (search_cli.py, run_search.ps1, model\, README.md)
└── demo\    (app.py, run_demo.ps1, README.md)
```

`run_container.ps1` runs:
`docker run --rm --gpus all --ipc=host --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864
-e HF_TOKEN -e WANDB_DISABLED=true -e WANDB_MODE=disabled -e HUGGINGFACE_HUB_CACHE=/hf_cache
-v <workspace mounts> -v <exp>\scripts:/exp/scripts:ro -v <exp>\deploy:/exp/deploy:ro -v <exp>\demo:/exp/demo:ro
<image> bash -lc "python -m pip install 'protobuf<7' && <Cmd>"` (+ `-p` only for demo).

## Data pipeline

- **Sources**: `http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar` (+ splits rar for
  provenance only — we do our own stratified 80/20, seed 42).
- **Sanitized filenames** are mandatory (HMDB names contain `#;( ` etc.); loader matches metadata `video`
  against mp4 filenames exactly.
- **Captions**: one gerund caption per class, template `"a video of a person {gerund}"`
  (`brush_hair` → "a video of a person brushing hair"). **Invariant**: identical strings in metadata
  captions, `caption_to_label`, `class_prompts.json`, and the demo — drift silently breaks topk eval.
- Val (2/class) is carved from the 80% train side for the training-time validation callback; test (6/class)
  is untouched until evaluation.

## Spec files (deltas from repo templates; field names verified)

- **train.yaml**: seed 42; `max_iter: 2000` (~7 epochs @ batch 4 over 1,122 clips); `validation_iter: 250`;
  `checkpoint_iter: 500`; optim adamw `lr: 1e-4` (LoRA ≈ 10× the full-FT template lr), cosine,
  `warmup_steps: 100`, `lr_decay_iters: 2000`; bf16; `freeze_visual_encoder: true`;
  `use_captioning_loss: false` (51 template captions carry no captioning signal; saves VRAM);
  `model.pretrained_model_path: nvidia/Cosmos-Embed1-224p`; LoRA rank 8 / alpha 16 / dropout 0.1;
  train/val datasets → `/data/train.json` / `/data/val.json`, batch 4/8.
  VRAM ladder: batch 4 → `checkpoint_activations: true` → batch 2. If smoke shows <10GB peak: batch 8, `max_iter: 1000`.
- **evaluate_zeroshot.yaml**: `checkpoint: null`; `pretrained_model_path: nvidia/Cosmos-Embed1-224p`;
  `save_dataset_pkl`; callbacks `topk_classification: true`, `top_k_values: [1,3,5,10]`;
  `test_dataset` → `/data/test.json`, batch 8, `caption_to_label: {51 entries}` (generated by 04).
  Run with `results_dir=/results/evaluate_zeroshot`.
- **evaluate_finetuned.yaml**: same + `model.lora` block (adapter-checkpoint insurance);
  checkpoint passed at CLI as the exact resolved `/results/train/checkpoints/iter_#########.pt`
  (from `latest_checkpoint.txt`); `results_dir=/results/evaluate_finetuned`.
- **inference.yaml**: `inference.k: 10`, `save_dataset_pkl` (search DB cache), test metadata; sanity checks.
- **export_hf.yaml**: `export.hf_output_dir: /results/export_hf/cosmos_embed1_hmdb51_hf`, `on_cpu: true`,
  + pretrained path + lora block; checkpoint via CLI.
- **export_onnx.yaml** (optional): `mode: combined`, opset 17.

## Metrics protocol (per phase: baseline, finetuned — identical 306-clip test set)

Two paths: (1) container `evaluate` → official `metrics.json` (top-k classification via caption_to_label);
(2) custom `10_extract_embeddings.py` → `11_compute_metrics.py` — authoritative for everything else, and for
the finetuned phase it runs on the **exported HF dir**, validating the exact deploy artifact.

From `S = video_embs @ class_prompt_embs.T` (cosine, 306×51):
- Zero-shot classification: argmax → **accuracy**, macro+weighted **precision/recall/F1**, per-class P/R/F1/support CSV, **51×51 confusion matrix** (CSV + PNG heatmap).
- **Video→text R@{1,3,5,10}** (≡ top-k classification accuracy here — stated in report).
- **Text→video R@{1,5,10}** multi-positive (hit iff ≥1 of the 6 relevant clips in top-K), **mAP**, median rank.
- **Score logs**: full similarity matrix → `sim_video_x_class.csv` + `similarity_scores.npz`; `retrieval_log.csv` (top-10 classes + scores per video).
- `12_compare_phases.py`: baseline-vs-finetuned Δ table → `reports\comparison.md`; cross-check custom top-k vs container metrics.json (flag |Δ|>0.01). Container caption-R@K with duplicate captions is ill-defined → custom multi-positive numbers are the reported retrieval results.

## Execution phases & gates

0. **Env**: create exp tree + PLAN.md (Step 0); user starts Docker Desktop; pull image (20+GB; `docker login nvcr.io` with NGC key if needed); user sets `HF_TOKEN` + accepts model terms; **sm_120 gate**: in-container torch CUDA matmul smoke — stop/escalate if Blackwell kernels missing.
1. **Data**: run 00→04. Gates: 51 classes / ~6,766 avi; manifest 1,530 = 1,122+102+306 (22/2/6 per class); ffprobe passes all; zero metadata↔file mismatches.
2. **Zero-shot baseline**: container evaluate (first run downloads weights into /hf_cache) → gate: metrics.json over 306 samples, top-1 ≫ 2% chance; then custom baseline metrics.
3. **Smoke chain**: 1-iter train (SKILL.md smoke overrides, `results_dir=/results/train_smoke`) → evaluate on 8 samples → export HF → `AutoModel.from_pretrained` load. Proves LoRA-ckpt→eval→export→HF-load chain before real training. Watch VRAM peak.
4. **Real train**: 2,000 iters (~30–90 min). Gates: loss trending down, val retrieval improving, checkpoint + `latest_checkpoint.txt` written.
5. **Finetuned eval + export**: evaluate with exact ckpt; export HF; custom metrics on exported dir; compare. Gate: finetuned ≥ baseline on top-1 / t→v R@1 (else drop lr to 5e-5 / adjust epochs). Optional ONNX.
6. **Deploy**: copy exported HF dir → `deploy\model\`; `search_cli.py index` (embeds test clips → index.npz) and `search --query ... --topk` (encodes text live, cosine top-k, table/JSON with scores); gate: "riding a bike" query returns `ride_bike_*` at rank 1.
7. **Demo**: Gradio `app.py` in-container (`pip install gradio` at launch), loads deploy model + precomputed index; textbox + top-k slider → `gr.Video` gallery with class + score (`allowed_paths=["/data/video"]`, `server_name=0.0.0.0`, `-p 7860:7860`); gate: 3 spot-check queries correct in host browser.
8. **Report**: finalize `reports\comparison.md` + summary; update PLAN.md status.

## Risks

1. **Blackwell sm_120 vs TAO 7.0.1 container torch** — gated first (Phase 0); fallback: newer image tag or Brev cloud dispatch for training only.
2. In-batch false negatives from class-level captions (~11% of batches at batch 4) — accepted, documented; don't raise batch size without noting this.
3. LoRA adapter-only checkpoint at export/evaluate — mitigated via lora block in specs + Phase 3 smoke of full chain.
4. Serre Lab download flaky — resume + URL override + manual mirror fallback.
5. Gradio not in image — pinned pip install at launch; fallback: host UI shelling out to search CLI.
6. 12GB shared with display (~11GB usable) — VRAM ladder above.
7. Exported-model processor API contract — confirmed at Phase 3 smoke; one shared embedding helper for 10_/search_cli.

## Critical reference files

- `skills\models\tao-finetune-cosmos-embed\SKILL.md` — run pattern, protobuf preamble, LoRA/FSDP, smoke overrides, pitfalls
- `skills\models\tao-finetune-cosmos-embed\references\spec_template_{train,evaluate,inference,export_hf,export_onnx}.yaml` — spec bases
- `versions.yaml` — image key `tao_toolkit.cosmos_embed`
- `skills\applications\tao-finetune-huggingface-model\examples\convnext-tiny-cifar10\run_eval.py` — reusable eval-report pattern

## Verification

- Every phase has an explicit gate (listed above); nothing proceeds past a failed gate.
- End-to-end proof = Phase 3 smoke chain (train→evaluate→export→HF-load) before any expensive run.
- Final acceptance: `reports\comparison.md` shows baseline vs finetuned metrics on the same 306-clip test set; demo at `http://localhost:7860` retrieves correct-class test videos for spot-check queries; `deploy\model\` + `deploy\search_cli.py` run standalone via `run_search.ps1`.
- Also: add `exp/` to `D:\tao-skill-bank\.gitignore` so experiment artifacts never pollute the skill-bank repo.

## Status

- [x] Phase 0 - Environment — image pulled (no NGC auth needed), sm_120 GATE PASSED (torch 2.7.0+cu12.9, capability (12,0), bf16 conv3d ok). HF token not needed (model ungated).
- [x] Phase 1 - Data pipeline (download/extract/sample/transcode/metadata) — 1530 clips: 1122/102/306, all gates passed. NOTE: Serre Lab URLs are dead; used HF mirror jili5044/hmdb51 (hmdb51.zip). Model is NOT gated — no HF_TOKEN required.
- [x] Phase 2 - Zero-shot baseline — container: top-1 66.01%, top-5 89.22%, MAP 69.9%; custom: acc 65.69%, macro-F1 62.4%, t2v R@1 82.4%, t2v mAP 70.1%. Cross-check delta 0.3% (pass).
- [x] Phase 3 - Smoke chain — 1-iter train OK (LoRA 9.8M/1.18B trainable), eval-with-ckpt OK, export HF merged LoRA + parity cosine 1.0, HF load+embed OK. FIX FOUND: export omits processor_config.json (processor defaults to 448p) — always copy the 224p snapshot's processor_config.json into exported dirs.
- [x] Phase 4 - LoRA training — rescoped 2000→600 iters (user-approved; compute-bound ~20s/iter on RTX 5070). Completed 600/600 in ~3.5h; final val: top-1 78.43% (vs 66.67% zero-shot), MAP 87.97%.
- [x] Phase 5 - Finetuned eval + export + comparison — TEST SET (306 clips): acc 77.45% (+11.8), macro-F1 76.4 (+14.0), t2v R@1 94.1 (+11.8), t2v mAP 82.7 (+12.6). HF export parity cosine 1.0; container cross-check deltas <0.4%. reports/comparison.md written.
- [x] Phase 6 - Deploy — exported HF model in deploy\model; index over 306 test clips; gate: bike query 5/5 correct at ranks 1-5, free-text paraphrase 3/3.
- [x] Phase 7 - Web demo (test data only) — gradio impossible in container (pip ResolutionImpossible on pinned httpx/httpcore) → zero-dependency stdlib server at http://127.0.0.1:7860 (IPv4 only; Docker Desktop drops ::1). Gate: 3 paraphrase queries all correct + video streaming OK.
- [x] Phase 8 - Final report — reports\final_report.md + comparison.md. TEST RESULT: acc 65.7%→77.5%, macro-F1 62.4→76.4, t2v mAP 70.1→82.7.
