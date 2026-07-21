# Final report — Cosmos-Embed1 video-text search on HMDB51

Date: 2026-07-17 · Hardware: RTX 5070 12GB (Blackwell sm_120), Windows 11 + Docker Desktop/WSL2
Engine: `nvcr.io/nvidia/tao/tao-toolkit:7.0.1-cosmos-embed` via the repo skill `tao-finetune-cosmos-embed`

## Headline result (306-clip held-out test set, 51 classes)

| metric | zero-shot baseline | fine-tuned (LoRA, 600 iters) | delta |
|---|---:|---:|---:|
| accuracy (top-1) | 65.7% | **77.5%** | **+11.8** |
| macro precision | 64.8% | 79.6% | +14.8 |
| macro recall | 65.7% | 77.5% | +11.8 |
| macro F1 | 62.4% | **76.4%** | **+14.0** |
| video→text R@5 | 89.2% | 95.1% | +5.9 |
| text→video R@1 (multi-positive) | 82.4% | **94.1%** | +11.8 |
| text→video mAP | 70.1% | **82.7%** | +12.6 |

Full table + container cross-checks (deltas <0.4%): `comparison.md`.
Confusion matrices: `confusion_matrix_{baseline,finetuned}.png` (+ CSVs next to each phase's metrics).
Per-class P/R/F1: `per_class_metrics_{baseline,finetuned}.csv`.
All matching scores logged per phase: `workspace\results\metrics\<phase>\{sim_video_x_class.csv, similarity_scores.npz, retrieval_log.csv}` (306×51 similarity matrix + per-video top-10 with scores).

## Experiment setup

- **Data**: HMDB51 (via HF mirror `jili5044/hmdb51` — original Serre Lab URLs are dead). Sampled
  30 clips/class × 51 classes = 1,530, seed 42, stratified **1,122 train / 102 val / 6-per-class = 306 test (80/20)**.
  Clips transcoded `.avi`→h264 mp4 (short side 256), all ffprobe-verified. Captions templated per class
  ("a video of a person riding a bike"); identical strings used in training metadata, evaluation
  `caption_to_label`, and the demo.
- **Model**: `nvidia/Cosmos-Embed1-224p` (ungated), LoRA rank 8 / alpha 16 / dropout 0.1 on ViT + Q-Former +
  projections (9.8M trainable / 1.18B total, 0.83%), visual encoder frozen, bf16.
- **Training**: 600 iterations, batch 4, AdamW lr 1e-4 cosine (decay 600, warmup 50) — rescoped from 2,000
  after measuring ~20s/iter compute-bound on the 5070 (user-approved). Wall time ≈ 3.5h.
  Val top-1 improved 66.7% → 78.4% during training.
- **Zero-shot protocol**: `evaluate` with `checkpoint: null` (pretrained weights) on the same test set;
  custom metrics computed from L2-normalized embeddings via the HF API (`get_video_embeddings` /
  `get_text_embeddings`, cosine similarity).

## Deliverables

- **Saved model**: `deploy\model\` — LoRA-merged, HF format (`AutoModel.from_pretrained(dir, trust_remote_code=True)`);
  export parity vs the training checkpoint verified at cosine 1.000000. Raw checkpoint:
  `workspace\results\train\train\checkpoints\iter_000000600.pt`.
- **Deployable search CLI**: `deploy\search_cli.py` (`index` / `search`) + `deploy\run_search.ps1`.
  Gate: "a video of a person riding a bike" → 5/5 ride_bike clips; free-text paraphrase
  "someone swinging a baseball bat" → 3/3 swing_baseball.
- **Demo (test data only)**: `demo\app.py` — stdlib HTTP server + embedded HTML UI at
  `http://127.0.0.1:7860` (`demo\run_demo.ps1`). 306-clip precomputed index, live text encoding,
  playable videos with cosine scores. Gate: 3 paraphrased spot-check queries returned only correct-class clips.

## Gotchas discovered (recorded for reuse)

1. HF export omits `processor_config.json` → exported processor silently defaults to 448p and video
   embedding crashes. Fix: copy the 224p snapshot's `processor_config.json` into every exported dir.
2. `pip install gradio` (any version) fails in the container (`ResolutionImpossible` on pinned
   httpx/httpcore) → demo uses a zero-dependency stdlib server instead.
3. Docker Desktop on this host drops IPv6 (`localhost`/`::1`) connections to published ports → use `127.0.0.1`.
4. Serre Lab HMDB51 URLs serve their SPA homepage (~5KB HTML) instead of archives → HF mirror + a
   minimum-size sanity check in the downloader.
5. Windows bind-mounts break symlinks: `cosmos_embed1_model_latest.pth` is 0 bytes → always pass the
   exact `iter_#########.pt`.

## Reproduce

```powershell
# data:      scripts\00..04 (download → extract → sample/split → transcode → metadata+specs)
# baseline:  run_container.ps1 -Cmd "cosmos-embed1 evaluate -e /specs/evaluate_zeroshot.yaml results_dir=/results/evaluate_zeroshot"
#            + scripts\10_extract_embeddings.py / 11_compute_metrics.py --phase baseline
# train:     run_container.ps1 -Cmd "cosmos-embed1 train -e /specs/train.yaml results_dir=/results/train"
# eval+ship: cosmos-embed1 evaluate (ckpt) → export (hf) → copy processor_config.json → 10/11 --phase finetuned → 12_compare_phases
# deploy:    deploy\run_search.ps1 -Index ; deploy\run_search.ps1 -Query "..."
# demo:      demo\run_demo.ps1  → http://127.0.0.1:7860
```
