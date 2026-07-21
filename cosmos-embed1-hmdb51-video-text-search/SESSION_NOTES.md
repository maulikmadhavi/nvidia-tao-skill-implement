# Session reference — Cosmos-Embed1 video-text search experiment

Session date: 2026-07-17 (single session, plan → data → baseline → train → eval → deploy → demo).
Keep this file as the entry point when returning to this work.

## What was accomplished

End-to-end video-text search experiment, all gates passed:
- HMDB51 curated (1,530 clips, 51 classes, seed-42 stratified 1,122 train / 102 val / 306 test).
- Zero-shot baseline (pretrained `nvidia/Cosmos-Embed1-224p`): test top-1 **65.7%**, macro-F1 62.4, t2v mAP 70.1.
- LoRA fine-tune (r=8, 600 iters, batch 4, lr 1e-4 cosine, ~3.5h on RTX 5070): test top-1 **77.5%**, macro-F1 76.4, t2v mAP 82.7.
- Model exported to HF format (LoRA merged, parity cosine 1.0) → `deploy\model\`.
- Deploy search CLI + precomputed test index; web demo (stdlib server) on the 306 test clips.

## Where everything lives (this folder)

| File | What it is |
|---|---|
| `PLAN.md` | Approved experiment spec + phase-by-phase status log with results |
| `PIPELINE.md` | 7-step reproduction guide (input/output per step) + design for the next (driving-violation) experiment |
| `reports\final_report.md` | Headline results, setup, deliverables, and the 5 gotchas discovered |
| `reports\comparison.md` | Full baseline-vs-finetuned metric table + container cross-checks |
| `reports\confusion_matrix_*.png`, `per_class_metrics_*.csv`, `metrics_*.json` | Metric artifacts, both phases |
| `workspace\results\metrics\{baseline,finetuned}\` | ALL matching scores: 306×51 similarity CSV/NPZ + per-video top-10 logs |
| `workspace\results\train\train\checkpoints\iter_000000600.pt` | Raw fine-tuned checkpoint (2.5GB) |
| `deploy\` | Shippable: HF model dir + `search_cli.py` + `run_search.ps1` + README |
| `demo\` | Web demo: `app.py` (stdlib server) + `run_demo.ps1` + README |
| `scripts\` | Numbered pipeline scripts 00–04 (host data prep), 10–12 (metrics), `run_container.ps1` (canonical docker entrypoint), `embed_lib.py` (shared model API) |

## Quick resume commands

```powershell
# demo (http://127.0.0.1:7860 — IPv4 only, never localhost):
demo\run_demo.ps1
# search from CLI:
deploy\run_search.ps1 -Query "a video of a person riding a bike" -TopK 5
# find/stop a running demo container:
docker ps --filter ancestor=nvcr.io/nvidia/tao/tao-toolkit:7.0.1-cosmos-embed
docker stop <name>
```

At session end the demo container (`epic_hoover`) was left RUNNING and holds a few GB of VRAM —
stop it before other GPU work.

## Five gotchas (cost real debugging time — details in final_report.md)

1. HF export omits `processor_config.json` → copy it from the 224p snapshot into every exported dir.
2. `pip install gradio` is impossible in this container (pinned httpx/httpcore) → stdlib demo server.
3. Docker Desktop drops IPv6 → always `127.0.0.1`, never `localhost`.
4. Serre Lab HMDB51 URLs are dead (serve HTML) → HF mirror `jili5044/hmdb51`.
5. Windows mounts zero symlinks → pass exact `iter_#########.pt`, never the `latest` symlink.

## Planned next experiment (not started)

Driving-school lesson violation detection (phone use, seatbelt, …) on 10s chunks.
Design is fully worked out in `PIPELINE.md` §"Next experiment": multi-label scoring with per-class
thresholds (not argmax), source-video-level stratified splits, temporal-jitter positive augmentation,
PR-AUC/P@R metrics, one metadata row per (chunk, violation), multi-window score max-pooling.
Blocked on: the driving-lesson videos + violation annotations (event timestamps at minimum).
