# Driving-lesson violation detection — Cosmos-Embed1 (air-gapped kit)

Multi-label violation detection on driving-school lesson videos (2s chunks, 1s stride):
zero-shot baseline → per-violation heads on the frozen encoder (`HEADS_DESIGN.md`) →
LoRA fine-tune (only if heads plateau) → two inference modes:
**isolated** (per-chunk thresholding) and **glue** (median-smoothed score trajectories →
per-class tuned thresholds → hysteresis → event segments with timestamps).

Built and fully validated on an internet machine (synthetic self-test: glue recovered
all planted events; every container step also gated under `--network none`), designed
to run on an air-gapped server.

## Read in this order

1. `MIGRATION.md` — build the bundle here, load it there (one-time).
2. `annotations_schema.md` — the annotations.csv contract you must provide.
3. `RUNBOOK.md` — phases 0–7 on the air-gapped server, gates included.
4. `scripts/taxonomy.py` — THE place to adjust classes/captions/glue defaults.

## Layout

| dir | contents |
|---|---|
| `scripts/` | **`phase1..5_*.sh` + `phase4a_heads.sh` — one bash script per RUNBOOK phase (the intended path on the Ubuntu server)** · 00–01 data prep (host, stdlib+ffmpeg) · 10 score matrices + embedding cache (container) · 11 glue/isolated post-processing · 12 chunk-level PR-AUC eval (container) · 13 event-level F1 eval · 14 threshold tuning · 15 heads fit / 16 heads scores + review queue (container, CPU) · gates (incl. `check_bash_syntax.sh`) · wrappers (`run_container.sh` / `.ps1`) · `make_bundle.sh` / `.ps1` · `fetch_snapshots.py` |
| `specs/` | train / evaluate / export specs (all model paths = `/model/...`, air-gap safe) |
| `demo/` | stdlib timeline review app (trajectories + event bars + click-to-seek video) |
| `selftest/` | synthetic-lesson end-to-end validation (uses HMDB clips as pseudo-violations) |
| `workspace/` | data / specs / results / model — the runtime area (`model/` ships in the bundle) |

## Validated on the build machine (2026-07-17)

- Offline image self-sufficiency (`--network none`): deps + model load + text embed — PASSED
- Synthetic self-test, zero-shot: chunking gates, glue event recovery (micro F1 0.86,
  all planted events found), chunk mAP 0.887, demo renders — PASSED
- 1-iter LoRA smoke train under `--network none` with multi-label duplicate-row
  metadata: loader OK, checkpoint written — PASSED
- Phase 4a heads chain (15 fit → 16 score → 14 prob-grid tune → glue → event eval):
  heads beat prompts on val AP for every class, heads glue recovered planted events
  (micro F1 0.44 from 4–11 positive train windows per class) — PASSED

## Air-gap gotchas already engineered around

1. The cosmos-embed1 CLI needs the FULL HF repo (incl. `model_converted.pth`) and does not
   read the hub cache → full snapshot ships in `workspace/model/`, specs use `/model/...`.
2. The Q-Former init resolves `google-bert/bert-base-uncased` to a fixed path inside the
   image and downloads if absent → bert is BAKED into the offline image (Dockerfile).
3. The model's remote code also does a hub lookup for bert under its legacy alias →
   hub cache ships entries under BOTH names (`fetch_snapshots.py`).
4. No runtime pip: protobuf<7 + sklearn + matplotlib baked into the image.
5. Demo/deploy are Python-stdlib only; host scripts too (no host pip anywhere).
