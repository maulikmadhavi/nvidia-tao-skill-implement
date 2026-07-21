# Distracted-driving clip classification — Cosmos-Embed1

Single-label 7-way classification of ~3s dashcam clips into 6 distracted-driving
violations + `no_violation`, using **Cosmos-Embed1-224p** (8-frame, 224², 256-d
L2-normalized video/text embeddings). Two tiers, cheapest first:

1. **Zero-shot** — cosine similarity of each clip embedding to 7 text prompts,
   `argmax` → predicted class. No training.
2. **Linear probe (finetune tier 1)** — multinomial logistic regression on the
   frozen embeddings. Seconds on CPU, sklearn baked into the offline image. Only
   escalate to encoder LoRA if this plateaus.

## Dataset

`D:\research_data\driving_violation\shortclips_pos_neg\shortclips_pos_neg`

| class | folder | clips |
|---|---|---|
| no_violation | `negative/` | 605 |
| drinking, eating, interacting_with_phone, reading_newspaper, talking_on_phone, working_on_laptop | `positive/<class>/` | 50 each (300) |

**Known artifact:** negatives are 1920×1080@30fps, positives 960×540@15fps — a
resolution/fps confound. Cosmos-Embed resizes to 224² and uniformly samples 8
frames, so it is largely (not perfectly) normalized away. Watch the negative
column of the confusion matrix for leakage.

Stratified 80/20 split per class → **724 train / 181 val** (`workspace/splits/`).

## Run (Windows build machine)

```powershell
cd D:\tao-skill-bank\exp\cosmos-embed1-driving-distracted

# 0. split + prompts (host, stdlib only)
python scripts\00_prepare_split.py `
  --data-root "D:\research_data\driving_violation\shortclips_pos_neg\shortclips_pos_neg" `
  --out-dir workspace\splits --val-frac 0.2 --seed 0

# 1. embed all clips + prompts (container, GPU) — one cached NPZ
.\scripts\run_container.ps1 -Cmd "python /exp/scripts/10_embed.py --model /model/Cosmos-Embed1-224p --metadata /splits/metadata.json --prompts /splits/class_prompts.json --videos /data --out /results/baseline/embeddings.npz --phase baseline"

# 2. zero-shot metrics on val (container)
.\scripts\run_container.ps1 -Cmd "python /exp/scripts/11_zeroshot_metrics.py --embeddings /results/baseline/embeddings.npz --split val --out /results/baseline/zeroshot"

# 3. linear probe finetune (container, CPU)
.\scripts\run_container.ps1 -Cmd "python /exp/scripts/20_linear_probe.py --embeddings /results/baseline/embeddings.npz --out /results/baseline/probe"
```

The container reuses the model snapshot + hf_cache from the sibling
`cosmos-embed1-driving-violations` experiment (no 7.6 GB duplicate).

## Layout

| path | what |
|---|---|
| `scripts/00_prepare_split.py` | host: stratified split, metadata.json + class_prompts.json |
| `scripts/embed_lib.py` | Cosmos-Embed1 load / frame-sample / embed helpers |
| `scripts/10_embed.py` | container: embed clips + prompts → embeddings.npz |
| `scripts/11_zeroshot_metrics.py` | container: zero-shot argmax metrics + confusion |
| `scripts/20_linear_probe.py` | container: LR finetune on frozen embeddings |
| `scripts/run_container.ps1` | docker wrapper (shared model/hf_cache mounts) |
| `workspace/splits/` | metadata.json, class_prompts.json, split_stats.json |
| `workspace/results/baseline/` | embeddings.npz, zeroshot/, probe/ |

## Captions

The zero-shot prompts live in `scripts/00_prepare_split.py` (`CAPTIONS`) — the one
place to edit wording. The two phone folders are deliberately separated:
`interacting_with_phone` = looking at / operating a handheld phone (texting,
browsing); `talking_on_phone` = phone held to the ear on a call.
