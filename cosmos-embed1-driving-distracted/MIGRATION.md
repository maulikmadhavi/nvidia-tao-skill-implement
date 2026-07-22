# Migration guide — running driving-distracted on a fresh GPU server

Validated 2026-07-22 on a clean Linux + H100 NVL box: reproduced probe val acc
**0.989** (bit-identical to the original Windows run) from a `git clone` plus the
steps below. This guide is the single source of truth for standing the experiment
up somewhere new.

## Prerequisites

| need | why |
|---|---|
| Docker + NVIDIA Container Toolkit | GPU passthrough (`docker run --gpus all …`) |
| NGC account + `docker login nvcr.io` | pull the base image the offline image builds on |
| Python 3 + `huggingface_hub[cli]` on the **host** | download the model/bert snapshots |
| ffmpeg/ffprobe NOT needed | data is pre-cut mp4 clips; no transcode step here |

The scripts are path-agnostic — nothing is hard-coded to the original machine.
Three env knobs point them at your layout:

| var | what it is | mounted at |
|---|---|---|
| `DATA_ROOT` | dataset root (`shortclips_pos_neg` with `negative/` + `positive/<class>/`) | `/data` (ro) |
| `MODEL_DIR` | the `Cosmos-Embed1-224p` snapshot dir | `/model/Cosmos-Embed1-224p` (ro) |
| `HF_CACHE`  | HF hub cache holding `bert-base-uncased` | `/hf_cache` (rw) |

## Step 1 — build the offline image

The base is `nvcr.io/nvidia/tao/tao-toolkit:7.0.1-cosmos-embed` (from nvcr.io) and the
build downloads bert, so the **build** needs network + an NGC login even though the
resulting image runs offline.

```bash
docker login nvcr.io          # username: $oauthtoken   password: <NGC API key>
cd cosmos-embed1-driving-distracted            # or the sibling driving-violations dir — same Dockerfile
docker build -f Dockerfile.offline -t cosmos-embed-offline:7.0.1 .
```

## Step 2 — stage the weights (model + bert)

The offline image runs `HF_HUB_OFFLINE=1`, so every weight must already be on disk.
`download_model.sh` pulls **both** the Cosmos snapshot (real files, for the bind mount)
**and** `bert-base-uncased` into the HF cache — the latter is mandatory (see Gotchas).

```bash
export MODEL_DIR=/data/you/hf_cache/Cosmos-Embed1-224p
export HF_CACHE=/data/you/hf_cache/cosmos_embed_hfcache
MODEL_DIR=$MODEL_DIR HF_CACHE=$HF_CACHE ./scripts/download_model.sh
```

Behind a corporate TLS-inspecting proxy the pull fails with
`SSL CERTIFICATE_VERIFY_FAILED`; fix it on the host before retrying:

```bash
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt        # RHEL: /etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

## Step 3 — run the pipeline

```bash
cd cosmos-embed1-driving-distracted/scripts
DATA_ROOT=/data/you/shortclips_pos_neg/shortclips_pos_neg \
MODEL_DIR=$MODEL_DIR \
HF_CACHE=$HF_CACHE \
./run_steps.sh
```

`run_steps.sh` phases: `preflight` (checks docker, GPU passthrough, image, dataset,
`MODEL_DIR/config.json`, staged bert) → `split` (host) → `embed` (GPU) → `zeroshot`
→ `probe`. Pick phases explicitly (`./run_steps.sh embed zeroshot`) or dry-run with
`DRY_RUN=1`. Preflight fails fast with the exact fix command if anything is missing.

Expected: **zero-shot ~0.95, linear-probe val acc ~0.989.** Results land in
`workspace/results/baseline/{zeroshot,probe}/`.

## Gotchas that bite a fresh server

1. **bert-base-uncased must be in the mounted `/hf_cache`.** The Cosmos Q-Former calls
   `BertConfig.from_pretrained("bert-base-uncased")` (bare alias) at load. `Dockerfile.offline`
   bakes bert only into the *CLI's* `pretrained_checkpoints/` path — NOT the HF cache the
   `AutoModel` path reads. `download_model.sh` stages it; preflight hard-checks it. (On the
   original Windows box it silently worked because that shared hf_cache already held bert
   from a prior HMDB run.)
2. **Corp-proxy SSL.** See Step 2's CA-bundle fix. Host-side only — the pre-staged weights
   mean the container never needs network.
3. **`best_C` may differ (e.g. 10 vs 100) across sklearn versions/platforms.** It's a
   CV macro-F1 tiebreak on a flat plateau; val accuracy is unchanged. Not a regression.
4. **Zero-shot may shift by ~1 clip** (0.945 vs 0.950 on 181 val) because the Linux
   container's frame-decode backend (decord/PyAV/OpenCV precedence) samples 8 frames
   slightly differently. Expected, harmless.
5. **PowerShell-only:** the probe step's sklearn `FutureWarning` on stderr makes PowerShell
   report exit 1 on success — check `metrics_probe.json` exists. Under bash `set -e` this
   never trips (only docker's real exit code matters).

## What is NOT covered here

Deploy CLI + web demo (their `deploy/`/`demo/` runner scripts are not committed in this
tree). The pipeline ends at the linear probe + `probe.npz`, which is sufficient to score
new clips.
