# Migration to the air-gapped server

## On the INTERNET machine (this PC) — build the bundle

```powershell
cd D:\tao-skill-bank\exp\cosmos-embed1-driving-violations
docker build -f Dockerfile.offline -t cosmos-embed-offline:7.0.1 .        # already done + gated
docker run --rm -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 `
  -v "$PWD\workspace\model:/model" -v "$PWD\scripts:/exp/scripts:ro" `
  cosmos-embed-offline:7.0.1 python /exp/scripts/fetch_snapshots.py       # already done
.\scripts\make_bundle.ps1 -Full        # ~30-35 GB total
```

On a Linux/Ubuntu internet machine use the bash twin (same output, same layout):

```bash
./scripts/make_bundle.sh --full        # dry-run without --full; --model-source /path to override
```

`bundle\` then contains:

| item | size | purpose |
|---|---|---|
| `cosmos-embed-offline.tar` | ~20+ GB | TAO cosmos-embed image with protobuf<7 + sklearn + matplotlib BAKED IN and `HF_HUB_OFFLINE=1` set — no pip, no internet ever needed at runtime |
| `model\` | ~10 GB | FULL local snapshots: `Cosmos-Embed1-224p` (incl. `model_converted.pth` — the cosmos-embed1 CLI needs the complete repo and does NOT read the HF hub cache) + `bert-base-uncased` (Q-Former init at train time). Every spec/script uses `/model/...` paths |
| `kit\` | ~1 MB | scripts, specs, demo, deploy, selftest, docs, Dockerfile.offline |
| `MANIFEST.txt` | — | sizes + SHA256 of the tar — verify after transfer |

Transfer the `bundle\` directory by approved media (verify the tar's SHA256 on arrival).

## On the AIR-GAPPED server — one-time setup

Prereqs: Docker + nvidia-container-toolkit + NVIDIA driver, ffmpeg, Python 3.10+ (stdlib
only — no pip needed on the host). Linux assumed below; on Windows use the `.ps1` twins.

```bash
# 1. load the image
docker load -i bundle/cosmos-embed-offline.tar
docker images | grep cosmos-embed-offline      # expect 7.0.1

# 2. lay out the experiment
mkdir -p ~/exp/cosmos-embed1-driving-violations
cp -r bundle/kit/*  ~/exp/cosmos-embed1-driving-violations/
cd ~/exp/cosmos-embed1-driving-violations
mkdir -p workspace/{data/video,data/full,specs,results,hf_cache}
cp -r /path/to/bundle/model workspace/model          # the full snapshots
chmod +x scripts/*.sh demo/*.sh selftest/*.sh

# 3. GPU gate (must print "GATE PASSED")
./scripts/run_container.sh "python /exp/scripts/gate_sm120.py"

# 4. offline self-sufficiency gate (must print "GATE PASSED: offline image is self-sufficient")
docker run --rm --gpus all --network none \
  -v "$PWD/workspace/model:/model" -v "$PWD/scripts:/exp/scripts:ro" \
  cosmos-embed-offline:7.0.1 python /exp/scripts/gate_offline.py
```

Both gates were already verified on the build machine (RTX 5070 / sm_120). If gate_sm120
fails on the server's GPU generation, the torch build in the image (2.7.0+cu12.9) lacks
kernels for it — escalate before proceeding.

Then follow `RUNBOOK.md` phase by phase.

## Known environment gotchas (carried from the HMDB experiment)

1. Use `127.0.0.1`, never `localhost`, for the demo URL (Docker Desktop drops IPv6; harmless on Linux).
2. Never rely on `*_latest.pth` symlinks through bind mounts — pass exact `iter_#########.pt` files.
3. If ffmpeg/ffprobe are launched from background shells and hang at 0% CPU: the scripts
   already set `stdin=DEVNULL` + timeouts; if you add new subprocess calls, do the same.
4. `--shm-size=16g` assumes >= 32 GB host RAM; scale down proportionally if the server has less.
