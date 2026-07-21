"""One-time (INTERNET machine): download full model snapshots into /model.

The cosmos-embed1 CLI needs the COMPLETE repo (incl. model_converted.pth, which
transformers never caches), so the air-gap bundle carries full local snapshots:
  /model/Cosmos-Embed1-224p      (specs: model.pretrained_model_path)
  /model/bert-base-uncased       (train spec: model.network.qformer_pretrain_ckpt)

Run inside the container WITH network (override the baked offline env):
  docker run --rm -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
    -v <exp>/workspace/model:/model cosmos-embed-offline:7.0.1 \
    python /exp/scripts/fetch_snapshots.py
"""

from huggingface_hub import snapshot_download

for repo, dest in [("nvidia/Cosmos-Embed1-224p", "/model/Cosmos-Embed1-224p"),
                   ("google-bert/bert-base-uncased", "/model/bert-base-uncased")]:
    print(f"fetching {repo} -> {dest}")
    snapshot_download(repo, local_dir=dest)

# The Cosmos remote code calls BertConfig.from_pretrained on a bert repo id —
# a HUB lookup even when weights load from a local dir. Offline resolution needs
# the hub cache (mounted at /hf_cache) to contain the entry under BOTH names:
# the canonical "google-bert/bert-base-uncased" AND the legacy alias
# "bert-base-uncased" (the remote code uses the alias; online the hub redirects,
# offline the cache lookup is name-exact).
for repo in ("google-bert/bert-base-uncased", "bert-base-uncased"):
    print(f"populating hub cache with {repo}")
    snapshot_download(repo)
print("done")
