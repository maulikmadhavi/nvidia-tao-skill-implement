# Deployable Cosmos-Embed1 HMDB51 video-text search

Artifacts:
- `model\` — the fine-tuned model exported in HuggingFace format (load with
  `AutoModel.from_pretrained(dir, trust_remote_code=True)`).
- `search_cli.py` — index + search CLI (runs inside the cosmos-embed container).
- `run_search.ps1` — Windows wrapper around both subcommands.

## Usage

```powershell
# one-time: build the search index over the test split
.\run_search.ps1 -Index

# search
.\run_search.ps1 -Query "a video of a person riding a bike" -TopK 5
.\run_search.ps1 -Query "someone doing a cartwheel" -TopK 5 -Json
```

Requirements: Docker Desktop running, GPU available, the
`nvcr.io/nvidia/tao/tao-toolkit:7.0.1-cosmos-embed` image pulled
(all docker flags live in `..\scripts\run_container.ps1`).

The index (`workspace\results\deploy\index.npz`) contains only the 306
held-out test clips; search never touches training data. Only the text query
is embedded at request time.
