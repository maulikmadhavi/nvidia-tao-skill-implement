"""Deployable Cosmos-Embed1 video-text search CLI (container-side).

Two subcommands:
  index   Embed every video of a metadata JSON into an index NPZ.
  search  Encode a free-text query live and print the top-k videos by cosine.

--model accepts either the exported fine-tuned HF dir (default deploy artifact)
or nvidia/Cosmos-Embed1-224p (for zero-shot A/B). Videos are only decoded at
index time; search embeds nothing but the query text.

Examples (via scripts/run_container.ps1):
  python /exp/deploy/search_cli.py index --model /exp/deploy/model \
      --metadata /data/test.json --videos /data/video --out /results/deploy/index.npz
  python /exp/deploy/search_cli.py search --model /exp/deploy/model \
      --index /results/deploy/index.npz --query "a video of a person riding a bike" --topk 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/exp/scripts")
from embed_lib import embed_texts, embed_video, load_model  # noqa: E402


def cmd_index(args) -> int:
    rows = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    model, processor = load_model(args.model, args.device)
    embs, ids, captions, files = [], [], [], []
    for i, row in enumerate(rows, 1):
        embs.append(embed_video(model, processor, str(Path(args.videos) / row["video"]), args.device))
        ids.append(row["video_id"])
        captions.append(row["caption"])
        files.append(row["video"])
        if i % 25 == 0 or i == len(rows):
            print(f"  indexed {i}/{len(rows)}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, embeddings=np.stack(embs), video_ids=np.array(ids),
             captions=np.array(captions), files=np.array(files), model=np.array(str(args.model)))
    print(f"wrote {out} ({len(ids)} videos)")
    return 0


def cmd_search(args) -> int:
    z = np.load(args.index, allow_pickle=False)
    model, processor = load_model(args.model, args.device)
    q = embed_texts(model, processor, [args.query], args.device)[0]
    scores = z["embeddings"] @ q
    order = np.argsort(-scores)[: args.topk]
    hits = [{"rank": r + 1,
             "video_id": str(z["video_ids"][i]),
             "file": str(z["files"][i]),
             "caption": str(z["captions"][i]),
             "score": float(scores[i])}
            for r, i in enumerate(order)]
    if args.json:
        print(json.dumps({"query": args.query, "results": hits}, indent=2))
    else:
        print(f'query: "{args.query}"')
        for h in hits:
            print(f'  {h["rank"]:2}. {h["video_id"]:28} {h["score"]:.4f}  ({h["caption"]})')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="embed videos into an index NPZ")
    p.add_argument("--model", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--videos", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.set_defaults(fn=cmd_index)

    p = sub.add_parser("search", help="text query -> top-k videos")
    p.add_argument("--model", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--json", action="store_true")
    p.add_argument("--device", default="cuda")
    p.set_defaults(fn=cmd_search)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
