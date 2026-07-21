"""Score full lesson videos into per-window score matrices (container-side).

For each video: sliding 2s windows at 1s stride, embed each window, cosine
score against every class prompt -> S[W x C]. This IS the "isolated" mode
output; glue (11_glue_postprocess.py) post-processes the same files.

Outputs per video into --out:
  scores_<stem>.csv   start_sec + one column per class id  (host scripts read this)
  scores_<stem>.npz   S, starts, V (window embeddings), class_ids
                      (container eval reads this; 15_train_heads consumes V)

Run (via run_container wrapper):
  python /exp/scripts/10_infer_chunks.py --model nvidia/Cosmos-Embed1-224p \
    --videos /data/full --list /data/test_videos.txt --prompts /data/prompts.json \
    --out /results/scores/baseline
--list = file of video stems (one per line), or "all".
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_lib import embed_texts, embed_video_windows, load_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--videos", type=Path, required=True, help="dir of full lesson mp4s")
    ap.add_argument("--list", default="all", help='"all" or file with one video stem per line')
    ap.add_argument("--prompts", type=Path, required=True, help="prompts.json from 01")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk-sec", type=float, default=2.0)
    ap.add_argument("--stride-sec", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    class_ids = sorted(prompts, key=lambda c: prompts[c]["label"])
    captions = [prompts[c]["caption"] for c in class_ids]

    if args.list == "all":
        stems = sorted(p.stem for p in args.videos.glob("*.mp4"))
    else:
        stems = [ln.strip() for ln in Path(args.list).read_text(encoding="utf-8").splitlines() if ln.strip()]

    print(f"loading model: {args.model}")
    model, processor = load_model(args.model, args.device)
    T = embed_texts(model, processor, captions, args.device)  # [C, D]

    args.out.mkdir(parents=True, exist_ok=True)
    for i, stem in enumerate(stems, 1):
        path = args.videos / f"{stem}.mp4"
        if not path.exists():
            print(f"[FAIL] missing {path}", file=sys.stderr)
            return 1
        V, starts = embed_video_windows(model, processor, str(path), args.chunk_sec,
                                        args.stride_sec, args.batch, args.device)
        S = V @ T.T  # [W, C] cosine
        np.savez(args.out / f"scores_{stem}.npz", S=S, starts=starts, V=V,
                 class_ids=np.array(class_ids), model=np.array(str(args.model)),
                 chunk_sec=args.chunk_sec, stride_sec=args.stride_sec)
        with open(args.out / f"scores_{stem}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["start_sec"] + class_ids)
            for j, t0 in enumerate(starts):
                w.writerow([f"{t0:.3f}"] + [f"{s:.6f}" for s in S[j]])
        print(f"  [{i}/{len(stems)}] {stem}: {len(starts)} windows", flush=True)

    print(f"done: {len(stems)} videos -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
