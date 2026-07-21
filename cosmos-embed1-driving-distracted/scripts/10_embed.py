"""Embed every clip + the class prompts with Cosmos-Embed1 (CONTAINER-side; GPU).

One pass over all clips (train+val together — the split lives in the metadata and
is carried through as an array, so 11_zeroshot and 20_linear_probe both read a
single cached NPZ). --model is the LOCAL snapshot (/model/Cosmos-Embed1-224p) for
the baseline, or an exported HF dir after LoRA.

Writes --out NPZ:
  video_embs [N,D] f32 (L2-norm), video_ids [N], labels [N] i64, splits [N],
  class_prompt_embs [C,D], class_names [C], prompts [C], model, phase

Run via run_container.ps1:
  python /exp/scripts/10_embed.py --model /model/Cosmos-Embed1-224p \
    --metadata /splits/metadata.json --prompts /splits/class_prompts.json \
    --videos /data --out /results/baseline/embeddings.npz --phase baseline
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_lib import embed_texts, embed_video, load_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="local snapshot or exported HF dir")
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--videos", type=Path, required=True, help="dataset root (/data)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--phase", default="baseline")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rows = json.loads(args.metadata.read_text(encoding="utf-8"))
    prompts_map = json.loads(args.prompts.read_text(encoding="utf-8"))
    class_names = sorted(prompts_map, key=lambda c: prompts_map[c]["label"])
    prompts = [prompts_map[c]["caption"] for c in class_names]

    print(f"[{args.phase}] loading model: {args.model}", flush=True)
    model, processor = load_model(args.model, args.device)

    print(f"embedding {len(prompts)} class prompts", flush=True)
    class_prompt_embs = embed_texts(model, processor, prompts, args.device)

    video_embs, video_ids, labels, splits = [], [], [], []
    n = len(rows)
    for i, row in enumerate(rows, 1):
        path = args.videos / row["video"]
        try:
            emb = embed_video(model, processor, str(path), args.device)
        except Exception as e:  # noqa: BLE001 — one bad clip must not kill the run
            print(f"  WARN skipping {row['video']}: {e}", flush=True)
            continue
        video_embs.append(emb)
        video_ids.append(row["video_id"])
        labels.append(int(row["label"]))
        splits.append(row["split"])
        if i % 50 == 0 or i == n:
            print(f"  embedded {i}/{n}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        video_embs=np.stack(video_embs),
        video_ids=np.array(video_ids),
        labels=np.array(labels, dtype=np.int64),
        splits=np.array(splits),
        class_prompt_embs=class_prompt_embs,
        class_names=np.array(class_names),
        prompts=np.array(prompts),
        phase=np.array(args.phase),
        model=np.array(str(args.model)),
    )
    D = video_embs[0].shape[0]
    print(f"wrote {args.out}: {len(video_ids)} clips x {D}d, {len(class_names)} classes",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
