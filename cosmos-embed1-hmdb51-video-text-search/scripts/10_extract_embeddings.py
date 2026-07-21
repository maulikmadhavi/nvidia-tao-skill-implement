"""Extract test-set video + class-prompt embeddings (container-side).

Loads Cosmos-Embed1 as a HuggingFace model: --model is either the pretrained
repo id (baseline phase) or the exported HF dir (finetuned phase). Embeds
every video in --metadata plus the 51 class prompts, and writes one NPZ with:
  video_embs [N, D], video_ids [N], labels [N],
  class_prompt_embs [C, D], class_names [C], prompts [C]

Run via run_container.ps1, e.g.:
  python /exp/scripts/10_extract_embeddings.py --model nvidia/Cosmos-Embed1-224p
    --metadata /data/test.json --videos /data/video --prompts /data/class_prompts.json
    --out /results/metrics/baseline/embeddings.npz --phase baseline
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
    ap.add_argument("--model", required=True, help="HF repo id or exported HF dir")
    ap.add_argument("--metadata", type=Path, required=True, help="test.json")
    ap.add_argument("--videos", type=Path, required=True, help="/data/video")
    ap.add_argument("--prompts", type=Path, required=True, help="class_prompts.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--phase", required=True, choices=["baseline", "finetuned"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rows = json.loads(args.metadata.read_text(encoding="utf-8"))
    prompts_map = json.loads(args.prompts.read_text(encoding="utf-8"))
    class_names = sorted(prompts_map, key=lambda c: prompts_map[c]["label"])
    prompts = [prompts_map[c]["caption"] for c in class_names]
    caption_to_label = {prompts_map[c]["caption"]: prompts_map[c]["label"] for c in class_names}

    print(f"[{args.phase}] loading model: {args.model}")
    model, processor = load_model(args.model, args.device)

    print(f"embedding {len(prompts)} class prompts")
    class_prompt_embs = embed_texts(model, processor, prompts, args.device)

    video_embs, video_ids, labels = [], [], []
    for i, row in enumerate(rows, 1):
        path = args.videos / row["video"]
        video_embs.append(embed_video(model, processor, str(path), args.device))
        video_ids.append(row["video_id"])
        labels.append(caption_to_label[row["caption"]])
        if i % 25 == 0 or i == len(rows):
            print(f"  embedded {i}/{len(rows)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        video_embs=np.stack(video_embs),
        video_ids=np.array(video_ids),
        labels=np.array(labels, dtype=np.int64),
        class_prompt_embs=class_prompt_embs,
        class_names=np.array(class_names),
        prompts=np.array(prompts),
        phase=np.array(args.phase),
        model=np.array(str(args.model)),
    )
    print(f"wrote {args.out}: videos {len(video_ids)} x {video_embs[0].shape[0]}d, "
          f"classes {len(class_names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
