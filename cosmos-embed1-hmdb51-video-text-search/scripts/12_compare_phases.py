"""Compare baseline vs finetuned metrics -> reports/comparison.md (host, stdlib only).

Reads the custom metrics.json of both phases, plus (optionally) the container's
own evaluate metrics.json files for cross-checking top-k classification
accuracy (|delta| > 0.01 gets flagged — small drift from frame-sampling
differences is expected).
"""

import argparse
import json
import sys
from pathlib import Path

HEADLINE = [
    "accuracy", "precision_macro", "recall_macro", "f1_macro",
    "precision_weighted", "recall_weighted", "f1_weighted",
    "v2t_recall@1", "v2t_recall@3", "v2t_recall@5", "v2t_recall@10",
    "t2v_recall@1", "t2v_recall@5", "t2v_recall@10", "t2v_mAP",
    "t2v_median_rank_first_relevant",
]


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def find_topk(container_metrics: dict) -> dict:
    """Pull top-k accuracy values out of the container's metrics.json (schema tolerant)."""
    flat = {}

    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}{k}.".lower())
        elif isinstance(obj, (int, float)):
            flat[prefix.rstrip(".")] = float(obj)

    walk(container_metrics)
    return {k: v for k, v in flat.items() if "top" in k}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True, help="custom metrics.json (baseline)")
    ap.add_argument("--finetuned", type=Path, required=True, help="custom metrics.json (finetuned)")
    ap.add_argument("--container-baseline", type=Path, help="container evaluate metrics.json (zeroshot)")
    ap.add_argument("--container-finetuned", type=Path, help="container evaluate metrics.json (finetuned)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    base, fine = load(args.baseline), load(args.finetuned)

    lines = [
        "# Cosmos-Embed1 HMDB51 video-text search — baseline vs finetuned",
        "",
        f"Test set: {base['num_test_videos']} clips, {base['num_classes']} classes "
        "(identical for both phases).",
        "",
        "| metric | baseline (zero-shot) | finetuned (LoRA) | delta |",
        "|---|---:|---:|---:|",
    ]
    for m in HEADLINE:
        b, f = base.get(m), fine.get(m)
        if b is None or f is None:
            continue
        lines.append(f"| {m} | {b:.4f} | {f:.4f} | {f - b:+.4f} |")

    for name, path in (("baseline", args.container_baseline), ("finetuned", args.container_finetuned)):
        if not path or not path.exists():
            continue
        topk = find_topk(load(path))
        if not topk:
            continue
        lines += ["", f"## Container evaluate top-k ({name})", "",
                  "| container metric | value |", "|---|---:|"]
        lines += [f"| {k} | {v:.4f} |" for k, v in sorted(topk.items())]
        custom = (base if name == "baseline" else fine).get("v2t_recall@1")
        top1 = next((v for k, v in sorted(topk.items()) if "top_1" in k or "top1" in k), None)
        if custom is not None and top1 is not None:
            flag = "  <-- FLAG: |delta| > 0.01" if abs(custom - top1) > 0.01 else ""
            lines.append(f"\ncross-check top-1: custom {custom:.4f} vs container {top1:.4f}{flag}")

    lines += ["", "## Notes", "",
              "- v2t_recall@k equals top-k classification accuracy (one relevant class prompt per video).",
              "- t2v metrics are multi-positive (6 relevant clips per class query); the container's",
              "  caption-level retrieval R@K is ill-defined with duplicate captions and is not reported.",
              "- Full similarity matrices and per-video top-10 score logs live next to each phase's",
              "  metrics.json (sim_video_x_class.csv, similarity_scores.npz, retrieval_log.csv)."]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
