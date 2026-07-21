"""Zero-shot clip classification metrics from the cached embeddings NPZ (CONTAINER).

S = video_embs @ class_prompt_embs.T  (cosine; both L2-normalized).
pred = argmax over the 7 class prompts. Reported on the requested --split
(default val) so the number is comparable to the linear probe. Also prints the
train-split accuracy for reference (zero-shot uses no training, so train == an
unbiased second sample).

Writes into --out: metrics_zeroshot.json, per_class_zeroshot.csv,
confusion_zeroshot.csv, confusion_zeroshot.png, sim_scores.csv

Run via run_container.ps1:
  python /exp/scripts/11_zeroshot_metrics.py \
    --embeddings /results/baseline/embeddings.npz --split val \
    --out /results/baseline/zeroshot
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def eval_split(S, labels, C):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    pred = S.argmax(axis=1)
    acc = accuracy_score(labels, pred)
    mp, mr, mf1, _ = precision_recall_fscore_support(
        labels, pred, average="macro", zero_division=0)
    return pred, acc, mp, mr, mf1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--split", default="val", choices=["val", "train", "all"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from sklearn.metrics import (confusion_matrix,
                                 precision_recall_fscore_support)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(args.embeddings, allow_pickle=False)
    V, T = z["video_embs"], z["class_prompt_embs"]
    labels_all, splits = z["labels"], z["splits"].astype(str)
    class_names = [str(c) for c in z["class_names"]]
    C = T.shape[0]
    S_all = V @ T.T

    mask = np.ones(len(labels_all), bool) if args.split == "all" else (splits == args.split)
    S, labels = S_all[mask], labels_all[mask]
    N = len(labels)

    pred, acc, mp, mr, mf1 = eval_split(S, labels, C)
    # reference accuracy on the other split
    other = "train" if args.split == "val" else "val"
    omask = splits == other
    oacc = float((S_all[omask].argmax(1) == labels_all[omask]).mean()) if omask.any() else None

    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
        labels, pred, labels=range(C), average=None, zero_division=0)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "per_class_zeroshot.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "label", "precision", "recall", "f1", "support"])
        for i, name in enumerate(class_names):
            w.writerow([name, i, f"{per_p[i]:.4f}", f"{per_r[i]:.4f}",
                        f"{per_f1[i]:.4f}", int(per_sup[i])])

    cm = confusion_matrix(labels, pred, labels=range(C))
    with open(args.out / "confusion_zeroshot.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + class_names)
        for i, name in enumerate(class_names):
            w.writerow([name] + cm[i].tolist())

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_xticks(range(C), class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(C), class_names, fontsize=8)
    for i in range(C):
        for j in range(C):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] < cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Zero-shot ({args.split}, N={N}, acc={acc:.3f})")
    fig.colorbar(im, shrink=0.7); fig.tight_layout()
    fig.savefig(args.out / "confusion_zeroshot.png", dpi=150); plt.close(fig)

    with open(args.out / "sim_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "true_class", "pred_class", "correct"] + class_names)
        for i in range(N):
            w.writerow([i, class_names[labels[i]], class_names[pred[i]],
                        int(pred[i] == labels[i])] + [f"{s:.4f}" for s in S[i]])

    metrics = {
        "phase": "baseline_zeroshot",
        "split": args.split,
        "num_clips": int(N),
        "num_classes": int(C),
        "accuracy": float(acc),
        "precision_macro": float(mp),
        "recall_macro": float(mr),
        "f1_macro": float(mf1),
        f"reference_accuracy_{other}": oacc,
        "per_class_f1": {class_names[i]: float(per_f1[i]) for i in range(C)},
    }
    (args.out / "metrics_zeroshot.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote zero-shot metrics to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
