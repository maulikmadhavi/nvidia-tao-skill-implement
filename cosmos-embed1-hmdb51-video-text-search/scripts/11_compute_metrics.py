"""Compute all evaluation metrics + score logs from an embeddings NPZ (container-side).

Install deps at invocation:  python -m pip install --quiet scikit-learn matplotlib

From S = video_embs @ class_prompt_embs.T (cosine, N x C):
  - zero-shot classification (argmax): accuracy, macro+weighted P/R/F1,
    per-class P/R/F1/support CSV, CxC confusion matrix (CSV + PNG heatmap)
  - video->text Recall@{1,3,5,10} (== top-k classification accuracy here,
    since each video has exactly one relevant class prompt)
  - text->video Recall@{1,5,10} multi-positive (hit iff >=1 of the class's
    clips in top-K), mAP over class queries, median rank of first relevant
  - score logs: full similarity matrix CSV + NPZ, per-video top-10 retrieval CSV

Outputs into --out: metrics.json, per_class_metrics.csv, confusion_matrix.csv,
confusion_matrix.png, sim_video_x_class.csv, similarity_scores.npz, retrieval_log.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def video_to_text_recall(S: np.ndarray, labels: np.ndarray, ks=(1, 3, 5, 10)) -> dict:
    order = np.argsort(-S, axis=1)  # class indices, best first
    ranks = np.argmax(order == labels[:, None], axis=1)  # 0-based rank of true class
    return {f"v2t_recall@{k}": float((ranks < k).mean()) for k in ks}


def text_to_video_metrics(S: np.ndarray, labels: np.ndarray, ks=(1, 5, 10)) -> dict:
    """Multi-positive retrieval over class queries: S.T is C x N."""
    St = S.T
    C = St.shape[0]
    recalls = {k: [] for k in ks}
    aps, first_ranks = [], []
    for c in range(C):
        order = np.argsort(-St[c])  # video indices, best first
        rel = (labels[order] == c).astype(np.float64)
        n_rel = int(rel.sum())
        if n_rel == 0:
            continue
        for k in ks:
            recalls[k].append(float(rel[:k].max()))
        hits = np.cumsum(rel)
        precision_at = hits / np.arange(1, len(rel) + 1)
        aps.append(float((precision_at * rel).sum() / n_rel))
        first_ranks.append(int(np.argmax(rel) + 1))  # 1-based
    out = {f"t2v_recall@{k}": float(np.mean(recalls[k])) for k in ks}
    out["t2v_mAP"] = float(np.mean(aps))
    out["t2v_median_rank_first_relevant"] = float(np.median(first_ranks))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--phase", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(args.embeddings, allow_pickle=False)
    V, T = z["video_embs"], z["class_prompt_embs"]
    labels, video_ids = z["labels"], z["video_ids"]
    class_names = [str(c) for c in z["class_names"]]
    N, C = V.shape[0], T.shape[0]
    S = V @ T.T  # embeddings are L2-normalized -> cosine similarity

    args.out.mkdir(parents=True, exist_ok=True)

    # --- classification ---
    pred = S.argmax(axis=1)
    acc = accuracy_score(labels, pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, pred, average="macro", zero_division=0)
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(
        labels, pred, average="weighted", zero_division=0)
    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
        labels, pred, labels=range(C), average=None, zero_division=0)

    with open(args.out / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "label", "precision", "recall", "f1", "support"])
        for i, name in enumerate(class_names):
            w.writerow([name, i, f"{per_p[i]:.4f}", f"{per_r[i]:.4f}",
                        f"{per_f1[i]:.4f}", int(per_sup[i])])

    cm = confusion_matrix(labels, pred, labels=range(C))
    with open(args.out / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + class_names)
        for i, name in enumerate(class_names):
            w.writerow([name] + cm[i].tolist())

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_xticks(range(C), class_names, rotation=90, fontsize=6)
    ax.set_yticks(range(C), class_names, fontsize=6)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(f"Cosmos-Embed1 HMDB51 zero-shot confusion matrix — {args.phase} "
                 f"(N={N}, acc={acc:.3f})")
    fig.colorbar(im, shrink=0.7)
    fig.tight_layout()
    fig.savefig(args.out / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # --- retrieval ---
    metrics = {
        "phase": args.phase,
        "num_test_videos": int(N),
        "num_classes": int(C),
        "accuracy": float(acc),
        "precision_macro": float(macro_p),
        "recall_macro": float(macro_r),
        "f1_macro": float(macro_f1),
        "precision_weighted": float(w_p),
        "recall_weighted": float(w_r),
        "f1_weighted": float(w_f1),
        **video_to_text_recall(S, labels),
        **text_to_video_metrics(S, labels),
        "note_v2t": "video->text recall@k == top-k classification accuracy (one relevant prompt per video)",
        "note_t2v": "text->video recall@k is multi-positive (6 relevant clips per class query)",
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # --- score logs ---
    with open(args.out / "sim_video_x_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "true_class"] + class_names)
        for i in range(N):
            w.writerow([str(video_ids[i]), class_names[labels[i]]]
                       + [f"{s:.6f}" for s in S[i]])

    np.savez(args.out / "similarity_scores.npz", S=S, labels=labels,
             video_ids=video_ids, class_names=np.array(class_names), phase=np.array(args.phase))

    with open(args.out / "retrieval_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "true_class", "rank", "predicted_class", "score", "correct"])
        order = np.argsort(-S, axis=1)
        for i in range(N):
            for r in range(10):
                c = order[i, r]
                w.writerow([str(video_ids[i]), class_names[labels[i]], r + 1,
                            class_names[c], f"{S[i, c]:.6f}", int(c == labels[i])])

    print(json.dumps(metrics, indent=2))
    print(f"wrote metrics + score logs to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
