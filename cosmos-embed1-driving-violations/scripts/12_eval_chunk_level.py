"""Chunk(window)-level evaluation: PR-AUC/AP per class + co-occurrence matrix (container-side).

Ground truth is derived at the SAME window grid as the scores (overlap >= 0.5s
with an annotated event), so scores and labels align exactly. Reports, per
violation class: Average Precision (PR-AUC), precision at recall >= 0.9,
support; plus a co-occurrence error matrix at the tuned thresholds
(rows = GT class active, cols = predicted class active).

Needs sklearn + matplotlib (baked into the offline image).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glue_lib import read_annotations  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path, required=True, help="dir with scores_*.npz")
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--videos", type=Path, required=True, help="split video stems, one per line")
    ap.add_argument("--thresholds", type=Path, default=None, help="thresholds.json (for the co-occurrence matrix)")
    ap.add_argument("--min-overlap", type=float, default=0.5)
    ap.add_argument("--phase", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from sklearn.metrics import average_precision_score, precision_recall_curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_all = read_annotations(args.annotations)
    stems = [ln.strip() for ln in args.videos.read_text(encoding="utf-8").splitlines() if ln.strip()]

    S_list, Y_list = [], []
    class_ids = None
    chunk_sec = None
    for stem in stems:
        z = np.load(args.scores / f"scores_{stem}.npz", allow_pickle=False)
        if class_ids is None:
            class_ids = [str(c) for c in z["class_ids"]]
            chunk_sec = float(z["chunk_sec"])
        S = z["S"]
        starts = z["starts"]
        Y = np.zeros_like(S, dtype=np.int8)
        for ci, cls in enumerate(class_ids):
            for s, e in gt_all.get(stem, {}).get(cls, []):
                ov = np.minimum(starts + chunk_sec, e) - np.maximum(starts, s)
                Y[ov >= args.min_overlap, ci] = 1
        S_list.append(S)
        Y_list.append(Y)
    S = np.concatenate(S_list)
    Y = np.concatenate(Y_list)
    viol = [i for i, c in enumerate(class_ids) if c != "no_violation"]

    args.out.mkdir(parents=True, exist_ok=True)
    per_class = {}
    fig, ax = plt.subplots(figsize=(7, 6))
    for i in viol:
        cls = class_ids[i]
        support = int(Y[:, i].sum())
        if support == 0:
            per_class[cls] = {"ap": None, "p_at_r90": None, "support": 0}
            continue
        ap_score = float(average_precision_score(Y[:, i], S[:, i]))
        prec, rec, _ = precision_recall_curve(Y[:, i], S[:, i])
        p_at_r90 = float(prec[rec >= 0.9].max()) if (rec >= 0.9).any() else 0.0
        per_class[cls] = {"ap": round(ap_score, 4), "p_at_r90": round(p_at_r90, 4), "support": support}
        ax.plot(rec, prec, label=f"{cls} (AP {ap_score:.2f}, n={support})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"per-class PR curves — {args.phase} ({len(stems)} videos, {S.shape[0]} windows)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out / "pr_curves.png", dpi=150)
    plt.close(fig)

    aps = [v["ap"] for v in per_class.values() if v["ap"] is not None]
    metrics = {"phase": args.phase, "n_videos": len(stems), "n_windows": int(S.shape[0]),
               "mAP": round(float(np.mean(aps)), 4) if aps else None,
               "per_class": per_class}

    if args.thresholds and Path(args.thresholds).exists():
        thr = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
        P = np.zeros_like(S, dtype=np.int8)
        for i in viol:
            P[:, i] = (S[:, i] >= thr.get(class_ids[i], {}).get("threshold", 0.25)).astype(np.int8)
        co = np.zeros((len(viol), len(viol)), dtype=np.int64)
        for r, i in enumerate(viol):
            mask = Y[:, i] == 1
            for c, j in enumerate(viol):
                co[r, c] = int(P[mask, j].sum())
        with open(args.out / "cooccurrence_matrix.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["gt\\pred"] + [class_ids[j] for j in viol])
            for r, i in enumerate(viol):
                w.writerow([class_ids[i]] + co[r].tolist())
        metrics["cooccurrence_note"] = "rows: windows where GT class active; cols: predicted-active counts at tuned thresholds"

    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
