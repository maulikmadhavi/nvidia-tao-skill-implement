"""Finetune tier 1: linear probe (multinomial logistic regression) on the frozen
Cosmos-Embed1 embeddings (CONTAINER-side; CPU, seconds).

Fits LogisticRegression on the TRAIN clip embeddings, evaluates on VAL, and prints
the lift over zero-shot. This is the cheap finetune the heads-design ladder calls
for before touching the encoder with LoRA: no new deps (sklearn is baked in),
retrainable in seconds, and it can only use signal the frozen encoder already
carries — so if it clears the bar, encoder LoRA is unnecessary.

class_weight='balanced' handles the 605-vs-50 negative/positive imbalance.
C is picked by 5-fold CV macro-F1 on train (falls back to no-CV for tiny classes).

Writes into --out: metrics_probe.json, per_class_probe.csv, confusion_probe.csv,
confusion_probe.png, probe.npz (coef/intercept/classes for later scoring).

Run via run_container.ps1:
  python /exp/scripts/20_linear_probe.py \
    --embeddings /results/baseline/embeddings.npz --out /results/baseline/probe
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cv", type=int, default=5)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(args.embeddings, allow_pickle=False)
    V = z["video_embs"]
    labels, splits = z["labels"], z["splits"].astype(str)
    T = z["class_prompt_embs"]
    class_names = [str(c) for c in z["class_names"]]
    C = len(class_names)

    tr, va = splits == "train", splits == "val"
    Xtr, ytr = V[tr], labels[tr]
    Xva, yva = V[va], labels[va]

    # C selected by CV macro-F1; min class count caps the fold count.
    min_per_class = np.bincount(ytr, minlength=C).min()
    n_splits = int(max(2, min(args.cv, min_per_class)))
    grid = {"C": [0.1, 1.0, 10.0, 100.0]}
    base = LogisticRegression(class_weight="balanced", max_iter=5000)
    search = GridSearchCV(base, grid, scoring="f1_macro",
                          cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0))
    search.fit(Xtr, ytr)
    clf = search.best_estimator_
    best_C = search.best_params_["C"]

    pred = clf.predict(Xva)
    acc = accuracy_score(yva, pred)
    mp, mr, mf1, _ = precision_recall_fscore_support(yva, pred, average="macro", zero_division=0)
    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
        yva, pred, labels=range(C), average=None, zero_division=0)

    # zero-shot on the same val split, for the lift line
    Szs = Xva @ T.T
    zs_pred = Szs.argmax(1)
    zs_acc = accuracy_score(yva, zs_pred)
    _, _, zs_mf1, _ = precision_recall_fscore_support(yva, zs_pred, average="macro", zero_division=0)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "per_class_probe.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "label", "precision", "recall", "f1", "support"])
        for i, name in enumerate(class_names):
            w.writerow([name, i, f"{per_p[i]:.4f}", f"{per_r[i]:.4f}",
                        f"{per_f1[i]:.4f}", int(per_sup[i])])

    cm = confusion_matrix(yva, pred, labels=range(C))
    with open(args.out / "confusion_probe.csv", "w", newline="", encoding="utf-8") as f:
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
    ax.set_title(f"Linear probe (val, N={len(yva)}, acc={acc:.3f}, C={best_C})")
    fig.colorbar(im, shrink=0.7); fig.tight_layout()
    fig.savefig(args.out / "confusion_probe.png", dpi=150); plt.close(fig)

    # persist the probe so it can score new clips without refitting
    coef = np.zeros((C, V.shape[1]), np.float32)
    intercept = np.zeros(C, np.float32)
    for row, cls in enumerate(clf.classes_):
        coef[cls] = clf.coef_[row] if clf.coef_.shape[0] > 1 else clf.coef_[0]
        intercept[cls] = clf.intercept_[row] if len(clf.intercept_) > 1 else clf.intercept_[0]
    np.savez(args.out / "probe.npz", coef=coef, intercept=intercept,
             class_names=np.array(class_names), best_C=np.array(best_C),
             model=z["model"])

    metrics = {
        "phase": "linear_probe",
        "best_C": float(best_C),
        "cv_folds": n_splits,
        "num_train": int(tr.sum()),
        "num_val": int(va.sum()),
        "val_accuracy": float(acc),
        "val_precision_macro": float(mp),
        "val_recall_macro": float(mr),
        "val_f1_macro": float(mf1),
        "zeroshot_val_accuracy": float(zs_acc),
        "zeroshot_val_f1_macro": float(zs_mf1),
        "accuracy_lift": float(acc - zs_acc),
        "f1_macro_lift": float(mf1 - zs_mf1),
        "per_class_f1": {class_names[i]: float(per_f1[i]) for i in range(C)},
    }
    (args.out / "metrics_probe.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"\nlinear probe val acc {acc:.3f} (zero-shot {zs_acc:.3f}, "
          f"lift {acc - zs_acc:+.3f}); macro-F1 {mf1:.3f} (zs {zs_mf1:.3f}, "
          f"lift {mf1 - zs_mf1:+.3f})")
    print(f"wrote probe + metrics to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
