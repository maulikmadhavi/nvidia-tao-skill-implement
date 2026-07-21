"""Fit per-violation logistic-regression heads on frozen window embeddings (container-side).

Phase 4a middle tier (HEADS_DESIGN.md): the encoder is untouched; each class
gets a linear head over the L2-normalized window embeddings that
10_infer_chunks.py caches in scores_*.npz under the V key (re-run 10 on this
split if your npz files predate that key). Labels are derived at the window
grid with the same overlap >= 0.5s rule as 12_eval_chunk_level.py; the
no_violation head trains on the complement (windows where no violation is
active), so the output keeps the exact class column set of 10.

Per class: LogisticRegression(class_weight='balanced', C in {0.01, 0.1, 1},
max_iter=2000); C picked by val Average Precision. Unknown band from the val
PR curve: t_lo = highest threshold keeping recall >= 0.95 (below -> confident
negative), t_hi = lowest threshold reaching precision >= 0.90 (above ->
confident positive); 16_score_heads.py flags t_lo <= p < t_hi for review.

Outputs:
  heads.npz          W [C,D], b [C], class_ids, t_lo, t_hi, C_reg, encoder
  heads_report.json  per class: positives, chosen C, head vs prompt val AP,
                     band, val unknown-rate  (the Phase 4b go/no-go table)

GATE (warn only): any class with < 10 positive train windows -> head
unreliable, keep the zero-shot prompt score for that class.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glue_lib import read_annotations  # noqa: E402
from taxonomy import NEGATIVE_CLASS  # noqa: E402

C_GRID = (0.01, 0.1, 1.0)


def load_split(scores_dir: Path, stems: list[str], gt_all: dict, min_overlap: float):
    """Concatenate (V, S, Y) over stems; Y at the window grid via the overlap rule."""
    Vs, Ss, Ys = [], [], []
    meta = None  # (class_ids, chunk_sec, encoder)
    for stem in stems:
        path = scores_dir / f"scores_{stem}.npz"
        if not path.exists():
            raise FileNotFoundError(f"missing {path} — run 10_infer_chunks on this split first")
        z = np.load(path, allow_pickle=False)
        if "V" not in z.files:
            raise KeyError(f"{path} has no V key — re-run 10_infer_chunks.py (it now caches embeddings)")
        class_ids = [str(c) for c in z["class_ids"]]
        encoder = str(z["model"])
        if meta is None:
            meta = (class_ids, float(z["chunk_sec"]), encoder)
        elif class_ids != meta[0] or encoder != meta[2]:
            raise ValueError(f"{path}: class_ids/encoder differ from the other scores files")
        starts = z["starts"]
        Y = np.zeros((len(starts), len(class_ids)), dtype=np.int8)
        for ci, cls in enumerate(class_ids):
            if cls == NEGATIVE_CLASS:
                continue
            for s, e in gt_all.get(stem, {}).get(cls, []):
                ov = np.minimum(starts + meta[1], e) - np.maximum(starts, s)
                Y[ov >= min_overlap, ci] = 1
        if NEGATIVE_CLASS in class_ids:
            ni = class_ids.index(NEGATIVE_CLASS)
            viol = [i for i in range(len(class_ids)) if i != ni]
            Y[:, ni] = (Y[:, viol].max(axis=1) == 0).astype(np.int8)
        Vs.append(z["V"])
        Ss.append(z["S"])
        Ys.append(Y)
    return np.concatenate(Vs), np.concatenate(Ss), np.concatenate(Ys), meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path, required=True, help="dir with scores_*.npz incl. V (from 10)")
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--train-videos", type=Path, required=True, help="train stems, one per line")
    ap.add_argument("--val-videos", type=Path, required=True, help="val stems, one per line")
    ap.add_argument("--min-overlap", type=float, default=0.5)
    ap.add_argument("--recall-floor", type=float, default=0.95, help="val recall floor defining t_lo")
    ap.add_argument("--precision-floor", type=float, default=0.90, help="val precision floor defining t_hi")
    ap.add_argument("--out", type=Path, required=True, help="heads.npz path")
    ap.add_argument("--report", type=Path, default=None, help="heads_report.json path")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, precision_recall_curve

    gt_all = read_annotations(args.annotations)
    tr_stems = [ln.strip() for ln in args.train_videos.read_text(encoding="utf-8").splitlines() if ln.strip()]
    va_stems = [ln.strip() for ln in args.val_videos.read_text(encoding="utf-8").splitlines() if ln.strip()]

    Vtr, _Str, Ytr, meta = load_split(args.scores, tr_stems, gt_all, args.min_overlap)
    Vva, Sva, Yva, meta_va = load_split(args.scores, va_stems, gt_all, args.min_overlap)
    if meta_va[0] != meta[0] or meta_va[2] != meta[2]:
        print("[FAIL] train and val scores disagree on class_ids/encoder", file=sys.stderr)
        return 1
    class_ids, _, encoder = meta
    C = len(class_ids)
    D = Vtr.shape[1]
    print(f"train: {Vtr.shape[0]} windows ({len(tr_stems)} videos)  "
          f"val: {Vva.shape[0]} windows ({len(va_stems)} videos)  dim {D}  encoder {encoder}")

    W = np.zeros((C, D), dtype=np.float32)
    b = np.zeros(C, dtype=np.float32)
    t_lo = np.ones(C, dtype=np.float32)
    t_hi = np.ones(C, dtype=np.float32)
    C_reg = np.zeros(C, dtype=np.float32)
    per_class = {}
    warns = []
    for ci, cls in enumerate(class_ids):
        ytr, yva = Ytr[:, ci], Yva[:, ci]
        n_tr, n_va = int(ytr.sum()), int(yva.sum())
        row = {"train_pos": n_tr, "val_pos": n_va, "C": None, "val_ap_head": None,
               "val_ap_prompt": None, "t_lo": None, "t_hi": None, "val_unknown_rate": None}
        if n_va:
            row["val_ap_prompt"] = round(float(average_precision_score(yva, Sva[:, ci])), 4)
        if n_tr < 10:
            warns.append(f"{cls}: only {n_tr} positive train windows — head unreliable, keep the prompt score")
        if n_tr == 0 or n_tr == len(ytr):
            # single-class train set: no separating head; pin the sigmoid to the prior
            b[ci] = -10.0 if n_tr == 0 else 10.0
            row["t_lo"], row["t_hi"] = 1.0, 1.0
            per_class[cls] = row
            continue

        # AP ties (AP is rank-based) MUST break toward the larger C: heavy L2 +
        # balanced weights pin every sigmoid near 0.5, a range the downstream
        # 0.02-step threshold grid cannot resolve. No val positives -> keep the
        # most regularized C.
        best = None  # (val_ap, C, clf, p_va)
        for c in C_GRID:
            clf = LogisticRegression(class_weight="balanced", C=c, max_iter=2000)
            clf.fit(Vtr, ytr)
            p_va = clf.predict_proba(Vva)[:, 1]
            ap_val = float(average_precision_score(yva, p_va)) if n_va else -1.0
            if best is None or (n_va and ap_val >= best[0]):
                best = (ap_val, c, clf, p_va)
        ap_val, c, clf, p_va = best
        W[ci] = clf.coef_[0]
        b[ci] = float(clf.intercept_[0])
        C_reg[ci] = c
        row["C"] = c
        if n_va:
            row["val_ap_head"] = round(ap_val, 4)
            prec, rec, thr = precision_recall_curve(yva, p_va)
            lo_idx = np.where(rec[:-1] >= args.recall_floor)[0]
            hi_idx = np.where(prec[:-1] >= args.precision_floor)[0]
            t_lo[ci] = float(thr[lo_idx].max()) if lo_idx.size else 0.0
            t_hi[ci] = float(thr[hi_idx].min()) if hi_idx.size else 1.0
            if t_lo[ci] >= t_hi[ci]:
                warns.append(f"{cls}: degenerate unknown band (t_lo {t_lo[ci]:.3f} >= t_hi {t_hi[ci]:.3f}) — band is empty")
            row["val_unknown_rate"] = round(float(np.mean((p_va >= t_lo[ci]) & (p_va < t_hi[ci]))), 4)
        else:
            warns.append(f"{cls}: no val positives — C falls back to {c}, no unknown band (t_lo=t_hi=1)")
        row["t_lo"] = round(float(t_lo[ci]), 4)
        row["t_hi"] = round(float(t_hi[ci]), 4)
        per_class[cls] = row

    def fmt(v, spec):
        return format(v, spec) if v is not None else "-".rjust(len(format(0, spec)))

    print(f"\n{'class':28} {'trP':>6} {'vaP':>5} {'C':>5} {'headAP':>7} {'promptAP':>8} {'t_lo':>6} {'t_hi':>6} {'unk':>6}")
    for cls in class_ids:
        r = per_class[cls]
        print(f"{cls:28} {r['train_pos']:>6} {r['val_pos']:>5} {fmt(r['C'], '.2f'):>5} "
              f"{fmt(r['val_ap_head'], '.4f'):>7} {fmt(r['val_ap_prompt'], '.4f'):>8} "
              f"{fmt(r['t_lo'], '.3f'):>6} {fmt(r['t_hi'], '.3f'):>6} {fmt(r['val_unknown_rate'], '.3f'):>6}")
    for w in warns:
        print(f"GATE WARN: {w}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, W=W, b=b, class_ids=np.array(class_ids),
             t_lo=t_lo, t_hi=t_hi, C_reg=C_reg, encoder=np.array(encoder))
    print(f"wrote {args.out}")
    if args.report:
        report = {"encoder": encoder,
                  "n_train_windows": int(Vtr.shape[0]), "n_val_windows": int(Vva.shape[0]),
                  "recall_floor": args.recall_floor, "precision_floor": args.precision_floor,
                  "per_class": per_class, "warnings": warns}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
