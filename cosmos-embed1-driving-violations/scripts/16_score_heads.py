"""Apply trained heads to cached window embeddings -> probabilities + unknown queue (container, CPU).

Reads every scores_*.npz in --scores (must carry the V key from 10), applies
P = sigmoid(V @ W.T + b), and writes scores_<stem>.csv/.npz into --out in the
exact 10_infer_chunks.py format (S = head probabilities, V passed through), so
11/12/13/14 and the demo run unchanged on the output dir.

Refuses to run if the heads were trained on a different encoder than the
scores (heads.npz `encoder` vs npz `model` — never apply v1 heads to v2
embeddings; refit instead).

Reviewer worklist (violation classes only; unknown = t_lo <= p < t_hi from 15):
  unknown_<stem>.csv   per window x class: 0/1 unknown flags
  review_queue.json    contiguous unknown runs (>= --min-run windows) merged
                       into {video, class, start_sec, end_sec, peak_p},
                       sorted by peak_p descending

WARN: per-class unknown-rate > 20% (a queue nobody can clear helps nobody).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy import NEGATIVE_CLASS  # noqa: E402


def unknown_runs(flags: np.ndarray, min_run: int) -> list[tuple[int, int]]:
    """0/1 vector -> [(first_idx, last_idx)] of contiguous 1-runs with length >= min_run."""
    runs, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start >= min_run:
                runs.append((start, i - 1))
            start = None
    if start is not None and len(flags) - start >= min_run:
        runs.append((start, len(flags) - 1))
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path, required=True, help="dir with scores_*.npz incl. V (from 10)")
    ap.add_argument("--heads", type=Path, required=True, help="heads.npz from 15")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-run", type=int, default=2, help="min consecutive unknown windows per review item")
    ap.add_argument("--warn-unknown", type=float, default=0.20)
    args = ap.parse_args()

    h = np.load(args.heads, allow_pickle=False)
    W, b, t_lo, t_hi = h["W"], h["b"], h["t_lo"], h["t_hi"]
    head_classes = [str(c) for c in h["class_ids"]]
    encoder = str(h["encoder"])

    files = sorted(args.scores.glob("scores_*.npz"))
    if not files:
        print(f"[FAIL] no scores_*.npz in {args.scores}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    unk_count = np.zeros(len(head_classes), dtype=np.int64)
    n_windows = 0
    queue = []
    for path in files:
        z = np.load(path, allow_pickle=False)
        model = str(z["model"])
        if model != encoder:
            print(f"[FAIL] encoder mismatch: heads were trained on '{encoder}' but {path.name} was scored "
                  f"with '{model}' — rescore with the matching encoder or refit the heads (HEADS_DESIGN.md D8)",
                  file=sys.stderr)
            return 1
        if "V" not in z.files:
            print(f"[FAIL] {path.name} has no V key — re-run 10_infer_chunks.py (it now caches embeddings)",
                  file=sys.stderr)
            return 1
        class_ids = [str(c) for c in z["class_ids"]]
        if class_ids != head_classes:
            print(f"[FAIL] class_ids mismatch between {path.name} and {args.heads.name}", file=sys.stderr)
            return 1
        V, starts, chunk_sec = z["V"], z["starts"], float(z["chunk_sec"])
        P = 1.0 / (1.0 + np.exp(-(V @ W.T + b)))

        stem = path.stem[len("scores_"):]
        np.savez(args.out / f"scores_{stem}.npz", S=P, starts=starts, V=V,
                 class_ids=z["class_ids"], model=z["model"],
                 chunk_sec=z["chunk_sec"], stride_sec=z["stride_sec"])
        with open(args.out / f"scores_{stem}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["start_sec"] + class_ids)
            for j, t0 in enumerate(starts):
                w.writerow([f"{t0:.3f}"] + [f"{p:.6f}" for p in P[j]])

        U = ((P >= t_lo) & (P < t_hi)).astype(np.int8)
        with open(args.out / f"unknown_{stem}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["start_sec"] + class_ids)
            for j, t0 in enumerate(starts):
                w.writerow([f"{t0:.3f}"] + U[j].tolist())
        unk_count += U.sum(axis=0)
        n_windows += len(starts)
        for ci, cls in enumerate(class_ids):
            if cls == NEGATIVE_CLASS:
                continue
            for i0, i1 in unknown_runs(U[:, ci], args.min_run):
                queue.append({"video": stem, "class": cls,
                              "start_sec": round(float(starts[i0]), 3),
                              "end_sec": round(float(starts[i1]) + chunk_sec, 3),
                              "peak_p": round(float(P[i0:i1 + 1, ci].max()), 6)})
        print(f"  {stem}: {len(starts)} windows, {int(U.sum())} unknown flags")

    queue.sort(key=lambda q: -q["peak_p"])
    (args.out / "review_queue.json").write_text(json.dumps(queue, indent=1), encoding="utf-8")

    print(f"\n{'class':28} {'unknown-rate':>12}")
    for ci, cls in enumerate(head_classes):
        rate = unk_count[ci] / n_windows if n_windows else 0.0
        flag = "  WARN > 20% — band too wide, review queue not clearable" if rate > args.warn_unknown else ""
        print(f"{cls:28} {rate:>12.4f}{flag}")
    print(f"wrote {len(files)} score files + {len(queue)} review items -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
