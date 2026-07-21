"""Tune per-class glue parameters on the VAL split (host, stdlib).

Sweeps threshold x median-window x min_consec per class, maximizing event-level
F1 (IoU >= 0.3) on the val videos' score trajectories. Writes thresholds.json
consumed by 11_glue_postprocess.py and 12_eval_chunk_level.py.

Never tune on test — that's the whole point of the val split.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glue_lib import extract_events, match_events, median_filter, prf, read_annotations, read_scores_csv  # noqa: E402
from taxonomy import NEGATIVE_CLASS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path, required=True, help="dir with scores_*.csv (val videos)")
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--videos", type=Path, required=True, help="val video stems, one per line")
    ap.add_argument("--chunk-sec", type=float, default=2.0)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--thr-grid", default="0.10:0.50:0.01")
    ap.add_argument("--windows", default="1,3,5,9")
    ap.add_argument("--min-consec", default="1,2,3")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    lo, hi, step = (float(x) for x in args.thr_grid.split(":"))
    thresholds = []
    t = lo
    while t <= hi + 1e-9:
        thresholds.append(round(t, 4))
        t += step
    windows = [int(x) for x in args.windows.split(",")]
    consecs = [int(x) for x in args.min_consec.split(",")]

    gt_all = read_annotations(args.annotations)
    stems = [ln.strip() for ln in args.videos.read_text(encoding="utf-8").splitlines() if ln.strip()]
    trajs = {}  # stem -> (starts, {class: traj})
    for stem in stems:
        path = args.scores / f"scores_{stem}.csv"
        if not path.exists():
            print(f"[FAIL] missing {path} — run 10_infer_chunks on the val videos first", file=sys.stderr)
            return 1
        trajs[stem] = read_scores_csv(path)

    class_ids = [c for c in trajs[stems[0]][1] if c != NEGATIVE_CLASS]
    best = {}
    for cls in class_ids:
        best_cfg, best_f1 = None, -1.0
        smoothed_cache = {}  # (stem, window) -> traj
        for w in windows:
            for stem in stems:
                starts, cols = trajs[stem]
                smoothed_cache[(stem, w)] = median_filter(cols[cls], w)
            for thr in thresholds:
                for k in consecs:
                    tp = fp = fn = 0
                    for stem in stems:
                        starts, _ = trajs[stem]
                        events = extract_events(starts, smoothed_cache[(stem, w)], thr, k, args.chunk_sec)
                        gt = gt_all.get(stem, {}).get(cls, [])
                        a, b, c = match_events(events, gt, args.iou)
                        tp, fp, fn = tp + a, fp + b, fn + c
                    f1 = prf(tp, fp, fn)["f1"]
                    if f1 > best_f1:
                        best_f1 = f1
                        best_cfg = {"threshold": thr, "window": w, "min_consec": k,
                                    "val_f1": f1, "val_tp": tp, "val_fp": fp, "val_fn": fn}
        best[cls] = best_cfg
        print(f"  {cls}: {best_cfg}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
