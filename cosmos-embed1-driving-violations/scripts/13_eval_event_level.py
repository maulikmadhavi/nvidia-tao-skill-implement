"""Event-level evaluation: predicted events vs GT annotations (host, stdlib).

Greedy temporal-IoU matching (default IoU >= 0.3) per class per video:
per-class precision/recall/F1 + micro totals. Run once per (phase, mode)
combination, e.g. baseline/isolated, baseline/glue, finetuned/glue.

Inputs: events.json (from 11), annotations.csv, a video list restricting to
one split. Output: JSON report.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glue_lib import match_events, prf, read_annotations  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", type=Path, required=True, help="events.json from 11")
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--videos", type=Path, required=True, help="file: one video stem per line (split filter)")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--label", default="", help="free-text tag stored in the report (phase/mode)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pred_all = json.loads(args.events.read_text(encoding="utf-8"))
    gt_all = read_annotations(args.annotations)
    stems = [ln.strip() for ln in args.videos.read_text(encoding="utf-8").splitlines() if ln.strip()]

    totals = defaultdict(lambda: [0, 0, 0])  # class -> [tp, fp, fn]
    for stem in stems:
        pred_video = pred_all.get(stem, {})
        gt_video = gt_all.get(stem, {})
        for cls in set(pred_video) | set(gt_video):
            tp, fp, fn = match_events(pred_video.get(cls, []), gt_video.get(cls, []), args.iou)
            totals[cls][0] += tp
            totals[cls][1] += fp
            totals[cls][2] += fn

    per_class = {cls: prf(*counts) for cls, counts in sorted(totals.items())}
    micro = prf(sum(c[0] for c in totals.values()),
                sum(c[1] for c in totals.values()),
                sum(c[2] for c in totals.values()))
    report = {"label": args.label, "iou_threshold": args.iou, "n_videos": len(stems),
              "micro": micro, "per_class": per_class}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
