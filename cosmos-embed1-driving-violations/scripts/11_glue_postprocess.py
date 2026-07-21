"""Post-process score matrices into flags/events — isolated or glue mode (host, stdlib).

  isolated: threshold RAW per-window scores (no smoothing, min_consec=1)
  glue:     median-filter trajectories (per-class window), threshold,
            hysteresis (min_consec), merge into event segments

Inputs: scores_<stem>.csv files from 10_infer_chunks.py, prompts.json,
optional thresholds.json (from 14_tune_thresholds.py; overrides taxonomy defaults).
Outputs into --out:
  events_<stem>.json  per-video events per class (violations only)
  events.json         combined {video: {class: [events]}}
  flags_<stem>.csv    per-window 0/1 flags after this mode's rule (audit trail)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glue_lib import extract_events, median_filter, read_scores_csv  # noqa: E402
from taxonomy import NEGATIVE_CLASS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path, required=True, help="dir with scores_*.csv")
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--thresholds", type=Path, default=None, help="thresholds.json from 14 (optional)")
    ap.add_argument("--mode", choices=["isolated", "glue"], required=True)
    ap.add_argument("--chunk-sec", type=float, default=2.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    params = {c: {"threshold": p["threshold"], "window": p["window"], "min_consec": p["min_consec"]}
              for c, p in prompts.items()}
    if args.thresholds and args.thresholds.exists():
        for c, p in json.loads(args.thresholds.read_text(encoding="utf-8")).items():
            params[c].update({k: p[k] for k in ("threshold", "window", "min_consec") if k in p})

    args.out.mkdir(parents=True, exist_ok=True)
    combined = {}
    for csv_path in sorted(args.scores.glob("scores_*.csv")):
        stem = csv_path.stem[len("scores_"):]
        starts, cols = read_scores_csv(csv_path)
        per_class_events = {}
        flags = {}
        for cls, traj in cols.items():
            if cls == NEGATIVE_CLASS:
                continue
            p = params[cls]
            if args.mode == "glue":
                smoothed = median_filter(traj, int(p["window"]))
                events = extract_events(starts, smoothed, p["threshold"], int(p["min_consec"]), args.chunk_sec)
                flags[cls] = [1 if s >= p["threshold"] else 0 for s in smoothed]
            else:
                events = extract_events(starts, traj, p["threshold"], 1, args.chunk_sec)
                flags[cls] = [1 if s >= p["threshold"] else 0 for s in traj]
            if events:
                per_class_events[cls] = events
        combined[stem] = per_class_events
        (args.out / f"events_{stem}.json").write_text(json.dumps(per_class_events, indent=1), encoding="utf-8")
        with open(args.out / f"flags_{stem}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            keys = sorted(flags)
            w.writerow(["start_sec"] + keys)
            for i, t0 in enumerate(starts):
                w.writerow([f"{t0:.3f}"] + [flags[k][i] for k in keys])
        n_ev = sum(len(v) for v in per_class_events.values())
        print(f"  {stem}: {n_ev} events ({args.mode})")

    (args.out / "events.json").write_text(json.dumps(combined, indent=1), encoding="utf-8")
    print(f"wrote {args.out / 'events.json'} ({args.mode} mode)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
