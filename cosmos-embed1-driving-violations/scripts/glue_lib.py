"""Shared glue-mode logic (pure stdlib — host machines have no numpy/torch).

Used by 11_glue_postprocess.py (apply), 14_tune_thresholds.py (sweep) and
13_eval_event_level.py (matching), so smoothing/eventing/matching exist once.
"""

import csv
from pathlib import Path
from statistics import median


def read_scores_csv(path: Path) -> tuple[list[float], dict[str, list[float]]]:
    """scores_<stem>.csv -> (starts, {class_id: trajectory})."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        class_ids = header[1:]
        starts, cols = [], {c: [] for c in class_ids}
        for row in reader:
            starts.append(float(row[0]))
            for c, v in zip(class_ids, row[1:]):
                cols[c].append(float(v))
    return starts, cols


def median_filter(traj: list[float], window: int) -> list[float]:
    """Centered median filter; window <= 1 is identity. Edges use shrunken windows."""
    if window <= 1:
        return list(traj)
    half = window // 2
    n = len(traj)
    return [median(traj[max(0, i - half):min(n, i + half + 1)]) for i in range(n)]


def extract_events(starts: list[float], traj: list[float], threshold: float,
                   min_consec: int, chunk_sec: float) -> list[dict]:
    """Contiguous above-threshold runs of >= min_consec windows -> event segments."""
    events = []
    run = []
    for i, s in enumerate(traj):
        if s >= threshold:
            run.append(i)
        else:
            if len(run) >= min_consec:
                events.append(_run_to_event(run, starts, traj, chunk_sec))
            run = []
    if len(run) >= min_consec:
        events.append(_run_to_event(run, starts, traj, chunk_sec))
    return events


def _run_to_event(run: list[int], starts: list[float], traj: list[float], chunk_sec: float) -> dict:
    scores = [traj[i] for i in run]
    return {"start_sec": round(starts[run[0]], 3),
            "end_sec": round(starts[run[-1]] + chunk_sec, 3),
            "peak": round(max(scores), 6),
            "mean": round(sum(scores) / len(scores), 6),
            "n_windows": len(run)}


def temporal_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def match_events(pred: list[dict], gt: list[tuple[float, float]], iou_thr: float) -> tuple[int, int, int]:
    """Greedy match (pred sorted by peak desc) -> (tp, fp, fn)."""
    pred_sorted = sorted(pred, key=lambda e: -e["peak"])
    unmatched = list(range(len(gt)))
    tp = 0
    for e in pred_sorted:
        best_j, best_iou = -1, iou_thr
        for j in unmatched:
            iou = temporal_iou(e["start_sec"], e["end_sec"], gt[j][0], gt[j][1])
            if iou >= best_iou:
                best_j, best_iou = j, iou
        if best_j >= 0:
            unmatched.remove(best_j)
            tp += 1
    return tp, len(pred) - tp, len(unmatched)


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def read_annotations(path: Path, colmap: dict[str, str] | None = None) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """annotations.csv -> {video_stem: {class: [(start, end), ...]}}."""
    colmap = colmap or {"video": "video", "class": "class", "start": "start_sec", "end": "end_sec"}
    out: dict[str, dict[str, list]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stem = Path(row[colmap["video"]].strip()).stem
            out.setdefault(stem, {}).setdefault(row[colmap["class"]].strip(), []).append(
                (float(row[colmap["start"]]), float(row[colmap["end"]])))
    return out
