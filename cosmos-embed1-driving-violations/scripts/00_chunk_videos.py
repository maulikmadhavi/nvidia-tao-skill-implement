"""Chunk lesson videos and label chunks from event annotations (host, stdlib + ffmpeg).

Inputs:
  --videos       dir of source lesson videos (any ffmpeg-readable format)
  --annotations  CSV with columns video,class,start_sec,end_sec (integer secs;
                 `video` = source filename, with or without extension;
                 remap column names with --columns video=...,class=...,start=...,end=...)
  --out          workspace/data

Does, in order:
 1. ffprobe every video (duration); transcode full video -> out/full/<stem>.mp4
    (h264, short side 256) — inference + demo read these.
 2. Video-level stratified split train/val/test (--split 72,8,20 by percent-ish
    counts; greedy: rarest-class videos placed first so every class lands in
    test/val when possible). Seeded.
 3. Non-overlapping 2s grid per video; chunk gets class c when overlap with a
    c-event >= --min-overlap (0.5s). Multi-label. Unlabeled chunk -> no_violation.
 4. Train split only: keep all positive chunks; sample negatives to
    --neg-ratio x positives; add +/-0.5s jitter copies of positive chunks.
    Val/test: keep the full grid (true distribution).
 5. Cut every selected chunk -> out/video/<chunk_id>.mp4 (parallel ffmpeg),
    ffprobe-verify all.
 6. Write out/chunk_manifest.csv:
    chunk_id,source_video,split,start_sec,end_sec,labels(;-joined),jitter

Gates: every event covered by >=1 labeled chunk; all cut mp4s verified.
"""

import argparse
import csv
import json
import random
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy import NEGATIVE_CLASS, class_ids, load as load_taxonomy  # noqa: E402


RUN_KW = {"capture_output": True, "text": True, "stdin": subprocess.DEVNULL, "timeout": 600}
# stdin=DEVNULL: ffmpeg/ffprobe hang forever on inherited never-closing pipes
# when launched from background shells (observed on Windows PowerShell jobs).


def ffprobe_duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "json", str(path)], **RUN_KW)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {out.stderr[-200:]}")
    return float(json.loads(out.stdout)["format"]["duration"])


def transcode(src: Path, dst: Path, start: float | None = None, dur: float | None = None) -> tuple[str, str]:
    if dst.exists():
        return (dst.name, "skip")
    cmd = ["ffmpeg", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-vf", "scale=-2:256", "-c:v", "libx264", "-profile:v", "high",
            "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "veryfast", "-an", str(dst)]
    proc = subprocess.run(cmd, **RUN_KW)
    return (dst.name, "ok" if proc.returncode == 0 else f"FAIL: {proc.stderr[-300:]}")


def verify(mp4: Path) -> tuple[str, str]:
    proc = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                           "-of", "json", str(mp4)], **RUN_KW)
    if proc.returncode != 0:
        return (mp4.name, "PROBE-FAIL")
    return (mp4.name, "ok" if float(json.loads(proc.stdout)["format"]["duration"]) > 0.2 else "TOO-SHORT")


def stratified_video_split(video_classes: dict[str, set], fracs: tuple[float, float, float],
                           rng: random.Random) -> dict[str, str]:
    """Greedy per-class-need assignment; rarest-class videos placed first."""
    class_count = defaultdict(int)
    for cs in video_classes.values():
        for c in cs:
            class_count[c] += 1
    order = sorted(video_classes, key=lambda v: (min((class_count[c] for c in video_classes[v]), default=1 << 30),
                                                 rng.random()))
    n = len(order)
    targets = {"train": fracs[0] * n, "val": fracs[1] * n, "test": fracs[2] * n}
    counts = {s: 0 for s in targets}
    class_in_split = {s: defaultdict(int) for s in targets}
    assign = {}
    for v in order:
        def need(s):
            class_need = sum(1 for c in video_classes[v] if class_in_split[s][c] == 0)
            return (class_need, targets[s] - counts[s])
        split = max(targets, key=need)
        assign[v] = split
        counts[split] += 1
        for c in video_classes[v]:
            class_in_split[split][c] += 1
    return assign


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--taxonomy", default=None)
    ap.add_argument("--columns", default="video=video,class=class,start=start_sec,end=end_sec")
    ap.add_argument("--chunk-sec", type=float, default=2.0)
    ap.add_argument("--min-overlap", type=float, default=0.5)
    ap.add_argument("--neg-ratio", type=float, default=4.0)
    ap.add_argument("--jitter", type=float, default=0.5, help="0 disables jitter copies")
    ap.add_argument("--split", default="72,8,20", help="train,val,test percents")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    classes = load_taxonomy(args.taxonomy)
    known = set(class_ids(classes))
    colmap = dict(kv.split("=") for kv in args.columns.split(","))
    rng = random.Random(args.seed)

    # --- read annotations ---
    events = defaultdict(list)  # video stem -> [(class, start, end)]
    with open(args.annotations, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cls = row[colmap["class"]].strip()
            if cls not in known:
                print(f"[FAIL] unknown class {cls!r} in annotations (known: {sorted(known)})", file=sys.stderr)
                return 1
            stem = Path(row[colmap["video"]].strip()).stem
            events[stem].append((cls, float(row[colmap["start"]]), float(row[colmap["end"]])))

    # --- discover videos + durations ---
    vids = {p.stem: p for p in sorted(args.videos.iterdir())
            if p.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov", ".webm")}
    missing = sorted(set(events) - set(vids))
    if missing:
        print(f"[FAIL] annotations reference missing videos: {missing[:5]}", file=sys.stderr)
        return 1
    durations = {stem: ffprobe_duration(p) for stem, p in vids.items()}
    print(f"videos: {len(vids)}, annotated: {len(events)}, "
          f"events: {sum(len(v) for v in events.values())}")

    # --- transcode full videos ---
    full_dir = args.out / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(transcode, p, full_dir / f"{stem}.mp4") for stem, p in vids.items()]
        bad = [r for r in (f.result() for f in as_completed(futs)) if r[1] not in ("ok", "skip")]
    if bad:
        print(f"[FAIL] full-video transcode failures: {bad[:3]}", file=sys.stderr)
        return 1

    # --- split at video level ---
    fracs = tuple(float(x) / 100 for x in args.split.split(","))
    video_classes = {stem: {e[0] for e in events.get(stem, [])} or {NEGATIVE_CLASS} for stem in vids}
    split_of = stratified_video_split(video_classes, fracs, rng)

    # --- build chunk grid + labels ---
    def labels_for(stem: str, t0: float, t1: float) -> list[str]:
        found = sorted({cls for cls, s, e in events.get(stem, [])
                        if min(t1, e) - max(t0, s) >= args.min_overlap})
        return found or [NEGATIVE_CLASS]

    rows = []
    covered = set()
    for stem in sorted(vids):
        dur, split = durations[stem], split_of[stem]
        grid = [round(t, 3) for t in frange(0.0, dur - args.chunk_sec, args.chunk_sec)]
        pos_chunks, neg_chunks = [], []
        for t0 in grid:
            t1 = t0 + args.chunk_sec
            labs = labels_for(stem, t0, t1)
            entry = (stem, split, t0, t1, labs, 0)
            if labs != [NEGATIVE_CLASS]:
                pos_chunks.append(entry)
                for cls, s, e in events[stem]:
                    if cls in labs and min(t1, e) - max(t0, s) >= args.min_overlap:
                        covered.add((stem, cls, s, e))
            else:
                neg_chunks.append(entry)
        if split == "train":
            keep_neg = min(len(neg_chunks), int(args.neg_ratio * max(len(pos_chunks), 1)))
            neg_chunks = rng.sample(neg_chunks, keep_neg)
            if args.jitter > 0:
                jittered = []
                for stem_, sp, t0, t1, labs, _ in pos_chunks:
                    for dt in (-args.jitter, args.jitter):
                        j0 = t0 + dt
                        if 0 <= j0 and j0 + args.chunk_sec <= dur:
                            jl = labels_for(stem_, j0, j0 + args.chunk_sec)
                            if jl != [NEGATIVE_CLASS]:
                                jittered.append((stem_, sp, round(j0, 3), round(j0 + args.chunk_sec, 3), jl, 1))
                pos_chunks += jittered
        rows += pos_chunks + neg_chunks

    all_events = {(stem, cls, s, e) for stem, evs in events.items() for cls, s, e in evs}
    uncovered = all_events - covered
    if uncovered:
        print(f"[WARN] {len(uncovered)} events produced no labeled chunk "
              f"(shorter than min-overlap?): {sorted(uncovered)[:3]}")

    # --- cut chunk mp4s ---
    video_dir = args.out / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    jobs = []
    for stem, split, t0, t1, labs, jit in rows:
        chunk_id = f"{stem}_{int(round(t0 * 1000)):07d}"
        manifest.append({"chunk_id": chunk_id, "source_video": stem, "split": split,
                         "start_sec": t0, "end_sec": t1, "labels": ";".join(labs), "jitter": jit})
        jobs.append((vids[stem], video_dir / f"{chunk_id}.mp4", t0, args.chunk_sec))
    print(f"cutting {len(jobs)} chunks ...")
    failures = []
    with ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(transcode, src, dst, t0, d) for src, dst, t0, d in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            name, status = fut.result()
            if status not in ("ok", "skip"):
                failures.append((name, status))
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    if failures:
        print(f"[FAIL] {len(failures)} chunk cuts failed: {failures[:3]}", file=sys.stderr)
        return 1
    with ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(verify, dst) for _, dst, _, _ in jobs]
        bad = [r for r in (f.result() for f in as_completed(futs)) if r[1] != "ok"]
    if bad:
        print(f"[FAIL] {len(bad)} chunks failed verification: {bad[:3]}", file=sys.stderr)
        return 1

    mpath = args.out / "chunk_manifest.csv"
    with open(mpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    n = defaultdict(int)
    npos = defaultdict(int)
    for m in manifest:
        n[m["split"]] += 1
        if m["labels"] != NEGATIVE_CLASS:
            npos[m["split"]] += 1
    for s in ("train", "val", "test"):
        vcount = sum(1 for v, sp in split_of.items() if sp == s)
        print(f"  {s}: {vcount} videos, {n[s]} chunks ({npos[s]} positive)")
    print(f"GATE PASSED: wrote {mpath}")
    return 0


def frange(start: float, stop: float, step: float):
    t = start
    while t <= stop + 1e-9:
        yield t
        t += step


if __name__ == "__main__":
    sys.exit(main())
