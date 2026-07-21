"""Transcode sampled HMDB51 .avi clips to .mp4 (stdlib + ffmpeg/ffprobe).

Reads split_manifest.csv, writes workspace/data/video/{video_id}.mp4 with:
  ffmpeg -y -i in.avi -vf scale=-2:256 -c:v libx264 -profile:v high
         -pix_fmt yuv420p -crf 20 -preset veryfast -an out.mp4
(short side 256 -> clean 224x224 model resize headroom; audio dropped; native
fps/duration kept — the model samples 8 frames itself).

Then ffprobe-verifies every output: decodable, h264/yuv420p, duration > 0.5 s.
Fails loudly listing bad clips.
"""

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def transcode_one(src: Path, dst: Path, force: bool) -> tuple[str, str]:
    if dst.exists() and not force:
        return (dst.name, "skip")
    cmd = ["ffmpeg", "-y", "-i", str(src), "-vf", "scale=-2:256",
           "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
           "-crf", "20", "-preset", "veryfast", "-an", str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return (dst.name, f"FFMPEG-FAIL: {proc.stderr[-400:]}")
    return (dst.name, "ok")


def probe_one(mp4: Path) -> tuple[str, str]:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=codec_name,pix_fmt:format=duration",
           "-of", "json", str(mp4)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return (mp4.name, f"PROBE-FAIL: {proc.stderr[-200:]}")
    try:
        info = json.loads(proc.stdout)
        stream = info["streams"][0]
        duration = float(info["format"]["duration"])
    except (KeyError, IndexError, ValueError) as e:
        return (mp4.name, f"PROBE-PARSE-FAIL: {e}")
    if stream.get("codec_name") != "h264" or stream.get("pix_fmt") != "yuv420p":
        return (mp4.name, f"BAD-CODEC: {stream.get('codec_name')}/{stream.get('pix_fmt')}")
    if duration <= 0.5:
        return (mp4.name, f"TOO-SHORT: {duration:.2f}s")
    return (mp4.name, "ok")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True, help="root that original_relpath is relative to")
    ap.add_argument("--out", type=Path, required=True, help="workspace/data/video")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = [(args.raw / r["original_relpath"], args.out / f"{r['video_id']}.mp4") for r in rows]
    missing_src = [str(src) for src, _ in jobs if not src.exists()]
    if missing_src:
        print(f"[FAIL] {len(missing_src)} source .avi missing, first: {missing_src[:3]}", file=sys.stderr)
        return 1

    failures = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(transcode_one, src, dst, args.force): dst for src, dst in jobs}
        for fut in as_completed(futs):
            name, status = fut.result()
            done += 1
            if status not in ("ok", "skip"):
                failures.append((name, status))
            if done % 100 == 0:
                print(f"  transcoded {done}/{len(jobs)}", flush=True)
    if failures:
        print(f"[FAIL] {len(failures)} transcode failures:", file=sys.stderr)
        for name, status in failures[:10]:
            print(f"  {name}: {status}", file=sys.stderr)
        return 1
    print(f"transcode complete: {len(jobs)} clips")

    print("verifying with ffprobe ...")
    bad = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(probe_one, dst) for _, dst in jobs]
        for fut in as_completed(futs):
            name, status = fut.result()
            if status != "ok":
                bad.append((name, status))
    if bad:
        print(f"[FAIL] {len(bad)} clips failed verification:", file=sys.stderr)
        for name, status in bad[:10]:
            print(f"  {name}: {status}", file=sys.stderr)
        return 1
    print(f"GATE PASSED: all {len(jobs)} mp4s verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
