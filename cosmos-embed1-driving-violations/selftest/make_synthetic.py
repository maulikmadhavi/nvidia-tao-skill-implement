"""Build synthetic 'lesson' videos + annotations from HMDB clips (host, stdlib + ffmpeg).

Purpose: validate the whole driving-violations pipeline end-to-end WITHOUT real
driving data. HMDB actions stand in for violations (the pretrained model scores
them well zero-shot, so glue must recover the planted events):

  pseudo-violation  <- HMDB class   caption
  smoking           <- smoke        "a video of a person smoking"
  drinking          <- drink        "a video of a person drinking"
  pullups           <- pullup       "a video of a person doing pull ups"
  no_violation      <- walk/talk    "a video of a person walking"

Each lesson = normalized background clips (walk/talk) with violation clips
planted at known offsets. Emits:
  <out>/videos/lesson_XX.mp4
  <out>/annotations.csv          video,class,start_sec,end_sec  (ints, real schema)
  <out>/taxonomy_selftest.json   taxonomy override for the pipeline scripts
"""

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from pathlib import Path

PSEUDO = {"smoking": "smoke", "drinking": "drink", "pullups": "pullup"}
BACKGROUND = ["walk", "talk"]

TAXONOMY = [
    {"id": "smoking", "caption": "a video of a person smoking",
     "kind": "action", "window": 3, "threshold": 0.25, "min_consec": 2},
    {"id": "drinking", "caption": "a video of a person drinking",
     "kind": "action", "window": 3, "threshold": 0.25, "min_consec": 2},
    {"id": "pullups", "caption": "a video of a person doing pull ups",
     "kind": "action", "window": 3, "threshold": 0.25, "min_consec": 2},
    {"id": "no_violation", "caption": "a video of a person walking",
     "kind": "state", "window": 5, "threshold": 0.5, "min_consec": 1},
]


def run(cmd: list[str]) -> None:
    # stdin=DEVNULL + timeout: ffmpeg/ffprobe can hang forever on an inherited
    # never-closing pipe when launched from a background shell on Windows.
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:4])}... failed: {proc.stderr[-300:]}")


def duration_of(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True,
                         stdin=subprocess.DEVNULL, timeout=60)
    return float(json.loads(out.stdout)["format"]["duration"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hmdb", type=Path,
                    default=Path(__file__).resolve().parents[2] / "cosmos-embed1-hmdb51-video-text-search" / "workspace" / "data" / "video",
                    help="dir with HMDB chunk mp4s (default: sibling HMDB experiment)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lessons", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    norm_dir = args.out / "_norm"
    videos_dir = args.out / "videos"
    for d in (norm_dir, videos_dir):
        d.mkdir(parents=True, exist_ok=True)

    def pick(prefix: str, n: int) -> list[Path]:
        pool = sorted(args.hmdb.glob(f"{prefix}_*.mp4"))
        if len(pool) < n:
            raise RuntimeError(f"not enough {prefix} clips in {args.hmdb}")
        return rng.sample(pool, n)

    def normalize(src: Path) -> Path:
        dst = norm_dir / f"n_{src.name}"
        if not dst.exists():
            run(["ffmpeg", "-y", "-i", str(src), "-vf", "scale=320:240,fps=25",
                 "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
                 "-crf", "20", "-preset", "veryfast", "-an", str(dst)])
        return dst

    annotations = []
    for li in range(args.lessons):
        stem = f"lesson_{li:02d}"
        # sequence: bg bg V bg V bg V bg  (one event of each pseudo class, shuffled)
        viol_order = list(PSEUDO)
        rng.shuffle(viol_order)
        seq = []
        bg_iter = iter(normalize(p) for p in pick(rng.choice(BACKGROUND), 4) + pick(rng.choice(BACKGROUND), 4))
        for cls in viol_order:
            seq.append(("bg", next(bg_iter)))
            seq.append((cls, normalize(pick(PSEUDO[cls], 1)[0])))
        seq.append(("bg", next(bg_iter)))
        seq.append(("bg", next(bg_iter)))

        t = 0.0
        concat_lines = []
        for cls, clip in seq:
            d = duration_of(clip)
            if cls != "bg":
                annotations.append({"video": f"{stem}.mp4", "class": cls,
                                    "start_sec": math.floor(t), "end_sec": math.ceil(t + d)})
            concat_lines.append(f"file '{clip.resolve().as_posix()}'")
            t += d
        listfile = args.out / f"_{stem}.txt"
        listfile.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
             "-c", "copy", str(videos_dir / f"{stem}.mp4")])
        print(f"{stem}: {t:.1f}s, {len(viol_order)} planted events", flush=True)

    with open(args.out / "annotations.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video", "class", "start_sec", "end_sec"])
        w.writeheader()
        w.writerows(annotations)
    (args.out / "taxonomy_selftest.json").write_text(json.dumps(TAXONOMY, indent=2), encoding="utf-8")
    print(f"wrote {len(annotations)} annotations, taxonomy override, {args.lessons} lessons -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
