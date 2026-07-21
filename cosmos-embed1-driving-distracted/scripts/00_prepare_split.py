"""Scan the shortclips_pos_neg dataset, make a stratified train/val split, emit
metadata + class prompts (HOST-side, Python stdlib only — no torch/pip).

Dataset layout (each .mp4 is one ~3s single-activity clip):
    <root>/negative/*.mp4
    <root>/positive/<class>/*.mp4        class in the 6 distracted-driving folders

Single-label 7-way problem: 6 violations + `no_violation` (the negatives).
Split is stratified per class (val fraction applied within each class so rare
classes keep val examples). Deterministic given --seed.

Writes into --out-dir:
  metadata.json       list of {video, video_id, label_name, label, split}
                      `video` is POSIX-relative to the dataset root (mounted /data)
  class_prompts.json  {class_name: {caption, label}}  — the zero-shot captions
  split_stats.json    per-class train/val counts

Run:
  python scripts/00_prepare_split.py \
    --data-root "D:/research_data/driving_violation/shortclips_pos_neg/shortclips_pos_neg" \
    --out-dir workspace/splits --val-frac 0.2 --seed 0
"""

import argparse
import json
import random
from pathlib import Path

NEGATIVE_CLASS = "no_violation"

# Zero-shot captions. INVARIANT: these are the only caption strings in the pipeline.
# Phrased to separate the two phone folders (operating/looking vs. held-to-ear call).
CAPTIONS = {
    "drinking": "the driver is drinking from a bottle or cup while driving",
    "eating": "the driver is eating food while driving",
    "interacting_with_phone": "the driver is looking at and operating a handheld mobile phone, texting or browsing, while driving",
    "reading_newspaper": "the driver is reading a newspaper or paper document while driving",
    "talking_on_phone": "the driver is holding a phone to their ear and talking on a phone call while driving",
    "working_on_laptop": "the driver is using a laptop computer while driving",
    NEGATIVE_CLASS: "a person driving normally with both hands on the steering wheel, attentive and not distracted",
}

# label ints are assigned by sorted class name so every downstream file agrees.
CLASS_ORDER = sorted(CAPTIONS)  # alphabetical; no_violation lands among them, fine


def collect(data_root: Path) -> dict[str, list[Path]]:
    """class_name -> list of clip paths."""
    out: dict[str, list[Path]] = {}
    neg = data_root / "negative"
    if not neg.is_dir():
        raise SystemExit(f"missing {neg}")
    out[NEGATIVE_CLASS] = sorted(neg.glob("*.mp4"))
    pos = data_root / "positive"
    for d in sorted(p for p in pos.iterdir() if p.is_dir()):
        out[d.name] = sorted(d.glob("*.mp4"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = args.data_root.resolve()
    per_class = collect(root)

    unexpected = set(per_class) - set(CAPTIONS)
    if unexpected:
        raise SystemExit(f"folders with no caption defined: {sorted(unexpected)}")
    label_of = {name: CLASS_ORDER.index(name) for name in CLASS_ORDER}

    rng = random.Random(args.seed)
    rows, stats = [], {}
    for name in sorted(per_class):
        clips = per_class[name][:]
        rng.shuffle(clips)
        n_val = max(1, round(len(clips) * args.val_frac)) if clips else 0
        val_set = set(clips[:n_val])
        for p in clips:
            rel = p.relative_to(root).as_posix()
            rows.append({
                "video": rel,
                "video_id": rel.replace("/", "__")[:-4],  # drop .mp4
                "label_name": name,
                "label": label_of[name],
                "split": "val" if p in val_set else "train",
            })
        stats[name] = {"total": len(clips), "train": len(clips) - n_val, "val": n_val}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metadata.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    prompts = {name: {"caption": CAPTIONS[name], "label": label_of[name]}
               for name in CLASS_ORDER}
    (args.out_dir / "class_prompts.json").write_text(
        json.dumps(prompts, indent=2), encoding="utf-8")
    (args.out_dir / "split_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")

    tr = sum(s["train"] for s in stats.values())
    va = sum(s["val"] for s in stats.values())
    print(f"dataset root: {root}")
    print(f"classes ({len(CLASS_ORDER)}): {CLASS_ORDER}")
    print(f"{'class':<26}{'label':>6}{'total':>8}{'train':>8}{'val':>6}")
    for name in CLASS_ORDER:
        s = stats[name]
        print(f"{name:<26}{label_of[name]:>6}{s['total']:>8}{s['train']:>8}{s['val']:>6}")
    print(f"{'TOTAL':<26}{'':>6}{tr+va:>8}{tr:>8}{va:>6}")
    print(f"wrote metadata.json ({len(rows)} clips), class_prompts.json, split_stats.json "
          f"to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
