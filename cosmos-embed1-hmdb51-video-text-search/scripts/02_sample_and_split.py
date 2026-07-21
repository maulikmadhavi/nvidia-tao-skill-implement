"""Sample HMDB51 clips and build the stratified train/val/test split (stdlib only).

Deterministic: filenames are sorted before random.Random(seed) sampling.
Per class: min(per_class, available) clips; split 22 train / 2 val / 6 test
(val carved from the 80% train side; test = 20%). Assigns sanitized ids
{class}_{0001..} because raw HMDB filenames contain #;()&' and spaces.

Outputs (into --out):
  split_manifest.csv   video_id,class,label,split,caption,original_relpath
  class_prompts.json   {class: {label, caption}}
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

# 51 hand-written gerund captions. INVARIANT: these exact strings are used in
# train/val/test metadata captions, caption_to_label in the evaluate specs,
# class_prompts.json, and the demo. Template: "a video of a person/people ..."
CAPTIONS = {
    "brush_hair": "a video of a person brushing hair",
    "cartwheel": "a video of a person doing a cartwheel",
    "catch": "a video of a person catching something",
    "chew": "a video of a person chewing",
    "clap": "a video of a person clapping hands",
    "climb": "a video of a person climbing",
    "climb_stairs": "a video of a person climbing stairs",
    "dive": "a video of a person diving",
    "draw_sword": "a video of a person drawing a sword",
    "dribble": "a video of a person dribbling a basketball",
    "drink": "a video of a person drinking",
    "eat": "a video of a person eating",
    "fall_floor": "a video of a person falling on the floor",
    "fencing": "a video of people fencing",
    "flic_flac": "a video of a person doing a flic flac backflip",
    "golf": "a video of a person swinging a golf club",
    "handstand": "a video of a person doing a handstand",
    "hit": "a video of a person hitting something",
    "hug": "a video of people hugging",
    "jump": "a video of a person jumping",
    "kick": "a video of a person kicking",
    "kick_ball": "a video of a person kicking a ball",
    "kiss": "a video of people kissing",
    "laugh": "a video of a person laughing",
    "pick": "a video of a person picking something up",
    "pour": "a video of a person pouring liquid",
    "pullup": "a video of a person doing pull ups",
    "punch": "a video of a person punching",
    "push": "a video of a person pushing something",
    "pushup": "a video of a person doing push ups",
    "ride_bike": "a video of a person riding a bike",
    "ride_horse": "a video of a person riding a horse",
    "run": "a video of a person running",
    "shake_hands": "a video of people shaking hands",
    "shoot_ball": "a video of a person shooting a basketball",
    "shoot_bow": "a video of a person shooting a bow and arrow",
    "shoot_gun": "a video of a person shooting a gun",
    "sit": "a video of a person sitting down",
    "situp": "a video of a person doing sit ups",
    "smile": "a video of a person smiling",
    "smoke": "a video of a person smoking",
    "somersault": "a video of a person doing a somersault",
    "stand": "a video of a person standing up",
    "swing_baseball": "a video of a person swinging a baseball bat",
    "sword": "a video of people sword fighting",
    "sword_exercise": "a video of a person doing sword exercises",
    "talk": "a video of a person talking",
    "throw": "a video of a person throwing something",
    "turn": "a video of a person turning around",
    "walk": "a video of a person walking",
    "wave": "a video of a person waving",
}

TRAIN_N, VAL_N, TEST_N = 22, 2, 6  # per class; 22+2 = 80%, 6 = 20% of 30


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True, help="dir containing class_dirs.json (from 01)")
    ap.add_argument("--out", type=Path, required=True, help="workspace/data dir")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-class", type=int, default=30)
    args = ap.parse_args()

    class_dirs = json.loads((args.raw / "class_dirs.json").read_text(encoding="utf-8"))
    missing = sorted(set(CAPTIONS) - set(class_dirs))
    extra = sorted(set(class_dirs) - set(CAPTIONS))
    if missing or extra:
        print(f"[FAIL] class mismatch. missing from disk: {missing}; unknown on disk: {extra}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    rows = []
    for label, cls in enumerate(sorted(CAPTIONS)):
        avis = sorted(Path(class_dirs[cls]).glob("*.avi"), key=lambda p: p.name)
        if len(avis) < args.per_class:
            print(f"[warn] {cls}: only {len(avis)} clips (< {args.per_class})")
        picked = rng.sample(avis, min(args.per_class, len(avis)))
        rng.shuffle(picked)
        for i, avi in enumerate(picked):
            if i < TRAIN_N:
                split = "train"
            elif i < TRAIN_N + VAL_N:
                split = "val"
            elif i < TRAIN_N + VAL_N + TEST_N:
                split = "test"
            else:
                break
            rows.append({
                "video_id": f"{cls}_{i + 1:04d}",
                "class": cls,
                "label": label,
                "split": split,
                "caption": CAPTIONS[cls],
                "original_relpath": str(avi.relative_to(args.raw)),
            })

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "split_manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    prompts = {cls: {"label": label, "caption": CAPTIONS[cls]}
               for label, cls in enumerate(sorted(CAPTIONS))}
    (args.out / "class_prompts.json").write_text(json.dumps(prompts, indent=2), encoding="utf-8")

    counts = {s: sum(1 for r in rows if r["split"] == s) for s in ("train", "val", "test")}
    total = sum(counts.values())
    print(f"total {total} = train {counts['train']} + val {counts['val']} + test {counts['test']}")
    per_class_ok = all(
        [sum(1 for r in rows if r["class"] == c and r["split"] == "train") == TRAIN_N,
         sum(1 for r in rows if r["class"] == c and r["split"] == "val") == VAL_N,
         sum(1 for r in rows if r["class"] == c and r["split"] == "test") == TEST_N]
        for c in CAPTIONS
    )
    expected_total = len(CAPTIONS) * (TRAIN_N + VAL_N + TEST_N)
    if total != expected_total or not per_class_ok:
        print(f"[FAIL] split gate not met (expected {expected_total} with "
              f"{TRAIN_N}/{VAL_N}/{TEST_N} per class — a class may have <30 clips)", file=sys.stderr)
        return 1
    print(f"GATE PASSED: wrote {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
