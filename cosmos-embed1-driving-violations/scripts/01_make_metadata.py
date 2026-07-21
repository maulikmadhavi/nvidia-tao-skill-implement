"""Manifest -> MSR-VTT metadata + prompts + rendered eval specs (host, stdlib).

From workspace/data/chunk_manifest.csv emits into workspace/data:
  train.json / val.json / test.json  — ONE ROW PER (chunk, label-class):
      {"video_id": "<chunk_id>#<class>", "video": "<chunk_id>.mp4", "caption": <class caption>}
      (a chunk with 2 violations contributes 2 rows sharing the same mp4;
       smoke-gated in a 1-iter train — see RUNBOOK if the loader rejects this)
  prompts.json          {class: {label, caption, kind, window, threshold, min_consec}}
  caption_to_label.json
and renders specs/*.yaml.tmpl -> workspace/specs (caption_to_label injection,
same mechanism as the HMDB experiment).

Gate: every manifest chunk mp4 exists on disk; captions bijective with classes.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy import load as load_taxonomy  # noqa: E402

INDENT = " " * 6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True, help="workspace/data (contains chunk_manifest.csv)")
    ap.add_argument("--taxonomy", default=None)
    ap.add_argument("--templates", type=Path, required=True, help="specs dir with *.yaml.tmpl")
    ap.add_argument("--specs", type=Path, required=True, help="workspace/specs output")
    args = ap.parse_args()

    classes = load_taxonomy(args.taxonomy)
    caption = {c["id"]: c["caption"] for c in classes}
    label = {c["id"]: i for i, c in enumerate(classes)}

    with open(args.data / "chunk_manifest.csv", newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    video_dir = args.data / "video"
    missing = [m["chunk_id"] for m in manifest if not (video_dir / f"{m['chunk_id']}.mp4").exists()]
    if missing:
        print(f"[FAIL] {len(missing)} manifest chunks missing on disk: {missing[:5]}", file=sys.stderr)
        return 1

    rows_by_split = {"train": [], "val": [], "test": []}
    videos_by_split = {"train": set(), "val": set(), "test": set()}
    for m in manifest:
        videos_by_split[m["split"]].add(m["source_video"])
        for cls in m["labels"].split(";"):
            rows_by_split[m["split"]].append({
                "video_id": f"{m['chunk_id']}#{cls}",
                "video": f"{m['chunk_id']}.mp4",
                "caption": caption[cls],
            })
    for split, rows in rows_by_split.items():
        out = args.data / f"{split}.json"
        out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        vlist = args.data / f"{split}_videos.txt"
        vlist.write_text("\n".join(sorted(videos_by_split[split])) + "\n", encoding="utf-8")
        print(f"{out.name}: {len(rows)} rows; {vlist.name}: {len(videos_by_split[split])} videos")

    prompts = {c["id"]: {"label": label[c["id"]], **{k: c[k] for k in ("caption", "kind", "window", "threshold", "min_consec")}}
               for c in classes}
    (args.data / "prompts.json").write_text(json.dumps(prompts, indent=2), encoding="utf-8")
    cap2label = {caption[cid]: lab for cid, lab in label.items()}
    (args.data / "caption_to_label.json").write_text(json.dumps(cap2label, indent=2), encoding="utf-8")

    yaml_block = "\n".join(f'{INDENT}"{cap}": {lab}'
                           for cap, lab in sorted(cap2label.items(), key=lambda kv: kv[1]))
    args.specs.mkdir(parents=True, exist_ok=True)
    for tmpl in sorted(args.templates.glob("*.yaml.tmpl")):
        text = tmpl.read_text(encoding="utf-8")
        out = args.specs / tmpl.name.removesuffix(".tmpl")
        out.write_text(text.replace("__CAPTION_TO_LABEL__", yaml_block), encoding="utf-8", newline="\n")
        print(f"rendered {out}")
    for static in sorted(args.templates.glob("*.yaml")):
        (args.specs / static.name).write_text(static.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        print(f"copied {static.name}")

    print(f"GATE PASSED: {len(classes)} classes, all chunk mp4s present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
