"""Emit MSR-VTT-style metadata JSONs and render the evaluate specs (stdlib only).

From split_manifest.csv produces, in workspace/data:
  train.json / val.json / test.json   [{"video_id","video","caption"}, ...]
  caption_to_label.json               {caption: int label} (51 entries)
and renders scripts/spec_templates/evaluate_*.yaml.tmpl -> workspace/specs/
replacing the __CAPTION_TO_LABEL__ line with the 51 YAML mapping entries.

Hard gate: every metadata `video` filename must exist in workspace/data/video/
(the loader matches metadata rows against the mp4 glob by filename — a
mismatch silently yields "0 videos found").
"""

import argparse
import csv
import json
import sys
from pathlib import Path

INDENT = " " * 6  # caption_to_label entries sit under dataset.test_dataset (4) + 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="workspace/data")
    ap.add_argument("--templates", type=Path, required=True, help="scripts/spec_templates")
    ap.add_argument("--specs", type=Path, required=True, help="workspace/specs")
    args = ap.parse_args()

    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    video_dir = args.data / "video"
    missing = [r["video_id"] for r in rows if not (video_dir / f"{r['video_id']}.mp4").exists()]
    if missing:
        print(f"[FAIL] {len(missing)} manifest rows have no mp4 on disk, first: {missing[:5]}",
              file=sys.stderr)
        return 1

    for split in ("train", "val", "test"):
        items = [{"video_id": r["video_id"], "video": f"{r['video_id']}.mp4", "caption": r["caption"]}
                 for r in rows if r["split"] == split]
        out = args.data / f"{split}.json"
        out.write_text(json.dumps(items, indent=1), encoding="utf-8")
        print(f"{out.name}: {len(items)} rows")

    cap2label = {}
    for r in rows:
        label = int(r["label"])
        if cap2label.setdefault(r["caption"], label) != label:
            print(f"[FAIL] caption maps to two labels: {r['caption']!r}", file=sys.stderr)
            return 1
    (args.data / "caption_to_label.json").write_text(json.dumps(cap2label, indent=2), encoding="utf-8")

    yaml_block = "\n".join(
        f'{INDENT}"{cap}": {label}' for cap, label in sorted(cap2label.items(), key=lambda kv: kv[1])
    )
    args.specs.mkdir(parents=True, exist_ok=True)
    for tmpl in sorted(args.templates.glob("*.yaml.tmpl")):
        text = tmpl.read_text(encoding="utf-8")
        if "__CAPTION_TO_LABEL__" not in text:
            print(f"[FAIL] {tmpl.name} lacks __CAPTION_TO_LABEL__ placeholder", file=sys.stderr)
            return 1
        rendered = text.replace("__CAPTION_TO_LABEL__", yaml_block)
        out = args.specs / tmpl.name.removesuffix(".tmpl")
        out.write_text(rendered, encoding="utf-8", newline="\n")
        print(f"rendered {out}")

    print(f"GATE PASSED: {len(cap2label)} classes, all metadata videos present on disk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
