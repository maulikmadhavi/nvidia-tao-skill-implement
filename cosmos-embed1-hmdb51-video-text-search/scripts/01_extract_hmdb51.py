"""Extract the HMDB51 archive (stdlib zipfile + 7z.exe for any inner rars).

Layout is auto-detected: hmdb51.zip is extracted first (skipped if already
done); any inner per-class .rar files are then extracted with 7-Zip; finally
the class directories (dirs containing *.avi) are located anywhere under the
extraction root and reported.

Writes <raw>/class_dirs.json mapping class name -> absolute dir path for
downstream scripts, so they don't need to re-discover the layout.

Gate: 51 classes, ~6766 total .avi clips.
"""

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

EXPECTED_CLASSES = 51
MIN_AVI = 6000


def run_7z(sevenzip: str, archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([sevenzip, "x", "-y", f"-o{out_dir}", str(archive)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"7z failed on {archive.name}:\n{proc.stdout}\n{proc.stderr}")


def find_class_dirs(root: Path) -> dict[str, Path]:
    """Every directory that directly contains .avi files, keyed by dir name."""
    class_dirs: dict[str, Path] = {}
    for d in root.rglob("*"):
        if d.is_dir() and any(f.suffix.lower() == ".avi" for f in d.iterdir() if f.is_file()):
            if d.name in class_dirs:
                raise RuntimeError(f"duplicate class dir name {d.name}: {class_dirs[d.name]} vs {d}")
            class_dirs[d.name] = d
    return class_dirs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True, help="dir containing hmdb51.zip")
    ap.add_argument("--sevenzip", default=r"C:\Program Files\7-Zip\7z.exe")
    args = ap.parse_args()

    outer = args.raw / "hmdb51.zip"
    extract_root = args.raw / "hmdb51_extracted"

    if not extract_root.exists() or not any(extract_root.iterdir()):
        if not outer.exists():
            print(f"missing {outer}", file=sys.stderr)
            return 1
        print(f"[stage 1] unzipping {outer.name} -> {extract_root} (this takes a few minutes)")
        with zipfile.ZipFile(outer) as zf:
            zf.extractall(extract_root)
    else:
        print(f"[stage 1] {extract_root} already populated, skipping unzip")

    inner_rars = sorted(extract_root.rglob("*.rar"))
    if inner_rars:
        print(f"[stage 2] extracting {len(inner_rars)} inner rars")
        for i, rar in enumerate(inner_rars, 1):
            marker_dir = extract_root / "_from_rars"
            done = (marker_dir / rar.stem).is_dir() and any((marker_dir / rar.stem).rglob("*.avi"))
            if not done:
                run_7z(args.sevenzip, rar, marker_dir)
            print(f"  [{i:2}/{len(inner_rars)}] {rar.stem}", flush=True)
    else:
        print("[stage 2] no inner rars — zip contained videos directly")

    class_dirs = find_class_dirs(extract_root)
    avi_total = sum(len(list(d.glob("*.avi"))) for d in class_dirs.values())
    print(f"classes: {len(class_dirs)}  avi clips: {avi_total}")
    for name in sorted(class_dirs):
        print(f"  {name}: {len(list(class_dirs[name].glob('*.avi')))}")

    if len(class_dirs) != EXPECTED_CLASSES or avi_total < MIN_AVI:
        print("[FAIL] extraction gate not met "
              f"(want {EXPECTED_CLASSES} classes, >={MIN_AVI} avi)", file=sys.stderr)
        return 1

    index = {name: str(path) for name, path in sorted(class_dirs.items())}
    (args.raw / "class_dirs.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"GATE PASSED: wrote {args.raw / 'class_dirs.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
