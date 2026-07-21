"""Download the HMDB51 archive (stdlib only).

Primary source: the HuggingFace mirror jili5044/hmdb51 (hmdb51.zip, ~2.1 GB,
6,766 .avi clips in 51 classes — the original Serre Lab content re-packaged;
the original serre-lab.clps.brown.edu URLs are dead as of 2026: the site
redesign serves its SPA homepage for every old wp-content path).
Resume-on-partial support and size verification. Skips complete files.
"""

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

FILES = {
    "hmdb51.zip": "https://huggingface.co/datasets/jili5044/hmdb51/resolve/main/hmdb51.zip",
}

MIN_PLAUSIBLE_SIZE = 1 << 30  # anything under 1 GiB is an error page, not the dataset

CHUNK = 1 << 20  # 1 MiB


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", 0))


def download(url: str, dest: Path) -> None:
    total = remote_size(url)
    have = dest.stat().st_size if dest.exists() else 0
    if total and have == total:
        print(f"[skip] {dest.name} already complete ({have:,} bytes)")
        return
    if have > total > 0:
        print(f"[warn] {dest.name} larger than remote ({have:,} > {total:,}); restarting")
        dest.unlink()
        have = 0

    headers = {}
    mode = "wb"
    if have:
        headers["Range"] = f"bytes={have}-"
        mode = "ab"
        print(f"[resume] {dest.name} from {have:,}/{total:,} bytes")
    else:
        print(f"[start] {dest.name} ({total:,} bytes)" if total else f"[start] {dest.name}")

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, mode) as f:
            done = have
            while True:
                chunk = r.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total and done % (100 * CHUNK) < CHUNK:
                    print(f"  {dest.name}: {done / total:6.1%} ({done:,}/{total:,})", flush=True)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # range not satisfiable -> already complete
            print(f"[skip] {dest.name} range complete")
            return
        raise

    final = dest.stat().st_size
    if total and final != total:
        raise RuntimeError(f"{dest.name}: size mismatch after download ({final:,} != {total:,}); rerun to resume")
    if final < MIN_PLAUSIBLE_SIZE:
        raise RuntimeError(f"{dest.name}: only {final:,} bytes — server returned an error page, not the dataset")
    print(f"[done] {dest.name} ({final:,} bytes)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="output dir for the .rar files")
    ap.add_argument("--url-override", nargs=2, action="append", metavar=("FILENAME", "URL"),
                    help="override the URL for a given target filename (mirror fallback)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    urls = dict(FILES)
    for name, url in (args.url_override or []):
        urls[name] = url

    failures = []
    for name, url in urls.items():
        try:
            download(url, args.out / name)
        except Exception as e:  # noqa: BLE001 - report and continue to next file
            print(f"[FAIL] {name}: {e}", file=sys.stderr)
            failures.append(name)
    if failures:
        print(f"FAILED: {', '.join(failures)} -- rerun to resume, or pass --url-override", file=sys.stderr)
        return 1
    print("All downloads complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
