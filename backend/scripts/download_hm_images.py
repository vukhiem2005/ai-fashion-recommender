"""Download selected H&M article images via the authenticated Kaggle CLI.

Run from the backend directory with its virtualenv Python:
    python -m scripts.download_hm_images --limit 20

Files are stored only under data/private/hm/images/. The SQLite database and
frontend/public/ directory are never modified by this script.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

COMPETITION = "h-and-m-personalized-fashion-recommendations"
REPO_ROOT = Path(__file__).resolve().parents[2]
HM_ROOT = REPO_ROOT / "data" / "private" / "hm"
MANIFEST = HM_ROOT / "image_manifest.csv"
IMAGE_PATH_PATTERN = re.compile(r"^images/(\d{3})/(\d{10})\.jpg$")


def read_manifest(limit: int) -> list[tuple[str, Path]]:
    if not MANIFEST.is_file():
        raise SystemExit(f"Manifest not found: {MANIFEST}")

    entries: list[tuple[str, Path]] = []
    seen: set[str] = set()

    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {"article_id", "image_path"}.issubset(reader.fieldnames):
            raise SystemExit("Manifest must have article_id and image_path columns.")

        for row in reader:
            article_id = (row.get("article_id") or "").strip()
            image_path = (row.get("image_path") or "").strip()
            match = IMAGE_PATH_PATTERN.fullmatch(image_path)

            if not match or article_id != match.group(2) or article_id[:3] != match.group(1):
                raise SystemExit(f"Unexpected manifest image path for article {article_id}: {image_path}")

            if image_path in seen:
                continue
            seen.add(image_path)
            entries.append((article_id, Path(*image_path.split("/"))))
            if len(entries) >= limit:
                break

    return entries


def kaggle_executable() -> str:
    executable = Path(sys.executable).with_name("kaggle.exe" if sys.platform == "win32" else "kaggle")
    if executable.is_file():
        return str(executable)
    found = shutil.which("kaggle")
    if found:
        return found
    raise SystemExit("Kaggle CLI not found in the active Python environment.")


def is_valid_jpeg(file_path: Path) -> bool:
    if not file_path.is_file() or file_path.stat().st_size < 4:
        return False
    with file_path.open("rb") as handle:
        return handle.read(3) == b"\xff\xd8\xff"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download images listed in the H&M manifest.")
    parser.add_argument("--limit", type=int, default=20, help="First N manifest items to process (default: 20).")
    parser.add_argument("--pause", type=float, default=1.0, help="Seconds between download attempts (default: 1).")
    parser.add_argument("--timeout", type=int, default=180, help="Timeout per image in seconds (default: 180).")
    args = parser.parse_args()

    if args.limit <= 0 or args.pause < 0 or args.timeout < 1:
        parser.error("--limit and --timeout must be positive; --pause cannot be negative")

    entries = read_manifest(args.limit)
    kaggle = kaggle_executable()
    counts = {"already_present": 0, "copied_from_sample": 0, "downloaded": 0, "missing_or_failed": 0}
    failures: list[str] = []

    print(f"Manifest: {MANIFEST}")
    print(f"Selected: {len(entries)} images; destination: {HM_ROOT / 'images'}")
    print("This script does not modify SQLite or the public website directory.\n")

    for index, (article_id, relative_path) in enumerate(entries, 1):
        destination = HM_ROOT / relative_path
        label = f"[{index}/{len(entries)}] {article_id}"
        if is_valid_jpeg(destination):
            counts["already_present"] += 1
            print(f"{label}: EXISTS")
            continue

        # Reuse a JPG previously fetched during the single-image test.
        sample = HM_ROOT / "image_sample" / relative_path
        if is_valid_jpeg(sample):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sample, destination)
            counts["copied_from_sample"] += 1
            print(f"{label}: COPIED FROM SAMPLE")
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            kaggle,
            "competitions",
            "download",
            COMPETITION,
            "-f",
            relative_path.as_posix(),
            "-p",
            str(HM_ROOT),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=args.timeout,
                check=False,
            )
            output = (completed.stdout + "\n" + completed.stderr).strip()
            if completed.returncode == 0 and is_valid_jpeg(destination):
                counts["downloaded"] += 1
                print(f"{label}: DOWNLOADED ({destination.stat().st_size:,} bytes)")
            else:
                counts["missing_or_failed"] += 1
                reason = output.splitlines()[-1][:200] if output else f"Exit code {completed.returncode}"
                failures.append(f"{article_id}: {reason}")
                print(f"{label}: FAILED ({reason})")
                if "429" in output or "too many requests" in output.lower() or "rate limit" in output.lower():
                    print("Stopping after possible rate limit. Rerun later; existing files will be skipped.")
                    break
        except subprocess.TimeoutExpired:
            counts["missing_or_failed"] += 1
            failures.append(f"{article_id}: timed out after {args.timeout} seconds")
            print(f"{label}: TIMEOUT")

        if index < len(entries) and args.pause:
            time.sleep(args.pause)

    print("\nSUMMARY")
    for label, count in counts.items():
        print(f"{label}: {count}")
    if failures:
        print("Failed items (up to first 10):")
        for failure in failures[:10]:
            print(f"  {failure}")
    print("SQLite was NOT modified.")


if __name__ == "__main__":
    main()
