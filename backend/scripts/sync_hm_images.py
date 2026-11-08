r"""Link downloaded H&M JPGs to the local ELAN storefront without altering original products.

Use from backend directory:
    .\.venv\Scripts\python.exe -m scripts.sync_hm_images --dry-run
    .\.venv\Scripts\python.exe -m scripts.sync_hm_images --apply

Source images stay in data/private/hm/images/. In apply mode, matching files are
copied to frontend/public/products/hm-<article_id>.jpg, and ProductVariant.image_url
is set in SQLite for H&M variants with no previous image. Missing images remain null.
The 8 existing ELAN products and their variants are not modified.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import re
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "backend" / "data" / "elan.db"
MANIFEST_PATH = ROOT / "data" / "private" / "hm" / "image_manifest.csv"
SOURCE_DIR = ROOT / "data" / "private" / "hm" / "images"
PUBLIC_DIR = ROOT / "frontend" / "public" / "products"
ARTICLE_ID_RE = re.compile(r"\d{10}\Z")


@dataclass(frozen=True)
class ImageEntry:
    product_id: str
    article_id: str
    source: Path
    destination: Path
    image_url: str


def load_manifest() -> list[ImageEntry]:
    if not MANIFEST_PATH.is_file():
        raise RuntimeError(f"Image manifest not found: {MANIFEST_PATH}")

    items: list[ImageEntry] = []
    seen: set[str] = set()
    with MANIFEST_PATH.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"product_id", "article_id", "image_path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"Image manifest must contain: {', '.join(sorted(required))}")

        for row in reader:
            article_id = (row.get("article_id") or "").strip()
            product_id = (row.get("product_id") or "").strip()
            image_path = (row.get("image_path") or "").strip()
            if not ARTICLE_ID_RE.fullmatch(article_id):
                raise RuntimeError(f"Invalid article_id in manifest: {article_id!r}")
            if product_id != f"HM-{article_id}":
                raise RuntimeError(f"Product ID mismatch for article {article_id}")
            expected_path = f"images/{article_id[:3]}/{article_id}.jpg"
            if image_path != expected_path:
                raise RuntimeError(f"Unexpected image path for {article_id}: {image_path}")
            if article_id in seen:
                raise RuntimeError(f"Duplicate article_id in manifest: {article_id}")
            seen.add(article_id)
            filename = f"hm-{article_id}.jpg"
            items.append(ImageEntry(
                product_id=product_id,
                article_id=article_id,
                source=SOURCE_DIR / article_id[:3] / f"{article_id}.jpg",
                destination=PUBLIC_DIR / filename,
                image_url=f"/products/{filename}",
            ))
    return items


def is_plausible_jpeg(file_path: Path) -> bool:
    """Cheap safeguard against missing, empty, or non-JPEG files; not full decoding."""
    if not file_path.is_file() or file_path.stat().st_size < 100:
        return False
    with file_path.open("rb") as stream:
        return stream.read(3) == b"\xff\xd8\xff"


def validate_database(connection: sqlite3.Connection, entries: list[ImageEntry], expected: int) -> None:
    hm_ids = {
        row[0] for row in connection.execute("SELECT id FROM product WHERE id LIKE 'HM-%'")
    }
    elan_count = connection.execute(
        "SELECT COUNT(*) FROM product WHERE id LIKE 'ELAN-%'"
    ).fetchone()[0]
    manifest_ids = {entry.product_id for entry in entries}
    if len(entries) != expected or hm_ids != manifest_ids or elan_count != 8:
        raise RuntimeError(
            "Catalog/manifest mismatch: "
            f"manifest={len(entries)}, database HM={len(hm_ids)}, "
            f"original ELAN={elan_count}. Expected {expected} HM + 8 ELAN. "
            "No changes made."
        )


def backup_database(connection: sqlite3.Connection) -> Path:
    backup_folder = DB_PATH.parent / "backups"
    backup_folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = backup_folder / f"elan_before_image_sync_{timestamp}.db"
    with sqlite3.connect(backup_path) as destination:
        connection.backup(destination)
    return backup_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Link H&M images to product variants in local ELAN.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Analyze without writing or copying files")
    mode.add_argument("--apply", action="store_true", help="Copy images and update SQLite")
    parser.add_argument("--expected-hm", type=int, default=320, help="Expected H&M catalog size (default 320)")
    args = parser.parse_args()

    if not DB_PATH.is_file():
        parser.error(f"Database not found: {DB_PATH}")
    if args.expected_hm < 1:
        parser.error("--expected-hm must be a positive integer")

    entries = load_manifest()
    existing_images: list[ImageEntry] = []
    missing_images: list[str] = []
    for entry in entries:
        if is_plausible_jpeg(entry.source):
            existing_images.append(entry)
        else:
            missing_images.append(entry.article_id)

    planned_copies: list[ImageEntry] = []
    for entry in existing_images:
        if entry.destination.exists():
            if not entry.destination.is_file() or not filecmp.cmp(
                entry.source, entry.destination, shallow=False
            ):
                raise RuntimeError(
                    "A different file already exists at "
                    f"{entry.destination}. No changes made."
                )
        else:
            planned_copies.append(entry)

    print("Mode:", "APPLY" if args.apply else "DRY RUN")
    print("SQLite:", DB_PATH)
    print("Manifest entries:", len(entries))
    print("Valid-looking local JPEG files:", len(existing_images))
    print("Missing/unusable JPEG files:", len(missing_images))
    print("Files to copy into frontend/public/products:", len(planned_copies))
    if missing_images:
        print("Missing article IDs:", ", ".join(missing_images[:15]))

    connection = sqlite3.connect(DB_PATH)
    copied: list[Path] = []
    try:
        validate_database(connection, entries, args.expected_hm)
        link_map = {entry.product_id: entry for entry in existing_images}

        variants = connection.execute(
            "SELECT sku, product_id, image_url FROM productvariant "
            "WHERE product_id LIKE 'HM-%'"
        ).fetchall()
        variants_by_product: dict[str, int] = {}
        for _, product_id, _ in variants:
            variants_by_product[product_id] = variants_by_product.get(product_id, 0) + 1
        missing_variants = [entry.product_id for entry in entries if not variants_by_product.get(entry.product_id)]
        if missing_variants:
            raise RuntimeError(f"H&M products with no variants: {missing_variants[:5]}. No changes made.")

        updates: list[tuple[str, str]] = []
        unchanged_links = 0
        custom_links = 0
        for sku, product_id, current_url in variants:
            entry = link_map.get(product_id)
            if entry is None:
                continue
            if current_url is None:
                updates.append((entry.image_url, sku))
            elif current_url == entry.image_url:
                unchanged_links += 1
            else:
                custom_links += 1  # Preserve image paths set through the Admin UI.

        print("Variants to link:", len(updates))
        print("Variants already linked:", unchanged_links)
        print("Variants with custom images (preserved):", custom_links)

        if args.dry_run:
            print("DRY RUN complete. SQLite and frontend/public were NOT modified.")
            return

        backup = backup_database(connection)
        print("SQLite backup:", backup)
        PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
        for entry in planned_copies:
            # Skip source content with invalid JPEG header; filtered during preflight.
            shutil.copy2(entry.source, entry.destination)
            copied.append(entry.destination)

        connection.execute("BEGIN IMMEDIATE")
        for image_url, sku in updates:
            result = connection.execute(
                "UPDATE productvariant SET image_url = ? WHERE sku = ? AND image_url IS NULL",
                (image_url, sku),
            )
            if result.rowcount != 1:
                raise RuntimeError(f"Concurrent change detected for SKU {sku}; rolling back")
        connection.commit()

        print("Copied files:", len(copied))
        print("Updated SKU image paths:", len(updates))
        print("Original ELAN products: unchanged")
        print("APPLY complete.")
    except Exception:
        connection.rollback()
        for item in copied:
            item.unlink(missing_ok=True)
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, sqlite3.Error, RuntimeError) as error:
        raise SystemExit(f"ERROR: {error}") from None
