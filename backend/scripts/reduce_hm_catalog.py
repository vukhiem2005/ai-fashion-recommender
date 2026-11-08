r"""Safely reduce the local H&M catalog to 320 products (plus existing ÉLAN products).

Usage from the repository's backend/ directory:
    .\.venv\Scripts\python.exe -m scripts.reduce_hm_catalog --dry-run
    .\.venv\Scripts\python.exe -m scripts.reduce_hm_catalog --apply

The selection favors already-downloaded images, then balances product colors.
It never deletes image files or modifies non-H&M products.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = PROJECT_ROOT / "backend" / "data" / "elan.db"
DEFAULT_HM_DIR = PROJECT_ROOT / "data" / "private" / "hm"

# The six category quotas total 320 H&M products.
QUOTAS = {
    "Tops": 70,
    "Bottoms": 55,
    "Dresses": 55,
    "Outerwear": 50,
    "Knitwear": 50,
    "Accessories": 40,
}
EXPECTED_ORIGINAL_PRODUCTS = 8


@dataclass(frozen=True)
class Article:
    id: str
    name: str
    category: str
    color: str
    has_downloaded_image: bool

    @property
    def article_id(self) -> str:
        return self.id.removeprefix("HM-")

    @property
    def image_path(self) -> str:
        code = self.article_id
        return f"images/{code[:3]}/{code}.jpg"


def read_catalog(connection: sqlite3.Connection, hm_dir: Path) -> list[Article]:
    rows = connection.execute(
        "SELECT id, name, category FROM product "
        "WHERE id LIKE 'HM-%' ORDER BY id"
    ).fetchall()

    first_colors: dict[str, str] = {}
    for product_id, color in connection.execute(
        "SELECT product_id, color FROM productvariant "
        "WHERE product_id LIKE 'HM-%' ORDER BY product_id, sku"
    ):
        first_colors.setdefault(product_id, (color or "Unknown").strip() or "Unknown")

    results: list[Article] = []
    for product_id, name, category in rows:
        article_id = product_id.removeprefix("HM-")
        if len(article_id) != 10 or not article_id.isdigit():
            raise ValueError(f"Unexpected H&M product ID: {product_id}")

        image_file = hm_dir / "images" / article_id[:3] / f"{article_id}.jpg"
        has_image = image_file.is_file() and image_file.stat().st_size > 0

        results.append(
            Article(
                id=product_id,
                name=name,
                category=category,
                color=first_colors.get(product_id, "Unknown"),
                has_downloaded_image=has_image,
            )
        )
    return results


def distribute_by_color(items: list[Article]) -> list[Article]:
    """Deterministic round-robin through colors, with stable article ordering."""
    groups: dict[str, deque[Article]] = defaultdict(deque)
    for article in sorted(items, key=lambda p: p.id):
        groups[article.color.casefold()].append(article)

    # Larger groups first makes ties deterministic, then alternate colors.
    colors = sorted(groups, key=lambda color: (-len(groups[color]), color))
    result: list[Article] = []
    while True:
        added = False
        for color in colors:
            if groups[color]:
                result.append(groups[color].popleft())
                added = True
        if not added:
            return result


def select_articles(articles: list[Article]) -> list[Article]:
    by_category: dict[str, list[Article]] = defaultdict(list)
    for article in articles:
        by_category[article.category].append(article)

    selected: list[Article] = []
    for category, quota in QUOTAS.items():
        candidates = by_category.get(category, [])
        if len(candidates) < quota:
            raise ValueError(
                f"Not enough H&M products in {category}: "
                f"need {quota}, found {len(candidates)}"
            )

        downloaded = [p for p in candidates if p.has_downloaded_image]
        remaining = [p for p in candidates if not p.has_downloaded_image]
        prioritized = distribute_by_color(downloaded) + distribute_by_color(remaining)
        selected.extend(prioritized[:quota])

    if len(selected) != 320 or len({p.id for p in selected}) != 320:
        raise ValueError("Selection must contain 320 unique H&M products")
    return sorted(selected, key=lambda p: p.id)


def write_manifest(articles: list[Article], hm_dir: Path) -> Path:
    """Update image download plan after DB changes; never delete existing images."""
    hm_dir.mkdir(parents=True, exist_ok=True)
    destination = hm_dir / "image_manifest.csv"
    temporary = hm_dir / "image_manifest.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["product_id", "article_id", "name", "image_path"])
        for product in articles:
            writer.writerow(
                [product.id, product.article_id, product.name, product.image_path]
            )
    temporary.replace(destination)
    return destination


def create_backup(source_path: Path) -> Path:
    backup_dir = source_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination_path = backup_dir / f"elan_before_hm_reduction_{timestamp}.db"
    with sqlite3.connect(str(source_path), timeout=30) as source:
        with sqlite3.connect(str(destination_path), timeout=30) as destination:
            source.backup(destination)
    if not destination_path.is_file() or destination_path.stat().st_size == 0:
        raise RuntimeError("Database backup failed")
    return destination_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true", help="Print plan; change nothing")
    action.add_argument("--apply", action="store_true", help="Back up and reduce SQLite")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=argparse.SUPPRESS)
    parser.add_argument("--hm-dir", type=Path, default=DEFAULT_HM_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args()

    database = args.db.resolve()
    hm_dir = args.hm_dir.resolve()
    if not database.is_file():
        parser.error(f"SQLite database not found: {database}")

    with sqlite3.connect(str(database), timeout=30) as connection:
        existing_articles = read_catalog(connection, hm_dir)
        original_count = connection.execute(
            "SELECT COUNT(*) FROM product WHERE id NOT LIKE 'HM-%'"
        ).fetchone()[0]
        if original_count != EXPECTED_ORIGINAL_PRODUCTS:
            raise RuntimeError(
                f"Expected {EXPECTED_ORIGINAL_PRODUCTS} original ÉLAN products; "
                f"found {original_count}. Refusing to continue."
            )

        selection = select_articles(existing_articles)
        keep_ids = {product.id for product in selection}
        to_remove = [p.id for p in existing_articles if p.id not in keep_ids]
        kept_images = sum(product.has_downloaded_image for product in selection)
        all_downloaded = sum(product.has_downloaded_image for product in existing_articles)
        print("Mode:", "DRY RUN" if args.dry_run else "APPLY")
        print("SQLite:", database)
        print("H&M products before:", len(existing_articles))
        print("ÉLAN products preserved:", original_count)
        print("H&M products to keep:", len(selection))
        print("H&M products to remove:", len(to_remove))
        print("Total after reduction:", original_count + len(selection))
        print("Downloaded images retained:", kept_images, "/", all_downloaded)
        print("\nSelected H&M category counts:")
        counts = Counter(product.category for product in selection)
        for category, quota in QUOTAS.items():
            print(f"  {category}: {counts[category]} / {quota}")
        print("\nSample selected products:")
        for item in selection[:5]:
            label = " [image downloaded]" if item.has_downloaded_image else ""
            print(f"  {item.id} | {item.name} | {item.category}{label}")

        if args.dry_run:
            print("\nDRY RUN complete. SQLite and image_manifest.csv were NOT modified.")
            return 0

        backup = create_backup(database)
        print("\nBackup created:", backup)

        try:
            connection.execute("BEGIN IMMEDIATE")
            # Refuse deletion if the database changed between planning and write lock.
            locked_hm_ids = {
                row[0] for row in connection.execute(
                    "SELECT id FROM product WHERE id LIKE 'HM-%'"
                )
            }
            if locked_hm_ids != {p.id for p in existing_articles}:
                raise RuntimeError("H&M catalog changed during selection; retry")

            for product_id in to_remove:
                connection.execute(
                    "DELETE FROM productvariant WHERE product_id = ?", (product_id,)
                )
                connection.execute("DELETE FROM product WHERE id = ?", (product_id,))

            final_hm_count = connection.execute(
                "SELECT COUNT(*) FROM product WHERE id LIKE 'HM-%'"
            ).fetchone()[0]
            final_original_count = connection.execute(
                "SELECT COUNT(*) FROM product WHERE id NOT LIKE 'HM-%'"
            ).fetchone()[0]
            if final_hm_count != 320 or final_original_count != original_count:
                raise RuntimeError("Unexpected product counts; rolling back")
            connection.commit()
        except Exception:
            connection.rollback()
            print("Database transaction rolled back; backup remains available.")
            raise

    manifest = write_manifest(selection, hm_dir)
    print("Image manifest updated:", manifest)
    print("APPLY complete. Original ÉLAN products and downloaded images preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
