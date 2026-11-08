"""Import a reproducible sample of H&M article metadata into the local ELAN SQLite DB.

Default: preview only (no database writes).
Run from backend/ as: python -m scripts.import_hm_articles --dry-run
After reviewing and backing up: python -m scripts.import_hm_articles --apply
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import sqlite3

import pandas as pd
from sqlmodel import Session, select

from app.database import DATABASE_FILE, engine
from app.models import Product, ProductVariant

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "private" / "hm" / "articles.csv"

# Planned distribution for an initial 1,000-product demo catalog.
CATEGORY_TARGETS = {
    "Tops": 210,
    "Outerwear": 160,
    "Bottoms": 170,
    "Dresses": 170,
    "Knitwear": 150,
    "Accessories": 140,
}

# Illustrative prices, not actual H&M retail prices.
DEMO_PRICE_CENTS = {
    "Tops": 1999,
    "Outerwear": 5999,
    "Bottoms": 2999,
    "Dresses": 3999,
    "Knitwear": 3499,
    "Accessories": 1499,
}

REQUIRED_COLUMNS = {
    "article_id",
    "prod_name",
    "product_group_name",
    "product_type_name",
    "garment_group_name",
    "colour_group_name",
    "detail_desc",
}


def classify(row: pd.Series) -> str | None:
    group = str(row["product_group_name"]).strip().lower()
    kind = str(row["product_type_name"]).strip().lower()
    garment = str(row["garment_group_name"]).strip().lower()

    if group in {"accessories", "bags"}:
        return "Accessories"
    if "dress" in kind and group == "garment full body":
        return "Dresses"
    if group == "garment upper body" and (
        "knitwear" in garment
        or any(word in kind for word in ("sweater", "cardigan", "jumper"))
    ):
        return "Knitwear"
    if group == "garment upper body" and (
        "outdoor" in garment
        or any(
            word in kind
            for word in ("jacket", "coat", "blazer", "parka", "waistcoat", "vest")
        )
    ):
        return "Outerwear"
    if group == "garment lower body":
        return "Bottoms"
    if group == "garment upper body":
        return "Tops"
    return None


def choose_articles(limit: int) -> pd.DataFrame:
    if not CSV_PATH.is_file():
        raise SystemExit(f"Cannot find H&M dataset: {CSV_PATH}")

    articles = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS.difference(articles.columns)
    if missing:
        raise SystemExit(f"Missing required CSV columns: {sorted(missing)}")

    # Preserve leading zeros: article IDs are strings, never integers.
    for column in REQUIRED_COLUMNS:
        articles[column] = articles[column].str.strip()

    articles = articles[
        articles["article_id"].str.fullmatch(r"[0-9]{10}")
        & articles["prod_name"].ne("")
        & articles["colour_group_name"].ne("")
    ].drop_duplicates(subset="article_id").copy()

    articles["elan_category"] = articles.apply(classify, axis=1)
    articles = articles[articles["elan_category"].notna()].copy()

    if len(articles) < limit:
        raise SystemExit(
            f"Only {len(articles)} eligible articles; requested {limit}."
        )

    chosen_parts: list[pd.DataFrame] = []
    used_ids: set[str] = set()

    for category, target in CATEGORY_TARGETS.items():
        desired = (limit * target) // 1000
        pool = articles[articles["elan_category"] == category]
        if desired == 0 or pool.empty:
            continue
        sampled = pool.sample(n=min(desired, len(pool)), random_state=42)
        chosen_parts.append(sampled)
        used_ids.update(sampled["article_id"].tolist())

    chosen = pd.concat(chosen_parts, ignore_index=True) if chosen_parts else articles.iloc[:0]
    remaining_count = limit - len(chosen)
    if remaining_count > 0:
        remaining = articles[~articles["article_id"].isin(used_ids)]
        extra = remaining.sample(n=remaining_count, random_state=42)
        chosen = pd.concat([chosen, extra], ignore_index=True)

    return chosen.sort_values("article_id").reset_index(drop=True)


def make_description(row: pd.Series) -> str:
    original = str(row["detail_desc"]).strip()
    metadata = (
        f"Product type: {row['product_type_name']}; "
        f"garment group: {row['garment_group_name']}; "
        f"colour: {row['colour_group_name']}."
    )
    return f"{original} {metadata}".strip()


def back_up_database() -> Path:
    # SQLite's backup API is safer than directly copying a database in use.
    backup_dir = DATABASE_FILE.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = backup_dir / f"elan-before-hm-{stamp}.db"
    with sqlite3.connect(DATABASE_FILE) as source:
        with sqlite3.connect(backup_path) as destination:
            source.backup(destination)
    return backup_path


def import_articles(articles: pd.DataFrame) -> None:
    if not DATABASE_FILE.is_file():
        raise SystemExit(
            f"The existing ELAN database was not found: {DATABASE_FILE}. "
            "Import cancelled to prevent creating a different database."
        )

    backup_path = back_up_database()
    print(f"SQLite backup created: {backup_path}")

    created = 0
    skipped = 0
    with Session(engine) as session:
        existing_ids = set(session.exec(select(Product.id)).all())
        existing_slugs = set(session.exec(select(Product.slug)).all())

        for row in articles.to_dict(orient="records"):
            article_id = row["article_id"]
            product_id = f"HM-{article_id}"
            slug = f"hm-{article_id}"

            # Never modify an existing product, including the original ELAN items.
            if product_id in existing_ids or slug in existing_slugs:
                skipped += 1
                continue

            category = row["elan_category"]
            session.add(
                Product(
                    id=product_id,
                    slug=slug,
                    name=row["prod_name"],
                    category=category,
                    description=make_description(row),
                    price_cents=DEMO_PRICE_CENTS[category],
                    is_active=True,
                )
            )
            session.add(
                ProductVariant(
                    sku=f"{product_id}-BASE",
                    product_id=product_id,
                    color=row["colour_group_name"],
                    size="Not specified",  # No size/stock data in articles.csv.
                    image_url=None,  # Images will be linked in a later phase.
                )
            )
            existing_ids.add(product_id)
            existing_slugs.add(slug)
            created += 1

        session.commit()

    print(f"Import completed: {created} inserted; {skipped} already existed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="H&M -> ELAN catalog import")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", help="Preview only (default)")
    parser.add_argument("--apply", action="store_true", help="Write into SQLite")
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.apply and args.dry_run:
        parser.error("Choose either --dry-run or --apply")

    selected = choose_articles(args.limit)
    counts = Counter(selected["elan_category"])

    print(f"Source: {CSV_PATH}")
    print(f"Selected: {len(selected)} unique H&M articles")
    for category in CATEGORY_TARGETS:
        print(f"  {category}: {counts.get(category, 0)}")
    print("Sample (ID | name | category | colour):")
    for row in selected.head(5).to_dict(orient="records"):
        print(
            f"  HM-{row['article_id']} | {row['prod_name']} | "
            f"{row['elan_category']} | {row['colour_group_name']}"
        )

    if not args.apply:
        print("DRY RUN complete. SQLite was NOT modified.")
        return

    import_articles(selected)


if __name__ == "__main__":
    main()
