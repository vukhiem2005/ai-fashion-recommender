r"""Select new, stratified ELAN references for frozen V1/V2/V3/V4 comparison.

Windows PowerShell, from repository root:
    .\backend\.venv\Scripts\python.exe .\backend\evaluation\select_holdout_v4.py

Reads local SQLite and two completed development-label CSVs. Excludes every
product previously seen as a labeled reference OR recommended candidate from
new reference selection. This does NOT exclude those products from the catalog
when models later generate recommendations.

Output: backend/evaluation/holdout_v4_references.csv (new file only).
No SQLite, dataset, model, API, or existing evaluation file is modified.
"""
from __future__ import annotations

import csv
import hashlib
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
EVAL = BACKEND / "evaluation"
DATABASE = BACKEND / "data" / "elan.db"
OUTPUT = EVAL / "holdout_v4_references.csv"
V4_SOURCE = BACKEND / "app" / "recommender_v4.py"
DEVELOPMENT_LABELS = (
    EVAL / "v1_v2_candidate_union.csv",
    EVAL / "holdout_v1_v2_candidate_union.csv",
)
PREVIOUS_REFERENCES = EVAL / "holdout_references.csv"
CATEGORIES = ("Accessories", "Bottoms", "Dresses", "Knitwear", "Outerwear", "Tops")
REFS_PER_CATEGORY = 3
SEED = 2026110804
FIELDS = ("product_id", "name", "category", "slug")


def read_csv(path: Path, required_fields: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required development evidence missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            raise ValueError(f"Missing required columns in: {path}")
        return list(reader)


def main() -> int:
    try:
        if OUTPUT.exists():
            raise FileExistsError(f"Holdout file already exists; will not overwrite: {OUTPUT}")
        if not V4_SOURCE.is_file():
            raise FileNotFoundError(f"V4 model source missing: {V4_SOURCE}")
        previously_reviewed: set[str] = set()
        label_counts: list[int] = []
        for path in DEVELOPMENT_LABELS:
            rows = read_csv(path, {"reference_id", "recommended_id", "relevance_0_1_2"})
            invalid = [r for r in rows if r["relevance_0_1_2"].strip() not in {"0", "1", "2"}]
            if invalid:
                raise ValueError(f"Development labels incomplete in {path.name}: {len(invalid)}")
            for row in rows:
                previously_reviewed.add(row["reference_id"].strip())
                previously_reviewed.add(row["recommended_id"].strip())
            label_counts.append(len(rows))

        old_references = read_csv(PREVIOUS_REFERENCES, {"product_id", "category"})
        if len(old_references) != 12:
            raise ValueError(f"Expected 12 earlier holdout references; found {len(old_references)}")
        previously_reviewed.update(r["product_id"].strip() for r in old_references)
        if not DATABASE.is_file():
            raise FileNotFoundError(f"SQLite database missing: {DATABASE}")
        with sqlite3.connect(DATABASE) as conn:
            products = conn.execute(
                "SELECT id, name, category, slug FROM product "
                "WHERE is_active = 1 AND id LIKE 'HM-%' ORDER BY id"
            ).fetchall()
        if len(products) != 320:
            raise ValueError(f"Expected unchanged 320-product H&M catalog, got {len(products)}")

        rng = random.Random(SEED)
        selected: list[dict[str, str]] = []
        remaining_by_category: dict[str, int] = {}
        for category in CATEGORIES:
            candidates = [p for p in products if p[2] == category and p[0] not in previously_reviewed]
            remaining_by_category[category] = len(candidates)
            if len(candidates) < REFS_PER_CATEGORY:
                raise ValueError(
                    f"Too few unseen reference candidates for {category}: "
                    f"{len(candidates)} < {REFS_PER_CATEGORY}"
                )
            for product_id, name, cat, slug in rng.sample(candidates, REFS_PER_CATEGORY):
                selected.append({
                    "product_id": str(product_id),
                    "name": str(name),
                    "category": str(cat),
                    "slug": str(slug),
                })
        ids = [r["product_id"] for r in selected]
        assert len(ids) == 18 and len(ids) == len(set(ids))
        assert all(pid not in previously_reviewed for pid in ids)
        assert Counter(row["category"] for row in selected) == Counter({c: 3 for c in CATEGORIES})

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT.open("x", newline="", encoding="utf-8-sig") as target:
            writer = csv.DictWriter(target, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(selected)
        model_hash = hashlib.sha256(V4_SOURCE.read_bytes()).hexdigest()
        print("V4 NEW HOLDOUT SELECTION: COMPLETE")
        print("Active H&M catalog:", len(products))
        print("Previous labeled pairs checked:", "+".join(map(str, label_counts)))
        print("Previously reviewed products excluded as references:", len(previously_reviewed))
        print("New reference products:", len(selected))
        print("Reference products per category:", REFS_PER_CATEGORY)
        print("Selection seed:", SEED)
        print("V4 model SHA256:", model_hash)
        print("Eligible unseen references by category:", remaining_by_category)
        for row in selected:
            print(f"{row['category']} | {row['product_id']} | {row['name']}")
        print("Output:", OUTPUT)
        print("Existing labels, SQLite, models, and production API were NOT modified.")
        return 0
    except (FileNotFoundError, FileExistsError, OSError, ValueError, AssertionError, sqlite3.Error) as exc:
        print("ERROR:", exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
