"""ÉLAN content-based product recommendation baseline.

Reads the existing SQLite product catalog without changing data. Builds
TF-IDF vectors from product attributes and ranks candidates with cosine
similarity. Intended for local experiments before adding an API endpoint.

Run from the backend directory:
    python -m app.recommender --product-id HM-0225286014 --top-k 5
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_DATABASE = Path(__file__).resolve().parent.parent / "data" / "elan.db"


def _load_catalog(database_path: Path) -> list[dict[str, Any]]:
    """Load only active products and their variants; never modify SQLite."""
    if not database_path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {database_path}")

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, slug, name, category, description, price_cents
            FROM product
            WHERE is_active = 1
            ORDER BY id
            """
        ).fetchall()
        variant_rows = connection.execute(
            """
            SELECT product_id, color, image_url
            FROM productvariant
            ORDER BY sku
            """
        ).fetchall()

    colors: dict[str, set[str]] = defaultdict(set)
    images: dict[str, str] = {}

    for variant in variant_rows:
        product_id = variant["product_id"]
        color = (variant["color"] or "").strip()
        image_url = variant["image_url"]
        if color:
            colors[product_id].add(color)
        if image_url and product_id not in images:
            images[product_id] = image_url

    return [
        {
            **dict(row),
            "colors": sorted(colors[row["id"]]),
            "image_url": images.get(row["id"]),
        }
        for row in rows
    ]


def _document(product: dict[str, Any]) -> str:
    """Combine attributes with simple, transparent feature weights."""
    category = str(product["category"] or "").strip()
    name = str(product["name"] or "").strip()
    description = str(product["description"] or "").strip()
    colors = " ".join(product["colors"])
    return " ".join(
        [category] * 3 + [name] * 2 + [description] + [colors] * 2
    ).strip()


def recommend_products(
    product_id: str,
    top_k: int = 5,
    database_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return the top-K most similar *other* active products.

    Similarity is a retrieval score, not a measured recommendation accuracy.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    products = _load_catalog(database_path or DEFAULT_DATABASE)
    if len(products) < 2:
        raise ValueError("At least two active products are required")

    product_ids = {p["id"]: index for index, p in enumerate(products)}
    if product_id not in product_ids:
        raise ValueError(f"Active product not found: {product_id}")

    documents = [_document(product) for product in products]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(documents)

    target_index = product_ids[product_id]
    target = products[target_index]
    scores = cosine_similarity(matrix[target_index], matrix).ravel()
    candidates = sorted(
        (index for index in range(len(products)) if index != target_index),
        key=lambda index: (-float(scores[index]), products[index]["id"]),
    )

    recommendations: list[dict[str, Any]] = []
    for index in candidates[:top_k]:
        candidate = products[index]
        shared_colors = sorted(set(target["colors"]) & set(candidate["colors"]))
        shared_category = target["category"] == candidate["category"]

        reasons: list[str] = []
        if shared_category:
            reasons.append("same category")
        if shared_colors:
            reasons.append("shared colors: " + ", ".join(shared_colors))
        if not reasons:
            reasons.append("similar product text")

        recommendations.append(
            {
                "id": candidate["id"],
                "slug": candidate["slug"],
                "name": candidate["name"],
                "category": candidate["category"],
                "price_cents": candidate["price_cents"],
                "image_url": candidate["image_url"],
                "similarity": round(float(scores[index]), 4),
                "reason": "; ".join(reasons),
            }
        )

    return recommendations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ÉLAN TF-IDF + cosine similarity recommendation baseline"
    )
    parser.add_argument("--product-id", required=True, help="Existing active product ID")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    try:
        results = recommend_products(args.product_id, args.top_k)
    except (ValueError, FileNotFoundError, sqlite3.Error) as error:
        parser.exit(1, f"ERROR: {error}\n")

    print(f"Reference product: {args.product_id}")
    print(f"Recommendation count: {len(results)}\n")
    for number, product in enumerate(results, start=1):
        print(
            f"{number}. {product['id']} | {product['name']} | "
            f"{product['category']} | similarity={product['similarity']:.4f}"
        )
        print(f"   Reason: {product['reason']}")


if __name__ == "__main__":
    main()
