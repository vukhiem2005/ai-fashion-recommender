"""ELAN recommendation v2 experiment: category/subtype-aware TF-IDF reranking.

This file does NOT replace the stable v1 recommender or production API.

Run from backend/: 
    python -m app.recommender_v2 --product-id HM-0292453013 --top-k 5
Scores:
    similarity: pure v1 TF-IDF cosine similarity (0..1)
    ranking_score: experimental score (not cosine and not accuracy)

Product subtype is inferred from NAME only using explicit, inspectable rules.
This is a baseline heuristic, not a trained subtype classifier.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.recommender import DEFAULT_DATABASE, _document, _load_catalog, recommend_products

# Rules are intentionally conservative. Unknown subtype -> no subtype reward/penalty.
# Match longer/more specific expressions first to avoid accidental matches.
SUBTYPE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("sunglasses", r"\b(sunglasses|sun glasses|shades)\b"),
    ("earrings", r"\b(earring|earrings|hoops?|studs?|huggies?)\b"),
    ("necklaces", r"\b(necklaces?|pendants?)\b"),
    ("headwear", r"\b(headbands?|hairbands?|hair clips?|scrunchies?)\b"),
    ("patches", r"\b(patches?|badges?)\b"),
    ("bags", r"\b(bags?|handbags?|crossbody|totes?|backpacks?|purses?)\b"),
    ("shorts", r"\b(shorts?|bermudas?)\b"),
    ("joggers", r"\b(joggers?|sweatpants?|track pants?)\b"),
    ("jeans", r"\b(jeans?|denims?)\b"),
    ("skirts", r"\b(skirts?)\b"),
    ("trousers", r"\b(trousers?|pants?|chinos?|leggings?)\b"),
    ("dresses", r"\b(dresses?|dress)\b"),
    ("tunics", r"\b(tunics?)\b"),
    ("bodysuits", r"\b(bodysuits?|tanktop body|tank top body|\bbody\b)\b"),
    ("t-shirts", r"\b(t-shirts?|tees?|tshirt)\b"),
    ("shirts", r"\b(shirts?|blouses?)\b"),
    ("sweaters", r"\b(sweaters?|jumpers?|pullovers?)\b"),
    ("cardigans", r"\b(cardigans?)\b"),
    ("outerwear", r"\b(jackets?|coats?|blazers?|parkas?|windproof|windbreakers?|waistcoats?|vests?)\b"),
)

_COMPILED = [(label, re.compile(regex, flags=re.IGNORECASE)) for label, regex in SUBTYPE_PATTERNS]


def infer_subtype(product: dict[str, Any]) -> str | None:
    """Heuristic name-only garment type; return None if no supported match.

    Name-only avoids treating words in styling suggestions within a description
    as the article's actual type. Improve with curated product_type later.
    """
    name = str(product.get("name") or "")
    for subtype, pattern in _COMPILED:
        if pattern.search(name):
            return subtype
    return None


def recommend_products_v2(
    product_id: str,
    top_k: int = 5,
    database_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Experimentally rerank candidates; do not update any SQLite records."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    products = _load_catalog(database_path or DEFAULT_DATABASE)
    if len(products) < 2:
        raise ValueError("At least two active products are required")

    index_by_id = {product["id"]: idx for idx, product in enumerate(products)}
    if product_id not in index_by_id:
        raise ValueError(f"Active product not found: {product_id}")

    matrix = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), sublinear_tf=True
    ).fit_transform([_document(p) for p in products])

    target_index = index_by_id[product_id]
    target = products[target_index]
    target_type = infer_subtype(target)
    text_scores = cosine_similarity(matrix[target_index], matrix).ravel()

    ranked: list[tuple[float, int, str | None]] = []
    for index, candidate in enumerate(products):
        if index == target_index:
            continue
        candidate_type = infer_subtype(candidate)
        same_category = target["category"] == candidate["category"]

        # Keep the actual cosine separate from the experimental reranking score.
        score = float(text_scores[index])
        if same_category:
            score += 0.12
        if target_type is not None and candidate_type is not None:
            if target_type == candidate_type:
                score += 0.22
            elif same_category:
                score -= 0.18
        ranked.append((score, index, candidate_type))

    ranked.sort(key=lambda row: (-row[0], products[row[1]]["id"]))

    results: list[dict[str, Any]] = []
    for score, index, candidate_type in ranked[:top_k]:
        candidate = products[index]
        matched_category = target["category"] == candidate["category"]
        shared_colors = sorted(set(target["colors"]) & set(candidate["colors"]))
        reasons: list[str] = []
        if target_type and target_type == candidate_type:
            reasons.append("same product type: " + target_type)
        if matched_category:
            reasons.append("same category")
        if shared_colors:
            reasons.append("shared colors: " + ", ".join(shared_colors))
        if not reasons:
            reasons.append("similar product text")
        results.append(
            {
                "id": candidate["id"],
                "slug": candidate["slug"],
                "name": candidate["name"],
                "category": candidate["category"],
                "price_cents": candidate["price_cents"],
                "image_url": candidate["image_url"],
                "similarity": round(float(text_scores[index]), 4),
                "ranking_score": round(score, 4),
                "product_type": candidate_type,
                "reason": "; ".join(reasons),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="ELAN TF-IDF v2 offline reranking experiment")
    parser.add_argument("--product-id", required=True, help="Active product ID")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    try:
        baseline = recommend_products(args.product_id, args.top_k)
        experimental = recommend_products_v2(args.product_id, args.top_k)
    except (ValueError, FileNotFoundError, sqlite3.Error) as exc:
        parser.exit(1, f"ERROR: {exc}\n")

    print("\nV1 - CURRENT PRODUCTION BASELINE")
    for rank, product in enumerate(baseline, 1):
        print(f"{rank}. {product['id']} | {product['name']} | {product['category']} | cosine={product['similarity']:.4f}")
    print("\nV2 - OFFLINE EXPERIMENT (not deployed)")
    for rank, product in enumerate(experimental, 1):
        print(
            f"{rank}. {product['id']} | {product['name']} | {product['category']} "
            f"| cosine={product['similarity']:.4f} | rank_score={product['ranking_score']:.4f}"
        )
        print("   Reason: " + product["reason"])
    overlap = len({p["id"] for p in baseline} & {p["id"] for p in experimental})
    print(f"\nTop-{args.top_k} overlap: {overlap}/{min(args.top_k, len(baseline), len(experimental))}")
    print("SQLite was NOT modified. Production recommender/API were NOT modified.")


if __name__ == "__main__":
    main()
