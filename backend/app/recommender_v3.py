"""ELAN recommendation v3: conservative product-type TF-IDF reranking.

Offline experiment only. Does NOT replace v1/v2 or the production API.

From the backend directory:
    python -m app.recommender_v3 --product-id HM-0341242026 --top-k 5
    python -m app.recommender_v3 --self-test

The underlying cosine similarity is identical to v1. ``ranking_score`` adds
transparent, *untuned* heuristic boosts/penalties for garment types. Product
type is inferred from the product name only; unknown types stay unknown.

IMPORTANT: The 12-reference holdout used to diagnose v2 must not be reused as
an independent test set after making these changes. Evaluate v3 on NEW data.
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
from app.recommender_v2 import recommend_products_v2


# Priority matters. More specific garment nouns precede potentially ambiguous
# words (e.g. "body dress" should be a dress, not a bodysuit).
# Do NOT label "windproof" as outerwear: it describes a property, not a type.
_SUBTYPE_RULES: tuple[tuple[str, str], ...] = (
    ("sunglasses", r"\b(?:sunglasses|sun[ -]?glasses|shades)\b"),
    ("earrings", r"\b(?:earrings?|hoops?|ear[ -]?studs?|huggies?)\b"),
    ("necklaces", r"\b(?:necklaces?|pendants?)\b"),
    ("headwear", r"\b(?:headbands?|hairbands?|hair[ -]?clips?|scrunchies?)\b"),
    ("patches", r"\b(?:patches|badges?)\b"),
    ("bags", r"\b(?:bags?|handbags?|cross[ -]?body(?: bags?)?|totes?|backpacks?|purses?)\b"),
    ("shorts", r"\b(?:shorts?|bermudas?)\b"),
    ("joggers", r"\b(?:joggers?|sweatpants?|track[ -]?pants?)\b"),
    ("jeans", r"\b(?:jeans?|denims?)\b"),
    ("skirts", r"\bskirts?\b"),
    ("trousers", r"\b(?:trousers?|pants?|chinos?|leggings?)\b"),
    ("dresses", r"\b(?:dresses|dress)\b"),
    ("tunics", r"\btunics?\b"),
    ("bodysuits", r"\b(?:bodysuits?|body[ -]?suits?|tank[ -]?top[ -]?body|body)\b"),
    ("tshirts", r"\b(?:t[ -]?shirts?|tees?|tshirt)\b"),
    ("shirts", r"\b(?:shirts?|blouses?)\b"),
    ("cardigans", r"\bcardigans?\b"),
    ("sweaters", r"\b(?:sweaters?|jumpers?|pullovers?)\b"),
    ("hoodies", r"\bhoodies?\b"),
    ("sweatshirts", r"\bsweatshirts?\b"),
    ("zip_tops", r"\b(?:zip(?:ped)?[ -]+tops?|zip[ -]?neck(?:[ -]+tops?)?)\b"),
    ("waistcoats", r"\bwaistcoats?\b"),
    ("windbreakers", r"\bwindbreakers?\b"),
    ("parkas", r"\bparkas?\b"),
    ("coats", r"\bcoats?\b"),
    ("blazers", r"\bblazers?\b"),
    ("jackets", r"\bjackets?\b"),
    ("vests", r"\bvests?\b"),
)
_COMPILED_RULES = tuple(
    (label, re.compile(regex, re.IGNORECASE)) for label, regex in _SUBTYPE_RULES
)

# Weak compatibility means related *forms*, not identical product types.
# In particular: shorts != joggers; coat != waistcoat; bodysuit != ordinary top.
_RELATED_TYPES = frozenset(
    {
        frozenset(("coats", "parkas")),
        frozenset(("jackets", "windbreakers")),
        frozenset(("jackets", "parkas")),
        frozenset(("jackets", "zip_tops")),
        frozenset(("zip_tops", "windbreakers")),
        frozenset(("waistcoats", "vests")),
        frozenset(("sweaters", "cardigans")),
        frozenset(("sweaters", "sweatshirts")),
        frozenset(("tshirts", "shirts")),
        frozenset(("dresses", "tunics")),
    }
)

# Deliberately fixed, interpretable starting parameters; not model-fit weights.
CATEGORY_BONUS = 0.10
EXACT_TYPE_BONUS = 0.22
RELATED_TYPE_BONUS = 0.05
SAME_CATEGORY_TYPE_MISMATCH_PENALTY = 0.14
CROSS_CATEGORY_TYPE_MISMATCH_PENALTY = 0.06


def infer_subtype(product: dict[str, Any]) -> str | None:
    """Conservative subtype prediction from the name; unknown stays None."""
    name = str(product.get("name") or "")
    category = str(product.get("category") or "").strip().lower()
    for label, pattern in _COMPILED_RULES:
        if pattern.search(name):
            # A bare "body" is ambiguous outside Tops; explicit bodysuits
            # remain recognized regardless of catalog category.
            if label == "bodysuits" and re.search(r"\bbody\b", name, re.IGNORECASE):
                explicit = re.search(
                    r"\b(?:bodysuits?|body[ -]?suits?|tank[ -]?top[ -]?body)\b",
                    name,
                    re.IGNORECASE,
                )
                if not explicit and category not in ("", "tops"):
                    continue
            return label
    return None


def _type_relation(target_type: str | None, candidate_type: str | None) -> str:
    """Return 'unknown', 'exact', 'related', or 'different'."""
    if target_type is None or candidate_type is None:
        return "unknown"
    if target_type == candidate_type:
        return "exact"
    if frozenset((target_type, candidate_type)) in _RELATED_TYPES:
        return "related"
    return "different"


def _ranking_score(cosine: float, same_category: bool, relation: str) -> float:
    score = cosine + (CATEGORY_BONUS if same_category else 0.0)
    if relation == "exact":
        score += EXACT_TYPE_BONUS
    elif relation == "related":
        score += RELATED_TYPE_BONUS
    elif relation == "different":
        score -= (
            SAME_CATEGORY_TYPE_MISMATCH_PENALTY
            if same_category else CROSS_CATEGORY_TYPE_MISMATCH_PENALTY
        )
    return score


def recommend_products_v3(
    product_id: str,
    top_k: int = 5,
    database_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Retrieve and rerank other active catalog products; read-only SQLite."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    products = _load_catalog(database_path or DEFAULT_DATABASE)
    if len(products) < 2:
        raise ValueError("At least two active products are required")

    index_by_id = {product["id"]: idx for idx, product in enumerate(products)}
    if product_id not in index_by_id:
        raise ValueError(f"Active product not found: {product_id}")

    documents = [_document(product) for product in products]
    matrix = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), sublinear_tf=True
    ).fit_transform(documents)

    source_idx = index_by_id[product_id]
    source = products[source_idx]
    source_type = infer_subtype(source)
    cosine_scores = cosine_similarity(matrix[source_idx], matrix).ravel()

    ranked: list[tuple[float, int, str | None, str]] = []
    for idx, candidate in enumerate(products):
        if idx == source_idx:
            continue
        candidate_type = infer_subtype(candidate)
        relation = _type_relation(source_type, candidate_type)
        same_category = source["category"] == candidate["category"]
        score = _ranking_score(float(cosine_scores[idx]), same_category, relation)
        ranked.append((score, idx, candidate_type, relation))

    ranked.sort(key=lambda row: (-row[0], products[row[1]]["id"]))
    results: list[dict[str, Any]] = []
    for rank_score, idx, candidate_type, relation in ranked[:top_k]:
        candidate = products[idx]
        same_category = source["category"] == candidate["category"]
        shared_colors = sorted(set(source["colors"]) & set(candidate["colors"]))
        reasons: list[str] = []
        if relation == "exact":
            reasons.append(f"same product type: {source_type}")
        elif relation == "related":
            reasons.append(f"related product types: {source_type} / {candidate_type}")
        if same_category:
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
                "similarity": round(float(cosine_scores[idx]), 4),
                "ranking_score": round(rank_score, 4),
                "product_type": candidate_type,
                "type_relation": relation,
                "reason": "; ".join(reasons),
            }
        )
    return results


def _self_test() -> None:
    examples = {
        "ELVIRA jacket": "jackets",
        "Capri double face coat": "coats",
        "Leia Parka": "parkas",
        "Dave waistcoat": "waistcoats",
        "Sage Fur Vest": "vests",
        "Gauss windproof zip top": "zip_tops",
        "2P Tanktop body Mixed": "bodysuits",
        "CAKE body dress": "dresses",
        "Sigge shorts": "shorts",
        "AXEL JOGGERS": "joggers",
        "Flirty Chantal earring": "earrings",
        "Astrid round sunglasses": "sunglasses",
        "Vanessa": None,
    }
    for name, expected in examples.items():
        actual = infer_subtype({"name": name})
        assert actual == expected, f"{name!r}: expected {expected}, got {actual}"
    assert _type_relation("coats", "waistcoats") == "different"
    assert _type_relation("jackets", "windbreakers") == "related"
    assert _type_relation("dresses", "dresses") == "exact"
    assert _type_relation(None, "dresses") == "unknown"
    assert _ranking_score(0.2, True, "exact") > _ranking_score(0.2, True, "different")
    print(f"Self-test: PASS ({len(examples)} name rules + relation checks)")


def main() -> None:
    parser = argparse.ArgumentParser(description="ELAN v3 offline type-aware reranker")
    parser.add_argument("--product-id", help="Active product ID")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.product_id:
        parser.error("--product-id is required unless --self-test is set")

    try:
        catalog = _load_catalog(DEFAULT_DATABASE)
        target = next((p for p in catalog if p["id"] == args.product_id), None)
        baseline = recommend_products(args.product_id, args.top_k)
        v2 = recommend_products_v2(args.product_id, args.top_k)
        v3 = recommend_products_v3(args.product_id, args.top_k)
    except (ValueError, FileNotFoundError, sqlite3.Error) as exc:
        parser.exit(1, f"ERROR: {exc}\n")

    print(f"\nReference: {args.product_id} | {target['name']} | category={target['category']}")
    print(f"Name-derived subtype (v3): {infer_subtype(target) or 'UNKNOWN'}")
    for label, items in (("V1 - PRODUCTION", baseline), ("V2 - SAVED EXPERIMENT", v2), ("V3 - NEW OFFLINE EXPERIMENT", v3)):
        print(f"\n{label}")
        for position, item in enumerate(items, 1):
            summary = (
                f"{position}. {item['id']} | {item['name']} | {item['category']} "
                f"| cosine={item['similarity']:.4f}"
            )
            if "ranking_score" in item:
                summary += f" | ranking_score={item['ranking_score']:.4f}"
            if "type_relation" in item:
                summary += f" | type={item['product_type'] or 'UNKNOWN'} | relation={item['type_relation']}"
            print(summary)
    print("\nSQLite, V1, V2, and production API were NOT modified.")
    print("Previous holdout is now development evidence, not an independent v3 holdout.")


if __name__ == "__main__":
    main()
