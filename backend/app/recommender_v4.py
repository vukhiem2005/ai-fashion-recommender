"""ELAN V4: metadata-aware, content-based recommendation (OFFLINE ONLY).

Uses the *original H&M product_type_name* as an explicit reranking signal.
TF-IDF/cosine is intentionally identical to V1 (same documents/vectorizer).
It reads the private H&M CSV and local SQLite; it writes to neither.

Local usage from backend/:
    python -m app.recommender_v4 --self-test
    python -m app.recommender_v4 --product-id HM-0225286014 --top-k 5

Important limitations:
- The private data/private/hm/articles.csv is NOT distributed with GitHub and
  must NOT be required by the existing production API.
- Original ELAN demo products have no H&M metadata; their type remains UNKNOWN.
- Exact H&M product type is taxonomy agreement, not proof of visual similarity.
- The earlier V1/V2 holdout has influenced V4 design; evaluate V4 on NEW labels.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.recommender import DEFAULT_DATABASE, _document, _load_catalog, recommend_products

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_CSV = PROJECT_ROOT / "data" / "private" / "hm" / "articles.csv"

# Predefined, conservative relations between distinct types in H&M taxonomy.
# 'Related' means potentially useful alternatives, NOT exact substitutes.
# These were not fitted to the prior V1/V2 holdout labels.
_RELATED_FAMILIES = (
    frozenset({"Top", "T-shirt", "Vest top", "Polo shirt"}),
    frozenset({"Shirt", "Blouse"}),
    frozenset({"Sweater", "Cardigan"}),
    frozenset({"Outdoor Waistcoat", "Tailored Waistcoat"}),
)

# Transparent heuristic coefficients; ranking_score is NOT a probability.
CATEGORY_BONUS = 0.04
EXACT_TYPE_BONUS = 0.24
RELATED_TYPE_BONUS = 0.06
DIFFERENT_TYPE_PENALTY = 0.08


def _normalized_type(value: str | None) -> str | None:
    """Normalize taxonomy labels for comparisons, without guessing a type."""
    normalized = " ".join(str(value or "").split())
    return normalized.casefold() or None


def _load_hm_product_types(
    csv_path: Path, active_product_ids: set[str]
) -> dict[str, str]:
    """Map HM-article IDs to original type labels. Fail on incomplete HM coverage.

    IDs remain strings to preserve leading zeros. Non-HM demo products are
    deliberately excluded rather than assigned an inferred H&M type.
    """
    hm_ids = {pid for pid in active_product_ids if pid.startswith("HM-")}
    if not hm_ids:
        return {}
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"H&M metadata not found: {csv_path}. "
            "V4 is an offline experiment and requires the private articles.csv."
        )

    expected_article_ids = {pid.removeprefix("HM-") for pid in hm_ids}
    seen: dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"article_id", "product_type_name"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"H&M CSV requires columns {sorted(required)}")
        for row in reader:
            aid = str(row["article_id"] or "").strip()
            if aid not in expected_article_ids:
                continue
            label = str(row["product_type_name"] or "").strip()
            if not label:
                raise ValueError(f"Missing product_type_name for HM-{aid}")
            if aid in seen and _normalized_type(seen[aid]) != _normalized_type(label):
                raise ValueError(f"Conflicting product_type_name for HM-{aid}")
            seen[aid] = label

    missing = sorted(expected_article_ids - set(seen))
    if missing:
        preview = ", ".join("HM-" + aid for aid in missing[:5])
        raise ValueError(
            f"Missing H&M metadata for {len(missing)} active products: {preview}"
        )
    return {"HM-" + aid: label for aid, label in seen.items()}


def _relation(type_a: str | None, type_b: str | None) -> str:
    """Classify type agreement, independent of possibly noisy ELAN category."""
    left = _normalized_type(type_a)
    right = _normalized_type(type_b)
    if not left or not right:
        return "unknown"
    if left == right:
        return "exact"
    if any(
        left in {_normalized_type(label) for label in family}
        and right in {_normalized_type(label) for label in family}
        for family in _RELATED_FAMILIES
    ):
        return "related"
    return "different"


def _ranking_score(cosine: float, same_category: bool, relation: str) -> float:
    score = float(cosine) + (CATEGORY_BONUS if same_category else 0.0)
    if relation == "exact":
        score += EXACT_TYPE_BONUS
    elif relation == "related":
        score += RELATED_TYPE_BONUS
    elif relation == "different":
        score -= DIFFERENT_TYPE_PENALTY
    elif relation != "unknown":
        raise ValueError(f"Unexpected type relation: {relation}")
    return score


def recommend_products_v4(
    product_id: str,
    top_k: int = 5,
    database_path: Path | None = None,
    metadata_csv: Path | None = None,
) -> list[dict[str, Any]]:
    """Rank active catalog items; never write the catalog or CSV."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    products = _load_catalog(database_path or DEFAULT_DATABASE)
    if len(products) < 2:
        raise ValueError("At least two active products are required")
    index_by_id = {item["id"]: index for index, item in enumerate(products)}
    if product_id not in index_by_id:
        raise ValueError(f"Active product not found: {product_id}")

    types = _load_hm_product_types(
        metadata_csv or DEFAULT_METADATA_CSV, set(index_by_id)
    )
    matrix = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), sublinear_tf=True
    ).fit_transform([_document(item) for item in products])
    target_index = index_by_id[product_id]
    target = products[target_index]
    target_type = types.get(product_id)
    similarities = cosine_similarity(matrix[target_index], matrix).ravel()

    ranked: list[tuple[float, int, str]] = []
    for idx, candidate in enumerate(products):
        if idx == target_index:
            continue
        candidate_type = types.get(candidate["id"])
        relation = _relation(target_type, candidate_type)
        same_category = target["category"] == candidate["category"]
        score = _ranking_score(float(similarities[idx]), same_category, relation)
        ranked.append((score, idx, relation))

    ranked.sort(key=lambda entry: (-entry[0], products[entry[1]]["id"]))
    results: list[dict[str, Any]] = []
    for rank_score, idx, relation in ranked[:top_k]:
        candidate = products[idx]
        candidate_type = types.get(candidate["id"])
        same_category = target["category"] == candidate["category"]
        shared_colors = sorted(set(target["colors"]) & set(candidate["colors"]))
        reasons: list[str] = []
        if relation == "exact":
            reasons.append(f"exact H&M product type: {target_type}")
        elif relation == "related":
            reasons.append(f"related H&M product types: {target_type} / {candidate_type}")
        if same_category:
            reasons.append("same ELAN category")
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
                "similarity": round(float(similarities[idx]), 4),
                "ranking_score": round(rank_score, 4),
                "product_type": candidate_type,
                "type_relation": relation,
                "reason": "; ".join(reasons),
            }
        )
    return results


def _self_test() -> None:
    # Product type matching follows H&M labels, not vague name keywords.
    assert _relation("Sweater", "Sweater") == "exact"
    assert _relation("Sweater", "Cardigan") == "related"
    assert _relation("Tailored Waistcoat", "Outdoor Waistcoat") == "related"
    assert _relation("Tailored Waistcoat", "Jacket") == "different"
    assert _relation("Bodysuit", "Top") == "different"
    assert _relation("Vest top", "Top") == "related"
    assert _relation("Jacket", "T-shirt") == "different"
    assert _relation(None, "Sweater") == "unknown"
    assert _relation("  t-SHIRT ", "T-shirt") == "exact"
    assert _ranking_score(0.1, True, "exact") > _ranking_score(0.1, True, "related")
    assert _ranking_score(0.1, True, "related") > _ranking_score(0.1, True, "different")
    assert _ranking_score(0.1, False, "exact") > _ranking_score(0.1, True, "different")
    print("V4 self-test: PASS (12 checks)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ELAN V4 offline metadata-aware TF-IDF reranking"
    )
    parser.add_argument("--product-id", help="Active catalog product ID")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.product_id:
        parser.error("--product-id is required unless --self-test is set")

    try:
        products = _load_catalog(DEFAULT_DATABASE)
        current = next((p for p in products if p["id"] == args.product_id), None)
        if current is None:
            raise ValueError(f"Active product not found: {args.product_id}")
        type_lookup = _load_hm_product_types(
            DEFAULT_METADATA_CSV, {p["id"] for p in products}
        )
        v1 = recommend_products(args.product_id, args.top_k)
        v4 = recommend_products_v4(args.product_id, args.top_k)
    except (OSError, ValueError, sqlite3.Error, csv.Error) as exc:
        parser.exit(1, f"ERROR: {exc}\n")

    print(
        f"\nReference: {args.product_id} | {current['name']} | "
        f"ELAN category={current['category']} | "
        f"H&M type={type_lookup.get(args.product_id, 'UNKNOWN')}"
    )
    print("\nV1 - CURRENT PRODUCTION (unchanged)")
    for rank, item in enumerate(v1, 1):
        print(
            f"{rank}. {item['id']} | {item['name']} | "
            f"{item['category']} | cosine={item['similarity']:.4f}"
        )
    print("\nV4 - METADATA-AWARE OFFLINE EXPERIMENT")
    for rank, item in enumerate(v4, 1):
        print(
            f"{rank}. {item['id']} | {item['name']} | {item['category']} | "
            f"cosine={item['similarity']:.4f} | "
            f"rank_score={item['ranking_score']:.4f} | "
            f"H&M type={item['product_type'] or 'UNKNOWN'} | "
            f"relation={item['type_relation']}"
        )
    overlap = len({item["id"] for item in v1} & {item["id"] for item in v4})
    print(f"\nV1/V4 Top-{args.top_k} overlap: {overlap}")
    print("SQLite, V1/V2/V3, and production API were NOT modified.")
    print("Older holdouts are development evidence; new independent labels required.")


if __name__ == "__main__":
    main()
