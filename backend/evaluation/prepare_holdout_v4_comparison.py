r"""Export a BLIND review bundle for ELAN's frozen V1/V2/V3/V4 holdout.

Run from the repository root (PowerShell):
    .\backend\.venv\Scripts\python.exe .\backend\evaluation\prepare_holdout_v4_comparison.py

Prerequisites:
    backend/evaluation/holdout_v4_references.csv  (18 frozen references)
    backend/app/recommender.py, recommender_v2.py, recommender_v3.py, recommender_v4.py
    backend/evaluation/prepare_relevance_review.py
    backend/data/elan.db, data/private/hm/articles.csv, frontend/public/products/

Creates (does not overwrite):
    backend/evaluation/holdout_v4_candidate_union.csv   (MODEL RANKS; keep local)
    data/private/hm/holdout_v4_blind_review_bundle.zip  (safe for blind annotation;
        contains blind_review.csv + review_index.html + images, NO MODEL RANKS)

Neither SQLite, the private H&M dataset, nor any recommendation model/API is modified.
Do not put the private image ZIP in public Git/GitHub.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import sqlite3
import sys
import zipfile
from collections import Counter
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
EVAL = BACKEND / "evaluation"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.recommender import _load_catalog, recommend_products
from app.recommender_v2 import recommend_products_v2
from app.recommender_v3 import recommend_products_v3
from app.recommender_v4 import recommend_products_v4
from evaluation import prepare_relevance_review as images

REFERENCES_CSV = EVAL / "holdout_v4_references.csv"
UNION_CSV = EVAL / "holdout_v4_candidate_union.csv"
BUNDLE_ZIP = ROOT / "data" / "private" / "hm" / "holdout_v4_blind_review_bundle.zip"
DATABASE = BACKEND / "data" / "elan.db"
METADATA_CSV = ROOT / "data" / "private" / "hm" / "articles.csv"
MODEL_FILES = {
    "v1": BACKEND / "app" / "recommender.py",
    "v2": BACKEND / "app" / "recommender_v2.py",
    "v3": BACKEND / "app" / "recommender_v3.py",
    "v4": BACKEND / "app" / "recommender_v4.py",
}
EXPECTED_V4_SHA256 = "e6b6b3214d9e57cb14e4243625261e1a6c2a823929386324eceeb2d0fc37bfa6"
CATEGORIES = {"Accessories", "Bottoms", "Dresses", "Knitwear", "Outerwear", "Tops"}
TOP_K = 5
REFERENCES_EXPECTED = 18
REFERENCES_PER_CATEGORY = 3
REVIEW_SEED = 2026110807

# This unblinded CSV must NOT be placed inside the image ZIP.
UNION_COLUMNS = [
    "reference_id", "reference_name", "reference_category", "reference_url",
    "recommended_id", "recommended_name", "recommended_category", "recommended_url",
    "blind_order", "v1_rank", "v2_rank", "v3_rank", "v4_rank",
    "relevance_0_1_2", "notes",
]
# Match prepare_relevance_review.create_html's expected 'rank' name.
BLIND_COLUMNS = [
    "reference_id", "reference_name", "reference_category", "reference_url",
    "rank", "recommended_id", "recommended_name", "recommended_category",
    "recommended_url", "relevance_0_1_2", "notes",
]


def sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required input missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_text(rows: list[dict[str, str]], columns: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def get_references(active: dict[str, dict]) -> list[dict[str, str]]:
    if not REFERENCES_CSV.is_file():
        raise FileNotFoundError(f"Missing frozen holdout: {REFERENCES_CSV}")
    with REFERENCES_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"product_id", "name", "category", "slug"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Holdout reference CSV missing required columns")
        references = list(reader)
    if len(references) != REFERENCES_EXPECTED:
        raise ValueError(f"Expected {REFERENCES_EXPECTED} references, got {len(references)}")
    ids = [r["product_id"].strip() for r in references]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate reference IDs")
    counts = Counter(r["category"].strip() for r in references)
    if set(counts) != CATEGORIES or any(n != REFERENCES_PER_CATEGORY for n in counts.values()):
        raise ValueError(f"Expected 3 references per category; got {dict(counts)}")
    for ref in references:
        pid = ref["product_id"].strip()
        product = active.get(pid)
        if not product or not pid.startswith("HM-"):
            raise ValueError(f"Selected H&M reference no longer active: {pid}")
        for key, original in (("category", "category"), ("name", "name"), ("slug", "slug")):
            if ref[key].strip() != str(product[original]).strip():
                raise ValueError(f"Catalog changed for {pid}, column {key}")
    return references


def assert_unseen_references(references: list[dict[str, str]]) -> None:
    """Repeat leakage check: no new reference in the 130 earlier labeled pairs."""
    prior = (
        EVAL / "v1_v2_candidate_union.csv",
        EVAL / "holdout_v1_v2_candidate_union.csv",
    )
    reviewed_ids: set[str] = set()
    for path in prior:
        if not path.is_file():
            raise FileNotFoundError(f"Prior evaluated CSV missing: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not {
                "reference_id", "recommended_id", "relevance_0_1_2"
            }.issubset(reader.fieldnames):
                raise ValueError(f"Invalid prior labeled CSV: {path}")
            for row in reader:
                if row["relevance_0_1_2"].strip() not in {"0", "1", "2"}:
                    raise ValueError(f"Incomplete prior labels in {path}")
                reviewed_ids.add(row["reference_id"].strip())
                reviewed_ids.add(row["recommended_id"].strip())
    prior_refs = EVAL / "holdout_references.csv"
    if not prior_refs.is_file():
        raise FileNotFoundError(f"Prior reference CSV missing: {prior_refs}")
    with prior_refs.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "product_id" not in reader.fieldnames:
            raise ValueError("Prior reference CSV missing product_id")
        reviewed_ids.update(row["product_id"].strip() for row in reader)
    leaked = sorted(r["product_id"] for r in references if r["product_id"] in reviewed_ids)
    if leaked:
        raise ValueError("Previously reviewed products selected as references: " + ", ".join(leaked))


def validate_recommendations(
    results: list[dict], source_id: str, model: str, active: dict[str, dict]
) -> dict[str, int]:
    if len(results) != TOP_K:
        raise ValueError(f"{model} returned {len(results)} rather than {TOP_K} for {source_id}")
    ids = [str(r["id"]) for r in results]
    if len(set(ids)) != TOP_K or source_id in ids:
        raise ValueError(f"{model} has duplicate/self recommendations for {source_id}")
    for item in results:
        pid = str(item["id"])
        if pid not in active:
            raise ValueError(f"{model} returned inactive/unknown candidate {pid}")
        if any(str(item.get(k, "")) != str(active[pid][k]) for k in ("name", "category", "slug")):
            raise ValueError(f"{model} returned catalog fields inconsistent with SQLite: {pid}")
    return {pid: i for i, pid in enumerate(ids, start=1)}


def main() -> int:
    try:
        # Fail before computing anything if outputs already exist.
        existing = [str(p) for p in (UNION_CSV, BUNDLE_ZIP) if p.exists()]
        if existing:
            raise FileExistsError("Outputs exist; refusing to overwrite: " + ", ".join(existing))

        frozen_sha = {name: sha256(path) for name, path in MODEL_FILES.items()}
        if frozen_sha["v4"] != EXPECTED_V4_SHA256:
            raise ValueError(
                "V4 source differs from the frozen selection snapshot. "
                f"Expected {EXPECTED_V4_SHA256}; got {frozen_sha['v4']}. "
                "Do not change the model for this evaluation."
            )
        input_sha = {
            "holdout_references": sha256(REFERENCES_CSV),
            "sqlite": sha256(DATABASE),
            "hm_articles": sha256(METADATA_CSV),
        }
        products = _load_catalog(DATABASE)
        active = {p["id"]: p for p in products}
        if sum(pid.startswith("HM-") for pid in active) != 320:
            raise ValueError("H&M active catalog no longer has 320 products")
        references = get_references(active)
        assert_unseen_references(references)

        models = {
            "v1": recommend_products,
            "v2": recommend_products_v2,
            "v3": recommend_products_v3,
            "v4": recommend_products_v4,
        }
        rng = random.Random(REVIEW_SEED)
        union: list[dict[str, str]] = []
        blind: list[dict[str, str]] = []
        overlaps_v1_v4 = 0
        for ref in references:
            source_id = ref["product_id"].strip()
            outputs: dict[str, list[dict]] = {}
            ranks: dict[str, dict[str, int]] = {}
            for model_name, recommend in models.items():
                # Use the model's own unchanged default DB and metadata paths.
                outputs[model_name] = recommend(source_id, top_k=TOP_K)
                ranks[model_name] = validate_recommendations(
                    outputs[model_name], source_id, model_name, active
                )
            overlaps_v1_v4 += len(set(ranks["v1"]) & set(ranks["v4"]))
            candidate_info = {str(p["id"]): p for batch in outputs.values() for p in batch}
            candidate_ids = sorted(candidate_info)
            rng.shuffle(candidate_ids)
            source_url = "http://localhost:3000/shop/" + ref["slug"].strip()
            for blind_order, candidate_id in enumerate(candidate_ids, 1):
                candidate = active[candidate_id]
                common = {
                    "reference_id": source_id,
                    "reference_name": ref["name"].strip(),
                    "reference_category": ref["category"].strip(),
                    "reference_url": source_url,
                    "recommended_id": candidate_id,
                    "recommended_name": str(candidate["name"]),
                    "recommended_category": str(candidate["category"]),
                    "recommended_url": "http://localhost:3000/shop/" + str(candidate["slug"]),
                    "relevance_0_1_2": "",
                    "notes": "",
                }
                union.append({
                    **common,
                    "blind_order": str(blind_order),
                    **{f"{m}_rank": str(ranks[m].get(candidate_id, "")) for m in models},
                })
                blind.append({**common, "rank": str(blind_order)})

        total_positions = len(references) * TOP_K * len(models)
        if not (len(references) * TOP_K <= len(union) <= total_positions):
            raise AssertionError("Candidate union has impossible size")
        unique_ids = {r["reference_id"] for r in blind} | {r["recommended_id"] for r in blind}
        image_map = images.select_images(unique_ids)
        missing = sorted(pid for pid, file in image_map.items() if file is None)
        missing_refs = sorted(set(r["product_id"] for r in references) & set(missing))
        if missing_refs:
            raise ValueError(
                "Reference photos missing; cannot perform a complete blind visual review: "
                + ", ".join(missing_refs)
            )
        asset_map = {
            pid: f"images/{pid}{path.suffix.lower()}" if path else None
            for pid, path in image_map.items()
        }
        # Check that the models, references and data did not change during export.
        if {m: sha256(p) for m, p in MODEL_FILES.items()} != frozen_sha:
            raise ValueError("At least one model file changed during export")
        if {
            "holdout_references": sha256(REFERENCES_CSV),
            "sqlite": sha256(DATABASE),
            "hm_articles": sha256(METADATA_CSV),
        } != input_sha:
            raise ValueError("Holdout references or underlying data changed during export")
        manifest = {
            "purpose": "Private blind visual relevance assessment; do not publish H&M images",
            "reference_count": len(references),
            "model_count": len(models),
            "top_k": TOP_K,
            "recommendation_positions_per_model": len(references) * TOP_K,
            "candidate_union_pairs": len(union),
            "unique_products": len(unique_ids),
            "images_included": len(unique_ids) - len(missing),
            "images_missing": missing,
            "label_scale": {
                "0": "not a reasonable substitute",
                "1": "partly related / potentially useful alternative",
                "2": "close, convincing substitute",
            },
            "blind_review_seed": REVIEW_SEED,
            "frozen_v4_sha256": EXPECTED_V4_SHA256,
            "model_sha256": frozen_sha,
            "source_sha256": input_sha,
            "important": (
                "Candidate order is randomized. This ZIP contains no model-rank mapping. "
                "Keep holdout_v4_candidate_union.csv locally for later metrics; "
                "assign relevance labels before accessing rank mappings."
            ),
        }
        UNION_CSV.parent.mkdir(parents=True, exist_ok=True)
        BUNDLE_ZIP.parent.mkdir(parents=True, exist_ok=True)
        csv_temp = UNION_CSV.with_name(UNION_CSV.name + ".tmp")
        zip_temp = BUNDLE_ZIP.with_name(BUNDLE_ZIP.name + ".tmp")
        try:
            with csv_temp.open("x", encoding="utf-8-sig", newline="") as stream:
                stream.write(csv_text(union, UNION_COLUMNS))
            with zipfile.ZipFile(zip_temp, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr("blind_review.csv", csv_text(blind, BLIND_COLUMNS))
                bundle.writestr(
                    "review_index.html",
                    images.create_html(blind, asset_map).replace(
                        "Top-5 suggested products", "Blinded candidate pool (shuffled, not model rank)"
                    ),
                )
                bundle.writestr(
                    "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2)
                )
                bundle.writestr("holdout_references.csv", csv_text(references, list(references[0])))
                for product_id in sorted(unique_ids):
                    image_path = image_map[product_id]
                    if image_path is not None:
                        bundle.write(image_path, asset_map[product_id])
            # Publish ZIP and local-union only after successful construction.
            zip_temp.replace(BUNDLE_ZIP)
            csv_temp.replace(UNION_CSV)
        finally:
            csv_temp.unlink(missing_ok=True)
            zip_temp.unlink(missing_ok=True)

        print("V4 FOUR-MODEL HOLDOUT EXPORT: COMPLETE")
        print("Reference products:", len(references))
        for name in models:
            print(f"{name.upper()} Top-5 positions:", len(references) * TOP_K)
        print("Total model recommendation positions:", total_positions)
        print("Unique reference-candidate pairs (need labels):", len(union))
        print("V1/V4 Top-5 overlap:", overlaps_v1_v4, "/", len(references) * TOP_K)
        print("Unique product images required:", len(unique_ids))
        print("Images included in private ZIP:", len(unique_ids) - len(missing))
        print("Images missing:", missing)
        print("Frozen V4 SHA256:", EXPECTED_V4_SHA256)
        print("Local full-rank union CSV:", UNION_CSV)
        print("Private blind-review ZIP:", BUNDLE_ZIP)
        print("IMPORTANT: Upload ZIP only for blind labeling; do not upload full-rank union CSV yet.")
        print("Existing labels, SQLite, V1/V2/V3/V4, and production API were NOT modified.")
        return 0
    except (OSError, ValueError, KeyError, AssertionError, sqlite3.Error, zipfile.BadZipFile) as exc:
        print("ERROR:", exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
