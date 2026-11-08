r"""Build a private, blind image-review bundle for ELAN's V1/V2 holdout evaluation.

Windows PowerShell (from repository root):
    .\backend\.venv\Scripts\python.exe .\backend\evaluation\prepare_holdout_v1_v2.py

Inputs (read only):
    backend/evaluation/holdout_references.csv
    backend/data/elan.db
    backend/app/recommender.py
    backend/app/recommender_v2.py
    backend/evaluation/prepare_relevance_review.py

Outputs (new files only):
    backend/evaluation/holdout_v1_v2_candidate_union.csv
    data/private/hm/holdout_v1_v2_review_bundle.zip

Neither model, the API, SQLite nor existing evaluation files are modified.
Do not publish the private ZIP: it contains local H&M catalog images.
"""

from __future__ import annotations

import csv
import io
import json
import random
import sqlite3
import sys
import zipfile
from collections import Counter, OrderedDict
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.recommender import _load_catalog, recommend_products
from app.recommender_v2 import recommend_products_v2
from evaluation import prepare_relevance_review as images

REFERENCES_CSV = BACKEND / "evaluation" / "holdout_references.csv"
UNION_CSV = BACKEND / "evaluation" / "holdout_v1_v2_candidate_union.csv"
BUNDLE_ZIP = ROOT / "data" / "private" / "hm" / "holdout_v1_v2_review_bundle.zip"
EXPECTED_CATEGORIES = {"Accessories", "Bottoms", "Dresses", "Knitwear", "Outerwear", "Tops"}
TOP_K = 5
SEED = 20261108

COLUMNS = [
    "reference_id", "reference_name", "reference_category", "reference_url",
    "recommended_id", "recommended_name", "recommended_category", "recommended_url",
    "v1_rank", "v2_rank", "relevance_0_1_2", "notes",
]
BLIND_COLUMNS = [
    "reference_id", "reference_name", "reference_category", "reference_url",
    "rank", "recommended_id", "recommended_name", "recommended_category",
    "recommended_url", "relevance_0_1_2", "notes",
]


def as_csv(rows: list[dict[str, str]], fieldnames: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def load_references() -> list[dict[str, str]]:
    if not REFERENCES_CSV.is_file():
        raise FileNotFoundError(f"Holdout references not found: {REFERENCES_CSV}")
    with REFERENCES_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 12:
        raise ValueError(f"Expected 12 holdout reference products, got {len(rows)}")
    if not {"product_id", "name", "category", "slug"}.issubset(rows[0]):
        raise ValueError("holdout_references.csv has missing required columns")
    ids = [r["product_id"].strip() for r in rows]
    if len(set(ids)) != 12:
        raise ValueError("Duplicate holdout product IDs")
    counts = Counter(r["category"].strip() for r in rows)
    if set(counts) != EXPECTED_CATEGORIES or any(n != 2 for n in counts.values()):
        raise ValueError(f"Expected exactly 2 references per category: {dict(counts)}")

    active = {row["id"]: row for row in _load_catalog(BACKEND / "data" / "elan.db")}
    for row in rows:
        pid = row["product_id"].strip()
        if pid not in active:
            raise ValueError(f"Holdout product is not active: {pid}")
        if row["category"].strip() != active[pid]["category"]:
            raise ValueError(f"Category changed for {pid}")
    return rows


def main() -> int:
    try:
        references = load_references()
        if UNION_CSV.exists() or BUNDLE_ZIP.exists():
            existing = [str(p) for p in (UNION_CSV, BUNDLE_ZIP) if p.exists()]
            raise FileExistsError(
                "Output already exists; not overwriting: " + ", ".join(existing)
            )

        union: list[dict[str, str]] = []
        blind: list[dict[str, str]] = []
        overlap_total = 0
        rng = random.Random(SEED)

        for ref in references:
            pid = ref["product_id"].strip()
            v1 = recommend_products(pid, top_k=TOP_K)
            v2 = recommend_products_v2(pid, top_k=TOP_K)
            v1_ids = [p["id"] for p in v1]
            v2_ids = [p["id"] for p in v2]
            if len(v1) != TOP_K or len(v2) != TOP_K:
                raise ValueError(f"Not enough recommendations for {pid}")
            if len(set(v1_ids)) != TOP_K or len(set(v2_ids)) != TOP_K:
                raise ValueError(f"Duplicate recommendations for {pid}")
            if pid in v1_ids or pid in v2_ids:
                raise ValueError(f"Model recommended the reference itself: {pid}")

            v1_ranks = {p["id"]: rank for rank, p in enumerate(v1, 1)}
            v2_ranks = {p["id"]: rank for rank, p in enumerate(v2, 1)}
            overlap_total += len(set(v1_ids) & set(v2_ids))
            # The annotator sees randomly ordered candidates, without model identity.
            by_id = {p["id"]: p for p in (v1 + v2)}
            candidates = sorted(by_id)
            rng.shuffle(candidates)
            ref_url = "http://localhost:3000/shop/" + ref["slug"].strip()

            for display_rank, candidate_id in enumerate(candidates, 1):
                candidate = by_id[candidate_id]
                common = {
                    "reference_id": pid,
                    "reference_name": ref["name"].strip(),
                    "reference_category": ref["category"].strip(),
                    "reference_url": ref_url,
                    "recommended_id": candidate_id,
                    "recommended_name": candidate["name"],
                    "recommended_category": candidate["category"],
                    "recommended_url": "http://localhost:3000/shop/" + candidate["slug"],
                    "relevance_0_1_2": "",
                    "notes": "",
                }
                union.append({
                    **common,
                    "v1_rank": str(v1_ranks.get(candidate_id, "")),
                    "v2_rank": str(v2_ranks.get(candidate_id, "")),
                })
                blind.append({**common, "rank": str(display_rank)})

        unique_ids = {r["reference_id"] for r in blind} | {
            r["recommended_id"] for r in blind
        }
        image_files = images.select_images(unique_ids)
        asset_names = {
            pid: (f"images/{pid}{path.suffix.lower()}" if path else None)
            for pid, path in image_files.items()
        }
        missing_images = sorted(pid for pid, p in image_files.items() if p is None)
        manifest = {
            "purpose": "Blind offline relevance review; images are private, do not publish",
            "reference_count": len(references),
            "v1_top5_pairs": len(references) * TOP_K,
            "v2_top5_pairs": len(references) * TOP_K,
            "top5_overlap": overlap_total,
            "candidate_union_pairs": len(union),
            "unique_products": len(unique_ids),
            "images_included": len(unique_ids) - len(missing_images),
            "images_missing": missing_images,
            "label_scale": {"0": "not similar", "1": "partially similar", "2": "highly similar"},
            "annotation": "Use blind_review.csv or review_index.html before viewing model ranks",
        }
        if len(union) != len(references) * (2 * TOP_K) - overlap_total:
            raise AssertionError("Union size is inconsistent with V1/V2 overlap")

        UNION_CSV.parent.mkdir(parents=True, exist_ok=True)
        BUNDLE_ZIP.parent.mkdir(parents=True, exist_ok=True)
        temp_csv = UNION_CSV.with_suffix(".csv.tmp")
        temp_zip = BUNDLE_ZIP.with_suffix(".zip.tmp")
        try:
            temp_csv.write_text(as_csv(union, COLUMNS), encoding="utf-8-sig")
            with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("blind_review.csv", as_csv(blind, BLIND_COLUMNS))
                archive.writestr("holdout_candidate_union.csv", as_csv(union, COLUMNS))
                archive.writestr("holdout_references.csv", as_csv(references, list(references[0])))
                page = images.create_html(blind, asset_names).replace(
                    "Top-5 suggested products", "Blinded candidate pool (V1 or V2)"
                )
                archive.writestr("review_index.html", page)
                archive.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
                for product_id in sorted(unique_ids):
                    image_file = image_files[product_id]
                    if image_file:
                        archive.write(image_file, asset_names[product_id])
            temp_csv.replace(UNION_CSV)
            temp_zip.replace(BUNDLE_ZIP)
        finally:
            temp_csv.unlink(missing_ok=True)
            temp_zip.unlink(missing_ok=True)

        print("Holdout V1/V2 export: COMPLETE")
        print("Reference products:", len(references))
        print("V1 Top-5 pairs:", len(references) * TOP_K)
        print("V2 Top-5 pairs:", len(references) * TOP_K)
        print("Top-5 overlap:", overlap_total)
        print("Union pairs needing independent labels:", len(union))
        print("Unique products in review:", len(unique_ids))
        print("Images in private bundle:", manifest["images_included"])
        print("Images missing:", missing_images)
        print("Evaluation CSV:", UNION_CSV)
        print("Private ZIP:", BUNDLE_ZIP)
        print("SQLite, holdout references, V1/V2 models and production API were NOT modified.")
        return 0
    except (OSError, ValueError, AssertionError, sqlite3.Error, zipfile.BadZipFile) as exc:
        print("ERROR:", exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
