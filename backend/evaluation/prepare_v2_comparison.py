r"""Prepare a fair, offline V1/V2 candidate-pool review for ELAN.

Run from the repository root (Windows PowerShell):
    .\backend\.venv\Scripts\python.exe .\backend\evaluation\prepare_v2_comparison.py

Inputs:
    backend/evaluation/relevance_labels_v1.csv  (30 labeled baseline pairs)
    backend/data/elan.db                         (read-only)
    backend/app/recommender.py                   (stable V1)
    backend/app/recommender_v2.py                (experimental V2)
    backend/evaluation/prepare_relevance_review.py (reusable image helpers)

Outputs:
    backend/evaluation/v1_v2_candidate_union.csv (all pairs; original labels preserved)
    data/private/hm/v2_new_pairs_review_bundle.zip (private, only new candidate images)

Does not overwrite existing outputs, modify V1 labels, modify SQLite, or deploy V2.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path

# Allow direct execution from the repository root, not only `python -m`.
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.recommender import recommend_products
from app.recommender_v2 import recommend_products_v2
from evaluation import prepare_relevance_review as images

ROOT = Path(__file__).resolve().parents[2]
LABEL_CSV = ROOT / "backend" / "evaluation" / "relevance_labels_v1.csv"
UNION_CSV = ROOT / "backend" / "evaluation" / "v1_v2_candidate_union.csv"
BUNDLE_ZIP = ROOT / "data" / "private" / "hm" / "v2_new_pairs_review_bundle.zip"

COLUMNS = [
    "reference_id", "reference_name", "reference_category", "reference_url",
    "v1_rank", "v2_rank", "recommended_id", "recommended_name",
    "recommended_category", "recommended_url", "relevance_0_1_2", "notes",
    "label_source",
]


def _as_csv(rows: list[dict[str, str]], fieldnames: list[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def _load_baseline() -> OrderedDict[str, list[dict[str, str]]]:
    if not LABEL_CSV.is_file():
        raise FileNotFoundError(f"Baseline labels not found: {LABEL_CSV}")
    with LABEL_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 30:
        raise ValueError(f"Expected 30 baseline pairs, got {len(rows)}")
    required = {"reference_id", "reference_name", "reference_category",
                "rank", "recommended_id", "recommended_name",
                "recommended_category", "relevance_0_1_2", "notes"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Baseline CSV does not contain required columns")

    grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for row in rows:
        if row["relevance_0_1_2"].strip() not in {"0", "1", "2"}:
            raise ValueError(f"Missing/invalid baseline label: {row['reference_id']}, {row['recommended_id']}")
        grouped.setdefault(row["reference_id"], []).append(row)
    if len(grouped) != 6:
        raise ValueError(f"Expected 6 reference products, got {len(grouped)}")
    for reference_id, group in grouped.items():
        if len(group) != 5:
            raise ValueError(f"Expected 5 baseline results for {reference_id}, got {len(group)}")
        group.sort(key=lambda row: int(row["rank"]))
        if [int(row["rank"]) for row in group] != [1, 2, 3, 4, 5]:
            raise ValueError(f"Invalid baseline ranks for {reference_id}")
    return grouped


def main() -> int:
    try:
        baseline = _load_baseline()
        if UNION_CSV.exists() or BUNDLE_ZIP.exists():
            raise FileExistsError(
                "Output already exists. Review it before rerunning; nothing was overwritten: "
                + ", ".join(str(p) for p in (UNION_CSV, BUNDLE_ZIP) if p.exists())
            )

        union_rows: list[dict[str, str]] = []
        review_rows: list[dict[str, str]] = []
        overlap_total = 0

        for reference_id, old in baseline.items():
            # Guard against changes in data/model since V1 labels were created.
            current_v1 = recommend_products(reference_id, top_k=5)
            original_ids = [row["recommended_id"] for row in old]
            current_ids = [row["id"] for row in current_v1]
            if current_ids != original_ids:
                raise ValueError(
                    f"V1 Top-5 changed for {reference_id}; do not reuse old labels blindly."
                    f"\nSaved: {original_ids}\nCurrent: {current_ids}"
                )

            current_v2 = recommend_products_v2(reference_id, top_k=5)
            v2_by_id = {row["id"]: (rank, row) for rank, row in enumerate(current_v2, 1)}
            old_by_id = {row["recommended_id"]: row for row in old}
            overlap_total += len(set(original_ids) & set(v2_by_id))
            first = old[0]
            # Existing five baseline results, followed by V2-only candidates.
            ordered_ids = original_ids + [p["id"] for p in current_v2 if p["id"] not in old_by_id]

            for candidate_id in ordered_ids:
                old_pair = old_by_id.get(candidate_id)
                v2_rank, v2_product = v2_by_id.get(candidate_id, (None, None))
                name = old_pair["recommended_name"] if old_pair else v2_product["name"]
                category = old_pair["recommended_category"] if old_pair else v2_product["category"]
                slug = v2_product["slug"] if v2_product else None
                # Existing baseline URLs are preserved; V2-only URL comes from live SQLite data.
                url = (
                    old_pair.get("recommended_url", "") if old_pair else
                    "http://localhost:3000/shop/" + slug
                )
                row = {
                    "reference_id": reference_id,
                    "reference_name": first["reference_name"],
                    "reference_category": first["reference_category"],
                    "reference_url": first.get("reference_url", ""),
                    "v1_rank": old_pair["rank"] if old_pair else "",
                    "v2_rank": str(v2_rank) if v2_rank else "",
                    "recommended_id": candidate_id,
                    "recommended_name": name,
                    "recommended_category": category,
                    "recommended_url": url,
                    "relevance_0_1_2": old_pair["relevance_0_1_2"] if old_pair else "",
                    "notes": old_pair.get("notes", "") if old_pair else "",
                    "label_source": "v1_preserved" if old_pair else "v2_new_unrated",
                }
                union_rows.append(row)
                if not old_pair:
                    review_rows.append({
                        "reference_id": reference_id,
                        "reference_name": first["reference_name"],
                        "reference_category": first["reference_category"],
                        "reference_url": first.get("reference_url", ""),
                        "rank": str(v2_rank),
                        "recommended_id": candidate_id,
                        "recommended_name": name,
                        "recommended_category": category,
                        "recommended_url": url,
                        "relevance_0_1_2": "",
                        "notes": "",
                    })

        if len(union_rows) != 60 - overlap_total or len(review_rows) != 30 - overlap_total:
            raise AssertionError("Unexpected candidate-union count")
        if not review_rows:
            raise ValueError("V2 has no new candidates to review; all Top-5 pairs are already labeled")

        ids = {row["reference_id"] for row in review_rows} | {
            row["recommended_id"] for row in review_rows
        }
        found_images = images.select_images(ids)
        paths: dict[str, str | None] = {}
        for pid, image_path in found_images.items():
            paths[pid] = f"images/{pid}{image_path.suffix.lower()}" if image_path else None
        manifest = {
            "purpose": "Offline review of V2-only candidates; do not publish the image bundle",
            "baseline_pairs_labeled": 30,
            "v1_v2_top5_overlap": overlap_total,
            "new_unrated_pairs": len(review_rows),
            "candidate_union_pairs": len(union_rows),
            "reference_count": len(baseline),
            "images_found": sum(p is not None for p in found_images.values()),
            "images_missing": sorted(pid for pid, image_path in found_images.items() if image_path is None),
            "pairs": review_rows,
        }
        UNION_CSV.parent.mkdir(parents=True, exist_ok=True)
        BUNDLE_ZIP.parent.mkdir(parents=True, exist_ok=True)
        # Avoid leaving partially written outputs if ZIP creation fails.
        temp_csv = UNION_CSV.with_suffix(".csv.tmp")
        temp_zip = BUNDLE_ZIP.with_suffix(".zip.tmp")
        try:
            temp_csv.write_text(_as_csv(union_rows, COLUMNS), encoding="utf-8-sig")
            with zipfile.ZipFile(temp_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr("v2_new_pairs.csv", _as_csv(review_rows, list(review_rows[0].keys())))
                bundle.writestr("review_index.html", images.create_html(review_rows, paths))
                bundle.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
                for pid in sorted(ids):
                    if found_images[pid] is not None:
                        bundle.write(found_images[pid], paths[pid])
            temp_csv.replace(UNION_CSV)
            temp_zip.replace(BUNDLE_ZIP)
        finally:
            temp_csv.unlink(missing_ok=True)
            temp_zip.unlink(missing_ok=True)

        print("V1/V2 candidate comparison: COMPLETE")
        print("Reference products:", len(baseline))
        print("V1 labeled pairs retained: 30")
        print("Top-5 overlap (V1 vs V2):", f"{overlap_total}/30")
        print("V2-only pairs requiring labels:", len(review_rows))
        print("Union pairs (for comparison):", len(union_rows))
        print("Images in private bundle:", manifest["images_found"])
        print("Images missing from private bundle:", manifest["images_missing"])
        print("Comparison CSV:", UNION_CSV)
        print("Private image ZIP:", BUNDLE_ZIP)
        print("SQLite, V1 label CSV, production API, and models were NOT modified.")
        return 0
    except (OSError, ValueError, AssertionError, sqlite3.Error, zipfile.BadZipFile) as error:
        print("ERROR:", error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
