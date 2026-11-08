r"""Create a private, offline image-review bundle for ELAN relevance labels.

Run from any directory:
    .\backend\.venv\Scripts\python.exe backend\evaluation\prepare_relevance_review.py

Uses only Python's standard library. Never changes the evaluation CSV or SQLite.
The output is placed in the Git-ignored data/private/hm directory.
"""

from __future__ import annotations

import csv
import html
import io
import json
import re
import sqlite3
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
CSV_FILE = ROOT / "backend" / "evaluation" / "relevance_labels_v1.csv"
DATABASE = ROOT / "backend" / "data" / "elan.db"
PUBLIC = (ROOT / "frontend" / "public").resolve()
OUTPUT = ROOT / "data" / "private" / "hm" / "relevance_review_bundle.zip"
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def load_pairs() -> list[dict[str, str]]:
    if not CSV_FILE.is_file():
        raise FileNotFoundError(f"Missing evaluation CSV: {CSV_FILE}")

    with CSV_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    required = {"reference_id", "reference_name", "reference_category", "rank",
                "recommended_id", "recommended_name", "recommended_category"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Evaluation CSV is empty or missing required columns")

    for row in rows:
        for key in ("reference_id", "recommended_id"):
            if not ID_PATTERN.fullmatch(row[key].strip()):
                raise ValueError(f"Invalid {key}: {row[key]!r}")
    return rows


def resolve_image(url: str | None) -> Path | None:
    if not url or not url.strip():
        return None
    parts = urlsplit(url.strip())
    if parts.scheme or parts.netloc or not parts.path.startswith("/"):
        return None  # Only package local public assets
    target = (PUBLIC / unquote(parts.path).lstrip("/")).resolve()
    try:
        target.relative_to(PUBLIC)
    except ValueError:
        return None
    if target.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    return target if target.is_file() and target.stat().st_size else None


def select_images(ids: set[str]) -> dict[str, Path | None]:
    if not DATABASE.is_file():
        raise FileNotFoundError(f"Missing SQLite database: {DATABASE}")
    images: dict[str, Path | None] = {product_id: None for product_id in ids}
    placeholders = ",".join("?" for _ in ids)
    with sqlite3.connect(DATABASE) as conn:
        candidates = conn.execute(
            f"""SELECT product_id, sku, image_url
                FROM productvariant
                WHERE product_id IN ({placeholders})
                ORDER BY product_id, sku""",
            sorted(ids),
        ).fetchall()
    for product_id, _sku, url in candidates:
        if images[product_id] is None:
            images[product_id] = resolve_image(url)
    return images


def create_html(rows: list[dict[str, str]], asset_paths: dict[str, str | None]) -> str:
    grouped: dict[str, list[dict[str, str]]] = OrderedDict()
    for row in rows:
        grouped.setdefault(row["reference_id"], []).append(row)

    def esc(value: str) -> str:
        return html.escape(str(value), quote=True)

    def picture(product_id: str) -> str:
        src = asset_paths.get(product_id)
        if src:
            return f'<img src="{esc(src)}" alt="{esc(product_id)}" loading="lazy">'
        return '<div class="missing">Image unavailable locally</div>'

    blocks = []
    for reference_id, results in grouped.items():
        ref = results[0]
        cards = []
        for item in results:
            candidate_id = item["recommended_id"]
            cards.append(
                '<div class="card">'
                f'{picture(candidate_id)}'
                f'<div class="caption"><b>#{esc(item["rank"])} — {esc(item["recommended_name"])}</b>'
                f'<br><small>{esc(candidate_id)} · {esc(item["recommended_category"])}</small>'
                f'<br><small>Current label: {esc(item.get("relevance_0_1_2") or "unlabeled")}</small></div>'
                '</div>'
            )
        blocks.append(
            '<section>'
            f'<h2>{esc(ref["reference_category"])} · {esc(ref["reference_name"])}</h2>'
            f'<p class="meta">Reference: {esc(reference_id)}</p>'
            f'<div class="reference">{picture(reference_id)}</div>'
            '<h3>Top-5 suggested products</h3>'
            f'<div class="cards">{"".join(cards)}</div>'
            '</section>'
        )

    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>ÉLAN Recommendation Relevance Review</title>'
            '<style>body{margin:24px auto;max-width:1140px;padding:0 18px;font:15px/1.5 Arial,sans-serif;color:#29231d;background:#f9f6f1}'
            'h1{font-family:Georgia,serif;font-size:32px}h2{font-family:Georgia,serif;font-size:25px;margin-bottom:2px}'
            'h3{font-size:17px}section{border-top:2px solid #ccc3b8;margin-top:35px;padding-top:18px}'
            '.meta{color:#777}.reference img{width:220px;max-height:290px;object-fit:contain;background:white}'
            '.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:16px}'
            '.card{background:white;border:1px solid #e2dcd5;padding:8px}'
            '.card img{width:100%;height:210px;object-fit:contain}'
            '.caption{padding-top:8px}small{color:#6f6a64;word-break:break-word}'
            '.missing{height:180px;display:flex;align-items:center;justify-content:center;background:#eee6db;color:#846b55;padding:10px;text-align:center}'
            '</style></head><body><h1>ÉLAN · Recommendation Review</h1>'
            '<p>Private image-review bundle. Labels are 0 (not a substitute), 1 (partly similar), 2 (highly similar). '
            'Compare each candidate with the reference image above it. These are local catalog assets, not web images.</p>'
            f'{"".join(blocks)}</body></html>')


def main() -> int:
    try:
        rows = load_pairs()
        ids = {r["reference_id"] for r in rows} | {r["recommended_id"] for r in rows}
        images = select_images(ids)

        asset_paths: dict[str, str | None] = {}
        for product_id, image_file in images.items():
            asset_paths[product_id] = (
                f"images/{product_id}{image_file.suffix.lower()}" if image_file else None
            )

        manifest = {
            "purpose": "Private visual relevance assessment for TF-IDF recommender",
            "reference_count": len({row["reference_id"] for row in rows}),
            "pair_count": len(rows),
            "images_found": sum(path is not None for path in images.values()),
            "images_missing": sorted(pid for pid, path in images.items() if path is None),
            "product_images": asset_paths,
            "pairs": rows,
        }

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.write(CSV_FILE, "relevance_labels_v1.csv")
            bundle.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            bundle.writestr("review_index.html", create_html(rows, asset_paths))
            for product_id in sorted(ids):
                image_file = images[product_id]
                if image_file:
                    bundle.write(image_file, asset_paths[product_id])

        print(f"Output: {OUTPUT}")
        print(f"Reference products: {manifest['reference_count']}")
        print(f"Evaluation pairs: {len(rows)}")
        print(f"Unique product IDs: {len(ids)}")
        print(f"Images included: {manifest['images_found']}")
        print(f"Images unavailable: {len(manifest['images_missing'])}")
        for pid in manifest["images_missing"]:
            print(f"  MISSING: {pid}")
        print(f"ZIP size: {OUTPUT.stat().st_size / 1024 / 1024:.2f} MB")
        print("SQLite and evaluation CSV were NOT modified.")
        print("Do not commit or publish this private image bundle to GitHub.")
        return 0
    except (OSError, ValueError, sqlite3.Error, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
