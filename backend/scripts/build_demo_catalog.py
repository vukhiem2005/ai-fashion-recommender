r"""Build an independent, deterministic synthetic fashion catalog for ELAN.

PowerShell (from repository root):
    .\backend\.venv\Scripts\python.exe .\backend\scripts\build_demo_catalog.py --preview
    .\backend\.venv\Scripts\python.exe .\backend\scripts\build_demo_catalog.py --create

Only writes backend/data/elan_public_demo.db on --create. Never writes elan.db.
The public demo catalog is independent of the H&M dataset and AI holdout.
No third-party Python packages are required.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "backend" / "data" / "elan_public_demo.db"

SIZES = ("S", "M", "L", "XL")
ONE_SIZE = ("One Size",)
COLOR_CODES = {
    "Beige": "BGE", "Black": "BLK", "Blue": "BLU", "Brown": "BRN",
    "Burgundy": "BRG", "Charcoal": "CHR", "Cream": "CRM",
    "Denim Blue": "DBL", "Forest Green": "FGR", "Grey": "GRY",
    "Ivory": "IVR", "Khaki": "KHK", "Navy": "NVY", "Olive": "OLV",
    "Rust": "RST", "Sand": "SND", "Stone": "STN", "White": "WHT",
}


@dataclass(frozen=True)
class Seed:
    category: str
    name: str
    description: str
    price_cents: int
    colors: tuple[str, ...]
    sizes: tuple[str, ...] = SIZES


def items(category: str, entries: list[tuple[str, str, int, str, str]],
          sizes: tuple[str, ...] = SIZES) -> list[Seed]:
    return [
        Seed(category, name, description, price, (color_a, color_b), sizes)
        for name, description, price, color_a, color_b in entries
    ]


CATALOG = [
    *items("Outerwear", [
        ("Tailored City Blazer", "Structured woven blazer with notched lapels and a relaxed office-ready silhouette.", 7999, "Beige", "Navy"),
        ("Relaxed Linen Blazer", "Lightweight linen blend blazer with a soft unstructured shape for warm days.", 6999, "Sand", "Ivory"),
        ("Classic Belted Trench", "Long cotton-blend trench coat with a waist belt and storm flap for layered outfits.", 9999, "Khaki", "Black"),
        ("Cropped Denim Jacket", "Cropped denim jacket with classic seams, front pockets and a casual silhouette.", 5499, "Denim Blue", "Black"),
        ("Quilted Utility Jacket", "Lightly quilted utility jacket with patch pockets and practical everyday layering.", 7599, "Olive", "Charcoal"),
        ("Wool Blend Overcoat", "Longline wool blend overcoat with clean shoulders and a minimal winter profile.", 11999, "Grey", "Brown"),
        ("Minimal Bomber Jacket", "Clean zip-front bomber jacket with ribbed cuffs and a relaxed streetwear fit.", 6499, "Black", "Forest Green"),
        ("Soft Suede Overshirt", "Suede-look overshirt with a pointed collar and chest pockets for transitional layering.", 8499, "Brown", "Stone"),
    ]),
    *items("Tops", [
        ("Essential Cotton Tee", "Soft breathable cotton crewneck tee for versatile everyday outfits.", 1599, "White", "Black"),
        ("Ribbed Everyday Tee", "Fitted ribbed cotton tee with a close neckline and comfortable stretch.", 1899, "Ivory", "Grey"),
        ("Linen Button Shirt", "Airy linen button-down shirt with a relaxed collar and light summer texture.", 3299, "White", "Sand"),
        ("Oversized Poplin Shirt", "Crisp poplin shirt with a roomy fit, dropped shoulders and a long hem.", 3499, "Blue", "White"),
        ("Fine Knit Mock Neck Top", "Soft fine-gauge mock neck top designed for understated layered looks.", 2999, "Black", "Cream"),
        ("Satin Square Neck Camisole", "Light satin camisole with a square neckline and adjustable straps for evening styling.", 2599, "Burgundy", "Ivory"),
        ("Relaxed Breton Stripe Top", "Classic striped jersey top with long sleeves and a casual maritime look.", 2899, "Navy", "Cream"),
        ("Lightweight Oxford Shirt", "Textured oxford cotton shirt with button cuffs and a versatile tailored fit.", 3799, "White", "Blue"),
    ]),
    *items("Bottoms", [
        ("Straight Leg Denim Jeans", "Classic straight leg denim jeans with five pockets and everyday structure.", 4599, "Denim Blue", "Black"),
        ("Wide Leg Linen Trousers", "Breathable linen trousers with a flowing wide leg and soft waistband.", 4299, "Sand", "Black"),
        ("Pleated Tailored Pants", "Tailored trousers with front pleats and a clean full-length drape.", 4999, "Charcoal", "Navy"),
        ("Relaxed Cargo Trousers", "Cotton cargo trousers with roomy legs and side utility pockets.", 4499, "Olive", "Khaki"),
        ("Slim Ankle Chinos", "Stretch cotton chinos with a slim ankle shape for smart casual outfits.", 3899, "Navy", "Stone"),
        ("High Rise Wide Jeans", "High rise wide leg jeans with classic denim seams and a relaxed shape.", 4899, "Denim Blue", "Grey"),
        ("Denim Midi Skirt", "Structured denim midi skirt with a front slit and a clean everyday silhouette.", 3599, "Denim Blue", "Black"),
        ("Cotton Drawstring Shorts", "Casual cotton shorts with an adjustable drawstring and relaxed summer fit.", 2799, "Beige", "Black"),
    ]),
    *items("Dresses", [
        ("Minimal Midi Dress", "Clean minimal midi dress with a soft drape and timeless everyday fit.", 5499, "Black", "Navy"),
        ("Satin Wrap Dress", "Fluid satin wrap dress with a tied waist and soft evening silhouette.", 6599, "Burgundy", "Black"),
        ("Relaxed Shirt Dress", "Button-front shirt dress with a removable belt and easy daytime styling.", 5299, "Blue", "White"),
        ("Ribbed Knit Midi Dress", "Stretch ribbed knit midi dress with a fitted shape and round neckline.", 5799, "Cream", "Charcoal"),
        ("Linen Summer Dress", "Breathable sleeveless linen dress with a relaxed summer shape.", 4899, "Sand", "White"),
        ("Long Sleeve Shift Dress", "Simple long sleeve shift dress with a straight silhouette and subtle structure.", 5199, "Navy", "Olive"),
        ("Pleated Evening Dress", "Elegant pleated dress with a floating skirt and refined evening finish.", 7999, "Black", "Burgundy"),
        ("Cotton Poplin Dress", "Cotton poplin dress with a defined waist and full airy skirt.", 5599, "Ivory", "Blue"),
    ]),
    *items("Knitwear", [
        ("Soft Knit Cardigan", "Soft button-front cardigan with fine ribbing for effortless layering.", 4299, "Cream", "Grey"),
        ("Chunky Cable Sweater", "Cozy chunky cable knit sweater with a relaxed winter silhouette.", 5999, "Ivory", "Brown"),
        ("Lightweight V Neck Knit", "Fine lightweight v neck knit in soft yarn for transitional outfits.", 3899, "Navy", "Stone"),
        ("Relaxed Crewneck Pullover", "Relaxed crewneck pullover in a soft knit for everyday comfort.", 4499, "Beige", "Forest Green"),
        ("Ribbed Mock Neck Sweater", "Textured ribbed mock neck sweater with long sleeves and a fitted shape.", 4699, "Black", "Cream"),
        ("Cropped Button Cardigan", "Cropped button cardigan with soft knit texture and a classic neckline.", 4199, "Ivory", "Burgundy"),
        ("Textured Knit Vest", "Sleeveless textured knit vest for layered shirts and minimalist outfits.", 3599, "Olive", "Grey"),
        ("Fine Gauge Merino Sweater", "Fine gauge merino blend sweater with a clean slim profile and soft touch.", 6499, "Charcoal", "Navy"),
    ]),
    *items("Accessories", [
        ("Everyday Crossbody Bag", "Compact crossbody bag with an adjustable strap and simple everyday silhouette.", 3499, "Black", "Brown"),
        ("Minimal Canvas Tote", "Durable canvas tote bag with roomy interior and everyday utility.", 2499, "Ivory", "Olive"),
        ("Structured Shoulder Bag", "Structured shoulder bag with a curved handle and refined minimal look.", 4999, "Beige", "Black"),
        ("Classic Narrow Belt", "Minimal narrow belt with a polished metal buckle for tailored outfits.", 1999, "Brown", "Black"),
        ("Soft Woven Scarf", "Soft woven scarf with a light textured finish for cool-weather layering.", 2399, "Cream", "Grey"),
        ("Cotton Bucket Hat", "Casual cotton bucket hat with an easy soft brim for summer styling.", 1899, "Khaki", "Navy"),
        ("Everyday Knit Beanie", "Fine knit beanie with a folded cuff for understated winter warmth.", 1699, "Charcoal", "Burgundy"),
        ("Slim Card Holder", "Compact card holder with clean slots and a sleek everyday profile.", 2199, "Black", "Brown"),
    ], ONE_SIZE),
]


def slugify(name: str) -> str:
    return "demo-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def build_rows() -> tuple[list[tuple], list[tuple]]:
    products: list[tuple] = []
    variants: list[tuple] = []
    seen_slugs: set[str] = set()
    for index, seed in enumerate(CATALOG, start=1):
        pid = f"DEMO-{index:03d}"
        slug = slugify(seed.name)
        if slug in seen_slugs:
            raise ValueError(f"Duplicate slug: {slug}")
        seen_slugs.add(slug)
        if seed.category not in {"Outerwear", "Tops", "Bottoms", "Dresses", "Knitwear", "Accessories"}:
            raise ValueError(f"Invalid category: {seed.category}")
        if seed.price_cents <= 0 or not seed.description:
            raise ValueError(f"Invalid product attributes: {pid}")
        products.append((pid, slug, seed.name, seed.category, seed.description, seed.price_cents, 1))
        for color in seed.colors:
            code = COLOR_CODES[color]
            for size in seed.sizes:
                sku = f"{pid}-{code}-{size.upper().replace(' ', '-')}"
                variants.append((sku, pid, color, size, None))
    if len(products) != 48 or len({x[0] for x in variants}) != len(variants):
        raise ValueError("Expected 48 unique products and unique variant SKUs")
    return products, variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true", help="Show plan only (default)")
    mode.add_argument("--create", action="store_true", help="Create isolated demo SQLite; never overwrite")
    args = parser.parse_args()
    products, variants = build_rows()
    counts = Counter(x[3] for x in products)
    print("PUBLIC DEMO CATALOG")
    print("Target:", OUTPUT)
    print("Products:", len(products), "| Variants:", len(variants))
    for category in sorted(counts):
        print(f"  {category}: {counts[category]}")
    print("Source: original synthetic demo descriptions, no H&M files")

    if not args.create:
        print("PREVIEW complete: no files or databases changed")
        return

    if OUTPUT.exists():
        parser.error(f"Refusing to overwrite existing demo database: {OUTPUT}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        with closing(sqlite3.connect(OUTPUT)) as db:
            db.execute("PRAGMA foreign_keys=ON")
            db.executescript("""
                CREATE TABLE product (
                    id VARCHAR PRIMARY KEY NOT NULL,
                    slug VARCHAR NOT NULL,
                    name VARCHAR NOT NULL,
                    category VARCHAR NOT NULL,
                    description VARCHAR NOT NULL,
                    price_cents INTEGER NOT NULL,
                    is_active BOOLEAN NOT NULL
                );
                CREATE INDEX ix_product_slug ON product (slug);
                CREATE TABLE productvariant (
                    sku VARCHAR PRIMARY KEY NOT NULL,
                    product_id VARCHAR NOT NULL REFERENCES product(id),
                    color VARCHAR NOT NULL,
                    size VARCHAR NOT NULL,
                    image_url VARCHAR
                );
                CREATE INDEX ix_productvariant_product_id ON productvariant (product_id);
            """)
            db.executemany("INSERT INTO product VALUES (?, ?, ?, ?, ?, ?, ?)", products)
            db.executemany("INSERT INTO productvariant VALUES (?, ?, ?, ?, ?)", variants)
            actual = db.execute("SELECT COUNT(*) FROM product").fetchone()[0]
            if actual != len(products):
                raise RuntimeError(f"Row count mismatch: {actual}")
            errors = db.execute("PRAGMA foreign_key_check").fetchall()
            if errors:
                raise RuntimeError(f"Foreign key errors: {errors}")
            db.commit()
    except Exception:
        OUTPUT.unlink(missing_ok=True)
        raise
    print("CREATE complete: isolated demo catalog built")
    print("Original backend/data/elan.db NOT MODIFIED")
    print("All product image_url values are NULL (placeholders until assets are prepared)")
    print("Production API still uses elan.db until an explicit database switch is implemented")


if __name__ == "__main__":
    main()
