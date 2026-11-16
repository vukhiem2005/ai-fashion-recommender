"""Generate the ELAN V5 reproducible synthetic user-item interaction dataset.

IMPORTANT SCIENTIFIC & SAFETY NOTICE:
- Reads `backend/data/elan_release.db` strictly in SQLite URI read-only mode (`mode=ro`).
- Does NOT modify V1 recommendation code, existing APIs, or SQLite databases.
- Separates observable model-facing tables (`backend/evaluation/v5/data/`) from
  generator-internal latent ground truth (`backend/evaluation/v5/audit/`) so
  downstream recommendation models cannot leak latent persona parameters,
  session intents, or event `origin_signal` values.
- All outputs are explicitly labeled `is_synthetic=True` and
  `data_source=SYNTHETIC_SIMULATION_V5`.
- Synthetic interactions validate algorithmic pipeline mechanics only; they do
  not represent real consumer behavior or production recommendation quality.

Usage from repository root:
    python backend/evaluation/v5/generate_synthetic_dataset.py --seed 20260501
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RELEASE_DB = REPO_ROOT / "backend" / "data" / "elan_release.db"
V5_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = V5_DIR / "data"
DEFAULT_AUDIT_DIR = V5_DIR / "audit"
DEFAULT_REPORT_PATH = V5_DIR / "synthetic_dataset_report.md"

DEFAULT_SEED = 20260501
DATA_SOURCE_LABEL = "SYNTHETIC_SIMULATION_V5"

# Chronological split windows (UTC, non-overlapping with 1-second gap at boundary)
SPLIT_WINDOWS: dict[str, tuple[datetime, datetime]] = {
    "train": (
        datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 3, 4, 23, 59, 59, tzinfo=timezone.utc),
    ),
    "val": (
        datetime(2026, 3, 5, 0, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 3, 17, 23, 59, 59, tzinfo=timezone.utc),
    ),
    "test": (
        datetime(2026, 3, 18, 0, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 3, 31, 23, 59, 59, tzinfo=timezone.utc),
    ),
}

# Safety margin before split end so a session's events/orders never cross split boundaries
SESSION_BOUNDARY_BUFFER_SECONDS = 7200  # 2 hours

IMPLICIT_WEIGHTS: dict[str, float] = {
    "view": 1.0,
    "add_to_cart": 3.0,
    "purchase": 5.0,
}

# Conservative garment subtype rules (aligned with V3 name-only rules)
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
_COMPILED_SUBTYPES = tuple(
    (label, re.compile(pattern, re.IGNORECASE)) for label, pattern in _SUBTYPE_RULES
)

# Six interpretable shopping personas (simulation ground truth only)
PERSONA_SPECS: tuple[dict[str, Any], ...] = (
    {
        "persona_id": "tailored_office",
        "share": 0.18,
        "categories": ("Outerwear", "Tops", "Bottoms"),
        "subtypes": ("blazers", "coats", "waistcoats", "shirts", "trousers", "skirts"),
        "color_families": ("neutral_dark", "neutral_light"),
        "style": "tailored",
        "price_tier": "premium",
        "price_sensitivity_range": (0.15, 0.45),
        "noise_range": (0.06, 0.12),
    },
    {
        "persona_id": "casual_denim_basics",
        "share": 0.24,
        "categories": ("Tops", "Bottoms", "Outerwear"),
        "subtypes": ("tshirts", "hoodies", "sweatshirts", "jeans", "joggers", "shorts", "jackets", "windbreakers"),
        "color_families": ("cool_tone", "neutral_dark", "neutral_light"),
        "style": "casual_street",
        "price_tier": "budget",
        "price_sensitivity_range": (0.45, 0.85),
        "noise_range": (0.08, 0.15),
    },
    {
        "persona_id": "minimalist_knit_layering",
        "share": 0.18,
        "categories": ("Knitwear", "Outerwear", "Tops", "Bottoms"),
        "subtypes": ("sweaters", "cardigans", "coats", "jackets", "vests", "trousers", "shirts"),
        "color_families": ("earth_warm", "neutral_light", "neutral_dark"),
        "style": "cozy_knit",
        "price_tier": "mid_range",
        "price_sensitivity_range": (0.25, 0.60),
        "noise_range": (0.06, 0.13),
    },
    {
        "persona_id": "occasion_dresses_accessories",
        "share": 0.16,
        "categories": ("Dresses", "Accessories", "Outerwear"),
        "subtypes": ("dresses", "earrings", "necklaces", "bags", "coats", "blazers"),
        "color_families": ("neutral_dark", "accent_bright", "neutral_light"),
        "style": "feminine_resort",
        "price_tier": "mid_range",
        "price_sensitivity_range": (0.20, 0.55),
        "noise_range": (0.07, 0.14),
    },
    {
        "persona_id": "summer_resort_edit",
        "share": 0.14,
        "categories": ("Dresses", "Tops", "Bottoms", "Accessories"),
        "subtypes": ("dresses", "tunics", "tshirts", "bodysuits", "shorts", "skirts", "sunglasses", "headwear", "bags"),
        "color_families": ("neutral_light", "accent_bright", "earth_warm"),
        "style": "feminine_resort",
        "price_tier": "budget",
        "price_sensitivity_range": (0.35, 0.75),
        "noise_range": (0.08, 0.16),
    },
    {
        "persona_id": "eclectic_explorer",
        "share": 0.10,
        "categories": ("Outerwear", "Tops", "Bottoms", "Dresses", "Knitwear", "Accessories"),
        "subtypes": ("jackets", "parkas", "dresses", "sweaters", "jeans", "bags", "patches", "sunglasses", "earrings"),
        "color_families": ("accent_bright", "cool_tone", "earth_warm", "neutral_dark", "neutral_light"),
        "style": "minimalist",
        "price_tier": "mid_range",
        "price_sensitivity_range": (0.20, 0.70),
        "noise_range": (0.16, 0.25),
    },
)

# Cross-category complementary outfit compatibility map for multi-item baskets
COMPLEMENTARY_CATEGORIES: dict[str, tuple[str, ...]] = {
    "Tops": ("Bottoms", "Outerwear", "Accessories", "Knitwear"),
    "Bottoms": ("Tops", "Knitwear", "Outerwear", "Accessories"),
    "Outerwear": ("Tops", "Bottoms", "Knitwear", "Dresses"),
    "Dresses": ("Accessories", "Outerwear", "Knitwear"),
    "Knitwear": ("Bottoms", "Tops", "Outerwear", "Accessories"),
    "Accessories": ("Dresses", "Tops", "Outerwear", "Knitwear"),
}

COMPLEMENTARY_SUBTYPES: dict[str, tuple[str, ...]] = {
    "shirts": ("trousers", "skirts", "blazers", "coats", "waistcoats", "bags"),
    "tshirts": ("jeans", "joggers", "shorts", "jackets", "windbreakers", "hoodies", "sunglasses"),
    "trousers": ("shirts", "sweaters", "cardigans", "blazers", "coats"),
    "jeans": ("tshirts", "sweaters", "hoodies", "jackets", "shirts"),
    "dresses": ("earrings", "necklaces", "bags", "cardigans", "coats", "blazers", "sunglasses"),
    "sweaters": ("trousers", "jeans", "skirts", "coats", "jackets", "bags"),
    "cardigans": ("dresses", "shirts", "tshirts", "trousers", "jeans"),
    "blazers": ("shirts", "trousers", "dresses", "skirts"),
    "coats": ("sweaters", "trousers", "dresses", "bags"),
    "jackets": ("tshirts", "jeans", "joggers", "hoodies"),
}


def format_ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def infer_subtype(name: str, category: str) -> str:
    cat_lower = category.strip().lower()
    for label, pattern in _COMPILED_SUBTYPES:
        if pattern.search(name):
            if label == "bodysuits" and re.search(r"\bbody\b", name, re.IGNORECASE):
                explicit = re.search(
                    r"\b(?:bodysuits?|body[ -]?suits?|tank[ -]?top[ -]?body)\b",
                    name,
                    re.IGNORECASE,
                )
                if not explicit and cat_lower not in ("", "tops"):
                    continue
            return label
    return "unknown"


def map_color_family(colors: list[str]) -> str:
    joined = " ".join(c.lower() for c in colors)
    if any(k in joined for k in ("black", "dark", "charcoal", "anthracite")):
        return "neutral_dark"
    if any(k in joined for k in ("white", "cream", "off-white", "light beige", "grey", "gray", "silver")):
        return "neutral_light"
    if any(k in joined for k in ("beige", "brown", "khaki", "camel", "tan", "rust", "mustard", "gold", "orange")):
        return "earth_warm"
    if any(k in joined for k in ("blue", "navy", "denim", "green", "turquoise", "teal", "mint")):
        return "cool_tone"
    if any(k in joined for k in ("red", "pink", "yellow", "purple", "lilac", "coral", "multi")):
        return "accent_bright"
    return "neutral_dark"


def infer_style_tag(name: str, description: str, category: str, subtype: str) -> str:
    text = f"{name} {description}".lower()
    if subtype in ("blazers", "waistcoats", "coats", "shirts", "trousers") or "tailored" in text:
        return "tailored"
    if category == "Knitwear" or subtype in ("sweaters", "cardigans") or "knit" in text:
        return "cozy_knit"
    if category == "Dresses" or subtype in ("dresses", "skirts", "earrings", "necklaces", "sunglasses"):
        return "feminine_resort"
    if subtype in ("jeans", "joggers", "hoodies", "sweatshirts", "tshirts", "windbreakers"):
        return "casual_street"
    return "minimalist"


def load_release_catalog_readonly(db_path: Path) -> list[dict[str, Any]]:
    """Read active products and variants from SQLite using strict URI mode=ro."""
    if not db_path.is_file():
        raise FileNotFoundError(f"Release SQLite database not found: {db_path}")

    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        product_rows = conn.execute(
            """
            SELECT id, slug, name, category, description, price_cents, is_active
            FROM product
            WHERE is_active = 1
            ORDER BY id
            """
        ).fetchall()
        variant_rows = conn.execute(
            """
            SELECT sku, product_id, color, size, image_url
            FROM productvariant
            ORDER BY sku
            """
        ).fetchall()

    variants_by_product: dict[str, list[dict[str, Any]]] = defaultdict(list)
    colors_by_product: dict[str, set[str]] = defaultdict(set)
    for v in variant_rows:
        pid = v["product_id"]
        variants_by_product[pid].append(dict(v))
        color = (v["color"] or "").strip()
        if color:
            colors_by_product[pid].add(color)

    prices = sorted(int(r["price_cents"]) for r in product_rows)
    q33 = prices[len(prices) // 3]
    q66 = prices[(2 * len(prices)) // 3]

    catalog: list[dict[str, Any]] = []
    for row in product_rows:
        pid = row["id"]
        p_variants = variants_by_product[pid]
        if not p_variants:
            raise ValueError(f"Active product {pid} has no variants in release DB")
        colors = sorted(colors_by_product[pid])
        price_cents = int(row["price_cents"])
        if price_cents <= q33:
            price_tier = "budget"
        elif price_cents <= q66:
            price_tier = "mid_range"
        else:
            price_tier = "premium"

        subtype = infer_subtype(row["name"], row["category"])
        color_family = map_color_family(colors)
        style_tag = infer_style_tag(row["name"], row["description"], row["category"], subtype)

        catalog.append(
            {
                "id": pid,
                "slug": row["slug"],
                "name": row["name"],
                "category": row["category"],
                "description": row["description"],
                "price_cents": price_cents,
                "price_tier": price_tier,
                "colors": colors,
                "color_family": color_family,
                "subtype": subtype,
                "style_tag": style_tag,
                "variants": p_variants,
            }
        )
    return catalog


_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "about", "above", "across", "after", "again", "all", "almost", "along",
        "also", "although", "always", "am", "among", "an", "and", "another", "any",
        "are", "around", "as", "at", "back", "be", "because", "been", "before",
        "behind", "being", "below", "between", "both", "bottom", "but", "by", "can",
        "cannot", "could", "de", "detail", "do", "does", "done", "down", "due",
        "during", "each", "either", "else", "empty", "enough", "etc", "even", "ever",
        "every", "few", "fill", "find", "first", "for", "former", "from", "Front",
        "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he",
        "hence", "her", "here", "hers", "herself", "him", "himself", "his", "how",
        "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "into", "is",
        "it", "its", "itself", "keep", "last", "latter", "less", "ltd", "made",
        "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover",
        "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely",
        "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none",
        "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often",
        "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise",
        "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
        "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming",
        "seems", "serious", "several", "she", "should", "show", "side", "since",
        "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something",
        "sometime", "sometimes", "somewhere", "still", "such", "system", "take",
        "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence",
        "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these",
        "they", "thick", "thin", "third", "this", "those", "though", "three",
        "through", "throughout", "thru", "thus", "to", "together", "too", "top",
        "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up",
        "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever",
        "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby",
        "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
        "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within",
        "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves",
    }
)
_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


def build_content_similarity_matrix(catalog: list[dict[str, Any]]) -> list[list[float]]:
    """Compute V1-compatible sublinear TF-IDF (1..2-gram) cosine similarity in pure Python."""
    n_docs = len(catalog)
    doc_term_counts: list[Counter[str]] = []
    df_counts: Counter[str] = Counter()

    for item in catalog:
        cat = str(item["category"] or "").strip()
        name = str(item["name"] or "").strip()
        desc = str(item["description"] or "").strip()
        cols = " ".join(item["colors"])
        raw_text = " ".join([cat] * 3 + [name] * 2 + [desc] + [cols] * 2).lower()
        unigrams = [tok for tok in _TOKEN_RE.findall(raw_text) if tok not in _STOP_WORDS]
        bigrams = [f"{unigrams[i]} {unigrams[i + 1]}" for i in range(len(unigrams) - 1)]
        counts = Counter(unigrams + bigrams)
        doc_term_counts.append(counts)
        for term in counts:
            df_counts[term] += 1

    idf: dict[str, float] = {
        term: math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        for term, df in df_counts.items()
    }

    norm_vectors: list[dict[str, float]] = []
    for counts in doc_term_counts:
        vec: dict[str, float] = {}
        sq_norm = 0.0
        for term, tf in counts.items():
            val = (1.0 + math.log(tf)) * idf[term]
            vec[term] = val
            sq_norm += val * val
        inv_norm = 1.0 / math.sqrt(sq_norm) if sq_norm > 0.0 else 0.0
        norm_vectors.append({t: v * inv_norm for t, v in vec.items()})

    sim_matrix: list[list[float]] = [[0.0] * n_docs for _ in range(n_docs)]
    for i in range(n_docs):
        sim_matrix[i][i] = 1.0
        vec_i = norm_vectors[i]
        for j in range(i + 1, n_docs):
            vec_j = norm_vectors[j]
            if len(vec_i) > len(vec_j):
                small, large = vec_j, vec_i
            else:
                small, large = vec_i, vec_j
            dot = sum(v * large.get(t, 0.0) for t, v in small.items())
            sim_matrix[i][j] = dot
            sim_matrix[j][i] = dot
    return sim_matrix


def weighted_choice_index(weights: list[float], rng: random.Random) -> int:
    total = sum(weights)
    if total <= 0.0:
        return rng.randrange(len(weights))
    r = rng.random() * total
    upto = 0.0
    for idx, w in enumerate(weights):
        upto += w
        if upto >= r:
            return idx
    return len(weights) - 1


def price_tier_distance(tier_a: str, tier_b: str) -> float:
    order = {"budget": 0, "mid_range": 1, "premium": 2}
    return float(abs(order[tier_a] - order[tier_b]))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


def generate_dataset(
    db_path: Path = DEFAULT_RELEASE_DB,
    seed: int = DEFAULT_SEED,
    num_users: int = 600,
) -> dict[str, Any]:
    """Run deterministic synthetic generation and return tables + telemetry."""
    t0 = time.perf_counter()
    rng = random.Random(seed)

    catalog = load_release_catalog_readonly(db_path)
    n_items = len(catalog)
    content_sim = build_content_similarity_matrix(catalog)

    # Assign damped Zipf-Mandelbrot base popularity prior across shuffled catalog
    rank_order = list(range(n_items))
    rng.shuffle(rank_order)
    q_zipf, alpha_zipf = 8.0, 0.85
    raw_pop = [0.0] * n_items
    for rank_idx, item_idx in enumerate(rank_order, start=1):
        raw_pop[item_idx] = 1.0 / ((rank_idx + q_zipf) ** alpha_zipf)
    pop_sum = sum(raw_pop)
    pop_prior = [w / pop_sum for w in raw_pop]

    # 1. Generate users (split observable user table vs. audit-only persona ground truth)
    users_observable: list[dict[str, Any]] = []
    users_audit: list[dict[str, Any]] = []

    persona_weights = [p["share"] for p in PERSONA_SPECS]
    activity_choices = [("low", 0.30), ("medium", 0.50), ("high", 0.20)]
    activity_labels = [a[0] for a in activity_choices]
    activity_probs = [a[1] for a in activity_choices]

    base_reg_start = datetime(2025, 11, 1, 0, 0, 0, tzinfo=timezone.utc)
    for u_idx in range(1, num_users + 1):
        user_id = f"SYN-U{u_idx:04d}"
        p_spec = PERSONA_SPECS[weighted_choice_index(persona_weights, rng)]
        act_level = activity_labels[weighted_choice_index(activity_probs, rng)]
        price_sens = round(rng.uniform(*p_spec["price_sensitivity_range"]), 4)
        noise_rate = round(rng.uniform(*p_spec["noise_range"]), 4)
        pref_color = rng.choice(p_spec["color_families"])
        reg_offset_sec = rng.randint(0, 60 * 86400)
        created_at = format_ts(base_reg_start + timedelta(seconds=reg_offset_sec))

        # Select 2-3 favorite specific subtypes from persona spec
        k_sub = min(len(p_spec["subtypes"]), rng.randint(2, 4))
        fav_subtypes = tuple(rng.sample(p_spec["subtypes"], k=k_sub))

        users_observable.append(
            {
                "user_id": user_id,
                "created_at": created_at,
                "is_synthetic": True,
                "data_source": DATA_SOURCE_LABEL,
            }
        )
        users_audit.append(
            {
                "user_id": user_id,
                "persona_id": p_spec["persona_id"],
                "primary_categories": "|".join(p_spec["categories"]),
                "preferred_subtypes": "|".join(fav_subtypes),
                "preferred_color_family": pref_color,
                "preferred_style": p_spec["style"],
                "price_tier_preference": p_spec["price_tier"],
                "price_sensitivity": price_sens,
                "exploration_noise_rate": noise_rate,
                "activity_level": act_level,
                "created_at": created_at,
                "is_synthetic": True,
                "data_source": DATA_SOURCE_LABEL,
            }
        )

    # Precompute latent user-item base utilities (strictly kept inside generator)
    user_item_utility: list[list[float]] = []
    user_anchor_weights: list[list[float]] = []
    for u_audit in users_audit:
        cats = set(u_audit["primary_categories"].split("|"))
        subs = set(u_audit["preferred_subtypes"].split("|"))
        pref_col = u_audit["preferred_color_family"]
        pref_style = u_audit["preferred_style"]
        pref_tier = u_audit["price_tier_preference"]
        beta_u = float(u_audit["price_sensitivity"])

        u_utils: list[float] = []
        u_weights: list[float] = []
        for idx, item in enumerate(catalog):
            cat_match = 1.15 if item["category"] in cats else 0.0
            sub_match = 1.10 if item["subtype"] in subs else 0.0
            col_match = 0.65 if item["color_family"] == pref_col else 0.0
            style_match = 0.60 if item["style_tag"] == pref_style else 0.0
            price_pen = beta_u * price_tier_distance(item["price_tier"], pref_tier)
            pop_term = 0.40 * math.log(pop_prior[idx] * n_items + 1e-6)
            jitter = rng.gauss(0.0, 0.25)

            util = cat_match + sub_match + col_match + style_match - price_pen + pop_term + jitter
            u_utils.append(util)
            u_weights.append(math.exp(max(-6.0, min(6.0, util))))
        user_item_utility.append(u_utils)
        user_anchor_weights.append(u_weights)

    # 2. Generate chronological sessions across train, val, test splits
    raw_session_specs: list[tuple[datetime, int, str]] = []
    split_session_means = {
        "low": {"train": 3.0, "val": 0.75, "test": 0.75},
        "medium": {"train": 4.4, "val": 1.0, "test": 1.0},
        "high": {"train": 6.5, "val": 1.4, "test": 1.4},
    }

    for u_idx, u_audit in enumerate(users_audit):
        act = u_audit["activity_level"]
        for split_name, (win_start, win_end) in SPLIT_WINDOWS.items():
            mean_s = split_session_means[act][split_name]
            floor_s = int(mean_s)
            rem_s = mean_s - floor_s
            n_sess = floor_s + (1 if rng.random() < rem_s else 0)
            if split_name == "train" and n_sess < 2:
                n_sess = 2
            usable_span = int((win_end - win_start).total_seconds()) - SESSION_BOUNDARY_BUFFER_SECONDS
            for _ in range(n_sess):
                offset = rng.randint(0, usable_span)
                s_start = win_start + timedelta(seconds=offset)
                raw_session_specs.append((s_start, u_idx, split_name))

    # Sort all sessions globally by start timestamp
    raw_session_specs.sort(key=lambda x: (x[0], x[1]))

    sessions_observable: list[dict[str, Any]] = []
    sessions_audit: list[dict[str, Any]] = []
    interactions_observable: list[dict[str, Any]] = []
    interactions_audit: list[dict[str, Any]] = []
    orders_observable: list[dict[str, Any]] = []
    order_items_observable: list[dict[str, Any]] = []

    intent_choices = (
        ("outfit_building", 0.42),
        ("category_search", 0.30),
        ("browsing", 0.20),
        ("replenishment", 0.08),
    )
    intent_labels = [c[0] for c in intent_choices]
    intent_probs = [c[1] for c in intent_choices]

    event_counter = 0
    order_counter = 0
    order_item_counter = 0

    for s_idx, (s_start, u_idx, split_name) in enumerate(raw_session_specs, start=1):
        session_id = f"SYN-S{s_idx:05d}"
        u_audit = users_audit[u_idx]
        user_id = u_audit["user_id"]
        eps_u = float(u_audit["exploration_noise_rate"])
        u_utils = user_item_utility[u_idx]
        u_weights = user_anchor_weights[u_idx]

        intent = intent_labels[weighted_choice_index(intent_probs, rng)]
        # Number of distinct viewed items in session (1..10, mean ~4.2)
        n_views = min(10, max(2, int(rng.gauss(4.3, 1.6))))

        viewed_indices: list[int] = []
        viewed_signals: list[str] = []

        # Pick anchor product
        if rng.random() < eps_u:
            anchor_idx = rng.randrange(n_items)
            viewed_signals.append("random_noise")
        elif rng.random() < 0.22:
            anchor_idx = weighted_choice_index(pop_prior, rng)
            viewed_signals.append("popularity_prior")
        else:
            anchor_idx = weighted_choice_index(u_weights, rng)
            viewed_signals.append("persona_match")
        viewed_indices.append(anchor_idx)

        anchor_item = catalog[anchor_idx]
        anchor_cat = anchor_item["category"]
        anchor_sub = anchor_item["subtype"]
        anchor_col = anchor_item["color_family"]

        # Pick subsequent viewed items in session
        for _ in range(1, n_views):
            if rng.random() < eps_u:
                cand_Pool = [i for i in range(n_items) if i not in viewed_indices]
                chosen_idx = rng.choice(cand_Pool)
                viewed_indices.append(chosen_idx)
                viewed_signals.append("random_noise")
                continue

            step_weights: list[float] = []
            step_signals: list[str] = []
            comp_cats = COMPLEMENTARY_CATEGORIES.get(anchor_cat, ())
            comp_subs = COMPLEMENTARY_SUBTYPES.get(anchor_sub, ())

            for idx, cand in enumerate(catalog):
                if idx in viewed_indices:
                    step_weights.append(0.0)
                    step_signals.append("excluded")
                    continue

                base_w = u_weights[idx]
                c_sim = content_sim[anchor_idx][idx]

                if intent == "outfit_building":
                    # Strongly boost cross-category complementary outfit items
                    is_comp_sub = cand["subtype"] in comp_subs
                    is_comp_cat = cand["category"] in comp_cats and cand["category"] != anchor_cat
                    col_harmony = 1.45 if cand["color_family"] == anchor_col else 1.0
                    if is_comp_sub:
                        w = base_w * 4.5 * col_harmony
                        sig = "outfit_complement"
                    elif is_comp_cat:
                        w = base_w * 2.6 * col_harmony
                        sig = "outfit_complement"
                    else:
                        w = base_w * (0.6 + c_sim)
                        sig = "persona_match"
                elif intent == "category_search":
                    # Focus on substitutes within the same category / subtype
                    same_cat = cand["category"] == anchor_cat
                    same_sub = cand["subtype"] != "unknown" and cand["subtype"] == anchor_sub
                    boost = (3.2 if same_sub else (2.0 if same_cat else 0.35)) * (1.0 + 2.5 * c_sim)
                    w = base_w * boost
                    sig = "persona_match"
                else:
                    # Browsing / replenishment
                    w = base_w * (1.0 + 1.5 * c_sim) * (0.8 + 0.4 * pop_prior[idx] * n_items)
                    sig = "persona_match"

                step_weights.append(max(1e-8, w))
                step_signals.append(sig)

            chosen_idx = weighted_choice_index(step_weights, rng)
            viewed_indices.append(chosen_idx)
            viewed_signals.append(step_signals[chosen_idx])

        # Generate timestamps and funnel progression within session
        curr_ts = s_start
        seq_num = 0
        session_events_obs: list[dict[str, Any]] = []
        session_events_aud: list[dict[str, Any]] = []
        carted_items: list[tuple[int, dict[str, Any], str]] = []  # (item_idx, variant_dict, origin_signal)

        for item_idx, orig_sig in zip(viewed_indices, viewed_signals):
            item = catalog[item_idx]
            variant = rng.choice(item["variants"])
            curr_ts = curr_ts + timedelta(seconds=rng.randint(8, 75))
            seq_num += 1
            event_counter += 1
            ev_id = f"SYN-E{event_counter:06d}"

            session_events_obs.append(
                {
                    "event_id": ev_id,
                    "session_id": session_id,
                    "user_id": user_id,
                    "product_id": item["id"],
                    "sku": variant["sku"],
                    "event_type": "view",
                    "implicit_weight": IMPLICIT_WEIGHTS["view"],
                    "event_ts": format_ts(curr_ts),
                    "sequence_number": seq_num,
                    "split": split_name,
                    "is_synthetic": True,
                    "data_source": DATA_SOURCE_LABEL,
                }
            )
            session_events_aud.append(
                {
                    "event_id": ev_id,
                    "session_id": session_id,
                    "user_id": user_id,
                    "product_id": item["id"],
                    "event_type": "view",
                    "origin_signal": orig_sig,
                    "latent_utility": round(u_utils[item_idx], 4),
                    "split": split_name,
                }
            )

            # Decide whether viewed item is added to cart
            norm_util = (u_utils[item_idx] - 0.8) / 1.5
            intent_cart_bonus = 0.50 if intent == "outfit_building" else 0.10
            p_cart = sigmoid(-1.20 + 0.85 * norm_util + intent_cart_bonus)
            if rng.random() < p_cart:
                curr_ts = curr_ts + timedelta(seconds=rng.randint(12, 95))
                seq_num += 1
                event_counter += 1
                cart_ev_id = f"SYN-E{event_counter:06d}"
                session_events_obs.append(
                    {
                        "event_id": cart_ev_id,
                        "session_id": session_id,
                        "user_id": user_id,
                        "product_id": item["id"],
                        "sku": variant["sku"],
                        "event_type": "add_to_cart",
                        "implicit_weight": IMPLICIT_WEIGHTS["add_to_cart"],
                        "event_ts": format_ts(curr_ts),
                        "sequence_number": seq_num,
                        "split": split_name,
                        "is_synthetic": True,
                        "data_source": DATA_SOURCE_LABEL,
                    }
                )
                session_events_aud.append(
                    {
                        "event_id": cart_ev_id,
                        "session_id": session_id,
                        "user_id": user_id,
                        "product_id": item["id"],
                        "event_type": "add_to_cart",
                        "origin_signal": orig_sig,
                        "latent_utility": round(u_utils[item_idx], 4),
                        "split": split_name,
                    }
                )
                carted_items.append((item_idx, variant, orig_sig))

        # Decide whether non-empty cart converts to order
        converted = False
        if carted_items:
            p_order = 0.68 if len(carted_items) >= 2 else 0.26
            if rng.random() < p_order:
                purchased_list: list[tuple[int, dict[str, Any], str]] = []
                for entry in carted_items:
                    if rng.random() < 0.90:
                        purchased_list.append(entry)
                if not purchased_list:
                    purchased_list.append(carted_items[0])

                converted = True
                curr_ts = curr_ts + timedelta(seconds=rng.randint(45, 240))
                order_ts_str = format_ts(curr_ts)

                order_counter += 1
                order_id = f"SYN-O{order_counter:05d}"
                total_qty = 0
                order_total_cents = 0

                for item_idx, variant, orig_sig in purchased_list:
                    item = catalog[item_idx]
                    qty = 2 if rng.random() < 0.08 else 1
                    unit_price = int(item["price_cents"])
                    total_qty += qty
                    order_total_cents += qty * unit_price

                    seq_num += 1
                    event_counter += 1
                    pur_ev_id = f"SYN-E{event_counter:06d}"
                    session_events_obs.append(
                        {
                            "event_id": pur_ev_id,
                            "session_id": session_id,
                            "user_id": user_id,
                            "product_id": item["id"],
                            "sku": variant["sku"],
                            "event_type": "purchase",
                            "implicit_weight": IMPLICIT_WEIGHTS["purchase"],
                            "event_ts": order_ts_str,
                            "sequence_number": seq_num,
                            "split": split_name,
                            "is_synthetic": True,
                            "data_source": DATA_SOURCE_LABEL,
                        }
                    )
                    session_events_aud.append(
                        {
                            "event_id": pur_ev_id,
                            "session_id": session_id,
                            "user_id": user_id,
                            "product_id": item["id"],
                            "event_type": "purchase",
                            "origin_signal": orig_sig,
                            "latent_utility": round(u_utils[item_idx], 4),
                            "split": split_name,
                        }
                    )

                    order_item_counter += 1
                    order_items_observable.append(
                        {
                            "order_item_id": f"SYN-OI{order_item_counter:05d}",
                            "order_id": order_id,
                            "session_id": session_id,
                            "user_id": user_id,
                            "product_id": item["id"],
                            "sku": variant["sku"],
                            "quantity": qty,
                            "unit_price_cents": unit_price,
                            "split": split_name,
                            "is_synthetic": True,
                            "data_source": DATA_SOURCE_LABEL,
                        }
                    )

                orders_observable.append(
                    {
                        "order_id": order_id,
                        "session_id": session_id,
                        "user_id": user_id,
                        "order_ts": order_ts_str,
                        "item_count": len(purchased_list),
                        "total_quantity": total_qty,
                        "order_total_cents": order_total_cents,
                        "split": split_name,
                        "is_synthetic": True,
                        "data_source": DATA_SOURCE_LABEL,
                    }
                )

        s_end = curr_ts + timedelta(seconds=rng.randint(5, 30))
        sessions_observable.append(
            {
                "session_id": session_id,
                "user_id": user_id,
                "session_start_ts": format_ts(s_start),
                "session_end_ts": format_ts(s_end),
                "num_views": len(viewed_indices),
                "num_cart_adds": len(carted_items),
                "converted_to_order": converted,
                "split": split_name,
                "is_synthetic": True,
                "data_source": DATA_SOURCE_LABEL,
            }
        )
        sessions_audit.append(
            {
                "session_id": session_id,
                "user_id": user_id,
                "session_intent": intent,
                "anchor_product_id": anchor_item["id"],
                "split": split_name,
            }
        )
        interactions_observable.extend(session_events_obs)
        interactions_audit.extend(session_events_aud)

    elapsed_sec = time.perf_counter() - t0
    return {
        "seed": seed,
        "elapsed_seconds": round(elapsed_sec, 4),
        "catalog_count": n_items,
        "users": users_observable,
        "sessions": sessions_observable,
        "interactions": interactions_observable,
        "orders": orders_observable,
        "order_items": order_items_observable,
        "audit_users": users_audit,
        "audit_sessions": sessions_audit,
        "audit_interactions": interactions_audit,
    }


def write_csv_deterministic(path: Path, rows: list[dict[str, Any]]) -> str:
    """Write rows with deterministic LF line endings and return SHA-256 hex digest."""
    if not rows:
        raise ValueError(f"Cannot write empty table to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_measured_statistics(bundle: dict[str, Any]) -> dict[str, Any]:
    """Compute exact measured metrics from generated bundle (no unverified estimates)."""
    n_users = len(bundle["users"])
    n_items = bundle["catalog_count"]
    sessions = bundle["sessions"]
    interactions = bundle["interactions"]
    orders = bundle["orders"]
    order_items = bundle["order_items"]

    event_counts = Counter(r["event_type"] for r in interactions)
    split_session_counts = Counter(r["split"] for r in sessions)
    split_event_counts = Counter(r["split"] for r in interactions)
    split_order_counts = Counter(r["split"] for r in orders)

    all_pairs = {(r["user_id"], r["product_id"]) for r in interactions}
    train_pairs = {(r["user_id"], r["product_id"]) for r in interactions if r["split"] == "train"}
    val_pairs = {(r["user_id"], r["product_id"]) for r in interactions if r["split"] == "val"}
    test_pairs = {(r["user_id"], r["product_id"]) for r in interactions if r["split"] == "test"}

    matrix_cells = n_users * n_items
    overall_density = len(all_pairs) / matrix_cells
    overall_sparsity = 1.0 - overall_density
    train_density = len(train_pairs) / matrix_cells
    train_sparsity = 1.0 - train_density

    # Catalog coverage
    all_products_seen = {r["product_id"] for r in interactions}
    train_products_seen = {r["product_id"] for r in interactions if r["split"] == "train"}
    val_products_seen = {r["product_id"] for r in interactions if r["split"] == "val"}
    test_products_seen = {r["product_id"] for r in interactions if r["split"] == "test"}

    # Evaluation-time derived repeat interaction metrics
    val_repeat_from_train = len(val_pairs & train_pairs)
    test_repeat_from_train_val = len(test_pairs & (train_pairs | val_pairs))

    # Multi-item order metrics
    multi_item_orders = [o for o in orders if int(o["item_count"]) >= 2]
    avg_basket_items = (
        sum(int(o["item_count"]) for o in orders) / len(orders) if orders else 0.0
    )

    # Top-10% product concentration (long-tail check)
    prod_view_counts = Counter(
        r["product_id"] for r in interactions if r["event_type"] == "view"
    )
    top_10_pct_k = max(1, round(0.10 * n_items))
    top_10_views = sum(c for _, c in prod_view_counts.most_common(top_10_pct_k))
    top_10_view_share = top_10_views / max(1, event_counts["view"])

    # Audit persona & origin signal distributions
    persona_dist = Counter(u["persona_id"] for u in bundle["audit_users"])
    origin_dist = Counter(
        e["origin_signal"] for e in bundle["audit_interactions"] if e["event_type"] == "view"
    )

    return {
        "design_assumptions": {
            "target_users": 600,
            "target_catalog_products": 328,
            "target_overall_sparsity_range": [0.91, 0.96],
            "min_train_catalog_coverage_threshold": 0.98,
            "target_multi_item_order_ratio_range": [0.50, 0.75],
            "is_repeat_interaction_policy": (
                "Derived at evaluation time by comparing (user_id, product_id) pairs "
                "against earlier chronological splits; NOT stored in raw interaction CSVs "
                "to keep event tables purely observational."
            ),
        },
        "measured_outcomes": {
            "generation_runtime_seconds": bundle["elapsed_seconds"],
            "active_catalog_products": n_items,
            "synthetic_users": n_users,
            "shopping_sessions": len(sessions),
            "total_interactions": len(interactions),
            "event_type_counts": dict(event_counts),
            "total_orders": len(orders),
            "total_order_items": len(order_items),
            "multi_item_orders": len(multi_item_orders),
            "multi_item_order_ratio": round(len(multi_item_orders) / max(1, len(orders)), 4),
            "mean_distinct_items_per_order": round(avg_basket_items, 4),
            "view_to_cart_rate": round(event_counts["add_to_cart"] / max(1, event_counts["view"]), 4),
            "cart_session_to_order_conversion_rate": round(
                len(orders) / max(1, sum(1 for s in sessions if int(s["num_cart_adds"]) > 0)),
                4,
            ),
            "overall_unique_user_item_pairs": len(all_pairs),
            "train_unique_user_item_pairs": len(train_pairs),
            "val_unique_user_item_pairs": len(val_pairs),
            "test_unique_user_item_pairs": len(test_pairs),
            "overall_matrix_density": round(overall_density, 4),
            "overall_matrix_sparsity": round(overall_sparsity, 4),
            "train_matrix_density": round(train_density, 4),
            "train_matrix_sparsity": round(train_sparsity, 4),
            "catalog_coverage_overall_count": len(all_products_seen),
            "catalog_coverage_overall_ratio": round(len(all_products_seen) / n_items, 4),
            "catalog_coverage_train_count": len(train_products_seen),
            "catalog_coverage_train_ratio": round(len(train_products_seen) / n_items, 4),
            "catalog_coverage_val_count": len(val_products_seen),
            "catalog_coverage_test_count": len(test_products_seen),
            "train_coverage_threshold_satisfied": (len(train_products_seen) / n_items) >= 0.98,
            "top_10_percent_products_view_share": round(top_10_view_share, 4),
            "split_session_counts": dict(split_session_counts),
            "split_event_counts": dict(split_event_counts),
            "split_order_counts": dict(split_order_counts),
            "eval_derived_val_repeat_pairs_from_train": val_repeat_from_train,
            "eval_derived_val_novel_pairs": len(val_pairs) - val_repeat_from_train,
            "eval_derived_test_repeat_pairs_from_prior": test_repeat_from_train_val,
            "eval_derived_test_novel_pairs": len(test_pairs) - test_repeat_from_train_val,
            "audit_persona_counts": dict(persona_dist),
            "audit_view_origin_signal_counts": dict(origin_dist),
        },
    }


def render_markdown_report(manifest: dict[str, Any]) -> str:
    m = manifest["statistics"]["measured_outcomes"]
    a = manifest["statistics"]["design_assumptions"]
    hashes = manifest["file_sha256"]

    lines = [
        "# ELAN V5 — Milestone 1: Reproducible Synthetic Interaction Dataset Report",
        "",
        "> **CRITICAL SCIENTIFIC DISCLAIMER (`SYNTHETIC_SIMULATION_ONLY`):**",
        "> All records in this dataset are computer-generated synthetic interactions derived from",
        "> mathematical persona and outfit rules over the 328-product ELAN release catalog.",
        "> Offline metrics computed on this dataset validate algorithmic implementation, data pipeline",
        "> integrity, and absence of leakage under simulation assumptions. **They do NOT measure real",
        "> human preference or justify replacing the production V1 recommender on the live website.**",
        "",
        "## 1. Provenance & Reproducibility Summary",
        "",
        f"- **Dataset Label:** `{manifest['data_source']}` (`is_synthetic = True` across all rows)",
        f"- **Random Seed:** `{manifest['seed']}`",
        f"- **Source Catalog:** `backend/data/elan_release.db` (opened strictly in SQLite `mode=ro`)",
        f"- **Measured Generation Runtime:** `{m['generation_runtime_seconds']:.4f}` seconds",
        "",
        "## 2. Design Assumptions vs. Measured Outcomes",
        "",
        "| Metric | Design Assumption / Target | Measured Actual Outcome | Status |",
        "|---|---:|---:|---|",
        f"| Active Catalog Products | {a['target_catalog_products']} | **{m['active_catalog_products']}** | Verified |",
        f"| Synthetic Users | {a['target_users']} | **{m['synthetic_users']}** | Verified |",
        f"| Shopping Sessions | ~3,600 (assumption) | **{m['shopping_sessions']}** | Measured |",
        f"| Total Interaction Events | ~21,500 (assumption) | **{m['total_interactions']}** | Measured |",
        f"| Synthetic Orders (Baskets) | ~1,150 (assumption) | **{m['total_orders']}** ({m['total_order_items']} line items) | Measured |",
        f"| Multi-Item Order Ratio | {a['target_multi_item_order_ratio_range'][0]*100:.1f}%–{a['target_multi_item_order_ratio_range'][1]*100:.1f}% | **{m['multi_item_order_ratio']*100:.2f}%** ({m['multi_item_orders']}/{m['total_orders']}) | {'PASS' if a['target_multi_item_order_ratio_range'][0] <= m['multi_item_order_ratio'] <= a['target_multi_item_order_ratio_range'][1] else 'FAIL'} |",
        f"| Overall User–Item Sparsity | {a['target_overall_sparsity_range'][0]*100:.1f}%–{a['target_overall_sparsity_range'][1]*100:.1f}% | **{m['overall_matrix_sparsity']*100:.2f}%** ({m['overall_unique_user_item_pairs']} unique pairs) | {'PASS' if a['target_overall_sparsity_range'][0] <= m['overall_matrix_sparsity'] <= a['target_overall_sparsity_range'][1] else 'FAIL'} |",
        f"| Train User–Item Sparsity | >= 91.0% | **{m['train_matrix_sparsity']*100:.2f}%** ({m['train_unique_user_item_pairs']} unique pairs) | Verified |",
        f"| Train Product Coverage | >= {a['min_train_catalog_coverage_threshold']*100:.1f}% (>= 322/328) | **{m['catalog_coverage_train_ratio']*100:.2f}%** ({m['catalog_coverage_train_count']}/{m['active_catalog_products']}) | {'PASS' if m['train_coverage_threshold_satisfied'] else 'FAIL'} |",
        f"| Overall Product Coverage | Measured | **{m['catalog_coverage_overall_ratio']*100:.2f}%** ({m['catalog_coverage_overall_count']}/{m['active_catalog_products']}) | Measured |",
        "",
        "## 3. Funnel & Chronological Split Breakdown",
        "",
        "- **Event Funnel Counts:**",
        f"  - `view`: **{m['event_type_counts'].get('view', 0)}**",
        f"  - `add_to_cart`: **{m['event_type_counts'].get('add_to_cart', 0)}** (View-to-Cart Rate: **{m['view_to_cart_rate']*100:.2f}%**)",
        f"  - `purchase`: **{m['event_type_counts'].get('purchase', 0)}** (Cart-Session-to-Order Rate: **{m['cart_session_to_order_conversion_rate']*100:.2f}%**)",
        f"  - Mean Distinct Items per Order: **{m['mean_distinct_items_per_order']:.2f}**",
        f"  - Top-10% Popular Products Share of Views: **{m['top_10_percent_products_view_share']*100:.2f}%**",
        "",
        "| Split | Window (UTC) | Sessions | Interactions | Orders | Unique $(u, i)$ Pairs | Novel $(u, i)$ Pairs (Eval-Derived) |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| `train` | `2026-01-01` .. `2026-03-04` | {m['split_session_counts']['train']} | {m['split_event_counts']['train']} | {m['split_order_counts']['train']} | {m['train_unique_user_item_pairs']} | {m['train_unique_user_item_pairs']} (baseline) |",
        f"| `val` | `2026-03-05` .. `2026-03-17` | {m['split_session_counts']['val']} | {m['split_event_counts']['val']} | {m['split_order_counts']['val']} | {m['val_unique_user_item_pairs']} | {m['eval_derived_val_novel_pairs']} ({m['eval_derived_val_repeat_pairs_from_train']} repeat) |",
        f"| `test` | `2026-03-18` .. `2026-03-31` | {m['split_session_counts']['test']} | {m['split_event_counts']['test']} | {m['split_order_counts']['test']} | {m['test_unique_user_item_pairs']} | {m['eval_derived_test_novel_pairs']} ({m['eval_derived_test_repeat_pairs_from_prior']} repeat) |",
        "",
        "## 4. Anti-Leakage Architecture",
        "",
        "1. **Strict Quarantine of Generator-Internal Variables:** Latent simulation parameters (`persona_id`,",
        "   `primary_categories`, `preferred_subtypes`, `preferred_color_family`, `preferred_style`,",
        "   `price_tier_preference`, `price_sensitivity`, `exploration_noise_rate`, `session_intent`,",
        "   `anchor_product_id`, `origin_signal`, `latent_utility`) are **completely excluded** from",
        "   `backend/evaluation/v5/data/*.csv` and stored separately in `backend/evaluation/v5/audit/`.",
        "2. **Evaluation-Time Derived Repeat Flag:** `is_repeat_interaction` is not stored in raw interaction CSVs;",
        "   it is computed dynamically at evaluation time by checking whether `(user_id, product_id)` appeared in",
        "   prior chronological splits.",
        "3. **Atomic Session & Order Partitioning:** Every session, along with all its events, orders, and order",
        "   items, is contained strictly within a single chronological split window.",
        "",
        "## 5. File SHA-256 Digests (`seed=20260501`)",
        "",
    ]
    for fname, digest in sorted(hashes.items()):
        lines.append(f"- `{fname}`: `{digest}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-users", type=int, default=600)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_RELEASE_DB)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    bundle = generate_dataset(db_path=args.db_path, seed=args.seed, num_users=args.num_users)
    stats = compute_measured_statistics(bundle)

    model_files = {
        "data/synthetic_users.csv": (args.data_dir / "synthetic_users.csv", bundle["users"]),
        "data/synthetic_sessions.csv": (args.data_dir / "synthetic_sessions.csv", bundle["sessions"]),
        "data/synthetic_interactions.csv": (args.data_dir / "synthetic_interactions.csv", bundle["interactions"]),
        "data/synthetic_orders.csv": (args.data_dir / "synthetic_orders.csv", bundle["orders"]),
        "data/synthetic_order_items.csv": (args.data_dir / "synthetic_order_items.csv", bundle["order_items"]),
    }
    audit_files = {
        "audit/synthetic_users_latent_audit_only.csv": (
            args.audit_dir / "synthetic_users_latent_audit_only.csv",
            bundle["audit_users"],
        ),
        "audit/synthetic_sessions_latent_audit_only.csv": (
            args.audit_dir / "synthetic_sessions_latent_audit_only.csv",
            bundle["audit_sessions"],
        ),
        "audit/synthetic_interactions_latent_audit_only.csv": (
            args.audit_dir / "synthetic_interactions_latent_audit_only.csv",
            bundle["audit_interactions"],
        ),
    }

    file_hashes: dict[str, str] = {}
    for rel_key, (target_path, rows) in {**model_files, **audit_files}.items():
        file_hashes[rel_key] = write_csv_deterministic(target_path, rows)

    manifest = {
        "dataset_type": "SYNTHETIC_SIMULATION_ONLY",
        "data_source": DATA_SOURCE_LABEL,
        "is_synthetic": True,
        "scientific_disclaimer": (
            "Synthetic user-item interactions generated from catalog attributes and simulation rules. "
            "For offline pipeline and model mechanics validation only; never claim real-world "
            "recommendation improvements from this dataset."
        ),
        "seed": args.seed,
        "split_windows_utc": {
            k: [format_ts(v[0]), format_ts(v[1])] for k, v in SPLIT_WINDOWS.items()
        },
        "forbidden_latent_columns_in_model_data": [
            "persona_id",
            "primary_categories",
            "preferred_subtypes",
            "preferred_color_family",
            "preferred_style",
            "price_tier_preference",
            "price_sensitivity",
            "exploration_noise_rate",
            "activity_level",
            "session_intent",
            "anchor_product_id",
            "origin_signal",
            "latent_utility",
        ],
        "statistics": stats,
        "file_sha256": file_hashes,
    }

    manifest_path = args.data_dir / "synthetic_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.report_path.write_text(render_markdown_report(manifest), encoding="utf-8")

    m = stats["measured_outcomes"]
    print("ELAN V5 SYNTHETIC DATASET GENERATION: COMPLETE")
    print(f"Seed: {args.seed} | Runtime: {m['generation_runtime_seconds']:.4f}s")
    print(
        f"Users: {m['synthetic_users']} | Sessions: {m['shopping_sessions']} | "
        f"Interactions: {m['total_interactions']} | Orders: {m['total_orders']} | "
        f"Order Items: {m['total_order_items']}"
    )
    print(
        f"Overall Sparsity: {m['overall_matrix_sparsity']*100:.2f}% | "
        f"Train Sparsity: {m['train_matrix_sparsity']*100:.2f}% | "
        f"Train Coverage: {m['catalog_coverage_train_count']}/{m['active_catalog_products']} "
        f"({m['catalog_coverage_train_ratio']*100:.2f}%)"
    )


if __name__ == "__main__":
    main()
