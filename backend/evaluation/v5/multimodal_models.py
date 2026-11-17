"""Lightweight Visual & Multimodal Product Similarity Models for ELAN V5 (Milestone 4, Phase 4A).

Implements:
1. Deterministic catalog-to-image mapping (`resolve_catalog_images_readonly`) against
   `backend/data/elan_release.db` (strictly `mode=ro`) and `frontend/public/products/`.
2. Handcrafted studio-garment visual feature extractor (`extract_handcrafted_visual_features`)
   using Pillow (`PIL`), NumPy, and Scikit-learn (`TruncatedSVD`):
   - 4-zone spatial HSV & RGB color histograms and moments (112 features)
   - Studio foreground garment segmentation mask & silhouette geometry (12 features)
   - 3-zone Sobel gradient orientation & edge-texture histograms (24 features)
   NOTE: These are interpretable handcrafted visual descriptors, NOT pretrained
   neural/deep multimodal embeddings (CLIP/SigLIP).
3. Offline embedding & manifest cache (`save_multimodal_cache`, `load_multimodal_cache`)
   so recommendation calls never touch SQLite or decode images at inference time.
4. `ExactV1TFIDFRecommender`: Exact reproduction of production `backend/app/recommender.py`.
5. `ImageOnlyRecommender`: Visual cosine similarity with explicit V1 text fallback
   when either the query product or candidate lacks a valid image (`HM-0398947001`).
6. `EarlyFusionRecommender`: Feature-level concatenation of L2-normalized text and
   visual embeddings with modality-normalized fallback for missing images.
7. `LateFusionRecommender`: Modality-normalized, non-attenuating category/text-gated
   score fusion with exact missing-image fallback.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote, urlsplit

import numpy as np
from PIL import Image
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RELEASE_DB = REPO_ROOT / "backend" / "data" / "elan_release.db"
DEFAULT_PUBLIC_DIR = (REPO_ROOT / "frontend" / "public").resolve()
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = THIS_DIR / "data"
DEFAULT_CACHE_NPZ = DEFAULT_DATA_DIR / "multimodal_embeddings.npz"
DEFAULT_MANIFEST_JSON = DEFAULT_DATA_DIR / "multimodal_manifest.json"

DEFAULT_SEED = 20260501
ALLOWED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
STANDARD_IMAGE_SIZE = (128, 128)


@dataclass(frozen=True)
class ProductImageRecord:
    """Deterministic mapping between an active catalog product and its local image asset."""

    product_id: str
    slug: str
    name: str
    category: str
    description: str
    price_cents: int
    colors: tuple[str, ...]
    primary_image_url: str | None
    primary_image_path: Path | None
    all_variant_image_urls: tuple[str, ...]
    has_valid_image: bool
    image_sha256: str | None
    image_width: int | None
    image_height: int | None
    image_format: str | None


@dataclass(frozen=True)
class MultimodalEmbeddingCache:
    """In-memory precomputed text and visual representations for all 328 active products."""

    product_ids: tuple[str, ...]
    categories: tuple[str, ...]
    names: tuple[str, ...]
    slugs: tuple[str, ...]
    product_to_idx: dict[str, int]
    has_valid_image: np.ndarray        # Shape (N,), bool
    text_embeddings: np.ndarray        # Shape (N, D_text), L2-normalized V1 TF-IDF vectors
    text_sim_matrix: np.ndarray        # Shape (N, N), exact V1 TF-IDF cosine similarity
    raw_visual_features: np.ndarray    # Shape (N, D_raw=148), standardized + L2-normalized
    svd_visual_embeddings: dict[int, np.ndarray]  # k -> Shape (N, k), L2-normalized
    visual_sim_matrices: dict[int, np.ndarray]    # k -> Shape (N, N) cosine similarity (0 on missing)
    extraction_time_ms: float
    peak_extraction_memory_kb: float


def _resolve_local_image_path(
    image_url: str | None, public_dir: Path = DEFAULT_PUBLIC_DIR
) -> Path | None:
    if not image_url or not image_url.strip():
        return None
    parts = urlsplit(image_url.strip())
    if parts.scheme or parts.netloc or not parts.path.startswith("/"):
        return None
    candidate = (public_dir / unquote(parts.path).lstrip("/")).resolve()
    try:
        candidate.relative_to(public_dir)
    except ValueError:
        return None
    if candidate.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
        return None
    if not candidate.is_file() or candidate.stat().st_size == 0:
        return None
    return candidate


def resolve_catalog_images_readonly(
    db_path: Path = DEFAULT_RELEASE_DB,
    public_dir: Path = DEFAULT_PUBLIC_DIR,
) -> tuple[list[ProductImageRecord], dict[str, Any]]:
    """Load active products from SQLite (`mode=ro`) and verify local image assets."""
    if not db_path.is_file():
        raise FileNotFoundError(f"Release SQLite database not found: {db_path}")

    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        product_rows = conn.execute(
            """
            SELECT id, slug, name, category, description, price_cents
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

    colors_by_pid: dict[str, set[str]] = defaultdict(set)
    primary_url_by_pid: dict[str, str] = {}
    all_urls_by_pid: dict[str, list[str]] = defaultdict(list)

    total_variants = len(variant_rows)
    variants_with_null_url: list[tuple[str, str]] = []
    variants_with_missing_file: list[tuple[str, str, str]] = []
    variants_with_valid_file = 0

    for v in variant_rows:
        pid = str(v["product_id"])
        sku = str(v["sku"])
        color = (v["color"] or "").strip()
        url = v["image_url"]
        if color:
            colors_by_pid[pid].add(color)
        if not url or not str(url).strip():
            variants_with_null_url.append((sku, pid))
            continue
        url_str = str(url).strip()
        resolved = _resolve_local_image_path(url_str, public_dir=public_dir)
        if resolved is None:
            variants_with_missing_file.append((sku, pid, url_str))
        else:
            variants_with_valid_file += 1
            if pid not in primary_url_by_pid:
                primary_url_by_pid[pid] = url_str
            if url_str not in all_urls_by_pid[pid]:
                all_urls_by_pid[pid].append(url_str)

    records: list[ProductImageRecord] = []
    products_with_image = 0
    products_missing_image: list[str] = []

    for row in product_rows:
        pid = str(row["id"])
        p_url = primary_url_by_pid.get(pid)
        p_path = _resolve_local_image_path(p_url, public_dir=public_dir) if p_url else None

        if p_path is not None:
            img_bytes = p_path.read_bytes()
            sha = hashlib.sha256(img_bytes).hexdigest()
            with Image.open(p_path) as im:
                width, height = im.size
                img_fmt = str(im.format or p_path.suffix.lstrip(".")).upper()
            has_img = True
            products_with_image += 1
        else:
            sha = None
            width = None
            height = None
            img_fmt = None
            has_img = False
            products_missing_image.append(pid)

        records.append(
            ProductImageRecord(
                product_id=pid,
                slug=str(row["slug"]),
                name=str(row["name"]),
                category=str(row["category"]),
                description=str(row["description"]),
                price_cents=int(row["price_cents"]),
                colors=tuple(sorted(colors_by_pid[pid])),
                primary_image_url=p_url,
                primary_image_path=p_path,
                all_variant_image_urls=tuple(all_urls_by_pid[pid]),
                has_valid_image=has_img,
                image_sha256=sha,
                image_width=width,
                image_height=height,
                image_format=img_fmt,
            )
        )

    products_dir = public_dir / "products"
    disk_files = sorted(
        f"/products/{p.name}"
        for p in products_dir.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_SUFFIXES
    )
    referenced_urls = {u for urls in all_urls_by_pid.values() for u in urls}
    unreferenced_files = sorted(set(disk_files) - referenced_urls)
    multi_image_products = {
        r.product_id: list(r.all_variant_image_urls)
        for r in records
        if len(r.all_variant_image_urls) > 1
    }

    audit_summary = {
        "active_products_total": len(records),
        "active_products_with_valid_image": products_with_image,
        "active_products_missing_image_count": len(products_missing_image),
        "active_products_missing_image_ids": products_missing_image,
        "product_image_coverage_ratio": round(
            products_with_image / max(1, len(records)), 4
        ),
        "total_variants": total_variants,
        "variants_with_valid_file": variants_with_valid_file,
        "variants_with_null_image_url": variants_with_null_url,
        "variants_with_missing_file_on_disk": variants_with_missing_file,
        "total_image_files_in_public_products": len(disk_files),
        "distinct_referenced_image_files": len(referenced_urls),
        "unreferenced_image_files_on_disk": unreferenced_files,
        "multi_image_products": multi_image_products,
    }
    return records, audit_summary


def _rgb_to_hsv_array(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized RGB in [0, 1] (shape H x W x 3) -> (H, S, V) each in [0, 1]."""
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    v = maxc
    s = np.where(maxc > 1e-12, delta / np.maximum(maxc, 1e-12), 0.0)

    h = np.zeros_like(maxc)
    mask = delta > 1e-12
    r_max = mask & (r == maxc)
    g_max = mask & (g == maxc) & ~r_max
    b_max = mask & ~r_max & ~g_max

    h[r_max] = ((g[r_max] - b[r_max]) / delta[r_max]) % 6.0
    h[g_max] = ((b[g_max] - r[g_max]) / delta[g_max]) + 2.0
    h[b_max] = ((r[b_max] - g[b_max]) / delta[b_max]) + 4.0
    h = (h / 6.0) % 1.0
    return h, s, v


def extract_single_image_descriptor(image_path: Path) -> np.ndarray:
    """Extract a 148-dim handcrafted studio-garment visual feature vector from `image_path`.

    Feature blocks (total = 148 dimensions):
    1. 4-Zone Spatial HSV & RGB Color Histograms + Moments (4 * 28 = 112 dims):
       - 12-bin saturation-weighted Hue histogram
       - 6-bin Saturation histogram
       - 6-bin Value (brightness) histogram
       - 4 color moments (mean R, mean G, mean B, std luminance)
    2. Studio Foreground Garment Silhouette & Geometry (12 dims):
       - Foreground mask ratio, bounding-box height/width ratios, aspect ratio,
         vertical center of mass, horizontal symmetry, and 6-band vertical width profile
    3. 3-Zone Sobel Edge Orientation & Gradient Texture Histograms (3 * 8 = 24 dims):
       - 8-bin unsigned gradient orientation histogram weighted by gradient magnitude
         across upper, middle, and lower garment regions
    """
    with Image.open(image_path) as im:
        rgb_im = im.convert("RGB").resize(
            STANDARD_IMAGE_SIZE, resample=Image.Resampling.BILINEAR
        )
        rgb = np.asarray(rgb_im, dtype=np.float64) / 255.0

    height, width, _ = rgb.shape
    h_arr, s_arr, v_arr = _rgb_to_hsv_array(rgb)
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    # Block 1: 4 horizontal spatial zones (32 rows each)
    zone_features: list[float] = []
    zone_h = height // 4
    hue_bins = np.linspace(0.0, 1.0, 13)
    sv_bins = np.linspace(0.0, 1.0, 7)

    for z_idx in range(4):
        r0 = z_idx * zone_h
        r1 = height if z_idx == 3 else (z_idx + 1) * zone_h
        z_rgb = rgb[r0:r1, :, :]
        z_h = h_arr[r0:r1, :].ravel()
        z_s = s_arr[r0:r1, :].ravel()
        z_v = v_arr[r0:r1, :].ravel()
        z_lum = lum[r0:r1, :].ravel()

        h_hist, _ = np.histogram(z_h, bins=hue_bins, weights=z_s + 0.05)
        h_hist = h_hist / (np.sum(h_hist) + 1e-12)
        s_hist, _ = np.histogram(z_s, bins=sv_bins)
        s_hist = s_hist / (np.sum(s_hist) + 1e-12)
        v_hist, _ = np.histogram(z_v, bins=sv_bins)
        v_hist = v_hist / (np.sum(v_hist) + 1e-12)

        moments = [
            float(np.mean(z_rgb[:, :, 0])),
            float(np.mean(z_rgb[:, :, 1])),
            float(np.mean(z_rgb[:, :, 2])),
            float(np.std(z_lum)),
        ]
        zone_features.extend(float(x) for x in h_hist)
        zone_features.extend(float(x) for x in s_hist)
        zone_features.extend(float(x) for x in v_hist)
        zone_features.extend(moments)

    # Block 2: Studio foreground garment silhouette & geometry (12 dims)
    border_pixels = np.concatenate(
        [
            rgb[:4, :, :].reshape(-1, 3),
            rgb[-4:, :, :].reshape(-1, 3),
            rgb[:, :4, :].reshape(-1, 3),
            rgb[:, -4:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg_rgb = np.median(border_pixels, axis=0)
    bg_dist = np.linalg.norm(rgb - bg_rgb[np.newaxis, np.newaxis, :], axis=2)
    fg_mask = bg_dist > 0.08
    fg_fill = float(np.mean(fg_mask))

    if np.any(fg_mask):
        ys, xs = np.where(fg_mask)
        bbox_h = float(np.max(ys) - np.min(ys) + 1) / float(height)
        bbox_w = float(np.max(xs) - np.min(xs) + 1) / float(width)
        aspect = bbox_h / max(bbox_w, 0.05)
        aspect_norm = min(aspect / 3.0, 1.0)
        y_com = float(np.mean(ys)) / float(height)
    else:
        bbox_h, bbox_w, aspect_norm, y_com = 1.0, 1.0, 0.3333, 0.5

    left_half = lum[:, : width // 2]
    right_half_flipped = lum[:, width // 2 :][:, ::-1]
    symmetry = float(1.0 - min(1.0, np.mean(np.abs(left_half - right_half_flipped))))

    band_widths: list[float] = []
    band_h = height // 6
    for b_idx in range(6):
        br0 = b_idx * band_h
        br1 = height if b_idx == 5 else (b_idx + 1) * band_h
        band_widths.append(float(np.mean(fg_mask[br0:br1, :])))

    silhouette_features = [
        fg_fill,
        bbox_h,
        bbox_w,
        aspect_norm,
        y_com,
        symmetry,
        *band_widths,
    ]

    # Block 3: 3-Zone Sobel gradient orientation & texture histograms (24 dims)
    gx = np.zeros_like(lum)
    gy = np.zeros_like(lum)
    gx[:, 1:-1] = lum[:, 2:] - lum[:, :-2]
    gy[1:-1, :] = lum[2:, :] - lum[:-2, :]
    mag = np.hypot(gx, gy)
    ori = np.mod(np.arctan2(gy, gx), np.pi)
    ori_bins = np.linspace(0.0, np.pi, 9)

    texture_features: list[float] = []
    tex_h = height // 3
    for t_idx in range(3):
        tr0 = t_idx * tex_h
        tr1 = height if t_idx == 2 else (t_idx + 1) * tex_h
        t_ori = ori[tr0:tr1, :].ravel()
        t_mag = mag[tr0:tr1, :].ravel()
        o_hist, _ = np.histogram(t_ori, bins=ori_bins, weights=t_mag)
        o_hist = o_hist / (np.sum(o_hist) + 1e-12)
        texture_features.extend(float(x) for x in o_hist)

    # Concatenate blocks with balanced block L2 normalization
    b1 = np.array(zone_features, dtype=np.float64)
    b2 = np.array(silhouette_features, dtype=np.float64)
    b3 = np.array(texture_features, dtype=np.float64)

    b1 = b1 / (np.linalg.norm(b1) + 1e-12) * np.sqrt(0.50)
    b2 = b2 / (np.linalg.norm(b2) + 1e-12) * np.sqrt(0.25)
    b3 = b3 / (np.linalg.norm(b3) + 1e-12) * np.sqrt(0.25)

    combined = np.concatenate([b1, b2, b3], axis=0)
    norm = float(np.linalg.norm(combined))
    return combined / max(norm, 1e-12)


def _v1_document(record: ProductImageRecord) -> str:
    """Exact document weighting formula from `backend/app/recommender.py`."""
    cat = str(record.category or "").strip()
    name = str(record.name or "").strip()
    desc = str(record.description or "").strip()
    cols = " ".join(record.colors)
    return " ".join([cat] * 3 + [name] * 2 + [desc] + [cols] * 2).strip()


def build_multimodal_embedding_cache(
    records: Sequence[ProductImageRecord],
    svd_dimensions: Sequence[int] = (32, 48, 64),
    seed: int = DEFAULT_SEED,
) -> MultimodalEmbeddingCache:
    """Extract and compute V1 TF-IDF text vectors and handcrafted visual descriptors."""
    tracemalloc.start()
    t0 = time.perf_counter()

    n_items = len(records)
    product_ids = tuple(r.product_id for r in records)
    categories = tuple(r.category for r in records)
    names = tuple(r.name for r in records)
    slugs = tuple(r.slug for r in records)
    product_to_idx = {pid: idx for idx, pid in enumerate(product_ids)}

    # 1. Exact V1 TF-IDF text vectors and cosine similarity matrix
    docs = [_v1_document(r) for r in records]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    sparse_tfidf = vectorizer.fit_transform(docs)
    text_embeddings = sparse_tfidf.toarray().astype(np.float64)
    text_sim_matrix = cosine_similarity(sparse_tfidf).astype(np.float64)
    np.fill_diagonal(text_sim_matrix, 0.0)

    # 2. Extract 148-dim handcrafted visual descriptors for all products with valid images
    raw_dim = 148
    raw_visual = np.zeros((n_items, raw_dim), dtype=np.float64)
    has_valid_image = np.zeros(n_items, dtype=bool)

    for idx, r in enumerate(records):
        if r.has_valid_image and r.primary_image_path is not None:
            raw_visual[idx, :] = extract_single_image_descriptor(r.primary_image_path)
            has_valid_image[idx] = True

    # Center and standardize across imaged products only, then L2-normalize each imaged row
    valid_rows = raw_visual[has_valid_image]
    mean_vec = np.mean(valid_rows, axis=0, keepdims=True)
    std_vec = np.std(valid_rows, axis=0, keepdims=True) + 1e-6
    centered_valid = (valid_rows - mean_vec) / std_vec
    row_norms = np.linalg.norm(centered_valid, axis=1, keepdims=True)
    row_norms = np.where(row_norms > 1e-12, row_norms, 1.0)
    normalized_valid = centered_valid / row_norms

    standardized_visual = np.zeros_like(raw_visual)
    standardized_visual[has_valid_image] = normalized_valid

    svd_embeddings: dict[int, np.ndarray] = {raw_dim: standardized_visual.copy()}
    visual_sim_matrices: dict[int, np.ndarray] = {}

    for k in svd_dimensions:
        if k >= raw_dim:
            continue
        svd = TruncatedSVD(
            n_components=k,
            algorithm="randomized",
            n_iter=15,
            random_state=seed,
        )
        reduced_valid = svd.fit_transform(normalized_valid)
        red_norms = np.linalg.norm(reduced_valid, axis=1, keepdims=True)
        red_norms = np.where(red_norms > 1e-12, red_norms, 1.0)
        reduced_valid = reduced_valid / red_norms

        mat_k = np.zeros((n_items, k), dtype=np.float64)
        mat_k[has_valid_image] = reduced_valid
        svd_embeddings[int(k)] = mat_k

    # Compute visual cosine similarity matrices (zeroed out for any missing-image row/col)
    for dim_key, emb_mat in svd_embeddings.items():
        sim_mat = cosine_similarity(emb_mat).astype(np.float64)
        np.fill_diagonal(sim_mat, 0.0)
        invalid_mask = ~has_valid_image
        if np.any(invalid_mask):
            sim_mat[invalid_mask, :] = 0.0
            sim_mat[:, invalid_mask] = 0.0
        visual_sim_matrices[int(dim_key)] = sim_mat

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return MultimodalEmbeddingCache(
        product_ids=product_ids,
        categories=categories,
        names=names,
        slugs=slugs,
        product_to_idx=product_to_idx,
        has_valid_image=has_valid_image,
        text_embeddings=text_embeddings,
        text_sim_matrix=text_sim_matrix,
        raw_visual_features=standardized_visual,
        svd_visual_embeddings=svd_embeddings,
        visual_sim_matrices=visual_sim_matrices,
        extraction_time_ms=round(elapsed_ms, 2),
        peak_extraction_memory_kb=round(peak_bytes / 1024.0, 2),
    )


def save_multimodal_cache(
    cache: MultimodalEmbeddingCache,
    records: Sequence[ProductImageRecord],
    audit_summary: dict[str, Any],
    npz_path: Path = DEFAULT_CACHE_NPZ,
    manifest_path: Path = DEFAULT_MANIFEST_JSON,
    seed: int = DEFAULT_SEED,
) -> dict[str, str]:
    """Save precomputed embeddings to `.npz` and write reproducible manifest JSON."""
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    arrays_to_save: dict[str, np.ndarray] = {
        "product_ids": np.array(cache.product_ids, dtype="<U32"),
        "categories": np.array(cache.categories, dtype="<U32"),
        "names": np.array(cache.names, dtype="<U128"),
        "slugs": np.array(cache.slugs, dtype="<U128"),
        "has_valid_image": cache.has_valid_image,
        "text_embeddings": cache.text_embeddings,
        "text_sim_matrix": cache.text_sim_matrix,
        "raw_visual_features": cache.raw_visual_features,
    }
    for dim_k, mat in sorted(cache.svd_visual_embeddings.items()):
        arrays_to_save[f"svd_visual_{dim_k}"] = mat
        arrays_to_save[f"visual_sim_{dim_k}"] = cache.visual_sim_matrices[dim_k]

    np.savez_compressed(npz_path, **arrays_to_save)
    npz_sha256 = hashlib.sha256(npz_path.read_bytes()).hexdigest()

    image_manifest_entries = [
        {
            "product_id": r.product_id,
            "category": r.category,
            "has_valid_image": r.has_valid_image,
            "primary_image_url": r.primary_image_url,
            "variant_image_count": len(r.all_variant_image_urls),
            "image_format": r.image_format,
            "image_size": [r.image_width, r.image_height] if r.has_valid_image else None,
            "image_sha256": r.image_sha256,
        }
        for r in records
    ]

    manifest_data = {
        "milestone": "Milestone 4 Phase 4A — Lightweight Visual Embedding Cache",
        "feature_type": "HANDCRAFTED_STUDIO_VISUAL_DESCRIPTORS_NOT_NEURAL",
        "seed": seed,
        "standard_image_size": list(STANDARD_IMAGE_SIZE),
        "raw_visual_feature_dim": int(cache.raw_visual_features.shape[1]),
        "available_visual_dimensions": sorted(int(k) for k in cache.svd_visual_embeddings.keys()),
        "text_tfidf_feature_dim": int(cache.text_embeddings.shape[1]),
        "npz_file_size_bytes": int(npz_path.stat().st_size),
        "npz_sha256": npz_sha256,
        "image_audit_summary": audit_summary,
        "product_image_manifest": image_manifest_entries,
    }
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return {"npz_sha256": npz_sha256, "manifest_sha256": manifest_sha256}


def load_multimodal_cache(npz_path: Path = DEFAULT_CACHE_NPZ) -> MultimodalEmbeddingCache:
    """Load precomputed multimodal cache in memory without touching SQLite or image files."""
    if not npz_path.is_file():
        raise FileNotFoundError(f"Precomputed cache not found: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        product_ids = tuple(str(x) for x in data["product_ids"].tolist())
        categories = tuple(str(x) for x in data["categories"].tolist())
        names = tuple(str(x) for x in data["names"].tolist())
        slugs = tuple(str(x) for x in data["slugs"].tolist())
        has_valid_image = data["has_valid_image"].astype(bool)
        text_embeddings = data["text_embeddings"].astype(np.float64)
        text_sim_matrix = data["text_sim_matrix"].astype(np.float64)
        raw_visual_features = data["raw_visual_features"].astype(np.float64)

        svd_embeddings: dict[int, np.ndarray] = {}
        visual_sim_matrices: dict[int, np.ndarray] = {}
        for key in data.files:
            if key.startswith("svd_visual_"):
                k = int(key.removeprefix("svd_visual_"))
                svd_embeddings[k] = data[key].astype(np.float64)
            elif key.startswith("visual_sim_"):
                k = int(key.removeprefix("visual_sim_"))
                visual_sim_matrices[k] = data[key].astype(np.float64)

    return MultimodalEmbeddingCache(
        product_ids=product_ids,
        categories=categories,
        names=names,
        slugs=slugs,
        product_to_idx={pid: idx for idx, pid in enumerate(product_ids)},
        has_valid_image=has_valid_image,
        text_embeddings=text_embeddings,
        text_sim_matrix=text_sim_matrix,
        raw_visual_features=raw_visual_features,
        svd_visual_embeddings=svd_embeddings,
        visual_sim_matrices=visual_sim_matrices,
        extraction_time_ms=0.0,
        peak_extraction_memory_kb=0.0,
    )


def _rank_candidates_v1_exact(
    row_scores: np.ndarray,
    product_ids: tuple[str, ...],
    target_idx: int,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Rank candidates using production V1's exact full-precision float key `(-float(score), id)`."""
    n_items = len(product_ids)
    candidates = sorted(
        (idx for idx in range(n_items) if idx != target_idx),
        key=lambda idx: (-float(row_scores[idx]), product_ids[idx]),
    )
    return [(product_ids[idx], float(row_scores[idx])) for idx in candidates[:top_k]]


class ExactV1TFIDFRecommender:
    """Exact reproduction of production `backend/app/recommender.py` over precomputed cache."""

    def __init__(self) -> None:
        self.product_ids: tuple[str, ...] = ()
        self.product_to_idx: dict[str, int] = {}
        self.sim_matrix: np.ndarray = np.array([], dtype=np.float64)
        self._is_fitted: bool = False

    def fit(self, cache: MultimodalEmbeddingCache) -> "ExactV1TFIDFRecommender":
        self.product_ids = cache.product_ids
        self.product_to_idx = cache.product_to_idx
        self.sim_matrix = cache.text_sim_matrix
        self._is_fitted = True
        return self

    def recommend_similar_items(
        self, product_id: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        if not self._is_fitted:
            raise RuntimeError("ExactV1TFIDFRecommender must be fitted before inference")
        if product_id not in self.product_to_idx:
            raise KeyError(f"Active product not found: {product_id}")
        t_idx = self.product_to_idx[product_id]
        return _rank_candidates_v1_exact(
            self.sim_matrix[t_idx], self.product_ids, t_idx, top_k=top_k
        )


class ImageOnlyRecommender:
    """Image-only product similarity recommender with explicit V1 fallback for missing images.

    - If the query product `i` is missing a local image (`has_valid_image[i] == False`,
      e.g., `HM-0398947001`), falls back deterministically to `ExactV1TFIDFRecommender`
      and sets `fallback_used = True`.
    - For an imaged query product `i`, candidates `j` with valid images are ranked by
      visual cosine similarity `S_vis(i, j)`. Any candidate `j` missing an image
      receives `-1.0` visual similarity so imaged candidates rank first, with deterministic
      V1 text similarity tie-breaking.
    """

    def __init__(self, visual_dim: int = 48) -> None:
        self.visual_dim = int(visual_dim)
        self.product_ids: tuple[str, ...] = ()
        self.product_to_idx: dict[str, int] = {}
        self.has_valid_image: np.ndarray = np.array([], dtype=bool)
        self.vis_sim_matrix: np.ndarray = np.array([], dtype=np.float64)
        self.text_sim_matrix: np.ndarray = np.array([], dtype=np.float64)
        self._is_fitted: bool = False

    def fit(self, cache: MultimodalEmbeddingCache) -> "ImageOnlyRecommender":
        if self.visual_dim not in cache.visual_sim_matrices:
            raise ValueError(f"Unsupported visual_dim={self.visual_dim}")
        self.product_ids = cache.product_ids
        self.product_to_idx = cache.product_to_idx
        self.has_valid_image = cache.has_valid_image
        self.vis_sim_matrix = cache.visual_sim_matrices[self.visual_dim]
        self.text_sim_matrix = cache.text_sim_matrix
        self._is_fitted = True
        return self

    def recommend_similar_items_with_meta(
        self, product_id: str, top_k: int = 5
    ) -> dict[str, Any]:
        if not self._is_fitted:
            raise RuntimeError("ImageOnlyRecommender must be fitted before inference")
        if product_id not in self.product_to_idx:
            raise KeyError(f"Active product not found: {product_id}")

        t_idx = self.product_to_idx[product_id]
        if not bool(self.has_valid_image[t_idx]):
            v1_recs = _rank_candidates_v1_exact(
                self.text_sim_matrix[t_idx], self.product_ids, t_idx, top_k=top_k
            )
            return {
                "reference_product_id": product_id,
                "model": f"image_only_d{self.visual_dim}_fallback_v1",
                "query_has_image": False,
                "fallback_used": True,
                "recommendations": v1_recs,
            }

        n_items = len(self.product_ids)
        vis_row = self.vis_sim_matrix[t_idx]
        txt_row = self.text_sim_matrix[t_idx]
        candidates = sorted(
            (idx for idx in range(n_items) if idx != t_idx),
            key=lambda idx: (
                -1 if bool(self.has_valid_image[idx]) else 0,
                -float(vis_row[idx]) if bool(self.has_valid_image[idx]) else 0.0,
                -float(txt_row[idx]),
                self.product_ids[idx],
            ),
        )
        recs = [
            (
                self.product_ids[idx],
                float(vis_row[idx]) if bool(self.has_valid_image[idx]) else float(txt_row[idx]),
            )
            for idx in candidates[:top_k]
        ]
        return {
            "reference_product_id": product_id,
            "model": f"image_only_d{self.visual_dim}",
            "query_has_image": True,
            "fallback_used": False,
            "recommendations": recs,
        }

    def recommend_similar_items(
        self, product_id: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        return self.recommend_similar_items_with_meta(product_id, top_k=top_k)[
            "recommendations"
        ]


class EarlyFusionRecommender:
    """Feature-level concatenation of L2-normalized text and visual embeddings.

    For products with valid images:
        z_i = [sqrt(1 - alpha_vis) * v_i^text  ||  sqrt(alpha_vis) * v_i^vis]
    Since ||v_i^text||_2 = 1 and ||v_i^vis||_2 = 1, ||z_i||_2 = 1, and for two
    imaged products (i, j):
        <z_i, z_j> = (1 - alpha_vis) S_text(i, j) + alpha_vis S_vis(i, j).
    For any pair where i or j lacks a valid image, only the text modality is active,
    so modality-normalized similarity falls back cleanly to S_text(i, j) without
    attenuating the text vector by sqrt(1 - alpha_vis).
    """

    def __init__(self, visual_dim: int = 48, alpha_vis: float = 0.20) -> None:
        if not (0.0 <= alpha_vis <= 1.0):
            raise ValueError("alpha_vis must be in [0.0, 1.0]")
        self.visual_dim = int(visual_dim)
        self.alpha_vis = float(alpha_vis)
        self.product_ids: tuple[str, ...] = ()
        self.product_to_idx: dict[str, int] = {}
        self.has_valid_image: np.ndarray = np.array([], dtype=bool)
        self.fused_embeddings: np.ndarray = np.array([], dtype=np.float64)
        self.sim_matrix: np.ndarray = np.array([], dtype=np.float64)
        self._is_fitted: bool = False

    def fit(self, cache: MultimodalEmbeddingCache) -> "EarlyFusionRecommender":
        if self.visual_dim not in cache.svd_visual_embeddings:
            raise ValueError(f"Unsupported visual_dim={self.visual_dim}")
        self.product_ids = cache.product_ids
        self.product_to_idx = cache.product_to_idx
        self.has_valid_image = cache.has_valid_image

        w_txt = np.sqrt(1.0 - self.alpha_vis)
        w_vis = np.sqrt(self.alpha_vis)
        vis_emb = cache.svd_visual_embeddings[self.visual_dim]
        self.fused_embeddings = np.concatenate(
            [w_txt * cache.text_embeddings, w_vis * vis_emb], axis=1
        )
        raw_sim = (self.fused_embeddings @ self.fused_embeddings.T).astype(np.float64)

        # Modality normalization: if either i or j has no image, only text is active -> S_text(i, j)
        both_imaged = np.outer(self.has_valid_image, self.has_valid_image)
        self.sim_matrix = np.where(both_imaged, raw_sim, cache.text_sim_matrix)
        np.fill_diagonal(self.sim_matrix, 0.0)
        self._is_fitted = True
        return self

    def recommend_similar_items(
        self, product_id: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        if not self._is_fitted:
            raise RuntimeError("EarlyFusionRecommender must be fitted before inference")
        if product_id not in self.product_to_idx:
            raise KeyError(f"Active product not found: {product_id}")
        t_idx = self.product_to_idx[product_id]
        return _rank_candidates_v1_exact(
            self.sim_matrix[t_idx], self.product_ids, t_idx, top_k=top_k
        )


def compute_late_fusion_similarity_matrix(
    text_sim_matrix: np.ndarray,
    visual_sim_matrix: np.ndarray,
    has_valid_image: np.ndarray,
    w_vis: float = 0.15,
    text_gate_threshold: float = 0.05,
    same_category_mask: np.ndarray | None = None,
    require_same_category_for_visual: bool = False,
) -> np.ndarray:
    """Compute modality-normalized, non-attenuating late-fusion similarity matrix.

    Mandatory Engineering Correction B guarantees:
    1. When query `i` or candidate `j` lacks a valid image (`~has_valid_image[i]` or
       `~has_valid_image[j]`), only the text modality is active (`w_text=1.0, w_vis=0.0`),
       so `S_late(i, j) = S_text(i, j)` (exact V1 score, zero attenuation).
    2. When the semantic gate is inactive (`S_text(i, j) < text_gate_threshold`, or
       category mismatch when `require_same_category_for_visual=True`), only the text
       modality is active (`w_text=1.0, w_vis=0.0`), so `S_late(i, j) = S_text(i, j)`
       (un-gated text scores are NEVER attenuated by `1 - w_vis`).
    3. When the gate is active (`has_img(i) & has_img(j) & (S_text >= tau)`), both
       modalities are active with normalized weights `(1 - w_vis) + w_vis = 1.0`, and
       visual similarity is mapped into `[tau, 1.0]` via:
           S_vis_mapped(i, j) = tau + (1.0 - tau) * clip(S_vis(i, j), 0.0, 1.0)
           S_late(i, j) = (1.0 - w_vis) * S_text(i, j) + w_vis * S_vis_mapped(i, j)
       This ensures:
       - Active modality weights sum strictly to 1.0.
       - Every candidate passing the gate (`S_text >= tau`) has `S_late >= tau`, so an
         active visual gate NEVER pushes a gated candidate below an un-gated candidate
         (`S_text < tau`).
       - When `tau = 0.0`, `S_late(i, j) = (1 - w_vis) * S_text(i, j) + w_vis * max(0, S_vis(i, j))`.
       - When `w_vis = 0.0`, `S_late(i, j) == S_text(i, j)` everywhere.
    """
    if not (0.0 <= w_vis <= 1.0):
        raise ValueError("w_vis must be in [0.0, 1.0]")
    if not (0.0 <= text_gate_threshold < 1.0):
        raise ValueError("text_gate_threshold must be in [0.0, 1.0)")

    both_imaged = np.outer(has_valid_image, has_valid_image)
    gate_active = both_imaged & (text_sim_matrix >= text_gate_threshold)
    if require_same_category_for_visual and same_category_mask is not None:
        gate_active = gate_active & same_category_mask

    clipped_vis = np.clip(visual_sim_matrix, 0.0, 1.0)
    mapped_vis = text_gate_threshold + (1.0 - text_gate_threshold) * clipped_vis
    fused_active = (1.0 - w_vis) * text_sim_matrix + w_vis * mapped_vis

    result = np.where(gate_active, fused_active, text_sim_matrix).astype(np.float64)
    np.fill_diagonal(result, 0.0)
    return result


class LateFusionRecommender:
    """Category/text-gated late-fusion recommender over precomputed text & visual similarities."""

    def __init__(
        self,
        visual_dim: int = 48,
        w_vis: float = 0.15,
        text_gate_threshold: float = 0.05,
        require_same_category_for_visual: bool = False,
    ) -> None:
        self.visual_dim = int(visual_dim)
        self.w_vis = float(w_vis)
        self.text_gate_threshold = float(text_gate_threshold)
        self.require_same_category_for_visual = bool(require_same_category_for_visual)

        self.product_ids: tuple[str, ...] = ()
        self.product_to_idx: dict[str, int] = {}
        self.has_valid_image: np.ndarray = np.array([], dtype=bool)
        self.sim_matrix: np.ndarray = np.array([], dtype=np.float64)
        self._is_fitted: bool = False

    def fit(self, cache: MultimodalEmbeddingCache) -> "LateFusionRecommender":
        if self.visual_dim not in cache.visual_sim_matrices:
            raise ValueError(f"Unsupported visual_dim={self.visual_dim}")
        self.product_ids = cache.product_ids
        self.product_to_idx = cache.product_to_idx
        self.has_valid_image = cache.has_valid_image

        cat_arr = np.array(cache.categories)
        same_cat_mask = cat_arr[:, np.newaxis] == cat_arr[np.newaxis, :]
        self.sim_matrix = compute_late_fusion_similarity_matrix(
            text_sim_matrix=cache.text_sim_matrix,
            visual_sim_matrix=cache.visual_sim_matrices[self.visual_dim],
            has_valid_image=cache.has_valid_image,
            w_vis=self.w_vis,
            text_gate_threshold=self.text_gate_threshold,
            same_category_mask=same_cat_mask,
            require_same_category_for_visual=self.require_same_category_for_visual,
        )
        self._is_fitted = True
        return self

    def recommend_similar_items_with_meta(
        self, product_id: str, top_k: int = 5
    ) -> dict[str, Any]:
        if not self._is_fitted:
            raise RuntimeError("LateFusionRecommender must be fitted before inference")
        if product_id not in self.product_to_idx:
            raise KeyError(f"Active product not found: {product_id}")
        t_idx = self.product_to_idx[product_id]
        q_has_img = bool(self.has_valid_image[t_idx])
        recs = _rank_candidates_v1_exact(
            self.sim_matrix[t_idx], self.product_ids, t_idx, top_k=top_k
        )
        return {
            "reference_product_id": product_id,
            "model": f"late_fusion_d{self.visual_dim}_w{self.w_vis:.2f}",
            "query_has_image": q_has_img,
            "fallback_used": not q_has_img,
            "recommendations": recs,
        }

    def recommend_similar_items(
        self, product_id: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        return self.recommend_similar_items_with_meta(product_id, top_k=top_k)[
            "recommendations"
        ]
