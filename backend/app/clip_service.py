"""Isolated CLIP Late Fusion recommendation service for ELAN.

Loads precomputed CLIP ViT-B/32 image embeddings (`clip_embeddings.npz`) once per
process and combines them with the exact V1 TF-IDF text similarity matrix using
the Dev-18-selected frozen configuration:
    - text_source = "v1_tfidf"
    - w_vis = 0.5
    - text_gate_threshold = 0.0
    - require_same_category_for_visual = False

Design Guarantees:
- Controlled by `ELAN_CLIP_RECOMMENDATIONS_ENABLED` (default: "false").
- Zero runtime PyTorch, Transformers, or image-decoding dependencies.
- Preserves exact V1 TF-IDF fallback for missing-image product `HM-0398947001`
  and for any missing/corrupted/incompatible cache file.
- Never modifies SQLite or `backend/app/recommender.py`.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.recommender import DEFAULT_DATABASE, _document, _load_catalog, recommend_products

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CLIP_NPZ_PATH = (
    BACKEND_DIR / "evaluation" / "v5" / "data" / "clip_embeddings.npz"
)

# Frozen Dev-18-selected CLIP Late Fusion configuration
FROZEN_W_VIS = 0.5
FROZEN_TEXT_GATE = 0.0


def is_clip_recommendations_enabled() -> bool:
    """Return True only when `ELAN_CLIP_RECOMMENDATIONS_ENABLED` is explicitly truthy."""
    raw = os.getenv("ELAN_CLIP_RECOMMENDATIONS_ENABLED", "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


class _CLIPLateFusionCache:
    """In-memory singleton holding precomputed CLIP Late Fusion scores and catalog metadata."""

    def __init__(self, database_path: Path, clip_npz_path: Path) -> None:
        if not clip_npz_path.is_file():
            raise FileNotFoundError(f"CLIP cache not found: {clip_npz_path}")

        products = _load_catalog(database_path)
        if len(products) < 2:
            raise ValueError("At least two active products are required")

        product_ids = tuple(str(p["id"]) for p in products)
        n_items = len(product_ids)

        with np.load(clip_npz_path, allow_pickle=False) as data:
            npz_pids = tuple(str(x) for x in data["product_ids"].tolist())
            if npz_pids != product_ids:
                raise ValueError(
                    "CLIP cache product_ids do not match active SQLite catalog"
                )
            image_embeddings = data["image_embeddings"].astype(np.float64)
            image_valid = data["image_valid"].astype(bool)

        if image_embeddings.shape != (n_items, 512) or image_valid.shape != (n_items,):
            raise ValueError("Unexpected CLIP embedding or validity-mask shape")
        if np.any(image_valid) and not np.all(np.isfinite(image_embeddings[image_valid])):
            raise ValueError("Non-finite values in CLIP image embeddings")

        # Exact V1 TF-IDF text similarity matrix
        documents = [_document(product) for product in products]
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(documents)
        text_sim = cosine_similarity(tfidf_matrix).astype(np.float64)
        np.fill_diagonal(text_sim, 0.0)

        # CLIP visual cosine similarity (masked on valid images)
        vis_sim = image_embeddings @ image_embeddings.T
        both_imaged = np.outer(image_valid, image_valid)
        clipped_vis = np.clip(vis_sim, 0.0, 1.0)
        mapped_vis = FROZEN_TEXT_GATE + (1.0 - FROZEN_TEXT_GATE) * clipped_vis
        fused = (1.0 - FROZEN_W_VIS) * text_sim + FROZEN_W_VIS * mapped_vis

        gate_active = both_imaged & (text_sim >= FROZEN_TEXT_GATE)
        sim_matrix = np.where(gate_active, fused, text_sim).astype(np.float64)
        np.fill_diagonal(sim_matrix, 0.0)

        self.products = products
        self.product_ids = product_ids
        self.product_to_idx = {pid: idx for idx, pid in enumerate(product_ids)}
        self.image_valid = image_valid
        self.text_sim_matrix = text_sim
        self.sim_matrix = sim_matrix

    def recommend(self, product_id: str, top_k: int) -> tuple[list[dict[str, Any]], str]:
        if product_id not in self.product_to_idx:
            raise ValueError(f"Active product not found: {product_id}")

        target_index = self.product_to_idx[product_id]
        target = self.products[target_index]
        q_has_image = bool(self.image_valid[target_index])

        # Missing-image query (e.g., HM-0398947001) falls back cleanly to exact V1 scores
        scores = (
            self.sim_matrix[target_index]
            if q_has_image
            else self.text_sim_matrix[target_index]
        )
        candidates = sorted(
            (idx for idx in range(len(self.products)) if idx != target_index),
            key=lambda idx: (-float(scores[idx]), self.products[idx]["id"]),
        )

        recommendations: list[dict[str, Any]] = []
        for idx in candidates[:top_k]:
            candidate = self.products[idx]
            shared_colors = sorted(set(target["colors"]) & set(candidate["colors"]))
            shared_category = target["category"] == candidate["category"]

            reasons: list[str] = []
            if shared_category:
                reasons.append("same category")
            if shared_colors:
                reasons.append("shared colors: " + ", ".join(shared_colors))
            if q_has_image and bool(self.image_valid[idx]):
                reasons.append("visual and style similarity")
            elif not reasons:
                reasons.append("similar product text")

            recommendations.append(
                {
                    "id": candidate["id"],
                    "slug": candidate["slug"],
                    "name": candidate["name"],
                    "category": candidate["category"],
                    "price_cents": candidate["price_cents"],
                    "image_url": candidate["image_url"],
                    "similarity": round(float(scores[idx]), 4),
                    "reason": "; ".join(reasons),
                }
            )

        model_name = "clip_late_fusion" if q_has_image else "tfidf_cosine_v1"
        return recommendations, model_name


@lru_cache(maxsize=4)
def _get_cached_clip_service(
    database_path_str: str,
    clip_npz_path_str: str,
) -> _CLIPLateFusionCache:
    return _CLIPLateFusionCache(
        database_path=Path(database_path_str),
        clip_npz_path=Path(clip_npz_path_str),
    )


def recommend_products_with_clip_fallback(
    product_id: str,
    top_k: int = 5,
    database_path: Path | None = None,
    clip_npz_path: Path | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Return `(recommendations, model_name)` using CLIP Late Fusion or V1 fallback."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    db_path = (database_path or DEFAULT_DATABASE).resolve()
    npz_path = (clip_npz_path or Path(os.getenv("ELAN_CLIP_NPZ_PATH", str(DEFAULT_CLIP_NPZ_PATH)))).resolve()

    if not is_clip_recommendations_enabled():
        return recommend_products(product_id=product_id, top_k=top_k, database_path=db_path), "tfidf_cosine_v1"

    try:
        service = _get_cached_clip_service(str(db_path), str(npz_path))
        return service.recommend(product_id=product_id, top_k=top_k)
    except ValueError as exc:
        if str(exc).startswith("Active product not found:"):
            raise
        logger.warning("CLIP service error (%s); falling back to V1 TF-IDF.", exc)
    except Exception as exc:
        logger.warning("CLIP cache unavailable (%s); falling back to V1 TF-IDF.", exc)

    return recommend_products(product_id=product_id, top_k=top_k, database_path=db_path), "tfidf_cosine_v1"
