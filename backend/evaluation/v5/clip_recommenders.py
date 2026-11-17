"""Offline CLIP Recommendation Strategies for ELAN V5 (Milestone 4B.3A).

Implements three independent recommendation strategies over precomputed
`clip_embeddings.npz` and `clip_manifest.json` (plus optional precomputed
`multimodal_embeddings.npz` for exact V1 TF-IDF fallback on `HM-0398947001`):

1. `CLIPImageOnlyRecommender`: CLIP image-to-image cosine similarity.
2. `CLIPTextImageRecommender`: CLIP shared-space text-to-image (and optional
   symmetric image-to-text / text-to-text) cosine similarity.
3. `CLIPLateFusionRecommender`: Modality-normalized, non-attenuating gated
   late fusion of text similarity (V1 TF-IDF or CLIP text) and CLIP visual
   similarity.

Design Guarantees:
- Zero runtime PyTorch, Transformers, image decoding, or SQLite queries.
- Strict respect for `image_valid` and `text_valid` boolean masks (zero-vector
  placeholders are never treated as valid CLIP similarity features).
- Exact deterministic ranking and tie-breaking: `(-float(score), product_id)`.
- Query product is always excluded from its own recommendations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = THIS_DIR / "data"
DEFAULT_CLIP_NPZ = DEFAULT_DATA_DIR / "clip_embeddings.npz"
DEFAULT_CLIP_MANIFEST = DEFAULT_DATA_DIR / "clip_manifest.json"
DEFAULT_V1_CACHE_NPZ = DEFAULT_DATA_DIR / "multimodal_embeddings.npz"
EXPECTED_CLIP_DIM = 512


@dataclass(frozen=True)
class OfflineCLIPData:
    """Pure-NumPy representation of precomputed CLIP embeddings and optional V1 cache."""

    product_ids: tuple[str, ...]
    product_to_idx: dict[str, int]
    image_embeddings: np.ndarray  # Shape (N, 512), float32/float64
    text_embeddings: np.ndarray   # Shape (N, 512), float32/float64
    image_valid: np.ndarray       # Shape (N,), bool
    text_valid: np.ndarray        # Shape (N,), bool
    categories: tuple[str, ...] | None = None
    v1_text_sim_matrix: np.ndarray | None = None


def load_offline_clip_data(
    clip_npz_path: Path = DEFAULT_CLIP_NPZ,
    clip_manifest_path: Path = DEFAULT_CLIP_MANIFEST,
    v1_cache_npz_path: Path | None = DEFAULT_V1_CACHE_NPZ,
) -> OfflineCLIPData:
    """Load precomputed CLIP NPZ/manifest (and optional V1 text matrix) using NumPy only."""
    if not clip_npz_path.is_file():
        raise FileNotFoundError(f"CLIP NPZ cache not found: {clip_npz_path}")
    if not clip_manifest_path.is_file():
        raise FileNotFoundError(f"CLIP manifest not found: {clip_manifest_path}")

    manifest = json.loads(clip_manifest_path.read_text(encoding="utf-8"))
    expected_count = int(manifest["source_catalog"]["total_active_products"])
    expected_dim = int(manifest["embedding_specification"]["embedding_dim"])

    with np.load(clip_npz_path, allow_pickle=False) as data:
        product_ids = tuple(str(x) for x in data["product_ids"].tolist())
        image_embeddings = data["image_embeddings"].astype(np.float32)
        text_embeddings = data["text_embeddings"].astype(np.float32)
        image_valid = data["image_valid"].astype(bool)
        text_valid = data["text_valid"].astype(bool)

    if len(product_ids) != expected_count:
        raise ValueError(
            f"Product count mismatch: NPZ={len(product_ids)}, manifest={expected_count}"
        )
    if expected_dim != EXPECTED_CLIP_DIM:
        raise ValueError(f"Unexpected embedding dim in manifest: {expected_dim}")

    categories: tuple[str, ...] | None = None
    v1_text_sim_matrix: np.ndarray | None = None
    if v1_cache_npz_path is not None and v1_cache_npz_path.is_file():
        with np.load(v1_cache_npz_path, allow_pickle=False) as v1_data:
            v1_pids = tuple(str(x) for x in v1_data["product_ids"].tolist())
            if v1_pids != product_ids:
                raise ValueError("Product ID alignment mismatch between CLIP NPZ and V1 NPZ")
            categories = tuple(str(x) for x in v1_data["categories"].tolist())
            v1_text_sim_matrix = v1_data["text_sim_matrix"].astype(np.float64)

    return OfflineCLIPData(
        product_ids=product_ids,
        product_to_idx={pid: idx for idx, pid in enumerate(product_ids)},
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        image_valid=image_valid,
        text_valid=text_valid,
        categories=categories,
        v1_text_sim_matrix=v1_text_sim_matrix,
    )


def _validate_clip_data(data: OfflineCLIPData) -> None:
    """Validate shapes, uniqueness, finite values, and unit L2 norms on valid rows."""
    n_items = len(data.product_ids)
    if n_items < 2:
        raise ValueError("At least two products are required for recommendation")
    if len(set(data.product_ids)) != n_items:
        raise ValueError("Duplicate product_ids detected in OfflineCLIPData")

    for name, emb, mask in (
        ("image_embeddings", data.image_embeddings, data.image_valid),
        ("text_embeddings", data.text_embeddings, data.text_valid),
    ):
        if emb.ndim != 2 or emb.shape != (n_items, EXPECTED_CLIP_DIM):
            raise ValueError(
                f"{name} must have shape ({n_items}, {EXPECTED_CLIP_DIM}), got {emb.shape}"
            )
        if mask.shape != (n_items,):
            raise ValueError(f"Validity mask for {name} must have shape ({n_items},)")
        if np.any(mask):
            valid_rows = emb[mask]
            if not np.all(np.isfinite(valid_rows)):
                raise ValueError(f"Non-finite values found in valid rows of {name}")
            norms = np.linalg.norm(valid_rows.astype(np.float64), axis=1)
            if not np.allclose(norms, 1.0, atol=1e-4):
                raise ValueError(
                    f"Valid rows of {name} must be L2-normalized (norm range [{norms.min():.6f}, {norms.max():.6f}])"
                )

    if data.v1_text_sim_matrix is not None:
        if data.v1_text_sim_matrix.shape != (n_items, n_items):
            raise ValueError("v1_text_sim_matrix shape mismatch")
        if not np.all(np.isfinite(data.v1_text_sim_matrix)):
            raise ValueError("Non-finite values in v1_text_sim_matrix")

    if data.categories is not None and len(data.categories) != n_items:
        raise ValueError("categories length mismatch")


def _compute_masked_cosine_matrix(
    left_emb: np.ndarray,
    right_emb: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
) -> np.ndarray:
    """Compute float64 cosine similarity `(left @ right.T)` zeroed on invalid pairs."""
    sim = (left_emb.astype(np.float64) @ right_emb.astype(np.float64).T)
    valid_pair = np.outer(left_valid, right_valid)
    sim = np.where(valid_pair, sim, 0.0)
    np.fill_diagonal(sim, 0.0)
    return sim


def _rank_deterministic(
    scores: np.ndarray,
    product_ids: tuple[str, ...],
    target_idx: int,
    top_k: int,
    valid_candidate_mask: np.ndarray | None = None,
    tie_break_scores: np.ndarray | None = None,
) -> list[tuple[str, float]]:
    """Rank candidates excluding `target_idx` with deterministic tie-breaking.

    If `valid_candidate_mask` is provided, valid candidates (`True`) always rank ahead
    of invalid-feature candidates (`False`), which are ordered by `tie_break_scores`
    (e.g., exact V1 TF-IDF score) and `product_id`.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    n_items = len(product_ids)

    if valid_candidate_mask is None:
        ordered = sorted(
            (idx for idx in range(n_items) if idx != target_idx),
            key=lambda idx: (-float(scores[idx]), product_ids[idx]),
        )
        return [(product_ids[idx], float(scores[idx])) for idx in ordered[:top_k]]

    tb = tie_break_scores if tie_break_scores is not None else np.zeros(n_items, dtype=np.float64)
    ordered = sorted(
        (idx for idx in range(n_items) if idx != target_idx),
        key=lambda idx: (
            0 if bool(valid_candidate_mask[idx]) else 1,
            -float(scores[idx]) if bool(valid_candidate_mask[idx]) else -float(tb[idx]),
            -float(tb[idx]),
            product_ids[idx],
        ),
    )
    return [
        (
            product_ids[idx],
            float(scores[idx]) if bool(valid_candidate_mask[idx]) else float(tb[idx]),
        )
        for idx in ordered[:top_k]
    ]


class BaseCLIPRecommender:
    """Shared interface for offline CLIP recommendation strategies."""

    model_name: str = "base_clip"

    def __init__(self) -> None:
        self.data: OfflineCLIPData | None = None
        self._is_fitted: bool = False

    def fit(self, data: OfflineCLIPData) -> Self:
        _validate_clip_data(data)
        self.data = data
        self._fit_impl(data)
        self._is_fitted = True
        return self

    def _fit_impl(self, data: OfflineCLIPData) -> None:
        raise NotImplementedError

    def recommend_similar_items_with_meta(
        self, product_id: str, top_k: int = 5
    ) -> dict[str, Any]:
        raise NotImplementedError

    def recommend_similar_items(
        self, product_id: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        return self.recommend_similar_items_with_meta(product_id, top_k=top_k)[
            "recommendations"
        ]

    def _require_fitted_target(self, product_id: str, top_k: int) -> tuple[OfflineCLIPData, int]:
        if not self._is_fitted or self.data is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before inference")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if product_id not in self.data.product_to_idx:
            raise KeyError(f"Active product not found: {product_id}")
        return self.data, self.data.product_to_idx[product_id]


class CLIPImageOnlyRecommender(BaseCLIPRecommender):
    """Rank candidates using CLIP image-to-image cosine similarity.

    - For valid-image queries, ranks valid-image candidates by `<e_img[i], e_img[j]>`,
      placing missing-image candidates (`image_valid[j] == False`) strictly after valid
      candidates using V1 TF-IDF tie-breaking (when provided) without assigning an
      artificial CLIP visual score.
    - For missing-image queries (`HM-0398947001`), falls back deterministically to
      exact V1 TF-IDF similarity when available (or CLIP text-to-text if V1 matrix is
      not supplied in synthetic unit tests).
    """

    model_name = "clip_image_only"

    def __init__(self) -> None:
        super().__init__()
        self.img_sim_matrix: np.ndarray = np.array([], dtype=np.float64)
        self.txt_sim_matrix: np.ndarray = np.array([], dtype=np.float64)

    def _fit_impl(self, data: OfflineCLIPData) -> None:
        self.img_sim_matrix = _compute_masked_cosine_matrix(
            data.image_embeddings,
            data.image_embeddings,
            data.image_valid,
            data.image_valid,
        )
        self.txt_sim_matrix = _compute_masked_cosine_matrix(
            data.text_embeddings,
            data.text_embeddings,
            data.text_valid,
            data.text_valid,
        )

    def recommend_similar_items_with_meta(
        self, product_id: str, top_k: int = 5
    ) -> dict[str, Any]:
        data, t_idx = self._require_fitted_target(product_id, top_k)
        q_has_img = bool(data.image_valid[t_idx])

        if not q_has_img:
            if data.v1_text_sim_matrix is not None:
                recs = _rank_deterministic(
                    data.v1_text_sim_matrix[t_idx], data.product_ids, t_idx, top_k=top_k
                )
                fb_mode = "exact_v1_tfidf"
            else:
                recs = _rank_deterministic(
                    self.txt_sim_matrix[t_idx],
                    data.product_ids,
                    t_idx,
                    top_k=top_k,
                    valid_candidate_mask=data.text_valid,
                )
                fb_mode = "clip_text_only"
            return {
                "reference_product_id": product_id,
                "model": self.model_name,
                "query_image_valid": False,
                "fallback_used": True,
                "fallback_mode": fb_mode,
                "recommendations": recs,
            }

        tie_break = (
            data.v1_text_sim_matrix[t_idx]
            if data.v1_text_sim_matrix is not None
            else self.txt_sim_matrix[t_idx]
        )
        recs = _rank_deterministic(
            self.img_sim_matrix[t_idx],
            data.product_ids,
            t_idx,
            top_k=top_k,
            valid_candidate_mask=data.image_valid,
            tie_break_scores=tie_break,
        )
        return {
            "reference_product_id": product_id,
            "model": self.model_name,
            "query_image_valid": True,
            "fallback_used": False,
            "fallback_mode": None,
            "recommendations": recs,
        }


class CLIPTextImageRecommender(BaseCLIPRecommender):
    """Shared-space CLIP text-to-image (and optional symmetric i2t/t2t) recommender.

    Computes modality-normalized weighted similarity over active valid modalities:
        S(i, j) = sum_{m in active(i,j)} w_m * S_m(i, j) / sum_{m in active(i,j)} w_m
    where:
        - t2i: `<e_txt[i], e_img[j]>` active iff `text_valid[i] & image_valid[j]`
        - i2t: `<e_img[i], e_txt[j]>` active iff `image_valid[i] & text_valid[j]`
        - t2t: `<e_txt[i], e_txt[j]>` active iff `text_valid[i] & text_valid[j]`

    Default `(w_t2i=1.0, w_i2t=0.0, w_t2t=0.0)` measures pure text-to-image alignment
    in CLIP's shared embedding space. Any pair `(i, j)` with zero active weight
    (e.g., candidate `HM-0398947001` when `w_t2t=0.0`) is ranked strictly after active
    pairs without assigning an artificial similarity score.
    """

    model_name = "clip_text_image"

    def __init__(
        self,
        w_t2i: float = 1.0,
        w_i2t: float = 0.0,
        w_t2t: float = 0.0,
        fallback_missing_query_to_v1: bool = False,
    ) -> None:
        super().__init__()
        if w_t2i < 0.0 or w_i2t < 0.0 or w_t2t < 0.0:
            raise ValueError("CLIPTextImageRecommender weights must be non-negative")
        if (w_t2i + w_i2t + w_t2t) <= 0.0:
            raise ValueError("At least one weight must be strictly positive")
        self.w_t2i = float(w_t2i)
        self.w_i2t = float(w_i2t)
        self.w_t2t = float(w_t2t)
        self.fallback_missing_query_to_v1 = bool(fallback_missing_query_to_v1)
        self.sim_matrix: np.ndarray = np.array([], dtype=np.float64)
        self.active_pair_mask: np.ndarray = np.array([], dtype=bool)

    def _fit_impl(self, data: OfflineCLIPData) -> None:
        s_t2i = _compute_masked_cosine_matrix(
            data.text_embeddings, data.image_embeddings, data.text_valid, data.image_valid
        )
        s_i2t = _compute_masked_cosine_matrix(
            data.image_embeddings, data.text_embeddings, data.image_valid, data.text_valid
        )
        s_t2t = _compute_masked_cosine_matrix(
            data.text_embeddings, data.text_embeddings, data.text_valid, data.text_valid
        )

        m_t2i = np.outer(data.text_valid, data.image_valid) & (self.w_t2i > 0.0)
        m_i2t = np.outer(data.image_valid, data.text_valid) & (self.w_i2t > 0.0)
        m_t2t = np.outer(data.text_valid, data.text_valid) & (self.w_t2t > 0.0)

        num = (
            np.where(m_t2i, self.w_t2i * s_t2i, 0.0)
            + np.where(m_i2t, self.w_i2t * s_i2t, 0.0)
            + np.where(m_t2t, self.w_t2t * s_t2t, 0.0)
        )
        den = (
            np.where(m_t2i, self.w_t2i, 0.0)
            + np.where(m_i2t, self.w_i2t, 0.0)
            + np.where(m_t2t, self.w_t2t, 0.0)
        )

        self.active_pair_mask = den > 0.0
        np.fill_diagonal(self.active_pair_mask, False)
        self.sim_matrix = np.where(self.active_pair_mask, num / np.maximum(den, 1e-12), 0.0)
        np.fill_diagonal(self.sim_matrix, 0.0)

    def recommend_similar_items_with_meta(
        self, product_id: str, top_k: int = 5
    ) -> dict[str, Any]:
        data, t_idx = self._require_fitted_target(product_id, top_k)
        q_has_img = bool(data.image_valid[t_idx])

        if (not q_has_img) and self.fallback_missing_query_to_v1 and (data.v1_text_sim_matrix is not None):
            recs = _rank_deterministic(
                data.v1_text_sim_matrix[t_idx], data.product_ids, t_idx, top_k=top_k
            )
            return {
                "reference_product_id": product_id,
                "model": self.model_name,
                "query_image_valid": False,
                "fallback_used": True,
                "fallback_mode": "exact_v1_tfidf",
                "recommendations": recs,
            }

        tie_break = (
            data.v1_text_sim_matrix[t_idx]
            if data.v1_text_sim_matrix is not None
            else np.zeros(len(data.product_ids), dtype=np.float64)
        )
        recs = _rank_deterministic(
            self.sim_matrix[t_idx],
            data.product_ids,
            t_idx,
            top_k=top_k,
            valid_candidate_mask=self.active_pair_mask[t_idx],
            tie_break_scores=tie_break,
        )
        return {
            "reference_product_id": product_id,
            "model": self.model_name,
            "query_image_valid": q_has_img,
            "fallback_used": False,
            "fallback_mode": None,
            "recommendations": recs,
        }


class CLIPLateFusionRecommender(BaseCLIPRecommender):
    """Modality-normalized, non-attenuating gated late fusion for CLIP embeddings.

    Combines a base text similarity matrix (`text_source="v1_tfidf"` or `"clip_text"`)
    with CLIP image-to-image cosine similarity `S_vis(i, j)`:
    - When `image_valid[i] == False` or `image_valid[j] == False`, or when the
      semantic gate is inactive (`S_text(i, j) < text_gate_threshold` or category
      mismatch when `require_same_category_for_visual=True`), only the text modality
      is active (`w_text=1.0, w_vis=0.0`), so `S_late(i, j) = S_text(i, j)` with
      zero attenuation.
    - When the gate is active, `S_vis` is mapped into `[tau, 1.0]` via
      `tau + (1.0 - tau) * clip(S_vis, 0.0, 1.0)` and fused with weights
      `(1.0 - w_vis) * S_text + w_vis * S_vis_mapped`.
    """

    model_name = "clip_late_fusion"

    def __init__(
        self,
        w_vis: float = 0.15,
        text_gate_threshold: float = 0.05,
        require_same_category_for_visual: bool = False,
        text_source: str = "v1_tfidf",
    ) -> None:
        super().__init__()
        if not (0.0 <= w_vis <= 1.0):
            raise ValueError("w_vis must be in [0.0, 1.0]")
        if not (0.0 <= text_gate_threshold < 1.0):
            raise ValueError("text_gate_threshold must be in [0.0, 1.0)")
        if text_source not in {"v1_tfidf", "clip_text"}:
            raise ValueError("text_source must be 'v1_tfidf' or 'clip_text'")
        self.w_vis = float(w_vis)
        self.text_gate_threshold = float(text_gate_threshold)
        self.require_same_category_for_visual = bool(require_same_category_for_visual)
        self.text_source = text_source
        self.sim_matrix: np.ndarray = np.array([], dtype=np.float64)

    def _fit_impl(self, data: OfflineCLIPData) -> None:
        if self.text_source == "v1_tfidf":
            if data.v1_text_sim_matrix is None:
                raise ValueError(
                    "v1_text_sim_matrix is required when text_source='v1_tfidf'"
                )
            text_sim = data.v1_text_sim_matrix
        else:
            text_sim = _compute_masked_cosine_matrix(
                data.text_embeddings,
                data.text_embeddings,
                data.text_valid,
                data.text_valid,
            )

        vis_sim = _compute_masked_cosine_matrix(
            data.image_embeddings,
            data.image_embeddings,
            data.image_valid,
            data.image_valid,
        )

        both_imaged = np.outer(data.image_valid, data.image_valid)
        gate_active = both_imaged & (text_sim >= self.text_gate_threshold)
        if self.require_same_category_for_visual:
            if data.categories is None:
                raise ValueError(
                    "categories are required when require_same_category_for_visual=True"
                )
            cat_arr = np.array(data.categories)
            same_cat = cat_arr[:, np.newaxis] == cat_arr[np.newaxis, :]
            gate_active = gate_active & same_cat

        clipped_vis = np.clip(vis_sim, 0.0, 1.0)
        mapped_vis = self.text_gate_threshold + (
            1.0 - self.text_gate_threshold
        ) * clipped_vis
        fused = (1.0 - self.w_vis) * text_sim + self.w_vis * mapped_vis

        self.sim_matrix = np.where(gate_active, fused, text_sim).astype(np.float64)
        np.fill_diagonal(self.sim_matrix, 0.0)

    def recommend_similar_items_with_meta(
        self, product_id: str, top_k: int = 5
    ) -> dict[str, Any]:
        data, t_idx = self._require_fitted_target(product_id, top_k)
        q_has_img = bool(data.image_valid[t_idx])
        recs = _rank_deterministic(
            self.sim_matrix[t_idx], data.product_ids, t_idx, top_k=top_k
        )
        return {
            "reference_product_id": product_id,
            "model": self.model_name,
            "query_image_valid": q_has_img,
            "fallback_used": not q_has_img,
            "fallback_mode": self.text_source if not q_has_img else None,
            "recommendations": recs,
        }
