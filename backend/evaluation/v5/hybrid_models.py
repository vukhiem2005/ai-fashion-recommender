"""Configurable Offline Hybrid Recommender for ELAN V5 (Milestone 5).

Integrates three isolated V5 recommendation signals over the shared 328-product catalog:
1. Content-Based / Multimodal: Phase 4B `CLIPLateFusionRecommender` (`text_source="v1_tfidf"`,
   `w_vis=0.5`, `tau=0.0`, `same_cat=False`) with exact V1 TF-IDF fallback for `HM-0398947001`.
2. Collaborative Filtering: Milestone 2 `ItemKNNRecommender` (`log_sum`, `iuf_exponent=0.5`,
   `shrinkage=20.0`, `k_neighbors=100`).
3. Association Rule Mining: Milestone 3 `AprioriAssociationRuleMiner` (`min_support_count=2`,
   `min_confidence=0.05`, `min_lift=1.05`, `beta_lift=1.0`) activated only when basket/cart
   context is provided and matching rules exist.

Normalization & Fusion Strategy:
- Raw scores across heterogeneous modalities (cosine similarity, weighted KNN sums,
  smoothed confidence * lift) are NOT added directly.
- Each active modality is min-max normalized into `[0.0, 1.0]` over the eligible
  candidate set (if `max == min`, the modality carries zero discrimination and is marked
  inactive).
- Active normalized scores are combined with modality-normalized weights summing to `1.0`.
- Zero runtime PyTorch, Transformers, image decoding, or SQLite queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from backend.evaluation.v5.arm_models import AprioriAssociationRuleMiner
from backend.evaluation.v5.cf_models import ItemKNNRecommender
from backend.evaluation.v5.clip_recommenders import (
    CLIPLateFusionRecommender,
    OfflineCLIPData,
)


@dataclass(frozen=True)
class HybridFusionConfig:
    """Fixed default weights for modality-normalized hybrid score fusion."""

    w_content: float = 0.50
    w_cf: float = 0.35
    w_arm: float = 0.15


def _minmax_normalize_eligible(
    raw_scores: np.ndarray,
    eligible_mask: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Min-max normalize `raw_scores` into `[0.0, 1.0]` over `eligible_mask`.

    Returns `(normalized_scores, is_informative)`. If no eligible candidates exist or
    all eligible scores are identical (`span <= 1e-12`), returns zeros and `False`
    so the uninformative modality is not given artificial weight.
    """
    n_items = raw_scores.shape[0]
    out = np.zeros(n_items, dtype=np.float64)
    if not np.any(eligible_mask):
        return out, False

    vals = raw_scores[eligible_mask]
    v_min = float(np.min(vals))
    v_max = float(np.max(vals))
    span = v_max - v_min
    if span <= 1e-12:
        return out, False

    out[eligible_mask] = (vals - v_min) / span
    return out, True


class HybridRecommender:
    """Unified, deterministic offline hybrid recommender combining Content, CF, and ARM."""

    def __init__(
        self,
        clip_late_fusion: CLIPLateFusionRecommender,
        cf_model: ItemKNNRecommender,
        arm_model: AprioriAssociationRuleMiner,
        clip_data: OfflineCLIPData,
        config: HybridFusionConfig | None = None,
    ) -> None:
        self.clip_late = clip_late_fusion
        self.cf_model = cf_model
        self.arm_model = arm_model
        self.clip_data = clip_data
        self.config = config or HybridFusionConfig()

        # Verify strict product_id alignment across all integrated components
        if clip_data.product_ids != cf_model.product_ids:
            raise ValueError("Product ID alignment mismatch between CLIP and CF models")
        if clip_data.v1_text_sim_matrix is None:
            raise ValueError("OfflineCLIPData must include v1_text_sim_matrix")

        self.product_ids: tuple[str, ...] = clip_data.product_ids
        self.product_to_idx: dict[str, int] = clip_data.product_to_idx
        self.n_items: int = len(self.product_ids)
        self.content_sim_matrix: np.ndarray = clip_late_fusion.sim_matrix
        self.v1_sim_matrix: np.ndarray = clip_data.v1_text_sim_matrix
        self.fallback_popularity: np.ndarray = cf_model.fallback_popularity

    def _raw_content_scores(
        self,
        anchor_product_id: str | None = None,
        user_profile: np.ndarray | None = None,
        use_v1_only: bool = False,
    ) -> tuple[np.ndarray, bool]:
        """Compute raw content scores from anchor item and/or user interaction profile."""
        sim_mat = self.v1_sim_matrix if use_v1_only else self.content_sim_matrix
        if anchor_product_id is not None:
            if anchor_product_id not in self.product_to_idx:
                raise KeyError(f"Unknown anchor_product_id: {anchor_product_id}")
            t_idx = self.product_to_idx[anchor_product_id]
            return sim_mat[t_idx].copy(), True

        if user_profile is not None and float(np.linalg.norm(user_profile)) > 1e-12:
            raw = (user_profile.astype(np.float64) @ sim_mat) / max(
                float(np.sum(np.abs(user_profile))), 1e-12
            )
            return raw, True

        return np.zeros(self.n_items, dtype=np.float64), False

    def _raw_cf_scores(
        self,
        anchor_product_id: str | None = None,
        user_profile: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool]:
        """Compute raw ItemKNN CF scores when valid user history or anchor support exists."""
        if user_profile is not None and float(np.linalg.norm(user_profile)) > 1e-12:
            raw = (user_profile.astype(np.float64) @ self.cf_model.sim_topk)
            return raw, bool(float(np.max(raw)) > 1e-12)

        if anchor_product_id is not None:
            t_idx = self.product_to_idx[anchor_product_id]
            raw = self.cf_model.sim_topk[t_idx].copy()
            return raw, bool(float(np.max(raw)) > 1e-12)

        return np.zeros(self.n_items, dtype=np.float64), False

    def _raw_arm_scores(
        self,
        basket_product_ids: Sequence[str] | None,
    ) -> tuple[np.ndarray, bool]:
        """Compute raw Apriori association rule scores when basket context triggers rules."""
        if not basket_product_ids:
            return np.zeros(self.n_items, dtype=np.float64), False

        unique_basket = tuple(dict.fromkeys(str(p) for p in basket_product_ids if str(p) in self.product_to_idx))
        if not unique_basket:
            return np.zeros(self.n_items, dtype=np.float64), False

        # Score all rule-matched candidates using Milestone 3 miner's exact scoring formula
        arm_out = self.arm_model.recommend_for_basket(
            unique_basket, top_k=self.n_items, allow_fallback=False
        )
        rule_recs = arm_out.get("recommendations", [])
        if not rule_recs:
            return np.zeros(self.n_items, dtype=np.float64), False

        raw = np.zeros(self.n_items, dtype=np.float64)
        for item in rule_recs:
            pid = str(item["product_id"])
            if pid in self.product_to_idx:
                raw[self.product_to_idx[pid]] = float(item["score"])
        return raw, bool(float(np.max(raw)) > 1e-12)

    def recommend(
        self,
        *,
        anchor_product_id: str | None = None,
        user_profile: np.ndarray | None = None,
        seen_mask_row: np.ndarray | None = None,
        basket_product_ids: Sequence[str] | None = None,
        top_k: int = 5,
        exclude_seen: bool = True,
        enable_content: bool = True,
        enable_cf: bool = True,
        enable_arm: bool = True,
        use_v1_content_only: bool = False,
    ) -> dict[str, Any]:
        """Generate deduplicated, deterministic Top-K hybrid recommendations."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        # Build eligible candidate mask (exclude anchor, basket items, and seen items)
        eligible = np.ones(self.n_items, dtype=bool)
        if anchor_product_id is not None:
            if anchor_product_id not in self.product_to_idx:
                raise KeyError(f"Unknown anchor_product_id: {anchor_product_id}")
            eligible[self.product_to_idx[anchor_product_id]] = False

        if basket_product_ids:
            for pid in basket_product_ids:
                if pid in self.product_to_idx:
                    eligible[self.product_to_idx[pid]] = False

        if exclude_seen and seen_mask_row is not None:
            eligible = eligible & (~seen_mask_row.astype(bool))

        # 1. Compute raw modality signals
        raw_content, has_content = (
            self._raw_content_scores(
                anchor_product_id=anchor_product_id,
                user_profile=user_profile,
                use_v1_only=use_v1_content_only,
            )
            if enable_content
            else (np.zeros(self.n_items, dtype=np.float64), False)
        )
        raw_cf, has_cf = (
            self._raw_cf_scores(
                anchor_product_id=anchor_product_id,
                user_profile=user_profile,
            )
            if enable_cf
            else (np.zeros(self.n_items, dtype=np.float64), False)
        )
        raw_arm, has_arm = (
            self._raw_arm_scores(basket_product_ids)
            if enable_arm
            else (np.zeros(self.n_items, dtype=np.float64), False)
        )

        # 2. Min-max normalize active modalities over eligible candidates
        norm_content, inf_content = (
            _minmax_normalize_eligible(raw_content, eligible)
            if has_content
            else (np.zeros(self.n_items, dtype=np.float64), False)
        )
        norm_cf, inf_cf = (
            _minmax_normalize_eligible(raw_cf, eligible)
            if has_cf
            else (np.zeros(self.n_items, dtype=np.float64), False)
        )
        norm_arm, inf_arm = (
            _minmax_normalize_eligible(raw_arm, eligible)
            if has_arm
            else (np.zeros(self.n_items, dtype=np.float64), False)
        )

        # 3. Modality-normalized weighted fusion (or cold-start popularity fallback)
        active_modalities: list[str] = []
        total_w = 0.0
        fused = np.zeros(self.n_items, dtype=np.float64)

        if inf_content and self.config.w_content > 0.0:
            fused += self.config.w_content * norm_content
            total_w += self.config.w_content
            active_modalities.append("content")
        if inf_cf and self.config.w_cf > 0.0:
            fused += self.config.w_cf * norm_cf
            total_w += self.config.w_cf
            active_modalities.append("cf")
        if inf_arm and self.config.w_arm > 0.0:
            fused += self.config.w_arm * norm_arm
            total_w += self.config.w_arm
            active_modalities.append("arm")

        cold_start_fallback = False
        if total_w > 0.0:
            final_scores = fused / total_w
        else:
            norm_pop, _ = _minmax_normalize_eligible(self.fallback_popularity, eligible)
            final_scores = norm_pop
            cold_start_fallback = True
            active_modalities.append("popularity_fallback")

        # 4. Deterministic ranking & tie-breaking: (-round(score, 12), -popularity, product_id)
        cand_indices = [idx for idx in range(self.n_items) if bool(eligible[idx])]
        cand_indices.sort(
            key=lambda idx: (
                -round(float(final_scores[idx]), 12),
                -round(float(self.fallback_popularity[idx]), 12),
                self.product_ids[idx],
            )
        )
        recs = [(self.product_ids[idx], float(final_scores[idx])) for idx in cand_indices[:top_k]]

        missing_image_fallback = False
        if anchor_product_id is not None:
            a_idx = self.product_to_idx[anchor_product_id]
            missing_image_fallback = not bool(self.clip_data.image_valid[a_idx])

        return {
            "recommendations": recs,
            "active_modalities": active_modalities,
            "cold_start_fallback": cold_start_fallback,
            "missing_image_fallback": missing_image_fallback,
            "arm_activated": "arm" in active_modalities,
            "cf_activated": "cf" in active_modalities,
            "content_activated": "content" in active_modalities,
        }
