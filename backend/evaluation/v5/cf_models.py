"""Collaborative Filtering models and baselines for ELAN V5 (Milestone 2).

Implements:
1. User-item implicit interaction matrix builder (max_weight / log_sum deduplication,
   Inverse User Frequency weighting, user L2 normalization, and strict seen-item masks).
2. PopularityBaseline (damped/total train implicit weight ranker).
3. V1TFIDFBaseline (exact V1 TF-IDF + cosine similarity for item-to-item and
   user-profile centroid ranking).
4. ItemKNNRecommender (Item-Based Collaborative Filtering with cosine similarity,
   co-occurrence shrinkage, and Top-K neighbor pruning).
5. TruncatedSVDRecommender (Low-Rank Matrix Factorization via Scikit-learn TruncatedSVD).

Safety & Isolation:
- Reads `backend/data/elan_release.db` strictly in SQLite URI read-only mode (`mode=ro`).
- Never imports or modifies production V1 API endpoints or database records.
- Never reads quarantined generator audit files (`backend/evaluation/v5/audit/*`).
"""

from __future__ import annotations

import sqlite3
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RELEASE_DB = REPO_ROOT / "backend" / "data" / "elan_release.db"
DEFAULT_SEED = 20260501

EVENT_WEIGHTS: dict[str, float] = {
    "view": 1.0,
    "add_to_cart": 3.0,
    "purchase": 5.0,
}


@dataclass(frozen=True)
class MatrixBundle:
    """Container for fitted train matrix statistics and user/item index mappings."""

    user_ids: tuple[str, ...]
    product_ids: tuple[str, ...]
    user_to_idx: dict[str, int]
    product_to_idx: dict[str, int]
    raw_weights: np.ndarray          # Shape (U, I), deduplicated raw weights W
    normalized_matrix: np.ndarray    # Shape (U, I), IUF + L2 normalized matrix R
    seen_mask: np.ndarray            # Shape (U, I), bool mask (W > 0)
    iuf_weights: np.ndarray          # Shape (I,), fitted on train only
    dedup_mode: str
    iuf_exponent: float
    l2_normalize_users: bool


def load_active_catalog_readonly(db_path: Path = DEFAULT_RELEASE_DB) -> list[dict[str, Any]]:
    """Load the 328 active products and colors from SQLite in strict mode=ro."""
    if not db_path.is_file():
        raise FileNotFoundError(f"Release SQLite database not found: {db_path}")

    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, slug, name, category, description, price_cents
            FROM product
            WHERE is_active = 1
            ORDER BY id
            """
        ).fetchall()
        variant_rows = conn.execute(
            """
            SELECT product_id, color, image_url
            FROM productvariant
            ORDER BY sku
            """
        ).fetchall()

    colors: dict[str, set[str]] = defaultdict(set)
    images: dict[str, str] = {}
    for v in variant_rows:
        pid = v["product_id"]
        c = (v["color"] or "").strip()
        if c:
            colors[pid].add(c)
        if v["image_url"] and pid not in images:
            images[pid] = v["image_url"]

    return [
        {
            **dict(r),
            "colors": sorted(colors[r["id"]]),
            "image_url": images.get(r["id"]),
        }
        for r in rows
    ]


def _aggregate_raw_weights(
    interactions: list[dict[str, str]],
    user_to_idx: dict[str, int],
    product_to_idx: dict[str, int],
    dedup_mode: str,
) -> np.ndarray:
    """Aggregate raw interaction rows into a (U, I) matrix using max_weight or log_sum."""
    n_users = len(user_to_idx)
    n_items = len(product_to_idx)
    mat = np.zeros((n_users, n_items), dtype=np.float64)

    if dedup_mode not in ("max_weight", "log_sum"):
        raise ValueError(f"Unsupported dedup_mode: {dedup_mode}")

    for row in interactions:
        uid = row["user_id"]
        pid = row["product_id"]
        if uid not in user_to_idx or pid not in product_to_idx:
            raise KeyError(f"Unknown user_id={uid} or product_id={pid} in interaction row")
        u_idx = user_to_idx[uid]
        i_idx = product_to_idx[pid]
        w = float(row.get("implicit_weight") or EVENT_WEIGHTS[row["event_type"]])
        if dedup_mode == "max_weight":
            if w > mat[u_idx, i_idx]:
                mat[u_idx, i_idx] = w
        else:
            mat[u_idx, i_idx] += w

    if dedup_mode == "log_sum":
        nonzero = mat > 0.0
        mat[nonzero] = np.log1p(mat[nonzero])

    return mat


def build_train_matrix(
    train_interactions: list[dict[str, str]],
    user_ids: list[str],
    product_ids: list[str],
    dedup_mode: str = "max_weight",
    iuf_exponent: float = 0.5,
    l2_normalize_users: bool = True,
) -> MatrixBundle:
    """Build user-item matrix strictly from training split interactions."""
    for row in train_interactions:
        if row.get("split") != "train":
            raise ValueError(
                f"Data leakage error: non-train split '{row.get('split')}' passed to build_train_matrix"
            )

    sorted_users = tuple(sorted(user_ids))
    sorted_products = tuple(sorted(product_ids))
    user_to_idx = {u: idx for idx, u in enumerate(sorted_users)}
    product_to_idx = {p: idx for idx, p in enumerate(sorted_products)}

    raw_weights = _aggregate_raw_weights(
        train_interactions, user_to_idx, product_to_idx, dedup_mode=dedup_mode
    )
    seen_mask = raw_weights > 0.0

    n_users = len(sorted_users)
    doc_freq = np.sum(seen_mask, axis=0).astype(np.float64)
    base_iuf = np.log((1.0 + n_users) / (1.0 + doc_freq)) + 1.0
    iuf_weights = np.power(base_iuf, float(iuf_exponent))

    weighted = raw_weights * iuf_weights[np.newaxis, :]
    if l2_normalize_users:
        norms = np.linalg.norm(weighted, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        normalized = weighted / norms
    else:
        normalized = weighted

    return MatrixBundle(
        user_ids=sorted_users,
        product_ids=sorted_products,
        user_to_idx=user_to_idx,
        product_to_idx=product_to_idx,
        raw_weights=raw_weights,
        normalized_matrix=normalized,
        seen_mask=seen_mask,
        iuf_weights=iuf_weights,
        dedup_mode=dedup_mode,
        iuf_exponent=float(iuf_exponent),
        l2_normalize_users=l2_normalize_users,
    )


def build_conditioning_profiles(
    conditioning_interactions: list[dict[str, str]],
    train_bundle: MatrixBundle,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform arbitrary conditioning history (e.g., train or train+val) using frozen train IUF.

    Does NOT refit IUF weights or model parameters. Returns (normalized_profiles, seen_mask).
    """
    raw_weights = _aggregate_raw_weights(
        conditioning_interactions,
        train_bundle.user_to_idx,
        train_bundle.product_to_idx,
        dedup_mode=train_bundle.dedup_mode,
    )
    seen_mask = raw_weights > 0.0
    weighted = raw_weights * train_bundle.iuf_weights[np.newaxis, :]
    if train_bundle.l2_normalize_users:
        norms = np.linalg.norm(weighted, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        normalized = weighted / norms
    else:
        normalized = weighted
    return normalized, seen_mask


def rank_top_k_from_scores(
    scores: np.ndarray,
    product_ids: tuple[str, ...],
    exclude_mask: np.ndarray | None,
    top_k: int = 5,
    fallback_scores: np.ndarray | None = None,
) -> list[tuple[str, float]]:
    """Deterministically rank Top-K items descending by score, breaking ties by ascending product_id."""
    n_items = len(product_ids)
    effective_scores = np.array(scores, dtype=np.float64, copy=True)

    # If the user received all-zero scores (e.g., cold-start user), use fallback popularity scores
    if fallback_scores is not None:
        valid_mask = ~exclude_mask if exclude_mask is not None else np.ones(n_items, dtype=bool)
        if np.all(np.abs(effective_scores[valid_mask]) <= 1e-12):
            effective_scores = np.array(fallback_scores, dtype=np.float64, copy=True)

    candidate_indices = [
        idx
        for idx in range(n_items)
        if exclude_mask is None or not bool(exclude_mask[idx])
    ]
    candidate_indices.sort(
        key=lambda idx: (-round(float(effective_scores[idx]), 12), product_ids[idx])
    )
    return [
        (product_ids[idx], float(effective_scores[idx]))
        for idx in candidate_indices[:top_k]
    ]


class PopularityBaseline:
    """Non-personalized popularity baseline ranked by total train implicit interaction weight."""

    def __init__(self) -> None:
        self.product_ids: tuple[str, ...] = ()
        self.product_to_idx: dict[str, int] = {}
        self.item_popularity: np.ndarray = np.array([], dtype=np.float64)
        self.fit_time_seconds: float = 0.0

    def fit(self, bundle: MatrixBundle) -> "PopularityBaseline":
        t0 = time.perf_counter()
        self.product_ids = bundle.product_ids
        self.product_to_idx = bundle.product_to_idx
        totals = np.sum(bundle.raw_weights, axis=0).astype(np.float64)
        max_tot = float(np.max(totals)) if totals.size > 0 else 1.0
        self.item_popularity = totals / max(max_tot, 1e-12)
        self.fit_time_seconds = time.perf_counter() - t0
        return self

    def score_all_users(self, user_profiles: np.ndarray) -> np.ndarray:
        n_users = user_profiles.shape[0]
        return np.tile(self.item_popularity, (n_users, 1))

    def recommend_for_user(
        self,
        user_profile: np.ndarray,
        seen_mask_row: np.ndarray | None = None,
        top_k: int = 5,
        exclude_seen: bool = True,
    ) -> list[tuple[str, float]]:
        mask = seen_mask_row if exclude_seen else None
        return rank_top_k_from_scores(
            self.item_popularity, self.product_ids, mask, top_k=top_k
        )

    def recommend_similar_items(
        self, product_id: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        if product_id not in self.product_to_idx:
            raise KeyError(f"Unknown product_id: {product_id}")
        target_idx = self.product_to_idx[product_id]
        exclude = np.zeros(len(self.product_ids), dtype=bool)
        exclude[target_idx] = True
        return rank_top_k_from_scores(
            self.item_popularity, self.product_ids, exclude, top_k=top_k
        )


class V1TFIDFBaseline:
    """Content-based baseline using exact V1 TF-IDF document representation and cosine similarity."""

    def __init__(self) -> None:
        self.product_ids: tuple[str, ...] = ()
        self.product_to_idx: dict[str, int] = {}
        self.tfidf_matrix: np.ndarray = np.array([], dtype=np.float64)
        self.item_sim_matrix: np.ndarray = np.array([], dtype=np.float64)
        self.fallback_popularity: np.ndarray = np.array([], dtype=np.float64)
        self.fit_time_seconds: float = 0.0

    @staticmethod
    def _document(product: dict[str, Any]) -> str:
        category = str(product["category"] or "").strip()
        name = str(product["name"] or "").strip()
        description = str(product["description"] or "").strip()
        colors = " ".join(product["colors"])
        return " ".join(
            [category] * 3 + [name] * 2 + [description] + [colors] * 2
        ).strip()

    def fit(
        self, catalog: list[dict[str, Any]], bundle: MatrixBundle
    ) -> "V1TFIDFBaseline":
        t0 = time.perf_counter()
        catalog_by_id = {p["id"]: p for p in catalog}
        self.product_ids = bundle.product_ids
        self.product_to_idx = bundle.product_to_idx

        docs = [self._document(catalog_by_id[pid]) for pid in self.product_ids]
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        sparse_mat = vectorizer.fit_transform(docs)
        self.tfidf_matrix = sparse_mat.toarray().astype(np.float64)
        self.item_sim_matrix = cosine_similarity(sparse_mat).astype(np.float64)
        np.fill_diagonal(self.item_sim_matrix, 0.0)

        totals = np.sum(bundle.raw_weights, axis=0).astype(np.float64)
        self.fallback_popularity = totals / max(float(np.max(totals)), 1e-12)
        self.fit_time_seconds = time.perf_counter() - t0
        return self

    def score_all_users(self, user_profiles: np.ndarray) -> np.ndarray:
        user_centroids = user_profiles @ self.tfidf_matrix
        norms = np.linalg.norm(user_centroids, axis=1, keepdims=True)
        valid = (norms.ravel() > 1e-12)
        scores = np.zeros((user_profiles.shape[0], len(self.product_ids)), dtype=np.float64)
        if np.any(valid):
            normalized_centroids = user_centroids[valid] / norms[valid]
            scores[valid] = normalized_centroids @ self.tfidf_matrix.T
        if np.any(~valid):
            scores[~valid] = self.fallback_popularity[np.newaxis, :]
        return scores

    def recommend_for_user(
        self,
        user_profile: np.ndarray,
        seen_mask_row: np.ndarray | None = None,
        top_k: int = 5,
        exclude_seen: bool = True,
    ) -> list[tuple[str, float]]:
        scores = self.score_all_users(user_profile[np.newaxis, :])[0]
        mask = seen_mask_row if exclude_seen else None
        return rank_top_k_from_scores(
            scores,
            self.product_ids,
            mask,
            top_k=top_k,
            fallback_scores=self.fallback_popularity,
        )

    def recommend_similar_items(
        self, product_id: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        if product_id not in self.product_to_idx:
            raise KeyError(f"Unknown product_id: {product_id}")
        target_idx = self.product_to_idx[product_id]
        exclude = np.zeros(len(self.product_ids), dtype=bool)
        exclude[target_idx] = True
        return rank_top_k_from_scores(
            self.item_sim_matrix[target_idx],
            self.product_ids,
            exclude,
            top_k=top_k,
        )


class ItemKNNRecommender:
    """Item-Based Collaborative Filtering using cosine similarity, shrinkage, and Top-K pruning."""

    def __init__(
        self,
        k_neighbors: int = 30,
        shrinkage: float = 10.0,
        normalize_sim_columns: bool = False,
    ) -> None:
        self.k_neighbors = int(k_neighbors)
        self.shrinkage = float(shrinkage)
        self.normalize_sim_columns = bool(normalize_sim_columns)
        self.product_ids: tuple[str, ...] = ()
        self.product_to_idx: dict[str, int] = {}
        self.sim_shrunk: np.ndarray = np.array([], dtype=np.float64)
        self.sim_topk: np.ndarray = np.array([], dtype=np.float64)
        self.fallback_popularity: np.ndarray = np.array([], dtype=np.float64)
        self.fallback_content_sim: np.ndarray | None = None
        self.item_support: np.ndarray = np.array([], dtype=np.float64)
        self.fit_time_seconds: float = 0.0
        self.peak_memory_bytes: int = 0

    def fit(
        self,
        bundle: MatrixBundle,
        fallback_content_sim: np.ndarray | None = None,
    ) -> "ItemKNNRecommender":
        tracemalloc.start()
        t0 = time.perf_counter()

        self.product_ids = bundle.product_ids
        self.product_to_idx = bundle.product_to_idx
        self.fallback_content_sim = fallback_content_sim

        # 1. Raw item-item cosine similarity on normalized train matrix R (shape U x I)
        raw_cos = cosine_similarity(bundle.normalized_matrix.T).astype(np.float64)
        np.fill_diagonal(raw_cos, 0.0)

        # 2. Co-interaction count matrix N_ij = sum_u I(W_ui > 0 and W_uj > 0)
        binary_seen = bundle.seen_mask.astype(np.float64)
        co_counts = binary_seen.T @ binary_seen
        np.fill_diagonal(co_counts, 0.0)
        self.item_support = np.sum(binary_seen, axis=0)

        # 3. Shrinkage regularization
        if self.shrinkage > 0.0:
            shrink_factor = co_counts / (co_counts + self.shrinkage)
            self.sim_shrunk = raw_cos * shrink_factor
        else:
            self.sim_shrunk = raw_cos.copy()
        np.fill_diagonal(self.sim_shrunk, 0.0)

        # 4. Top-K neighborhood pruning per target column j
        n_items = len(self.product_ids)
        k = min(self.k_neighbors, n_items - 1)
        sim_topk = np.zeros_like(self.sim_shrunk)
        if k > 0:
            for j in range(n_items):
                col = self.sim_shrunk[:, j]
                top_idx = np.argpartition(col, -k)[-k:]
                for idx in top_idx:
                    if col[idx] > 0.0:
                        sim_topk[idx, j] = col[idx]

        if self.normalize_sim_columns:
            col_sums = np.sum(sim_topk, axis=0, keepdims=True)
            col_sums = np.where(col_sums > 1e-12, col_sums, 1.0)
            sim_topk = sim_topk / col_sums

        self.sim_topk = sim_topk
        totals = np.sum(bundle.raw_weights, axis=0).astype(np.float64)
        self.fallback_popularity = totals / max(float(np.max(totals)), 1e-12)

        self.fit_time_seconds = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_memory_bytes = int(peak_bytes)
        return self

    def score_all_users(self, user_profiles: np.ndarray) -> np.ndarray:
        scores = user_profiles @ self.sim_topk
        row_norms = np.linalg.norm(user_profiles, axis=1)
        cold_users = row_norms <= 1e-12
        if np.any(cold_users):
            scores[cold_users] = self.fallback_popularity[np.newaxis, :]
        return scores

    def recommend_for_user(
        self,
        user_profile: np.ndarray,
        seen_mask_row: np.ndarray | None = None,
        top_k: int = 5,
        exclude_seen: bool = True,
    ) -> list[tuple[str, float]]:
        scores = self.score_all_users(user_profile[np.newaxis, :])[0]
        mask = seen_mask_row if exclude_seen else None
        return rank_top_k_from_scores(
            scores,
            self.product_ids,
            mask,
            top_k=top_k,
            fallback_scores=self.fallback_popularity,
        )

    def recommend_similar_items(
        self, product_id: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        if product_id not in self.product_to_idx:
            raise KeyError(f"Unknown product_id: {product_id}")
        target_idx = self.product_to_idx[product_id]
        exclude = np.zeros(len(self.product_ids), dtype=bool)
        exclude[target_idx] = True

        row_scores = self.sim_shrunk[target_idx]
        fallback = (
            self.fallback_content_sim[target_idx]
            if self.fallback_content_sim is not None
            else self.fallback_popularity
        )
        return rank_top_k_from_scores(
            row_scores,
            self.product_ids,
            exclude,
            top_k=top_k,
            fallback_scores=fallback,
        )


class TruncatedSVDRecommender:
    """Low-Rank Matrix Factorization using Scikit-learn TruncatedSVD.

    Mathematical Note on `singular_value_power`:
    - Track A (personalized user ranking via `score_all_users`) projects user
      history vectors onto the rank-k orthonormal item subspace:
          y_hat_u = (r_u @ V_k) @ V_k.T
      Because V_k has orthonormal columns, this folding-in projection is
      invariant to `singular_value_power`.
    - Track B (item-to-item similarity via `recommend_similar_items`) computes
      cosine similarity between singular-value-weighted item embeddings
          F = V_k @ diag(sigma_k ** singular_value_power)
      where `singular_value_power` controls how strongly leading singular
      axes weight item-to-item cosine similarity.
    """

    def __init__(
        self,
        n_components: int = 24,
        singular_value_power: float = 0.5,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.n_components = int(n_components)
        self.singular_value_power = float(singular_value_power)
        self.seed = int(seed)
        self.product_ids: tuple[str, ...] = ()
        self.product_to_idx: dict[str, int] = {}
        self.item_factors: np.ndarray = np.array([], dtype=np.float64)    # Shape (I, k)
        self.singular_values: np.ndarray = np.array([], dtype=np.float64) # Shape (k,)
        self.latent_item_sim: np.ndarray = np.array([], dtype=np.float64) # Shape (I, I)
        self.fallback_popularity: np.ndarray = np.array([], dtype=np.float64)
        self.fallback_content_sim: np.ndarray | None = None
        self.explained_variance_ratio_sum: float = 0.0
        self.fit_time_seconds: float = 0.0
        self.peak_memory_bytes: int = 0

    def fit(
        self,
        bundle: MatrixBundle,
        fallback_content_sim: np.ndarray | None = None,
    ) -> "TruncatedSVDRecommender":
        tracemalloc.start()
        t0 = time.perf_counter()

        self.product_ids = bundle.product_ids
        self.product_to_idx = bundle.product_to_idx
        self.fallback_content_sim = fallback_content_sim

        max_rank = min(bundle.normalized_matrix.shape) - 1
        k = max(1, min(self.n_components, max_rank))
        svd = TruncatedSVD(
            n_components=k,
            algorithm="randomized",
            n_iter=15,
            random_state=self.seed,
        )
        svd.fit(bundle.normalized_matrix)

        self.item_factors = svd.components_.T.astype(np.float64)  # (I, k)
        self.singular_values = svd.singular_values_.astype(np.float64)
        self.explained_variance_ratio_sum = float(np.sum(svd.explained_variance_ratio_))

        scaled_factors = self.item_factors * np.power(
            np.maximum(self.singular_values, 1e-12), self.singular_value_power
        )[np.newaxis, :]
        self.latent_item_sim = cosine_similarity(scaled_factors).astype(np.float64)
        np.fill_diagonal(self.latent_item_sim, 0.0)

        totals = np.sum(bundle.raw_weights, axis=0).astype(np.float64)
        self.fallback_popularity = totals / max(float(np.max(totals)), 1e-12)

        self.fit_time_seconds = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_memory_bytes = int(peak_bytes)
        return self

    def score_all_users(self, user_profiles: np.ndarray) -> np.ndarray:
        latent_users = user_profiles @ self.item_factors
        scores = latent_users @ self.item_factors.T
        row_norms = np.linalg.norm(user_profiles, axis=1)
        cold_users = row_norms <= 1e-12
        if np.any(cold_users):
            scores[cold_users] = self.fallback_popularity[np.newaxis, :]
        return scores

    def recommend_for_user(
        self,
        user_profile: np.ndarray,
        seen_mask_row: np.ndarray | None = None,
        top_k: int = 5,
        exclude_seen: bool = True,
    ) -> list[tuple[str, float]]:
        scores = self.score_all_users(user_profile[np.newaxis, :])[0]
        mask = seen_mask_row if exclude_seen else None
        return rank_top_k_from_scores(
            scores,
            self.product_ids,
            mask,
            top_k=top_k,
            fallback_scores=self.fallback_popularity,
        )

    def recommend_similar_items(
        self, product_id: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        if product_id not in self.product_to_idx:
            raise KeyError(f"Unknown product_id: {product_id}")
        target_idx = self.product_to_idx[product_id]
        exclude = np.zeros(len(self.product_ids), dtype=bool)
        exclude[target_idx] = True

        row_scores = self.latent_item_sim[target_idx]
        fallback = (
            self.fallback_content_sim[target_idx]
            if self.fallback_content_sim is not None
            else self.fallback_popularity
        )
        return rank_top_k_from_scores(
            row_scores,
            self.product_ids,
            exclude,
            top_k=top_k,
            fallback_scores=fallback,
        )
