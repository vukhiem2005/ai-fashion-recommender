"""Chronological evaluation and benchmark runner for ELAN V5 Milestone 2 (CF).

Evaluation Tracks:
1. Track A — Personalized User-to-Item Ranking (`[SIMULATION VALIDATION ONLY]`):
   - Fits all models strictly on `train` interactions (`2026-01-01` .. `2026-03-04`).
   - Selects hyperparameters for `ItemKNN_CF` (192 configs) and `TruncatedSVD_MF`
     (60 configs) on `val` (`2026-03-05` .. `2026-03-17`) enforcing
     `MIN_VAL_COVERAGE = 0.25` (25.0% Catalog Coverage@5 floor) with a
     documented deterministic fallback if zero configurations in a model family
     satisfy the 25.0% floor.
   - Evaluates locked models once on `test` (`2026-03-18` .. `2026-03-31`) under:
     * Protocol A1 (Primary `frozen_train_history`): user profiles, seen-item
       masking, and target novelty defined strictly relative to `train`.
     * Protocol A2 (Optional `updated_train_val_history`): user profiles, seen-item
       masking, and target novelty defined relative to `train ∪ val`, while model
       parameters remain strictly frozen from `train` (no refitting).
2. Track B — Item-to-Item Similarity Evaluation vs. V1 TF-IDF Baseline:
   - Evaluates on the 18-reference holdout (`backend/evaluation/holdout_v4_blind_labeled.csv`).
   - Never treats unjudged catalog products as irrelevant: computes Precision@5,
     Strict P@5, and nDCG@5 strictly over the labeled candidate pool per reference,
     and separately reports unconstrained full-catalog overlap with V1 and the
     unjudged candidate retrieval rate.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from cf_models import (  # noqa: E402
    DEFAULT_RELEASE_DB,
    DEFAULT_SEED,
    ItemKNNRecommender,
    MatrixBundle,
    PopularityBaseline,
    TruncatedSVDRecommender,
    V1TFIDFBaseline,
    build_conditioning_profiles,
    build_train_matrix,
    load_active_catalog_readonly,
)

REPO_ROOT = THIS_DIR.parents[2]
DEFAULT_DATA_DIR = THIS_DIR / "data"
DEFAULT_HOLDOUT_LABELS = REPO_ROOT / "backend" / "evaluation" / "holdout_v4_blind_labeled.csv"
DEFAULT_METRICS_PATH = DEFAULT_DATA_DIR / "cf_milestone2_metrics.json"
DEFAULT_REPORT_PATH = THIS_DIR / "cf_milestone2_report.md"

# Approved validation selection diversity floor (25.0% of 328 active products = 82 items)
MIN_VAL_COVERAGE: float = 0.25

# Complete Cartesian hyperparameter grid axes
DEDUP_MODES: tuple[str, ...] = ("max_weight", "log_sum")
IUF_EXPONENTS: tuple[float, ...] = (0.0, 0.5, 1.0)
MATRIX_CONFIGS: tuple[tuple[str, float], ...] = tuple(
    (d, iuf) for d in DEDUP_MODES for iuf in IUF_EXPONENTS
)
KNN_SHRINKAGES: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)
KNN_NEIGHBORS: tuple[int, ...] = (15, 30, 50, 100)
KNN_NORM_COLS: tuple[bool, ...] = (False, True)

SVD_COMPONENTS: tuple[int, ...] = (8, 16, 24, 32, 48)
SVD_SV_POWERS: tuple[float, ...] = (0.5, 1.0)

EXPECTED_ITEMKNN_GRID_COUNT: int = (
    len(MATRIX_CONFIGS) * len(KNN_SHRINKAGES) * len(KNN_NEIGHBORS) * len(KNN_NORM_COLS)
)  # 6 * 4 * 4 * 2 = 192
EXPECTED_SVD_GRID_COUNT: int = (
    len(MATRIX_CONFIGS) * len(SVD_COMPONENTS) * len(SVD_SV_POWERS)
)  # 6 * 5 * 2 = 60

FUNNEL_GRADE_MAP: dict[str, int] = {
    "view": 1,
    "add_to_cart": 2,
    "purchase": 3,
}


def load_csv_records(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_target_ground_truth(
    target_interactions: list[dict[str, str]],
    conditioning_seen_pairs: set[tuple[str, str]],
    require_novel: bool = True,
) -> dict[str, dict[str, int]]:
    """Map user_id -> {product_id: max_funnel_grade (1..3)} for evaluation split.

    When `require_novel=True`, any (user_id, product_id) pair present in
    `conditioning_seen_pairs` is excluded so target novelty matches seen-item masking.
    """
    gt: dict[str, dict[str, int]] = defaultdict(dict)
    for row in target_interactions:
        uid = row["user_id"]
        pid = row["product_id"]
        if require_novel and (uid, pid) in conditioning_seen_pairs:
            continue
        grade = FUNNEL_GRADE_MAP[row["event_type"]]
        prev = gt[uid].get(pid, 0)
        if grade > prev:
            gt[uid][pid] = grade
    return dict(gt)


def dcg_at_k(gains: list[float], k: int = 5) -> float:
    return sum(
        float(g) / math.log2(rank_idx + 2)
        for rank_idx, g in enumerate(gains[:k])
    )


def _config_performance_key(cfg: dict[str, Any]) -> tuple[float, float, float, float]:
    """Deterministic validation ranking key: (ndcg_graded@5, ndcg_binary@5, recall@5, coverage@5)."""
    vm = cfg["val_metrics"]
    return (
        float(vm["ndcg_graded_at_5"]),
        float(vm["ndcg_binary_at_5"]),
        float(vm["recall_at_5"]),
        float(vm["catalog_coverage_at_5"]),
    )


def select_best_config_with_coverage_floor(
    grid_results: list[dict[str, Any]],
    min_coverage: float = MIN_VAL_COVERAGE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select best hyperparameter configuration enforcing `catalog_coverage_at_5 >= min_coverage`.

    Protocol:
    1. Filter `grid_results` to configurations with `val_metrics["catalog_coverage_at_5"] >= min_coverage`.
    2. If one or more configurations are eligible, select the eligible configuration maximizing
       `_config_performance_key` (highest validation `ndcg_graded_at_5`, breaking ties by
       `ndcg_binary_at_5`, `recall_at_5`, and `catalog_coverage_at_5`).
    3. If zero configurations satisfy `min_coverage`, do NOT silently lower the threshold:
       use a documented deterministic fallback that selects the configuration maximizing
       `_config_performance_key` across all evaluated configurations, and explicitly report
       `coverage_threshold_satisfied = False` and `fallback_used = True`.
    """
    if not grid_results:
        raise ValueError("grid_results cannot be empty")

    eligible = [
        cfg
        for cfg in grid_results
        if float(cfg["val_metrics"]["catalog_coverage_at_5"]) >= min_coverage
    ]
    all_coverages = [float(c["val_metrics"]["catalog_coverage_at_5"]) for c in grid_results]
    max_cov_cfg = max(
        grid_results,
        key=lambda c: (
            float(c["val_metrics"]["catalog_coverage_at_5"]),
            float(c["val_metrics"]["ndcg_graded_at_5"]),
        ),
    )

    if eligible:
        selected = max(eligible, key=_config_performance_key)
        status = {
            "min_val_coverage_threshold": round(float(min_coverage), 4),
            "total_configs_evaluated": len(grid_results),
            "eligible_configs_count": len(eligible),
            "coverage_threshold_satisfied": True,
            "fallback_used": False,
            "selection_policy_applied": (
                "highest_val_ndcg_graded_among_configs_meeting_25pct_coverage_floor"
            ),
            "selected_val_coverage_at_5": float(
                selected["val_metrics"]["catalog_coverage_at_5"]
            ),
            "selected_val_ndcg_graded_at_5": float(
                selected["val_metrics"]["ndcg_graded_at_5"]
            ),
            "min_val_coverage_in_grid": round(min(all_coverages), 4),
            "max_val_coverage_in_grid": round(max(all_coverages), 4),
            "highest_coverage_config_in_grid": {
                k: v for k, v in max_cov_cfg.items() if k != "val_metrics"
            },
        }
    else:
        selected = max(grid_results, key=_config_performance_key)
        status = {
            "min_val_coverage_threshold": round(float(min_coverage), 4),
            "total_configs_evaluated": len(grid_results),
            "eligible_configs_count": 0,
            "coverage_threshold_satisfied": False,
            "fallback_used": True,
            "selection_policy_applied": (
                "FALLBACK_unconstrained_highest_val_ndcg_graded_because_zero_configs_met_25pct_coverage_floor"
            ),
            "selected_val_coverage_at_5": float(
                selected["val_metrics"]["catalog_coverage_at_5"]
            ),
            "selected_val_ndcg_graded_at_5": float(
                selected["val_metrics"]["ndcg_graded_at_5"]
            ),
            "min_val_coverage_in_grid": round(min(all_coverages), 4),
            "max_val_coverage_in_grid": round(max(all_coverages), 4),
            "highest_coverage_config_in_grid": {
                k: v for k, v in max_cov_cfg.items() if k != "val_metrics"
            },
        }

    return selected, status


def evaluate_user_recommendations(
    model: Any,
    user_ids: tuple[str, ...],
    product_ids: tuple[str, ...],
    user_profiles: np.ndarray,
    seen_mask: np.ndarray,
    ground_truth: dict[str, dict[str, int]],
    top_k: int = 5,
    exclude_seen: bool = True,
) -> dict[str, Any]:
    """Compute Precision@5, Recall@5, Binary nDCG@5, Graded nDCG@5, and Coverage@5."""
    t0 = time.perf_counter()
    score_matrix = model.score_all_users(user_profiles)
    n_users, n_items = score_matrix.shape

    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs_binary: list[float] = []
    ndcgs_graded: list[float] = []
    recommended_items_union: set[str] = set()

    for u_idx, uid in enumerate(user_ids):
        user_gt = ground_truth.get(uid)
        if not user_gt:
            continue

        row_scores = score_matrix[u_idx]
        mask_row = seen_mask[u_idx] if exclude_seen else None

        cand_indices = [
            i for i in range(n_items) if mask_row is None or not bool(mask_row[i])
        ]
        cand_indices.sort(
            key=lambda i: (-round(float(row_scores[i]), 12), product_ids[i])
        )
        top_indices = cand_indices[:top_k]
        top_pids = [product_ids[i] for i in top_indices]
        recommended_items_union.update(top_pids)

        binary_hits = [1.0 if pid in user_gt else 0.0 for pid in top_pids]
        hit_count = sum(binary_hits)
        precisions.append(hit_count / float(top_k))
        recalls.append(hit_count / float(len(user_gt)))

        ideal_bin_len = min(top_k, len(user_gt))
        idcg_bin = dcg_at_k([1.0] * ideal_bin_len, k=top_k)
        dcg_bin = dcg_at_k(binary_hits, k=top_k)
        ndcgs_binary.append(dcg_bin / idcg_bin if idcg_bin > 0.0 else 0.0)

        graded_gains = [
            float((2 ** user_gt[pid]) - 1) if pid in user_gt else 0.0
            for pid in top_pids
        ]
        ideal_grades = sorted(user_gt.values(), reverse=True)[:top_k]
        ideal_graded_gains = [float((2 ** g) - 1) for g in ideal_grades]
        idcg_grad = dcg_at_k(ideal_graded_gains, k=top_k)
        dcg_grad = dcg_at_k(graded_gains, k=top_k)
        ndcgs_graded.append(dcg_grad / idcg_grad if idcg_grad > 0.0 else 0.0)

    inference_sec = time.perf_counter() - t0
    eval_users = len(precisions)
    return {
        "evaluated_users": eval_users,
        "precision_at_5": round(float(np.mean(precisions)) if eval_users else 0.0, 4),
        "recall_at_5": round(float(np.mean(recalls)) if eval_users else 0.0, 4),
        "ndcg_binary_at_5": round(float(np.mean(ndcgs_binary)) if eval_users else 0.0, 4),
        "ndcg_graded_at_5": round(float(np.mean(ndcgs_graded)) if eval_users else 0.0, 4),
        "catalog_coverage_at_5": round(len(recommended_items_union) / float(n_items), 4),
        "unique_recommended_items_at_5": len(recommended_items_union),
        "inference_time_ms": round(inference_sec * 1000.0, 2),
    }


def evaluate_track_b_item_to_item(
    models: dict[str, Any],
    v1_model: V1TFIDFBaseline,
    product_ids: tuple[str, ...],
    holdout_csv_path: Path = DEFAULT_HOLDOUT_LABELS,
    top_k: int = 5,
) -> dict[str, Any]:
    """Evaluate item-to-item similarity against V1 on the 18-reference holdout.

    Correction C enforcement:
    1. Labeled-Pool Reranking: Ranks only the labeled candidate pool for each of the
       18 references so unjudged catalog products are NEVER treated as irrelevant (0).
    2. Full-Catalog Retrieval Diagnostics: Computes Top-5 overlap with V1 across all
       328 products, and reports the unjudged candidate retrieval rate on the 18
       holdout references without computing biased precision on unjudged items.
    """
    rows = load_csv_records(holdout_csv_path)
    by_ref: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_ref[r["reference_id"].strip()].append(r)

    v1_top5_all: dict[str, set[str]] = {
        pid: {rec_id for rec_id, _ in v1_model.recommend_similar_items(pid, top_k=top_k)}
        for pid in product_ids
    }

    results: dict[str, Any] = {}
    for model_name, model in models.items():
        p_at_5_list: list[float] = []
        strict_p_at_5_list: list[float] = []
        ndcg_at_5_list: list[float] = []
        unjudged_slots = 0
        total_holdout_slots = len(by_ref) * top_k

        for ref_id in sorted(by_ref):
            pool = by_ref[ref_id]
            labeled_map = {
                r["recommended_id"].strip(): int(r["relevance_0_1_2"].strip())
                for r in pool
            }

            full_recs = model.recommend_similar_items(ref_id, top_k=len(product_ids) - 1)
            full_top5_ids = [cid for cid, _ in full_recs[:top_k]]
            unjudged_slots += sum(1 for cid in full_top5_ids if cid not in labeled_map)

            pool_ranked = [
                (cid, score) for cid, score in full_recs if cid in labeled_map
            ][:top_k]
            grades = [labeled_map[cid] for cid, _ in pool_ranked]

            p_at_5_list.append(sum(1 for g in grades if g >= 1) / float(top_k))
            strict_p_at_5_list.append(sum(1 for g in grades if g == 2) / float(top_k))

            ideal_grades = sorted(labeled_map.values(), reverse=True)[:top_k]
            idcg = dcg_at_k([float((2 ** g) - 1) for g in ideal_grades], k=top_k)
            if idcg > 0.0:
                dcg = dcg_at_k([float((2 ** g) - 1) for g in grades], k=top_k)
                ndcg_at_5_list.append(dcg / idcg)

        overlaps: list[float] = []
        for pid in product_ids:
            m_top5 = {cid for cid, _ in model.recommend_similar_items(pid, top_k=top_k)}
            overlaps.append(len(m_top5 & v1_top5_all[pid]) / float(top_k))

        results[model_name] = {
            "labeled_pool_reranking_18_refs": {
                "precision_at_5": round(float(np.mean(p_at_5_list)), 4),
                "strict_precision_at_5": round(float(np.mean(strict_p_at_5_list)), 4),
                "ndcg_at_5_defined_queries": round(float(np.mean(ndcg_at_5_list)), 4),
                "defined_queries": len(ndcg_at_5_list),
                "total_references": len(by_ref),
            },
            "full_catalog_retrieval_diagnostics": {
                "holdout_18_refs_unjudged_top5_slot_rate": round(
                    unjudged_slots / float(total_holdout_slots), 4
                ),
                "holdout_18_refs_unjudged_slots": f"{unjudged_slots}/{total_holdout_slots}",
                "mean_top5_overlap_with_v1_all_328_products": round(
                    float(np.mean(overlaps)), 4
                ),
            },
        }
    return results


def run_milestone2_evaluation(
    db_path: Path = DEFAULT_RELEASE_DB,
    data_dir: Path = DEFAULT_DATA_DIR,
    holdout_csv_path: Path = DEFAULT_HOLDOUT_LABELS,
    seed: int = DEFAULT_SEED,
    min_val_coverage: float = MIN_VAL_COVERAGE,
) -> dict[str, Any]:
    catalog = load_active_catalog_readonly(db_path)
    product_ids = [p["id"] for p in catalog]

    users_rows = load_csv_records(data_dir / "synthetic_users.csv")
    user_ids = [u["user_id"] for u in users_rows]

    all_interactions = load_csv_records(data_dir / "synthetic_interactions.csv")
    train_rows = [r for r in all_interactions if r["split"] == "train"]
    val_rows = [r for r in all_interactions if r["split"] == "val"]
    test_rows = [r for r in all_interactions if r["split"] == "test"]

    train_seen_pairs = {(r["user_id"], r["product_id"]) for r in train_rows}
    train_val_seen_pairs = train_seen_pairs | {
        (r["user_id"], r["product_id"]) for r in val_rows
    }

    val_gt_novel = build_target_ground_truth(val_rows, train_seen_pairs, require_novel=True)
    test_gt_novel_from_train = build_target_ground_truth(
        test_rows, train_seen_pairs, require_novel=True
    )
    test_gt_novel_from_train_val = build_target_ground_truth(
        test_rows, train_val_seen_pairs, require_novel=True
    )

    # Pre-build all 6 Cartesian (dedup_mode, iuf_exponent) train MatrixBundles
    bundles: dict[tuple[str, float], MatrixBundle] = {
        (dedup, iuf_exp): build_train_matrix(
            train_rows,
            user_ids=user_ids,
            product_ids=product_ids,
            dedup_mode=dedup,
            iuf_exponent=iuf_exp,
            l2_normalize_users=True,
        )
        for dedup, iuf_exp in MATRIX_CONFIGS
    }

    default_bundle = bundles[("max_weight", 0.5)]

    # Fit baselines strictly on train
    pop_model = PopularityBaseline().fit(default_bundle)
    v1_model = V1TFIDFBaseline().fit(catalog, default_bundle)

    # 1. Complete 192-configuration Validation Grid Search for ItemKNN_CF
    itemknn_grid_results: list[dict[str, Any]] = []
    for dedup, iuf_exp in MATRIX_CONFIGS:
        b = bundles[(dedup, iuf_exp)]
        for shrinkage in KNN_SHRINKAGES:
            for k_neighbors in KNN_NEIGHBORS:
                for norm_cols in KNN_NORM_COLS:
                    candidate_model = ItemKNNRecommender(
                        k_neighbors=k_neighbors,
                        shrinkage=shrinkage,
                        normalize_sim_columns=norm_cols,
                    ).fit(b, fallback_content_sim=v1_model.item_sim_matrix)
                    val_metrics = evaluate_user_recommendations(
                        candidate_model,
                        b.user_ids,
                        b.product_ids,
                        b.normalized_matrix,
                        b.seen_mask,
                        val_gt_novel,
                        top_k=5,
                        exclude_seen=True,
                    )
                    itemknn_grid_results.append(
                        {
                            "dedup_mode": dedup,
                            "iuf_exponent": iuf_exp,
                            "shrinkage": shrinkage,
                            "k_neighbors": k_neighbors,
                            "normalize_sim_columns": norm_cols,
                            "meets_coverage_floor": bool(
                                val_metrics["catalog_coverage_at_5"] >= min_val_coverage
                            ),
                            "val_metrics": val_metrics,
                        }
                    )

    best_itemknn_cfg, itemknn_selection_status = select_best_config_with_coverage_floor(
        itemknn_grid_results, min_coverage=min_val_coverage
    )

    # 2. Complete 60-configuration Validation Grid Search for TruncatedSVD_MF
    # Note on `singular_value_power`: Track A user scoring uses orthogonal subspace
    # projection `(r_u @ V_k) @ V_k.T`, which is mathematically invariant to
    # `singular_value_power` (which scales item embeddings for Track B item-to-item
    # cosine similarity). We therefore fit and evaluate Track A once per unique
    # `(dedup_mode, iuf_exponent, n_components)` (30 SVD fits) and record all 60
    # grid configurations without redundant duplicate Track A matrix multiplications.
    svd_grid_results: list[dict[str, Any]] = []
    for dedup, iuf_exp in MATRIX_CONFIGS:
        b = bundles[(dedup, iuf_exp)]
        for n_comp in SVD_COMPONENTS:
            base_svd = TruncatedSVDRecommender(
                n_components=n_comp,
                singular_value_power=SVD_SV_POWERS[0],
                seed=seed,
            ).fit(b, fallback_content_sim=v1_model.item_sim_matrix)
            base_val_metrics = evaluate_user_recommendations(
                base_svd,
                b.user_ids,
                b.product_ids,
                b.normalized_matrix,
                b.seen_mask,
                val_gt_novel,
                top_k=5,
                exclude_seen=True,
            )
            evr_sum = round(base_svd.explained_variance_ratio_sum, 4)
            meets_floor = bool(
                base_val_metrics["catalog_coverage_at_5"] >= min_val_coverage
            )
            for sv_power in SVD_SV_POWERS:
                svd_grid_results.append(
                    {
                        "dedup_mode": dedup,
                        "iuf_exponent": iuf_exp,
                        "n_components": n_comp,
                        "singular_value_power": sv_power,
                        "singular_value_power_scope": (
                            "track_b_item_to_item_only_track_a_orthogonal_projection_invariant"
                        ),
                        "explained_variance_ratio_sum": evr_sum,
                        "meets_coverage_floor": meets_floor,
                        "val_metrics": copy.deepcopy(base_val_metrics),
                    }
                )

    best_svd_cfg, svd_selection_status = select_best_config_with_coverage_floor(
        svd_grid_results, min_coverage=min_val_coverage
    )

    # Refit final selected models strictly on train with validation-chosen hyperparameters
    best_knn_bundle = bundles[
        (best_itemknn_cfg["dedup_mode"], best_itemknn_cfg["iuf_exponent"])
    ]
    final_itemknn = ItemKNNRecommender(
        k_neighbors=best_itemknn_cfg["k_neighbors"],
        shrinkage=best_itemknn_cfg["shrinkage"],
        normalize_sim_columns=best_itemknn_cfg["normalize_sim_columns"],
    ).fit(best_knn_bundle, fallback_content_sim=v1_model.item_sim_matrix)

    best_svd_bundle = bundles[
        (best_svd_cfg["dedup_mode"], best_svd_cfg["iuf_exponent"])
    ]
    final_svd = TruncatedSVDRecommender(
        n_components=best_svd_cfg["n_components"],
        singular_value_power=best_svd_cfg["singular_value_power"],
        seed=seed,
    ).fit(best_svd_bundle, fallback_content_sim=v1_model.item_sim_matrix)

    model_specs = [
        ("PopularityBaseline", pop_model, default_bundle),
        ("V1_TFIDF_Content", v1_model, default_bundle),
        ("ItemKNN_CF", final_itemknn, best_knn_bundle),
        ("TruncatedSVD_MF", final_svd, best_svd_bundle),
    ]

    track_a_val: dict[str, Any] = {}
    track_a_test_frozen_train: dict[str, Any] = {}
    track_a_test_updated_history: dict[str, Any] = {}
    performance_telemetry: dict[str, Any] = {}

    train_plus_val_rows = train_rows + val_rows

    for name, model, bundle in model_specs:
        track_a_val[name] = evaluate_user_recommendations(
            model,
            bundle.user_ids,
            bundle.product_ids,
            bundle.normalized_matrix,
            bundle.seen_mask,
            val_gt_novel,
            top_k=5,
            exclude_seen=True,
        )
        track_a_test_frozen_train[name] = evaluate_user_recommendations(
            model,
            bundle.user_ids,
            bundle.product_ids,
            bundle.normalized_matrix,
            bundle.seen_mask,
            test_gt_novel_from_train,
            top_k=5,
            exclude_seen=True,
        )
        cond_profiles, cond_seen_mask = build_conditioning_profiles(
            train_plus_val_rows, bundle
        )
        track_a_test_updated_history[name] = evaluate_user_recommendations(
            model,
            bundle.user_ids,
            bundle.product_ids,
            cond_profiles,
            cond_seen_mask,
            test_gt_novel_from_train_val,
            top_k=5,
            exclude_seen=True,
        )
        performance_telemetry[name] = {
            "fit_time_ms": round(getattr(model, "fit_time_seconds", 0.0) * 1000.0, 2),
            "peak_fit_memory_kb": round(
                getattr(model, "peak_memory_bytes", 0) / 1024.0, 2
            ),
            "test_inference_time_ms": track_a_test_frozen_train[name]["inference_time_ms"],
        }

    track_b_results = evaluate_track_b_item_to_item(
        models={
            "PopularityBaseline": pop_model,
            "V1_TFIDF_Content": v1_model,
            "ItemKNN_CF": final_itemknn,
            "TruncatedSVD_MF": final_svd,
        },
        v1_model=v1_model,
        product_ids=default_bundle.product_ids,
        holdout_csv_path=holdout_csv_path,
        top_k=5,
    )

    def _selection_sort_key(cfg: dict[str, Any]) -> tuple[bool, float, float, float, float]:
        return (
            bool(cfg["meets_coverage_floor"]),
            *_config_performance_key(cfg),
        )

    return {
        "milestone": "Milestone 2 — Collaborative Filtering",
        "data_source": "SYNTHETIC_SIMULATION_V5",
        "scientific_disclaimer": (
            "Track A personalized ranking metrics are computed on computer-generated synthetic "
            "interactions (SYNTHETIC_SIMULATION_V5). They validate algorithmic implementation "
            "and pipeline hygiene under simulation rules, NOT real-world consumer behavior. "
            "Track B item-to-item metrics use the 18-reference AI-labeled holdout."
        ),
        "seed": seed,
        "dataset_summary": {
            "users": len(user_ids),
            "active_products": len(product_ids),
            "train_events": len(train_rows),
            "val_events": len(val_rows),
            "test_events": len(test_rows),
            "val_users_with_novel_targets": len(val_gt_novel),
            "test_users_with_novel_targets_vs_train": len(test_gt_novel_from_train),
            "test_users_with_novel_targets_vs_train_val": len(test_gt_novel_from_train_val),
        },
        "selected_hyperparameters": {
            "ItemKNN_CF": {
                **{k: v for k, v in best_itemknn_cfg.items() if k != "val_metrics"},
                "selection_status": itemknn_selection_status,
            },
            "TruncatedSVD_MF": {
                **{k: v for k, v in best_svd_cfg.items() if k != "val_metrics"},
                "selection_status": svd_selection_status,
            },
        },
        "validation_grid_search_summary": {
            "min_val_coverage_threshold": min_val_coverage,
            "matrix_configs_evaluated": [list(pair) for pair in MATRIX_CONFIGS],
            "itemknn_configs_evaluated": len(itemknn_grid_results),
            "itemknn_eligible_configs_count": itemknn_selection_status[
                "eligible_configs_count"
            ],
            "svd_configs_evaluated": len(svd_grid_results),
            "svd_unique_track_a_projections_evaluated": len(MATRIX_CONFIGS)
            * len(SVD_COMPONENTS),
            "svd_eligible_configs_count": svd_selection_status["eligible_configs_count"],
            "top_5_itemknn_val_configs": sorted(
                itemknn_grid_results, key=_selection_sort_key, reverse=True
            )[:5],
            "top_5_svd_val_configs": sorted(
                svd_grid_results, key=_selection_sort_key, reverse=True
            )[:5],
            "top_5_svd_val_configs_by_coverage": sorted(
                svd_grid_results,
                key=lambda c: (
                    c["val_metrics"]["catalog_coverage_at_5"],
                    c["val_metrics"]["ndcg_graded_at_5"],
                ),
                reverse=True,
            )[:5],
        },
        "track_a_personalized_ranking_simulation_only": {
            "validation_split_frozen_train_history": track_a_val,
            "test_split_protocol_a1_frozen_train_history": track_a_test_frozen_train,
            "test_split_protocol_a2_updated_train_val_history": track_a_test_updated_history,
        },
        "track_b_item_to_item_18_ref_holdout": track_b_results,
        "performance_telemetry": performance_telemetry,
    }


def render_milestone2_report(metrics: dict[str, Any], metrics_sha256: str) -> str:
    ds = metrics["dataset_summary"]
    sel_knn = metrics["selected_hyperparameters"]["ItemKNN_CF"]
    sel_svd = metrics["selected_hyperparameters"]["TruncatedSVD_MF"]
    knn_stat = sel_knn["selection_status"]
    svd_stat = sel_svd["selection_status"]
    gs = metrics["validation_grid_search_summary"]
    val_res = metrics["track_a_personalized_ranking_simulation_only"][
        "validation_split_frozen_train_history"
    ]
    test_a1 = metrics["track_a_personalized_ranking_simulation_only"][
        "test_split_protocol_a1_frozen_train_history"
    ]
    test_a2 = metrics["track_a_personalized_ranking_simulation_only"][
        "test_split_protocol_a2_updated_train_val_history"
    ]
    track_b = metrics["track_b_item_to_item_18_ref_holdout"]
    perf = metrics["performance_telemetry"]

    lines = [
        "# ELAN V5 — Milestone 2: Collaborative Filtering Benchmark Report",
        "",
        "> **CRITICAL SCIENTIFIC DISCLAIMER (`SYNTHETIC_SIMULATION_ONLY`):**",
        "> Track A personalized user-to-item metrics in this report are evaluated on the",
        "> Milestone 1 synthetic interaction dataset (`SYNTHETIC_SIMULATION_V5`). They measure",
        "> how effectively each algorithm recovers simulated persona preferences and outfit",
        "> co-occurrences from sparse training interactions. **They do NOT measure real human",
        "> preference or justify replacing the production V1 recommender.**",
        "",
        "## 1. Experimental Setup & Chronological Hygiene",
        "",
        f"- **Random Seed:** `{metrics['seed']}`",
        f"- **Catalog Size:** `{ds['active_products']}` active products (`backend/data/elan_release.db`, read-only `mode=ro`)",
        f"- **Training Set (`train`):** `{ds['train_events']}` events across `{ds['users']}` users (`2026-01-01` .. `2026-03-04`)",
        f"- **Validation Set (`val`):** `{ds['val_events']}` events (`{ds['val_users_with_novel_targets']}` users with novel targets unseen in `train`)",
        f"- **Test Set (`test`):** `{ds['test_events']}` events (`{ds['test_users_with_novel_targets_vs_train']}` users with novel targets vs. `train`; `{ds['test_users_with_novel_targets_vs_train_val']}` users with novel targets vs. `train ∪ val`)",
        "- **Model Fitting Policy:** All model parameters (`PopularityBaseline`, `V1_TFIDF_Content`, `ItemKNN_CF`, `TruncatedSVD_MF`) are fitted **exclusively on `train`**. No model parameters are refitted on `val` or `test`.",
        "",
        "## 2. Validation Hyperparameter Selection (`MIN_VAL_COVERAGE = 25.0%`)",
        "",
        f"- **Selection Protocol:** Maximize validation `Graded nDCG@5` among configurations satisfying `Catalog Coverage@5 >= {gs['min_val_coverage_threshold']*100:.1f}%` (`82 / 328` items). If `0` configurations satisfy the coverage floor, deterministically fall back to the highest validation `Graded nDCG@5` across the grid and flag `coverage_threshold_satisfied = False`.",
        f"- **Matrix Configurations Evaluated (`6` pairs):** `dedup_mode ∈ {{'max_weight', 'log_sum'}} × iuf_exponent ∈ {{0.0, 0.5, 1.0}}`",
        "",
        f"### 2.1. `ItemKNN_CF` Grid Selection (`{gs['itemknn_configs_evaluated']}` configurations evaluated)",
        f"- **Coverage Floor Status:** `{knn_stat['eligible_configs_count']} / {knn_stat['total_configs_evaluated']}` configurations met `Coverage@5 >= 25.0%` (`coverage_threshold_satisfied = {knn_stat['coverage_threshold_satisfied']}`, `fallback_used = {knn_stat['fallback_used']}`).",
        f"- **Validation Coverage Range Across Grid:** `{knn_stat['min_val_coverage_in_grid']*100:.2f}%` .. `{knn_stat['max_val_coverage_in_grid']*100:.2f}%`",
        f"- **Selected Configuration:** `dedup_mode = {sel_knn['dedup_mode']!r}`, `iuf_exponent = {sel_knn['iuf_exponent']}`, `shrinkage = {sel_knn['shrinkage']}`, `k_neighbors = {sel_knn['k_neighbors']}`, `normalize_sim_columns = {sel_knn['normalize_sim_columns']}` (Validation `Coverage@5 = {knn_stat['selected_val_coverage_at_5']*100:.2f}%`, `Graded nDCG@5 = {knn_stat['selected_val_ndcg_graded_at_5']:.4f}`).",
        "",
        f"### 2.2. `TruncatedSVD_MF` Grid Selection (`{gs['svd_configs_evaluated']}` configurations evaluated; `{gs['svd_unique_track_a_projections_evaluated']}` unique Track A subspace projections)",
        f"- **Coverage Floor Status:** `{svd_stat['eligible_configs_count']} / {svd_stat['total_configs_evaluated']}` configurations met `Coverage@5 >= 25.0%` (`coverage_threshold_satisfied = {svd_stat['coverage_threshold_satisfied']}`, `fallback_used = {svd_stat['fallback_used']}`).",
        f"- **Validation Coverage Range Across Grid:** `{svd_stat['min_val_coverage_in_grid']*100:.2f}%` .. `{svd_stat['max_val_coverage_in_grid']*100:.2f}%`",
        f"- **Selection Policy Applied:** `{svd_stat['selection_policy_applied']}`",
        f"- **Selected Configuration:** `dedup_mode = {sel_svd['dedup_mode']!r}`, `iuf_exponent = {sel_svd['iuf_exponent']}`, `n_components = {sel_svd['n_components']}`, `singular_value_power = {sel_svd['singular_value_power']}` (Explained Variance Ratio Sum: `{sel_svd['explained_variance_ratio_sum']*100:.2f}%`, Validation `Coverage@5 = {svd_stat['selected_val_coverage_at_5']*100:.2f}%`, `Graded nDCG@5 = {svd_stat['selected_val_ndcg_graded_at_5']:.4f}`).",
        "- **Parameter Semantics Note (`singular_value_power`):** Track A personalized user scoring projects user profiles onto the rank-$k$ orthonormal subspace `(r_u @ V_k) @ V_k.T`, which is mathematically invariant to `singular_value_power`. The parameter `singular_value_power` only scales item embeddings `V_k @ diag(sigma_k ** p)` for Track B item-to-item cosine similarity.",
        "",
        "## 3. Track A — Personalized Next-Item Recommendation (`[SIMULATION VALIDATION ONLY]`)",
        "",
        "### 3.1. Validation Split Results (`conditioning = train`, `targets = val novel`)",
        "",
        "| Model | Precision@5 | Recall@5 | Binary nDCG@5 | Graded nDCG@5 | Catalog Coverage@5 | Meets >=25% Floor |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for m_name in ("PopularityBaseline", "V1_TFIDF_Content", "ItemKNN_CF", "TruncatedSVD_MF"):
        r = val_res[m_name]
        floor_flag = "Yes" if r["catalog_coverage_at_5"] >= gs["min_val_coverage_threshold"] else "No"
        lines.append(
            f"| **{m_name}** | {r['precision_at_5']*100:.2f}% | {r['recall_at_5']*100:.2f}% | "
            f"{r['ndcg_binary_at_5']:.4f} | **{r['ndcg_graded_at_5']:.4f}** | "
            f"{r['catalog_coverage_at_5']*100:.2f}% ({r['unique_recommended_items_at_5']}/328) | {floor_flag} |"
        )

    lines.extend(
        [
            "",
            "### 3.2. Test Split — Protocol A1: Frozen Train History (`conditioning = train`, `targets = test novel vs. train`)",
            "",
            "| Model | Precision@5 | Recall@5 | Binary nDCG@5 | Graded nDCG@5 | Catalog Coverage@5 | Unique Items |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for m_name in ("PopularityBaseline", "V1_TFIDF_Content", "ItemKNN_CF", "TruncatedSVD_MF"):
        r = test_a1[m_name]
        lines.append(
            f"| **{m_name}** | {r['precision_at_5']*100:.2f}% | {r['recall_at_5']*100:.2f}% | "
            f"{r['ndcg_binary_at_5']:.4f} | **{r['ndcg_graded_at_5']:.4f}** | "
            f"{r['catalog_coverage_at_5']*100:.2f}% | {r['unique_recommended_items_at_5']}/328 |"
        )

    lines.extend(
        [
            "",
            "### 3.3. Test Split — Protocol A2: Folded-In `train + val` History (`conditioning = train ∪ val`, `targets = test novel vs. train ∪ val`, frozen `train` model parameters)",
            "",
            "| Model | Precision@5 | Recall@5 | Binary nDCG@5 | Graded nDCG@5 | Catalog Coverage@5 | Unique Items |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for m_name in ("PopularityBaseline", "V1_TFIDF_Content", "ItemKNN_CF", "TruncatedSVD_MF"):
        r = test_a2[m_name]
        lines.append(
            f"| **{m_name}** | {r['precision_at_5']*100:.2f}% | {r['recall_at_5']*100:.2f}% | "
            f"{r['ndcg_binary_at_5']:.4f} | **{r['ndcg_graded_at_5']:.4f}** | "
            f"{r['catalog_coverage_at_5']*100:.2f}% | {r['unique_recommended_items_at_5']}/328 |"
        )

    lines.extend(
        [
            "",
            "## 4. Track B — Item-to-Item Similarity vs. V1 on the 18-Reference Holdout",
            "",
            "> **Methodological Note (No Unjudged-as-Zero Bias):**",
            "> The frozen holdout (`holdout_v4_blind_labeled.csv`) contains relevance judgments (`0=irrelevant`,",
            "> `1=related`, `2=close substitute`) ONLY for the **140 candidate pairs** pooled from content-based",
            "> models V1–V4 across 18 reference products. In full-catalog retrieval, collaborative models retrieve",
            "> cross-category complementary items that are **unjudged** in this pool. Therefore, Table 4.1 evaluates",
            "> **Labeled-Pool Reranking** (ranking only the judged candidates for each reference), while Table 4.2",
            "> reports **Full-Catalog Retrieval Diagnostics** (unjudged slot rate and Top-5 overlap with V1).",
            "",
            "### 4.1. Labeled-Pool Reranking (18 References, 140 Judged Candidate Pairs)",
            "",
            "| Model | Labeled-Pool P@5 | Labeled-Pool Strict P@5 | Labeled-Pool nDCG@5 (n=16) |",
            "|---|---:|---:|---:|",
        ]
    )
    for m_name in ("PopularityBaseline", "V1_TFIDF_Content", "ItemKNN_CF", "TruncatedSVD_MF"):
        lp = track_b[m_name]["labeled_pool_reranking_18_refs"]
        lines.append(
            f"| **{m_name}** | {lp['precision_at_5']*100:.2f}% | "
            f"{lp['strict_precision_at_5']*100:.2f}% | **{lp['ndcg_at_5_defined_queries']:.4f}** |"
        )

    lines.extend(
        [
            "",
            "### 4.2. Unconstrained Full-Catalog Retrieval Diagnostics (328 Products)",
            "",
            "| Model | Unjudged Top-5 Slot Rate (18 Holdout Refs) | Mean Top-5 Overlap with V1 (All 328 Products) |",
            "|---|---:|---:|",
        ]
    )
    for m_name in ("PopularityBaseline", "V1_TFIDF_Content", "ItemKNN_CF", "TruncatedSVD_MF"):
        diag = track_b[m_name]["full_catalog_retrieval_diagnostics"]
        lines.append(
            f"| **{m_name}** | {diag['holdout_18_refs_unjudged_top5_slot_rate']*100:.2f}% "
            f"({diag['holdout_18_refs_unjudged_slots']}) | "
            f"{diag['mean_top5_overlap_with_v1_all_328_products']*100:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## 5. Runtime & Memory Telemetry",
            "",
            "| Model | Fit Time (ms) | Peak Fit Memory (KB) | Full-Split Inference Time (ms) |",
            "|---|---:|---:|---:|",
        ]
    )
    for m_name in ("PopularityBaseline", "V1_TFIDF_Content", "ItemKNN_CF", "TruncatedSVD_MF"):
        p = perf[m_name]
        lines.append(
            f"| **{m_name}** | {p['fit_time_ms']:.2f} ms | {p['peak_fit_memory_kb']:.2f} KB | "
            f"{p['test_inference_time_ms']:.2f} ms |"
        )

    lines.extend(
        [
            "",
            "## 6. Reproducibility & Verification",
            "",
            f"- **`data/cf_milestone2_metrics.json` SHA-256:** `{metrics_sha256}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_RELEASE_DB)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--holdout-csv", type=Path, default=DEFAULT_HOLDOUT_LABELS)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    metrics = run_milestone2_evaluation(
        db_path=args.db_path,
        data_dir=args.data_dir,
        holdout_csv_path=args.holdout_csv,
        seed=args.seed,
    )

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    args.metrics_out.write_text(serialized, encoding="utf-8", newline="\n")
    metrics_sha256 = hashlib.sha256(args.metrics_out.read_bytes()).hexdigest()

    report_md = render_milestone2_report(metrics, metrics_sha256)
    args.report_out.write_text(report_md, encoding="utf-8", newline="\n")

    print("ELAN V5 MILESTONE 2 CF EVALUATION: COMPLETE")
    print(f"Metrics saved: {args.metrics_out} (SHA-256: {metrics_sha256})")
    print(f"Report saved:  {args.report_out}")
    t_a1 = metrics["track_a_personalized_ranking_simulation_only"][
        "test_split_protocol_a1_frozen_train_history"
    ]
    for m_name, r in t_a1.items():
        print(
            f"  [Test A1] {m_name:20s} | P@5={r['precision_at_5']*100:5.2f}% | "
            f"R@5={r['recall_at_5']*100:5.2f}% | nDCG@5(grad)={r['ndcg_graded_at_5']:.4f} | "
            f"Cov@5={r['catalog_coverage_at_5']*100:5.2f}%"
        )


if __name__ == "__main__":
    main()
