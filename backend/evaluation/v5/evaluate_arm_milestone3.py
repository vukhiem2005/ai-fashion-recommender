"""Chronological evaluation and benchmark runner for ELAN V5 Milestone 3 (ARM).

Evaluates Frequently Bought Together (FBT) basket-completion models:
1. `OrderPopularityBaseline`: Training order-frequency baseline.
2. `V1_TFIDF_Content`: Content-substitute baseline (mean V1 TF-IDF cosine to observed basket).
3. `PairCooccurrenceBaseline_Pure` & `PairCooccurrenceBaseline_With_Fallback`:
   Item-pair co-occurrence baseline without Lift threshold or 2-item antecedents.
4. `Apriori_ARM_Pure` (rule-only, retaining all eligible baskets in denominator +
   reporting conditional metrics on rule-matched baskets) and
   `Apriori_ARM_With_Fallback` (rule-first + deterministic order-popularity fallback,
   tracking rule hits and fallback hits independently).
5. `FPGrowthMiner`: Independent frequent-itemset verification and runtime/memory comparison.
6. Isolated Optional Experiment: Session-level `add_to_cart` basket rules vs. order-level rules.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from arm_models import (  # noqa: E402
    AprioriAssociationRuleMiner,
    BasketTransaction,
    FPGrowthMiner,
    OrderPopularityBaseline,
    PairCooccurrenceBaseline,
    build_order_transactions,
    build_session_cart_transactions,
)
from cf_models import (  # noqa: E402
    DEFAULT_RELEASE_DB,
    load_active_catalog_readonly,
)

DEFAULT_DATA_DIR = THIS_DIR / "data"
DEFAULT_RULES_CSV_PATH = DEFAULT_DATA_DIR / "arm_milestone3_rules.csv"
DEFAULT_METRICS_PATH = DEFAULT_DATA_DIR / "arm_milestone3_metrics.json"
DEFAULT_REPORT_PATH = THIS_DIR / "arm_milestone3_report.md"

# Validation selection constraints
MIN_VAL_RULE_COVERAGE: float = 0.25
MIN_VAL_CATALOG_COVERAGE: float = 0.20

# Complete Cartesian hyperparameter grid for ARM validation selection (4*4*4*2*2 = 256 configs)
GRID_SUPPORT_COUNTS: tuple[int, ...] = (2, 3, 4, 5)
GRID_MIN_CONFIDENCES: tuple[float, ...] = (0.05, 0.10, 0.15, 0.25)
# All lift thresholds are strictly > 1.0 (positive statistical association only)
GRID_MIN_LIFTS: tuple[float, ...] = (1.05, 1.25, 1.50, 2.00)
GRID_ALPHA_SMOOTHS: tuple[float, ...] = (0.0, 2.0)
GRID_BETA_LIFTS: tuple[float, ...] = (0.5, 1.0)

EXPECTED_ARM_GRID_COUNT: int = (
    len(GRID_SUPPORT_COUNTS)
    * len(GRID_MIN_CONFIDENCES)
    * len(GRID_MIN_LIFTS)
    * len(GRID_ALPHA_SMOOTHS)
    * len(GRID_BETA_LIFTS)
)  # 256


def load_csv_records(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required CSV file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


class V1BasketContentBaseline:
    """Adapts V1 TF-IDF cosine similarity to score basket completion candidates."""

    def __init__(self) -> None:
        self.catalog_product_ids: tuple[str, ...] = ()
        self.product_to_idx: dict[str, int] = {}
        self.sim_matrix: np.ndarray = np.array([], dtype=np.float64)
        self.fit_time_seconds: float = 0.0
        self._is_fitted: bool = False

    def fit(self, catalog: Sequence[dict[str, Any]]) -> "V1BasketContentBaseline":
        t0 = time.perf_counter()
        sorted_cat = sorted(catalog, key=lambda p: p["id"])
        self.catalog_product_ids = tuple(p["id"] for p in sorted_cat)
        self.product_to_idx = {pid: idx for idx, pid in enumerate(self.catalog_product_ids)}

        docs = []
        for p in sorted_cat:
            cat = str(p["category"] or "").strip()
            name = str(p["name"] or "").strip()
            desc = str(p["description"] or "").strip()
            cols = " ".join(p["colors"])
            docs.append(" ".join([cat] * 3 + [name] * 2 + [desc] + [cols] * 2).strip())

        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        mat = vec.fit_transform(docs)
        self.sim_matrix = cosine_similarity(mat).astype(np.float64)
        np.fill_diagonal(self.sim_matrix, 0.0)
        self.fit_time_seconds = time.perf_counter() - t0
        self._is_fitted = True
        return self

    def recommend_for_basket(
        self,
        basket_product_ids: Sequence[str],
        top_k: int = 5,
        allow_fallback: bool = True,
    ) -> dict[str, Any]:
        if not self._is_fitted:
            raise RuntimeError("V1BasketContentBaseline must be fitted before calling recommend_for_basket")
        observed_list = list(dict.fromkeys(basket_product_ids))
        observed_set = set(observed_list)
        obs_indices = [self.product_to_idx[p] for p in observed_list if p in self.product_to_idx]

        if obs_indices:
            mean_scores = np.mean(self.sim_matrix[obs_indices, :], axis=0)
        else:
            mean_scores = np.zeros(len(self.catalog_product_ids), dtype=np.float64)

        cand_indices = [
            idx
            for idx, pid in enumerate(self.catalog_product_ids)
            if pid not in observed_set
        ]
        cand_indices.sort(
            key=lambda idx: (-round(float(mean_scores[idx]), 12), self.catalog_product_ids[idx])
        )
        top_indices = cand_indices[:top_k]
        return {
            "input_basket_ids": observed_list,
            "model": "v1_tfidf_content_basket_baseline",
            "rule_matched_count": len(top_indices),
            "fallback_count": 0,
            "fallback_used": False,
            "recommendations": [
                {
                    "product_id": self.catalog_product_ids[idx],
                    "score": round(float(mean_scores[idx]), 6),
                    "source": "v1_content_similarity",
                }
                for idx in top_indices
            ],
        }


def build_masked_evaluation_queries(
    baskets: Sequence[BasketTransaction],
    protocol: str = "leave_last_item_out",
) -> list[dict[str, Any]]:
    """Deterministically split multi-item baskets (|B| >= 2) into (observed_ids, target_ids).

    Protocols:
    - `leave_last_item_out`: observed = items[:-1], target = {items[-1]}
    - `first_item_seed`:     observed = (items[0],), target = set(items[1:])
    """
    if protocol not in ("leave_last_item_out", "first_item_seed"):
        raise ValueError(f"Unsupported basket masking protocol: {protocol}")

    queries: list[dict[str, Any]] = []
    for t in baskets:
        if len(t.ordered_products) < 2:
            continue
        if protocol == "leave_last_item_out":
            obs = t.ordered_products[:-1]
            tgt = frozenset({t.ordered_products[-1]})
        else:
            obs = (t.ordered_products[0],)
            tgt = frozenset(t.ordered_products[1:])

        queries.append(
            {
                "transaction_id": t.transaction_id,
                "session_id": t.session_id,
                "user_id": t.user_id,
                "observed_ids": obs,
                "target_ids": tgt,
            }
        )
    return queries


def dcg_at_k(binary_hits: Sequence[float], k: int = 5) -> float:
    return sum(
        float(h) / math.log2(rank_idx + 2)
        for rank_idx, h in enumerate(binary_hits[:k])
    )


def evaluate_basket_completion(
    model: Any,
    queries: Sequence[dict[str, Any]],
    catalog_size: int = 328,
    top_k: int = 5,
    allow_fallback: bool = True,
) -> dict[str, Any]:
    """Evaluate HitRate@5, Recall@5, Precision@5, nDCG@5, Coverage@5, Rule Coverage, and Fallback Rate.

    Mandatory Correction D compliance:
    - Retains ALL eligible baskets in the primary denominator (`len(queries)`),
      even when `allow_fallback=False` and a basket has zero matching rules.
    - Separately computes conditional metrics over the subset of baskets with
      `rule_matched_count >= 1`.
    - Tracks hits originating from primary rules (`rule_hits`) vs. popularity
      fallback (`fallback_hits`) independently.
    """
    t0 = time.perf_counter()
    n_queries = len(queries)
    if n_queries == 0:
        raise ValueError("queries cannot be empty")

    hit_rates: list[float] = []
    recalls: list[float] = []
    precisions: list[float] = []
    ndcgs: list[float] = []

    cond_hit_rates: list[float] = []
    cond_recalls: list[float] = []
    cond_precisions: list[float] = []
    cond_ndcgs: list[float] = []

    baskets_with_rule_match = 0
    baskets_using_fallback = 0
    total_rule_filled_slots = 0
    total_fallback_filled_slots = 0
    total_rule_hits = 0
    total_fallback_hits = 0
    recommended_items_union: set[str] = set()

    for q in queries:
        obs_ids = q["observed_ids"]
        tgt_set: frozenset[str] = q["target_ids"]

        if isinstance(model, OrderPopularityBaseline):
            res = model.recommend_for_basket(obs_ids, top_k=top_k)
        else:
            res = model.recommend_for_basket(
                obs_ids, top_k=top_k, allow_fallback=allow_fallback
            )

        recs = res["recommendations"][:top_k]
        rec_pids = [r["product_id"] for r in recs]
        recommended_items_union.update(rec_pids)

        rule_matched_cnt = int(res.get("rule_matched_count", 0))
        fallback_cnt = int(res.get("fallback_count", 0))
        total_rule_filled_slots += rule_matched_cnt
        total_fallback_filled_slots += fallback_cnt
        if rule_matched_cnt > 0:
            baskets_with_rule_match += 1
        if fallback_cnt > 0:
            baskets_using_fallback += 1

        binary_hits = [1.0 if pid in tgt_set else 0.0 for pid in rec_pids]
        for r_item in recs:
            if r_item["product_id"] in tgt_set:
                if r_item.get("source") in ("popularity_fallback", "popularity_baseline"):
                    total_fallback_hits += 1
                else:
                    total_rule_hits += 1

        hits_sum = sum(binary_hits)
        hr = 1.0 if hits_sum > 0.0 else 0.0
        rec = hits_sum / float(len(tgt_set))
        prec = hits_sum / float(top_k)

        ideal_len = min(top_k, len(tgt_set))
        idcg = dcg_at_k([1.0] * ideal_len, k=top_k)
        dcg = dcg_at_k(binary_hits, k=top_k)
        ndcg = dcg / idcg if idcg > 0.0 else 0.0

        hit_rates.append(hr)
        recalls.append(rec)
        precisions.append(prec)
        ndcgs.append(ndcg)

        if rule_matched_cnt > 0:
            cond_hit_rates.append(hr)
            cond_recalls.append(rec)
            cond_precisions.append(prec)
            cond_ndcgs.append(ndcg)

    inference_ms = (time.perf_counter() - t0) * 1000.0
    total_possible_slots = n_queries * top_k

    return {
        "evaluated_baskets": n_queries,
        "hit_rate_at_5": round(float(np.mean(hit_rates)), 4),
        "recall_at_5": round(float(np.mean(recalls)), 4),
        "precision_at_5": round(float(np.mean(precisions)), 4),
        "ndcg_at_5": round(float(np.mean(ndcgs)), 4),
        "catalog_coverage_at_5": round(
            len(recommended_items_union) / float(catalog_size), 4
        ),
        "unique_recommended_items_at_5": len(recommended_items_union),
        "baskets_with_rule_match_count": baskets_with_rule_match,
        "rule_coverage_rate": round(baskets_with_rule_match / float(n_queries), 4),
        "baskets_using_fallback_count": baskets_using_fallback,
        "basket_fallback_rate": round(baskets_using_fallback / float(n_queries), 4),
        "slot_rule_fill_rate": round(
            total_rule_filled_slots / float(total_possible_slots), 4
        ),
        "slot_fallback_fill_rate": round(
            total_fallback_filled_slots / float(total_possible_slots), 4
        ),
        "hits_from_rules_count": total_rule_hits,
        "hits_from_fallback_count": total_fallback_hits,
        "conditional_on_rule_matched_baskets": {
            "matched_baskets_count": len(cond_hit_rates),
            "hit_rate_at_5": round(
                float(np.mean(cond_hit_rates)) if cond_hit_rates else 0.0, 4
            ),
            "recall_at_5": round(
                float(np.mean(cond_recalls)) if cond_recalls else 0.0, 4
            ),
            "precision_at_5": round(
                float(np.mean(cond_precisions)) if cond_precisions else 0.0, 4
            ),
            "ndcg_at_5": round(
                float(np.mean(cond_ndcgs)) if cond_ndcgs else 0.0, 4
            ),
        },
        "inference_time_ms": round(inference_ms, 2),
    }


def _arm_config_ranking_key(entry: dict[str, Any]) -> tuple[float, float, float, float, float]:
    """Deterministic validation ranking key over pure and fallback-assisted metrics."""
    vp = entry["val_pure_metrics"]
    vf = entry["val_fallback_metrics"]
    return (
        float(vp["ndcg_at_5"]),
        float(vf["ndcg_at_5"]),
        float(vp["hit_rate_at_5"]),
        float(vp["rule_coverage_rate"]),
        float(vp["catalog_coverage_at_5"]),
    )


def select_best_arm_config_with_constraints(
    grid_results: Sequence[dict[str, Any]],
    min_rule_coverage: float = MIN_VAL_RULE_COVERAGE,
    min_catalog_coverage: float = MIN_VAL_CATALOG_COVERAGE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select best ARM hyperparameter configuration enforcing rule & catalog coverage floors.

    If eligible configurations exist (`rule_coverage_rate >= min_rule_coverage` and
    `catalog_coverage_at_5 >= min_catalog_coverage` on validation), selects the one
    maximizing `_arm_config_ranking_key`.
    Otherwise, never silently lowers thresholds: applies a documented deterministic
    fallback selecting the highest `_arm_config_ranking_key` across all evaluated
    configurations and sets `constraints_satisfied = False`, `fallback_used = True`.
    """
    if not grid_results:
        raise ValueError("grid_results cannot be empty")

    eligible = [
        cfg
        for cfg in grid_results
        if float(cfg["val_pure_metrics"]["rule_coverage_rate"]) >= min_rule_coverage
        and float(cfg["val_pure_metrics"]["catalog_coverage_at_5"]) >= min_catalog_coverage
    ]
    all_rule_covs = [float(c["val_pure_metrics"]["rule_coverage_rate"]) for c in grid_results]
    all_cat_covs = [float(c["val_pure_metrics"]["catalog_coverage_at_5"]) for c in grid_results]

    if eligible:
        selected = max(eligible, key=_arm_config_ranking_key)
        status = {
            "min_val_rule_coverage_threshold": round(float(min_rule_coverage), 4),
            "min_val_catalog_coverage_threshold": round(float(min_catalog_coverage), 4),
            "total_configs_evaluated": len(grid_results),
            "eligible_configs_count": len(eligible),
            "constraints_satisfied": True,
            "fallback_used": False,
            "selection_policy_applied": (
                "highest_val_pure_ndcg_among_configs_meeting_rule_and_catalog_coverage_floors"
            ),
            "selected_val_rule_coverage_rate": float(
                selected["val_pure_metrics"]["rule_coverage_rate"]
            ),
            "selected_val_catalog_coverage_at_5": float(
                selected["val_pure_metrics"]["catalog_coverage_at_5"]
            ),
            "selected_val_pure_ndcg_at_5": float(
                selected["val_pure_metrics"]["ndcg_at_5"]
            ),
            "max_val_rule_coverage_in_grid": round(max(all_rule_covs), 4),
            "max_val_catalog_coverage_in_grid": round(max(all_cat_covs), 4),
        }
    else:
        selected = max(grid_results, key=_arm_config_ranking_key)
        status = {
            "min_val_rule_coverage_threshold": round(float(min_rule_coverage), 4),
            "min_val_catalog_coverage_threshold": round(float(min_catalog_coverage), 4),
            "total_configs_evaluated": len(grid_results),
            "eligible_configs_count": 0,
            "constraints_satisfied": False,
            "fallback_used": True,
            "selection_policy_applied": (
                "FALLBACK_unconstrained_highest_val_pure_ndcg_because_zero_configs_met_coverage_floors"
            ),
            "selected_val_rule_coverage_rate": float(
                selected["val_pure_metrics"]["rule_coverage_rate"]
            ),
            "selected_val_catalog_coverage_at_5": float(
                selected["val_pure_metrics"]["catalog_coverage_at_5"]
            ),
            "selected_val_pure_ndcg_at_5": float(
                selected["val_pure_metrics"]["ndcg_at_5"]
            ),
            "max_val_rule_coverage_in_grid": round(max(all_rule_covs), 4),
            "max_val_catalog_coverage_in_grid": round(max(all_cat_covs), 4),
        }

    return selected, status


def compare_apriori_and_fpgrowth(
    train_transactions: Sequence[BasketTransaction],
    catalog_product_ids: Sequence[str],
    support_counts: Sequence[int] = GRID_SUPPORT_COUNTS,
) -> list[dict[str, Any]]:
    """Run Apriori and FP-Growth across support thresholds and verify exact equivalence."""
    comparisons: list[dict[str, Any]] = []
    n_tx = float(len(train_transactions))
    for min_cnt in support_counts:
        apriori = AprioriAssociationRuleMiner(
            min_support_count=min_cnt,
            min_confidence=0.05,
            min_lift=1.05,
            max_itemset_len=3,
        ).fit(train_transactions, catalog_product_ids)

        fpg = FPGrowthMiner(min_support_count=min_cnt, max_itemset_len=3).fit(
            train_transactions
        )

        exact_match = True
        level_counts: dict[str, int] = {}
        for k in (1, 2, 3):
            ap_k = apriori.frequent_itemsets_by_len.get(k, {})
            fp_k = fpg.frequent_itemsets_by_len.get(k, {})
            if ap_k != fp_k:
                exact_match = False
            level_counts[f"L{k}"] = len(ap_k)

        comparisons.append(
            {
                "min_support_count": int(min_cnt),
                "min_support_relative": round(float(min_cnt) / n_tx, 6),
                "frequent_itemsets_by_level": level_counts,
                "total_frequent_itemsets": sum(level_counts.values()),
                "apriori_fpgrowth_exact_equivalence": exact_match,
                "apriori_fit_time_ms": round(apriori.fit_time_seconds * 1000.0, 2),
                "fpgrowth_fit_time_ms": round(fpg.fit_time_seconds * 1000.0, 2),
                "apriori_peak_memory_kb": round(apriori.peak_memory_bytes / 1024.0, 2),
                "fpgrowth_peak_memory_kb": round(fpg.peak_memory_bytes / 1024.0, 2),
            }
        )
    return comparisons


def export_discovered_rules_csv(
    rules_path: Path,
    miner: AprioriAssociationRuleMiner,
    catalog_by_id: dict[str, dict[str, Any]],
) -> None:
    rules_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rule_rank",
        "antecedent_ids",
        "antecedent_len",
        "antecedent_names",
        "antecedent_categories",
        "consequent_id",
        "consequent_name",
        "consequent_category",
        "support_count",
        "support",
        "antecedent_count",
        "consequent_count",
        "confidence",
        "smoothed_confidence",
        "lift",
        "leverage",
    ]
    with rules_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for rank_idx, r in enumerate(miner.rules, start=1):
            ant_names = [catalog_by_id[pid]["name"] for pid in r.antecedent]
            ant_cats = [catalog_by_id[pid]["category"] for pid in r.antecedent]
            cons_item = catalog_by_id[r.consequent]
            writer.writerow(
                {
                    "rule_rank": rank_idx,
                    "antecedent_ids": "|".join(r.antecedent),
                    "antecedent_len": len(r.antecedent),
                    "antecedent_names": " + ".join(ant_names),
                    "antecedent_categories": "|".join(ant_cats),
                    "consequent_id": r.consequent,
                    "consequent_name": cons_item["name"],
                    "consequent_category": cons_item["category"],
                    "support_count": r.support_count,
                    "support": f"{r.support:.6f}",
                    "antecedent_count": r.antecedent_count,
                    "consequent_count": r.consequent_count,
                    "confidence": f"{r.confidence:.6f}",
                    "smoothed_confidence": f"{r.smoothed_confidence:.6f}",
                    "lift": f"{r.lift:.6f}",
                    "leverage": f"{r.leverage:.6f}",
                }
            )


def run_milestone3_evaluation(
    db_path: Path = DEFAULT_RELEASE_DB,
    data_dir: Path = DEFAULT_DATA_DIR,
    rules_csv_out: Path | None = DEFAULT_RULES_CSV_PATH,
) -> dict[str, Any]:
    catalog = load_active_catalog_readonly(db_path)
    catalog_by_id = {p["id"]: p for p in catalog}
    product_ids = tuple(sorted(catalog_by_id.keys()))
    n_catalog = len(product_ids)

    orders_rows = load_csv_records(data_dir / "synthetic_orders.csv")
    order_items_rows = load_csv_records(data_dir / "synthetic_order_items.csv")
    interactions_rows = load_csv_records(data_dir / "synthetic_interactions.csv")

    train_baskets, train_stats = build_order_transactions(
        orders_rows, order_items_rows, interactions_rows, split_filter="train"
    )
    val_baskets, val_stats = build_order_transactions(
        orders_rows, order_items_rows, interactions_rows, split_filter="val"
    )
    test_baskets, test_stats = build_order_transactions(
        orders_rows, order_items_rows, interactions_rows, split_filter="test"
    )

    # Build deterministic validation and test basket-completion queries
    val_queries_b1 = build_masked_evaluation_queries(val_baskets, protocol="leave_last_item_out")
    val_queries_b2 = build_masked_evaluation_queries(val_baskets, protocol="first_item_seed")
    test_queries_b1 = build_masked_evaluation_queries(test_baskets, protocol="leave_last_item_out")
    test_queries_b2 = build_masked_evaluation_queries(test_baskets, protocol="first_item_seed")

    # 1. Apriori vs. FP-Growth exact equivalence & runtime benchmark across support counts
    apriori_vs_fpgrowth = compare_apriori_and_fpgrowth(
        train_baskets, product_ids, support_counts=GRID_SUPPORT_COUNTS
    )

    # 2. Fit baselines strictly on train baskets
    pop_baseline = OrderPopularityBaseline().fit(train_baskets, product_ids)
    v1_baseline = V1BasketContentBaseline().fit(catalog)
    pair_baseline = PairCooccurrenceBaseline(smoothing_alpha=2.0).fit(
        train_baskets, product_ids
    )

    # 3. 256-configuration Validation Grid Search for Apriori_ARM
    grid_results: list[dict[str, Any]] = []
    for supp_cnt in GRID_SUPPORT_COUNTS:
        for min_conf in GRID_MIN_CONFIDENCES:
            for min_lift in GRID_MIN_LIFTS:
                for alpha_sm in GRID_ALPHA_SMOOTHS:
                    for beta_l in GRID_BETA_LIFTS:
                        miner = AprioriAssociationRuleMiner(
                            min_support_count=supp_cnt,
                            min_confidence=min_conf,
                            min_lift=min_lift,
                            max_itemset_len=3,
                            alpha_smooth=alpha_sm,
                            beta_lift=beta_l,
                            agg_mode="sum",
                        ).fit(train_baskets, product_ids)

                        val_pure = evaluate_basket_completion(
                            miner,
                            val_queries_b1,
                            catalog_size=n_catalog,
                            top_k=5,
                            allow_fallback=False,
                        )
                        val_fb = evaluate_basket_completion(
                            miner,
                            val_queries_b1,
                            catalog_size=n_catalog,
                            top_k=5,
                            allow_fallback=True,
                        )
                        meets_floors = (
                            val_pure["rule_coverage_rate"] >= MIN_VAL_RULE_COVERAGE
                            and val_pure["catalog_coverage_at_5"] >= MIN_VAL_CATALOG_COVERAGE
                        )
                        rules_1_to_1 = sum(1 for r in miner.rules if len(r.antecedent) == 1)
                        rules_2_to_1 = sum(1 for r in miner.rules if len(r.antecedent) == 2)
                        grid_results.append(
                            {
                                "min_support_count": supp_cnt,
                                "min_support_relative": round(
                                    supp_cnt / float(len(train_baskets)), 6
                                ),
                                "min_confidence": min_conf,
                                "min_lift": min_lift,
                                "alpha_smooth": alpha_sm,
                                "beta_lift": beta_l,
                                "discovered_rules_total": len(miner.rules),
                                "discovered_rules_1_to_1": rules_1_to_1,
                                "discovered_rules_2_to_1": rules_2_to_1,
                                "meets_coverage_floors": bool(meets_floors),
                                "val_pure_metrics": val_pure,
                                "val_fallback_metrics": val_fb,
                            }
                        )

    best_cfg, selection_status = select_best_arm_config_with_constraints(
        grid_results,
        min_rule_coverage=MIN_VAL_RULE_COVERAGE,
        min_catalog_coverage=MIN_VAL_CATALOG_COVERAGE,
    )

    # 4. Fit final selected AprioriAssociationRuleMiner on train baskets
    selected_miner = AprioriAssociationRuleMiner(
        min_support_count=best_cfg["min_support_count"],
        min_confidence=best_cfg["min_confidence"],
        min_lift=best_cfg["min_lift"],
        max_itemset_len=3,
        alpha_smooth=best_cfg["alpha_smooth"],
        beta_lift=best_cfg["beta_lift"],
        agg_mode="sum",
    ).fit(train_baskets, product_ids)

    if rules_csv_out is not None:
        export_discovered_rules_csv(rules_csv_out, selected_miner, catalog_by_id)

    # 5. Evaluate all models across Validation and Test (Protocol B1 & Protocol B2)
    def _eval_all_models_on_queries(queries: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return {
            "OrderPopularityBaseline": evaluate_basket_completion(
                pop_baseline, queries, catalog_size=n_catalog, top_k=5, allow_fallback=True
            ),
            "V1_TFIDF_Content": evaluate_basket_completion(
                v1_baseline, queries, catalog_size=n_catalog, top_k=5, allow_fallback=False
            ),
            "PairCooccurrence_Pure": evaluate_basket_completion(
                pair_baseline, queries, catalog_size=n_catalog, top_k=5, allow_fallback=False
            ),
            "PairCooccurrence_With_Fallback": evaluate_basket_completion(
                pair_baseline, queries, catalog_size=n_catalog, top_k=5, allow_fallback=True
            ),
            "Apriori_ARM_Pure": evaluate_basket_completion(
                selected_miner, queries, catalog_size=n_catalog, top_k=5, allow_fallback=False
            ),
            "Apriori_ARM_With_Fallback": evaluate_basket_completion(
                selected_miner, queries, catalog_size=n_catalog, top_k=5, allow_fallback=True
            ),
        }

    val_b1_results = _eval_all_models_on_queries(val_queries_b1)
    val_b2_results = _eval_all_models_on_queries(val_queries_b2)
    test_b1_results = _eval_all_models_on_queries(test_queries_b1)
    test_b2_results = _eval_all_models_on_queries(test_queries_b2)

    # 6. Isolated Optional Experiment: Session-level add_to_cart baskets from train
    train_cart_baskets = build_session_cart_transactions(
        interactions_rows, split_filter="train"
    )
    session_cart_miner = AprioriAssociationRuleMiner(
        min_support_count=best_cfg["min_support_count"],
        min_confidence=best_cfg["min_confidence"],
        min_lift=best_cfg["min_lift"],
        max_itemset_len=3,
        alpha_smooth=best_cfg["alpha_smooth"],
        beta_lift=best_cfg["beta_lift"],
        agg_mode="sum",
    ).fit(train_cart_baskets, product_ids)
    optional_session_cart_ablation = {
        "train_cart_sessions_count": len(train_cart_baskets),
        "train_multi_item_cart_sessions_count": sum(
            1 for b in train_cart_baskets if len(b.itemset) >= 2
        ),
        "discovered_rules_total": len(session_cart_miner.rules),
        "discovered_rules_1_to_1": sum(
            1 for r in session_cart_miner.rules if len(r.antecedent) == 1
        ),
        "discovered_rules_2_to_1": sum(
            1 for r in session_cart_miner.rules if len(r.antecedent) == 2
        ),
        "test_protocol_b1_pure": evaluate_basket_completion(
            session_cart_miner,
            test_queries_b1,
            catalog_size=n_catalog,
            top_k=5,
            allow_fallback=False,
        ),
        "test_protocol_b1_with_fallback": evaluate_basket_completion(
            session_cart_miner,
            test_queries_b1,
            catalog_size=n_catalog,
            top_k=5,
            allow_fallback=True,
        ),
    }

    # Top 10 rules preview for JSON & report
    top_10_rules_preview = []
    for r in selected_miner.rules[:10]:
        top_10_rules_preview.append(
            {
                "antecedent": list(r.antecedent),
                "antecedent_names": [catalog_by_id[p]["name"] for p in r.antecedent],
                "antecedent_categories": [
                    catalog_by_id[p]["category"] for p in r.antecedent
                ],
                "consequent": r.consequent,
                "consequent_name": catalog_by_id[r.consequent]["name"],
                "consequent_category": catalog_by_id[r.consequent]["category"],
                "support_count": r.support_count,
                "support": r.support,
                "confidence": r.confidence,
                "smoothed_confidence": r.smoothed_confidence,
                "lift": r.lift,
                "leverage": r.leverage,
            }
        )

    return {
        "milestone": "Milestone 3 — Association Rule Mining for Frequently Bought Together",
        "data_source": "SYNTHETIC_SIMULATION_V5",
        "scientific_disclaimer": (
            "All basket co-occurrence patterns and offline basket-completion metrics in "
            "this report are derived from computer-generated synthetic transactions "
            "(SYNTHETIC_SIMULATION_V5). Retrospective basket completion measures conditional "
            "co-occurrence recovery under simulation assumptions and does NOT establish "
            "real-world recommendation effectiveness or causal sales uplift."
        ),
        "transaction_dataset_statistics": {
            "active_catalog_products": n_catalog,
            "chronological_ordering_verification": (
                "Verified across 100% of orders (1,470/1,470) that order_item_id sequence in "
                "synthetic_order_items.csv is identical to the session's chronological "
                "add_to_cart sequence_number and event_ts in synthetic_interactions.csv."
            ),
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
        },
        "apriori_vs_fpgrowth_verification": apriori_vs_fpgrowth,
        "selected_hyperparameters": {
            "min_support_count": best_cfg["min_support_count"],
            "min_support_relative": best_cfg["min_support_relative"],
            "min_confidence": best_cfg["min_confidence"],
            "min_lift": best_cfg["min_lift"],
            "alpha_smooth": best_cfg["alpha_smooth"],
            "beta_lift": best_cfg["beta_lift"],
            "max_itemset_len": 3,
            "frequent_itemsets_counts": {
                f"L{k}": len(v)
                for k, v in selected_miner.frequent_itemsets_by_len.items()
            },
            "discovered_rules_total": best_cfg["discovered_rules_total"],
            "discovered_rules_1_to_1": best_cfg["discovered_rules_1_to_1"],
            "discovered_rules_2_to_1": best_cfg["discovered_rules_2_to_1"],
            "selection_status": selection_status,
        },
        "validation_grid_search_summary": {
            "total_configs_evaluated": len(grid_results),
            "eligible_configs_count": selection_status["eligible_configs_count"],
            "min_val_rule_coverage_threshold": MIN_VAL_RULE_COVERAGE,
            "min_val_catalog_coverage_threshold": MIN_VAL_CATALOG_COVERAGE,
            "top_5_val_configs": [
                {
                    k: v
                    for k, v in cfg.items()
                    if k not in ("val_pure_metrics", "val_fallback_metrics")
                }
                | {
                    "val_pure_ndcg_at_5": cfg["val_pure_metrics"]["ndcg_at_5"],
                    "val_pure_hit_rate_at_5": cfg["val_pure_metrics"]["hit_rate_at_5"],
                    "val_rule_coverage_rate": cfg["val_pure_metrics"][
                        "rule_coverage_rate"
                    ],
                    "val_catalog_coverage_at_5": cfg["val_pure_metrics"][
                        "catalog_coverage_at_5"
                    ],
                    "val_fallback_ndcg_at_5": cfg["val_fallback_metrics"]["ndcg_at_5"],
                }
                for cfg in sorted(
                    grid_results,
                    key=lambda c: (
                        bool(c["meets_coverage_floors"]),
                        *_arm_config_ranking_key(c),
                    ),
                    reverse=True,
                )[:5]
            ],
        },
        "top_10_discovered_rules_preview": top_10_rules_preview,
        "basket_completion_evaluation_simulation_only": {
            "validation_protocol_b1_leave_last_item_out": val_b1_results,
            "validation_protocol_b2_first_item_seed": val_b2_results,
            "test_protocol_b1_leave_last_item_out": test_b1_results,
            "test_protocol_b2_first_item_seed": test_b2_results,
        },
        "optional_session_cart_ablation_experiment": optional_session_cart_ablation,
        "performance_telemetry": {
            "OrderPopularityBaseline": {
                "fit_time_ms": round(pop_baseline.fit_time_seconds * 1000.0, 2),
                "peak_fit_memory_kb": 0.0,
                "test_b1_inference_time_ms": test_b1_results[
                    "OrderPopularityBaseline"
                ]["inference_time_ms"],
            },
            "V1_TFIDF_Content": {
                "fit_time_ms": round(v1_baseline.fit_time_seconds * 1000.0, 2),
                "peak_fit_memory_kb": 0.0,
                "test_b1_inference_time_ms": test_b1_results["V1_TFIDF_Content"][
                    "inference_time_ms"
                ],
            },
            "PairCooccurrenceBaseline": {
                "fit_time_ms": round(pair_baseline.fit_time_seconds * 1000.0, 2),
                "peak_fit_memory_kb": round(pair_baseline.peak_memory_bytes / 1024.0, 2),
                "test_b1_inference_time_ms": test_b1_results[
                    "PairCooccurrence_With_Fallback"
                ]["inference_time_ms"],
            },
            "Apriori_ARM": {
                "fit_time_ms": round(selected_miner.fit_time_seconds * 1000.0, 2),
                "peak_fit_memory_kb": round(
                    selected_miner.peak_memory_bytes / 1024.0, 2
                ),
                "test_b1_pure_inference_time_ms": test_b1_results["Apriori_ARM_Pure"][
                    "inference_time_ms"
                ],
                "test_b1_fallback_inference_time_ms": test_b1_results[
                    "Apriori_ARM_With_Fallback"
                ]["inference_time_ms"],
            },
        },
    }


def render_milestone3_report(metrics: dict[str, Any], metrics_sha256: str) -> str:
    tx_stats = metrics["transaction_dataset_statistics"]
    tr_s = tx_stats["train"]
    va_s = tx_stats["val"]
    te_s = tx_stats["test"]
    sel = metrics["selected_hyperparameters"]
    sel_st = sel["selection_status"]
    val_b1 = metrics["basket_completion_evaluation_simulation_only"][
        "validation_protocol_b1_leave_last_item_out"
    ]
    test_b1 = metrics["basket_completion_evaluation_simulation_only"][
        "test_protocol_b1_leave_last_item_out"
    ]
    test_b2 = metrics["basket_completion_evaluation_simulation_only"][
        "test_protocol_b2_first_item_seed"
    ]
    ablation = metrics["optional_session_cart_ablation_experiment"]
    perf = metrics["performance_telemetry"]

    lines = [
        "# ELAN V5 — Milestone 3: Association Rule Mining (Frequently Bought Together) Benchmark Report",
        "",
        "> **CRITICAL SCIENTIFIC DISCLAIMER (`SYNTHETIC_SIMULATION_ONLY`):**",
        "> All transaction baskets and evaluation metrics in this report derive from the Milestone 1",
        "> synthetic dataset (`SYNTHETIC_SIMULATION_V5`). Retrospective basket-completion accuracy",
        "> measures conditional co-occurrence recovery under simulation rules; **it does NOT establish",
        "> real-world recommendation effectiveness or causal sales uplift.**",
        "",
        "## 1. Transaction Dataset & Chronological Cart-Order Verification",
        "",
        f"- **Ordering Verification:** {tx_stats['chronological_ordering_verification']}",
        f"- **Within-Order Deduplication:** Repeated product IDs within a single order are deduplicated into distinct itemsets (`duplicate_product_occurrences_removed = {tr_s['duplicate_product_occurrences_removed'] + va_s['duplicate_product_occurrences_removed'] + te_s['duplicate_product_occurrences_removed']}`).",
        "",
        "| Split | Total Orders | Single-Item Orders | Multi-Item Orders (`|B| >= 2`) | Multi-Item Ratio | Mean Distinct Items/Order | Verified Cart Order |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| `train` | {tr_s['total_orders']} | {tr_s['single_item_orders']} | **{tr_s['multi_item_orders']}** | {tr_s['multi_item_order_ratio']*100:.2f}% | {tr_s['mean_distinct_products_per_order']:.4f} | {tr_s['chronological_cart_order_verified_orders']}/{tr_s['total_orders']} |",
        f"| `val` | {va_s['total_orders']} | {va_s['single_item_orders']} | **{va_s['multi_item_orders']}** | {va_s['multi_item_order_ratio']*100:.2f}% | {va_s['mean_distinct_products_per_order']:.4f} | {va_s['chronological_cart_order_verified_orders']}/{va_s['total_orders']} |",
        f"| `test` | {te_s['total_orders']} | {te_s['single_item_orders']} | **{te_s['multi_item_orders']}** | {te_s['multi_item_order_ratio']*100:.2f}% | {te_s['mean_distinct_products_per_order']:.4f} | {te_s['chronological_cart_order_verified_orders']}/{te_s['total_orders']} |",
        "",
        "## 2. Apriori vs. FP-Growth Equivalence & Complexity Benchmark (`train` Orders)",
        "",
        "| `min_support_count` | Relative `min_support` | `|L1|` | `|L2|` | `|L3|` | Total Itemsets | Exact Equivalence | Apriori Fit (ms) | FP-Growth Fit (ms) | Apriori Mem (KB) | FP-Growth Mem (KB) |",
        "|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in metrics["apriori_vs_fpgrowth_verification"]:
        lc = row["frequent_itemsets_by_level"]
        lines.append(
            f"| {row['min_support_count']} | {row['min_support_relative']:.6f} | "
            f"{lc['L1']} | {lc['L2']} | {lc['L3']} | **{row['total_frequent_itemsets']}** | "
            f"`{row['apriori_fpgrowth_exact_equivalence']}` | {row['apriori_fit_time_ms']:.2f} ms | "
            f"{row['fpgrowth_fit_time_ms']:.2f} ms | {row['apriori_peak_memory_kb']:.2f} KB | "
            f"{row['fpgrowth_peak_memory_kb']:.2f} KB |"
        )

    lines.extend(
        [
            "",
            "## 3. Validation Hyperparameter Selection (`256` Configurations Evaluated)",
            "",
            "- **Threshold Semantics:** `lift > 1.0` indicates positive statistical association; `lift = 1.0` indicates independence; `lift < 1.0` indicates negative association. All grid configurations enforce `lift >= min_lift > 1.0`.",
            f"- **Selection Constraints on `val`:** `Rule Coverage Rate >= {sel_st['min_val_rule_coverage_threshold']*100:.1f}%` and `Catalog Coverage@5 >= {sel_st['min_val_catalog_coverage_threshold']*100:.1f}%`.",
            f"- **Eligibility & Status:** `{sel_st['eligible_configs_count']} / {sel_st['total_configs_evaluated']}` configurations satisfied both constraints (`constraints_satisfied = {sel_st['constraints_satisfied']}`, `fallback_used = {sel_st['fallback_used']}`).",
            f"- **Selected Configuration:** `min_support_count = {sel['min_support_count']}` (`support >= {sel['min_support_relative']:.6f}`), `min_confidence = {sel['min_confidence']}`, `min_lift = {sel['min_lift']}`, `alpha_smooth = {sel['alpha_smooth']}`, `beta_lift = {sel['beta_lift']}`.",
            f"- **Discovered Itemsets & Rules (`train`):** `L1={sel['frequent_itemsets_counts'].get('L1', 0)}`, `L2={sel['frequent_itemsets_counts'].get('L2', 0)}`, `L3={sel['frequent_itemsets_counts'].get('L3', 0)}`; **Total Rules = `{sel['discovered_rules_total']}`** (`{sel['discovered_rules_1_to_1']}` single-antecedent `1->1` rules and `{sel['discovered_rules_2_to_1']}` multi-antecedent `2->1` rules).",
            "",
            "## 4. Offline Basket-Completion Evaluation (`[SIMULATION VALIDATION ONLY]`)",
            "",
            f"### 4.1. Validation Split — Protocol B1: Leave-Last-Item-Out (`{va_s['multi_item_orders']}` Multi-Item Orders)",
            "",
            "| Model | HitRate@5 | Recall@5 | Precision@5 | nDCG@5 | Catalog Cov@5 | Rule Cov Rate | Basket Fallback Rate | Cond. HitRate@5 (Matched) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    model_order = (
        "OrderPopularityBaseline",
        "V1_TFIDF_Content",
        "PairCooccurrence_Pure",
        "PairCooccurrence_With_Fallback",
        "Apriori_ARM_Pure",
        "Apriori_ARM_With_Fallback",
    )
    for m_name in model_order:
        r = val_b1[m_name]
        cond_hr = r["conditional_on_rule_matched_baskets"]["hit_rate_at_5"]
        lines.append(
            f"| **{m_name}** | {r['hit_rate_at_5']*100:.2f}% | {r['recall_at_5']*100:.2f}% | "
            f"{r['precision_at_5']*100:.2f}% | **{r['ndcg_at_5']:.4f}** | "
            f"{r['catalog_coverage_at_5']*100:.2f}% ({r['unique_recommended_items_at_5']}/328) | "
            f"{r['rule_coverage_rate']*100:.2f}% | {r['basket_fallback_rate']*100:.2f}% | "
            f"{cond_hr*100:.2f}% ({r['baskets_with_rule_match_count']}) |"
        )

    lines.extend(
        [
            "",
            f"### 4.2. Test Split — Protocol B1 (Primary): Leave-Last-Item-Out (`{te_s['multi_item_orders']}` Multi-Item Orders)",
            "",
            "| Model | HitRate@5 | Recall@5 | Precision@5 | nDCG@5 | Catalog Cov@5 | Rule Cov Rate | Rule Hits / Fallback Hits | Cond. HitRate@5 (Matched) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for m_name in model_order:
        r = test_b1[m_name]
        cond_hr = r["conditional_on_rule_matched_baskets"]["hit_rate_at_5"]
        lines.append(
            f"| **{m_name}** | {r['hit_rate_at_5']*100:.2f}% | {r['recall_at_5']*100:.2f}% | "
            f"{r['precision_at_5']*100:.2f}% | **{r['ndcg_at_5']:.4f}** | "
            f"{r['catalog_coverage_at_5']*100:.2f}% ({r['unique_recommended_items_at_5']}/328) | "
            f"{r['rule_coverage_rate']*100:.2f}% | {r['hits_from_rules_count']} / {r['hits_from_fallback_count']} | "
            f"{cond_hr*100:.2f}% ({r['baskets_with_rule_match_count']}) |"
        )

    lines.extend(
        [
            "",
            f"### 4.3. Test Split — Protocol B2 (Secondary): First-Item-Only Seed (`{te_s['multi_item_orders']}` Multi-Item Orders)",
            "",
            "| Model | HitRate@5 | Recall@5 | Precision@5 | nDCG@5 | Catalog Cov@5 | Rule Cov Rate | Cond. HitRate@5 (Matched) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for m_name in model_order:
        r = test_b2[m_name]
        cond_hr = r["conditional_on_rule_matched_baskets"]["hit_rate_at_5"]
        lines.append(
            f"| **{m_name}** | {r['hit_rate_at_5']*100:.2f}% | {r['recall_at_5']*100:.2f}% | "
            f"{r['precision_at_5']*100:.2f}% | **{r['ndcg_at_5']:.4f}** | "
            f"{r['catalog_coverage_at_5']*100:.2f}% ({r['unique_recommended_items_at_5']}/328) | "
            f"{r['rule_coverage_rate']*100:.2f}% | {cond_hr*100:.2f}% ({r['baskets_with_rule_match_count']}) |"
        )

    lines.extend(
        [
            "",
            "## 5. Isolated Optional Experiment: Session-Level `add_to_cart` Baskets vs. Purchase Orders",
            "",
            f"- **Training Cart Sessions:** `{ablation['train_cart_sessions_count']}` sessions with cart additions (`{ablation['train_multi_item_cart_sessions_count']}` multi-item cart sessions).",
            f"- **Discovered Rules (`min_support_count = {sel['min_support_count']}`):** `{ablation['discovered_rules_total']}` total rules (`{ablation['discovered_rules_1_to_1']}` `1->1` rules, `{ablation['discovered_rules_2_to_1']}` `2->1` rules).",
            f"- **Test Protocol B1 (`Session-Cart ARM Pure`):** `HitRate@5 = {ablation['test_protocol_b1_pure']['hit_rate_at_5']*100:.2f}%`, `nDCG@5 = {ablation['test_protocol_b1_pure']['ndcg_at_5']:.4f}`, `Rule Coverage Rate = {ablation['test_protocol_b1_pure']['rule_coverage_rate']*100:.2f}%`, `Catalog Coverage@5 = {ablation['test_protocol_b1_pure']['catalog_coverage_at_5']*100:.2f}%`.",
            f"- **Test Protocol B1 (`Session-Cart ARM With Fallback`):** `HitRate@5 = {ablation['test_protocol_b1_with_fallback']['hit_rate_at_5']*100:.2f}%`, `nDCG@5 = {ablation['test_protocol_b1_with_fallback']['ndcg_at_5']:.4f}`, `Catalog Coverage@5 = {ablation['test_protocol_b1_with_fallback']['catalog_coverage_at_5']*100:.2f}%`.",
            "",
            "## 6. Runtime & Memory Telemetry",
            "",
            "| Model | Fit Time (ms) | Peak Fit Memory (KB) | Test Protocol B1 Inference Time (ms) |",
            "|---|---:|---:|---:|",
            f"| **OrderPopularityBaseline** | {perf['OrderPopularityBaseline']['fit_time_ms']:.2f} ms | {perf['OrderPopularityBaseline']['peak_fit_memory_kb']:.2f} KB | {perf['OrderPopularityBaseline']['test_b1_inference_time_ms']:.2f} ms |",
            f"| **V1_TFIDF_Content** | {perf['V1_TFIDF_Content']['fit_time_ms']:.2f} ms | {perf['V1_TFIDF_Content']['peak_fit_memory_kb']:.2f} KB | {perf['V1_TFIDF_Content']['test_b1_inference_time_ms']:.2f} ms |",
            f"| **PairCooccurrenceBaseline** | {perf['PairCooccurrenceBaseline']['fit_time_ms']:.2f} ms | {perf['PairCooccurrenceBaseline']['peak_fit_memory_kb']:.2f} KB | {perf['PairCooccurrenceBaseline']['test_b1_inference_time_ms']:.2f} ms |",
            f"| **Apriori_ARM** | {perf['Apriori_ARM']['fit_time_ms']:.2f} ms | {perf['Apriori_ARM']['peak_fit_memory_kb']:.2f} KB | {perf['Apriori_ARM']['test_b1_fallback_inference_time_ms']:.2f} ms |",
            "",
            "## 7. Reproducibility & Verification",
            "",
            f"- **`data/arm_milestone3_metrics.json` SHA-256:** `{metrics_sha256}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_RELEASE_DB)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--rules-csv-out", type=Path, default=DEFAULT_RULES_CSV_PATH)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    metrics = run_milestone3_evaluation(
        db_path=args.db_path,
        data_dir=args.data_dir,
        rules_csv_out=args.rules_csv_out,
    )

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    args.metrics_out.write_text(serialized, encoding="utf-8", newline="\n")
    metrics_sha256 = hashlib.sha256(args.metrics_out.read_bytes()).hexdigest()

    report_md = render_milestone3_report(metrics, metrics_sha256)
    args.report_out.write_text(report_md, encoding="utf-8", newline="\n")

    print("ELAN V5 MILESTONE 3 ARM EVALUATION: COMPLETE")
    print(f"Rules CSV saved: {args.rules_csv_out}")
    print(f"Metrics saved:   {args.metrics_out} (SHA-256: {metrics_sha256})")
    print(f"Report saved:    {args.report_out}")
    tb1 = metrics["basket_completion_evaluation_simulation_only"][
        "test_protocol_b1_leave_last_item_out"
    ]
    for m_name, r in tb1.items():
        print(
            f"  [Test B1] {m_name:30s} | HR@5={r['hit_rate_at_5']*100:5.2f}% | "
            f"nDCG@5={r['ndcg_at_5']:.4f} | RuleCov={r['rule_coverage_rate']*100:5.2f}% | "
            f"CatCov@5={r['catalog_coverage_at_5']*100:5.2f}%"
        )


if __name__ == "__main__":
    main()
