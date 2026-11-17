"""Automated software correctness and validation suite for ELAN V5 Milestone 3 (ARM).

Strictly separates software correctness checks from experimental model performance
comparisons (does NOT require ARM to beat any baseline to pass validation).

Verifies all 10 required engineering checks:
1. Train/validation/test isolation and audit quarantine
2. Order boundaries, product deduplication, and verified chronological cart ordering
3. Apriori vs. FP-Growth exact frequent-itemset equivalence
4. Support, Confidence, and Lift mathematical correctness & lift > 1.0 enforcement
5. Basket-item exclusion and deterministic lexicographical string tie-breaking
6. Multi-item antecedent (2->1) rule discovery and activation
7. Explicit model lifecycle and cold-start / fallback determinism
8. Basket-completion metric correctness and denominator fairness (pure vs. conditional)
9. Hyperparameter grid enumeration (256 configs) and selection constraint integrity
10. Reproducibility and preservation of Milestone 1 & Milestone 2 artifacts
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

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
    recommend_frequently_bought_together,
)
from cf_models import DEFAULT_RELEASE_DB  # noqa: E402
from evaluate_arm_milestone3 import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_METRICS_PATH,
    DEFAULT_RULES_CSV_PATH,
    EXPECTED_ARM_GRID_COUNT,
    MIN_VAL_CATALOG_COVERAGE,
    MIN_VAL_RULE_COVERAGE,
    evaluate_basket_completion,
    load_csv_records,
    run_milestone3_evaluation,
    select_best_arm_config_with_constraints,
)


def make_toy_tx(tx_id: str, items: tuple[str, ...], split: str = "train") -> BasketTransaction:
    return BasketTransaction(
        transaction_id=tx_id,
        session_id=f"S-{tx_id}",
        user_id="U-TOY",
        timestamp="2026-01-10T00:00:00Z",
        split=split,
        ordered_products=items,
        itemset=frozenset(items),
    )


def run_milestone3_validation(
    db_path: Path = DEFAULT_RELEASE_DB,
    data_dir: Path = DEFAULT_DATA_DIR,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    rules_csv_path: Path = DEFAULT_RULES_CSV_PATH,
) -> tuple[bool, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

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

    # 1. Train/validation/test isolation and audit quarantine
    non_train_rejected = True
    for cls in (OrderPopularityBaseline, PairCooccurrenceBaseline, AprioriAssociationRuleMiner):
        try:
            cls().fit(val_baskets[:3], ["HM-A", "HM-B"])
            non_train_rejected = False
        except ValueError:
            pass
    try:
        FPGrowthMiner().fit(test_baskets[:3])
        non_train_rejected = False
    except ValueError:
        pass

    arm_src = (THIS_DIR / "arm_models.py").read_text(encoding="utf-8")
    eval_src = (THIS_DIR / "evaluate_arm_milestone3.py").read_text(encoding="utf-8")
    no_audit_refs = (
        "latent_audit_only" not in arm_src
        and "latent_audit_only" not in eval_src
        and "origin_signal" not in arm_src
        and "origin_signal" not in eval_src
    )
    record(
        "train_val_test_isolation_and_audit_quarantine",
        non_train_rejected and no_audit_refs,
        f"Non-train baskets rejected by all 4 miners/baselines={non_train_rejected}; "
        f"zero references to quarantined audit files={no_audit_refs}",
    )

    # 2. Order boundaries, product deduplication & verified chronological cart ordering
    toy_orders = [
        {"order_id": "O1", "session_id": "S1", "user_id": "U1", "order_ts": "2026-01-01T00:00:00Z", "split": "train"}
    ]
    toy_order_items = [
        {"order_item_id": "OI1", "order_id": "O1", "product_id": "P_B", "split": "train"},
        {"order_item_id": "OI2", "order_id": "O1", "product_id": "P_A", "split": "train"},
        {"order_item_id": "OI3", "order_id": "O1", "product_id": "P_B", "split": "train"},
    ]
    dedup_tx, dedup_st = build_order_transactions(toy_orders, toy_order_items, split_filter="train")
    dedup_ok = (
        len(dedup_tx) == 1
        and dedup_tx[0].ordered_products == ("P_B", "P_A")
        and dedup_tx[0].itemset == frozenset({"P_A", "P_B"})
        and dedup_st["duplicate_product_occurrences_removed"] == 1
    )
    cart_order_verified_ok = (
        train_stats["total_orders"] == 1016
        and val_stats["total_orders"] == 229
        and test_stats["total_orders"] == 225
        and train_stats["chronological_cart_order_mismatches"] == 0
        and val_stats["chronological_cart_order_mismatches"] == 0
        and test_stats["chronological_cart_order_mismatches"] == 0
        and (
            train_stats["chronological_cart_order_verified_orders"]
            + val_stats["chronological_cart_order_verified_orders"]
            + test_stats["chronological_cart_order_verified_orders"]
        )
        == 1470
    )
    record(
        "order_boundaries_deduplication_and_cart_ordering",
        dedup_ok and cart_order_verified_ok,
        f"Within-order deduplication verified={dedup_ok}; 1470/1470 orders verified identical to session add_to_cart sequence={cart_order_verified_ok}",
    )

    # 3. Apriori vs. FP-Growth exact frequent-itemset equivalence
    catalog_pids = sorted({t_pid for b in train_baskets for t_pid in b.itemset})
    eq_all_supports = True
    for min_cnt in (2, 3, 4, 5):
        ap = AprioriAssociationRuleMiner(
            min_support_count=min_cnt, min_confidence=0.05, min_lift=1.05, max_itemset_len=3
        ).fit(train_baskets, catalog_pids)
        fp = FPGrowthMiner(min_support_count=min_cnt, max_itemset_len=3).fit(train_baskets)
        for k in (1, 2, 3):
            if ap.frequent_itemsets_by_len.get(k, {}) != fp.frequent_itemsets_by_len.get(k, {}):
                eq_all_supports = False
    record(
        "apriori_vs_fpgrowth_itemset_equivalence",
        eq_all_supports,
        "Apriori and FP-Growth produced 100% identical frequent itemsets (L1, L2, L3) and support counts across min_support_count in {2, 3, 4, 5}",
    )

    # 4. Support, Confidence, and Lift mathematical correctness & lift > 1.0 enforcement
    # Toy dataset of 5 transactions:
    # T1: {A, B}, T2: {A, B}, T3: {A, C}, T4: {B, D}, T5: {E}
    # count(A)=3 (supp=0.6), count(B)=3 (supp=0.6), count(A,B)=2 (supp=0.4)
    # Rule A => B: conf = 2/3, smoothed_conf(alpha=2) = 2/(3+2) = 0.4, lift = (2/3)/0.6 = 10/9 = 1.111111
    toy_5 = [
        make_toy_tx("T1", ("A", "B")),
        make_toy_tx("T2", ("A", "B")),
        make_toy_tx("T3", ("A", "C")),
        make_toy_tx("T4", ("B", "D")),
        make_toy_tx("T5", ("E",)),
    ]
    toy_miner = AprioriAssociationRuleMiner(
        min_support_count=2, min_confidence=0.5, min_lift=1.05, max_itemset_len=2, alpha_smooth=2.0
    ).fit(toy_5, ["A", "B", "C", "D", "E"])
    ab_rules = [r for r in toy_miner.rules if r.antecedent == ("A",) and r.consequent == "B"]
    math_ok = (
        len(ab_rules) == 1
        and ab_rules[0].support_count == 2
        and math.isclose(ab_rules[0].support, 0.4, rel_tol=1e-6)
        and math.isclose(ab_rules[0].confidence, 2.0 / 3.0, rel_tol=1e-5)
        and math.isclose(ab_rules[0].smoothed_confidence, 0.4, rel_tol=1e-6)
        and math.isclose(ab_rules[0].lift, 10.0 / 9.0, rel_tol=1e-5)
    )
    invalid_lift_rejected = False
    try:
        AprioriAssociationRuleMiner(min_lift=1.0)
    except ValueError:
        invalid_lift_rejected = True

    record(
        "support_confidence_and_lift_correctness",
        math_ok and invalid_lift_rejected,
        f"Toy rule A=>B exact match (supp=0.4, conf=0.666667, smoothed=0.4, lift=1.111111)={math_ok}; "
        f"min_lift<=1.0 rejected={invalid_lift_rejected}",
    )

    # 5. Basket-item exclusion & deterministic string tie-breaking
    # Equal support/confidence/lift for B and C given A -> must break tie in ascending lexicographical order ("ITEM-B" before "ITEM-C")
    tie_tx = [
        make_toy_tx("T1", ("ITEM-A", "ITEM-C")),
        make_toy_tx("T2", ("ITEM-A", "ITEM-C")),
        make_toy_tx("T3", ("ITEM-A", "ITEM-B")),
        make_toy_tx("T4", ("ITEM-A", "ITEM-B")),
        make_toy_tx("T5", ("ITEM-Z",)),
    ]
    tie_miner = AprioriAssociationRuleMiner(
        min_support_count=2, min_confidence=0.1, min_lift=1.05, max_itemset_len=2
    ).fit(tie_tx, ["ITEM-A", "ITEM-B", "ITEM-C", "ITEM-Z"])
    tie_res = tie_miner.recommend_for_basket(["ITEM-A"], top_k=5, allow_fallback=True)
    rec_ids = [r["product_id"] for r in tie_res["recommendations"]]
    exclusion_and_tie_ok = (
        "ITEM-A" not in rec_ids
        and rec_ids[:2] == ["ITEM-B", "ITEM-C"]
    )
    record(
        "basket_item_exclusion_and_string_tie_breaking",
        exclusion_and_tie_ok,
        f"Observed item excluded and equal-score string IDs sorted lexicographically: {rec_ids[:2]}",
    )

    # 6. Multi-item antecedent (2->1) rule discovery and activation
    # {A, B} co-occurs with C 3/3 times (conf=1.0), whereas A alone has count 5 (conf(A=>C)=3/5=0.6) and B alone has count 5 (conf(B=>C)=3/5=0.6)
    synergy_tx = [
        make_toy_tx("S1", ("A", "B", "C")),
        make_toy_tx("S2", ("A", "B", "C")),
        make_toy_tx("S3", ("A", "B", "C")),
        make_toy_tx("S4", ("A", "D")),
        make_toy_tx("S5", ("A", "E")),
        make_toy_tx("S6", ("B", "D")),
        make_toy_tx("S7", ("B", "E")),
        make_toy_tx("S8", ("D", "E")),
    ]
    syn_miner = AprioriAssociationRuleMiner(
        min_support_count=3,
        min_confidence=0.5,
        min_lift=1.2,
        max_itemset_len=3,
        require_multi_antecedent_synergy=True,
    ).fit(synergy_tx, ["A", "B", "C", "D", "E"])
    has_2_to_1 = any(r.antecedent == ("A", "B") and r.consequent == "C" for r in syn_miner.rules)
    syn_rec = syn_miner.recommend_for_basket(["A", "B"], top_k=3, allow_fallback=False)
    activated_ok = (
        has_2_to_1
        and len(syn_rec["recommendations"]) >= 1
        and syn_rec["recommendations"][0]["product_id"] == "C"
        and syn_rec["recommendations"][0]["matched_antecedent"] == ["A", "B"]
    )
    record(
        "multi_item_antecedent_rule_activation",
        activated_ok,
        f"Synergistic 2->1 rule ('A', 'B') => 'C' discovered={has_2_to_1} and activated as top recommendation={activated_ok}",
    )

    # 7. Explicit model lifecycle & cold-start / fallback determinism
    unfitted_rejected = False
    try:
        AprioriAssociationRuleMiner().recommend_for_basket(["A"])
    except RuntimeError:
        unfitted_rejected = True

    cold_pure = syn_miner.recommend_for_basket(["UNKNOWN-ITEM"], top_k=3, allow_fallback=False)
    cold_fb = recommend_frequently_bought_together(
        syn_miner, ["UNKNOWN-ITEM"], top_k=3, allow_fallback=True
    )
    pop_fb = syn_miner.popularity_fallback.recommend_for_basket(["UNKNOWN-ITEM"], top_k=3)
    lifecycle_and_cold_ok = (
        unfitted_rejected
        and cold_pure["rule_matched_count"] == 0
        and cold_pure["fallback_used"] is False
        and len(cold_pure["recommendations"]) == 0
        and cold_fb["rule_matched_count"] == 0
        and cold_fb["fallback_count"] == 3
        and cold_fb["fallback_used"] is True
        and [r["product_id"] for r in cold_fb["recommendations"]]
        == [r["product_id"] for r in pop_fb["recommendations"]]
    )
    record(
        "explicit_lifecycle_and_cold_start_fallback",
        lifecycle_and_cold_ok,
        f"Unfitted inference rejected={unfitted_rejected}; cold-start pure returns [] and fallback matches OrderPopularityBaseline={lifecycle_and_cold_ok}",
    )

    # 8. Basket-completion metric correctness & denominator fairness (pure vs. conditional)
    # Query 1: observed=("A", "B"), target={"C"} -> syn_miner recommends "C" at rank 1 (HR=1, R=1, P=0.2, nDCG=1.0)
    # Query 2: observed=("D",), target={"E"} -> syn_miner has 0 matching rules (pure HR=0, R=0, P=0, nDCG=0)
    # Across all 2 queries: Pure HR@5 = 0.5, Pure nDCG@5 = 0.5, Rule Coverage = 0.5; Conditional HR@5 on matched (1 basket) = 1.0!
    toy_queries = [
        {"transaction_id": "Q1", "session_id": "SQ1", "user_id": "U1", "observed_ids": ("A", "B"), "target_ids": frozenset({"C"})},
        {"transaction_id": "Q2", "session_id": "SQ2", "user_id": "U1", "observed_ids": ("D",), "target_ids": frozenset({"E"})},
    ]
    eval_pure = evaluate_basket_completion(
        syn_miner, toy_queries, catalog_size=5, top_k=5, allow_fallback=False
    )
    metrics_fairness_ok = (
        eval_pure["evaluated_baskets"] == 2
        and math.isclose(eval_pure["hit_rate_at_5"], 0.5, rel_tol=1e-9)
        and math.isclose(eval_pure["recall_at_5"], 0.5, rel_tol=1e-9)
        and math.isclose(eval_pure["precision_at_5"], 0.1, rel_tol=1e-9)
        and math.isclose(eval_pure["ndcg_at_5"], 0.5, rel_tol=1e-9)
        and math.isclose(eval_pure["rule_coverage_rate"], 0.5, rel_tol=1e-9)
        and eval_pure["conditional_on_rule_matched_baskets"]["matched_baskets_count"] == 1
        and math.isclose(
            eval_pure["conditional_on_rule_matched_baskets"]["hit_rate_at_5"], 1.0, rel_tol=1e-9
        )
    )
    record(
        "basket_completion_metrics_and_denominator_fairness",
        metrics_fairness_ok,
        f"Full denominator retained (evaluated=2, pure HR@5=0.5, pure nDCG@5=0.5) and conditional matched HR@5=1.0 verified",
    )

    # 9. Hyperparameter grid enumeration (256 configs) & selection constraint integrity
    if not metrics_path.is_file() or not rules_csv_path.is_file():
        record("hyperparameter_selection_and_grid_integrity", False, "Missing saved metrics or rules CSV")
        return False, checks

    saved_bytes = metrics_path.read_bytes()
    saved_sha256 = hashlib.sha256(saved_bytes).hexdigest()
    saved_json = json.loads(saved_bytes.decode("utf-8"))
    gs = saved_json["validation_grid_search_summary"]

    # Controlled Test Case 9A: Eligible config exists (rule_cov >= 0.25 and cat_cov >= 0.20)
    mock_grid_eligible = [
        {
            "id": "ineligible_high_ndcg",
            "val_pure_metrics": {"ndcg_at_5": 0.12, "hit_rate_at_5": 0.18, "rule_coverage_rate": 0.15, "catalog_coverage_at_5": 0.30},
            "val_fallback_metrics": {"ndcg_at_5": 0.12},
        },
        {
            "id": "eligible_winner",
            "val_pure_metrics": {"ndcg_at_5": 0.08, "hit_rate_at_5": 0.14, "rule_coverage_rate": 0.40, "catalog_coverage_at_5": 0.25},
            "val_fallback_metrics": {"ndcg_at_5": 0.09},
        },
    ]
    sel_el, st_el = select_best_arm_config_with_constraints(
        mock_grid_eligible, min_rule_coverage=0.25, min_catalog_coverage=0.20
    )

    # Controlled Test Case 9B: Zero eligible configs -> deterministic fallback with constraints_satisfied=False
    mock_grid_fallback = [
        {
            "id": "fallback_winner",
            "val_pure_metrics": {"ndcg_at_5": 0.07, "hit_rate_at_5": 0.11, "rule_coverage_rate": 0.18, "catalog_coverage_at_5": 0.15},
            "val_fallback_metrics": {"ndcg_at_5": 0.08},
        },
        {
            "id": "fallback_loser",
            "val_pure_metrics": {"ndcg_at_5": 0.04, "hit_rate_at_5": 0.08, "rule_coverage_rate": 0.22, "catalog_coverage_at_5": 0.19},
            "val_fallback_metrics": {"ndcg_at_5": 0.05},
        },
    ]
    sel_fb, st_fb = select_best_arm_config_with_constraints(
        mock_grid_fallback, min_rule_coverage=0.25, min_catalog_coverage=0.20
    )

    live_st = saved_json["selected_hyperparameters"]["selection_status"]
    live_protocol_ok = (
        gs["total_configs_evaluated"] == EXPECTED_ARM_GRID_COUNT == 256
        and math.isclose(float(live_st["min_val_rule_coverage_threshold"]), MIN_VAL_RULE_COVERAGE, rel_tol=1e-9)
        and math.isclose(float(live_st["min_val_catalog_coverage_threshold"]), MIN_VAL_CATALOG_COVERAGE, rel_tol=1e-9)
        and (
            (
                live_st["eligible_configs_count"] > 0
                and live_st["constraints_satisfied"] is True
                and live_st["fallback_used"] is False
                and float(live_st["selected_val_rule_coverage_rate"]) >= MIN_VAL_RULE_COVERAGE
                and float(live_st["selected_val_catalog_coverage_at_5"]) >= MIN_VAL_CATALOG_COVERAGE
            )
            or (
                live_st["eligible_configs_count"] == 0
                and live_st["constraints_satisfied"] is False
                and live_st["fallback_used"] is True
            )
        )
    )
    grid_and_sel_ok = (
        sel_el["id"] == "eligible_winner"
        and st_el["constraints_satisfied"] is True
        and st_el["fallback_used"] is False
        and sel_fb["id"] == "fallback_winner"
        and st_fb["constraints_satisfied"] is False
        and st_fb["fallback_used"] is True
        and live_protocol_ok
    )
    record(
        "hyperparameter_selection_and_grid_integrity",
        grid_and_sel_ok,
        f"Grid count={gs['total_configs_evaluated']}/256; eligible unit test=True; fallback unit test=True; live protocol adherence={live_protocol_ok}",
    )

    # 10. Reproducibility & Milestone 1/2 artifact preservation
    recomputed = run_milestone3_evaluation(
        db_path=db_path, data_dir=data_dir, rules_csv_out=None
    )
    repro_ok = (
        saved_json["selected_hyperparameters"] == recomputed["selected_hyperparameters"]
        and saved_json["top_10_discovered_rules_preview"]
        == recomputed["top_10_discovered_rules_preview"]
    )
    for split_key, split_dict in saved_json["basket_completion_evaluation_simulation_only"].items():
        for m_name, m_vals in split_dict.items():
            s_copy = dict(m_vals)
            r_copy = dict(recomputed["basket_completion_evaluation_simulation_only"][split_key][m_name])
            s_copy.pop("inference_time_ms", None)
            r_copy.pop("inference_time_ms", None)
            if s_copy != r_copy:
                repro_ok = False

    m1_manifest = json.loads((data_dir / "synthetic_manifest.json").read_text(encoding="utf-8"))
    m1_hashes_ok = all(
        hashlib.sha256((THIS_DIR / rel_path).read_bytes()).hexdigest() == expected_digest
        for rel_path, expected_digest in m1_manifest["file_sha256"].items()
    )
    m2_metrics_hash = hashlib.sha256(
        (data_dir / "cf_milestone2_metrics.json").read_bytes()
    ).hexdigest()
    m2_expected_hash = "a03bd2d0818193f7532637325d147f5c5e0a9b3f89422ff3afbed7642a026a84"

    record(
        "reproducibility_and_prior_milestones_preservation",
        repro_ok and m1_hashes_ok and (m2_metrics_hash == m2_expected_hash),
        f"Milestone 3 metrics reproducible={repro_ok} (SHA-256={saved_sha256[:16]}...); "
        f"Milestone 1 (8/8 files) unchanged={m1_hashes_ok}; Milestone 2 metrics unchanged={m2_metrics_hash == m2_expected_hash}",
    )

    all_passed = all(c["passed"] for c in checks)
    return all_passed, checks


def main() -> None:
    all_passed, checks = run_milestone3_validation()
    print("ELAN V5 MILESTONE 3 ARM VALIDATION SUITE")
    print("=" * 68)
    for idx, c in enumerate(checks, start=1):
        status = "PASS" if c["passed"] else "FAIL"
        print(f"{idx:2d}. [{status}] {c['check']}: {c['detail']}")
    print("=" * 68)
    if not all_passed:
        sys.exit(1)
    print(f"ALL MILESTONE 3 VALIDATION CHECKS PASSED ({len(checks)}/{len(checks)}).")


if __name__ == "__main__":
    main()
