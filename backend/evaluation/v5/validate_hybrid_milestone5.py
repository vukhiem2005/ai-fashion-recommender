"""Focused 10-Check Validation Suite for Milestone 5 (Hybrid Recommender)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

from backend.evaluation.v5.evaluate_hybrid_milestone5 import (
    DEFAULT_M5_METRICS_JSON,
    build_fitted_hybrid_system,
)
from backend.evaluation.v5.multimodal_models import DEFAULT_DATA_DIR

THIS_DIR = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_milestone5_validation() -> tuple[bool, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    hybrid, train_bundle, interactions_rows, _, _, _ = build_fitted_hybrid_system()

    # 1. Product ID alignment across CLIP, CF, and Hybrid
    align_ok = (
        len(hybrid.product_ids) == 328
        and hybrid.product_ids == hybrid.cf_model.product_ids
        and hybrid.product_ids == hybrid.clip_data.product_ids
    )
    record("product_id_alignment", align_ok, f"328 product IDs aligned across CLIP, CF, and Hybrid={align_ok}")

    # 2. Candidate deduplication & query/seen/basket exclusion
    out_excl = hybrid.recommend(
        anchor_product_id="ELAN-001",
        user_profile=train_bundle.normalized_matrix[0],
        seen_mask_row=train_bundle.seen_mask[0],
        basket_product_ids=["ELAN-001", "ELAN-002"],
        top_k=10,
        exclude_seen=True,
    )
    rec_ids = [cid for cid, _ in out_excl["recommendations"]]
    seen_pids = {
        hybrid.product_ids[i] for i in range(hybrid.n_items) if bool(train_bundle.seen_mask[0, i])
    }
    excl_ok = (
        len(rec_ids) == len(set(rec_ids)) == 10
        and "ELAN-001" not in rec_ids
        and "ELAN-002" not in rec_ids
        and len(set(rec_ids) & seen_pids) == 0
    )
    record("deduplication_and_exclusion", excl_ok, f"10 unique recs, anchor/basket/seen excluded={excl_ok}")

    # 3. Deterministic ranking
    r1 = hybrid.recommend(anchor_product_id="ELAN-001", top_k=5)["recommendations"]
    r2 = hybrid.recommend(anchor_product_id="ELAN-001", top_k=5)["recommendations"]
    det_ok = r1 == r2
    record("deterministic_ranking", det_ok, f"Repeated inference identical={det_ok}")

    # 4. Missing-image fallback (HM-0398947001 exact V1 match when CF/ARM disabled)
    miss_out = hybrid.recommend(
        anchor_product_id="HM-0398947001",
        enable_content=True,
        enable_cf=False,
        enable_arm=False,
        top_k=5,
    )
    v1_row = hybrid.v1_sim_matrix[hybrid.product_to_idx["HM-0398947001"]]
    v1_top5 = [
        hybrid.product_ids[i]
        for i in sorted(
            (idx for idx in range(hybrid.n_items) if idx != hybrid.product_to_idx["HM-0398947001"]),
            key=lambda idx: (-float(v1_row[idx]), hybrid.product_ids[idx]),
        )[:5]
    ]
    miss_ok = (
        miss_out["missing_image_fallback"] is True
        and [cid for cid, _ in miss_out["recommendations"]] == v1_top5
    )
    record("missing_image_fallback", miss_ok, f"HM-0398947001 exact V1 Top-5 match={miss_ok}")

    # 5. Cold-start routing (zero profile, no anchor/basket -> popularity fallback)
    cold_out = hybrid.recommend(
        user_profile=np.zeros(hybrid.n_items, dtype=np.float64), top_k=5
    )
    cold_ok = (
        cold_out["cold_start_fallback"] is True
        and cold_out["active_modalities"] == ["popularity_fallback"]
    )
    record("cold_start_routing", cold_ok, f"Zero-history user routed to popularity_fallback={cold_ok}")

    # 6. Missing user history (`user_profile=None` with anchor item)
    no_hist_out = hybrid.recommend(anchor_product_id="ELAN-001", user_profile=None, top_k=5)
    no_hist_ok = (
        no_hist_out["cold_start_fallback"] is False
        and no_hist_out["content_activated"] is True
        and len(no_hist_out["recommendations"]) == 5
    )
    record("missing_user_history_handling", no_hist_ok, f"Anchor-only without user_profile succeeded={no_hist_ok}")

    # 7. Appropriate activation of basket association rules
    rule_ant = hybrid.arm_model.rules[0].antecedent
    arm_on = hybrid.recommend(basket_product_ids=list(rule_ant), top_k=5)
    arm_off = hybrid.recommend(anchor_product_id="ELAN-001", basket_product_ids=None, top_k=5)
    arm_ok = arm_on["arm_activated"] is True and arm_off["arm_activated"] is False
    record("basket_arm_activation", arm_ok, f"ARM active on rule antecedent={arm_on['arm_activated']}, inactive without basket={not arm_off['arm_activated']}")

    # 8. Finite, normalized fusion scores in [0.0, 1.0]
    scores = [sc for _, sc in out_excl["recommendations"]]
    scores_ok = all(np.isfinite(sc) and 0.0 <= sc <= 1.0 + 1e-12 for sc in scores)
    record("finite_normalized_fusion_scores", scores_ok, f"Scores in [0, 1] and finite={scores_ok}")

    # 9. Train/test isolation (CF support & ARM transactions strictly from train)
    n_train_tx = sum(1 for r in interactions_rows if r["split"].strip() == "train")
    iso_ok = (
        hybrid.arm_model.n_transactions == 1016
        and int(np.sum(train_bundle.seen_mask)) == 9821
        and n_train_tx == 16091
    )
    record("train_test_isolation", iso_ok, f"ARM tx=1016 (train only), CF pairs=9821 (train only)={iso_ok}")

    # 10. Preservation of V1 and Milestones 1-4B artifacts
    clip_ok = _sha256(DEFAULT_DATA_DIR / "clip_embeddings.npz") == "4265a3a69611ef57c3f3350cdcdd971e31e10d56db4005b07a328ec410af1e7f"
    p4a_ok = _sha256(DEFAULT_DATA_DIR / "multimodal_embeddings.npz") == "cfa7af1cabed44767f610b724c3d28683a079eae38c79a84327e0c06afba1ae6"
    m2_ok = _sha256(DEFAULT_DATA_DIR / "cf_milestone2_metrics.json") == "a03bd2d0818193f7532637325d147f5c5e0a9b3f89422ff3afbed7642a026a84"
    m3_ok = _sha256(DEFAULT_DATA_DIR / "arm_milestone3_metrics.json") == "a6b5eaaabf3c0e4df49b5ddcaac7f5dbc5a23cd769810a1bb4c9256a58341ee7"
    h18_ok = DEFAULT_M5_METRICS_JSON.is_file()
    record("prior_artifacts_preserved", clip_ok and p4a_ok and m2_ok and m3_ok and h18_ok, f"CLIP={clip_ok}, P4A={p4a_ok}, M2={m2_ok}, M3={m3_ok}, M5_metrics={h18_ok}")

    return all(c["passed"] for c in checks), checks


def main() -> None:
    all_passed, checks = run_milestone5_validation()
    print("ELAN V5 MILESTONE 5 — HYBRID RECOMMENDER VALIDATION")
    print("=" * 72)
    for idx, c in enumerate(checks, start=1):
        status = "PASS" if c["passed"] else "FAIL"
        print(f"{idx:2d}. [{status}] {c['check']}: {c['detail']}")
    print("=" * 72)
    if not all_passed:
        sys.exit(1)
    print(f"ALL MILESTONE 5 VALIDATION CHECKS PASSED ({len(checks)}/{len(checks)}).")


if __name__ == "__main__":
    main()
