"""Focused 10-Check Validation Suite for Milestone 4B.3B (Dev-18 Tuning Only)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from backend.evaluation.v5.evaluate_multimodal_milestone4 import (
    DEV_CSV_1,
    DEV_CSV_2,
    load_judged_pool,
)
from backend.evaluation.v5.evaluate_neural_multimodal_milestone4b import (
    DEFAULT_DEV18_METRICS_JSON,
    DEFAULT_FINAL_REPORT_MD,
    DEFAULT_HOLDOUT18_METRICS_JSON,
    DEFAULT_LOCKED_CONFIG_JSON,
    HOLDOUT_EVALUATION_ENABLED,
    _canonical_json_sha256,
    _sha256_file,
    run_dev18_tuning,
)
from backend.evaluation.v5.multimodal_models import DEFAULT_DATA_DIR

THIS_DIR = Path(__file__).resolve().parent


def run_milestone4b3b_validation() -> tuple[bool, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    saved = json.loads(DEFAULT_DEV18_METRICS_JSON.read_text(encoding="utf-8"))

    # 1. Dev-18 query and label loading
    dev_pool = load_judged_pool([DEV_CSV_1, DEV_CSV_2])
    dev_ok = len(dev_pool) == 18 and sum(len(m) for m in dev_pool.values()) == 130
    record("dev18_dataset_counts", dev_ok, f"refs={len(dev_pool)}, pairs={sum(len(m) for m in dev_pool.values())}")

    # 2. No Holdout-18 access (runtime file-open guard)
    holdout_flag_ok = (HOLDOUT_EVALUATION_ENABLED is False) and (saved["holdout_18_accessed"] is False)
    opened_paths: list[str] = []
    orig_open = Path.open

    def _guarded_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        opened_paths.append(str(self))
        if "holdout_v4_blind_labeled" in str(self):
            raise AssertionError("Holdout-18 CSV accessed during Dev-18 tuning!")
        return orig_open(self, *args, **kwargs)

    from unittest.mock import patch

    with patch.object(Path, "open", _guarded_open):
        recomputed = run_dev18_tuning()

    no_holdout_opened = not any("holdout_v4_blind_labeled" in p for p in opened_paths)
    record(
        "no_holdout18_access",
        holdout_flag_ok and no_holdout_opened,
        f"HOLDOUT_EVALUATION_ENABLED={HOLDOUT_EVALUATION_ENABLED}, runtime Holdout-18 file opens=0",
    )

    # 3. Exact reuse of V1 and Phase 4A baselines
    p4a_metrics = json.loads((DEFAULT_DATA_DIR / "multimodal_milestone4_metrics.json").read_text(encoding="utf-8"))
    m = saved["models_dev18"]
    v1_match = (
        m["1_exact_v1_tfidf"]["ndcg_at_5"] == p4a_metrics["models"]["v1_tfidf_exact"]["dev_18_protocol_m1"]["ndcg_at_5"]
        and m["4_phase4a_late_fusion"]["ndcg_at_5"] == p4a_metrics["models"]["late_fusion"]["dev_18_protocol_m1"]["ndcg_at_5"]
    )
    record("exact_reuse_of_v1_and_phase4a", v1_match, f"Exact V1 & Phase 4A Dev-18 metrics match={v1_match}")

    # 4. Consistent candidate pools across all 7 models
    pools_ok = all(
        v["num_references"] == 18 and v["num_judged_pairs"] == 130 for v in m.values()
    )
    record("consistent_candidate_pools", pools_ok, f"All 7 models evaluated on 18 refs / 130 pairs={pools_ok}")

    # 5. Deterministic parameter selection
    det_ok = saved["models_dev18"] == recomputed["models_dev18"]
    record("deterministic_parameter_selection", det_ok, f"Re-run identical={det_ok}")

    # 6. Valid ranking metrics in [0, 1]
    metrics_ok = all(
        0.0 <= float(v[k]) <= 1.0
        for v in m.values()
        for k in ("precision_at_5", "strict_precision_at_5", "ndcg_at_5")
    )
    record("valid_ranking_metrics_bounds", metrics_ok, f"All metrics in [0.0, 1.0]={metrics_ok}")

    # 7. Missing-image fallback preservation (HM-0398947001 in Dev-18 pool of HM-0292453013)
    per_q = {r["reference_id"]: r for r in saved["per_query_comparison_dev18"]}
    fb_ok = "HM-0292453013" in per_q and np.isfinite(per_q["HM-0292453013"]["clip_late_ndcg5"])
    record("missing_image_candidate_in_dev18", fb_ok, f"HM-0292453013 evaluated cleanly={fb_ok}")

    # 8. No NaN or infinite scores
    finite_ok = all(
        np.isfinite(float(v[k]))
        for v in m.values()
        for k in ("precision_at_5", "strict_precision_at_5", "ndcg_at_5")
    )
    record("no_nan_or_inf_scores", finite_ok, f"All aggregate scores finite={finite_ok}")

    # 9. Existing CLIP & Phase 4A embedding artifacts unchanged
    clip_sha = hashlib.sha256((DEFAULT_DATA_DIR / "clip_embeddings.npz").read_bytes()).hexdigest()
    p4a_sha = hashlib.sha256((DEFAULT_DATA_DIR / "multimodal_embeddings.npz").read_bytes()).hexdigest()
    art_ok = (
        clip_sha == "4265a3a69611ef57c3f3350cdcdd971e31e10d56db4005b07a328ec410af1e7f"
        and p4a_sha == "cfa7af1cabed44767f610b724c3d28683a079eae38c79a84327e0c06afba1ae6"
    )
    record("existing_embedding_artifacts_preserved", art_ok, f"CLIP & Phase 4A NPZ SHA-256 unchanged={art_ok}")

    # 10. Prior Milestones 1–3 SHA-256 preservation
    m2_ok = hashlib.sha256((DEFAULT_DATA_DIR / "cf_milestone2_metrics.json").read_bytes()).hexdigest() == "a03bd2d0818193f7532637325d147f5c5e0a9b3f89422ff3afbed7642a026a84"
    m3_ok = hashlib.sha256((DEFAULT_DATA_DIR / "arm_milestone3_metrics.json").read_bytes()).hexdigest() == "a6b5eaaabf3c0e4df49b5ddcaac7f5dbc5a23cd769810a1bb4c9256a58341ee7"
    record("prior_milestones_preserved", m2_ok and m3_ok, f"Milestone 2={m2_ok}, Milestone 3={m3_ok}")

    return all(c["passed"] for c in checks), checks


def run_milestone4b3c_validation() -> tuple[bool, list[dict[str, Any]]]:
    """Validate frozen config, one-shot Holdout-18 evaluation, and report consistency."""
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    lock_data = json.loads(DEFAULT_LOCKED_CONFIG_JSON.read_text(encoding="utf-8"))
    holdout_data = json.loads(DEFAULT_HOLDOUT18_METRICS_JSON.read_text(encoding="utf-8"))
    report_text = DEFAULT_FINAL_REPORT_MD.read_text(encoding="utf-8")

    # 1. Frozen config SHA-256 unchanged
    canon_ok = (
        _canonical_json_sha256(lock_data["locked_models"])
        == lock_data["locked_models_canonical_sha256"]
    )
    file_sha_ok = _sha256_file(DEFAULT_LOCKED_CONFIG_JSON) == holdout_data["locked_config_sha256"]
    record("frozen_config_sha256_valid", canon_ok and file_sha_ok, f"canonical={canon_ok}, file_sha={file_sha_ok}")

    # 2. One-shot Holdout-18 & no post-Holdout tuning
    oneshot_ok = (
        holdout_data["holdout18_evaluation_count"] == 1
        and holdout_data["post_holdout_tuning"] is False
    )
    record("oneshot_holdout_no_retuning", oneshot_ok, f"count={holdout_data['holdout18_evaluation_count']}, post_tuning={holdout_data['post_holdout_tuning']}")

    # 3. Holdout-18 dataset stats & 0 Dev-18 overlap
    hs = holdout_data["holdout_18_stats"]
    h_ok = hs["num_references"] == 18 and hs["num_judged_pairs"] == 140 and hs["reference_overlap_with_dev18"] == 0
    record("holdout18_dataset_and_disjointness", h_ok, f"refs={hs['num_references']}, pairs={hs['num_judged_pairs']}, overlap={hs['reference_overlap_with_dev18']}")

    # 4. Consistent pools & finite metrics across all 7 models
    hm = holdout_data["models_holdout18_protocol_m1"]
    pools_ok = len(hm) == 7 and all(
        v["num_references"] == 18
        and v["num_judged_pairs"] == 140
        and all(0.0 <= float(v[k]) <= 1.0 and np.isfinite(float(v[k])) for k in ("precision_at_5", "strict_precision_at_5", "ndcg_at_5"))
        for v in hm.values()
    )
    record("all_7_models_holdout_metrics_valid", pools_ok, f"7 models finite & pooled (18/140)={pools_ok}")

    # 5. Missing-image fallback preserved
    mi = holdout_data["missing_image_fallback_check"]
    mi_ok = mi["clip_image_only_exact_v1_match"] is True and mi["clip_late_fusion_exact_v1_match"] is True
    record("missing_image_fallback_preserved", mi_ok, f"HM-0398947001 exact V1 match: ImageOnly={mi['clip_image_only_exact_v1_match']}, LateFusion={mi['clip_late_fusion_exact_v1_match']}")

    # 6. Report matches metrics JSON
    lf_ndcg = f"{hm['7_phase4b_clip_late_fusion']['ndcg_at_5']:.4f}"
    v1_ndcg = f"{hm['1_exact_v1_tfidf']['ndcg_at_5']:.4f}"
    rep_ok = (
        holdout_data["locked_config_sha256"] in report_text
        and lf_ndcg in report_text
        and v1_ndcg in report_text
    )
    record("final_report_matches_metrics_json", rep_ok, f"SHA & key metrics ({lf_ndcg}, {v1_ndcg}) present in report={rep_ok}")

    # 7. Original CLIP & Phase 4A caches + Milestones 1-3 unchanged
    clip_ok = _sha256_file(DEFAULT_DATA_DIR / "clip_embeddings.npz") == "4265a3a69611ef57c3f3350cdcdd971e31e10d56db4005b07a328ec410af1e7f"
    p4a_ok = _sha256_file(DEFAULT_DATA_DIR / "multimodal_embeddings.npz") == "cfa7af1cabed44767f610b724c3d28683a079eae38c79a84327e0c06afba1ae6"
    m2_ok = _sha256_file(DEFAULT_DATA_DIR / "cf_milestone2_metrics.json") == "a03bd2d0818193f7532637325d147f5c5e0a9b3f89422ff3afbed7642a026a84"
    m3_ok = _sha256_file(DEFAULT_DATA_DIR / "arm_milestone3_metrics.json") == "a6b5eaaabf3c0e4df49b5ddcaac7f5dbc5a23cd769810a1bb4c9256a58341ee7"
    record("prior_caches_and_milestones_preserved", clip_ok and p4a_ok and m2_ok and m3_ok, f"CLIP={clip_ok}, P4A={p4a_ok}, M2={m2_ok}, M3={m3_ok}")

    return all(c["passed"] for c in checks), checks


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ELAN V5 Milestone 4B Validation")
    parser.add_argument("--mode", choices=("dev18", "holdout18"), default="dev18")
    args = parser.parse_args()

    if args.mode == "dev18":
        all_passed, checks = run_milestone4b3b_validation()
        header = "ELAN V5 MILESTONE 4B.3B — DEV-18 TUNING VALIDATION"
    else:
        all_passed, checks = run_milestone4b3c_validation()
        header = "ELAN V5 MILESTONE 4B.3C — LOCKED HOLDOUT-18 VALIDATION"

    print(header)
    print("=" * 72)
    for idx, c in enumerate(checks, start=1):
        status = "PASS" if c["passed"] else "FAIL"
        print(f"{idx:2d}. [{status}] {c['check']}: {c['detail']}")
    print("=" * 72)
    if not all_passed:
        sys.exit(1)
    print(f"ALL VALIDATION CHECKS PASSED ({len(checks)}/{len(checks)}).")


if __name__ == "__main__":
    main()
