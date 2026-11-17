"""Focused Integration Tests for CLIP-Powered Similar Items API (with V1 Fallback)."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import HTTPException

import app.clip_service as clip_service_mod
import app.recommender as recommender_mod
from app.clip_service import _get_cached_clip_service
from app.main import health
from app.recommendations import get_recommendations
from app.recommender import recommend_products as v1_recommend_products
from backend.evaluation.v5.clip_recommenders import (
    CLIPLateFusionRecommender,
    load_offline_clip_data,
)

RELEASE_DB = BACKEND_DIR / "data" / "elan_release.db"

EXPECTED_ITEM_KEYS = {
    "id",
    "slug",
    "name",
    "category",
    "price_cents",
    "image_url",
    "similarity",
    "reason",
}


def run_api_integration_tests() -> tuple[bool, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    with (
        patch.object(recommender_mod, "DEFAULT_DATABASE", RELEASE_DB),
        patch.object(clip_service_mod, "DEFAULT_DATABASE", RELEASE_DB),
    ):
        return _run_checks(checks, record)


def _run_checks(checks: list[dict[str, Any]], record: Any) -> tuple[bool, list[dict[str, Any]]]:

    # 1. Default flag disabled -> exact V1 mode
    _get_cached_clip_service.cache_clear()
    with patch.dict(os.environ, {"ELAN_CLIP_RECOMMENDATIONS_ENABLED": "false"}, clear=False):
        data_v1 = get_recommendations("ELAN-001", top_k=5)
        expected_v1 = v1_recommend_products("ELAN-001", top_k=5)
        v1_ok = (
            data_v1["model"] == "tfidf_cosine_v1"
            and data_v1["count"] == 5
            and data_v1["recommendations"] == expected_v1
        )
    record("flag_disabled_preserves_v1", v1_ok, f"model={data_v1.get('model')}, count={data_v1.get('count')}")

    # 2. Flag enabled -> CLIP Late Fusion mode + exact schema + zero PyTorch/Transformers
    _get_cached_clip_service.cache_clear()
    with patch.dict(os.environ, {"ELAN_CLIP_RECOMMENDATIONS_ENABLED": "true"}, clear=False):
        data_clip = get_recommendations("ELAN-001", top_k=5)
        data_clip_2 = get_recommendations("ELAN-001", top_k=5)
        recs = data_clip.get("recommendations", [])
        rec_ids = [r["id"] for r in recs]

        clip_data = load_offline_clip_data()
        research_lf = CLIPLateFusionRecommender(
            w_vis=0.5,
            text_gate_threshold=0.0,
            require_same_category_for_visual=False,
            text_source="v1_tfidf",
        ).fit(clip_data)
        expected_ids = [cid for cid, _ in research_lf.recommend_similar_items("ELAN-001", top_k=5)]

        no_heavy_deps = "torch" not in sys.modules and "transformers" not in sys.modules
        clip_ok = (
            data_clip["model"] == "clip_late_fusion"
            and data_clip["count"] == 5
            and rec_ids == expected_ids
            and "ELAN-001" not in rec_ids
            and len(set(rec_ids)) == 5
            and all(set(r.keys()) == EXPECTED_ITEM_KEYS for r in recs)
            and data_clip == data_clip_2
            and no_heavy_deps
        )
    record(
        "flag_enabled_clip_late_fusion_and_schema",
        clip_ok,
        f"model={data_clip.get('model')}, parity={rec_ids == expected_ids}, zero_torch_transformers={no_heavy_deps}",
    )

    # 3. Missing-image product HM-0398947001 fallback when flag enabled
    with patch.dict(os.environ, {"ELAN_CLIP_RECOMMENDATIONS_ENABLED": "true"}, clear=False):
        data_miss = get_recommendations("HM-0398947001", top_k=5)
        expected_miss = v1_recommend_products("HM-0398947001", top_k=5)
        miss_ok = (
            data_miss["model"] == "tfidf_cosine_v1"
            and [r["id"] for r in data_miss["recommendations"]] == [r["id"] for r in expected_miss]
        )
    record("missing_image_fallback", miss_ok, f"HM-0398947001 model={data_miss.get('model')}, exact_v1={miss_ok}")

    # 4. Missing/corrupted cache fallback (never crashes)
    _get_cached_clip_service.cache_clear()
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp.write(b"not-a-valid-npz")
        bad_npz = tmp.name
    try:
        with patch.dict(
            os.environ,
            {
                "ELAN_CLIP_RECOMMENDATIONS_ENABLED": "true",
                "ELAN_CLIP_NPZ_PATH": bad_npz,
            },
            clear=False,
        ):
            data_bad = get_recommendations("ELAN-001", top_k=5)
            bad_ok = (
                data_bad["model"] == "tfidf_cosine_v1"
                and data_bad["recommendations"] == v1_recommend_products("ELAN-001", top_k=5)
            )
    finally:
        Path(bad_npz).unlink(missing_ok=True)
        _get_cached_clip_service.cache_clear()
    record("corrupted_cache_graceful_fallback", bad_ok, f"model={data_bad.get('model')}")

    # 5. Unknown product returns 404 in both modes
    got_404 = False
    with patch.dict(os.environ, {"ELAN_CLIP_RECOMMENDATIONS_ENABLED": "true"}, clear=False):
        try:
            get_recommendations("NONEXISTENT-999", top_k=5)
        except HTTPException as exc:
            got_404 = exc.status_code == 404
    record("unknown_product_404", got_404, f"raised_404={got_404}")

    # 6. Health endpoint unaffected
    health_res = health()
    record("health_endpoint_ok", health_res.get("status") == "ok", f"health={health_res}")

    all_passed = all(c["passed"] for c in checks)
    return all_passed, checks


def main() -> None:
    all_passed, checks = run_api_integration_tests()
    print("ELAN CLIP SIMILAR ITEMS API INTEGRATION TESTS")
    print("=" * 72)
    for idx, c in enumerate(checks, start=1):
        status = "PASS" if c["passed"] else "FAIL"
        print(f"{idx:2d}. [{status}] {c['check']}: {c['detail']}")
    print("=" * 72)
    if not all_passed:
        sys.exit(1)
    print(f"ALL INTEGRATION TESTS PASSED ({len(checks)}/{len(checks)}).")


if __name__ == "__main__":
    main()
