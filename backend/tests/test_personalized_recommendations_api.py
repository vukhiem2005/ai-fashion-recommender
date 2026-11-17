"""Focused Integration Tests for Synthetic ItemKNN Personalized Recommendations API."""

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
import app.personalization_service as pers_mod
import app.recommender as recommender_mod
from app.personalization_service import _get_itemknn_demo_cache
from app.recommendations import (
    get_personalized_demo_users,
    get_personalized_recommendations,
    get_recommendations,
)
from backend.evaluation.v5.cf_models import (
    ItemKNNRecommender,
    build_train_matrix,
    load_active_catalog_readonly,
)
from backend.evaluation.v5.evaluate_cf_milestone2 import load_csv_records
from backend.evaluation.v5.multimodal_models import DEFAULT_DATA_DIR

RELEASE_DB = BACKEND_DIR / "data" / "elan_release.db"
EXPECTED_KEYS = {"id", "slug", "name", "category", "price_cents", "image_url", "reason"}


def run_personalized_api_tests() -> tuple[bool, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    _get_itemknn_demo_cache.cache_clear()
    clip_service_mod._get_cached_clip_service.cache_clear()
    with (
        patch.object(recommender_mod, "DEFAULT_DATABASE", RELEASE_DB),
        patch.object(pers_mod, "DEFAULT_DATABASE", RELEASE_DB),
        patch.object(clip_service_mod, "DEFAULT_DATABASE", RELEASE_DB),
    ):
        # 1. List synthetic profiles endpoint
        users_payload = get_personalized_demo_users()
        profiles = users_payload.get("profiles", [])
        uids = [p["user_id"] for p in profiles]
        users_ok = (
            users_payload["is_synthetic_simulation"] is True
            and len(profiles) == 7
            and "SYN-U0001" in uids
            and "SYN-COLD-START" in uids
        )
        record("synthetic_profiles_list", users_ok, f"count={len(profiles)}, ids={uids}")

        # 2. ItemKNN parity against Milestone 2 cf_models for SYN-U0001 & SYN-U0002
        api_u1 = get_personalized_recommendations(user_id="SYN-U0001", top_k=5)
        api_u1_repeat = get_personalized_recommendations(user_id="SYN-U0001", top_k=5)

        users_rows = load_csv_records(DEFAULT_DATA_DIR / "synthetic_users.csv")
        inter_rows = load_csv_records(DEFAULT_DATA_DIR / "synthetic_interactions.csv")
        cat = load_active_catalog_readonly(RELEASE_DB)
        train_evs = [r for r in inter_rows if r["split"].strip() == "train"]
        bundle = build_train_matrix(
            train_interactions=train_evs,
            user_ids=tuple(sorted(r["user_id"].strip() for r in users_rows)),
            product_ids=tuple(str(p["id"]) for p in cat),
            dedup_mode="log_sum",
            iuf_exponent=0.5,
            l2_normalize_users=True,
        )
        cf = ItemKNNRecommender(k_neighbors=100, shrinkage=20.0, normalize_sim_columns=False).fit(bundle)
        u1_idx = bundle.user_to_idx["SYN-U0001"]
        expected_u1 = [
            pid
            for pid, _ in cf.recommend_for_user(
                bundle.normalized_matrix[u1_idx],
                seen_mask_row=bundle.seen_mask[u1_idx],
                top_k=5,
                exclude_seen=True,
            )
        ]

        seen_u1_pids = {
            bundle.product_ids[i]
            for i in range(len(bundle.product_ids))
            if bool(bundle.seen_mask[u1_idx, i])
        }
        api_u1_ids = [r["id"] for r in api_u1["recommendations"]]
        no_torch = "torch" not in sys.modules and "transformers" not in sys.modules
        parity_ok = (
            api_u1["is_synthetic_simulation"] is True
            and api_u1["model"] == "itemknn_cf_synthetic_demo"
            and api_u1["cold_start_fallback"] is False
            and api_u1["count"] == 5
            and api_u1_ids == expected_u1
            and api_u1 == api_u1_repeat
            and len(set(api_u1_ids) & seen_u1_pids) == 0
            and all(set(r.keys()) == EXPECTED_KEYS for r in api_u1["recommendations"])
            and no_torch
        )
        record(
            "itemknn_parity_and_seen_exclusion",
            parity_ok,
            f"SYN-U0001 parity={api_u1_ids == expected_u1}, seen_overlap=0, zero_torch={no_torch}",
        )

        # 3. Cold-start visitor triggers popularity fallback
        cold = get_personalized_recommendations(user_id="SYN-COLD-START", top_k=5)
        cold_ok = (
            cold["cold_start_fallback"] is True
            and cold["is_synthetic_simulation"] is True
            and cold["count"] == 5
            and len({r["id"] for r in cold["recommendations"]}) == 5
        )
        record("cold_start_popularity_fallback", cold_ok, f"cold_start_fallback={cold['cold_start_fallback']}")

        # 4. Unknown user_id raises 404
        got_404 = False
        try:
            get_personalized_recommendations(user_id="UNKNOWN-USER", top_k=5)
        except HTTPException as exc:
            got_404 = exc.status_code == 404
        record("unknown_user_returns_404", got_404, f"raised_404={got_404}")

        # 5. Route collision guard: `/recommendations/{product_id}` still works!
        with patch.dict(os.environ, {"ELAN_CLIP_RECOMMENDATIONS_ENABLED": "false"}, clear=False):
            v1_pdp = get_recommendations("ELAN-001", top_k=5)
        pdp_ok = v1_pdp["model"] == "tfidf_cosine_v1" and v1_pdp["count"] == 5
        record("existing_pdp_route_intact", pdp_ok, f"ELAN-001 model={v1_pdp['model']}")

        # 6. Missing demo JSON raises controlled 503 without crashing app
        _get_itemknn_demo_cache.cache_clear()
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_file = str(Path(tmpdir) / "missing.json")
            with patch.dict(os.environ, {"ELAN_ITEMKNN_DEMO_JSON": missing_file}, clear=False):
                got_503 = False
                try:
                    get_personalized_recommendations(user_id="SYN-U0001", top_k=5)
                except HTTPException as exc:
                    got_503 = exc.status_code == 503
        _get_itemknn_demo_cache.cache_clear()
        record("missing_cache_returns_503", got_503, f"raised_503={got_503}")

    all_passed = all(c["passed"] for c in checks)
    return all_passed, checks


def main() -> None:
    all_passed, checks = run_personalized_api_tests()
    print("ELAN SYNTHETIC ITEMKNN PERSONALIZATION API TESTS")
    print("=" * 72)
    for idx, c in enumerate(checks, start=1):
        status = "PASS" if c["passed"] else "FAIL"
        print(f"{idx:2d}. [{status}] {c['check']}: {c['detail']}")
    print("=" * 72)
    if not all_passed:
        sys.exit(1)
    print(f"ALL PERSONALIZATION API TESTS PASSED ({len(checks)}/{len(checks)}).")


if __name__ == "__main__":
    main()
