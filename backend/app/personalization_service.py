"""Synthetic ItemKNN Personalized Recommendations Demo Service for ELAN.

Serves precomputed Milestone 2 `ItemKNNRecommender` (`log_sum`, `iuf_exponent=0.5`,
`shrinkage=20.0`, `k_neighbors=100`) outputs for curated synthetic shopper profiles
and a zero-history cold-start visitor, joined live against active SQLite catalog
products. Strictly identified as a synthetic simulation demonstration.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.recommender import DEFAULT_DATABASE, _load_catalog

BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ITEMKNN_DEMO_PATH = (
    BACKEND_DIR / "evaluation" / "v5" / "data" / "itemknn_personalized_demo.json"
)


class _ItemKNNDemoCache:
    """In-memory singleton holding synthetic ItemKNN demo profiles and live catalog index."""

    def __init__(self, database_path: Path, demo_json_path: Path) -> None:
        if not demo_json_path.is_file():
            raise FileNotFoundError(
                f"ItemKNN demo cache not found: {demo_json_path}"
            )

        products = _load_catalog(database_path)
        self.products_by_id = {str(p["id"]): p for p in products}

        raw = json.loads(demo_json_path.read_text(encoding="utf-8"))
        self.model_name = str(raw.get("model", "itemknn_cf_synthetic_demo"))
        self.data_source = str(raw.get("data_source", "SYNTHETIC_SIMULATION_V5"))
        self.profiles_list = list(raw.get("profiles", []))
        self.profiles_by_id = {
            str(p["user_id"]): p for p in self.profiles_list
        }

    def list_profiles(self) -> list[dict[str, Any]]:
        return [
            {
                "user_id": p["user_id"],
                "display_label": p["display_label"],
                "is_cold_start": bool(p["is_cold_start"]),
                "interaction_summary": p["interaction_summary"],
            }
            for p in self.profiles_list
        ]

    def recommend(self, user_id: str, top_k: int) -> dict[str, Any]:
        if user_id not in self.profiles_by_id:
            raise ValueError(f"Synthetic user profile not found: {user_id}")

        profile = self.profiles_by_id[user_id]
        recent_history: list[dict[str, Any]] = []
        for h in profile.get("recent_history", []):
            pid = str(h["product_id"])
            prod = self.products_by_id.get(pid)
            if prod is None:
                continue
            recent_history.append(
                {
                    "product_id": pid,
                    "slug": prod["slug"],
                    "name": prod["name"],
                    "category": prod["category"],
                    "event_type": h["event_type"],
                }
            )

        recs: list[dict[str, Any]] = []
        seen_out: set[str] = set()
        for entry in profile.get("recommendations", []):
            pid = str(entry["product_id"])
            if pid in seen_out:
                continue
            prod = self.products_by_id.get(pid)
            if prod is None:
                continue
            seen_out.add(pid)
            recs.append(
                {
                    "id": prod["id"],
                    "slug": prod["slug"],
                    "name": prod["name"],
                    "category": prod["category"],
                    "price_cents": prod["price_cents"],
                    "image_url": prod["image_url"],
                    "reason": str(entry["reason"]),
                }
            )
            if len(recs) >= top_k:
                break

        return {
            "reference_user_id": user_id,
            "display_label": profile["display_label"],
            "model": self.model_name,
            "is_synthetic_simulation": True,
            "data_source": self.data_source,
            "cold_start_fallback": bool(profile["is_cold_start"]),
            "interaction_summary": profile["interaction_summary"],
            "recent_history": recent_history,
            "count": len(recs),
            "recommendations": recs,
        }


@lru_cache(maxsize=4)
def _get_itemknn_demo_cache(
    database_path_str: str, demo_json_path_str: str
) -> _ItemKNNDemoCache:
    return _ItemKNNDemoCache(
        database_path=Path(database_path_str),
        demo_json_path=Path(demo_json_path_str),
    )


def _resolve_paths(
    database_path: Path | None = None,
    demo_json_path: Path | None = None,
) -> tuple[Path, Path]:
    db_path = (database_path or DEFAULT_DATABASE).resolve()
    json_path = (
        demo_json_path
        or Path(os.getenv("ELAN_ITEMKNN_DEMO_JSON", str(DEFAULT_ITEMKNN_DEMO_PATH)))
    ).resolve()
    return db_path, json_path


def list_synthetic_user_profiles(
    database_path: Path | None = None,
    demo_json_path: Path | None = None,
) -> dict[str, Any]:
    db_path, json_path = _resolve_paths(database_path, demo_json_path)
    cache = _get_itemknn_demo_cache(str(db_path), str(json_path))
    return {
        "model": cache.model_name,
        "is_synthetic_simulation": True,
        "data_source": cache.data_source,
        "profiles": cache.list_profiles(),
    }


def get_synthetic_personalized_recommendations(
    user_id: str = "SYN-U0001",
    top_k: int = 5,
    database_path: Path | None = None,
    demo_json_path: Path | None = None,
) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    db_path, json_path = _resolve_paths(database_path, demo_json_path)
    cache = _get_itemknn_demo_cache(str(db_path), str(json_path))
    return cache.recommend(user_id=user_id, top_k=top_k)
