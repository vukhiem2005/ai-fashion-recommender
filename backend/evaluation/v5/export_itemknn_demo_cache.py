"""Export Precomputed Milestone 2 ItemKNN Synthetic Personalization Demo Cache.

Reuses Milestone 2 `build_train_matrix` and `ItemKNNRecommender` (`log_sum`,
`iuf_exponent=0.5`, `shrinkage=20.0`, `k_neighbors=100`) fitted strictly on the
`train` split of `SYNTHETIC_SIMULATION_V5`, exporting deterministic Top-K
recommendations and simulated interaction summaries for 6 category-representative
synthetic profiles plus a zero-history cold-start visitor (`SYN-COLD-START`).

Does NOT modify any Milestone 1–5 datasets or benchmark reports.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from backend.evaluation.v5.cf_models import (
    ItemKNNRecommender,
    build_train_matrix,
    load_active_catalog_readonly,
)
from backend.evaluation.v5.evaluate_cf_milestone2 import load_csv_records
from backend.evaluation.v5.multimodal_models import DEFAULT_DATA_DIR, DEFAULT_RELEASE_DB

DEFAULT_ITEMKNN_DEMO_JSON = DEFAULT_DATA_DIR / "itemknn_personalized_demo.json"

CURATED_SYNTHETIC_USERS: tuple[tuple[str, str], ...] = (
    ("SYN-U0001", "Simulated Shopper 01 — Tops & Casual Edit"),
    ("SYN-U0002", "Simulated Shopper 02 — Dresses & Occasion Edit"),
    ("SYN-U0003", "Simulated Shopper 03 — Outerwear & Tailoring"),
    ("SYN-U0007", "Simulated Shopper 07 — Bottoms & Denim"),
    ("SYN-U0016", "Simulated Shopper 16 — Knitwear Layering"),
    ("SYN-U0048", "Simulated Shopper 48 — Accessories Focus"),
)


def export_itemknn_demo_cache(
    output_path: Path = DEFAULT_ITEMKNN_DEMO_JSON,
    top_n_stored: int = 20,
) -> dict[str, Any]:
    users_rows = load_csv_records(DEFAULT_DATA_DIR / "synthetic_users.csv")
    interactions_rows = load_csv_records(DEFAULT_DATA_DIR / "synthetic_interactions.csv")
    catalog = load_active_catalog_readonly(DEFAULT_RELEASE_DB)

    user_ids = tuple(sorted(r["user_id"].strip() for r in users_rows))
    product_ids = tuple(str(p["id"]) for p in catalog)
    prod_by_id = {str(p["id"]): p for p in catalog}

    train_events = [r for r in interactions_rows if r["split"].strip() == "train"]
    train_bundle = build_train_matrix(
        train_interactions=train_events,
        user_ids=user_ids,
        product_ids=product_ids,
        dedup_mode="log_sum",
        iuf_exponent=0.5,
        l2_normalize_users=True,
    )

    cf_model = ItemKNNRecommender(
        k_neighbors=100, shrinkage=20.0, normalize_sim_columns=False
    ).fit(train_bundle)

    events_by_user: dict[str, list[dict[str, str]]] = defaultdict(list)
    for ev in train_events:
        events_by_user[ev["user_id"].strip()].append(ev)

    profiles: list[dict[str, Any]] = []
    for uid, label in CURATED_SYNTHETIC_USERS:
        u_idx = train_bundle.user_to_idx[uid]
        u_evs = events_by_user[uid]
        ev_counts = Counter(e["event_type"].strip() for e in u_evs)
        cat_counts = Counter(prod_by_id[e["product_id"].strip()]["category"] for e in u_evs)

        # Highlight up to 4 strongest/most recent simulated train items
        item_best: dict[str, tuple[float, str, str]] = {}
        for ev in u_evs:
            pid = ev["product_id"].strip()
            w = float(ev["implicit_weight"])
            ts = ev["event_ts"].strip()
            etype = ev["event_type"].strip()
            prev = item_best.get(pid)
            if prev is None or (w, ts) > (prev[0], prev[1]):
                item_best[pid] = (w, ts, etype)

        history_sorted = sorted(
            item_best.items(),
            key=lambda kv: (-kv[1][0], kv[1][1], kv[0]),
            reverse=False,
        )
        # Re-sort descending by (weight, ts), ascending pid
        history_sorted = sorted(
            item_best.items(),
            key=lambda kv: (-kv[1][0], kv[0]),
        )
        recent_history = [
            {
                "product_id": pid,
                "name": str(prod_by_id[pid]["name"]),
                "category": str(prod_by_id[pid]["category"]),
                "event_type": meta[2],
            }
            for pid, meta in history_sorted[:4]
        ]

        user_profile = train_bundle.normalized_matrix[u_idx]
        seen_row = train_bundle.seen_mask[u_idx]
        recs = cf_model.recommend_for_user(
            user_profile=user_profile,
            seen_mask_row=seen_row,
            top_k=top_n_stored,
            exclude_seen=True,
        )

        rec_entries: list[dict[str, Any]] = []
        for rec_pid, score in recs:
            rec_idx = train_bundle.product_to_idx[rec_pid]
            # Find top contributing train item from this user's history via sim_topk column
            contrib = user_profile * cf_model.sim_topk[:, rec_idx]
            best_src_idx = int(np.argmax(contrib))
            if float(contrib[best_src_idx]) > 1e-12:
                src_pid = product_ids[best_src_idx]
                reason = f"Co-interacted pattern with {prod_by_id[src_pid]['name']}"
            else:
                reason = "Popular item among similar catalog shoppers"

            rec_entries.append(
                {
                    "product_id": rec_pid,
                    "score": round(float(score), 6),
                    "reason": reason,
                }
            )

        profiles.append(
            {
                "user_id": uid,
                "display_label": label,
                "is_cold_start": False,
                "interaction_summary": {
                    "total_train_events": len(u_evs),
                    "distinct_products_seen": int(np.sum(seen_row)),
                    "views": int(ev_counts.get("view", 0)),
                    "add_to_carts": int(ev_counts.get("add_to_cart", 0)),
                    "purchases": int(ev_counts.get("purchase", 0)),
                    "top_categories": [c for c, _ in cat_counts.most_common(2)],
                },
                "recent_history": recent_history,
                "recommendations": rec_entries,
            }
        )

    # Zero-history Cold-Start Visitor demonstrating ItemKNN popularity fallback
    cold_profile = np.zeros(len(product_ids), dtype=np.float64)
    cold_recs = cf_model.recommend_for_user(
        user_profile=cold_profile,
        seen_mask_row=None,
        top_k=top_n_stored,
        exclude_seen=False,
    )
    profiles.append(
        {
            "user_id": "SYN-COLD-START",
            "display_label": "Cold-Start Visitor — Zero History (Popularity Fallback)",
            "is_cold_start": True,
            "interaction_summary": {
                "total_train_events": 0,
                "distinct_products_seen": 0,
                "views": 0,
                "add_to_carts": 0,
                "purchases": 0,
                "top_categories": [],
            },
            "recent_history": [],
            "recommendations": [
                {
                    "product_id": rec_pid,
                    "score": round(float(score), 6),
                    "reason": "Popular across simulated catalog sessions (cold-start fallback)",
                }
                for rec_pid, score in cold_recs
            ],
        }
    )

    payload: dict[str, Any] = {
        "model": "itemknn_cf_synthetic_demo",
        "is_synthetic_simulation": True,
        "data_source": "SYNTHETIC_SIMULATION_V5 (train split only)",
        "itemknn_config": {
            "dedup_mode": "log_sum",
            "iuf_exponent": 0.5,
            "shrinkage": 20.0,
            "k_neighbors": 100,
            "normalize_sim_columns": False,
        },
        "profiles": profiles,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload


if __name__ == "__main__":
    res = export_itemknn_demo_cache()
    print(
        json.dumps(
            {
                "status": "OK",
                "profiles_exported": len(res["profiles"]),
                "user_ids": [p["user_id"] for p in res["profiles"]],
            },
            indent=2,
        )
    )
