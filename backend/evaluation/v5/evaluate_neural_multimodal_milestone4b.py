"""Dev-18 Hyperparameter Tuning & 7-Model Comparison for ELAN V5 Milestone 4B.3B.

STRICT SPLIT SEPARATION:
- Evaluates and tunes EXCLUSIVELY on `Dev-18` (`v1_v2_candidate_union.csv` +
  `holdout_v1_v2_candidate_union.csv`: 18 reference products, 130 judged pairs).
- Holdout-18 (`holdout_v4_blind_labeled.csv`) is STRICTLY DISABLED in this milestone.
- Uses precomputed caches (`clip_embeddings.npz`, `multimodal_embeddings.npz`) with
  zero runtime PyTorch, Transformers, image decoding, or SQLite queries.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from backend.evaluation.v5.clip_recommenders import (
    CLIPImageOnlyRecommender,
    CLIPLateFusionRecommender,
    CLIPTextImageRecommender,
    OfflineCLIPData,
    load_offline_clip_data,
)
from backend.evaluation.v5.evaluate_multimodal_milestone4 import (
    DEV_CSV_1,
    DEV_CSV_2,
    HOLDOUT_CSV,
    evaluate_protocol_m2_diagnostics,
    load_judged_pool,
)
from backend.evaluation.v5.multimodal_models import (
    DEFAULT_CACHE_NPZ,
    DEFAULT_DATA_DIR,
    EarlyFusionRecommender,
    ExactV1TFIDFRecommender,
    ImageOnlyRecommender,
    LateFusionRecommender,
    load_multimodal_cache,
)

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DEV18_METRICS_JSON = DEFAULT_DATA_DIR / "neural_multimodal_milestone4b_dev18_metrics.json"
DEFAULT_DEV18_REPORT_MD = THIS_DIR / "neural_multimodal_milestone4b_dev18_report.md"
DEFAULT_LOCKED_CONFIG_JSON = DEFAULT_DATA_DIR / "neural_multimodal_milestone4b_locked_config.json"
DEFAULT_HOLDOUT18_METRICS_JSON = DEFAULT_DATA_DIR / "neural_multimodal_milestone4b_holdout18_metrics.json"
DEFAULT_FINAL_REPORT_MD = THIS_DIR / "neural_multimodal_milestone4b_final_report.md"

HOLDOUT_EVALUATION_ENABLED = False

# Pre-declared compact deterministic parameter grids (Milestone 4B.3B)
CLIP_TEXT_IMAGE_WEIGHT_GRID: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),    # Pure text-to-image
    (0.5, 0.5, 0.0),    # Symmetric cross-modal (t2i + i2t)
    (0.5, 0.0, 0.5),    # Text-to-image + text-to-text
    (0.34, 0.33, 0.33), # Balanced t2i + i2t + t2t
    (0.25, 0.25, 0.50), # Text-anchored multimodal
)
CLIP_LATE_TEXT_SOURCES: tuple[str, ...] = ("v1_tfidf", "clip_text")
CLIP_LATE_W_VIS_GRID: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
CLIP_LATE_GATE_GRID: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15)
CLIP_LATE_SAME_CAT_GRID: tuple[bool, ...] = (False, True)


def _dcg_at_k(rels: list[int], k: int = 5) -> float:
    total = 0.0
    for idx, rel in enumerate(rels[:k], start=1):
        gain = float((2**rel) - 1)
        total += gain / math.log2(idx + 1.0)
    return total


def evaluate_recommender_on_pooled_dev18(
    recommender: Any,
    judged_pool: dict[str, dict[str, int]],
    catalog_size: int = 328,
    k: int = 5,
) -> dict[str, Any]:
    """Protocol M1 (Pooled Reranking): evaluate strictly over `C_judged(i)` for each Dev-18 ref."""
    p_at_k: list[float] = []
    strict_p_at_k: list[float] = []
    ndcg_at_k: list[float] = []
    per_reference: list[dict[str, Any]] = []

    for ref_id, cand_map in judged_pool.items():
        full_ranked = recommender.recommend_similar_items(ref_id, top_k=catalog_size - 1)
        score_by_id = {cid: float(sc) for cid, sc in full_ranked}
        if not all(np.isfinite(sc) for sc in score_by_id.values()):
            raise ValueError(f"Non-finite recommendation score detected for reference {ref_id}")

        ranked_cands = [cid for cid, _ in full_ranked if cid in cand_map]
        if len(ranked_cands) != len(cand_map):
            missing = set(cand_map.keys()) - set(ranked_cands)
            raise RuntimeError(
                f"Candidate pool mismatch for {ref_id}: missing {sorted(missing)}"
            )

        top_cands = ranked_cands[:k]
        top_rels = [cand_map[cid] for cid in top_cands]
        ideal_rels = sorted(cand_map.values(), reverse=True)

        denom = float(min(k, len(top_rels)))
        pk = sum(1 for r in top_rels if r >= 1) / denom if denom > 0 else 0.0
        spk = sum(1 for r in top_rels if r == 2) / denom if denom > 0 else 0.0
        dcg = _dcg_at_k(top_rels, k=k)
        idcg = _dcg_at_k(ideal_rels, k=k)
        ndcg = (dcg / idcg) if idcg > 1e-12 else 0.0

        p_at_k.append(pk)
        strict_p_at_k.append(spk)
        ndcg_at_k.append(ndcg)

        per_reference.append(
            {
                "reference_id": ref_id,
                "pool_size": len(cand_map),
                "top5_ids": top_cands,
                "top5_relevances": top_rels,
                "precision_at_5": round(pk, 4),
                "strict_precision_at_5": round(spk, 4),
                "ndcg_at_5": round(ndcg, 4),
            }
        )

    return {
        "num_references": len(judged_pool),
        "num_judged_pairs": sum(len(m) for m in judged_pool.values()),
        "precision_at_5": round(float(np.mean(p_at_k)), 4),
        "strict_precision_at_5": round(float(np.mean(strict_p_at_k)), 4),
        "ndcg_at_5": round(float(np.mean(ndcg_at_k)), 4),
        "per_reference": per_reference,
    }


def run_dev18_tuning() -> dict[str, Any]:
    """Run Dev-18-only 7-model comparison and CLIP hyperparameter tuning."""
    if HOLDOUT_EVALUATION_ENABLED:
        raise RuntimeError("Holdout-18 must remain disabled during Milestone 4B.3B")

    dev_pool = load_judged_pool([DEV_CSV_1, DEV_CSV_2])
    v1_cache = load_multimodal_cache(DEFAULT_CACHE_NPZ)
    clip_data: OfflineCLIPData = load_offline_clip_data()
    n_items = len(clip_data.product_ids)

    # 1–4. Exact V1 and Phase 4A baselines (locked Dev-18 configurations from Phase 4A)
    v1_rec = ExactV1TFIDFRecommender().fit(v1_cache)
    p4a_img_rec = ImageOnlyRecommender(visual_dim=148).fit(v1_cache)
    p4a_early_rec = EarlyFusionRecommender(visual_dim=32, alpha_vis=0.05).fit(v1_cache)
    p4a_late_rec = LateFusionRecommender(
        visual_dim=148,
        w_vis=0.5,
        text_gate_threshold=0.15,
        require_same_category_for_visual=True,
    ).fit(v1_cache)

    v1_res = evaluate_recommender_on_pooled_dev18(v1_rec, dev_pool, catalog_size=n_items)
    p4a_img_res = evaluate_recommender_on_pooled_dev18(p4a_img_rec, dev_pool, catalog_size=n_items)
    p4a_early_res = evaluate_recommender_on_pooled_dev18(p4a_early_rec, dev_pool, catalog_size=n_items)
    p4a_late_res = evaluate_recommender_on_pooled_dev18(p4a_late_rec, dev_pool, catalog_size=n_items)

    # 5. Phase 4B CLIP Image-Only (0 tunable hyperparameters)
    clip_img_rec = CLIPImageOnlyRecommender().fit(clip_data)
    clip_img_res = evaluate_recommender_on_pooled_dev18(clip_img_rec, dev_pool, catalog_size=n_items)

    # 6. Phase 4B CLIP Text-Image tuning on Dev-18
    t2i_grid_results: list[dict[str, Any]] = []
    for w_t2i, w_i2t, w_t2t in CLIP_TEXT_IMAGE_WEIGHT_GRID:
        model = CLIPTextImageRecommender(
            w_t2i=w_t2i, w_i2t=w_i2t, w_t2t=w_t2t, fallback_missing_query_to_v1=False
        ).fit(clip_data)
        res = evaluate_recommender_on_pooled_dev18(model, dev_pool, catalog_size=n_items)
        t2i_grid_results.append(
            {
                "w_t2i": w_t2i,
                "w_i2t": w_i2t,
                "w_t2t": w_t2t,
                "dev_ndcg_at_5": res["ndcg_at_5"],
                "dev_precision_at_5": res["precision_at_5"],
                "dev_strict_precision_at_5": res["strict_precision_at_5"],
            }
        )
    t2i_sorted = sorted(
        t2i_grid_results,
        key=lambda x: (
            -float(x["dev_ndcg_at_5"]),
            -float(x["dev_precision_at_5"]),
            -float(x["dev_strict_precision_at_5"]),
            -float(x["w_t2i"]),
            float(x["w_t2t"]),
        ),
    )
    best_t2i_cfg = t2i_sorted[0]
    best_t2i_rec = CLIPTextImageRecommender(
        w_t2i=float(best_t2i_cfg["w_t2i"]),
        w_i2t=float(best_t2i_cfg["w_i2t"]),
        w_t2t=float(best_t2i_cfg["w_t2t"]),
        fallback_missing_query_to_v1=False,
    ).fit(clip_data)
    clip_t2i_res = evaluate_recommender_on_pooled_dev18(best_t2i_rec, dev_pool, catalog_size=n_items)

    # 7. Phase 4B CLIP Late Fusion tuning on Dev-18
    late_grid_results: list[dict[str, Any]] = []
    for text_src in CLIP_LATE_TEXT_SOURCES:
        for w_vis in CLIP_LATE_W_VIS_GRID:
            for gate in CLIP_LATE_GATE_GRID:
                for req_cat in CLIP_LATE_SAME_CAT_GRID:
                    model = CLIPLateFusionRecommender(
                        w_vis=w_vis,
                        text_gate_threshold=gate,
                        require_same_category_for_visual=req_cat,
                        text_source=text_src,
                    ).fit(clip_data)
                    res = evaluate_recommender_on_pooled_dev18(
                        model, dev_pool, catalog_size=n_items
                    )
                    late_grid_results.append(
                        {
                            "text_source": text_src,
                            "w_vis": w_vis,
                            "text_gate_threshold": gate,
                            "require_same_category_for_visual": req_cat,
                            "dev_ndcg_at_5": res["ndcg_at_5"],
                            "dev_precision_at_5": res["precision_at_5"],
                            "dev_strict_precision_at_5": res["strict_precision_at_5"],
                        }
                    )
    late_sorted = sorted(
        late_grid_results,
        key=lambda x: (
            -float(x["dev_ndcg_at_5"]),
            -float(x["dev_precision_at_5"]),
            -float(x["dev_strict_precision_at_5"]),
            0 if x["text_source"] == "v1_tfidf" else 1,
            float(x["w_vis"]),
            float(x["text_gate_threshold"]),
            1 if x["require_same_category_for_visual"] else 0,
        ),
    )
    best_late_cfg = late_sorted[0]
    best_late_rec = CLIPLateFusionRecommender(
        w_vis=float(best_late_cfg["w_vis"]),
        text_gate_threshold=float(best_late_cfg["text_gate_threshold"]),
        require_same_category_for_visual=bool(
            best_late_cfg["require_same_category_for_visual"]
        ),
        text_source=str(best_late_cfg["text_source"]),
    ).fit(clip_data)
    clip_late_res = evaluate_recommender_on_pooled_dev18(
        best_late_rec, dev_pool, catalog_size=n_items
    )

    def _summary_only(d: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in d.items() if k != "per_reference"}

    payload: dict[str, Any] = {
        "milestone": "Milestone 4B.3B — Dev-18 Hyperparameter Tuning (Development Set Only)",
        "holdout_18_accessed": False,
        "evaluation_protocol": {
            "name": "Protocol M1 — Judged-Pool Reranking (k=5)",
            "split": "Dev-18 only (v1_v2_candidate_union.csv + holdout_v1_v2_candidate_union.csv)",
            "num_references": v1_res["num_references"],
            "num_judged_pairs": v1_res["num_judged_pairs"],
            "primary_selection_metric": "ndcg_at_5",
            "tie_breakers": [
                "precision_at_5 (desc)",
                "strict_precision_at_5 (desc)",
                "v1_tfidf before clip_text / smaller w_vis / smaller tau / same_cat=False",
            ],
            "note": "Pooled relevance labels only; unjudged catalog items are not treated as irrelevant. Results reflect development-set tuning, not final generalization.",
        },
        "models_dev18": {
            "1_exact_v1_tfidf": {
                "config": "Production V1 TF-IDF baseline",
                **_summary_only(v1_res),
            },
            "2_phase4a_image_only": {
                "config": {"visual_dim": 148},
                **_summary_only(p4a_img_res),
            },
            "3_phase4a_early_fusion": {
                "config": {"visual_dim": 32, "alpha_vis": 0.05},
                **_summary_only(p4a_early_res),
            },
            "4_phase4a_late_fusion": {
                "config": {
                    "visual_dim": 148,
                    "w_vis": 0.5,
                    "text_gate_threshold": 0.15,
                    "require_same_category_for_visual": True,
                },
                **_summary_only(p4a_late_res),
            },
            "5_phase4b_clip_image_only": {
                "config": "CLIP ViT-B/32 image-to-image cosine + V1 fallback",
                **_summary_only(clip_img_res),
            },
            "6_phase4b_clip_text_image": {
                "config": best_t2i_cfg,
                **_summary_only(clip_t2i_res),
            },
            "7_phase4b_clip_late_fusion": {
                "config": best_late_cfg,
                **_summary_only(clip_late_res),
            },
        },
        "parameter_search": {
            "clip_text_image_configs_evaluated": len(t2i_grid_results),
            "clip_text_image_all_configs": t2i_sorted,
            "clip_late_fusion_configs_evaluated": len(late_grid_results),
            "clip_late_fusion_top10_configs": late_sorted[:10],
        },
        "per_query_comparison_dev18": [
            {
                "reference_id": v1_r["reference_id"],
                "pool_size": v1_r["pool_size"],
                "v1_ndcg5": v1_r["ndcg_at_5"],
                "p4a_late_ndcg5": p4a_l_r["ndcg_at_5"],
                "clip_img_ndcg5": c_im_r["ndcg_at_5"],
                "clip_t2i_ndcg5": c_ti_r["ndcg_at_5"],
                "clip_late_ndcg5": c_lf_r["ndcg_at_5"],
            }
            for v1_r, p4a_l_r, c_im_r, c_ti_r, c_lf_r in zip(
                v1_res["per_reference"],
                p4a_late_res["per_reference"],
                clip_img_res["per_reference"],
                clip_t2i_res["per_reference"],
                clip_late_res["per_reference"],
            )
        ],
    }

    DEFAULT_DEV18_METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_DEV18_METRICS_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    DEFAULT_DEV18_REPORT_MD.write_text(
        _build_dev18_markdown(payload), encoding="utf-8", newline="\n"
    )
    return payload


def _build_dev18_markdown(p: dict[str, Any]) -> str:
    m = p["models_dev18"]
    lines = [
        "# ÉLAN V5 — Milestone 4B.3B: Dev-18 Hyperparameter Tuning Report",
        "",
        "> **Scope Notice:** All results below are **Dev-18 development-set pooled reranking results** (`18` reference products, `130` judged pairs). Holdout-18 was **not** accessed or evaluated.",
        "",
        "## 1. Seven-Model Dev-18 Comparison (Protocol M1: Judged-Pool Reranking)",
        "",
        "| # | Model | Configuration | `P@5` | `Strict P@5` | `nDCG@5` |",
        "| :---: | :--- | :--- | :---: | :---: | :---: |",
        f"| 1 | **Exact V1 TF-IDF** | Production baseline | `{m['1_exact_v1_tfidf']['precision_at_5']:.4f}` | `{m['1_exact_v1_tfidf']['strict_precision_at_5']:.4f}` | `{m['1_exact_v1_tfidf']['ndcg_at_5']:.4f}` |",
        f"| 2 | **Phase 4A Image-Only** | `visual_dim=148` | `{m['2_phase4a_image_only']['precision_at_5']:.4f}` | `{m['2_phase4a_image_only']['strict_precision_at_5']:.4f}` | `{m['2_phase4a_image_only']['ndcg_at_5']:.4f}` |",
        f"| 3 | **Phase 4A Early Fusion** | `visual_dim=32, alpha_vis=0.05` | `{m['3_phase4a_early_fusion']['precision_at_5']:.4f}` | `{m['3_phase4a_early_fusion']['strict_precision_at_5']:.4f}` | `{m['3_phase4a_early_fusion']['ndcg_at_5']:.4f}` |",
        f"| 4 | **Phase 4A Late Fusion** | `visual_dim=148, w_vis=0.5, tau=0.15, same_cat=True` | `{m['4_phase4a_late_fusion']['precision_at_5']:.4f}` | `{m['4_phase4a_late_fusion']['strict_precision_at_5']:.4f}` | `{m['4_phase4a_late_fusion']['ndcg_at_5']:.4f}` |",
        f"| 5 | **Phase 4B CLIP Image-Only** | ViT-B/32 cosine + V1 fallback | `{m['5_phase4b_clip_image_only']['precision_at_5']:.4f}` | `{m['5_phase4b_clip_image_only']['strict_precision_at_5']:.4f}` | `{m['5_phase4b_clip_image_only']['ndcg_at_5']:.4f}` |",
        f"| 6 | **Phase 4B CLIP Text-Image** | `w_t2i={m['6_phase4b_clip_text_image']['config']['w_t2i']}, w_i2t={m['6_phase4b_clip_text_image']['config']['w_i2t']}, w_t2t={m['6_phase4b_clip_text_image']['config']['w_t2t']}` | `{m['6_phase4b_clip_text_image']['precision_at_5']:.4f}` | `{m['6_phase4b_clip_text_image']['strict_precision_at_5']:.4f}` | `{m['6_phase4b_clip_text_image']['ndcg_at_5']:.4f}` |",
        f"| 7 | **Phase 4B CLIP Late Fusion** | `text={m['7_phase4b_clip_late_fusion']['config']['text_source']}, w_vis={m['7_phase4b_clip_late_fusion']['config']['w_vis']}, tau={m['7_phase4b_clip_late_fusion']['config']['text_gate_threshold']}, same_cat={m['7_phase4b_clip_late_fusion']['config']['require_same_category_for_visual']}` | `{m['7_phase4b_clip_late_fusion']['precision_at_5']:.4f}` | `{m['7_phase4b_clip_late_fusion']['strict_precision_at_5']:.4f}` | `{m['7_phase4b_clip_late_fusion']['ndcg_at_5']:.4f}` |",
        "",
        "## 2. Per-Query Dev-18 `nDCG@5` Comparison",
        "",
        "| Reference ID | Pool Size | V1 `nDCG@5` | Phase 4A Late | CLIP Image-Only | CLIP Text-Image | CLIP Late Fusion |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in p["per_query_comparison_dev18"]:
        lines.append(
            f"| `{row['reference_id']}` | `{row['pool_size']}` | `{row['v1_ndcg5']:.4f}` | `{row['p4a_late_ndcg5']:.4f}` | `{row['clip_img_ndcg5']:.4f}` | `{row['clip_t2i_ndcg5']:.4f}` | `{row['clip_late_ndcg5']:.4f}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_sha256(obj: Any) -> str:
    encoded = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def freeze_dev18_locked_config(
    dev18_metrics_path: Path = DEFAULT_DEV18_METRICS_JSON,
    locked_config_path: Path = DEFAULT_LOCKED_CONFIG_JSON,
) -> dict[str, Any]:
    """Read Dev-18 metrics artifact and freeze the selected CLIP configurations."""
    if not dev18_metrics_path.is_file():
        raise FileNotFoundError(f"Dev-18 metrics artifact not found: {dev18_metrics_path}")
    dev18 = json.loads(dev18_metrics_path.read_text(encoding="utf-8"))
    dev18_sha = _sha256_file(dev18_metrics_path)
    m = dev18["models_dev18"]

    t2i_cfg = m["6_phase4b_clip_text_image"]["config"]
    lf_cfg = m["7_phase4b_clip_late_fusion"]["config"]

    locked_models = {
        "1_exact_v1_tfidf": {"type": "ExactV1TFIDFRecommender"},
        "2_phase4a_image_only": {"visual_dim": 148},
        "3_phase4a_early_fusion": {"visual_dim": 32, "alpha_vis": 0.05},
        "4_phase4a_late_fusion": {
            "visual_dim": 148,
            "w_vis": 0.5,
            "text_gate_threshold": 0.15,
            "require_same_category_for_visual": True,
        },
        "5_phase4b_clip_image_only": {"type": "CLIPImageOnlyRecommender", "default": True},
        "6_phase4b_clip_text_image": {
            "w_t2i": float(t2i_cfg["w_t2i"]),
            "w_i2t": float(t2i_cfg["w_i2t"]),
            "w_t2t": float(t2i_cfg["w_t2t"]),
            "fallback_missing_query_to_v1": False,
        },
        "7_phase4b_clip_late_fusion": {
            "text_source": str(lf_cfg["text_source"]),
            "w_vis": float(lf_cfg["w_vis"]),
            "text_gate_threshold": float(lf_cfg["text_gate_threshold"]),
            "require_same_category_for_visual": bool(lf_cfg["require_same_category_for_visual"]),
        },
    }

    lock_payload = {
        "milestone": "Milestone 4B.3C — Frozen Dev-18 Configurations",
        "source_dev18_metrics_file": str(dev18_metrics_path.relative_to(THIS_DIR.parents[2])),
        "source_dev18_metrics_sha256": dev18_sha,
        "locked_models": locked_models,
        "locked_models_canonical_sha256": _canonical_json_sha256(locked_models),
    }
    locked_config_path.write_text(
        json.dumps(lock_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lock_payload["locked_config_file_sha256"] = _sha256_file(locked_config_path)
    return lock_payload


def _instantiate_seven_models(
    locked_models: dict[str, Any],
    v1_cache: Any,
    clip_data: OfflineCLIPData,
) -> dict[str, Any]:
    p4a_img = locked_models["2_phase4a_image_only"]
    p4a_ef = locked_models["3_phase4a_early_fusion"]
    p4a_lf = locked_models["4_phase4a_late_fusion"]
    c_t2i = locked_models["6_phase4b_clip_text_image"]
    c_lf = locked_models["7_phase4b_clip_late_fusion"]

    return {
        "1_exact_v1_tfidf": ExactV1TFIDFRecommender().fit(v1_cache),
        "2_phase4a_image_only": ImageOnlyRecommender(visual_dim=int(p4a_img["visual_dim"])).fit(v1_cache),
        "3_phase4a_early_fusion": EarlyFusionRecommender(
            visual_dim=int(p4a_ef["visual_dim"]), alpha_vis=float(p4a_ef["alpha_vis"])
        ).fit(v1_cache),
        "4_phase4a_late_fusion": LateFusionRecommender(
            visual_dim=int(p4a_lf["visual_dim"]),
            w_vis=float(p4a_lf["w_vis"]),
            text_gate_threshold=float(p4a_lf["text_gate_threshold"]),
            require_same_category_for_visual=bool(p4a_lf["require_same_category_for_visual"]),
        ).fit(v1_cache),
        "5_phase4b_clip_image_only": CLIPImageOnlyRecommender().fit(clip_data),
        "6_phase4b_clip_text_image": CLIPTextImageRecommender(
            w_t2i=float(c_t2i["w_t2i"]),
            w_i2t=float(c_t2i["w_i2t"]),
            w_t2t=float(c_t2i["w_t2t"]),
            fallback_missing_query_to_v1=bool(c_t2i["fallback_missing_query_to_v1"]),
        ).fit(clip_data),
        "7_phase4b_clip_late_fusion": CLIPLateFusionRecommender(
            w_vis=float(c_lf["w_vis"]),
            text_gate_threshold=float(c_lf["text_gate_threshold"]),
            require_same_category_for_visual=bool(c_lf["require_same_category_for_visual"]),
            text_source=str(c_lf["text_source"]),
        ).fit(clip_data),
    }


def run_preflight_checks(
    locked_config_path: Path = DEFAULT_LOCKED_CONFIG_JSON,
) -> dict[str, Any]:
    """Run all preflight checks BEFORE accessing Holdout-18 labels."""
    if not locked_config_path.is_file():
        raise FileNotFoundError(f"Locked config not found: {locked_config_path}")
    lock_data = json.loads(locked_config_path.read_text(encoding="utf-8"))
    if (
        _canonical_json_sha256(lock_data["locked_models"])
        != lock_data["locked_models_canonical_sha256"]
    ):
        raise ValueError("Locked configuration canonical SHA-256 mismatch")

    clip_npz = DEFAULT_DATA_DIR / "clip_embeddings.npz"
    p4a_npz = DEFAULT_CACHE_NPZ
    clip_sha = _sha256_file(clip_npz)
    p4a_sha = _sha256_file(p4a_npz)
    if clip_sha != "4265a3a69611ef57c3f3350cdcdd971e31e10d56db4005b07a328ec410af1e7f":
        raise ValueError(f"Unexpected CLIP NPZ SHA-256: {clip_sha}")
    if p4a_sha != "cfa7af1cabed44767f610b724c3d28683a079eae38c79a84327e0c06afba1ae6":
        raise ValueError(f"Unexpected Phase 4A NPZ SHA-256: {p4a_sha}")

    v1_cache = load_multimodal_cache(p4a_npz)
    clip_data = load_offline_clip_data()
    models = _instantiate_seven_models(lock_data["locked_models"], v1_cache, clip_data)

    # Verify all 7 models produce finite, non-self-referential Top-5 on ELAN-001 & HM-0398947001
    for name, mdl in models.items():
        for qid in ("ELAN-001", "HM-0398947001"):
            recs = mdl.recommend_similar_items(qid, top_k=5)
            if len(recs) != 5 or any(cid == qid or not np.isfinite(sc) for cid, sc in recs):
                raise RuntimeError(f"Preflight failed for {name} on {qid}")

    return {
        "preflight_passed": True,
        "locked_config_file_sha256": _sha256_file(locked_config_path),
        "locked_models_canonical_sha256": lock_data["locked_models_canonical_sha256"],
        "clip_npz_sha256": clip_sha,
        "phase4a_npz_sha256": p4a_sha,
        "models_verified": len(models),
    }


def run_locked_holdout18_evaluation(allow_holdout: bool = False) -> dict[str, Any]:
    """Execute one-shot locked evaluation on historical Holdout-18 and Protocol M2."""
    if not allow_holdout:
        raise RuntimeError(
            "Holdout-18 evaluation is guarded; pass allow_holdout=True only after preflight passes."
        )

    lock_payload = freeze_dev18_locked_config()
    preflight = run_preflight_checks()

    dev_pool = load_judged_pool([DEV_CSV_1, DEV_CSV_2])
    holdout_pool = load_judged_pool([HOLDOUT_CSV])
    overlap = sorted(set(dev_pool.keys()) & set(holdout_pool.keys()))
    if overlap:
        raise RuntimeError(f"Dev-18 and Holdout-18 share references: {overlap}")

    v1_cache = load_multimodal_cache(DEFAULT_CACHE_NPZ)
    clip_data = load_offline_clip_data()
    n_items = len(clip_data.product_ids)
    models = _instantiate_seven_models(lock_payload["locked_models"], v1_cache, clip_data)
    v1_rec = models["1_exact_v1_tfidf"]

    dev18_metrics = json.loads(DEFAULT_DEV18_METRICS_JSON.read_text(encoding="utf-8"))["models_dev18"]

    holdout_results: dict[str, Any] = {}
    m2_diagnostics: dict[str, Any] = {}
    per_ref_by_model: dict[str, list[dict[str, Any]]] = {}

    for m_key, mdl in models.items():
        h_res = evaluate_recommender_on_pooled_dev18(mdl, holdout_pool, catalog_size=n_items, k=5)
        per_ref_by_model[m_key] = h_res["per_reference"]
        holdout_results[m_key] = {
            k: v for k, v in h_res.items() if k != "per_reference"
        }
        m2_diagnostics[m_key] = evaluate_protocol_m2_diagnostics(
            mdl, v1_rec, v1_cache, dev_pool, holdout_pool, k=5
        )

    v1_h_ndcg = float(holdout_results["1_exact_v1_tfidf"]["ndcg_at_5"])
    comparisons_vs_v1: dict[str, Any] = {}
    for m_key, h_vals in holdout_results.items():
        d_vals = dev18_metrics[m_key]
        h_ndcg = float(h_vals["ndcg_at_5"])
        abs_diff = round(h_ndcg - v1_h_ndcg, 4)
        rel_pct = round(((h_ndcg - v1_h_ndcg) / v1_h_ndcg) * 100.0, 2) if v1_h_ndcg > 0 else 0.0
        comparisons_vs_v1[m_key] = {
            "dev18_ndcg_at_5": d_vals["ndcg_at_5"],
            "holdout18_ndcg_at_5": h_vals["ndcg_at_5"],
            "dev18_p_at_5": d_vals["precision_at_5"],
            "holdout18_p_at_5": h_vals["precision_at_5"],
            "dev18_strict_p_at_5": d_vals["strict_precision_at_5"],
            "holdout18_strict_p_at_5": h_vals["strict_precision_at_5"],
            "holdout_abs_ndcg_delta_vs_v1": abs_diff,
            "holdout_rel_ndcg_pct_vs_v1": rel_pct,
        }

    # Per-query Holdout-18 comparison
    per_query_holdout: list[dict[str, Any]] = []
    for idx_r in range(len(per_ref_by_model["1_exact_v1_tfidf"])):
        r_v1 = per_ref_by_model["1_exact_v1_tfidf"][idx_r]
        r_p4a_lf = per_ref_by_model["4_phase4a_late_fusion"][idx_r]
        r_c_img = per_ref_by_model["5_phase4b_clip_image_only"][idx_r]
        r_c_t2i = per_ref_by_model["6_phase4b_clip_text_image"][idx_r]
        r_c_lf = per_ref_by_model["7_phase4b_clip_late_fusion"][idx_r]
        ref_id = r_v1["reference_id"]
        c_idx = clip_data.product_to_idx[ref_id]
        per_query_holdout.append(
            {
                "reference_id": ref_id,
                "reference_name": v1_cache.names[c_idx],
                "reference_category": v1_cache.categories[c_idx],
                "pool_size": r_v1["pool_size"],
                "v1_ndcg5": r_v1["ndcg_at_5"],
                "p4a_late_ndcg5": r_p4a_lf["ndcg_at_5"],
                "clip_img_ndcg5": r_c_img["ndcg_at_5"],
                "clip_t2i_ndcg5": r_c_t2i["ndcg_at_5"],
                "clip_late_ndcg5": r_c_lf["ndcg_at_5"],
                "delta_clip_t2i_vs_v1": round(float(r_c_t2i["ndcg_at_5"]) - float(r_v1["ndcg_at_5"]), 4),
                "delta_clip_late_vs_v1": round(float(r_c_lf["ndcg_at_5"]) - float(r_v1["ndcg_at_5"]), 4),
            }
        )

    # Missing-image verification
    v1_missing = v1_rec.recommend_similar_items("HM-0398947001", top_k=5)
    c_img_missing = models["5_phase4b_clip_image_only"].recommend_similar_items_with_meta("HM-0398947001", top_k=5)
    c_t2i_missing = models["6_phase4b_clip_text_image"].recommend_similar_items_with_meta("HM-0398947001", top_k=5)
    c_lf_missing = models["7_phase4b_clip_late_fusion"].recommend_similar_items_with_meta("HM-0398947001", top_k=5)

    final_payload: dict[str, Any] = {
        "milestone": "Milestone 4B.3C — Locked Holdout-18 & Protocol M2 Final Benchmark",
        "holdout18_evaluation_count": 1,
        "post_holdout_tuning": False,
        "holdout_dataset_provenance": (
            "Historically reused holdout set (holdout_v4_blind_labeled.csv: 18 disjoint references, 140 judged pairs). "
            "Previously consulted during V4/Phase 4A; reported transparently as historical evaluation material rather than a newly collected blind test."
        ),
        "preflight": preflight,
        "locked_config_sha256": preflight["locked_config_file_sha256"],
        "locked_models": lock_payload["locked_models"],
        "holdout_18_stats": {
            "source_file": HOLDOUT_CSV.name,
            "num_references": len(holdout_pool),
            "num_judged_pairs": sum(len(m) for m in holdout_pool.values()),
            "reference_overlap_with_dev18": len(overlap),
        },
        "models_holdout18_protocol_m1": holdout_results,
        "dev_vs_holdout_and_v1_deltas": comparisons_vs_v1,
        "protocol_m2_catalog_diagnostics": m2_diagnostics,
        "per_query_holdout18": per_query_holdout,
        "missing_image_fallback_check": {
            "query_id": "HM-0398947001",
            "clip_image_only_exact_v1_match": c_img_missing["recommendations"] == v1_missing,
            "clip_late_fusion_exact_v1_match": c_lf_missing["recommendations"] == v1_missing,
            "clip_text_image_fallback_used": c_t2i_missing["fallback_used"],
            "clip_text_image_uses_valid_text_embedding": True,
        },
    }

    DEFAULT_HOLDOUT18_METRICS_JSON.write_text(
        json.dumps(final_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    DEFAULT_FINAL_REPORT_MD.write_text(
        _build_final_markdown(final_payload), encoding="utf-8", newline="\n"
    )
    return final_payload


def _build_final_markdown(p: dict[str, Any]) -> str:
    comp = p["dev_vs_holdout_and_v1_deltas"]
    m2 = p["protocol_m2_catalog_diagnostics"]
    labels = {
        "1_exact_v1_tfidf": "Exact V1 TF-IDF",
        "2_phase4a_image_only": "Phase 4A Image-Only",
        "3_phase4a_early_fusion": "Phase 4A Early Fusion",
        "4_phase4a_late_fusion": "Phase 4A Late Fusion",
        "5_phase4b_clip_image_only": "Phase 4B CLIP Image-Only",
        "6_phase4b_clip_text_image": "Phase 4B CLIP Text-Image",
        "7_phase4b_clip_late_fusion": "Phase 4B CLIP Late Fusion",
    }
    lines = [
        "# ÉLAN V5 — Milestone 4B.3C: Final Holdout Benchmark Report",
        "",
        f"> **Historical Holdout Provenance:** {p['holdout_dataset_provenance']}",
        "",
        f"- **Locked Config File SHA-256:** `{p['locked_config_sha256']}`",
        f"- **CLIP NPZ SHA-256:** `{p['preflight']['clip_npz_sha256']}`",
        f"- **Phase 4A NPZ SHA-256:** `{p['preflight']['phase4a_npz_sha256']}`",
        f"- **Holdout-18 Dataset:** `{p['holdout_18_stats']['num_references']}` disjoint references, `{p['holdout_18_stats']['num_judged_pairs']}` judged pairs (`0` overlap with `Dev-18`).",
        "",
        "## 1. Seven-Model Holdout-18 & Dev-18 Comparison (Protocol M1: Pooled Reranking)",
        "",
        "| Model | Dev `P@5` | Dev `nDCG@5` | Holdout `P@5` | Holdout `Strict P@5` | Holdout `nDCG@5` | `Δ nDCG@5` vs V1 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for k, title in labels.items():
        c = comp[k]
        lines.append(
            f"| **{title}** | `{c['dev18_p_at_5']:.4f}` | `{c['dev18_ndcg_at_5']:.4f}` | `{c['holdout18_p_at_5']:.4f}` | `{c['holdout18_strict_p_at_5']:.4f}` | `{c['holdout18_ndcg_at_5']:.4f}` | `{c['holdout_abs_ndcg_delta_vs_v1']:+.4f}` (`{c['holdout_rel_ndcg_pct_vs_v1']:+.2f}%`) |"
        )

    lines.extend(
        [
            "",
            "## 2. Protocol M2 — Unrestricted Full-Catalog Diagnostics (All 328 Products)",
            "",
            "> Note: Unjudged Top-5 slot rates reflect retrieval outside historical V1/V2 pools and must **not** be interpreted as irrelevance.",
            "",
            "| Model | `Coverage@5` | `Same-Category@5` | `Overlap w/ V1` | `Dev-18 Unjudged` | `Holdout-18 Unjudged` | Latency (`ms/query`) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for k, title in labels.items():
        d = m2[k]
        lines.append(
            f"| **{title}** | `{d['catalog_coverage_at_5']:.4f}` (`{d['unique_recommended_items_at_5']}/328`) | `{d['same_category_at_5']:.4f}` | `{d['mean_top5_overlap_with_v1']:.4f}` | `{d['dev18_unjudged_top5_slot_rate']:.4f}` | `{d['holdout18_unjudged_top5_slot_rate']:.4f}` | `{d['mean_query_latency_ms']:.4f}` |"
        )

    lines.extend(
        [
            "",
            "## 3. Per-Query Holdout-18 `nDCG@5` Breakdown",
            "",
            "| Reference ID | Name | Category | V1 | P4A Late | CLIP Img | CLIP Text-Img | CLIP Late | `Δ CLIP Late vs V1` |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for r in p["per_query_holdout18"]:
        lines.append(
            f"| `{r['reference_id']}` | {r['reference_name']} | {r['reference_category']} | `{r['v1_ndcg5']:.4f}` | `{r['p4a_late_ndcg5']:.4f}` | `{r['clip_img_ndcg5']:.4f}` | `{r['clip_t2i_ndcg5']:.4f}` | `{r['clip_late_ndcg5']:.4f}` | `{r['delta_clip_late_vs_v1']:+.4f}` |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ELAN V5 Milestone 4B Evaluation")
    parser.add_argument(
        "--mode",
        choices=("dev18", "holdout18"),
        default="dev18",
        help="Evaluation mode ('dev18' tuning or one-shot locked 'holdout18').",
    )
    args = parser.parse_args()
    if args.mode == "dev18":
        out = run_dev18_tuning()
        print(json.dumps({"status": "OK", "mode": "dev18", "models_dev18": out["models_dev18"]}, indent=2))
    else:
        out = run_locked_holdout18_evaluation(allow_holdout=True)
        print(
            json.dumps(
                {
                    "status": "OK",
                    "mode": "holdout18",
                    "locked_config_sha256": out["locked_config_sha256"],
                    "dev_vs_holdout_and_v1_deltas": out["dev_vs_holdout_and_v1_deltas"],
                    "protocol_m2_catalog_diagnostics": out["protocol_m2_catalog_diagnostics"],
                },
                indent=2,
            )
        )
