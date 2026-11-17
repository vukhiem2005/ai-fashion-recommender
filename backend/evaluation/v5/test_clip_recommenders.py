"""Focused Automated Validation Suite for Milestone 4B.3A (Offline CLIP Recommenders).

Covers all 10 required verification areas using the cached `clip_embeddings.npz`
and controlled synthetic test embeddings, without running Dev-18 tuning, Holdout-18
evaluation, or Milestone 4B.2C extraction.
"""

from __future__ import annotations

import hashlib
import sys
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


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return (mat / np.maximum(norms, 1e-12)).astype(np.float32)


def run_milestone4b3a_tests() -> tuple[bool, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    # 1. Cache loading & zero PyTorch / Transformers / SQLite / PIL dependencies
    data = load_offline_clip_data()
    no_heavy_imports = (
        "torch" not in sys.modules
        and "transformers" not in sys.modules
        and "PIL" not in sys.modules
        and "sqlite3" not in sys.modules
    )
    cache_ok = (
        len(data.product_ids) == 328
        and data.image_embeddings.shape == (328, 512)
        and data.text_embeddings.shape == (328, 512)
        and int(np.sum(data.image_valid)) == 327
        and int(np.sum(data.text_valid)) == 328
        and data.v1_text_sim_matrix is not None
        and no_heavy_imports
    )
    record(
        "cache_loading_and_runtime_isolation",
        cache_ok,
        f"Loaded 328 products (327 valid images, 328 valid texts); torch/transformers/PIL/sqlite3 not imported={no_heavy_imports}",
    )

    # 2. Product ID alignment
    align_ok = (
        len(set(data.product_ids)) == 328
        and data.product_to_idx["HM-0398947001"] == 27
        and not bool(data.image_valid[27])
        and bool(data.text_valid[27])
    )
    record(
        "product_id_alignment",
        align_ok,
        f"328 unique IDs; HM-0398947001 aligned at idx 27 with image_valid={bool(data.image_valid[27])}",
    )

    # 3. Cosine similarity correctness (synthetic orthonormal + known angle vectors)
    syn_pids = ("P0", "P1", "P2", "P3_MISSING")
    syn_img = _normalize_rows(
        np.array(
            [
                [1.0, 0.0, 0.0] + [0.0] * 509,
                [0.8, 0.6, 0.0] + [0.0] * 509,  # cos(P0, P1) = 0.8
                [0.0, 1.0, 0.0] + [0.0] * 509,  # cos(P0, P2) = 0.0
                [0.0] * 512,                    # missing image placeholder
            ],
            dtype=np.float32,
        )
    )
    syn_img[3, :] = 0.0
    syn_txt = _normalize_rows(
        np.array(
            [
                [0.0, 1.0, 0.0] + [0.0] * 509,  # t2i prefers P2 (1.0) > P1 (0.6) > P0 (0.0)
                [1.0, 0.0, 0.0] + [0.0] * 509,
                [0.0, 0.0, 1.0] + [0.0] * 509,
                [0.6, 0.8, 0.0] + [0.0] * 509,
            ],
            dtype=np.float32,
        )
    )
    syn_v1 = np.array(
        [
            [0.0, 0.05, 0.25, 0.90],
            [0.05, 0.0, 0.15, 0.20],
            [0.25, 0.15, 0.0, 0.30],
            [0.90, 0.20, 0.30, 0.0],
        ],
        dtype=np.float64,
    )
    syn_data = OfflineCLIPData(
        product_ids=syn_pids,
        product_to_idx={p: i for i, p in enumerate(syn_pids)},
        image_embeddings=syn_img,
        text_embeddings=syn_txt,
        image_valid=np.array([True, True, True, False], dtype=bool),
        text_valid=np.array([True, True, True, True], dtype=bool),
        categories=("Tops", "Tops", "Tops", "Tops"),
        v1_text_sim_matrix=syn_v1,
    )

    im_syn = CLIPImageOnlyRecommender().fit(syn_data)
    im_recs = im_syn.recommend_similar_items("P0", top_k=3)
    cos_ok = (
        im_recs[0][0] == "P1"
        and abs(im_recs[0][1] - 0.8) < 1e-6
        and im_recs[1][0] == "P2"
        and abs(im_recs[1][1] - 0.0) < 1e-6
        and im_recs[2][0] == "P3_MISSING"
        and abs(im_recs[2][1] - 0.90) < 1e-6  # fallback V1 score, not artificial 0.0 CLIP score
    )
    record(
        "cosine_similarity_and_image_only_ranking",
        cos_ok,
        f"Synthetic ImageOnly P0 ranking={[(p, round(s, 4)) for p, s in im_recs]}",
    )

    # 4. Text-Image ranking & missing candidate image handling
    t2i_syn = CLIPTextImageRecommender(w_t2i=1.0, w_i2t=0.0, w_t2t=0.0).fit(syn_data)
    t2i_recs = t2i_syn.recommend_similar_items("P0", top_k=3)
    t2i_ok = (
        t2i_recs[0][0] == "P2"
        and abs(t2i_recs[0][1] - 1.0) < 1e-6
        and t2i_recs[1][0] == "P1"
        and abs(t2i_recs[1][1] - 0.6) < 1e-6
        and t2i_recs[2][0] == "P3_MISSING"
    )
    record(
        "text_image_ranking_and_invalid_mask",
        t2i_ok,
        f"Synthetic TextImage P0 ranking={[(p, round(s, 4)) for p, s in t2i_recs]}",
    )

    # 5. Late-fusion non-attenuating scoring
    tau = 0.10
    w_vis = 0.20
    lf_syn = CLIPLateFusionRecommender(
        w_vis=w_vis,
        text_gate_threshold=tau,
        require_same_category_for_visual=True,
        text_source="v1_tfidf",
    ).fit(syn_data)
    # P0->P1: S_text=0.05 < tau=0.10 -> unattenuated 0.05
    # P0->P2: S_text=0.25 >= tau=0.10, S_vis=0.0 -> mapped_vis=0.10 -> 0.8*0.25 + 0.2*0.10 = 0.22
    # P0->P3_MISSING: image_valid=False -> unattenuated S_text=0.90
    lf_ok = (
        abs(float(lf_syn.sim_matrix[0, 1]) - 0.05) < 1e-12
        and abs(float(lf_syn.sim_matrix[0, 2]) - 0.22) < 1e-12
        and abs(float(lf_syn.sim_matrix[0, 3]) - 0.90) < 1e-12
    )
    record(
        "late_fusion_non_attenuating_scoring",
        lf_ok,
        f"P0->P1={lf_syn.sim_matrix[0,1]:.4f} (ungated), P0->P2={lf_syn.sim_matrix[0,2]:.4f} (gated), P0->P3={lf_syn.sim_matrix[0,3]:.4f} (missing img)",
    )

    # 6. Real-catalog missing-image fallback for HM-0398947001
    im_real = CLIPImageOnlyRecommender().fit(data)
    lf_real = CLIPLateFusionRecommender().fit(data)
    t2i_real = CLIPTextImageRecommender().fit(data)

    assert data.v1_text_sim_matrix is not None
    v1_row = data.v1_text_sim_matrix[27]
    expected_v1_top5 = [
        (data.product_ids[idx], float(v1_row[idx]))
        for idx in sorted(
            (i for i in range(328) if i != 27),
            key=lambda i: (-float(v1_row[i]), data.product_ids[i]),
        )[:5]
    ]
    im_miss_meta = im_real.recommend_similar_items_with_meta("HM-0398947001", top_k=5)
    lf_miss_meta = lf_real.recommend_similar_items_with_meta("HM-0398947001", top_k=5)
    fallback_ok = (
        im_miss_meta["fallback_used"] is True
        and im_miss_meta["fallback_mode"] == "exact_v1_tfidf"
        and im_miss_meta["recommendations"] == expected_v1_top5
        and lf_miss_meta["fallback_used"] is True
        and lf_miss_meta["recommendations"] == expected_v1_top5
    )
    record(
        "missing_image_exact_v1_fallback",
        fallback_ok,
        f"HM-0398947001 exact V1 fallback match: ImageOnly={im_miss_meta['recommendations'] == expected_v1_top5}, LateFusion={lf_miss_meta['recommendations'] == expected_v1_top5}",
    )

    # 7. Query-product exclusion across all 328 products & models
    excl_ok = True
    for pid in data.product_ids:
        for model in (im_real, t2i_real, lf_real):
            rec_ids = [cid for cid, _ in model.recommend_similar_items(pid, top_k=5)]
            if pid in rec_ids or len(rec_ids) != 5 or len(set(rec_ids)) != 5:
                excl_ok = False
                break
    record(
        "query_product_exclusion_and_topk",
        excl_ok,
        "Verified self-exclusion and 5 distinct recommendations across all 328 products x 3 models",
    )

    # 8. Deterministic Top-K ranking & ascending ID tie-breaking
    tie_img = np.zeros((4, 512), dtype=np.float32)
    tie_img[:, 0] = 1.0
    tie_txt = tie_img.copy()
    tie_data = OfflineCLIPData(
        product_ids=("TARGET", "ITEM-C", "ITEM-A", "ITEM-B"),
        product_to_idx={"TARGET": 0, "ITEM-C": 1, "ITEM-A": 2, "ITEM-B": 3},
        image_embeddings=tie_img,
        text_embeddings=tie_txt,
        image_valid=np.ones(4, dtype=bool),
        text_valid=np.ones(4, dtype=bool),
    )
    tie_recs = CLIPImageOnlyRecommender().fit(tie_data).recommend_similar_items("TARGET", top_k=3)
    det_ok = [p for p, _ in tie_recs] == ["ITEM-A", "ITEM-B", "ITEM-C"] and (
        im_real.recommend_similar_items("ELAN-001", top_k=5)
        == im_real.recommend_similar_items("ELAN-001", top_k=5)
    )
    record(
        "deterministic_ranking_and_tie_breaking",
        det_ok,
        f"Tie-break order={[p for p, _ in tie_recs]}; repeated inference identical=True",
    )

    # 9. Candidate missing-image placement on real catalog (HM-0292453013 -> HM-0398947001)
    im_all = im_real.recommend_similar_items("HM-0292453013", top_k=327)
    t2i_all = t2i_real.recommend_similar_items("HM-0292453013", top_k=327)
    cand_missing_ok = im_all[-1][0] == "HM-0398947001" and t2i_all[-1][0] == "HM-0398947001"
    record(
        "candidate_missing_image_placement",
        cand_missing_ok,
        f"HM-0398947001 placed last in ImageOnly ({im_all[-1][0]}) and pure TextImage ({t2i_all[-1][0]})",
    )

    # 10. Existing CLIP cache SHA-256 unchanged (not regenerated)
    clip_npz_sha = hashlib.sha256(
        (Path(__file__).resolve().parent / "data" / "clip_embeddings.npz").read_bytes()
    ).hexdigest()
    expected_clip_sha = "4265a3a69611ef57c3f3350cdcdd971e31e10d56db4005b07a328ec410af1e7f"
    record(
        "existing_clip_cache_preserved",
        clip_npz_sha == expected_clip_sha,
        f"clip_embeddings.npz SHA-256 unchanged={clip_npz_sha == expected_clip_sha}",
    )

    all_passed = all(c["passed"] for c in checks)
    return all_passed, checks


def main() -> None:
    all_passed, checks = run_milestone4b3a_tests()
    print("ELAN V5 MILESTONE 4B.3A — OFFLINE CLIP RECOMMENDER TESTS")
    print("=" * 72)
    for idx, c in enumerate(checks, start=1):
        status = "PASS" if c["passed"] else "FAIL"
        print(f"{idx:2d}. [{status}] {c['check']}: {c['detail']}")
    print("=" * 72)
    if not all_passed:
        sys.exit(1)
    print(f"ALL MILESTONE 4B.3A TESTS PASSED ({len(checks)}/{len(checks)}).")


if __name__ == "__main__":
    main()
