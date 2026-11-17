"""CLI Entry Point & 10-Check Data-Integrity Validator for Milestone 4B.2C.

Extracts offline CPU CLIP ViT-B/32 embeddings for the active 328-product ELAN catalog,
writes `data/clip_embeddings.npz` and `data/clip_manifest.json`, and executes 10
data-integrity checks on the persisted artifacts.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from backend.evaluation.v5.multimodal_models import (
    DEFAULT_PUBLIC_DIR,
    DEFAULT_RELEASE_DB,
    resolve_catalog_images_readonly,
)
from backend.evaluation.v5.neural_multimodal_models import (
    CLIP_EMBEDDING_DIM,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLIP_MANIFEST,
    DEFAULT_CLIP_NPZ,
    compute_file_sha256,
    load_clip_cache,
    run_extraction_pipeline,
)


def validate_clip_artifacts(
    npz_path: Path = DEFAULT_CLIP_NPZ,
    manifest_path: Path = DEFAULT_CLIP_MANIFEST,
) -> tuple[bool, list[dict[str, Any]]]:
    """Run all 10 required Milestone 4B.2C data-integrity checks."""
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    records, _audit = resolve_catalog_images_readonly(
        db_path=DEFAULT_RELEASE_DB, public_dir=DEFAULT_PUBLIC_DIR
    )
    expected_ids = tuple(r.product_id for r in records)
    n_expected = len(expected_ids)

    # 9. Reopen saved NPZ and JSON
    reopen_ok = True
    try:
        cache = load_clip_cache(npz_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_npz_sha = compute_file_sha256(npz_path)
        sha_match = manifest["artifacts"]["npz_sha256"] == actual_npz_sha
        reopen_ok = sha_match
    except Exception as exc:
        reopen_ok = False
        record("reopen_npz_and_manifest", False, f"Failed to reopen artifacts: {exc}")
        return False, checks

    # 1. Exactly one embedding row per catalog product
    row_count_ok = (
        len(cache.product_ids) == n_expected == 328
        and cache.image_embeddings.shape[0] == n_expected
        and cache.text_embeddings.shape[0] == n_expected
        and cache.image_valid.shape[0] == n_expected
        and cache.text_valid.shape[0] == n_expected
    )
    record(
        "one_row_per_catalog_product",
        row_count_ok,
        f"Catalog={n_expected}, NPZ rows={len(cache.product_ids)}",
    )

    # 2. No duplicate or missing product IDs
    unique_ok = len(set(cache.product_ids)) == n_expected and set(cache.product_ids) == set(
        expected_ids
    )
    record(
        "no_duplicate_or_missing_product_ids",
        unique_ok,
        f"Unique IDs={len(set(cache.product_ids))}/{n_expected}",
    )

    # 3. Image/text arrays have shape (N, 512)
    shape_ok = cache.image_embeddings.shape == (
        n_expected,
        CLIP_EMBEDDING_DIM,
    ) and cache.text_embeddings.shape == (n_expected, CLIP_EMBEDDING_DIM)
    record(
        "embedding_shapes_512",
        shape_ok,
        f"image={cache.image_embeddings.shape}, text={cache.text_embeddings.shape}",
    )

    # 4. Both arrays use float32
    dtype_ok = (
        cache.image_embeddings.dtype == np.float32
        and cache.text_embeddings.dtype == np.float32
    )
    record(
        "embedding_dtype_float32",
        dtype_ok,
        f"image_dtype={cache.image_embeddings.dtype}, text_dtype={cache.text_embeddings.dtype}",
    )

    # 5. All valid embeddings contain finite values
    finite_ok = bool(
        np.all(np.isfinite(cache.image_embeddings[cache.image_valid]))
        and np.all(np.isfinite(cache.text_embeddings[cache.text_valid]))
    )
    record(
        "valid_embeddings_finite",
        finite_ok,
        f"valid_images={int(np.sum(cache.image_valid))}, valid_texts={int(np.sum(cache.text_valid))}, all_finite={finite_ok}",
    )

    # 6. Valid embeddings have L2 norm approximately 1
    img_norms = np.linalg.norm(cache.image_embeddings[cache.image_valid], axis=1)
    txt_norms = np.linalg.norm(cache.text_embeddings[cache.text_valid], axis=1)
    norm_ok = bool(
        np.allclose(img_norms, 1.0, atol=1e-5)
        and np.allclose(txt_norms, 1.0, atol=1e-5)
    )
    record(
        "valid_embeddings_unit_l2_norm",
        norm_ok,
        f"image_norm_range=[{float(img_norms.min()):.6f}, {float(img_norms.max()):.6f}], "
        f"text_norm_range=[{float(txt_norms.min()):.6f}, {float(txt_norms.max()):.6f}]",
    )

    # 7. Missing-image records are correctly masked (HM-0398947001)
    missing_idx = cache.product_ids.index("HM-0398947001")
    missing_norm = float(np.linalg.norm(cache.image_embeddings[missing_idx]))
    missing_ok = (
        int(np.sum(~cache.image_valid)) == 1
        and not bool(cache.image_valid[missing_idx])
        and abs(missing_norm) < 1e-12
        and bool(cache.text_valid[missing_idx])
    )
    record(
        "missing_image_mask_and_zero_placeholder",
        missing_ok,
        f"HM-0398947001 idx={missing_idx}, image_valid={bool(cache.image_valid[missing_idx])}, "
        f"image_norm={missing_norm:.6f}, text_valid={bool(cache.text_valid[missing_idx])}",
    )

    # 8. Product ID and embedding alignment preserved
    align_ok = cache.product_ids == expected_ids
    record(
        "product_id_order_and_alignment",
        align_ok,
        f"Exact match with SQLite `ORDER BY id`={align_ok}",
    )

    # 9. Reopenable NPZ and JSON + SHA-256 match
    record(
        "reopen_npz_and_manifest",
        reopen_ok,
        f"NPZ & JSON reopened cleanly; NPZ SHA-256 verified={reopen_ok}",
    )

    # 10. Source catalog, images, and prior milestones untouched
    m4a_manifest = json.loads(
        (npz_path.parent / "multimodal_manifest.json").read_text(encoding="utf-8")
    )
    m4a_ok = (
        compute_file_sha256(npz_path.parent / "multimodal_embeddings.npz")
        == m4a_manifest["npz_sha256"]
    )
    record(
        "prior_artifacts_preserved",
        m4a_ok,
        f"Phase 4A NPZ SHA-256 unchanged={m4a_ok}",
    )

    all_passed = all(c["passed"] for c in checks)
    return all_passed, checks


def main() -> None:
    summary = run_extraction_pipeline(batch_size=DEFAULT_BATCH_SIZE)
    all_passed, checks = validate_clip_artifacts()
    disk_free_gb = round(
        float(shutil.disk_usage(DEFAULT_CLIP_NPZ.parent).free) / (1024.0**3), 3
    )

    print("ELAN V5 MILESTONE 4B.2C — OFFLINE CLIP EXTRACTION & VALIDATION")
    print("=" * 72)
    print(json.dumps(summary, indent=2))
    print(f"Remaining disk free: {disk_free_gb} GB")
    print("-" * 72)
    for idx, c in enumerate(checks, start=1):
        status = "PASS" if c["passed"] else "FAIL"
        print(f"{idx:2d}. [{status}] {c['check']}: {c['detail']}")
    print("=" * 72)
    if not all_passed:
        sys.exit(1)
    print(f"ALL MILESTONE 4B.2C CHECKS PASSED ({len(checks)}/{len(checks)}).")


if __name__ == "__main__":
    main()
