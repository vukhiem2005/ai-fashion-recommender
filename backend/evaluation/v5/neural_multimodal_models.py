"""Pretrained CLIP ViT-B/32 Offline Embedding Extractor & Cache for ELAN V5 (Milestone 4B.2C).

Strictly offline, CPU-only extraction using the verified local snapshot of
`openai/clip-vit-base-patch32` (`d15b5f29721ca72dac15f8526b284be910de18be`,
`model.safetensors` SHA-256 verified before loading).

Reuses Phase 4A's read-only catalog-to-image resolver (`resolve_catalog_images_readonly`)
and preserves exact catalog ordering (`ORDER BY id`) with explicit boolean validity
masks (`image_valid`, `text_valid`) and zero-vector placeholders for missing assets
(`HM-0398947001`).
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import PIL
import sklearn
import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from backend.evaluation.v5.multimodal_models import (
    DEFAULT_DATA_DIR,
    DEFAULT_PUBLIC_DIR,
    DEFAULT_RELEASE_DB,
    ProductImageRecord,
    resolve_catalog_images_readonly,
)

CLIP_REPO_ID = "openai/clip-vit-base-patch32"
CLIP_REVISION = "d15b5f29721ca72dac15f8526b284be910de18be"
CLIP_WEIGHT_SHA256 = "99d28a652e6ec46629ab7047a0ac82c69b1fe11e0ce672c43af65d3a9a3fc05d"
DEFAULT_CLIP_SNAPSHOT_DIR = Path(
    r"C:\Users\vuvan\Projects\_hf_cache\elan-v5-clip\models--openai--clip-vit-base-patch32\snapshots\d15b5f29721ca72dac15f8526b284be910de18be"
)
DEFAULT_CLIP_NPZ = DEFAULT_DATA_DIR / "clip_embeddings.npz"
DEFAULT_CLIP_MANIFEST = DEFAULT_DATA_DIR / "clip_manifest.json"

CLIP_EMBEDDING_DIM = 512
CLIP_TEXT_MAX_LENGTH = 77
DEFAULT_BATCH_SIZE = 8
TEXT_TEMPLATE_DESCRIPTION = (
    '"{category}: {name}. Colors: {colors_csv}. {description}" '
    "(omitting empty segments; whitespace-normalized; BPE-truncated to 77 tokens)"
)


@dataclass(frozen=True)
class CLIPCatalogCache:
    """Aligned CLIP ViT-B/32 image and text embeddings for all active catalog products."""

    product_ids: tuple[str, ...]
    image_embeddings: np.ndarray  # Shape (N, 512), float32, L2-normalized if valid else 0
    text_embeddings: np.ndarray   # Shape (N, 512), float32, L2-normalized if valid else 0
    image_valid: np.ndarray       # Shape (N,), bool
    text_valid: np.ndarray        # Shape (N,), bool


def get_process_rss_mb() -> tuple[float | None, float | None]:
    """Return `(current_rss_mb, peak_rss_mb)` on Windows via `psapi.GetProcessMemoryInfo`."""
    if os.name != "nt":
        return None, None
    try:

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        if not ok:
            return None, None
        curr_mb = round(float(counters.WorkingSetSize) / (1024.0 * 1024.0), 2)
        peak_mb = round(float(counters.PeakWorkingSetSize) / (1024.0 * 1024.0), 2)
        return curr_mb, peak_mb
    except Exception:
        return None, None


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file using 8 MiB streaming chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_clip_product_text(record: ProductImageRecord) -> str:
    """Construct concise English fashion text from existing catalog metadata only.

    Places `category`, `name`, and `colors` ahead of `description` so key attributes
    remain within CLIP's 77-token BPE context window.
    """
    category = " ".join(str(record.category or "").split())
    name = " ".join(str(record.name or "").split())
    colors = ", ".join(" ".join(str(c).split()) for c in record.colors if str(c).strip())
    description = " ".join(str(record.description or "").split())

    parts: list[str] = []
    if category and name:
        parts.append(f"{category}: {name}.")
    elif name:
        parts.append(f"{name}.")
    elif category:
        parts.append(f"{category}.")

    if colors:
        parts.append(f"Colors: {colors}.")
    if description:
        parts.append(description)
    return " ".join(parts).strip()


def _extract_tensor_output(output: Any) -> torch.Tensor:
    """Safely extract a `torch.Tensor` from Transformers feature-extraction outputs."""
    if isinstance(output, torch.Tensor):
        return output
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        val = getattr(output, attr, None)
        if isinstance(val, torch.Tensor):
            return val
    raise TypeError(f"Unsupported CLIP feature output type: {type(output)!r}")


def load_offline_clip(
    snapshot_dir: Path = DEFAULT_CLIP_SNAPSHOT_DIR,
) -> tuple[CLIPModel, CLIPProcessor]:
    """Load verified local CLIP snapshot strictly offline on CPU."""
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(f"CLIP snapshot directory not found: {snapshot_dir}")

    weights_path = snapshot_dir / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"SafeTensors weights not found: {weights_path}")

    actual_sha = compute_file_sha256(weights_path)
    if actual_sha != CLIP_WEIGHT_SHA256:
        raise ValueError(
            f"SHA-256 mismatch for {weights_path}: got {actual_sha}, expected {CLIP_WEIGHT_SHA256}"
        )

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    model = CLIPModel.from_pretrained(
        str(snapshot_dir),
        local_files_only=True,
        use_safetensors=True,
    )
    model.to("cpu")
    model.eval()

    processor = CLIPProcessor.from_pretrained(
        str(snapshot_dir),
        local_files_only=True,
        use_fast=False,
    )
    return model, processor


def extract_catalog_clip_embeddings(
    records: Sequence[ProductImageRecord],
    snapshot_dir: Path = DEFAULT_CLIP_SNAPSHOT_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[CLIPCatalogCache, dict[str, Any]]:
    """Extract L2-normalized float32 CLIP image & text embeddings for all catalog products."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    t_total_0 = time.perf_counter()
    model, processor = load_offline_clip(snapshot_dir=snapshot_dir)

    n_items = len(records)
    product_ids = tuple(r.product_id for r in records)
    image_embeddings = np.zeros((n_items, CLIP_EMBEDDING_DIM), dtype=np.float32)
    text_embeddings = np.zeros((n_items, CLIP_EMBEDDING_DIM), dtype=np.float32)
    image_valid = np.zeros(n_items, dtype=bool)
    text_valid = np.zeros(n_items, dtype=bool)
    failed_or_skipped: list[dict[str, str]] = []

    # 1. Image embedding extraction (CPU batched, only for products with valid local image)
    imaged_indices = [
        idx
        for idx, rec in enumerate(records)
        if rec.has_valid_image and rec.primary_image_path is not None
    ]
    for idx, rec in enumerate(records):
        if not (rec.has_valid_image and rec.primary_image_path is not None):
            failed_or_skipped.append(
                {
                    "product_id": rec.product_id,
                    "modality": "image",
                    "status": "skipped_missing_image",
                    "reason": "Product has no valid local image asset (image_url is NULL or missing); zero placeholder + image_valid=False.",
                }
            )

    t_img_0 = time.perf_counter()
    for start in range(0, len(imaged_indices), batch_size):
        batch_indices = imaged_indices[start : start + batch_size]
        pil_images: list[Image.Image] = []
        loaded_indices: list[int] = []

        for idx in batch_indices:
            img_path = records[idx].primary_image_path
            assert img_path is not None
            try:
                with Image.open(img_path) as im:
                    pil_images.append(im.convert("RGB"))
                loaded_indices.append(idx)
            except Exception as exc:
                failed_or_skipped.append(
                    {
                        "product_id": records[idx].product_id,
                        "modality": "image",
                        "status": "failed_decode",
                        "reason": str(exc),
                    }
                )

        if not pil_images:
            continue

        inputs = processor(images=pil_images, return_tensors="pt")
        for im in pil_images:
            im.close()
        del pil_images

        with torch.inference_mode():
            raw_feats = _extract_tensor_output(
                model.get_image_features(pixel_values=inputs["pixel_values"])
            )
            norm_feats = F.normalize(raw_feats.to(torch.float32), p=2, dim=-1)
            batch_np = norm_feats.cpu().numpy().astype(np.float32, copy=False)

        for row_i, cat_idx in enumerate(loaded_indices):
            image_embeddings[cat_idx, :] = batch_np[row_i]
            image_valid[cat_idx] = True

        del inputs, raw_feats, norm_feats, batch_np

    image_duration_s = time.perf_counter() - t_img_0

    # 2. Text embedding extraction (CPU batched)
    text_items: list[tuple[int, str]] = []
    for idx, rec in enumerate(records):
        prompt = build_clip_product_text(rec)
        if prompt:
            text_items.append((idx, prompt))
        else:
            failed_or_skipped.append(
                {
                    "product_id": rec.product_id,
                    "modality": "text",
                    "status": "skipped_empty_text",
                    "reason": "Constructed product text was empty; zero placeholder + text_valid=False.",
                }
            )

    t_txt_0 = time.perf_counter()
    for start in range(0, len(text_items), batch_size):
        batch_pairs = text_items[start : start + batch_size]
        batch_indices = [idx for idx, _ in batch_pairs]
        batch_texts = [txt for _, txt in batch_pairs]

        inputs = processor(
            text=batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CLIP_TEXT_MAX_LENGTH,
        )
        with torch.inference_mode():
            raw_feats = _extract_tensor_output(
                model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
            )
            norm_feats = F.normalize(raw_feats.to(torch.float32), p=2, dim=-1)
            batch_np = norm_feats.cpu().numpy().astype(np.float32, copy=False)

        for row_i, cat_idx in enumerate(batch_indices):
            text_embeddings[cat_idx, :] = batch_np[row_i]
            text_valid[cat_idx] = True

        del inputs, raw_feats, norm_feats, batch_np

    text_duration_s = time.perf_counter() - t_txt_0
    total_duration_s = time.perf_counter() - t_total_0
    curr_rss_mb, peak_rss_mb = get_process_rss_mb()

    del model, processor

    cache = CLIPCatalogCache(
        product_ids=product_ids,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        image_valid=image_valid,
        text_valid=text_valid,
    )
    stats: dict[str, Any] = {
        "batch_size": int(batch_size),
        "total_products": n_items,
        "valid_image_embeddings": int(np.sum(image_valid)),
        "missing_or_failed_image_embeddings": int(n_items - np.sum(image_valid)),
        "valid_text_embeddings": int(np.sum(text_valid)),
        "missing_or_failed_text_embeddings": int(n_items - np.sum(text_valid)),
        "image_extraction_duration_s": round(image_duration_s, 3),
        "text_extraction_duration_s": round(text_duration_s, 3),
        "total_extraction_duration_s": round(total_duration_s, 3),
        "current_process_rss_mb": curr_rss_mb,
        "peak_process_rss_mb": peak_rss_mb,
        "failed_or_skipped_records": failed_or_skipped,
    }
    return cache, stats


def save_clip_cache(
    cache: CLIPCatalogCache,
    stats: dict[str, Any],
    snapshot_dir: Path = DEFAULT_CLIP_SNAPSHOT_DIR,
    npz_path: Path = DEFAULT_CLIP_NPZ,
    manifest_path: Path = DEFAULT_CLIP_MANIFEST,
    db_path: Path = DEFAULT_RELEASE_DB,
) -> dict[str, str]:
    """Persist aligned CLIP embeddings `.npz` and `clip_manifest.json`."""
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        npz_path,
        product_ids=np.array(cache.product_ids, dtype="<U32"),
        image_embeddings=cache.image_embeddings.astype(np.float32, copy=False),
        text_embeddings=cache.text_embeddings.astype(np.float32, copy=False),
        image_valid=cache.image_valid.astype(bool, copy=False),
        text_valid=cache.text_valid.astype(bool, copy=False),
    )
    npz_sha256 = compute_file_sha256(npz_path)

    disk_free_bytes = shutil.disk_usage(npz_path.parent).free
    manifest = {
        "milestone": "Milestone 4B.2C — Offline CLIP Catalog Embedding Cache",
        "model_repository": CLIP_REPO_ID,
        "model_revision_sha": CLIP_REVISION,
        "model_weight_file": "model.safetensors",
        "model_weight_sha256": CLIP_WEIGHT_SHA256,
        "external_snapshot_path": str(snapshot_dir),
        "source_catalog": {
            "database_path": str(db_path.relative_to(Path(__file__).resolve().parents[3])),
            "query_filter": "is_active = 1 ORDER BY id",
            "total_active_products": len(cache.product_ids),
        },
        "library_versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "numpy": str(np.__version__),
            "pillow": str(PIL.__version__),
            "scikit_learn": str(sklearn.__version__),
        },
        "embedding_specification": {
            "embedding_dim": CLIP_EMBEDDING_DIM,
            "dtype": "float32",
            "normalization": "L2 unit norm (p=2, dim=-1) on valid rows; zero vector placeholder on invalid rows",
            "text_construction_template": TEXT_TEMPLATE_DESCRIPTION,
            "text_max_bpe_tokens": CLIP_TEXT_MAX_LENGTH,
            "processor_use_fast": False,
        },
        "extraction_statistics": stats,
        "artifacts": {
            "npz_path": str(npz_path.relative_to(Path(__file__).resolve().parents[3])),
            "npz_file_size_bytes": int(npz_path.stat().st_size),
            "npz_sha256": npz_sha256,
            "remaining_disk_free_gb": round(float(disk_free_bytes) / (1024.0**3), 3),
        },
        "reproduction_command": (
            r"C:\Users\vuvan\Projects\_venvs\elan-v5-clip\Scripts\python.exe "
            "-m backend.evaluation.v5.extract_clip_embeddings"
        ),
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest_sha256 = compute_file_sha256(manifest_path)
    return {"npz_sha256": npz_sha256, "manifest_sha256": manifest_sha256}


def load_clip_cache(npz_path: Path = DEFAULT_CLIP_NPZ) -> CLIPCatalogCache:
    """Reopen precomputed `clip_embeddings.npz` strictly offline."""
    if not npz_path.is_file():
        raise FileNotFoundError(f"CLIP embedding cache not found: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        return CLIPCatalogCache(
            product_ids=tuple(str(x) for x in data["product_ids"].tolist()),
            image_embeddings=data["image_embeddings"].astype(np.float32),
            text_embeddings=data["text_embeddings"].astype(np.float32),
            image_valid=data["image_valid"].astype(bool),
            text_valid=data["text_valid"].astype(bool),
        )


def run_extraction_pipeline(
    batch_size: int = DEFAULT_BATCH_SIZE,
    snapshot_dir: Path = DEFAULT_CLIP_SNAPSHOT_DIR,
    db_path: Path = DEFAULT_RELEASE_DB,
    public_dir: Path = DEFAULT_PUBLIC_DIR,
    npz_path: Path = DEFAULT_CLIP_NPZ,
    manifest_path: Path = DEFAULT_CLIP_MANIFEST,
) -> dict[str, Any]:
    """End-to-end offline catalog CLIP extraction and serialization."""
    records, _audit = resolve_catalog_images_readonly(
        db_path=db_path, public_dir=public_dir
    )
    cache, stats = extract_catalog_clip_embeddings(
        records=records, snapshot_dir=snapshot_dir, batch_size=batch_size
    )
    digests = save_clip_cache(
        cache=cache,
        stats=stats,
        snapshot_dir=snapshot_dir,
        npz_path=npz_path,
        manifest_path=manifest_path,
        db_path=db_path,
    )
    return {
        "stats": stats,
        "digests": digests,
        "npz_size_bytes": int(npz_path.stat().st_size),
        "manifest_size_bytes": int(manifest_path.stat().st_size),
    }
