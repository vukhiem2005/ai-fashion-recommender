
"""Public recommendation API for ELAN."""

import sqlite3

from fastapi import APIRouter, HTTPException, Query

from app.clip_service import recommend_products_with_clip_fallback
from app.personalization_service import (
    get_synthetic_personalized_recommendations,
    list_synthetic_user_profiles,
)


router = APIRouter(
    prefix="/recommendations",
    tags=["AI Recommendations"],
)


@router.get("/personalized/users")
def get_personalized_demo_users():
    """Return curated synthetic shopper profiles for the ItemKNN demonstration."""
    try:
        return list_synthetic_user_profiles()
    except (FileNotFoundError, sqlite3.Error, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Synthetic personalization demo data is unavailable.",
        ) from exc


@router.get("/personalized")
def get_personalized_recommendations(
    user_id: str = Query(default="SYN-U0001", min_length=1),
    top_k: int = Query(default=5, ge=1, le=20),
):
    """Return Top-K ItemKNN recommendations for a deterministic synthetic user profile."""
    try:
        return get_synthetic_personalized_recommendations(
            user_id=user_id,
            top_k=top_k,
        )
    except ValueError as exc:
        message = str(exc)
        if message.startswith("Synthetic user profile not found:"):
            raise HTTPException(status_code=404, detail=message) from exc
        raise HTTPException(status_code=400, detail=message) from exc
    except (FileNotFoundError, sqlite3.Error) as exc:
        raise HTTPException(
            status_code=503,
            detail="Synthetic personalization demo data is unavailable.",
        ) from exc


@router.get("/{product_id}")
def get_recommendations(
    product_id: str,
    top_k: int = Query(default=5, ge=1, le=20),
):
    """Return similar products using V1 TF-IDF or feature-flagged CLIP Late Fusion."""

    try:
        results, model_name = recommend_products_with_clip_fallback(
            product_id=product_id,
            top_k=top_k,
        )

    except ValueError as exc:
        message = str(exc)

        if message.startswith("Active product not found:"):
            raise HTTPException(
                status_code=404,
                detail=message,
            ) from exc

        raise HTTPException(
            status_code=400,
            detail=message,
        ) from exc

    except (FileNotFoundError, sqlite3.Error) as exc:
        raise HTTPException(
            status_code=503,
            detail="Recommendation data is unavailable.",
        ) from exc

    return {
        "reference_product_id": product_id,
        "model": model_name,
        "count": len(results),
        "recommendations": results,
    }
