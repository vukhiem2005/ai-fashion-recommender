
"""Public recommendation API for ELAN."""

import sqlite3

from fastapi import APIRouter, HTTPException, Query

from app.clip_service import recommend_products_with_clip_fallback


router = APIRouter(
    prefix="/recommendations",
    tags=["AI Recommendations"],
)


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
