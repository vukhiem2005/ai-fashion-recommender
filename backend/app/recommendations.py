
"""Public recommendation API for ELAN."""

import sqlite3

from fastapi import APIRouter, HTTPException, Query

from app.recommender import recommend_products


router = APIRouter(
    prefix="/recommendations",
    tags=["AI Recommendations"],
)


@router.get("/{product_id}")
def get_recommendations(
    product_id: str,
    top_k: int = Query(default=5, ge=1, le=20),
):
    """Return similar products using TF-IDF + cosine similarity."""

    try:
        results = recommend_products(
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
        "model": "tfidf_cosine_v1",
        "count": len(results),
        "recommendations": results,
    }
