
import hmac
import os
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from .database import engine
from .models import Product, ProductVariant


ProductCategory = Literal[
    "Outerwear",
    "Tops",
    "Bottoms",
    "Dresses",
    "Knitwear",
    "Accessories",
]


class VariantCreate(BaseModel):
    sku: str = Field(min_length=1, max_length=80)
    color: str = Field(min_length=1)
    size: str = Field(min_length=1)
    image_url: str | None = None


class ProductCreate(BaseModel):
    id: str = Field(min_length=1, max_length=50)
    slug: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    )
    name: str = Field(min_length=1)
    category: ProductCategory
    description: str = Field(min_length=1)
    price_cents: int = Field(gt=0)
    variants: list[VariantCreate] = Field(min_length=1)


def require_admin_key(
    x_admin_key: str | None = Header(default=None),
):
    expected_key = os.getenv("ELAN_ADMIN_KEY")

    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="Admin API is not configured",
        )

    if not x_admin_key or not hmac.compare_digest(
        x_admin_key, expected_key
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid admin credentials",
        )


router = APIRouter(
    prefix="/admin",
    tags=["Admin"],
    dependencies=[Depends(require_admin_key)],
)


@router.post("/products", status_code=201)
def create_product(payload: ProductCreate):
    skus = [variant.sku for variant in payload.variants]

    if len(skus) != len(set(skus)):
        raise HTTPException(
            status_code=422,
            detail="Duplicate variant SKUs",
        )

    for variant in payload.variants:
        if (
            variant.image_url is not None
            and not variant.image_url.startswith("/products/")
        ):
            raise HTTPException(
                status_code=422,
                detail="Invalid product image path",
            )

    with Session(engine) as session:
        if session.get(Product, payload.id):
            raise HTTPException(
                status_code=409,
                detail="Product ID already exists",
            )

        existing_slug = session.exec(
            select(Product).where(Product.slug == payload.slug)
        ).first()

        if existing_slug:
            raise HTTPException(
                status_code=409,
                detail="Product slug already exists",
            )

        for sku in skus:
            if session.get(ProductVariant, sku):
                raise HTTPException(
                    status_code=409,
                    detail=f"Variant SKU already exists: {sku}",
                )

        product = Product(
            id=payload.id,
            slug=payload.slug,
            name=payload.name,
            category=payload.category,
            description=payload.description,
            price_cents=payload.price_cents,
            is_active=True,
        )

        session.add(product)

        for variant in payload.variants:
            session.add(
                ProductVariant(
                    sku=variant.sku,
                    product_id=payload.id,
                    color=variant.color,
                    size=variant.size,
                    image_url=variant.image_url,
                )
            )

        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            raise HTTPException(
                status_code=409,
                detail="Product or variant conflict",
            ) from None

    return {
        "message": "Product created successfully",
        "id": payload.id,
        "variant_count": len(payload.variants),
    }

class ProductUpdate(BaseModel):
    name: str | None = Field(
        default=None,
        min_length=1,
    )
    category: ProductCategory | None = None
    description: str | None = Field(
        default=None,
        min_length=1,
    )
    price_cents: int | None = Field(
        default=None,
        gt=0,
    )


@router.patch("/products/{product_id}")
def update_product(
    product_id: str,
    payload: ProductUpdate,
):
    updates = payload.model_dump(exclude_unset=True)

    if not updates:
        raise HTTPException(
            status_code=422,
            detail="No fields provided for update",
        )

    if any(value is None for value in updates.values()):
        raise HTTPException(
            status_code=422,
            detail="Updated fields cannot be null",
        )

    with Session(engine) as session:
        product = session.get(Product, product_id)

        if product is None:
            raise HTTPException(
                status_code=404,
                detail="Product not found",
            )

        for field, value in updates.items():
            setattr(product, field, value)

        session.add(product)
        session.commit()
        session.refresh(product)

        return {
            "message": "Product updated successfully",
            "id": product.id,
            "name": product.name,
            "category": product.category,
            "description": product.description,
            "price_cents": product.price_cents,
        }

class ProductStatusUpdate(BaseModel):
    is_active: bool


@router.patch("/products/{product_id}/status")
def update_product_status(
    product_id: str,
    payload: ProductStatusUpdate,
):
    with Session(engine) as session:
        product = session.get(Product, product_id)

        if product is None:
            raise HTTPException(
                status_code=404,
                detail="Product not found",
            )

        product.is_active = payload.is_active

        session.add(product)
        session.commit()
        session.refresh(product)

        return {
            "message": "Product status updated successfully",
            "id": product.id,
            "name": product.name,
            "is_active": product.is_active,
        }

@router.get("/products")
def list_admin_products():
    """
    Return all products, including inactive products.
    This endpoint requires an admin key.
    """

    with Session(engine) as session:
        all_products = session.exec(
            select(Product).order_by(Product.id)
        ).all()

        results = []

        for product in all_products:
            variants = session.exec(
                select(ProductVariant)
                .where(
                    ProductVariant.product_id == product.id
                )
                .order_by(ProductVariant.sku)
            ).all()

            results.append(
                {
                    "id": product.id,
                    "slug": product.slug,
                    "name": product.name,
                    "category": product.category,
                    "description": product.description,
                    "price_cents": product.price_cents,
                    "is_active": product.is_active,
                    "variants": [
                        {
                            "sku": variant.sku,
                            "color": variant.color,
                            "size": variant.size,
                            "image_url": variant.image_url,
                        }
                        for variant in variants
                    ],
                }
            )

        return results
