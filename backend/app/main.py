
from contextlib import asynccontextmanager
from app.recommendations import router as recommendations_router

from fastapi import FastAPI, HTTPException
from sqlalchemy import text
from sqlmodel import Session, select

from .admin import router as admin_router
from .database import create_db_and_tables, engine
from .models import Product, ProductVariant


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield


app = FastAPI(
    title="ELAN Fashion Store API",
    description="Backend API for ELAN E-Commerce",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(recommendations_router)
app.include_router(admin_router)


@app.get("/")
def root():
    return {
        "application": "ELAN Fashion Store API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
def health():
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    return {
        "status": "ok",
        "database": "connected",
    }


def serialize_product(session: Session, product: Product):
    variants = session.exec(
        select(ProductVariant).where(
            ProductVariant.product_id == product.id
        )
    ).all()

    return {
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


@app.get("/products")
def list_products():
    with Session(engine) as session:
        products = session.exec(select(Product)).all()

        return [
            serialize_product(session, product)
            for product in products
            if product.is_active
        ]


@app.get("/products/{slug}")
def get_product(slug: str):
    with Session(engine) as session:
        product = session.exec(
            select(Product).where(Product.slug == slug)
        ).first()

        if product is None or not product.is_active:
            raise HTTPException(
                status_code=404,
                detail="Product not found",
            )

        return serialize_product(session, product)
