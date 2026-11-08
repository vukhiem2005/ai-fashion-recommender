from sqlmodel import SQLModel, Field


class Product(SQLModel, table=True):
    id: str = Field(primary_key=True)
    slug: str = Field(index=True)
    name: str
    category: str
    description: str
    price_cents: int
    is_active: bool = Field(default=True)


class ProductVariant(SQLModel, table=True):
    sku: str = Field(primary_key=True)
    product_id: str = Field(
        foreign_key="product.id",
        index=True,
    )
    color: str
    size: str
    image_url: str | None = None
