
from sqlmodel import Session

from .database import create_db_and_tables, engine
from .models import Product, ProductVariant


# Initial demo catalog.
# Prices are stored in USD cents.
# Images refer to local frontend assets.

SEED_PRODUCTS = [
    (
        "ELAN-001",
        "classic-beige-blazer",
        "Classic Beige Blazer",
        "Outerwear",
        "A timeless tailored blazer with a modern silhouette.",
        2999,
        "Beige",
        "BGE",
        ["S", "M", "L"],
        "/products/classic-beige-blazer.jpg",
    ),
    (
        "ELAN-002",
        "essential-white-shirt",
        "Essential White Shirt",
        "Tops",
        "A versatile white shirt designed for everyday styling.",
        1999,
        "White",
        "WHT",
        ["S", "M", "L", "XL"],
        "/products/essential-white-shirt.jpg",
    ),
    (
        "ELAN-003",
        "straight-leg-denim",
        "Straight Leg Denim",
        "Bottoms",
        "Classic straight-leg jeans for a relaxed modern look.",
        2799,
        "Blue",
        "BLU",
        ["S", "M", "L", "XL"],
        None,
    ),
    (
        "ELAN-004",
        "minimal-midi-dress",
        "Minimal Midi Dress",
        "Dresses",
        "An elegant midi dress with a clean minimalist design.",
        3499,
        "Black",
        "BLK",
        ["S", "M", "L"],
        None,
    ),
    (
        "ELAN-005",
        "soft-knit-cardigan",
        "Soft Knit Cardigan",
        "Knitwear",
        "A comfortable knit cardigan for effortless layering.",
        2499,
        "Cream",
        "CRM",
        ["S", "M", "L"],
        None,
    ),
    (
        "ELAN-006",
        "everyday-crossbody-bag",
        "Everyday Crossbody Bag",
        "Accessories",
        "A compact crossbody bag with a refined everyday style.",
        2199,
        "Black",
        "BLK",
        ["One Size"],
        None,
    ),
]


def seed_products() -> None:
    create_db_and_tables()

    created_products = 0
    created_variants = 0

    with Session(engine) as session:
        for (
            product_id,
            slug,
            name,
            category,
            description,
            price_cents,
            color,
            color_code,
            sizes,
            image_url,
        ) in SEED_PRODUCTS:

            # Do not overwrite products that already exist.
            if session.get(Product, product_id):
                continue

            product = Product(
                id=product_id,
                slug=slug,
                name=name,
                category=category,
                description=description,
                price_cents=price_cents,
                is_active=True,
            )

            session.add(product)
            created_products += 1

            for size in sizes:
                size_code = size.upper().replace(" ", "-")

                variant = ProductVariant(
                    sku=f"{product_id}-{color_code}-{size_code}",
                    product_id=product_id,
                    color=color,
                    size=size,
                    image_url=image_url,
                )

                session.add(variant)
                created_variants += 1

        session.commit()

    print(
        f"Seed completed: {created_products} products, "
        f"{created_variants} variants added."
    )


if __name__ == "__main__":
    seed_products()
