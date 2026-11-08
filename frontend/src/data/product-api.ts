import type {
  Product,
  ProductCategory,
  ProductVariant,
} from "@/data/products";

export interface BackendProductVariant {
  sku: string;
  color: string;
  size: string;
  image_url: string | null;
}

export interface BackendProduct {
  id: string;
  slug: string;
  name: string;
  category: ProductCategory;
  description: string;
  price_cents: number;
  is_active: boolean;
  variants: BackendProductVariant[];
}

export function toStorefrontProduct(product: BackendProduct): Product {
  const variants: ProductVariant[] = product.variants.map((variant) => ({
    sku: variant.sku,
    color: variant.color,
    size: variant.size,
    imageUrl: variant.image_url,
  }));

  return {
    id: product.id,
    slug: product.slug,
    name: product.name,
    category: product.category,
    description: product.description,
    price: product.price_cents / 100,
    variants,
    colors: [...new Set(variants.map((variant) => variant.color))],
    sizes: [...new Set(variants.map((variant) => variant.size))],
    imageUrl:
      variants.find((variant) => variant.imageUrl !== null)?.imageUrl ?? null,
  };
}
