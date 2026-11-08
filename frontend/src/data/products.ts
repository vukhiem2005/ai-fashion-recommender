
export type ProductCategory =
  | "Outerwear"
  | "Tops"
  | "Bottoms"
  | "Dresses"
  | "Knitwear"
  | "Accessories";

export interface ProductVariant {
  sku: string;
  color: string;
  size: string;
  imageUrl: string | null;
}

export interface Product {
  id: string;
  slug: string;
  name: string;
  category: ProductCategory;
  description: string;
  price: number;

  // Product variants
  variants: ProductVariant[];

  // Derived fields for existing UI compatibility
  colors: string[];
  sizes: string[];
  imageUrl: string | null;
}

interface ColorwayDefinition {
  color: string;
  code: string;
  imageUrl: string | null;
  sizes: string[];
}

type ProductDefinition = Omit<
  Product,
  "variants" | "colors" | "sizes" | "imageUrl"
> & {
  colorways: ColorwayDefinition[];
};

// Central product definitions
const productDefinitions: ProductDefinition[] = [
  {
    id: "ELAN-001",
    slug: "classic-beige-blazer",
    name: "Classic Beige Blazer",
    category: "Outerwear",
    description:
      "A timeless tailored blazer with a modern silhouette.",
    price: 29.99,
    colorways: [
      {
        color: "Beige",
        code: "BGE",
        imageUrl: "/products/classic-beige-blazer.jpg",
        sizes: ["S", "M", "L"],
      },
    ],
  },
  {
    id: "ELAN-002",
    slug: "essential-white-shirt",
    name: "Essential White Shirt",
    category: "Tops",
    description:
      "A versatile white shirt designed for everyday styling.",
    price: 19.99,
    colorways: [
      {
        color: "White",
        code: "WHT",
        imageUrl: "/products/essential-white-shirt.jpg",
        sizes: ["S", "M", "L", "XL"],
      },
    ],
  },
  {
    id: "ELAN-003",
    slug: "straight-leg-denim",
    name: "Straight Leg Denim",
    category: "Bottoms",
    description:
      "Classic straight-leg jeans for a relaxed modern look.",
    price: 27.99,
    colorways: [
      {
        color: "Blue",
        code: "BLU",
        imageUrl: null,
        sizes: ["S", "M", "L", "XL"],
      },
    ],
  },
  {
    id: "ELAN-004",
    slug: "minimal-midi-dress",
    name: "Minimal Midi Dress",
    category: "Dresses",
    description:
      "An elegant midi dress with a clean minimalist design.",
    price: 34.99,
    colorways: [
      {
        color: "Black",
        code: "BLK",
        imageUrl: null,
        sizes: ["S", "M", "L"],
      },
    ],
  },
  {
    id: "ELAN-005",
    slug: "soft-knit-cardigan",
    name: "Soft Knit Cardigan",
    category: "Knitwear",
    description:
      "A comfortable knit cardigan for effortless layering.",
    price: 24.99,
    colorways: [
      {
        color: "Cream",
        code: "CRM",
        imageUrl: null,
        sizes: ["S", "M", "L"],
      },
    ],
  },
  {
    id: "ELAN-006",
    slug: "everyday-crossbody-bag",
    name: "Everyday Crossbody Bag",
    category: "Accessories",
    description:
      "A compact crossbody bag with a refined everyday style.",
    price: 21.99,
    colorways: [
      {
        color: "Black",
        code: "BLK",
        imageUrl: null,
        sizes: ["One Size"],
      },
    ],
  },
];

// Generate size-specific product variants
export const products: Product[] = productDefinitions.map(
  (definition) => {
    const { colorways, ...baseProduct } = definition;

    const variants: ProductVariant[] = colorways.flatMap(
      (colorway) =>
        colorway.sizes.map((size) => ({
          sku: `${definition.id}-${colorway.code}-${size
            .toUpperCase()
            .replace(/\s+/g, "-")}`,
          color: colorway.color,
          size,
          imageUrl: colorway.imageUrl,
        }))
    );

    return {
      ...baseProduct,
      variants,
      colors: colorways.map((colorway) => colorway.color),
      sizes: [...new Set(variants.map((variant) => variant.size))],
      imageUrl: colorways[0]?.imageUrl ?? null,
    };
  }
);

// Product lookup by URL slug
export function getProductBySlug(
  slug: string
): Product | undefined {
  return products.find((product) => product.slug === slug);
}

// Product variant lookup
export function getVariantBySku(
  sku: string
): ProductVariant | undefined {
  for (const product of products) {
    const variant = product.variants.find(
      (item) => item.sku === sku
    );

    if (variant) {
      return variant;
    }
  }

  return undefined;
}

// USD price formatting
export function formatPrice(price: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(price);
}
