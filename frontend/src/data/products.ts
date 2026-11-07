export type ProductCategory =
  | "Outerwear"
  | "Tops"
  | "Bottoms"
  | "Dresses"
  | "Knitwear"
  | "Accessories";

export interface Product {
  id: string;
  slug: string;
  name: string;
  category: ProductCategory;
  description: string;
  price: number;
  colors: string[];
  sizes: string[];
  imageUrl: string | null;
}

export const products: Product[] = [
  {
    id: "ELAN-001",
    slug: "classic-beige-blazer",
    name: "Classic Beige Blazer",
    category: "Outerwear",
    description:
      "A timeless tailored blazer with a modern silhouette.",
    price: 649000,
    colors: ["Beige", "Black"],
    sizes: ["S", "M", "L"],
    imageUrl: null,
  },
  {
    id: "ELAN-002",
    slug: "essential-white-shirt",
    name: "Essential White Shirt",
    category: "Tops",
    description:
      "A versatile white shirt designed for everyday styling.",
    price: 399000,
    colors: ["White", "Light Blue"],
    sizes: ["S", "M", "L", "XL"],
    imageUrl: null,
  },
  {
    id: "ELAN-003",
    slug: "straight-leg-denim",
    name: "Straight Leg Denim",
    category: "Bottoms",
    description:
      "Classic straight-leg jeans for a relaxed modern look.",
    price: 599000,
    colors: ["Blue", "Dark Blue"],
    sizes: ["S", "M", "L", "XL"],
    imageUrl: null,
  },
  {
    id: "ELAN-004",
    slug: "minimal-midi-dress",
    name: "Minimal Midi Dress",
    category: "Dresses",
    description:
      "An elegant midi dress with a clean minimalist design.",
    price: 749000,
    colors: ["Black", "Cream"],
    sizes: ["S", "M", "L"],
    imageUrl: null,
  },
  {
    id: "ELAN-005",
    slug: "soft-knit-cardigan",
    name: "Soft Knit Cardigan",
    category: "Knitwear",
    description:
      "A comfortable knit cardigan for effortless layering.",
    price: 549000,
    colors: ["Cream", "Brown"],
    sizes: ["S", "M", "L"],
    imageUrl: null,
  },
  {
    id: "ELAN-006",
    slug: "everyday-crossbody-bag",
    name: "Everyday Crossbody Bag",
    category: "Accessories",
    description:
      "A compact crossbody bag with a refined everyday style.",
    price: 459000,
    colors: ["Black", "Tan"],
    sizes: ["One Size"],
    imageUrl: null,
  },
];

export function getProductBySlug(
  slug: string
): Product | undefined {
  return products.find((product) => product.slug === slug);
}

export function formatPrice(price: number): string {
  return new Intl.NumberFormat("vi-VN", {
    style: "currency",
    currency: "VND",
    maximumFractionDigits: 0,
  }).format(price);
}
