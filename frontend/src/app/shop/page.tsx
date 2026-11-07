
"use client";

import Image from "next/image";
import Link from "next/link";
import { useState } from "react";
import {
  products,
  formatPrice,
  type ProductCategory,
} from "@/data/products";

type CategoryFilter = "All" | ProductCategory;
type SortOption = "featured" | "price-asc" | "price-desc";

const categories: CategoryFilter[] = [
  "All",
  ...Array.from(new Set(products.map((p) => p.category))),
];

export default function ShopPage() {
  const [search, setSearch] = useState("");
  const [category, setCategory] =
    useState<CategoryFilter>("All");
  const [sort, setSort] =
    useState<SortOption>("featured");

  const query = search.trim().toLowerCase();

  const visibleProducts = products
    .filter((product) => {
      const matchesCategory =
        category === "All" ||
        product.category === category;

      const searchableText = [
        product.name,
        product.category,
        product.description,
        ...product.colors,
      ]
        .join(" ")
        .toLowerCase();

      return (
        matchesCategory &&
        searchableText.includes(query)
      );
    })
    .sort((a, b) => {
      if (sort === "price-asc") {
        return a.price - b.price;
      }

      if (sort === "price-desc") {
        return b.price - a.price;
      }

      return a.id.localeCompare(b.id);
    });

  function resetFilters() {
    setSearch("");
    setCategory("All");
    setSort("featured");
  }

  return (
    <div className="min-h-screen bg-[#faf9f6] text-[#24221f]">
      <div className="bg-[#24221f] px-4 py-2 text-center text-xs tracking-wider text-white">
        ÉLAN Fashion Store — Portfolio Demo
      </div>

      <header className="border-b border-[#e9e6e1] bg-white">
        <nav className="mx-auto flex max-w-7xl items-center justify-between px-6 py-6">
          <Link
            href="/"
            className="font-serif text-3xl font-bold tracking-widest"
          >
            ÉLAN.
          </Link>

          <div className="flex items-center gap-6 text-sm">
            <Link href="/" className="hover:text-[#ae805f]">
              Home
            </Link>
            <Link
              href="/shop"
              className="font-semibold text-[#ae805f]"
            >
              Shop
            </Link>
          </div>

          <span className="hidden text-xs text-[#80776e] sm:block">
            Curated Fashion
          </span>
        </nav>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-12 md:py-16">
        <div className="mb-12">
          <p className="mb-3 text-xs font-semibold uppercase tracking-[0.25em] text-[#ae805f]">
            The ÉLAN Collection
          </p>

          <h1 className="font-serif text-5xl md:text-6xl">
            Shop the Collection
          </h1>

          <p className="mt-5 max-w-2xl leading-7 text-[#77716b]">
            Discover contemporary essentials designed
            for effortless everyday styling.
          </p>
        </div>

        <section
          aria-label="Product filters"
          className="mb-10 space-y-6 border-y border-[#e7e2dc] py-6"
        >
          <div className="flex flex-col gap-4 md:flex-row">
            <input
              type="search"
              value={search}
              onChange={(event) =>
                setSearch(event.target.value)
              }
              placeholder="Search products..."
              aria-label="Search products"
              className="w-full border border-[#ddd6cf] bg-white px-5 py-3 text-sm outline-none focus:border-[#ae805f] md:max-w-md"
            />

            <select
              value={sort}
              onChange={(event) =>
                setSort(event.target.value as SortOption)
              }
              aria-label="Sort products"
              className="border border-[#ddd6cf] bg-white px-5 py-3 text-sm outline-none md:ml-auto"
            >
              <option value="featured">Featured</option>
              <option value="price-asc">
                Price: Low to High
              </option>
              <option value="price-desc">
                Price: High to Low
              </option>
            </select>
          </div>

          <div className="flex flex-wrap gap-2">
            {categories.map((item) => (
              <button
                key={item}
                type="button"
                onClick={() => setCategory(item)}
                aria-pressed={category === item}
                className={`border px-4 py-2 text-xs transition ${
                  category === item
                    ? "border-[#24221f] bg-[#24221f] text-white"
                    : "border-[#ddd6cf] bg-white hover:border-[#ae805f]"
                }`}
              >
                {item === "All" ? "All Products" : item}
              </button>
            ))}
          </div>

          <p className="text-sm text-[#817a72]">
            Showing {visibleProducts.length} of{" "}
            {products.length} products
          </p>
        </section>

        {visibleProducts.length === 0 ? (
          <div className="py-20 text-center">
            <h2 className="font-serif text-3xl">
              No products found
            </h2>
            <p className="mt-3 text-sm text-[#817a72]">
              Try another search or category.
            </p>
            <button
              type="button"
              onClick={resetFilters}
              className="mt-6 bg-[#24221f] px-6 py-3 text-sm text-white"
            >
              Clear Filters
            </button>
          </div>
        ) : (
          <section
            aria-label="Product catalog"
            className="grid grid-cols-1 gap-x-7 gap-y-12 sm:grid-cols-2 lg:grid-cols-3"
          >
            {visibleProducts.map((product) => (
              <article key={product.id}>
                <div className="relative aspect-[3/4] overflow-hidden bg-[#eee8e0]">
                  {product.imageUrl ? (
                    <Image
                      src={product.imageUrl}
                      alt={`Product image: ${product.name}`}
                      fill
                      sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
                      className="object-cover"
                    />
                  ) : (
                    <div className="flex h-full flex-col items-center justify-center px-6 text-center">
                      <span className="font-serif text-4xl tracking-widest text-[#b5a79a]">
                        ÉLAN.
                      </span>
                      <p className="mt-5 text-xs uppercase tracking-widest text-[#a39385]">
                        Product Image Coming Soon
                      </p>
                    </div>
                  )}

                  <span className="absolute left-4 top-4 bg-white px-3 py-2 text-[10px] font-semibold tracking-wider">
                    DEMO
                  </span>
                </div>

                <div className="mt-5">
                  <p className="text-xs uppercase tracking-widest text-[#a58a76]">
                    {product.category}
                  </p>

                  <h2 className="mt-2 text-xl font-medium">
                    {product.name}
                  </h2>

                  <p className="mt-2 text-base font-semibold">
                    {formatPrice(product.price)}
                  </p>

                  <p className="mt-2 text-xs text-[#817a72]">
                    {product.id}
                  </p>

                  <details className="mt-5 border-t border-[#e7e2dc] pt-4">
                    <summary className="cursor-pointer text-sm font-medium">
                      Product Information
                    </summary>

                    <div className="space-y-3 pt-4 text-sm text-[#77716b]">
                      <p>{product.description}</p>
                      <p>
                        <strong>Colors:</strong>{" "}
                        {product.colors.join(", ")}
                      </p>
                      <p>
                        <strong>Sizes:</strong>{" "}
                        {product.sizes.join(", ")}
                      </p>
                    </div>
                  </details>
                </div>
              </article>
            ))}
          </section>
        )}

        <p className="mt-16 border-t border-[#e7e2dc] pt-6 text-xs leading-6 text-[#817a72]">
          Demonstration catalog with illustrative prices.
          Product images will be added after asset
          verification. No real purchases are processed.
        </p>
      </main>

      <footer className="bg-[#24221f] px-6 py-8 text-center text-xs text-[#c2bab2]">
        ÉLAN. — AI Fashion Recommendation E-Commerce
      </footer>
    </div>
  );
}
