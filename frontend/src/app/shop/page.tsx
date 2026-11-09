"use client";

import Image from "next/image";
import Link from "next/link";
import StoreHeader from "@/components/StoreHeader";
import { useEffect, useRef, useState } from "react";

import {
  formatPrice,
  type Product,
  type ProductCategory,
} from "@/data/products";

import {
  toStorefrontProduct,
  type BackendProduct,
} from "@/data/product-api";

type CategoryFilter = "All" | ProductCategory;
type SortOption = "featured" | "price-asc" | "price-desc";

const PAGE_SIZE = 24;

export default function ShopPage() {
  const [search, setSearch] = useState("");
  const [category, setCategory] =
    useState<CategoryFilter>("All");
  const [sort, setSort] =
    useState<SortOption>("featured");
  const [products, setProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const filtersRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const controller = new AbortController();

    async function loadProducts() {
      try {
        const response = await fetch("/api/products", {
          cache: "no-store",
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`Failed to load products: ${response.status}`);
        }

        const data: unknown = await response.json();

        if (!Array.isArray(data)) {
          throw new Error("Invalid product response");
        }

        const activeProducts = (data as BackendProduct[])
          .filter((product) => product.is_active)
          .map(toStorefrontProduct);

        if (!controller.signal.aborted) {
          setProducts(activeProducts);
          setLoadError(null);
        }
      } catch {
        if (!controller.signal.aborted) {
          setLoadError("Unable to load products. Please refresh the page.");
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    void loadProducts();

    return () => controller.abort();
  }, []);

  const categories: CategoryFilter[] = [
    "All",
    ...Array.from(new Set(products.map((product) => product.category))),
  ];

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

  // Filter and sort the full catalog before selecting the current page.
  const totalPages = Math.max(
    1,
    Math.ceil(visibleProducts.length / PAGE_SIZE)
  );
  const currentPage = Math.min(page, totalPages);
  const startIndex = (currentPage - 1) * PAGE_SIZE;
  const pageProducts = visibleProducts.slice(
    startIndex,
    startIndex + PAGE_SIZE
  );
  const firstShown = visibleProducts.length === 0 ? 0 : startIndex + 1;
  const lastShown = Math.min(
    startIndex + PAGE_SIZE,
    visibleProducts.length
  );

  function changePage(nextPage: number) {
    setPage(Math.max(1, Math.min(nextPage, totalPages)));
    filtersRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }

  function resetFilters() {
    setSearch("");
    setCategory("All");
    setSort("featured");
    setPage(1);
  }

  return (
    <div className="min-h-screen bg-[#faf9f6] text-[#24221f]">
      {/* Announcement Bar */}
      <div className="bg-[#24221f] px-4 py-2 text-center text-xs tracking-wider text-white">
        ÉLAN Fashion Store — Portfolio Demo
      </div>

      {/* Navigation */}
      <StoreHeader />

      <main className="mx-auto max-w-7xl px-6 py-12 md:py-16">
        {/* Page Heading */}
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

        {/* Search, Filters, Sorting */}
        <section
          ref={filtersRef}
          aria-label="Product filters"
          className="mb-10 space-y-6 border-y border-[#e7e2dc] py-6"
        >
          <div className="flex flex-col gap-4 md:flex-row">
            <input
              type="search"
              value={search}
              onChange={(event) => {
                setSearch(event.target.value);
                setPage(1);
              }}
              placeholder="Search products..."
              aria-label="Search products"
              className="w-full border border-[#ddd6cf] bg-white px-5 py-3 text-sm outline-none focus:border-[#ae805f] md:max-w-md"
            />

            <select
              value={sort}
              onChange={(event) => {
                setSort(event.target.value as SortOption);
                setPage(1);
              }}
              aria-label="Sort products"
              className="border border-[#ddd6cf] bg-white px-5 py-3 text-sm outline-none md:ml-auto"
            >
              <option value="featured">
                Featured
              </option>
              <option value="price-asc">
                Price: Low to High
              </option>
              <option value="price-desc">
                Price: High to Low
              </option>
            </select>
          </div>

          {/* Category Filters */}
          <div className="flex flex-wrap gap-2">
            {categories.map((item) => (
              <button
                key={item}
                type="button"
                onClick={() => {
                  setCategory(item);
                  setPage(1);
                }}
                aria-pressed={category === item}
                className={`border px-4 py-2 text-xs transition ${
                  category === item
                    ? "border-[#24221f] bg-[#24221f] text-white"
                    : "border-[#ddd6cf] bg-white hover:border-[#ae805f]"
                }`}
              >
                {item === "All"
                  ? "All Products"
                  : item}
              </button>
            ))}
          </div>

          <p className="text-sm text-[#817a72]">
            {loading
              ? "Loading collection..."
              : loadError
                ? "Product list unavailable"
                : (
                  <>
                    Showing {firstShown}–{lastShown} of{" "}
                    {visibleProducts.length} products
                    {visibleProducts.length !== products.length && (
                      <> (filtered from {products.length} total)</>
                    )}
                  </>
                )}
          </p>
        </section>

        {/* Product Grid */}
        {loading ? (
          <p role="status" className="py-20 text-center text-[#817a72]">
            Loading products...
          </p>
        ) : loadError ? (
          <p role="alert" className="py-20 text-center text-red-700">
            {loadError}
          </p>
        ) : visibleProducts.length === 0 ? (
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
            {pageProducts.map((product) => (
              <article
                key={product.id}
                className="group"
              >
                {/* Clickable Product Image */}
                <Link
                  href={`/shop/${product.slug}`}
                  aria-label={`View ${product.name}`}
                  className="relative block aspect-[3/4] overflow-hidden bg-[#eee8e0]"
                >
                  {product.imageUrl ? (
                    <Image
                      src={product.imageUrl}
                      alt={`Product image: ${product.name}`}
                      fill
                      sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
                      className="object-cover transition duration-500 group-hover:scale-105"
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

                  <span className="pointer-events-none absolute left-4 top-4 bg-white px-3 py-2 text-[10px] font-semibold tracking-wider">
                    DEMO
                  </span>
                </Link>

                {/* Product Information */}
                <div className="mt-5">
                  <p className="text-xs uppercase tracking-widest text-[#a58a76]">
                    {product.category}
                  </p>

                  {/* Clickable Product Name */}
                  <h2 className="mt-2 text-xl font-medium">
                    <Link
                      href={`/shop/${product.slug}`}
                      className="transition hover:text-[#ae805f]"
                    >
                      {product.name}
                    </Link>
                  </h2>

                  <p className="mt-2 text-base font-semibold">
                    {formatPrice(product.price)}
                  </p>

                  <p className="mt-2 text-xs text-[#817a72]">
                    {product.id}
                  </p>

                  {/* Product Details Summary */}
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

                  {/* Product Detail Navigation */}
                  <Link
                    href={`/shop/${product.slug}`}
                    className="mt-5 inline-block border-b border-[#24221f] pb-1 text-sm font-medium transition hover:border-[#ae805f] hover:text-[#ae805f]"
                  >
                    View Product Details
                  </Link>
                </div>
              </article>
            ))}
          </section>
        )}

        {/* Client-side pagination: only 24 cards render per page. */}
        {!loading && !loadError && visibleProducts.length > PAGE_SIZE && (
          <nav
            aria-label="Product pagination"
            className="mt-12 flex flex-col items-center justify-between gap-4 border-t border-[#e7e2dc] pt-6 sm:flex-row"
          >
            <p className="text-sm text-[#817a72]">
              Page {currentPage} of {totalPages}
            </p>

            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => changePage(currentPage - 1)}
                disabled={currentPage === 1}
                className="border border-[#ddd6cf] bg-white px-5 py-3 text-sm transition hover:border-[#ae805f] disabled:cursor-not-allowed disabled:opacity-40"
              >
                Previous
              </button>

              <button
                type="button"
                onClick={() => changePage(currentPage + 1)}
                disabled={currentPage === totalPages}
                className="border border-[#ddd6cf] bg-white px-5 py-3 text-sm transition hover:border-[#ae805f] disabled:cursor-not-allowed disabled:opacity-40"
              >
                Next
              </button>
            </div>
          </nav>
        )}

        {/* Demo Disclaimer */}
        <p className="mt-16 border-t border-[#e7e2dc] pt-6 text-xs leading-6 text-[#817a72]">
          Demonstration catalog with illustrative prices.
          Product images are for demonstration only.
          No real purchases are processed.
        </p>
      </main>

      <footer className="bg-[#24221f] px-6 py-8 text-center text-xs text-[#c2bab2]">
        ÉLAN. — AI Fashion Recommendation E-Commerce
      </footer>
    </div>
  );
}
