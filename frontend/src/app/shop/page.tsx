"use client";

import Image from "next/image";
import Link from "next/link";
import PersonalizedDemo from "@/components/PersonalizedDemo";
import StoreFooter from "@/components/StoreFooter";
import StoreHeader from "@/components/StoreHeader";
import { useEffect, useId, useRef, useState } from "react";

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
const PERSONALIZED_BREAK_COUNT = 8;

function ProductCard({ product }: { product: Product }) {
  return (
    <article className="elan-product-card group flex flex-col">
      <div className="relative aspect-[3/4] overflow-hidden border border-[#e7e2dc] bg-[#eee8e0]">
        {product.imageUrl ? (
          <Image
            src={product.imageUrl}
            alt={product.name}
            fill
            sizes="(max-width: 640px) 50vw, (max-width: 1024px) 33vw, 25vw"
            className="object-cover object-top transition duration-500 group-hover:scale-105"
          />
        ) : (
          <div className="flex h-full flex-col items-center justify-center px-4 text-center">
            <span className="font-serif text-3xl tracking-widest text-[#b5a79a]">
              ÉLAN.
            </span>
            <p className="mt-3 text-xs uppercase tracking-widest text-[#a39385]">
              Image Coming Soon
            </p>
          </div>
        )}
      </div>

      <p className="mt-4 text-[11px] font-semibold uppercase tracking-[0.18em] text-[#9a6b48]">
        {product.category}
      </p>

      <h2 className="mt-1.5 font-serif text-lg leading-snug font-medium text-[#24221f] transition-colors group-hover:text-[#9a6b48] sm:text-xl">
        <Link
          href={`/shop/${product.slug}`}
          className="elan-card-link"
        >
          {product.name}
        </Link>
      </h2>

      <p className="mt-1.5 text-sm font-medium text-[#24221f]">
        {formatPrice(product.price)}
      </p>
    </article>
  );
}

export default function ShopPage() {
  const searchInputId = useId();
  const sortSelectId = useId();
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState<CategoryFilter>("All");
  const [sort, setSort] = useState<SortOption>("featured");
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
        category === "All" || product.category === category;

      const searchableText = [
        product.name,
        product.category,
        product.description,
        ...product.colors,
      ]
        .join(" ")
        .toLowerCase();

      return matchesCategory && searchableText.includes(query);
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

  const hasActiveFilters =
    search.trim() !== "" || category !== "All" || sort !== "featured";

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

  const firstBatch = pageProducts.slice(0, PERSONALIZED_BREAK_COUNT);
  const remainingBatch = pageProducts.slice(PERSONALIZED_BREAK_COUNT);

  return (
    <div className="flex min-h-screen flex-col bg-[#faf9f6] text-[#24221f]">
      <StoreHeader />

      <main id="main-content" className="elan-container flex-1 py-10 md:py-16">
        {/* Page Heading */}
        <div className="mb-10 flex flex-col justify-between gap-6 lg:flex-row lg:items-end">
          <div className="max-w-2xl">
            <p className="text-xs font-semibold uppercase tracking-[0.26em] text-[#9a6b48]">
              The ÉLAN Collection
            </p>

            <h1 className="mt-3 font-serif text-4xl sm:text-5xl md:text-6xl">
              Shop the Collection
            </h1>

            <p className="mt-4 text-sm leading-7 text-[#686159] sm:text-base">
              Discover contemporary essentials designed for effortless everyday
              styling across outerwear, tailoring, knitwear, and accessories.
            </p>
          </div>

          <a
            href="#personalized-demo-heading"
            className="inline-flex min-h-11 w-fit items-center gap-2 border border-[#d5cdc3] bg-white px-4 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-[#24221f] transition-colors hover:border-[#9a6b48] hover:text-[#9a6b48]"
          >
            <span>Explore Your Style Edit</span>
            <span aria-hidden="true">↓</span>
          </a>
        </div>

        {/* Search, Category Filters, and Sorting */}
        <search
          ref={filtersRef}
          aria-label="Catalog search and filters"
          className="mb-10 space-y-5 border-y border-[#e7e2dc] bg-[#faf9f6] py-6"
        >
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div className="w-full md:max-w-md">
              <label
                htmlFor={searchInputId}
                className="block text-[11px] font-semibold uppercase tracking-[0.18em] text-[#686159]"
              >
                Search Pieces
              </label>
              <div className="relative mt-1.5 flex items-center">
                <input
                  id={searchInputId}
                  type="search"
                  value={search}
                  onChange={(event) => {
                    setSearch(event.target.value);
                    setPage(1);
                  }}
                  placeholder="Search by piece, category, or color..."
                  autoCorrect="off"
                  autoCapitalize="none"
                  spellCheck={false}
                  className="min-h-11 w-full border border-[#d5cdc3] bg-white px-4 py-2.5 text-sm text-[#24221f] transition-colors focus:border-[#9a6b48]"
                />
              </div>
            </div>

            <div className="w-full sm:w-60">
              <label
                htmlFor={sortSelectId}
                className="block text-[11px] font-semibold uppercase tracking-[0.18em] text-[#686159]"
              >
                Sort By
              </label>
              <select
                id={sortSelectId}
                value={sort}
                onChange={(event) => {
                  setSort(event.target.value as SortOption);
                  setPage(1);
                }}
                className="mt-1.5 min-h-11 w-full border border-[#d5cdc3] bg-white px-4 py-2.5 text-sm text-[#24221f] transition-colors hover:border-[#9a6b48]"
              >
                <option value="featured">Featured</option>
                <option value="price-asc">Price: Low to High</option>
                <option value="price-desc">Price: High to Low</option>
              </select>
            </div>
          </div>

          {/* Category Filter Pills */}
          <div
            role="group"
            aria-label="Filter by category"
            className="flex flex-wrap gap-2 pt-1"
          >
            {categories.map((item) => {
              const active = category === item;
              return (
                <button
                  key={item}
                  type="button"
                  onClick={() => {
                    setCategory(item);
                    setPage(1);
                  }}
                  aria-pressed={active}
                  className={`inline-flex min-h-10 items-center border px-4 py-2 text-xs font-medium uppercase tracking-[0.14em] transition-colors ${
                    active
                      ? "border-[#24221f] bg-[#24221f] font-semibold text-white"
                      : "border-[#d5cdc3] bg-white text-[#686159] hover:border-[#9a6b48] hover:text-[#24221f]"
                  }`}
                >
                  {item === "All" ? "All Pieces" : item}
                </button>
              );
            })}
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 pt-1 text-xs text-[#686159]">
            <p className="m-0">
              {loading ? (
                "Loading collection..."
              ) : loadError ? (
                "Product list unavailable"
              ) : (
                <>
                  Showing{" "}
                  <strong className="font-semibold text-[#24221f]">
                    {firstShown}–{lastShown}
                  </strong>{" "}
                  of{" "}
                  <strong className="font-semibold text-[#24221f]">
                    {visibleProducts.length}
                  </strong>{" "}
                  pieces
                  {visibleProducts.length !== products.length && (
                    <> (filtered from {products.length} total)</>
                  )}
                </>
              )}
            </p>

            {hasActiveFilters && (
              <button
                type="button"
                onClick={resetFilters}
                className="font-semibold uppercase tracking-[0.14em] text-[#9a6b48] underline underline-offset-4 hover:text-[#24221f]"
              >
                Reset Filters
              </button>
            )}
          </div>
        </search>

        {/* Product Grid + In-Journey Personalized Discovery */}
        {loading ? (
          <>
            <p role="status" className="py-16 text-center text-sm text-[#686159]">
              Loading collection...
            </p>
            <PersonalizedDemo />
          </>
        ) : loadError ? (
          <>
            <p role="alert" className="py-16 text-center text-sm text-red-700">
              {loadError}
            </p>
            <PersonalizedDemo />
          </>
        ) : visibleProducts.length === 0 ? (
          <>
            <div className="border border-[#e7e2dc] bg-white px-6 py-16 text-center">
              <h2 className="font-serif text-3xl">No matching pieces found</h2>

              <p className="mt-3 text-sm text-[#686159]">
                Try adjusting your search keywords or selecting another
                category.
              </p>

              <button
                type="button"
                onClick={resetFilters}
                className="elan-btn-primary mt-6"
              >
                Clear Filters
              </button>
            </div>
            <PersonalizedDemo />
          </>
        ) : (
          <>
            <section
              aria-label="Product catalog"
              className="grid grid-cols-2 gap-x-4 gap-y-9 sm:gap-x-6 sm:gap-y-11 md:grid-cols-3 lg:grid-cols-4"
            >
              {firstBatch.map((product) => (
                <ProductCard key={product.id} product={product} />
              ))}
            </section>

            <PersonalizedDemo />

            {remainingBatch.length > 0 && (
              <section
                aria-label="More catalog pieces"
                className="mt-12 grid grid-cols-2 gap-x-4 gap-y-9 sm:gap-x-6 sm:gap-y-11 md:grid-cols-3 lg:grid-cols-4"
              >
                {remainingBatch.map((product) => (
                  <ProductCard key={product.id} product={product} />
                ))}
              </section>
            )}
          </>
        )}

        {/* Client-side pagination: 24 cards per page */}
        {!loading && !loadError && visibleProducts.length > PAGE_SIZE && (
          <nav
            aria-label="Product pagination"
            className="mt-16 flex flex-col items-center justify-between gap-4 border-t border-[#e7e2dc] pt-8 sm:flex-row"
          >
            <p className="text-xs font-medium uppercase tracking-[0.16em] text-[#686159]">
              Page {currentPage} of {totalPages}
            </p>

            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => changePage(currentPage - 1)}
                disabled={currentPage === 1}
                className="elan-btn-secondary disabled:cursor-not-allowed disabled:opacity-40"
              >
                Previous
              </button>

              <button
                type="button"
                onClick={() => changePage(currentPage + 1)}
                disabled={currentPage === totalPages}
                className="elan-btn-secondary disabled:cursor-not-allowed disabled:opacity-40"
              >
                Next
              </button>
            </div>
          </nav>
        )}
      </main>

      <StoreFooter />
    </div>
  );
}
