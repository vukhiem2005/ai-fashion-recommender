"use client";

import YouMayAlsoLike from "@/components/YouMayAlsoLike";
import StoreFooter from "@/components/StoreFooter";
import StoreHeader from "@/components/StoreHeader";
import { addToCart } from "@/lib/cart";
import Image from "next/image";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import {
  formatPrice,
  type Product,
} from "@/data/products";
import {
  toStorefrontProduct,
  type BackendProduct,
} from "@/data/product-api";

function isMeaningfulOption(value: string): boolean {
  const normalized = value.trim().toLowerCase();
  return normalized !== "" && normalized !== "not specified";
}

export default function ProductDetailPage() {
  const { slug } = useParams<{ slug: string }>();

  // Remount when the URL changes so that color/size selection
  // never carries over to a different product.
  return <ProductDetailContent key={slug} slug={slug} />;
}

function ProductDetailContent({ slug }: { slug: string }) {
  const [product, setProduct] = useState<Product | null>(null);
  const [selectedSku, setSelectedSku] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [cartFeedback, setCartFeedback] = useState<{
    type: "success" | "error";
    message: string;
  } | null>(null);

  useEffect(() => {
    const controller = new AbortController();

    async function loadProduct() {
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

        const backendProduct = (data as BackendProduct[]).find(
          (item) => item.slug === slug && item.is_active
        );

        if (!controller.signal.aborted) {
          setProduct(
            backendProduct ? toStorefrontProduct(backendProduct) : null
          );
          setLoadError(null);
        }
      } catch {
        if (!controller.signal.aborted) {
          setLoadError("Unable to load product details. Please try again.");
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    void loadProduct();

    return () => controller.abort();
  }, [slug]);

  if (loading) {
    return (
      <div className="flex min-h-screen flex-col bg-[#faf9f6] text-[#24221f]">
        <StoreHeader />
        <main
          id="main-content"
          className="elan-container flex flex-1 flex-col items-center justify-center py-20 text-center"
        >
          <p role="status" className="text-sm text-[#686159]">
            Loading piece details...
          </p>
        </main>
        <StoreFooter />
      </div>
    );
  }

  if (loadError) {
    return (
      <div className="flex min-h-screen flex-col bg-[#faf9f6] text-[#24221f]">
        <StoreHeader />
        <main
          id="main-content"
          className="elan-container flex flex-1 flex-col items-center justify-center py-20 text-center"
        >
          <h1 className="font-serif text-4xl">Unable to load piece</h1>
          <p role="alert" className="mt-4 text-sm text-red-700">
            {loadError}
          </p>
          <Link href="/shop" className="elan-btn-primary mt-8">
            Back to Collection
          </Link>
        </main>
        <StoreFooter />
      </div>
    );
  }

  if (!product) {
    return (
      <div className="flex min-h-screen flex-col bg-[#faf9f6] text-[#24221f]">
        <StoreHeader />
        <main
          id="main-content"
          className="elan-container flex flex-1 flex-col items-center justify-center py-20 text-center"
        >
          <h1 className="font-serif text-4xl">Piece not found</h1>
          <p className="mt-4 text-sm text-[#686159]">
            This piece is not available in the current collection.
          </p>
          <Link href="/shop" className="elan-btn-primary mt-8">
            Back to Collection
          </Link>
        </main>
        <StoreFooter />
      </div>
    );
  }

  const selectedVariant =
    product.variants.find((variant) => variant.sku === selectedSku) ??
    product.variants[0];

  if (!selectedVariant) {
    return (
      <div className="flex min-h-screen flex-col bg-[#faf9f6] text-[#24221f]">
        <StoreHeader />
        <main
          id="main-content"
          className="elan-container flex flex-1 flex-col items-center justify-center py-20 text-center"
        >
          <p className="text-sm text-[#686159]">
            No product variants available.
          </p>
          <Link href="/shop" className="elan-btn-secondary mt-8">
            Back to Collection
          </Link>
        </main>
        <StoreFooter />
      </div>
    );
  }

  const availableColors = [
    ...new Set(product.variants.map((variant) => variant.color)),
  ];

  const availableSizes = [
    ...new Set(
      product.variants
        .filter((variant) => variant.color === selectedVariant.color)
        .map((variant) => variant.size)
    ),
  ];

  const selectableColors = availableColors.filter(isMeaningfulOption);
  const selectableSizes = availableSizes.filter(isMeaningfulOption);

  function chooseColor(color: string) {
    if (!product) return;

    const variant =
      product.variants.find(
        (item) =>
          item.color === color && item.size === selectedVariant.size
      ) ?? product.variants.find((item) => item.color === color);

    if (variant) {
      setSelectedSku(variant.sku);
      setCartFeedback(null);
    }
  }

  function chooseSize(size: string) {
    if (!product) return;

    const variant = product.variants.find(
      (item) =>
        item.color === selectedVariant.color && item.size === size
    );

    if (variant) {
      setSelectedSku(variant.sku);
      setCartFeedback(null);
    }
  }

  function handleAddToCart() {
    if (!product || !selectedVariant) {
      setCartFeedback({
        type: "error",
        message: "Please select an available option.",
      });
      return;
    }

    try {
      addToCart({
        productId: product.id,
        sku: selectedVariant.sku,
        name: product.name,
        color: selectedVariant.color,
        size: selectedVariant.size,
        priceCents: Math.round(product.price * 100),
        imageUrl: selectedVariant.imageUrl,
      });

      const variantDetails = [
        isMeaningfulOption(selectedVariant.color)
          ? selectedVariant.color
          : null,
        isMeaningfulOption(selectedVariant.size)
          ? `Size ${selectedVariant.size}`
          : null,
      ]
        .filter(Boolean)
        .join(" · ");

      setCartFeedback({
        type: "success",
        message: variantDetails
          ? `${product.name} (${variantDetails}) added to your bag.`
          : `${product.name} added to your bag.`,
      });
    } catch {
      setCartFeedback({
        type: "error",
        message: "Unable to save your shopping bag. Please try again.",
      });
    }
  }

  return (
    <div className="flex min-h-screen flex-col bg-[#faf9f6] text-[#24221f]">
      <StoreHeader />

      <main id="main-content" className="elan-container flex-1 py-10 pb-20 md:py-14">
        {/* Breadcrumb Navigation */}
        <nav aria-label="Breadcrumb" className="mb-8 text-xs uppercase tracking-[0.16em] text-[#686159]">
          <ol className="m-0 flex flex-wrap items-center gap-2 p-0 list-none">
            <li>
              <Link href="/shop" className="transition-colors hover:text-[#24221f]">
                Collection
              </Link>
            </li>
            <li aria-hidden="true" className="text-[#b8afa6]">
              /
            </li>
            <li>{product.category}</li>
            <li aria-hidden="true" className="text-[#b8afa6]">
              /
            </li>
            <li aria-current="page" className="font-semibold text-[#24221f]">
              {product.name}
            </li>
          </ol>
        </nav>

        <div className="grid gap-10 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,0.95fr)] lg:gap-16">
          {/* Product Gallery */}
          <div className="relative aspect-[3/4] w-full overflow-hidden border border-[#e7e2dc] bg-[#eee8e0]">
            {selectedVariant.imageUrl ? (
              <Image
                src={selectedVariant.imageUrl}
                alt={
                  isMeaningfulOption(selectedVariant.color)
                    ? `${product.name} in ${selectedVariant.color}`
                    : product.name
                }
                fill
                sizes="(max-width: 1024px) 100vw, 50vw"
                className="object-cover object-top"
                priority
                fetchPriority="high"
              />
            ) : (
              <div className="flex h-full flex-col items-center justify-center px-6 text-center">
                <span className="font-serif text-5xl tracking-widest text-[#b5a79a]">
                  ÉLAN.
                </span>
                <p className="mt-4 text-xs uppercase tracking-[0.2em] text-[#a39385]">
                  Image Coming Soon
                </p>
              </div>
            )}
          </div>

          {/* Primary Purchase & Product Information Panel */}
          <div className="flex flex-col justify-center lg:py-4">
            <p className="text-xs font-semibold uppercase tracking-[0.26em] text-[#9a6b48]">
              {product.category}
            </p>

            <h1 className="mt-3 font-serif text-3xl leading-tight sm:text-4xl md:text-5xl">
              {product.name}
            </h1>

            <p className="mt-4 text-xl font-medium text-[#24221f] sm:text-2xl">
              {formatPrice(product.price)}
            </p>

            <p className="mt-6 max-w-xl text-sm leading-7 text-[#686159] sm:text-base sm:leading-8">
              {product.description}
            </p>

            {/* Color Presentation: selector only when multiple colors exist */}
            {selectableColors.length > 0 && (
              <div className="mt-8 border-t border-[#e7e2dc] pt-7">
                {selectableColors.length === 1 ? (
                  <p className="text-xs uppercase tracking-[0.18em] text-[#686159]">
                    Color:{" "}
                    <span className="font-semibold text-[#24221f]">
                      {selectableColors[0]}
                    </span>
                  </p>
                ) : (
                  <div>
                    <p className="mb-3 text-xs uppercase tracking-[0.18em] text-[#686159]">
                      Color:{" "}
                      <span className="font-semibold text-[#24221f]">
                        {selectedVariant.color}
                      </span>
                    </p>

                    <div
                      role="group"
                      aria-label="Select color"
                      className="flex flex-wrap gap-2.5"
                    >
                      {selectableColors.map((color) => {
                        const active = selectedVariant.color === color;
                        return (
                          <button
                            key={color}
                            type="button"
                            onClick={() => chooseColor(color)}
                            aria-pressed={active}
                            className={`inline-flex min-h-11 items-center border px-5 py-2.5 text-xs font-medium uppercase tracking-[0.14em] transition-colors ${
                              active
                                ? "border-[#24221f] bg-[#24221f] font-semibold text-white"
                                : "border-[#d5cdc3] bg-white text-[#24221f] hover:border-[#9a6b48]"
                            }`}
                          >
                            {color}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Size Presentation: hide "Not specified", selector only when multiple sizes exist */}
            {selectableSizes.length > 0 && (
              <div
                className={
                  selectableColors.length > 0
                    ? "mt-6"
                    : "mt-8 border-t border-[#e7e2dc] pt-7"
                }
              >
                {selectableSizes.length === 1 ? (
                  <p className="text-xs uppercase tracking-[0.18em] text-[#686159]">
                    Size:{" "}
                    <span className="font-semibold text-[#24221f]">
                      {selectableSizes[0]}
                    </span>
                  </p>
                ) : (
                  <div>
                    <p className="mb-3 text-xs uppercase tracking-[0.18em] text-[#686159]">
                      Select Size:{" "}
                      <span className="font-semibold text-[#24221f]">
                        {selectedVariant.size}
                      </span>
                    </p>

                    <div
                      role="group"
                      aria-label="Select size"
                      className="flex flex-wrap gap-2.5"
                    >
                      {selectableSizes.map((size) => {
                        const active = selectedVariant.size === size;
                        return (
                          <button
                            key={size}
                            type="button"
                            onClick={() => chooseSize(size)}
                            aria-pressed={active}
                            className={`inline-flex min-h-11 min-w-13 items-center justify-center border px-4 py-2.5 text-xs font-medium uppercase tracking-[0.14em] transition-colors ${
                              active
                                ? "border-[#24221f] bg-[#24221f] font-semibold text-white"
                                : "border-[#d5cdc3] bg-white text-[#24221f] hover:border-[#9a6b48]"
                            }`}
                          >
                            {size}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            )}

            <button
              type="button"
              onClick={handleAddToCart}
              className="elan-btn-primary mt-9 w-full py-4"
            >
              Add to Bag
            </button>

            {cartFeedback && (
              <div
                role={cartFeedback.type === "error" ? "alert" : "status"}
                aria-live="polite"
                className={`mt-4 flex flex-wrap items-center justify-between gap-3 border px-4 py-3 text-sm ${
                  cartFeedback.type === "success"
                    ? "border-[#d5cdc3] bg-[#f4efe8] text-[#24221f]"
                    : "border-red-200 bg-red-50 text-red-700"
                }`}
              >
                <span>{cartFeedback.message}</span>
                {cartFeedback.type === "success" && (
                  <Link
                    href="/cart"
                    className="text-xs font-semibold uppercase tracking-[0.16em] text-[#9a6b48] underline underline-offset-4 hover:text-[#24221f]"
                  >
                    View Shopping Bag →
                  </Link>
                )}
              </div>
            )}

            <p className="mt-5 text-xs leading-6 text-[#756d65]">
              Your shopping bag is saved in this browser. Checkout and payments
              are not active.
            </p>
          </div>
        </div>

        <YouMayAlsoLike productId={product.id} />
      </main>

      <StoreFooter />
    </div>
  );
}
