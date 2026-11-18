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
        }
      } catch {
        if (!controller.signal.aborted) {
          setLoadError("Unable to load product information. Please try again.");
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
      <main className="flex min-h-screen items-center justify-center bg-[#faf9f6] px-6">
        <p role="status" className="text-sm text-[#77716b]">
          Loading product...
        </p>
      </main>
    );
  }

  if (loadError) {
    return (
      <main className="flex min-h-screen flex-col items-center justify-center bg-[#faf9f6] px-6 text-center">
        <h1 className="font-serif text-4xl">Unable to load product</h1>
        <p role="alert" className="mt-4 text-[#77716b]">
          {loadError}
        </p>
        <Link
          href="/shop"
          className="mt-8 bg-[#24221f] px-8 py-3 text-sm text-white"
        >
          Back to Shop
        </Link>
      </main>
    );
  }

  if (!product) {
    return (
      <main className="flex min-h-screen flex-col items-center justify-center bg-[#faf9f6] px-6 text-center">
        <h1 className="font-serif text-4xl">Product not found</h1>
        <p className="mt-4 text-[#77716b]">
          This product is not available in our demo catalog.
        </p>
        <Link
          href="/shop"
          className="mt-8 bg-[#24221f] px-8 py-3 text-sm text-white"
        >
          Back to Shop
        </Link>
      </main>
    );
  }

  const selectedVariant =
    product.variants.find((variant) => variant.sku === selectedSku) ??
    product.variants[0];

  if (!selectedVariant) {
    return (
      <main className="flex min-h-screen flex-col items-center justify-center bg-[#faf9f6] px-6 text-center">
        <p className="text-[#77716b]">No product variants available.</p>
        <Link href="/shop" className="mt-8 underline">
          Back to Shop
        </Link>
      </main>
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
    // The click handler is a separate closure: guard both values here
    // so TypeScript knows they exist when the button is clicked.
    if (!product || !selectedVariant) {
      setCartFeedback({
        type: "error",
        message: "Please select an available product variant.",
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

      setCartFeedback({
        type: "success",
        message: `${product.name} (${selectedVariant.color} / ${selectedVariant.size}) added to your bag.`,
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
      <main id="main-content" className="mx-auto w-full max-w-7xl flex-1 px-6 py-12 pb-20">
        <div className="mb-10 text-sm text-[#80776e]">
          <Link href="/shop" className="hover:underline">
            Shop
          </Link>
          {" / "}
          {product.category}
          {" / "}
          <span className="text-[#24221f]">{product.name}</span>
        </div>

        <div className="grid gap-12 lg:grid-cols-2 lg:gap-20">
          <div className="relative aspect-[3/4] overflow-hidden bg-[#eee8e0]">
            {selectedVariant.imageUrl ? (
              <Image
                src={selectedVariant.imageUrl}
                alt={`${product.name} - ${selectedVariant.color}`}
                fill
                sizes="(max-width: 1024px) 100vw, 50vw"
                className="object-cover"
                priority
              />
            ) : (
              <div className="flex h-full flex-col items-center justify-center">
                <span className="font-serif text-5xl text-[#b5a79a]">
                  ÉLAN.
                </span>

                <p className="mt-5 text-xs uppercase tracking-wider text-[#a39385]">
                  Product Image Coming Soon
                </p>
              </div>
            )}

            <span className="absolute left-5 top-5 bg-white px-3 py-2 text-xs">
              DEMO
            </span>
          </div>

          <div className="lg:py-8">
            <p className="text-xs uppercase tracking-[0.25em] text-[#ae805f]">
              {product.category}
            </p>

            <h1 className="mt-5 font-serif text-4xl md:text-5xl">
              {product.name}
            </h1>

            <p className="mt-6 text-2xl font-semibold">
              {formatPrice(product.price)}
            </p>

            <p className="mt-8 leading-8 text-[#77716b]">
              {product.description}
            </p>

            <div className="mt-10 border-t border-[#e7e2dc] pt-8">
              <p className="mb-4 text-sm font-medium">
                Color: {selectedVariant.color}
              </p>

              <div className="flex flex-wrap gap-3">
                {availableColors.map((color) => (
                  <button
                    key={color}
                    type="button"
                    onClick={() => chooseColor(color)}
                    aria-pressed={selectedVariant.color === color}
                    className={`border px-5 py-3 text-sm transition ${
                      selectedVariant.color === color
                        ? "border-[#24221f] bg-[#24221f] text-white"
                        : "border-[#ddd6cf] bg-white hover:border-[#ae805f]"
                    }`}
                  >
                    {color}
                  </button>
                ))}
              </div>
            </div>

            <div className="mt-8">
              <p className="mb-4 text-sm font-medium">Select Size</p>

              <div className="flex flex-wrap gap-3">
                {availableSizes.map((size) => (
                  <button
                    key={size}
                    type="button"
                    onClick={() => chooseSize(size)}
                    aria-pressed={selectedVariant.size === size}
                    className={`min-w-14 border px-4 py-3 text-sm transition ${
                      selectedVariant.size === size
                        ? "border-[#24221f] bg-[#24221f] text-white"
                        : "border-[#ddd6cf] bg-white hover:border-[#ae805f]"
                    }`}
                  >
                    {size}
                  </button>
                ))}
              </div>
            </div>

            <div className="mt-10 border-t border-[#e7e2dc] pt-6">
              <p className="text-sm text-[#817a72]">
                Product ID: {product.id}
              </p>

              <p className="mt-2 text-sm text-[#817a72]">
                Selected SKU: {selectedVariant.sku}
              </p>
            </div>

            <button
              type="button"
              onClick={handleAddToCart}
              className="mt-10 w-full bg-[#24221f] px-8 py-5 text-sm font-medium text-white transition hover:bg-[#ae805f]"
            >
              Add to Cart
            </button>

            {cartFeedback && (
              <p
                role={cartFeedback.type === "error" ? "alert" : "status"}
                aria-live="polite"
                className={`mt-4 text-sm ${
                  cartFeedback.type === "success"
                    ? "text-green-700"
                    : "text-red-700"
                }`}
              >
                {cartFeedback.message}
              </p>
            )}
{cartFeedback?.type === "success" && (
  <Link
    href="/cart"
    className="mt-3 inline-block text-sm font-semibold text-[#ae805f] underline"
  >
    View Shopping Bag
  </Link>
)}


            <p className="mt-5 text-xs leading-6 text-[#817a72]">
              Portfolio demo only. Your shopping bag is saved in this browser.
              Checkout and payments are not active.
            </p>
          </div>
        </div>
        <YouMayAlsoLike productId={product.id} />
      </main>

      <StoreFooter />
    </div>
  );
}
