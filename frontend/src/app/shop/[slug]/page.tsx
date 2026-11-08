
"use client";

import Image from "next/image";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useState } from "react";

import {
  formatPrice,
  getProductBySlug,
} from "@/data/products";

export default function ProductDetailPage() {
  const { slug } = useParams<{ slug: string }>();

  const product = getProductBySlug(slug);

  const [selectedSku, setSelectedSku] = useState<
    string | null
  >(null);

  if (!product) {
    return (
      <main className="flex min-h-screen flex-col items-center justify-center bg-[#faf9f6] px-6">
        <h1 className="font-serif text-4xl">
          Product not found
        </h1>

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
    product.variants.find(
      (variant) => variant.sku === selectedSku
    ) ?? product.variants[0];

  if (!selectedVariant) {
    return (
      <main className="p-12 text-center">
        No product variants available.
      </main>
    );
  }

  const availableColors = [
    ...new Set(
      product.variants.map((variant) => variant.color)
    ),
  ];

  const availableSizes = [
    ...new Set(
      product.variants
        .filter(
          (variant) =>
            variant.color === selectedVariant.color
        )
        .map((variant) => variant.size)
    ),
  ];

  function chooseColor(color: string) {
    const variant =
      product!.variants.find(
        (item) =>
          item.color === color &&
          item.size === selectedVariant.size
      ) ??
      product!.variants.find(
        (item) => item.color === color
      );

    if (variant) {
      setSelectedSku(variant.sku);
    }
  }

  function chooseSize(size: string) {
    const variant = product!.variants.find(
      (item) =>
        item.color === selectedVariant.color &&
        item.size === size
    );

    if (variant) {
      setSelectedSku(variant.sku);
    }
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

          <div className="flex gap-8 text-sm">
            <Link href="/" className="hover:text-[#ae805f]">
              Home
            </Link>

            <Link
              href="/shop"
              className="hover:text-[#ae805f]"
            >
              Shop
            </Link>
          </div>

          <span className="hidden text-xs text-[#80776e] sm:block">
            Fashion Discovery
          </span>
        </nav>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-12">
        <div className="mb-10 text-sm text-[#80776e]">
          <Link href="/shop" className="hover:underline">
            Shop
          </Link>
          {" / "}
          {product.category}
          {" / "}
          <span className="text-[#24221f]">
            {product.name}
          </span>
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
                    aria-pressed={
                      selectedVariant.color === color
                    }
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
              <p className="mb-4 text-sm font-medium">
                Select Size
              </p>

              <div className="flex flex-wrap gap-3">
                {availableSizes.map((size) => (
                  <button
                    key={size}
                    type="button"
                    onClick={() => chooseSize(size)}
                    aria-pressed={
                      selectedVariant.size === size
                    }
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
              disabled
              className="mt-10 w-full cursor-not-allowed bg-[#24221f] px-8 py-5 text-sm font-medium text-white opacity-60"
            >
              Add to Cart — Coming Soon
            </button>

            <p className="mt-5 text-xs leading-6 text-[#817a72]">
              Portfolio demonstration only. Product
              availability and checkout are not active.
            </p>
          </div>
        </div>
      </main>

      <footer className="mt-20 bg-[#24221f] px-6 py-10 text-center text-xs text-white">
        ÉLAN. — AI Fashion Recommendation E-Commerce
      </footer>
    </div>
  );
}
