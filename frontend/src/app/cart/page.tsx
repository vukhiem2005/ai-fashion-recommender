"use client";
import StoreHeader from "@/components/StoreHeader";
import Image from "next/image";
import Link from "next/link";
import { useEffect, useState } from "react";

import { formatPrice } from "@/data/products";
import {
  cartSubtotal,
  clearCart,
  readCart,
  removeFromCart,
  subscribeToCart,
  updateCartQuantity,
  type CartItem,
} from "@/lib/cart";

export default function CartPage() {
  const [items, setItems] = useState<CartItem[]>([]);
  const [isLoaded, setIsLoaded] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const refresh = () => {
      setItems(readCart());
      setIsLoaded(true);
    };

    // Read browser storage after mounting to avoid server/client mismatch.
    const frame = window.requestAnimationFrame(refresh);
    const unsubscribe = subscribeToCart(refresh);

    return () => {
      window.cancelAnimationFrame(frame);
      unsubscribe();
    };
  }, []);

  function performCartAction(action: () => void) {
    try {
      action();
      setError("");
    } catch {
      setError("Unable to update your shopping bag. Please try again.");
    }
  }

  const itemCount = items.reduce((count, item) => count + item.quantity, 0);
  const subtotalCents = cartSubtotal(items);

  return (
    <div className="min-h-screen bg-[#faf9f6] text-[#24221f]">
      <div className="bg-[#24221f] px-4 py-2 text-center text-xs tracking-wider text-white">
        ÉLAN Fashion Store — Portfolio Demo
      </div>

      <StoreHeader />

      <main className="mx-auto max-w-7xl px-6 py-12 md:py-16">
        <p className="text-xs font-semibold uppercase tracking-[0.25em] text-[#ae805f]">
          The ÉLAN Collection
        </p>
        <h1 className="mt-4 font-serif text-4xl md:text-5xl">
          Your Shopping Bag
        </h1>
        <p className="mt-3 text-sm text-[#77716b]">
          {isLoaded
            ? `${itemCount} ${itemCount === 1 ? "item" : "items"} in your bag`
            : "Loading your bag..."}
        </p>

        {error && (
          <p role="alert" className="mt-6 text-sm text-red-700">
            {error}
          </p>
        )}

        {!isLoaded ? (
          <p role="status" className="mt-12 text-sm text-[#77716b]">
            Loading saved products...
          </p>
        ) : items.length === 0 ? (
          <div className="mt-12 border border-[#e7e2dc] bg-white px-6 py-16 text-center">
            <h2 className="font-serif text-3xl">Your bag is empty</h2>
            <p className="mt-4 text-sm text-[#77716b]">
              Discover styles you love and add them to your bag.
            </p>
            <Link
              href="/shop"
              className="mt-8 inline-block bg-[#24221f] px-8 py-4 text-sm font-medium text-white hover:bg-[#ae805f]"
            >
              Explore Collection
            </Link>
          </div>
        ) : (
          <div className="mt-12 grid gap-10 lg:grid-cols-[minmax(0,1fr)_320px]">
            <section aria-label="Items in your shopping bag" className="space-y-6">
              {items.map((item) => (
                <article
                  key={item.sku}
                  className="flex gap-5 border-b border-[#e7e2dc] pb-6"
                >
                  <div className="relative h-36 w-28 shrink-0 overflow-hidden bg-[#eee8e0] sm:h-44 sm:w-32">
                    {item.imageUrl?.startsWith("/products/") ? (
                      <Image
                        src={item.imageUrl}
                        alt={item.name}
                        fill
                        sizes="(max-width: 640px) 112px, 128px"
                        className="object-cover"
                      />
                    ) : (
                      <div className="flex h-full items-center justify-center font-serif text-xl text-[#b5a79a]">
                        ÉLAN.
                      </div>
                    )}
                  </div>

                  <div className="min-w-0 flex-1">
                    <h2 className="font-serif text-xl sm:text-2xl">
                      {item.name}
                    </h2>
                    <p className="mt-2 text-sm text-[#817a72]">
                      {item.color} / Size {item.size}
                    </p>
                    <p className="mt-1 break-all text-xs text-[#817a72]">
                      SKU: {item.sku}
                    </p>
                    <p className="mt-3 text-sm font-semibold">
                      {formatPrice(item.priceCents / 100)} each
                    </p>

                    <div className="mt-4 flex flex-wrap items-center gap-4">
                      <div className="flex items-center border border-[#ddd6cf] bg-white">
                        <button
                          type="button"
                          aria-label={`Decrease quantity of ${item.name}`}
                          disabled={item.quantity <= 1}
                          onClick={() =>
                            performCartAction(() =>
                              updateCartQuantity(item.sku, item.quantity - 1)
                            )
                          }
                          className="px-3 py-2 text-sm disabled:cursor-not-allowed disabled:opacity-30"
                        >
                          −
                        </button>
                        <span className="min-w-8 text-center text-sm" aria-label="Quantity">
                          {item.quantity}
                        </span>
                        <button
                          type="button"
                          aria-label={`Increase quantity of ${item.name}`}
                          disabled={item.quantity >= 99}
                          onClick={() =>
                            performCartAction(() =>
                              updateCartQuantity(item.sku, item.quantity + 1)
                            )
                          }
                          className="px-3 py-2 text-sm disabled:cursor-not-allowed disabled:opacity-30"
                        >
                          +
                        </button>
                      </div>
                      <button
                        type="button"
                        onClick={() =>
                          performCartAction(() => removeFromCart(item.sku))
                        }
                        className="text-xs text-[#817a72] underline hover:text-red-700"
                      >
                        Remove
                      </button>
                    </div>
                    <p className="mt-4 text-sm font-semibold">
                      Item total: {formatPrice((item.priceCents * item.quantity) / 100)}
                    </p>
                  </div>
                </article>
              ))}
            </section>

            <aside className="h-fit border border-[#e7e2dc] bg-white p-6">
              <h2 className="font-serif text-2xl">Order Summary</h2>
              <div className="mt-6 flex justify-between border-b border-[#e7e2dc] pb-4 text-sm">
                <span>Items</span>
                <span>{itemCount}</span>
              </div>
              <div className="mt-5 flex justify-between text-lg font-semibold">
                <span>Subtotal</span>
                <span>{formatPrice(subtotalCents / 100)}</span>
              </div>
              <p className="mt-5 text-xs leading-6 text-[#817a72]">
                Portfolio demo only. No checkout, payments, real orders, or inventory reservations.
              </p>
              <Link
                href="/shop"
                className="mt-7 block bg-[#24221f] px-6 py-4 text-center text-sm font-medium text-white hover:bg-[#ae805f]"
              >
                Continue Shopping
              </Link>
              <button
                type="button"
                onClick={() => performCartAction(clearCart)}
                className="mt-4 w-full py-3 text-sm text-[#817a72] underline hover:text-[#24221f]"
              >
                Clear Bag
              </button>
            </aside>
          </div>
        )}
      </main>
    </div>
  );
}
