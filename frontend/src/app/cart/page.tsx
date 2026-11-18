"use client";

import StoreFooter from "@/components/StoreFooter";
import StoreHeader from "@/components/StoreHeader";
import Image from "next/image";
import Link from "next/link";
import { useEffect, useId, useRef, useState } from "react";

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

function isMeaningfulOption(value: string): boolean {
  const normalized = value.trim().toLowerCase();
  return normalized !== "" && normalized !== "not specified";
}

function formatVariantAttributes(color: string, size: string): string {
  const parts: string[] = [];
  if (isMeaningfulOption(color)) {
    parts.push(color.trim());
  }
  if (isMeaningfulOption(size)) {
    parts.push(`Size ${size.trim()}`);
  }
  return parts.join(" · ");
}

export default function CartPage() {
  const confirmDialogTitleId = useId();
  const confirmDialogDescId = useId();
  const confirmDialogRef = useRef<HTMLDialogElement | null>(null);
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

  function openClearConfirmation() {
    const dialog = confirmDialogRef.current;
    if (!dialog) return;
    dialog.returnValue = "";
    if (!dialog.open) {
      dialog.showModal();
    }
  }

  function handleConfirmDialogClose() {
    const dialog = confirmDialogRef.current;
    if (!dialog) return;
    if (dialog.returnValue === "clear") {
      dialog.returnValue = "";
      performCartAction(clearCart);
    } else {
      dialog.returnValue = "";
    }
  }

  const itemCount = items.reduce((count, item) => count + item.quantity, 0);
  const subtotalCents = cartSubtotal(items);

  return (
    <div className="flex min-h-screen flex-col bg-[#faf9f6] text-[#24221f]">
      <StoreHeader />

      <main id="main-content" className="elan-container flex-1 py-10 md:py-16">
        <div className="border-b border-[#e7e2dc] pb-6">
          <h1 className="font-serif text-4xl sm:text-5xl">
            Your Shopping Bag
          </h1>
          <p className="mt-2.5 text-sm text-[#686159]">
            {isLoaded
              ? `${itemCount} ${itemCount === 1 ? "piece" : "pieces"} in your bag`
              : "Loading your bag..."}
          </p>
        </div>

        {error && (
          <p
            role="alert"
            className="mt-6 border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
          >
            {error}
          </p>
        )}

        {!isLoaded ? (
          <p role="status" className="py-16 text-sm text-[#686159]">
            Loading saved pieces...
          </p>
        ) : items.length === 0 ? (
          <div className="mt-10 border border-[#e7e2dc] bg-white px-6 py-16 text-center sm:py-20">
            <h2 className="font-serif text-3xl sm:text-4xl">
              Your Bag Is Empty
            </h2>
            <p className="mx-auto mt-3 max-w-md text-sm leading-7 text-[#686159]">
              Find something you&apos;ll love in the collection.
            </p>
            <Link href="/shop" className="elan-btn-primary mt-8">
              Explore Collection
            </Link>
          </div>
        ) : (
          <div className="mt-10 grid items-start gap-10 lg:grid-cols-[minmax(0,1fr)_360px] lg:gap-14">
            {/* Left Column: Product List */}
            <section
              aria-label="Items in your shopping bag"
              className="divide-y divide-[#e7e2dc] border-b border-[#e7e2dc]"
            >
              {items.map((item) => {
                const variantAttributes = formatVariantAttributes(
                  item.color,
                  item.size
                );
                const lineTotalCents = item.priceCents * item.quantity;

                return (
                  <article
                    key={item.sku}
                    className="flex gap-4 py-6 first:pt-0 sm:gap-6 sm:py-7"
                  >
                    <div className="relative aspect-[3/4] w-24 shrink-0 overflow-hidden border border-[#e7e2dc] bg-[#eee8e0] sm:w-32">
                      {item.imageUrl?.startsWith("/products/") ? (
                        <Image
                          src={item.imageUrl}
                          alt={item.name}
                          fill
                          sizes="(max-width: 640px) 96px, 128px"
                          className="object-cover object-top"
                        />
                      ) : (
                        <div className="flex h-full items-center justify-center font-serif text-xl text-[#b5a79a]">
                          ÉLAN.
                        </div>
                      )}
                    </div>

                    <div className="flex min-w-0 flex-1 flex-col justify-between">
                      <div className="flex flex-col justify-between gap-2 sm:flex-row sm:items-start">
                        <div className="min-w-0">
                          <h2 className="font-serif text-xl leading-snug text-[#24221f] sm:text-2xl">
                            {item.name}
                          </h2>

                          {variantAttributes && (
                            <p className="mt-1.5 text-xs font-medium uppercase tracking-[0.14em] text-[#686159]">
                              {variantAttributes}
                            </p>
                          )}

                          <p className="mt-2 text-xs text-[#756d65]">
                            {formatPrice(item.priceCents / 100)} each
                          </p>
                        </div>

                        <p className="font-serif text-lg font-medium text-[#24221f] sm:text-xl">
                          {formatPrice(lineTotalCents / 100)}
                        </p>
                      </div>

                      <div className="mt-5 flex flex-wrap items-center justify-between gap-4">
                        <div
                          role="group"
                          aria-label={`Quantity for ${item.name}`}
                          className="inline-flex items-center border border-[#d5cdc3] bg-white"
                        >
                          <button
                            type="button"
                            aria-label={`Decrease quantity of ${item.name}`}
                            disabled={item.quantity <= 1}
                            onClick={() =>
                              performCartAction(() =>
                                updateCartQuantity(item.sku, item.quantity - 1)
                              )
                            }
                            className="inline-flex min-h-10 min-w-10 items-center justify-center text-sm text-[#24221f] transition-colors hover:bg-[#f4efe8] disabled:cursor-not-allowed disabled:opacity-35"
                          >
                            −
                          </button>

                          <span
                            className="min-w-9 px-2 text-center text-sm font-medium text-[#24221f]"
                            aria-label="Quantity"
                          >
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
                            className="inline-flex min-h-10 min-w-10 items-center justify-center text-sm text-[#24221f] transition-colors hover:bg-[#f4efe8] disabled:cursor-not-allowed disabled:opacity-35"
                          >
                            +
                          </button>
                        </div>

                        <button
                          type="button"
                          onClick={() =>
                            performCartAction(() => removeFromCart(item.sku))
                          }
                          aria-label={`Remove ${item.name} from bag`}
                          className="inline-flex min-h-10 items-center px-2 text-xs font-medium uppercase tracking-[0.14em] text-[#686159] underline underline-offset-4 transition-colors hover:text-red-700"
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                  </article>
                );
              })}
            </section>

            {/* Right Column: Order Summary */}
            <aside
              aria-labelledby="order-summary-heading"
              className="border border-[#e7e2dc] bg-white p-6 sm:p-8 lg:sticky lg:top-28"
            >
              <h2
                id="order-summary-heading"
                className="font-serif text-2xl sm:text-3xl"
              >
                Order Summary
              </h2>

              <dl className="mt-6 space-y-4 border-b border-[#e7e2dc] pb-5 text-sm">
                <div className="flex items-center justify-between text-[#686159]">
                  <dt>Pieces</dt>
                  <dd className="font-medium text-[#24221f]">{itemCount}</dd>
                </div>
              </dl>

              <div className="mt-5 flex items-baseline justify-between">
                <span className="text-xs font-semibold uppercase tracking-[0.18em] text-[#686159]">
                  Subtotal
                </span>
                <span className="font-serif text-3xl font-medium text-[#24221f]">
                  {formatPrice(subtotalCents / 100)}
                </span>
              </div>

              <div className="mt-7 flex flex-col gap-3">
                <Link
                  href="/shop"
                  className="elan-btn-primary w-full text-center"
                >
                  Continue Shopping
                </Link>

                <button
                  type="button"
                  onClick={openClearConfirmation}
                  className="elan-btn-secondary w-full"
                >
                  Clear Bag
                </button>
              </div>

              <p className="mt-6 border-t border-[#e7e2dc] pt-5 text-xs leading-6 text-[#756d65]">
                Your shopping bag is saved in this browser. Checkout and real
                payments are not active in this simulation.
              </p>
            </aside>
          </div>
        )}

        {/* Native Modal Confirmation Dialog for Clear Bag */}
        <dialog
          ref={confirmDialogRef}
          aria-labelledby={confirmDialogTitleId}
          aria-describedby={confirmDialogDescId}
          onClose={handleConfirmDialogClose}
          className="elan-confirm-dialog"
        >
          <h2
            id={confirmDialogTitleId}
            className="font-serif text-2xl text-[#24221f]"
          >
            Clear your shopping bag?
          </h2>
          <p
            id={confirmDialogDescId}
            className="mt-3 text-sm leading-6 text-[#686159]"
          >
            This will remove all {itemCount}{" "}
            {itemCount === 1 ? "piece" : "pieces"} currently saved in your
            shopping bag.
          </p>
          <form
            method="dialog"
            className="mt-7 flex flex-wrap items-center justify-end gap-3"
          >
            <button
              type="submit"
              value="cancel"
              autoFocus
              className="elan-btn-secondary"
            >
              Keep Items
            </button>
            <button
              type="submit"
              value="clear"
              className="elan-btn-primary"
            >
              Clear Bag
            </button>
          </form>
        </dialog>
      </main>

      <StoreFooter />
    </div>
  );
}
