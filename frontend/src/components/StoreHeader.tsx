"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useId, useRef, useState, type MouseEvent } from "react";
import { readCart, subscribeToCart } from "@/lib/cart";

const primaryNavigation = [
  { label: "Home", href: "/" },
  { label: "Collection", href: "/shop" },
  { label: "About", href: "/#about" },
];

const utilityNavigation = [
  { label: "Account", href: "/account" },
  { label: "Shopping Bag", href: "/cart" },
];

const allNavigation = [...primaryNavigation, ...utilityNavigation];

export default function StoreHeader() {
  const pathname = usePathname();
  const drawerId = useId();
  const dialogRef = useRef<HTMLDialogElement | null>(null);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [bagCount, setBagCount] = useState(0);

  useEffect(() => {
    const syncBagCount = () => {
      try {
        const items = readCart();
        setBagCount(items.reduce((sum, item) => sum + item.quantity, 0));
      } catch {
        setBagCount(0);
      }
    };

    const frame = window.requestAnimationFrame(syncBagCount);
    const unsubscribe = subscribeToCart(syncBagCount);

    return () => {
      window.cancelAnimationFrame(frame);
      unsubscribe();
    };
  }, []);

  function isActive(href: string): boolean {
    if (href === "/#about") {
      return false;
    }

    if (href === "/shop") {
      return pathname === "/shop" || pathname.startsWith("/shop/");
    }

    if (href === "/account") {
      return pathname === "/account" || pathname.startsWith("/admin");
    }

    return pathname === href;
  }

  function openMobileMenu() {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (!dialog.open) {
      dialog.showModal();
    }
    setIsMenuOpen(true);
  }

  function closeMobileMenu() {
    const dialog = dialogRef.current;
    if (dialog?.open) {
      dialog.close();
    }
    setIsMenuOpen(false);
  }

  useEffect(() => {
    const dialog = dialogRef.current;
    if (dialog?.open) {
      dialog.close();
    }
    const frame = window.requestAnimationFrame(() => {
      setIsMenuOpen(false);
    });
    return () => window.cancelAnimationFrame(frame);
  }, [pathname]);

  useEffect(() => {
    const mql = window.matchMedia("(min-width: 768px)");
    const handleViewportChange = (event: MediaQueryListEvent) => {
      if (event.matches) {
        const dialog = dialogRef.current;
        if (dialog?.open) {
          dialog.close();
        }
        setIsMenuOpen(false);
      }
    };

    mql.addEventListener("change", handleViewportChange);
    return () => mql.removeEventListener("change", handleViewportChange);
  }, []);

  function handleDialogBackdropClick(event: MouseEvent<HTMLDialogElement>) {
    if (event.target === dialogRef.current) {
      closeMobileMenu();
    }
  }

  return (
    <header className="sticky top-0 z-40 border-b border-[#e7e2dc] bg-[#faf9f6]/95 backdrop-blur-sm">
      <a href="#main-content" className="elan-skip-link">
        Skip to main content
      </a>

      <div className="elan-container flex min-h-18 items-center justify-between gap-4 py-3.5 md:min-h-20 md:py-4">
        <Link
          href="/"
          className="font-serif text-2xl font-bold tracking-[0.24em] text-[#24221f] transition-colors hover:text-[#9a6b48] sm:text-3xl"
        >
          ÉLAN.
        </Link>

        {/* Desktop Primary & Utility Navigation */}
        <nav
          aria-label="Primary"
          className="hidden md:flex md:items-center md:gap-8 lg:gap-10"
        >
          <ul className="m-0 flex list-none items-center gap-7 p-0 lg:gap-9">
            {primaryNavigation.map((item) => {
              const active = isActive(item.href);
              return (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    aria-current={active ? "page" : undefined}
                    className={`inline-flex min-h-11 items-center border-b-2 py-1 text-xs font-medium uppercase tracking-[0.18em] whitespace-nowrap transition-colors ${
                      active
                        ? "border-[#9a6b48] font-semibold text-[#24221f]"
                        : "border-transparent text-[#686159] hover:border-[#d5cdc3] hover:text-[#24221f]"
                    }`}
                  >
                    {item.label}
                  </Link>
                </li>
              );
            })}
          </ul>

          <span
            aria-hidden="true"
            className="h-4 w-px bg-[#dfd8cf]"
          />

          <ul className="m-0 flex list-none items-center gap-6 p-0 lg:gap-8">
            {utilityNavigation.map((item) => {
              const active = isActive(item.href);
              const isBag = item.href === "/cart";
              return (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    aria-current={active ? "page" : undefined}
                    className={`inline-flex min-h-11 items-center gap-2 border-b-2 py-1 text-xs font-medium uppercase tracking-[0.18em] whitespace-nowrap transition-colors ${
                      active
                        ? "border-[#9a6b48] font-semibold text-[#24221f]"
                        : "border-transparent text-[#686159] hover:border-[#d5cdc3] hover:text-[#24221f]"
                    }`}
                  >
                    {isBag && (
                      <svg
                        aria-hidden="true"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.6"
                        className="h-4 w-4 shrink-0"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M6 8h12l-1 12H7L6 8Z"
                        />
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M9 8V6a3 3 0 0 1 6 0v2"
                        />
                      </svg>
                    )}
                    <span>{item.label}</span>
                    {isBag && (
                      <span className="inline-flex min-w-5 items-center justify-center rounded-full bg-[#24221f] px-1.5 py-0.5 text-[11px] font-semibold tracking-normal text-white">
                        {bagCount}
                      </span>
                    )}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>

        {/* Mobile Compact Actions: Direct Shopping Bag Entry + Menu Drawer Trigger */}
        <div className="flex items-center gap-2.5 md:hidden">
          <Link
            href="/cart"
            aria-current={isActive("/cart") ? "page" : undefined}
            className={`inline-flex min-h-11 items-center gap-1.5 border px-3 py-2 text-xs font-medium uppercase tracking-[0.14em] whitespace-nowrap transition-colors ${
              isActive("/cart")
                ? "border-[#9a6b48] bg-white font-semibold text-[#24221f]"
                : "border-[#e7e2dc] bg-white/80 text-[#24221f] hover:border-[#9a6b48]"
            }`}
          >
            <svg
              aria-hidden="true"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
              className="h-4 w-4 shrink-0"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M6 8h12l-1 12H7L6 8Z"
              />
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M9 8V6a3 3 0 0 1 6 0v2"
              />
            </svg>
            <span>Shopping Bag</span>
            <span className="inline-flex min-w-5 items-center justify-center rounded-full bg-[#24221f] px-1.5 py-0.5 text-[11px] font-semibold tracking-normal text-white">
              {bagCount}
            </span>
          </Link>

          <button
            type="button"
            aria-expanded={isMenuOpen}
            aria-controls={drawerId}
            onClick={openMobileMenu}
            className="inline-flex min-h-11 items-center gap-2 border border-[#d5cdc3] bg-white px-3.5 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-[#24221f] transition-colors hover:border-[#9a6b48]"
          >
            <svg
              aria-hidden="true"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
              className="h-4 w-4 shrink-0"
            >
              <path
                strokeLinecap="round"
                d="M4 7h16M4 12h16M4 17h16"
              />
            </svg>
            <span>Menu</span>
          </button>
        </div>
      </div>

      {/* Mobile Navigation Modal Dialog */}
      <dialog
        ref={dialogRef}
        id={drawerId}
        aria-label="Site navigation"
        {...({ closedby: "any" } as Record<string, string>)}
        onClose={() => setIsMenuOpen(false)}
        onClick={handleDialogBackdropClick}
        className="elan-mobile-drawer md:hidden"
      >
        <div className="elan-mobile-drawer-sheet">
          <div className="flex items-center justify-between border-b border-[#e7e2dc] px-6 py-4">
            <span className="font-serif text-2xl font-bold tracking-[0.22em] text-[#24221f]">
              ÉLAN.
            </span>

            <button
              type="button"
              onClick={closeMobileMenu}
              className="inline-flex min-h-11 items-center gap-1.5 border border-[#d5cdc3] bg-white px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-[#24221f] transition-colors hover:border-[#9a6b48]"
            >
              <svg
                aria-hidden="true"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                className="h-4 w-4 shrink-0"
              >
                <path
                  strokeLinecap="round"
                  d="M6 6l12 12M18 6L6 18"
                />
              </svg>
              <span>Close</span>
            </button>
          </div>

          <nav aria-label="Mobile primary" className="flex-1 px-6 py-6">
            <ul className="m-0 flex list-none flex-col divide-y divide-[#e7e2dc] p-0">
              {allNavigation.map((item) => {
                const active = isActive(item.href);
                const isBag = item.href === "/cart";
                return (
                  <li key={item.href}>
                    <Link
                      href={item.href}
                      onClick={closeMobileMenu}
                      aria-current={active ? "page" : undefined}
                      className={`flex min-h-13 items-center justify-between py-3 text-sm tracking-[0.16em] uppercase transition-colors ${
                        active
                          ? "border-s-2 border-[#9a6b48] ps-3 font-semibold text-[#24221f]"
                          : "text-[#686159] hover:text-[#24221f]"
                      }`}
                    >
                      <span>{item.label}</span>
                      {isBag && (
                        <span className="inline-flex min-w-6 items-center justify-center rounded-full bg-[#24221f] px-2 py-0.5 text-xs font-semibold tracking-normal text-white">
                          {bagCount}
                        </span>
                      )}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </nav>

          <div className="border-t border-[#e7e2dc] bg-[#f2ece4]/60 px-6 py-5">
            <Link
              href="/shop"
              onClick={closeMobileMenu}
              className="elan-btn-primary w-full text-center"
            >
              Explore Collection
            </Link>
          </div>
        </div>
      </dialog>
    </header>
  );
}
