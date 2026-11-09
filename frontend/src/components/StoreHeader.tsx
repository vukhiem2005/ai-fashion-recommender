
"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navigation = [
  { label: "Home", href: "/" },
  { label: "Collection", href: "/shop" },
  { label: "About", href: "/#about" },
  { label: "Account", href: "/account" },
  { label: "Shopping Bag", href: "/cart" },
];

export default function StoreHeader() {
  const pathname = usePathname();

  function isActive(href: string): boolean {
    if (href === "/#about") {
      return false;
    }

    if (href === "/shop") {
      return pathname === "/shop" ||
        pathname.startsWith("/shop/");
    }

    return pathname === href;
  }

  return (
    <header className="border-b border-[#e9e6e1] bg-white">
      <nav className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-5 px-6 py-6">
        <Link
          href="/"
          className="font-serif text-3xl font-bold tracking-widest"
        >
          ÉLAN.
        </Link>

        <div className="flex flex-wrap items-center justify-end gap-x-6 gap-y-3 text-sm">
          {navigation.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              aria-current={
                isActive(item.href) ? "page" : undefined
              }
              className={
                isActive(item.href)
                  ? "font-semibold text-[#ae805f]"
                  : "hover:text-[#ae805f]"
              }
            >
              {item.label}
            </Link>
          ))}
        </div>
      </nav>
    </header>
  );
}
