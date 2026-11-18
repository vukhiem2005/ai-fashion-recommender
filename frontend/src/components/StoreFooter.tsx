import Link from "next/link";

const collectionLinks = [
  { label: "All Collection", href: "/shop" },
  { label: "Style Inspirations", href: "/#collection" },
  { label: "Shopping Bag", href: "/cart" },
  { label: "Account", href: "/account" },
];

const brandLinks = [
  { label: "About ÉLAN", href: "/#about" },
  {
    label: "Project Documentation (GitHub)",
    href: "https://github.com/vukhiem2005/ai-fashion-recommender",
    external: true,
  },
];

export default function StoreFooter() {
  return (
    <footer className="mt-auto border-t border-[#e7e2dc] bg-[#24221f] text-[#f4efe9]">
      <div className="elan-container py-14 md:py-16">
        <div className="grid gap-10 md:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)_minmax(0,1fr)] lg:gap-16">
          <div className="max-w-sm">
            <Link
              href="/"
              className="inline-block font-serif text-3xl font-bold tracking-[0.24em] text-white transition-colors hover:text-[#d5b295]"
            >
              ÉLAN.
            </Link>
            <p className="mt-4 text-sm leading-7 text-[#c7bfb6]">
              Contemporary silhouettes and intelligent style discovery, curated
              with a minimalist editorial sensibility.
            </p>
          </div>

          <nav aria-label="Footer collection">
            <h2 className="text-xs font-semibold uppercase tracking-[0.22em] text-[#d5b295]">
              Collection
            </h2>
            <ul className="mt-4 space-y-2.5 p-0 list-none">
              {collectionLinks.map((item) => (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    className="inline-flex min-h-8 items-center text-sm text-[#ddd6ce] transition-colors hover:text-white"
                  >
                    {item.label}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>

          <nav aria-label="Footer brand and documentation">
            <h2 className="text-xs font-semibold uppercase tracking-[0.22em] text-[#d5b295]">
              House of ÉLAN
            </h2>
            <ul className="mt-4 space-y-2.5 p-0 list-none">
              {brandLinks.map((item) => (
                <li key={item.href}>
                  {item.external ? (
                    <a
                      href={item.href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex min-h-8 items-center gap-1.5 text-sm text-[#ddd6ce] transition-colors hover:text-white"
                    >
                      <span>{item.label}</span>
                      <span aria-hidden="true" className="text-xs text-[#b8afa6]">
                        ↗
                      </span>
                    </a>
                  ) : (
                    <Link
                      href={item.href}
                      className="inline-flex min-h-8 items-center text-sm text-[#ddd6ce] transition-colors hover:text-white"
                    >
                      {item.label}
                    </Link>
                  )}
                </li>
              ))}
            </ul>
          </nav>
        </div>

        <div className="mt-12 flex flex-col items-start justify-between gap-4 border-t border-[#383530] pt-6 text-xs text-[#a9a198] sm:flex-row sm:items-center">
          <p className="m-0">© 2026 ÉLAN. Contemporary Fashion Storefront.</p>
          <div className="flex flex-wrap items-center gap-6">
            <Link href="/shop" className="hover:text-white">
              Collection
            </Link>
            <Link href="/#about" className="hover:text-white">
              About
            </Link>
            <a
              href="https://github.com/vukhiem2005/ai-fashion-recommender"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-white"
            >
              GitHub
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
