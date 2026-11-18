import Link from "next/link";
import Image from "next/image";
import StoreFooter from "@/components/StoreFooter";
import StoreHeader from "@/components/StoreHeader";

const editorialLooks = [
  {
    id: "demo-001",
    name: "Modern City Style",
    category: "Women's Fashion",
    note: "Tailored outerwear and structured city silhouettes.",
    href: "/shop",
    cta: "Shop Tailored Edit",
    image:
      "https://images.unsplash.com/photo-1539109136881-3be0616acf4b?w=900&q=85",
  },
  {
    id: "demo-002",
    name: "Everyday Essentials",
    category: "Casual Collection",
    note: "Relaxed staples crafted for effortless daily layering.",
    href: "/shop",
    cta: "Shop Essentials",
    image:
      "https://images.unsplash.com/photo-1512436991641-6745cdb1723f?w=900&q=85",
  },
  {
    id: "demo-003",
    name: "Minimal Wardrobe",
    category: "Contemporary Fashion",
    note: "Refined monochromatic separates and understated knitwear.",
    href: "/shop#personalized-demo-heading",
    cta: "Explore Curated Edit",
    image:
      "https://images.unsplash.com/photo-1490481651871-ab68de25d43d?w=900&q=85",
  },
  {
    id: "demo-004",
    name: "Urban Collection",
    category: "New Arrivals",
    note: "Contemporary statement pieces with modern proportions.",
    href: "/shop",
    cta: "Browse New Arrivals",
    image:
      "https://images.unsplash.com/photo-1529139574466-a303027c1d8b?w=900&q=85",
  },
];

const styleHighlights = [
  {
    eyebrow: "01 · The Collection",
    title: "328 Curated Pieces",
    description:
      "Browse outerwear, tops, bottoms, dresses, knitwear, and accessories with instant search, category filtering, and price sorting.",
    href: "/shop",
    cta: "Browse All Pieces",
  },
  {
    eyebrow: "02 · Visual Harmony",
    title: "Complete the Look",
    description:
      "Every product page surfaces five visually and textually harmonious pieces so you can compare silhouettes, palettes, and textures side by side.",
    href: "/shop/classic-beige-blazer",
    cta: "View Featured Piece",
  },
  {
    eyebrow: "03 · Tailored Edits",
    title: "Personalized Style Profiles",
    description:
      "Preview how curated shopper profiles shape individualized wardrobe recommendations across casual, tailored, and seasonal edits.",
    href: "/shop#personalized-demo-heading",
    cta: "Try Style Profiles",
  },
];

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col bg-[#faf9f6] text-[#24221f]">
      <StoreHeader />

      <main id="main-content" className="flex-1">
        {/* Editorial Hero */}
        <section
          id="home"
          className="border-b border-[#e7e2dc] bg-gradient-to-b from-[#faf9f6] to-[#f5f0e8]"
        >
          <div className="elan-container grid items-center gap-10 py-12 md:py-16 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,0.95fr)] lg:gap-16 lg:py-22">
            <div className="max-w-xl">
              <p className="text-xs font-semibold uppercase tracking-[0.28em] text-[#9a6b48]">
                Contemporary Wardrobe · Seasonal Edit
              </p>

              <h1 className="mt-5 font-serif text-5xl leading-[1.06] tracking-tight sm:text-6xl lg:text-7xl">
                Style that
                <br />
                speaks for
                <br />
                <span className="italic font-normal text-[#9a6b48]">you.</span>
              </h1>

              <p className="mt-6 max-w-md text-base leading-8 text-[#686159]">
                Timeless silhouettes, tactile neutrals, and effortless wardrobe
                essentials curated for modern dressing. Explore the collection
                and discover pieces tailored to your aesthetic.
              </p>

              <div className="mt-9 flex flex-wrap items-center gap-4">
                <Link href="/shop" className="elan-btn-primary">
                  Explore Collection
                </Link>
                <Link href="/about" className="elan-btn-secondary">
                  House of ÉLAN
                </Link>
              </div>

              <dl className="mt-12 grid grid-cols-3 gap-6 border-t border-[#e2dbd2] pt-7">
                <div>
                  <dt className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[#756d65]">
                    Categories
                  </dt>
                  <dd className="mt-1.5 font-serif text-2xl text-[#24221f]">
                    6 Edits
                  </dd>
                </div>
                <div>
                  <dt className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[#756d65]">
                    Catalog
                  </dt>
                  <dd className="mt-1.5 font-serif text-2xl text-[#24221f]">
                    328 Pieces
                  </dd>
                </div>
                <div>
                  <dt className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[#756d65]">
                    Discovery
                  </dt>
                  <dd className="mt-1.5 font-serif text-2xl text-[#24221f]">
                    Curated Top-5
                  </dd>
                </div>
              </dl>
            </div>

            <div className="relative aspect-[4/5] w-full overflow-hidden border border-[#e7e2dc] bg-[#e8ded3] sm:aspect-[16/11] lg:aspect-[4/5]">
              <Image
                src="https://images.unsplash.com/photo-1483985988355-763728e1935b?w=1400&q=85"
                alt="Fashion editorial collection"
                fill
                priority
                fetchPriority="high"
                sizes="(max-width: 1024px) 100vw, 50vw"
                className="object-cover"
              />

              <div className="absolute inset-x-4 bottom-4 flex items-center justify-between border border-[#e7e2dc]/80 bg-[#faf9f6]/95 px-5 py-3.5 backdrop-blur-sm sm:inset-x-6 sm:bottom-6">
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-[0.24em] text-[#9a6b48]">
                    Featured Lookbook
                  </p>
                  <p className="mt-0.5 font-serif text-lg text-[#24221f]">
                    The Modern Edit
                  </p>
                </div>
                <Link
                  href="/shop"
                  className="text-xs font-semibold uppercase tracking-[0.18em] text-[#24221f] underline decoration-[#9a6b48] underline-offset-4 transition-colors hover:text-[#9a6b48]"
                >
                  Shop Now
                </Link>
              </div>
            </div>
          </div>
        </section>

        {/* Editorial Style Inspirations */}
        <section id="collection" className="elan-container py-16 md:py-24">
          <div className="mb-12 flex flex-col items-start justify-between gap-4 border-b border-[#e7e2dc] pb-7 sm:flex-row sm:items-end">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.26em] text-[#9a6b48]">
                Curated Collection
              </p>
              <h2 className="mt-3 font-serif text-3xl sm:text-4xl md:text-5xl">
                Explore Style Inspirations
              </h2>
            </div>

            <Link
              href="/shop"
              className="inline-flex min-h-11 items-center gap-2 text-xs font-semibold uppercase tracking-[0.2em] text-[#24221f] transition-colors hover:text-[#9a6b48]"
            >
              <span>View Full Collection</span>
              <span aria-hidden="true">→</span>
            </Link>
          </div>

          <div className="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-4">
            {editorialLooks.map((look) => (
              <article key={look.id} className="group flex flex-col">
                <Link
                  href={look.href}
                  className="relative aspect-[3/4] overflow-hidden border border-[#e7e2dc] bg-[#e9e5df]"
                >
                  <Image
                    src={look.image}
                    alt={`Editorial look for ${look.name}`}
                    fill
                    sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 25vw"
                    className="object-cover transition duration-500 group-hover:scale-105"
                  />
                </Link>

                <div className="mt-5 flex flex-1 flex-col justify-between">
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[#9a6b48]">
                      {look.category}
                    </p>
                    <h3 className="mt-2 font-serif text-2xl text-[#24221f]">
                      <Link
                        href={look.href}
                        className="transition-colors group-hover:text-[#9a6b48]"
                      >
                        {look.name}
                      </Link>
                    </h3>
                    <p className="mt-2 text-sm leading-6 text-[#686159]">
                      {look.note}
                    </p>
                  </div>

                  <div className="mt-4 pt-2">
                    <Link
                      href={look.href}
                      className="inline-flex min-h-9 items-center gap-1.5 border-b border-[#24221f] pb-1 text-xs font-semibold uppercase tracking-[0.16em] text-[#24221f] transition-colors hover:border-[#9a6b48] hover:text-[#9a6b48]"
                    >
                      <span>{look.cta}</span>
                      <span aria-hidden="true">→</span>
                    </Link>
                  </div>
                </div>
              </article>
            ))}
          </div>
        </section>

        {/* Discover Your Style — Concise Shopping Experience Section */}
        <section className="border-t border-[#e7e2dc] bg-[#f2ece4] py-16 md:py-24">
          <div className="elan-container">
            <div className="mx-auto max-w-2xl text-center">
              <p className="text-xs font-semibold uppercase tracking-[0.26em] text-[#9a6b48]">
                Effortless Wardrobe Curation
              </p>
              <h2 className="mt-4 font-serif text-3xl leading-tight sm:text-4xl md:text-5xl">
                Discover Your Style
              </h2>
              <p className="mt-4 text-sm leading-7 text-[#686159] sm:text-base">
                From tailored outerwear to everyday knitwear, move seamlessly
                between curated catalog browsing, visual piece matching, and
                personalized style edits.
              </p>
            </div>

            <div className="mt-12 grid gap-6 md:grid-cols-3">
              {styleHighlights.map((item) => (
                <div
                  key={item.title}
                  className="flex flex-col justify-between border border-[#e2dbd2] bg-[#faf9f6] p-7 sm:p-8"
                >
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[#9a6b48]">
                      {item.eyebrow}
                    </p>
                    <h3 className="mt-3 font-serif text-2xl text-[#24221f]">
                      {item.title}
                    </h3>
                    <p className="mt-3 text-sm leading-7 text-[#686159]">
                      {item.description}
                    </p>
                  </div>

                  <div className="mt-7 pt-2">
                    <Link
                      href={item.href}
                      className="inline-flex min-h-10 items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-[#24221f] underline decoration-[#9a6b48] underline-offset-4 transition-colors hover:text-[#9a6b48]"
                    >
                      <span>{item.cta}</span>
                      <span aria-hidden="true">→</span>
                    </Link>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-12 flex flex-wrap items-center justify-center gap-4">
              <Link href="/shop" className="elan-btn-primary">
                Shop the Collection
              </Link>
              <Link
                href="/shop#personalized-demo-heading"
                className="elan-btn-secondary"
              >
                Explore Personalized Edits
              </Link>
              <Link
                href="/about"
                className="inline-flex min-h-11 items-center px-4 text-xs font-semibold uppercase tracking-[0.18em] text-[#686159] underline underline-offset-4 hover:text-[#24221f]"
              >
                Read About ÉLAN &amp; Research
              </Link>
            </div>
          </div>
        </section>
      </main>

      <StoreFooter />
    </div>
  );
}
