
import Link from "next/link";
import Image from "next/image";
import StoreFooter from "@/components/StoreFooter";
import StoreHeader from "@/components/StoreHeader";
const products = [
  {
    id: "demo-001",
    name: "Modern City Style",
    category: "Women's Fashion",
    image:
      "https://images.unsplash.com/photo-1539109136881-3be0616acf4b?w=900&q=85",
  },
  {
    id: "demo-002",
    name: "Everyday Essentials",
    category: "Casual Collection",
    image:
      "https://images.unsplash.com/photo-1512436991641-6745cdb1723f?w=900&q=85",
  },
  {
    id: "demo-003",
    name: "Minimal Wardrobe",
    category: "Contemporary Fashion",
    image:
      "https://images.unsplash.com/photo-1490481651871-ab68de25d43d?w=900&q=85",
  },
  {
    id: "demo-004",
    name: "Urban Collection",
    category: "New Arrivals",
    image:
      "https://images.unsplash.com/photo-1529139574466-a303027c1d8b?w=900&q=85",
  },
];

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col bg-[#faf9f6] text-[#24221f]">
      <StoreHeader />
      <main id="main-content" className="flex-1">
        <section
          id="home"
          className="mx-auto grid max-w-7xl items-center gap-8 px-6 py-12 lg:grid-cols-2 lg:py-20"
        >
          <div className="max-w-lg">
            <p className="mb-5 text-xs font-semibold uppercase tracking-[0.3em] text-[#ad8266]">
              Modern Fashion · AI Discovery
            </p>

            <h1 className="font-serif text-5xl leading-tight md:text-7xl">
              Style that
              <br />
              speaks for
              <br />
              <span className="italic text-[#ae805f]">
                you.
              </span>
            </h1>

            <p className="mt-7 max-w-md leading-8 text-[#77716b]">
              Discover contemporary fashion through a
              smarter shopping experience. Explore
              curated styles and personalized
              recommendations.
            </p>

            <Link
              href="/shop"
              className="mt-9 inline-block bg-[#24221f] px-9 py-4 text-sm font-medium text-white transition hover:bg-[#ae805f]"
            >
              Explore Collection
            </Link>
          </div>

          <div className="relative h-[480px] overflow-hidden bg-[#e8ded3] md:h-[600px]">
            <Image
              src="https://images.unsplash.com/photo-1483985988355-763728e1935b?w=1400&q=85"
              alt="Fashion editorial collection"
              fill
              sizes="(max-width: 1024px) 100vw, 50vw"
              className="object-cover"
            />

            <div className="absolute bottom-6 left-6 bg-white/90 px-5 py-3 text-xs uppercase tracking-[0.2em]">
              The Modern Edit
            </div>
          </div>
        </section>

        <section
          id="collection"
          className="mx-auto max-w-7xl px-6 py-16"
        >
          <div className="mb-10 flex flex-wrap items-end justify-between gap-4">
            <div>
              <p className="mb-3 text-xs font-semibold uppercase tracking-[0.25em] text-[#ae805f]">
                Curated Collection
              </p>

              <h2 className="font-serif text-4xl">
              Explore Style Inspirations
              </h2>
            </div>

            <span className="text-sm text-[#817a72]">
  Curated Fashion Editorial · Style Inspiration
</span>
          </div>

          <div className="grid grid-cols-1 gap-7 sm:grid-cols-2 lg:grid-cols-4">
            {products.map((product) => (
              <article
                key={product.id}
                className="group"
              >
                <div className="relative aspect-[3/4] overflow-hidden bg-[#e9e5df]">
                  <Image
                    src={product.image}
                    alt={`Editorial image for ${product.name}`}
                    fill
                    sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 25vw"
                    className="object-cover transition duration-500 group-hover:scale-105"
                  />

                  <span className="absolute left-4 top-4 bg-white px-3 py-2 text-[10px] font-semibold tracking-wider">
                    DEMO
                  </span>
                </div>

                <div className="mt-5">
                  <p className="text-xs uppercase tracking-wider text-[#9b9289]">
                    {product.category}
                  </p>

                  <h3 className="mt-2 text-lg font-medium">
                    {product.name}
                  </h3>

                  <Link
  href="/shop"
  className="mt-3 inline-block text-sm font-medium text-[#ae805f] underline hover:text-[#24221f]"
>
  Explore the Collection
</Link>
                </div>
              </article>
            ))}
          </div>
        </section>
<section
  id="about"
  className="bg-[#eee8e0] px-6 py-20"
>
  <div className="mx-auto max-w-6xl">
    <div className="mx-auto max-w-2xl text-center">
      <p className="text-xs font-semibold uppercase tracking-[0.25em] text-[#a27b60]">
        Intelligent Fashion Discovery
      </p>

      <h2 className="mt-5 font-serif text-4xl leading-tight md:text-5xl">
        Fashion discovery, powered by AI.
      </h2>

      <p className="mt-6 text-sm leading-7 text-[#706961]">
        Explore fashion through content-based recommendations.
        ÉLAN compares product descriptions, categories and colors
        using TF-IDF and cosine similarity to discover related styles.
      </p>
    </div>

    <div className="mt-12 grid gap-5 md:grid-cols-3">
      <div className="bg-white/75 p-7">
        <p className="text-xs font-semibold uppercase tracking-widest text-[#ae805f]">
          01 / Explore
        </p>
        <h3 className="mt-4 font-serif text-2xl">
          Discover the Collection
        </h3>
        <p className="mt-3 text-sm leading-7 text-[#706961]">
          Browse fashion products, explore categories,
          and find styles that match your interests.
        </p>
      </div>

      <div className="bg-white/75 p-7">
        <p className="text-xs font-semibold uppercase tracking-widest text-[#ae805f]">
          02 / Intelligence
        </p>
        <h3 className="mt-4 font-serif text-2xl">
          AI Similarity Matching
        </h3>
        <p className="mt-3 text-sm leading-7 text-[#706961]">
          Our recommendation engine compares product
          attributes to rank similar items.
        </p>
      </div>

      <div className="bg-white/75 p-7">
        <p className="text-xs font-semibold uppercase tracking-widest text-[#ae805f]">
          03 / Recommendation
        </p>
        <h3 className="mt-4 font-serif text-2xl">
          You May Also Like
        </h3>
        <p className="mt-3 text-sm leading-7 text-[#706961]">
          View five recommended products and understand
          the similarities behind each suggestion.
        </p>
      </div>
    </div>
  </div>
</section>

      </main>

      <StoreFooter />
    </div>
  );
}
