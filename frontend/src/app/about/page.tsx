import type { Metadata } from "next";
import Link from "next/link";
import StoreFooter from "@/components/StoreFooter";
import StoreHeader from "@/components/StoreHeader";

export const metadata: Metadata = {
  title: "About ÉLAN | Contemporary Fashion & AI Discovery",
  description:
    "Learn about ÉLAN, an academic AI-powered fashion e-commerce simulation combining editorial design with intelligent style discovery.",
};

const technologies = [
  {
    label: "AI Recommendations",
    value: "CLIP Late Fusion, ItemKNN, TF-IDF + Cosine Similarity",
  },
  {
    label: "Frontend",
    value: "Next.js, React, TypeScript",
  },
  {
    label: "Backend",
    value: "Python, FastAPI",
  },
  {
    label: "Database",
    value: "SQLite",
  },
  {
    label: "Deployment",
    value: "Vercel and Render",
  },
];

export default function AboutPage() {
  return (
    <div className="flex min-h-screen flex-col bg-[#faf9f6] text-[#24221f]">
      <StoreHeader />

      <main id="main-content" className="flex-1">
        {/* Section 1 — About ÉLAN */}
        <section className="border-b border-[#e7e2dc] bg-gradient-to-b from-[#faf9f6] to-[#f4efe8] py-14 md:py-20">
          <div className="elan-container max-w-4xl">
            <p className="text-xs font-semibold uppercase tracking-[0.26em] text-[#9a6b48]">
              About ÉLAN
            </p>
            <h1 className="mt-3 font-serif text-4xl leading-tight sm:text-5xl md:text-6xl">
              A more thoughtful way to discover style.
            </h1>
            <p className="mt-5 max-w-2xl text-base leading-8 text-[#686159]">
              ÉLAN is an academic AI-powered fashion e-commerce simulation
              designed around a calm, editorial shopping experience. Visitors
              can browse curated seasonal pieces across outerwear, tops,
              bottoms, dresses, knitwear, and accessories while exploring how
              modern recommendation systems surface complementary silhouettes
              and personalized style edits.
            </p>
          </div>
        </section>

        {/* Section 2 — Intelligent Fashion Discovery */}
        <section className="elan-container py-14 md:py-18">
          <div className="max-w-2xl">
            <p className="text-xs font-semibold uppercase tracking-[0.24em] text-[#9a6b48]">
              Curated Discovery
            </p>
            <h2 className="mt-2.5 font-serif text-3xl sm:text-4xl">
              Intelligent Fashion Discovery
            </h2>
          </div>

          <div className="mt-8 grid gap-6 md:grid-cols-2">
            <article className="border border-[#e7e2dc] bg-white p-7 sm:p-8">
              <h3 className="font-serif text-2xl text-[#24221f]">
                Similar Styles
              </h3>
              <p className="mt-3 text-sm leading-7 text-[#686159]">
                On every product page, ÉLAN uses CLIP-based visual and textual
                similarity to suggest related fashion items that complement the
                piece&apos;s silhouette, color palette, and attributes, with
                TF-IDF as a reliable fallback.
              </p>
            </article>

            <article className="border border-[#e7e2dc] bg-white p-7 sm:p-8">
              <h3 className="font-serif text-2xl text-[#24221f]">
                Your Style Edit
              </h3>
              <p className="mt-3 text-sm leading-7 text-[#686159]">
                Within the collection, ItemKNN Collaborative Filtering curates
                tailored product recommendations for selectable simulated
                shopper profiles—letting you explore personalized style
                discovery without live customer tracking or personal data
                collection.
              </p>
            </article>
          </div>
        </section>

        {/* Section 3 — Technology Behind ÉLAN */}
        <section className="border-y border-[#e7e2dc] bg-[#f4efe8]/70 py-14 md:py-18">
          <div className="elan-container">
            <p className="text-xs font-semibold uppercase tracking-[0.24em] text-[#9a6b48]">
              System Overview
            </p>
            <h2 className="mt-2.5 font-serif text-3xl sm:text-4xl">
              Technology Behind ÉLAN
            </h2>

            <dl className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
              {technologies.map((item) => (
                <div
                  key={item.label}
                  className="border border-[#e2dbd2] bg-white p-5"
                >
                  <dt className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[#9a6b48]">
                    {item.label}
                  </dt>
                  <dd className="mt-2 font-serif text-lg leading-snug text-[#24221f]">
                    {item.value}
                  </dd>
                </div>
              ))}
            </dl>
          </div>
        </section>

        {/* Section 4 — Explore the Project */}
        <section className="elan-container py-14 md:py-18">
          <div className="flex flex-col items-start justify-between gap-8 border border-[#e7e2dc] bg-white p-7 sm:p-10 lg:flex-row lg:items-center">
            <div className="max-w-2xl">
              <p className="text-xs font-semibold uppercase tracking-[0.24em] text-[#9a6b48]">
                Academic Simulation
              </p>
              <h2 className="mt-2.5 font-serif text-3xl sm:text-4xl">
                Explore the Project
              </h2>
              <p className="mt-3 text-sm leading-7 text-[#686159]">
                ÉLAN is an academic simulation using synthetic shopper
                interactions, illustrative catalog prices, and a browser-saved
                shopping bag with no real payment processing. Detailed offline
                evaluations, benchmark comparisons, and architectural
                documentation are available in the project repository.
              </p>
            </div>

            <div className="flex w-full flex-col gap-3 sm:w-auto sm:flex-row lg:flex-col">
              <a
                href="https://github.com/vukhiem2005/ai-fashion-recommender#readme"
                target="_blank"
                rel="noopener noreferrer"
                className="elan-btn-primary text-center"
              >
                View Research &amp; Technical Documentation ↗
              </a>
              <Link href="/shop" className="elan-btn-secondary text-center">
                Explore Collection
              </Link>
            </div>
          </div>
        </section>
      </main>

      <StoreFooter />
    </div>
  );
}
