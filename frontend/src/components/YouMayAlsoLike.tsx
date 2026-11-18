"use client";

import Image from "next/image";
import Link from "next/link";
import { useEffect, useState } from "react";

import { formatPrice } from "@/data/products";

type Recommendation = {
  id: string;
  slug: string;
  name: string;
  category: string;
  price_cents: number;
  image_url: string | null;
  similarity: number;
  reason: string;
};

type RecommendationResponse = {
  reference_product_id?: string;
  model?: string;
  count?: number;
  recommendations: Recommendation[];
};

export default function YouMayAlsoLike({ productId }: { productId: string }) {
  const [items, setItems] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    const controller = new AbortController();

    async function loadRecommendations() {
      setLoading(true);
      setError(false);
      setItems([]);

      try {
        const response = await fetch(
          `/api/recommendations/${encodeURIComponent(productId)}`,
          { cache: "no-store", signal: controller.signal }
        );

        if (!response.ok) {
          throw new Error(`Recommendation request failed: ${response.status}`);
        }

        const data: RecommendationResponse = await response.json();
        if (!Array.isArray(data.recommendations)) {
          throw new Error("Invalid recommendation response");
        }

        if (!controller.signal.aborted) {
          setItems(data.recommendations);
        }
      } catch {
        if (!controller.signal.aborted) {
          setError(true);
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    void loadRecommendations();
    return () => controller.abort();
  }, [productId]);

  return (
    <section
      className="mt-20 border-t border-[#e7e2dc] pt-14"
      aria-labelledby="recommendation-heading"
    >
      <div className="max-w-2xl">
        <p className="text-xs font-semibold uppercase tracking-[0.26em] text-[#9a6b48]">
          CURATED FOR YOUR STYLE
        </p>
        <h2
          id="recommendation-heading"
          className="mt-3 font-serif text-3xl text-[#24221f] md:text-4xl"
        >
          You May Also Like
        </h2>
        <p className="mt-2.5 text-sm leading-6 text-[#686159]">
          Discover more pieces that complement your style.
        </p>
      </div>

      {loading ? (
        <p className="mt-8 text-sm text-[#686159]" aria-busy="true">
          Curating complementary pieces...
        </p>
      ) : error ? (
        <p role="alert" className="mt-8 text-sm text-[#686159]">
          Recommendations are temporarily unavailable.
        </p>
      ) : items.length === 0 ? (
        <p className="mt-8 text-sm text-[#686159]">
          No complementary pieces available yet.
        </p>
      ) : (
        <ul
          role="list"
          className="mt-9 grid list-none grid-cols-2 gap-4 p-0 sm:grid-cols-3 sm:gap-6 lg:grid-cols-5"
        >
          {items.map((item) => {
            const imageUrl = item.image_url?.startsWith("/")
              ? item.image_url
              : null;

            return (
              <li key={item.id} className="min-w-0">
                <article className="elan-product-card group flex h-full flex-col">
                  <div className="relative aspect-[3/4] overflow-hidden border border-[#e7e2dc] bg-[#eee8e0]">
                    {imageUrl ? (
                      <Image
                        src={imageUrl}
                        alt={item.name}
                        fill
                        sizes="(max-width: 640px) 50vw, (max-width: 1024px) 33vw, 20vw"
                        className="object-cover object-top transition-transform duration-500 group-hover:scale-105"
                      />
                    ) : (
                      <div className="flex h-full flex-col items-center justify-center text-center">
                        <span className="font-serif text-3xl text-[#b5a79a]">
                          ÉLAN.
                        </span>
                        <span className="mt-2 px-3 text-xs text-[#a39385]">
                          Image Coming Soon
                        </span>
                      </div>
                    )}
                  </div>

                  <p className="mt-3.5 text-[11px] font-semibold uppercase tracking-[0.18em] text-[#9a6b48]">
                    {item.category}
                  </p>

                  <h3 className="mt-1 font-serif text-lg leading-snug font-medium text-[#24221f] transition-colors group-hover:text-[#9a6b48]">
                    <Link
                      href={`/shop/${encodeURIComponent(item.slug)}`}
                      className="elan-card-link"
                    >
                      {item.name}
                    </Link>
                  </h3>

                  <p className="mt-1 text-sm font-medium text-[#24221f]">
                    {formatPrice(item.price_cents / 100)}
                  </p>
                </article>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
