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

function getRecommendationPresentation(model: string | null): {
  badge: string | null;
  subtitle: string;
} {
  if (model === "clip_late_fusion") {
    return {
      badge: "Visual + Text AI",
      subtitle:
        "Similar styles matched using visual photography and product attribute signals.",
    };
  }
  if (model === "tfidf_cosine_v1") {
    return {
      badge: "Attribute Match",
      subtitle:
        "Similar styles selected from product categories, colors, and descriptions.",
    };
  }
  return {
    badge: null,
    subtitle: "Discover complementary pieces from our curated collection.",
  };
}

export default function YouMayAlsoLike({ productId }: { productId: string }) {
  const [items, setItems] = useState<Recommendation[]>([]);
  const [activeModel, setActiveModel] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    const controller = new AbortController();

    async function loadRecommendations() {
      setLoading(true);
      setError(false);
      setItems([]);
      setActiveModel(null);

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
          setActiveModel(typeof data.model === "string" ? data.model : null);
        }
      } catch {
        if (!controller.signal.aborted) {
          setError(true);
          setActiveModel(null);
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

  const { badge, subtitle } = getRecommendationPresentation(activeModel);

  return (
    <section
      className="mt-20 border-t border-[#e7e2dc] pt-12"
      aria-labelledby="recommendation-heading"
    >
      <div className="flex flex-wrap items-center gap-3">
        <p className="text-xs uppercase tracking-[0.25em] text-[#ae805f]">
          AI-Powered Fashion Discovery
        </p>
        {!loading && !error && badge ? (
          <span className="border border-[#e7e2dc] bg-white px-2.5 py-0.5 text-[11px] uppercase tracking-[0.16em] text-[#706961]">
            {badge}
          </span>
        ) : null}
      </div>
      <h2 id="recommendation-heading" className="mt-3 font-serif text-3xl md:text-4xl">
        You May Also Like
      </h2>
      <p className="mt-3 text-sm text-[#817a72]">{subtitle}</p>

      {loading ? (
        <p className="mt-8 text-sm text-[#817a72]" aria-busy="true">
          Finding similar styles...
        </p>
      ) : error ? (
        <p role="alert" className="mt-8 text-sm text-[#817a72]">
          Recommendations are temporarily unavailable.
        </p>
      ) : items.length === 0 ? (
        <p className="mt-8 text-sm text-[#817a72]">
          No similar products available yet.
        </p>
      ) : (
        <ul
          role="list"
          className="mt-8 grid list-none grid-cols-2 gap-5 p-0 sm:grid-cols-3 lg:grid-cols-5"
        >
          {items.map((item) => {
            const imageUrl = item.image_url?.startsWith("/")
              ? item.image_url
              : null;

            return (
              <li key={item.id} className="min-w-0">
                <Link
                  href={`/shop/${encodeURIComponent(item.slug)}`}
                  className="group block min-w-0 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[#24221f]"
                >
                  <div className="relative aspect-[3/4] overflow-hidden bg-[#eee8e0]">
                    {imageUrl ? (
                      <Image
                        src={imageUrl}
                        alt={item.name}
                        fill
                        sizes="(max-width: 640px) 50vw, (max-width: 1024px) 33vw, 20vw"
                        className="object-cover transition-transform duration-300 group-hover:scale-105"
                      />
                    ) : (
                      <div className="flex h-full flex-col items-center justify-center text-center">
                        <span className="font-serif text-3xl text-[#b5a79a]">ÉLAN.</span>
                        <span className="mt-3 px-3 text-xs text-[#a39385]">
                          Image Coming Soon
                        </span>
                      </div>
                    )}
                  </div>
                  <p className="mt-4 text-xs uppercase tracking-wider text-[#ae805f]">
                    {item.category}
                  </p>
                  <h3 className="mt-1 text-sm font-medium leading-6 group-hover:underline">
                    {item.name}
                  </h3>
                  <p className="mt-1 text-sm font-semibold">
                    {formatPrice(item.price_cents / 100)}
                  </p>
                  <p className="mt-2 text-xs leading-5 text-[#817a72]">
                    {item.reason}
                  </p>
                </Link>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
