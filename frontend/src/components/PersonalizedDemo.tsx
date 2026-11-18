"use client";

import Image from "next/image";
import Link from "next/link";
import { useEffect, useId, useState } from "react";

import { formatPrice } from "@/data/products";

type InteractionSummary = {
  total_train_events: number;
  distinct_products_seen: number;
  views: number;
  add_to_carts: number;
  purchases: number;
  top_categories: string[];
};

type SyntheticProfile = {
  user_id: string;
  display_label: string;
  is_cold_start: boolean;
  interaction_summary: InteractionSummary;
};

type HistoryItem = {
  product_id: string;
  slug: string;
  name: string;
  category: string;
  event_type: string;
};

type PersonalizedItem = {
  id: string;
  slug: string;
  name: string;
  category: string;
  price_cents: number;
  image_url: string | null;
  reason: string;
};

type PersonalizedResponse = {
  reference_user_id: string;
  display_label: string;
  model: string;
  is_synthetic_simulation: boolean;
  cold_start_fallback: boolean;
  interaction_summary: InteractionSummary;
  recent_history: HistoryItem[];
  count: number;
  recommendations: PersonalizedItem[];
};

const EDITORIAL_PROFILE_LABELS: Record<string, string> = {
  "SYN-U0001": "Tops & Casual Edit",
  "SYN-U0002": "Dresses & Occasion Edit",
  "SYN-U0003": "Outerwear & Tailoring Edit",
  "SYN-U0007": "Bottoms & Denim Edit",
  "SYN-U0016": "Knitwear & Layering Edit",
  "SYN-U0048": "Accessories & Finishing Edit",
  "SYN-COLD-START": "New Visitor — Most-Loved Essentials",
};

function formatStyleProfileLabel(userId: string, rawLabel: string): string {
  const mapped = EDITORIAL_PROFILE_LABELS[userId];
  if (mapped) return mapped;
  return rawLabel
    .replace(/^Simulated Shopper \d+\s*—\s*/i, "")
    .replace(/\s*\(Popularity Fallback\)/i, "")
    .trim();
}

export default function PersonalizedDemo() {
  const selectId = useId();
  const [profiles, setProfiles] = useState<SyntheticProfile[]>([]);
  const [selectedUserId, setSelectedUserId] = useState("SYN-U0001");
  const [payload, setPayload] = useState<PersonalizedResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    const controller = new AbortController();

    async function loadProfiles() {
      try {
        const res = await fetch("/api/recommendations/personalized?users=1", {
          cache: "no-store",
          signal: controller.signal,
        });
        if (!res.ok) return;
        const data = (await res.json()) as { profiles?: SyntheticProfile[] };
        if (!controller.signal.aborted && Array.isArray(data.profiles)) {
          setProfiles(data.profiles);
        }
      } catch {
        // Non-blocking: default SYN-U0001 still loads below
      }
    }

    void loadProfiles();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();

    async function loadRecommendations() {
      setLoading(true);
      setError(false);

      try {
        const res = await fetch(
          `/api/recommendations/personalized?user_id=${encodeURIComponent(
            selectedUserId
          )}&top_k=5`,
          { cache: "no-store", signal: controller.signal }
        );
        if (!res.ok) {
          throw new Error(`Personalization request failed: ${res.status}`);
        }
        const data = (await res.json()) as PersonalizedResponse;
        if (!Array.isArray(data.recommendations)) {
          throw new Error("Invalid personalized recommendation response");
        }
        if (!controller.signal.aborted) {
          setPayload(data);
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
  }, [selectedUserId]);

  const items = payload?.recommendations ?? [];

  return (
    <section
      className="my-14 scroll-mt-24 border border-[#e7e2dc] bg-[#f4efe8]/80 px-6 py-9 sm:px-8 sm:py-11"
      aria-labelledby="personalized-demo-heading"
    >
      <div className="flex flex-col justify-between gap-6 lg:flex-row lg:items-end">
        <div className="max-w-2xl">
          <div className="flex flex-wrap items-center gap-3">
            <p className="text-xs font-semibold uppercase tracking-[0.26em] text-[#9a6b48]">
              PERSONALIZED DISCOVERY
            </p>
            <span className="border border-[#dfd8cf] bg-[#faf9f6] px-2.5 py-0.5 text-[11px] font-medium uppercase tracking-[0.14em] text-[#686159]">
              Demo style profiles
            </span>
          </div>

          <h2
            id="personalized-demo-heading"
            className="mt-3 font-serif text-3xl text-[#24221f] sm:text-4xl"
          >
            Your Style Edit
          </h2>

          <p className="mt-2.5 text-sm leading-6 text-[#686159]">
            Discover pieces curated around different style profiles.
          </p>
        </div>

        <div className="w-full sm:w-80">
          <label
            htmlFor={selectId}
            className="block text-xs font-semibold uppercase tracking-[0.18em] text-[#24221f]"
          >
            Explore a Style Profile
          </label>
          <select
            id={selectId}
            value={selectedUserId}
            onChange={(e) => setSelectedUserId(e.target.value)}
            className="mt-2 min-h-11 w-full border border-[#d5cdc3] bg-white px-3.5 py-2.5 text-sm text-[#24221f] transition-colors hover:border-[#9a6b48]"
          >
            {profiles.length > 0 ? (
              profiles.map((p) => (
                <option key={p.user_id} value={p.user_id}>
                  {formatStyleProfileLabel(p.user_id, p.display_label)}
                </option>
              ))
            ) : (
              <option value="SYN-U0001">Tops &amp; Casual Edit</option>
            )}
          </select>
        </div>
      </div>

      {loading ? (
        <p className="mt-8 py-8 text-sm text-[#686159]" aria-busy="true">
          Curating your style edit...
        </p>
      ) : error ? (
        <p role="alert" className="mt-8 py-8 text-sm text-[#686159]">
          Style profile recommendations are temporarily unavailable.
        </p>
      ) : items.length === 0 ? (
        <p className="mt-8 py-8 text-sm text-[#686159]">
          No curated pieces available for this style profile.
        </p>
      ) : (
        <ul
          role="list"
          className="mt-8 grid list-none grid-cols-2 gap-4 p-0 sm:grid-cols-3 sm:gap-5 lg:grid-cols-5"
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
