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
      className="mt-20 border-t border-[#e7e2dc] pt-12"
      aria-labelledby="personalized-demo-heading"
    >
      <div className="flex flex-wrap items-end justify-between gap-6">
        <div className="max-w-2xl">
          <div className="flex flex-wrap items-center gap-3">
            <p className="text-xs font-semibold uppercase tracking-[0.25em] text-[#ae805f]">
              Synthetic Behavioral AI Demo
            </p>
            <span className="border border-[#e7e2dc] bg-white px-2.5 py-0.5 text-[11px] uppercase tracking-[0.16em] text-[#706961]">
              {payload?.cold_start_fallback
                ? "ItemKNN · Popularity Fallback"
                : "ItemKNN Collaborative Filtering"}
            </span>
          </div>
          <h2
            id="personalized-demo-heading"
            className="mt-3 font-serif text-3xl md:text-4xl"
          >
            Recommended for Synthetic Shopper
          </h2>
        </div>

        <div className="w-full sm:w-80">
          <label
            htmlFor={selectId}
            className="block text-xs font-semibold uppercase tracking-[0.18em] text-[#706961]"
          >
            Simulated Shopper Profile
          </label>
          <select
            id={selectId}
            value={selectedUserId}
            onChange={(e) => setSelectedUserId(e.target.value)}
            className="mt-2 w-full border border-[#d9d2c9] bg-white px-3.5 py-2.5 text-sm text-[#24221f] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#24221f]"
          >
            {profiles.length > 0 ? (
              profiles.map((p) => (
                <option key={p.user_id} value={p.user_id}>
                  {p.display_label}
                </option>
              ))
            ) : (
              <option value="SYN-U0001">
                Simulated Shopper 01 — Tops &amp; Casual Edit
              </option>
            )}
          </select>
        </div>
      </div>

      {payload && !loading && !error ? (
        <div className="mt-6 border border-[#e7e2dc] bg-white/80 p-4 text-xs text-[#706961]">
          <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
            <span>
              <strong className="text-[#24221f]">Profile:</strong>{" "}
              {payload.reference_user_id}
            </span>
            <span>
              <strong className="text-[#24221f]">Simulated Train Events:</strong>{" "}
              {payload.interaction_summary.total_train_events} (
              {payload.interaction_summary.views} views,{" "}
              {payload.interaction_summary.add_to_carts} cart adds,{" "}
              {payload.interaction_summary.purchases} purchases)
            </span>
          </div>
          {payload.recent_history.length > 0 ? (
            <p className="mt-2 leading-6">
              <strong className="text-[#24221f]">Sampled History:</strong>{" "}
              {payload.recent_history
                .map((h) => `${h.name} (${h.event_type.replaceAll("_", " ")})`)
                .join(" · ")}
            </p>
          ) : (
            <p className="mt-2 leading-6">
              Zero prior interactions — demonstrating deterministic cold-start
              popularity fallback.
            </p>
          )}
        </div>
      ) : null}

      {loading ? (
        <p className="mt-8 text-sm text-[#817a72]" aria-busy="true">
          Loading synthetic personalized recommendations...
        </p>
      ) : error ? (
        <p role="alert" className="mt-8 text-sm text-[#817a72]">
          Synthetic personalized recommendations are temporarily unavailable.
        </p>
      ) : items.length === 0 ? (
        <p className="mt-8 text-sm text-[#817a72]">
          No personalized recommendations available for this simulated profile.
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
                        <span className="font-serif text-3xl text-[#b5a79a]">
                          ÉLAN.
                        </span>
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
