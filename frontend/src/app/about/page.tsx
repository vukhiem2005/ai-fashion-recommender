import type { Metadata } from "next";
import Link from "next/link";
import StoreFooter from "@/components/StoreFooter";
import StoreHeader from "@/components/StoreHeader";

export const metadata: Metadata = {
  title: "About ÉLAN | Academic AI Fashion Storefront & Research",
  description:
    "Explore the architecture, offline evaluation benchmarks, evidence-based model selection, and synthetic data provenance behind the ÉLAN fashion recommendation storefront.",
};

const activeSystems = [
  {
    badge: "Primary Similar-Items Engine · Deployed",
    title: "CLIP Late Fusion",
    modelId: "clip_late_fusion",
    surface: "Product Detail Pages (/shop/[slug]) — You May Also Like",
    summary:
      "Combines normalized 512-D CLIP ViT-B/32 visual embeddings (w_vis = 0.5) with V1 TF-IDF attribute cosine similarity (w_text = 0.5, tau = 0.0) across 327 image-backed catalog products.",
    rationale:
      "Served from a precomputed NumPy (.npz) embedding bundle with 0.485 ms/query latency and zero runtime PyTorch or Transformers memory overhead.",
  },
  {
    badge: "Content Baseline & Exact Fallback · Deployed",
    title: "TF-IDF + Cosine Similarity (V1)",
    modelId: "tfidf_cosine_v1",
    surface: "Product Detail Pages (/shop/[slug]) — Attribute Match Fallback",
    summary:
      "Ranks active products by cosine similarity over unigram/bigram TF-IDF vectors constructed from weighted product metadata (category × 3, name × 2, description × 1, colors × 2).",
    rationale:
      "Provides deterministic 100% catalog coverage, automatically serving products without an image embedding (such as HM-0398947001) or whenever multimodal cache mode is disabled.",
  },
  {
    badge: "Synthetic Behavioral Demo · Deployed on /shop",
    title: "ItemKNN Collaborative Filtering",
    modelId: "itemknn_cf_synthetic_demo",
    surface: "Collection Page (/shop) — Recommended for Synthetic Shopper",
    summary:
      "Item-to-item collaborative filtering (log_sum deduplication, IUF exponent = 0.5, shrinkage = 20.0, k = 100 neighbors) fitted exclusively on chronological training interactions from SYNTHETIC_SIMULATION_V5.",
    rationale:
      "Demonstrates personalized top-5 ranking for deterministic simulated shopper profiles (SYN-U0001 through SYN-U0048), strictly excluding already-seen training items.",
  },
  {
    badge: "Cold-Start Guardrail · Deployed on /shop",
    title: "Deterministic Popularity Fallback",
    modelId: "itemknn_cf_synthetic_demo (cold_start_fallback = true)",
    surface: "Collection Page (/shop) — Profile SYN-COLD-START",
    summary:
      "When a simulated profile has zero prior training interactions, the personalization service routes deterministically to top training-popularity items.",
    rationale:
      "Prevents empty recommendation states for new or unconditioned profiles while surfacing an explicit ItemKNN · Popularity Fallback disclosure badge.",
  },
];

const similarItemBenchmarks = [
  {
    model: "Phase 4B CLIP Late Fusion",
    status: "Deployed Primary",
    devP5: "68.89%",
    devNdcg5: "0.8933",
    holdoutP5: "71.11%",
    holdoutStrictP5: "26.67%",
    holdoutNdcg5: "0.7625",
    deltaVsV1: "+9.44% (+0.0658)",
    coverage5: "95.43% (313/328)",
  },
  {
    model: "Phase 4A Early Fusion",
    status: "Research Only",
    devP5: "58.89%",
    devNdcg5: "0.7781",
    holdoutP5: "71.11%",
    holdoutStrictP5: "25.56%",
    holdoutNdcg5: "0.7338",
    deltaVsV1: "+5.33% (+0.0371)",
    coverage5: "96.04% (315/328)",
  },
  {
    model: "Phase 4A Late Fusion",
    status: "Research Only",
    devP5: "61.11%",
    devNdcg5: "0.8023",
    holdoutP5: "70.00%",
    holdoutStrictP5: "24.44%",
    holdoutNdcg5: "0.7282",
    deltaVsV1: "+4.52% (+0.0315)",
    coverage5: "97.26% (319/328)",
  },
  {
    model: "Phase 4B CLIP Image-Only",
    status: "Research Only",
    devP5: "70.00%",
    devNdcg5: "0.8425",
    holdoutP5: "72.22%",
    holdoutStrictP5: "23.33%",
    holdoutNdcg5: "0.7091",
    deltaVsV1: "+1.78% (+0.0124)",
    coverage5: "91.16% (299/328)",
  },
  {
    model: "Phase 4B CLIP Text-Image",
    status: "Research Only",
    devP5: "75.56%",
    devNdcg5: "0.8819",
    holdoutP5: "76.67%",
    holdoutStrictP5: "23.33%",
    holdoutNdcg5: "0.7066",
    deltaVsV1: "+1.42% (+0.0099)",
    coverage5: "85.98% (282/328)",
  },
  {
    model: "Exact V1 TF-IDF Baseline",
    status: "Deployed Fallback",
    devP5: "56.67%",
    devNdcg5: "0.7699",
    holdoutP5: "66.67%",
    holdoutStrictP5: "24.44%",
    holdoutNdcg5: "0.6967",
    deltaVsV1: "Baseline (0.00%)",
    coverage5: "96.65% (317/328)",
  },
  {
    model: "Phase 4A Lightweight Image-Only",
    status: "Research Only",
    devP5: "61.11%",
    devNdcg5: "0.6731",
    holdoutP5: "70.00%",
    holdoutStrictP5: "23.33%",
    holdoutNdcg5: "0.6823",
    deltaVsV1: "-2.07% (-0.0144)",
    coverage5: "96.34% (316/328)",
  },
];

const personalizationBenchmarks = [
  {
    model: "ItemKNN_CF (log_sum, IUF=0.5, k=100)",
    status: "Deployed Demo",
    a1P5: "4.11%",
    a1Recall5: "5.61%",
    a1GradedNdcg5: "0.0514",
    a2GradedNdcg5: "0.0544",
    coverage5: "36.59% (120/328)",
    floorMet: "Yes (192/192 configs)",
  },
  {
    model: "Hybrid_Content_CF (w_content=0.50, w_cf=0.35)",
    status: "Deferred (Offline)",
    a1P5: "3.96%",
    a1Recall5: "5.68%",
    a1GradedNdcg5: "0.0501",
    a2GradedNdcg5: "—",
    coverage5: "45.73% (150/328)",
    floorMet: "Yes (Milestone 5)",
  },
  {
    model: "Hybrid_Content_CF_ARM (0.50 / 0.35 / 0.15)",
    status: "Deferred (Offline)",
    a1P5: "3.89%",
    a1Recall5: "5.59%",
    a1GradedNdcg5: "0.0493",
    a2GradedNdcg5: "—",
    coverage5: "46.95% (154/328)",
    floorMet: "Yes (Milestone 5)",
  },
  {
    model: "TruncatedSVD_MF (n_components=16)",
    status: "Deferred (Offline)",
    a1P5: "3.71%",
    a1Recall5: "4.86%",
    a1GradedNdcg5: "0.0464",
    a2GradedNdcg5: "0.0485",
    coverage5: "33.54% (110/328)",
    floorMet: "Yes (48/60 configs)",
  },
  {
    model: "PopularityBaseline",
    status: "Cold-Start Fallback",
    a1P5: "3.27%",
    a1Recall5: "4.59%",
    a1GradedNdcg5: "0.0383",
    a2GradedNdcg5: "0.0363",
    coverage5: "3.35% (11/328)",
    floorMet: "No (11/328 items)",
  },
  {
    model: "V1_TFIDF_Content (User Profile)",
    status: "Baseline Reference",
    a1P5: "2.35%",
    a1Recall5: "2.82%",
    a1GradedNdcg5: "0.0220",
    a2GradedNdcg5: "0.0240",
    coverage5: "80.79% (265/328)",
    floorMet: "Yes",
  },
];

export default function AboutPage() {
  return (
    <div className="flex min-h-screen flex-col bg-[#faf9f6] text-[#24221f]">
      <StoreHeader />

      <main id="main-content" className="flex-1">
        {/* 1. About ÉLAN — Editorial Header */}
        <section className="border-b border-[#e7e2dc] bg-gradient-to-b from-[#faf9f6] to-[#f3ede4] py-16 md:py-24">
          <div className="elan-container">
            <div className="max-w-3xl">
              <p className="text-xs font-semibold uppercase tracking-[0.28em] text-[#9a6b48]">
                House of ÉLAN · Academic AI Storefront Simulation
              </p>
              <h1 className="mt-4 font-serif text-4xl leading-tight sm:text-5xl md:text-6xl">
                Where contemporary fashion meets evidence-based AI discovery.
              </h1>
              <p className="mt-6 text-base leading-8 text-[#686159]">
                ÉLAN is an academic AI-powered fashion e-commerce simulation and
                full-stack engineering project. It pairs a minimalist
                contemporary storefront with reproducible offline evaluation
                across content-based, multimodal, collaborative filtering,
                association rule mining, and hybrid recommendation algorithms.
              </p>
              <div className="mt-8 flex flex-wrap items-center gap-4">
                <Link href="/shop" className="elan-btn-primary">
                  Explore the Collection
                </Link>
                <a
                  href="https://github.com/vukhiem2005/ai-fashion-recommender"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="elan-btn-secondary"
                >
                  View GitHub Repository ↗
                </a>
              </div>
            </div>

            <dl className="mt-14 grid grid-cols-2 gap-6 border-t border-[#ded6cc] pt-8 sm:grid-cols-4">
              <div>
                <dt className="text-xs font-semibold uppercase tracking-[0.18em] text-[#756d65]">
                  Release Catalog
                </dt>
                <dd className="mt-2 font-serif text-3xl text-[#24221f]">
                  328 Products · 343 SKUs
                </dd>
              </div>
              <div>
                <dt className="text-xs font-semibold uppercase tracking-[0.18em] text-[#756d65]">
                  Visual Embeddings
                </dt>
                <dd className="mt-2 font-serif text-3xl text-[#24221f]">
                  327 / 328 CLIP ViT-B/32
                </dd>
              </div>
              <div>
                <dt className="text-xs font-semibold uppercase tracking-[0.18em] text-[#756d65]">
                  Synthetic Benchmark
                </dt>
                <dd className="mt-2 font-serif text-3xl text-[#24221f]">
                  600 Users · 23,335 Events
                </dd>
              </div>
              <div>
                <dt className="text-xs font-semibold uppercase tracking-[0.18em] text-[#756d65]">
                  Deployment Principle
                </dt>
                <dd className="mt-2 font-serif text-3xl text-[#24221f]">
                  Evidence-Gated Only
                </dd>
              </div>
            </dl>
          </div>
        </section>

        {/* 2. The Shopping Experience */}
        <section
          id="shopping-experience"
          className="elan-container py-16 md:py-20"
        >
          <div className="max-w-2xl">
            <p className="text-xs font-semibold uppercase tracking-[0.25em] text-[#9a6b48]">
              01 · Storefront Capabilities
            </p>
            <h2 className="mt-3 font-serif text-3xl sm:text-4xl">
              The Shopping Experience
            </h2>
            <p className="mt-4 text-sm leading-7 text-[#686159] sm:text-base">
              Every customer-facing surface is designed to feel like a modern
              editorial fashion house while remaining transparent about how
              recommendations and simulated interactions work.
            </p>
          </div>

          <div className="mt-10 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
            <article className="border border-[#e7e2dc] bg-white p-6">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#9a6b48]">
                Catalog Discovery
              </p>
              <h3 className="mt-3 font-serif text-2xl">
                Curated Collection Browsing
              </h3>
              <p className="mt-3 text-sm leading-6 text-[#686159]">
                Filter 328 active products across Outerwear, Tops, Bottoms,
                Dresses, Knitwear, and Accessories with instant keyword search,
                price sorting, and 24-item pagination.
              </p>
            </article>

            <article className="border border-[#e7e2dc] bg-white p-6">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#9a6b48]">
                Product Detail Pages
              </p>
              <h3 className="mt-3 font-serif text-2xl">
                Similar-Item Discovery
              </h3>
              <p className="mt-3 text-sm leading-6 text-[#686159]">
                Each product page presents color and size SKU selection alongside
                five{" "}
                <strong className="font-medium text-[#24221f]">
                  You May Also Like
                </strong>{" "}
                recommendations labeled with{" "}
                <code className="text-xs">Visual + Text AI</code> or{" "}
                <code className="text-xs">Attribute Match</code>.
              </p>
            </article>

            <article className="border border-[#e7e2dc] bg-white p-6">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#9a6b48]">
                Interactive Simulation
              </p>
              <h3 className="mt-3 font-serif text-2xl">
                Personalized Shopper Profiles
              </h3>
              <p className="mt-3 text-sm leading-6 text-[#686159]">
                Switch between 7 deterministic synthetic profiles on{" "}
                <Link href="/shop#personalized-demo-heading" className="underline">
                  /shop
                </Link>{" "}
                to inspect training interaction counts, sampled product history,
                and Top-5 collaborative filtering recommendations.
              </p>
            </article>

            <article className="border border-[#e7e2dc] bg-white p-6">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#9a6b48]">
                Bag &amp; Account State
              </p>
              <h3 className="mt-3 font-serif text-2xl">
                Simulated Shopping Bag
              </h3>
              <p className="mt-3 text-sm leading-6 text-[#686159]">
                Add SKU variants to a persistent browser shopping bag, adjust
                item quantities (1–99), or save a local guest display profile on{" "}
                <Link href="/account" className="underline">
                  /account
                </Link>{" "}
                without creating an external account.
              </p>
            </article>
          </div>
        </section>

        {/* 3. AI Recommendation Systems */}
        <section
          id="ai-systems"
          className="border-y border-[#e7e2dc] bg-[#f2ece4]/70 py-16 md:py-20"
        >
          <div className="elan-container">
            <div className="max-w-2xl">
              <p className="text-xs font-semibold uppercase tracking-[0.25em] text-[#9a6b48]">
                02 · Production &amp; Simulation Engines
              </p>
              <h2 className="mt-3 font-serif text-3xl sm:text-4xl">
                AI Recommendation Systems
              </h2>
              <p className="mt-4 text-sm leading-7 text-[#686159] sm:text-base">
                ÉLAN separates item-to-item catalog similarity from synthetic
                behavioral personalization. Synthetic recommendations are never
                represented as live customer tracking.
              </p>
            </div>

            <div className="mt-10 grid gap-6 md:grid-cols-2">
              {activeSystems.map((system) => (
                <article
                  key={system.title}
                  className="flex flex-col justify-between border border-[#dfd8cf] bg-white p-7"
                >
                  <div>
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[#9a6b48]">
                        {system.badge}
                      </span>
                      <code className="border border-[#e7e2dc] bg-[#faf9f6] px-2 py-0.5 text-xs text-[#686159]">
                        {system.modelId}
                      </code>
                    </div>

                    <h3 className="mt-3 font-serif text-2xl text-[#24221f]">
                      {system.title}
                    </h3>

                    <p className="mt-1 text-xs font-medium text-[#756d65]">
                      Active Surface: {system.surface}
                    </p>

                    <p className="mt-4 text-sm leading-7 text-[#686159]">
                      {system.summary}
                    </p>
                  </div>

                  <p className="mt-5 border-t border-[#e7e2dc] pt-4 text-xs leading-6 text-[#686159]">
                    <strong className="font-semibold text-[#24221f]">
                      Engineering Role:
                    </strong>{" "}
                    {system.rationale}
                  </p>
                </article>
              ))}
            </div>
          </div>
        </section>

        {/* 4. Research & Model Selection */}
        <section
          id="research-selection"
          className="elan-container py-16 md:py-20"
        >
          <div className="max-w-3xl">
            <p className="text-xs font-semibold uppercase tracking-[0.25em] text-[#9a6b48]">
              03 · Offline Evaluation &amp; Ablation Evidence
            </p>
            <h2 className="mt-3 font-serif text-3xl sm:text-4xl">
              Research &amp; Evidence-Based Model Selection
            </h2>
            <p className="mt-4 text-sm leading-7 text-[#686159] sm:text-base">
              Not every researched model was integrated into the live website.
              Across Milestones 1–5, candidate algorithms were evaluated offline
              and promoted only when they demonstrated measurable ranking gains
              and met catalog coverage floors.
            </p>
          </div>

          {/* Table 1: Similar-Item Multimodal Evaluation */}
          <div className="mt-10">
            <h3
              id="table-similar-items-heading"
              className="font-serif text-2xl text-[#24221f]"
            >
              Task 1 — Similar-Item Retrieval (Milestone 4A &amp; 4B)
            </h3>
            <p className="mt-2 text-xs leading-6 text-[#686159]">
              Protocol M1 (Pooled Reranking across 18 disjoint Holdout references
              and 140 judged candidate pairs from{" "}
              <code>holdout_v4_blind_labeled.csv</code>; 0 overlap with Dev-18)
              and Protocol M2 (Full-Catalog 328-product diagnostics). Note:
              Holdout-18 is a historically reused evaluation set reported
              transparently for cross-milestone comparison.
            </p>

            <div
              tabIndex={0}
              role="region"
              aria-labelledby="table-similar-items-heading"
              className="mt-4 overflow-x-auto border border-[#e7e2dc] bg-white"
            >
              <table className="w-full min-w-[760px] border-collapse text-left text-xs sm:text-sm">
                <caption className="sr-only">
                  Milestone 4A and 4B similar-item benchmark comparison across
                  Dev-18 and Holdout-18 sets
                </caption>
                <thead className="border-b border-[#e7e2dc] bg-[#f2ece4] text-[11px] uppercase tracking-[0.14em] text-[#686159]">
                  <tr>
                    <th scope="col" className="px-4 py-3.5">
                      Model
                    </th>
                    <th scope="col" className="px-4 py-3.5">
                      Decision
                    </th>
                    <th scope="col" className="px-4 py-3.5">
                      Dev nDCG@5
                    </th>
                    <th scope="col" className="px-4 py-3.5">
                      Holdout P@5
                    </th>
                    <th scope="col" className="px-4 py-3.5">
                      Strict P@5
                    </th>
                    <th scope="col" className="px-4 py-3.5">
                      Holdout nDCG@5
                    </th>
                    <th scope="col" className="px-4 py-3.5">
                      Δ nDCG@5 vs V1
                    </th>
                    <th scope="col" className="px-4 py-3.5">
                      Coverage@5
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#eee9e2]">
                  {similarItemBenchmarks.map((row) => (
                    <tr
                      key={row.model}
                      className={
                        row.status.startsWith("Deployed")
                          ? "bg-[#faf6f0] font-medium"
                          : "bg-white"
                      }
                    >
                      <th scope="row" className="px-4 py-3.5 font-medium text-[#24221f]">
                        {row.model}
                      </th>
                      <td className="px-4 py-3.5 text-xs text-[#9a6b48]">
                        {row.status}
                      </td>
                      <td className="px-4 py-3.5">{row.devNdcg5}</td>
                      <td className="px-4 py-3.5">{row.holdoutP5}</td>
                      <td className="px-4 py-3.5">{row.holdoutStrictP5}</td>
                      <td className="px-4 py-3.5 font-semibold">
                        {row.holdoutNdcg5}
                      </td>
                      <td className="px-4 py-3.5">{row.deltaVsV1}</td>
                      <td className="px-4 py-3.5 text-[#686159]">
                        {row.coverage5}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Table 2: Personalized Collaborative & Hybrid Evaluation */}
          <div className="mt-12">
            <h3
              id="table-personalization-heading"
              className="font-serif text-2xl text-[#24221f]"
            >
              Task 2 — Personalized Next-Item Ranking (Milestone 2 &amp; 5 Synthetic Evaluation)
            </h3>
            <p className="mt-2 text-xs leading-6 text-[#686159]">
              Evaluated on <code>SYNTHETIC_SIMULATION_V5</code> (random seed{" "}
              <code>20260501</code>; 16,091 training events across 600 users;
              545 test users in Protocol A1 with frozen training history; 542
              test users in Protocol A2 with folded-in train ∪ val history).
              Validation selection required{" "}
              <code>Catalog Coverage@5 &gt;= 25.0%</code> (82 / 328 products).
            </p>

            <div
              tabIndex={0}
              role="region"
              aria-labelledby="table-personalization-heading"
              className="mt-4 overflow-x-auto border border-[#e7e2dc] bg-white"
            >
              <table className="w-full min-w-[760px] border-collapse text-left text-xs sm:text-sm">
                <caption className="sr-only">
                  Milestone 2 and Milestone 5 synthetic user-item personalized
                  ranking benchmark results
                </caption>
                <thead className="border-b border-[#e7e2dc] bg-[#f2ece4] text-[11px] uppercase tracking-[0.14em] text-[#686159]">
                  <tr>
                    <th scope="col" className="px-4 py-3.5">
                      Model
                    </th>
                    <th scope="col" className="px-4 py-3.5">
                      Decision
                    </th>
                    <th scope="col" className="px-4 py-3.5">
                      Test A1 P@5
                    </th>
                    <th scope="col" className="px-4 py-3.5">
                      Test A1 Recall@5
                    </th>
                    <th scope="col" className="px-4 py-3.5">
                      A1 Graded nDCG@5
                    </th>
                    <th scope="col" className="px-4 py-3.5">
                      A2 Graded nDCG@5
                    </th>
                    <th scope="col" className="px-4 py-3.5">
                      Catalog Coverage@5
                    </th>
                    <th scope="col" className="px-4 py-3.5">
                      &gt;=25% Val Floor
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#eee9e2]">
                  {personalizationBenchmarks.map((row) => (
                    <tr
                      key={row.model}
                      className={
                        row.status.startsWith("Deployed")
                          ? "bg-[#faf6f0] font-medium"
                          : "bg-white"
                      }
                    >
                      <th scope="row" className="px-4 py-3.5 font-medium text-[#24221f]">
                        {row.model}
                      </th>
                      <td className="px-4 py-3.5 text-xs text-[#9a6b48]">
                        {row.status}
                      </td>
                      <td className="px-4 py-3.5">{row.a1P5}</td>
                      <td className="px-4 py-3.5">{row.a1Recall5}</td>
                      <td className="px-4 py-3.5 font-semibold">
                        {row.a1GradedNdcg5}
                      </td>
                      <td className="px-4 py-3.5">{row.a2GradedNdcg5}</td>
                      <td className="px-4 py-3.5 text-[#686159]">
                        {row.coverage5}
                      </td>
                      <td className="px-4 py-3.5 text-xs text-[#686159]">
                        {row.floorMet}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Expandable Technical Details on Deferred / Research-Only Models */}
          <div className="mt-12">
            <h3
              id="deferred-models-heading"
              className="font-serif text-2xl text-[#24221f]"
            >
              Why Other Researched Algorithms Remain Offline
            </h3>
            <p className="mt-2 text-sm text-[#686159]">
              Expand each research milestone below to inspect the exact
              experimental findings that governed deployment selection.
            </p>

            <div
              aria-labelledby="deferred-models-heading"
              className="elan-accordion mt-5"
            >
              <details>
                <summary>
                  Milestone 3 — Apriori &amp; FP-Growth Association Rule Mining (Deferred from Live UI)
                </summary>
                <div className="px-5 pb-6 pt-2 text-sm leading-7 text-[#686159]">
                  <p>
                    Apriori and FP-Growth were implemented from scratch and
                    verified for exact mathematical equivalence across 1,016
                    training orders (692 multi-item baskets, mean 1.97 distinct
                    items/order). At <code>min_support_count = 2</code>, both
                    mined exactly 351 frequent itemsets (<code>|L1| = 299</code>,{" "}
                    <code>|L2| = 52</code>, <code>|L3| = 0</code>) and 104
                    association rules, with FP-Growth fitting in{" "}
                    <code>79.76 ms</code> vs. Apriori&apos;s{" "}
                    <code>3,435.46 ms</code>.
                  </p>
                  <p className="mt-3">
                    However, across all 256 evaluated hyperparameter
                    configurations, <strong>0 / 256 configurations</strong> met
                    the joint validation coverage floors (
                    <code>Rule Coverage &gt;= 25.0%</code> and{" "}
                    <code>Catalog Coverage@5 &gt;= 20.0%</code>). On the 155
                    multi-item test orders (Protocol B1 Leave-Last-Item-Out),{" "}
                    <code>Apriori_ARM_Pure</code> achieved only{" "}
                    <code>HitRate@5 = 1.29%</code> (2 / 155 orders),{" "}
                    <code>nDCG@5 = 0.0097</code>,{" "}
                    <code>Rule Coverage = 38.71%</code>, and{" "}
                    <code>Catalog Coverage@5 = 14.02% (46/328)</code>. Although{" "}
                    <code>Apriori_ARM_With_Fallback</code> reached{" "}
                    <code>HitRate@5 = 7.74%</code>, 10 of its 12 hits came from
                    the <code>OrderPopularityBaseline</code> fallback (which
                    scored <code>7.74%</code> on its own). Consequently, a
                    Frequently Bought Together widget was not deployed to the
                    website.
                  </p>
                </div>
              </details>

              <details>
                <summary>
                  Milestone 2 — TruncatedSVD Matrix Factorization vs. ItemKNN (Deferred)
                </summary>
                <div className="px-5 pb-6 pt-2 text-sm leading-7 text-[#686159]">
                  <p>
                    Across 60 evaluated configurations (30 unique rank-
                    <em>k</em> subspace projections), <code>TruncatedSVD_MF</code>{" "}
                    (<code>n_components = 16</code>,{" "}
                    <code>dedup_mode = &apos;log_sum&apos;</code>,{" "}
                    <code>18.93%</code> explained variance) satisfied the 25%
                    coverage floor on 48 / 60 grid points and reached{" "}
                    <code>Graded nDCG@5 = 0.0464</code> on Test Protocol A1 (
                    <code>0.0485</code> on Protocol A2,{" "}
                    <code>Coverage@5 = 33.54%</code>).
                  </p>
                  <p className="mt-3">
                    <code>ItemKNN_CF</code> outperformed{" "}
                    <code>TruncatedSVD_MF</code> on every personalized ranking
                    metric (<code>Graded nDCG@5 = 0.0514</code> on A1,{" "}
                    <code>0.0544</code> on A2, <code>P@5 = 4.11%</code> vs.{" "}
                    <code>3.71%</code>, and <code>Coverage@5 = 36.59%</code> vs.{" "}
                    <code>33.54%</code>) while offering directly interpretable
                    item-to-item neighborhood explanations.
                  </p>
                </div>
              </details>

              <details>
                <summary>
                  Milestone 5 — Calibrated Hybrid Recommendation Pipeline (Deferred as Offline Benchmark)
                </summary>
                <div className="px-5 pb-6 pt-2 text-sm leading-7 text-[#686159]">
                  <p>
                    Milestone 5 evaluated a calibrated linear hybrid combining{" "}
                    <code>CLIPLateFusionRecommender</code> (
                    <code>w_content = 0.50</code>),{" "}
                    <code>ItemKNNRecommender</code> (<code>w_cf = 0.35</code>),
                    and conditional <code>AprioriAssociationRuleMiner</code> (
                    <code>w_arm = 0.15</code>) with per-query min-max score
                    normalization.
                  </p>
                  <p className="mt-3">
                    While the hybrid increased catalog diversity (
                    <code>Coverage@5 = 45.73%–46.95%</code>), its Track 1 Test
                    Protocol A1 ranking accuracy (
                    <code>Graded nDCG@5 = 0.0501</code> for Content+CF and{" "}
                    <code>0.0493</code> for Content+CF+ARM) did not surpass
                    standalone <code>ItemKNN_CF</code> (<code>0.0514</code>),
                    and Track 2 basket completion reached{" "}
                    <code>HitRate@5 = 6.45%</code> (<code>nDCG@5 = 0.0379</code>
                    ). Keeping PDP similarity (<code>CLIP Late Fusion</code>)
                    and synthetic profile personalization (<code>ItemKNN</code>)
                    as dedicated, task-specific services yielded higher
                    precision and clearer user transparency.
                  </p>
                </div>
              </details>

              <details>
                <summary>
                  Earlier Content Rerankers (V2, V3, V4) &amp; Phase 4A Handcrafted Multimodal Models
                </summary>
                <div className="px-5 pb-6 pt-2 text-sm leading-7 text-[#686159]">
                  <p>
                    Before adopting pretrained <code>clip-vit-B-32</code>{" "}
                    embeddings in Phase 4B, earlier experiments tested
                    rule-based subtype rerankers (V2–V4) and lightweight
                    handcrafted HSV/edge visual features (Phase 4A). Phase 4B{" "}
                    <code>CLIP Late Fusion</code> improved Holdout-18{" "}
                    <code>nDCG@5</code> from <code>0.6967</code> (Exact V1
                    TF-IDF) and <code>0.7282</code> (Phase 4A Late Fusion) to{" "}
                    <code>0.7625</code> (<code>+9.44%</code> over V1) while
                    lifting <code>Strict P@5</code> to <code>26.67%</code>.
                  </p>
                </div>
              </details>
            </div>
          </div>
        </section>

        {/* 5. Technology & Architecture */}
        <section
          id="architecture"
          className="border-y border-[#e7e2dc] bg-white py-16 md:py-20"
        >
          <div className="elan-container">
            <div className="max-w-2xl">
              <p className="text-xs font-semibold uppercase tracking-[0.25em] text-[#9a6b48]">
                04 · System Design
              </p>
              <h2 className="mt-3 font-serif text-3xl sm:text-4xl">
                Technology &amp; Architecture
              </h2>
              <p className="mt-4 text-sm leading-7 text-[#686159] sm:text-base">
                Designed for low-cost, deterministic cloud deployment by
                separating offline embedding extraction and model fitting from
                lightweight runtime serving.
              </p>
            </div>

            <div className="mt-10 grid gap-6 md:grid-cols-3">
              <div className="border border-[#e7e2dc] bg-[#faf9f6] p-7">
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#9a6b48]">
                  Frontend Layer · Vercel
                </p>
                <h3 className="mt-3 font-serif text-2xl">
                  Next.js 16 &amp; TypeScript
                </h3>
                <ul className="mt-4 space-y-2 pl-4 text-sm leading-6 text-[#686159]">
                  <li>Next.js 16 App Router with React 19 and Tailwind CSS 4</li>
                  <li>
                    Server-side route handlers (<code>/api/products</code>,{" "}
                    <code>/api/recommendations/[productId]</code>,{" "}
                    <code>/api/recommendations/personalized</code>) proxying to{" "}
                    <code>ELAN_BACKEND_URL</code>
                  </li>
                  <li>
                    Browser <code>localStorage</code> persistence for the
                    simulated shopping bag and guest profile
                  </li>
                </ul>
              </div>

              <div className="border border-[#e7e2dc] bg-[#faf9f6] p-7">
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#9a6b48]">
                  Backend Service · Render
                </p>
                <h3 className="mt-3 font-serif text-2xl">
                  FastAPI &amp; SQLite Snapshot
                </h3>
                <ul className="mt-4 space-y-2 pl-4 text-sm leading-6 text-[#686159]">
                  <li>
                    Python FastAPI + SQLModel / SQLAlchemy served via Uvicorn (
                    <code>GET /health</code>, <code>GET /products</code>,{" "}
                    <code>GET /recommendations/*</code>)
                  </li>
                  <li>
                    Versioned release database (
                    <code>backend/data/elan_release.db</code>,{" "}
                    <code>ELAN_CATALOG_MODE=release</code>) with 328 active
                    products and 343 variants
                  </li>
                  <li>
                    Header-authenticated admin endpoints (
                    <code>x-admin-key</code>) restricted to local catalog
                    management
                  </li>
                </ul>
              </div>

              <div className="border border-[#e7e2dc] bg-[#faf9f6] p-7">
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#9a6b48]">
                  Inference Artifacts
                </p>
                <h3 className="mt-3 font-serif text-2xl">
                  Zero-Torch Runtime Serving
                </h3>
                <ul className="mt-4 space-y-2 pl-4 text-sm leading-6 text-[#686159]">
                  <li>
                    Precomputed <code>clip_vitb32_embeddings.npz</code> bundle
                    loaded with NumPy + scikit-learn for sub-millisecond cosine
                    fusion
                  </li>
                  <li>
                    Precomputed <code>itemknn_personalized_demo.json</code>{" "}
                    joined live against active SQLite catalog records
                  </li>
                  <li>
                    Automatic per-product and service-level fallback to{" "}
                    <code>tfidf_cosine_v1</code>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* 6. Academic Simulation & Documentation */}
        <section id="provenance" className="elan-container py-16 md:py-20">
          <div className="grid gap-10 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,0.8fr)] lg:gap-14">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.25em] text-[#9a6b48]">
                05 · Transparency &amp; Governance
              </p>
              <h2 className="mt-3 font-serif text-3xl sm:text-4xl">
                Academic Simulation &amp; Data Provenance
              </h2>
              <p className="mt-4 text-sm leading-7 text-[#686159] sm:text-base">
                To uphold scientific integrity, all synthetic datasets,
                historical evaluation labels, and storefront boundaries are
                explicitly documented:
              </p>

              <ul className="mt-6 space-y-3.5 pl-5 text-sm leading-7 text-[#686159]">
                <li>
                  <strong className="text-[#24221f]">
                    Synthetic Interaction Profiles (
                    <code>SYNTHETIC_SIMULATION_V5</code>):
                  </strong>{" "}
                  All 600 shopper profiles, 3,827 sessions, 23,335 interaction
                  events, and 1,470 orders were generated deterministically
                  (seed <code>20260501</code>). Latent generator variables are
                  quarantined in <code>backend/evaluation/v5/audit/</code> to
                  prevent training/test leakage.
                </li>
                <li>
                  <strong className="text-[#24221f]">
                    Catalog &amp; Image Provenance:
                  </strong>{" "}
                  The release catalog comprises 320 sampled articles from the{" "}
                  <em>H&amp;M Personalized Fashion Recommendations</em> dataset
                  and 8 original ÉLAN demo products, backed by 335 tracked
                  product image files under <code>frontend/public/products/</code>{" "}
                  (341 / 343 variants have image URLs; 2 variants intentionally
                  omit image URLs to validate fallback handling). Homepage
                  editorial photography uses Unsplash imagery.
                </li>
                <li>
                  <strong className="text-[#24221f]">
                    Illustrative Pricing &amp; Simulated Checkout:
                  </strong>{" "}
                  All product prices are illustrative. The shopping bag saves
                  items locally in your browser; no real orders, payments, or
                  shipments are processed.
                </li>
                <li>
                  <strong className="text-[#24221f]">
                    Known Evaluation Limitations:
                  </strong>{" "}
                  Holdout-18 relevance labels were assigned by an AI evaluator
                  from product imagery and titles without human annotator
                  validation, and synthetic interaction metrics measure recovery
                  of simulated persona rules rather than real-world customer
                  conversion uplift.
                </li>
              </ul>
            </div>

            <aside className="flex flex-col justify-between border border-[#e7e2dc] bg-[#f2ece4] p-7 sm:p-8">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[#9a6b48]">
                  Repository &amp; Documentation
                </p>
                <h3 className="mt-3 font-serif text-2xl text-[#24221f]">
                  Inspect the Source Code &amp; Benchmark Reports
                </h3>
                <p className="mt-3 text-sm leading-7 text-[#686159]">
                  Full evaluation scripts, SHA-256 verified JSON metrics, frozen
                  candidate pools, and reproducible PowerShell commands are
                  committed in the public GitHub repository under{" "}
                  <code>backend/evaluation/</code> and{" "}
                  <code>backend/evaluation/v5/</code>.
                </p>
              </div>

              <div className="mt-8 flex flex-col gap-3">
                <a
                  href="https://github.com/vukhiem2005/ai-fashion-recommender"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="elan-btn-primary w-full text-center"
                >
                  Open GitHub Repository ↗
                </a>
                <Link
                  href="/shop"
                  className="elan-btn-secondary w-full text-center"
                >
                  Browse the Collection
                </Link>
              </div>
            </aside>
          </div>
        </section>
      </main>

      <StoreFooter />
    </div>
  );
}
