"use client";

import StoreFooter from "@/components/StoreFooter";
import StoreHeader from "@/components/StoreHeader";
import Link from "next/link";
import { useEffect, useState, type FormEvent } from "react";
import { readCart, subscribeToCart } from "@/lib/cart";

const PROFILE_KEY = "elan_demo_profile_v1";

export default function AccountPage() {
  const [displayName, setDisplayName] = useState("");
  const [savedName, setSavedName] = useState<string | null>(null);
  const [itemCount, setItemCount] = useState(0);
  const [loaded, setLoaded] = useState(false);
  const [feedback, setFeedback] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);

  useEffect(() => {
    function initialize() {
      try {
        const storedName = window.localStorage.getItem(PROFILE_KEY);
        const name = storedName?.trim().slice(0, 40) || null;

        setSavedName(name);
        setDisplayName(name ?? "");
      } catch {
        setFeedback({
          type: "error",
          text: "Unable to access saved profile preferences.",
        });
      }

      const items = readCart();
      setItemCount(
        items.reduce((total, item) => total + item.quantity, 0)
      );
      setLoaded(true);
    }

    const frame = window.requestAnimationFrame(initialize);

    const unsubscribe = subscribeToCart(() => {
      const items = readCart();
      setItemCount(
        items.reduce((total, item) => total + item.quantity, 0)
      );
    });

    return () => {
      window.cancelAnimationFrame(frame);
      unsubscribe();
    };
  }, []);

  function saveProfile(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const name = displayName.trim().replace(/\s+/g, " ");

    if (name.length < 2) {
      setFeedback({
        type: "error",
        text: "Please enter at least two characters for your display name.",
      });
      return;
    }

    try {
      window.localStorage.setItem(PROFILE_KEY, name);
      setSavedName(name);
      setDisplayName(name);
      setFeedback({
        type: "success",
        text: "Your display name has been saved in this browser.",
      });
    } catch {
      setFeedback({
        type: "error",
        text: "Unable to save your display name.",
      });
    }
  }

  function clearProfile() {
    try {
      window.localStorage.removeItem(PROFILE_KEY);
      setSavedName(null);
      setDisplayName("");
      setFeedback({
        type: "success",
        text: "Display name cleared.",
      });
    } catch {
      setFeedback({
        type: "error",
        text: "Unable to clear your display name.",
      });
    }
  }

  return (
    <div className="flex min-h-screen flex-col bg-[#faf9f6] text-[#24221f]">
      <StoreHeader />

      <main id="main-content" className="elan-container flex-1 py-10 md:py-16">
        <div className="max-w-2xl">
          <p className="text-xs font-semibold uppercase tracking-[0.26em] text-[#9a6b48]">
            Personal Preferences
          </p>

          <h1 className="mt-3 font-serif text-4xl sm:text-5xl">
            Your Account
          </h1>

          <p className="mt-4 text-sm leading-7 text-[#686159] sm:text-base">
            Manage your local display preferences and review your shopping bag.
          </p>
        </div>

        {!loaded ? (
          <p role="status" className="mt-12 text-sm text-[#686159]">
            Loading your preferences...
          </p>
        ) : (
          <div className="mt-10 grid items-start gap-8 lg:grid-cols-[minmax(0,1fr)_340px] lg:gap-12">
            {/* Primary Section: Your Profile */}
            <section
              aria-labelledby="profile-section-heading"
              className="border border-[#e7e2dc] bg-white p-6 sm:p-9"
            >
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[#9a6b48]">
                Guest Profile
              </p>

              <h2
                id="profile-section-heading"
                className="mt-3 font-serif text-2xl sm:text-3xl"
              >
                {savedName ? `Welcome, ${savedName}` : "Your Profile"}
              </h2>

              <p className="mt-3 text-sm leading-7 text-[#686159]">
                {savedName
                  ? "Your display name is stored locally in this browser for your shopping session."
                  : "Set a preferred display name for this browser, or continue browsing as a guest."}
              </p>

              <form onSubmit={saveProfile} className="mt-7 max-w-md">
                <label
                  htmlFor="display-name"
                  className="block text-xs font-semibold uppercase tracking-[0.18em] text-[#24221f]"
                >
                  Display Name
                </label>

                <input
                  id="display-name"
                  name="displayName"
                  type="text"
                  value={displayName}
                  onChange={(event) => {
                    setDisplayName(event.target.value);
                    if (feedback?.type === "error") {
                      setFeedback(null);
                    }
                  }}
                  maxLength={40}
                  autoComplete="name"
                  placeholder="Enter your preferred name"
                  aria-describedby="display-name-hint"
                  className="mt-2 min-h-11 w-full border border-[#d5cdc3] bg-[#faf9f6] px-4 py-2.5 text-sm text-[#24221f] transition-colors focus:border-[#9a6b48] focus:bg-white"
                />

                <p
                  id="display-name-hint"
                  className="mt-2 text-xs text-[#756d65]"
                >
                  2–40 characters. Saved locally in your browser without
                  registration or passwords.
                </p>

                <div className="mt-6 flex flex-wrap items-center gap-3">
                  <button type="submit" className="elan-btn-primary">
                    Save Profile
                  </button>

                  {savedName && (
                    <button
                      type="button"
                      onClick={clearProfile}
                      className="elan-btn-secondary"
                    >
                      Clear Profile
                    </button>
                  )}
                </div>
              </form>

              {feedback && (
                <p
                  role={feedback.type === "error" ? "alert" : "status"}
                  aria-live="polite"
                  className={`mt-5 border px-4 py-3 text-sm ${
                    feedback.type === "error"
                      ? "border-red-200 bg-red-50 text-red-700"
                      : "border-[#d5cdc3] bg-[#f4efe8] text-[#24221f]"
                  }`}
                >
                  {feedback.text}
                </p>
              )}

              <p className="mt-8 border-t border-[#e7e2dc] pt-5 text-xs leading-6 text-[#756d65]">
                Local guest profile only. This simulation does not create an
                authenticated customer account; please do not enter sensitive
                personal information.
              </p>
            </section>

            {/* Secondary Section: Your Shopping Bag */}
            <aside
              aria-labelledby="account-bag-heading"
              className="border border-[#e7e2dc] bg-white p-6 sm:p-8"
            >
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[#9a6b48]">
                Bag Summary
              </p>

              <h2
                id="account-bag-heading"
                className="mt-3 font-serif text-2xl sm:text-3xl"
              >
                Your Shopping Bag
              </h2>

              <p className="mt-3 text-sm leading-6 text-[#686159]">
                {itemCount === 0
                  ? "Your shopping bag is currently empty."
                  : `You have ${itemCount} ${
                      itemCount === 1 ? "piece" : "pieces"
                    } saved in your shopping bag.`}
              </p>

              <div className="mt-7 flex flex-col gap-3">
                <Link
                  href="/cart"
                  className="elan-btn-primary w-full text-center"
                >
                  View Shopping Bag ({itemCount})
                </Link>

                <Link
                  href="/shop"
                  className="elan-btn-secondary w-full text-center"
                >
                  Explore Collection
                </Link>
              </div>
            </aside>
          </div>
        )}

        {/* Discreet Administrator Access Footer Section */}
        <div className="mt-16 flex flex-col items-start justify-between gap-3 border-t border-[#e7e2dc] pt-6 text-xs text-[#756d65] sm:flex-row sm:items-center">
          <p className="m-0">
            Catalog management tools are restricted to authorized local
            administrators.
          </p>
          <Link
            href="/admin/products"
            className="inline-flex min-h-9 items-center gap-1.5 font-medium uppercase tracking-[0.16em] text-[#686159] underline underline-offset-4 transition-colors hover:text-[#24221f]"
          >
            <span>Catalog Administration</span>
            <span aria-hidden="true">→</span>
          </Link>
        </div>
      </main>

      <StoreFooter />
    </div>
  );
}
