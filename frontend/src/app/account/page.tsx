
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
  const [message, setMessage] = useState("");

  useEffect(() => {
    function initialize() {
      try {
        const storedName = window.localStorage.getItem(PROFILE_KEY);
        const name = storedName?.trim().slice(0, 40) || null;

        setSavedName(name);
        setDisplayName(name ?? "");
      } catch {
        setMessage("Unable to access saved demo profile.");
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
      setMessage("Please enter at least two characters.");
      return;
    }

    try {
      window.localStorage.setItem(PROFILE_KEY, name);
      setSavedName(name);
      setDisplayName(name);
      setMessage("Your demo profile has been saved.");
    } catch {
      setMessage("Unable to save your demo profile.");
    }
  }

  function clearProfile() {
    try {
      window.localStorage.removeItem(PROFILE_KEY);
      setSavedName(null);
      setDisplayName("");
      setMessage("Demo profile cleared. You can continue as a guest.");
    } catch {
      setMessage("Unable to clear your demo profile.");
    }
  }

  return (
    <div className="flex min-h-screen flex-col bg-[#faf9f6] text-[#24221f]">
      <StoreHeader />
      <main id="main-content" className="mx-auto w-full max-w-5xl flex-1 px-6 py-14 md:py-20">
        <p className="text-xs font-semibold uppercase tracking-[0.25em] text-[#ae805f]">
          ÉLAN Member Experience
        </p>

        <h1 className="mt-4 font-serif text-4xl md:text-5xl">
          Your Account
        </h1>

<p className="mt-5 max-w-2xl text-sm leading-7 text-[#77716b]">
  Choose how you would like to explore ÉLAN.
  Continue as a guest or access the product administration area.
</p>

<div className="mt-9 grid gap-4 sm:grid-cols-2">
  <a
    href="#customer-profile"
    className="border border-[#e7e2dc] bg-white p-6 transition hover:border-[#ae805f]"
  >
    <p className="text-xs font-semibold uppercase tracking-widest text-[#ae805f]">
      Customer Experience
    </p>

    <h2 className="mt-3 font-serif text-2xl">
      Continue as Guest
    </h2>

    <p className="mt-3 text-sm leading-6 text-[#77716b]">
      Create a local demo profile and explore the collection.
      No password or registration required.
    </p>

    <p className="mt-5 text-sm font-semibold">
      Explore Customer Profile →
    </p>
  </a>

  <Link
    href="/admin/products"
    className="border border-[#e7e2dc] bg-white p-6 transition hover:border-[#ae805f]"
  >
    <p className="text-xs font-semibold uppercase tracking-widest text-[#ae805f]">
      Store Management
    </p>

    <h2 className="mt-3 font-serif text-2xl">
      Administrator
    </h2>

    <p className="mt-3 text-sm leading-6 text-[#77716b]">
      Manage products, prices and product images.
      An administrator key is required.
    </p>

    <p className="mt-5 text-sm font-semibold">
      Admin Access →
    </p>
  </Link>
</div>
        {!loaded ? (
          <p role="status" className="mt-12 text-sm text-[#817a72]">
            Loading demo profile...
          </p>
        ) : (
          <div className="mt-12 grid gap-8 md:grid-cols-[minmax(0,1fr)_300px]">
            <section
  id="customer-profile"
  className="scroll-mt-8 border border-[#e7e2dc] bg-white p-7 md:p-10"
>
            
              <p className="text-xs font-semibold uppercase tracking-widest text-[#ae805f]">
                Customer Profile
              </p>

              <h2 className="mt-4 font-serif text-3xl">
                {savedName ? `Welcome, ${savedName}` : "Continue as Guest"}
              </h2>

              <p className="mt-4 text-sm leading-7 text-[#77716b]">
                {savedName
                  ? "Your demo display name is saved in this browser."
                  : "Save a display name or continue browsing as a guest."}
              </p>

              <form onSubmit={saveProfile} className="mt-9">
                <label
                  htmlFor="display-name"
                  className="block text-sm font-medium"
                >
                  Display name
                </label>

                <input
                  id="display-name"
                  type="text"
                  value={displayName}
                  onChange={(event) => setDisplayName(event.target.value)}
                  maxLength={40}
                  autoComplete="off"
                  placeholder="Enter your display name"
                  className="mt-3 w-full border border-[#ddd6cf] bg-[#faf9f6] px-4 py-3 text-sm outline-none focus:border-[#ae805f]"
                />

                <button
                  type="submit"
                  className="mt-6 bg-[#24221f] px-7 py-3 text-sm font-medium text-white transition hover:bg-[#ae805f]"
                >
                  Save Demo Profile
                </button>

                {savedName && (
                  <button
                    type="button"
                    onClick={clearProfile}
                    className="ml-4 mt-4 text-sm text-[#817a72] underline hover:text-[#24221f]"
                  >
                    Clear Profile
                  </button>
                )}
              </form>

              {message && (
                <p role="status" aria-live="polite" className="mt-5 text-sm text-[#706961]">
                  {message}
                </p>
              )}

              <p className="mt-9 border-t border-[#e7e2dc] pt-6 text-xs leading-6 text-[#817a72]">
                Demo profile only. This does not create an authenticated
                account. Do not enter sensitive personal information.
              </p>
            </section>

            <aside className="h-fit border border-[#e7e2dc] bg-white p-7">
              <p className="text-xs font-semibold uppercase tracking-widest text-[#ae805f]">
                Shopping Overview
              </p>

              <h2 className="mt-4 font-serif text-2xl">
                Your Shopping Bag
              </h2>

              <p className="mt-5 text-sm text-[#77716b]">
                {itemCount} {itemCount === 1 ? "item" : "items"} saved
              </p>

              <Link
                href="/cart"
                className="mt-7 block bg-[#24221f] px-5 py-4 text-center text-sm font-medium text-white hover:bg-[#ae805f]"
              >
                View Shopping Bag
              </Link>

              <Link
                href="/shop"
                className="mt-4 block text-center text-sm text-[#817a72] underline hover:text-[#24221f]"
              >
                Explore Collection
              </Link>
            </aside>
          </div>
        )}
      </main>

      <StoreFooter />
    </div>
  );
}
