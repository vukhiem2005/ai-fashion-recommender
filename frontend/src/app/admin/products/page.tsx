
"use client";

import Link from "next/link";
import { useState } from "react";
import AddProductForm from "@/components/AddProductForm";
import EditProductForm from "@/components/EditProductForm";
import ManageProductImages from "@/components/ManageProductImages";

interface AdminVariant {
  sku: string;
  color: string;
  size: string;
  image_url: string | null;
}

interface AdminProduct {
  id: string;
  slug: string;
  name: string;
  category: string;
  description: string;
  price_cents: number;
  is_active: boolean;
  variants: AdminVariant[];
}

type StatusFilter = "all" | "active" | "inactive";

const currencyFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
});

export default function AdminProductsPage() {
  const [adminKey, setAdminKey] = useState("");
  const [products, setProducts] = useState<
    AdminProduct[] | null
  >(null);

  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] =
    useState<StatusFilter>("all");

  const [loading, setLoading] = useState(false);

const [updatingId, setUpdatingId] = useState<string | null>(
  null
);

const [editingProduct, setEditingProduct] =
  useState<AdminProduct | null>(null);

const [imageProduct, setImageProduct] =
  useState<AdminProduct | null>(null);


  const [error, setError] = useState("");

  async function loadProducts() {
    if (!adminKey.trim()) {
      setError("Please enter your admin key.");
      setProducts(null);
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch("/api/admin/products", {
        headers: {
          "x-admin-key": adminKey,
        },
        cache: "no-store",
      });

      const data: unknown = await response.json();

      if (!response.ok) {
        const message =
          typeof data === "object" &&
          data !== null &&
          "detail" in data &&
          typeof data.detail === "string"
            ? data.detail
            : "Unable to load products.";

        throw new Error(message);
      }

      if (!Array.isArray(data)) {
        throw new Error("Unexpected API response.");
      }

      setProducts(data as AdminProduct[]);
    } catch (err) {
      setProducts(null);
      setError(
        err instanceof Error
          ? err.message
          : "An unexpected error occurred."
      );
    } finally {
      setLoading(false);
    }
  }

async function changeProductStatus(
  product: AdminProduct
) {
  const nextStatus = !product.is_active;
  const action = nextStatus ? "Restore" : "Hide";

  const confirmed = window.confirm(
    `${action} product "${product.name}"?`
  );

  if (!confirmed) {
    return;
  }

  setUpdatingId(product.id);
  setError("");

  try {
    const response = await fetch(
      `/api/admin/products/${encodeURIComponent(product.id)}/status`,
      {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          "x-admin-key": adminKey,
        },
        body: JSON.stringify({
          is_active: nextStatus,
        }),
        cache: "no-store",
      }
    );

    const data: unknown = await response.json();

    if (!response.ok) {
      const message =
        typeof data === "object" &&
        data !== null &&
        "detail" in data &&
        typeof data.detail === "string"
          ? data.detail
          : "Failed to update product status.";

      throw new Error(message);
    }

    setProducts((current) =>
      current === null
        ? null
        : current.map((item) =>
            item.id === product.id
              ? { ...item, is_active: nextStatus }
              : item
          )
    );
  } catch (err) {
    setError(
      err instanceof Error
        ? err.message
        : "An unexpected error occurred."
    );
  } finally {
    setUpdatingId(null);
  }
}

  const allProducts = products ?? [];

  const activeCount = allProducts.filter(
    (product) => product.is_active
  ).length;

  const inactiveCount =
    allProducts.length - activeCount;

  const visibleProducts = allProducts.filter((product) => {
    const query = search.trim().toLowerCase();

    const matchesSearch = [
      product.id,
      product.name,
      product.category,
      product.slug,
    ]
      .join(" ")
      .toLowerCase()
      .includes(query);

    const matchesStatus =
      statusFilter === "all" ||
      (statusFilter === "active" && product.is_active) ||
      (statusFilter === "inactive" && !product.is_active);

    return matchesSearch && matchesStatus;
  });

  return (
    <div className="min-h-screen bg-[#f8f7f5] text-[#24221f]">
      <header className="border-b border-[#e7e2dc] bg-white">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-6">
          <Link
            href="/"
            className="font-serif text-3xl font-bold tracking-widest"
          >
            ÉLAN.
          </Link>

          <span className="text-sm font-medium">
            Administration
          </span>

          <Link
            href="/shop"
            className="text-sm text-[#80776e] hover:text-black"
          >
            View Store
          </Link>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-12">
        <div className="mb-10">
          <p className="mb-3 text-xs font-semibold uppercase tracking-[0.25em] text-[#ae805f]">
            Product Management
          </p>

          <h1 className="font-serif text-4xl md:text-5xl">
            Product Inventory
          </h1>

          <p className="mt-4 text-sm text-[#77716b]">
            Manage your fashion catalog from one place.
          </p>
        </div>

        <section className="mb-10 border border-[#e7e2dc] bg-white p-6">
          <h2 className="mb-3 text-lg font-medium">
            Admin Access
          </h2>

          <p className="mb-5 text-sm text-[#77716b]">
            Enter your local admin key to load inventory
            data from the ELAN Backend.
          </p>

          <form
            onSubmit={(event) => {
              event.preventDefault();
              void loadProducts();
            }}
            className="flex flex-col gap-3 sm:flex-row"
          >
            <input
              type="password"
              value={adminKey}
              onChange={(event) =>
                setAdminKey(event.target.value)
              }
              placeholder="Enter Admin Key"
              aria-label="Admin Key"
              autoComplete="off"
              className="w-full border border-[#ddd6cf] px-4 py-3 text-sm outline-none focus:border-[#ae805f] sm:max-w-md"
            />

            <button
              type="submit"
              disabled={loading}
              className="bg-[#24221f] px-7 py-3 text-sm font-medium text-white disabled:opacity-50"
            >
              {loading ? "Loading..." : "Load Products"}
            </button>
          </form>

          <p className="mt-4 text-xs text-[#817a72]">
            Local development only. The key is kept in
            memory and is not saved to browser storage.
          </p>

          {error && (
            <p
              role="alert"
              className="mt-5 border border-red-200 bg-red-50 p-4 text-sm text-red-700"
            >
              {error}
            </p>
          )}
        </section>


{products !== null && (
  <>
    <div className="mb-10">
      <AddProductForm
        adminKey={adminKey}
        onCreated={() => void loadProducts()}
      />
    </div>



            <section
              aria-label="Inventory statistics"
              className="mb-10 grid gap-5 sm:grid-cols-3"
            >
              {[
                {
                  label: "Total Products",
                  value: allProducts.length,
                },
                {
                  label: "Active Products",
                  value: activeCount,
                },
                {
                  label: "Inactive Products",
                  value: inactiveCount,
                },
              ].map((stat) => (
                <div
                  key={stat.label}
                  className="border border-[#e7e2dc] bg-white p-6"
                >
                  <p className="text-sm text-[#80776e]">
                    {stat.label}
                  </p>

                  <p className="mt-3 text-3xl font-semibold">
                    {stat.value}
                  </p>
                </div>
              ))}
            </section>

            <section className="border border-[#e7e2dc] bg-white p-6">
              <div className="mb-6 flex flex-col justify-between gap-4 md:flex-row">
                <div>
                  <h2 className="text-xl font-semibold">
                    All Products
                  </h2>

                  <p className="mt-2 text-sm text-[#80776e]">
                    Showing {visibleProducts.length} products
                  </p>
                </div>

                <div className="flex flex-col gap-3 sm:flex-row">
                  <input
                    type="search"
                    value={search}
                    onChange={(event) =>
                      setSearch(event.target.value)
                    }
                    placeholder="Search inventory..."
                    aria-label="Search inventory"
                    className="border border-[#ddd6cf] px-4 py-3 text-sm"
                  />

                  <select
                    value={statusFilter}
                    onChange={(event) =>
                      setStatusFilter(
                        event.target.value as StatusFilter
                      )
                    }
                    aria-label="Filter product status"
                    className="border border-[#ddd6cf] px-4 py-3 text-sm"
                  >
                    <option value="all">All Status</option>
                    <option value="active">Active</option>
                    <option value="inactive">Inactive</option>
                  </select>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full min-w-[700px] text-left text-sm">
                  <thead className="border-b border-[#e7e2dc] bg-[#faf9f6]">
                    <tr>
                      <th className="px-4 py-4">Product</th>
                      <th className="px-4 py-4">Category</th>
                      <th className="px-4 py-4">Price</th>
                      <th className="px-4 py-4">Variants</th>
                      <th className="px-4 py-4">Status</th>
                      <th className="px-4 py-4">Actions</th>

                    </tr>
                  </thead>

                  <tbody>
                    {visibleProducts.map((product) => (
                      <tr
                        key={product.id}
                        className="border-b border-[#eeeae5]"
                      >
                        <td className="px-4 py-5">
                          <p className="font-medium">
                            {product.name}
                          </p>
                          <p className="mt-1 text-xs text-[#80776e]">
                            {product.id}
                          </p>
                        </td>

                        <td className="px-4 py-5">
                          {product.category}
                        </td>

                        <td className="px-4 py-5 font-medium">
                          {currencyFormatter.format(
                            product.price_cents / 100
                          )}
                        </td>

                        <td className="px-4 py-5">
                          {product.variants.length}
                        </td>

                        <td className="px-4 py-5">
                          <span
                            className={`inline-block px-3 py-1 text-xs ${
                              product.is_active
                                ? "bg-green-50 text-green-700"
                                : "bg-gray-100 text-gray-600"
                            }`}
                          >
                            {product.is_active
                              ? "Active"
                              : "Inactive"}
                          </span>
                        </td>

<td className="px-4 py-5">

<button
  type="button"
  onClick={() => setImageProduct(product)}
  className="mr-2 border border-[#ddd6cf] px-3 py-2 text-xs font-medium transition hover:border-[#ae805f]"
>
  Images
</button>

<button
  type="button"
  onClick={() => setEditingProduct(product)}
  className="mr-2 border border-[#ddd6cf] px-4 py-2 text-xs font-medium transition hover:border-[#ae805f]"
>
  Edit
</button>

  <button
    type="button"
    onClick={() => void changeProductStatus(product)}
    disabled={updatingId !== null}
    className={`border px-4 py-2 text-xs font-medium transition disabled:opacity-50 ${
      product.is_active
        ? "border-red-200 text-red-700 hover:bg-red-50"
        : "border-green-200 text-green-700 hover:bg-green-50"
    }`}
  >
    {updatingId === product.id
      ? "Updating..."
      : product.is_active
        ? "Hide"
        : "Restore"}
  </button>
</td>

                      </tr>
                    ))}
                  </tbody>
                </table>

                {visibleProducts.length === 0 && (
                  <p className="py-12 text-center text-sm text-[#80776e]">
                    No matching products found.
                  </p>
                )}
              </div>
            </section>


<p className="mt-5 text-xs text-[#817a72]">
  Portfolio demo catalog. Product creation, editing,
  and status management are enabled in local development.
  No real purchases or payments are processed.
</p>

          </>
        )}
      </main>

{editingProduct !== null && (
  <EditProductForm
    key={editingProduct.id}
    product={editingProduct}
    adminKey={adminKey}
    onSaved={() => loadProducts()}
    onClose={() => setEditingProduct(null)}
  />
)}

{imageProduct !== null && (
  <ManageProductImages
    key={imageProduct.id}
    product={imageProduct}
    adminKey={adminKey}
    onSaved={() => loadProducts()}
    onClose={() => setImageProduct(null)}
  />
)}

    </div>
  );
}
