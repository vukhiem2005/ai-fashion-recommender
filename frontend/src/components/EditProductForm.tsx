
"use client";

import { useState, type FormEvent } from "react";

const categories = [
  "Outerwear",
  "Tops",
  "Bottoms",
  "Dresses",
  "Knitwear",
  "Accessories",
] as const;

interface EditableProduct {
  id: string;
  name: string;
  category: string;
  description: string;
  price_cents: number;
}

interface EditProductFormProps {
  product: EditableProduct;
  adminKey: string;
  onSaved: () => void | Promise<void>;
  onClose: () => void;
}

export default function EditProductForm({
  product,
  adminKey,
  onSaved,
  onClose,
}: EditProductFormProps) {
  const [name, setName] = useState(product.name);
  const [category, setCategory] = useState(
    product.category
  );
  const [description, setDescription] = useState(
    product.description
  );
  const [price, setPrice] = useState(
    (product.price_cents / 100).toFixed(2)
  );

  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(
    event: FormEvent<HTMLFormElement>
  ) {
    event.preventDefault();
    setError("");

    const productName = name.trim();
    const productDescription = description.trim();
    const priceText = price.trim();

    if (!adminKey.trim()) {
      setError("Admin Key is required.");
      return;
    }

    if (!productName || !productDescription) {
      setError("Name and description are required.");
      return;
    }

    if (!/^\d+(?:\.\d{1,2})?$/.test(priceText)) {
      setError("Enter a valid USD price.");
      return;
    }

    const [dollars, fraction = ""] =
      priceText.split(".");

    const priceCents =
      Number(dollars) * 100 +
      Number(fraction.padEnd(2, "0"));

    if (
      !Number.isSafeInteger(priceCents) ||
      priceCents <= 0
    ) {
      setError("Price must be greater than zero.");
      return;
    }

    setSaving(true);

    try {
      const response = await fetch(
        `/api/admin/products/${encodeURIComponent(product.id)}`,
        {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
            "x-admin-key": adminKey,
          },
          body: JSON.stringify({
            name: productName,
            category,
            description: productDescription,
            price_cents: priceCents,
          }),
          cache: "no-store",
        }
      );

      const result: unknown = await response.json();

      if (!response.ok) {
        const message =
          typeof result === "object" &&
          result !== null &&
          "detail" in result &&
          typeof result.detail === "string"
            ? result.detail
            : "Failed to update product.";

        throw new Error(message);
      }

      await onSaved();
      onClose();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "An unexpected error occurred."
      );
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-black/50 px-4 py-8">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="edit-product-title"
        className="mx-auto w-full max-w-2xl border border-[#e7e2dc] bg-[#faf9f6] p-6 shadow-xl md:p-8"
      >
        <div className="mb-8 flex items-start justify-between gap-4">
          <div>
            <p className="mb-2 text-xs uppercase tracking-[0.2em] text-[#ae805f]">
              Catalog Management
            </p>

            <h2
              id="edit-product-title"
              className="font-serif text-3xl"
            >
              Edit Product
            </h2>

            <p className="mt-2 text-sm text-[#817a72]">
              Product ID: {product.id}
            </p>
          </div>

          <button
            type="button"
            onClick={onClose}
            disabled={saving}
            aria-label="Close edit form"
            className="text-2xl text-[#817a72] hover:text-black disabled:opacity-50"
          >
            ×
          </button>
        </div>

        <form
          onSubmit={handleSubmit}
          className="space-y-6"
        >
          <label className="block text-sm">
            Product Name
            <input
              required
              value={name}
              onChange={(event) =>
                setName(event.target.value)
              }
              className="mt-2 w-full border border-[#ddd6cf] bg-white p-3 outline-none focus:border-[#ae805f]"
            />
          </label>

          <div className="grid gap-5 md:grid-cols-2">
            <label className="block text-sm">
              Category
              <select
                value={category}
                onChange={(event) =>
                  setCategory(event.target.value)
                }
                className="mt-2 w-full border border-[#ddd6cf] bg-white p-3"
              >
                {categories.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>

            <label className="block text-sm">
              Price (USD)
              <input
                required
                value={price}
                onChange={(event) =>
                  setPrice(event.target.value)
                }
                inputMode="decimal"
                className="mt-2 w-full border border-[#ddd6cf] bg-white p-3 outline-none focus:border-[#ae805f]"
              />
            </label>
          </div>

          <label className="block text-sm">
            Description
            <textarea
              required
              value={description}
              onChange={(event) =>
                setDescription(event.target.value)
              }
              rows={4}
              className="mt-2 w-full border border-[#ddd6cf] bg-white p-3 outline-none focus:border-[#ae805f]"
            />
          </label>

          {error && (
            <p
              role="alert"
              className="border border-red-200 bg-red-50 p-3 text-sm text-red-700"
            >
              {error}
            </p>
          )}

          <div className="flex justify-end gap-3 border-t border-[#e7e2dc] pt-6">
            <button
              type="button"
              onClick={onClose}
              disabled={saving}
              className="border border-[#ddd6cf] px-6 py-3 text-sm disabled:opacity-50"
            >
              Cancel
            </button>

            <button
              type="submit"
              disabled={saving}
              className="bg-[#24221f] px-6 py-3 text-sm font-medium text-white disabled:opacity-50"
            >
              {saving
                ? "Saving..."
                : "Save Changes"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
