
"use client";

import { useState } from "react";

const categories = [
  "Outerwear",
  "Tops",
  "Bottoms",
  "Dresses",
  "Knitwear",
  "Accessories",
] as const;

const availableSizes = [
  "XS",
  "S",
  "M",
  "L",
  "XL",
  "One Size",
];

interface AddProductFormProps {
  adminKey: string;
  onCreated: () => void;
}

function slugify(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/đ/g, "d")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

function skuCode(value: string): string {
  return slugify(value).toUpperCase();
}

export default function AddProductForm({
  adminKey,
  onCreated,
}: AddProductFormProps) {
  const [id, setId] = useState("");
  const [name, setName] = useState("");
  const [category, setCategory] =
    useState<(typeof categories)[number]>("Tops");

  const [description, setDescription] = useState("");
  const [price, setPrice] = useState("");
  const [color, setColor] = useState("");
  const [sizes, setSizes] = useState<string[]>([]);

  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  function toggleSize(size: string) {
    setSizes((current) =>
      current.includes(size)
        ? current.filter((item) => item !== size)
        : [...current, size]
    );
  }

  async function createProduct() {
    setError("");
    setSuccess("");

    const productId = id.trim().toUpperCase();
    const productName = name.trim();
    const productColor = color.trim();
    const slug = slugify(productName);

    if (!adminKey.trim()) {
      setError("Please enter your Admin Key first.");
      return;
    }

    if (!/^ELAN-[A-Z0-9]+(?:-[A-Z0-9]+)*$/.test(productId)) {
      setError("Product ID must look like ELAN-007.");
      return;
    }

    if (!productName || !slug) {
      setError("Please enter a valid product name.");
      return;
    }

    if (!description.trim()) {
      setError("Please enter a product description.");
      return;
    }

    if (!/^\d+(?:\.\d{1,2})?$/.test(price.trim())) {
      setError("Enter a valid USD price, such as 24.99.");
      return;
    }

    const [dollars, fraction = ""] = price.trim().split(".");
    const priceCents =
      Number(dollars) * 100 +
      Number(fraction.padEnd(2, "0"));

    if (!Number.isSafeInteger(priceCents) || priceCents <= 0) {
      setError("Product price must be greater than zero.");
      return;
    }

    if (!productColor || !skuCode(productColor)) {
      setError("Please enter a valid color.");
      return;
    }

    if (sizes.length === 0) {
      setError("Select at least one size.");
      return;
    }

    const payload = {
      id: productId,
      slug,
      name: productName,
      category,
      description: description.trim(),
      price_cents: priceCents,
      variants: sizes.map((size) => ({
        sku: `${productId}-${skuCode(productColor)}-${skuCode(size)}`,
        color: productColor,
        size,
        image_url: null,
      })),
    };

    setSaving(true);

    try {
      const response = await fetch("/api/admin/products", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-admin-key": adminKey,
        },
        body: JSON.stringify(payload),
        cache: "no-store",
      });

      const result: unknown = await response.json();

      if (!response.ok) {
        let message = "Unable to create product.";

        if (
          typeof result === "object" &&
          result !== null &&
          "detail" in result
        ) {
          if (typeof result.detail === "string") {
            message = result.detail;
          } else if (Array.isArray(result.detail)) {
            message = JSON.stringify(result.detail);
          }
        }

        throw new Error(message);
      }

      setSuccess(
        `Product ${productId} created successfully.`
      );

      setId("");
      setName("");
      setDescription("");
      setPrice("");
      setColor("");
      setSizes([]);

      onCreated();
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
    <section className="border border-[#e7e2dc] bg-white p-6 md:p-8">
      <div className="mb-8">
        <p className="mb-2 text-xs uppercase tracking-[0.2em] text-[#ae805f]">
          Catalog Management
        </p>

        <h2 className="font-serif text-3xl">
          Add New Product
        </h2>

        <p className="mt-3 text-sm text-[#817a72]">
          Create a product with one color and multiple
          sizes. Product images can be added later.
        </p>
      </div>

      <form
        onSubmit={(event) => {
          event.preventDefault();
          void createProduct();
        }}
        className="space-y-6"
      >
        <div className="grid gap-5 md:grid-cols-2">
          <label className="block text-sm">
            Product ID
            <input
              required
              value={id}
              onChange={(event) => setId(event.target.value)}
              placeholder="ELAN-007"
              className="mt-2 w-full border border-[#ddd6cf] p-3 outline-none focus:border-[#ae805f]"
            />
          </label>

          <label className="block text-sm">
            Product Name
            <input
              required
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="Relaxed Cotton Tee"
              className="mt-2 w-full border border-[#ddd6cf] p-3 outline-none focus:border-[#ae805f]"
            />
          </label>

          <label className="block text-sm">
            Category
            <select
              value={category}
              onChange={(event) =>
                setCategory(
                  event.target.value as typeof category
                )
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
              onChange={(event) => setPrice(event.target.value)}
              placeholder="24.99"
              inputMode="decimal"
              className="mt-2 w-full border border-[#ddd6cf] p-3 outline-none focus:border-[#ae805f]"
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
            placeholder="Describe this fashion product..."
            rows={3}
            className="mt-2 w-full border border-[#ddd6cf] p-3 outline-none focus:border-[#ae805f]"
          />
        </label>

        <label className="block text-sm">
          Product Color
          <input
            required
            value={color}
            onChange={(event) => setColor(event.target.value)}
            placeholder="White"
            className="mt-2 w-full border border-[#ddd6cf] p-3 outline-none focus:border-[#ae805f]"
          />
        </label>

        <fieldset>
          <legend className="mb-3 text-sm">
            Available Sizes
          </legend>

          <div className="flex flex-wrap gap-3">
            {availableSizes.map((size) => (
              <button
                key={size}
                type="button"
                onClick={() => toggleSize(size)}
                aria-pressed={sizes.includes(size)}
                className={`border px-4 py-2 text-sm ${
                  sizes.includes(size)
                    ? "border-[#24221f] bg-[#24221f] text-white"
                    : "border-[#ddd6cf] bg-white"
                }`}
              >
                {size}
              </button>
            ))}
          </div>
        </fieldset>

        {name.trim() && (
          <p className="text-xs text-[#817a72]">
            Generated slug: {slugify(name)}
          </p>
        )}

        {error && (
          <p role="alert" className="text-sm text-red-700">
            {error}
          </p>
        )}

        {success && (
          <p role="status" className="text-sm text-green-700">
            {success}
          </p>
        )}

        <button
          type="submit"
          disabled={saving}
          className="bg-[#24221f] px-8 py-4 text-sm font-medium text-white transition hover:bg-[#ae805f] disabled:opacity-50"
        >
          {saving ? "Creating Product..." : "Create Product"}
        </button>
      </form>
    </section>
  );
}
