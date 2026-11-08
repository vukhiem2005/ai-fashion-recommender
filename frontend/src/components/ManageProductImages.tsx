"use client";

import { useState } from "react";

interface ImageVariant {
  sku: string;
  color: string;
  size: string;
  image_url: string | null;
}

interface ImageProduct {
  id: string;
  name: string;
  variants: ImageVariant[];
}

interface ManageProductImagesProps {
  product: ImageProduct;
  adminKey: string;
  onSaved: () => void | Promise<void>;
  onClose: () => void;
}

const IMAGE_PATH_PATTERN =
  /^\/products\/[A-Za-z0-9][A-Za-z0-9._-]*\.(jpg|jpeg|png|webp)$/i;

const MAX_IMAGE_SIZE = 5 * 1024 * 1024;
const ACCEPTED_IMAGE_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/webp",
]);

function getErrorMessage(result: unknown, fallback: string): string {
  if (
    result !== null &&
    typeof result === "object" &&
    "detail" in result &&
    typeof result.detail === "string"
  ) {
    return result.detail;
  }

  return fallback;
}

export default function ManageProductImages({
  product,
  adminKey,
  onSaved,
  onClose,
}: ManageProductImagesProps) {
  const [imagePaths, setImagePaths] = useState<Record<string, string>>(
    () =>
      Object.fromEntries(
        product.variants.map((variant) => [
          variant.sku,
          variant.image_url ?? "",
        ])
      )
  );

  const [selectedFiles, setSelectedFiles] = useState<
    Record<string, File | null>
  >({});
  const [savingSku, setSavingSku] = useState<string | null>(null);
  const [uploadingSku, setUploadingSku] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const isBusy = savingSku !== null || uploadingSku !== null;

  async function saveImage(sku: string) {
    const imagePath = (imagePaths[sku] ?? "").trim();

    if (imagePath !== "" && !IMAGE_PATH_PATTERN.test(imagePath)) {
      setError(
        "Image path must start with /products/ " +
          "and end with .jpg, .jpeg, .png or .webp."
      );
      setSuccess(null);
      return;
    }

    const imageUrl = imagePath === "" ? null : imagePath;

    setSavingSku(sku);
    setError(null);
    setSuccess(null);

    try {
      const response = await fetch(
        `/api/admin/products/${encodeURIComponent(
          product.id
        )}/variants/${encodeURIComponent(sku)}/image`,
        {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
            "X-Admin-Key": adminKey,
          },
          body: JSON.stringify({ image_url: imageUrl }),
        }
      );

      const result: unknown = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(getErrorMessage(result, "Unable to update image."));
      }

      await onSaved();
      setSuccess(`Image updated for SKU: ${sku}`);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Unable to update image."
      );
    } finally {
      setSavingSku(null);
    }
  }

  async function uploadImage(sku: string) {
    const file = selectedFiles[sku];

    if (!file) {
      setError("Choose an image file before uploading.");
      setSuccess(null);
      return;
    }

    if (!ACCEPTED_IMAGE_TYPES.has(file.type)) {
      setError("Only JPG, PNG and WebP images are supported.");
      setSuccess(null);
      return;
    }

    if (file.size === 0 || file.size > MAX_IMAGE_SIZE) {
      setError("Image must be non-empty and at most 5 MB.");
      setSuccess(null);
      return;
    }

    const formData = new FormData();
    formData.append("image", file);

    setUploadingSku(sku);
    setError(null);
    setSuccess(null);

    try {
      const response = await fetch(
        `/api/admin/products/${encodeURIComponent(
          product.id
        )}/variants/${encodeURIComponent(sku)}/upload`,
        {
          method: "POST",
          headers: { "X-Admin-Key": adminKey },
          body: formData,
        }
      );

      const result: unknown = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(getErrorMessage(result, "Unable to upload image."));
      }

      if (
        result === null ||
        typeof result !== "object" ||
        !("image_url" in result) ||
        typeof result.image_url !== "string"
      ) {
        throw new Error("Upload response did not include an image path.");
      }

      const uploadedPath = result.image_url;
      setImagePaths((previous) => ({
        ...previous,
        [sku]: uploadedPath,
      }));

      await onSaved();
      setSuccess(`Image uploaded for SKU: ${sku}`);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Unable to upload image."
      );
    } finally {
      setUploadingSku(null);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center overflow-y-auto bg-black/50 px-4 py-8">
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="manage-images-title"
        className="w-full max-w-3xl bg-[#faf9f6] p-6 shadow-xl md:p-10"
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="mb-2 text-xs uppercase tracking-[0.2em] text-[#ae805f]">
              Catalog Management
            </p>

            <h2 id="manage-images-title" className="font-serif text-3xl">
              Manage Product Images
            </h2>

            <p className="mt-3 text-sm text-[#817a72]">
              {product.name} — {product.id}
            </p>
          </div>

          <button
            type="button"
            onClick={onClose}
            disabled={isBusy}
            aria-label="Close image manager"
            className="text-xl text-[#77716b] hover:text-black disabled:opacity-50"
          >
            ×
          </button>
        </div>

        <p className="mt-7 text-sm leading-6 text-[#77716b]">
          Upload a JPG, PNG or WebP image (up to 5 MB), or enter a path to an
          existing file in frontend/public/products. Leave the path empty and
          choose Save Image to remove an image association. Local uploads are
          available only during development.
        </p>

        <div className="mt-6 max-h-[55vh] space-y-5 overflow-y-auto pr-1">
          {product.variants.map((variant) => (
            <div
              key={variant.sku}
              className="border border-[#e7e2dc] bg-white p-5"
            >
              <div className="mb-4">
                <p className="font-medium">
                  {variant.color} / {variant.size}
                </p>
                <p className="mt-1 text-xs text-[#817a72]">
                  SKU: {variant.sku}
                </p>
              </div>

              <label
                htmlFor={`image-${variant.sku}`}
                className="mb-2 block text-xs font-medium"
              >
                Image Path
              </label>

              <div className="flex flex-col gap-3 sm:flex-row">
                <input
                  id={`image-${variant.sku}`}
                  type="text"
                  value={imagePaths[variant.sku] ?? ""}
                  onChange={(event) => {
                    setImagePaths((previous) => ({
                      ...previous,
                      [variant.sku]: event.target.value,
                    }));
                    setError(null);
                    setSuccess(null);
                  }}
                  disabled={isBusy}
                  placeholder="/products/example.jpg"
                  className="min-w-0 flex-1 border border-[#ddd6cf] px-4 py-3 text-sm outline-none focus:border-[#ae805f] disabled:opacity-50"
                />

                <button
                  type="button"
                  disabled={isBusy}
                  onClick={() => void saveImage(variant.sku)}
                  className="bg-[#24221f] px-5 py-3 text-xs font-medium text-white transition hover:bg-[#ae805f] disabled:opacity-50"
                >
                  {savingSku === variant.sku ? "Saving..." : "Save Image"}
                </button>
              </div>

              <div className="mt-5 border-t border-[#e7e2dc] pt-4">
                <label
                  htmlFor={`upload-${variant.sku}`}
                  className="mb-2 block text-xs font-medium"
                >
                  Upload Image from Computer
                </label>
                <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                  <div className="flex min-w-0 flex-1 items-center gap-3">
                    <input
                      id={`upload-${variant.sku}`}
                      type="file"
                      accept="image/jpeg,image/png,image/webp"
                      disabled={isBusy}
                      onChange={(event) => {
                        const file = event.target.files?.[0] ?? null;
                        setSelectedFiles((previous) => ({
                          ...previous,
                          [variant.sku]: file,
                        }));
                        setError(null);
                        setSuccess(null);
                      }}
                      className="peer sr-only"
                    />
                    <label
                      htmlFor={`upload-${variant.sku}`}
                      className={`shrink-0 border border-[#ddd6cf] bg-white px-3 py-2 text-xs font-medium transition peer-focus-visible:outline peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-[#ae805f] ${
                        isBusy
                          ? "cursor-not-allowed opacity-50"
                          : "cursor-pointer hover:border-[#ae805f]"
                      }`}
                    >
                      Choose File
                    </label>
                    <span
                      className="min-w-0 truncate text-xs text-[#817a72]"
                      title={selectedFiles[variant.sku]?.name ?? "No file selected"}
                    >
                      {selectedFiles[variant.sku]?.name ?? "No file selected"}
                    </span>
                  </div>
                  <button
                    type="button"
                    disabled={isBusy || !selectedFiles[variant.sku]}
                    onClick={() => void uploadImage(variant.sku)}
                    className="bg-[#ae805f] px-5 py-3 text-xs font-medium text-white transition hover:bg-[#8e6548] disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {uploadingSku === variant.sku
                      ? "Uploading..."
                      : "Upload Image"}
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        {error && (
          <p role="alert" className="mt-5 text-sm text-red-700">
            {error}
          </p>
        )}

        {success && (
          <p role="status" className="mt-5 text-sm text-green-700">
            {success}
          </p>
        )}

        <div className="mt-8 flex justify-end border-t border-[#e7e2dc] pt-6">
          <button
            type="button"
            onClick={onClose}
            disabled={isBusy}
            className="border border-[#ddd6cf] px-6 py-3 text-sm transition hover:border-[#ae805f] disabled:opacity-50"
          >
            Close
          </button>
        </div>
      </section>
    </div>
  );
}
