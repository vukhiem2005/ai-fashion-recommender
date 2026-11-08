
import { randomUUID } from "node:crypto";
import { mkdir, unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

const BACKEND_URL =
  process.env.ELAN_BACKEND_URL ??
  "http://127.0.0.1:8001";

const MAX_IMAGE_SIZE = 5 * 1024 * 1024;

interface RouteContext {
  params: Promise<{
    productId: string;
    sku: string;
  }>;
}

interface AdminProduct {
  id: string;
  variants: { sku: string }[];
}

function detectImageType(
  bytes: Uint8Array
): "jpg" | "png" | "webp" | null {
  if (
    bytes.length >= 3 &&
    bytes[0] === 0xff &&
    bytes[1] === 0xd8 &&
    bytes[2] === 0xff
  ) {
    return "jpg";
  }

  const pngSignature = [
    137, 80, 78, 71, 13, 10, 26, 10,
  ];

  if (
    bytes.length >= 8 &&
    pngSignature.every(
      (value, index) => bytes[index] === value
    )
  ) {
    return "png";
  }

  if (
    bytes.length >= 12 &&
    String.fromCharCode(...bytes.slice(0, 4)) ===
      "RIFF" &&
    String.fromCharCode(...bytes.slice(8, 12)) ===
      "WEBP"
  ) {
    return "webp";
  }

  return null;
}

export async function POST(
  request: Request,
  context: RouteContext
) {
  if (process.env.NODE_ENV !== "development") {
    return NextResponse.json(
      {
        detail:
          "Local image upload is only available in development.",
      },
      { status: 501 }
    );
  }

  const adminKey = request.headers.get("x-admin-key");

  if (!adminKey) {
    return NextResponse.json(
      { detail: "Admin key is required." },
      { status: 401 }
    );
  }

  const { productId, sku } = await context.params;

  let savedFilePath: string | null = null;

  try {
    // Validate credentials before processing the upload.
    const inventoryResponse = await fetch(
      `${BACKEND_URL}/admin/products`,
      {
        headers: {
          "X-Admin-Key": adminKey,
        },
        cache: "no-store",
      }
    );

    if (!inventoryResponse.ok) {
      return NextResponse.json(
        { detail: "Admin authorization failed." },
        { status: inventoryResponse.status }
      );
    }

    const products =
      (await inventoryResponse.json()) as AdminProduct[];

    const validVariant =
      Array.isArray(products) &&
      products.some(
        (product) =>
          product.id === productId &&
          Array.isArray(product.variants) &&
          product.variants.some(
            (variant) => variant.sku === sku
          )
      );

    if (!validVariant) {
      return NextResponse.json(
        { detail: "Product or SKU not found." },
        { status: 404 }
      );
    }

    const formData = await request.formData();
    const file = formData.get("image");

    if (!(file instanceof File)) {
      return NextResponse.json(
        { detail: "Image file is required." },
        { status: 422 }
      );
    }

    if (file.size === 0 || file.size > MAX_IMAGE_SIZE) {
      return NextResponse.json(
        { detail: "Image must be under 5 MB." },
        { status: 413 }
      );
    }

    const bytes = new Uint8Array(
      await file.arrayBuffer()
    );

    const extension = detectImageType(bytes);

    const acceptedTypes = {
      jpg: "image/jpeg",
      png: "image/png",
      webp: "image/webp",
    };

    if (
      extension === null ||
      file.type !== acceptedTypes[extension]
    ) {
      return NextResponse.json(
        {
          detail:
            "Only valid JPG, PNG and WebP images are supported.",
        },
        { status: 415 }
      );
    }

    // Generate a safe, unique filename.
    const safeSku = sku
      .replace(/[^a-zA-Z0-9_-]/g, "-")
      .slice(0, 60);

    const filename =
      `${safeSku}-${randomUUID()}.${extension}`;

    const uploadDirectory = path.join(
      process.cwd(),
      "public",
      "products"
    );

    await mkdir(uploadDirectory, {
      recursive: true,
    });

    savedFilePath = path.join(
      uploadDirectory,
      filename
    );

    await writeFile(savedFilePath, bytes, {
      flag: "wx",
    });

    const imageUrl = `/products/${filename}`;

    // Link the uploaded image to the selected SKU.
    const updateResponse = await fetch(
      `${BACKEND_URL}/admin/products/` +
        `${encodeURIComponent(productId)}/variants/` +
        `${encodeURIComponent(sku)}/image`,
      {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          "X-Admin-Key": adminKey,
        },
        body: JSON.stringify({
          image_url: imageUrl,
        }),
        cache: "no-store",
      }
    );

    if (!updateResponse.ok) {
      await unlink(savedFilePath);
      savedFilePath = null;

      return NextResponse.json(
        {
          detail:
            "Image saved locally but database update failed.",
        },
        { status: updateResponse.status }
      );
    }

    return NextResponse.json(
      {
        message: "Image uploaded successfully.",
        product_id: productId,
        sku,
        image_url: imageUrl,
      },
      { status: 201 }
    );
  } catch {
    // Remove a newly uploaded file if the operation fails.
    if (savedFilePath !== null) {
      await unlink(savedFilePath).catch(() => {});
    }

    return NextResponse.json(
      {
        detail: "Unable to upload image.",
      },
      { status: 500 }
    );
  }
}
