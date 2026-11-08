
import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND_URL =
  process.env.ELAN_BACKEND_URL ??
  "http://127.0.0.1:8001";

type RouteContext = {
  params: Promise<{ productId: string }>;
};

const allowedFields = [
  "name",
  "category",
  "description",
  "price_cents",
];

export async function PATCH(
  request: NextRequest,
  context: RouteContext
) {
  const adminKey = request.headers.get("x-admin-key");

  if (!adminKey?.trim()) {
    return NextResponse.json(
      { detail: "Admin key is required" },
      { status: 401 }
    );
  }

  let payload: unknown;

  try {
    payload = await request.json();
  } catch {
    return NextResponse.json(
      { detail: "Invalid JSON request" },
      { status: 400 }
    );
  }

  if (
    !payload ||
    typeof payload !== "object" ||
    Array.isArray(payload)
  ) {
    return NextResponse.json(
      { detail: "Invalid product data" },
      { status: 422 }
    );
  }

  const fields = Object.keys(payload);

  if (
    fields.length === 0 ||
    fields.some((field) => !allowedFields.includes(field))
  ) {
    return NextResponse.json(
      { detail: "Invalid update fields" },
      { status: 422 }
    );
  }

  const { productId } = await context.params;

  try {
    const response = await fetch(
      `${BACKEND_URL}/admin/products/${encodeURIComponent(productId)}`,
      {
        method: "PATCH",
        headers: {
          "x-admin-key": adminKey,
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify(payload),
        cache: "no-store",
      }
    );

    const data = await response.json();

    return NextResponse.json(data, {
      status: response.status,
      headers: {
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return NextResponse.json(
      {
        detail: "Cannot connect to ELAN Backend.",
      },
      { status: 502 }
    );
  }
}
