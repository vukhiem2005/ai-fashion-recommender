
import { NextResponse } from "next/server";

export const runtime = "nodejs";

const BACKEND_URL =
  process.env.ELAN_BACKEND_URL ??
  "http://127.0.0.1:8001";

interface RouteContext {
  params: Promise<{
    productId: string;
    sku: string;
  }>;
}

export async function PATCH(
  request: Request,
  context: RouteContext
) {
  const adminKey = request.headers.get("x-admin-key");

  if (!adminKey) {
    return NextResponse.json(
      { detail: "Admin key is required." },
      { status: 401 }
    );
  }

  const payload: unknown = await request
    .json()
    .catch(() => null);

  if (
    typeof payload !== "object" ||
    payload === null ||
    !("image_url" in payload) ||
    !(
      typeof payload.image_url === "string" ||
      payload.image_url === null
    )
  ) {
    return NextResponse.json(
      {
        detail:
          "image_url must be a string or null.",
      },
      { status: 422 }
    );
  }

  const { productId, sku } = await context.params;

  const url =
    `${BACKEND_URL}/admin/products/` +
    `${encodeURIComponent(productId)}/variants/` +
    `${encodeURIComponent(sku)}/image`;

  try {
    const response = await fetch(url, {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        "X-Admin-Key": adminKey,
      },
      body: JSON.stringify({
        image_url: payload.image_url,
      }),
      cache: "no-store",
    });

    const result = await response.json().catch(() => ({
      detail: "Backend returned an invalid response.",
    }));

    return NextResponse.json(result, {
      status: response.status,
    });
  } catch {
    return NextResponse.json(
      {
        detail:
          "Cannot connect to ELAN Backend.",
      },
      { status: 502 }
    );
  }
}
