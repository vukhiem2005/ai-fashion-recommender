
import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND_URL =
  process.env.ELAN_BACKEND_URL ??
  "http://127.0.0.1:8001";

type RouteContext = {
  params: Promise<{ productId: string }>;
};

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

  let body: unknown;

  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { detail: "Invalid JSON request" },
      { status: 400 }
    );
  }

  if (
    !body ||
    typeof body !== "object" ||
    !("is_active" in body) ||
    typeof body.is_active !== "boolean"
  ) {
    return NextResponse.json(
      { detail: "is_active must be a boolean" },
      { status: 422 }
    );
  }

  const { productId } = await context.params;

  try {
    const response = await fetch(
      `${BACKEND_URL}/admin/products/${encodeURIComponent(productId)}/status`,
      {
        method: "PATCH",
        headers: {
          "x-admin-key": adminKey,
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({
          is_active: body.is_active,
        }),
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
