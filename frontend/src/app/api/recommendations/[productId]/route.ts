import { NextResponse } from "next/server";

type RouteContext = {
  params: Promise<{ productId: string }>;
};

const backendUrl = (
  process.env.ELAN_BACKEND_URL ?? "http://127.0.0.1:8001"
).replace(/\/$/, "");

export async function GET(_request: Request, { params }: RouteContext) {
  const { productId } = await params;

  if (!productId.trim()) {
    return NextResponse.json(
      { detail: "Product ID is required." },
      { status: 400 }
    );
  }

  try {
    const response = await fetch(
      `${backendUrl}/recommendations/${encodeURIComponent(productId)}?top_k=5`,
      { cache: "no-store" }
    );

    if (!response.ok) {
      return NextResponse.json(
        { detail: "Unable to fetch product recommendations." },
        { status: response.status }
      );
    }

    const data: unknown = await response.json();
    return NextResponse.json(data, {
      headers: { "Cache-Control": "no-store" },
    });
  } catch {
    return NextResponse.json(
      { detail: "Recommendation service is unavailable." },
      { status: 502 }
    );
  }
}
