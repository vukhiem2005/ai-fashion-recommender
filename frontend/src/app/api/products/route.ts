
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND_URL =
  process.env.ELAN_BACKEND_URL ??
  "http://127.0.0.1:8001";

export async function GET() {
  try {
    const response = await fetch(
      `${BACKEND_URL}/products`,
      {
        method: "GET",
        headers: {
          Accept: "application/json",
        },
        cache: "no-store",
      }
    );

    if (!response.ok) {
      return NextResponse.json(
        {
          detail: "Unable to load products.",
        },
        { status: response.status }
      );
    }

    const products = await response.json();

    return NextResponse.json(products, {
      status: 200,
      headers: {
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return NextResponse.json(
      {
        detail:
          "Cannot connect to ELAN Backend on port 8001.",
      },
      { status: 502 }
    );
  }
}
