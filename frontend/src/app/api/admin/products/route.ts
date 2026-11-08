
import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND_URL =
  process.env.ELAN_BACKEND_URL ??
  "http://127.0.0.1:8001";

export async function GET(request: NextRequest) {
  const adminKey = request.headers.get("x-admin-key");

  if (!adminKey?.trim()) {
    return NextResponse.json(
      { detail: "Admin key is required" },
      { status: 401 }
    );
  }

  try {
    const response = await fetch(
      `${BACKEND_URL}/admin/products`,
      {
        method: "GET",
        headers: {
          "x-admin-key": adminKey,
          Accept: "application/json",
        },
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
        detail:
          "Cannot connect to ELAN Backend. Check port 8001.",
      },
      { status: 502 }
    );
  }
}

export async function POST(request: NextRequest) {
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

  try {
    const response = await fetch(
      `${BACKEND_URL}/admin/products`,
      {
        method: "POST",
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
        detail:
          "Cannot connect to ELAN Backend on port 8001.",
      },
      { status: 502 }
    );
  }
}
