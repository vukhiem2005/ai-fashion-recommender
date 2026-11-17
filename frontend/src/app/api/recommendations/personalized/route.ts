import { NextResponse } from "next/server";

const backendUrl = (
  process.env.ELAN_BACKEND_URL ?? "http://127.0.0.1:8001"
).replace(/\/$/, "");

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const wantsUsers = searchParams.get("users") === "1";
  const userId = searchParams.get("user_id") ?? "SYN-U0001";
  const topK = searchParams.get("top_k") ?? "5";

  const targetUrl = wantsUsers
    ? `${backendUrl}/recommendations/personalized/users`
    : `${backendUrl}/recommendations/personalized?user_id=${encodeURIComponent(
        userId
      )}&top_k=${encodeURIComponent(topK)}`;

  try {
    const response = await fetch(targetUrl, { cache: "no-store" });
    if (!response.ok) {
      return NextResponse.json(
        { detail: "Unable to load synthetic personalized recommendations." },
        { status: response.status }
      );
    }

    const data: unknown = await response.json();
    return NextResponse.json(data, {
      headers: { "Cache-Control": "no-store" },
    });
  } catch {
    return NextResponse.json(
      { detail: "Personalization service is unavailable." },
      { status: 502 }
    );
  }
}
