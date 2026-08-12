import { backendUrl, extractDetail } from "@/lib/backend";

export const dynamic = "force-dynamic";

export async function GET() {
  const base = backendUrl("/health");
  try {
    const res = await fetch(base, {
      signal: AbortSignal.timeout(10_000),
      cache: "no-store",
    });
    if (!res.ok) {
      const body = await res.json().catch(() => null);
      return Response.json(
        {
          ok: false,
          status: res.status,
          message: extractDetail(body, `Backend health check failed (${res.status})`),
        },
        { status: res.status },
      );
    }
    const health = await res.json();
    return Response.json({ ok: true, data: health });
  } catch (e) {
    return Response.json({
      ok: false,
      status: 0,
      message: `Could not reach the backend at ${base}: ${
        e instanceof Error ? e.message : String(e)
      }`,
    });
  }
}
