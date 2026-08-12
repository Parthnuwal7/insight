import { backendUrl, extractDetail } from "@/lib/backend";

export const dynamic = "force-dynamic";

/** GET /api/jobs/[jobId] — job status, stage, chunk progress. */
export async function GET(
  _request: Request,
  ctx: RouteContext<"/api/jobs/[jobId]">,
) {
  const { jobId } = await ctx.params;
  try {
    const res = await fetch(backendUrl(`/jobs/${encodeURIComponent(jobId)}`), {
      cache: "no-store",
      signal: AbortSignal.timeout(10_000),
    });
    const json = await res.json().catch(() => null);
    if (!res.ok) {
      return Response.json(
        {
          ok: false,
          status: res.status,
          message: extractDetail(json, `Job lookup failed (${res.status})`),
        },
        { status: res.status },
      );
    }
    return Response.json({ ok: true, data: json });
  } catch (e) {
    return Response.json({
      ok: false,
      status: 0,
      message: `Could not reach the backend: ${e instanceof Error ? e.message : String(e)}`,
    });
  }
}
