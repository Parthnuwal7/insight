import { backendUrl, extractDetail } from "@/lib/backend";

/** POST /api/jobs/[jobId]/cancel — request cancellation. */
export async function POST(
  _request: Request,
  ctx: RouteContext<"/api/jobs/[jobId]/cancel">,
) {
  const { jobId } = await ctx.params;
  try {
    const res = await fetch(
      backendUrl(`/jobs/${encodeURIComponent(jobId)}/cancel`),
      { method: "POST", signal: AbortSignal.timeout(10_000) },
    );
    const json = await res.json().catch(() => null);
    if (!res.ok) {
      return Response.json(
        {
          ok: false,
          status: res.status,
          message: extractDetail(json, `Cancellation failed (${res.status})`),
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
