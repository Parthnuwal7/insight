import { backendUrl, extractDetail } from "@/lib/backend";

export const dynamic = "force-dynamic";

/** GET /api/jobs/[jobId]/results — a completed job's merged result.
 *
 * The backend returns 409 while the job is still running; that is passed
 * through with the backend's message so the caller can treat it as
 * "poll again" rather than an error.
 */
export async function GET(
  _request: Request,
  ctx: RouteContext<"/api/jobs/[jobId]/results">,
) {
  const { jobId } = await ctx.params;
  try {
    const res = await fetch(
      backendUrl(`/jobs/${encodeURIComponent(jobId)}/results`),
      { cache: "no-store", signal: AbortSignal.timeout(60_000) },
    );
    const json = await res.json().catch(() => null);
    if (!res.ok) {
      return Response.json(
        {
          ok: false,
          status: res.status,
          message: extractDetail(json, `Fetching results failed (${res.status})`),
        },
        { status: res.status },
      );
    }
    // Backend shape: { status, job_id, data: {...} }. The frontend wants
    // the merged result object directly.
    return Response.json({ ok: true, data: json.data ?? json });
  } catch (e) {
    return Response.json({
      ok: false,
      status: 0,
      message: `Could not reach the backend: ${e instanceof Error ? e.message : String(e)}`,
    });
  }
}
