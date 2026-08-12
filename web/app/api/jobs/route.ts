import { backendUrl, extractDetail } from "@/lib/backend";

const JSON_HEADERS = { "Content-Type": "application/json" };

/** POST /api/jobs — submit a durable background job.
 *
 * Forwarded to the backend's POST /jobs, which returns `{ job_id }`
 * immediately; the browser then polls /api/jobs/[jobId] for progress.
 */
export async function POST(request: Request) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return Response.json(
      { ok: false, status: 400, message: "Request body must be valid JSON." },
      { status: 400 },
    );
  }

  const data = (body as { data?: unknown })?.data;
  if (!Array.isArray(data) || data.length === 0) {
    return Response.json(
      { ok: false, status: 400, message: "data must not be empty." },
      { status: 400 },
    );
  }

  try {
    const res = await fetch(backendUrl("/jobs"), {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(60_000),
    });
    const json = await res.json().catch(() => null);
    if (!res.ok) {
      return Response.json(
        {
          ok: false,
          status: res.status,
          message: extractDetail(json, `Job submission failed (${res.status})`),
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

/** GET /api/jobs?user_id=... — recent runs for a user, newest first. */
export async function GET(request: Request) {
  const user_id = new URL(request.url).searchParams.get("user_id") ?? "web";
  try {
    const res = await fetch(
      backendUrl(`/jobs?user_id=${encodeURIComponent(user_id)}`),
      { cache: "no-store", signal: AbortSignal.timeout(10_000) },
    );
    const json = await res.json().catch(() => null);
    if (!res.ok) {
      return Response.json(
        {
          ok: false,
          status: res.status,
          message: extractDetail(json, `Listing jobs failed (${res.status})`),
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
