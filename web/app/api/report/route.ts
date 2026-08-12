import { backendUrl, extractDetail } from "@/lib/backend";

const JSON_HEADERS = { "Content-Type": "application/json" };

// /insights/report is slow — measured to exceed 600s on a cold server. The
// browser must never hold this request itself (a proxy or the user gives
// up long before it finishes). The route handler absorbs the wait: it holds
// the connection to FastAPI open for up to 15 minutes while the browser
// shows an honest "this can take several minutes" state.
//
// Note: serverless hosts (e.g. Vercel) impose their own request ceilings
// well under this. This app is intended for local / self-hosted use where
// the Node server holds the connection for as long as needed.
const REPORT_TIMEOUT_MS = 15 * 60 * 1000;

/** POST /api/report — run the grounded report over already-processed rows. */
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

  const processedData = (body as { processed_data?: unknown })?.processed_data;
  if (!Array.isArray(processedData) || processedData.length === 0) {
    return Response.json(
      {
        ok: false,
        status: 400,
        message: "processed_data must not be empty — process reviews before requesting a report.",
      },
      { status: 400 },
    );
  }

  try {
    const res = await fetch(backendUrl("/insights/report"), {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(REPORT_TIMEOUT_MS),
    });
    const json = await res.json().catch(() => null);
    if (!res.ok) {
      return Response.json(
        {
          ok: false,
          status: res.status,
          message: extractDetail(json, `Report generation failed (${res.status})`),
        },
        { status: res.status },
      );
    }
    // Backend shape: { status: "success", data: Report }. An empty report
    // is a valid, successful response — it flows through as data.
    return Response.json({ ok: true, data: (json as { data?: unknown })?.data ?? json });
  } catch (e) {
    return Response.json({
      ok: false,
      status: 0,
      message: `Report request failed: ${e instanceof Error ? e.message : String(e)}`,
    });
  }
}
