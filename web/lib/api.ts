/**
 * Typed client for the Next.js route handlers.
 *
 * Every function talks to a same-origin `/api/*` route handler, which in
 * turn proxies to the FastAPI backend server-side. The client itself never
 * needs the backend URL and never hits it directly, so CORS is irrelevant
 * for the real deployment.
 *
 * Every function returns an `ApiResult<T>` — a discriminated union, not a
 * thrown exception. The cases worth handling explicitly:
 *
 * - `409` from job results  — the job isn't finished yet; poll again.
 * - `404`                   — unknown job id; authoritative (the backend is
 *                             careful never to 200 with a null body here).
 * - `400`                   — bad input, e.g. an empty `data` array.
 */
import type {
  ApiResult,
  Health,
  InsightReportRequest,
  JobListResponse,
  JobResult,
  JobStatus,
  ProcessRequest,
  Report,
  SubmitJobResponse,
} from "./types";

const JSON_HEADERS = { "Content-Type": "application/json" };

async function request<T>(
  url: string,
  init?: RequestInit,
): Promise<ApiResult<T>> {
  try {
    const res = await fetch(url, init);
    const body = await res.json().catch(() => null);
    // Route handlers return the ApiResult envelope directly.
    return body as ApiResult<T>;
  } catch (e) {
    return {
      ok: false,
      status: 0,
      message: e instanceof Error ? e.message : "Network error",
    };
  }
}

export function getHealth(): Promise<ApiResult<Health>> {
  return request<Health>("/api/health", { cache: "no-store" });
}

export function submitJob(
  req: ProcessRequest,
): Promise<ApiResult<SubmitJobResponse>> {
  return request<SubmitJobResponse>("/api/jobs", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(req),
  });
}

export function getJobStatus(jobId: string): Promise<ApiResult<JobStatus>> {
  return request<JobStatus>(`/api/jobs/${encodeURIComponent(jobId)}`, {
    cache: "no-store",
  });
}

export function getJobResults(jobId: string): Promise<ApiResult<JobResult>> {
  return request<JobResult>(
    `/api/jobs/${encodeURIComponent(jobId)}/results`,
    { cache: "no-store" },
  );
}

export function cancelJob(
  jobId: string,
): Promise<ApiResult<{ status: string; job_id: string }>> {
  return request<{ status: string; job_id: string }>(
    `/api/jobs/${encodeURIComponent(jobId)}/cancel`,
    { method: "POST" },
  );
}

export function listJobs(userId: string): Promise<ApiResult<JobListResponse>> {
  return request<JobListResponse>(
    `/api/jobs?user_id=${encodeURIComponent(userId)}`,
    { cache: "no-store" },
  );
}

/** Kick off report generation. The route handler holds this request open
 * for up to 15 minutes — see Task 4. */
export function generateReport(
  req: InsightReportRequest,
): Promise<ApiResult<Report>> {
  return request<Report>("/api/report", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(req),
  });
}
