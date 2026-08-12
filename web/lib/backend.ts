/**
 * Server-only helpers for route handlers that proxy to the FastAPI backend.
 *
 * The backend URL lives here, server-side, and is never shipped to the
 * browser — that is the point of proxying through Next.js route handlers
 * rather than calling FastAPI directly (see docs/superpowers/plans/
 * 2026-08-12-phase-d-nextjs-frontend.md, Task 4).
 *
 * Importing this module from client code will pull `process.env` into the
 * browser bundle; keep it in route handlers only.
 */

export function getBackendBaseUrl(): string {
  return process.env.BACKEND_API_URL ?? "http://localhost:7860";
}

/** Pull a human-readable error string out of an error response body. */
export function extractDetail(json: unknown, fallback: string): string {
  if (json && typeof json === "object") {
    const detail = (json as { detail?: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (detail && typeof detail === "object") return JSON.stringify(detail);
  }
  return fallback;
}

/** Build the backend URL for one path, e.g. `/jobs/abc/results`. */
export function backendUrl(path: string): string {
  return `${getBackendBaseUrl()}${path}`;
}
