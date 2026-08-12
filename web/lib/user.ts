/**
 * A stable-per-browser user id for job submission and the recent-runs list.
 *
 * There is no auth anywhere in this product (deliberately out of scope).
 * Scoping jobs to a per-browser id keeps one browser's recent runs from
 * showing another's — without pretending the id means anything beyond that.
 */

const STORAGE_KEY = "insights:user_id";

export function getUserId(): string {
  if (typeof window === "undefined") return "web";
  const existing = window.localStorage.getItem(STORAGE_KEY);
  if (existing) return existing;
  const fresh = `web_${Math.random().toString(36).slice(2, 10)}`;
  try {
    window.localStorage.setItem(STORAGE_KEY, fresh);
  } catch {
    // Storage unavailable (private mode); fall back to a stable default.
    return "web";
  }
  return fresh;
}
