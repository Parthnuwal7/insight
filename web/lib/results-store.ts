/**
 * Client-side persistence of a completed job's results.
 *
 * Task 5 (the report view) needs `processed_data` to resolve citations.
 * Re-fetching a large result set just to render one quote is wasteful, so
 * a completed job's results are stashed in sessionStorage keyed by job id.
 * The report route reads them back; if they're missing (fresh browser
 * session) it re-fetches from the backend instead.
 */
import type { AspectRow, JobResult, ProcessedRow } from "./types";

const KEY_PREFIX = "insights:results:";

export interface StoredResults {
  processed_data: ProcessedRow[];
  aspect_level_data: AspectRow[];
  savedAt: number;
}

export function resultsKey(jobId: string): string {
  return `${KEY_PREFIX}${jobId}`;
}

export function storeResults(jobId: string, result: JobResult): void {
  if (typeof window === "undefined") return;
  const payload: StoredResults = {
    processed_data: result.processed_data ?? [],
    aspect_level_data: result.aspect_level_data ?? [],
    savedAt: Date.now(),
  };
  try {
    window.sessionStorage.setItem(resultsKey(jobId), JSON.stringify(payload));
  } catch {
    // Quota exceeded on a very large run; the report route can re-fetch.
  }
}

export function loadResults(jobId: string): StoredResults | null {
  if (typeof window === "undefined") return null;
  const raw = window.sessionStorage.getItem(resultsKey(jobId));
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as StoredResults;
    if (!Array.isArray(parsed.processed_data)) return null;
    return parsed;
  } catch {
    return null;
  }
}
