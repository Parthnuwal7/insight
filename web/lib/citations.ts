/**
 * Citation resolution: `Claim.review_ids` -> the review text behind it.
 *
 * `id` is the join key for the whole product. Every claim carries
 * `review_ids`; we resolve those against the `processed_data` kept in
 * client state after processing. If ids don't line up, the citation chain
 * silently renders empty — so ids are normalized (compared as strings) to
 * absorb `number` vs `string` drift between runs.
 */
import type { Claim, ProcessedRow } from "./types";

/** Compare review ids the way the backend does: loosely, by string form. */
export function idsEqual(a: number | string, b: number | string): boolean {
  return String(a) === String(b);
}

export function resolveReview(
  processed: ProcessedRow[],
  reviewId: number | string,
): ProcessedRow | undefined {
  return processed.find((row) => idsEqual(row.id, reviewId));
}

export interface ResolvedCitation {
  reviewId: number | string;
  row: ProcessedRow | null;
}

/** Resolve every id a claim cites, in order. Missing rows come back null —
 * the UI must say so rather than silently skipping. */
export function resolveCitations(
  claim: Claim,
  processed: ProcessedRow[],
): ResolvedCitation[] {
  return claim.review_ids.map((reviewId) => ({
    reviewId,
    row: resolveReview(processed, reviewId) ?? null,
  }));
}

/** True when the review actually needed translation (and we have it). */
export function isTranslated(row: ProcessedRow): boolean {
  return (
    row.detected_language !== "en" &&
    typeof row.translated_review === "string" &&
    row.translated_review.length > 0 &&
    row.translated_review !== row.review
  );
}

const LANGUAGE_LABELS: Record<string, string> = {
  en: "English",
  hi: "Hindi",
  te: "Telugu",
  ta: "Tamil",
  bn: "Bengali",
  mr: "Marathi",
  gu: "Gujarati",
  kn: "Kannada",
  ml: "Malayalam",
  pa: "Punjabi",
};

export function languageLabel(code: string | null | undefined): string {
  if (!code) return "unknown language";
  return LANGUAGE_LABELS[code] ?? code;
}
