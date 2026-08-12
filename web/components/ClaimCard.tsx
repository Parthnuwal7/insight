"use client";

import { useState } from "react";
import type { Claim, ProcessedRow } from "@/lib/types";
import { isTranslated, languageLabel, resolveCitations } from "@/lib/citations";

/** A claim plus its expandable citations.
 *
 * This is the point of the whole product: a finding the user cannot expand
 * to the review text it rests on is indistinguishable from the templated
 * prose this system replaced. Every review id is resolved against the
 * run's own processed_data, so nothing here is fabricated.
 *
 * The evidence badge exists because "1 review cited" and "4 reviews cited"
 * used to render as identical grey text in the corner, which let a
 * one-off remark and a repeated pattern claim the same authority. The
 * count is the single most decision-relevant fact about a claim, so it is
 * now weighted, coloured, and stated as a share of the run.
 */
export default function ClaimCard({
  claim,
  processed,
  totalReviews,
}: {
  claim: Claim;
  processed: ProcessedRow[];
  totalReviews?: number;
}) {
  const [open, setOpen] = useState(false);
  const citations = resolveCitations(claim, processed);
  const missing = citations.filter((c) => c.row === null).length;
  const cited = claim.review_ids.length;
  const total = totalReviews ?? processed.length;
  const evidence = evidenceLabel(cited, total);

  return (
    <div className="rounded-lg border border-slate-200 bg-white shadow-sm">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-start justify-between gap-3 px-4 py-3 text-left"
        aria-expanded={open}
      >
        <p className="text-sm leading-relaxed text-slate-800">{claim.text}</p>
        <span className="flex shrink-0 items-center gap-2">
          <span
            className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${evidence.className}`}
            title={evidence.title}
          >
            {evidence.text}
          </span>
          <span className="text-xs text-slate-400">{open ? "hide" : "show"}</span>
        </span>
      </button>

      {open ? (
        <div className="space-y-3 border-t border-slate-100 px-4 py-3">
          {claim.reason ? (
            <p className="text-xs leading-relaxed text-slate-600">
              <span className="font-medium text-slate-500">Verifier:</span>{" "}
              {claim.reason}
            </p>
          ) : null}

          {missing > 0 ? (
            <p className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
              {missing} cited review id(s) could not be resolved against this
              run&apos;s data. The claim was verified against them regardless,
              but the text can&apos;t be shown here.
            </p>
          ) : null}

          {citations.map(({ reviewId, row }) =>
            row ? (
              <div
                key={String(reviewId)}
                className="rounded-md border border-slate-100 bg-slate-50 p-3"
              >
                <p className="text-xs font-medium text-slate-500">
                  Review #{reviewId} · {languageLabel(row.detected_language)} ·
                  sentiment{" "}
                  <span className="text-slate-700">{row.overall_sentiment}</span>
                </p>
                <p className="mt-1 text-sm text-slate-800">{row.review}</p>
                {isTranslated(row) ? (
                  <p className="mt-1 text-xs italic text-slate-600">
                    Translated from {languageLabel(row.detected_language)}:{" "}
                    {row.translated_review}
                  </p>
                ) : null}
              </div>
            ) : (
              <p key={String(reviewId)} className="text-xs text-slate-500">
                Review text unavailable locally for id {String(reviewId)}.
              </p>
            ),
          )}
        </div>
      ) : null}
    </div>
  );
}

/**
 * How much weight a reader should put on a claim, from its citation count.
 *
 * The thresholds are deliberately blunt and stated in the label rather
 * than hidden in a score: one review is an anecdote, two or three is a
 * repeat, four or more is a pattern. The share of the run is shown
 * alongside because "4 reviews" means something very different in a run of
 * 30 than in a run of 3,000.
 */
function evidenceLabel(cited: number, total: number) {
  const share = total > 0 ? Math.round((cited / total) * 100) : 0;
  const suffix = total > 0 ? ` of ${total} reviews (${share}%)` : "";
  if (cited <= 1) {
    return {
      text: "1 review",
      className: "bg-slate-100 text-slate-600",
      title: `Single review${suffix}. One person said this once — treat as an anecdote, not a pattern.`,
    };
  }
  if (cited <= 3) {
    return {
      text: `${cited} reviews`,
      className: "bg-sky-100 text-sky-800",
      title: `Cited by ${cited}${suffix}. Repeated, but on thin evidence.`,
    };
  }
  return {
    text: `${cited} reviews`,
    className: "bg-violet-100 text-violet-800",
    title: `Cited by ${cited}${suffix}. The strongest evidence in this report.`,
  };
}
