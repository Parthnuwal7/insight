"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { generateReport, getJobResults } from "@/lib/api";
import type { AspectRow, Claim, ProcessedRow, Report } from "@/lib/types";
import { loadResults, storeResults } from "@/lib/results-store";
import ClaimCard from "./ClaimCard";

const REPORT_STORAGE_PREFIX = "insights:report:";

type ReportPhase =
  | { kind: "loading" }
  | { kind: "ready" }
  | { kind: "generating" }
  | { kind: "done"; report: Report }
  | { kind: "failed"; message: string };

type SectionKey = "complaints" | "strengths" | "actions" | "findings";

interface SectionConfig {
  key: SectionKey;
  title: string;
  icon: string;
  tone: string;
  /** Sections the product promises by name are rendered even when empty —
   *  silently omitting "Suggested actions" reads as "there were none to
   *  make" when the truth is usually "the agent never produced any". */
  alwaysShow?: boolean;
  emptyNote?: string;
}

const SECTIONS: SectionConfig[] = [
  { key: "complaints", title: "Complaints", icon: "⚠️", tone: "text-red-800" },
  { key: "strengths", title: "Strengths", icon: "✅", tone: "text-emerald-800" },
  { key: "findings", title: "Other findings", icon: "🔎", tone: "text-slate-900" },
  {
    key: "actions",
    title: "Suggested actions",
    icon: "🛠️",
    tone: "text-violet-800",
    alwaysShow: true,
    emptyNote:
      "The investigation produced no action claims for this run. Actions are only ever generated from verified complaints — they are not templated from the complaint list, so an empty section here means the agent did not propose any, not that none exist.",
  },
];

/** Strongest evidence first. Emission order carries no meaning — it is the
 *  order the agent happened to think of things — so presenting it as the
 *  reading order buried the best-supported claim under anecdotes. */
function byEvidence(a: Claim, b: Claim) {
  return b.review_ids.length - a.review_ids.length;
}

export default function ReportView({ jobId }: { jobId: string }) {
  const [phase, setPhase] = useState<ReportPhase>({ kind: "loading" });
  const [processed, setProcessed] = useState<ProcessedRow[]>([]);
  const [aspectLevel, setAspectLevel] = useState<AspectRow[]>([]);

  const loadRunData = useCallback(async () => {
    const cached = loadResults(jobId);
    if (cached) {
      setProcessed(cached.processed_data);
      setAspectLevel(cached.aspect_level_data);
      return true;
    }
    const res = await getJobResults(jobId);
    if (res.ok) {
      storeResults(jobId, res.data);
      setProcessed(res.data.processed_data ?? []);
      setAspectLevel(res.data.aspect_level_data ?? []);
      return true;
    }
    return false;
  }, [jobId]);

  useEffect(() => {
    let active = true;

    (async () => {
      const reportKey = `${REPORT_STORAGE_PREFIX}${jobId}`;
      if (typeof window !== "undefined") {
        const cachedReport = window.sessionStorage.getItem(reportKey);
        if (cachedReport) {
          try {
            const report = JSON.parse(cachedReport) as Report;
            if (active && report && typeof report === "object") {
              setPhase({ kind: "done", report });
            }
          } catch {
            // Corrupt cache; fall through and load run data.
          }
        }
      }

      const hasData = await loadRunData();
      if (!active) return;
      if (!hasData) {
        setPhase({
          kind: "failed",
          message:
            "This run's results could not be loaded. Make sure the job is complete and the backend is running.",
        });
        return;
      }
      setPhase((prev) => (prev.kind === "done" ? prev : { kind: "ready" }));
    })();

    return () => {
      active = false;
    };
  }, [jobId, loadRunData]);

  const report = phase.kind === "done" ? phase.report : null;

  const statsSummary = useMemo(() => {
    if (!report) return null;
    const verify = report.stats?.verification;
    const investigation = report.stats?.investigation;
    const health = report.stats?.extraction_health;
    return {
      produced: (verify?.kept ?? 0) + (verify?.dropped ?? 0),
      kept: verify?.kept ?? 0,
      dropped: verify?.dropped ?? 0,
      droppedByReason: verify?.dropped_by_reason ?? {},
      toolCalls: investigation?.tool_calls_made ?? 0,
      stoppedReason: investigation?.stopped_reason ?? "n/a",
      elapsedSeconds: investigation?.elapsed_seconds ?? 0,
      deduplicated: investigation?.claims_deduplicated ?? 0,
      reviews: health?.total ?? 0,
      degradedFraction: health?.degraded_fraction ?? 0,
    };
  }, [report]);

  /** The one claim a reader should see if they read nothing else: the
   *  complaint resting on the most reviews. Chosen by citation count only
   *  — no scoring, no LLM, nothing that could invent an emphasis the data
   *  does not support. */
  const headline = useMemo(() => {
    if (!report) return null;
    const ranked = [...(report.complaints ?? [])].sort(byEvidence);
    const top = ranked[0];
    if (!top || top.review_ids.length < 2) return null;
    return top;
  }, [report]);

  async function handleGenerate() {
    setPhase({ kind: "generating" });
    const res = await generateReport({
      processed_data: processed,
      aspect_level_data: aspectLevel.length > 0 ? aspectLevel : null,
      max_tool_calls: 20,
      max_seconds: 120,
    });
    if (res.ok) {
      try {
        window.sessionStorage.setItem(
          `${REPORT_STORAGE_PREFIX}${jobId}`,
          JSON.stringify(res.data),
        );
      } catch {
        // Non-fatal: report still renders this session.
      }
      setPhase({ kind: "done", report: res.data });
    } else {
      setPhase({ kind: "failed", message: res.message });
    }
  }

  const hasAnyClaim =
    report != null && SECTIONS.some((s) => (report[s.key] ?? []).length > 0);

  return (
    <div className="mt-6 space-y-6">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">Insight report</h2>
          <p className="text-sm text-slate-500">
            Every claim below is checked against the reviews it cites. An empty
            report is a valid report — nothing here is ever invented to fill
            space.
          </p>
        </div>
        <Link
          href={`/runs/${jobId}`}
          className="shrink-0 text-sm text-slate-500 transition hover:text-slate-700"
        >
          ← Back to results
        </Link>
      </div>

      {phase.kind === "loading" ? (
        <p className="text-sm text-slate-500">Loading run data…</p>
      ) : null}

      {phase.kind === "failed" ? (
        <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-900">
          {phase.message}
        </div>
      ) : null}

      {(phase.kind === "ready" ||
        phase.kind === "generating" ||
        phase.kind === "done") &&
      !report ? (
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          {phase.kind === "generating" ? (
            <div className="text-center">
              <div className="mx-auto mb-3 h-8 w-8 animate-spin rounded-full border-2 border-slate-300 border-t-slate-900" />
              <p className="text-sm font-medium text-slate-900">
                Investigating and verifying findings…
              </p>
              <p className="mx-auto mt-1 max-w-md text-sm text-slate-500">
                This can take several minutes — the agent reads review text and
                every claim is re-checked against its citations before it
                appears. A page refresh loses the in-flight request; the report
                will persist once it finishes.
              </p>
            </div>
          ) : (
            <div className="flex flex-col items-start gap-3">
              <p className="text-sm text-slate-600">
                Ready to run over {processed.length} review(s).
              </p>
              <button
                type="button"
                onClick={() => void handleGenerate()}
                className="rounded-lg bg-violet-700 px-4 py-2 text-sm font-medium text-white transition hover:bg-violet-600"
              >
                Generate insight report
              </button>
            </div>
          )}
        </div>
      ) : null}

      {report && statsSummary ? (
        <>
          {/* Headline — the single best-supported complaint, quoted. This is
              the top of the page because it is the only thing on it a reader
              can act on without reading further. */}
          {headline ? (
            <div className="rounded-xl border-l-4 border-l-red-500 border-y border-r border-slate-200 bg-white p-5 shadow-sm">
              <p className="text-xs font-medium uppercase tracking-wide text-red-700">
                Best-supported complaint
              </p>
              <p className="mt-1.5 text-base leading-relaxed text-slate-900">
                {headline.text}
              </p>
              <p className="mt-1.5 text-xs text-slate-500">
                Rests on {headline.review_ids.length} of {statsSummary.reviews}{" "}
                reviews — the most-cited claim in this report.
              </p>
            </div>
          ) : null}

          {/* Section counts, not pipeline counts. */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {SECTIONS.map((s) => {
              const n = (report[s.key] ?? []).length;
              return (
                <a
                  key={s.key}
                  href={`#section-${s.key}`}
                  className="rounded-lg border border-slate-200 bg-white p-3 shadow-sm transition hover:border-slate-300"
                >
                  <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
                    {s.title}
                  </p>
                  <p className={`mt-0.5 text-2xl font-bold ${n === 0 ? "text-slate-300" : "text-slate-900"}`}>
                    {n}
                  </p>
                </a>
              );
            })}
          </div>

          {report.caveats.length > 0 ? (
            <div className="space-y-2">
              {report.caveats.map((caveat) => (
                <div
                  key={caveat}
                  className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-2.5 text-xs leading-relaxed text-amber-900"
                >
                  {caveat}
                </div>
              ))}
            </div>
          ) : null}

          {!hasAnyClaim ? (
            <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
              <h3 className="text-base font-semibold text-slate-900">
                Nothing could be substantiated for this run.
              </h3>
              <p className="mt-2 max-w-2xl text-sm text-slate-600">
                That is a valid outcome, not an error. {statsSummary.produced}{" "}
                claim(s) were produced by the investigation, and all{" "}
                {statsSummary.dropped} were dropped in verification. The run
                details below say exactly why. We are not going to invent prose
                to fill the page.
              </p>
            </div>
          ) : (
            <div className="space-y-8">
              {SECTIONS.map((section) => {
                const claims = [...(report[section.key] ?? [])].sort(byEvidence);
                if (claims.length === 0 && !section.alwaysShow) return null;
                return (
                  <section key={section.key} id={`section-${section.key}`}>
                    <h3 className={`mb-2 text-base font-semibold ${section.tone}`}>
                      {section.icon} {section.title} ({claims.length})
                    </h3>
                    {claims.length === 0 ? (
                      <p className="rounded-lg border border-dashed border-slate-300 bg-slate-50 px-4 py-3 text-sm leading-relaxed text-slate-600">
                        {section.emptyNote}
                      </p>
                    ) : (
                      <div className="space-y-2">
                        {claims.map((claim, i) => (
                          <ClaimCard
                            key={`${section.key}-${i}`}
                            claim={claim}
                            processed={processed}
                            totalReviews={statsSummary.reviews || processed.length}
                          />
                        ))}
                      </div>
                    )}
                  </section>
                );
              })}
            </div>
          )}

          {/* Run details — the pipeline's own bookkeeping. The user is
              entitled to all of it, but it is evidence about the machine,
              not about their reviews, so it no longer occupies the top of
              the page ahead of the findings. */}
          <details className="rounded-xl border border-slate-200 bg-white shadow-sm">
            <summary className="cursor-pointer px-4 py-3 text-sm font-medium text-slate-700">
              Run details — how this report was produced
            </summary>
            <div className="border-t border-slate-100 px-4 py-4">
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                <Stat label="Claims produced" value={String(statsSummary.produced)} />
                <Stat label="Kept after verification" value={String(statsSummary.kept)} />
                <Stat label="Dropped in verification" value={String(statsSummary.dropped)} />
                <Stat label="Duplicates collapsed" value={String(statsSummary.deduplicated)} />
                <Stat label="Tool calls made" value={String(statsSummary.toolCalls)} />
                <Stat label="Investigation ended" value={statsSummary.stoppedReason} />
                <Stat label="Elapsed" value={`${Math.round(statsSummary.elapsedSeconds)}s`} />
                <Stat label="Reviews in run" value={String(statsSummary.reviews)} />
              </div>
              {Object.keys(statsSummary.droppedByReason).length > 0 ? (
                <p className="mt-4 border-t border-slate-100 pt-3 text-xs text-slate-500">
                  Why claims were dropped:{" "}
                  {Object.entries(statsSummary.droppedByReason)
                    .map(([reason, count]) => `${reason} (${count})`)
                    .join(", ")}
                </p>
              ) : null}
            </div>
          </details>

          <button
            type="button"
            onClick={() => void handleGenerate()}
            className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
          >
            Regenerate report
          </button>
        </>
      ) : null}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
        {label}
      </p>
      <p className="mt-0.5 truncate text-base font-semibold text-slate-900" title={value}>
        {value}
      </p>
    </div>
  );
}
