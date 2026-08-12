"use client";

import { useMemo } from "react";
import type { JobResult } from "@/lib/types";
import { buildAspectGroups } from "@/lib/aspect-groups";

interface Tile {
  label: string;
  value: string;
  detail?: string;
  /** Word-valued tiles wrap at a smaller size instead of truncating. A
   *  tile reading "interfa…" has told the user nothing. */
  text?: boolean;
  tone?: "neutral" | "warn";
}

/** Most-cited negative concept, grouped the same way the heatmap groups so
 *  the headline tile and the chart below it cannot disagree. */
function topComplaintAspect(result: JobResult): { label: string; count: number } | null {
  const rows = result.aspect_level_data ?? [];
  const groups = buildAspectGroups(rows.map((r) => r.aspect_canonical || r.aspect));
  const counts = new Map<string, number>();
  for (const row of rows) {
    if (row.aspect_sentiment !== "Negative") continue;
    const canonical = (row.aspect_canonical || row.aspect || "").toLowerCase();
    if (!canonical) continue;
    const label = groups.get(canonical)?.label ?? canonical;
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }
  let top: { label: string; count: number } | null = null;
  for (const [label, count] of counts) {
    if (!top || count > top.count) top = { label, count };
  }
  return top;
}

/**
 * Extraction health, counted the way `InsightTools.get_extraction_health`
 * counts it — by `extraction_method`, never by `degraded_reason`.
 *
 * This tile used to count rows with a truthy `degraded_reason`, which is a
 * different population: it swept in rows that escalation had repaired and
 * rows whose method was fine. The result was the dashboard reporting "5"
 * while the report page reported "13%" (= 4 of 30) for what reads as the
 * same quantity. That method docstring names itself the single source of
 * truth precisely to stop this; the frontend now reads from it too.
 */
function extractionHealth(result: JobResult) {
  const rows = result.processed_data ?? [];
  let healthy = 0;
  let escalated = 0;
  for (const r of rows) {
    if (r.extraction_method === "pyabsa") healthy += 1;
    else if (r.extraction_method === "llm_escalated") escalated += 1;
  }
  const degraded = rows.length - healthy - escalated;
  return { total: rows.length, degraded, escalated };
}

export default function KpiTiles({ result }: { result: JobResult }) {
  const summary = result.summary;
  const tiles = useMemo<Tile[]>(() => {
    const total = summary.total_reviews;
    const dist = summary.sentiment_distribution ?? {};
    const pct = (key: string) =>
      total > 0 ? Math.round(((dist[key] ?? 0) / total) * 100) : 0;

    const intentDist = summary.intents_distribution ?? {};
    const topIntent = Object.entries(intentDist).sort((a, b) => b[1] - a[1])[0];
    const intentShare =
      topIntent && total > 0 ? Math.round((topIntent[1] / total) * 100) : 0;

    const complaint = topComplaintAspect(result);
    const health = extractionHealth(result);

    return [
      { label: "Total reviews", value: total.toLocaleString(), detail: "processed" },
      {
        label: "Positive sentiment",
        value: `${pct("Positive")}%`,
        detail: `${dist["Positive"] ?? 0} reviews`,
      },
      {
        label: "Negative sentiment",
        value: `${pct("Negative")}%`,
        detail: `${dist["Negative"] ?? 0} reviews`,
      },
      {
        label: "Top complaint",
        value: complaint?.label ?? "—",
        detail: complaint
          ? `${complaint.count} negative mention${complaint.count === 1 ? "" : "s"}`
          : "no negative aspects",
        text: true,
      },
      {
        label: "Dominant intent",
        value: topIntent?.[0] ?? "—",
        detail: topIntent ? `${intentShare}% of reviews` : "—",
        text: true,
      },
      {
        label: "Extraction",
        value:
          health.degraded === 0
            ? "clean"
            : `${Math.round((health.degraded / Math.max(1, health.total)) * 100)}% degraded`,
        detail:
          health.degraded === 0
            ? health.escalated > 0
              ? `${health.escalated} repaired by LLM`
              : "all rows real ABSA"
            : `${health.degraded} of ${health.total} not from real ABSA`,
        text: health.degraded > 0,
        tone: health.degraded > 0 ? "warn" : "neutral",
      },
    ];
  }, [result, summary]);

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
      {tiles.map((tile) => (
        <div
          key={tile.label}
          className={`rounded-lg border bg-white p-4 shadow-sm ${
            tile.tone === "warn" ? "border-amber-300" : "border-slate-200"
          }`}
        >
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
            {tile.label}
          </p>
          <p
            className={
              tile.text
                ? "mt-1 break-words text-base font-semibold leading-tight text-slate-900"
                : "mt-1 text-2xl font-bold text-slate-900"
            }
            title={tile.value}
          >
            {tile.value}
          </p>
          {tile.detail ? (
            <p className="mt-0.5 text-xs leading-tight text-slate-500">{tile.detail}</p>
          ) : null}
        </div>
      ))}
    </div>
  );
}
