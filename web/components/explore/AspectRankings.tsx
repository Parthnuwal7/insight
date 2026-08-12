"use client";

import type { JobResult } from "@/lib/types";

/** Below this many mentions, a "100% negative" aspect is one person saying
 *  one thing once. Those rows are kept — a single complaint can matter —
 *  but they are moved out of the ranked list, because `priority_score` does
 *  not discount for sample size and so lets them outrank aspects with real
 *  repetition behind them. See the note rendered under each table. */
const THIN_EVIDENCE_BELOW = 2;

interface RankedRow {
  aspect: string;
  score: number;
  pct: number;
  frequency: number;
}

function RankingTable({
  title,
  tone,
  barClass,
  scoreHeader,
  pctHeader,
  rows,
}: {
  title: string;
  tone: string;
  barClass: string;
  scoreHeader: string;
  pctHeader: string;
  rows: RankedRow[];
}) {
  const ranked = rows.filter((r) => r.frequency >= THIN_EVIDENCE_BELOW);
  const thin = rows.filter((r) => r.frequency < THIN_EVIDENCE_BELOW);
  const maxScore = Math.max(1, ...ranked.map((r) => r.score));

  return (
    <div className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
      <h3 className={`px-4 py-3 text-sm font-semibold ${tone}`}>{title}</h3>

      {ranked.length === 0 ? (
        <p className="px-4 py-3 text-sm text-slate-500">
          No aspect was mentioned more than once in this run.
        </p>
      ) : (
        <table className="min-w-full divide-y divide-slate-100 text-sm">
          <thead className="bg-slate-50">
            <tr>
              <th className="px-4 py-2 text-left font-medium text-slate-500">Aspect</th>
              <th className="px-4 py-2 text-left font-medium text-slate-500">
                {scoreHeader}
              </th>
              <th className="px-4 py-2 text-right font-medium text-slate-500">
                {pctHeader}
              </th>
              <th className="px-4 py-2 text-right font-medium text-slate-500">
                Mentions
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-50">
            {ranked.map((row) => (
              <tr key={row.aspect} className="hover:bg-slate-50">
                <td className="px-4 py-2 font-medium text-slate-800">{row.aspect}</td>
                <td className="px-4 py-2">
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-16 overflow-hidden rounded-full bg-slate-100">
                      <div
                        className={`h-full rounded-full ${barClass}`}
                        style={{ width: `${(row.score / maxScore) * 100}%` }}
                      />
                    </div>
                    <span className="tabular-nums text-slate-600">
                      {row.score.toFixed(1)}
                    </span>
                  </div>
                </td>
                <td className="px-4 py-2 text-right tabular-nums text-slate-600">
                  {row.pct}%
                </td>
                <td className="px-4 py-2 text-right tabular-nums text-slate-600">
                  {row.frequency}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {thin.length > 0 ? (
        <details className="border-t border-slate-100">
          <summary className="cursor-pointer px-4 py-2 text-xs text-slate-500">
            {thin.length} aspect{thin.length === 1 ? "" : "s"} mentioned once —
            shown separately, not ranked
          </summary>
          <div className="flex flex-wrap gap-1.5 px-4 pb-3 pt-1">
            {thin.map((row) => (
              <span
                key={row.aspect}
                className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-600"
                title={`${row.pct}% · 1 mention`}
              >
                {row.aspect}
              </span>
            ))}
          </div>
        </details>
      ) : null}
    </div>
  );
}

/** The dual ranking the backend computes: areas of improvement (priority
 *  score) vs strength anchors (strength score). Both are grouped on the
 *  canonical aspect server-side. Note that an aspect can legitimately
 *  appear in both tables — the thresholds overlap (>10% negative, >30%
 *  positive), so a divisive aspect shows up as both a problem and a
 *  strength. That is real, not a bug. */
export default function AspectRankings({ result }: { result: JobResult }) {
  const improvement: RankedRow[] = (result.areas_of_improvement ?? []).map((r) => ({
    aspect: r.aspect,
    score: Number(r.priority_score),
    pct: r.negativity_pct,
    frequency: r.frequency,
  }));
  const strengths: RankedRow[] = (result.strength_anchors ?? []).map((r) => ({
    aspect: r.aspect,
    score: Number(r.strength_score),
    pct: r.positivity_pct,
    frequency: r.frequency,
  }));

  const both = improvement
    .filter((i) => strengths.some((s) => s.aspect === i.aspect))
    .filter((i) => i.frequency >= THIN_EVIDENCE_BELOW)
    .map((i) => i.aspect);

  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-2">
        <RankingTable
          title="Areas of improvement — ranked by priority"
          tone="text-red-800"
          barClass="bg-red-500"
          scoreHeader="Priority"
          pctHeader="Negativity"
          rows={improvement}
        />
        <RankingTable
          title="Strength anchors — ranked by strength"
          tone="text-emerald-800"
          barClass="bg-emerald-600"
          scoreHeader="Strength"
          pctHeader="Positivity"
          rows={strengths}
        />
      </div>

      {both.length > 0 ? (
        <p className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-2.5 text-xs leading-relaxed text-slate-600">
          <span className="font-medium text-slate-700">Divisive:</span>{" "}
          {both.join(", ")} appear in both tables — reviewers disagree about
          them. Worth reading directly rather than trusting either score.
        </p>
      ) : null}
    </div>
  );
}
