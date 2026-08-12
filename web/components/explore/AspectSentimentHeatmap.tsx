"use client";

import { useMemo, useState } from "react";
import type { JobResult } from "@/lib/types";
import { buildAspectGroups, isContainerAspect } from "@/lib/aspect-groups";

const SENTIMENTS = ["Positive", "Neutral", "Negative"] as const;
type Sentiment = (typeof SENTIMENTS)[number];

/** Per-sentiment hue. The previous version painted every column with the
 *  same red and scaled it by the row total, so a 3-Positive row rendered as
 *  the most alarming cell on the board — the colour channel said "problem"
 *  about the run's strongest aspect. Colour now encodes polarity and
 *  saturation encodes count within that polarity. */
const HUES: Record<Sentiment, { rgb: string; text: string }> = {
  Positive: { rgb: "22, 163, 74", text: "#14532d" },
  Neutral: { rgb: "100, 116, 139", text: "#1e293b" },
  Negative: { rgb: "220, 38, 38", text: "#7f1d1d" },
};

interface Row {
  label: string;
  members: string[];
  counts: Record<Sentiment, number>;
  total: number;
  /** Positive − Negative. The number the eye should find first. */
  net: number;
}

export default function AspectSentimentHeatmap({ result }: { result: JobResult }) {
  const [grouped, setGrouped] = useState(true);
  const [minMentions, setMinMentions] = useState(2);

  const { rows, hiddenThin, maxCell } = useMemo(() => {
    const aspectRows = result.aspect_level_data ?? [];
    const groups = buildAspectGroups(
      aspectRows.map((r) => r.aspect_canonical || r.aspect),
    );

    const acc = new Map<string, Row>();
    for (const row of aspectRows) {
      const canonical = (row.aspect_canonical || row.aspect || "").toLowerCase();
      if (!canonical) continue;
      const group = groups.get(canonical);
      const label = grouped && group ? group.label : canonical;
      const members = grouped && group ? group.members : [canonical];

      if (!acc.has(label)) {
        acc.set(label, {
          label,
          members,
          counts: { Positive: 0, Neutral: 0, Negative: 0 },
          total: 0,
          net: 0,
        });
      }
      const entry = acc.get(label)!;
      if ((SENTIMENTS as readonly string[]).includes(row.aspect_sentiment)) {
        entry.counts[row.aspect_sentiment as Sentiment] += 1;
        entry.total += 1;
      }
    }

    const all = [...acc.values()];
    for (const r of all) r.net = r.counts.Positive - r.counts.Negative;

    const kept = all
      .filter((r) => r.total >= minMentions)
      // Loudest signal first: strongest net polarity, then volume. Sorting
      // by raw volume alone buried a 3-mention 67%-negative aspect under
      // singletons that happened to tie on count.
      .sort((a, b) => Math.abs(b.net) - Math.abs(a.net) || b.total - a.total)
      .slice(0, 15);

    return {
      rows: kept,
      hiddenThin: all.length - all.filter((r) => r.total >= minMentions).length,
      maxCell: Math.max(1, ...kept.flatMap((r) => Object.values(r.counts))),
    };
  }, [result, grouped, minMentions]);

  const maxBar = Math.max(1, ...rows.map((r) => Math.max(r.counts.Positive, r.counts.Negative)));

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <h3 className="text-sm font-semibold text-slate-900">
          Aspect sentiment{" "}
          <span className="font-normal text-slate-500">
            ({rows.length} shown)
          </span>
        </h3>
        <div className="flex items-center gap-4 text-xs text-slate-600">
          <label className="flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={grouped}
              onChange={(e) => setGrouped(e.target.checked)}
              className="h-3.5 w-3.5 accent-slate-700"
            />
            Group synonyms
          </label>
          <label className="flex items-center gap-1.5">
            Min mentions
            <select
              value={minMentions}
              onChange={(e) => setMinMentions(Number(e.target.value))}
              className="rounded border border-slate-300 bg-white px-1.5 py-0.5"
            >
              {[1, 2, 3, 5].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>

      {rows.length === 0 ? (
        <p className="py-6 text-center text-sm text-slate-500">
          No aspect reached {minMentions} mention{minMentions === 1 ? "" : "s"} in
          this run. Lower the threshold to see thinner evidence.
        </p>
      ) : (
        <div className="space-y-1">
          <div className="grid grid-cols-[minmax(7rem,1.2fr)_repeat(3,2rem)_minmax(6rem,1fr)] items-center gap-x-1 pb-1 text-[11px] font-medium text-slate-500">
            <div />
            {SENTIMENTS.map((s) => (
              <div key={s} className="text-center" title={s}>
                {s[0]}
              </div>
            ))}
            <div className="pl-2">← negative · positive →</div>
          </div>

          {rows.map((row) => (
            <div
              key={row.label}
              className="grid grid-cols-[minmax(7rem,1.2fr)_repeat(3,2rem)_minmax(6rem,1fr)] items-center gap-x-1"
            >
              <div className="truncate pr-2 text-xs">
                <span
                  className="font-medium text-slate-700"
                  title={
                    row.members.length > 1
                      ? `grouped: ${row.members.join(", ")}`
                      : row.label
                  }
                >
                  {row.label}
                </span>
                {row.members.length > 1 ? (
                  <span className="ml-1 rounded bg-slate-100 px-1 text-[10px] text-slate-500">
                    +{row.members.length - 1}
                  </span>
                ) : null}
                {isContainerAspect(row.label) ? (
                  <span
                    className="ml-1 text-[10px] text-slate-400"
                    title="Names the product, not a facet of it — not actionable on its own."
                  >
                    ⌾
                  </span>
                ) : null}
                <span className="ml-1 text-slate-400">({row.total})</span>
              </div>

              {SENTIMENTS.map((s) => {
                const count = row.counts[s];
                const hue = HUES[s];
                return (
                  <div
                    key={s}
                    className="flex h-6 items-center justify-center rounded text-xs tabular-nums"
                    style={{
                      backgroundColor:
                        count === 0
                          ? "transparent"
                          : `rgba(${hue.rgb}, ${0.15 + (count / maxCell) * 0.6})`,
                      color: count === 0 ? "#cbd5e1" : hue.text,
                    }}
                  >
                    {count}
                  </div>
                );
              })}

              <DivergingBar
                positive={row.counts.Positive}
                negative={row.counts.Negative}
                max={maxBar}
              />
            </div>
          ))}
        </div>
      )}

      <p className="mt-3 border-t border-slate-100 pt-2 text-[11px] leading-relaxed text-slate-500">
        {hiddenThin > 0 ? (
          <>
            {hiddenThin} aspect{hiddenThin === 1 ? "" : "s"} mentioned fewer than{" "}
            {minMentions} time{minMentions === 1 ? "" : "s"} hidden.{" "}
          </>
        ) : null}
        {grouped
          ? "Synonym grouping is display-only — the stored aspects are unchanged, and a grouped row lists its members on hover."
          : "Showing raw canonical aspects; one concept may appear under several spellings."}
      </p>
    </div>
  );
}

/** Counts as length, polarity as direction. Length is the honest encoding
 *  for a count — colour saturation across a 0–4 range carries almost none. */
function DivergingBar({
  positive,
  negative,
  max,
}: {
  positive: number;
  negative: number;
  max: number;
}) {
  return (
    <div className="flex h-4 items-center pl-2" aria-hidden>
      <div className="flex w-1/2 justify-end">
        <div
          className="h-2.5 rounded-l-sm bg-red-500"
          style={{ width: `${(negative / max) * 100}%` }}
        />
      </div>
      <div className="h-3 w-px shrink-0 bg-slate-300" />
      <div className="w-1/2">
        <div
          className="h-2.5 rounded-r-sm bg-emerald-600"
          style={{ width: `${(positive / max) * 100}%` }}
        />
      </div>
    </div>
  );
}
