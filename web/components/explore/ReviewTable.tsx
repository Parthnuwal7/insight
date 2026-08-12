"use client";

import { useMemo, useState } from "react";
import type { JobResult, ProcessedRow } from "@/lib/types";
import { languageLabel } from "@/lib/citations";

const SENTIMENT_TONE: Record<string, string> = {
  Positive: "bg-emerald-50 text-emerald-700 border-emerald-200",
  Negative: "bg-red-50 text-red-700 border-red-200",
  Neutral: "bg-blue-50 text-blue-700 border-blue-200",
};

const EXTRACTION_LABEL: Record<string, string> = {
  pyabsa: "real ABSA",
  none: "none",
  llm_escalated: "LLM-repaired",
};

export default function ReviewTable({ result }: { result: JobResult }) {
  const [filter, setFilter] = useState<string>("all");
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  const reviews = useMemo<ProcessedRow[]>(() => result.processed_data ?? [], [result]);

  const filtered = useMemo(() => {
    if (filter === "all") return reviews;
    return reviews.filter((r) => r.overall_sentiment === filter);
  }, [reviews, filter]);

  function toggle(id: number) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-900">
          Reviews ({reviews.length})
        </h3>
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="rounded-md border border-slate-300 px-2 py-1 text-sm text-slate-700"
        >
          <option value="all">All sentiments</option>
          <option value="Positive">Positive</option>
          <option value="Neutral">Neutral</option>
          <option value="Negative">Negative</option>
        </select>
      </div>

      {filtered.length === 0 ? (
        <p className="text-sm text-slate-500">No reviews match this filter.</p>
      ) : (
        <ul className="max-h-[28rem] divide-y divide-slate-100 overflow-y-auto">
          {filtered.map((row) => {
            const isOpen = expanded.has(row.id);
            const sentiment = row.overall_sentiment;
            const degraded = Boolean(row.degraded_reason);
            return (
              <li key={row.id} className="py-3">
                <button
                  type="button"
                  onClick={() => toggle(row.id)}
                  className="block w-full text-left"
                >
                  <div className="flex items-start justify-between gap-3">
                    <p className="text-sm text-slate-800">{row.review}</p>
                    <span
                      className={`shrink-0 rounded-full border px-2 py-0.5 text-xs font-medium ${SENTIMENT_TONE[sentiment] ?? "bg-slate-100 text-slate-600"}`}
                    >
                      {sentiment}
                    </span>
                  </div>
                  <p className="mt-1 text-xs text-slate-500">
                    #{row.id} · {languageLabel(row.detected_language)} · intent{" "}
                    {row.intent}
                    {degraded ? " · degraded" : ""}
                  </p>
                </button>

                {isOpen ? (
                  <div className="mt-2 space-y-1 rounded-md border border-slate-100 bg-slate-50 p-3 text-xs text-slate-600">
                    {row.translated_review &&
                    row.translated_review !== row.review ? (
                      <p>
                        <span className="font-medium text-slate-700">
                          Translated:{" "}
                        </span>
                        {row.translated_review}
                      </p>
                    ) : null}
                    {row.aspects.length > 0 ? (
                      <p>
                        <span className="font-medium text-slate-700">Aspects: </span>
                        {row.aspects.map((a, i) => (
                          <span key={i}>
                            {a} ({row.aspect_sentiments[i] ?? "?"})
                            {i < row.aspects.length - 1 ? ", " : ""}
                          </span>
                        ))}
                      </p>
                    ) : null}
                    <p>
                      <span className="font-medium text-slate-700">Extraction: </span>
                      {EXTRACTION_LABEL[row.extraction_method] ?? row.extraction_method}
                      {row.degraded_reason ? (
                        <span className="text-red-600">
                          {" "}
                          — {row.degraded_reason}
                        </span>
                      ) : null}
                    </p>
                  </div>
                ) : null}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
