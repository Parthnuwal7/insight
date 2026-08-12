"use client";

import { useState } from "react";
import Link from "next/link";
import type { JobResult } from "@/lib/types";
import AspectRankings from "./explore/AspectRankings";
import AspectSentimentHeatmap from "./explore/AspectSentimentHeatmap";
import KpiTiles from "./explore/KpiTiles";
import ReviewTable from "./explore/ReviewTable";
import SentimentDistribution from "./explore/SentimentDistribution";

type Tab = "overview" | "rankings" | "reviews";

const TABS: Array<{ key: Tab; label: string }> = [
  { key: "overview", label: "Overview" },
  { key: "rankings", label: "Aspect rankings" },
  { key: "reviews", label: "Reviews" },
];

export default function ResultsView({
  jobId,
  result,
}: {
  jobId: string;
  result: JobResult;
}) {
  const [tab, setTab] = useState<Tab>("overview");

  return (
    <div className="mt-6 space-y-6">
      <KpiTiles result={result} />

      <div className="flex items-center justify-between gap-4 rounded-xl border border-violet-200 bg-violet-50 p-4">
        <div>
          <p className="text-sm font-semibold text-violet-900">
            Insight report
          </p>
          <p className="text-sm text-violet-700">
            Grounded findings drawn from this run&apos;s reviews, each traceable
            to the reviews it rests on. Runs the investigation agent and
            verifier — can take several minutes.
          </p>
        </div>
        <Link
          href={`/runs/${jobId}/report`}
          className="shrink-0 rounded-lg bg-violet-700 px-4 py-2 text-sm font-medium text-white transition hover:bg-violet-600"
        >
          Open report
        </Link>
      </div>

      <nav className="flex gap-1 border-b border-slate-200">
        {TABS.map((t) => (
          <button
            key={t.key}
            type="button"
            onClick={() => setTab(t.key)}
            className={`-mb-px rounded-t-lg border-b-2 px-4 py-2 text-sm font-medium transition ${
              tab === t.key
                ? "border-slate-900 text-slate-900"
                : "border-transparent text-slate-500 hover:text-slate-700"
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {tab === "overview" ? (
        <div className="grid gap-4 lg:grid-cols-2">
          <SentimentDistribution result={result} />
          <AspectSentimentHeatmap result={result} />
        </div>
      ) : null}

      {tab === "rankings" ? <AspectRankings result={result} /> : null}

      {tab === "reviews" ? <ReviewTable result={result} /> : null}
    </div>
  );
}
