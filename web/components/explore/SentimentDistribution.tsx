"use client";

import { useMemo } from "react";
import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { JobResult } from "@/lib/types";

const SENTIMENT_COLORS: Record<string, string> = {
  Positive: "#16a34a",
  Neutral: "#2563eb",
  Negative: "#dc2626",
};

const ORDER = ["Positive", "Neutral", "Negative"];

export default function SentimentDistribution({ result }: { result: JobResult }) {
  const data = useMemo(() => {
    const dist = result.summary?.sentiment_distribution ?? {};
    return ORDER.filter((key) => (dist[key] ?? 0) > 0).map((key) => ({
      name: key,
      count: dist[key] ?? 0,
      color: SENTIMENT_COLORS[key] ?? "#64748b",
    }));
  }, [result]);

  if (data.length === 0) {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-4 text-sm text-slate-500">
        No sentiment data for this run.
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <h3 className="mb-3 text-sm font-semibold text-slate-900">
        Sentiment distribution
      </h3>
      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 8 }}>
            <XAxis type="number" allowDecimals={false} fontSize={12} tickLine={false} />
            <YAxis
              type="category"
              dataKey="name"
              width={80}
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip cursor={{ fill: "rgba(0,0,0,0.03)" }} />
            <Bar dataKey="count" name="Reviews" radius={[0, 4, 4, 0]}>
              {data.map((entry) => (
                <Cell key={entry.name} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
