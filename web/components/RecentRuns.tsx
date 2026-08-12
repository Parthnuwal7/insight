"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { listJobs } from "@/lib/api";
import type { JobStatus } from "@/lib/types";
import { getUserId } from "@/lib/user";

const STATUS_TONE: Record<JobStatus["status"], string> = {
  pending: "bg-slate-100 text-slate-600",
  running: "bg-blue-100 text-blue-700",
  completed: "bg-emerald-100 text-emerald-700",
  failed: "bg-red-100 text-red-700",
  cancelled: "bg-slate-200 text-slate-600",
};

export default function RecentRuns() {
  const [jobs, setJobs] = useState<JobStatus[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    listJobs(getUserId()).then((res) => {
      if (!active) return;
      if (res.ok) setJobs(res.data.jobs ?? []);
      else setError(res.message);
    });
    return () => {
      active = false;
    };
  }, []);

  return (
    <section className="mt-8">
      <h2 className="text-lg font-semibold text-slate-900">Recent runs</h2>
      {error ? (
        <p className="mt-2 text-sm text-slate-500">
          Could not load recent runs ({error}).
        </p>
      ) : jobs === null ? (
        <p className="mt-2 text-sm text-slate-500">Loading…</p>
      ) : jobs.length === 0 ? (
        <p className="mt-2 text-sm text-slate-500">
          Nothing here yet — process a CSV above and it will show up.
        </p>
      ) : (
        <ul className="mt-3 divide-y divide-slate-200 rounded-xl border border-slate-200 bg-white">
          {jobs.slice(0, 10).map((job) => (
            <li key={job.id}>
              <Link
                href={`/runs/${job.id}`}
                className="flex items-center justify-between gap-4 px-4 py-3 text-sm transition hover:bg-slate-50"
              >
                <span className="font-mono text-xs text-slate-500">{job.id}</span>
                <span
                  className={`rounded-full px-2 py-0.5 text-xs font-medium capitalize ${STATUS_TONE[job.status]}`}
                >
                  {job.status}
                </span>
                <span className="text-xs text-slate-500">
                  {new Date(job.created_at * 1000).toLocaleString()}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
