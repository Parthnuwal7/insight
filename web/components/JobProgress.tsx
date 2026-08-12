"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { cancelJob, getJobResults, getJobStatus } from "@/lib/api";
import type { JobResult, JobStatus } from "@/lib/types";
import { storeResults } from "@/lib/results-store";
import ResultsView from "./ResultsView";

/**
 * The pipeline's stages, in the order `absa/pipeline.py` reports them,
 * with each one's approximate share of a chunk's wall time.
 *
 * These shares are ESTIMATES, not measurements — the backend reports which
 * stage it is in, never how far through it is (`JobStoreProgress.advance`
 * is a deliberate no-op, because within-chunk progress would fight the
 * runner for the same field). Extraction dominates because it is the stage
 * that runs the 1.1GB PyABSA checkpoint.
 *
 * Driving the bar off these is what makes a single-chunk run legible at
 * all: with the default CHUNK_SIZE of 100, any run under 100 reviews is
 * exactly one chunk, so a purely chunk-driven bar sat at 0% and then
 * jumped to 100% with nothing in between. It was not wrong — there really
 * was one chunk — it just had nothing to say.
 */
const STAGES: Array<{ key: string; label: string; weight: number }> = [
  { key: "validation", label: "Validating data", weight: 0.02 },
  { key: "translating", label: "Translating reviews", weight: 0.15 },
  { key: "classifying_intent", label: "Classifying intent", weight: 0.08 },
  { key: "extracting", label: "Extracting aspects", weight: 0.65 },
  { key: "combining_results", label: "Combining results", weight: 0.05 },
  { key: "analytics", label: "Computing analytics", weight: 0.05 },
];

const TERMINAL_STAGE = "completed";

function stageIndex(stage: string | null): number {
  if (!stage) return -1;
  return STAGES.findIndex((s) => s.key === stage);
}

function stageLabel(stage: string | null): string {
  if (!stage) return "Preparing";
  if (stage === TERMINAL_STAGE) return "Completed";
  return STAGES.find((s) => s.key === stage)?.label ?? stage;
}

/** Fraction of the CURRENT chunk that is done, from its reported stage.
 *  A stage is credited only for the stages *before* it — the one in flight
 *  counts as zero, so the bar never claims progress the backend has not
 *  actually reported reaching the end of. */
function fractionThroughChunk(stage: string | null): number {
  if (stage === TERMINAL_STAGE) return 1;
  const idx = stageIndex(stage);
  if (idx < 0) return 0;
  return STAGES.slice(0, idx).reduce((sum, s) => sum + s.weight, 0);
}

const POLL_INTERVAL_MS = 2000;

export default function JobProgress({ jobId }: { jobId: string }) {
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [result, setResult] = useState<JobResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [cancelledNotice, setCancelledNotice] = useState(false);
  const [loading, setLoading] = useState(true);

  const stopped = useRef(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    stopped.current = false;

    async function poll() {
      const statusRes = await getJobStatus(jobId);
      if (stopped.current) return;

      if (!statusRes.ok) {
        setError(statusRes.message);
        setStatus(null);
        setLoading(false);
        return;
      }

      const s = statusRes.data;
      setStatus(s);
      setLoading(false);

      // True once the run reached a terminal state we can render.
      let finished = true;

      if (s.status === "completed") {
        const resultRes = await getJobResults(jobId);
        if (stopped.current) return;
        if (resultRes.ok) {
          setResult(resultRes.data);
          storeResults(jobId, resultRes.data);
        } else if (resultRes.status === 409) {
          // Results not merged yet — edge race; poll again.
          finished = false;
        } else {
          setError(resultRes.message);
        }
      } else if (s.status === "pending" || s.status === "running") {
        finished = false;
      }

      if (finished) return;
      timerRef.current = setTimeout(() => void poll(), POLL_INTERVAL_MS);
    }

    void poll();
    return () => {
      stopped.current = true;
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [jobId]);

  async function handleCancel() {
    if (!status) return;
    setCancelling(true);
    const res = await cancelJob(jobId);
    if (res.ok) {
      setCancelledNotice(true);
    } else {
      setError(res.message);
    }
    setCancelling(false);
  }

  if (result) {
    return <ResultsView jobId={jobId} result={result} />;
  }

  if (error) {
    return (
      <div className="mt-6 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-900">
        {error}
        <p className="mt-1 text-red-700">
          Double-check the job id in the URL, and that the backend is running.
        </p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="mt-6 rounded-xl border border-slate-200 bg-white p-6 text-sm text-slate-500">
        Loading job status…
      </div>
    );
  }

  if (!status) return null;

  if (status.status === "failed") {
    return (
      <div className="mt-6 rounded-lg border border-red-300 bg-red-50 px-4 py-4 text-sm text-red-900">
        <p className="font-semibold">This run failed.</p>
        {status.error ? (
          <p className="mt-1 whitespace-pre-wrap font-mono text-xs text-red-700">
            {status.error}
          </p>
        ) : null}
        <p className="mt-2 text-red-700">
          Nothing was discarded silently — the failure is shown above. You can{" "}
          <Link href="/" className="underline">
            try again with a fresh upload
          </Link>
          .
        </p>
      </div>
    );
  }

  if (status.status === "cancelled") {
    return (
      <div className="mt-6 rounded-lg border border-amber-300 bg-amber-50 px-4 py-4 text-sm text-amber-900">
        <p className="font-semibold">This run was cancelled.</p>
        <p className="mt-1">
          Partial results that had already been persisted per chunk are still
          in the job store, but no more work will happen.{" "}
          <Link href="/" className="underline">
            Start a new run
          </Link>{" "}
          if you still need answers.
        </p>
      </div>
    );
  }

  // Chunk-level progress, refined by how far through the current chunk the
  // reported stage implies we are. With one chunk this is entirely
  // stage-driven; with many it is mostly chunk-driven, which is the right
  // weighting in both cases.
  const totalChunks = status.total_chunks;
  const percent =
    totalChunks > 0
      ? Math.min(
          99,
          Math.round(
            ((status.completed_chunks + fractionThroughChunk(status.stage)) /
              totalChunks) *
              100,
          ),
        )
      : 0;

  const multiChunk = totalChunks > 1;

  return (
    <div className="mt-6 rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium text-slate-900">
          {stageLabel(status.stage)}
          {multiChunk ? (
            <span className="ml-2 text-slate-500">
              batch {Math.min(status.completed_chunks + 1, totalChunks)} of{" "}
              {totalChunks}
            </span>
          ) : null}
        </p>
        {cancelledNotice ? (
          <span className="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-600">
            Cancellation requested…
          </span>
        ) : (
          <button
            type="button"
            disabled={cancelling}
            onClick={() => void handleCancel()}
            className="rounded-lg border border-red-300 px-3 py-1.5 text-sm text-red-700 transition hover:bg-red-50 disabled:opacity-50"
          >
            {cancelling ? "Requesting…" : "Cancel run"}
          </button>
        )}
      </div>

      <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-slate-100">
        <div
          className="h-full rounded-full bg-slate-900 transition-all duration-500"
          style={{ width: totalChunks > 0 ? `${Math.max(percent, 4)}%` : "4%" }}
        />
      </div>

      {/* Stage checklist. For a single-chunk run this is the only real
          progress signal there is, and it was already being reported by the
          backend — it just never reached the bar. */}
      <ol className="mt-4 flex flex-wrap gap-x-4 gap-y-1.5">
        {STAGES.map((s, i) => {
          const current = stageIndex(status.stage);
          const done = status.stage === TERMINAL_STAGE || (current >= 0 && i < current);
          const active = i === current;
          return (
            <li
              key={s.key}
              className={`flex items-center gap-1.5 text-xs ${
                active
                  ? "font-medium text-slate-900"
                  : done
                    ? "text-slate-500"
                    : "text-slate-300"
              }`}
            >
              <span aria-hidden>{done ? "✓" : active ? "●" : "○"}</span>
              {s.label}
            </li>
          );
        })}
      </ol>

      <p className="mt-3 text-xs leading-relaxed text-slate-500">
        {totalChunks > 0 ? (
          <>
            {percent}% done.{" "}
            {multiChunk
              ? `Split into ${totalChunks} batches — each batch's result is persisted as it completes, so a restart resumes rather than restarts.`
              : "This run fits in a single batch, so the bar tracks pipeline stages rather than batches. Stage shares are estimates; the backend reports which stage it is in, not how far through it is."}
          </>
        ) : (
          "Preparing batches…"
        )}
      </p>
    </div>
  );
}
