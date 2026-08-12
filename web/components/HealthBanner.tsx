"use client";

import { useEffect, useState } from "react";
import { getHealth } from "@/lib/api";
import type { Health } from "@/lib/types";

type BannerState =
  | { kind: "loading" }
  | { kind: "healthy"; health: Health }
  | { kind: "degraded"; health: Health }
  | { kind: "unreachable"; message: string }
  | { kind: "error"; message: string };

const TRANSLATION_LABEL = {
  available: "translation available",
  unavailable: "translation unavailable",
} as const;

export default function HealthBanner() {
  const [state, setState] = useState<BannerState>({ kind: "loading" });

  useEffect(() => {
    let active = true;
    getHealth().then((res) => {
      if (!active) return;
      if (res.ok) {
        const health = res.data as Health;
        if (health.status === "healthy") setState({ kind: "healthy", health });
        else if (health.status === "degraded")
          setState({ kind: "degraded", health });
        else
          setState({
            kind: "error",
            message: health.absa_error ?? "Unknown health status.",
          });
      } else {
        setState({ kind: "unreachable", message: res.message });
      }
    });
    return () => {
      active = false;
    };
  }, []);

  if (state.kind === "loading") {
    return (
      <div className="mb-8 rounded-lg border border-slate-200 bg-white px-4 py-3 text-sm text-slate-500">
        Checking backend health…
      </div>
    );
  }

  if (state.kind === "healthy" || state.kind === "degraded") {
    const { health } = state;
    const tone =
      state.kind === "healthy"
        ? "border-emerald-200 bg-emerald-50 text-emerald-800"
        : "border-amber-300 bg-amber-50 text-amber-900";
    return (
      <div className={`mb-8 rounded-lg border px-4 py-3 text-sm ${tone}`}>
        <span className="font-semibold">
          {state.kind === "healthy"
            ? "Backend healthy — real ABSA extraction is available"
            : "Backend degraded — keyword matching, not real ABSA"}
        </span>
        <span className="opacity-90">
          {" · "}
          {TRANSLATION_LABEL[health.translation_service]}
        </span>
        {health.absa_error ? (
          <span className="mt-1 block font-mono text-xs opacity-70">
            {health.absa_error}
          </span>
        ) : null}
      </div>
    );
  }

  if (state.kind === "error") {
    return (
      <div className="mb-8 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-900">
        <span className="font-semibold">Backend error</span> — {state.message}
      </div>
    );
  }

  return (
    <div className="mb-8 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-900">
      <span className="font-semibold">Backend unreachable</span> — {state.message}.
      <span className="mt-1 block text-red-700">
        Start it with{" "}
        <code className="font-mono text-xs">
          cd ABSA && python -m uvicorn app:app --port 7860
        </code>{" "}
        then refresh.
      </span>
    </div>
  );
}
