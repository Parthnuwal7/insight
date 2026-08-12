"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import Papa from "papaparse";
import { submitJob } from "@/lib/api";
import type { ProcessRow } from "@/lib/types";
import { getUserId } from "@/lib/user";

const REQUIRED_COLUMNS = ["review"] as const;
const OPTIONAL_COLUMNS = ["id", "reviews_title", "date", "user_id"] as const;

interface SampleOption {
  label: string;
  file: string;
}

const SAMPLE_OPTIONS: SampleOption[] = [
  { label: "Mobile App Reviews (30 rows)", file: "test_data_app_reviews.csv" },
  { label: "E-Commerce Product Reviews (22 rows)", file: "test_data_ecommerce.csv" },
  { label: "Restaurant & Dining Reviews (15 rows)", file: "test_data_restaurant.csv" },
];

interface ParsedCsv {
  fileName: string;
  rows: ProcessRow[];
  errors: string[];
  warnings: string[];
}

const today = new Date().toISOString().slice(0, 10);

/** Build ProcessRow[] from parsed rows, injecting missing/awkward columns
 * and reporting exactly what was changed. */
function toProcessRows(
  records: Array<Record<string, unknown>>,
): Pick<ParsedCsv, "rows" | "errors" | "warnings"> {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (records.length === 0) {
    return { rows: [], errors: ["The file contains no data rows."], warnings };
  }

  const headers = Object.keys(records[0]);
  const missingRequired = REQUIRED_COLUMNS.filter((c) => !headers.includes(c));
  if (missingRequired.length > 0) {
    errors.push(`Missing required column: ${missingRequired.join(", ")}.`);
    return { rows: [], errors, warnings };
  }

  const missingOptional = OPTIONAL_COLUMNS.filter((c) => !headers.includes(c));
  if (missingOptional.length > 0) {
    warnings.push(
      `Missing optional column(s): ${missingOptional.join(", ")}. Defaults will be applied.`,
    );
  }

  const hasIdColumn = headers.includes("id");
  const idRewrites = new Set<string>();

  const rows: ProcessRow[] = records.map((rec, index) => {
    const rowIndex = index + 1;
    let id = rowIndex;

    if (hasIdColumn) {
      const raw = rec.id;
      const parsed = Number(raw);
      if (Number.isInteger(parsed) && parsed > 0) {
        id = parsed;
      } else {
        idRewrites.add(String(raw ?? ""));
      }
    }

    const review = String(rec.review ?? "").trim();
    const reviewsTitle = String(rec.reviews_title ?? `Review ${id}`).trim() || `Review ${id}`;
    const date = String(rec.date ?? "").trim() || today;
    const userId = String(rec.user_id ?? `user_${id}`).trim() || `user_${id}`;

    return { id, reviews_title: reviewsTitle, review, date, user_id: userId };
  });

  if (idRewrites.size > 0) {
    warnings.push(
      "The CSV has an `id` column, but some values were not positive integers " +
        "(the backend requires integer ids). Those rows were renumbered by row index; " +
        "citations resolve against the renumbered ids.",
    );
  }

  const emptyReviews = rows.filter((r) => !r.review).length;
  if (emptyReviews > 0) {
    warnings.push(
      `${emptyReviews} row(s) have an empty review; the backend will still process them but they contribute nothing.`,
    );
  }

  return { rows, errors, warnings };
}

/** Decode bytes strictly as UTF-8; returns null if the file is not valid UTF-8. */
function decodeUtf8Strict(buffer: ArrayBuffer): string | null {
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(buffer);
  } catch {
    return null;
  }
}

export default function UploadCsv() {
  const router = useRouter();
  const [parsed, setParsed] = useState<ParsedCsv | null>(null);
  const [parsing, setParsing] = useState(false);
  const [parseError, setParseError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const previewRows = useMemo(() => parsed?.rows.slice(0, 5) ?? [], [parsed]);

  function handleParsed(
    next: { fileName: string; rows: ProcessRow[]; errors: string[]; warnings: string[] },
  ) {
    setParsed({
      fileName: next.fileName,
      rows: next.rows,
      errors: next.errors,
      warnings: next.warnings,
    });
    setParseError(null);
    setSubmitError(null);
  }

  async function handleFile(file: File) {
    setParsing(true);
    setParseError(null);
    try {
      const buffer = await file.arrayBuffer();
      const text = decodeUtf8Strict(buffer);
      if (text === null) {
        setParseError(
          "The file is not valid UTF-8. Re-save it as UTF-8 (no BOM) and try again.",
        );
        setParsed(null);
        return;
      }
      const result = Papa.parse<Record<string, unknown>>(text, {
        header: true,
        skipEmptyLines: true,
      });
      const { rows, errors, warnings } = toProcessRows(result.data);
      handleParsed({ fileName: file.name, rows, errors, warnings });
    } catch (e) {
      setParseError(
        `Could not read the file: ${e instanceof Error ? e.message : String(e)}`,
      );
    } finally {
      setParsing(false);
    }
  }

  async function handleSample(option: SampleOption) {
    setParsing(true);
    setParseError(null);
    try {
      const res = await fetch(`/sample-data/${option.file}`);
      if (!res.ok) {
        setParseError(`Could not load sample dataset (HTTP ${res.status}).`);
        return;
      }
      const text = await res.text();
      const result = Papa.parse<Record<string, unknown>>(text, {
        header: true,
        skipEmptyLines: true,
      });
      const { rows, errors, warnings } = toProcessRows(result.data);
      handleParsed({ fileName: option.file, rows, errors, warnings });
    } catch (e) {
      setParseError(
        `Could not load the sample: ${e instanceof Error ? e.message : String(e)}`,
      );
    } finally {
      setParsing(false);
    }
  }

  async function handleSubmit() {
    if (!parsed || parsed.rows.length === 0) return;
    setSubmitting(true);
    setSubmitError(null);
    const res = await submitJob({
      data: parsed.rows,
      user_id: getUserId(),
      options: { include_translation: true, include_aspects: true },
    });
    if (res.ok) {
      router.push(`/runs/${res.data.job_id}`);
      return;
    }
    setSubmitError(res.message);
    setSubmitting(false);
  }

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
      <h2 className="text-lg font-semibold text-slate-900">
        Process review data
      </h2>

      <div className="mt-4 space-y-4">
        <label className="block">
          <span className="mb-1 block text-sm font-medium text-slate-700">
            Upload a CSV
          </span>
          <input
            type="file"
            accept=".csv,text/csv"
            disabled={parsing || submitting}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void handleFile(file);
            }}
            className="block w-full cursor-pointer rounded-lg border border-slate-300 bg-white text-sm text-slate-600 file:mr-3 file:cursor-pointer file:rounded-md file:border-0 file:bg-slate-900 file:px-3 file:py-2 file:text-sm file:font-medium file:text-white"
          />
          <span className="mt-1 block text-xs text-slate-500">
            Requires a <code className="font-mono">review</code> column.
            Optional: <code className="font-mono">id</code>,{" "}
            <code className="font-mono">reviews_title</code>,{" "}
            <code className="font-mono">date</code>,{" "}
            <code className="font-mono">user_id</code> — sensible defaults are
            applied and the changes are shown below.
          </span>
        </label>

        <div>
          <span className="mb-1 block text-sm font-medium text-slate-700">
            Or try a sample dataset
          </span>
          <div className="flex flex-wrap gap-2">
            {SAMPLE_OPTIONS.map((option) => (
              <button
                key={option.file}
                type="button"
                disabled={parsing || submitting}
                onClick={() => void handleSample(option)}
                className="rounded-full border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-700 transition hover:border-slate-400 hover:bg-slate-50 disabled:opacity-50"
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {parsing ? (
        <p className="mt-4 text-sm text-slate-500">Parsing…</p>
      ) : null}

      {parseError ? (
        <div className="mt-4 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-900">
          {parseError}
        </div>
      ) : null}

      {parsed ? (
        <div className="mt-6 border-t border-slate-100 pt-4">
          {parsed.errors.length > 0 ? (
            <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-900">
              {parsed.errors.map((e) => (
                <p key={e}>{e}</p>
              ))}
            </div>
          ) : (
            <>
              <p className="text-sm text-slate-700">
                <span className="font-semibold">{parsed.fileName}</span> —{" "}
                {parsed.rows.length} review(s) ready to submit.
              </p>
              {parsed.warnings.map((w) => (
                <p
                  key={w}
                  className="mt-1 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800"
                >
                  {w}
                </p>
              ))}

              <div className="mt-3 overflow-x-auto rounded-lg border border-slate-200">
                <table className="min-w-full divide-y divide-slate-200 text-sm">
                  <thead className="bg-slate-50">
                    <tr>
                      <th className="px-3 py-2 text-left font-medium text-slate-500">id</th>
                      <th className="px-3 py-2 text-left font-medium text-slate-500">review</th>
                      <th className="px-3 py-2 text-left font-medium text-slate-500">date</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {previewRows.map((row) => (
                      <tr key={row.id}>
                        <td className="px-3 py-2 text-slate-600">{row.id}</td>
                        <td className="max-w-md truncate px-3 py-2 text-slate-800">
                          {row.review}
                        </td>
                        <td className="px-3 py-2 text-slate-600">{row.date}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {parsed.rows.length > 5 ? (
                  <p className="px-3 py-2 text-xs text-slate-500">
                    …and {parsed.rows.length - 5} more.
                  </p>
                ) : null}
              </div>

              {submitError ? (
                <div className="mt-3 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-900">
                  Submission failed: {submitError}
                </div>
              ) : null}

              <button
                type="button"
                disabled={submitting || parsed.rows.length === 0}
                onClick={() => void handleSubmit()}
                className="mt-4 rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {submitting ? "Submitting…" : "Process reviews"}
              </button>
            </>
          )}
        </div>
      ) : null}
    </section>
  );
}
