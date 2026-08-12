import Link from "next/link";
import ReportView from "@/components/ReportView";

export default async function ReportPage(
  props: PageProps<"/runs/[jobId]/report">,
) {
  const { jobId } = await props.params;
  return (
    <main className="mx-auto w-full max-w-4xl flex-1 px-4 py-10">
      <Link
        href="/"
        className="text-sm text-slate-500 transition hover:text-slate-700"
      >
        ← Home
      </Link>
      <h1 className="mt-2 font-mono text-lg font-semibold text-slate-900">
        Run {jobId}
      </h1>
      <ReportView jobId={jobId} />
    </main>
  );
}
