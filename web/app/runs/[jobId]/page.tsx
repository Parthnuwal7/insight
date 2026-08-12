import Link from "next/link";
import JobProgress from "@/components/JobProgress";

export default async function JobPage(
  props: PageProps<"/runs/[jobId]">,
) {
  const { jobId } = await props.params;
  return (
    <main className="mx-auto w-full max-w-4xl flex-1 px-4 py-10">
      <Link
        href="/"
        className="text-sm text-slate-500 transition hover:text-slate-700"
      >
        ← Back to upload
      </Link>
      <h1 className="mt-2 font-mono text-lg font-semibold text-slate-900">
        Run {jobId}
      </h1>
      <JobProgress jobId={jobId} />
    </main>
  );
}
