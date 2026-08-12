import HealthBanner from "@/components/HealthBanner";
import RecentRuns from "@/components/RecentRuns";
import UploadCsv from "@/components/UploadCsv";

export default function Home() {
  return (
    <main className="mx-auto w-full max-w-4xl flex-1 px-4 py-10">
      <header className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-slate-900">
          Insights
        </h1>
        <p className="mt-2 text-slate-600">
          Aspect-based sentiment analysis over your review data. Upload a CSV,
          and every finding in the report is traceable to the reviews it came
          from.
        </p>
      </header>

      <HealthBanner />

      <UploadCsv />

      <RecentRuns />
    </main>
  );
}
