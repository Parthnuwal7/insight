# Insights frontend

Next.js (App Router) frontend for the Insights aspect-based sentiment analysis
platform. Pure presentation plus orchestration: the FastAPI backend does all the
work, reached through route handlers in `app/api/` that proxy over JSON.

## Run

Requires the backend on `http://localhost:7860` (see the repo root `README.md`):

```bash
npm install
cp .env.example .env.local   # set BACKEND_API_URL if not http://localhost:7860
npm run dev
```

Open http://localhost:3000.

## Layout

```
app/
├── page.tsx                   upload + health + recent runs
├── runs/[jobId]/page.tsx      job progress → results (charts)
├── runs/[jobId]/report/       the grounded insight report
└── api/                       route handlers proxying to FastAPI
    ├── health/                GET /health
    ├── jobs/                  submit + list jobs
    ├── jobs/[jobId]/          status, results, cancel
    └── report/                POST /insights/report (15-min server timeout)
lib/
├── api.ts                     typed client → route handlers
├── types.ts                   backend contract types
├── citations.ts               claim review_ids → review text resolution
└── results-store.ts           processed_data persisted for citation rendering
components/
├── UploadCsv.tsx              client-side CSV parse + validation + submit
├── JobProgress.tsx            polling, stage/chunk progress, cancel
├── ResultsView.tsx            KPIs, charts, aspect rankings, review drill-down
├── ReportView.tsx             sections + caveats + honest empty state
├── ClaimCard.tsx              a claim + its expandable citations
└── explore/                   the charts
```

## Design notes

- **`id` is the join key.** Every claim's `review_ids` resolve against the
  run's `processed_data`, which is stashed in sessionStorage after processing
  and re-read by the report route — no re-fetch to render a quote.
- **The slow report call stays server-side.** `/api/report` holds the
  connection to `/insights/report` for up to 15 minutes; the browser shows an
  honest "this can take several minutes" state instead of a hung spinner.
- **An empty report is rendered honestly.** Nothing is substituted with
  encouraging prose; caveats and stats carry the explanation.
- **Degradation is visible.** `extraction_method` / `degraded_reason` are
  surfaced in the review drill-down and KPI tiles.
