# Phase D — Next.js Frontend Implementation Plan

**Audience:** you, implementing by hand. This is written as a working document, not
a subagent brief — it assumes you know TypeScript and React but nothing about this
backend's quirks. Everything about the backend below was verified against the code
at `ABSA/app.py` on 2026-08-12, not taken from the design doc.

**Goal:** replace the Streamlit dashboard with a decoupled Next.js app that talks to
the FastAPI backend over JSON — and, in doing so, make the citation chain a
first-class UI affordance rather than an expander buried in a tab.

---

## Read this before you start

Three things will cost you an afternoon each if you discover them yourself.

### 1. `/insights/report` is synchronous and can take many minutes

It was measured **exceeding a 600-second HTTP timeout** against a cold server. As of
this writing the bounds are being fixed, but even bounded it is: investigation
(up to 120s) + verification (one LLM call per claim) + a cold embedding-model load
that is *not* bounded at all.

**Do not call it from the browser with a naive `fetch`.** This is the single most
important architectural decision in Phase D, and it is covered in Task 4.

The current Streamlit tab sets a 240-second client timeout, which has no headroom
over even the nominal budgets. Don't copy that.

### 2. There are two processing paths, and you want the async one

| Path | Shape | Use it for |
|---|---|---|
| `POST /process-reviews` | synchronous, returns everything | small batches, quick tests |
| `POST /jobs` | async, chunked, durable, resumable | **anything real** |

The job path persists results per chunk and survives a server restart. The sync path
does not, and has a server-side timeout of `300s + 0.3s/review` capped at 900s. Use
`/jobs` as the primary path; `/process-reviews` is fine for a dev-mode shortcut.

### 3. `id` is the join key for the whole product

Every claim carries `review_ids`. The frontend resolves those against the
`processed_data` it already holds to render citations. If ids don't line up, the
citation chain silently renders empty and the product's core promise is gone.

The backend injects an `id` if the uploaded CSV lacks one. **Keep `processed_data`
in client state after processing** — you need it to resolve citations later.

---

## Verified backend contract

Base URL default: `http://localhost:7860`. CORS is currently `allow_origins=["*"]`
(`app.py:90`) — fine for local dev, **must be tightened before any deployment**.

### Endpoints you will use

```
GET  /health                     component health, incl. whether real ABSA is available
POST /jobs                       submit → { job_id }
GET  /jobs/{job_id}              status, stage, chunk progress
GET  /jobs/{job_id}/results      merged results (409 if not finished)
POST /jobs/{job_id}/cancel       request cancellation
POST /insights/report            generate the grounded report (SLOW — see Task 4)
POST /process-reviews            synchronous processing (small batches only)
```

Endpoints that exist but you can ignore: `/cancel-task/*`, `/task-status/*`,
`/user-tasks/*`, `/task-stats`, `/cleanup-old-tasks` — legacy, backed by the job
store, superseded by `/jobs/*`.

### Request shapes

```ts
// POST /jobs  and  POST /process-reviews
type ProcessRequest = {
  data: Array<{
    id: number;
    reviews_title: string;
    review: string;
    date: string;      // ISO-ish; the backend parses it
    user_id: string;
  }>;
  options?: Record<string, unknown>;
  user_id?: string;
};

// POST /insights/report
type InsightReportRequest = {
  processed_data: ProcessedRow[];        // from the job results
  aspect_level_data?: AspectRow[] | null;
  max_tool_calls?: number | null;        // default 20
  max_seconds?: number | null;           // default 120
};
```

### Response shapes

The envelope is `{ status, data, message }` for processing endpoints.

```ts
type ProcessedRow = {
  id: number;
  review: string;
  reviews_title: string;
  date: string;
  user_id: string;
  translated_review: string;
  detected_language: string;             // "en" | "hi" | ...
  intent: string;                        // "complaint" | "praise" | ...
  intent_severity: string;
  intent_confidence: number;
  aspects: string[];                     // surface forms, e.g. "Battery life"
  aspects_canonical: string[];           // normalised, e.g. "battery life"
  aspect_sentiments: string[];           // parallel to aspects: Positive|Negative|Neutral
  overall_sentiment: string;
  extraction_method: string;             // "pyabsa" | "none" | "llm_escalated"
  degraded_reason: string | null;
};

// GET /jobs/{job_id}
type JobStatus = {
  id: string;
  user_id: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  stage: string | null;                  // "translating" | "extracting" | ...
  total_chunks: number;
  completed_chunks: number;
  error: string | null;
  created_at: number;
  updated_at: number;
};

// POST /insights/report → { status: "success", data: Report }
type Claim = {
  text: string;
  kind: "finding" | "complaint" | "strength" | "action";
  review_ids: Array<number | string>;    // THE citation chain
  reason: string;                        // "supported"
};

type Report = {
  findings: Claim[];
  complaints: Claim[];
  strengths: Claim[];
  actions: Claim[];
  stats: Record<string, unknown>;        // extraction_health, clusters, verification, investigation
  caveats: string[];                     // render these prominently — see Task 5
  is_empty?: boolean;
};
```

> **Contract in flux:** a fix wave is currently adding LLM-escalation stats to the
> report and changing how `extraction_health` counts escalated rows. Re-check
> `Report["stats"]` against `ABSA/src/insights/report.py` when you start Task 5.

### Progress stages

`/jobs/{job_id}.stage` cycles through: `validation`, `translating`,
`classifying_intent`, `extracting`, `analytics`, `combining_results`, `completed`.
Use `completed_chunks / total_chunks` for the bar and `stage` for the label.

---

## Architecture

**Next.js App Router, TypeScript, server components where they help.** The backend
does all the work; this app is presentation plus orchestration.

Suggested shape:

```
web/
├── app/
│   ├── page.tsx                  upload + recent runs
│   ├── runs/[jobId]/page.tsx     progress → results
│   ├── runs/[jobId]/report/      the report view
│   └── api/                      route handlers (see Task 4 — this is why)
├── lib/
│   ├── api.ts                    typed client for the backend
│   ├── types.ts                  the types above
│   └── citations.ts              id → review resolution
└── components/
    ├── UploadCsv.tsx
    ├── JobProgress.tsx
    ├── ReportView.tsx
    ├── ClaimCard.tsx             a claim + its expandable citations
    └── explore/                  the charts
```

**Why route handlers rather than calling FastAPI directly from the browser:** it
gives you one place to set timeouts, keeps the backend URL server-side, and lets you
turn the slow report call into something the browser can poll. It also means CORS
stops mattering for the real deployment.

---

## Tasks

Each is independently shippable. Do them in order — 1–3 give you a working app,
4–5 give you the thing that makes this project interesting.

### Task 1 — Scaffold and the typed client

Set up Next.js, then write `lib/types.ts` (paste the types above) and `lib/api.ts`.

The client should be thin and total: every function returns a discriminated result
rather than throwing, because the interesting cases here are all "the backend said
no in a specific way".

```ts
type ApiResult<T> =
  | { ok: true; data: T }
  | { ok: false; status: number; message: string };
```

**Handle these explicitly** — each corresponds to a real backend behaviour:
- `409` from `/jobs/{id}/results` — the job isn't finished. Not an error; poll again.
- `404` — unknown job id. The backend is careful never to return `200` with a null
  body here (an earlier bug did exactly that and crashed the old frontend), so treat
  a 404 as authoritative.
- `400` — bad input, e.g. an empty `data` array.

**Checkpoint:** `GET /health` renders in the UI, showing whether real ABSA is
available or the backend has degraded to keyword matching. That flag matters — it
tells you whether the numbers you're about to show mean anything.

### Task 2 — Upload and job submission

CSV upload, parsed client-side (papaparse is fine), mapped to `ProcessRequest`.

The backend requires `id`, `reviews_title`, `review`, `date`, `user_id` per row.
**Inject `id` yourself if the CSV lacks one** — the row index is fine, but be
consistent, because those ids are what citations resolve against later.

Validate before submitting and show the user what's wrong: missing required column,
empty file, non-UTF8. A 400 from the backend after a 20-second upload is a poor
substitute for a client-side check.

Sample files to test against: `streamlit-deployment/test_data_app_reviews.csv` (30
rows), `test_data_ecommerce.csv` (22), `test_data_restaurant.csv` (15).

**Checkpoint:** upload a CSV, get a `job_id` back, see it in the UI.

### Task 3 — Progress and results

Poll `GET /jobs/{job_id}` on an interval (2s is fine; the work is minutes long).
Render `stage` and `completed_chunks / total_chunks`. Offer cancel.

When `status === "completed"`, fetch `/jobs/{job_id}/results`. **Persist
`processed_data` in client state or session storage** — Task 5 needs it to resolve
citations, and re-fetching a large result set to render one quote is wasteful.

Handle `failed` (show `error`) and `cancelled` distinctly from `completed`.

**Checkpoint:** a full upload → progress → results round trip, with the aspect table
rendering. At this point you have parity with the useful half of the Streamlit app.

### Task 4 — The report call, done properly

**This is the task that needs thought.** `/insights/report` can run for minutes and
returns in one shot. Browsers, proxies, and users all give up before it does.

Three options, in the order I'd try them:

**(a) Route handler with a long timeout + optimistic UI.** A Next.js route handler
calls the backend with a generous timeout (Node's default fetch has none; set one
explicitly, e.g. 15 minutes) while the client shows an indeterminate "investigating"
state. Simplest thing that works. Weakness: no progress, and a page refresh loses it.

**(b) Kick off server-side, poll a local job record.** The route handler starts the
work and returns an id immediately; the client polls your own endpoint. Requires
somewhere to keep state (a module-level Map is enough for single-instance dev).
Gives you a refreshable page and a progress affordance.

**(c) Ask the backend to put reports on the job machinery.** The backend already has
a durable job store with chunked progress, resumption, and cancellation — used by
`/jobs`. Moving `/insights/report` onto it is tracked as a known follow-up
(`Task 7(b)` in the Phase C ledger) and would give you a real progress bar for free.

**My recommendation: build (a) now, and raise (c) as a backend change.** (b) is real
work that (c) makes redundant. If you do (a), set the client's expectations honestly
in the UI — "this can take several minutes" beats a spinner that looks hung.

Whatever you choose, **do not** put a short timeout on this call and call it done.

**Checkpoint:** the report request completes without the browser or a proxy killing
it, on a real ~20-review dataset.

### Task 5 — The report view

This is the payoff. Everything else in this project exists to make this screen
truthful.

**Four sections** — findings, complaints, strengths, actions — rendered from the
`Report` object.

**Every claim expands to its citations.** `ClaimCard` takes a `Claim`, resolves each
`review_ids` entry against the `processed_data` you kept in Task 3, and shows the
actual review text. This is not a nice-to-have: a finding you can't trace is
indistinguishable from the templated prose this system replaced.

Show `translated_review` when it differs from `review`, and label the original
language — a Hindi review's claim should be checkable by someone reading English.

**Render `caveats` prominently, not in a footer.** They carry things like "N of M
reviews came from a degraded extraction path" and "this analysis was truncated". A
caveat the user doesn't see defeats its purpose.

**The empty report is a first-class state, and you must not decorate it.** When
`is_empty` is true, or all four sections are empty:
- Say plainly that nothing could be substantiated.
- Show `stats` and `caveats`, which are still real.
- **Never** substitute encouraging prose, a "no significant issues found" message,
  or a placeholder. A class doing exactly that was deleted from this codebase on
  purpose. If you add it back in the UI layer, the whole phase was pointless.

Surface the dropped-claim count from `stats` — "11 claims produced, 4 dropped in
verification" is information the user is entitled to.

**Checkpoint:** click a finding, see the reviews it rests on. Then force an empty
report (a tiny CSV, or a backend with `OPENROUTER_API_KEY` unset) and confirm the UI
says so honestly.

### Task 6 — Explore: the charts

Rebuild what's worth keeping from the Streamlit dashboard against the JSON API.
Recharts or visx; the data is already aggregated server-side.

Worth porting: KPI tiles, sentiment distribution, the aspect ranking tables (areas of
improvement vs strength anchors — both come back in the job results), and the
aspect-sentiment heatmap.

Lower value: the Sankey and the network graph. They demo well and are rarely acted
on. Your call.

**Checkpoint:** parity on the things a user would actually act on.

### Task 7 — Cut over

Retire the Streamlit app only once Tasks 1–6 are done and you've used the new one on
a real dataset. Then update `README.md`'s architecture section and quick-start.

Keep `streamlit-deployment/` in git history; delete the directory rather than letting
two frontends drift.

---

## Gotchas

Things that have already bitten this project, listed so they don't bite you.

**Non-finite floats.** `NaN` and `Infinity` are valid Python floats and **invalid
JSON** — `JSON.parse` rejects them outright. The backend now converts them to `null`
in both `app.py` and `jobs/runner.py`, but this bug shipped once and was only caught
because Starlette's `JSONResponse` happens to reject `NaN` on the sync path. If you
ever see `Unexpected token N in JSON`, this is why. A defensive check in your client
is cheap.

**Two repos.** `ABSA/` is a nested git repo with its own history, tracked in the
parent as a gitlink. If you change backend code, you commit in `ABSA/` **and** bump
the gitlink in the parent, or the parent tree points at the wrong backend commit.
This has already caused one stale-pointer incident.

**Degraded extraction is invisible unless you show it.** Every row carries
`extraction_method` and `degraded_reason`. A row with `extraction_method: "none"`
contributed no aspects; one with `"llm_escalated"` was repaired by an LLM, not the
aspect model. Consider surfacing this in the explore view — the backend went to
considerable trouble to make degradation visible, and dropping that in the UI undoes
the work.

**The `aspects` / `aspect_sentiments` arrays are positional.** `aspects[i]` pairs
with `aspect_sentiments[i]`. Don't sort one without the other.

**`aspects` vs `aspects_canonical`.** Show the surface form to users, group and count
on the canonical form. Ranking on surface forms splits one concept across five rows —
this was a real measured bug in the analytics.

**No auth exists.** Deliberately out of scope. The backend has no authentication, no
rate limiting (the Redis-backed limiter was removed in Phase A), and CORS is wide
open. Fine for local use; do not expose it to the internet as-is.

---

## What is not ready

Be aware of these before you build on them.

**Phase C has open Critical findings** as of 2026-08-12, currently being fixed:
- LLM escalation is implemented but not wired into the report path, so
  `extraction_method` will never be `"llm_escalated"` until that lands.
- The agent could emit duplicate claims; report rendering should tolerate near-
  identical claims gracefully regardless.
- The report endpoint's time bounds did not actually bound. This is exactly why
  Task 4 matters.

**The groundedness baseline is unestablished** — blocked on an API quota. Citation
validity is measured at 1.000 (every claim cites a review that exists), but whether
the cited text *supports* the claim has not yet been scored.

**Nothing is merged to main.** Phases A–C live on `benchmark/absa-baseline` and
`fix/silent-fallback`.

---

## Definition of done

- Upload → job → progress → results → report works end to end on a real CSV.
- Every claim in the report expands to the review text behind it.
- An empty report renders as an honest empty state, with no substituted prose.
- Caveats are visible without scrolling past the findings.
- No Python renders UI.
- `README.md` describes the Next.js app, and Streamlit is deleted rather than left
  to rot beside it.
