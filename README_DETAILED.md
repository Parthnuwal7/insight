# Insights — Aspect-Based Sentiment Analysis Platform

**A technical reference for the system as it is actually built.**

This document describes the current architecture, module-by-module responsibilities, tech stack, and operational context of the `insights` repository. It is written for someone who needs to *work on* the system — extend it, deploy it, or debug it — rather than evaluate it. For the marketing-oriented overview, see [`README.md`](README.md).

> **Accuracy note:** everything below was derived by reading the source in this working tree (`main` @ `841c1fe`). Where the code and the older docs disagree, this document follows the code, and the disagreement is called out in [Known Drift & Gotchas](#12-known-drift--gotchas).

---

## Table of Contents

1. [What the system does](#1-what-the-system-does)
2. [Context: why it is split in two](#2-context-why-it-is-split-in-two)
3. [Repository layout](#3-repository-layout)
4. [High-level architecture](#4-high-level-architecture)
5. [Request lifecycle](#5-request-lifecycle)
6. [The NLP pipeline, stage by stage](#6-the-nlp-pipeline-stage-by-stage)
7. [Backend module reference (`ABSA/`)](#7-backend-module-reference-absa)
8. [Frontend module reference (`streamlit-deployment/`)](#8-frontend-module-reference-streamlit-deployment)
9. [Data contracts](#9-data-contracts)
10. [Tech stack](#10-tech-stack)
11. [Configuration & environment](#11-configuration--environment)
12. [Known drift & gotchas](#12-known-drift--gotchas)
13. [Running it locally](#13-running-it-locally)
14. [Extension points](#14-extension-points)

---

## 1. What the system does

The platform ingests a CSV of customer reviews and returns **aspect-level** sentiment rather than a single score per review. One review such as *"Battery is great but delivery took two weeks"* produces two independent verdicts — `Battery → Positive`, `Delivery → Negative` — plus an intent label, a confidence score, and a place in several aggregate analyses.

Concretely, one processing run yields:

| Output | Shape | Produced by |
|---|---|---|
| Per-review records | DataFrame, one row per review | `DataProcessor.process_uploaded_data` |
| Aspect-level records | DataFrame, one row per *(review, aspect)* pair | same, reshape step |
| Mixed-sentiment reviews | Reviews containing both Positive and Negative aspects | same, reshape step |
| Areas of improvement | Ranked table by `priority_score` | `AspectAnalytics.calculate_aspect_scores` |
| Strength anchors | Ranked table by `strength_score` | same |
| Aspect co-occurrence graph | NetworkX graph, nodes = aspects, edges = co-mentions | `AspectAnalytics.calculate_aspect_cooccurrence` |
| Sentiment spike alerts | Week-over-week negativity jumps per aspect | `AspectAnalytics.detect_sentiment_spikes` |
| LLM narrative insights | Markdown from an LLM | Frontend, `generate_llm_insights` |

The distinguishing capability is the **aspect-level reshape**: because the pipeline emits one row per aspect mention, the dashboard can do multi-aspect relationship analysis (co-occurrence, mixed sentiment, per-aspect drill-down) that a review-level dataset cannot support.

---

## 2. Context: why it is split in two

The system is deliberately split across **two deployment targets with two separate hosting providers**, and nearly every architectural decision follows from that split:

- **PyABSA requires the full PyTorch + Transformers stack** (~1.5 GB of dependencies plus a downloaded model checkpoint). That does not fit comfortably in Streamlit Cloud's free tier.
- **Streamlit Cloud is excellent at hosting the UI** and free for public apps.

So the ML lives in a Docker container on **HuggingFace Spaces** (`parthnuwal7/ABSA`, port 7860), and the dashboard lives on **Streamlit Cloud**, talking to the backend over REST. The frontend's `requirements.txt` deliberately contains **no ML libraries at all** — it is ~80 MB instead of ~1.5 GB, and it comments on this fact in the file itself.

A consequence worth internalising: **the frontend never computes sentiment.** It is a rendering and orchestration layer over a remote API response. All the analysis logic that matters lives in `ABSA/src/utils/data_processor.py`.

A second consequence: `ABSA/` is a **nested, independent git repository** tracked in the parent repo as a gitlink (mode `160000`), whose `origin` is the HuggingFace Space itself. Pushing the backend deploys it. The parent repo does not contain the backend's file history.

---

## 3. Repository layout

```
insights/
├── ABSA/                              ← nested git repo → HF Space (deploys on push)
│   ├── app.py                         FastAPI application (the deployed entrypoint)
│   ├── api_server.py                  byte-identical copy of app.py (see §12)
│   ├── admin_dashboard.py             standalone Streamlit admin viewer
│   ├── validate_setup.py              preflight checker for env/Mongo/Redis/API
│   ├── Dockerfile                     python:3.10-slim, installs git+git-lfs, CMD python app.py
│   ├── requirements.txt               full ML stack (torch, pyabsa, transformers…)
│   ├── requirements-backend.txt       same list minus python-dotenv
│   ├── .streamlit/                    config.toml + secrets.toml.template
│   ├── PYABSA_FIX.md                  deployment notes on getting PyABSA to load
│   └── src/
│       ├── utils/
│       │   ├── data_processor.py      ★ the entire NLP pipeline (1170 lines)
│       │   ├── task_manager.py        in-process task registry + cancellation
│       │   ├── task_queue.py          Redis-backed async job queue + worker thread
│       │   ├── redis_service.py       rate limiting, IP gate, queue primitives
│       │   ├── mongodb_service.py     append-only telemetry event store
│       │   ├── ip_location_service.py IPinfo lookup, Redis-gated
│       │   ├── rate_limit_middleware.py  ASGI middleware
│       │   ├── admin_endpoints.py     /admin/metrics/* router
│       │   └── data_management.py     ⚠ not imported anywhere (see §12)
│       ├── components/
│       │   └── visualizations.py      ⚠ not imported anywhere (see §12)
│       └── streamlit_app.py           ⚠ Streamlit scaffolding leftover
│
├── streamlit-deployment/              ← deployed to Streamlit Cloud
│   ├── app_a.py                       ★ the dashboard (2869 lines, 3 pages)
│   ├── dashboard_components.py        chart + KPI builders used by the Analytics page
│   ├── diagnostic_component.py        aspect diagnostics panel (call site commented out)
│   ├── frontend_helpers.py            device id, telemetry, async job client
│   ├── requirements.txt               UI-only dependencies, no ML
│   ├── AI_INSIGHTS_SETUP.md           OpenRouter setup guide
│   └── test_data_*.csv                3 bundled sample datasets
│
├── requirements.txt                   full local/dev stack (superset of both)
├── insights_arc.png                   architecture diagram image
├── logs.md                            captured PyABSA checkpoint-loading log
└── README.md                          overview-oriented readme
```

**Three requirements files, three purposes.** Root = local development with everything. `ABSA/requirements.txt` = the container image. `streamlit-deployment/requirements.txt` = the Streamlit Cloud slug. They are intentionally not unified.

---

## 4. High-level architecture

```mermaid
flowchart TB
    subgraph SC["Streamlit Cloud"]
        UI["app_a.py<br/>Home · Analytics · Admin"]
        DC["dashboard_components.py<br/>charts & KPIs"]
        FH["frontend_helpers.py<br/>telemetry & job client"]
        UI --- DC
        UI --- FH
    end

    subgraph HF["HuggingFace Spaces — Docker, port 7860"]
        MW["RateLimitMiddleware<br/>+ CORS"]
        API["FastAPI app.py"]
        TM["TaskManager<br/>in-process, cancellable"]
        TQ["TaskQueue<br/>worker thread"]
        DP["DataProcessor<br/>the NLP pipeline"]
        MW --> API
        API --> TM
        API --> TQ
        API --> DP
    end

    subgraph EXT["External services"]
        RD[("Redis<br/>limits · gate · queue")]
        MG[("MongoDB<br/>events")]
        HFAPI["HF Inference API<br/>IndicTrans2"]
        IPI["IPinfo"]
        ORT["OpenRouter<br/>Nemotron"]
    end

    UI -- "POST /process-reviews" --> MW
    FH -- "/log-session · /log-event · /submit-job" --> MW
    UI -- "GET /admin/metrics/*  (Bearer)" --> MW
    UI -- "insight generation" --> ORT

    API --> RD
    API --> MG
    TQ --> RD
    DP --> HFAPI
    API --> IPI
```

### Layer responsibilities

| Layer | Where | Owns |
|---|---|---|
| **Presentation** | `app_a.py`, `dashboard_components.py` | Upload, filtering, ~15 chart types, CSV export, admin views |
| **Client/telemetry** | `frontend_helpers.py` | Device identity, event logging, async job submission & polling |
| **Edge** | `rate_limit_middleware.py`, CORS | Per-identity throttling, rate-limit headers, admin/health bypass |
| **API** | `app.py` | Request validation (Pydantic), timeout budgeting, thread offload, response serialization |
| **Orchestration** | `task_manager.py`, `task_queue.py` | Task IDs, progress %, cooperative cancellation, Redis job lifecycle |
| **Pipeline** | `data_processor.py` | Validation, translation, ABSA, intent, analytics, summarization |
| **Infrastructure** | `redis_service.py`, `mongodb_service.py`, `ip_location_service.py` | Singleton connections, fail-open behaviour, append-only events |
| **Analytics API** | `admin_endpoints.py` | Token-gated aggregation over the event store |

---

## 5. Request lifecycle

The **synchronous path is the one the dashboard actually uses.** Walking it end to end:

1. **Upload.** The user picks a CSV or one of three bundled samples. The frontend backfills missing optional columns (`id`, `reviews_title`, `date`, `user_id`) so only `review` is truly required from the user, then converts rows to `ReviewData` records.

2. **Dispatch.** `call_ml_backend` POSTs to `{HF_SPACES_API_URL}/process-reviews` with a **900-second client timeout** — deliberately matching the server's absolute ceiling.

3. **Middleware.** `RateLimitMiddleware` derives an identity with priority `X-User-Id` → `X-Device-Id` → client IP, and does an `INCR`+`EXPIRE` window in Redis. `/admin/*`, `/health`, `/`, `/docs`, `/openapi.json` bypass it. Every response gains `X-RateLimit-{Limit,Remaining,Reset}` headers.

4. **Endpoint-level limit.** `/process-reviews` applies a *second*, stricter Redis check: **10 requests per minute per `user_id`**, because this is the expensive endpoint. A rejection here logs a `RATE_LIMIT_HIT` event to MongoDB and returns HTTP 429.

5. **Task creation.** `TaskManager.create_task` mints a UUID, registers status/progress/stage, and creates a `threading.Event` cancellation flag.

6. **Timeout budgeting.** `calculate_timeout(n) = min(300 + 0.3n, 900)` seconds — a floor of 5 minutes, 0.3 s of headroom per review, hard-capped at 15 minutes.

7. **Execution.** The pipeline runs on a `ThreadPoolExecutor` (`MAX_WORKERS`, default 2) wrapped in `asyncio.wait_for`, so the event loop stays responsive and the timeout is enforceable. On `TimeoutError` the task is marked failed, cleaned up, and a `status: "timeout"` response is returned — not an exception.

8. **Cancellation.** The pipeline checks `task_manager.is_cancelled(task_id)` at every stage boundary and every ABSA batch of 5. On cancellation it deletes intermediate structures, calls `gc.collect()`, and returns `{'status': 'cancelled'}`. This is **cooperative** cancellation — nothing is killed mid-inference; the request is simply abandoned at the next checkpoint. The frontend triggers it via `POST /cancel-task/{task_id}`.

9. **Serialization.** `serialize_for_api` converts DataFrames to `records` dicts and the NetworkX graph via `nx.node_link_data`, then attaches `task_id` and `timeout_used`.

10. **Rendering.** The frontend parses the payload, normalizes column names, and stashes `processed_data`, `aspect_level_data`, `mixed_sentiment_reviews`, `analysis_summary`, and `aspect_network` in `st.session_state`. Everything the Analytics page draws comes from that session state.

### The asynchronous path

A parallel design exists — `POST /submit-job` → Redis list `absa_tasks` → background worker thread → `GET /job-status/{job_id}` → result under `job:{id}:result` (TTL 1 hour) — with statuses `PENDING`/`RUNNING`/`DONE`/`FAILED`, client helpers in `frontend_helpers.py`, and telemetry at each transition. **It is wired but not functional**; see [§12](#12-known-drift--gotchas). Treat it as scaffolding for the next scaling step, not as a working code path.

---

## 6. The NLP pipeline, stage by stage

All of this is `DataProcessor.process_uploaded_data`. Note that **intent classification runs before aspect extraction**, not after.

| # | Stage | Implementation | Progress | Notes |
|---|---|---|---|---|
| 1 | **Validate** | `DataValidator.validate_csv` | 5% | Requires `id`, `reviews_title`, `review`, `date`, `user_id`. Rejects empty reviews and unparseable dates. Returns `{'error': [...]}`, no exception. |
| 2 | **Clean** | `DataValidator.clean_data` | — | Coerces dates, strips text, drops null reviews, **de-duplicates on review text**, resets index. Row counts can shrink here. |
| 3 | **Detect + translate** | `TranslationService` | 10–40% | `langdetect` per review; only `hi` is sent to the HF Inference API (`ai4bharat/indictrans2-en-indic-1.3B`). 10-second timeout, **silent fallback to the original text** on any failure. Batches of 10, cancellable between batches. |
| 4 | **Classify intent** | `IntentClassifier` | 40% | Keyword rules over 6 intents (`complaint`, `praise`, `question`, `suggestion`, `comparison`, `neutral`). Complaints get `high`/`medium`/`low` severity; praise gets a positivity tier. Confidence is a normalized keyword hit count. |
| 5 | **Extract aspects + sentiment** | `ABSAProcessor` | 50–90% | PyABSA `AspectTermExtraction.AspectExtractor('multilingual')` (FAST-LCF-ATEPC). Batches of 5 for responsive cancellation. Per-review try/except falls back to rules. |
| 5b | *(fallback)* | `_extract_with_fallback` | — | If PyABSA fails to load *or* fails on a review: 14 keyword aspect buckets (OTP/Verification, Login/Account, App Performance, Payment, Quality, Price, Service, Delivery, Design, Performance, Usability, Features, Size, Battery — with Hindi variants) + rule-based polarity, fixed confidence `0.7`. |
| 6 | **Combine** | inline | 90% | Adds `translated_review`, `detected_language`, `intent*`, `aspects`, `aspect_sentiments`, and a majority-vote `overall_sentiment`. |
| 7 | **Analytics** | `AspectAnalytics` | 95–100% | Priority/strength scores, co-occurrence graph, spike detection. Formulae below. |
| 8 | **Reshape** | inline | — | Explodes to one row per *(review, aspect)*; flags reviews holding both Positive and Negative aspects as mixed. |

### Scoring formulae

```
priority_score  = negativity_ratio × frequency × (1 + severity_weight)
strength_score  = positivity_ratio × frequency × (1 + 2 × positivity_ratio)
severity_weight = mean over complaints of {high:3, medium:2, low:1, standard:1}
```

An aspect enters **Areas of Improvement** only above 10% negativity, and **Strength Anchors** only above 30% positivity. Both multiply ratio by raw frequency, so a widely-mentioned aspect outranks a rarely-mentioned one at the same ratio — the ranking is intentionally volume-weighted for business triage.

### Graph and alerting rules

- **Co-occurrence graph** — nodes carry `frequency`, `sentiment_score`, `color`, `positive_pct`, `negative_pct`. Edges are added only at **weight ≥ 2**, so a single coincidental co-mention never draws a line.
- **Spike detection** — needs ≥ 4 mentions and ≥ 14 days of history. Fires when the 7-day mean negative count exceeds the prior 7-day mean by **>50% *and* by ≥ 2 absolute complaints** — the absolute floor exists to suppress noise like 1 → 2. Severity is `high` above 100% growth, else `medium`.

### The LLM layer (frontend-side)

Narrative insights are generated in the **frontend**, not the pipeline, in `generate_llm_insights`:

- Provider **OpenRouter**, model **`nvidia/nemotron-3-nano-30b-a3b:free`**, `temperature 0.7`, `max_tokens 800`, 30-second timeout.
- `prepare_analysis_context` compresses aspect-level data into a compact text brief: totals, top-8 aspects with sentiment percentages, top-5 co-mention pairs, mixed-sentiment count. **Raw review text is not sent** — only aggregates.
- Without `OPENROUTER_API_KEY` the function returns `""` and the UI falls back to `generate_rag_insights`, a deterministic pattern-based generator capped at 6 insights. The LLM is an enhancement, never a dependency.

---

## 7. Backend module reference (`ABSA/`)

### `app.py` — FastAPI application

Entry point (`CMD ["python", "app.py"]` → uvicorn on `0.0.0.0:7860`). Mounts CORS (`allow_origins=["*"]`), `RateLimitMiddleware(max_requests=100, window_seconds=60)`, and the admin router. The `DataProcessor` is created **lazily** by `get_processor()` so container startup isn't blocked by loading the PyABSA checkpoint.

**Endpoints (as implemented):**

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Liveness stub |
| GET | `/health` | Reports translator, ABSA, MongoDB, Redis availability |
| POST | `/process-reviews` | **Synchronous pipeline** — the path the dashboard uses |
| POST | `/submit-job` | Enqueue async job → `job_id` |
| GET | `/job-status/{job_id}` | Queue status, plus `result` when `DONE` |
| POST | `/cancel-task/{task_id}` | Request cooperative cancellation |
| GET | `/task-status/{task_id}` | Progress, stage, message |
| POST | `/cancel-user-tasks/{user_id}` | Cancel all of a user's active tasks |
| GET | `/user-tasks/{user_id}` | List a user's tasks |
| GET | `/task-stats` | Aggregate counts by status and user |
| POST | `/cleanup-old-tasks` | Evict finished tasks older than N hours |
| POST | `/log-session` | Session metadata + IP geolocation (Redis-gated) |
| POST | `/log-event` | Telemetry event → MongoDB |
| GET | `/admin/metrics/summary` | Events by type, unique devices/users, time range |
| GET | `/admin/metrics/events` | Daily event timeline |
| GET | `/admin/metrics/funnel` | View → Request → Queued → Completed conversions |
| GET | `/admin/metrics/rate-limits` | Total hits, top 10 devices, daily timeline |

### `src/utils/data_processor.py` — the pipeline

Five classes: `DataValidator`, `TranslationService`, `ABSAProcessor`, `IntentClassifier`, `AspectAnalytics`, coordinated by `DataProcessor`. This is where essentially all analytical behaviour lives; see [§6](#6-the-nlp-pipeline-stage-by-stage).

### `src/utils/task_manager.py` — in-process task registry

Thread-safe (`threading.Lock`) dict of tasks plus a `threading.Event` per task as the cancellation flag. Tracks `status`, `stage`, `progress`, timestamps, and messages; supports per-user cancellation and age-based cleanup. **Process-local** — it does not survive a restart and does not span replicas.

### `src/utils/task_queue.py` — Redis job queue

`RPUSH`/`BLPOP` on the list `absa_tasks`, a daemon worker thread, statuses in `job:{id}:status` and results in `job:{id}:result` (both TTL 3600 s), and `ANALYSIS_REQUEST` / `TASK_QUEUED` / `TASK_COMPLETED` telemetry at each transition. See [§12](#12-known-drift--gotchas) before relying on it.

### `src/utils/redis_service.py` — Redis primitives

Singleton client with 5-second connect/socket timeouts. Three responsibilities:

- **Rate limiting** — `INCR` + `EXPIRE` sliding window under `ratelimit:{identifier}`.
- **IP-logging gate** — `SET key 1 NX EX 300` under `iplog:{device_id}`, so IPinfo is called at most once per device per 5 minutes.
- **Queue primitives** — enqueue/dequeue/status/result.

**Fail-open by design:** if Redis is unreachable, `check_rate_limit` returns `(True, 0)` and logs a warning. Availability is preferred over enforcement — worth knowing before treating the limiter as a security control.

### `src/utils/mongodb_service.py` — telemetry store

Singleton client over an **append-only** `events` collection. Six whitelisted event types — `SESSION_METADATA`, `DASHBOARD_VIEW`, `ANALYSIS_REQUEST`, `TASK_QUEUED`, `TASK_COMPLETED`, `RATE_LIMIT_HIT` — anything else is rejected with a warning. Six indexes are created on connect (`created_at`, `device_id`, `user_id`, `event_type`, and two compounds) to keep the admin aggregations cheap. Absent `MONGO_URI`, every call becomes a no-op returning `False`; telemetry is never load-bearing.

### `src/utils/ip_location_service.py` — geolocation

Behind the Redis gate, calls `ipinfo.io` (5-second timeout) and stores city/region/country/coords/org/timezone inside the `SESSION_METADATA` event. Disabled entirely without `IPINFO_TOKEN`.

### `src/utils/rate_limit_middleware.py` — ASGI middleware

Identity resolution (`user_id` → `device_id` → IP), Redis check, `RATE_LIMIT_HIT` logging on rejection, HTTP 429 with a `retry_after`, and rate-limit headers on success.

### `src/utils/admin_endpoints.py` — analytics router

`APIRouter(prefix="/admin")` guarded by a `Bearer <ADMIN_TOKEN>` dependency, exposing four MongoDB aggregations. Returns 500 (not 401) when `ADMIN_TOKEN` is unset — a deliberate distinction between *misconfigured* and *unauthorized*.

### Supporting scripts

- **`admin_dashboard.py`** — a standalone Streamlit admin viewer (`streamlit run admin_dashboard.py`), duplicating the frontend's Admin page for operators who don't want to load the main dashboard.
- **`validate_setup.py`** — preflight checker: env vars present, MongoDB reachable and writable, Redis reachable, service singletons constructible, `/health` responding. Run this first when something is misbehaving.

---

## 8. Frontend module reference (`streamlit-deployment/`)

### `app_a.py` — the dashboard

Three pages via `streamlit-option-menu`:

**🏠 Home** — data source selection (upload vs. three bundled samples), preview, `🚀 Process Reviews with AI`, a debug expander showing the exact API request, then quick KPIs and aspect-level statistics after completion. Only `review` is genuinely required in an uploaded CSV; the rest is backfilled.

**📈 Analytics** — enhanced KPI cards over three tabs, each with **independent filter state** (deliberate: filters in one tab don't disturb another):

- **Overview** — sentiment pie, intent×aspect and sentiment×aspect heatmaps, review timeline, priority leaderboard, co-occurrence heatmap, confidence funnel, plus filtered CSV export.
- **Multi-Aspect Analysis** — operates on the aspect-level DataFrame: relationship patterns, mixed-sentiment inspection, and the LLM insight panel. Degrades with a clear warning if the payload predates aspect-level output.
- **Deep Dive** — pick an aspect, see every review mentioning it with per-mention sentiment, mention/positive/negative/confidence metrics, and CSV export.

**🔒 Admin** — token entry stored in session state, then `Bearer`-authenticated calls to `/admin/metrics/*` rendering summary metrics, event timeline, funnel conversions, and rate-limit statistics with a 1–30 day window.

Also holds `SessionManager` (lightweight, `st.session_state`-only — **no server-side persistence**; refreshing the browser loses history), `normalize_backend_columns` for schema tolerance across backend versions, and standalone chart builders including the WordCloud and the Plotly aspect-network renderer.

### `dashboard_components.py`

Twelve pure builders — `extract_aspects_list`, `get_all_unique_aspects`, `get_top_aspects_by_frequency`, `calculate_kpi_metrics`, `create_enhanced_kpi_cards`, and the seven Plotly figures listed above. `extract_aspects_list` is the load-bearing one: aspects arrive variously as lists, stringified lists, or scalars depending on serialization path, and it normalizes all of them.

### `frontend_helpers.py`

Device identity (`uuid4` in session state), `log_session_metadata` / `log_event`, `submit_analysis_job` / `get_job_status` / `poll_job_until_complete`, and `initialize_telemetry` (called once at import). Every function is wrapped so telemetry failure never breaks the UI — `app_a.py` even defines no-op stubs if this module fails to import.

### `diagnostic_component.py`

`show_aspect_diagnostics` — an aspect-extraction debugging panel. **Currently inactive**: its call site at `app_a.py:1949` is commented out.

### Sample data

`test_data_ecommerce.csv` (22 reviews), `test_data_restaurant.csv` (15), `test_data_app_reviews.csv` (30). Each mixes positive/negative/neutral, includes Hindi reviews, and contains deliberate mixed-sentiment cases so the multi-aspect features have something to show without an upload.

---

## 9. Data contracts

### Input CSV

| Column | Required by backend | Required by UI | Notes |
|---|---|---|---|
| `id` | ✅ | auto-filled | Sequential if absent |
| `reviews_title` | ✅ | auto-filled | `"Review {id}"` if absent |
| `review` | ✅ | ✅ | The only genuinely mandatory column |
| `date` | ✅ | auto-filled | Today if absent; must parse |
| `user_id` | ✅ | auto-filled | `"user_{id}"` if absent |

### `POST /process-reviews`

```jsonc
// request
{
  "data": [
    { "id": 1, "reviews_title": "Great", "review": "Battery life is amazing!",
      "date": "2024-01-15", "user_id": "u1" }
  ],
  "options": { "include_translation": true, "include_aspects": true },
  "user_id": "demo_user"
}

// response
{
  "status": "success",           // | "cancelled" | "timeout"
  "data": {
    "processed_data": [ /* one row per review */ ],
    "aspect_level_data": [ /* one row per (review, aspect) */ ],
    "mixed_sentiment_reviews": [ /* both-polarity reviews */ ],
    "absa_details": [ /* raw extractor output */ ],
    "areas_of_improvement": [ /* ranked by priority_score */ ],
    "strength_anchors": [ /* ranked by strength_score */ ],
    "aspect_network": { "nodes": [], "links": [] },   // nx.node_link_data
    "sentiment_alerts": [ /* spike alerts */ ],
    "summary": {
      "total_reviews": 0, "total_aspects": 0,
      "mixed_sentiment_count": 0, "mixed_sentiment_pct": 0.0,
      "languages_detected": [], "intents_distribution": {},
      "sentiment_distribution": {}, "top_problem_areas": 0,
      "top_strength_anchors": 0, "active_alerts": 0
    },
    "task_id": "uuid", "timeout_used": 300.0
  }
}
```

### `aspect_level_data` row

```jsonc
{
  "review_id": 1, "review": "…", "aspect": "Battery",
  "aspect_sentiment": "Positive",   // Positive | Negative | Neutral
  "overall_sentiment": "Positive",  // majority vote across the review's aspects
  "intent": "praise",               // complaint | praise | question | suggestion | comparison | neutral
  "intent_severity": "high_positive",
  "date": "2024-01-15", "language": "en"
}
```

### Telemetry event

```jsonc
{
  "user_id": "…|null", "device_id": "uuid",
  "event_type": "DASHBOARD_VIEW",     // one of six whitelisted types
  "created_at": "2026-08-08T00:00:00Z",
  "metadata": { }                      // e.g. IP/geo for SESSION_METADATA
}
```

---

## 10. Tech stack

### Backend (`ABSA/requirements.txt`)

| Concern | Choice | Version |
|---|---|---|
| Web framework | FastAPI + Uvicorn | `>=0.104` / `>=0.24` |
| Validation | Pydantic v2 (`model_dump`) | `>=2.0` |
| ABSA model | **PyABSA** ATEPC multilingual (FAST-LCF-ATEPC) | `>=2.4,<3.0` |
| DL runtime | PyTorch | `>=2.0,<2.2` |
| Transformers | HuggingFace Transformers | `>=4.30,<4.37` |
| Tokenization | sentencepiece, sacremoses | — |
| Vector ops | faiss-cpu | `>=1.7.4` |
| Language ID | langdetect | `>=1.0.9` |
| Data | pandas / numpy / scikit-learn | `<2.1` / `<1.26` / `<1.4` |
| Graphs | NetworkX | `>=3.0` |
| Cache/queue | redis-py | `>=5.0` |
| Telemetry | pymongo | `>=4.6` |
| Runtime | Python 3.10-slim, Docker | — |

Version ceilings are load-bearing — PyABSA 2.4 is sensitive to the Transformers and PyTorch versions it sits on. Loosen them only with a deliberate test.

### Frontend (`streamlit-deployment/requirements.txt`)

Streamlit, Plotly, pandas, numpy, requests, `streamlit-option-menu`, `streamlit-aggrid`, NetworkX, WordCloud, Matplotlib, `python-dotenv`, Pillow, openpyxl. **No ML libraries** — that omission is the whole point of the split.

### External services

| Service | Used for | Required? |
|---|---|---|
| HuggingFace Spaces | Backend hosting (Docker, port 7860) | Yes, for the deployed system |
| Streamlit Cloud | Frontend hosting | Yes, for the deployed system |
| HF Inference API | IndicTrans2 translation | No — silently falls back to source text |
| MongoDB Atlas | Telemetry event store | No — telemetry becomes no-op |
| Redis | Rate limits, IP gate, queue | No — limiter fails open |
| IPinfo | Geolocation | No — geo fields omitted |
| OpenRouter | LLM narrative insights | No — pattern-based fallback |

**Every external service degrades rather than fails.** Only the two hosting platforms are hard requirements. This is a consistent, deliberate property of the design, and the single most useful thing to know when debugging: a missing key produces quieter output, not an error.

---

## 11. Configuration & environment

### Backend (`ABSA/.env`)

```env
MONGO_URI=mongodb+srv://...      # telemetry; omit to disable
REDIS_HOST=localhost             # rate limiting / queue / IP gate
REDIS_PORT=6379
REDIS_PASSWORD=
ADMIN_TOKEN=                     # Bearer token for /admin/*; openssl rand -hex 32
IPINFO_TOKEN=                    # geolocation; free tier 50k/month
HF_TOKEN=                        # HF Inference API (translation)
MAX_WORKERS=2                    # ThreadPoolExecutor size
```

### Frontend (`streamlit-deployment/.env`)

```env
OPENROUTER_API_KEY=sk-or-v1-...  # LLM insights; omit for pattern-based fallback
HF_SPACES_API_URL=https://parthnuwal7-absa.hf.space   # currently hardcoded in app_a.py:55
```

`TranslationService` and `ABSAProcessor` read `HF_TOKEN` from `st.secrets` first and fall back to the environment — a holdover from when the pipeline ran inside Streamlit. In the FastAPI container the `st.secrets` lookup simply fails and falls through, which is why `.streamlit/secrets.toml.template` still exists in the backend.

### Deployment

| Target | Mechanism |
|---|---|
| Backend | `git push` inside `ABSA/` → HF Space rebuilds from the `Dockerfile`. Secrets set in Space settings. |
| Frontend | Streamlit Cloud tracks the repo; main file `streamlit-deployment/app_a.py`. Secrets in app settings. |

The Dockerfile installs `git` and `git-lfs` explicitly — PyABSA needs them to fetch checkpoints at runtime. `PYABSA_FIX.md` and `logs.md` document that episode; `logs.md` is a captured checkpoint-resolution log, useful when the model silently falls back to rules.

---

## 12. Known drift & gotchas

Real observations from the current tree. Nothing here is fatal to the working system, but each will cost time if discovered the hard way.

1. **The async queue path is broken.** `TaskQueue._process_task` calls `self.data_processor.process_data(csv_data)`, but `DataProcessor` exposes `process_uploaded_data(df, task_id)`. Any job submitted via `/submit-job` raises `AttributeError`, gets marked `FAILED`, and stores the error as its result. The queue, worker, status keys, and client helpers are all correct — only the call is wrong. The dashboard is unaffected because it uses the synchronous endpoint.

2. **`app.py` and `api_server.py` are byte-identical.** `diff` reports no differences. The Dockerfile runs `app.py`; `ABSA/.env.example` tells users to run `api_server.py`. Editing one silently leaves the other stale.

3. **Rate-limit numbers disagree across three places.** `RateLimitMiddleware`'s docstring and default say 2/min; `app.py` mounts it at **100/min**; `/process-reviews` separately enforces **10/min per user**. The effective policy is 100/min globally and 10/min on the expensive endpoint.

4. **Dead attributes in `ABSAProcessor`.** `_call_hf_api` and `_get_hf_sentiment` reference `self.api_token`, `self.base_url`, and `self.sentiment_model`, none of which `__init__` assigns. Both methods would raise `AttributeError` — they are unreachable today (the class uses the local PyABSA model), but they are traps for anyone re-enabling the HF-API sentiment route.

5. **The translation model direction looks inverted.** `ai4bharat/indictrans2-en-indic-1.3B` is an English→Indic model, but it is used for Hindi→English. Because failures fall back silently to the original text, a wrong or unavailable model is *invisible at runtime* — Hindi reviews simply pass through untranslated and are then analysed as-is. Worth verifying before trusting multilingual results.

6. **The backend imports Streamlit.** `data_processor.py` calls `st.spinner(...)` around each stage inside the FastAPI process. It works (Streamlit no-ops in bare mode) but couples the ML layer to a UI framework and prints bare-mode warnings into container logs.

7. **Documented endpoint names don't match the code.** The current `README.md` lists `/process`, `/submit_job`, `/job/{id}`, `/log_session`; the real routes are `/process-reviews`, `/submit-job`, `/job-status/{id}`, `/log-session`. It also credits **Google Gemini** for summaries — the code uses **OpenRouter + Nvidia Nemotron**, and the in-pipeline summaries are template strings, not LLM output.

8. **Unused modules carry ~1,700 lines.** `ABSA/src/components/visualizations.py` (`KPIEngine`, `AdvancedVisualizationEngine`, `ExportEngine`, `FilterEngine`) and `ABSA/src/utils/data_management.py` (`DataManager`, `TestDataGenerator`, `SessionManager`, `ConfigManager`) are imported nowhere. `ABSA/src/streamlit_app.py` is untouched Streamlit scaffolding. Notably, **PDF export lives only in the unused `ExportEngine`** — the deployed frontend exports CSV only, despite the older docs promising PDF reports.

9. **`.env` files are present in the working tree** at `ABSA/.env` and `streamlit-deployment/.env`. They are gitignored, but they exist locally and hold real credentials — worth rotating if this tree has ever been shared or archived.

10. **Performance figures in the old README are unverified.** "~2s per review", "99.5%+ uptime", and "max batch size 100" appear nowhere in the code. The only real limits are the `min(300 + 0.3n, 900)` timeout and the rate limits above.

11. **Sessions are not persisted.** `SessionManager` in the frontend writes to `st.session_state` only. A browser refresh discards all analysis history; the "History" page described in the sub-READMEs does not exist in `app_a.py` (navigation is Home / Analytics / Admin).

12. **`TaskManager` state is process-local.** Task IDs, progress, and cancellation flags live in memory. They do not survive a container restart, and they would not work correctly across multiple replicas.

---

## 13. Running it locally

### Backend

```bash
cd ABSA
pip install -r requirements.txt          # ~1.5 GB; torch + pyabsa
cp .env.example .env                     # then fill in what you have

python validate_setup.py                 # preflight: env, Mongo, Redis, /health
python app.py                            # uvicorn on http://0.0.0.0:7860
```

Docs at `http://localhost:7860/docs`. The **first** ABSA request downloads the PyABSA checkpoint and takes several minutes; watch for `✅ PyABSA model loaded successfully`. If you see `⚠️ Using fallback method for aspect extraction`, the model did not load and you are getting keyword-based results — check that `git`/`git-lfs` are on `PATH` (see `PYABSA_FIX.md`).

Via Docker, which matches the deployed environment more closely:

```bash
cd ABSA
docker build -t absa-backend .
docker run -p 7860:7860 --env-file .env absa-backend
```

### Frontend

```bash
cd streamlit-deployment
pip install -r requirements.txt          # ~80 MB, UI only
streamlit run app_a.py                   # http://localhost:8501
```

To point at a local backend, change `HF_SPACES_API_URL` at `app_a.py:55` — it is currently hardcoded to the deployed Space, and the `.env` variable of the same name is not read by the code.

### Admin dashboard (standalone)

```bash
cd ABSA
API_URL=http://localhost:7860 streamlit run admin_dashboard.py
```

Requires `ADMIN_TOKEN` set on the backend; enter the same token in the UI.

### No test suite

There are no automated tests in the repository. `validate_setup.py` is an infrastructure preflight, not a test suite, and the three sample CSVs serve as the de-facto smoke test — process each one and confirm the Analytics tabs populate.

---

## 14. Extension points

Where to work, given the structure above:

| Goal | Where to go |
|---|---|
| Add a language | `TranslationService.process_reviews` — currently gates on `lang == 'hi'`; widen the condition and pick the correct direction-appropriate model. |
| Swap the ABSA model | `ABSAProcessor._load_pyabsa_model` / `_extract_with_pyabsa`. Keep the return schema (`aspects`, `sentiments`, `positions`, `confidence_scores`) and everything downstream keeps working. |
| Add aspect keywords | `_extract_simple_aspects` — the fallback taxonomy; also worth extending because it runs whenever PyABSA misses. |
| Change ranking | `AspectAnalytics.calculate_aspect_scores` — the two formulae and the 10% / 30% inclusion thresholds. |
| Tune alerting | `detect_sentiment_spikes` — `window_days`, the 1.5× ratio, and the ≥ 2 absolute floor. |
| New intent | `IntentClassifier.INTENT_KEYWORDS`; severity tiers are only defined for `complaint` and `praise`. |
| New chart | Add a builder to `dashboard_components.py`, call it from the relevant tab in `app_a.py`. |
| New telemetry event | Add to `MongoDBService.EVENT_TYPES` (unlisted types are rejected), then emit via `frontend_helpers.log_event`. |
| New admin metric | Add an aggregation to `admin_endpoints.py` and a fetcher + renderer in `app_a.py`'s admin section. |
| Fix async processing | `TaskQueue._process_task` — build a DataFrame and call `process_uploaded_data`; see [§12.1](#12-known-drift--gotchas). |
| Change the LLM | `LLM_MODEL` at `app_a.py:58` and the prompt in `generate_llm_insights`. |

---

## License

MIT — see [`LICENSE`](LICENSE).

## Acknowledgments

[PyABSA](https://github.com/yangheng95/PyABSA) · [AI4Bharat IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) · [HuggingFace](https://huggingface.co/) · [Streamlit](https://streamlit.io/) · [Plotly](https://plotly.com/) · [OpenRouter](https://openrouter.ai/)
