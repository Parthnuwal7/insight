# Agentic Insights — Design

**Date:** 2026-08-11
**Status:** Approved for planning
**Scope:** Four phases, A → D, each with its own implementation plan

---

## 1. Why

Insights takes review data and is supposed to return what was found, where the
complaints are, where the product did well, and what to do about it. Today it
returns aspect tables and charts. The interpretation layer that turns those into
findings does not exist in any real sense.

This is not a judgement about polish — it is verifiable in the code:

| Claim | Evidence |
|---|---|
| Summaries are hardcoded, not derived | `data_processor.py:1077+` — `key_issues` is one of three constant strings chosen by whether negative outnumbers positive. Identical for every dataset and every aspect. |
| `recommendation` is a fixed sentence | `generate_macro_summary` returns the same advice regardless of input. |
| The LLM never sees a review | `prepare_analysis_context` (`app_a.py:2441`) passes only counts and percentages for the top 8 aspects. `generate_llm_insights` is asked for "key findings" while holding a frequency table. |
| The report does not exist | `README.md:118` promises "PDF reports"; only CSV `download_button` calls exist and `reportlab` is never imported. |
| The pipeline cannot look twice | Seven fixed stages. When extraction yields nothing — 5 reviews in the eval set — nothing retries, escalates, or flags it. |
| Intent has no negation handling | `IntentClassifier.INTENT_KEYWORDS` is substring matching, so "not a waste of money" scores `high_severity` complaint. |

The benchmark already names the ceiling. `benchmarks/BASELINE_FINDINGS.md`
concludes: *"Every remaining item is model capability. The ops and data-hygiene
work is done."* Implicit aspects F1 0.250, sarcasm 0.500, negation polarity
0.500, Hinglish polarity 0.600, `out_of_taxonomy` recall 0.522. These are
exactly the cases an LLM handles well, which is where the agentic budget goes.

Reliability findings from the 2026-08-11 session, all reproduced:

- `ABSA/src` imports Streamlit and uses it for real work — `st.spinner` around
  every stage, `st.secrets` as a config source (`:136`, `:403`), plus 22
  `st.session_state` / 12 `st.metric` / 4 `st.columns` across the package. The
  backend logs `missing ScriptRunContext!` on every request and resolves config
  differently depending on working directory (`token=MISSING` from one cwd,
  `token=present` from another).
- `extract_aspects_and_sentiments` has a batch loop that does not batch: it
  iterates and calls `self.model.predict(review, ...)` once per review
  (`:506`), paying tokenization and model overhead N times.
- Blanket `except Exception` handlers convert real errors into unusable
  strings. One turned a legitimate timeout into
  `argument of type 'NoneType' is not iterable`, hiding the actual message.
- `redis_service.check_rate_limit` fails open (`:80-83`), so rate limiting
  currently enforces nothing.

Measured on this machine: **2.3s/review** for the full pipeline, so 5,000
reviews is roughly 3.2 hours — far past the 900s synchronous ceiling.

---

## 2. Goals and non-goals

**Goals**

1. Every statement in a report traces to specific reviews.
2. Degraded output is always labelled as degraded, never dressed as good output.
3. Runs of 200–5,000 reviews complete without the user chunking input by hand.
4. The analysis surface is reusable — agent, frontend, and benchmark harness
   call the same tools.
5. Quality is measured, not asserted. Both extraction and interpretation have
   regression baselines.

**Non-goals**

- Authentication, multi-tenancy, user accounts. Single-user for now.
- Real-time or streaming ingestion. Batch uploads only.
- Retraining or fine-tuning any model.
- Restoring Redis or MongoDB. Removed in Phase A, revisited later if needed.
- Volumes above 5,000 reviews per run.

---

## 3. Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Scale target | 200–5,000 reviews | Stated requirement; forces async and map-reduce |
| Extraction | PyABSA first, LLM escalation on weak rows | Spends model budget exactly where the benchmark proves quality is lost |
| Insight architecture | Cluster-then-read | Finds themes nobody thought to query; cost stays flat as volume grows |
| Autonomy | Grounded agent with tools + verification | Agentic, but every claim is checkable against the eval set |
| Report format | In-app section, not a file export | Tightest scope. Rendered in Streamlit during C, rebuilt in Next.js in D. Same generator can emit standalone HTML later |
| Job store | Local, no external services | Free-tier lapses caused today's busy-wait bug and dead telemetry |
| Redis / MongoDB | Removed | Reinstated later if rate limiting or telemetry is wanted |
| Batching | Phase A | The code is being rewritten there anyway; doing it later means editing twice |

---

## 4. Target architecture

```
ABSA/src/absa/            extraction — what customers mentioned
  config.py               Settings dataclass, env-loaded, validated at startup
  progress.py             ProgressReporter protocol (no-op default)
  validation.py           DataValidator
  translation.py          TranslationService
  extraction.py           ABSAProcessor (batched)
  intent.py               IntentClassifier
  analytics.py            AspectAnalytics
  pipeline.py             orchestrator -> ExtractionResult

ABSA/src/insights/        interpretation — what it means
  embed.py                review -> vector
  cluster.py              vectors -> clusters + representatives
  tools.py                retrieval surface (single source of truth)
  mcp_server.py           exposes tools.py over MCP
  agent.py                bounded investigation loop
  verify.py               claim -> supported / dropped
  report.py               verified claims -> Report

ABSA/src/jobs/            execution (Phase B)
  store.py                SQLite job store
  runner.py               chunked, staged, concurrent execution

ABSA/app.py               HTTP layer only
web/                      Next.js frontend (Phase D)
```

Dependency direction is one-way: `insights/` may import `absa/`; `absa/` never
imports `insights/`. Neither imports Streamlit, FastAPI, or any UI framework.
That inversion is what makes a decoupled frontend possible and what lets the
benchmark harness keep importing the real pipeline directly.

`SummaryGenerator` is deleted. Its hardcoded strings are precisely what the
insight engine replaces, and keeping both would leave two competing answers to
"what did we find."

---

## 5. Phase A — Decouple and batch

**Blocks every other phase.**

### Removals

| Removed | Consequence |
|---|---|
| `redis_service.py` | Rate limiting goes. No behavioural loss — it already fails open. |
| `task_queue.py` | Redis-backed queue goes, and with it the busy-wait defect. |
| `mongodb_service.py`, `ip_location_service.py` | Telemetry, session logs, geo. Removes the 5s stall per call. |
| `admin_dashboard.py`, admin endpoints | Lose their data source, so they go too. |
| `/log-session`, `/log-event`, `/submit-job`, `/job/{id}`, admin routes | Matching calls in `frontend_helpers.py` removed in the same change so Streamlit does not start 404ing. |

`task_manager.py` stays — in-process, no Redis dependency, already provides
progress and cancellation.

### Additions

**`config.py`** — one `Settings` dataclass built from environment at startup,
replacing `st.secrets` and scattered `os.getenv` calls. Validates eagerly: a
missing `HF_TOKEN` raises at boot rather than silently disabling Hindi
translation mid-run.

**`progress.py`** — `ProgressReporter` protocol with `stage(name)` and
`advance(n)`. No-op default; the API layer passes one writing to `TaskManager`;
tests pass a recording one and assert on stages. Replaces all five `st.spinner`
blocks.

**Batched extraction** — replace per-review `model.predict(review)` with list
input so one forward pass covers a batch. Provenance fields
(`extraction_method`, `degraded_reason`) are preserved exactly.

### Acceptance criteria

- No module under `ABSA/src` imports `streamlit`. Enforced by a test.
- `python -c "import absa"` works from any working directory with identical
  config resolution.
- Benchmark harness reports **aspect F1 0.746 and sentiment accuracy 0.873**,
  unchanged. Batching that moves either number does not land.
- Batched extraction is measurably faster than per-review on the 46-review eval
  set; the speedup is recorded in the run manifest.

---

## 6. Phase B — Throughput

**Depends on A.** Makes 200–5,000 review runs practical.

### Job store

SQLite (`store.py`), WAL mode, one file beside the app. Jobs and their chunk
results survive process restarts, which the current in-memory `TaskManager`
cannot do. Schema covers: job id, status, stage, progress, chunk results,
error, timestamps.

### Execution model

The key insight is that **stages have different bottlenecks and therefore need
different concurrency**:

| Stage | Bottleneck | Concurrency |
|---|---|---|
| Validation | trivial | serial |
| Language detection | CPU, cheap | serial |
| Translation | network I/O | 8 concurrent requests, configurable |
| Extraction | CPU + 1.1 GB model per process | bounded process pool, default 2 |
| Embedding | CPU, batches well | serial, batched |
| Clustering | CPU, single pass | serial |

The embedding and clustering rows apply once Phase C exists. If B lands first,
those stages are simply absent and the table describes the pipeline as it stands.

Threads do not help extraction — measured identical times on main thread versus
worker thread, as expected under the GIL. Parallelism there must be process-based,
and each process loads its own 1.1 GB checkpoint, so pool size is bounded by RAM
rather than cores. Default 2, configurable, documented as a memory tradeoff.

Input is split into chunks (default 100 reviews). Each chunk's result is
persisted on completion, so a crash loses at most one chunk rather than the
whole run. Progress aggregates across chunks. Cancellation is checked at chunk
boundaries.

### Acceptance criteria

- A 5,000-review run completes without manual chunking.
- Killing the process mid-run and restarting resumes from the last completed
  chunk.
- Cancellation takes effect within one chunk.
- Per-stage concurrency is configurable and defaults are documented with their
  memory cost.

---

## 7. Phase C — Insight engine

**Depends on A.** Independent of B.

### Components

**`embed.py`** — local `all-MiniLM-L6-v2` (~90 MB) rather than an embedding
API. Torch is already installed, it runs on CPU, batches naturally, and keeps
the no-external-services property. Embeddings run on **translated English
text**, so a Hindi and an English review about the same thing cluster together.

**`cluster.py`** — `sklearn.cluster.HDBSCAN`, available in the pinned
scikit-learn 1.3, so no new dependency. Chosen over KMeans because it needs no
guessed `k` and explicitly labels outliers as noise — and those outliers are
the one-off complaints nobody else made, which are interesting rather than
garbage. Each cluster yields a medoid plus nearest neighbours as
representatives.

**`tools.py`** — the retrieval surface, plain Python over one run's data:

| Tool | Returns |
|---|---|
| `list_clusters()` | id, size, dominant sentiment, top aspects |
| `get_cluster_reviews(id, limit, sentiment?)` | representative texts + review ids |
| `get_reviews_for_aspect(aspect, sentiment?, limit)` | texts + ids |
| `search_reviews(query, limit)` | semantic search over embeddings |
| `get_aspect_stats()` | `AspectAnalytics` priority / strength scores |
| `get_extraction_health()` | counts by `extraction_method` / `degraded_reason` |

`get_extraction_health` exists so the agent can caveat its own conclusions when
a meaningful share of rows came from keyword fallback, instead of asserting
confidently over degraded data.

**`mcp_server.py`** — thin adapter exposing exactly those functions. `tools.py`
stays plain Python so tests call it directly with no MCP transport, while the
agent, the frontend, and the harness all go through MCP. Shared surface without
MCP becoming a testing burden.

**`agent.py`** — bounded loop. The agent sees cluster and aspect summaries,
chooses what to investigate, and pulls actual review text. Hard bounds on tool
calls, tokens, and wall-clock. Every claim carries the `review_ids` it rests on.

**`verify.py`** — re-fetches each claim's cited reviews and checks at
temperature 0 whether the text supports the claim. Unsupported claims are
**dropped, not softened**. The report states how many were dropped.

**`report.py`** — assembles verified claims into the four things the product
promises: what was found, areas of complaint, where the product did well, and
action items. Each carries citations, so every sentence traces back to real
reviews.

### Escalation

Rows where `extraction_method != 'pyabsa'` or where extraction returned nothing
are re-processed by the LLM **before** analytics and clustering, so scores and
clusters are never computed over data already known to be degraded.

Escalation and the agent share one provider configured in `Settings`
(OpenRouter today, model id configurable). Escalated rows keep provenance:
`extraction_method` becomes `llm_escalated`, so the benchmark can score them
separately and the report can report on them honestly.

### Rendering during Phase C

Phase C ends with a `Report` object and a **minimal Streamlit section** that
renders it — narrative, findings with expandable citations, action items. This
keeps C independently shippable and testable without waiting for D. Phase D
rebuilds that view in Next.js; the `Report` object does not change.

### Acceptance criteria

- Every claim in a rendered report carries at least one review id.
- Claims whose cited text does not support them are absent from the report, and
  the dropped count is displayed.
- Report generation with the LLM disabled produces statistics and clusters with
  the narrative section explicitly marked unavailable — never canned prose.
- `tools.py` has unit tests over a fixed fixture requiring no LLM and no MCP
  transport.

---

## 8. Phase D — Next.js frontend

**Depends on A and C.**

Next.js (App Router, TypeScript) talking to FastAPI over JSON. No shared Python
state, which is only possible once Phase A removes Streamlit from the backend.

- **Upload** — CSV in, returns a job id.
- **Progress** — server-sent events from the job store; falls back to polling.
- **Report view** — renders the `Report` object. Citations are expandable:
  clicking a finding reveals the review text it rests on. This is the payoff of
  carrying ids through the whole pipeline.
- **Explore** — the existing aspect tables and charts, rebuilt against the JSON
  API.

The Streamlit app keeps working throughout A, B, and C, and is retired only when
D reaches parity. Authentication stays out of scope.

### Acceptance criteria

- No Python renders UI.
- Upload → progress → report works end to end against a running backend.
- Every finding in the report view expands to its supporting review text.
- Streamlit is deleted only after parity is demonstrated.

---

## 9. Failure behaviour

The governing rule, established on the `fix/silent-fallback` branch: **never let
degraded output impersonate good output.** The insight engine raises the stakes,
because a fluent paragraph is more convincing than a table and therefore more
damaging when built on keyword-matched rows.

| Failure | Behaviour |
|---|---|
| LLM unavailable | Statistics and clusters render; narrative marked unavailable. Never canned prose. |
| Embeddings fail | Aspect axis only; theme section marked absent. |
| PyABSA unavailable | Keyword fallback runs, and the report header states the data is keyword-derived. |
| Translation unavailable | Non-English rows marked untranslated and excluded from clustering, not silently embedded in the wrong language. |
| Agent hits bounds | Returns what it verified, marked partial. |
| Verification fails a claim | Claim dropped; count surfaced. |

Exception handling becomes narrow. Blanket `except Exception` is what turned a
legitimate timeout into `argument of type 'NoneType' is not iterable` and hid
the real message. Specific catches only; anything unexpected propagates with
context and becomes a typed error response with a machine-readable code.

---

## 10. Testing

1. **Phase A moves no numbers.** Decoupling and batching are refactors. The
   harness must still report aspect F1 0.746 and sentiment accuracy 0.873 on
   `eval_reviews_v1.csv`. This is the gate on batching.
2. **Claim groundedness** — a new metric scoring whether each report claim is
   supported by the reviews it cites. Labelled once over the eval set, then
   tracked as a regression baseline. The insight layer's analogue of F1; it is
   what stops prompt edits from quietly degrading honesty.
3. **`tools.py`** — ordinary unit tests over a fixed fixture. No LLM, no MCP.
4. **`agent.py`** — tested against a stubbed LLM with recorded transcripts, so
   control flow is deterministic and testable without spending tokens.
5. **Import hygiene** — a test asserting no module under `ABSA/src` imports
   Streamlit, so the decoupling cannot silently regress.
6. **Job store** — restart-mid-run resumption is an explicit test, not a manual
   check.

---

## 11. Sequencing

```
A (decouple + batch)
├─> B (throughput)          independent of C
└─> C (insight engine)  ──> D (Next.js + report view)
```

A blocks everything. B and C may proceed in either order or together. D needs
both A's API contract and C's `Report` object.

Each phase gets its own implementation plan.

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| Batching changes PyABSA output subtly | Gated on benchmark parity; does not land if F1 moves |
| Process pool exhausts RAM (1.1 GB per worker) | Default 2 workers, configurable, memory cost documented |
| Agent costs balloon on large runs | Hard bounds on tool calls, tokens, wall-clock; caching keyed by content hash |
| Clustering quality depends on translation | Already-fixed sentence-splitting matters more now; untranslated rows are excluded rather than mis-clustered |
| Groundedness labelling is manual effort | Label once over the 46-review eval set; thereafter it is a regression check |
| Phase D is a large rewrite | Streamlit stays functional until parity; retirement is the last step |
