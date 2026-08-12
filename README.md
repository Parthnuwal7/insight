# 🔍 Insights — Aspect-Based Sentiment Analysis Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

**Give it your review data. It tells you what customers actually said — which
aspects they complained about, where the product did well, and what to do about
it — with every finding traceable back to the reviews it came from.**

</div>

---

## Project Overview

Most sentiment tools return one polarity score per review. Insights returns
per-*aspect* sentiment: not "this review is negative" but "battery is negative,
price is positive, camera is negative." That granularity is what makes the output
actionable — you can rank what to fix.

The system ingests a CSV of reviews (English or Hindi), extracts aspects and their
sentiments, ranks problem areas against strengths, and renders the result as an
interactive dashboard.

**Current state:** the extraction and throughput layers are built and measured. The
interpretation layer — turning aspect tables into written findings — is specified
and planned, not yet built. See [Roadmap](#roadmap) for the honest breakdown.

---

## How It Works

```
CSV ──▶ validate ──▶ detect language ──▶ translate (hi→en) ──▶ extract aspects
                                                                     │
                            ┌────────────────────────────────────────┘
                            ▼
                   classify intent ──▶ rank aspects ──▶ dashboard
```

| Stage | Implementation | Output |
|-------|----------------|--------|
| **Validation** | `DataValidator` | Clean DataFrame, schema enforced |
| **Language detection** | `langdetect` | `hi` / `en` per review |
| **Translation** | `Helsinki-NLP/opus-mt-hi-en` via HF Inference API | English text, translated sentence-by-sentence with caching |
| **Aspect extraction** | PyABSA ATEPC (multilingual checkpoint) | Aspect spans per review |
| **Sentiment** | PyABSA ATEPC | Positive / Negative / Neutral per aspect |
| **Intent** | Keyword classifier with severity tiers | Complaint / praise / question / suggestion |
| **Analytics** | `AspectAnalytics` | Priority scores, strength anchors, co-occurrence graph |

Every extracted row carries **provenance**: `extraction_method` records whether the
result came from real ABSA or a degraded fallback, and `degraded_reason` records
why. Degraded output is never presented as if it were good output.

---

## Measured Quality

The pipeline is benchmarked against a 46-review evaluation set with per-category
probes (multi-aspect, mixed sentiment, sarcasm, negation, Hindi, out-of-taxonomy).
Results are reproducible via `benchmarks/harness/run_benchmark.py`.

| Metric | Value |
|---|---|
| Aspect F1 | **0.746** |
| Sentiment accuracy | **0.873** |
| Keyword-fallback rows | **0%** |
| Pipeline time (46 reviews) | **~35s** |

These numbers are a regression gate, not a marketing claim: every change touching
the pipeline must reproduce them or it does not land. Remaining weaknesses are
documented honestly in `benchmarks/BASELINE_FINDINGS.md` — implicit aspects
(F1 0.250), sarcasm (0.500), and negation polarity (0.500) are model-capability
limits, not tuning gaps.

---

## Architecture

The backend is a **pure Python library with no UI framework and no external service
dependencies**. It does not import Streamlit, FastAPI, or the job store. That
inversion is what makes the pipeline independently testable and lets the API,
benchmark harness, and future frontends all consume the same code.

```
ABSA/
├── app.py                  FastAPI layer — HTTP only, no business logic
└── src/
    ├── absa/               EXTRACTION — what customers mentioned
    │   ├── validation.py       schema validation
    │   ├── translation.py      hi→en, concurrent, sentence-wise cache
    │   ├── extraction.py       PyABSA aspects + sentiment (batched)
    │   ├── intent.py           intent + severity
    │   ├── analytics.py        priority scores, co-occurrence
    │   ├── aspect_canonical.py surface-form → canonical aspect
    │   ├── pipeline.py         orchestration
    │   ├── progress.py         ProgressReporter protocol (no UI dependency)
    │   └── config.py           env-driven Settings, validated at startup
    └── jobs/               EXECUTION — durable long-running work
        ├── store.py            SQLite job store, thread-safe
        ├── runner.py           chunked execution, resumption, cancellation
        ├── pool.py             optional process pool for extraction
        └── progress.py         job-store progress adapter

streamlit-deployment/       Dashboard (Streamlit)
benchmarks/                 Evaluation set, harness, recorded runs
docs/superpowers/           Design spec and phase-by-phase implementation plans
```

**Dependency direction is one-way and test-enforced:** `jobs/` may import `absa/`;
`absa/` may import neither `jobs/` nor the API layer. A guard test fails the build
if that is ever violated.

---

## Durable Jobs

A 5,000-review run takes hours, which makes a synchronous request untenable.

- **Chunked execution** — work splits into chunks (default 100 reviews); each
  chunk's result is persisted as it completes.
- **Restart resumption** — if the process is killed mid-run, restarting resumes
  from the last persisted chunk. Verified by hard-killing a live server mid-job and
  confirming the replacement process reprocessed nothing.
- **Cancellation** — checked at chunk boundaries.
- **Per-stage concurrency** — translation is network-bound and uses threads;
  extraction is CPU-bound and offers an optional process pool.

Job state lives in SQLite beside the app. No Redis, no MongoDB — earlier versions
depended on both, and both lapsing took the system down with them.

---

## Features

**Dashboard**
- KPI metrics, sentiment distribution, aspect coverage
- Timeline analysis with anomaly detection
- Aspect-sentiment heatmaps and co-occurrence network graphs
- Sankey diagrams: intent → aspect → sentiment
- Sentiment-filtered word clouds
- Impact simulation (what-if analysis on aspect improvement)

**Analytics**
- Dual ranking: areas of improvement vs. strength anchors
- Priority scoring weighted by complaint severity
- Aspect canonicalisation, so one concept is not split across five surface forms
- CSV export

---

## Quick Start

**Prerequisites:** Python 3.11, ~4 GB RAM, ~2 GB disk for the model checkpoint.

```bash
# Backend
python -m venv .venv
.venv/Scripts/pip install -r ABSA/requirements.txt
cd ABSA && python -m uvicorn app:app --port 7860

# Dashboard (separate terminal, separate venv — no ML dependencies)
python -m venv .venv-ui
.venv-ui/Scripts/pip install -r streamlit-deployment/requirements.txt
.venv-ui/Scripts/streamlit run streamlit-deployment/app_a.py
```

Sample data lives in `streamlit-deployment/test_data_*.csv`.

> **Dependency note:** `ABSA/requirements.txt` carries load-bearing pins
> (`update_checker<1.0`, `spacy>=3.7,<3.9`, `transformers<4.37`). Relaxing any of
> them makes PyABSA fail *silently* and degrade to keyword matching. Each pin
> carries a comment explaining why.

### Environment

```env
HF_TOKEN=your_huggingface_token      # required — Hindi translation
OPENROUTER_API_KEY=your_key          # optional — LLM features
ABSA_ALLOW_NO_TRANSLATION=0          # set 1 to run deliberately without translation
CHUNK_SIZE=100                       # reviews per job chunk
EXTRACTION_WORKERS=1                 # process pool size (see note below)
TRANSLATION_WORKERS=8                # concurrent translation requests
```

`EXTRACTION_WORKERS` defaults to 1 by measurement, not assumption: two workers were
~11% faster on a 230-review benchmark (inside run-to-run variance) while peak
memory rose from 2.8 GB to 6.0 GB. The speedup was not distinguishable from noise;
the memory cost was.

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Service info |
| `/health` | GET | Component health, including whether real ABSA is available |
| `/process-reviews` | POST | Synchronous processing (small batches) |
| `/jobs` | POST | Submit an async job → job id |
| `/jobs` | GET | List jobs for a user |
| `/jobs/{job_id}` | GET | Job status, stage, chunk progress |
| `/jobs/{job_id}/results` | GET | Merged results of a completed job |
| `/jobs/{job_id}/cancel` | POST | Request cancellation |

```bash
curl -X POST http://localhost:7860/jobs \
  -H "Content-Type: application/json" \
  -d '{"data":[{"id":1,"review":"Battery life is amazing but the camera is poor.",
                "reviews_title":"Great","date":"2024-01-15","user_id":"u1"}],
       "user_id":"demo"}'
```

---

## Sample Output

**Input** — `"Battery bahut achi hai lekin camera quality thodi kam hai."`

```json
{
  "review": "Battery bahut achi hai lekin camera quality thodi kam hai.",
  "translated_review": "The battery is very good but the camera quality is a bit low.",
  "detected_language": "hi",
  "aspects": ["battery", "camera quality"],
  "aspects_canonical": ["battery", "camera"],
  "aspect_sentiments": ["Positive", "Negative"],
  "intent": "complaint",
  "extraction_method": "pyabsa",
  "degraded_reason": null
}
```

---

## Tech Stack

| Category | Technologies |
|---|---|
| **Backend** | FastAPI, Uvicorn, Pydantic v2 |
| **ML/NLP** | PyABSA (ATEPC), HuggingFace Transformers, PyTorch, spaCy, langdetect |
| **Translation** | Helsinki-NLP opus-mt via HF Inference API |
| **Persistence** | SQLite (WAL) |
| **Frontend** | Streamlit, Plotly, NetworkX, WordCloud |
| **Testing** | pytest — 302 tests by default (2 `slow` deselected), plus a 20-test groundedness benchmark suite and a reproducible accuracy benchmark |

---

## Roadmap

Work is planned in four phases. Specs and task-level plans live in
`docs/superpowers/`.

| Phase | Scope | Status |
|---|---|---|
| **A** | Decouple the backend from Streamlit; batch inference | ✅ **Done** — 2.5× faster extraction, accuracy unchanged |
| **B** | Durable SQLite job store, chunked resumption, per-stage concurrency | ✅ **Done** — restart-resumption proven against a killed process |
| **C** | Insight engine: embeddings, clustering, grounded agent, verifier, report | ✅ **Done** — 9 tasks shipped, plus a whole-branch review pass |
| **D** | Decoupled Next.js frontend | 📝 **Specified** |

### What Phase C changes

Today the "insights" the dashboard shows are **templated**: a hardcoded sentence
selected by whether negatives outnumber positives, identical for every dataset. The
one LLM call receives aggregate counts only — never review text — so it can restate
arithmetic but cannot explain anything.

Phase C replaces that with findings derived from the reviews themselves:

- **Embeddings + HDBSCAN clustering** add a *theme* axis alongside the aspect axis,
  surfacing patterns the aspect taxonomy cannot express — and treating outliers as
  interesting rather than noise.
- **LLM escalation** re-processes only the rows PyABSA handled poorly, spending
  model budget where the benchmark proves quality is lost.
- **A bounded agent** queries both axes through one retrieval surface (also exposed
  over MCP), choosing what to investigate and reading actual review text.
- **A fail-closed verifier** re-checks every claim against its cited reviews.
  Unsupported claims are **dropped, not softened** — three grounded findings beat
  ten plausible ones.
- **Groundedness** becomes a tracked metric, the interpretation layer's analogue of
  F1, so prompt changes cannot quietly degrade honesty.

The design rule throughout: **an empty report is a valid report.** If nothing
survives verification, the system says so rather than falling back to prose.

---

## Engineering Notes

A few decisions that shaped the codebase, recorded because the reasoning is not
obvious from the code alone:

- **Import order is load-bearing.** On Windows, importing `pandas` before `pyabsa`
  segfaults the interpreter (exit 139) inside sentencepiece's native extension. A
  preload guard in `absa/__init__.py` enforces the order for every consumer,
  including spawned worker processes.
- **Degradation is always labelled.** `extraction_method` and `degraded_reason`
  travel with every row from extraction through to the API. An earlier version
  silently emitted keyword-matched output indistinguishable from real ABSA — 17% of
  rows in a "healthy" run.
- **Benchmark parity gates every pipeline change.** Refactors must reproduce
  F1 0.746 / accuracy 0.873 exactly. The `absa` package split was validated this
  way: five of six moved classes were byte-identical, and the metrics file matched
  the baseline byte-for-byte.

---

## Acknowledgments

- **[PyABSA](https://github.com/yangheng95/PyABSA)** — aspect-based sentiment analysis framework
- **[HuggingFace](https://huggingface.co/)** — transformers and model hosting
- **[Helsinki-NLP](https://huggingface.co/Helsinki-NLP)** — opus-mt translation models
- **[Streamlit](https://streamlit.io/)** — dashboard framework

---

## License

MIT — see [LICENSE](LICENSE).
