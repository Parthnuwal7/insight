# ABSA Accuracy Benchmark

A persisted, reproducible baseline for the current ABSA pipeline — built so that
every future accuracy claim can be checked against a number rather than an
impression.

The harness imports and runs the **real** pipeline from
`ABSA/src/utils/data_processor.py` unmodified. Nothing in `ABSA/` is changed;
instrumentation is applied by wrapping methods at runtime and restoring them
afterwards.

---

## Why this exists

`ABSAProcessor` has four routes to a result, and the production output records
none of them:

| Route | What actually ran |
|---|---|
| `pyabsa` | Real aspect-based sentiment analysis |
| `pyabsa_empty_to_fallback` | PyABSA found nothing → 14-entry keyword taxonomy |
| `pyabsa_error_to_fallback` | PyABSA raised on this review → keyword taxonomy |
| `model_unavailable_fallback` | Checkpoint never loaded → keyword taxonomy |

Only the first is ABSA. The other three emit a fixed 14-bucket vocabulary with a
hardcoded `0.7` confidence, land in the same table, and are indistinguishable
downstream. An accuracy number computed over a blend of both is measuring two
different systems at once — so the benchmark reports **everything split by route**.

---

## Layout

```
benchmarks/
├── eval_set/
│   └── eval_reviews_v1.csv        46 reviews, tagged by probe category
├── harness/
│   ├── run_benchmark.py           runs the real pipeline + label-free metrics
│   ├── instrument.py              runtime provenance recording
│   ├── matching.py                aspect normalization + fuzzy alignment
│   ├── metrics_unlabeled.py       metrics that need no ground truth
│   ├── make_judge_packet.py       builds the blind LLM judging prompt
│   ├── score_judgments.py         scores judge output → triage.md
│   ├── run_insight_report.py      generates a Report over a run's extraction output
│   ├── score_groundedness.py      scores a Report's claims → groundedness.json
│   └── test_score_groundedness.py unit tests for the groundedness scorer (no network)
├── judgments/                     drop LLM replies here as <run_id>.json
└── runs/
    ├── LATEST                     run id of the most recent run
    └── <run_id>/
        ├── manifest.json          git sha, versions, timings, eval set hash
        ├── review_level.csv       one row per review + provenance
        ├── aspect_level.csv       one row per (review, aspect) + provenance
        ├── predictions.json       machine-readable predictions
        ├── metrics_unlabeled.json label-free metrics
        ├── metrics_unlabeled.md   human-readable summary
        ├── judge_packet.md        prompt to paste into an LLM
        ├── metrics_labeled.json   (after scoring)
        ├── triage.md              (after scoring) the decision document
        ├── report.json            (after run_insight_report.py) a generated Report
        ├── groundedness.json      (after score_groundedness.py) per-claim verdicts
        └── groundedness.md        (after score_groundedness.py) human-readable summary
```

---

## The evaluation set

46 reviews in `eval_set/eval_reviews_v1.csv`. Two provenances:

- **`bundled_*`** (32) — drawn verbatim from the three sample CSVs already in
  `streamlit-deployment/`, so results are traceable to data the project shipped.
- **`authored_probe`** (14) — written to cover phenomena the bundled samples do
  not exercise at all.

Every row carries a `probe_category`, which is what makes the triage diagnostic
rather than just a score:

| Category | n | What it tests |
|---|---:|---|
| `multi_aspect` | 5 | Several distinct aspects in one review |
| `mixed_sentiment` | 10 | Opposing polarities in one review |
| `out_of_taxonomy` | 11 | Aspects with no keyword bucket (ads, privacy, noise, export) |
| `hindi` | 3 | Devanagari — translation path |
| `hinglish` | 3 | Romanized code-mix — `langdetect` blind spot |
| `negation` | 3 | Polarity inverted by negation |
| `sarcasm` | 3 | Positive surface words, negative intent |
| `implicit_aspect` | 3 | Aspect evaluated but never named |
| `comparative` | 1 | Judgment relative to a prior version |
| `single_aspect_control` | 2 | Trivial cases; failure here means something is badly wrong |

**Swapping in real reviews is a one-file change.** Point `--eval-set` at any CSV
with columns `id, review, reviews_title, date, user_id` (plus optional
`source, probe_category, lang_expected`). Review text must be unique —
`clean_data()` silently drops duplicates, which would break row alignment.

---

## Running it

### Setup

The benchmark uses an isolated venv so it never disturbs your global packages.

```bash
python -m venv .venv-bench
.venv-bench/Scripts/python.exe -m pip install --upgrade pip
.venv-bench/Scripts/python.exe -m pip install \
    "torch>=2.0.0,<2.2.0" "transformers>=4.30.0,<4.37.0" "pyabsa>=2.4.0,<3.0.0" \
    "pandas>=1.5.0,<2.1.0" "numpy>=1.24.0,<1.26.0" "scikit-learn>=1.3.0,<1.4.0" \
    "update_checker<1.0" \
    langdetect networkx streamlit python-dotenv sentencepiece sacremoses
```

> **`update_checker<1.0` is required.** See [Environment notes](#environment-notes).

### 1. Run the pipeline

```bash
.venv-bench/Scripts/python.exe benchmarks/harness/run_benchmark.py --label baseline
```

Writes a timestamped directory under `runs/` and prints the label-free metrics.
The first run downloads the PyABSA checkpoint and is slow; later runs are not.

### 2. Get the judge labels

Open `runs/<run_id>/judge_packet.md`, paste the block between the PROMPT markers
into a strong LLM, and save the raw JSON reply to
`benchmarks/judgments/<run_id>.json`.

The judge sees **only the review text** — never the pipeline's predictions.
Showing predictions would anchor the model into ratifying what it was shown and
inflate the score.

### 3. Score

```bash
.venv-bench/Scripts/python.exe benchmarks/harness/score_judgments.py --run <run_id>
```

Writes `metrics_labeled.json` and `triage.md`.

### 4. Generate and score a report's groundedness (Phase C)

The metrics above score the ABSA extraction layer. `insights.report.Report` —
the investigation agent's cited findings/complaints/strengths/actions — is a
separate layer with its own failure mode: a claim that reads as grounded but
whose citation does not actually support it. `insights.verify` already checks
this once, internally, before a claim is allowed into a `Report` — but that
check cannot catch a regression in the prompt that produces the check itself.
These two scripts are the external, independent gate:

```bash
.venv-bench/Scripts/python.exe benchmarks/harness/run_insight_report.py --run <run_id>
.venv-bench/Scripts/python.exe benchmarks/harness/score_groundedness.py --run <run_id>
```

`run_insight_report.py` re-shapes an existing run's `predictions.json` into
the `processed_data` / `aspect_level_data` records `InsightTools` expects,
runs `insights.agent.investigate` → `insights.verify.verify_claims` →
`insights.report.build_report` (the same sequence `ABSA/app.py`'s
`POST /insights/report` runs), and writes the result to `report.json`. It
needs `OPENROUTER_API_KEY` and skips theme clustering (no embedding model
load) — see the script's docstring.

`score_groundedness.py` then takes each claim in `report.json`, re-fetches
its cited reviews from `predictions.json`, and asks an independent LLM judge
(temperature 0, its own prompt — deliberately **not** imported from
`insights.verify`, so a prompt regression there cannot silently pass this
gate too) whether the text actually supports the claim. Writes
`groundedness.json` and `groundedness.md`. See the script's module docstring
for the full design rationale, and "Groundedness" below for the current
baseline.

`test_score_groundedness.py` covers the scorer's own logic (unknown citation
handling, the zero-claims denominator, fail-closed behaviour, verdict
parsing) against a fixture report, with the judge call stubbed — no network,
no API key required:

```bash
.venv-bench/Scripts/python.exe -m pytest benchmarks/harness/test_score_groundedness.py -v
```

---

## What gets measured

### Without labels

| Metric | Symptom it explains |
|---|---|
| Route distribution / fallback rate | Aspects look generic and coarse |
| Fragmentation ratio | Same aspect appears under several surface forms |
| Translation effectiveness | Non-English reviews handled badly |
| Confidence distribution | Share of rows carrying the hardcoded `0.7` |
| Coverage | Aspects missed entirely |
| Heuristic provenance check | Whether production can detect fallback rows from the API payload alone |

### With labels

| Metric | Note |
|---|---|
| Aspect precision / recall / F1 | Fuzzy matching, not exact string equality |
| Sentiment accuracy | Computed **only over correctly matched aspects** |
| Coverage ratio | Predicted aspects ÷ gold aspects |
| Sentiment confusion matrix | Which polarity flips dominate |

### Report groundedness (Phase C)

| Metric | Value (2026-08-12) | Note |
|---|---|---|
| Citation validity | **1.000** (11/11 claims) | Every cited review id resolves. Needs no judge. |
| Groundedness fraction | **not established** | Judge unreachable — see below |
| Claims scored | 11 (8 complaints, 3 strengths) | Report over the 46-review eval set |

**The groundedness baseline is not yet established, and the number is
`null` rather than `0.0`.** A report was generated end to end over the
evaluation set and produced 11 verified claims, but scoring hit the
OpenRouter free-tier daily quota before any claim could be judged. Re-run
`score_groundedness.py --run 20260811T180751Z-task4-pool-default-serial`
once quota or credit is available; `report.json` is committed, so no
re-processing is needed.

Three decisions, all deliberate, mirroring the discipline in
`insights.verify` (see its module docstring) but applied by a scorer that
does not import from it:

**A claim citing a review id absent from the run's data is ungrounded, not
skipped.** Excluding it from the denominator would flatter the fraction
exactly where the report is least trustworthy.

**Zero claims means an undefined fraction, not a perfect one.** A report
that found nothing to say is not "100% grounded" — `groundedness_fraction`
is `null` in that case, with an explanatory `groundedness_note`.

**A claim the judge could not reach is `unjudged`, not `ungrounded`, and is
excluded from the denominator.** This distinction is the difference between
"we measured nothing" and "everything failed". The first version of this
scorer conflated them and emitted `groundedness_fraction: 0.0` for a run in
which zero claims had actually been evaluated — a damning number for a
measurement that never ran. When no claim can be judged the fraction is
`null`; `citation_validity_fraction` still reports what is knowable without
a judge.

Two measurement decisions, both deliberate:

**Aspect matching is fuzzy.** Exact comparison would score a predicted
`"battery life"` against a gold `"battery"` as *both* a false positive and a
false negative — punishing the model twice for a normalization problem. Fuzzy
matching separates "didn't find it" from "found it, named it differently".

**Sentiment accuracy is conditional.** Measured unconditionally, a missed aspect
and a flipped polarity blur into one number and you cannot tell which is moving.

---

## Environment notes

**`update_checker` must be pinned below 1.0.** The dependency chain is
`pyabsa → metric_visualizer → update_checker`. `update_checker 1.0.0` changed
`UpdateChecker.check()` to take no arguments, while `metric_visualizer 0.9.17`
still calls `checker.check(__name__, __version__)`. The result is a `TypeError`
at `import pyabsa` — which `_load_pyabsa_model()` catches with a bare
`except Exception`, logs, and swallows, leaving `self.model = None`.

Nothing raises. The service starts, answers requests, and returns keyword-bucket
output shaped exactly like ABSA output. **This is reachable purely through
dependency drift** — no code change required — because none of these transitive
dependencies are pinned in `ABSA/requirements.txt`.

`logs.md` in the repo root records a *different* PyABSA import failure
(`ImportError: cannot import name 'checkpoint_utils'`) hitting the same bare
`except`, which is evidence this failure mode has already occurred in the
deployed environment.

**`en_core_web_sm` must be installed.** PyABSA's polarity-classification path
calls `spacy.load("en_core_web_sm")`. That model is not declared in
`ABSA/requirements.txt`. When it is missing, *every* `_extract_with_pyabsa` call
raises `OSError [E050]`, the per-review `except` in `ABSAProcessor` swallows it,
and the review takes the keyword fallback — while the checkpoint reports loaded
and healthy. Install with:

```bash
.venv-bench/Scripts/python.exe -m spacy download en_core_web_sm
```

**Import order is load-bearing on Windows.** Importing `pandas` before `pyabsa`
segfaults the interpreter (exit 139), either at `import pyabsa` or later during
model construction — a native init-order conflict between the numpy/pandas and
torch/pyabsa stacks. `run_benchmark.py` imports `pyabsa` on its first line for
this reason; do not reorder it. This is an artifact of the local Windows
environment and is not expected to affect the Linux container.

**`HF_TOKEN` is absent from `ABSA/.env`.** `TranslationService` therefore returns
source text unchanged for every review, with no HTTP request made and no error
surfaced.

---

## Known limitations of the metrics

**Fragmentation clustering over-merges on generic head nouns.** `matching.py`
treats a token-subset as a match, so `Quality` ⊂ `sound quality` and
`Quality` ⊂ `Food quality` chain transitively into one cluster, producing a
`food` group that wrongly contains `sound quality`. Two consequences:

- the fragmentation ratio is a slight **over**-estimate wherever forms share a
  generic head noun;
- aspect matching in the scorer is correspondingly **lenient**, which biases
  aspect F1 *upward*.

Both biases are generous to the current pipeline, which is the safe direction
for a baseline: a future improvement cannot be overstated by them. Tightening
this (requiring head-noun agreement, or dropping subset matching for
single-token forms) is worth doing before the numbers are used to compare two
candidate models against each other rather than against this baseline.

---

## Groundedness baseline (Phase C)

**Status: not yet established.** A report was generated end-to-end against
the full 46-review eval set on **2026-08-12**, run
`20260811T180751Z-task4-pool-default-serial`
(`.venv-bench/Scripts/python.exe benchmarks/harness/run_insight_report.py --run 20260811T180751Z-task4-pool-default-serial`,
judge/agent model `nvidia/nemotron-3-nano-30b-a3b:free`, no theme clustering
— see the script's docstring). The investigation agent produced 11 claims
(8 complaints, 3 strengths), `insights.verify` kept 11 of 12 raised (1
dropped as `not_supported`), and the result is committed at
`runs/20260811T180751Z-task4-pool-default-serial/report.json`.

Scoring that report with `score_groundedness.py`, however, could not
complete: OpenRouter's free-tier daily cap (`free-models-per-day`, 50
requests across all `:free`-suffixed models, account-wide) was already
exhausted before scoring started — confirmed via `GET
/api/v1/credits` returning `total_credits: 0`, so no paid fallback is
available either. Every one of the 11 claims was correctly recorded as
`ungrounded` / `llm_unavailable` by the scorer's fail-closed behaviour
(see `runs/20260811T180751Z-task4-pool-default-serial/groundedness.json`)
— **that `0.0` is a quota artifact demonstrating the fail-closed path
works, not a groundedness measurement.** Treat it as unscored, not as a
baseline of zero.

To establish the real baseline once the quota resets (`X-RateLimit-Reset`
on the 429 response was `2026-08-13T00:00:00Z`) or credits are added:

```bash
.venv-bench/Scripts/python.exe benchmarks/harness/score_groundedness.py --run 20260811T180751Z-task4-pool-default-serial
```

(`report.json` and `predictions.json` are already committed for this run —
no need to regenerate the report, only to re-run the scorer.)

---

## Comparing runs

Each run directory is self-contained and stamped with the git sha, package
versions, and a hash of the eval set. To compare a future pipeline against this
baseline, run both against the same eval set and diff `metrics_labeled.json`.
Re-use the same judgments file where the eval set is unchanged — the gold labels
do not depend on the pipeline.
