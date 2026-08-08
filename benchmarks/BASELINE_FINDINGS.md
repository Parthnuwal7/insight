# Baseline findings

Two runs against the same 46-review evaluation set, differing only in whether
PyABSA was functional. Both are persisted under `runs/`.

| | Run A | Run B |
|---|---|---|
| id | `20260807T202423Z-baseline` | `20260808T121303Z-pyabsa-working` |
| PyABSA | broken (dependency drift) | working |
| Real ABSA rows | **0%** | **82.6%** |
| Keyword-fallback rows | **100%** | 17.4% |
| Wall clock | 0.5s (0.01s/review) | 78.1s (1.7s/review) |

Run A was not contrived. It is what a clean `pip install` of
`ABSA/requirements.txt` produced on 2026-08-08.

---

## 1. The pipeline fails silently, and it is easy to reach

Three independent faults each cause the pipeline to emit keyword-bucket output
while reporting success. None raises. All three are invisible downstream.

**a. Unpinned transitive dependency breaks `import pyabsa`.**
`pyabsa → metric_visualizer → update_checker`. `update_checker 1.0.0` changed
`UpdateChecker.check()` to take no arguments; `metric_visualizer 0.9.17` still
calls `checker.check(__name__, __version__)`. Result: `TypeError` at
`import pyabsa`, caught by the bare `except Exception` in
`_load_pyabsa_model()`, logged, swallowed. `self.model = None`, service healthy,
100% of reviews take the fallback.

No code change is required to trigger this — only time. None of these transitive
dependencies are pinned.

**b. Missing spaCy model breaks every prediction.**
PyABSA's polarity path calls `spacy.load("en_core_web_sm")`. That model is not in
`ABSA/requirements.txt`. Missing, it raises `OSError [E050]` on *every* review,
caught by the per-review `except` at `data_processor.py:337`. The checkpoint
loads, `/health` reports `absa_service: available`, and output is 100% keyword
buckets.

**c. Translation is inert.** `HF_TOKEN` is absent from `ABSA/.env`, so
`_call_hf_translation_api` returns the input unchanged before making any request.
Measured translation effectiveness: **0 of 6** non-English reviews had their text
altered, in both runs.

`logs.md` in the repo root records a *fourth* variant of the same class
(`ImportError: cannot import name 'checkpoint_utils'`) hitting the same bare
`except`, which is evidence this has already happened in the deployed
environment.

### This is directly detectable, and worth fixing first

Fallback rows are identifiable from the API payload alone: confidence is exactly
`0.7` and the aspect is one of 14 known strings. The harness scored that
heuristic against instrumented ground truth in Run B:

**precision 100%, recall 100%** (TP 8, FP 0, FN 0, TN 38).

So a provenance field could be added to the API response and surfaced in the
dashboard with no ML work at all. Until that exists, no one — including you — can
tell from the output whether they are looking at ABSA or at keyword matching.

---

## 2. With PyABSA healthy, 17.4% of reviews still fall back

Run B: 38 reviews took the `pyabsa` route, **8 took `pyabsa_empty_to_fallback`** —
PyABSA ran, returned zero aspects, and `_extract_with_pyabsa` quietly delegated to
the keyword extractor (`data_processor.py:370-371`).

This is a distinct problem from (1). Even a perfectly deployed model produces a
table blending two labelling schemes. Any aggregate computed over it — the
priority ranking, the co-occurrence graph, the heatmaps — mixes model output with
keyword output.

Confidence distribution makes the blend visible: mean `0.894`, median `0.977`,
range `0.499–0.999`, with exactly **10 rows at precisely `0.7`** — the hardcoded
fallback constant, and exactly the fallback row count.

---

## 3. Aspect fragmentation is real and concentrated

Run B, PyABSA rows only: **51 surface forms → 44 concepts (1.16x)**. The headline
ratio is mild, but fragmentation is concentrated in the aspects that matter most:

| Concept | Forms | Surface variants |
|---|---:|---|
| service | 5 | `Customer service`, `Service`, `Service staff`, `customer service`, `service` |
| battery | 4 | `battery`, `battery backup`, `battery drain`, `battery life` |
| performance | 3 | `App Performance`, `Performance`, `performance` |
| atmosphere | 2 | `Atmosphere`, `atmosphere` |
| delivery | 2 | `Delivery`, `delivery` |

Note that **case alone** splits `Delivery`/`delivery` and `Atmosphere`/`atmosphere`.
A large share of the fragmentation needs no ML to fix — casefolding and lemmatising
would collapse it. `service` at 5 forms means the single most business-relevant
aspect in restaurant data is split five ways across every ranking.

> **Caveat.** The `food` cluster reported by the tool wrongly absorbs
> `sound quality`, because `Quality` is a token-subset of both `Food quality` and
> `sound quality` and the clustering is transitive. The fragmentation ratio is
> therefore a slight over-estimate. See "Known limitations" in `README.md`.

---

## 4. Where the model is actually weak

From `review_level.csv`, Run B. These are the qualitative failures behind the
"sentiment is just wrong" and "aspects get missed" symptoms.

**Negation is not handled.**

> *"The battery does not drain quickly at all, and the screen isn't dim even outdoors."*
> → `battery: Negative`, `screen: Negative`. Both should be Positive.

**Sarcasm is not handled.**

> *"The battery life is fantastic if you enjoy charging twice before lunch."*
> → `battery life: Positive`. Should be Negative.

> *"Brilliant. Waited three weeks for a cable that arrived snapped in half. Outstanding work."*
> → `cable: Negative` (correct), `work: Positive` (sarcasm missed, and `work` is spurious).

**It extracts surface nouns rather than evaluated aspects.** This is the deepest
issue, and it explains "aspects get missed entirely":

> *"I had to ask twice for water and once for the bill."*
> → `water: Negative`, `bill: Neutral`. The actual aspect is **service**, which never appears.

> *"Everyone in the room could hear my private call through the speaker."*
> → `room: Neutral`, `speaker: Positive`. Should be a negative judgment on audio privacy.

Spurious aspects observed: `work`, `lunch`, `room`, `water`, `bill`. These are
nouns, not opinion targets — a precision problem that no amount of
canonicalisation fixes.

**Hinglish works better than expected.** The multilingual checkpoint handled
romanized Hindi without translation: `Delivery: Negative` correct,
`Service staff: Negative` correct. One error — `quality: Negative` where
*"quality ekdum mast hai"* is strongly positive. Given translation is inert
anyway, the multilingual model is carrying this case on its own.

**Controls both passed**, confirming the harness is not manufacturing failure:
`screen: Positive` ✓, `Delivery: Negative` ✓.

**`langdetect` misfires on code-mix.** One Hinglish review was detected as
Indonesian (`id`), two as English. None would ever reach translation even with a
working token.

---

## 5. What this changes about the overhaul

The accuracy problem is **not one problem**, and the cheap half is not ML:

| Layer | Fix | Needs ML work? |
|---|---|---|
| Silent fallback (100% of output at risk) | Pin deps, add `en_core_web_sm`, fail loudly, expose provenance in the API | No |
| 17.4% empty-result fallback | Stop delegating to keyword buckets; emit "no aspects found" honestly | No |
| Case/lemma fragmentation | Canonicalise before aggregation | No |
| Negation and sarcasm | Better model or LLM extraction | Yes |
| Surface nouns instead of opinion targets | Better model or LLM extraction | Yes |

Rows 1–3 are ops and data-hygiene work that would measurably improve every
dashboard number without touching the model. Rows 4–5 are the genuine ML case,
and they are the ones that justify considering an LLM-based extractor.

**Still unmeasured:** aspect precision/recall/F1 and sentiment accuracy. Those
need the judge labels. Run the packet in
`runs/20260808T121303Z-pyabsa-working/judge_packet.md` and score it to fill in
the rest.
