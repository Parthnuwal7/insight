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

---

## 6. Scored results (judge labels applied)

All 46 reviews judged, none missing. Full detail in `runs/20260808T121303Z-pyabsa-working/triage.md`.

| Metric | Value |
|---|---:|
| Aspect F1 (fuzzy) | **0.713** |
| Aspect precision | 0.795 |
| Aspect recall | 0.646 |
| Sentiment accuracy (matched aspects) | **0.839** |
| Coverage | 78 predicted / 96 gold (0.81) |

**Recall is the weak axis, not precision.** The pipeline misses roughly a third
of the aspects reviewers actually evaluate (34 false negatives vs 16 false
positives). When it does name an aspect it is usually right; it simply does not
name enough of them.

### The fallback rows are measurably much worse

| Route | Reviews | Aspect F1 | Precision | Recall | Sentiment acc |
|---|---:|---:|---:|---:|---:|
| `pyabsa` | 38 | **0.773** | 0.853 | 0.707 | **0.862** |
| `pyabsa_empty_to_fallback` | 8 | **0.333** | 0.400 | 0.286 | **0.500** |

Fallback rows score less than half the F1 and their sentiment is a coin flip.
Holding everything else fixed, if those 8 reviews performed at the `pyabsa` rate
the overall F1 would rise from **0.713 to 0.773** — a 6-point gain from deleting
one silent code path, with no model change.

### The fallback does not merely miss — it inverts

Two of the F1-zero reviews were labelled **Positive** by the keyword scorer when
the gold is unambiguously negative:

> *"Product stopped working after a month. Tried to return but process is complicated."*
> → `General: Positive`

> *"Used to be great but recent updates made it worse. Bring back the old version!"*
> → `General: Positive`

Both are explained by the wordlists in `_get_rule_based_sentiment`:

- `'working'` is in `positive_words`, so **"stopped working" scores positive**.
- `'worse'` is absent from `negative_words` (only `'worst'` is present), so
  *"made it worse"* contributes nothing and *"great"* wins.

Substring matching against a hand-written wordlist produces confidently wrong
polarity on plainly negative text.

### Sentiment error is asymmetric — the model leans negative

| Gold | n | Errors | Error rate |
|---|---:|---:|---:|
| Positive | 25 | 7 | **28.0%** |
| Negative | 35 | 3 | 8.6% |
| Neutral | 2 | 0 | 0.0% |

Positives are misread more than three times as often as negatives (5 flipped to
Negative, 2 to Neutral). This has direct business consequence: the *Areas of
Improvement* table is systematically inflated and *Strength Anchors* is
systematically understated. The dashboard's core recommendation is biased
pessimistic.

### Extraction vs polarity: the split the metrics were built to expose

| Category | n | Aspect F1 | Sentiment acc | Diagnosis |
|---|---:|---:|---:|---|
| `negation` | 3 | **1.000** | **0.500** | Finds every aspect, reads polarity wrong |
| `sarcasm` | 3 | 0.667 | **0.333** | Mostly finds aspects, polarity badly wrong |
| `hinglish` | 3 | **0.909** | 0.600 | Extraction excellent, polarity mediocre |
| `mixed_sentiment` | 11 | 0.739 | **1.000** | Polarity perfect, misses aspects |
| `multi_aspect` | 6 | 0.812 | **1.000** | Polarity perfect, misses aspects |
| `out_of_taxonomy` | 11 | 0.615 | 0.917 | Recall 0.52 — misses over half |
| `hindi` | 3 | 0.545 | 0.667 | Devanagari genuinely weak |
| `implicit_aspect` | 3 | **0.250** | **0.000** | Fails completely |
| `comparative` | 1 | 0.000 | n/a | n=1, no signal |
| `single_aspect_control` | 2 | 1.000 | 1.000 | Controls pass |

These are two different failures needing two different fixes:

- **Polarity failures** (`negation`, `sarcasm`) have *perfect or near-perfect
  extraction*. A canonicalisation layer or better aspect model would not touch
  them. They need a polarity model that handles scope and intent.
- **Recall failures** (`mixed_sentiment`, `multi_aspect`, `out_of_taxonomy`) have
  *perfect polarity*. The model judges correctly whatever it finds; it just
  finds too little.

**`implicit_aspect` fails on both axes** (F1 0.25, sentiment 0.00) and is the
clearest argument for an LLM-based extractor. The pipeline reports the literal
noun rather than the thing being evaluated:

> *"I had to ask twice for water and once for the bill."*
> → `water: Negative`, `bill: Neutral`. Gold: `service: Negative`.

**Devanagari underperforms romanized Hindi.** `hindi` scores F1 0.545 / recall
0.429 while `hinglish` scores 0.909. Two of three Devanagari reviews took the
empty-to-fallback route. With translation inert, the multilingual checkpoint
handles romanized Hindi unaided but struggles with Devanagari script — so fixing
translation matters more for Devanagari than for code-mix.

### What the numbers say about sequencing (superseded by §7 — see below)

Roughly 6 F1 points are recoverable with **no ML work at all** (removing the
silent fallback), plus the wordlist inversions and the case-only fragmentation.
The remaining gap — recall on multi-aspect reviews, implicit aspects, negation
and sarcasm polarity — is the genuine ML case and is where an LLM-based
extractor would earn its cost.

---

## 7. After the cheap fixes — measured, not predicted

Run `20260808T220751Z-cheap-fixes`, same eval set, same judge labels.

**I predicted F1 would rise from 0.713 to 0.773. It did not.** That estimate
assumed the 8 fallback reviews would start producing ABSA-quality aspects.
They don't — they now produce *no* aspects, so recall fell instead.

| Metric | Before | After | Change |
|---|---:|---:|:--|
| Aspect F1 | 0.713 | 0.707 | −0.006 |
| **Precision** | 0.795 | **0.853** | **+0.058** |
| Recall | 0.646 | 0.604 | −0.042 |
| **Sentiment accuracy** | 0.839 | **0.862** | **+0.023** |
| Keyword rows in output | 12.8% | **0%** | — |
| Rows with fake `0.7` confidence | 10 | **0** | — |

The fixes **removed wrong data rather than adding right data**. Every aspect the
system now reports is more likely to be correct, and its polarity more likely to
be right. Nothing was done to make it find more.

### F1 is the wrong scoreboard for this change

F1 penalises a missing aspect and a wrong aspect equally. For this product they
are not equal: a wrong aspect becomes a phantom row in *Areas of Improvement*
that somebody acts on, while a missing one is a gap. Precision up 5.8 points and
sentiment up 2.3 points is the trade that matters, and it is invisible in F1.

### Regression: Devanagari now returns nothing

All 8 reviews that now return zero aspects, by category:

| Category | n | Note |
|---|---:|---|
| `hindi` | 3 | **All Devanagari reviews in the set** |
| `out_of_taxonomy` | 3 | Return process, privacy, export — no keyword bucket |
| `comparative` | 1 | |
| `sarcasm` | 1 | |

`hindi` went from F1 0.545 to **0.000**. PyABSA finds no aspects in Devanagari,
and translation is inert because `HF_TOKEN` is unset — so Hindi reviews now
produce no output at all.

This is not a fluke of removing the fallback: the keyword taxonomy contains
Devanagari terms (`गुणवत्ता`, `डिलीवरी`, `सेवा`), so it was genuinely carrying
Hindi coverage. Removing it exposed that **Hindi was only ever working by
accident, through keyword matching, not through ABSA.**

Two ways forward, and this is a product call:

1. **Set `HF_TOKEN` and fix translation.** Hindi becomes English before
   extraction, and the multilingual model handles it. This is the real fix, and
   it also explains why romanized Hinglish scores 0.909 while Devanagari scores
   0.000 — Hinglish never needed translation.
2. **Re-enable the fallback for now** via `ABSA_ALLOW_KEYWORD_FALLBACK=true`,
   accepting inverted polarity on some English text in exchange for coarse Hindi
   coverage.

Option 1 is correct. Option 2 is available if Hindi content is in production
today and cannot go dark while translation is fixed.

### What did improve, that the benchmark cannot see

Canonicalization affects aggregation, not extraction, so it does not move F1 —
the scorer compares raw surface forms. Verified separately by unit test:
`Delivery`, `delivery`, and `the delivery` now produce **one** ranking entry with
frequency 3, where before they produced three entries of frequency 1 and each
ranked too low to surface. Co-occurrence edges accumulate across variants
instead of splitting below the weight-2 threshold.

### Honest summary

| Claim | Verdict |
|---|---|
| Silent fallback removed | ✅ 0% keyword rows, provenance on every row |
| Pipeline fails loudly | ✅ `/health` reports `degraded`, was always wrong before |
| Polarity inversions fixed | ✅ "stopped working" now Negative |
| Aspect fragmentation fixed | ✅ verified by test, invisible to F1 |
| **Accuracy improved** | ⚠️ **precision and sentiment yes; F1 no; recall worse** |
| Hindi | ❌ **regressed to zero — needs `HF_TOKEN`** |

The remaining gap is unchanged and is the real ML work: recall on multi-aspect
reviews, implicit aspects, and negation/sarcasm polarity.

---

## 8. After fixing translation

Run `20260809T201004Z-translation-fixed`. `HF_TOKEN` was supplied, which exposed
three further faults — the token alone changed nothing.

**a. The endpoint no longer exists.** `_call_hf_translation_api` hardcoded
`https://api-inference.huggingface.co/models/{model}`. That host has been retired
and fails at DNS resolution. The bare `except` caught the `ConnectionError` and
logged it at DEBUG, so translation had been dead regardless of credentials.
Tellingly, `__init__` already set `base_url` to `router.huggingface.co` — the
correct current host — and nothing used it. A migration was started and never
finished.

**b. The model was wrong in two ways.** `ai4bharat/indictrans2-en-indic-1.3B`
translates English *into* Indic languages; it was being used for Hindi→English.
It is also not served: both IndicTrans2 directions return
`"Model not supported by provider hf-inference"`. The call could never have
succeeded. Replaced with `Helsinki-NLP/opus-mt-hi-en`, which is the right
direction and is served.

**c. Multi-sentence input was truncated.** opus-mt translates only the first
sentence and drops the rest:

> `"यह उत्पाद बहुत अच्छा है। गुणवत्ता शानदार है और डिलीवरी भी तेज़ थी।"`
> → `"This product is very good."`

The quality and delivery clauses vanished — two aspects the extractor then had
no chance of seeing. Translation is now sentence-by-sentence with a cache.

### Results

| Metric | Baseline | Cheap fixes | **+ translation** |
|---|---:|---:|---:|
| Aspect F1 | 0.713 | 0.707 | **0.746** |
| Precision | 0.795 | 0.853 | **0.863** |
| Recall | 0.646 | 0.604 | **0.656** |
| Sentiment accuracy | 0.839 | 0.862 | **0.873** |
| Keyword rows | 12.8% | 0% | **0%** |
| Reviews yielding nothing | 0 | 8 | **5** |

The Hindi regression is resolved and overtakes the original baseline:

| | Baseline | Cheap fixes | + translation |
|---|---:|---:|---:|
| `hindi` F1 | 0.545 | 0.000 | **0.833** |
| `hindi` sentiment | 0.667 | n/a | **1.000** |

This also confirms the earlier read: Devanagari was never handled by ABSA, only
by Devanagari keywords in the fallback. With real translation it now routes
through the model and scores better than it ever did.

Sentiment bias is reduced but not gone — Positive error 28.0% → **22.2%**,
Negative 8.6% → **5.9%**. The pipeline still reads complaints more reliably than
compliments, so *Areas of Improvement* remains somewhat inflated relative to
*Strength Anchors*.

### What is still broken

| Problem | Evidence | Fixable without ML? |
|---|---|---|
| Reports literal nouns, not opinion targets | `implicit_aspect` F1 0.250, sentiment 0.000 | No |
| Sarcasm | F1 0.500, sentiment 0.500 | No |
| Negation polarity | extraction F1 1.000, sentiment 0.500 | No |
| Recall on aspects outside common vocabulary | `out_of_taxonomy` recall 0.522 | No |
| 5 reviews still yield nothing | privacy, export, return process, comparative, sarcasm | No |
| Hinglish polarity | extraction 0.909, sentiment 0.600 | Partly — `langdetect` tags Hinglish as English, so it is never translated |

Every remaining item is model capability. The ops and data-hygiene work is done.

### Harness caveat

`api_called` now under-reports in `metrics_unlabeled.json`: the recorder keys on
the text passed to `_call_hf_translation_api`, which is a sentence since the
sentence-splitting change, while `translate_to_english` receives the whole
review. `text_changed` is still accurate and is the metric that matters.
