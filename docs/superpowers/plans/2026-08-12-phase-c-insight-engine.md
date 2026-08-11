# Phase C — Insight Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace templated, hardcoded "insights" with findings actually derived from review text — every claim traceable to the reviews it rests on, and any claim the text does not support dropped rather than softened.

**Architecture:** PyABSA output stays the aspect axis. Embeddings plus HDBSCAN add a theme axis that can express patterns the aspect taxonomy cannot. A bounded agent queries both through one retrieval surface, citing review ids; a fail-closed verifier re-checks each claim against its cited text before it may appear in a report.

**Tech Stack:** Python 3.11, sentence-transformers 3.0.1, scikit-learn 1.3.2 (HDBSCAN is built in), `mcp` 2.0.0, OpenRouter for LLM calls.

## Global Constraints

- **Interpreter:** `.venv-bench/Scripts/python.exe` (Python 3.11.9). Bare `python` may resolve to a broken path.
- **Two repositories.** `ABSA/` is a separate git repo on `fix/silent-fallback`; use `git -C ABSA`. Parent is on `benchmark/absa-baseline`; bump the ABSA gitlink there whenever `ABSA/` changes.
- **`import pyabsa` MUST precede `import pandas`** wherever both load. The guard lives in `ABSA/src/absa/__init__.py`; violating the order segfaults the interpreter on Windows (exit 139).
- **Dependency direction is one-way:** `insights/` may import `absa/`; `absa/` must never import `insights/`, `jobs/`, or the API layer. A guard test in `ABSA/tests/test_import_hygiene.py` enforces this — extend it, never weaken it.
- **DEPENDENCY INSTALL IS CONSTRAINED.** A bare `pip install sentence-transformers` upgrades `transformers` to 5.x and breaks PyABSA, silently degrading extraction to keyword matching. Verified working command:
  ```
  .venv-bench/Scripts/python.exe -m pip install "sentence-transformers" "transformers>=4.30,<4.37" "torch>=2.0,<2.2"
  ```
  This resolves to `sentence-transformers 3.0.1` and upgrades nothing. After installing, re-run `pip list | grep -E "transformers|torch"` and confirm `transformers 4.36.2` and `torch 2.1.2` are unchanged.
- **Load-bearing pins** in `ABSA/requirements.txt`, never relax: `update_checker<1.0`, `spacy>=3.7,<3.9`, plus the `en_core_web_sm` model.
- **Benchmark parity gate:** aspect F1 `0.746`, sentiment accuracy `0.873`. Extraction behaviour must not change in this phase except where LLM escalation deliberately repairs rows — and escalated rows carry their own provenance so they can be scored separately.
- **Provenance is sacred:** `extraction_method` and `degraded_reason` must survive every change. Escalated rows get `extraction_method = 'llm_escalated'`.
- **No LLM calls in unit tests.** Every test uses a stub. Real-model tests are explicitly marked and excluded from the default run.
- **Test baseline:** 148 tests pass at the start of this phase. The count only goes up.

---

### Task 1: Embeddings

Local model, no embedding API. Torch is already installed, it runs on CPU, and it keeps the no-external-services property the project chose deliberately.

**Files:**
- Create: `ABSA/src/insights/__init__.py`, `ABSA/src/insights/embed.py`
- Create: `ABSA/tests/test_embed.py`
- Modify: `ABSA/requirements.txt`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces:
  - `insights.embed.Embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2")`
  - `Embedder.encode(texts: list[str]) -> np.ndarray` — shape `(len(texts), 384)`, L2-normalised
  - `Embedder.is_available: bool` — False when the model could not load
  - `insights.embed.EmbeddingUnavailable` — raised by `encode` when unavailable

- [ ] **Step 1: Install the dependency under constraint**

```bash
.venv-bench/Scripts/python.exe -m pip install "sentence-transformers" "transformers>=4.30,<4.37" "torch>=2.0,<2.2"
```

Then **prove the pins survived** — this is the whole point of constraining:

```bash
.venv-bench/Scripts/python.exe -m pip list | grep -Ei "^(transformers|torch|sentence-transformers) "
```

Expected: `transformers 4.36.2`, `torch 2.1.2+cpu`, `sentence-transformers 3.0.1`. **If `transformers` moved to 5.x, stop immediately** — PyABSA is broken and extraction has silently degraded to keyword matching. Roll back and report BLOCKED.

Then confirm the extractor still works:

```bash
.venv-bench/Scripts/python.exe -m pytest ABSA/tests/ -q
```

Expected: 148 passing, unchanged.

- [ ] **Step 2: Write the failing test**

Create `ABSA/tests/test_embed.py`:

```python
"""Embeddings run locally on translated English text.

The model is ~90MB and downloads once. Tests that need real vectors are marked
`slow` and skipped by default so the suite stays fast.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from insights.embed import Embedder, EmbeddingUnavailable  # noqa: E402


def test_unavailable_embedder_raises_rather_than_returning_garbage():
    """A failed model load must not silently yield zero vectors -- downstream
    clustering would 'succeed' and produce meaningless themes."""
    embedder = Embedder.__new__(Embedder)
    embedder._model = None
    assert embedder.is_available is False
    with pytest.raises(EmbeddingUnavailable):
        embedder.encode(["anything"])


def test_empty_input_returns_empty_array_without_loading_a_model():
    embedder = Embedder.__new__(Embedder)
    embedder._model = None
    out = embedder.encode([])
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == 0


@pytest.mark.slow
def test_real_embeddings_are_normalised_and_semantically_sane():
    embedder = Embedder()
    if not embedder.is_available:
        pytest.skip("model unavailable offline")

    vectors = embedder.encode([
        "The battery life is terrible",
        "Battery drains far too quickly",
        "The delivery arrived on time",
    ])
    assert vectors.shape == (3, 384)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), "vectors must be L2-normalised"

    # Two battery complaints must be closer to each other than either is to
    # the delivery remark. If this fails, clustering cannot work.
    sim_battery = float(vectors[0] @ vectors[1])
    sim_unrelated = float(vectors[0] @ vectors[2])
    assert sim_battery > sim_unrelated + 0.15
```

Register the marker in `ABSA/pytest.ini` (create it if absent):

```ini
[pytest]
markers =
    slow: needs a real model download; excluded from the default run
addopts = -m "not slow"
```

- [ ] **Step 3: Run it to verify it fails**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/test_embed.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'insights'`

- [ ] **Step 4: Write the implementation**

Create `ABSA/src/insights/__init__.py` as an empty file.

Create `ABSA/src/insights/embed.py`:

```python
"""Sentence embeddings for theme discovery.

Local model rather than an embedding API: torch is already a dependency, it
runs fine on CPU, and it preserves the project's no-external-services
property. Embeddings are computed on TRANSLATED English text, so a Hindi and
an English review about the same thing land near each other.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class EmbeddingUnavailable(RuntimeError):
    """Raised when embeddings are requested but the model never loaded."""


class Embedder:
    """Encodes text to L2-normalised vectors."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model_name = model_name
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded: %s", model_name)
        except Exception as exc:  # noqa: BLE001
            # Degraded, not fatal: the aspect axis still works without themes.
            logger.warning(
                "Embedding model unavailable (%s: %s); theme discovery disabled",
                type(exc).__name__, exc,
            )

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        if self._model is None:
            raise EmbeddingUnavailable(
                f"embedding model {self._model_name!r} is not loaded; "
                "callers must check is_available and degrade explicitly"
            )
        return self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
```

- [ ] **Step 5: Run the tests**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/test_embed.py -v` — the two fast tests pass, the slow one is deselected.

Then run the slow one once explicitly to prove the model works:

```bash
.venv-bench/Scripts/python.exe -m pytest ABSA/tests/test_embed.py -v -m slow
```

Record the output in your report. If it downloads the model, that is expected on first run.

- [ ] **Step 6: Record the dependency and commit**

Add to `ABSA/requirements.txt`, beside the existing load-bearing pin comments:

```
# LOAD-BEARING: sentence-transformers must NOT pull transformers 5.x, which
# breaks PyABSA and silently degrades extraction to keyword matching. 3.0.1 is
# the newest release compatible with the transformers<4.37 pin above.
sentence-transformers>=3.0,<3.1
```

```bash
git -C ABSA add -A
git -C ABSA commit -m "feat: add local sentence embeddings for theme discovery"
```

---

### Task 2: Clustering

**Files:**
- Create: `ABSA/src/insights/cluster.py`, `ABSA/tests/test_cluster.py`

**Interfaces:**
- Consumes: `insights.embed.Embedder`
- Produces:
  - `insights.cluster.Cluster` — dataclass: `id: int`, `review_ids: list`, `size: int`, `representative_ids: list`
  - `insights.cluster.cluster_reviews(vectors, review_ids, min_cluster_size=3) -> list[Cluster]`
  - Noise points are returned as `Cluster(id=-1, ...)` — they are the one-off complaints nobody else made, which are interesting rather than garbage.

- [ ] **Step 1: Write the failing test**

Create `ABSA/tests/test_cluster.py`:

```python
"""Clustering turns vectors into themes.

HDBSCAN over KMeans: no guessed `k`, and it labels outliers explicitly rather
than forcing every review into a cluster.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from insights.cluster import Cluster, cluster_reviews  # noqa: E402


def _blob(centre, n, rng, spread=0.02):
    pts = rng.normal(centre, spread, size=(n, len(centre)))
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def test_two_separated_groups_become_two_clusters():
    rng = np.random.default_rng(0)
    vectors = np.vstack([_blob([1, 0, 0], 8, rng), _blob([0, 1, 0], 8, rng)])
    ids = list(range(16))

    clusters = cluster_reviews(vectors, ids, min_cluster_size=3)

    real = [c for c in clusters if c.id != -1]
    assert len(real) == 2
    assert sum(c.size for c in real) == 16


def test_every_review_id_appears_exactly_once():
    """No review may be lost or double-counted across clusters."""
    rng = np.random.default_rng(1)
    vectors = np.vstack([_blob([1, 0, 0], 6, rng), _blob([0, 1, 0], 6, rng)])
    ids = [f"r{i}" for i in range(12)]

    clusters = cluster_reviews(vectors, ids, min_cluster_size=3)

    seen = [rid for c in clusters for rid in c.review_ids]
    assert sorted(seen) == sorted(ids)


def test_representatives_are_drawn_from_the_cluster():
    rng = np.random.default_rng(2)
    vectors = _blob([1, 0, 0], 10, rng)
    ids = list(range(10))

    clusters = cluster_reviews(vectors, ids, min_cluster_size=3)

    for c in clusters:
        assert set(c.representative_ids) <= set(c.review_ids)
        assert len(c.representative_ids) <= len(c.review_ids)


def test_too_few_reviews_yields_a_single_noise_cluster_not_a_crash():
    rng = np.random.default_rng(3)
    vectors = _blob([1, 0, 0], 2, rng)
    clusters = cluster_reviews(vectors, [0, 1], min_cluster_size=3)
    assert sum(c.size for c in clusters) == 2


def test_empty_input_returns_no_clusters():
    assert cluster_reviews(np.empty((0, 384)), [], min_cluster_size=3) == []


def test_mismatched_lengths_raise():
    """A silent zip() truncation here would mis-attribute reviews to themes."""
    with pytest.raises(ValueError):
        cluster_reviews(np.zeros((3, 384)), [1, 2], min_cluster_size=3)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/test_cluster.py -v`
Expected: FAIL — no module `insights.cluster`.

- [ ] **Step 3: Write the implementation**

Create `ABSA/src/insights/cluster.py`:

```python
"""Group semantically similar reviews into themes.

HDBSCAN rather than KMeans: it needs no guessed cluster count and explicitly
labels outliers as noise (label -1). Those outliers are kept rather than
discarded -- a complaint nobody else made is often the most interesting thing
in a dataset.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

NOISE_LABEL = -1
MAX_REPRESENTATIVES = 5


@dataclass
class Cluster:
    id: int
    review_ids: list[Any]
    representative_ids: list[Any] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.review_ids)


def cluster_reviews(
    vectors: np.ndarray,
    review_ids: Sequence[Any],
    min_cluster_size: int = 3,
) -> list[Cluster]:
    """Cluster vectors, returning themes plus a noise group.

    Representatives are the points nearest their cluster's centroid -- the
    reviews that best express the theme, which is what the agent should read.
    """
    if len(review_ids) != vectors.shape[0]:
        raise ValueError(
            f"vectors and review_ids disagree: {vectors.shape[0]} vs {len(review_ids)}"
        )
    if len(review_ids) == 0:
        return []

    from sklearn.cluster import HDBSCAN

    effective_min = max(2, min(min_cluster_size, len(review_ids)))
    labels = HDBSCAN(min_cluster_size=effective_min, metric="euclidean").fit_predict(
        vectors
    )

    clusters: list[Cluster] = []
    for label in sorted(set(labels)):
        member_positions = [i for i, lab in enumerate(labels) if lab == label]
        members = [review_ids[i] for i in member_positions]

        if label == NOISE_LABEL:
            clusters.append(Cluster(id=NOISE_LABEL, review_ids=members,
                                    representative_ids=members[:MAX_REPRESENTATIVES]))
            continue

        centroid = vectors[member_positions].mean(axis=0)
        distances = np.linalg.norm(vectors[member_positions] - centroid, axis=1)
        nearest = np.argsort(distances)[:MAX_REPRESENTATIVES]
        clusters.append(
            Cluster(
                id=int(label),
                review_ids=members,
                representative_ids=[members[i] for i in nearest],
            )
        )

    logger.info(
        "Clustered %d reviews into %d themes (+%d noise)",
        len(review_ids),
        len([c for c in clusters if c.id != NOISE_LABEL]),
        sum(c.size for c in clusters if c.id == NOISE_LABEL),
    )
    return clusters
```

- [ ] **Step 4: Run the tests, then commit**

```bash
.venv-bench/Scripts/python.exe -m pytest ABSA/tests/ -v
git -C ABSA add -A
git -C ABSA commit -m "feat: cluster reviews into themes with HDBSCAN"
```

---

### Task 3: The retrieval surface

One source of truth for what the agent, the frontend, and the benchmark can ask. Plain Python so tests need no MCP transport.

**Files:**
- Create: `ABSA/src/insights/tools.py`, `ABSA/tests/test_tools.py`

**Interfaces:**
- Consumes: `insights.cluster.Cluster`
- Produces:
  - `insights.tools.InsightTools(processed_df, aspect_level_df, clusters, embedder=None)`
  - `.list_clusters() -> list[dict]` — id, size, dominant_sentiment, top_aspects
  - `.get_cluster_reviews(cluster_id, limit=5, sentiment=None) -> list[dict]` — each `{review_id, text, sentiment}`
  - `.get_reviews_for_aspect(aspect, sentiment=None, limit=10) -> list[dict]`
  - `.search_reviews(query, limit=10) -> list[dict]`
  - `.get_aspect_stats() -> dict`
  - `.get_extraction_health() -> dict` — counts by `extraction_method` / `degraded_reason`

- [ ] **Step 1: Write the failing test**

Create `ABSA/tests/test_tools.py`. Build a small fixture DataFrame with known content — at least 6 reviews, mixed sentiment, one Hindi-origin row, and one row with `extraction_method='none'` so health reporting has something to report. Cover:

- `list_clusters` returns one entry per cluster with correct sizes
- `get_cluster_reviews` returns only members of that cluster, respects `limit`, and every returned dict carries a `review_id`
- `get_cluster_reviews(sentiment='Negative')` returns only negative rows
- `get_reviews_for_aspect('battery')` returns only reviews mentioning battery
- an unknown aspect returns `[]` rather than raising
- an unknown cluster id returns `[]` rather than raising
- `get_extraction_health` reports the degraded row and the healthy ones separately
- **every returned review dict includes `review_id`** — assert this explicitly for each accessor, because citations are the foundation of the whole phase

- [ ] **Step 2: Run it to verify it fails, then implement**

Write `ABSA/src/insights/tools.py`. Requirements:

- Every accessor returns review ids alongside text. A tool that returns text without an id makes citation impossible, which makes verification impossible.
- No accessor raises on unknown input — return an empty list. The agent will probe for things that do not exist, and an exception ends its run.
- `search_reviews` uses the embedder when available; when it is not, fall back to case-insensitive substring matching and say so in the returned payload so the agent knows the retrieval was weaker.
- `get_extraction_health` returns at least: `total`, `by_method` (dict), `degraded_count`, `degraded_fraction`. The agent uses this to caveat conclusions drawn over partly-degraded data.

- [ ] **Step 3: Run the tests and commit**

---

### Task 4: LLM escalation

Rows PyABSA could not handle get a second look before anything aggregates them.

**Files:**
- Create: `ABSA/src/insights/llm.py`, `ABSA/src/insights/escalate.py`
- Create: `ABSA/tests/test_escalate.py`
- Modify: `ABSA/src/absa/config.py`

**Interfaces:**
- Consumes: `absa.config.get_settings`
- Produces:
  - `insights.llm.LLMClient(settings)` — `.complete(prompt: str, temperature: float = 0.0) -> str`, `.is_available: bool`
  - `insights.llm.LLMUnavailable`
  - `insights.escalate.escalate_weak_rows(df, client) -> tuple[pd.DataFrame, dict]` — returns the repaired frame plus a stats dict
  - Repaired rows get `extraction_method = 'llm_escalated'`

- [ ] **Step 1: Write the failing test**

`ABSA/tests/test_escalate.py`, using a stub client (never a real LLM):

- only rows with `extraction_method != 'pyabsa'` or empty aspects are sent to the LLM — assert the stub saw exactly those
- a healthy `pyabsa` row is never sent and is returned byte-identical
- a repaired row has `extraction_method == 'llm_escalated'` and `degraded_reason is None`
- **a malformed LLM response leaves the row degraded rather than inventing aspects** — this is the fail-closed rule and must be tested
- when the client is unavailable, every row passes through untouched and the stats dict says so
- the returned stats report how many rows were attempted, repaired, and left degraded

- [ ] **Step 2: Implement**

`llm.py` wraps OpenRouter using `settings.llm_api_key` and `settings.llm_model`, with a timeout and a single retry. `is_available` is False when no key is configured — and `escalate_weak_rows` must then be a no-op, not an error.

`escalate.py` prompts for aspects and sentiments on one review at a time, parses strict JSON, and **rejects anything that does not parse or does not match the expected shape**, leaving the row degraded. Never fabricate.

- [ ] **Step 3: Tests, then commit**

---

### Task 5: The agent

**Files:**
- Create: `ABSA/src/insights/agent.py`, `ABSA/tests/test_agent.py`

**Interfaces:**
- Consumes: `insights.tools.InsightTools`, `insights.llm.LLMClient`
- Produces:
  - `insights.agent.Claim` — dataclass: `text: str`, `kind: str` (`finding` | `complaint` | `strength` | `action`), `review_ids: list`
  - `insights.agent.InvestigationBudget` — `max_tool_calls: int = 20`, `max_seconds: float = 120.0`
  - `insights.agent.investigate(tools, client, budget) -> tuple[list[Claim], dict]` — claims plus a run-stats dict

- [ ] **Step 1: Write the failing test**

`ABSA/tests/test_agent.py` with a scripted stub client that returns a fixed sequence of tool calls and then claims. Cover:

- the agent calls tools and produces claims carrying `review_ids`
- **a claim with no `review_ids` is discarded before it leaves the agent** — uncited claims cannot be verified, so they must not exist
- the tool-call budget is enforced: a stub that always asks for another tool call stops at `max_tool_calls`
- the wall-clock budget is enforced
- an unparseable model response ends the run with whatever was already collected, rather than raising
- the stats dict reports tool calls made and whether a budget was hit

- [ ] **Step 2: Implement**

A loop: give the model the cluster list, aspect stats, and extraction health; let it request tools; feed results back; collect claims. Hard-stop on either budget. Every claim must carry ids or be dropped.

- [ ] **Step 3: Tests, then commit**

---

### Task 6: The verifier

**Files:**
- Create: `ABSA/src/insights/verify.py`, `ABSA/tests/test_verify.py`

**Interfaces:**
- Consumes: `insights.agent.Claim`, `insights.tools.InsightTools`, `insights.llm.LLMClient`
- Produces:
  - `insights.verify.VerifiedClaim` — `claim: Claim`, `supported: bool`, `reason: str`
  - `insights.verify.verify_claims(claims, tools, client) -> tuple[list[VerifiedClaim], dict]`

- [ ] **Step 1: Write the failing test**

The rule under test is **drop, do not soften**. Cover:

- a claim whose cited reviews support it is kept
- a claim whose cited reviews do not support it is **absent from the kept list** — not reworded, not hedged
- a claim citing a review id that does not exist is dropped
- verification runs at temperature 0
- when the client is unavailable, **no claim is marked supported** (fail closed — an unverifiable claim is not a verified claim)
- the stats dict reports kept and dropped counts

- [ ] **Step 2: Implement, test, commit**

---

### Task 7: The report, and rendering it

**Files:**
- Create: `ABSA/src/insights/report.py`, `ABSA/tests/test_report.py`
- Modify: `ABSA/src/absa/pipeline.py` or `ABSA/app.py` to expose the report
- Modify: `streamlit-deployment/app_a.py`

**Interfaces:**
- Produces:
  - `insights.report.Report` — `findings`, `complaints`, `strengths`, `actions` (each `list[VerifiedClaim]`), plus `stats: dict` and `caveats: list[str]`
  - `insights.report.build_report(verified, tools, run_stats) -> Report`

- [ ] **Step 1: Write the failing test**

- claims are routed to the correct section by `kind`
- **every claim in every section carries at least one review id**
- when a meaningful share of rows are degraded, a caveat naming that appears in `caveats`
- when no claims survive verification, the report is *empty with an explanation*, never filled with placeholder prose
- `stats` reports how many claims were dropped in verification

- [ ] **Step 2: Implement `report.py`**

- [ ] **Step 3: Render it in Streamlit**

Add a section to `streamlit-deployment/app_a.py` showing narrative, findings with expandable citations (clicking a finding reveals the review text it rests on), and action items. This is what makes the phase demonstrable rather than theoretical.

**Do not delete any existing dashboard section.** This adds a view.

- [ ] **Step 4: Verify end to end with a real run, then commit**

Start the backend, upload one of `streamlit-deployment/test_data_*.csv`, and confirm a report renders with citations that expand to real review text. Capture what you saw in your report.

---

### Task 8: MCP server

**Files:**
- Create: `ABSA/src/insights/mcp_server.py`, `ABSA/tests/test_mcp_server.py`
- Modify: `ABSA/requirements.txt`

**Interfaces:**
- Consumes: `insights.tools.InsightTools`
- Produces: an MCP server exposing exactly the `InsightTools` accessors

- [ ] **Step 1: Install `mcp`**

```bash
.venv-bench/Scripts/python.exe -m pip install "mcp"
```

Verified to resolve to `mcp 2.0.0` with no upgrades to existing packages. Re-check the pins afterwards exactly as in Task 1, and re-run the suite.

- [ ] **Step 2: Implement the adapter**

Thin: each MCP tool calls the matching `InsightTools` method and returns its result. **No logic lives here** — that is what keeps `tools.py` testable without a transport.

- [ ] **Step 3: Test that each exposed tool maps to a real method**

A test asserting the MCP tool list matches `InsightTools`' public accessors, so the two cannot drift apart silently.

- [ ] **Step 4: Commit**

---

### Task 9: Groundedness metric

The insight layer's analogue of F1. Without it, prompt edits can quietly degrade honesty and nothing will notice.

**Files:**
- Create: `benchmarks/harness/score_groundedness.py`
- Modify: `benchmarks/README.md`

- [ ] **Step 1: Implement the scorer**

For a completed run: for each claim, fetch its cited reviews and record whether the cited text supports it. Emit `groundedness.json` with per-claim verdicts and an overall supported fraction.

- [ ] **Step 2: Establish the baseline**

Run it over the eval set, record the number, and write it into `benchmarks/README.md` beside the existing F1 and sentiment-accuracy baselines so future changes have something to regress against.

- [ ] **Step 3: Commit both repos**
