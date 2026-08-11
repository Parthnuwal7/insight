# Phase A — Decouple and Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the ABSA backend a pure Python library with no UI framework, no external service dependencies, and batched inference — so Phases B, C, and D can build on it.

**Architecture:** Delete first, then restructure. Remove Redis, MongoDB, the admin/telemetry surface, and the Streamlit subprocess launcher; then introduce `Settings` and `ProgressReporter` to replace `st.secrets` and `st.spinner`; then split the 1,150-line `data_processor.py` into a focused `absa/` package; finally batch PyABSA inference, gated on benchmark parity.

**Tech Stack:** Python 3.11, FastAPI, PyABSA 2.4.2, pytest 9.1.1. No new dependencies in this phase.

## Global Constraints

- **Interpreter:** `.venv-bench/Scripts/python.exe` (Python 3.11.9). The root `venv/` no longer exists; bare `python` may resolve to a broken path.
- **`ABSA/` is a separate git repository** on branch `fix/silent-fallback`, tracked in the parent as a gitlink with no `.gitmodules`. Commits touching `ABSA/**` must be made with `git -C ABSA`. Commits touching `streamlit-deployment/**`, `benchmarks/**`, or `docs/**` are made in the parent repo.
- **Import order is load-bearing:** `import pyabsa` MUST precede `import pandas` in any module or script that touches both, or the interpreter segfaults on Windows (exit 139). See `benchmarks/harness/run_benchmark.py:17-30` and `ABSA/app.py`.
- **Benchmark parity is the gate:** aspect F1 `0.746`, sentiment accuracy `0.873` on `benchmarks/eval_set/eval_reviews_v1.csv`. Any task that moves either number does not land.
- **Load-bearing dependency pins** (from `ABSA/requirements.txt`, do not relax): `update_checker<1.0`, `spacy>=3.7,<3.9`, and the `en_core_web_sm` model must remain installed.
- **No new third-party dependencies in Phase A.**
- **Provenance fields are preserved everywhere:** `extraction_method` and `degraded_reason` must survive every refactor.

---

### Task 1: Streamlit import-hygiene gate

Establishes the test that drives the whole phase. It passes immediately via an explicit allowlist of modules that still import Streamlit; every later task shrinks that allowlist, so the suite stays green at every commit and the allowlist documents remaining debt.

**Files:**
- Create: `ABSA/tests/test_import_hygiene.py`

**Interfaces:**
- Consumes: nothing
- Produces: `STREAMLIT_ALLOWLIST: set[str]` in `ABSA/tests/test_import_hygiene.py`, a set of `src`-relative POSIX paths. Later tasks remove entries from it.

- [ ] **Step 1: Write the failing test**

Create `ABSA/tests/test_import_hygiene.py`:

```python
"""The backend must not depend on a UI framework.

STREAMLIT_ALLOWLIST is the remaining debt. Entries are removed as modules are
cleaned; when it is empty, delete it and the skip branch below.
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"

STREAMLIT_ALLOWLIST = {
    "streamlit_app.py",
    "components/visualizations.py",
    "utils/frontend_helpers.py",
    "utils/data_management.py",
    "utils/data_processor.py",
    "utils/admin_endpoints.py",
}


def _modules():
    for path in sorted(SRC.rglob("*.py")):
        yield path.relative_to(SRC).as_posix(), path


def test_no_streamlit_outside_allowlist():
    offenders = []
    for rel, path in _modules():
        if rel in STREAMLIT_ALLOWLIST:
            continue
        if "streamlit" in path.read_text(encoding="utf-8"):
            offenders.append(rel)
    assert offenders == [], f"streamlit imported outside allowlist: {offenders}"


def test_allowlist_has_no_stale_entries():
    """An allowlist entry for a clean or deleted file is misleading."""
    stale = []
    for rel in sorted(STREAMLIT_ALLOWLIST):
        path = SRC / rel
        if not path.exists() or "streamlit" not in path.read_text(encoding="utf-8"):
            stale.append(rel)
    assert stale == [], f"allowlist entries no longer need to be listed: {stale}"
```

- [ ] **Step 2: Run the test to see the current state**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/test_import_hygiene.py -v`

Expected: both tests PASS. If `test_no_streamlit_outside_allowlist` fails, add the reported paths to `STREAMLIT_ALLOWLIST`. If `test_allowlist_has_no_stale_entries` fails, remove the reported paths from it. The allowlist must exactly match reality before committing.

- [ ] **Step 3: Commit**

```bash
git -C ABSA add tests/test_import_hygiene.py
git -C ABSA commit -m "test: gate streamlit imports in backend with shrinking allowlist"
```

---

### Task 2: Stop the backend launching a UI

`ABSA/app.py:517-534` searches for a Streamlit app file and spawns `streamlit run` as a subprocess. That is why the deployed Space runs a second UI on port 8502. A library backend must not launch a UI.

**Files:**
- Modify: `ABSA/app.py` (the `__main__` block and its Streamlit launcher)
- Delete: `ABSA/src/streamlit_app.py`, `ABSA/src/components/visualizations.py`, `ABSA/src/utils/frontend_helpers.py`, `ABSA/src/utils/data_management.py`
- Modify: `ABSA/tests/test_import_hygiene.py`

**Interfaces:**
- Consumes: `STREAMLIT_ALLOWLIST` from Task 1
- Produces: nothing new

- [ ] **Step 1: Prove the deletion targets are unreferenced**

Run:

```bash
grep -rn "streamlit_app\|visualizations\|data_management\|frontend_helpers" ABSA/app.py ABSA/api_server.py ABSA/src --include="*.py" \
  | grep -v "^ABSA/src/streamlit_app.py:" \
  | grep -v "^ABSA/src/components/visualizations.py:" \
  | grep -v "^ABSA/src/utils/frontend_helpers.py:" \
  | grep -v "^ABSA/src/utils/data_management.py:"
```

Expected: only matches inside `ABSA/app.py` referring to the subprocess launcher (the `streamlit_app` local variable around lines 517-534). If anything else appears, stop and reassess — something still imports these modules.

- [ ] **Step 2: Read the launcher block before editing**

Run: `.venv-bench/Scripts/python.exe -c "print(open(r'ABSA/app.py').read()[:0])"` then open `ABSA/app.py` and read lines 505-545. Identify the full `if __name__ == "__main__":` block, the helper that locates a Streamlit file, and the `subprocess` call that runs it.

- [ ] **Step 3: Replace the `__main__` block**

Replace the entire Streamlit-launching `__main__` block in `ABSA/app.py` with:

```python
if __name__ == "__main__":
    import logging

    import uvicorn

    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info("Starting FastAPI server on http://0.0.0.0:7860")
    uvicorn.run(app, host="0.0.0.0", port=7860)
```

Then remove the now-unused `import subprocess` and `from threading import Thread` from the top of `ABSA/app.py` if nothing else uses them. Verify with:

```bash
grep -n "subprocess\|Thread(" ABSA/app.py
```

Expected: no matches, or only matches unrelated to the launcher (the git-availability check in `data_processor.py` is a different file and stays).

- [ ] **Step 4: Delete the unreferenced UI modules**

```bash
git -C ABSA rm src/streamlit_app.py src/components/visualizations.py src/utils/frontend_helpers.py src/utils/data_management.py
```

- [ ] **Step 5: Shrink the allowlist**

In `ABSA/tests/test_import_hygiene.py`, remove these four entries from `STREAMLIT_ALLOWLIST`:

```python
    "streamlit_app.py",
    "components/visualizations.py",
    "utils/frontend_helpers.py",
    "utils/data_management.py",
```

Leaving:

```python
STREAMLIT_ALLOWLIST = {
    "utils/data_processor.py",
    "utils/admin_endpoints.py",
}
```

- [ ] **Step 6: Run the tests**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/ -v`

Expected: all PASS, including both hygiene tests.

- [ ] **Step 7: Verify the server still starts**

Run from the `ABSA` directory: `../.venv-bench/Scripts/python.exe -m uvicorn app:app --host 127.0.0.1 --port 7860`

In another shell: `curl -s http://127.0.0.1:7860/`
Expected: `{"message":"ABSA ML Backend API","status":"running"}` and **no** Streamlit process on port 8502. Stop the server.

- [ ] **Step 8: Commit**

```bash
git -C ABSA add -A
git -C ABSA commit -m "refactor: stop backend spawning a streamlit UI, delete unreferenced UI modules"
```

---

### Task 3: Remove Redis

Redis is gone as a dependency. Rate limiting disappears with it, which costs nothing: `redis_service.check_rate_limit` already returns `(True, 0)` when the client is absent, so it enforces nothing today.

**Files:**
- Delete: `ABSA/src/utils/redis_service.py`, `ABSA/src/utils/task_queue.py`, `ABSA/src/utils/rate_limit_middleware.py`
- Modify: `ABSA/app.py`
- Modify: `streamlit-deployment/frontend_helpers.py`
- Modify: `streamlit-deployment/app_a.py`

**Interfaces:**
- Consumes: nothing
- Produces: `/process-reviews` remains the only processing endpoint; `/submit-job` and `/job-status/{job_id}` no longer exist.

- [ ] **Step 1: Write the failing test**

Create `ABSA/tests/test_no_external_services.py`:

```python
"""Redis and MongoDB were removed in Phase A. Their modules must be gone and
nothing in the backend may reference them.
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"

REMOVED_MODULES = [
    "utils/redis_service.py",
    "utils/task_queue.py",
    "utils/rate_limit_middleware.py",
]


def test_removed_modules_are_gone():
    still_present = [m for m in REMOVED_MODULES if (SRC / m).exists()]
    assert still_present == [], f"should have been deleted: {still_present}"


def test_no_redis_references_in_backend():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8").lower()
        if "redis" in text:
            offenders.append(path.relative_to(SRC).as_posix())
    assert offenders == [], f"redis still referenced: {offenders}"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/test_no_external_services.py -v`
Expected: FAIL — `test_removed_modules_are_gone` reports all three modules, `test_no_redis_references_in_backend` reports several files.

- [ ] **Step 3: Delete the Redis modules**

```bash
git -C ABSA rm src/utils/redis_service.py src/utils/task_queue.py src/utils/rate_limit_middleware.py
```

- [ ] **Step 4: Remove Redis from `ABSA/app.py`**

Remove these imports:

```python
from utils.rate_limit_middleware import RateLimitMiddleware
from utils.redis_service import get_redis_service
from utils.task_queue import get_task_queue
```

Remove the middleware registration (`app.add_middleware(RateLimitMiddleware...)`), the module-level `redis_service = get_redis_service()` assignment, and the `task_queue` global.

In `get_processor()`, delete the task-queue lines so the body becomes:

```python
def get_processor():
    """Get or initialize processor with task manager."""
    global processor
    if processor is None:
        processor = DataProcessor()
        processor.set_task_manager(task_manager)
    return processor
```

Delete the `@app.post("/submit-job")` and `@app.get("/job-status/{job_id}")` handlers entirely.

In `process_reviews`, delete the rate-limit block — the `redis_service.check_rate_limit(...)` call, the `if not is_allowed:` branch, and its `raise HTTPException(status_code=429, ...)`.

- [ ] **Step 5: Remove the matching frontend calls**

In `streamlit-deployment/frontend_helpers.py`, delete the functions `submit_analysis_job`, `get_job_status`, and `poll_job_until_complete`.

In `streamlit-deployment/app_a.py`, remove any import of those three names and any call sites. Find them with:

```bash
grep -n "submit_analysis_job\|get_job_status\|poll_job_until_complete" streamlit-deployment/app_a.py
```

Delete each call site. `call_ml_backend` (which posts to `/process-reviews`) is the remaining path and stays.

- [ ] **Step 6: Run the tests**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/ -v`
Expected: all PASS.

- [ ] **Step 7: Verify the app still compiles and serves**

```bash
.venv-bench/Scripts/python.exe -m py_compile streamlit-deployment/app_a.py streamlit-deployment/frontend_helpers.py
```

Then start the backend from `ABSA/` and `curl -s http://127.0.0.1:7860/`.
Expected: compiles clean; health endpoint responds; startup logs contain no Redis error.

- [ ] **Step 8: Commit**

```bash
git -C ABSA add -A
git -C ABSA commit -m "refactor: remove redis, task queue and rate limiting"
git add streamlit-deployment/
git commit -m "refactor: drop frontend calls to removed async job endpoints"
```

---

### Task 4: Remove MongoDB, telemetry and the admin surface

**Files:**
- Delete: `ABSA/src/utils/mongodb_service.py`, `ABSA/src/utils/ip_location_service.py`, `ABSA/src/utils/admin_endpoints.py`, `ABSA/admin_dashboard.py`
- Modify: `ABSA/app.py`, `ABSA/tests/test_no_external_services.py`, `ABSA/tests/test_import_hygiene.py`
- Modify: `streamlit-deployment/frontend_helpers.py`, `streamlit-deployment/app_a.py`

**Interfaces:**
- Consumes: `REMOVED_MODULES` from Task 3
- Produces: `/log-session`, `/log-event`, and all admin routes no longer exist.

- [ ] **Step 1: Extend the failing test**

In `ABSA/tests/test_no_external_services.py`, extend `REMOVED_MODULES` to:

```python
REMOVED_MODULES = [
    "utils/redis_service.py",
    "utils/task_queue.py",
    "utils/rate_limit_middleware.py",
    "utils/mongodb_service.py",
    "utils/ip_location_service.py",
    "utils/admin_endpoints.py",
]
```

And add:

```python
def test_no_mongo_references_in_backend():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8").lower()
        if "mongo" in text or "pymongo" in text:
            offenders.append(path.relative_to(SRC).as_posix())
    assert offenders == [], f"mongo still referenced: {offenders}"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/test_no_external_services.py -v`
Expected: FAIL, listing the mongo modules.

- [ ] **Step 3: Delete the modules**

```bash
git -C ABSA rm src/utils/mongodb_service.py src/utils/ip_location_service.py src/utils/admin_endpoints.py admin_dashboard.py
```

- [ ] **Step 4: Remove them from `ABSA/app.py`**

Remove these imports:

```python
from utils.mongodb_service import get_mongodb_service
from utils.ip_location_service import get_ip_location_service
from utils.admin_endpoints import router as admin_router
```

Remove the module-level `mongodb_service = ...` and `ip_location_service = ...` assignments and the `app.include_router(admin_router)` call. Delete the `@app.post("/log-session")` and `@app.post("/log-event")` handlers. Remove every remaining `mongodb_service.log_event(...)` call inside other handlers.

- [ ] **Step 5: Remove the matching frontend calls**

In `streamlit-deployment/frontend_helpers.py`, delete `log_session_metadata`, `log_event`, and `initialize_telemetry`.

In `streamlit-deployment/app_a.py`, remove the `initialize_telemetry(BACKEND_API_URL)` call and any `log_event` calls, plus the admin-analytics functions that call removed routes. Find them with:

```bash
grep -n "initialize_telemetry\|log_event\|log_session_metadata\|fetch_metrics_summary\|fetch_events_timeline\|fetch_funnel_analysis\|fetch_rate_limit_stats" streamlit-deployment/app_a.py streamlit-deployment/frontend_helpers.py
```

Delete each definition and call site reported.

- [ ] **Step 6: Shrink the hygiene allowlist**

In `ABSA/tests/test_import_hygiene.py`, `STREAMLIT_ALLOWLIST` becomes:

```python
STREAMLIT_ALLOWLIST = {
    "utils/data_processor.py",
}
```

- [ ] **Step 7: Run the tests**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/ -v`
Expected: all PASS.

- [ ] **Step 8: Verify startup is clean**

Start the backend from `ABSA/`. Expected: startup logs contain **no** MongoDB or Redis errors — previously the first two lines were both connection failures.

- [ ] **Step 9: Commit**

```bash
git -C ABSA add -A
git -C ABSA commit -m "refactor: remove mongodb, telemetry and admin surface"
git add streamlit-deployment/
git commit -m "refactor: drop frontend telemetry and admin analytics calls"
```

---

### Task 5: Settings replaces `st.secrets`

`data_processor.py:136` and `:403` read `st.secrets["HF_TOKEN"]`, which makes config resolution depend on whether Streamlit is running and on the working directory. One eagerly-validated `Settings` object replaces it.

**Files:**
- Create: `ABSA/src/absa/__init__.py`, `ABSA/src/absa/config.py`
- Create: `ABSA/tests/test_config.py`
- Modify: `ABSA/src/utils/data_processor.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `absa.config.Settings` — frozen dataclass with fields `hf_token: str | None`, `llm_api_key: str | None`, `llm_model: str`, `max_workers: int`
  - `absa.config.Settings.from_env() -> Settings`
  - `absa.config.ConfigError` — raised on invalid configuration
  - `absa.config.get_settings() -> Settings` — process-wide cached accessor

- [ ] **Step 1: Write the failing test**

Create `ABSA/tests/test_config.py`:

```python
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from absa.config import ConfigError, Settings  # noqa: E402


def test_reads_values_from_environment(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_abc")
    monkeypatch.setenv("LLM_MODEL", "some/model")
    monkeypatch.setenv("MAX_WORKERS", "3")
    settings = Settings.from_env()
    assert settings.hf_token == "hf_abc"
    assert settings.llm_model == "some/model"
    assert settings.max_workers == 3


def test_missing_hf_token_raises(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("ABSA_ALLOW_NO_TRANSLATION", raising=False)
    with pytest.raises(ConfigError) as exc:
        Settings.from_env()
    assert "HF_TOKEN" in str(exc.value)


def test_translation_can_be_explicitly_disabled(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("ABSA_ALLOW_NO_TRANSLATION", "1")
    settings = Settings.from_env()
    assert settings.hf_token is None


def test_invalid_max_workers_raises(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_abc")
    monkeypatch.setenv("MAX_WORKERS", "not-a-number")
    with pytest.raises(ConfigError) as exc:
        Settings.from_env()
    assert "MAX_WORKERS" in str(exc.value)


def test_settings_is_immutable(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_abc")
    settings = Settings.from_env()
    with pytest.raises(Exception):
        settings.hf_token = "changed"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'absa'`

- [ ] **Step 3: Create the package and implementation**

Create `ABSA/src/absa/__init__.py` as an empty file.

Create `ABSA/src/absa/config.py`:

```python
"""Process configuration, read once from the environment and validated eagerly.

Replaces st.secrets and scattered os.getenv calls. Validating at startup means
a missing token fails the run immediately instead of silently disabling
translation partway through a batch.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

DEFAULT_LLM_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
DEFAULT_MAX_WORKERS = 2


class ConfigError(RuntimeError):
    """Raised when the environment cannot produce a usable configuration."""


@dataclass(frozen=True)
class Settings:
    hf_token: str | None
    llm_api_key: str | None
    llm_model: str
    max_workers: int

    @classmethod
    def from_env(cls) -> "Settings":
        hf_token = os.getenv("HF_TOKEN") or None
        allow_no_translation = os.getenv("ABSA_ALLOW_NO_TRANSLATION") == "1"
        if hf_token is None and not allow_no_translation:
            raise ConfigError(
                "HF_TOKEN is not set. Non-English reviews cannot be translated. "
                "Set HF_TOKEN, or set ABSA_ALLOW_NO_TRANSLATION=1 to run without "
                "translation deliberately."
            )

        raw_workers = os.getenv("MAX_WORKERS", str(DEFAULT_MAX_WORKERS))
        try:
            max_workers = int(raw_workers)
        except ValueError as exc:
            raise ConfigError(f"MAX_WORKERS must be an integer, got {raw_workers!r}") from exc
        if max_workers < 1:
            raise ConfigError(f"MAX_WORKERS must be >= 1, got {max_workers}")

        return cls(
            hf_token=hf_token,
            llm_api_key=os.getenv("OPENROUTER_API_KEY") or None,
            llm_model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
            max_workers=max_workers,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Cached so validation happens exactly once."""
    return Settings.from_env()
```

- [ ] **Step 4: Run the tests**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/test_config.py -v`
Expected: all PASS.

- [ ] **Step 5: Replace the `st.secrets` reads**

In `ABSA/src/utils/data_processor.py`, find both token lookups (near lines 136 and 403). Each is a helper returning `st.secrets["HF_TOKEN"]` with an `os.getenv` fallback. Replace the body of each with:

```python
        from absa.config import get_settings

        return get_settings().hf_token
```

Delete any now-unused `try/except` around `st.secrets`.

- [ ] **Step 6: Run the full suite**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/ -v`
Expected: all PASS. `test_import_hygiene` still allows `utils/data_processor.py` because `import streamlit` remains at its top for `st.spinner` — removed in Task 6.

- [ ] **Step 7: Commit**

```bash
git -C ABSA add -A
git -C ABSA commit -m "feat: add absa.config.Settings, replace st.secrets token lookups"
```

---

### Task 6: ProgressReporter replaces `st.spinner`

Five `st.spinner` blocks wrap pipeline stages. In a FastAPI worker thread they emit `missing ScriptRunContext!` on every request and do nothing useful.

**Files:**
- Create: `ABSA/src/absa/progress.py`
- Create: `ABSA/tests/test_progress.py`
- Modify: `ABSA/src/utils/data_processor.py`
- Modify: `ABSA/tests/test_import_hygiene.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `absa.progress.ProgressReporter` — Protocol with `stage(name: str) -> None` and `advance(completed: int, total: int) -> None`
  - `absa.progress.NullProgress` — no-op implementation, the default
  - `absa.progress.RecordingProgress` — test double exposing `stages: list[str]` and `updates: list[tuple[int, int]]`
  - `DataProcessor.process_uploaded_data(df, task_id=None, progress=None)` — new optional keyword argument

- [ ] **Step 1: Write the failing test**

Create `ABSA/tests/test_progress.py`:

```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from absa.progress import NullProgress, RecordingProgress  # noqa: E402


def test_null_progress_accepts_calls_and_does_nothing():
    progress = NullProgress()
    progress.stage("translating")
    progress.advance(1, 10)


def test_recording_progress_captures_stages_and_updates():
    progress = RecordingProgress()
    progress.stage("translating")
    progress.advance(3, 10)
    progress.stage("extracting")
    progress.advance(10, 10)
    assert progress.stages == ["translating", "extracting"]
    assert progress.updates == [(3, 10), (10, 10)]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/test_progress.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'absa.progress'`

- [ ] **Step 3: Write the implementation**

Create `ABSA/src/absa/progress.py`:

```python
"""Progress reporting that does not assume a UI.

The pipeline announces what it is doing; the caller decides whether that
becomes a task-manager update, a log line, or nothing at all.
"""
from __future__ import annotations

from typing import Protocol


class ProgressReporter(Protocol):
    def stage(self, name: str) -> None:
        """Announce that a named pipeline stage has begun."""

    def advance(self, completed: int, total: int) -> None:
        """Report progress within the current stage."""


class NullProgress:
    """Default reporter. Discards everything."""

    def stage(self, name: str) -> None:
        return None

    def advance(self, completed: int, total: int) -> None:
        return None


class RecordingProgress:
    """Test double. Records what the pipeline reported."""

    def __init__(self) -> None:
        self.stages: list[str] = []
        self.updates: list[tuple[int, int]] = []

    def stage(self, name: str) -> None:
        self.stages.append(name)

    def advance(self, completed: int, total: int) -> None:
        self.updates.append((completed, total))
```

- [ ] **Step 4: Run the tests**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/test_progress.py -v`
Expected: PASS.

- [ ] **Step 5: Replace the spinners**

In `ABSA/src/utils/data_processor.py`, change the signature of `DataProcessor.process_uploaded_data` to:

```python
    def process_uploaded_data(self, df: pd.DataFrame, task_id: Optional[str] = None,
                              progress: Optional["ProgressReporter"] = None) -> Dict[str, Any]:
```

At the top of the method body add:

```python
        from absa.progress import NullProgress

        progress = progress or NullProgress()
```

Replace each of the five `with st.spinner("..."):` blocks with a `progress.stage(...)` call and de-indent the block body. The mapping is:

| Existing spinner text | Replacement |
|---|---|
| `"Translating reviews..."` | `progress.stage("translating")` |
| `"Classifying intents with severity analysis..."` | `progress.stage("classifying_intent")` |
| `"Extracting aspects and sentiments..."` | `progress.stage("extracting")` |
| `"Calculating aspect analytics and priority scores..."` | `progress.stage("analytics")` |
| `"Generating AI-powered summaries..."` | `progress.stage("summarising")` |

Then delete `import streamlit as st` from the top of the file. Confirm nothing else uses `st.`:

```bash
grep -n "st\." ABSA/src/utils/data_processor.py
```

Expected: no matches.

- [ ] **Step 6: Empty the hygiene allowlist**

In `ABSA/tests/test_import_hygiene.py`, set:

```python
STREAMLIT_ALLOWLIST: set[str] = set()
```

- [ ] **Step 7: Run the full suite**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/ -v`
Expected: all PASS. `test_no_streamlit_outside_allowlist` now guards the whole backend.

- [ ] **Step 8: Verify no ScriptRunContext warnings**

Start the backend from `ABSA/` and POST two reviews:

```bash
curl -s -X POST http://127.0.0.1:7860/process-reviews -H "Content-Type: application/json" \
  -d '{"data":[{"id":1,"reviews_title":"Great","review":"Battery life is amazing but the camera is poor.","date":"2024-01-15","user_id":"u1"}],"user_id":"verify"}'
```

Expected: HTTP 200, `extraction_method` is `pyabsa`, and the server log contains **no** `missing ScriptRunContext!` lines.

- [ ] **Step 9: Commit**

```bash
git -C ABSA add -A
git -C ABSA commit -m "feat: add ProgressReporter, remove streamlit from the backend entirely"
```

---

### Task 7: Split `data_processor.py` into the `absa` package

The file is ~1,150 lines holding six classes. Phases B and C add more; it has to stop being the place everything lives.

**Files:**
- Create: `ABSA/src/absa/validation.py`, `translation.py`, `extraction.py`, `intent.py`, `analytics.py`, `pipeline.py`
- Modify: `ABSA/src/utils/aspect_canonical.py` → moved to `ABSA/src/absa/aspect_canonical.py`
- Delete: `ABSA/src/utils/data_processor.py`
- Modify: `ABSA/tests/test_aspect_analytics_grouping.py`, `test_aspect_canonical.py`, `test_extraction_provenance.py`, `test_translation.py`
- Modify: `ABSA/app.py`, `benchmarks/harness/run_benchmark.py`

**Interfaces:**
- Consumes: `absa.config.get_settings`, `absa.progress.ProgressReporter`
- Produces:
  - `absa.validation.DataValidator`
  - `absa.translation.TranslationService`
  - `absa.extraction.ABSAProcessor`
  - `absa.intent.IntentClassifier`
  - `absa.analytics.AspectAnalytics`
  - `absa.pipeline.DataProcessor` — same public methods as today: `set_task_manager(tm)`, `process_uploaded_data(df, task_id=None, progress=None)`
  - `absa.aspect_canonical.canonicalize`, `canonicalize_list`

- [ ] **Step 1: Move each class to its own module**

Move classes verbatim — no behaviour changes in this task. Split `ABSA/src/utils/data_processor.py` by class:

| Class | New module |
|---|---|
| `DataValidator` | `absa/validation.py` |
| `TranslationService` | `absa/translation.py` |
| `ABSAProcessor` | `absa/extraction.py` |
| `IntentClassifier` | `absa/intent.py` |
| `AspectAnalytics` | `absa/analytics.py` |
| `DataProcessor` | `absa/pipeline.py` |

Move `aspect_canonical.py` too:

```bash
git -C ABSA mv src/utils/aspect_canonical.py src/absa/aspect_canonical.py
```

Each new module carries only the imports it needs. `absa/pipeline.py` imports the others:

```python
from absa.analytics import AspectAnalytics
from absa.aspect_canonical import canonicalize, canonicalize_list
from absa.extraction import ABSAProcessor
from absa.intent import IntentClassifier
from absa.progress import NullProgress
from absa.translation import TranslationService
from absa.validation import DataValidator
```

`absa/extraction.py` must keep the pyabsa-before-pandas ordering. Its import block starts:

```python
# IMPORT ORDER IS LOAD-BEARING: pyabsa must precede pandas or the interpreter
# segfaults on Windows. See benchmarks/harness/run_benchmark.py:17-30.
try:
    import pyabsa  # noqa: F401
except Exception:  # noqa: BLE001
    pyabsa = None

import pandas as pd
```

- [ ] **Step 2: Delete `SummaryGenerator`**

Do not move it. Delete the `SummaryGenerator` class outright — its `key_issues` strings are hardcoded constants that Phase C replaces. Remove its call site in `DataProcessor.process_uploaded_data` (the `progress.stage("summarising")` block from Task 6) and drop the resulting keys from the returned dict.

Find remaining references:

```bash
grep -rn "SummaryGenerator\|generate_macro_summary\|generate_aspect_micro_summaries" ABSA/ streamlit-deployment/ benchmarks/ --include="*.py"
```

Delete every call site reported. In `streamlit-deployment/app_a.py`, remove the UI section that renders those summary keys.

- [ ] **Step 3: Delete the old module**

```bash
git -C ABSA rm src/utils/data_processor.py
```

- [ ] **Step 4: Update the four existing tests**

Change the import lines:

| File | Old | New |
|---|---|---|
| `test_aspect_analytics_grouping.py:19` | `from utils.data_processor import AspectAnalytics` | `from absa.analytics import AspectAnalytics` |
| `test_aspect_canonical.py:12` | `from utils.aspect_canonical import canonicalize, canonicalize_list` | `from absa.aspect_canonical import canonicalize, canonicalize_list` |
| `test_extraction_provenance.py:18` | `from utils.data_processor import ABSAProcessor` | `from absa.extraction import ABSAProcessor` |
| `test_translation.py:21` | `from utils.data_processor import TranslationService` | `from absa.translation import TranslationService` |

- [ ] **Step 5: Update `ABSA/app.py`**

Replace `from utils.data_processor import DataProcessor` with `from absa.pipeline import DataProcessor`.

- [ ] **Step 6: Update the benchmark harness**

In `benchmarks/harness/run_benchmark.py:140`, replace:

```python
    from utils.data_processor import ABSAProcessor, DataProcessor, TranslationService
```

with:

```python
    from absa.extraction import ABSAProcessor
    from absa.pipeline import DataProcessor
    from absa.translation import TranslationService
```

Then check `benchmarks/harness/instrument.py` for the same import and update it identically:

```bash
grep -rn "utils.data_processor\|utils\.aspect_canonical" benchmarks/ ABSA/ streamlit-deployment/ --include="*.py"
```

Expected after edits: no matches.

- [ ] **Step 7: Run the full suite**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/ -v`
Expected: all PASS.

- [ ] **Step 8: Run the benchmark for parity**

Run: `.venv-bench/Scripts/python.exe benchmarks/harness/run_benchmark.py`

Then read the new run's `metrics_unlabeled.md` and compare against `benchmarks/runs/20260809T201004Z-translation-fixed/`.
Expected: aspect F1 `0.746`, sentiment accuracy `0.873`, keyword-fallback rows `0%`. **If any number moved, the split changed behaviour — stop and find out why before committing.**

- [ ] **Step 9: Commit**

```bash
git -C ABSA add -A
git -C ABSA commit -m "refactor: split data_processor into absa package, delete SummaryGenerator"
git add benchmarks/ streamlit-deployment/
git commit -m "refactor: point harness and frontend at the absa package"
```

---

### Task 8: Batched inference

`ABSAProcessor` has a batch loop that does not batch — it iterates and calls `self.model.predict(review, ...)` once per review. PyABSA's `predict()` accepts a list.

**Files:**
- Modify: `ABSA/src/absa/extraction.py`
- Create: `ABSA/tests/test_batched_extraction.py`

**Interfaces:**
- Consumes: `absa.extraction.ABSAProcessor`
- Produces:
  - `ABSAProcessor._extract_batch(reviews: list[str]) -> list[dict]` — one model call per batch, falling back to per-review on batch failure
  - `ABSAProcessor.extract_aspects_and_sentiments(reviews, task_id=None)` — unchanged signature and return shape

- [ ] **Step 1: Write the failing test**

Create `ABSA/tests/test_batched_extraction.py`:

```python
"""Batching must be a performance change only: same results, fewer model calls,
and one poisoned review must not destroy its whole batch.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from absa.extraction import ABSAProcessor  # noqa: E402


FOUND = {"aspect": ["battery"], "sentiment": ["Positive"],
         "confidence": [0.9], "position": [[0]]}
NOTHING_FOUND = {"aspect": [], "sentiment": [], "confidence": [], "position": []}


class FakeModel:
    """Records how it was called. Returns one result per input review."""

    def __init__(self, raise_on_batch=False, results=None):
        self.calls = []
        self.raise_on_batch = raise_on_batch
        self.results = results

    def predict(self, text, print_result=False, save_result=False):
        self.calls.append(text)
        if isinstance(text, list):
            if self.raise_on_batch:
                raise RuntimeError("batch exploded")
            if self.results is not None:
                return list(self.results)
            return [dict(FOUND) for _ in text]
        return dict(FOUND)


def _processor(model):
    # ABSAProcessor.__init__ loads a 1.1GB checkpoint, so build the instance
    # directly and set only the attributes the extraction path reads.
    # allow_keyword_fallback is required: _degraded_result branches on it.
    proc = ABSAProcessor.__new__(ABSAProcessor)
    proc.model = model
    proc.allow_keyword_fallback = False
    return proc


def test_batch_uses_one_model_call_for_many_reviews():
    model = FakeModel()
    proc = _processor(model)
    results = proc._extract_batch(["a", "b", "c"])
    assert len(results) == 3
    assert len(model.calls) == 1, "should be one batched call, not three"
    assert isinstance(model.calls[0], list)


def test_every_result_carries_provenance():
    model = FakeModel()
    proc = _processor(model)
    results = proc._extract_batch(["a", "b"])
    assert all(r["extraction_method"] == "pyabsa" for r in results)
    assert all(r["degraded_reason"] is None for r in results)


def test_batch_failure_falls_back_to_per_review():
    model = FakeModel(raise_on_batch=True)
    proc = _processor(model)
    results = proc._extract_batch(["a", "b"])
    assert len(results) == 2
    per_review_calls = [c for c in model.calls if not isinstance(c, list)]
    assert len(per_review_calls) == 2, "each review should be retried singly"


def test_empty_aspects_keep_their_provenance_in_a_batch():
    """A review PyABSA finds nothing in must still be reported honestly, not
    silently dropped or blended into the found rows."""
    model = FakeModel(results=[dict(FOUND), dict(NOTHING_FOUND)])
    proc = _processor(model)
    results = proc._extract_batch(["a", "b"])
    assert results[0]["extraction_method"] == "pyabsa"
    assert results[1]["aspects"] == []
    assert results[1]["degraded_reason"] == "pyabsa_empty"


def test_mismatched_result_count_falls_back_rather_than_misaligning():
    """If the model returns fewer results than reviews, pairing them by index
    would attach one review's aspects to another. Fall back instead."""
    model = FakeModel(results=[dict(FOUND)])
    proc = _processor(model)
    results = proc._extract_batch(["a", "b"])
    assert len(results) == 2
    per_review_calls = [c for c in model.calls if not isinstance(c, list)]
    assert len(per_review_calls) == 2
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/test_batched_extraction.py -v`
Expected: FAIL — `AttributeError: 'ABSAProcessor' object has no attribute '_extract_batch'`

- [ ] **Step 3: Implement `_extract_batch`**

Add to `ABSAProcessor` in `ABSA/src/absa/extraction.py`:

```python
    def _extract_batch(self, reviews: List[str]) -> List[Dict[str, Any]]:
        """Run one model call for the whole batch.

        A batch that raises is retried review-by-review so a single malformed
        input degrades one row instead of all of them.
        """
        try:
            raw_results = self.model.predict(
                reviews, print_result=False, save_result=False
            )
            if not isinstance(raw_results, list):
                raw_results = [raw_results]
            if len(raw_results) != len(reviews):
                raise ValueError(
                    f"model returned {len(raw_results)} results for {len(reviews)} reviews"
                )
            return [
                self._normalise_result(raw, review)
                for raw, review in zip(raw_results, reviews)
            ]
        except Exception as exc:  # noqa: BLE001 - batch failure is recoverable
            logger.warning(
                "Batched predict failed (%s: %s); retrying %d reviews individually",
                type(exc).__name__, exc, len(reviews),
            )
            results = []
            for review in reviews:
                try:
                    results.append(self._extract_with_pyabsa(review))
                except Exception as inner:  # noqa: BLE001
                    logger.warning(
                        "PyABSA raised on review: %s: %s", type(inner).__name__, inner
                    )
                    results.append(self._degraded_result(review, reason="pyabsa_error"))
            return results
```

Extract the result-shaping code currently inside `_extract_with_pyabsa` into a new method. Move **all** of it — the `aspect`/`sentiment`/`position`/`confidence` unpacking, the single-aspect normalisation, **the empty-aspects branch that returns `self._degraded_result(review, reason="pyabsa_empty")`**, and the final result dict. That empty branch is provenance-critical: it is what stopped 17% of rows being keyword output masquerading as ABSA, and it must apply per-review inside a batch.

```python
    def _normalise_result(self, raw: Dict[str, Any], review: str) -> Dict[str, Any]:
        """Shape one raw PyABSA result into our result dict.

        Returns a degraded result with reason 'pyabsa_empty' when the model
        found no opinion target, exactly as the per-review path did.
        """
```

Then rewrite `_extract_with_pyabsa` to use it:

```python
    def _extract_with_pyabsa(self, review: str) -> Dict[str, Any]:
        raw = self.model.predict(review, print_result=False, save_result=False)
        return self._normalise_result(raw, review)
```

- [ ] **Step 4: Run the tests**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/test_batched_extraction.py -v`
Expected: all three PASS.

- [ ] **Step 5: Use the batch path in the main loop**

In `extract_aspects_and_sentiments`, replace the inner `for i, review in enumerate(batch_reviews):` loop with a single call:

```python
            if self.model is not None:
                processed_results.extend(self._extract_batch(batch_reviews))
            else:
                processed_results.extend(
                    self._degraded_result(r, reason="model_unavailable")
                    for r in batch_reviews
                )
```

Keep the surrounding chunking, the progress update after each batch, and the cancellation check exactly as they are.

- [ ] **Step 6: Run the full suite**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/ -v`
Expected: all PASS, including `test_extraction_provenance.py`.

- [ ] **Step 7: Benchmark parity plus a speed measurement**

Run: `.venv-bench/Scripts/python.exe benchmarks/harness/run_benchmark.py`

Compare the new run's `metrics_unlabeled.md` and `manifest.json` against `benchmarks/runs/20260809T201004Z-translation-fixed/`:

- Aspect F1 must be `0.746` and sentiment accuracy `0.873`. **If either moved, revert the batching and investigate — this is the gate you approved.**
- `process_seconds` should be materially below the baseline `95.0`. Record the new value.

- [ ] **Step 8: Commit**

```bash
git -C ABSA add -A
git -C ABSA commit -m "perf: batch pyabsa inference with per-review fallback on batch failure"
```

---

### Task 9: Update docs and close out the phase

**Files:**
- Modify: `README.md`, `ABSA/README.md`, `ABSA/requirements.txt`, `ABSA/.env.example`

**Interfaces:**
- Consumes: everything above
- Produces: documentation matching reality

- [ ] **Step 1: Remove dependencies that are no longer imported**

From `ABSA/requirements.txt`, delete the `pymongo>=4.6.0` and `redis>=5.0.0` lines and the `# MongoDB and Redis for telemetry and task queue` comment above them. Also delete `streamlit>=1.36.0` — the backend no longer imports it. Keep every load-bearing pin and its comment untouched.

Verify nothing still needs them:

```bash
grep -rn "import redis\|import pymongo\|import streamlit" ABSA/ --include="*.py"
```

Expected: no matches.

- [ ] **Step 2: Fix the README claims that are now wrong**

In `README.md`:
- The architecture table lists Redis and MongoDB under "Backend Services". Remove that row and the "Backend Services" section describing rate limiting, session cache, task queue, and telemetry.
- Remove `- **Export Engine**: PDF reports and CSV data exports` from the Features list, or change it to `- **Export**: CSV data exports` — there is no PDF generation in the codebase and there never was.
- Update the Project Structure block: `src/utils/` becomes `src/absa/`, and the removed service modules go.
- Update the API Endpoints table to list only the endpoints that still exist: `GET /`, `GET /health`, `POST /process-reviews`, `POST /cancel-task/{task_id}`, `GET /task-status/{task_id}`, `POST /cancel-user-tasks/{user_id}`, `GET /user-tasks/{user_id}`, `GET /task-stats`, `POST /cleanup-old-tasks`.

- [ ] **Step 3: Update `.env.example`**

In `ABSA/.env.example`, delete `MONGO_URI`, `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`, `ADMIN_TOKEN`, and `IPINFO_TOKEN`. Add:

```
# Set to 1 to run deliberately without translation (HF_TOKEN unset).
ABSA_ALLOW_NO_TRANSLATION=0
```

- [ ] **Step 4: Verify config resolves identically from any working directory**

This is the spec acceptance criterion that caught a real bug: the same code
previously reported `token=MISSING` from one directory and `token=present` from
another, because `st.secrets` resolution depended on cwd.

Run from the repo root:

```bash
.venv-bench/Scripts/python.exe -c "import sys; sys.path.insert(0,'ABSA/src'); from absa.config import Settings; print('root:', Settings.from_env().hf_token is not None)"
```

Then run the equivalent from inside `ABSA/`:

```bash
cd ABSA && ../.venv-bench/Scripts/python.exe -c "import sys; sys.path.insert(0,'src'); from absa.config import Settings; print('absa:', Settings.from_env().hf_token is not None)"
```

Expected: both print the same boolean. If they differ, config is still
cwd-dependent — find the remaining implicit path assumption before continuing.

- [ ] **Step 5: Verify the whole phase end to end**

```bash
.venv-bench/Scripts/python.exe -m pytest ABSA/tests/ -v
```

Start the backend from `ABSA/`, then:

```bash
curl -s -X POST http://127.0.0.1:7860/process-reviews -H "Content-Type: application/json" \
  -d '{"data":[{"id":1,"reviews_title":"Great","review":"Battery life is amazing but the camera is poor.","date":"2024-01-15","user_id":"u1"},{"id":2,"reviews_title":"Ok","review":"Price is reasonable and delivery was fast.","date":"2024-01-16","user_id":"u2"}],"user_id":"verify"}'
```

Expected, all of which must hold:
- HTTP 200
- `extraction_method` is `pyabsa` and `degraded_reason` is `null` for both rows
- Startup logs contain no MongoDB, Redis, or ScriptRunContext lines
- No process listening on port 8502

- [ ] **Step 6: Commit**

```bash
git -C ABSA add -A
git -C ABSA commit -m "docs: align backend docs and requirements with phase A"
git add README.md
git commit -m "docs: remove redis/mongo/PDF claims, update structure and endpoints"
```
