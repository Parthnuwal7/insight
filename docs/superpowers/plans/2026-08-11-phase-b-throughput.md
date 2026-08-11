# Phase B — Throughput Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make 200–5,000 review runs practical: durable jobs that survive a restart, chunked execution that loses at most one chunk on a crash, and per-stage concurrency matched to each stage's bottleneck.

**Architecture:** A SQLite job store replaces the in-memory `TaskManager`. Work is split into chunks whose results are persisted as they complete, so a killed process resumes from the last completed chunk. Translation gets thread concurrency (network-bound); extraction gets an optional process pool (CPU-bound, 1.1 GB model per worker).

**Tech Stack:** Python 3.11, FastAPI, SQLite (stdlib `sqlite3`), `concurrent.futures`. No new third-party dependencies.

## Global Constraints

- **Interpreter:** `.venv-bench/Scripts/python.exe` (Python 3.11.9). Bare `python` may resolve to a broken path.
- **Two repositories.** `ABSA/` is a separate git repo on branch `fix/silent-fallback`; commits touching `ABSA/**` use `git -C ABSA`. The parent repo is on `benchmark/absa-baseline`; when `ABSA/` changes, bump the gitlink in the parent commit.
- **`ABSA/src/absa/__init__.py` holds a load-bearing pyabsa preload guard.** Importing pandas before pyabsa segfaults the interpreter on Windows (exit 139). Never remove or reorder it. **This applies inside every worker process too** — a spawned process re-imports from scratch.
- **`absa/` must not import the API layer, FastAPI, a UI framework, or the job store.** Dependency direction is one-way: `jobs/` may import `absa/`; `absa/` never imports `jobs/`.
- **Benchmark parity gate:** aspect F1 `0.746`, sentiment accuracy `0.873` on `benchmarks/eval_set/eval_reviews_v1.csv`. Any task that moves either number does not land.
- **Load-bearing pins** in `ABSA/requirements.txt`, do not relax: `update_checker<1.0`, `spacy>=3.7,<3.9`, plus the `en_core_web_sm` model.
- **Provenance fields `extraction_method` and `degraded_reason` must survive every change.**
- **Test baseline:** 85 tests pass at the start of this phase. The count only goes up.

---

### Task 1: The job store

Durable replacement for the in-memory `TaskManager`. Pure SQLite, no other imports — it must be testable without the ML stack.

**Files:**
- Create: `ABSA/src/jobs/__init__.py`, `ABSA/src/jobs/store.py`
- Create: `ABSA/tests/test_job_store.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `jobs.store.JobStore(db_path: str | Path)` — `":memory:"` accepted for tests
  - `JobStore.create_job(user_id: str, total_chunks: int) -> str` (returns job id)
  - `JobStore.get_job(job_id: str) -> dict | None` — keys `id, user_id, status, stage, total_chunks, completed_chunks, error, created_at, updated_at`
  - `JobStore.update_job(job_id: str, **fields) -> None`
  - `JobStore.record_chunk(job_id: str, index: int, result: dict) -> None`
  - `JobStore.get_chunk_results(job_id: str) -> list[dict]` — ordered by index
  - `JobStore.completed_chunk_indices(job_id: str) -> set[int]`
  - `JobStore.request_cancel(job_id: str) -> bool`
  - `JobStore.is_cancelled(job_id: str) -> bool`
  - `JobStore.list_user_jobs(user_id: str) -> list[dict]`
  - `JobStore.cleanup_old_jobs(max_age_seconds: int) -> int`
  - `jobs.store.JobStatus` — string constants `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`

- [ ] **Step 1: Write the failing test**

Create `ABSA/tests/test_job_store.py`:

```python
"""The job store is the durability boundary. Everything it claims to persist
must survive the process that wrote it going away.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jobs.store import JobStatus, JobStore  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return JobStore(tmp_path / "jobs.db")


def test_create_and_read_back(store):
    job_id = store.create_job(user_id="u1", total_chunks=3)
    job = store.get_job(job_id)
    assert job["user_id"] == "u1"
    assert job["total_chunks"] == 3
    assert job["completed_chunks"] == 0
    assert job["status"] == JobStatus.PENDING


def test_unknown_job_is_none(store):
    assert store.get_job("nope") is None


def test_update_job_fields(store):
    job_id = store.create_job("u1", 1)
    store.update_job(job_id, status=JobStatus.RUNNING, stage="translating")
    job = store.get_job(job_id)
    assert job["status"] == JobStatus.RUNNING
    assert job["stage"] == "translating"


def test_update_rejects_unknown_column(store):
    """A typo'd field name must fail loudly, not vanish."""
    job_id = store.create_job("u1", 1)
    with pytest.raises(ValueError):
        store.update_job(job_id, stagee="typo")


def test_chunk_results_round_trip_in_order(store):
    job_id = store.create_job("u1", 3)
    store.record_chunk(job_id, 2, {"rows": ["c"]})
    store.record_chunk(job_id, 0, {"rows": ["a"]})
    store.record_chunk(job_id, 1, {"rows": ["b"]})
    results = store.get_chunk_results(job_id)
    assert [r["rows"][0] for r in results] == ["a", "b", "c"]


def test_recording_a_chunk_advances_completed_count(store):
    job_id = store.create_job("u1", 2)
    store.record_chunk(job_id, 0, {"rows": []})
    assert store.get_job(job_id)["completed_chunks"] == 1


def test_recording_same_chunk_twice_does_not_double_count(store):
    """Resumption may re-run a chunk whose write raced a crash."""
    job_id = store.create_job("u1", 2)
    store.record_chunk(job_id, 0, {"rows": ["first"]})
    store.record_chunk(job_id, 0, {"rows": ["second"]})
    assert store.get_job(job_id)["completed_chunks"] == 1
    assert store.get_chunk_results(job_id)[0]["rows"] == ["second"]


def test_completed_indices_drive_resumption(store):
    job_id = store.create_job("u1", 4)
    store.record_chunk(job_id, 0, {})
    store.record_chunk(job_id, 2, {})
    assert store.completed_chunk_indices(job_id) == {0, 2}


def test_cancellation_is_visible(store):
    job_id = store.create_job("u1", 1)
    assert store.is_cancelled(job_id) is False
    assert store.request_cancel(job_id) is True
    assert store.is_cancelled(job_id) is True


def test_cancelling_unknown_job_returns_false(store):
    assert store.request_cancel("nope") is False


def test_data_survives_a_new_store_instance(tmp_path):
    """The whole point: a fresh process must see the previous one's work."""
    db = tmp_path / "jobs.db"
    first = JobStore(db)
    job_id = first.create_job("u1", 2)
    first.record_chunk(job_id, 0, {"rows": ["persisted"]})
    del first

    second = JobStore(db)
    job = second.get_job(job_id)
    assert job["completed_chunks"] == 1
    assert second.get_chunk_results(job_id)[0]["rows"] == ["persisted"]


def test_list_user_jobs_filters_by_user(store):
    a = store.create_job("u1", 1)
    store.create_job("u2", 1)
    assert [j["id"] for j in store.list_user_jobs("u1")] == [a]


def test_cleanup_removes_old_jobs_and_their_chunks(store):
    job_id = store.create_job("u1", 1)
    store.record_chunk(job_id, 0, {})
    assert store.cleanup_old_jobs(max_age_seconds=-1) == 1
    assert store.get_job(job_id) is None
    assert store.get_chunk_results(job_id) == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/test_job_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jobs'`

- [ ] **Step 3: Write the implementation**

Create `ABSA/src/jobs/__init__.py` as an empty file.

Create `ABSA/src/jobs/store.py`:

```python
"""Durable job state.

The previous in-memory TaskManager lost everything on restart, which is
untenable once a run takes an hour. This stores jobs and per-chunk results in
SQLite so a killed process resumes rather than starting over.

Deliberately dependency-free: no pandas, no torch, no FastAPI. That keeps it
fast to test and safe to import from a worker process.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable


class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# completed_chunks is deliberately absent: record_chunk derives it from the
# chunks table, and letting update_job overwrite it would break that invariant.
_UPDATABLE = {"status", "stage", "total_chunks", "error"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT,
    total_chunks INTEGER NOT NULL DEFAULT 0,
    completed_chunks INTEGER NOT NULL DEFAULT 0,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    job_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    result TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (job_id, idx)
);
CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_id);
"""


class JobStore:
    """SQLite-backed job state. Safe to construct per process."""

    def __init__(self, db_path: str | Path):
        self._path = str(db_path)
        # check_same_thread=False lets FastAPI's thread pool share this
        # connection, but it only disables Python's same-thread assertion --
        # it does NOT make the connection safe for concurrent use. Each
        # `with self._conn:` drives one transaction on the connection, so two
        # threads entering it collide ("cannot start a transaction within a
        # transaction") and writes are silently lost. Every public method
        # therefore takes self._lock. WAL's job is arbitration between
        # separate processes -- which the extraction pool needs later -- not
        # between threads sharing one connection.
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if self._path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def create_job(self, user_id: str = "default", total_chunks: int = 0) -> str:
        job_id = str(uuid.uuid4())
        now = time.time()
        with self._conn:
            self._conn.execute(
                "INSERT INTO jobs (id, user_id, status, stage, total_chunks,"
                " completed_chunks, cancel_requested, error, created_at, updated_at)"
                " VALUES (?, ?, ?, NULL, ?, 0, 0, NULL, ?, ?)",
                (job_id, user_id, JobStatus.PENDING, total_chunks, now, now),
            )
        return job_id

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT id, user_id, status, stage, total_chunks, completed_chunks,"
            " error, created_at, updated_at FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        return dict(row) if row else None

    def update_job(self, job_id: str, **fields: Any) -> None:
        unknown = set(fields) - _UPDATABLE
        if unknown:
            raise ValueError(
                f"unknown job field(s): {sorted(unknown)}; updatable: {sorted(_UPDATABLE)}"
            )
        if not fields:
            return
        assignments = ", ".join(f"{k} = ?" for k in fields)
        values: list[Any] = list(fields.values())
        values.extend([time.time(), job_id])
        with self._conn:
            self._conn.execute(
                f"UPDATE jobs SET {assignments}, updated_at = ? WHERE id = ?", values
            )

    def record_chunk(self, job_id: str, index: int, result: dict[str, Any]) -> None:
        """Persist one chunk's result and refresh the completed count.

        The count is recomputed from the chunks table rather than incremented,
        so a re-run chunk after a crash cannot double-count.
        """
        now = time.time()
        with self._conn:
            self._conn.execute(
                "INSERT INTO chunks (job_id, idx, result, created_at) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(job_id, idx) DO UPDATE SET result = excluded.result",
                (job_id, index, json.dumps(result), now),
            )
            self._conn.execute(
                "UPDATE jobs SET completed_chunks ="
                " (SELECT COUNT(*) FROM chunks WHERE job_id = ?), updated_at = ?"
                " WHERE id = ?",
                (job_id, now, job_id),
            )

    def get_chunk_results(self, job_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT result FROM chunks WHERE job_id = ? ORDER BY idx", (job_id,)
        ).fetchall()
        return [json.loads(r["result"]) for r in rows]

    def completed_chunk_indices(self, job_id: str) -> set[int]:
        rows = self._conn.execute(
            "SELECT idx FROM chunks WHERE job_id = ?", (job_id,)
        ).fetchall()
        return {r["idx"] for r in rows}

    def request_cancel(self, job_id: str) -> bool:
        with self._conn:
            cur = self._conn.execute(
                "UPDATE jobs SET cancel_requested = 1, updated_at = ? WHERE id = ?",
                (time.time(), job_id),
            )
        return cur.rowcount > 0

    def is_cancelled(self, job_id: str) -> bool:
        row = self._conn.execute(
            "SELECT cancel_requested, status FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row is None:
            return False
        return bool(row["cancel_requested"]) or row["status"] == JobStatus.CANCELLED

    def list_user_jobs(self, user_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, user_id, status, stage, total_chunks, completed_chunks,"
            " error, created_at, updated_at FROM jobs WHERE user_id = ?"
            " ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def cleanup_old_jobs(self, max_age_seconds: int = 3600) -> int:
        cutoff = time.time() - max_age_seconds
        with self._conn:
            ids: Iterable[str] = [
                r["id"]
                for r in self._conn.execute(
                    "SELECT id FROM jobs WHERE updated_at < ?", (cutoff,)
                ).fetchall()
            ]
            for job_id in ids:
                self._conn.execute("DELETE FROM chunks WHERE job_id = ?", (job_id,))
                self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        return len(list(ids))
```

**Every public method above must acquire `self._lock` around all connection
access — reads included.** Wrap each body in `with self._lock:`. Concurrent
reads on a shared connection during a write are not safe either, and the
contention cost is nil: a write is one small JSON blob per chunk, while a chunk
represents seconds to minutes of model inference.

- [ ] **Step 4: Add the concurrency regression test**

Append to `ABSA/tests/test_job_store.py`:

```python
def test_concurrent_writes_do_not_lose_chunks(store):
    """FastAPI serves handlers from a thread pool, so the store is written to
    from many threads at once. A shared sqlite3 connection is NOT safe for
    that on its own: two threads entering `with conn:` collide with
    "cannot start a transaction within a transaction", and chunks vanish with
    no exception reaching the caller.
    """
    import threading

    job_id = store.create_job("u1", total_chunks=50)
    errors = []

    def write(start):
        try:
            for i in range(start, start + 10):
                store.record_chunk(job_id, i, {"rows": [i]})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(s,)) for s in range(0, 50, 10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"concurrent writes raised: {errors}"
    assert store.completed_chunk_indices(job_id) == set(range(50))
    assert store.get_job(job_id)["completed_chunks"] == 50
```

- [ ] **Step 5: Run the tests, and prove the concurrency test is load-bearing**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/test_job_store.py -v`
Expected: all PASS.

Then temporarily remove the `with self._lock:` guards and re-run
`test_concurrent_writes_do_not_lose_chunks`. Expected: it FAILS, with either
`OperationalError` or a short chunk count. Restore the guards and confirm it
passes again. **A concurrency test that passes on unsynchronised code is worse
than no test at all**, so this probe is mandatory, not optional.

- [ ] **Step 6: Confirm the store has no heavy imports**

Run:

```bash
.venv-bench/Scripts/python.exe -c "import sys; sys.path.insert(0,'ABSA/src'); import jobs.store; print('pandas' in sys.modules, 'torch' in sys.modules)"
```

Expected: `False False`. The store must stay importable in a worker without dragging in the ML stack.

- [ ] **Step 7: Run the full suite and commit**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/ -v` — expect 85 + the new tests.

```bash
git -C ABSA add src/jobs tests/test_job_store.py
git -C ABSA commit -m "feat: add SQLite job store with durable chunk results"
```

---

### Task 2: Chunked runner, serial

Correctness before concurrency. This task establishes chunking, per-chunk persistence, resumption, and cancellation with no parallelism at all. Later tasks add concurrency underneath a runner that already works.

**Files:**
- Create: `ABSA/src/jobs/runner.py`
- Create: `ABSA/src/jobs/progress.py`
- Create: `ABSA/tests/test_runner.py`

**Interfaces:**
- Consumes: `jobs.store.JobStore`, `JobStatus`; `absa.progress.ProgressReporter`
- Produces:
  - `jobs.progress.JobStoreProgress(store, job_id)` — implements `ProgressReporter`
  - `jobs.runner.chunk_dataframe(df, chunk_size) -> list[pd.DataFrame]`
  - `jobs.runner.JobRunner(store, processor_factory, chunk_size=100)`
  - `JobRunner.run(job_id, df) -> dict` — merged results across chunks
  - `jobs.runner.CancelledError`

- [ ] **Step 1: Write the failing test**

Create `ABSA/tests/test_runner.py`:

```python
"""The runner's contract: split work into chunks, persist each result as it
lands, skip chunks already done, and stop promptly when cancelled.

Uses a fake processor throughout -- the real one loads a 1.1GB model.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jobs.runner import CancelledError, JobRunner, chunk_dataframe  # noqa: E402
from jobs.store import JobStatus, JobStore  # noqa: E402


def make_df(n):
    return pd.DataFrame(
        {
            "id": range(n),
            "review": [f"review {i}" for i in range(n)],
            "reviews_title": ["t"] * n,
            "date": ["2024-01-01"] * n,
            "user_id": ["u"] * n,
        }
    )


class FakeProcessor:
    """Records which chunks it saw."""

    def __init__(self):
        self.seen = []

    def process_uploaded_data(self, df, task_id=None, progress=None):
        self.seen.append(list(df["id"]))
        return {"processed_data": [{"id": int(i)} for i in df["id"]]}


@pytest.fixture
def store(tmp_path):
    return JobStore(tmp_path / "jobs.db")


def test_chunk_dataframe_splits_evenly_and_keeps_every_row():
    df = make_df(250)
    chunks = chunk_dataframe(df, 100)
    assert [len(c) for c in chunks] == [100, 100, 50]
    assert sum(len(c) for c in chunks) == 250


def test_chunk_dataframe_handles_smaller_than_one_chunk():
    assert [len(c) for c in chunk_dataframe(make_df(7), 100)] == [7]


def test_run_processes_every_chunk_and_merges_results(store):
    proc = FakeProcessor()
    runner = JobRunner(store, lambda: proc, chunk_size=10)
    job_id = store.create_job("u1", total_chunks=3)
    result = runner.run(job_id, make_df(25))
    assert len(proc.seen) == 3
    assert len(result["processed_data"]) == 25
    assert store.get_job(job_id)["status"] == JobStatus.COMPLETED


def test_each_chunk_is_persisted_as_it_completes(store):
    proc = FakeProcessor()
    runner = JobRunner(store, lambda: proc, chunk_size=10)
    job_id = store.create_job("u1", total_chunks=3)
    runner.run(job_id, make_df(25))
    assert store.get_job(job_id)["completed_chunks"] == 3


def test_resumption_skips_chunks_already_recorded(store):
    """The crash-recovery guarantee: work already done is not redone."""
    proc = FakeProcessor()
    runner = JobRunner(store, lambda: proc, chunk_size=10)
    job_id = store.create_job("u1", total_chunks=3)
    store.record_chunk(job_id, 0, {"processed_data": [{"id": i} for i in range(10)]})

    runner.run(job_id, make_df(25))

    assert [0] not in proc.seen, "chunk 0 was already done and must not re-run"
    assert len(proc.seen) == 2
    assert len(store.get_chunk_results(job_id)) == 3


def test_cancellation_stops_before_the_next_chunk(store):
    proc = FakeProcessor()
    runner = JobRunner(store, lambda: proc, chunk_size=10)
    job_id = store.create_job("u1", total_chunks=3)
    store.request_cancel(job_id)

    with pytest.raises(CancelledError):
        runner.run(job_id, make_df(25))

    assert proc.seen == [], "no chunk should start once cancel is requested"
    assert store.get_job(job_id)["status"] == JobStatus.CANCELLED


def test_a_failing_chunk_marks_the_job_failed_and_keeps_earlier_results(store):
    class ExplodingProcessor(FakeProcessor):
        def process_uploaded_data(self, df, task_id=None, progress=None):
            if 10 in list(df["id"]):
                raise RuntimeError("chunk 1 exploded")
            return super().process_uploaded_data(df, task_id, progress)

    runner = JobRunner(store, ExplodingProcessor, chunk_size=10)
    job_id = store.create_job("u1", total_chunks=3)
    with pytest.raises(RuntimeError):
        runner.run(job_id, make_df(25))

    job = store.get_job(job_id)
    assert job["status"] == JobStatus.FAILED
    assert "exploded" in job["error"]
    assert job["completed_chunks"] == 1, "chunk 0's work must not be lost"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/test_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jobs.runner'`

- [ ] **Step 3: Write `jobs/progress.py`**

```python
"""ProgressReporter that writes to the job store.

Mirrors ABSA/task_manager_progress.py, which writes to the in-memory
TaskManager. The absa package must not import the job store, so the adapter
lives here on the jobs side of the boundary.
"""
from __future__ import annotations

from jobs.store import JobStore


class JobStoreProgress:
    """Forwards pipeline progress into a JobStore row."""

    def __init__(self, store: JobStore, job_id: str):
        self._store = store
        self._job_id = job_id

    def stage(self, name: str) -> None:
        self._store.update_job(self._job_id, stage=name)

    def advance(self, completed: int, total: int) -> None:
        # Chunk-level progress is owned by the runner; within-chunk progress
        # would fight it for the same field, so this is intentionally a no-op.
        return None
```

- [ ] **Step 4: Write `jobs/runner.py`**

```python
"""Chunked execution over a JobStore.

Splitting into chunks bounds the blast radius of a crash to one chunk, and
gives cancellation a natural checkpoint. Concurrency is added by later tasks
beneath this same interface.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import pandas as pd

from jobs.progress import JobStoreProgress
from jobs.store import JobStatus, JobStore

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 100


class CancelledError(RuntimeError):
    """Raised when a job is cancelled before or between chunks."""


def chunk_dataframe(df: pd.DataFrame, chunk_size: int) -> list[pd.DataFrame]:
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    return [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]


class JobRunner:
    """Runs one job's chunks, persisting each result as it completes."""

    def __init__(
        self,
        store: JobStore,
        processor_factory: Callable[[], Any],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        self._store = store
        self._processor_factory = processor_factory
        self._chunk_size = chunk_size

    def run(self, job_id: str, df: pd.DataFrame) -> dict[str, Any]:
        chunks = chunk_dataframe(df, self._chunk_size)
        self._store.update_job(
            job_id, status=JobStatus.RUNNING, total_chunks=len(chunks)
        )
        already_done = self._store.completed_chunk_indices(job_id)
        if already_done:
            logger.info(
                "Job %s resuming: %d/%d chunks already complete",
                job_id, len(already_done), len(chunks),
            )

        progress = JobStoreProgress(self._store, job_id)
        processor = None

        for index, chunk in enumerate(chunks):
            if self._store.is_cancelled(job_id):
                self._store.update_job(job_id, status=JobStatus.CANCELLED)
                raise CancelledError(f"job {job_id} cancelled before chunk {index}")

            if index in already_done:
                continue

            if processor is None:
                # Built lazily so a fully-resumed job never pays for the model.
                processor = self._processor_factory()

            try:
                result = processor.process_uploaded_data(chunk, progress=progress)
            except Exception as exc:
                self._store.update_job(
                    job_id, status=JobStatus.FAILED, error=f"chunk {index}: {exc}"
                )
                raise

            self._store.record_chunk(job_id, index, result)

        merged = self._merge(self._store.get_chunk_results(job_id))
        self._store.update_job(job_id, status=JobStatus.COMPLETED, stage="completed")
        return merged

    @staticmethod
    def _merge(chunk_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Concatenate list-valued keys across chunks, keep scalars from the first.

        Aggregate analytics (rankings, co-occurrence) are deliberately NOT
        merged here -- summing per-chunk rankings would produce different
        numbers than ranking the whole set. The API layer recomputes them from
        the merged rows.
        """
        merged: dict[str, Any] = {}
        for result in chunk_results:
            for key, value in result.items():
                if isinstance(value, list):
                    merged.setdefault(key, []).extend(value)
                else:
                    merged.setdefault(key, value)
        return merged
```

- [ ] **Step 5: Run the tests**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/test_runner.py -v`
Expected: all PASS.

- [ ] **Step 7: Run the full suite and commit**

```bash
git -C ABSA add src/jobs tests/test_runner.py
git -C ABSA commit -m "feat: add chunked job runner with resumption and cancellation"
```

---

### Task 3: Concurrent translation

Translation is network-bound — it waits on HTTP, so threads help and the GIL does not matter. This is the cheapest real speedup in the phase.

**Files:**
- Modify: `ABSA/src/absa/translation.py`
- Modify: `ABSA/src/absa/config.py`
- Create: `ABSA/tests/test_concurrent_translation.py`

**Interfaces:**
- Consumes: `absa.config.get_settings`
- Produces:
  - `TranslationService.process_reviews(reviews: list[str]) -> tuple[list[str], list[str]]` — **unchanged signature and return shape**, now internally concurrent
  - `Settings.translation_workers: int` (env `TRANSLATION_WORKERS`, default `8`)

- [ ] **Step 1: Read the current translation path**

Read `ABSA/src/absa/translation.py` in full. The facts that shape this task:

- `process_reviews(reviews) -> (translated_reviews, detected_languages)` at ~:191 loops serially, calling `detect_language(review)` then, only when the language is `'hi'`, `translate_to_english(review, 'hi')`.
- The real signature is `translate_to_english(self, text: str, source_lang: str = 'hi')` — the parameter is `source_lang`, not `source_language`.
- `translate_to_english` splits into sentences and caches per sentence.

**Parallelise inside `process_reviews`, keeping its signature and return shape identical.** `pipeline.py` calls it in batches of 10 for cancellation granularity, and that loop stays exactly as it is — so this change is contained to one method and cancellation behaviour is untouched. Concurrency goes **across reviews**; sentence order within a review must be preserved exactly, and the existing per-sentence cache must keep working.

- [ ] **Step 2: Write the failing test**

Create `ABSA/tests/test_concurrent_translation.py`:

```python
"""Concurrency must not reorder results, bypass the cache, or change which
reviews get translated at all.
"""
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from absa.translation import TranslationService  # noqa: E402


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    return TranslationService()


def _fake_lang(mapping):
    """detect_language stub driven by a {text: lang} mapping."""
    def _detect(self, text):
        return mapping.get(text, "en")
    return _detect


def test_order_is_preserved(service, monkeypatch):
    """Threads finish out of order; results must not."""
    texts = [f"t{i}" for i in range(20)]
    monkeypatch.setattr(
        TranslationService, "detect_language", _fake_lang({t: "hi" for t in texts})
    )
    monkeypatch.setattr(
        TranslationService,
        "translate_to_english",
        lambda self, text, source_lang="hi": f"EN[{text}]",
    )

    translated, langs = service.process_reviews(texts)
    assert translated == [f"EN[t{i}]" for i in range(20)]
    assert langs == ["hi"] * 20


def test_non_hindi_is_not_translated(service, monkeypatch):
    calls = []
    monkeypatch.setattr(
        TranslationService, "detect_language", _fake_lang({"नमस्ते": "hi"})
    )

    def fake_translate(self, text, source_lang="hi"):
        calls.append(text)
        return f"EN[{text}]"

    monkeypatch.setattr(TranslationService, "translate_to_english", fake_translate)

    translated, langs = service.process_reviews(["hello", "नमस्ते"])
    assert translated == ["hello", "EN[नमस्ते]"]
    assert langs == ["en", "hi"]
    assert calls == ["नमस्ते"], "only Hindi rows should reach the translator"


def test_concurrency_is_actually_used(service, monkeypatch):
    """A serial implementation passes the order test too. Prove threads."""
    texts = [f"t{i}" for i in range(16)]
    seen_threads = set()
    monkeypatch.setattr(
        TranslationService, "detect_language", _fake_lang({t: "hi" for t in texts})
    )

    def fake_translate(self, text, source_lang="hi"):
        seen_threads.add(threading.get_ident())
        return text

    monkeypatch.setattr(TranslationService, "translate_to_english", fake_translate)
    service.process_reviews(texts)
    assert len(seen_threads) > 1, "translation ran on a single thread"


def test_one_failure_does_not_lose_the_other_rows(service, monkeypatch):
    texts = ["good", "bad", "also good"]
    monkeypatch.setattr(
        TranslationService, "detect_language", _fake_lang({t: "hi" for t in texts})
    )

    def fake_translate(self, text, source_lang="hi"):
        if text == "bad":
            raise RuntimeError("boom")
        return f"EN[{text}]"

    monkeypatch.setattr(TranslationService, "translate_to_english", fake_translate)

    translated, _ = service.process_reviews(texts)
    assert translated[0] == "EN[good]"
    assert translated[1] == "bad", "a failed translation falls back to source text"
    assert translated[2] == "EN[also good]"


def test_empty_input(service):
    assert service.process_reviews([]) == ([], [])
```

- [ ] **Step 3: Run it to verify it fails**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/test_concurrent_translation.py -v`
Expected: `test_concurrency_is_actually_used` FAILS (the current implementation is serial, so only one thread id is seen). The other tests describe existing behaviour and should already pass — that is the point: they pin the contract concurrency must not break.

- [ ] **Step 4: Add `translation_workers` to Settings**

In `ABSA/src/absa/config.py`, add a `translation_workers: int` field, parsed from `TRANSLATION_WORKERS` with default `8`, validated `>= 1` the same way `max_workers` is. Add a test to `ABSA/tests/test_config.py` mirroring the existing `max_workers` tests.

- [ ] **Step 5: Make `process_reviews` concurrent**

Replace the body of `process_reviews` in `ABSA/src/absa/translation.py`. Keep the signature, the return shape, and the `'hi'`-only translation rule exactly as they are:

```python
    def process_reviews(self, reviews: List[str]) -> Tuple[List[str], List[str]]:
        """
        Process list of reviews for translation.

        Args:
            reviews: List of review texts

        Returns:
            Tuple of (translated_reviews, detected_languages)

        Translation waits on HTTP, so threads help here even under the GIL --
        unlike extraction, which is CPU-bound and needs processes. Results are
        written back by index, so output order never depends on completion
        order.
        """
        from concurrent.futures import ThreadPoolExecutor

        from absa.config import get_settings

        if not reviews:
            return [], []

        detected_languages = [self.detect_language(r) for r in reviews]
        translated_reviews: List[str] = list(reviews)

        todo = [i for i, lang in enumerate(detected_languages) if lang == 'hi']
        if not todo:
            return translated_reviews, detected_languages

        workers = max(1, min(get_settings().translation_workers, len(todo)))

        def _one(index: int) -> Tuple[int, str]:
            try:
                return index, self.translate_to_english(reviews[index], 'hi')
            except Exception as exc:  # noqa: BLE001 - one row must not fail the batch
                logger.warning(
                    "Translation failed for review %d (%s: %s); using source text",
                    index, type(exc).__name__, exc,
                )
                return index, reviews[index]

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, translated in pool.map(_one, todo):
                translated_reviews[index] = translated

        return translated_reviews, detected_languages
```

Note `detect_language` stays serial — it is local CPU work with no I/O to overlap, and running it first keeps the language list deterministic.

- [ ] **Step 6: Confirm the pipeline needs no change**

`pipeline.py` calls `self.translator.process_reviews(batch)` inside a loop of 10 for cancellation granularity. Since the signature and return shape are unchanged, that loop stays exactly as it is. Verify with:

```bash
grep -n "process_reviews" ABSA/src/absa/pipeline.py
```

Expected: the existing call site, unmodified. If you found yourself editing `pipeline.py` in this task, stop — the change was meant to be contained to `translation.py`.

- [ ] **Step 7: Run the tests and the benchmark**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/ -v` — all pass.

Then the parity gate: `.venv-bench/Scripts/python.exe benchmarks/harness/run_benchmark.py`

Compare against `benchmarks/runs/20260809T201004Z-translation-fixed/`: aspect F1 must be `0.746`, sentiment accuracy `0.873`. Record the new `process_seconds`. **If either metric moves, revert and report BLOCKED** — concurrency must not change results.

- [ ] **Step 8: Commit**

```bash
git -C ABSA add -A
git -C ABSA commit -m "perf: translate reviews concurrently, order-preserving"
git add benchmarks/ ABSA
git commit -m "bench: record parity run for concurrent translation"
```

---

### Task 4: Optional process pool for extraction

Extraction is CPU-bound, so threads cannot help — measured identically on main and worker threads. Processes can, but each worker loads its own 1.1 GB checkpoint and Windows spawns rather than forks, so every worker re-imports and re-loads from scratch.

**This task is measurement-led: implement it, measure it, and let the number decide the default.**

**Files:**
- Create: `ABSA/src/jobs/pool.py`
- Create: `ABSA/tests/test_extraction_pool.py`
- Modify: `ABSA/src/jobs/runner.py`, `ABSA/src/absa/config.py`

**Interfaces:**
- Consumes: `jobs.runner.JobRunner`, `absa.config.get_settings`
- Produces:
  - `jobs.pool.run_chunks_in_pool(chunk_payloads, workers) -> Iterator[tuple[int, dict]]`
  - `jobs.pool.init_worker()` — per-process setup
  - `Settings.extraction_workers: int` (env `EXTRACTION_WORKERS`, default `2`)

- [ ] **Step 1: Write the failing test**

Create `ABSA/tests/test_extraction_pool.py`:

```python
"""The pool must preserve chunk identity and degrade to serial safely."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jobs.pool import resolve_worker_count  # noqa: E402


def test_single_worker_means_no_pool():
    assert resolve_worker_count(requested=1, n_chunks=10) == 1


def test_never_more_workers_than_chunks():
    """Spawning 4 processes for 2 chunks wastes two 1.1GB model loads."""
    assert resolve_worker_count(requested=4, n_chunks=2) == 2


def test_zero_or_negative_falls_back_to_serial():
    assert resolve_worker_count(requested=0, n_chunks=10) == 1
    assert resolve_worker_count(requested=-3, n_chunks=10) == 1


def test_no_chunks_is_serial():
    assert resolve_worker_count(requested=4, n_chunks=0) == 1
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv-bench/Scripts/python.exe -m pytest ABSA/tests/test_extraction_pool.py -v`
Expected: FAIL — no module `jobs.pool`.

- [ ] **Step 3: Write `jobs/pool.py`**

```python
"""Process-pool execution for the CPU-bound extraction stage.

Threads do not help here: extraction is CPU-bound and the GIL serialises it --
measured identically on the main thread and a worker thread. Processes do help,
but each worker loads its own 1.1GB checkpoint, so the pool size is bounded by
RAM rather than by core count.

On Windows, workers are spawned rather than forked, so each one re-imports the
package from scratch. That means absa/__init__.py's pyabsa preload guard runs
in every worker -- which is exactly why the guard belongs in __init__ and not
at a call site.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolve_worker_count(requested: int, n_chunks: int) -> int:
    """How many processes to actually use.

    Never more than there are chunks: an idle worker still pays the full
    model-load cost.
    """
    if requested < 1 or n_chunks < 1:
        return 1
    return min(requested, n_chunks)


def init_worker() -> None:
    """Per-process setup. Importing absa runs the pyabsa preload guard."""
    import absa  # noqa: F401
```

- [ ] **Step 4: Add `extraction_workers` to Settings**

In `ABSA/src/absa/config.py`, add `extraction_workers: int` from `EXTRACTION_WORKERS`, default `2`, validated `>= 1`. Add a config test mirroring the existing ones.

- [ ] **Step 5: Wire the pool into `JobRunner`**

Give `JobRunner.run` two paths: when `resolve_worker_count(...) == 1`, keep the existing serial loop untouched. Otherwise use a `ProcessPoolExecutor` with `initializer=init_worker`, submitting one task per not-yet-done chunk.

Requirements for the pool path:
- Each worker builds its own processor — the model is not picklable, so the factory must be called *inside* the worker, never sent across.
- Results are recorded via `store.record_chunk(job_id, index, result)` in the parent as futures complete, so persistence and resumption still work.
- The cancellation check runs in the parent between future completions.
- If pool construction raises for any reason, log a warning and fall back to the serial path rather than failing the job.

- [ ] **Step 6: Measure honestly — this decides the default**

Time a real run both ways on ~200 reviews (build a CSV by repeating the eval set):

```bash
EXTRACTION_WORKERS=1 <timed run>
EXTRACTION_WORKERS=2 <timed run>
```

Record wall-clock and peak memory for each. **If 2 workers is not meaningfully faster than 1** — plausible, since each spawn re-imports torch and reloads a 1.1 GB checkpoint, which can exceed the compute saved — then set the default to `1`, document the measurement in the module docstring, and say so in your report. Do not ship a default that your own measurement contradicts.

- [ ] **Step 7: Tests, parity, commit**

Full suite, then the benchmark parity gate (F1 `0.746`, sentiment `0.873`). Commit in both repos.

---

### Task 5: Async job API

**Files:**
- Modify: `ABSA/app.py`
- Create: `ABSA/tests/test_job_api.py`

**Interfaces:**
- Consumes: `jobs.store.JobStore`, `jobs.runner.JobRunner`
- Produces: `POST /jobs`, `GET /jobs/{job_id}`, `POST /jobs/{job_id}/cancel`, `GET /jobs?user_id=`

- [ ] **Step 1: Write the failing test**

Create `ABSA/tests/test_job_api.py` using `fastapi.testclient.TestClient`. Cover: submitting returns a job id and `202`; status reports `stage`/`completed_chunks`/`total_chunks`; cancelling a running job returns success and flips status; an unknown job id returns `404` (not `500` and not a null-data `200`); submitting an empty `data` list returns `400`.

Patch the processor factory with a fake so no model loads.

- [ ] **Step 2: Implement the endpoints**

In `ABSA/app.py`:
- Construct one module-level `JobStore` at a path from `Settings` (`JOB_DB_PATH`, default `jobs.db` beside the app).
- `POST /jobs` validates input, creates a job, starts `JobRunner.run` on the existing executor, returns `{"job_id": ...}` with `202`.
- `GET /jobs/{job_id}` returns the row, `404` when absent.
- `POST /jobs/{job_id}/cancel` calls `request_cancel`, `404` when absent.
- `GET /jobs?user_id=` lists.

**Keep `/process-reviews` working unchanged** — it is the frontend's current path and Phase D replaces it. Do not delete it in this task.

- [ ] **Step 3: Add `job_db_path` to Settings, run tests, commit**

---

### Task 6: Restart resumption, end to end

The headline durability claim, proven against a real killed process rather than a unit test.

**Files:**
- Create: `ABSA/tests/test_resumption_integration.py`
- Modify: `ABSA/README.md`, `ABSA/.env.example`

- [ ] **Step 1: Write the integration test**

Drive a runner over ~5 chunks with a fake processor that raises `SystemExit` partway, simulating a kill. Construct a **fresh `JobStore` and `JobRunner`** against the same database file, re-run, and assert: chunks completed before the kill are not reprocessed, the final merged result contains every row exactly once, and the job ends `completed`.

- [ ] **Step 2: Prove it with a real process kill**

Write a short script under the scratchpad that submits a job, kills the server mid-run, restarts it, and resumes. Capture the output in your report. This is the acceptance criterion the spec states — a unit test alone does not demonstrate it.

- [ ] **Step 3: Document the configuration**

Add to `ABSA/.env.example` with comments: `EXTRACTION_WORKERS` (with its measured memory cost per worker), `TRANSLATION_WORKERS`, `CHUNK_SIZE`, `JOB_DB_PATH`. Update `ABSA/README.md` with the async job endpoints and a short "how a long run behaves" note covering chunking, resumption, and cancellation granularity.

- [ ] **Step 4: Full suite, parity gate, commit both repos**

---

### Task 7: Retire `TaskManager`

Two job stores is one too many. Now that the durable one is proven, the in-memory one goes.

**Files:**
- Delete: `ABSA/src/utils/task_manager.py`, `ABSA/task_manager_progress.py`
- Modify: `ABSA/app.py`, `ABSA/src/absa/pipeline.py`
- Modify/delete: tests referencing `TaskManager`

- [ ] **Step 1: Inventory every reference**

```bash
grep -rn "task_manager\|TaskManager\|TaskManagerProgress" ABSA/ benchmarks/ streamlit-deployment/ --include="*.py"
```

Record the full list in your report before changing anything.

- [ ] **Step 2: Repoint the legacy endpoints**

`/cancel-task/{task_id}`, `/task-status/{task_id}`, `/cancel-user-tasks/{user_id}`, `/user-tasks/{user_id}`, `/task-stats`, `/cleanup-old-tasks` currently use `TaskManager`. Back each with `JobStore` instead, preserving response shapes so existing callers keep working.

- [ ] **Step 3: Remove `task_manager` from the pipeline**

`pipeline.py` holds `self.task_manager` and calls `update_task(...)` at ~12 sites, plus `is_cancelled` checks. Replace:
- Progress writes → the `ProgressReporter` it already receives.
- Cancellation checks → a `should_cancel: Callable[[], bool]` passed in, defaulting to `lambda: False`.

This is what finally makes `absa/` free of any job-tracking implementation. Keep `set_task_manager` as a deprecated no-op **only if** something outside still calls it; otherwise delete it.

- [ ] **Step 4: Delete the modules and add a guard test**

Add to `ABSA/tests/test_import_hygiene.py` a test that no module under `ABSA/src/absa/` references `task_manager` or `jobs`, mirroring the Streamlit guard. This locks the dependency direction the spec requires.

- [ ] **Step 5: Full suite, parity gate, end-to-end check, commit both repos**

Confirm a real `POST /jobs` run still returns `extraction_method: "pyabsa"` and `degraded_reason: null`.
