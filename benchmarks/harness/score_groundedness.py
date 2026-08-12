"""
Scores the groundedness of a generated insight report: the interpretation
layer's analogue of aspect F1.

`insights.verify` already checks every claim against its cited reviews
*before* a report is built -- only claims judged "supported" ever survive
into a `Report`. So why re-check them here? Because that check is internal
to the pipeline: if a future edit weakens the agent's claim-generation
prompt, or weakens `insights.verify`'s own judging prompt, the pipeline
would happily go on calling its own (now looser) output "verified" and
nothing would notice. This module is deliberately NOT wired to
`insights.verify` -- it does not import it, does not reuse its prompt
constant, and does not run inside the pipeline. It re-derives groundedness
from scratch, against a *persisted* report, using its own judgment. That
independence is the whole point: it is the external gate, not an internal
one, mirroring the relationship `score_judgments.py` has to the pipeline
it grades (a separate scorer, against separately-sourced text, that the
thing under test cannot quietly satisfy by changing its own definition of
"correct").

How it judges support: for each claim in the report, every cited
review id is re-fetched from a reviews file (`predictions.json` by
default -- plain JSON, id -> review text, no pipeline re-run required) and
handed to an LLM judge with the claim text, at temperature 0 (determinism
matters here more than anywhere else: the same claim against the same
evidence must get the same verdict). The model used is recorded in the
output so a number can always be traced to what produced it.

Two rules that keep the number honest, matching the discipline in
`insights.verify` and the brief this module was written against:

* A claim that cites a review id absent from the reviews file scores
  **ungrounded** (reason `"unknown_review_id"`), never skipped. Skipping it
  would drop it from both numerator and denominator and flatter the
  fraction. If *any* cited id is unknown the whole claim is ungrounded --
  a claim cannot be trusted to rest on the evidence it names if part of
  that evidence does not exist, same reasoning as
  `insights.verify._fetch_citations`.
* If the report contains zero claims, `groundedness_fraction` is `None`,
  not `1.0`. A report that says nothing has an undefined groundedness, not
  a perfect one -- see `totals.claims_total == 0` in the output.

Fails closed: if no API key is configured, or the judge call fails after
its one retry, the claim is scored ungrounded (reason `"llm_unavailable"`),
never silently excluded and never counted as grounded by default.

Runnable offline against a stored report: no ABSA/pyabsa/pandas import, no
pipeline re-run -- only `--report` (a persisted report JSON) and
`--reviews` (a persisted id -> text JSON) are read, plus one network call
per claim to the judge LLM. Re-running this script against the same two
files with the same model always asks the same questions.

Usage:
    .venv-bench/Scripts/python.exe benchmarks/harness/score_groundedness.py --run <run_id>
    .venv-bench/Scripts/python.exe benchmarks/harness/score_groundedness.py          # uses LATEST
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import requests

HARNESS_DIR = Path(__file__).resolve().parent
BENCH_DIR = HARNESS_DIR.parent
REPO_ROOT = BENCH_DIR.parent

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Matches absa.config.DEFAULT_LLM_MODEL, so a baseline recorded by this
# scorer used the same model insights.verify uses by default unless
# overridden -- but this constant is defined independently, not imported,
# so a change to that default does not silently retarget this scorer.
DEFAULT_JUDGE_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_WORKERS = 8

SECTIONS = ("findings", "complaints", "strengths", "actions")

_VALID_VERDICTS = frozenset({"supported", "not_supported"})

# Deliberately not imported from insights.verify -- see module docstring.
# Same discipline (strict, default to "not_supported" when in doubt), an
# independent copy so it cannot drift in lockstep with the thing it grades.
JUDGE_PROMPT_TEMPLATE = """You are a strict, independent auditor scoring whether a claim from an automated review-analysis report is actually grounded in the review text cited for it. You did not produce the claim and have no stake in it looking good.

Claim ({kind}): "{claim_text}"

Cited reviews:
{reviews}

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"verdict": "supported" | "not_supported"}}

Rules:
- "supported" means the cited review text, read plainly, actually backs the claim -- not merely plausible, not merely related.
- "not_supported" means the cited text is unrelated to the claim, contradicts it, or only weakly or ambiguously supports it. When in doubt, choose "not_supported".
- Judge only the review text given above. Do not use outside knowledge about the product, brand, or claim.
"""


class JudgeUnavailable(RuntimeError):
    """No API key configured, or the judge call failed even after retry."""


# ---- LLM client (self-contained: no ABSA import) --------------------------


def _call_judge(prompt: str, model: str, api_key: str, timeout: float) -> str:
    """One OpenRouter chat-completion call at temperature 0. Retries once,
    mirroring insights.llm.LLMClient.complete -- the same retry-once
    discipline, reimplemented here rather than imported so this module
    never depends on ABSA/src being importable."""
    last_exc: Optional[Exception] = None
    for _attempt in range(2):
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            return payload["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001 - any failure is retried once
            last_exc = exc
    raise JudgeUnavailable(f"OpenRouter request failed after retry: {last_exc}") from last_exc


def _parse_verdict(raw_response: str) -> tuple[Optional[str], Optional[str]]:
    """Mirrors insights.verify._parse_verdict's error taxonomy so a reader
    already familiar with that module recognises these reason strings."""
    try:
        parsed = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError):
        return None, "unparseable_response"
    if not isinstance(parsed, dict):
        return None, "unparseable_response"
    verdict = parsed.get("verdict")
    if not isinstance(verdict, str) or not verdict.strip():
        return None, "unparseable_response"
    normalized = verdict.strip().lower()
    if normalized not in _VALID_VERDICTS:
        return None, "ambiguous_verdict"
    return normalized, None


# ---- loading ---------------------------------------------------------------


def load_report(path: Path) -> dict[str, list]:
    """Load a report JSON, tolerant of either the bare report shape
    (`{"findings": [...], ...}`) or the `/insights/report` HTTP envelope
    (`{"status": "success", "data": {"findings": [...], ...}}`)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "findings" not in raw and isinstance(raw.get("data"), dict):
        raw = raw["data"]
    return {section: list(raw.get(section) or []) for section in SECTIONS}


def load_reviews(path: Path) -> dict[Any, str]:
    """Load an id -> review text lookup from a `predictions.json`-shaped
    file (a list of dicts, each with "id" and "review"). Both the int and
    str form of each id are stored, so a claim's review_ids can be matched
    regardless of whether JSON round-tripping preserved the original type."""
    records = json.loads(path.read_text(encoding="utf-8"))
    lookup: dict[Any, str] = {}
    for record in records:
        if "id" not in record:
            continue
        rid = record["id"]
        text = record.get("review", "")
        lookup[rid] = text
        lookup[str(rid)] = text
    return lookup


# ---- scoring ----------------------------------------------------------------


def _format_citations(review_ids: list, reviews_by_id: dict[Any, str]) -> str:
    lines = []
    for rid in review_ids:
        lines.append(f"[review {rid}]: \"{reviews_by_id[rid]}\"")
    return "\n".join(lines)


def score_one_claim(
    section: str,
    claim: dict,
    reviews_by_id: dict[Any, str],
    model: str,
    api_key: Optional[str],
    timeout: float,
) -> dict[str, Any]:
    """Score one claim. Never raises -- every failure mode is recorded as
    an ungrounded verdict with a reason, matching insights.verify's
    never-raises contract for the same reasons (a batch scoring run must
    not abort over one bad claim or one flaky network call)."""
    text = claim.get("text", "")
    kind = claim.get("kind", "")
    review_ids = list(claim.get("review_ids") or [])

    base = {"section": section, "kind": kind, "text": text, "review_ids": review_ids}

    if not review_ids:
        # Should be unreachable -- insights.report never emits an uncited
        # claim -- but a claim with no citation cannot be grounded by
        # definition, so this is checked rather than assumed.
        return {**base, "verdict": "ungrounded", "reason": "uncited"}

    for rid in review_ids:
        if rid not in reviews_by_id:
            return {**base, "verdict": "ungrounded", "reason": "unknown_review_id"}

    if not api_key:
        return {**base, "verdict": "ungrounded", "reason": "llm_unavailable"}

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        kind=kind, claim_text=text, reviews=_format_citations(review_ids, reviews_by_id)
    )
    try:
        raw_response = _call_judge(prompt, model, api_key, timeout)
    except JudgeUnavailable:
        return {**base, "verdict": "ungrounded", "reason": "llm_unavailable"}

    verdict, error_reason = _parse_verdict(raw_response)
    if error_reason is not None:
        return {**base, "verdict": "ungrounded", "reason": error_reason}
    if verdict == "supported":
        return {**base, "verdict": "grounded", "reason": "supported"}
    return {**base, "verdict": "ungrounded", "reason": "not_supported"}


def score_report(
    report: dict[str, list],
    reviews_by_id: dict[Any, str],
    model: str,
    api_key: Optional[str],
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    """Score every claim in `report`. Returns the full `groundedness.json`
    payload (see module docstring / README for the shape)."""
    jobs = [
        (section, claim)
        for section in SECTIONS
        for claim in report.get(section, [])
    ]
    claims_by_section = {section: len(report.get(section, [])) for section in SECTIONS}
    total = len(jobs)

    results: list[Optional[dict]] = [None] * total
    if jobs:
        workers = max(1, min(workers, total))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(score_one_claim, section, claim, reviews_by_id, model, api_key, timeout): i
                for i, (section, claim) in enumerate(jobs)
            }
            for future, i in futures.items():
                results[i] = future.result()

    grounded = sum(1 for r in results if r["verdict"] == "grounded") if results else 0
    ungrounded = total - grounded
    ungrounded_by_reason: dict[str, int] = {}
    for r in results:
        if r["verdict"] == "ungrounded":
            ungrounded_by_reason[r["reason"]] = ungrounded_by_reason.get(r["reason"], 0) + 1

    if total == 0:
        fraction = None
        note = (
            "The report contained zero claims. Groundedness is undefined here, "
            "not 1.0 -- a report that says nothing is not perfectly grounded."
        )
    else:
        fraction = round(grounded / total, 4)
        note = None

    return {
        "judge_model": model,
        "judge_temperature": 0.0,
        "totals": {
            "claims_total": total,
            "claims_by_section": claims_by_section,
            "grounded": grounded,
            "ungrounded": ungrounded,
            "ungrounded_by_reason": ungrounded_by_reason,
        },
        "groundedness_fraction": fraction,
        "groundedness_note": note,
        "claims": [r for r in results if r is not None],
    }


def render_markdown(result: dict[str, Any], run_id: str) -> str:
    t = result["totals"]
    frac = result["groundedness_fraction"]
    lines = [
        "# Groundedness",
        "",
        f"Run: `{run_id}`",
        f"Judge model: `{result['judge_model']}` (temperature {result['judge_temperature']})",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Claims in report | {t['claims_total']} |",
        f"| Grounded | {t['grounded']} |",
        f"| Ungrounded | {t['ungrounded']} |",
        f"| Groundedness fraction | "
        f"**{frac if frac is not None else 'undefined (0 claims)'}** |",
    ]
    if result["groundedness_note"]:
        lines += ["", f"> {result['groundedness_note']}"]

    lines += [
        "",
        "## By section",
        "",
        "| Section | Claims |",
        "|---|---:|",
    ]
    for section, n in t["claims_by_section"].items():
        lines.append(f"| `{section}` | {n} |")

    if t["ungrounded_by_reason"]:
        lines += ["", "## Ungrounded, by reason", "", "| Reason | Count |", "|---|---:|"]
        for reason, n in sorted(t["ungrounded_by_reason"].items()):
            lines.append(f"| `{reason}` | {n} |")

    ungrounded_claims = [c for c in result["claims"] if c["verdict"] == "ungrounded"]
    if ungrounded_claims:
        lines += ["", "## Ungrounded claims", ""]
        for c in ungrounded_claims:
            lines += [
                f"**{c['section']}** (`{c['reason']}`, cites {c['review_ids']})",
                "",
                f"> {c['text']}",
                "",
            ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="run id (defaults to runs/LATEST)")
    parser.add_argument("--report", default=None, help="path to report JSON (defaults to <run>/report.json)")
    parser.add_argument("--reviews", default=None, help="path to id->review JSON (defaults to <run>/predictions.json)")
    parser.add_argument("--model", default=os.getenv("GROUNDEDNESS_JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()

    run_id = args.run
    if not run_id:
        latest = BENCH_DIR / "runs" / "LATEST"
        if not latest.exists():
            raise SystemExit("No --run given and benchmarks/runs/LATEST is missing.")
        run_id = latest.read_text(encoding="utf-8").strip()

    run_dir = BENCH_DIR / "runs" / run_id
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    report_path = Path(args.report) if args.report else run_dir / "report.json"
    if not report_path.exists():
        raise SystemExit(
            f"Report not found: {report_path}\n"
            f"Generate one first, e.g. with run_insight_report.py --run {run_id}."
        )
    reviews_path = Path(args.reviews) if args.reviews else run_dir / "predictions.json"
    if not reviews_path.exists():
        raise SystemExit(f"Reviews file not found: {reviews_path}")

    # Best-effort: pick up OPENROUTER_API_KEY from ABSA/.env if the shell
    # environment doesn't already have it, mirroring run_benchmark.py.
    # Never fatal -- a missing .env or missing python-dotenv just means the
    # judge falls back to whatever (if anything) os.environ already has,
    # and an absent key is handled explicitly below (fail closed, not a
    # crash).
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / "ABSA" / ".env")
    except Exception:  # noqa: BLE001
        pass

    report = load_report(report_path)
    reviews_by_id = load_reviews(reviews_path)
    api_key = os.getenv(args.api_key_env) or None

    t0 = time.time()
    result = score_report(
        report, reviews_by_id, args.model, api_key, timeout=args.timeout, workers=args.workers
    )
    result["run_id"] = run_id
    result["report_path"] = str(report_path)
    result["reviews_path"] = str(reviews_path)
    result["scoring_seconds"] = round(time.time() - t0, 1)

    (run_dir / "groundedness.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_markdown(result, run_id)
    (run_dir / "groundedness.md").write_text(md, encoding="utf-8")

    try:
        print(md)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        print(md.encode(enc, errors="replace").decode(enc))

    print(f"\n[score] wrote {run_dir / 'groundedness.json'}")
    print(f"[score] wrote {run_dir / 'groundedness.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
