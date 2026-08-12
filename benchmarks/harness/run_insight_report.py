"""
Generates an insights.report.Report over an existing benchmark run's
extraction output, and persists it as `report.json` in that run's
directory -- input for score_groundedness.py.

Existing benchmark runs (see run_benchmark.py) capture extraction output
only: per-review aspects/sentiments/route, not an investigated, verified
report. This script takes that already-captured output, feeds it through
the real `insights` pipeline (InsightTools -> agent.investigate ->
verify.verify_claims -> report.build_report -- the same sequence
ABSA/app.py's `_build_insight_report` runs for POST /insights/report,
reimplemented here rather than imported since app.py boots a full FastAPI
app with job stores and thread pools this script has no use for), and
writes the resulting report next to the run's other artifacts.

Deliberately out of scope: theme clustering. `_build_insight_report` builds
clusters via `insights.embed.Embedder` (a local sentence-transformers
model) when available; this script always passes `clusters=[]`. Clustering
is supplementary context for the investigation agent, not required for it
to produce citable claims (get_reviews_for_aspect / get_aspect_stats /
search_reviews all still work without it), and skipping it avoids a model
download this script has no other reason to trigger. A report generated
this way therefore has no cluster section -- recorded as a condition of
whatever baseline is established from it, not hidden.

Usage:
    .venv-bench/Scripts/python.exe benchmarks/harness/run_insight_report.py --run <run_id>
    .venv-bench/Scripts/python.exe benchmarks/harness/run_insight_report.py          # uses LATEST
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# IMPORT ORDER IS LOAD-BEARING. Do not reorder.
# On Windows, importing pandas before pyabsa segfaults the interpreter
# (exit 139). This script imports insights.tools (pandas at module scope)
# and absa.aspect_canonical, so pyabsa must be preloaded first -- same
# pattern as run_benchmark.py and ABSA/app.py.
# ---------------------------------------------------------------------------
try:
    import pyabsa  # noqa: F401  # MUST precede pandas
except Exception:  # noqa: BLE001
    pass

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

HARNESS_DIR = Path(__file__).resolve().parent
BENCH_DIR = HARNESS_DIR.parent
REPO_ROOT = BENCH_DIR.parent
ABSA_SRC = REPO_ROOT / "ABSA" / "src"
sys.path.insert(0, str(ABSA_SRC))

import pandas as pd  # noqa: E402


def _extraction_fields(route: str) -> tuple[str, Optional[str]]:
    """Invert run_benchmark.py's route derivation: `route = method if
    method != 'none' else f'none:{reason}'` -> (method, reason)."""
    if route.startswith("none:"):
        return "none", route.split(":", 1)[1]
    return route, None


def build_records(review_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """`predictions.json` review rows -> (processed_records, aspect_records)
    shaped exactly like InsightTools expects (see ABSA/tests/test_report.py's
    `_processed_df` fixture and insights/tools.py's column contract)."""
    from absa.aspect_canonical import canonicalize

    processed_records: list[dict] = []
    aspect_records: list[dict] = []
    for row in review_rows:
        method, reason = _extraction_fields(row.get("route") or "")
        processed_records.append({
            "id": row["id"],
            "review": row["review"],
            "overall_sentiment": row.get("overall_sentiment"),
            "extraction_method": method,
            "degraded_reason": reason,
        })
        aspects = row.get("aspects") or []
        sentiments = row.get("aspect_sentiments") or []
        for i, aspect in enumerate(aspects):
            aspect_records.append({
                "review_id": row["id"],
                "aspect": aspect,
                "aspect_canonical": canonicalize(aspect),
                "aspect_sentiment": sentiments[i] if i < len(sentiments) else None,
                "review": row["review"],
                "extraction_method": method,
            })
    return processed_records, aspect_records


def _serialize_claim(vc: Any) -> dict:
    """Mirrors ABSA/app.py's `_serialize_verified_claim` exactly, so a
    report written by this script is byte-for-byte the shape
    /insights/report already returns under `data`."""
    return {
        "text": vc.claim.text,
        "kind": vc.claim.kind,
        "review_ids": list(vc.claim.review_ids),
        "reason": vc.reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="run id (defaults to runs/LATEST)")
    parser.add_argument("--max-tool-calls", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args()

    run_id = args.run
    if not run_id:
        latest = BENCH_DIR / "runs" / "LATEST"
        if not latest.exists():
            raise SystemExit("No --run given and benchmarks/runs/LATEST is missing.")
        run_id = latest.read_text(encoding="utf-8").strip()

    run_dir = BENCH_DIR / "runs" / run_id
    predictions_path = run_dir / "predictions.json"
    if not predictions_path.exists():
        raise SystemExit(
            f"No predictions.json in {run_dir}; run run_benchmark.py first."
        )

    review_rows = json.loads(predictions_path.read_text(encoding="utf-8"))
    processed_records, aspect_records = build_records(review_rows)

    # Load ABSA/.env so OPENROUTER_API_KEY (and everything else) matches the
    # rest of the harness -- mirrors run_benchmark.py. HF_TOKEN is
    # deliberately absent (see README's "Environment notes"); this script
    # never calls TranslationService, so ABSA_ALLOW_NO_TRANSLATION=1 opts
    # out of Settings.from_env()'s HF_TOKEN requirement rather than
    # fabricating a token this script has no use for.
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / "ABSA" / ".env")
    except Exception:  # noqa: BLE001
        pass
    os.environ.setdefault("ABSA_ALLOW_NO_TRANSLATION", "1")

    from absa.config import get_settings
    from insights.agent import InvestigationBudget, investigate
    from insights.llm import LLMClient
    from insights.report import build_report
    from insights.tools import InsightTools
    from insights.verify import verify_claims

    processed_df = pd.DataFrame(processed_records)
    aspect_df = pd.DataFrame(aspect_records) if aspect_records else pd.DataFrame()

    tools = InsightTools(processed_df, aspect_df, clusters=[], embedder=None)

    settings = get_settings()
    client = LLMClient(settings)
    if not client.is_available:
        raise SystemExit(
            "OPENROUTER_API_KEY is not configured; cannot generate a report "
            "(the agent and verifier both fail closed with no claims)."
        )

    budget_kwargs: dict[str, Any] = {}
    if args.max_tool_calls is not None:
        budget_kwargs["max_tool_calls"] = args.max_tool_calls
    if args.max_seconds is not None:
        budget_kwargs["max_seconds"] = args.max_seconds
    budget = InvestigationBudget(**budget_kwargs) if budget_kwargs else None

    print(f"[report] investigating run {run_id} ({len(processed_records)} reviews, "
          f"{len(aspect_records)} aspect rows)...")
    t0 = time.time()
    claims, agent_stats = investigate(tools, client, budget)
    print(f"[report] investigation stopped: {agent_stats['stopped_reason']} "
          f"({agent_stats['claims_returned']} claims returned, "
          f"{agent_stats['claims_discarded_uncited']} uncited, "
          f"{agent_stats['claims_discarded_malformed']} malformed, "
          f"{agent_stats['tool_calls_made']} tool calls, "
          f"{agent_stats['elapsed_seconds']:.1f}s)")

    verified, verify_stats = verify_claims(claims, tools, client)
    print(f"[report] verification: kept {verify_stats['kept']} / "
          f"dropped {verify_stats['dropped']} {verify_stats['dropped_by_reason']}")

    report = build_report(verified, tools, {"agent": agent_stats, "verify": verify_stats})
    elapsed = round(time.time() - t0, 1)

    payload = {
        "run_id": run_id,
        "llm_model": settings.llm_model,
        "generated_seconds": elapsed,
        "findings": [_serialize_claim(vc) for vc in report.findings],
        "complaints": [_serialize_claim(vc) for vc in report.complaints],
        "strengths": [_serialize_claim(vc) for vc in report.strengths],
        "actions": [_serialize_claim(vc) for vc in report.actions],
        "stats": report.stats,
        "caveats": report.caveats,
        "is_empty": report.is_empty,
    }

    out_path = run_dir / "report.json"
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"[report] wrote {out_path} ({elapsed}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
