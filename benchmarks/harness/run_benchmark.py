"""
Baseline benchmark runner.

Executes the REAL pipeline from ABSA/src/utils/data_processor.py -- unmodified,
imported directly -- against the evaluation set, with runtime instrumentation
recording which code path handled each review. Persists every artifact under
benchmarks/runs/<timestamp>/ so results are reproducible and comparable across
future pipeline versions.

Usage:
    .venv-bench/Scripts/python.exe benchmarks/harness/run_benchmark.py
    .venv-bench/Scripts/python.exe benchmarks/harness/run_benchmark.py --eval-set path/to.csv
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# IMPORT ORDER IS LOAD-BEARING. Do not reorder the next three lines.
#
# On Windows, importing pandas before pyabsa segfaults the interpreter
# (exit 139) either at `import pyabsa` or later during model construction --
# a native initialisation-order conflict between the numpy/pandas and
# torch/pyabsa native stacks. Importing pyabsa first is stable.
#
# ABSA/src/utils/data_processor.py imports pandas at module scope, so pyabsa
# must be imported before that module is touched. This affects only import
# sequencing, not pipeline behaviour.
# ---------------------------------------------------------------------------
try:
    import pyabsa  # noqa: F401  # MUST precede pandas
    _PYABSA_IMPORT_ERROR = None
except Exception as _exc:  # noqa: BLE001
    _PYABSA_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
BENCH_DIR = HARNESS_DIR.parent
REPO_ROOT = BENCH_DIR.parent
ABSA_SRC = REPO_ROOT / "ABSA" / "src"

sys.path.insert(0, str(HARNESS_DIR))
sys.path.insert(0, str(ABSA_SRC))


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _spacy_model_available(name: str = "en_core_web_sm") -> bool:
    """PyABSA's APC prediction path calls spacy.load('en_core_web_sm').

    That model is not declared in ABSA/requirements.txt. When it is missing,
    every _extract_with_pyabsa call raises, the per-review except clause in
    ABSAProcessor swallows it, and the review silently takes the keyword
    fallback -- with the checkpoint loaded and reporting healthy.
    """
    try:
        import spacy
        spacy.load(name)
        return True
    except Exception:
        return False


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for mod in ("torch", "transformers", "pyabsa", "pandas", "numpy", "langdetect", "networkx"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            out[mod] = "MISSING"
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default=str(BENCH_DIR / "eval_set" / "eval_reviews_v1.csv"))
    parser.add_argument("--label", default="baseline", help="short name for this run")
    args = parser.parse_args()

    # Load ABSA/.env so the run matches production configuration (HF_TOKEN etc).
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / "ABSA" / ".env")
    except Exception:
        pass

    import pandas as pd

    from instrument import PipelineRecorder, heuristic_is_fallback
    import metrics_unlabeled
    from make_judge_packet import build_packet

    eval_path = Path(args.eval_set)
    df_eval = pd.read_csv(eval_path)

    dupes = df_eval["review"][df_eval["review"].duplicated()].tolist()
    if dupes:
        # clean_data() silently drops duplicate review text, which would break
        # the row alignment this harness depends on.
        print(f"FATAL: duplicate review text in eval set: {dupes[:3]}", file=sys.stderr)
        return 2

    meta_cols = ["source", "probe_category", "lang_expected"]
    meta_by_review = {
        row["review"]: {c: row.get(c) for c in meta_cols}
        for _, row in df_eval.iterrows()
    }

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{args.label}"
    run_dir = BENCH_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_id}\n[run] artifacts -> {run_dir}")

    if _PYABSA_IMPORT_ERROR:
        print(f"[run] WARNING: import pyabsa failed -> {_PYABSA_IMPORT_ERROR}")
    spacy_ok = _spacy_model_available()
    print(f"[run] pyabsa import ok: {_PYABSA_IMPORT_ERROR is None} | "
          f"spacy en_core_web_sm available: {spacy_ok}")
    if not spacy_ok:
        print("[run] WARNING: without en_core_web_sm every PyABSA call will raise "
              "and fall back to keyword buckets.")

    from utils.data_processor import ABSAProcessor, DataProcessor, TranslationService

    recorder = PipelineRecorder()
    recorder.install(ABSAProcessor, TranslationService)

    print("[run] constructing DataProcessor (first run downloads the PyABSA checkpoint)...")
    t_init = time.time()
    processor = DataProcessor()
    init_secs = round(time.time() - t_init, 1)

    model_loaded = processor.absa_processor.model is not None
    recorder.model_loaded = model_loaded
    print(f"[run] PyABSA model loaded: {model_loaded}  ({init_secs}s)")
    if not model_loaded:
        print("[run] WARNING: every review will take the keyword-fallback path.")

    pipeline_input = df_eval[["id", "reviews_title", "review", "date", "user_id"]].copy()

    print(f"[run] processing {len(pipeline_input)} reviews...")
    t0 = time.time()
    result = processor.process_uploaded_data(pipeline_input)
    elapsed = round(time.time() - t0, 1)
    recorder.uninstall()

    if "error" in result:
        print(f"FATAL: pipeline returned validation errors: {result['error']}", file=sys.stderr)
        return 3
    if result.get("status") == "cancelled":
        print("FATAL: pipeline reported cancellation", file=sys.stderr)
        return 4

    print(f"[run] done in {elapsed}s ({round(elapsed / max(len(pipeline_input), 1), 2)}s/review)")

    processed = result["processed_data"]
    absa_details = result["absa_details"]

    review_rows, aspect_rows = [], []
    for i, (_, prow) in enumerate(processed.iterrows()):
        original = prow["review"]
        detail = absa_details[i] if i < len(absa_details) else {}
        aspects = list(detail.get("aspects") or [])
        sentiments = list(detail.get("sentiments") or [])
        confidences = list(detail.get("confidence_scores") or [])
        positions = list(detail.get("positions") or [])

        route = recorder.route_for(original)
        tinfo = recorder.translation_for(original)
        meta = meta_by_review.get(original, {})

        review_rows.append({
            "id": int(prow["id"]),
            "review": original,
            "translated_review": prow.get("translated_review"),
            "source": meta.get("source"),
            "probe_category": meta.get("probe_category"),
            "lang_expected": meta.get("lang_expected"),
            "detected_lang": prow.get("detected_language"),
            "route": route,
            "heuristic_fallback": heuristic_is_fallback(aspects, confidences, original, positions),
            "translate_attempted": tinfo["translate_attempted"],
            "api_called": tinfo["api_called"],
            "text_changed": tinfo["text_changed"],
            "intent": prow.get("intent"),
            "intent_severity": prow.get("intent_severity"),
            "overall_sentiment": prow.get("overall_sentiment"),
            "aspects": aspects,
            "aspect_sentiments": sentiments,
            "n_aspects": len(aspects),
        })

        for j, aspect in enumerate(aspects):
            aspect_rows.append({
                "review_id": int(prow["id"]),
                "aspect": aspect,
                "sentiment": sentiments[j] if j < len(sentiments) else None,
                "confidence": confidences[j] if j < len(confidences) else None,
                "route": route,
                "probe_category": meta.get("probe_category"),
                "review": original,
            })

    metrics = metrics_unlabeled.compute(review_rows, aspect_rows, model_loaded)

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "eval_set": str(eval_path.relative_to(REPO_ROOT)),
        "eval_set_sha256": hashlib.sha256(eval_path.read_bytes()).hexdigest()[:16],
        "n_reviews_in": int(len(df_eval)),
        "n_reviews_processed": len(review_rows),
        "rows_dropped_by_clean_data": int(len(df_eval) - len(review_rows)),
        "pyabsa_model_loaded": model_loaded,
        "pyabsa_import_error": _PYABSA_IMPORT_ERROR,
        "spacy_en_core_web_sm_available": spacy_ok,
        "hf_token_present": bool(os.getenv("HF_TOKEN")),
        "init_seconds": init_secs,
        "process_seconds": elapsed,
        "versions": _versions(),
    }

    import pandas as pd  # noqa: F811  (local rebind for clarity)

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "metrics_unlabeled.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "metrics_unlabeled.md").write_text(
        metrics_unlabeled.render_markdown(metrics, run_id), encoding="utf-8")
    pd.DataFrame(review_rows).to_csv(run_dir / "review_level.csv", index=False, encoding="utf-8")
    pd.DataFrame(aspect_rows).to_csv(run_dir / "aspect_level.csv", index=False, encoding="utf-8")
    (run_dir / "predictions.json").write_text(
        json.dumps(review_rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    packet = build_packet(review_rows, run_id)
    (run_dir / "judge_packet.md").write_text(packet, encoding="utf-8")

    latest = BENCH_DIR / "runs" / "LATEST"
    latest.write_text(run_id, encoding="utf-8")

    print("\n" + metrics_unlabeled.render_markdown(metrics, run_id))
    print(f"\n[run] artifacts written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
