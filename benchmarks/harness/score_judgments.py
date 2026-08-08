"""
Scores LLM judge output against pipeline predictions.

Produces the triage view: where the pipeline loses accuracy, sliced by the two
dimensions that matter for deciding what to fix -- which code path produced the
row, and which linguistic phenomenon the review exercises.

Two measurement decisions worth knowing:

* Aspect matching is fuzzy (see matching.py). Exact string comparison would
  count "battery life" against gold "battery" as both a miss and a false alarm.

* Sentiment accuracy is computed ONLY over correctly matched aspects. Measured
  unconditionally, a missed aspect and a flipped polarity blur into one number
  and you cannot tell which is moving.

Usage:
    .venv-bench/Scripts/python.exe benchmarks/harness/score_judgments.py --run <run_id>
    .venv-bench/Scripts/python.exe benchmarks/harness/score_judgments.py          # uses LATEST
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HARNESS_DIR = Path(__file__).resolve().parent
BENCH_DIR = HARNESS_DIR.parent
sys.path.insert(0, str(HARNESS_DIR))

from matching import align, normalize  # noqa: E402

SENTIMENTS = ("Positive", "Negative", "Neutral")


def load_judgments(path: Path) -> dict[int, dict]:
    """Tolerant loader -- LLMs wrap JSON in fences or add stray prose."""
    raw = path.read_text(encoding="utf-8").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise SystemExit(f"No JSON object found in {path}")
    data = json.loads(raw[start : end + 1])

    if "judgments" not in data:
        raise SystemExit(f"{path} has no top-level 'judgments' key")

    out: dict[int, dict] = {}
    for entry in data["judgments"]:
        rid = int(entry["review_id"])
        aspects = []
        for a in entry.get("aspects") or []:
            sentiment = str(a.get("sentiment", "")).strip().capitalize()
            if sentiment not in SENTIMENTS:
                sentiment = "Neutral"
            aspects.append({
                "aspect": str(a.get("aspect", "")).strip(),
                "sentiment": sentiment,
                "evidence": str(a.get("evidence", "")).strip(),
            })
        out[rid] = {"language": entry.get("language"), "aspects": aspects}
    return out


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


def score(predictions: list[dict], judgments: dict[int, dict]) -> dict[str, Any]:
    per_review = []
    confusion: Counter = Counter()
    by_route: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tot = defaultdict(int)
    missing = []

    for pred in predictions:
        rid = pred["id"]
        gold_entry = judgments.get(rid)
        if gold_entry is None:
            missing.append(rid)
            continue

        p_aspects = list(pred.get("aspects") or [])
        p_sents = list(pred.get("aspect_sentiments") or [])
        g_aspects = [a["aspect"] for a in gold_entry["aspects"]]
        g_sents = [a["sentiment"] for a in gold_entry["aspects"]]

        matched, unmatched_p, unmatched_g = align(p_aspects, g_aspects)
        tp, fp, fn = len(matched), len(unmatched_p), len(unmatched_g)

        sent_correct = 0
        for pi, gi in matched:
            ps = p_sents[pi] if pi < len(p_sents) else None
            gs = g_sents[gi] if gi < len(g_sents) else None
            confusion[(gs, ps)] += 1
            if ps == gs:
                sent_correct += 1

        route = pred.get("route", "unknown")
        category = pred.get("probe_category", "uncategorised")
        for bucket, key in ((by_route, route), (by_category, category)):
            bucket[key]["tp"] += tp
            bucket[key]["fp"] += fp
            bucket[key]["fn"] += fn
            bucket[key]["sent_correct"] += sent_correct
            bucket[key]["sent_total"] += len(matched)
            bucket[key]["reviews"] += 1

        tot["tp"] += tp
        tot["fp"] += fp
        tot["fn"] += fn
        tot["sent_correct"] += sent_correct
        tot["sent_total"] += len(matched)
        tot["pred_aspects"] += len(p_aspects)
        tot["gold_aspects"] += len(g_aspects)

        r = _prf(tp, fp, fn)
        per_review.append({
            "review_id": rid,
            "probe_category": category,
            "route": route,
            "review": pred["review"][:110],
            "predicted": [f"{a}:{p_sents[i] if i < len(p_sents) else '?'}"
                          for i, a in enumerate(p_aspects)],
            "gold": [f"{a['aspect']}:{a['sentiment']}" for a in gold_entry["aspects"]],
            "f1": r["f1"],
            "aspect_tp": tp, "aspect_fp": fp, "aspect_fn": fn,
            "sentiment_correct": sent_correct,
            "sentiment_of_matched": len(matched),
        })

    def summarise(d: dict[str, int]) -> dict[str, Any]:
        out = _prf(d["tp"], d["fp"], d["fn"])
        out["reviews"] = d["reviews"]
        out["sentiment_accuracy"] = (
            round(d["sent_correct"] / d["sent_total"], 4) if d["sent_total"] else None
        )
        out["sentiment_n"] = d["sent_total"]
        return out

    overall = _prf(tot["tp"], tot["fp"], tot["fn"])
    overall["sentiment_accuracy"] = (
        round(tot["sent_correct"] / tot["sent_total"], 4) if tot["sent_total"] else None
    )
    overall["sentiment_n"] = tot["sent_total"]
    overall["coverage_ratio"] = (
        round(tot["pred_aspects"] / tot["gold_aspects"], 3) if tot["gold_aspects"] else None
    )
    overall["predicted_aspects"] = tot["pred_aspects"]
    overall["gold_aspects"] = tot["gold_aspects"]

    return {
        "overall": overall,
        "by_route": {k: summarise(v) for k, v in sorted(by_route.items())},
        "by_probe_category": {k: summarise(v) for k, v in sorted(by_category.items())},
        "sentiment_confusion": {f"gold={g}|pred={p}": n for (g, p), n in sorted(
            confusion.items(), key=lambda kv: str(kv[0]))},
        "per_review": sorted(per_review, key=lambda r: r["f1"]),
        "reviews_missing_judgment": missing,
    }


def render_triage(s: dict[str, Any], run_id: str) -> str:
    o = s["overall"]
    lines = [
        "# Accuracy triage",
        "",
        f"Run: `{run_id}`",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Aspect F1 (fuzzy match) | **{o['f1']:.3f}** |",
        f"| Aspect precision | {o['precision']:.3f} |",
        f"| Aspect recall | {o['recall']:.3f} |",
        f"| Sentiment accuracy (matched aspects only) | "
        f"**{o['sentiment_accuracy'] if o['sentiment_accuracy'] is not None else 'n/a'}** |",
        f"| Coverage ratio (predicted / gold aspects) | {o['coverage_ratio']} |",
        f"| Aspects predicted / gold | {o['predicted_aspects']} / {o['gold_aspects']} |",
        "",
        "## By code path",
        "",
        "This is the split that decides whether the accuracy problem is an ML problem",
        "or an ops problem. `pyabsa` rows are real ABSA; every other route is the",
        "14-entry keyword taxonomy.",
        "",
        "| Route | Reviews | Aspect F1 | Precision | Recall | Sentiment acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for route, m in sorted(s["by_route"].items(), key=lambda kv: -kv[1]["reviews"]):
        sa = f"{m['sentiment_accuracy']:.3f}" if m["sentiment_accuracy"] is not None else "n/a"
        lines.append(
            f"| `{route}` | {m['reviews']} | {m['f1']:.3f} | "
            f"{m['precision']:.3f} | {m['recall']:.3f} | {sa} |"
        )

    lines += [
        "",
        "## By linguistic phenomenon",
        "",
        "Where the pipeline breaks down. Low F1 means aspects were missed or",
        "invented; low sentiment accuracy with healthy F1 means the aspect was found",
        "but the polarity was read wrong.",
        "",
        "| Probe category | Reviews | Aspect F1 | Recall | Sentiment acc |",
        "|---|---:|---:|---:|---:|",
    ]
    for cat, m in sorted(s["by_probe_category"].items(), key=lambda kv: kv[1]["f1"]):
        sa = f"{m['sentiment_accuracy']:.3f}" if m["sentiment_accuracy"] is not None else "n/a"
        lines.append(
            f"| `{cat}` | {m['reviews']} | {m['f1']:.3f} | {m['recall']:.3f} | {sa} |"
        )

    lines += ["", "## Sentiment confusion (matched aspects)", "",
              "| Gold | Predicted | Count |", "|---|---|---:|"]
    for key, n in s["sentiment_confusion"].items():
        gold, pred = key.split("|")
        lines.append(f"| {gold.split('=')[1]} | {pred.split('=')[1]} | {n} |")

    lines += ["", "## 12 worst reviews", "",
              "Sorted by aspect F1 ascending.", ""]
    for r in s["per_review"][:12]:
        lines += [
            f"**#{r['review_id']}** (`{r['probe_category']}`, `{r['route']}`) - F1 {r['f1']:.2f}",
            "",
            f"> {r['review']}",
            "",
            f"- predicted: {', '.join(r['predicted']) or '_none_'}",
            f"- gold: {', '.join(r['gold']) or '_none_'}",
            "",
        ]

    if s["reviews_missing_judgment"]:
        lines += [
            "## Warning",
            "",
            f"No judgment returned for review ids: {s['reviews_missing_judgment']}. "
            "These were excluded from all metrics.",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="run id (defaults to runs/LATEST)")
    parser.add_argument("--judgments", default=None, help="path to judge JSON")
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

    judge_path = Path(args.judgments) if args.judgments else BENCH_DIR / "judgments" / f"{run_id}.json"
    if not judge_path.exists():
        raise SystemExit(
            f"Judgments not found: {judge_path}\n"
            f"Run the prompt in {run_dir / 'judge_packet.md'} and save the reply there."
        )

    predictions = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    judgments = load_judgments(judge_path)

    result = score(predictions, judgments)
    (run_dir / "metrics_labeled.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    triage = render_triage(result, run_id)
    (run_dir / "triage.md").write_text(triage, encoding="utf-8")

    print(triage)
    print(f"\n[score] wrote {run_dir / 'metrics_labeled.json'}")
    print(f"[score] wrote {run_dir / 'triage.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
