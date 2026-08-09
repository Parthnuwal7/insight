"""
Label-free diagnostics.

Everything here is computable without ground truth. These metrics answer
"is the pipeline structurally doing what it claims?" rather than "is it
correct?", and they are deliberately the first thing measured: if a large
share of output never touched the ABSA model, an accuracy number computed
over the whole set is measuring a blend of two different systems.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any

from instrument import FALLBACK_CONFIDENCE, FALLBACK_TAXONOMY
from matching import cluster, normalize

# Legacy wrapper-derived route names, kept so older runs still score.
FALLBACK_ROUTES = {
    "pyabsa_empty_to_fallback",
    "pyabsa_error_to_fallback",
    "model_unavailable_fallback",
    "fallback_unattributed",
    "keyword_fallback",
}


def is_non_absa(route: str) -> bool:
    """True when a row did not come from real aspect-based sentiment analysis.

    Covers the legacy wrapper route names and the provenance the pipeline now
    reports natively ('keyword_fallback', and 'none:<reason>' for reviews it
    declined to guess at).
    """
    return route in FALLBACK_ROUTES or route.startswith("none:")


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


def compute(review_rows: list[dict], aspect_rows: list[dict], model_loaded: bool | None) -> dict[str, Any]:
    n_reviews = len(review_rows)
    n_aspects = len(aspect_rows)

    # ---- route distribution -------------------------------------------
    routes = Counter(r["route"] for r in review_rows)
    fallback_reviews = sum(c for r, c in routes.items() if is_non_absa(r))
    fallback_aspect_rows = sum(1 for a in aspect_rows if is_non_absa(a["route"]))

    # ---- fragmentation -------------------------------------------------
    surface_forms = [a["aspect"] for a in aspect_rows if a.get("aspect")]
    clusters = cluster(surface_forms)
    distinct_forms = len({normalize(f) for f in surface_forms})
    frag_detail = sorted(
        ({"concept": k, "surface_forms": v, "n_forms": len(v)} for k, v in clusters.items() if len(v) > 1),
        key=lambda d: -d["n_forms"],
    )

    # Fragmentation only within genuine ABSA output; the keyword fallback emits
    # a fixed vocabulary and cannot fragment, so including it dilutes the signal.
    absa_forms = [a["aspect"] for a in aspect_rows if a["route"] == "pyabsa" and a.get("aspect")]
    absa_clusters = cluster(absa_forms)
    absa_distinct = len({normalize(f) for f in absa_forms})

    # ---- translation ---------------------------------------------------
    by_lang: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in review_rows:
        lang = r.get("detected_lang") or "unknown"
        by_lang[lang]["reviews"] += 1
        by_lang[lang]["translate_attempted"] += int(bool(r.get("translate_attempted")))
        by_lang[lang]["api_called"] += int(bool(r.get("api_called")))
        by_lang[lang]["text_changed"] += int(bool(r.get("text_changed")))

    expected_non_en = [r for r in review_rows if r.get("lang_expected") in ("hi", "hi-latn")]
    non_en_translated = sum(1 for r in expected_non_en if r.get("text_changed"))

    # ---- confidence ----------------------------------------------------
    confs = [float(a["confidence"]) for a in aspect_rows if a.get("confidence") is not None]
    exactly_fallback_conf = sum(1 for c in confs if abs(c - FALLBACK_CONFIDENCE) < 1e-9)

    # ---- coverage ------------------------------------------------------
    per_review_counts = [r["n_aspects"] for r in review_rows]
    general_only = sum(
        1 for r in review_rows
        if r["n_aspects"] == 1 and r.get("aspects") and normalize(r["aspects"][0]) == "general"
    )
    zero_aspect = sum(1 for c in per_review_counts if c == 0)

    in_taxonomy = sum(1 for a in aspect_rows if a.get("aspect") in FALLBACK_TAXONOMY)

    # ---- heuristic provenance check ------------------------------------
    truth = [is_non_absa(r["route"]) for r in review_rows]
    guess = [bool(r.get("heuristic_fallback")) for r in review_rows]
    tp = sum(1 for t, g in zip(truth, guess) if t and g)
    fp = sum(1 for t, g in zip(truth, guess) if not t and g)
    fn = sum(1 for t, g in zip(truth, guess) if t and not g)
    tn = sum(1 for t, g in zip(truth, guess) if not t and not g)

    return {
        "totals": {
            "reviews_in": n_reviews,
            "aspect_rows_out": n_aspects,
            "pyabsa_model_loaded": model_loaded,
        },
        "routes": {
            "counts": dict(routes),
            "percent": {k: _pct(v, n_reviews) for k, v in routes.items()},
            "fallback_review_rate_pct": _pct(fallback_reviews, n_reviews),
            "fallback_aspect_row_rate_pct": _pct(fallback_aspect_rows, n_aspects),
        },
        "fragmentation": {
            "all_rows": {
                "distinct_surface_forms": distinct_forms,
                "distinct_concepts": len(clusters),
                "fragmentation_ratio": round(distinct_forms / len(clusters), 2) if clusters else 0.0,
            },
            "pyabsa_rows_only": {
                "distinct_surface_forms": absa_distinct,
                "distinct_concepts": len(absa_clusters),
                "fragmentation_ratio": round(absa_distinct / len(absa_clusters), 2) if absa_clusters else 0.0,
            },
            "fragmented_concepts": frag_detail[:15],
        },
        "translation": {
            "by_detected_language": {k: dict(v) for k, v in sorted(by_lang.items())},
            "expected_non_english_reviews": len(expected_non_en),
            "of_those_text_actually_changed": non_en_translated,
            "translation_effective_rate_pct": _pct(non_en_translated, len(expected_non_en)),
        },
        "confidence": {
            "n": len(confs),
            "mean": round(statistics.fmean(confs), 4) if confs else None,
            "median": round(statistics.median(confs), 4) if confs else None,
            "min": round(min(confs), 4) if confs else None,
            "max": round(max(confs), 4) if confs else None,
            "exactly_0.7_count": exactly_fallback_conf,
            "exactly_0.7_pct": _pct(exactly_fallback_conf, len(confs)),
        },
        "coverage": {
            "aspects_per_review_mean": round(statistics.fmean(per_review_counts), 2) if per_review_counts else 0,
            "aspects_per_review_median": statistics.median(per_review_counts) if per_review_counts else 0,
            "aspects_per_review_max": max(per_review_counts) if per_review_counts else 0,
            "reviews_with_zero_aspects": zero_aspect,
            "reviews_labelled_general_only": general_only,
            "reviews_labelled_general_only_pct": _pct(general_only, n_reviews),
            "aspect_rows_inside_fallback_taxonomy": in_taxonomy,
            "aspect_rows_inside_fallback_taxonomy_pct": _pct(in_taxonomy, n_aspects),
        },
        "heuristic_provenance_check": {
            "note": "Can a consumer detect fallback rows from the API payload alone?",
            "true_positive": tp, "false_positive": fp,
            "false_negative": fn, "true_negative": tn,
            "precision_pct": _pct(tp, tp + fp),
            "recall_pct": _pct(tp, tp + fn),
        },
    }


def render_markdown(m: dict[str, Any], run_id: str) -> str:
    t, r, f = m["totals"], m["routes"], m["fragmentation"]
    tr, c, cov = m["translation"], m["confidence"], m["coverage"]
    h = m["heuristic_provenance_check"]

    lines = [
        f"# Baseline diagnostic - label-free metrics",
        "",
        f"Run: `{run_id}`",
        "",
        f"- Reviews in: **{t['reviews_in']}**",
        f"- Aspect rows out: **{t['aspect_rows_out']}**",
        f"- PyABSA checkpoint loaded: **{t['pyabsa_model_loaded']}**",
        "",
        "## 1. Which code path produced the output?",
        "",
        "| Route | Reviews | % |",
        "|---|---:|---:|",
    ]
    for route, count in sorted(r["counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{route}` | {count} | {r['percent'][route]}% |")
    lines += [
        "",
        f"**Fallback review rate: {r['fallback_review_rate_pct']}%** "
        f"(aspect rows: {r['fallback_aspect_row_rate_pct']}%)",
        "",
        "## 2. Aspect fragmentation",
        "",
        "| Scope | Surface forms | Concepts | Ratio |",
        "|---|---:|---:|---:|",
        f"| All rows | {f['all_rows']['distinct_surface_forms']} | "
        f"{f['all_rows']['distinct_concepts']} | {f['all_rows']['fragmentation_ratio']}x |",
        f"| PyABSA rows only | {f['pyabsa_rows_only']['distinct_surface_forms']} | "
        f"{f['pyabsa_rows_only']['distinct_concepts']} | {f['pyabsa_rows_only']['fragmentation_ratio']}x |",
        "",
    ]
    if f["fragmented_concepts"]:
        lines += ["Most fragmented concepts:", ""]
        for item in f["fragmented_concepts"][:10]:
            forms = ", ".join(f"`{x}`" for x in item["surface_forms"])
            lines.append(f"- **{item['concept']}** ({item['n_forms']} forms): {forms}")
        lines.append("")
    else:
        lines += ["_No concept appeared under more than one surface form._", ""]

    lines += [
        "## 3. Translation",
        "",
        f"- Reviews expected non-English: **{tr['expected_non_english_reviews']}**",
        f"- Of those, text actually changed: **{tr['of_those_text_actually_changed']}** "
        f"({tr['translation_effective_rate_pct']}%)",
        "",
        "| Detected lang | Reviews | Translate attempted | API called | Text changed |",
        "|---|---:|---:|---:|---:|",
    ]
    for lang, d in tr["by_detected_language"].items():
        lines.append(
            f"| `{lang}` | {d.get('reviews', 0)} | {d.get('translate_attempted', 0)} | "
            f"{d.get('api_called', 0)} | {d.get('text_changed', 0)} |"
        )

    lines += [
        "",
        "## 4. Confidence distribution",
        "",
        f"- mean {c['mean']} / median {c['median']} / range {c['min']}-{c['max']}",
        f"- exactly `0.7` (the hardcoded fallback value): **{c['exactly_0.7_count']}** "
        f"({c['exactly_0.7_pct']}%)",
        "",
        "## 5. Coverage",
        "",
        f"- Aspects per review: mean **{cov['aspects_per_review_mean']}**, "
        f"median {cov['aspects_per_review_median']}, max {cov['aspects_per_review_max']}",
        f"- Reviews with zero aspects: **{cov['reviews_with_zero_aspects']}**",
        f"- Reviews labelled `General` only: **{cov['reviews_labelled_general_only']}** "
        f"({cov['reviews_labelled_general_only_pct']}%)",
        f"- Aspect rows inside the 14-bucket keyword taxonomy: "
        f"**{cov['aspect_rows_inside_fallback_taxonomy']}** "
        f"({cov['aspect_rows_inside_fallback_taxonomy_pct']}%)",
        "",
        "## 6. Can production detect fallback rows without instrumentation?",
        "",
        f"Heuristic precision **{h['precision_pct']}%**, recall **{h['recall_pct']}%** "
        f"(TP {h['true_positive']}, FP {h['false_positive']}, "
        f"FN {h['false_negative']}, TN {h['true_negative']}).",
        "",
    ]
    return "\n".join(lines)
