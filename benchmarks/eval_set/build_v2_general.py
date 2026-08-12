"""Builds the v2 general evaluation set: reviews CSV + gold judgments JSON.

WHY THIS IS A SCRIPT AND NOT TWO HAND-WRITTEN FILES
---------------------------------------------------
The harness reads reviews from `eval_set/*.csv` and gold labels from
`judgments/*.json`, joined on `review_id`. Maintaining those by hand lets them
drift: a review gets reworded, its gold evidence no longer appears in the
text, and the metric quietly measures something other than what the file says.
Here both artefacts are emitted from one list of records, so they cannot
disagree, and `validate()` refuses to emit anything if an evidence span is not
a literal substring of its review.

PROVENANCE -- READ THIS BEFORE QUOTING ANY NUMBER FROM THIS SET
--------------------------------------------------------------
These reviews are CONSTRUCTED, not sampled from real customer data, and the
gold labels were authored alongside them. That makes this a test of "does
extraction recover the aspects a review was built to contain", which is a
weaker claim than agreement with independently-annotated real reviews. It is
still a real measurement -- the reviews are written so the aspects are
unambiguous, and the extractor never sees the labels -- but it must not be
described as real-world accuracy, and a single author's labels carry no
inter-annotator agreement figure.

RELATIONSHIP TO v1
------------------
`eval_reviews_v1.csv` is the ADVERSARIAL probe set: 46 reviews deliberately
weighted toward sarcasm, implicit aspects, comparatives and out-of-taxonomy
mentions. It answers "where does the extractor break", and stays the internal
regression gate.

This set answers a different question -- "how does the extractor do on
ordinary review traffic" -- so it deliberately EXCLUDES those categories.
Reporting v1's number as headline accuracy understates the system on realistic
input; reporting this one as though it covered the hard cases would overstate
it. They are not interchangeable, and neither supersedes the other.

COMPOSITION (150 reviews, 5 domains x 30)
-----------------------------------------
Per domain: 6 single_aspect_control, 9 multi_aspect, 6 mixed_sentiment,
5 long_form, 2 hindi, 2 hinglish. Long-form reviews are 60-110 words carrying
4-7 aspects each, which is where multi-sentence handling and the
translate-then-extract path are actually exercised.

Usage:
    python benchmarks/eval_set/build_v2_general.py
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
BENCH_DIR = EVAL_DIR.parent
CSV_OUT = EVAL_DIR / "eval_reviews_v2_general.csv"
GOLD_OUT = BENCH_DIR / "judgments" / "eval_v2_general_gold.json"

VALID_SENTIMENTS = {"Positive", "Negative", "Neutral"}
# Categories v1 owns and this set deliberately excludes -- see module docstring.
EXCLUDED_CATEGORIES = {"sarcasm", "implicit_aspect", "comparative", "out_of_taxonomy",
                       "negation"}

sys.path.insert(0, str(EVAL_DIR))

RECORDS: list[dict] = []


def add(review, title, domain, category, lang, aspects):
    RECORDS.append({
        "review": review,
        "title": title,
        "domain": domain,
        "probe_category": category,
        "lang": lang,
        "aspects": [{"aspect": a, "sentiment": s, "evidence": e} for a, s, e in aspects],
    })


def collect() -> list[dict]:
    import _v2_part1_ecommerce
    import _v2_part2_app
    import _v2_part3_restaurant
    import _v2_part4_hotel
    import _v2_part5_electronics

    for module in (_v2_part1_ecommerce, _v2_part2_app, _v2_part3_restaurant,
                   _v2_part4_hotel, _v2_part5_electronics):
        module.load(add)
    return RECORDS


def validate(records: list[dict]) -> None:
    """Refuse to emit a set that would silently measure the wrong thing.

    Every check here corresponds to a way the two artefacts could disagree
    with each other or with their own documentation. A failure raises rather
    than warns: a benchmark that is quietly wrong is worse than no benchmark,
    because its number still gets quoted.
    """
    errors: list[str] = []

    if not records:
        raise SystemExit("no records collected -- part modules failed to load")

    seen_reviews: dict[str, int] = {}
    for i, rec in enumerate(records, start=1):
        review = rec["review"]

        # 1. Evidence must be literally present. This is the check that stops
        #    reviews and labels drifting apart when a review is reworded.
        for asp in rec["aspects"]:
            if asp["evidence"] not in review:
                errors.append(
                    f"review {i}: evidence {asp['evidence']!r} is not a substring "
                    f"of the review text"
                )
            if asp["sentiment"] not in VALID_SENTIMENTS:
                errors.append(f"review {i}: bad sentiment {asp['sentiment']!r}")

        # 2. Every review must carry at least one gold aspect, or it silently
        #    contributes nothing but still inflates the denominator.
        if not rec["aspects"]:
            errors.append(f"review {i}: no gold aspects")

        # 3. Aspect names must be unique within a review -- a duplicate would
        #    double-count in both precision and recall.
        names = [a["aspect"].lower() for a in rec["aspects"]]
        if len(names) != len(set(names)):
            errors.append(f"review {i}: duplicate aspect names {names}")

        # 4. Duplicate review text across the set would let one memorised
        #    result score twice.
        if review in seen_reviews:
            errors.append(f"review {i}: duplicate of review {seen_reviews[review]}")
        seen_reviews[review] = i

        # 5. Category discipline: this set must not quietly absorb v1's
        #    adversarial categories, or the two sets stop meaning different
        #    things and the headline number drifts without anyone noticing.
        if rec["probe_category"] in EXCLUDED_CATEGORIES:
            errors.append(
                f"review {i}: category {rec['probe_category']!r} belongs to the "
                f"v1 adversarial set, not the general set"
            )

        # 6. A review labelled mixed_sentiment must actually be mixed, or the
        #    per-category breakdown reports something other than its name.
        sentiments = {a["sentiment"] for a in rec["aspects"]}
        if rec["probe_category"] == "mixed_sentiment" and len(sentiments) < 2:
            errors.append(
                f"review {i}: labelled mixed_sentiment but all aspects are "
                f"{sentiments}"
            )
        if rec["probe_category"] == "single_aspect_control" and len(rec["aspects"]) != 1:
            errors.append(
                f"review {i}: labelled single_aspect_control but has "
                f"{len(rec['aspects'])} aspects"
            )
        if rec["probe_category"] == "multi_aspect" and len(rec["aspects"]) < 2:
            errors.append(f"review {i}: labelled multi_aspect but has one aspect")

        # 7. Hindi gold must name aspects in ENGLISH (evidence stays in the
        #    source script). The pipeline translates hi->en before extracting,
        #    so a Devanagari aspect name can never be matched and scores the
        #    review 0.00 no matter how well extraction did -- a labelling
        #    error that looks exactly like a model failure. This caught a real
        #    mistake in the first draft of this set; v1 already used the
        #    English convention.
        if rec["lang"] == "hi":
            for asp in rec["aspects"]:
                if any("ऀ" <= ch <= "ॿ" for ch in asp["aspect"]):
                    errors.append(
                        f"review {i}: Hindi gold aspect {asp['aspect']!r} is in "
                        f"Devanagari; aspect names must be English (evidence "
                        f"stays Devanagari)"
                    )

    if errors:
        raise SystemExit(
            f"eval set validation FAILED ({len(errors)} problem(s)):\n  "
            + "\n  ".join(errors)
        )


def emit(records: list[dict]) -> None:
    start = date(2026, 1, 1)
    rows = []
    judgments = []

    for i, rec in enumerate(records, start=1):
        # Dates spread across a year so any downstream time-series analysis
        # has something real to bucket, rather than 150 rows on one day.
        row_date = start + timedelta(days=(i * 2) % 365)
        rows.append({
            "id": i,
            "review": rec["review"],
            "reviews_title": rec["title"],
            "date": row_date.isoformat(),
            "user_id": f"u{i:03d}",
            "source": f"constructed_{rec['domain']}",
            "probe_category": rec["probe_category"],
            "lang_expected": rec["lang"],
        })
        judgments.append({
            "review_id": i,
            "language": rec["lang"],
            "aspects": rec["aspects"],
        })

    with CSV_OUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    GOLD_OUT.parent.mkdir(parents=True, exist_ok=True)
    with GOLD_OUT.open("w", encoding="utf-8") as fh:
        json.dump({"judgments": judgments}, fh, indent=2, ensure_ascii=False)


def summarise(records: list[dict]) -> None:
    from collections import Counter

    by_cat = Counter(r["probe_category"] for r in records)
    by_domain = Counter(r["domain"] for r in records)
    by_lang = Counter(r["lang"] for r in records)
    sentiments = Counter(a["sentiment"] for r in records for a in r["aspects"])
    n_aspects = sum(len(r["aspects"]) for r in records)
    words = [len(r["review"].split()) for r in records]

    print(f"reviews          : {len(records)}")
    print(f"gold aspects     : {n_aspects} ({n_aspects / len(records):.2f} per review)")
    print(f"words per review : min {min(words)}, median "
          f"{sorted(words)[len(words) // 2]}, max {max(words)}")
    print(f"by domain        : {dict(sorted(by_domain.items()))}")
    print(f"by category      : {dict(sorted(by_cat.items()))}")
    print(f"by language      : {dict(sorted(by_lang.items()))}")
    print(f"gold sentiment   : {dict(sorted(sentiments.items()))}")
    print()
    print(f"wrote {CSV_OUT.relative_to(BENCH_DIR.parent)}")
    print(f"wrote {GOLD_OUT.relative_to(BENCH_DIR.parent)}")


if __name__ == "__main__":
    records = collect()
    validate(records)
    emit(records)
    summarise(records)
