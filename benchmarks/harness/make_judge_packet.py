"""
Builds the LLM judging packet.

The judge sees ONLY the review text. It never sees what the pipeline predicted.
Showing predictions would anchor the model -- it would tend to ratify what it
was shown, inflating agreement and making the benchmark useless as a baseline.
Independent labels first, comparison second.
"""

from __future__ import annotations

import json

RUBRIC = """You are labelling customer reviews for an aspect-based sentiment analysis benchmark.

For each review, identify every ASPECT the reviewer actually evaluates, and the
sentiment they express toward that specific aspect.

RULES

1. An aspect is a thing being judged (battery, delivery speed, staff attitude,
   onboarding), not a topic the review merely mentions. If the reviewer expresses
   no stance toward it, it is not an aspect.

2. Name each aspect as a short noun phrase, lowercase, using the reviewer's own
   framing where possible. Prefer "delivery" over "the delivery was slow".
   Prefer "battery life" over "battery life issues".

3. One entry per distinct aspect. If the same aspect is praised twice, emit it once.
   If the same aspect is both praised and criticised, emit it once with the
   sentiment that dominates the reviewer's overall stance toward it.

4. Sentiment is strictly one of: Positive, Negative, Neutral.
   Judge the reviewer's INTENT, not surface vocabulary:
   - Negation flips polarity. "does not drain quickly" about battery is Positive.
   - Sarcasm flips polarity. "Brilliant, arrived snapped in half" is Negative.
   - Neutral means genuinely balanced or purely factual, not "mildly positive".

5. Include IMPLICIT aspects. If a reviewer says "already looking for the charger
   twice a day", the aspect is "battery life" with sentiment Negative, even
   though the word battery never appears.

6. For non-English or code-mixed (Hinglish) reviews, label the aspects in English
   but judge sentiment from the original meaning.

7. Evidence must be the shortest verbatim span from the review that supports your
   judgment. Copy it exactly; do not paraphrase.

8. If a review evaluates nothing, return an empty aspects list. Do not invent
   aspects to fill space.

OUTPUT FORMAT

Return ONE JSON object and nothing else. No prose before or after, no markdown
fences. It must validate against this shape:

{
  "judgments": [
    {
      "review_id": <integer, copied exactly from the input>,
      "language": "en" | "hi" | "hi-latn" | "other",
      "aspects": [
        {
          "aspect": "<short lowercase noun phrase>",
          "sentiment": "Positive" | "Negative" | "Neutral",
          "evidence": "<verbatim span from the review>"
        }
      ]
    }
  ]
}

Emit exactly one judgment object per input review, in the same order.
"""


def build_packet(review_rows: list[dict], run_id: str) -> str:
    payload = [
        {"review_id": r["id"], "review": r["review"], "title": r.get("reviews_title") or ""}
        for r in review_rows
    ]
    reviews_json = json.dumps({"reviews": payload}, indent=2, ensure_ascii=False)

    return f"""# LLM judge packet - run `{run_id}`

**{len(payload)} reviews.** The judge is deliberately blind to pipeline output.

## How to use

1. Paste everything between the PROMPT markers below into a strong LLM
   (Claude Opus, GPT-4-class, or similar). One shot, no follow-ups.
2. Save its raw JSON reply to `benchmarks/judgments/{run_id}.json`.
3. Score it:

   ```
   .venv-bench/Scripts/python.exe benchmarks/harness/score_judgments.py --run {run_id}
   ```

If the model truncates, split the review list into two halves, run each, then
concatenate the `judgments` arrays into a single JSON object before saving.

---

## ===== PROMPT START =====

{RUBRIC}

INPUT REVIEWS

```json
{reviews_json}
```

## ===== PROMPT END =====
"""
