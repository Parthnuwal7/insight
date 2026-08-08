# LLM judge packet - run `20260807T202423Z-baseline`

**46 reviews.** The judge is deliberately blind to pipeline output.

## How to use

1. Paste everything between the PROMPT markers below into a strong LLM
   (Claude Opus, GPT-4-class, or similar). One shot, no follow-ups.
2. Save its raw JSON reply to `benchmarks/judgments/20260807T202423Z-baseline.json`.
3. Score it:

   ```
   .venv-bench/Scripts/python.exe benchmarks/harness/score_judgments.py --run 20260807T202423Z-baseline
   ```

If the model truncates, split the review list into two halves, run each, then
concatenate the `judgments` arrays into a single JSON object before saving.

---

## ===== PROMPT START =====

You are labelling customer reviews for an aspect-based sentiment analysis benchmark.

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


INPUT REVIEWS

```json
{
  "reviews": [
    {
      "review_id": 1,
      "review": "Absolutely love this product! The quality is outstanding and delivery was super fast. Worth every penny.",
      "title": ""
    },
    {
      "review_id": 2,
      "review": "Terrible experience. Product arrived damaged and customer service was unhelpful. Very disappointed.",
      "title": ""
    },
    {
      "review_id": 3,
      "review": "Good product overall but the price is too high for what you get. Quality is decent though.",
      "title": ""
    },
    {
      "review_id": 4,
      "review": "यह उत्पाद बहुत अच्छा है। गुणवत्ता शानदार है और डिलीवरी भी तेज़ थी।",
      "title": ""
    },
    {
      "review_id": 5,
      "review": "Product quality is excellent but delivery took forever. Would buy again if shipping improves.",
      "title": ""
    },
    {
      "review_id": 6,
      "review": "The design is beautiful but functionality is lacking. Looks great but doesn't perform well.",
      "title": ""
    },
    {
      "review_id": 7,
      "review": "Misleading description. Product looks nothing like the pictures. Very disappointed.",
      "title": ""
    },
    {
      "review_id": 8,
      "review": "Delivery was delayed by 2 weeks. Product is fine but the wait was frustrating.",
      "title": ""
    },
    {
      "review_id": 9,
      "review": "Customer service needs improvement. Product is okay but support is terrible.",
      "title": ""
    },
    {
      "review_id": 10,
      "review": "Product stopped working after a month. Tried to return but process is complicated.",
      "title": ""
    },
    {
      "review_id": 11,
      "review": "Good product but packaging was poor. Item arrived safely but could have been damaged.",
      "title": ""
    },
    {
      "review_id": 12,
      "review": "Fantastic dining experience! Food quality was exceptional and service was prompt. Ambiance was perfect.",
      "title": ""
    },
    {
      "review_id": 13,
      "review": "Food was cold when served. Waited 45 minutes for our order. Very disappointing.",
      "title": ""
    },
    {
      "review_id": 14,
      "review": "Great taste but portions are too small for the price. Atmosphere is nice though.",
      "title": ""
    },
    {
      "review_id": 15,
      "review": "भोजन बहुत स्वादिष्ट था। सेवा भी उत्कृष्ट थी। मैं फिर से आऊंगा।",
      "title": ""
    },
    {
      "review_id": 16,
      "review": "Food poisoning after eating here. Never going back. Health department should inspect.",
      "title": ""
    },
    {
      "review_id": 17,
      "review": "Terrible service. Waiter was rude and forgot half our order. Food was mediocre.",
      "title": ""
    },
    {
      "review_id": 18,
      "review": "Cozy atmosphere and delicious desserts! Main course was okay but desserts were outstanding.",
      "title": ""
    },
    {
      "review_id": 19,
      "review": "Vegetarian options are limited. Food taste is good but needs more variety.",
      "title": ""
    },
    {
      "review_id": 20,
      "review": "Noise level is too high. Food is great but hard to have a conversation.",
      "title": ""
    },
    {
      "review_id": 21,
      "review": "This app is amazing! User interface is intuitive and performance is super fast. Best productivity app!",
      "title": ""
    },
    {
      "review_id": 22,
      "review": "Great features but battery drain is excessive. Fix the battery issue and it's 5 stars.",
      "title": ""
    },
    {
      "review_id": 23,
      "review": "यह ऐप बहुत उपयोगी है। इंटरफेस साफ और उपयोग में आसान है।",
      "title": ""
    },
    {
      "review_id": 24,
      "review": "Too many ads! Can't use the app without constant interruptions. Very annoying.",
      "title": ""
    },
    {
      "review_id": 25,
      "review": "Privacy concerns - app requests too many unnecessary permissions. Sketchy.",
      "title": ""
    },
    {
      "review_id": 26,
      "review": "Feature-rich but overwhelming for beginners. Needs better onboarding tutorial.",
      "title": ""
    },
    {
      "review_id": 27,
      "review": "App is good but notifications are broken. Doesn't alert when it should.",
      "title": ""
    },
    {
      "review_id": 28,
      "review": "Search function is terrible. Can't find anything easily. Navigation is confusing.",
      "title": ""
    },
    {
      "review_id": 29,
      "review": "Used to be great but recent updates made it worse. Bring back the old version!",
      "title": ""
    },
    {
      "review_id": 30,
      "review": "Glitchy animations and UI lag on older devices. Optimization needed for compatibility.",
      "title": ""
    },
    {
      "review_id": 31,
      "review": "Export function doesn't work. Tried multiple formats, all fail. Major bug.",
      "title": ""
    },
    {
      "review_id": 32,
      "review": "Cloud backup failed and lost weeks of work. No local backup option. Disaster.",
      "title": ""
    },
    {
      "review_id": 33,
      "review": "The battery does not drain quickly at all, and the screen isn't dim even outdoors.",
      "title": ""
    },
    {
      "review_id": 34,
      "review": "I wouldn't say the delivery was slow, but I can't call the packaging acceptable either.",
      "title": ""
    },
    {
      "review_id": 35,
      "review": "Not a single problem with the sound quality, though I can't recommend the build.",
      "title": ""
    },
    {
      "review_id": 36,
      "review": "Brilliant. Waited three weeks for a cable that arrived snapped in half. Outstanding work.",
      "title": ""
    },
    {
      "review_id": 37,
      "review": "Love how the app logs me out every ten minutes. Really adds excitement to my day.",
      "title": ""
    },
    {
      "review_id": 38,
      "review": "The battery life is fantastic if you enjoy charging twice before lunch.",
      "title": ""
    },
    {
      "review_id": 39,
      "review": "Delivery bohot slow thi lekin product ki quality ekdum mast hai.",
      "title": ""
    },
    {
      "review_id": 40,
      "review": "Camera thoda average hai but battery backup zabardast hai, paisa vasool.",
      "title": ""
    },
    {
      "review_id": 41,
      "review": "Service staff ka behaviour bahut kharab tha, khana theek tha.",
      "title": ""
    },
    {
      "review_id": 42,
      "review": "Three days in and I'm already looking for the charger twice a day.",
      "title": ""
    },
    {
      "review_id": 43,
      "review": "I had to ask twice for water and once for the bill.",
      "title": ""
    },
    {
      "review_id": 44,
      "review": "Everyone in the room could hear my private call through the speaker.",
      "title": ""
    },
    {
      "review_id": 45,
      "review": "The screen is bright and sharp.",
      "title": ""
    },
    {
      "review_id": 46,
      "review": "Delivery was late.",
      "title": ""
    }
  ]
}
```

## ===== PROMPT END =====
