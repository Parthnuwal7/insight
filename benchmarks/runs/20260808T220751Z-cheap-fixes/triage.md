# Accuracy triage

Run: `20260808T220751Z-cheap-fixes`

## Headline

| Metric | Value |
|---|---:|
| Aspect F1 (fuzzy match) | **0.707** |
| Aspect precision | 0.853 |
| Aspect recall | 0.604 |
| Sentiment accuracy (matched aspects only) | **0.8621** |
| Coverage ratio (predicted / gold aspects) | 0.708 |
| Aspects predicted / gold | 68 / 96 |

## By code path

This is the split that decides whether the accuracy problem is an ML problem
or an ops problem. `pyabsa` rows are real ABSA; every other route is the
14-entry keyword taxonomy.

| Route | Reviews | Aspect F1 | Precision | Recall | Sentiment acc |
|---|---:|---:|---:|---:|---:|
| `pyabsa` | 38 | 0.773 | 0.853 | 0.707 | 0.862 |
| `none:pyabsa_empty` | 8 | 0.000 | 0.000 | 0.000 | n/a |

## By linguistic phenomenon

Where the pipeline breaks down. Low F1 means aspects were missed or
invented; low sentiment accuracy with healthy F1 means the aspect was found
but the polarity was read wrong.

| Probe category | Reviews | Aspect F1 | Recall | Sentiment acc |
|---|---:|---:|---:|---:|
| `comparative` | 1 | 0.000 | 0.000 | n/a |
| `hindi` | 3 | 0.000 | 0.000 | n/a |
| `implicit_aspect` | 3 | 0.250 | 0.333 | 0.000 |
| `sarcasm` | 3 | 0.500 | 0.500 | 0.500 |
| `out_of_taxonomy` | 11 | 0.686 | 0.522 | 0.917 |
| `mixed_sentiment` | 11 | 0.739 | 0.654 | 1.000 |
| `multi_aspect` | 6 | 0.812 | 0.722 | 1.000 |
| `hinglish` | 3 | 0.909 | 0.833 | 0.600 |
| `negation` | 3 | 1.000 | 1.000 | 0.500 |
| `single_aspect_control` | 2 | 1.000 | 1.000 | 1.000 |

## Sentiment confusion (matched aspects)

| Gold | Predicted | Count |
|---|---|---:|
| Negative | Negative | 32 |
| Negative | Positive | 2 |
| Neutral | Neutral | 2 |
| Positive | Negative | 5 |
| Positive | Neutral | 1 |
| Positive | Positive | 16 |

## 12 worst reviews

Sorted by aspect F1 ascending.

**#4** (`hindi`, `none:pyabsa_empty`) - F1 0.00

> यह उत्पाद बहुत अच्छा है। गुणवत्ता शानदार है और डिलीवरी भी तेज़ थी।

- predicted: _none_
- gold: product:Positive, quality:Positive, delivery:Positive

**#10** (`out_of_taxonomy`, `none:pyabsa_empty`) - F1 0.00

> Product stopped working after a month. Tried to return but process is complicated.

- predicted: _none_
- gold: product:Negative, return process:Negative

**#15** (`hindi`, `none:pyabsa_empty`) - F1 0.00

> भोजन बहुत स्वादिष्ट था। सेवा भी उत्कृष्ट थी। मैं फिर से आऊंगा।

- predicted: _none_
- gold: food:Positive, service:Positive

**#23** (`hindi`, `none:pyabsa_empty`) - F1 0.00

> यह ऐप बहुत उपयोगी है। इंटरफेस साफ और उपयोग में आसान है।

- predicted: _none_
- gold: app:Positive, interface:Positive

**#25** (`out_of_taxonomy`, `none:pyabsa_empty`) - F1 0.00

> Privacy concerns - app requests too many unnecessary permissions. Sketchy.

- predicted: _none_
- gold: privacy:Negative, permissions:Negative

**#29** (`comparative`, `none:pyabsa_empty`) - F1 0.00

> Used to be great but recent updates made it worse. Bring back the old version!

- predicted: _none_
- gold: recent updates:Negative

**#31** (`out_of_taxonomy`, `none:pyabsa_empty`) - F1 0.00

> Export function doesn't work. Tried multiple formats, all fail. Major bug.

- predicted: _none_
- gold: export function:Negative

**#37** (`sarcasm`, `none:pyabsa_empty`) - F1 0.00

> Love how the app logs me out every ten minutes. Really adds excitement to my day.

- predicted: _none_
- gold: app:Negative

**#42** (`implicit_aspect`, `pyabsa`) - F1 0.00

> Three days in and I'm already looking for the charger twice a day.

- predicted: charger:Negative
- gold: battery life:Negative

**#43** (`implicit_aspect`, `pyabsa`) - F1 0.00

> I had to ask twice for water and once for the bill.

- predicted: water:Negative, bill:Neutral
- gold: service:Negative

**#2** (`multi_aspect`, `pyabsa`) - F1 0.50

> Terrible experience. Product arrived damaged and customer service was unhelpful. Very disappointed.

- predicted: customer service:Negative
- gold: experience:Negative, product:Negative, customer service:Negative

**#8** (`mixed_sentiment`, `pyabsa`) - F1 0.50

> Delivery was delayed by 2 weeks. Product is fine but the wait was frustrating.

- predicted: Delivery:Negative, wait:Negative
- gold: delivery:Negative, product:Positive
