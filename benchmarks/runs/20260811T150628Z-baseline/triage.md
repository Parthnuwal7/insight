# Accuracy triage

Run: `20260811T150628Z-baseline`

## Headline

| Metric | Value |
|---|---:|
| Aspect F1 (fuzzy match) | **0.746** |
| Aspect precision | 0.863 |
| Aspect recall | 0.656 |
| Sentiment accuracy (matched aspects only) | **0.873** |
| Coverage ratio (predicted / gold aspects) | 0.76 |
| Aspects predicted / gold | 73 / 96 |

## By code path

This is the split that decides whether the accuracy problem is an ML problem
or an ops problem. `pyabsa` rows are real ABSA; every other route is the
14-entry keyword taxonomy.

| Route | Reviews | Aspect F1 | Precision | Recall | Sentiment acc |
|---|---:|---:|---:|---:|---:|
| `pyabsa` | 41 | 0.778 | 0.863 | 0.708 | 0.873 |
| `none:pyabsa_empty` | 5 | 0.000 | 0.000 | 0.000 | n/a |

## By linguistic phenomenon

Where the pipeline breaks down. Low F1 means aspects were missed or
invented; low sentiment accuracy with healthy F1 means the aspect was found
but the polarity was read wrong.

| Probe category | Reviews | Aspect F1 | Recall | Sentiment acc |
|---|---:|---:|---:|---:|
| `comparative` | 1 | 0.000 | 0.000 | n/a |
| `implicit_aspect` | 3 | 0.250 | 0.333 | 0.000 |
| `sarcasm` | 3 | 0.500 | 0.500 | 0.500 |
| `out_of_taxonomy` | 11 | 0.686 | 0.522 | 0.917 |
| `mixed_sentiment` | 11 | 0.739 | 0.654 | 1.000 |
| `multi_aspect` | 6 | 0.812 | 0.722 | 1.000 |
| `hindi` | 3 | 0.833 | 0.714 | 1.000 |
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
| Positive | Positive | 21 |

## 12 worst reviews

Sorted by aspect F1 ascending.

**#10** (`out_of_taxonomy`, `none:pyabsa_empty`) - F1 0.00

> Product stopped working after a month. Tried to return but process is complicated.

- predicted: _none_
- gold: product:Negative, return process:Negative

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

**#13** (`multi_aspect`, `pyabsa`) - F1 0.50

> Food was cold when served. Waited 45 minutes for our order. Very disappointing.

- predicted: Food:Negative, served:Negative
- gold: food:Negative, wait time:Negative

**#26** (`mixed_sentiment`, `pyabsa`) - F1 0.50

> Feature-rich but overwhelming for beginners. Needs better onboarding tutorial.

- predicted: Feature:Positive, tutorial:Negative
- gold: features:Positive, onboarding tutorial:Negative

**#36** (`sarcasm`, `pyabsa`) - F1 0.50

> Brilliant. Waited three weeks for a cable that arrived snapped in half. Outstanding work.

- predicted: cable:Negative, work:Positive
- gold: delivery:Negative, cable:Negative
