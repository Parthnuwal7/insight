# Accuracy triage

Run: `20260812T212944Z-v2-general`

## Headline

| Metric | Value |
|---|---:|
| Aspect F1 (fuzzy match) | **0.752** |
| Aspect precision | 0.905 |
| Aspect recall | 0.642 |
| Sentiment accuracy (matched aspects only) | **0.9791** |
| Coverage ratio (predicted / gold aspects) | 0.71 |
| Aspects predicted / gold | 264 / 372 |

## By code path

This is the split that decides whether the accuracy problem is an ML problem
or an ops problem. `pyabsa` rows are real ABSA; every other route is the
14-entry keyword taxonomy.

| Route | Reviews | Aspect F1 | Precision | Recall | Sentiment acc |
|---|---:|---:|---:|---:|---:|
| `pyabsa` | 131 | 0.904 | 0.905 | 0.902 | 0.979 |
| `none:pyabsa_empty` | 19 | 0.000 | 0.000 | 0.000 | n/a |

## By linguistic phenomenon

Where the pipeline breaks down. Low F1 means aspects were missed or
invented; low sentiment accuracy with healthy F1 means the aspect was found
but the polarity was read wrong.

| Probe category | Reviews | Aspect F1 | Recall | Sentiment acc |
|---|---:|---:|---:|---:|
| `long_form` | 25 | 0.287 | 0.179 | 0.880 |
| `mixed_sentiment` | 30 | 0.898 | 0.883 | 1.000 |
| `hindi` | 10 | 0.923 | 0.900 | 1.000 |
| `single_aspect_control` | 30 | 0.923 | 1.000 | 1.000 |
| `multi_aspect` | 45 | 0.930 | 0.912 | 0.989 |
| `hinglish` | 10 | 1.000 | 1.000 | 0.950 |

## Sentiment confusion (matched aspects)

| Gold | Predicted | Count |
|---|---|---:|
| Negative | Negative | 100 |
| Negative | Neutral | 2 |
| Negative | Positive | 2 |
| Positive | Neutral | 1 |
| Positive | Positive | 134 |

## 12 worst reviews

Sorted by aspect F1 ascending.

**#22** (`long_form`, `none:pyabsa_empty`) - F1 0.00

> I ordered this set for my new apartment and I have mixed things to say. The delivery was genuinely impressive,

- predicted: _none_
- gold: delivery:Positive, Packaging:Positive, quality:Negative, Customer service:Positive, refund:Negative, price:Positive

**#23** (`long_form`, `none:pyabsa_empty`) - F1 0.00

> This was my third order from this seller and sadly the worst. The shipping took nineteen days with no tracking

- predicted: _none_
- gold: shipping:Negative, packaging:Negative, customer support:Negative, build quality:Positive

**#26** (`long_form`, `none:pyabsa_empty`) - F1 0.00

> Ordering was simple enough but everything after that went wrong. The delivery date moved three times. When it 

- predicted: _none_
- gold: delivery:Negative, product:Negative, seller:Positive, finish:Positive, courier:Negative

**#52** (`long_form`, `none:pyabsa_empty`) - F1 0.00

> I have been using this app daily for about eight months so I feel qualified to review it properly. The interfa

- predicted: _none_
- gold: interface:Positive, animations:Positive, sync:Positive, battery consumption:Negative, subscription price:Negative, customer support:Positive

**#53** (`long_form`, `none:pyabsa_empty`) - F1 0.00

> This started as a great app and has slowly got worse. Two years ago the performance was excellent and there we

- predicted: _none_
- gold: performance:Negative, ads:Negative, search function:Positive, offline mode:Positive, notification system:Negative

**#54** (`long_form`, `none:pyabsa_empty`) - F1 0.00

> Switched to this from a competitor last month and I am impressed. The onboarding was quick and it imported all

- predicted: _none_
- gold: onboarding:Positive, interface:Positive, performance:Positive, free tier:Positive, dark mode:Positive, tablet layout:Negative

**#55** (`long_form`, `none:pyabsa_empty`) - F1 0.00

> Mixed review because the app does one thing brilliantly and everything else poorly. The core editor is fast, s

- predicted: _none_
- gold: editor:Positive, cloud sync:Negative, settings menu:Negative, customer support:Negative, pricing:Positive

**#82** (`long_form`, `none:pyabsa_empty`) - F1 0.00

> We booked here for an anniversary dinner and it mostly lived up to expectations. The ambiance is genuinely spe

- predicted: _none_
- gold: ambiance:Positive, server:Positive, starters:Positive, lamb:Negative, wine list:Negative, staff:Positive

**#83** (`long_form`, `none:pyabsa_empty`) - F1 0.00

> Sadly this place has gone downhill since it changed hands. We used to come monthly. The menu has been cut in h

- predicted: _none_
- gold: menu:Negative, prices:Negative, food quality:Negative, service:Negative, dining room:Positive, location:Positive

**#86** (`long_form`, `none:pyabsa_empty`) - F1 0.00

> Booked a table for eight for a birthday and the restaurant handled it badly. Despite booking three weeks ahead

- predicted: _none_
- gold: manager:Positive, food:Positive, platters:Positive, service:Positive, noise level:Negative

**#112** (`long_form`, `none:pyabsa_empty`) - F1 0.00

> Stayed here for four nights on a work trip and it was a solid choice. The location is the standout feature, fi

- predicted: _none_
- gold: location:Positive, check in:Positive, room:Positive, bed:Positive, bathroom:Positive, breakfast:Negative, wifi:Positive

**#113** (`long_form`, `none:pyabsa_empty`) - F1 0.00

> I would not stay here again. The photographs online are clearly several years old. Our room was tired, the car

- predicted: _none_
- gold: room:Negative, carpet:Negative, reception:Positive, heating:Negative, Breakfast:Positive, location:Positive
