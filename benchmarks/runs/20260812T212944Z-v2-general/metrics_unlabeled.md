# Baseline diagnostic - label-free metrics

Run: `20260812T212944Z-v2-general`

- Reviews in: **150**
- Aspect rows out: **264**
- PyABSA checkpoint loaded: **True**

## 1. Which code path produced the output?

| Route | Reviews | % |
|---|---:|---:|
| `pyabsa` | 131 | 87.3% |
| `none:pyabsa_empty` | 19 | 12.7% |

**Fallback review rate: 12.7%** (aspect rows: 0.0%)

## 2. Aspect fragmentation

| Scope | Surface forms | Concepts | Ratio |
|---|---:|---:|---:|
| All rows | 143 | 127 | 1.13x |
| PyABSA rows only | 143 | 127 | 1.13x |

Most fragmented concepts:

- **battery** (5 forms): `Battery`, `Battery life`, `battery`, `battery jaldi`, `battery life`
- **camera** (4 forms): `Camera quality`, `camera`, `quality`, `sound quality`
- **room** (4 forms): `Room`, `dining room`, `room`, `rooms`
- **breakfast** (3 forms): `Breakfast`, `breakfast`, `breakfast spread`
- **service** (3 forms): `Customer service`, `Service`, `service`
- **staff** (3 forms): `Staff`, `reception staff`, `staff`
- **sync** (3 forms): `Sync`, `data sync`, `sync`
- **ambiance** (2 forms): `Ambiance`, `ambiance`
- **check** (2 forms): `Check in`, `check`
- **customer support** (2 forms): `Customer support`, `customer support`

## 3. Translation

- Reviews expected non-English: **10**
- Of those, text actually changed: **10** (100.0%)

| Detected lang | Reviews | Translate attempted | API called | Text changed |
|---|---:|---:|---:|---:|
| `en` | 138 | 0 | 0 | 0 |
| `hi` | 10 | 10 | 10 | 10 |
| `id` | 1 | 0 | 0 | 0 |
| `so` | 1 | 0 | 0 | 0 |

## 4. Confidence distribution

- mean 0.9688 / median 0.9948 / range 0.4282-0.9988
- exactly `0.7` (the hardcoded fallback value): **0** (0.0%)

## 5. Coverage

- Aspects per review: mean **1.76**, median 2.0, max 8
- Reviews with zero aspects: **19**
- Reviews labelled `General` only: **0** (0.0%)
- Aspect rows inside the 14-bucket keyword taxonomy: **10** (3.8%)

## 6. Can production detect fallback rows without instrumentation?

Heuristic precision **0.0%**, recall **0.0%** (TP 0, FP 0, FN 19, TN 131).
