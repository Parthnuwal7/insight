# Baseline diagnostic - label-free metrics

Run: `20260808T220751Z-cheap-fixes`

- Reviews in: **46**
- Aspect rows out: **68**
- PyABSA checkpoint loaded: **True**

## 1. Which code path produced the output?

| Route | Reviews | % |
|---|---:|---:|
| `pyabsa` | 38 | 82.6% |
| `none:pyabsa_empty` | 8 | 17.4% |

**Fallback review rate: 17.4%** (aspect rows: 0.0%)

## 2. Aspect fragmentation

| Scope | Surface forms | Concepts | Ratio |
|---|---:|---:|---:|
| All rows | 51 | 44 | 1.16x |
| PyABSA rows only | 51 | 44 | 1.16x |

Most fragmented concepts:

- **food** (5 forms): `Food`, `Food quality`, `Quality`, `quality`, `sound quality`
- **battery** (4 forms): `battery`, `battery backup`, `battery drain`, `battery life`
- **service** (3 forms): `Customer service`, `customer service`, `service`
- **atmosphere** (2 forms): `Atmosphere`, `atmosphere`
- **delivery** (2 forms): `Delivery`, `delivery`

## 3. Translation

- Reviews expected non-English: **6**
- Of those, text actually changed: **0** (0.0%)

| Detected lang | Reviews | Translate attempted | API called | Text changed |
|---|---:|---:|---:|---:|
| `en` | 42 | 0 | 0 | 0 |
| `hi` | 3 | 3 | 3 | 0 |
| `id` | 1 | 0 | 0 | 0 |

## 4. Confidence distribution

- mean 0.9223 / median 0.9879 / range 0.4991-0.9986
- exactly `0.7` (the hardcoded fallback value): **0** (0.0%)

## 5. Coverage

- Aspects per review: mean **1.48**, median 2.0, max 4
- Reviews with zero aspects: **8**
- Reviews labelled `General` only: **0** (0.0%)
- Aspect rows inside the 14-bucket keyword taxonomy: **4** (5.9%)

## 6. Can production detect fallback rows without instrumentation?

Heuristic precision **0.0%**, recall **0.0%** (TP 0, FP 0, FN 8, TN 38).
