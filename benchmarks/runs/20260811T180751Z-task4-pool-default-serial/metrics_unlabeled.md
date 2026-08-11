# Baseline diagnostic - label-free metrics

Run: `20260811T180751Z-task4-pool-default-serial`

- Reviews in: **46**
- Aspect rows out: **73**
- PyABSA checkpoint loaded: **True**

## 1. Which code path produced the output?

| Route | Reviews | % |
|---|---:|---:|
| `pyabsa` | 41 | 89.1% |
| `none:pyabsa_empty` | 5 | 10.9% |

**Fallback review rate: 10.9%** (aspect rows: 0.0%)

## 2. Aspect fragmentation

| Scope | Surface forms | Concepts | Ratio |
|---|---:|---:|---:|
| All rows | 52 | 43 | 1.21x |
| PyABSA rows only | 52 | 43 | 1.21x |

Most fragmented concepts:

- **food** (6 forms): `Food`, `Food quality`, `Quality`, `food`, `quality`, `sound quality`
- **service** (5 forms): `Customer service`, `Service`, `Service staff`, `customer service`, `service`
- **battery** (4 forms): `battery`, `battery backup`, `battery drain`, `battery life`
- **atmosphere** (2 forms): `Atmosphere`, `atmosphere`
- **delivery** (2 forms): `Delivery`, `delivery`
- **interface** (2 forms): `Interface`, `User interface`

## 3. Translation

- Reviews expected non-English: **6**
- Of those, text actually changed: **3** (50.0%)

| Detected lang | Reviews | Translate attempted | API called | Text changed |
|---|---:|---:|---:|---:|
| `en` | 42 | 0 | 0 | 0 |
| `hi` | 3 | 3 | 0 | 3 |
| `id` | 1 | 0 | 0 | 0 |

## 4. Confidence distribution

- mean 0.9274 / median 0.9903 / range 0.4991-0.9986
- exactly `0.7` (the hardcoded fallback value): **0** (0.0%)

## 5. Coverage

- Aspects per review: mean **1.59**, median 2.0, max 4
- Reviews with zero aspects: **5**
- Reviews labelled `General` only: **0** (0.0%)
- Aspect rows inside the 14-bucket keyword taxonomy: **5** (6.8%)

## 6. Can production detect fallback rows without instrumentation?

Heuristic precision **0.0%**, recall **0.0%** (TP 0, FP 0, FN 5, TN 41).
