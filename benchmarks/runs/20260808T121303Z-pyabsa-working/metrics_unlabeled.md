# Baseline diagnostic - label-free metrics

Run: `20260808T121303Z-pyabsa-working`

- Reviews in: **46**
- Aspect rows out: **78**
- PyABSA checkpoint loaded: **True**

## 1. Which code path produced the output?

| Route | Reviews | % |
|---|---:|---:|
| `pyabsa` | 38 | 82.6% |
| `pyabsa_empty_to_fallback` | 8 | 17.4% |

**Fallback review rate: 17.4%** (aspect rows: 12.8%)

## 2. Aspect fragmentation

| Scope | Surface forms | Concepts | Ratio |
|---|---:|---:|---:|
| All rows | 55 | 46 | 1.2x |
| PyABSA rows only | 51 | 44 | 1.16x |

Most fragmented concepts:

- **service** (5 forms): `Customer service`, `Service`, `Service staff`, `customer service`, `service`
- **food** (5 forms): `Food`, `Food quality`, `Quality`, `quality`, `sound quality`
- **battery** (4 forms): `battery`, `battery backup`, `battery drain`, `battery life`
- **performance** (3 forms): `App Performance`, `Performance`, `performance`
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

- mean 0.8938 / median 0.9768 / range 0.4991-0.9986
- exactly `0.7` (the hardcoded fallback value): **10** (12.8%)

## 5. Coverage

- Aspects per review: mean **1.7**, median 2.0, max 4
- Reviews with zero aspects: **0**
- Reviews labelled `General` only: **2** (4.3%)
- Aspect rows inside the 14-bucket keyword taxonomy: **14** (17.9%)

## 6. Can production detect fallback rows without instrumentation?

Heuristic precision **100.0%**, recall **100.0%** (TP 8, FP 0, FN 0, TN 38).
