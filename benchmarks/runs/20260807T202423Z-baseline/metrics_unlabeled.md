# Baseline diagnostic - label-free metrics

Run: `20260807T202423Z-baseline`

- Reviews in: **46**
- Aspect rows out: **75**
- PyABSA checkpoint loaded: **False**

## 1. Which code path produced the output?

| Route | Reviews | % |
|---|---:|---:|
| `model_unavailable_fallback` | 46 | 100.0% |

**Fallback review rate: 100.0%** (aspect rows: 100.0%)

## 2. Aspect fragmentation

| Scope | Surface forms | Concepts | Ratio |
|---|---:|---:|---:|
| All rows | 13 | 12 | 1.08x |
| PyABSA rows only | 0 | 0 | 0.0x |

Most fragmented concepts:

- **performance** (2 forms): `App Performance`, `Performance`

## 3. Translation

- Reviews expected non-English: **6**
- Of those, text actually changed: **0** (0.0%)

| Detected lang | Reviews | Translate attempted | API called | Text changed |
|---|---:|---:|---:|---:|
| `en` | 42 | 0 | 0 | 0 |
| `hi` | 3 | 3 | 3 | 0 |
| `id` | 1 | 0 | 0 | 0 |

## 4. Confidence distribution

- mean 0.7 / median 0.7 / range 0.7-0.7
- exactly `0.7` (the hardcoded fallback value): **75** (100.0%)

## 5. Coverage

- Aspects per review: mean **1.63**, median 1.0, max 4
- Reviews with zero aspects: **0**
- Reviews labelled `General` only: **8** (17.4%)
- Aspect rows inside the 14-bucket keyword taxonomy: **75** (100.0%)

## 6. Can production detect fallback rows without instrumentation?

Heuristic precision **100.0%**, recall **100.0%** (TP 46, FP 0, FN 0, TN 0).
