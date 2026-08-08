"""
Aspect string normalization and fuzzy matching.

Shared by the fragmentation metric and the judgment scorer so both agree on
what counts as "the same aspect". Exact string comparison would score a
predicted "battery life" against a gold "battery" as both a false positive and
a false negative, punishing the model twice for what is really a normalization
problem. Fuzzy matching separates "did not find it" from "found it, named it
differently" -- different failures with different fixes.
"""

from __future__ import annotations

import re
import unicodedata

_ARTICLES = {"the", "a", "an", "this", "that", "my", "our", "your", "its"}
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")

JACCARD_THRESHOLD = 0.5


def normalize(aspect: str) -> str:
    """Lowercase, strip punctuation/articles, and crudely singularize."""
    if aspect is None:
        return ""
    text = unicodedata.normalize("NFKC", str(aspect)).lower().strip()
    text = _PUNCT.sub(" ", text)
    text = _WS.sub(" ", text).strip()

    tokens = []
    for tok in text.split():
        if tok in _ARTICLES:
            continue
        # Crude singularization: enough for aspect nouns, avoids a dependency.
        if len(tok) > 3 and tok.endswith("ies"):
            tok = tok[:-3] + "y"
        elif len(tok) > 3 and tok.endswith("es") and not tok.endswith("ses"):
            tok = tok[:-2]
        elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        tokens.append(tok)
    return " ".join(tokens)


def tokens(aspect: str) -> frozenset[str]:
    return frozenset(normalize(aspect).split())


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def is_match(a: str, b: str) -> bool:
    """True when two aspect strings plausibly denote the same thing."""
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    ta, tb = tokens(a), tokens(b)
    # "battery" vs "battery life": one concept fully contained in the other.
    if ta <= tb or tb <= ta:
        return True
    return jaccard(a, b) >= JACCARD_THRESHOLD


def cluster(surface_forms: list[str]) -> dict[str, list[str]]:
    """Greedily group surface forms that denote the same concept.

    Returns {representative: [surface forms]}. The representative is the
    shortest normalized form in the group, which tends to be the head noun.
    """
    unique = sorted(set(f for f in surface_forms if normalize(f)))
    groups: list[list[str]] = []
    for form in unique:
        for group in groups:
            if any(is_match(form, member) for member in group):
                group.append(form)
                break
        else:
            groups.append([form])
    result = {}
    for group in groups:
        rep = min(group, key=lambda f: (len(normalize(f)), normalize(f)))
        result[normalize(rep)] = sorted(group)
    return result


def align(predicted: list[str], gold: list[str]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """One-to-one greedy alignment between predicted and gold aspects.

    Returns (matched_pairs, unmatched_predicted_idx, unmatched_gold_idx).
    Exact normalized matches are consumed first so a near-match cannot steal a
    partner from an exact one.
    """
    matched: list[tuple[int, int]] = []
    used_pred: set[int] = set()
    used_gold: set[int] = set()

    for exact_pass in (True, False):
        for pi, p in enumerate(predicted):
            if pi in used_pred:
                continue
            best, best_score = None, 0.0
            for gi, g in enumerate(gold):
                if gi in used_gold:
                    continue
                if exact_pass:
                    if normalize(p) == normalize(g):
                        best, best_score = gi, 1.0
                        break
                elif is_match(p, g):
                    score = max(jaccard(p, g), 0.5)
                    if score > best_score:
                        best, best_score = gi, score
            if best is not None:
                matched.append((pi, best))
                used_pred.add(pi)
                used_gold.add(best)

    unmatched_pred = [i for i in range(len(predicted)) if i not in used_pred]
    unmatched_gold = [i for i in range(len(gold)) if i not in used_gold]
    return matched, unmatched_pred, unmatched_gold
