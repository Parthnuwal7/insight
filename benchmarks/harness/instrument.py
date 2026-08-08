"""
Instrumentation wrappers around the real ABSA pipeline.

The point of this module is to answer one question exactly rather than
heuristically: for each review, which code path actually produced the output?

`ABSAProcessor` has four distinct routes to a result and the production output
records none of them:

    pyabsa                      PyABSA ran and returned aspects
    pyabsa_empty_to_fallback    PyABSA ran, found nothing, keyword buckets used
    pyabsa_error_to_fallback    PyABSA raised, keyword buckets used
    model_unavailable_fallback  checkpoint never loaded, keyword buckets used

Only the first is aspect-based sentiment analysis. The other three are a
14-entry keyword taxonomy with a hardcoded 0.7 confidence, and they land in the
same table with no marking.

Nothing in ABSA/ is modified. We wrap methods at runtime and restore them after.
"""

from __future__ import annotations

import functools
from collections import defaultdict
from typing import Any, Callable


class PipelineRecorder:
    """Records which code path handled each review, keyed by review text.

    Processing is sequential and single-threaded, so call order is review order.
    A review that falls back produces two events (the attempt, then the
    fallback); the event sequence is what identifies the route.
    """

    def __init__(self) -> None:
        self.absa_events: dict[str, list[str]] = defaultdict(list)
        self.translation: dict[str, dict[str, Any]] = {}
        self.model_loaded: bool | None = None
        self._patches: list[tuple[Any, str, Callable]] = []

    # ---------------------------------------------------------------- ABSA

    def _wrap_pyabsa(self, original: Callable) -> Callable:
        @functools.wraps(original)
        def wrapper(inner_self, review: str, *a, **kw):
            try:
                result = original(inner_self, review, *a, **kw)
            except Exception:
                self.absa_events[review].append("pyabsa_raised")
                raise
            # _extract_with_pyabsa delegates to the fallback when the model
            # returns zero aspects; that delegation fires the fallback wrapper
            # first, so an already-recorded fallback means PyABSA came up empty.
            if self.absa_events[review] and self.absa_events[review][-1] == "fallback_called":
                self.absa_events[review].append("pyabsa_returned_empty")
            else:
                self.absa_events[review].append("pyabsa_ok")
            return result

        return wrapper

    def _wrap_fallback(self, original: Callable) -> Callable:
        @functools.wraps(original)
        def wrapper(inner_self, review: str, *a, **kw):
            self.absa_events[review].append("fallback_called")
            return original(inner_self, review, *a, **kw)

        return wrapper

    def route_for(self, review: str) -> str:
        """Collapse the event sequence for one review into a single route."""
        events = self.absa_events.get(review, [])
        if not events:
            return "not_processed"
        if "pyabsa_ok" in events:
            return "pyabsa"
        if "pyabsa_returned_empty" in events:
            return "pyabsa_empty_to_fallback"
        if "pyabsa_raised" in events:
            return "pyabsa_error_to_fallback"
        if "fallback_called" in events:
            return (
                "model_unavailable_fallback"
                if self.model_loaded is False
                else "fallback_unattributed"
            )
        return "unknown"

    # --------------------------------------------------------- translation

    def _wrap_detect(self, original: Callable) -> Callable:
        @functools.wraps(original)
        def wrapper(inner_self, text: str, *a, **kw):
            lang = original(inner_self, text, *a, **kw)
            self.translation.setdefault(text, {})["detected_lang"] = lang
            return lang

        return wrapper

    def _wrap_translate(self, original: Callable) -> Callable:
        @functools.wraps(original)
        def wrapper(inner_self, text: str, *a, **kw):
            out = original(inner_self, text, *a, **kw)
            rec = self.translation.setdefault(text, {})
            rec["translate_attempted"] = True
            rec["text_changed"] = bool(out != text)
            return out

        return wrapper

    def _wrap_api_call(self, original: Callable) -> Callable:
        @functools.wraps(original)
        def wrapper(inner_self, text: str, *a, **kw):
            rec = self.translation.setdefault(text, {})
            rec["api_called"] = True
            out = original(inner_self, text, *a, **kw)
            rec["api_returned_new_text"] = bool(out != text)
            return out

        return wrapper

    def translation_for(self, text: str) -> dict[str, Any]:
        rec = dict(self.translation.get(text, {}))
        rec.setdefault("detected_lang", None)
        rec.setdefault("translate_attempted", False)
        rec.setdefault("api_called", False)
        rec.setdefault("api_returned_new_text", False)
        rec.setdefault("text_changed", False)
        return rec

    # ------------------------------------------------------------ lifecycle

    def install(self, absa_cls: type, translation_cls: type) -> None:
        pairs = [
            (absa_cls, "_extract_with_pyabsa", self._wrap_pyabsa),
            (absa_cls, "_extract_with_fallback", self._wrap_fallback),
            (translation_cls, "detect_language", self._wrap_detect),
            (translation_cls, "translate_to_english", self._wrap_translate),
            (translation_cls, "_call_hf_translation_api", self._wrap_api_call),
        ]
        for cls, name, factory in pairs:
            original = getattr(cls, name)
            self._patches.append((cls, name, original))
            setattr(cls, name, factory(original))

    def uninstall(self) -> None:
        for cls, name, original in reversed(self._patches):
            setattr(cls, name, original)
        self._patches.clear()


# ---------------------------------------------------------------------------
# Heuristic provenance, for environments where wrapping is impossible
# ---------------------------------------------------------------------------

# The 14 buckets in ABSAProcessor._extract_simple_aspects, plus its default.
FALLBACK_TAXONOMY = frozenset(
    {
        "OTP/Verification", "Login/Account", "App Performance", "Payment",
        "Quality", "Price", "Service", "Delivery", "Design", "Performance",
        "Usability", "Features", "Size", "Battery", "General",
    }
)

FALLBACK_CONFIDENCE = 0.7


def heuristic_is_fallback(aspects: list, confidences: list, review: str, positions: list) -> bool:
    """Guess whether a result came from the keyword fallback, using only output.

    This is what a production consumer could compute from an API response. The
    benchmark scores it against the recorder's ground truth so we know whether
    it can be trusted when instrumentation is unavailable.
    """
    if not aspects:
        return False

    signals = 0
    if confidences and all(abs(float(c) - FALLBACK_CONFIDENCE) < 1e-9 for c in confidences):
        signals += 1
    if aspects and all(a in FALLBACK_TAXONOMY for a in aspects):
        signals += 1
    whole_span = [0, len(review)]
    if positions and all(list(p) == whole_span for p in positions):
        signals += 1
    return signals >= 2
