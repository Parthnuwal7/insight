"""
Tests for score_groundedness.py.

No network: every test either monkeypatches `_call_judge` or omits the API
key, so the suite is fast, deterministic, and free. None of the other
harness modules (score_judgments.py, matching.py, metrics_unlabeled.py)
have a committed pytest file -- this is the first, because
score_groundedness.py's failure modes (denominator honesty, unknown-id
handling, fail-closed on no key) are exactly the kind of thing a future
edit could quietly break without a name to blame. It exists alongside,
not instead of, having actually run the scorer against a real generated
report (see run_insight_report.py and benchmarks/README.md's Groundedness
section for that run's outcome).

Run with the repo's benchmark venv from the repo root:

    .venv-bench/Scripts/python.exe -m pytest benchmarks/harness/test_score_groundedness.py -v

Not collected by the ABSA/tests suite -- ABSA/ is a separate git repo on
its own branch; "302 tests pass, 2 deselected" (see task brief) refers to
`pytest ABSA/tests` and is unaffected by this file living here.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_groundedness as sg  # noqa: E402

REVIEWS = {1: "The battery lasts all day.", 2: "Shipping took two weeks."}


def _claim(text: str = "a claim", kind: str = "finding", review_ids=(1,)) -> dict:
    return {"text": text, "kind": kind, "review_ids": list(review_ids)}


def _report(**sections) -> dict:
    base = {s: [] for s in sg.SECTIONS}
    base.update(sections)
    return base


def _boom(*_a, **_k):
    raise AssertionError("the judge should not have been called")


# ---- a claim citing a nonexistent review id must be ungrounded, not skipped


class TestUnknownReviewId:
    def test_unknown_id_scores_ungrounded_and_stays_in_the_denominator(self, monkeypatch):
        monkeypatch.setattr(sg, "_call_judge", _boom)
        report = _report(findings=[_claim(review_ids=[999])])

        result = sg.score_report(report, REVIEWS, "m", "key")

        assert result["totals"]["claims_total"] == 1
        assert result["totals"]["grounded"] == 0
        assert result["totals"]["ungrounded"] == 1
        assert result["claims"][0]["reason"] == "unknown_review_id"
        assert result["claims"][0]["verdict"] == "ungrounded"

    def test_one_bad_id_among_several_drops_the_whole_claim(self, monkeypatch):
        monkeypatch.setattr(sg, "_call_judge", _boom)
        report = _report(findings=[_claim(review_ids=[1, 999])])

        result = sg.score_report(report, REVIEWS, "m", "key")

        assert result["claims"][0]["verdict"] == "ungrounded"
        assert result["claims"][0]["reason"] == "unknown_review_id"

    def test_uncited_claim_is_ungrounded_not_crashed_on(self, monkeypatch):
        # insights.report never emits this, but the scorer must not assume it.
        monkeypatch.setattr(sg, "_call_judge", _boom)
        report = _report(findings=[_claim(review_ids=[])])

        result = sg.score_report(report, REVIEWS, "m", "key")

        assert result["claims"][0]["reason"] == "uncited"
        assert result["claims"][0]["verdict"] == "ungrounded"


# ---- denominator honesty: zero claims is undefined, not a perfect score


class TestDenominatorHonesty:
    def test_zero_claims_is_undefined_not_1_0(self):
        result = sg.score_report(_report(), REVIEWS, "m", "key")

        assert result["totals"]["claims_total"] == 0
        assert result["groundedness_fraction"] is None
        assert result["groundedness_note"] is not None
        assert "undefined" in result["groundedness_note"]

    def test_zero_claims_note_surfaces_in_the_markdown_summary(self):
        result = sg.score_report(_report(), REVIEWS, "m", "key")
        md = sg.render_markdown(result, "run-x")
        assert "undefined" in md


# ---- judging outcomes


class TestJudging:
    def test_supported_verdict_is_grounded(self, monkeypatch):
        monkeypatch.setattr(sg, "_call_judge", lambda *a, **k: '{"verdict": "supported"}')
        report = _report(findings=[_claim(review_ids=[1])])

        result = sg.score_report(report, REVIEWS, "m", "key")

        assert result["totals"]["grounded"] == 1
        assert result["groundedness_fraction"] == 1.0

    def test_not_supported_verdict_is_ungrounded(self, monkeypatch):
        monkeypatch.setattr(sg, "_call_judge", lambda *a, **k: '{"verdict": "not_supported"}')
        report = _report(findings=[_claim(review_ids=[1])])

        result = sg.score_report(report, REVIEWS, "m", "key")

        assert result["totals"]["ungrounded"] == 1
        assert result["claims"][0]["reason"] == "not_supported"

    def test_unparseable_response_is_ungrounded(self, monkeypatch):
        monkeypatch.setattr(sg, "_call_judge", lambda *a, **k: "not json at all")
        report = _report(findings=[_claim(review_ids=[1])])

        result = sg.score_report(report, REVIEWS, "m", "key")

        assert result["claims"][0]["reason"] == "unparseable_response"

    def test_ambiguous_verdict_is_ungrounded(self, monkeypatch):
        monkeypatch.setattr(sg, "_call_judge", lambda *a, **k: '{"verdict": "kind of"}')
        report = _report(findings=[_claim(review_ids=[1])])

        result = sg.score_report(report, REVIEWS, "m", "key")

        assert result["claims"][0]["reason"] == "ambiguous_verdict"

    def test_mixed_batch_computes_the_fraction_correctly(self, monkeypatch):
        responses = iter([
            '{"verdict": "supported"}',
            '{"verdict": "not_supported"}',
            '{"verdict": "supported"}',
        ])
        monkeypatch.setattr(sg, "_call_judge", lambda *a, **k: next(responses))
        report = _report(
            findings=[_claim(review_ids=[1]), _claim(review_ids=[1])],
            complaints=[_claim(review_ids=[1])],
        )

        # workers=1: single-worker pool processes submitted jobs in order,
        # so the shared `responses` iterator is consumed deterministically.
        result = sg.score_report(report, REVIEWS, "m", "key", workers=1)

        assert result["totals"]["claims_total"] == 3
        assert result["totals"]["grounded"] == 2
        assert result["groundedness_fraction"] == pytest.approx(2 / 3, abs=1e-4)


# ---- fail closed: no key, or a failing call, must never count as grounded


class TestFailsClosed:
    def test_no_api_key_configured_is_llm_unavailable(self, monkeypatch):
        monkeypatch.setattr(sg, "_call_judge", _boom)
        report = _report(findings=[_claim(review_ids=[1])])

        result = sg.score_report(report, REVIEWS, "m", None)

        assert result["claims"][0]["reason"] == "llm_unavailable"
        assert result["claims"][0]["verdict"] == "ungrounded"

    def test_judge_call_failure_after_retry_is_llm_unavailable(self, monkeypatch):
        def raise_unavailable(*_a, **_k):
            raise sg.JudgeUnavailable("network down")

        monkeypatch.setattr(sg, "_call_judge", raise_unavailable)
        report = _report(findings=[_claim(review_ids=[1])])

        result = sg.score_report(report, REVIEWS, "m", "key")

        assert result["claims"][0]["reason"] == "llm_unavailable"


# ---- provenance: the number must be traceable to what produced it


class TestProvenanceRecorded:
    def test_model_and_temperature_are_recorded_in_the_output(self, monkeypatch):
        monkeypatch.setattr(sg, "_call_judge", lambda *a, **k: '{"verdict": "supported"}')
        report = _report(findings=[_claim(review_ids=[1])])

        result = sg.score_report(report, REVIEWS, "some/model-name", "key")

        assert result["judge_model"] == "some/model-name"
        assert result["judge_temperature"] == 0.0

    def test_the_judge_call_itself_hardcodes_temperature_zero(self):
        # score_report only ever records 0.0; this asserts the call that
        # actually reaches the network cannot silently drift from it.
        src = inspect.getsource(sg._call_judge)
        assert '"temperature": 0.0' in src


# ---- loaders: tolerant, and runnable offline against a stored report


class TestLoaders:
    def test_load_report_unwraps_the_http_response_envelope(self, tmp_path):
        payload = {
            "status": "success",
            "data": {"findings": [_claim()], "complaints": [], "strengths": [], "actions": []},
        }
        path = tmp_path / "report.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        report = sg.load_report(path)

        assert len(report["findings"]) == 1

    def test_load_report_accepts_the_bare_report_shape(self, tmp_path):
        payload = {"findings": [], "complaints": [_claim()], "strengths": [], "actions": []}
        path = tmp_path / "report.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        report = sg.load_report(path)

        assert len(report["complaints"]) == 1

    def test_load_reviews_matches_both_int_and_str_ids(self, tmp_path):
        path = tmp_path / "predictions.json"
        path.write_text(json.dumps([{"id": 7, "review": "text"}]), encoding="utf-8")

        lookup = sg.load_reviews(path)

        assert lookup[7] == "text"
        assert lookup["7"] == "text"
