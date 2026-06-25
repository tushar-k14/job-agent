"""Tests for Phase 4: tracing, token capture, retry backoff, and deterministic fallback."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

os.environ.pop("DEEPSEEK_API_KEY", None)
os.environ.pop("GEMINI_API_KEY", None)


# --------------------------------------------------------------------------- #
# Retry / backoff
# --------------------------------------------------------------------------- #
class TestBackoff:
    def test_returns_on_success(self):
        from backend.guardrails.retry import with_backoff
        assert with_backoff(lambda: 42, sleep=lambda s: None) == 42

    def test_retries_transient_then_succeeds(self):
        from backend.guardrails.retry import with_backoff
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise requests.ConnectionError("boom")
            return "ok"

        assert with_backoff(flaky, sleep=lambda s: None) == "ok"
        assert calls["n"] == 3

    def test_non_transient_raises_immediately(self):
        from backend.guardrails.retry import with_backoff
        calls = {"n": 0}

        def bad():
            calls["n"] += 1
            raise ValueError("deterministic")

        with pytest.raises(ValueError):
            with_backoff(bad, sleep=lambda s: None)
        assert calls["n"] == 1  # not retried

    def test_gives_up_after_max_attempts(self):
        from backend.guardrails.retry import with_backoff
        calls = {"n": 0}

        def always_timeout():
            calls["n"] += 1
            raise requests.Timeout("slow")

        with pytest.raises(requests.Timeout):
            with_backoff(always_timeout, max_attempts=3, sleep=lambda s: None)
        assert calls["n"] == 3

    def test_http_429_is_transient(self):
        from backend.guardrails.retry import is_transient
        resp = MagicMock(); resp.status_code = 429
        exc = requests.HTTPError(response=resp)
        assert is_transient(exc) is True

    def test_http_400_not_transient(self):
        from backend.guardrails.retry import is_transient
        resp = MagicMock(); resp.status_code = 400
        exc = requests.HTTPError(response=resp)
        assert is_transient(exc) is False


# --------------------------------------------------------------------------- #
# Deterministic fallback letter
# --------------------------------------------------------------------------- #
class TestFallback:
    def test_template_is_grounded(self):
        from backend.guardrails.fallback import grounded_fallback_letter
        from backend.guardrails import check_cover_letter_grounding
        state = {
            "parsed_job": {"title": "Backend Engineer", "company": "Acme",
                           "required_skills": ["Python"], "responsibilities": ["Build"]},
            "matching_skills": ["Python", "FastAPI"],
            "transferable_experiences": [],
            "resume_text": "I use Python and FastAPI.",
        }
        letter = grounded_fallback_letter(state)
        result = check_cover_letter_grounding(letter, state["resume_text"], state["parsed_job"])
        assert result.grounded is True
        assert "Backend Engineer" in letter
        assert "Acme" in letter

    def test_template_handles_unknown_fields(self):
        from backend.guardrails.fallback import grounded_fallback_letter
        state = {"parsed_job": {"title": "Unknown", "company": "Unknown"},
                 "matching_skills": [], "resume_text": ""}
        letter = grounded_fallback_letter(state)
        assert "Dear Hiring Team" in letter
        assert "Sincerely" in letter

    def test_oxford_comma(self):
        from backend.guardrails.fallback import build_template_cover_letter
        letter = build_template_cover_letter(
            title="Engineer", company="Acme",
            matching_skills=["Python", "Go", "Rust"],
        )
        assert "Python, Go, and Rust" in letter


# --------------------------------------------------------------------------- #
# Tracing
# --------------------------------------------------------------------------- #
class TestTracing:
    def test_run_and_steps_recorded(self):
        from backend.observability import start_run, trace_step, record_llm_usage
        with start_run(job_url="https://x.com") as run:
            with trace_step("planner", "in"):
                pass
            with trace_step("writer", "in"):
                record_llm_usage(100, 50)
        assert len(run.steps) == 2
        assert run.steps[0].name == "planner"
        assert run.steps[1].total_tokens == 150
        assert run.total_tokens == 150
        assert run.total_llm_calls == 1

    def test_step_records_error(self):
        from backend.observability import start_run, trace_step
        with start_run() as run:
            with pytest.raises(ValueError):
                with trace_step("boom"):
                    raise ValueError("x")
        assert run.steps[0].error == "x"

    def test_trace_step_without_run_is_noop(self):
        from backend.observability import trace_step
        # Should not raise even with no active run.
        with trace_step("orphan"):
            pass

    def test_to_dict_shape(self):
        from backend.observability import start_run, trace_step
        with start_run(job_url="u") as run:
            with trace_step("planner"):
                pass
        d = run.to_dict()
        assert d["job_url"] == "u"
        assert "steps" in d and d["steps"][0]["name"] == "planner"
        assert "latency_ms" in d


# --------------------------------------------------------------------------- #
# Token capture in the LLM client
# --------------------------------------------------------------------------- #
class TestTokenCapture:
    def test_deepseek_usage_recorded(self):
        from backend.llm import client
        from backend.observability import start_run

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 80},
        }
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "k"}):
            with patch("backend.llm.client.requests.post", return_value=resp):
                with start_run() as run:
                    from backend.observability import trace_step
                    with trace_step("writer"):
                        client.llm_complete("s", "u")
        assert run.steps[0].prompt_tokens == 200
        assert run.steps[0].completion_tokens == 80


# --------------------------------------------------------------------------- #
# Verifier grounding pre-check (deterministic gate before LLM judge)
# --------------------------------------------------------------------------- #
class TestVerifierGroundingGate:
    def test_fabrication_fails_without_llm_call(self):
        from backend.agents.verifier import cover_letter_verifier_node
        state = {
            "cover_letter": "Dear Team, at Google I used Kubernetes and TensorFlow. Sincerely,",
            "resume_text": "I use Python and FastAPI.",
            "parsed_job": {"title": "Eng", "company": "Acme", "required_skills": ["Python"]},
        }
        # The LLM judge must NOT be called — grounding catches it first.
        with patch("backend.agents.verifier.llm_json") as mock_judge:
            out = cover_letter_verifier_node(state)
        mock_judge.assert_not_called()
        assert out["cover_letter_verdict"]["passed"] is False
        assert out["cover_letter_verdict"]["fabrication_detected"] is True

    def test_clean_letter_proceeds_to_judge(self):
        from backend.agents.verifier import cover_letter_verifier_node
        state = {
            "cover_letter": "Dear Team, I use Python and FastAPI. Sincerely,",
            "resume_text": "I use Python and FastAPI.",
            "parsed_job": {"title": "Eng", "company": "Acme", "required_skills": ["Python"]},
        }
        judge = {"fabrication_detected": False, "length_ok": True, "generic_filler": False,
                 "passed": True, "reason": "ok"}
        with patch("backend.agents.verifier.llm_json", return_value=judge) as mock_judge:
            out = cover_letter_verifier_node(state)
        mock_judge.assert_called_once()
        assert out["cover_letter_verdict"]["passed"] is True
