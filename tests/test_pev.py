"""Unit tests for the planner / executor / verifier nodes and routers."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

os.environ.pop("DEEPSEEK_API_KEY", None)
os.environ.pop("GEMINI_API_KEY", None)

from backend.agents.planner import planner_node, _jd_is_thin, _choose_strategy
from backend.agents.verifier import (
    _validate_scrape,
    scrape_verifier_node,
    cover_letter_verifier_node,
    MAX_SCRAPE_ATTEMPTS,
    MAX_COVER_LETTER_ATTEMPTS,
)
from backend.graph.pipeline import _route_after_scrape, _route_after_cover_letter
from backend.schemas import (
    ExtractionStrategy,
    ParsedJob,
    VerificationStage,
)


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #
class TestPlannerStrategy:
    def test_paste_path_uses_paste_text(self):
        assert _choose_strategy({"pasted": True}) == ExtractionStrategy.PASTE_TEXT

    def test_first_url_attempt_is_selector_structured(self):
        assert _choose_strategy({"scrape_attempts": 0}) == ExtractionStrategy.SELECTOR_STRUCTURED

    def test_second_attempt_escalates_to_generic(self):
        assert _choose_strategy({"scrape_attempts": 1}) == ExtractionStrategy.GENERIC_TEXT

    def test_third_attempt_escalates_to_raw_html(self):
        assert _choose_strategy({"scrape_attempts": 2}) == ExtractionStrategy.LLM_FROM_RAW_HTML

    def test_planner_emits_plan_dict(self):
        out = planner_node({"job_url": "https://x.com", "scrape_attempts": 0})
        assert "plan" in out
        assert out["plan"]["strategy"] == "selector_structured"
        assert out["plan"]["attempt"] == 1

    def test_planner_increments_attempt(self):
        out = planner_node({"job_url": "https://x.com", "scrape_attempts": 1})
        assert out["plan"]["attempt"] == 2


class TestThinJD:
    def test_thin_jd_flagged(self):
        thin = ParsedJob(title="X", company="Y", required_skills=[], responsibilities=["short"])
        assert _jd_is_thin(thin) is True

    def test_rich_jd_not_flagged(self):
        rich = ParsedJob(
            title="X", company="Y",
            required_skills=["Python", "SQL", "Docker"],
            responsibilities=["Build and maintain large scale backend services daily"],
        )
        assert _jd_is_thin(rich) is False

    def test_planner_sets_enrichment_for_thin_jd(self):
        state = {
            "job_url": "https://x.com", "scrape_attempts": 0,
            "parsed_job": {"title": "X", "company": "Y", "required_skills": [], "responsibilities": []},
        }
        out = planner_node(state)
        assert out["plan"]["needs_enrichment"] is True


# --------------------------------------------------------------------------- #
# Scrape verifier (deterministic)
# --------------------------------------------------------------------------- #
class TestScrapeVerifier:
    def _good(self):
        return ParsedJob(
            title="Backend Engineer", company="Acme",
            required_skills=["Python"], responsibilities=["Build APIs"],
        )

    def test_valid_scrape_passes(self):
        v = _validate_scrape(self._good(), "Backend Engineer at Acme. Python. Build APIs.")
        assert v.passed
        assert v.stage == VerificationStage.SCRAPE

    def test_missing_title_fails(self):
        p = self._good()
        p.title = "Unknown"
        v = _validate_scrape(p, "some text " * 60)
        assert not v.passed
        assert "title" in v.reason

    def test_missing_company_fails(self):
        p = self._good()
        p.company = "Unknown"
        v = _validate_scrape(p, "some text " * 60)
        assert not v.passed
        assert "company" in v.reason

    def test_empty_description_fails(self):
        p = ParsedJob(title="Engineer", company="Acme", required_skills=[], responsibilities=[])
        v = _validate_scrape(p, "short")
        assert not v.passed
        assert "description" in v.reason

    def test_captcha_page_fails(self):
        v = _validate_scrape(self._good(), "Please complete the CAPTCHA to continue")
        assert not v.passed
        assert "block" in v.reason.lower() or "captcha" in v.reason.lower()

    def test_node_sets_error_on_failure(self):
        state = {
            "parsed_job": {"title": "Unknown", "company": "Unknown",
                           "required_skills": [], "responsibilities": [], "nice_to_haves": [], "salary": None},
            "raw_job_text": "x",
        }
        out = scrape_verifier_node(state)
        assert out["scrape_verdict"]["passed"] is False
        assert out.get("error")

    def test_node_clears_error_on_pass(self):
        state = {
            "parsed_job": {"title": "Engineer", "company": "Acme",
                           "required_skills": ["Python"], "responsibilities": ["Build"],
                           "nice_to_haves": [], "salary": None},
            "raw_job_text": "Engineer at Acme. Python. Build things.",
            "error": "stale error from a prior attempt",
        }
        out = scrape_verifier_node(state)
        assert out["scrape_verdict"]["passed"] is True
        assert out["error"] is None


# --------------------------------------------------------------------------- #
# Cover-letter verifier (LLM-as-judge, mocked)
# --------------------------------------------------------------------------- #
class TestCoverLetterVerifier:
    _STATE = {
        "cover_letter": "Dear Hiring Team, I built APIs. Sincerely,",
        "resume_text": "Built APIs with FastAPI.",
        "parsed_job": {"title": "Eng", "company": "Acme"},
    }

    def test_passes_clean_letter(self):
        judge = {"fabrication_detected": False, "length_ok": True,
                 "generic_filler": False, "passed": True, "reason": "good"}
        with patch("backend.agents.verifier.llm_json", return_value=judge):
            out = cover_letter_verifier_node(self._STATE)
        assert out["cover_letter_verdict"]["passed"] is True

    def test_fabrication_forces_fail_even_if_passed_true(self):
        # Model contradicts itself: passed=True but fabrication=True → must fail.
        judge = {"fabrication_detected": True, "length_ok": True,
                 "generic_filler": False, "passed": True, "reason": "claims unsupported AWS"}
        with patch("backend.agents.verifier.llm_json", return_value=judge):
            out = cover_letter_verifier_node(self._STATE)
        assert out["cover_letter_verdict"]["passed"] is False
        assert out["cover_letter_verdict"]["fabrication_detected"] is True

    def test_judge_failure_passes_through(self):
        from backend.llm import LLMError
        with patch("backend.agents.verifier.llm_json", side_effect=LLMError("no keys")):
            out = cover_letter_verifier_node(self._STATE)
        # Judge unavailable should not block the pipeline.
        assert out["cover_letter_verdict"]["passed"] is True

    def test_no_letter_fails(self):
        out = cover_letter_verifier_node({"cover_letter": ""})
        assert out["cover_letter_verdict"]["passed"] is False


# --------------------------------------------------------------------------- #
# Routers
# --------------------------------------------------------------------------- #
class TestRouters:
    def test_scrape_pass_routes_to_analysis(self):
        assert _route_after_scrape({"scrape_verdict": {"passed": True}}) == "analysis"

    def test_scrape_fail_with_retries_routes_to_planner(self):
        state = {"scrape_verdict": {"passed": False}, "scrape_attempts": 1}
        assert _route_after_scrape(state) == "planner"

    def test_scrape_fail_exhausted_routes_to_tracker(self):
        state = {"scrape_verdict": {"passed": False}, "scrape_attempts": MAX_SCRAPE_ATTEMPTS}
        assert _route_after_scrape(state) == "tracker"

    def test_cover_pass_routes_to_tracker(self):
        assert _route_after_cover_letter({"cover_letter_verdict": {"passed": True}}) == "tracker"

    def test_cover_fail_with_retries_routes_to_writer(self):
        state = {"cover_letter_verdict": {"passed": False}, "cover_letter_attempts": 1}
        assert _route_after_cover_letter(state) == "writer"

    def test_cover_fail_exhausted_routes_to_fallback(self):
        # Phase 4: exhausted retries route to the deterministic grounded-template fallback
        # rather than shipping an unverified letter straight to the tracker.
        state = {"cover_letter_verdict": {"passed": False},
                 "cover_letter_attempts": MAX_COVER_LETTER_ATTEMPTS}
        assert _route_after_cover_letter(state) == "cover_letter_fallback"
