"""Tests for the agent node functions and extraction strategies.

All LLM calls are mocked; HTTP calls are also mocked. No API keys or network access
required.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

os.environ.pop("DEEPSEEK_API_KEY", None)
os.environ.pop("GEMINI_API_KEY", None)

from backend.agents.analysis import analysis_node
from backend.agents.tailor import tailor_node
from backend.agents.cover_letter import cover_letter_node
from backend.agents.tracker import tracker_node


# ---- shared fixtures --------------------------------------------------------

@pytest.fixture()
def parsed_job():
    return {
        "title": "ML Engineer",
        "company": "DeepCo",
        "required_skills": ["Python", "PyTorch"],
        "responsibilities": ["Train models", "Deploy APIs"],
        "nice_to_haves": ["Rust"],
        "salary": "$150k",
    }


@pytest.fixture()
def base_state(parsed_job):
    return {
        "job_url": "https://example.com/job/42",
        "raw_job_text": "ML Engineer at DeepCo...",
        "parsed_job": parsed_job,
        "resume_text": (
            "Led a team of 3 engineers. Built Python data pipelines. "
            "Deployed models with FastAPI. Used PyTorch for NLP."
        ),
        "match_score": 75,
        "matching_skills": ["Python", "PyTorch"],
        "missing_skills": ["Rust"],
        "transferable_experiences": ["FastAPI deployment"],
    }


@pytest.fixture(autouse=True)
def fresh_tracker_db(tmp_path):
    """Each test that exercises the tracker gets an isolated DB + memory store."""
    import backend.db.database as dbmod
    import backend.memory.store as ms
    db_path = str(tmp_path / "test.db")
    dbmod.DB_PATH = db_path
    dbmod.init_db()
    os.environ["JOB_AGENT_CHROMA_DIR"] = str(tmp_path / "chroma")
    ms._store = None
    yield
    ms._store = None
    # Let pytest's tmp_path cleanup handle deletion; Windows locks the file
    # while the sqlite3 module holds it open inside _connect(), so explicit
    # os.remove here would raise PermissionError on Windows.


# ---- JobScraperAgent --------------------------------------------------------

# ---- Extraction strategies (formerly the scraper) ---------------------------
class TestStrategies:
    """Behaviors that used to live in scraper.py now live in strategies.py."""

    def test_generic_text_strips_boilerplate(self):
        from backend.agents.strategies import text_from_generic
        html = (
            "<html><head><style>body{}</style></head>"
            "<body><nav>nav</nav><p>real content</p><footer>footer</footer>"
            "<script>var x = 1;</script></body></html>"
        )
        text = text_from_generic(html)
        assert "real content" in text
        assert "nav" not in text and "footer" not in text and "var x" not in text

    def test_generic_text_capped(self):
        from backend.agents.strategies import text_from_generic
        text = text_from_generic("<p>" + "x " * 10000 + "</p>")
        assert len(text) <= 12000

    def test_selector_prefers_known_container(self):
        from backend.agents.strategies import text_from_selectors
        html = (
            "<html><body><div class='sidebar'>junk junk junk</div>"
            "<main>" + ("Senior Engineer at Acme. Build and own backend services. " * 8) +
            "</main></body></html>"
        )
        text = text_from_selectors(html)
        assert "Senior Engineer" in text

    def test_blocked_domain_raises(self):
        from backend.agents.strategies import check_blocked_domain, BlockedDomainError
        with pytest.raises(BlockedDomainError):
            check_blocked_domain("https://www.linkedin.com/jobs/view/123")

    def test_non_blocked_domain_ok(self):
        from backend.agents.strategies import check_blocked_domain
        # Should not raise.
        check_blocked_domain("https://boards.greenhouse.io/acme/jobs/1")

    def test_extract_paste_text_calls_llm(self):
        from backend.agents import strategies
        from backend.schemas import ExtractionStrategy
        with patch.object(strategies, "llm_json", return_value={"title": "Eng", "company": "Acme"}):
            text, parsed = strategies.extract(ExtractionStrategy.PASTE_TEXT, raw_text="some jd text")
        assert parsed.title == "Eng"
        assert text == "some jd text"

    def test_extract_normalises_missing_fields(self):
        from backend.agents import strategies
        from backend.schemas import ExtractionStrategy
        with patch.object(strategies, "llm_json", return_value={"title": None, "company": None}):
            _, parsed = strategies.extract(ExtractionStrategy.PASTE_TEXT, raw_text="jd")
        assert parsed.title == "Unknown"
        assert parsed.required_skills == []


# ---- ResumeAnalysisAgent ----------------------------------------------------

class TestAnalysisNode:
    _RESULT = {
        "match_score": 78,
        "matching_skills": ["Python", "PyTorch"],
        "missing_skills": ["Rust"],
        "transferable_experiences": ["FastAPI → REST APIs"],
    }

    def test_returns_analysis_fields(self, base_state):
        with patch("backend.agents.analysis.llm_json", return_value=self._RESULT):
            result = analysis_node(base_state)
        assert result["match_score"] == 78
        assert "Python" in result["matching_skills"]
        assert "Rust" in result["missing_skills"]

    def test_score_clamped_to_0_100(self, base_state):
        with patch(
            "backend.agents.analysis.llm_json",
            return_value={**self._RESULT, "match_score": 150},
        ):
            result = analysis_node(base_state)
        assert result["match_score"] == 100

    def test_score_clamped_below_0(self, base_state):
        with patch(
            "backend.agents.analysis.llm_json",
            return_value={**self._RESULT, "match_score": -10},
        ):
            result = analysis_node(base_state)
        assert result["match_score"] == 0

    def test_skips_when_error_present(self, base_state):
        result = analysis_node({**base_state, "error": "upstream failure"})
        assert result == {}

    def test_error_on_missing_resume(self, base_state):
        result = analysis_node({**base_state, "resume_text": ""})
        assert "error" in result

    def test_llm_failure_returns_error(self, base_state):
        from backend.llm import LLMError
        with patch("backend.agents.analysis.llm_json", side_effect=LLMError("no keys")):
            result = analysis_node(base_state)
        assert "error" in result


# ---- ResumeTailiorAgent -----------------------------------------------------

class TestTailorNode:
    _BULLETS = {
        "tailored_bullets": [
            {
                "original": "Built Python data pipelines",
                "rewritten": "Engineered Python ETL pipelines processing 10GB+ daily",
                "rationale": "Aligns with data engineering focus in JD",
            }
        ]
    }

    def test_returns_bullet_list(self, base_state):
        with patch("backend.agents.tailor.llm_json", return_value=self._BULLETS):
            result = tailor_node(base_state)
        assert len(result["tailored_bullets"]) == 1
        assert result["tailored_bullets"][0]["original"] == "Built Python data pipelines"

    def test_filters_incomplete_bullets(self, base_state):
        bad = {"tailored_bullets": [{"original": "", "rewritten": "x", "rationale": ""}]}
        with patch("backend.agents.tailor.llm_json", return_value=bad):
            result = tailor_node(base_state)
        assert result["tailored_bullets"] == []

    def test_skips_when_error_present(self, base_state):
        result = tailor_node({**base_state, "error": "oops"})
        assert result == {}

    def test_llm_failure_returns_error(self, base_state):
        from backend.llm import LLMError
        with patch("backend.agents.tailor.llm_json", side_effect=LLMError("no keys")):
            result = tailor_node(base_state)
        assert "error" in result


# ---- CoverLetterAgent -------------------------------------------------------

class TestCoverLetterNode:
    _LETTER = (
        "Dear Hiring Team,\n\n"
        "I am excited to apply for the ML Engineer role at DeepCo.\n\n"
        "DeepCo's mission resonates deeply with my experience.\n\n"
        "My PyTorch and Python background makes me a strong fit.\n\n"
        "Sincerely,"
    )

    def test_returns_cover_letter(self, base_state):
        with patch("backend.agents.cover_letter.llm_complete", return_value=self._LETTER):
            result = cover_letter_node(base_state)
        assert result["cover_letter"] == self._LETTER.strip()

    def test_strips_whitespace(self, base_state):
        with patch(
            "backend.agents.cover_letter.llm_complete",
            return_value=f"\n\n{self._LETTER}\n\n",
        ):
            result = cover_letter_node(base_state)
        assert not result["cover_letter"].startswith("\n")
        assert not result["cover_letter"].endswith("\n")

    def test_skips_when_error_present(self, base_state):
        result = cover_letter_node({**base_state, "error": "upstream problem"})
        assert result == {}

    def test_llm_failure_returns_error(self, base_state):
        from backend.llm import LLMError
        with patch(
            "backend.agents.cover_letter.llm_complete",
            side_effect=LLMError("no keys"),
        ):
            result = cover_letter_node(base_state)
        assert "error" in result


# ---- TrackerAgent -----------------------------------------------------------

class TestTrackerNode:
    def test_saves_and_returns_id(self, base_state):
        result = tracker_node({
            **base_state,
            "cover_letter": "Dear team,",
            "tailored_bullets": [],
        })
        assert "application_id" in result
        assert isinstance(result["application_id"], int)
        assert result["status"] == "To Apply"

    def test_defaults_to_to_apply(self, base_state):
        state = {k: v for k, v in base_state.items() if k != "status"}
        result = tracker_node(state)
        assert result["status"] == "To Apply"

    def test_saves_with_partial_state(self):
        """Tracker should persist even when upstream agents failed."""
        result = tracker_node({
            "job_url": "https://failed.example",
            "resume_text": "my resume",
            "error": "scraper failed",
        })
        assert "application_id" in result

    def test_db_error_returns_error_field(self, base_state):
        from backend.db import save_application
        with patch("backend.agents.tracker.save_application", side_effect=RuntimeError("db full")):
            result = tracker_node(base_state)
        assert "error" in result
