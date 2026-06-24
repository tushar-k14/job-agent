"""Tests for the LangGraph pipeline — build_graph, run_single, run_batch."""

from __future__ import annotations

import contextlib
import os
from unittest.mock import MagicMock, patch

import pytest


@contextlib.contextmanager
def _apply(patches):
    """Enter every patch in a tuple regardless of how many there are."""
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    import backend.db.database as dbmod
    import backend.memory.store as ms
    db_path = str(tmp_path / "pipe.db")
    dbmod.DB_PATH = db_path
    dbmod.init_db()
    # Isolate Chroma memory per test so tracker's outcome writes don't leak into data/chroma.
    os.environ["JOB_AGENT_CHROMA_DIR"] = str(tmp_path / "chroma")
    ms._store = None
    yield
    ms._store = None
    # Windows SQLite lock: let pytest's tmp_path handle cleanup.


def _make_get_patch():
    """Patch requests.get used inside strategies.fetch_html."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.text = (
        "<html><body><main><h1>Software Engineer</h1>"
        "<p>Corp is hiring. Python required. Responsibilities: build APIs, ship "
        "features, write tests, and collaborate across the engineering organization "
        "on backend services.</p></main></body></html>"
    )
    return patch("backend.agents.strategies.requests.get", return_value=resp), resp


def _mock_full_pipeline(
    *,
    parsed_job=None,
    match_score=72,
    bullets=None,
    cover_letter="Dear Hiring Team,\n...",
    judge_passed=True,
):
    """Return tuple of context managers that stub every LLM call in the P/E/V graph."""
    parsed_job = parsed_job or {
        "title": "SWE", "company": "Corp",
        "required_skills": ["Python"], "responsibilities": ["code", "ship features"],
        "nice_to_haves": [], "salary": None,
    }
    bullets = bullets or [{"original": "a", "rewritten": "b", "rationale": "r"}]
    get_patch, _ = _make_get_patch()

    return (
        get_patch,
        patch("backend.agents.strategies.llm_json", return_value=parsed_job),
        patch("backend.agents.analysis.llm_json", return_value={
            "match_score": match_score,
            "matching_skills": ["Python"],
            "missing_skills": [],
            "transferable_experiences": [],
        }),
        patch("backend.agents.tailor.llm_json", return_value={"tailored_bullets": bullets}),
        patch("backend.agents.cover_letter.llm_complete", return_value=cover_letter),
        patch("backend.agents.verifier.llm_json", return_value={
            "fabrication_detected": False,
            "length_ok": True,
            "generic_filler": False,
            "passed": judge_passed,
            "reason": "ok" if judge_passed else "rejected",
        }),
    )


class TestBuildGraph:
    def test_graph_compiles(self):
        from backend.graph import build_graph
        g = build_graph()
        assert g is not None

    def test_graph_has_expected_nodes(self):
        from backend.graph import build_graph
        g = build_graph()
        node_names = set(g.nodes)
        for expected in (
            "planner", "executor", "scrape_verifier",
            "analysis", "tailor", "writer", "cover_letter_verifier", "tracker",
        ):
            assert expected in node_names, f"Missing node: {expected}"


class TestRunSingle:
    def test_full_pipeline_populates_state(self):
        patches = _mock_full_pipeline()
        with _apply(patches):
            from backend.graph import run_single
            state = run_single("https://example.com/job", "My resume text")
        assert state["match_score"] == 72
        assert state["parsed_job"]["company"] == "Corp"
        assert len(state["tailored_bullets"]) == 1
        assert "cover_letter" in state
        assert isinstance(state.get("application_id"), int)

    def test_error_in_scraper_propagates_gracefully(self):
        get_patch, resp = _make_get_patch()
        resp.raise_for_status.side_effect = ConnectionError("network error")
        with get_patch:
            from backend.graph import run_single
            state = run_single("https://bad.url", "resume")
        # Pipeline should complete (tracker still runs); error field is set
        assert state.get("error")
        assert "application_id" in state

    def test_blocked_domain_returns_error_with_id(self):
        from backend.graph import run_single
        state = run_single("https://www.linkedin.com/jobs/view/123", "resume")
        assert state.get("error")
        assert "application_id" in state

    def test_job_url_stripped_of_whitespace(self):
        patches = _mock_full_pipeline()
        with _apply(patches):
            from backend.graph import run_single
            state = run_single("  https://example.com/job  ", "resume")
        assert state["job_url"] == "https://example.com/job"

    def test_default_status_is_to_apply(self):
        patches = _mock_full_pipeline()
        with _apply(patches):
            from backend.graph import run_single
            state = run_single("https://example.com/job", "resume")
        assert state["status"] == "To Apply"


class TestRunBatch:
    def test_returns_result_per_url(self):
        patches = _mock_full_pipeline()
        with _apply(patches):
            from backend.graph import run_batch
            results = run_batch(
                ["https://a.com", "https://b.com"],
                "my resume",
                max_workers=2,
            )
        assert len(results) == 2

    def test_empty_input_returns_empty_list(self):
        from backend.graph import run_batch
        assert run_batch([], "resume") == []

    def test_skips_blank_urls(self):
        patches = _mock_full_pipeline()
        with _apply(patches):
            from backend.graph import run_batch
            results = run_batch(["https://a.com", "", "  "], "resume")
        assert len(results) == 1

    def test_on_complete_callback_called(self):
        patches = _mock_full_pipeline()
        calls = []
        with _apply(patches):
            from backend.graph import run_batch
            run_batch(
                ["https://a.com", "https://b.com"],
                "resume",
                max_workers=2,
                on_complete=lambda done, total, s: calls.append((done, total)),
            )
        assert len(calls) == 2
        assert {t for _, t in calls} == {2}

    def test_one_failure_does_not_abort_others(self):
        good_html = (
            "<html><body><main><h1>SWE</h1><p>Corp hiring. Python. "
            "Responsibilities: build APIs, ship features, write tests, collaborate "
            "across engineering on backend services.</p></main></body></html>"
        )

        def get_side_effect(url, **kwargs):
            r = MagicMock()
            r.text = good_html
            if "bad" in url:
                r.raise_for_status.side_effect = ConnectionError("bad url")
            else:
                r.raise_for_status = MagicMock()
            return r

        # Use the full LLM stub set, but override the HTTP fetch with our URL-aware mock.
        patches = _mock_full_pipeline()[1:]  # drop the default get_patch
        with patch("backend.agents.strategies.requests.get", side_effect=get_side_effect):
            with _apply(patches):
                from backend.graph import run_batch
                results = run_batch(
                    ["https://good.com", "https://bad.com"],
                    "resume",
                    max_workers=2,
                )
        assert len(results) == 2
        errors = [r for r in results if r.get("error")]
        successes = [r for r in results if not r.get("error")]
        assert len(errors) == 1
        assert len(successes) == 1

    def test_results_preserve_input_order(self):
        urls = [f"https://job{i}.com" for i in range(5)]
        patches = _mock_full_pipeline()
        with _apply(patches):
            from backend.graph import run_batch
            results = run_batch(urls, "resume", max_workers=5)
        for i, (url, result) in enumerate(zip(urls, results)):
            assert result["job_url"] == url, f"Order mismatch at index {i}"
