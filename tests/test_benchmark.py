"""Tests for the benchmark harness, scoring, and grounding guardrail."""

from __future__ import annotations

import os

import pytest

os.environ.pop("DEEPSEEK_API_KEY", None)
os.environ.pop("GEMINI_API_KEY", None)

from backend.guardrails import check_cover_letter_grounding, extract_candidate_entities
from eval.fixtures import all_tasks, RESUME
from eval.harness import run_task
from eval.scoring import score_task


@pytest.fixture(autouse=True)
def isolate(tmp_path):
    import backend.db.database as dbmod
    import backend.memory.store as ms
    dbmod.DB_PATH = str(tmp_path / "b.db")
    dbmod.init_db()
    os.environ["JOB_AGENT_CHROMA_DIR"] = str(tmp_path / "chroma")
    ms._store = None
    yield
    ms._store = None


# --------------------------------------------------------------------------- #
# Grounding guardrail
# --------------------------------------------------------------------------- #
class TestGrounding:
    _RESUME = "Built REST APIs with FastAPI and Python. Used PostgreSQL and Docker."
    _JOB = {"title": "Backend Engineer", "company": "Acme Corp",
            "required_skills": ["Python", "FastAPI"], "responsibilities": ["Build APIs"]}

    def test_clean_letter_grounded(self):
        letter = "Dear Hiring Team, I built REST APIs with FastAPI and Python at Acme Corp. Sincerely,"
        r = check_cover_letter_grounding(letter, self._RESUME, self._JOB)
        assert r.grounded is True
        assert r.ungrounded_entities == []

    def test_fabricated_entities_caught(self):
        letter = "Dear Hiring Team, At Google I used Kubernetes and TensorFlow. Sincerely,"
        r = check_cover_letter_grounding(letter, self._RESUME, self._JOB)
        assert r.grounded is False
        assert "google" in r.ungrounded_entities
        assert "kubernetes" in r.ungrounded_entities
        assert "tensorflow" in r.ungrounded_entities

    def test_prose_does_not_false_positive(self):
        letter = (
            "Dear Hiring Team, I am excited to apply. I believe my skills make me a strong "
            "candidate and I would love to contribute. Sincerely,"
        )
        r = check_cover_letter_grounding(letter, self._RESUME, self._JOB)
        assert r.grounded is True

    def test_salutation_words_ignored(self):
        ents = extract_candidate_entities("Dear Hiring Team, Sincerely, Best Regards")
        assert ents == set()

    def test_tech_detected_lowercase(self):
        ents = extract_candidate_entities("i used python and kubernetes daily")
        assert "python" in ents and "kubernetes" in ents


# --------------------------------------------------------------------------- #
# Harness + scoring
# --------------------------------------------------------------------------- #
class TestHarness:
    def test_all_tasks_loadable(self):
        tasks = all_tasks()
        assert len(tasks) >= 20  # spec: 20-30 tasks
        ids = [t.id for t in tasks]
        assert len(ids) == len(set(ids)), "duplicate task ids"

    def test_good_task_passes(self):
        task = next(t for t in all_tasks() if t.id == "good_main")
        out = run_task(task)
        failures = score_task(task, out["state"])
        assert failures == [], failures

    def test_bad_task_fails_scrape_and_records(self):
        task = next(t for t in all_tasks() if t.id == "bad_empty_page")
        out = run_task(task)
        failures = score_task(task, out["state"])
        assert failures == [], failures
        # The run itself should show a failed scrape with retries exhausted.
        assert out["state"]["scrape_attempts"] == 3
        assert not out["state"]["scrape_verdict"]["passed"]

    def test_recovery_task_escalates_strategy(self):
        task = next(t for t in all_tasks() if t.id == "recover_on_second_attempt")
        out = run_task(task)
        assert score_task(task, out["state"]) == []
        assert out["state"]["strategy_used"] == "generic_text"

    def test_blocked_task_short_circuits(self):
        task = next(t for t in all_tasks() if t.id == "blocked_1")
        out = run_task(task)
        assert score_task(task, out["state"]) == []
        assert out["state"].get("error")

    def test_fabrication_task_remediated_by_fallback(self):
        task = next(t for t in all_tasks() if t.id == "fabrication_detected")
        out = run_task(task)
        # Phase 4: fabrication detected -> retries exhaust -> deterministic fallback ->
        # final letter is grounded. Scoring must pass and the fallback flag must be set.
        assert score_task(task, out["state"]) == []
        assert out["state"]["cover_letter_fallback_used"] is True

    def test_harness_reports_tokens(self):
        task = next(t for t in all_tasks() if t.id == "good_main")
        out = run_task(task)
        assert out["tokens"] > 0
        assert out["latency_s"] >= 0


class TestFullSuitePasses:
    """The whole mock benchmark should pass at 100% — guards against fixture drift."""

    def test_pass_rate_is_full(self):
        tasks = all_tasks()
        failed = []
        for task in tasks:
            out = run_task(task)
            fails = score_task(task, out["state"])
            if fails:
                failed.append((task.id, fails))
        assert not failed, f"benchmark tasks failing: {failed}"
