"""Execution harness shared by the benchmark runner.

Provides a context manager that patches the pipeline's LLM + HTTP boundaries so a single
``BenchmarkTask`` runs deterministically and for free (MOCK tier), while recording a
synthetic token count for cost reporting. The LIVE tier skips the patches and uses the
real DeepSeek/Gemini client.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock, patch

from .fixtures import BenchmarkTask


@dataclass
class RunRecord:
    task_id: str
    category: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    approx_tokens: int = 0
    final_state: dict = field(default_factory=dict)


# Rough token estimate per LLM call type, for the mock tier's cost report.
_TOKENS = {"extract": 1200, "analysis": 900, "tailor": 1100, "cover": 700, "judge": 500}


class _MockLLM:
    """Deterministic stand-in for the LLM, scripted per task. Counts approx tokens."""

    def __init__(self, task: BenchmarkTask):
        self.task = task
        self.tokens = 0
        self._extraction_calls = 0

    # strategies.extract calls llm_json with the extraction prompt
    def extract_json(self, system, user, **kw):
        self.tokens += _TOKENS["extract"]
        self._extraction_calls += 1
        # Force-fail extraction (return empty) until the configured attempt.
        if self.task.fail_extraction_until_attempt:
            if self._extraction_calls < self.task.fail_extraction_until_attempt:
                return {"title": None, "company": None, "required_skills": [],
                        "responsibilities": [], "nice_to_haves": [], "salary": None}
        return dict(self.task.parsed_job)

    def analysis_json(self, system, user, **kw):
        self.tokens += _TOKENS["analysis"]
        return dict(self.task.analysis)

    def tailor_json(self, system, user, **kw):
        self.tokens += _TOKENS["tailor"]
        return {"tailored_bullets": [
            {"original": "Built REST APIs with FastAPI",
             "rewritten": "Engineered production REST APIs with FastAPI",
             "rationale": "keyword match"},
        ]}

    def cover_complete(self, system, user, **kw):
        self.tokens += _TOKENS["cover"]
        return self.task.cover_letter

    def judge_json(self, system, user, **kw):
        self.tokens += _TOKENS["judge"]
        return dict(self.task.judge)


@contextlib.contextmanager
def mock_environment(task: BenchmarkTask):
    """Patch all LLM + HTTP seams for one task. Yields the _MockLLM (for token count)."""
    mock = _MockLLM(task)

    # HTTP: strategies.requests.get returns the task HTML.
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.text = task.html or "<html><body></body></html>"

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("backend.agents.strategies.requests.get", return_value=resp))
        stack.enter_context(patch("backend.agents.strategies.llm_json", side_effect=mock.extract_json))
        stack.enter_context(patch("backend.agents.analysis.llm_json", side_effect=mock.analysis_json))
        stack.enter_context(patch("backend.agents.tailor.llm_json", side_effect=mock.tailor_json))
        stack.enter_context(patch("backend.agents.cover_letter.llm_complete", side_effect=mock.cover_complete))
        stack.enter_context(patch("backend.agents.verifier.llm_json", side_effect=mock.judge_json))
        yield mock


def run_task(task: BenchmarkTask, *, live: bool = False) -> dict:
    """Execute one task through the real graph; return the final state + token count."""
    from backend.graph import run_single, run_from_text

    t0 = time.time()
    if live:
        if task.pasted_text:
            state = run_from_text("", task.pasted_text, task.resume)
        else:
            state = run_single(task.url, task.resume)
        tokens = 0  # live token accounting would require client instrumentation (Phase 4)
    else:
        with mock_environment(task) as mock:
            if task.pasted_text:
                state = run_from_text("", task.pasted_text, task.resume)
            else:
                state = run_single(task.url, task.resume)
            tokens = mock.tokens
    latency = time.time() - t0
    return {"state": dict(state), "tokens": tokens, "latency_s": latency}
