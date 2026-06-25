"""Lightweight tracing for every agent step.

Design goals:
- Capture per-step input summary, output summary, tool/LLM calls, tokens, and latency.
- Zero hard dependencies: if ``structlog`` is installed we emit structured JSON logs; if
  not, we fall back to stdlib logging. Either way the structured ``RunTrace`` object is
  built in-process and persisted to SQLite for the dashboard.
- Optional LangSmith: if ``LANGSMITH_API_KEY`` (or ``LANGCHAIN_API_KEY``) is set we also
  flip on LangChain/LangGraph's native LangSmith env tracing. No code path depends on it.

A ``RunTrace`` is held in a ``ContextVar`` so nodes and the LLM client can attach to the
active run without threading an object through every signature. ContextVars are
per-thread/per-task, so the batch thread pool keeps runs isolated.
"""

from __future__ import annotations

import contextlib
import os
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional

# --- structured logger (structlog if available, else stdlib) ----------------- #
try:
    import structlog

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
    )

    def get_logger(name: str = "job_agent"):
        return structlog.get_logger(name)

    _HAVE_STRUCTLOG = True
except Exception:  # noqa: BLE001 - structlog optional
    import logging

    def get_logger(name: str = "job_agent"):
        return logging.getLogger(name)

    _HAVE_STRUCTLOG = False


def _maybe_enable_langsmith() -> bool:
    """Turn on LangSmith env tracing if a key is present. Returns whether enabled."""
    key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not key:
        return False
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", key)
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "job-agent"))
    return True


LANGSMITH_ENABLED = _maybe_enable_langsmith()


# --------------------------------------------------------------------------- #
# Trace data model
# --------------------------------------------------------------------------- #
@dataclass
class StepTrace:
    name: str
    started_at: float
    ended_at: Optional[float] = None
    input_summary: str = ""
    output_summary: str = ""
    error: Optional[str] = None
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def latency_ms(self) -> float:
        if self.ended_at is None:
            return 0.0
        return (self.ended_at - self.started_at) * 1000.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "latency_ms": round(self.latency_ms, 1),
            "input": self.input_summary,
            "output": self.output_summary,
            "error": self.error,
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class RunTrace:
    run_id: str
    job_url: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    steps: list[StepTrace] = field(default_factory=list)

    def step(self, name: str) -> StepTrace:
        st = StepTrace(name=name, started_at=time.time())
        self.steps.append(st)
        return st

    @property
    def latency_ms(self) -> float:
        end = self.ended_at or time.time()
        return (end - self.started_at) * 1000.0

    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.steps)

    @property
    def total_llm_calls(self) -> int:
        return sum(s.llm_calls for s in self.steps)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "job_url": self.job_url,
            "latency_ms": round(self.latency_ms, 1),
            "total_tokens": self.total_tokens,
            "total_llm_calls": self.total_llm_calls,
            "steps": [s.to_dict() for s in self.steps],
        }


_current_run: ContextVar[Optional[RunTrace]] = ContextVar("current_run", default=None)
_current_step: ContextVar[Optional[StepTrace]] = ContextVar("current_step", default=None)


def current_run() -> Optional[RunTrace]:
    return _current_run.get()


@contextlib.contextmanager
def start_run(job_url: str = ""):
    """Begin a run trace; yields the RunTrace. Resets the context on exit."""
    run = RunTrace(run_id=uuid.uuid4().hex[:12], job_url=job_url)
    token = _current_run.set(run)
    log = get_logger()
    try:
        if _HAVE_STRUCTLOG:
            log.info("run_start", run_id=run.run_id, job_url=job_url)
        yield run
    finally:
        run.ended_at = time.time()
        if _HAVE_STRUCTLOG:
            log.info(
                "run_end", run_id=run.run_id,
                latency_ms=round(run.latency_ms, 1),
                total_tokens=run.total_tokens,
                steps=len(run.steps),
            )
        _current_run.reset(token)


@contextlib.contextmanager
def trace_step(name: str, input_summary: str = ""):
    """Trace one node/step. Attaches to the active run (no-op if none)."""
    run = _current_run.get()
    if run is None:
        # Not inside a run (e.g. unit test calling a node directly) — yield a detached step.
        yield StepTrace(name=name, started_at=time.time())
        return
    st = run.step(name)
    st.input_summary = input_summary
    tok = _current_step.set(st)
    log = get_logger()
    try:
        yield st
    except Exception as exc:  # noqa: BLE001
        st.error = str(exc)
        raise
    finally:
        st.ended_at = time.time()
        _current_step.reset(tok)
        if _HAVE_STRUCTLOG:
            log.info("step", **st.to_dict())


def record_llm_usage(prompt_tokens: int, completion_tokens: int) -> None:
    """Called by the LLM client after each call to attribute tokens to the active step."""
    st = _current_step.get()
    if st is not None:
        st.llm_calls += 1
        st.prompt_tokens += int(prompt_tokens or 0)
        st.completion_tokens += int(completion_tokens or 0)
