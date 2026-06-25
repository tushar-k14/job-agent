"""LangGraph state machine: planner / executor / verifier agentic loop.

Flow (Phase 1):

    START → planner → executor → scrape_verifier ─(pass)→ analysis → tailor → writer
                ▲                          │                                      │
                └──────(retry, capped)─────┘                                      ▼
                                                                        cover_letter_verifier
                                                                                  │
                                              ┌────(pass)→ tracker → END           │
                                              └────(retry, capped)→ writer ◀───────┘

Two real verification gates with capped retries (N=2 retries ⇒ ≤3 attempts each):
- scrape_verifier routes back to planner with a failure reason for an alternate
  extraction strategy until MAX_SCRAPE_ATTEMPTS.
- cover_letter_verifier routes back to writer to regenerate with corrective
  instructions until MAX_COVER_LETTER_ATTEMPTS.

All hand-offs between nodes are validated through Pydantic models in ``backend.schemas``.
The paste-JD path is no longer a separate code path: it seeds ``raw_job_text`` + sets
``pasted=True`` and flows through the same graph (the planner picks PASTE_TEXT).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Optional

from langgraph.graph import END, START, StateGraph

from .state import ApplicationState

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Conditional routers
# --------------------------------------------------------------------------- #
def _route_after_scrape(state: ApplicationState) -> str:
    """After scrape verification: continue, retry via planner, or give up."""
    from ..agents.verifier import MAX_SCRAPE_ATTEMPTS

    verdict = state.get("scrape_verdict") or {}
    if verdict.get("passed"):
        return "analysis"
    if state.get("scrape_attempts", 0) < MAX_SCRAPE_ATTEMPTS:
        return "planner"  # retry with an escalated strategy
    return "tracker"  # exhausted retries — record the failed run


def _route_after_cover_letter(state: ApplicationState) -> str:
    """After cover-letter verification: continue, regenerate, or fall back deterministically."""
    from ..agents.verifier import MAX_COVER_LETTER_ATTEMPTS

    verdict = state.get("cover_letter_verdict") or {}
    if verdict.get("passed"):
        return "tracker"
    if state.get("cover_letter_attempts", 0) < MAX_COVER_LETTER_ATTEMPTS:
        return "writer"  # regenerate with corrective feedback
    # Retries exhausted and still failing → deterministic grounded-template fallback
    # rather than shipping an unverified (possibly fabricated) letter.
    return "cover_letter_fallback"


def _traced(name: str, fn):
    """Wrap a node fn so each invocation is captured as a trace step."""
    from ..observability import trace_step

    def wrapper(state: ApplicationState) -> dict:
        with trace_step(name, input_summary=_summarize_input(name, state)) as st:
            result = fn(state)
            st.output_summary = _summarize_output(name, result)
            return result

    wrapper.__name__ = getattr(fn, "__name__", name)
    return wrapper


def _summarize_input(name: str, state: ApplicationState) -> str:
    if name in ("planner", "executor"):
        return f"url={state.get('job_url', '')[:60]} attempt={state.get('scrape_attempts', 0)}"
    if name == "scrape_verifier":
        pj = state.get("parsed_job") or {}
        return f"title={pj.get('title')} company={pj.get('company')}"
    if name in ("writer", "cover_letter_verifier"):
        return f"cl_attempt={state.get('cover_letter_attempts', 0)}"
    return ""


def _summarize_output(name: str, result: dict) -> str:
    if not isinstance(result, dict):
        return ""
    if "plan" in result:
        p = result["plan"]
        return f"strategy={p.get('strategy')} enrich={p.get('needs_enrichment')}"
    if "scrape_verdict" in result:
        return f"scrape_passed={result['scrape_verdict'].get('passed')}"
    if "cover_letter_verdict" in result:
        return f"cl_passed={result['cover_letter_verdict'].get('passed')}"
    if "match_score" in result:
        return f"match_score={result.get('match_score')}"
    if result.get("error"):
        return f"error={result['error'][:60]}"
    return ", ".join(k for k in result if k != "cached_html")[:80]


def build_graph():
    """Construct and compile the planner/executor/verifier LangGraph."""
    # Lazy imports avoid the agents ↔ graph circular import at module load.
    from ..agents import (
        analysis_node,
        cover_letter_node,
        tailor_node,
        tracker_node,
    )
    from ..agents.planner import planner_node
    from ..agents.executor import executor_node
    from ..agents.verifier import (
        scrape_verifier_node,
        cover_letter_verifier_node,
        cover_letter_fallback_node,
    )

    g = StateGraph(ApplicationState)

    g.add_node("planner", _traced("planner", planner_node))
    g.add_node("executor", _traced("executor", executor_node))
    g.add_node("scrape_verifier", _traced("scrape_verifier", scrape_verifier_node))
    g.add_node("analysis", _traced("analysis", analysis_node))
    g.add_node("tailor", _traced("tailor", tailor_node))
    g.add_node("writer", _traced("writer", cover_letter_node))
    g.add_node("cover_letter_verifier", _traced("cover_letter_verifier", cover_letter_verifier_node))
    g.add_node("cover_letter_fallback", _traced("cover_letter_fallback", cover_letter_fallback_node))
    g.add_node("tracker", _traced("tracker", tracker_node))

    g.add_edge(START, "planner")
    g.add_edge("planner", "executor")
    g.add_edge("executor", "scrape_verifier")
    g.add_conditional_edges(
        "scrape_verifier",
        _route_after_scrape,
        {"analysis": "analysis", "planner": "planner", "tracker": "tracker"},
    )
    g.add_edge("analysis", "tailor")
    g.add_edge("tailor", "writer")
    g.add_edge("writer", "cover_letter_verifier")
    g.add_conditional_edges(
        "cover_letter_verifier",
        _route_after_cover_letter,
        {
            "writer": "writer",
            "cover_letter_fallback": "cover_letter_fallback",
            "tracker": "tracker",
        },
    )
    g.add_edge("cover_letter_fallback", "tracker")
    g.add_edge("tracker", END)

    return g.compile()


_APP_GRAPH = None


def get_graph():
    global _APP_GRAPH
    if _APP_GRAPH is None:
        _APP_GRAPH = build_graph()
    return _APP_GRAPH


def _invoke_traced(initial: ApplicationState) -> ApplicationState:
    """Invoke the graph inside a run trace and persist the trace alongside the result."""
    from ..observability import start_run

    with start_run(job_url=initial.get("job_url", "")) as run:
        result = get_graph().invoke(initial)
        # Persist the structured trace if we recorded an application id.
        try:
            from ..db import save_run_trace

            app_id = result.get("application_id")
            if app_id:
                save_run_trace(app_id, run.to_dict())
        except Exception:  # noqa: BLE001 - never fail the run over trace persistence
            pass
    return result


def run_single(job_url: str, resume_text: str) -> ApplicationState:
    """Run the full pipeline for one job URL and return the final state."""
    initial: ApplicationState = {
        "job_url": job_url.strip(),
        "resume_text": resume_text,
        "status": "To Apply",
        "pasted": False,
    }
    return _invoke_traced(initial)


def run_from_text(job_url: str, raw_jd_text: str, resume_text: str) -> ApplicationState:
    """Run the pipeline from pasted job description text.

    No longer a separate orchestration: we seed the raw text and set ``pasted=True`` so
    the planner selects the PASTE_TEXT strategy and the same graph handles the rest.
    """
    initial: ApplicationState = {
        "job_url": (job_url or "pasted").strip(),
        "resume_text": resume_text,
        "raw_job_text": raw_jd_text,
        "status": "To Apply",
        "pasted": True,
    }
    return _invoke_traced(initial)


def run_batch(
    job_urls: Iterable[str],
    resume_text: str,
    *,
    max_workers: int = 4,
    on_complete: Optional[Callable[[int, int, ApplicationState], None]] = None,
) -> list[ApplicationState]:
    """Map-reduce over many URLs in parallel. Results preserve input order."""
    urls = [u.strip() for u in job_urls if u and u.strip()]
    total = len(urls)
    results: list[Optional[ApplicationState]] = [None] * total

    if total == 0:
        return []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(run_single, url, resume_text): idx
            for idx, url in enumerate(urls)
        }
        done = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                state = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Batch item %s failed", idx)
                state = {
                    "job_url": urls[idx],
                    "resume_text": resume_text,
                    "status": "To Apply",
                    "error": f"Unhandled pipeline error: {exc}",
                }
            results[idx] = state
            done += 1
            if on_complete:
                on_complete(done, total, state)

    return [r for r in results if r is not None]
