"""LangGraph state machine wiring the five agents into a pipeline.

Single-application flow:
    scraper -> analysis -> tailor -> cover_letter -> tracker -> END

Each node returns a partial state dict that LangGraph merges. Nodes short-circuit
their own work when ``state["error"]`` is set, but the tracker still runs so a
failed run is recorded.

Batch mode uses a simple thread-pool map-reduce: each URL is compiled into its own
isolated run of the same graph, executed concurrently, then results are reduced into
a list. (LangGraph's `Send`/fan-out API can also express this; a thread pool keeps the
per-application state fully independent and is the most robust for I/O-bound scraping +
LLM calls.)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Optional

from langgraph.graph import END, START, StateGraph

from .state import ApplicationState

logger = logging.getLogger(__name__)


def build_graph():
    """Construct and compile the single-application LangGraph."""
    # Imported lazily inside the function to avoid a circular import:
    # agents -> graph.state -> graph package __init__ -> pipeline -> agents.
    from ..agents import (
        analysis_node,
        cover_letter_node,
        scraper_node,
        tailor_node,
        tracker_node,
    )

    graph = StateGraph(ApplicationState)

    graph.add_node("scraper", scraper_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("tailor", tailor_node)
    graph.add_node("cover_letter", cover_letter_node)
    graph.add_node("tracker", tracker_node)

    graph.add_edge(START, "scraper")
    graph.add_edge("scraper", "analysis")
    graph.add_edge("analysis", "tailor")
    graph.add_edge("tailor", "cover_letter")
    graph.add_edge("cover_letter", "tracker")
    graph.add_edge("tracker", END)

    return graph.compile()


# The compiled graph is stateless and safe to reuse, but we build it lazily on first
# use to avoid doing work at import time (which would re-enter the agents package
# before it finishes importing -> circular import).
_APP_GRAPH = None


def get_graph():
    global _APP_GRAPH
    if _APP_GRAPH is None:
        _APP_GRAPH = build_graph()
    return _APP_GRAPH


def run_single(job_url: str, resume_text: str) -> ApplicationState:
    """Run the full pipeline for one job URL and return the final state."""
    initial: ApplicationState = {
        "job_url": job_url.strip(),
        "resume_text": resume_text,
        "status": "To Apply",
    }
    result = get_graph().invoke(initial)
    return result


def run_batch(
    job_urls: Iterable[str],
    resume_text: str,
    *,
    max_workers: int = 4,
    on_complete: Optional[Callable[[int, int, ApplicationState], None]] = None,
) -> list[ApplicationState]:
    """Map-reduce over many URLs in parallel.

    ``on_complete(done_count, total, state)`` is invoked after each finishes, which
    the Streamlit UI uses to advance a progress bar. Results preserve input order.
    """
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
