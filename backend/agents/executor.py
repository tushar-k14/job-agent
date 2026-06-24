"""Executor node.

Performs the action the planner chose. In Phase 1 the executor owns the **extraction**
action (fetch the page once, run the selected strategy) and the optional **enrichment**
action. The downstream analysis / tailor / writer nodes remain separate graph nodes so
this refactor stays behavior-preserving for the happy path.

The executor fetches HTML at most once per run and caches it on the state, so when the
verifier routes back for a retry with a different strategy we re-parse the same HTML
instead of re-downloading it.
"""

from __future__ import annotations

import logging

from ..graph.state import ApplicationState
from ..schemas import ExtractionResult, ExtractionStrategy, ParsedJob, PlannerDecision
from . import strategies

logger = logging.getLogger(__name__)


def _run_extraction(
    state: ApplicationState, plan: PlannerDecision
) -> tuple[ExtractionResult, str]:
    """Returns (result, html_to_cache). html_to_cache is "" for the paste path."""
    strategy = plan.strategy

    # Paste path: text already on the state, no fetch.
    if strategy == ExtractionStrategy.PASTE_TEXT:
        raw_text = state.get("raw_job_text", "")
        if not raw_text.strip():
            return ExtractionResult(error="No pasted job text provided."), ""
        text, parsed = strategies.extract(strategy, raw_text=raw_text)
        return ExtractionResult(raw_text=text, parsed_job=parsed, strategy_used=strategy), ""

    # URL path: fetch HTML once, reuse the cached copy across retries.
    html = state.get("cached_html", "")
    if not html:
        try:
            html = strategies.fetch_html(state["job_url"])
        except strategies.BlockedDomainError as exc:
            return ExtractionResult(error=str(exc)), ""
        except Exception as exc:  # noqa: BLE001
            return ExtractionResult(error=f"Failed to fetch job posting: {exc}"), ""

    try:
        text, parsed = strategies.extract(strategy, url=state.get("job_url", ""), html=html)
    except Exception as exc:  # noqa: BLE001
        return (
            ExtractionResult(
                raw_text=html[:2000], strategy_used=strategy,
                error=f"Extraction failed ({strategy.value}): {exc}",
            ),
            html,
        )
    return (
        ExtractionResult(raw_text=text, parsed_job=parsed, strategy_used=strategy, error=None),
        html,
    )


def _maybe_enrich(parsed: ParsedJob, plan: PlannerDecision) -> ParsedJob:
    """Lightweight enrichment seam.

    When the planner flagged a thin JD, we annotate responsibilities so the downstream
    writer has *something* to anchor on rather than nothing. This is deliberately
    conservative (no fabrication of skills) — a real enrichment fetch (company about-page)
    is a documented future extension; doing it without inventing facts is the priority.
    """
    if not plan.needs_enrichment:
        return parsed
    note = (
        "Limited public detail was available for this posting; tailor against the "
        "candidate's transferable strengths rather than specific stated requirements."
    )
    if note not in parsed.responsibilities:
        parsed.responsibilities = parsed.responsibilities + [note]
    logger.info("Executor applied thin-JD enrichment note.")
    return parsed


def executor_node(state: ApplicationState) -> dict:
    plan_dict = state.get("plan")
    if not plan_dict:
        return {"error": state.get("error") or "Executor invoked without a plan."}
    plan = PlannerDecision(**plan_dict)

    result, html = _run_extraction(state, plan)

    update: dict = {
        "scrape_attempts": state.get("scrape_attempts", 0) + 1,
        "raw_job_text": result.raw_text or state.get("raw_job_text", ""),
        "strategy_used": result.strategy_used.value if result.strategy_used else None,
    }
    # Cache fetched HTML so retries re-parse instead of re-downloading.
    if html and not state.get("cached_html"):
        update["cached_html"] = html

    if result.error and result.parsed_job is None:
        update["error"] = result.error
        return update

    parsed = _maybe_enrich(result.parsed_job, plan)
    update["parsed_job"] = parsed.model_dump()
    # Clear any soft error from a previous failed attempt now that we have data.
    update["error"] = None
    return update
