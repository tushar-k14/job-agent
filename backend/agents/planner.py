"""Planner node.

Decides the next action from current state. Two responsibilities per the design:

1. **Extraction strategy** — which strategy the executor should use for THIS attempt:
   - paste path  → PASTE_TEXT
   - first URL attempt → SELECTOR_STRUCTURED (most specific)
   - subsequent attempts escalate: GENERIC_TEXT → LLM_FROM_RAW_HTML
   The escalation is driven by ``scrape_attempts`` and the verifier's failure reason,
   which the verifier writes back into state before routing here.

2. **Enrichment** — whether the JD is too thin to tailor against confidently. This is a
   cheap deterministic heuristic (few/no required skills AND short responsibilities);
   when true the executor is signalled to pull supplementary context. (Enrichment fetch
   itself is a documented, lightweight step in the executor.)

The planner is deterministic (no LLM call) — strategy selection is a routing decision,
not a generative one, which keeps it fast and predictable on the benchmark.
"""

from __future__ import annotations

import logging

from ..graph.state import ApplicationState
from ..schemas import ExtractionStrategy, ParsedJob, PlannerDecision

logger = logging.getLogger(__name__)


def _recall_strategy_hint(url: str) -> ExtractionStrategy | None:
    """Ask persistent memory whether a strategy has worked for this domain before.

    Returns a known-good strategy to try first, or None to fall back to the default
    escalation order. Imported lazily so the planner has no hard dependency on Chroma.
    """
    try:
        from ..memory import recall_site_strategy

        name = recall_site_strategy(url)
        if name:
            return ExtractionStrategy(name)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Memory recall skipped: %s", exc)
    return None


def _jd_is_thin(parsed: ParsedJob) -> bool:
    """Heuristic: too little structured content to tailor confidently."""
    skills = len(parsed.required_skills)
    resp_chars = sum(len(r) for r in parsed.responsibilities)
    return skills <= 1 and resp_chars < 120


def _choose_strategy(state: ApplicationState) -> ExtractionStrategy:
    if state.get("pasted"):
        return ExtractionStrategy.PASTE_TEXT

    attempts = state.get("scrape_attempts", 0)
    # Allow memory to override the first attempt (Phase 2).
    if attempts == 0:
        hint = _recall_strategy_hint(state.get("job_url", ""))
        if hint is not None:
            return hint
        return ExtractionStrategy.SELECTOR_STRUCTURED

    order = ExtractionStrategy.escalation_order()
    # attempts==1 → order[1] (GENERIC_TEXT); attempts>=2 → order[-1] (LLM_FROM_RAW_HTML)
    idx = min(attempts, len(order) - 1)
    return order[idx]


def planner_node(state: ApplicationState) -> dict:
    # The planner is the loop head / retry entry point. Whenever it is invoked we (re)plan
    # — the conditional routers, not the planner, decide when retries are exhausted. So we
    # deliberately do NOT short-circuit on a prior error here; a prior scrape error is
    # exactly the signal to escalate to a different extraction strategy.
    strategy = _choose_strategy(state)
    attempt = state.get("scrape_attempts", 0) + 1

    # Enrichment decision only meaningful once we have a parsed job to inspect.
    parsed_dict = state.get("parsed_job")
    needs_enrichment = False
    if parsed_dict:
        needs_enrichment = _jd_is_thin(ParsedJob(**parsed_dict))

    decision = PlannerDecision(
        strategy=strategy,
        needs_enrichment=needs_enrichment,
        reason=(
            f"attempt {attempt}: strategy={strategy.value}"
            + (", enrichment requested" if needs_enrichment else "")
        ),
        attempt=attempt,
    )
    logger.info("Planner → %s", decision.reason)
    return {"plan": decision.model_dump(mode="json")}
