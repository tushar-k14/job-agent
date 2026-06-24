"""TrackerAgent.

Persists the full application package to SQLite and stamps an initial status.
Runs even when an upstream error occurred, so failures are still recorded and
visible on the dashboard (with whatever partial data exists).
"""

from __future__ import annotations

import logging

from ..db import save_application
from ..graph.state import ApplicationState

logger = logging.getLogger(__name__)


def _record_memory(state: ApplicationState) -> None:
    """Write this run's outcome to persistent memory (best-effort, never raises)."""
    try:
        from ..memory import record_outcome

        scrape_ok = bool((state.get("scrape_verdict") or {}).get("passed"))
        # A run "succeeded" if we got a usable parsed job (scrape passed) and no fatal error.
        success = scrape_ok and not state.get("error")
        cl_verdict = state.get("cover_letter_verdict") or {}
        record_outcome(
            url=state.get("job_url", ""),
            strategy=state.get("strategy_used") or "unknown",
            success=success,
            reason=(state.get("error") or "ok"),
            match_score=state.get("match_score"),
            cover_letter_passed=cl_verdict.get("passed"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Memory write skipped: %s", exc)


def tracker_node(state: ApplicationState) -> dict:
    status = state.get("status") or "To Apply"
    record = dict(state)
    record["status"] = status
    try:
        app_id = save_application(record)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to persist application: %s", exc)
        return {"error": (state.get("error") or "") + f" | DB save failed: {exc}"}

    _record_memory(state)
    logger.info("Saved application id=%s status=%s", app_id, status)
    return {"application_id": app_id, "status": status}
