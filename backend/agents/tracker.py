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


def tracker_node(state: ApplicationState) -> dict:
    status = state.get("status") or "To Apply"
    record = dict(state)
    record["status"] = status
    try:
        app_id = save_application(record)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to persist application: %s", exc)
        return {"error": (state.get("error") or "") + f" | DB save failed: {exc}"}

    logger.info("Saved application id=%s status=%s", app_id, status)
    return {"application_id": app_id, "status": status}
