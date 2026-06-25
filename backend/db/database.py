"""SQLite persistence for application packages.

A single ``applications`` table stores everything the TrackerAgent produces, so
the dashboard can be rebuilt from the DB alone. JSON-shaped fields (parsed job,
tailored bullets, skill lists) are stored as TEXT containing JSON.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

DB_PATH = os.getenv("JOB_AGENT_DB", os.path.join("data", "applications.db"))

VALID_STATUSES = ("To Apply", "Applied", "Interview", "Rejected")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS applications (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    job_url           TEXT,
    company           TEXT,
    role              TEXT,
    match_score       INTEGER,
    parsed_job        TEXT,
    matching_skills   TEXT,
    missing_skills    TEXT,
    tailored_bullets  TEXT,
    cover_letter      TEXT,
    resume_text       TEXT,
    status            TEXT DEFAULT 'To Apply',
    notes             TEXT DEFAULT '',
    created_at        TEXT,
    updated_at        TEXT
);
"""

# Observability: one row per pipeline run, linked to an application.
_TRACE_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_traces (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id  INTEGER,
    run_id          TEXT,
    success         INTEGER,
    scrape_passed   INTEGER,
    cover_passed    INTEGER,
    strategy_used   TEXT,
    scrape_attempts INTEGER,
    latency_ms      REAL,
    total_tokens    INTEGER,
    total_llm_calls INTEGER,
    trace_json      TEXT,
    created_at      TEXT
);
"""


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(_SCHEMA)
        conn.execute(_TRACE_SCHEMA)
        conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def save_application(state: dict) -> int:
    """Insert a fully-processed application package; return its new id."""
    init_db()
    parsed = state.get("parsed_job", {}) or {}
    now = _now()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO applications (
                job_url, company, role, match_score, parsed_job,
                matching_skills, missing_skills, tailored_bullets,
                cover_letter, resume_text, status, notes, created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                state.get("job_url", ""),
                parsed.get("company", "Unknown"),
                parsed.get("title", "Unknown"),
                int(state.get("match_score", 0) or 0),
                _dumps(parsed),
                _dumps(state.get("matching_skills", [])),
                _dumps(state.get("missing_skills", [])),
                _dumps(state.get("tailored_bullets", [])),
                state.get("cover_letter", ""),
                state.get("resume_text", ""),
                state.get("status", "To Apply"),
                state.get("notes", ""),
                now,
                now,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for json_field in (
        "parsed_job",
        "matching_skills",
        "missing_skills",
        "tailored_bullets",
    ):
        try:
            d[json_field] = json.loads(d[json_field]) if d[json_field] else None
        except (TypeError, json.JSONDecodeError):
            d[json_field] = None
    return d


def list_applications() -> list[dict]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM applications ORDER BY created_at DESC"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_application(app_id: int) -> Optional[dict]:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM applications WHERE id = ?", (app_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def update_status(app_id: int, status: str) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status {status!r}; expected one of {VALID_STATUSES}")
    with _connect() as conn:
        conn.execute(
            "UPDATE applications SET status = ?, updated_at = ? WHERE id = ?",
            (status, _now(), app_id),
        )
        conn.commit()


def update_notes(app_id: int, notes: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE applications SET notes = ?, updated_at = ? WHERE id = ?",
            (notes, _now(), app_id),
        )
        conn.commit()


def update_cover_letter(app_id: int, cover_letter: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE applications SET cover_letter = ?, updated_at = ? WHERE id = ?",
            (cover_letter, _now(), app_id),
        )
        conn.commit()


# --------------------------------------------------------------------------- #
# Observability: run traces
# --------------------------------------------------------------------------- #
def save_run_trace(application_id: int, trace: dict) -> int:
    """Persist a structured run trace (dict from RunTrace.to_dict)."""
    init_db()
    steps = trace.get("steps") or []
    # Derive a couple of headline booleans from the step outputs for fast dashboard queries.
    scrape_passed = any(
        "scrape_passed=True" in (s.get("output") or "") for s in steps
    )
    cover_passed = any(
        "cl_passed=True" in (s.get("output") or "") for s in steps
    )
    strategy_used = ""
    scrape_attempts = 0
    for s in steps:
        out = s.get("output") or ""
        if s.get("name") == "executor":
            scrape_attempts += 1
        if "strategy=" in out and s.get("name") == "planner":
            strategy_used = out.split("strategy=")[1].split(" ")[0]
    success = scrape_passed and not any(s.get("error") for s in steps)
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO run_traces (
                application_id, run_id, success, scrape_passed, cover_passed,
                strategy_used, scrape_attempts, latency_ms, total_tokens,
                total_llm_calls, trace_json, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                application_id,
                trace.get("run_id", ""),
                int(success),
                int(scrape_passed),
                int(cover_passed),
                strategy_used,
                scrape_attempts,
                float(trace.get("latency_ms", 0.0)),
                int(trace.get("total_tokens", 0)),
                int(trace.get("total_llm_calls", 0)),
                _dumps(trace),
                _now(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_run_traces(limit: int = 100) -> list[dict]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM run_traces ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["trace"] = json.loads(d.pop("trace_json")) if d.get("trace_json") else None
        except (TypeError, json.JSONDecodeError):
            d["trace"] = None
        out.append(d)
    return out
