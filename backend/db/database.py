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


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(_SCHEMA)
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
