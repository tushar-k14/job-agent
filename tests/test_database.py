"""Tests for backend/db/database.py — all run against a temporary DB file."""

from __future__ import annotations

import os
import tempfile

import pytest

# Point every test to a fresh temp DB so tests are isolated.
_TMP = tempfile.mktemp(suffix=".db")
os.environ["JOB_AGENT_DB"] = _TMP

from backend.db.database import (
    VALID_STATUSES,
    get_application,
    init_db,
    list_applications,
    save_application,
    update_cover_letter,
    update_notes,
    update_status,
)

# ---- fixtures ---------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    """Each test gets its own isolated SQLite file."""
    db_path = str(tmp_path / "test.db")
    os.environ["JOB_AGENT_DB"] = db_path
    import backend.db.database as dbmod
    dbmod.DB_PATH = db_path
    init_db()
    yield
    # Windows keeps the SQLite file locked; let pytest's tmp_path handle cleanup.


_SAMPLE_STATE = {
    "job_url": "https://example.com/job/1",
    "parsed_job": {
        "company": "Acme Corp",
        "title": "Senior Engineer",
        "required_skills": ["Python", "SQL"],
        "responsibilities": ["build things"],
        "nice_to_haves": ["Go"],
        "salary": "$120k",
    },
    "match_score": 82,
    "matching_skills": ["Python", "SQL"],
    "missing_skills": ["Go"],
    "tailored_bullets": [
        {"original": "built APIs", "rewritten": "designed REST APIs", "rationale": "more specific"}
    ],
    "cover_letter": "Dear Hiring Team,\n...",
    "resume_text": "I am a developer.",
    "status": "To Apply",
    "notes": "",
}


# ---- tests ------------------------------------------------------------------

class TestSaveApplication:
    def test_returns_integer_id(self):
        app_id = save_application(_SAMPLE_STATE)
        assert isinstance(app_id, int)
        assert app_id >= 1

    def test_round_trip_fields(self):
        app_id = save_application(_SAMPLE_STATE)
        app = get_application(app_id)
        assert app["company"] == "Acme Corp"
        assert app["role"] == "Senior Engineer"
        assert app["match_score"] == 82
        assert app["status"] == "To Apply"

    def test_tailored_bullets_deserialised(self):
        app_id = save_application(_SAMPLE_STATE)
        app = get_application(app_id)
        bullets = app["tailored_bullets"]
        assert isinstance(bullets, list)
        assert bullets[0]["original"] == "built APIs"

    def test_skill_lists_deserialised(self):
        app_id = save_application(_SAMPLE_STATE)
        app = get_application(app_id)
        assert "Python" in app["matching_skills"]
        assert "Go" in app["missing_skills"]

    def test_missing_parsed_job_does_not_crash(self):
        state = {"job_url": "http://x", "match_score": 0}
        app_id = save_application(state)
        app = get_application(app_id)
        assert app["company"] == "Unknown"

    def test_multiple_applications_get_unique_ids(self):
        id1 = save_application(_SAMPLE_STATE)
        id2 = save_application({**_SAMPLE_STATE, "job_url": "https://other.com"})
        assert id1 != id2


class TestGetApplication:
    def test_returns_none_for_missing_id(self):
        assert get_application(9999) is None

    def test_returns_dict_for_valid_id(self):
        app_id = save_application(_SAMPLE_STATE)
        app = get_application(app_id)
        assert isinstance(app, dict)


class TestListApplications:
    def test_empty_db_returns_empty_list(self):
        assert list_applications() == []

    def test_returns_all_saved(self):
        save_application(_SAMPLE_STATE)
        save_application({**_SAMPLE_STATE, "job_url": "http://b"})
        apps = list_applications()
        assert len(apps) == 2

    def test_ordered_newest_first(self):
        id1 = save_application({**_SAMPLE_STATE, "job_url": "http://first"})
        id2 = save_application({**_SAMPLE_STATE, "job_url": "http://second"})
        apps = list_applications()
        # newest (id2) should be first in list
        assert apps[0]["id"] == id2
        assert apps[1]["id"] == id1


class TestUpdateStatus:
    def test_valid_status_persists(self):
        app_id = save_application(_SAMPLE_STATE)
        for status in VALID_STATUSES:
            update_status(app_id, status)
            assert get_application(app_id)["status"] == status

    def test_invalid_status_raises(self):
        app_id = save_application(_SAMPLE_STATE)
        with pytest.raises(ValueError, match="Invalid status"):
            update_status(app_id, "Ghosted")


class TestUpdateNotes:
    def test_notes_persisted(self):
        app_id = save_application(_SAMPLE_STATE)
        update_notes(app_id, "Follow up on Monday")
        assert get_application(app_id)["notes"] == "Follow up on Monday"

    def test_notes_can_be_cleared(self):
        app_id = save_application({**_SAMPLE_STATE, "notes": "old note"})
        update_notes(app_id, "")
        assert get_application(app_id)["notes"] == ""


class TestUpdateCoverLetter:
    def test_cover_letter_updated(self):
        app_id = save_application(_SAMPLE_STATE)
        update_cover_letter(app_id, "Revised letter text")
        assert get_application(app_id)["cover_letter"] == "Revised letter text"
