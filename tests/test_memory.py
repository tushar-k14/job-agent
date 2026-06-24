"""Tests for the Chroma-backed persistent memory (no model download — no-op embeddings)."""

from __future__ import annotations

import os

import pytest

from backend.memory.store import MemoryStore, domain_of


@pytest.fixture()
def store(tmp_path):
    # Force the fast no-op embedding path (default) and an isolated persist dir.
    os.environ.pop("JOB_AGENT_MEMORY_EMBED", None)
    return MemoryStore(persist_dir=str(tmp_path / "chroma"))


class TestDomainOf:
    def test_strips_www(self):
        assert domain_of("https://www.greenhouse.io/x") == "greenhouse.io"

    def test_keeps_subdomain(self):
        assert domain_of("https://boards.greenhouse.io/acme/jobs/1") == "boards.greenhouse.io"

    def test_paste_is_empty(self):
        assert domain_of("pasted") == ""
        assert domain_of("") == ""


class TestRecordAndRecall:
    def test_available(self, store):
        assert store.available is True

    def test_record_increments_count(self, store):
        store.record_outcome(url="https://x.io/1", strategy="generic_text", success=True, reason="ok")
        assert store.stats()["count"] == 1

    def test_best_strategy_by_success_rate(self, store):
        d = "https://jobs.lever.co/foo/"
        store.record_outcome(url=d + "1", strategy="selector_structured", success=False, reason="empty")
        store.record_outcome(url=d + "2", strategy="generic_text", success=True, reason="ok")
        store.record_outcome(url=d + "3", strategy="generic_text", success=True, reason="ok")
        assert store.best_strategy_for_domain(d + "9") == "generic_text"

    def test_no_recall_without_success(self, store):
        d = "https://hard.example/"
        store.record_outcome(url=d + "1", strategy="generic_text", success=False, reason="captcha")
        store.record_outcome(url=d + "2", strategy="llm_from_raw_html", success=False, reason="captcha")
        assert store.best_strategy_for_domain(d + "3") is None

    def test_paste_domain_returns_none(self, store):
        store.record_outcome(url="pasted", strategy="paste_text", success=True, reason="ok")
        assert store.best_strategy_for_domain("pasted") is None

    def test_known_bad_after_threshold(self, store):
        d = "https://blocked.example/"
        for i in range(3):
            store.record_outcome(url=f"{d}{i}", strategy="generic_text", success=False, reason="403")
        assert store.domain_known_bad(d + "x") is True

    def test_not_known_bad_with_a_success(self, store):
        d = "https://mixed.example/"
        store.record_outcome(url=d + "1", strategy="generic_text", success=False, reason="x")
        store.record_outcome(url=d + "2", strategy="generic_text", success=True, reason="ok")
        store.record_outcome(url=d + "3", strategy="generic_text", success=False, reason="x")
        assert store.domain_known_bad(d + "9") is False


class TestPersistence:
    def test_survives_reopen(self, tmp_path):
        path = str(tmp_path / "chroma")
        s1 = MemoryStore(persist_dir=path)
        s1.record_outcome(url="https://persist.io/1", strategy="generic_text", success=True, reason="ok")
        # New instance pointed at the same dir should see the prior write.
        s2 = MemoryStore(persist_dir=path)
        assert s2.best_strategy_for_domain("https://persist.io/2") == "generic_text"
