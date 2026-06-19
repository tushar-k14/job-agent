"""Tests for backend/llm/client.py — no real API calls."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

# Clear API keys so tests never hit real endpoints
os.environ.pop("DEEPSEEK_API_KEY", None)
os.environ.pop("GEMINI_API_KEY", None)

from backend.llm.client import (
    LLMError,
    _extract_json,
    _deepseek_complete,
    _gemini_complete,
    llm_complete,
    llm_json,
)


# --------------------------------------------------------------------------- #
# _extract_json
# --------------------------------------------------------------------------- #
class TestExtractJson:
    def test_bare_object(self):
        raw = '{"a": 1, "b": "hello"}'
        assert _extract_json(raw) == raw

    def test_fenced_json(self):
        raw = '```json\n{"key": "val"}\n```'
        assert _extract_json(raw) == '{"key": "val"}'

    def test_fenced_no_lang(self):
        raw = '```\n{"x": 99}\n```'
        assert _extract_json(raw) == '{"x": 99}'

    def test_json_buried_in_prose(self):
        raw = 'Here is the answer:\n{"score": 75}\nThat is all.'
        assert _extract_json(raw) == '{"score": 75}'

    def test_array_returned(self):
        raw = '[1, 2, 3]'
        assert _extract_json(raw) == '[1, 2, 3]'

    def test_no_json_returns_original(self):
        raw = "just some prose with no JSON"
        # Falls back to returning the raw text; JSON parsing will then fail
        assert _extract_json(raw) == raw


# --------------------------------------------------------------------------- #
# llm_json parsing
# --------------------------------------------------------------------------- #
class TestLlmJson:
    def _mock_response(self, text: str):
        """Patch llm_complete to return a fixed string."""
        return patch("backend.llm.client.llm_complete", return_value=text)

    def test_parses_clean_json(self):
        payload = {"match_score": 80, "skills": ["Python"]}
        with self._mock_response(json.dumps(payload)):
            result = llm_json("sys", "user")
        assert result == payload

    def test_parses_fenced_json(self):
        payload = {"title": "Engineer"}
        fenced = f"```json\n{json.dumps(payload)}\n```"
        with self._mock_response(fenced):
            result = llm_json("sys", "user")
        assert result == payload

    def test_raises_on_invalid_json(self):
        with self._mock_response("This is not JSON at all."):
            with pytest.raises(LLMError, match="valid JSON"):
                llm_json("sys", "user")


# --------------------------------------------------------------------------- #
# Provider fallback logic
# --------------------------------------------------------------------------- #
class TestProviderFallback:
    def _fake_response(self, text: str):
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        return mock

    def test_uses_deepseek_when_key_present(self):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": "hello from deepseek"}}]
        }
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            with patch("backend.llm.client.requests.post", return_value=resp) as mock_post:
                result = llm_complete("s", "u")
        assert result == "hello from deepseek"
        mock_post.assert_called_once()
        assert "deepseek" in mock_post.call_args[0][0]

    def test_falls_back_to_gemini_on_deepseek_failure(self):
        gemini_resp = MagicMock()
        gemini_resp.raise_for_status = MagicMock()
        gemini_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "hello from gemini"}]}}]
        }

        def side_effect(url, **kwargs):
            if "deepseek" in url:
                raise ConnectionError("deepseek down")
            return gemini_resp

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "k", "GEMINI_API_KEY": "gk"}):
            with patch("backend.llm.client.requests.post", side_effect=side_effect):
                result = llm_complete("s", "u")
        assert result == "hello from gemini"

    def test_raises_llm_error_when_both_fail(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "k", "GEMINI_API_KEY": "gk"}):
            with patch(
                "backend.llm.client.requests.post",
                side_effect=ConnectionError("network down"),
            ):
                with pytest.raises(LLMError, match="All LLM providers failed"):
                    llm_complete("s", "u")

    def test_raises_llm_error_when_no_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(LLMError):
                llm_complete("s", "u")

    def test_deepseek_missing_key_falls_back_to_gemini(self):
        gemini_resp = MagicMock()
        gemini_resp.raise_for_status = MagicMock()
        gemini_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "gemini fallback"}]}}]
        }
        env = {"GEMINI_API_KEY": "gk"}  # no DEEPSEEK_API_KEY
        with patch.dict(os.environ, env, clear=True):
            with patch("backend.llm.client.requests.post", return_value=gemini_resp):
                result = llm_complete("s", "u")
        assert result == "gemini fallback"
