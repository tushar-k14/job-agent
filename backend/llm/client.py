"""Unified LLM client.

Primary provider: DeepSeek (OpenAI-compatible chat completions API).
Fallback provider: Google Gemini (free tier).

The public surface is a single function, ``llm_complete``, which takes a system
prompt + user prompt and returns the model's text response. If the DeepSeek call
fails for any reason (missing key, network error, rate limit, 5xx), we transparently
fall back to Gemini. If both fail, the underlying exception is raised so callers can
surface a useful error.

A thin ``llm_json`` helper wraps ``llm_complete`` and parses a JSON object out of the
response, tolerating models that wrap JSON in markdown code fences.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
)

DEFAULT_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "90"))


class LLMError(RuntimeError):
    """Raised when every configured provider fails."""


def _deepseek_complete(
    system: str,
    user: str,
    *,
    temperature: float = 0.4,
    max_tokens: int = 2048,
) -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise LLMError("DEEPSEEK_API_KEY is not set")

    resp = requests.post(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        },
        timeout=DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _gemini_complete(
    system: str,
    user: str,
    *,
    temperature: float = 0.4,
    max_tokens: int = 2048,
) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is not set")

    url = f"{GEMINI_BASE_URL}/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        },
        timeout=DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:  # pragma: no cover - defensive
        raise LLMError(f"Unexpected Gemini response shape: {data}") from exc


def llm_complete(
    system: str,
    user: str,
    *,
    temperature: float = 0.4,
    max_tokens: int = 2048,
) -> str:
    """Run a completion, preferring DeepSeek and falling back to Gemini."""
    errors: list[str] = []

    try:
        return _deepseek_complete(
            system, user, temperature=temperature, max_tokens=max_tokens
        )
    except Exception as exc:  # noqa: BLE001 - fall back on any failure
        logger.warning("DeepSeek failed, falling back to Gemini: %s", exc)
        errors.append(f"deepseek: {exc}")

    try:
        return _gemini_complete(
            system, user, temperature=temperature, max_tokens=max_tokens
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Gemini fallback also failed: %s", exc)
        errors.append(f"gemini: {exc}")

    raise LLMError("All LLM providers failed -> " + " | ".join(errors))


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def _extract_json(text: str) -> str:
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        return fenced.group(1)
    bare = _JSON_OBJECT_RE.search(text)
    if bare:
        return bare.group(1)
    return text


def llm_json(
    system: str,
    user: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> Any:
    """Run a completion and parse a JSON value out of the response."""
    raw = llm_complete(
        system, user, temperature=temperature, max_tokens=max_tokens
    )
    candidate = _extract_json(raw)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"Model did not return valid JSON. Raw response:\n{raw}"
        ) from exc
