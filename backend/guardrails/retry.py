"""Transient-failure retry with exponential backoff.

Used to wrap individual provider HTTP calls in the LLM client. Only *transient* errors
(timeouts, connection errors, HTTP 429/500/502/503/504) are retried; deterministic
failures (missing key, 400/401/403) raise immediately so we don't waste time/backoff on
something that can't succeed.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

import requests

logger = logging.getLogger(__name__)

T = TypeVar("T")

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in _RETRYABLE_STATUS
    return False


def with_backoff(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn`` with retries on transient errors, exponential backoff + jitter.

    Non-transient exceptions propagate immediately. After ``max_attempts`` transient
    failures, the last exception is re-raised.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if not is_transient(exc) or attempt >= max_attempts:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, delay * 0.25)  # jitter
            logger.warning(
                "Transient error (attempt %d/%d), backing off %.2fs: %s",
                attempt, max_attempts, delay, exc,
            )
            sleep(delay)
