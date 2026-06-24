"""Extraction strategies for the executor.

Each strategy turns a job posting (URL or pre-fetched text) into a ``ParsedJob``. They
share the HTTP fetch + prompt helpers but differ in HOW they isolate the job content
before handing it to the LLM:

- PASTE_TEXT          : caller already supplied the text; no fetch.
- SELECTOR_STRUCTURED : look for known job-board container selectors first.
- GENERIC_TEXT        : strip boilerplate tags, use all visible text (legacy default).
- LLM_FROM_RAW_HTML   : feed a slice of raw HTML to the LLM (odd/unknown markup).

This module is intentionally free of LangGraph/state concerns so strategies are unit-
testable in isolation and reusable from the executor node.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from ..llm import llm_json
from ..schemas import ExtractionStrategy, ParsedJob

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

_BLOCKED_DOMAINS = {
    "linkedin.com",
    "indeed.com",
    "glassdoor.com",
    "ziprecruiter.com",
}

# Container selectors used by common ATS platforms, tried in order.
_JOB_CONTAINER_SELECTORS = [
    "div#content",                 # Greenhouse
    "div.content",                 # Greenhouse alt
    "div.posting",                 # Lever
    "div[data-automation-id='jobPostingPage']",  # Workday
    "section.job",                 # generic
    "div.job-description",         # generic
    "main",                        # last structural fallback
]

_MAX_CHARS = 12000

_EXTRACT_SYSTEM = (
    "You are a precise information extraction engine. Given the visible text of a "
    "job posting web page, extract the structured fields requested. Only use "
    "information present in the text. If a field is absent, use null (for salary) "
    "or an empty list. Respond with a single JSON object and nothing else."
)

_EXTRACT_USER_TMPL = """Extract the following fields from this job posting.

Return JSON with exactly these keys:
{{
  "title": string,                  // job title
  "company": string,                // hiring company
  "required_skills": [string],      // must-have skills/qualifications
  "responsibilities": [string],     // day-to-day duties
  "nice_to_haves": [string],        // preferred / bonus qualifications
  "salary": string or null          // salary or range if stated, else null
}}

JOB POSTING TEXT:
\"\"\"
{text}
\"\"\""""


class BlockedDomainError(ValueError):
    """Raised for domains known to hard-block scraping."""


def check_blocked_domain(url: str) -> None:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    for domain in _BLOCKED_DOMAINS:
        if host == domain or host.endswith("." + domain):
            raise BlockedDomainError(
                f"{domain} blocks automated scraping. Use the company's own careers "
                f"page URL, or paste the job description text instead."
            )


def fetch_html(url: str) -> str:
    """Download a URL and return raw HTML (blocked-domain check enforced)."""
    check_blocked_domain(url)
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def _clean_visible_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "svg"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned = "\n".join(ln for ln in lines if ln)
    return cleaned[:_MAX_CHARS]


def text_from_generic(html: str) -> str:
    """Legacy default: strip boilerplate tags, return all visible text."""
    return _clean_visible_text(BeautifulSoup(html, "html.parser"))


def text_from_selectors(html: str) -> str:
    """Try known ATS container selectors; fall back to generic if none match well."""
    soup = BeautifulSoup(html, "html.parser")
    for selector in _JOB_CONTAINER_SELECTORS:
        node = soup.select_one(selector)
        if node:
            inner = _clean_visible_text(BeautifulSoup(str(node), "html.parser"))
            # Only accept if the container actually has substance.
            if len(inner) >= 200:
                logger.info("Selector matched job container: %s", selector)
                return inner
    # No good container — defer to generic text.
    return _clean_visible_text(soup)


def _llm_extract(text: str) -> ParsedJob:
    raw = llm_json(
        _EXTRACT_SYSTEM,
        _EXTRACT_USER_TMPL.format(text=text[:_MAX_CHARS]),
        temperature=0.1,
    )
    return ParsedJob.normalized(raw)


def extract(
    strategy: ExtractionStrategy,
    *,
    url: str = "",
    raw_text: str = "",
    html: str = "",
) -> tuple[str, ParsedJob]:
    """Run one extraction strategy. Returns (text_used, parsed_job).

    The caller (executor) is responsible for fetching HTML once and passing it in, so
    retries with different strategies don't re-download the page.
    """
    if strategy == ExtractionStrategy.PASTE_TEXT:
        text = raw_text[:_MAX_CHARS]
        return text, _llm_extract(text)

    if strategy == ExtractionStrategy.SELECTOR_STRUCTURED:
        text = text_from_selectors(html)
        return text, _llm_extract(text)

    if strategy == ExtractionStrategy.GENERIC_TEXT:
        text = text_from_generic(html)
        return text, _llm_extract(text)

    if strategy == ExtractionStrategy.LLM_FROM_RAW_HTML:
        # Hand a trimmed slice of raw HTML directly to the LLM.
        text = html[:_MAX_CHARS]
        return text, _llm_extract(text)

    raise ValueError(f"Unknown extraction strategy: {strategy}")
