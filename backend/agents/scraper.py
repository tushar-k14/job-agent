"""JobScraperAgent.

Fetches a job posting URL with requests + BeautifulSoup, strips boilerplate, and
asks the LLM to extract a structured record. The raw page text is kept on the state
so downstream nodes (and debugging) can see what the model worked from.
"""

from __future__ import annotations

import logging

import requests
from bs4 import BeautifulSoup

from ..llm import llm_json
from ..graph.state import ApplicationState

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

_MAX_CHARS = 12000  # keep the prompt bounded

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


def fetch_page_text(url: str) -> str:
    """Download a URL and return cleaned, visible text."""
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "svg"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # collapse runs of blank lines / whitespace
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned = "\n".join(ln for ln in lines if ln)
    return cleaned[:_MAX_CHARS]


def scraper_node(state: ApplicationState) -> dict:
    url = state["job_url"]
    logger.info("Scraping job posting: %s", url)
    try:
        raw_text = fetch_page_text(url)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Failed to fetch job posting: {exc}"}

    if not raw_text.strip():
        return {"error": "Job posting page returned no readable text."}

    try:
        parsed = llm_json(
            _EXTRACT_SYSTEM,
            _EXTRACT_USER_TMPL.format(text=raw_text),
            temperature=0.1,
        )
    except Exception as exc:  # noqa: BLE001
        return {"raw_job_text": raw_text, "error": f"Failed to parse job posting: {exc}"}

    # normalize shape defensively
    parsed = {
        "title": parsed.get("title") or "Unknown",
        "company": parsed.get("company") or "Unknown",
        "required_skills": parsed.get("required_skills") or [],
        "responsibilities": parsed.get("responsibilities") or [],
        "nice_to_haves": parsed.get("nice_to_haves") or [],
        "salary": parsed.get("salary"),
    }
    return {"raw_job_text": raw_text, "parsed_job": parsed}
